#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "host/command_line.h"
#include "host/commands.h"
#include "host/fastrpc.h"
#include "host/files.h"
#include "host/model_file.h"
#include "host/timing.h"
#include "host/vggt_network.h"
#include "benzene/protocol.h"

namespace benzene::host {
namespace {

namespace dpt_stat = protocol::dpt_head::stat;
namespace gaussian_stat = protocol::anysplat::gaussian_stat;
using protocol::anysplat::kGaussianChannels;
using protocol::anysplat::kGaussianOutputChannels;

constexpr float kDefaultVoxelSize = 0.002f;  // AnySplat's voxel_size
constexpr int kShDegree = 4;                 // AnySplat's (most PLY viewers read up to degree 3)
constexpr int kShBases = (kShDegree + 1) * (kShDegree + 1);
// A pixel's raw channels (python/benzene/models/anysplat/network.py RAW_*): the opacity logit, the scales (3), the
// rotation (4, quaternion x y z w), the SH [3][kShBases], the confidence; a voxel averages all but the confidence.
constexpr int kRawOpacity = 0;
constexpr int kRawScales = 1;
constexpr int kRawRotation = 4;
constexpr int kRawHarmonics = 8;
constexpr int kRawConfidence = kGaussianChannels - 1;
constexpr int kAveraged = kRawConfidence;
static_assert(kRawHarmonics + 3 * kShBases == kRawConfidence);
constexpr int kPlyValues = 3 + 3 + 3 * kShBases + 1 + 3 + 4;  // x y z nx ny nz f_dc f_rest opacity scale rot
constexpr int kThreads = 4;

// A camera of the pose encoding [T (3), quaternion (x, y, z, w), fov_h, fov_w] as AnySplat decodes it (its copy of
// vggt's pose_encoding_to_extri_intri): world-to-camera R, t; focal lengths (size / 2) / (tan(fov / 2) + 1e-3).
struct Camera {
    float R[3][3];
    float t[3];
    float fx, fy, cx, cy;
};

Camera decode_camera(const float * pose, int height, int width) {
    Camera camera;
    const float i = pose[3], j = pose[4], k = pose[5], r = pose[6];
    const float two_s = 2.0f / (i * i + j * j + k * k + r * r);
    const float R[3][3] = {{1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r)},
                           {two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r)},
                           {two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)}};
    memcpy(camera.R, R, sizeof R);
    for (int c = 0; c < 3; c++) camera.t[c] = pose[c];
    camera.fy = (static_cast<float>(height) / 2.0f) / (tanf(pose[7] / 2.0f) + 1e-3f);
    camera.fx = (static_cast<float>(width) / 2.0f) / (tanf(pose[8] / 2.0f) + 1e-3f);
    camera.cx = static_cast<float>(width) / 2.0f;
    camera.cy = static_cast<float>(height) / 2.0f;
    return camera;
}

// The pixels' world points (the depth map through the camera, pixel (u, v) at its integer index) and voxels
// round(p / voxel_size) (half to even, as torch.round); `order` sorts the pixels by voxel, in the lexicographic order
// of torch.unique (by keys packing 21 bits per axis when every voxel fits).
struct Pixels {
    std::vector<float> points;    // [P][3]
    std::vector<int32_t> voxels;  // [P][3]
    std::vector<uint64_t> keys;   // [P]
    std::vector<uint32_t> order;
};

constexpr int32_t kPackedLimit = 1 << 20;

void unproject_frames(const float * pose, const float * depth, int first, int end, int height, int width,
                      float voxel_size, Pixels * pixels, bool * packed) {
    const size_t frame_pixels = static_cast<size_t>(height) * width;
    constexpr float kIntLimit = 2147483520.0f;  // the largest float below 2^31
    for (int s = first; s < end; s++) {
        const Camera camera = decode_camera(pose + static_cast<size_t>(s) * protocol::vggt::kPoseValues, height, width);
        for (int v = 0; v < height; v++) {
            for (int u = 0; u < width; u++) {
                const size_t p = s * frame_pixels + static_cast<size_t>(v) * width + u;
                const float d = depth[p * 2];
                const float cam[3] = {(static_cast<float>(u) - camera.cx) * d / camera.fx,
                                      (static_cast<float>(v) - camera.cy) * d / camera.fy, d};
                uint64_t key = 0;
                for (int c = 0; c < 3; c++) {  // R^T (cam - t)
                    const float world = camera.R[0][c] * (cam[0] - camera.t[0]) +
                                        camera.R[1][c] * (cam[1] - camera.t[1]) +
                                        camera.R[2][c] * (cam[2] - camera.t[2]);
                    float voxel = nearbyintf(world / voxel_size);  // clamped to int32 (NaN: the lowest)
                    voxel = voxel >= -kIntLimit ? (voxel <= kIntLimit ? voxel : kIntLimit) : -kIntLimit;
                    const auto index = static_cast<int32_t>(voxel);
                    pixels->points[p * 3 + c] = world;
                    pixels->voxels[p * 3 + c] = index;
                    if (index < -kPackedLimit || index >= kPackedLimit) *packed = false;
                    key = (key << 21) | (static_cast<uint64_t>(index + kPackedLimit) & ((uint64_t{1} << 21) - 1));
                }
                pixels->keys[p] = key;
            }
        }
    }
}

bool same_voxel(const Pixels & pixels, uint32_t a, uint32_t b) {
    return memcmp(&pixels.voxels[static_cast<size_t>(a) * 3], &pixels.voxels[static_cast<size_t>(b) * 3], 12) == 0;
}

// Sorts (key, pixel) pairs in kThreads parts on threads, then merges the parts.
void sort_pairs(std::vector<std::pair<uint64_t, uint32_t>> * pairs) {
    const size_t count = pairs->size();
    size_t bounds[kThreads + 1];
    for (int t = 0; t <= kThreads; t++) bounds[t] = count * t / kThreads;
    std::thread threads[kThreads];
    for (int t = 0; t < kThreads; t++) {
        threads[t] =
            std::thread([pairs, &bounds, t] { std::sort(pairs->begin() + bounds[t], pairs->begin() + bounds[t + 1]); });
    }
    for (std::thread & thread : threads) thread.join();
    for (int width = 1; width < kThreads; width *= 2) {
        for (int t = 0; t + width < kThreads; t += 2 * width) {
            const size_t end = bounds[t + 2 * width < kThreads ? t + 2 * width : kThreads];
            std::inplace_merge(pairs->begin() + bounds[t], pairs->begin() + bounds[t + width], pairs->begin() + end);
        }
    }
}

void sort_by_voxel(Pixels * pixels, bool packed) {
    const size_t count = pixels->keys.size();
    pixels->order.resize(count);
    if (packed) {
        std::vector<std::pair<uint64_t, uint32_t>> pairs(count);
        for (size_t i = 0; i < count; i++) pairs[i] = {pixels->keys[i], static_cast<uint32_t>(i)};
        sort_pairs(&pairs);
        for (size_t i = 0; i < count; i++) pixels->order[i] = pairs[i].second;
        return;
    }
    for (size_t i = 0; i < count; i++) pixels->order[i] = static_cast<uint32_t>(i);
    const std::vector<int32_t> & voxels = pixels->voxels;
    std::sort(pixels->order.begin(), pixels->order.end(), [&](uint32_t a, uint32_t b) {
        for (int c = 0; c < 3; c++) {
            const int32_t va = voxels[static_cast<size_t>(a) * 3 + c];
            const int32_t vb = voxels[static_cast<size_t>(b) * 3 + c];
            if (va != vb) return va < vb;
        }
        return a < b;
    });
}

float softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));  // torch's threshold 20
}

struct Fusion {
    const Pixels * pixels;
    const _Float16 * const * raw;  // per frame [H * W][kGaussianOutputChannels]
    size_t frame_pixels;
    const uint32_t * group_starts;  // voxel g: order[group_starts[g] .. group_starts[g + 1])
    float * ply;                    // [voxels][kPlyValues]
    float sh_mask[kShBases];
};

// AnySplat's voxelizaton_with_fusion and UnifiedGaussianAdapter for voxels [first, end): weights
// exp(c - max) / (sum + 1e-6) over the voxel's pixels; the PLY record of the 3DGS layout (opacity as its logit, log
// scales, quaternion w x y z, SH channel-major).
void fuse_voxels(const Fusion * f, size_t first, size_t end) {
    std::vector<float> weights;
    std::vector<const _Float16 *> raws;
    for (size_t g = first; g < end; g++) {
        const uint32_t begin = f->group_starts[g];
        const uint32_t pixels = f->group_starts[g + 1] - begin;
        const uint32_t * order = f->pixels->order.data() + begin;
        weights.resize(pixels);
        raws.resize(pixels);
        float peak = -INFINITY;
        for (uint32_t i = 0; i < pixels; i++) {
            raws[i] = f->raw[order[i] / f->frame_pixels] + (order[i] % f->frame_pixels) * kGaussianOutputChannels;
            weights[i] = static_cast<float>(raws[i][kRawConfidence]);
            peak = weights[i] > peak ? weights[i] : peak;
        }
        float total = 0.0f;
        for (uint32_t i = 0; i < pixels; i++) {
            weights[i] = expf(weights[i] - peak);
            total += weights[i];
        }
        float point[3] = {};
        float features[kAveraged] = {};
        for (uint32_t i = 0; i < pixels; i++) {
            const float w = weights[i] / (total + 1e-6f);
            for (int c = 0; c < 3; c++) point[c] += f->pixels->points[static_cast<size_t>(order[i]) * 3 + c] * w;
            const _Float16 * raw = raws[i];
            for (int c = 0; c < kAveraged; c++) features[c] += static_cast<float>(raw[c]) * w;
        }

        float * out = f->ply + g * kPlyValues;
        for (int c = 0; c < 3; c++) out[c] = point[c];
        out[3] = out[4] = out[5] = 0.0f;
        const float * sh = features + kRawHarmonics;  // [3][kShBases]
        for (int c = 0; c < 3; c++) out[6 + c] = sh[c * kShBases] * f->sh_mask[0];
        for (int c = 0; c < 3; c++) {
            for (int k = 1; k < kShBases; k++)
                out[9 + c * (kShBases - 1) + k - 1] = sh[c * kShBases + k] * f->sh_mask[k];
        }
        float * rest = out + 9 + 3 * (kShBases - 1);
        rest[0] = features[kRawOpacity];  // the opacity's logit
        for (int c = 0; c < 3; c++) {
            const float scale = 0.001f * softplus(features[kRawScales + c]);
            rest[1 + c] = logf(scale < 0.3f ? scale : 0.3f);
        }
        const float * q = features + kRawRotation;  // x y z w
        const float norm = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) + 1e-8f;
        rest[4] = q[3] / norm;
        rest[5] = q[0] / norm;
        rest[6] = q[1] / norm;
        rest[7] = q[2] / norm;
    }
}

// <directory>/gaussians.ply: the 3DGS PLY of the records with the SH up to degree sh_degree (f_rest: per channel
// (degree + 1)^2 - 1 values); the records are compacted in place when that drops SH. False and a message on failure.
bool write_ply(const char * directory, float * ply, size_t count, int sh_degree) {
    const int rest = (sh_degree + 1) * (sh_degree + 1) - 1;  // per channel
    const int values = kPlyValues - 3 * (kShBases - 1 - rest);
    std::string header = "ply\nformat binary_little_endian 1.0\nelement vertex " + std::to_string(count) + "\n";
    std::vector<std::string> names = {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"};
    for (int i = 0; i < 3 * rest; i++) names.push_back("f_rest_" + std::to_string(i));
    for (const char * name : {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"})
        names.emplace_back(name);
    for (const std::string & name : names) header += "property float " + name + "\n";
    header += "end_header\n";
    if (values != kPlyValues) {
        for (size_t g = 0; g < count; g++) {
            const float * source = ply + g * kPlyValues;
            float * destination = ply + g * values;  // at or before the source
            memmove(destination, source, 9 * sizeof(float));
            for (int c = 0; c < 3; c++) {
                memmove(destination + 9 + c * rest, source + 9 + c * (kShBases - 1), rest * sizeof(float));
            }
            memmove(destination + 9 + 3 * rest, source + 9 + 3 * (kShBases - 1), 8 * sizeof(float));
        }
    }
    const std::string path = join_path(directory, "gaussians.ply");
    const File file = open_file(path.c_str(), "wb");
    if (!file || fwrite(header.data(), 1, header.size(), file.get()) != header.size() ||
        fwrite(ply, sizeof(float), count * values, file.get()) != count * values) {
        fprintf(stderr, "write %s failed\n", path.c_str());
        return false;
    }
    return true;
}

// The Gaussians of the predictions (pose: the camera head's last iteration [frames][kPoseValues]; depth
// [frames][H][W][2]; raw per frame) into <directory>/gaussians.ply; prints the line of their times. False and a message
// on failure.
bool make_gaussians(const float * pose, const float * depth, const std::vector<_Float16 *> & raw, int frames,
                    int height, int width, float voxel_size, int sh_degree, const char * directory) {
    const double start = now_microseconds();
    const size_t frame_pixels = static_cast<size_t>(height) * width;
    const size_t count = frame_pixels * frames;
    Pixels pixels;
    pixels.points.resize(count * 3);
    pixels.voxels.resize(count * 3);
    pixels.keys.resize(count);
    bool packed = true;
    {
        std::thread threads[kThreads];
        bool fits[kThreads];
        for (int t = 0; t < kThreads; t++) {
            fits[t] = true;
            threads[t] = std::thread(unproject_frames, pose, depth, frames * t / kThreads, frames * (t + 1) / kThreads,
                                     height, width, voxel_size, &pixels, &fits[t]);
        }
        for (std::thread & thread : threads) thread.join();
        for (const bool fit : fits) packed = packed && fit;
    }
    const double unprojected = now_microseconds();
    sort_by_voxel(&pixels, packed);
    const double sorted = now_microseconds();

    std::vector<uint32_t> group_starts;
    for (size_t i = 0; i < count; i++) {
        if (i == 0 || !same_voxel(pixels, pixels.order[i], pixels.order[i - 1]))
            group_starts.push_back(static_cast<uint32_t>(i));
    }
    const size_t voxels = group_starts.size();
    group_starts.push_back(static_cast<uint32_t>(count));
    // Every value is written by the fusion threads (which also take the page faults of the fresh memory in parallel).
    const MallocBuffer<float> ply = allocate_buffer<float>(voxels * kPlyValues);
    if (!ply) {
        fprintf(stderr, "out of memory\n");
        return false;
    }
    Fusion fusion{&pixels, raw.data(), frame_pixels, group_starts.data(), ply.get(), {}};
    for (int degree = 0; degree <= kShDegree; degree++) {
        for (int k = degree * degree; k < (degree + 1) * (degree + 1); k++)
            fusion.sh_mask[k] = degree == 0 ? 1.0f : 0.1f * powf(0.25f, static_cast<float>(degree));
    }
    {
        std::thread threads[kThreads];
        for (int t = 0; t < kThreads; t++)
            threads[t] = std::thread(fuse_voxels, &fusion, voxels * t / kThreads, voxels * (t + 1) / kThreads);
        for (std::thread & thread : threads) thread.join();
    }
    const double fused = now_microseconds();
    if (!write_ply(directory, ply.get(), voxels, sh_degree)) return false;
    const double written = now_microseconds();
    printf(
        "gaussians %.0f ms | %zu of %zu pixels, voxel size %g, SH degree %d | points %.0f voxels %.0f fusion %.0f ply "
        "%.0f\n",
        (written - start) / 1e3, voxels, count, static_cast<double>(voxel_size), sh_degree, (unprojected - start) / 1e3,
        (sorted - unprojected) / 1e3, (fused - sorted) / 1e3, (written - fused) / 1e3);
    return true;
}

}  // namespace

int run_anysplat(int argc, char ** argv) {
    bool keep_raw = false;
    bool keep_tokens = false;
    const char * voxel_text = nullptr;
    const char * degree_text = nullptr;
    const Option options[] = {{"--raw", nullptr, &keep_raw},
                              {"--tokens", nullptr, &keep_tokens},
                              {"--voxel-size", &voxel_text},
                              {"--sh-degree", &degree_text}};
    if (!parse_command_line(
            argc, argv, 3, options, 4,
            "model.hexmodel bundle_dir out_dir [--raw] [--tokens] [--voxel-size SIZE] [--sh-degree N]")) {
        return 1;
    }
    float voxel_size = kDefaultVoxelSize;
    if (voxel_text != nullptr && !parse_float(voxel_text, "--voxel-size", 1e-6f, 1.0f, &voxel_size)) return 1;
    int sh_degree = kShDegree;  // the PLY's
    if (degree_text != nullptr && !parse_integer(degree_text, "--sh-degree", 0, kShDegree, &sh_degree)) return 1;
    const char * model = argv[2];
    const char * output = argv[4];

    ModelTable table;
    if (!load_table_of_model(model, &table) || !make_directory(output)) return 1;
    VggtBundle bundle;
    if (!read_vggt_bundle(argv[3], &bundle)) return 1;
    const int frames = bundle.frames;
    const int height = bundle.height;
    const int width = bundle.width;
    const size_t frame_pixels = static_cast<size_t>(height) * width;
    const int patch_rows = height / protocol::vggt::kPatchSize;
    const int patch_columns = width / protocol::vggt::kPatchSize;
    VggtWeights weights;
    int32_t depth_references[protocol::vggt::kDptReferences];
    int32_t gaussian_references[protocol::anysplat::kGaussianReferences];
    const DptHeadTensors heads[2] = {
        {"vggt.dpt.depth", "misc", protocol::vggt::dpt_head_part::kCount, depth_references},
        {"anysplat.gaussians", "head", protocol::anysplat::gaussian_head_part::kCount, gaussian_references},
    };
    if (!plan_vggt_weights(table, model, &weights) || !plan_dpt_heads(table, model, heads, 2, &weights)) {
        return 1;
    }

    const std::vector<float> uv = dpt_uv_tables(patch_rows, patch_columns, height, width);
    _Float16 * patches = nullptr;
    std::vector<float> grid_positions;
    float special[protocol::vggt::kSpecialTokens * protocol::vggt::kHidden];
    if (!load_vggt_weights(model, table, bundle, &weights, &patches, &grid_positions, special)) return 1;
    const size_t kept_values = keep_tokens ? static_cast<size_t>(protocol::vggt::kKeptOutputs) * frames *
                                                 (protocol::vggt::kSpecialTokens + patch_rows * patch_columns) *
                                                 protocol::vggt::kKeptOutputWidth
                                           : 0;
    auto * kept = keep_tokens ? static_cast<_Float16 *>(allocate_rpc_memory(kept_values * 2)) : nullptr;
    if (keep_tokens && kept == nullptr) return 1;
    if (!run_vggt_transformer("anysplat", weights, bundle, patches, grid_positions, special, kept, kept_values)) {
        return 1;
    }
    if (keep_tokens && !write_file(output, "inter.f16", kept, kept_values * 2)) return 1;
    if (stopped_by_signal()) return 1;

    std::vector<float> pose;
    if (!run_vggt_camera_head(weights, frames, &pose)) return 1;
    if (!write_file(output, "pose.f32", pose.data(), pose.size() * 4)) return 1;
    if (stopped_by_signal()) return 1;
    const size_t depth_values = frame_pixels * frames * 2;
    auto * depth = static_cast<float *>(allocate_rpc_memory(depth_values * 4));
    if (depth == nullptr || !run_vggt_dpt_head("depth", depth_references, 2, uv, depth, depth_values)) return 1;
    if (!write_file(output, "depth.f32", depth, depth_values * 4)) return 1;

    // The Gaussian head, frame by frame (its output [H][W][kGaussianOutputChannels] per call).
    const size_t raw_values = frame_pixels * kGaussianOutputChannels;
    std::vector<_Float16 *> raw(static_cast<size_t>(frames));
    auto * image = static_cast<uint8_t *>(allocate_rpc_memory(frame_pixels * 3));
    if (image == nullptr) return 1;
    int32_t totals[gaussian_stat::kCount] = {};
    double call_microseconds = 0.0;
    for (int frame = 0; frame < frames; frame++) {
        if (stopped_by_signal()) return 1;
        raw[static_cast<size_t>(frame)] = static_cast<_Float16 *>(allocate_rpc_memory(raw_values * 2));
        if (raw[static_cast<size_t>(frame)] == nullptr) return 1;
        memcpy(image, bundle.images.get() + static_cast<size_t>(frame) * frame_pixels * 3, frame_pixels * 3);
        int32_t s[gaussian_stat::kCount] = {};
        const double start = now_microseconds();
        const int error = benzene_anysplat_gaussian_head(
            session(), gaussian_references, protocol::anysplat::kGaussianReferences, uv.data(),
            static_cast<int>(uv.size()), static_cast<uint32_t>(frame), image, static_cast<int>(frame_pixels * 3),
            reinterpret_cast<uint16_t *>(raw[static_cast<size_t>(frame)]), static_cast<int>(raw_values), s,
            gaussian_stat::kCount);
        call_microseconds += now_microseconds() - start;
        if (!rpc_succeeded(error, "anysplat_gaussian_head (frame %d)", frame)) return 1;
        for (int i = 0; i < gaussian_stat::kCount; i++) totals[i] += s[i];
    }
    printf(
        "gaussian head %.0f ms (calls %.0f) | projects+resize %.1f layer_rn %.1f refinenets %.1f merger %.1f output "
        "convs %.1f elementwise %.1f workspace %.1f\n",
        totals[dpt_stat::kTotalMicroseconds] / 1e3, call_microseconds / 1e3,
        totals[dpt_stat::kProjectsMicroseconds] / 1e3, totals[dpt_stat::kLayerRnMicroseconds] / 1e3,
        totals[dpt_stat::kRefineNetsMicroseconds] / 1e3, totals[gaussian_stat::kMergerMicroseconds] / 1e3,
        totals[dpt_stat::kOutputConvolutionsMicroseconds] / 1e3, totals[dpt_stat::kElementwiseMicroseconds] / 1e3,
        totals[dpt_stat::kWorkspaceMicroseconds] / 1e3);
    if (keep_raw) {  // the calls' outputs as they are: [frames][H][W][kGaussianOutputChannels]
        const std::string path = join_path(output, "gaussian_head.f16");
        const File file = open_file(path.c_str(), "wb");
        for (const _Float16 * frame_raw : raw) {
            if (!file || fwrite(frame_raw, 2, raw_values, file.get()) != raw_values) {
                fprintf(stderr, "write %s failed\n", path.c_str());
                return 1;
            }
        }
    }
    if (stopped_by_signal()) return 1;

    const float * last_pose =
        pose.data() + static_cast<size_t>(protocol::vggt::kCameraIterations - 1) * frames * protocol::vggt::kPoseValues;
    return make_gaussians(last_pose, depth, raw, frames, height, width, voxel_size, sh_degree, output) ? 0 : 1;
}

}  // namespace benzene::host
