#include "host/vggt_network.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include "host/fastrpc.h"
#include "host/timing.h"

namespace benzene::host {
namespace {

namespace block_part = protocol::vision_transformer::block_part;
namespace camera_part = protocol::vggt::camera_part;
namespace convolution = protocol::dpt_head::convolution;
namespace convolution_part = protocol::dpt_head::convolution_part;
namespace dpt_stat = protocol::dpt_head::stat;
namespace tail_part = protocol::vggt::tail_part;
namespace transformer_stat = protocol::vggt::transformer_stat;
using protocol::vggt::kHidden;
using protocol::vggt::kPatchSize;
using protocol::vggt::kSpecialTokens;
using protocol::vision_transformer::BlockKind;

constexpr int kPatchValues = 3 * kPatchSize * kPatchSize;
constexpr int kDinov2Blocks = 24;
constexpr int kPositionGrid = 37;
constexpr uint64_t kGroupBytes = 240ull << 20;  // blocks per rpcmem buffer: within one 256 MiB IOVA window
constexpr int kMaximumRanges = 8;

// W[out][in]: PyTorch's antialiased bicubic weights (aten _compute_weights_aa, a = -0.5, align_corners false).
void antialiased_bicubic_weights(int in, int out, float * W) {
    const float scale = static_cast<float>(in) / static_cast<float>(out);
    const float support = scale >= 1.0f ? 2.0f * scale : 2.0f;
    const float inverse_scale = scale >= 1.0f ? 1.0f / scale : 1.0f;
    memset(W, 0, sizeof(float) * static_cast<size_t>(in) * out);

    for (int o = 0; o < out; o++) {
        const float center = scale * (static_cast<float>(o) + 0.5f);
        int low = static_cast<int>(center - support + 0.5f);
        int high = static_cast<int>(center + support + 0.5f);
        low = low < 0 ? 0 : low;
        high = high > in ? in : high;

        float total = 0.0f;
        for (int i = low; i < high; i++) {
            const float x = fabsf((static_cast<float>(i) - center + 0.5f) * inverse_scale);
            float w = 0.0f;
            if (x < 1.0f) {
                w = (1.5f * x - 2.5f) * x * x + 1.0f;  // ((a + 2) x - (a + 3)) x^2 + 1
            } else if (x < 2.0f) {
                w = ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;  // ((a x - 5 a) x + 8 a) x - 4 a
            }
            W[static_cast<size_t>(o) * in + i] = w;
            total += w;
        }
        if (total != 0.0f) {
            for (int i = low; i < high; i++) W[static_cast<size_t>(o) * in + i] /= total;
        }
    }
}

// positions [1 + M * M][1024] (cls, then the M x M grid) -> out [rows * columns][1024]
void resample_position_grid(const float * positions, int M, int rows, int columns, float * out) {
    const float * grid = positions + kHidden;
    if (rows == M && columns == M) {
        memcpy(out, grid, sizeof(float) * static_cast<size_t>(M) * M * kHidden);
        return;
    }

    std::vector<float> row_weights(static_cast<size_t>(M) * rows);
    std::vector<float> column_weights(static_cast<size_t>(M) * columns);
    std::vector<float> resampled_columns(static_cast<size_t>(M) * columns * kHidden);
    antialiased_bicubic_weights(M, rows, row_weights.data());
    antialiased_bicubic_weights(M, columns, column_weights.data());

    for (int i = 0; i < M; i++) {
        for (int x = 0; x < columns; x++) {
            for (int j = 0; j < M; j++) {
                const float w = column_weights[static_cast<size_t>(x) * M + j];
                if (w == 0.0f) continue;
                const float * source = grid + (static_cast<size_t>(i) * M + j) * kHidden;
                float * destination = resampled_columns.data() + (static_cast<size_t>(i) * columns + x) * kHidden;
                for (int c = 0; c < kHidden; c++) destination[c] += w * source[c];
            }
        }
    }

    memset(out, 0, sizeof(float) * static_cast<size_t>(rows) * columns * kHidden);
    for (int y = 0; y < rows; y++) {
        for (int i = 0; i < M; i++) {
            const float w = row_weights[static_cast<size_t>(y) * M + i];
            if (w == 0.0f) continue;
            for (int x = 0; x < columns; x++) {
                const float * source = resampled_columns.data() + (static_cast<size_t>(i) * columns + x) * kHidden;
                float * destination = out + (static_cast<size_t>(y) * columns + x) * kHidden;
                for (int c = 0; c < kHidden; c++) destination[c] += w * source[c];
            }
        }
    }
}

const float kMean[3] = {0.485f, 0.456f, 0.406f};
const float kStd[3] = {0.229f, 0.224f, 0.225f};

void write_patches(const uint8_t * images, int height, int width, int patch_rows, int patch_columns, int begin, int end,
                   _Float16 * patches) {
    const int patches_per_frame = patch_rows * patch_columns;
    for (int patch = begin; patch < end; patch++) {
        const int frame = patch / patches_per_frame;
        const int index = patch % patches_per_frame;
        const int py = index / patch_columns;
        const int px = index % patch_columns;
        _Float16 * destination = patches + static_cast<size_t>(patch) * protocol::vggt::kPatchInputs;
        for (int c = 0; c < 3; c++) {
            for (int ky = 0; ky < kPatchSize; ky++) {
                const uint8_t * source =
                    images + ((static_cast<size_t>(frame) * 3 + c) * height + py * kPatchSize + ky) * width +
                    px * kPatchSize;
                for (int kx = 0; kx < kPatchSize; kx++) {
                    destination[(c * kPatchSize + ky) * kPatchSize + kx] =
                        static_cast<_Float16>((static_cast<float>(source[kx]) / 255.0f - kMean[c]) / kStd[c]);
                }
            }
        }
        for (int k = kPatchValues; k < protocol::vggt::kPatchInputs; k++) destination[k] = static_cast<_Float16>(0.0f);
    }
}

bool dinov2_inputs(const char * model, const ModelTable & table, const uint8_t * images, int frames, int height,
                   int width, _Float16 ** patches, std::vector<float> * grid_positions, float * special) {
    const TensorRecord * position_record = table.find("vggt.pos_embed");
    const TensorRecord * cls_record = table.find("vggt.cls_token");
    const TensorRecord * register_record = table.find("vggt.register_tokens");
    if (position_record == nullptr || cls_record == nullptr || register_record == nullptr) {
        fprintf(stderr, "missing tensor vggt.pos_embed, vggt.cls_token or vggt.register_tokens\n");
        return false;
    }

    const int patch_rows = height / kPatchSize;
    const int patch_columns = width / kPatchSize;
    const MallocBuffer<float> positions = read_f32_tensor(model, *position_record);
    const MallocBuffer<float> cls = read_f32_tensor(model, *cls_record);
    const MallocBuffer<float> registers = read_f32_tensor(model, *register_record);
    if (!positions || !cls || !registers) return false;
    const int patch_count = frames * patch_rows * patch_columns;
    *patches = static_cast<_Float16 *>(
        allocate_rpc_memory(static_cast<uint64_t>(patch_count) * protocol::vggt::kPatchInputs * 2));
    grid_positions->resize(static_cast<size_t>(patch_rows) * patch_columns * kHidden);
    if (*patches == nullptr) return false;

    constexpr int kPatchThreads = 4;
    std::thread threads[kPatchThreads];
    for (int t = 0; t < kPatchThreads; t++) {
        threads[t] = std::thread(write_patches, images, height, width, patch_rows, patch_columns,
                                 patch_count * t / kPatchThreads, patch_count * (t + 1) / kPatchThreads, *patches);
    }
    resample_position_grid(positions.get(), kPositionGrid, patch_rows, patch_columns, grid_positions->data());
    for (std::thread & thread : threads) thread.join();

    for (int c = 0; c < kHidden; c++) special[c] = cls[c] + positions[c];
    memcpy(special + kHidden, registers.get(), sizeof(float) * 4 * kHidden);
    return true;
}

const TensorRecord * find_tensor(const ModelTable & table, const char * name, int parts) {
    const TensorRecord * record = table.find(name);
    if (record == nullptr || static_cast<int>(record->part_count) != parts) {
        fprintf(stderr, "missing tensor %s (%d parts)\n", name, parts);
        return nullptr;
    }
    return record;
}

// A new range (slot: its index) of `bytes` from `offset` on; false when there are kMaximumRanges.
bool add_range(const char * model, uint64_t offset, uint64_t bytes, std::vector<WeightRange> * ranges) {
    if (ranges->size() == kMaximumRanges) {
        fprintf(stderr, "more than %d weight buffers\n", kMaximumRanges);
        return false;
    }
    ranges->push_back({model, offset, bytes, nullptr});
    return true;
}

// The blocks in ranges of at most kGroupBytes, the tail with the last.
bool plan_blocks(const ModelTable & table, const char * model, VggtWeights * weights) {
    std::vector<WeightRange> & ranges = weights->ranges;
    for (int block = 0; block < kVggtBlocks; block++) {
        char name[48];
        snprintf(name, sizeof name, "vggt.blocks.%d", block);
        const TensorRecord * record = find_tensor(table, name, block_part::kCount);
        if (record == nullptr) return false;
        if ((ranges.empty() || record->offset + record->bytes - ranges.back().offset > kGroupBytes) &&
            !add_range(model, record->offset, 0, &ranges)) {
            return false;
        }
        WeightRange & group = ranges.back();
        group.bytes = record->offset + record->bytes - group.offset;

        int32_t * reference = weights->blocks + block * protocol::vggt::kBlockReferences;
        const BlockKind kind = block < kDinov2Blocks              ? BlockKind::kDinov2
                               : (block - kDinov2Blocks) % 2 != 0 ? BlockKind::kGlobal
                                                                  : BlockKind::kFrame;
        reference[0] = static_cast<int32_t>(ranges.size() - 1);
        reference[1] = static_cast<int32_t>(record->offset - group.offset);
        reference[2] = static_cast<int32_t>(kind);
        for (int i = 0; i < block_part::kCount; i++) reference[3 + i] = static_cast<int32_t>(record->parts[i]);
    }

    const TensorRecord * tail = find_tensor(table, "vggt.tail", tail_part::kCount);
    if (tail == nullptr) return false;
    WeightRange & last_group = ranges.back();
    if (tail->offset < last_group.offset) {
        fprintf(stderr, "vggt.tail before the last blocks\n");
        return false;
    }
    last_group.bytes = tail->offset + tail->bytes - last_group.offset;
    weights->tail[0] = static_cast<int32_t>(ranges.size() - 1);
    weights->tail[1] = static_cast<int32_t>(tail->offset - last_group.offset);
    for (int i = 0; i < tail_part::kCount; i++) weights->tail[2 + i] = static_cast<int32_t>(tail->parts[i]);
    return true;
}

// The camera head's trunk blocks and head in one range.
bool plan_camera_head(const ModelTable & table, const char * model, VggtWeights * weights) {
    constexpr int kTrunkBlocks = protocol::vggt::kCameraTrunkBlocks;
    const TensorRecord * camera[kTrunkBlocks + 1];
    for (int part = 0; part <= kTrunkBlocks; part++) {
        char name[48];
        snprintf(name, sizeof name, part < kTrunkBlocks ? "vggt.cam.blocks.%d" : "vggt.cam.head", part);
        const int parts =
            part < kTrunkBlocks ? static_cast<int>(block_part::kCount) : static_cast<int>(camera_part::kCount);
        camera[part] = find_tensor(table, name, parts);
        if (camera[part] == nullptr) return false;
        if (camera[part]->offset < camera[0]->offset) {
            fprintf(stderr, "%s before vggt.cam.blocks.0\n", name);
            return false;
        }
    }

    const TensorRecord & camera_head = *camera[kTrunkBlocks];
    if (!add_range(model, camera[0]->offset, camera_head.offset + camera_head.bytes - camera[0]->offset,
                   &weights->ranges)) {
        return false;
    }
    const auto slot = static_cast<int32_t>(weights->ranges.size() - 1);
    int32_t * reference = weights->camera;
    for (int part = 0; part <= kTrunkBlocks; part++) {
        *reference++ = slot;
        *reference++ = static_cast<int32_t>(camera[part]->offset - camera[0]->offset);
        for (uint32_t i = 0; i < camera[part]->part_count; i++)
            *reference++ = static_cast<int32_t>(camera[part]->parts[i]);
    }
    return true;
}

// As vggt/heads/utils.py: u = linspace(-span (n - 1) / n, span (n - 1) / n, n), then per u the C / 2 channels
// 0.1 sin(u w_d) | 0.1 cos(u w_d), w_d = 100^(-d / (C / 4)).
void uv_embedding(float * embedding, int n, double span, int channels) {
    const int quarter = channels / 4;
    for (int i = 0; i < n; i++) {
        const double u = n > 1 ? -span * (n - 1) / n + 2.0 * span * (n - 1) / n * i / (n - 1) : 0.0;
        for (int d = 0; d < quarter; d++) {
            const double angle = u * (1.0 / pow(100.0, static_cast<double>(d) / quarter));
            embedding[static_cast<size_t>(i) * 2 * quarter + d] =
                static_cast<float>(0.1 * static_cast<double>(static_cast<float>(sin(angle))));
            embedding[static_cast<size_t>(i) * 2 * quarter + quarter + d] =
                static_cast<float>(0.1 * static_cast<double>(static_cast<float>(cos(angle))));
        }
    }
}

}  // namespace

bool read_vggt_bundle(const char * directory, VggtBundle * bundle) {
    const std::string meta_path = join_path(directory, "meta.txt");
    const File meta = open_file(meta_path.c_str(), "r");
    if (!meta || fscanf(meta.get(), "%d %d %d", &bundle->frames, &bundle->height, &bundle->width) != 3 ||
        bundle->frames < 1 || bundle->height % kPatchSize != 0 || bundle->width % kPatchSize != 0) {
        fprintf(stderr, "bad %s\n", meta_path.c_str());
        return false;
    }
    const std::string images_path = join_path(directory, "images.u8");
    size_t image_bytes = 0;
    bundle->images = load_file(images_path.c_str(), &image_bytes);
    if (!bundle->images || image_bytes != static_cast<size_t>(bundle->frames) * 3 * bundle->height * bundle->width) {
        fprintf(stderr, "bad %s\n", images_path.c_str());
        return false;
    }
    return true;
}

bool plan_vggt_weights(const ModelTable & table, const char * model, VggtWeights * weights) {
    return plan_blocks(table, model, weights) && plan_camera_head(table, model, weights);
}

bool plan_dpt_heads(const ModelTable & table, const char * model, const DptHeadTensors * heads, int count,
                    VggtWeights * weights) {
    constexpr int kConvolutions = convolution::kCount;
    constexpr int kConvolutionReferences = protocol::blob_references(convolution_part::kCount);
    std::vector<const TensorRecord *> records;
    const TensorRecord * first = nullptr;
    const TensorRecord * last = nullptr;
    for (int head = 0; head < count; head++) {
        for (int index = 0; index <= kConvolutions; index++) {
            char name[48];
            if (index < kConvolutions) {
                snprintf(name, sizeof name, "%s.%d", heads[head].prefix, index);
            } else {
                snprintf(name, sizeof name, "%s.%s", heads[head].prefix, heads[head].own);
            }
            const TensorRecord * record =
                find_tensor(table, name,
                            index < kConvolutions ? static_cast<int>(convolution_part::kCount) : heads[head].own_parts);
            if (record == nullptr) return false;
            records.push_back(record);
            if (first == nullptr || record->offset < first->offset) first = record;
            if (last == nullptr || record->offset > last->offset) last = record;
        }
    }

    if (!add_range(model, first->offset, last->offset + last->bytes - first->offset, &weights->ranges)) return false;
    const auto slot = static_cast<int32_t>(weights->ranges.size() - 1);
    for (int head = 0; head < count; head++) {
        for (int index = 0; index <= kConvolutions; index++) {
            const TensorRecord & record = *records[static_cast<size_t>(head) * (kConvolutions + 1) + index];
            int32_t * reference = heads[head].references + index * kConvolutionReferences;
            reference[0] = slot;
            reference[1] = static_cast<int32_t>(record.offset - first->offset);
            for (uint32_t i = 0; i < record.part_count; i++) reference[2 + i] = static_cast<int32_t>(record.parts[i]);
        }
    }
    return true;
}

bool load_vggt_weights(const char * model, const ModelTable & table, const VggtBundle & bundle, VggtWeights * weights,
                       _Float16 ** patches, std::vector<float> * grid_positions, float * special) {
    const double start = now_microseconds();
    std::vector<std::thread> loaders;
    std::vector<char> loaded(weights->ranges.size());
    for (size_t g = 0; g < weights->ranges.size(); g++) {
        loaders.emplace_back([&weights, &loaded, g] {
            loaded[g] = allocate_range(&weights->ranges[g]) && read_range(weights->ranges[g]);
        });
    }
    bool ready = dinov2_inputs(model, table, bundle.images.get(), bundle.frames, bundle.height, bundle.width, patches,
                               grid_positions, special);
    for (std::thread & loader : loaders) loader.join();
    for (size_t g = 0; g < weights->ranges.size(); g++) {
        const WeightRange & range = weights->ranges[g];
        ready = ready && loaded[g] && map_into_slot(range.buffer, range.bytes, static_cast<int>(g));
    }
    if (!ready) return false;

    const WeightRange & first = weights->ranges.front();
    const WeightRange & last = weights->ranges.back();
    print_model_load(last.offset + last.bytes - first.offset, static_cast<int>(weights->ranges.size()),
                     now_microseconds() - start);
    return true;
}

std::vector<float> dpt_uv_tables(int patch_rows, int patch_columns, int height, int width) {
    static const int kProjectChannels[4] = {256, 512, 1024, 1024};
    std::vector<float> tables(
        static_cast<size_t>((patch_columns + patch_rows) * (128 + 256 + 512 + 512) + (width + height) * 64));
    const double aspect = static_cast<double>(width) / static_cast<double>(height);
    const double diagonal = sqrt(aspect * aspect + 1.0);
    const double span_x = aspect / diagonal;
    const double span_y = 1.0 / diagonal;

    float * table = tables.data();
    for (const int channels : kProjectChannels) {
        uv_embedding(table, patch_columns, span_x, channels);
        uv_embedding(table + static_cast<size_t>(patch_columns) * channels / 2, patch_rows, span_y, channels);
        table += static_cast<size_t>(patch_columns + patch_rows) * channels / 2;
    }

    uv_embedding(table, width, span_x, 128);
    uv_embedding(table + static_cast<size_t>(width) * 64, height, span_y, 128);
    return tables;
}

bool run_vggt_transformer(const char * name, const VggtWeights & weights, const VggtBundle & bundle,
                          const _Float16 * patches, const std::vector<float> & grid_positions, const float * special,
                          _Float16 * kept, size_t kept_values) {
    const int patch_rows = bundle.height / kPatchSize;
    const int patch_columns = bundle.width / kPatchSize;
    const int tokens = bundle.frames * (kSpecialTokens + patch_rows * patch_columns);
    int32_t statistics[transformer_stat::kCount] = {};
    const double start = now_microseconds();
    const int error = benzene_vggt_transformer(
        session(), weights.blocks, kVggtBlocks * protocol::vggt::kBlockReferences, weights.tail,
        protocol::vggt::kTailReferences, reinterpret_cast<const uint16_t *>(patches),
        bundle.frames * patch_rows * patch_columns * protocol::vggt::kPatchInputs, grid_positions.data(),
        patch_rows * patch_columns * kHidden, special, kSpecialTokens * kHidden, static_cast<uint32_t>(bundle.frames),
        static_cast<uint32_t>(patch_rows), static_cast<uint32_t>(patch_columns), reinterpret_cast<uint16_t *>(kept),
        static_cast<int>(kept_values), statistics, transformer_stat::kCount);
    const double end = now_microseconds();
    if (!rpc_succeeded(error, "vggt_transformer")) return false;
    printf("%s %d frames %dx%d (%d tokens): transformer %.0f ms (call %.0f) | ", name, bundle.frames, bundle.height,
           bundle.width, tokens, statistics[transformer_stat::kTotalMicroseconds] / 1e3, (end - start) / 1e3);
    print_block_stages(statistics);
    printf(" | embedding %.1f dinov2 output %.1f kept outputs %.1f workspace %.1f\n",
           statistics[transformer_stat::kEmbeddingMicroseconds] / 1e3,
           statistics[transformer_stat::kDinov2OutputMicroseconds] / 1e3,
           statistics[transformer_stat::kKeptOutputsMicroseconds] / 1e3,
           statistics[transformer_stat::kWorkspaceMicroseconds] / 1e3);
    return true;
}

bool run_vggt_camera_head(const VggtWeights & weights, int frames, std::vector<float> * pose) {
    pose->assign(static_cast<size_t>(protocol::vggt::kCameraIterations) * frames * protocol::vggt::kPoseValues, 0.0f);
    int32_t statistics[protocol::vggt::camera_stat::kCount] = {};
    const double start = now_microseconds();
    const int error =
        benzene_vggt_camera_head(session(), weights.camera, protocol::vggt::kCameraReferences, pose->data(),
                                 static_cast<int>(pose->size()), statistics, protocol::vggt::camera_stat::kCount);
    const double end = now_microseconds();
    if (!rpc_succeeded(error, "vggt_camera_head")) return false;
    printf("camera head %.0f ms (call %.0f)\n", statistics[protocol::vggt::camera_stat::kMicroseconds] / 1e3,
           (end - start) / 1e3);
    return true;
}

bool run_vggt_dpt_head(const char * head, const int32_t * references, int channels, const std::vector<float> & uv,
                       float * out, size_t values) {
    int32_t s[dpt_stat::kCount] = {};
    const double start = now_microseconds();
    const int error = benzene_vggt_dpt_head(session(), references, protocol::vggt::kDptReferences,
                                            static_cast<uint32_t>(channels), uv.data(), static_cast<int>(uv.size()),
                                            out, static_cast<int>(values), s, dpt_stat::kCount);
    const double end = now_microseconds();
    if (!rpc_succeeded(error, "vggt_dpt_head (%s)", head)) return false;
    printf(
        "%s head %.0f ms (call %.0f) | projects+resize %.1f layer_rn %.1f refinenets %.1f output convs %.1f "
        "elementwise %.1f workspace %.1f\n",
        head, s[dpt_stat::kTotalMicroseconds] / 1e3, (end - start) / 1e3, s[dpt_stat::kProjectsMicroseconds] / 1e3,
        s[dpt_stat::kLayerRnMicroseconds] / 1e3, s[dpt_stat::kRefineNetsMicroseconds] / 1e3,
        s[dpt_stat::kOutputConvolutionsMicroseconds] / 1e3, s[dpt_stat::kElementwiseMicroseconds] / 1e3,
        s[dpt_stat::kWorkspaceMicroseconds] / 1e3);
    return true;
}

}  // namespace benzene::host
