#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "host/command_line.h"
#include "host/commands.h"
#include "host/fastrpc.h"
#include "host/feature_matching.h"
#include "host/files.h"
#include "host/model_file.h"
#include "host/timing.h"
#include "benzene/protocol.h"

namespace benzene::host {
namespace {

namespace loma = protocol::loma;
namespace loma_stat = protocol::loma::stat;
using protocol::loma::kDinov2Hidden;
using protocol::loma::kPatchSize;

constexpr int kModelSlot = 0;
constexpr float kMatchThreshold = 0.1f;  // LoMa.Cfg.filter_threshold
constexpr int kPositionGrid = 37;
constexpr double kCubicA = -0.75;

std::string loma_tensor_name(int index) {
    const auto refiner_name = [](const char * prefix, int scale, int part, int hidden_blocks, bool with_context,
                                 const char * deltas) {
        std::string base = std::string(prefix) + ".dec" + std::to_string(scale) + ".";
        if (part == loma::refiner_part::kBlock1Conv1) return base + "block1.conv1";
        if (part == loma::refiner_part::kBlock1Conv2) return base + "block1.conv2";
        if (part < loma::refiner_part::context(hidden_blocks)) {
            const int block = (part - 2) / 2;
            return base + "hidden" + std::to_string(block) + ((part - 2) % 2 == 0 ? ".dw" : ".pw");
        }
        if (with_context && part == loma::refiner_part::context(hidden_blocks)) return base + "out.context";
        return base + "out." + deltas;
    };
    const int detector_scales[loma::kDetectorScales] = {8, 4, 2, 1};
    const int descriptor_scales[loma::kDescriptorScales] = {14, 8, 4, 2, 1};

    if (index < loma::detector_refiner(0)) return "det.enc." + std::to_string(index);
    for (int s = 0; s < loma::kDetectorScales; s++) {
        const bool context = s + 1 < loma::kDetectorScales;
        const int first = loma::detector_refiner(s);
        if (index >= first && index < first + loma::refiner_tensors(loma::kDetectorHiddenBlocks, context)) {
            return refiner_name("det", detector_scales[s], index - first, loma::kDetectorHiddenBlocks, context,
                                "logit");
        }
    }
    if (index < loma::kDetectorTensors) {
        const int s = index - loma::detector_context_quantizer(1) + 1;
        return "det.dec" + std::to_string(detector_scales[s]) + ".context.quantizer";
    }

    if (index < loma::descriptor_refiner(0)) return "desc.enc." + std::to_string(index - loma::descriptor_encoder(0));
    for (int s = 0; s < loma::kDescriptorScales; s++) {
        const bool context = s + 1 < loma::kDescriptorScales;
        const int first = loma::descriptor_refiner(s);
        if (index >= first && index < first + loma::refiner_tensors(loma::kDescriptorHiddenBlocks, context)) {
            return refiner_name("desc", descriptor_scales[s], index - first, loma::kDescriptorHiddenBlocks, context,
                                "deltas");
        }
    }
    if (index < loma::kDinov2OutputQuantizer) {
        const int s = index - loma::descriptor_context_quantizer(1) + 1;
        return "desc.dec" + std::to_string(descriptor_scales[s]) + ".context.quantizer";
    }

    if (index == loma::kDinov2OutputQuantizer) return "desc.dino.quantizer";
    if (index < loma::kDinov2Tail) return "desc.dinov2." + std::to_string(index - loma::dinov2_block(0));
    return "desc.dinov2.tail";
}

bool plan_loma_references(const ModelTable & table, uint64_t base, std::vector<int32_t> * references) {
    references->assign(static_cast<size_t>(loma::kReferences), 0);
    for (int t = 0; t < loma::kTensors; t++) {
        const std::string name = loma_tensor_name(t);
        const TensorRecord * record = table.find(name.c_str());
        if (record == nullptr) {
            fprintf(stderr, "missing tensor %s\n", name.c_str());
            return false;
        }
        if (record->part_count + 2 > static_cast<uint32_t>(loma::kReferenceStride)) {
            fprintf(stderr, "tensor %s has %u parts (at most %d)\n", name.c_str(), record->part_count,
                    loma::kReferenceStride - 2);
            return false;
        }

        int32_t * r = references->data() + static_cast<size_t>(t) * loma::kReferenceStride;
        r[0] = kModelSlot;
        r[1] = static_cast<int32_t>(record->offset - base);
        for (uint32_t p = 0; p < record->part_count; p++) r[2 + p] = static_cast<int32_t>(record->parts[p]);
    }
    return true;
}

// PyTorch's bicubic coefficients of the taps i0 - 1 .. i0 + 2 at fraction t.
void cubic_coefficients(double t, double * c) {
    const auto near = [](double x) { return ((kCubicA + 2.0) * x - (kCubicA + 3.0)) * x * x + 1.0; };
    const auto far = [](double x) { return ((kCubicA * x - 5.0 * kCubicA) * x + 8.0 * kCubicA) * x - 4.0 * kCubicA; };
    c[0] = far(t + 1.0);
    c[1] = near(t);
    c[2] = near(1.0 - t);
    c[3] = far(2.0 - t);
}

// The taps (clamped to the grid) and coefficients of interpolate_pos_encoding's resize: its scale_factor is
// (outputs + 0.1) / grid, and PyTorch maps the coordinates with 1 / scale_factor.
void position_taps(int outputs, int * taps, double * coefficients) {
    const double scale = 1.0 / ((outputs + 0.1) / kPositionGrid);
    for (int o = 0; o < outputs; o++) {
        const double source = scale * (o + 0.5) - 0.5;
        const double first = std::floor(source);
        cubic_coefficients(source - first, coefficients + 4 * o);
        for (int k = 0; k < 4; k++) {
            int tap = static_cast<int>(first) - 1 + k;
            tap = tap < 0 ? 0 : tap > kPositionGrid - 1 ? kPositionGrid - 1 : tap;
            taps[4 * o + k] = tap;
        }
    }
}

// As network.DinoVisionTransformer.position_embedding: grid [37 * 37][1024] (without the class token's entry) resized
// bicubically, without antialiasing, to out [rows * columns][1024].
void dinov2_positions(const float * grid, int rows, int columns, float * out) {
    if (rows == kPositionGrid && columns == kPositionGrid) {
        memcpy(out, grid, sizeof(float) * kPositionGrid * kPositionGrid * kDinov2Hidden);
        return;
    }

    std::vector<int> row_taps(static_cast<size_t>(4) * rows);
    std::vector<int> column_taps(static_cast<size_t>(4) * columns);
    std::vector<double> row_coefficients(static_cast<size_t>(4) * rows);
    std::vector<double> column_coefficients(static_cast<size_t>(4) * columns);
    position_taps(rows, row_taps.data(), row_coefficients.data());
    position_taps(columns, column_taps.data(), column_coefficients.data());

    std::vector<double> resized_columns(static_cast<size_t>(kPositionGrid) * columns * kDinov2Hidden);
    for (int i = 0; i < kPositionGrid; i++) {
        for (int x = 0; x < columns; x++) {
            double * destination = resized_columns.data() + (static_cast<size_t>(i) * columns + x) * kDinov2Hidden;
            for (int k = 0; k < 4; k++) {
                const double c = column_coefficients[4 * x + k];
                const float * source =
                    grid + (static_cast<size_t>(i) * kPositionGrid + column_taps[4 * x + k]) * kDinov2Hidden;
                for (int d = 0; d < kDinov2Hidden; d++) destination[d] += c * source[d];
            }
        }
    }

    std::vector<double> total(kDinov2Hidden);
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < columns; x++) {
            std::fill(total.begin(), total.end(), 0.0);
            for (int k = 0; k < 4; k++) {
                const double c = row_coefficients[4 * y + k];
                const double * source =
                    resized_columns.data() + (static_cast<size_t>(row_taps[4 * y + k]) * columns + x) * kDinov2Hidden;
                for (int d = 0; d < kDinov2Hidden; d++) total[d] += c * source[d];
            }
            float * destination = out + (static_cast<size_t>(y) * columns + x) * kDinov2Hidden;
            for (int d = 0; d < kDinov2Hidden; d++) destination[d] = static_cast<float>(total[d]);
        }
    }
}

void dinov2_patches(const uint8_t * gray, int height, int width, _Float16 * patches) {
    const int rows = height / kPatchSize;
    const int columns = width / kPatchSize;

    for (int py = 0; py < rows; py++) {
        for (int px = 0; px < columns; px++) {
            _Float16 * destination = patches + (static_cast<size_t>(py) * columns + px) * loma::kDinov2PatchInputs;
            for (int ky = 0; ky < kPatchSize; ky++) {
                const uint8_t * source = gray + static_cast<size_t>(py * kPatchSize + ky) * width + px * kPatchSize;
                for (int kx = 0; kx < kPatchSize; kx++) {
                    destination[ky * kPatchSize + kx] = static_cast<_Float16>(static_cast<float>(source[kx]) / 255.0f);
                }
            }
            for (int k = kPatchSize * kPatchSize; k < loma::kDinov2PatchInputs; k++) {
                destination[k] = static_cast<_Float16>(0.0f);
            }
        }
    }
}

void print_loma(int pair, int image, const int * sizes, const int32_t * s, double call_milliseconds) {
    const auto ms = [&](int index) { return s[index] / 1e3; };
    printf(
        "pair %d image %d %dx%d: loma %.2f ms (call %.2f) | %d keypoints | detector: encoder %.2f block1 %.2f hidden "
        "%.2f (depthwise %.2f) out %.2f contexts + logits %.2f | keypoints: softmax + NMS %.2f selection %.2f "
        "positions %.2f\n",
        pair, image, sizes[0], sizes[1], ms(loma_stat::kTotalMicroseconds), call_milliseconds, s[loma_stat::kKeypoints],
        ms(loma_stat::kDetectorEncoderMicroseconds), ms(loma_stat::kDetectorBlock1Microseconds),
        ms(loma_stat::kDetectorHiddenMicroseconds), ms(loma_stat::kDetectorDepthwiseMicroseconds),
        ms(loma_stat::kDetectorOutMicroseconds), ms(loma_stat::kDetectorContextMicroseconds),
        ms(loma_stat::kSoftmaxMicroseconds), ms(loma_stat::kSelectionMicroseconds),
        ms(loma_stat::kPositionMicroseconds));
    printf("  descriptor %dx%d: encoder %.2f DINOv2 %.2f (", sizes[2], sizes[3],
           ms(loma_stat::kDescriptorEncoderMicroseconds), ms(loma_stat::kDinov2Microseconds));
    print_block_stages(s + loma_stat::kDinov2BlockMicroseconds);
    printf(") block1 %.2f hidden %.2f (depthwise %.2f) out %.2f contexts + sums %.2f sparse %.2f\n",
           ms(loma_stat::kDescriptorBlock1Microseconds), ms(loma_stat::kDescriptorHiddenMicroseconds),
           ms(loma_stat::kDescriptorDepthwiseMicroseconds), ms(loma_stat::kDescriptorOutMicroseconds),
           ms(loma_stat::kDescriptorContextMicroseconds), ms(loma_stat::kSparseMicroseconds));
}

// rpcmem buffers of the calls.
struct Buffers {
    uint8_t * detector_image;
    uint8_t * descriptor_image;
    _Float16 * patches;
    float * positions;
    float * keypoints;
    float * scores;
    float * descriptors;
    int32_t * matches;
    float * match_scores;
};

bool allocate_buffers(const PairBundle & bundle, int keypoints, Buffers * buffers) {
    size_t largest_detector = 0;
    size_t largest_descriptor = 0;
    for (int image = 0; image < 2 * bundle.pairs; image++) {
        const int * sizes = bundle.image_sizes(image / 2, image % 2);
        const size_t detector_pixels = static_cast<size_t>(sizes[0]) * sizes[1];
        const size_t descriptor_pixels = static_cast<size_t>(sizes[2]) * sizes[3];
        largest_detector = detector_pixels > largest_detector ? detector_pixels : largest_detector;
        largest_descriptor = descriptor_pixels > largest_descriptor ? descriptor_pixels : largest_descriptor;
    }

    const size_t largest_patches = largest_descriptor / (kPatchSize * kPatchSize);
    const auto count = static_cast<uint64_t>(keypoints);
    buffers->detector_image = static_cast<uint8_t *>(allocate_rpc_memory(largest_detector));
    buffers->descriptor_image = static_cast<uint8_t *>(allocate_rpc_memory(largest_descriptor));
    buffers->patches = static_cast<_Float16 *>(allocate_rpc_memory(largest_patches * loma::kDinov2PatchInputs * 2));
    buffers->positions = static_cast<float *>(allocate_rpc_memory(largest_patches * kDinov2Hidden * 4));
    buffers->keypoints = static_cast<float *>(allocate_rpc_memory(count * 8));
    buffers->scores = static_cast<float *>(allocate_rpc_memory(count * 4));
    buffers->descriptors = static_cast<float *>(allocate_rpc_memory(count * loma::kDescriptorDimensions * 4));
    buffers->matches = static_cast<int32_t *>(allocate_rpc_memory(count * 4));
    buffers->match_scores = static_cast<float *>(allocate_rpc_memory(count * 4));
    return buffers->detector_image != nullptr && buffers->descriptor_image != nullptr && buffers->patches != nullptr &&
           buffers->positions != nullptr && buffers->keypoints != nullptr && buffers->scores != nullptr &&
           buffers->descriptors != nullptr && buffers->matches != nullptr && buffers->match_scores != nullptr;
}

struct Detection {
    uint32_t zero_point;
    int keypoints;
};

// The features of one image into feature slot `slot`: the call, its lines and the features file; the keypoint count,
// or -1.
int describe(const std::vector<int32_t> & references, const Detection & detection, const Buffers & buffers,
             const PairBundle & bundle, int pair, int slot, const float * resized_positions, const char * directory) {
    const int * sizes = bundle.image_sizes(pair, slot);
    const size_t detector_pixels = static_cast<size_t>(sizes[0]) * sizes[1];
    const size_t descriptor_pixels = static_cast<size_t>(sizes[2]) * sizes[3];
    const int patch_count = (sizes[2] / kPatchSize) * (sizes[3] / kPatchSize);
    memcpy(buffers.detector_image, bundle.image(pair, slot), detector_pixels);
    memcpy(buffers.descriptor_image, bundle.image(pair, slot) + detector_pixels, descriptor_pixels);
    memcpy(buffers.positions, resized_positions, static_cast<size_t>(patch_count) * kDinov2Hidden * 4);
    dinov2_patches(buffers.descriptor_image, sizes[2], sizes[3], buffers.patches);

    int32_t statistics[loma_stat::kCount] = {};
    const double start = now_microseconds();
    const int error = benzene_loma(
        session(), references.data(), static_cast<int>(references.size()), buffers.detector_image,
        static_cast<int>(detector_pixels), static_cast<uint32_t>(sizes[0]), static_cast<uint32_t>(sizes[1]),
        detection.zero_point, buffers.descriptor_image, static_cast<int>(descriptor_pixels),
        static_cast<uint32_t>(sizes[2]), static_cast<uint32_t>(sizes[3]),
        reinterpret_cast<const uint16_t *>(buffers.patches), patch_count * loma::kDinov2PatchInputs, buffers.positions,
        patch_count * kDinov2Hidden, static_cast<uint32_t>(detection.keypoints), static_cast<uint32_t>(slot),
        buffers.keypoints, 2 * detection.keypoints, buffers.scores, detection.keypoints, buffers.descriptors,
        detection.keypoints * loma::kDescriptorDimensions, statistics, loma_stat::kCount);
    const double end = now_microseconds();
    if (!rpc_succeeded(error, "loma (pair %d image %d)", pair, slot)) return -1;

    print_loma(pair, slot, sizes, statistics, (end - start) / 1e3);
    const int keypoints = statistics[loma_stat::kKeypoints];
    if (!write_features(directory, "lm", pair, slot, keypoints, buffers.keypoints, buffers.scores, buffers.descriptors,
                        loma::kDescriptorDimensions * 4)) {
        return -1;
    }
    return keypoints;
}

}  // namespace

int run_loma(int argc, char ** argv) {
    if (!parse_command_line(argc, argv, 3, nullptr, 0, "model.hexmodel bundle_dir out_dir")) return 1;
    const char * model = argv[2];
    const char * output = argv[4];

    ModelTable table;
    if (!load_table_of_model(model, &table) || !make_directory(output)) return 1;
    const TensorRecord * config_record = table.find("loma.config");
    const TensorRecord * position_record = table.find("desc.dinov2.positions");
    if (config_record == nullptr || position_record == nullptr) {
        fprintf(stderr, "missing tensor loma.config or desc.dinov2.positions\n");
        return 1;
    }
    const MallocBuffer<float> config = read_f32_tensor(model, *config_record);
    const MallocBuffer<float> position_grid = read_f32_tensor(model, *position_record);
    if (!config || !position_grid) return 1;
    const Detection detection{static_cast<uint32_t>(config[0]), static_cast<int>(config[2])};

    uint64_t begin = 0;
    uint64_t end = 0;
    table.tensor_range(&begin, &end);
    std::vector<int32_t> references;
    LightGlueSettings lightglue{protocol::lightglue::Model::kLoma, {}, kMatchThreshold, -1.0f, -1.0f};
    if (!plan_loma_references(table, begin, &references) ||
        !plan_lightglue_references(table, kModelSlot, begin, lightglue.model, &lightglue.references)) {
        return 1;
    }

    PairBundle bundle;
    if (!read_pair_bundle(argv[3], 4, &bundle)) return 1;
    for (int image = 0; image < 2 * bundle.pairs; image++) {
        const int * sizes = bundle.image_sizes(image / 2, image % 2);
        if (sizes[2] % kPatchSize != 0 || sizes[3] % kPatchSize != 0) {
            fprintf(stderr, "bad descriptor image size %dx%d (multiples of %d)\n", sizes[2], sizes[3], kPatchSize);
            return 1;
        }
    }

    const double load_start = now_microseconds();
    if (!load_into_slot(model, begin, end - begin, kModelSlot)) return 1;
    print_model_load(end - begin, 1, now_microseconds() - load_start);
    Buffers buffers{};
    if (!allocate_buffers(bundle, detection.keypoints, &buffers)) return 1;
    printf("loma: %d pairs, %d keypoints, detector zero point %u\n", bundle.pairs, detection.keypoints,
           detection.zero_point);

    std::map<std::pair<int, int>, std::vector<float>> resized_positions;
    for (int pair = 0; pair < bundle.pairs; pair++) {
        if (stopped_by_signal()) return 1;
        int keypoints[2];
        for (int slot = 0; slot < 2; slot++) {
            const int * sizes = bundle.image_sizes(pair, slot);
            const int rows = sizes[2] / kPatchSize;
            const int columns = sizes[3] / kPatchSize;
            std::vector<float> & resized = resized_positions[{rows, columns}];
            if (resized.empty()) {
                resized.resize(static_cast<size_t>(rows) * columns * kDinov2Hidden);
                dinov2_positions(position_grid.get(), rows, columns, resized.data());
            }
            keypoints[slot] = describe(references, detection, buffers, bundle, pair, slot, resized.data(), output);
            if (keypoints[slot] < 0) return 1;
        }
        if (!match_pair(lightglue, keypoints, pair, buffers.matches, buffers.match_scores, output)) return 1;
    }
    return 0;
}

}  // namespace benzene::host
