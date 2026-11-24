#include <cstdio>
#include <cstring>
#include <string>
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

namespace superpoint_stat = protocol::superpoint::stat;
namespace xfeat_stat = protocol::xfeat::stat;
namespace aliked_stat = protocol::aliked::stat;

const char * const kSuperPointTensors[protocol::superpoint::convolution::kCount] = {
    "sp.conv1a", "sp.conv1b", "sp.conv2a",   "sp.conv2b", "sp.conv3a", "sp.conv3b",
    "sp.conv4a", "sp.conv4b", "sp.convPaDa", "sp.convPb", "sp.convDb",
};
const char * const kXFeatTensors[protocol::xfeat::tensor::kCount] = {
    "xfeat.block1.0",   "xfeat.block1.1",      "xfeat.block1.2",   "xfeat.block1.3",   "xfeat.block2.0",
    "xfeat.block2.1",   "xfeat.block3.0",      "xfeat.block3.1",   "xfeat.block3.2",   "xfeat.block4.0",
    "xfeat.block4.1",   "xfeat.block4.2",      "xfeat.block5.0",   "xfeat.block5.1",   "xfeat.block5.2",
    "xfeat.block5.3",   "xfeat.fusion.0",      "xfeat.fusion.1",   "xfeat.fusion.2",   "xfeat.heatmap.0",
    "xfeat.heatmap.1",  "xfeat.heatmap.2",     "xfeat.keypoint.0", "xfeat.keypoint.1", "xfeat.keypoint.2",
    "xfeat.keypoint.3", "xfeat.fusion_scales",
};
const char * const kAlikedTensors[protocol::aliked::tensor::kCount] = {
    "aliked.block1.conv1",
    "aliked.block1.conv2",
    "aliked.block1.conv2.aux",
    "aliked.block2.conv1",
    "aliked.block2.conv1.aux",
    "aliked.block2.conv2",
    "aliked.block2.conv2.aux",
    "aliked.block3.conv1.offset",
    "aliked.block3.conv1.offset.aux",
    "aliked.block3.conv1",
    "aliked.block3.conv1.aux",
    "aliked.block3.conv2.offset",
    "aliked.block3.conv2.offset.aux",
    "aliked.block3.conv2",
    "aliked.block3.conv2.aux",
    "aliked.block4.conv1.offset",
    "aliked.block4.conv1.offset.aux",
    "aliked.block4.conv1",
    "aliked.block4.conv1.aux",
    "aliked.block4.conv2.offset",
    "aliked.block4.conv2.offset.aux",
    "aliked.block4.conv2",
    "aliked.block4.conv2.aux",
    "aliked.conv1",
    "aliked.conv1.aux",
    "aliked.conv2",
    "aliked.conv2.aux",
    "aliked.conv3",
    "aliked.conv3.aux",
    "aliked.conv4",
    "aliked.conv4.aux",
    "aliked.score_head.0.1",
    "aliked.score_head.0.2",
    "aliked.score_head.0.3",
    "aliked.score_head.0.4",
    "aliked.score_head.2",
    "aliked.score_head.4",
    "aliked.score_head.6",
    "aliked.desc.offset0",
    "aliked.desc.offset0.aux",
    "aliked.desc.offset2",
    "aliked.desc.sf",
    "aliked.desc.sf.aux",
    "aliked.desc.aggregation",
    "aliked.stages",
};

using ExtractCall = int (*)(remote_handle64, const int32_t *, int, const uint8_t *, int, uint32_t, uint32_t, uint32_t,
                            float, uint32_t, float *, int, float *, int, uint16_t *, int, int32_t *, int);
using PrintStatistics = void (*)(int pair, int image, int height, int width, const int32_t * statistics,
                                 double call_milliseconds);

double superpoint_microseconds(const int32_t * s) {
    double total = 0;
    for (int q = superpoint_stat::kFirstConvolutionMicroseconds; q <= superpoint_stat::kDescriptorsMicroseconds; q++) {
        total += s[q];
    }
    return total;
}

double xfeat_microseconds(const int32_t * s) {
    double total = 0;
    for (int q = xfeat_stat::kBlock1Microseconds; q <= xfeat_stat::kDescriptorsMicroseconds; q++) total += s[q];
    return total;
}

void print_superpoint(int pair, int image, int height, int width, const int32_t * s, double call_milliseconds) {
    double convolutions = 0;
    for (int i = superpoint_stat::kFirstConvolutionMicroseconds; i <= superpoint_stat::kHeadsMicroseconds; i++) {
        convolutions += s[i];
    }

    const auto ms = [&](int statistic) { return s[statistic] / 1e3; };
    const auto conv = [&](int layer) { return ms(superpoint_stat::kFirstConvolutionMicroseconds + layer); };
    printf(
        "pair %d image %d %dx%d: superpoint %.2f ms (call %.2f) | %d keypoints of %d | conv %.2f (1a %.2f 1b %.2f 2a "
        "%.2f 2b %.2f 3a %.2f 3b %.2f 4a %.2f 4b %.2f heads %.2f) softmax %.2f nms %.2f kp %.2f desc %.2f\n",
        pair, image, height, width, superpoint_microseconds(s) / 1e3, call_milliseconds, s[superpoint_stat::kKeypoints],
        s[superpoint_stat::kCandidates], convolutions / 1e3, conv(0), conv(1), conv(2), conv(3), conv(4), conv(5),
        conv(6), conv(7), ms(superpoint_stat::kHeadsMicroseconds), ms(superpoint_stat::kSoftmaxMicroseconds),
        ms(superpoint_stat::kNmsMicroseconds), ms(superpoint_stat::kKeypointSelectionMicroseconds),
        ms(superpoint_stat::kDescriptorsMicroseconds));
}

void print_xfeat(int pair, int image, int height, int width, const int32_t * s, double call_milliseconds) {
    const auto ms = [&](int statistic) { return s[statistic] / 1e3; };
    printf(
        "pair %d image %d %dx%d: xfeat %.2f ms (call %.2f) | %d keypoints of %d | block1 %.2f backbone %.2f fusion "
        "%.2f heads %.2f heatmap %.2f (reliability %.2f) nms %.2f kp %.2f desc %.2f (normalize %.2f)\n",
        pair, image, height, width, xfeat_microseconds(s) / 1e3, call_milliseconds, s[xfeat_stat::kKeypoints],
        s[xfeat_stat::kCandidates], ms(xfeat_stat::kBlock1Microseconds), ms(xfeat_stat::kBackboneMicroseconds),
        ms(xfeat_stat::kFusionMicroseconds), ms(xfeat_stat::kHeadsMicroseconds), ms(xfeat_stat::kHeatmapMicroseconds),
        ms(xfeat_stat::kReliabilityMicroseconds), ms(xfeat_stat::kNmsMicroseconds),
        ms(xfeat_stat::kKeypointSelectionMicroseconds), ms(xfeat_stat::kDescriptorsMicroseconds),
        ms(xfeat_stat::kCellNormalizationMicroseconds));
}

double aliked_microseconds(const int32_t * s) {
    double total = 0;
    for (int q = aliked_stat::kBlock1Microseconds; q <= aliked_stat::kDescriptorsMicroseconds; q++) total += s[q];
    return total;
}

void print_aliked(int pair, int image, int height, int width, const int32_t * s, double call_milliseconds) {
    const auto ms = [&](int statistic) { return s[statistic] / 1e3; };
    printf(
        "pair %d image %d %dx%d: aliked %.2f ms (call %.2f) | %d keypoints of %d | block1 %.2f (conv1 %.2f conv2 %.2f "
        "agg1 %.2f part1 %.2f) block2 %.2f (conv1 %.2f conv2 %.2f) blocks3-4 %.2f (block3 %.2f sampling %.2f) score "
        "head %.2f (upsampling %.2f) nms %.2f kp %.2f desc %.2f (patches %.2f samples %.2f convolutions %.2f)\n",
        pair, image, height, width, aliked_microseconds(s) / 1e3, call_milliseconds, s[aliked_stat::kKeypoints],
        s[aliked_stat::kCandidates], ms(aliked_stat::kBlock1Microseconds), ms(aliked_stat::kBlock1Conv1Microseconds),
        ms(aliked_stat::kBlock1Conv2Microseconds), ms(aliked_stat::kConv1Microseconds),
        ms(aliked_stat::kPart1Microseconds), ms(aliked_stat::kBlock2Microseconds),
        ms(aliked_stat::kBlock2Conv1Microseconds), ms(aliked_stat::kBlock2Conv2Microseconds),
        ms(aliked_stat::kDeformableMicroseconds), ms(aliked_stat::kBlock3Microseconds),
        ms(aliked_stat::kDeformableSamplingMicroseconds), ms(aliked_stat::kScoreHeadMicroseconds),
        ms(aliked_stat::kScoreUpsamplingMicroseconds), ms(aliked_stat::kNmsMicroseconds),
        ms(aliked_stat::kKeypointSelectionMicroseconds), ms(aliked_stat::kDescriptorsMicroseconds),
        ms(aliked_stat::kDescriptorPatchesMicroseconds), ms(aliked_stat::kDescriptorSamplesMicroseconds),
        ms(aliked_stat::kDescriptorConvolutionsMicroseconds));
}

struct Extractor {
    const char * name;
    const char * output_extension;
    protocol::lightglue::Model model;
    const char * const * tensors;  // the model file's tensors in the protocol's reference order
    int tensor_count;
    int statistics;
    int keypoints_statistic;
    int descriptor_dimensions;
    float default_threshold;
    float adaptive_depth_confidence;
    float adaptive_width_confidence;
    ExtractCall extract;
    PrintStatistics print;
    bool writes_normalization;  // the image normalization (xfeat::stat) into <pair>.<output_extension>n<image>
};

const Extractor kSuperPoint{
    .name = "superpoint",
    .output_extension = "sp",
    .model = protocol::lightglue::Model::kSuperPoint,
    .tensors = kSuperPointTensors,
    .tensor_count = protocol::superpoint::convolution::kCount,
    .statistics = superpoint_stat::kCount,
    .keypoints_statistic = superpoint_stat::kKeypoints,
    .descriptor_dimensions = protocol::superpoint::kDescriptorDimensions,
    .default_threshold = 0.0005f,
    .adaptive_depth_confidence = 0.95f,
    .adaptive_width_confidence = 0.99f,
    .extract = benzene_superpoint,
    .print = print_superpoint,
    .writes_normalization = false,
};

const Extractor kXFeat{
    .name = "xfeat",
    .output_extension = "xf",
    .model = protocol::lightglue::Model::kXFeat,
    .tensors = kXFeatTensors,
    .tensor_count = protocol::xfeat::tensor::kCount,
    .statistics = xfeat_stat::kCount,
    .keypoints_statistic = xfeat_stat::kKeypoints,
    .descriptor_dimensions = protocol::xfeat::kDescriptorDimensions,
    .default_threshold = 0.05f,
    .adaptive_depth_confidence = -1.0f,
    .adaptive_width_confidence = 0.95f,
    .extract = benzene_xfeat,
    .print = print_xfeat,
    .writes_normalization = true,
};

const Extractor kAliked{
    .name = "aliked",
    .output_extension = "al",
    .model = protocol::lightglue::Model::kAliked,
    .tensors = kAlikedTensors,
    .tensor_count = protocol::aliked::tensor::kCount,
    .statistics = aliked_stat::kCount,
    .keypoints_statistic = aliked_stat::kKeypoints,
    .descriptor_dimensions = protocol::aliked::kDescriptorDimensions,
    .default_threshold = 0.2f,
    .adaptive_depth_confidence = 0.95f,
    .adaptive_width_confidence = 0.99f,
    .extract = benzene_aliked,
    .print = print_aliked,
    .writes_normalization = false,
};

constexpr int kModelSlot = 0;
constexpr int kMaximumKeypoints = 1024;
constexpr float kMatchThreshold = 0.1f;  // LightGlue's filter_threshold

// rpcmem buffers of the calls: the image, per feature slot its keypoints, scores and descriptors, the matches.
struct Buffers {
    uint8_t * image;
    float * keypoints[protocol::kFeatureSlots];
    float * scores[protocol::kFeatureSlots];
    _Float16 * descriptors[protocol::kFeatureSlots];
    int32_t * matches;
    float * match_scores;
};

// The extractor's call on one image into feature slot `slot`, its line and features file; the keypoint count, or -1.
int extract(const Extractor & extractor, const std::vector<int32_t> & references, const Buffers & buffers,
            int maximum_keypoints, float threshold, const uint8_t * image, int height, int width, int slot, int pair,
            const char * directory) {
    memcpy(buffers.image, image, static_cast<size_t>(height) * width);
    std::vector<int32_t> statistics(static_cast<size_t>(extractor.statistics));
    const double start = now_microseconds();
    const int error = extractor.extract(
        session(), references.data(), static_cast<int>(references.size()), buffers.image, height * width,
        static_cast<uint32_t>(height), static_cast<uint32_t>(width), static_cast<uint32_t>(maximum_keypoints),
        threshold, static_cast<uint32_t>(slot), buffers.keypoints[slot], 2 * maximum_keypoints, buffers.scores[slot],
        maximum_keypoints, reinterpret_cast<uint16_t *>(buffers.descriptors[slot]),
        extractor.descriptor_dimensions * maximum_keypoints, statistics.data(), extractor.statistics);
    const double end = now_microseconds();
    if (!rpc_succeeded(error, "%s (pair %d image %d)", extractor.name, pair, slot)) return -1;

    extractor.print(pair, slot, height, width, statistics.data(), (end - start) / 1e3);
    const int keypoints = statistics[static_cast<size_t>(extractor.keypoints_statistic)];
    if (!write_features(directory, extractor.output_extension, pair, slot, keypoints, buffers.keypoints[slot],
                        buffers.scores[slot], buffers.descriptors[slot],
                        static_cast<size_t>(2 * extractor.descriptor_dimensions))) {
        return -1;
    }
    if (extractor.writes_normalization) {
        const int32_t normalization[2] = {statistics[xfeat_stat::kImageMeanBits],
                                          statistics[xfeat_stat::kImageInverseStdBits]};
        const std::string name = std::to_string(pair) + "." + extractor.output_extension + "n" + std::to_string(slot);
        if (!write_file(directory, name.c_str(), normalization, sizeof normalization)) return -1;
    }
    return keypoints;
}

bool plan_references(const Extractor & extractor, const ModelTable & table, uint64_t base,
                     std::vector<int32_t> * references, LightGlueSettings * lightglue) {
    references->assign(static_cast<size_t>(1 + extractor.tensor_count), 0);
    (*references)[0] = kModelSlot;
    for (int i = 0; i < extractor.tensor_count; i++) {
        const TensorRecord * record = table.find(extractor.tensors[i]);
        if (record == nullptr) {
            fprintf(stderr, "missing tensor %s\n", extractor.tensors[i]);
            return false;
        }
        (*references)[static_cast<size_t>(1 + i)] = static_cast<int32_t>(record->offset - base);
    }
    return plan_lightglue_references(table, kModelSlot, base, extractor.model, &lightglue->references);
}

bool allocate_buffers(size_t image_bytes, int maximum_keypoints, int descriptor_dimensions, Buffers * buffers) {
    const auto keypoints = static_cast<uint64_t>(maximum_keypoints);
    buffers->image = static_cast<uint8_t *>(allocate_rpc_memory(image_bytes));
    bool allocated = buffers->image != nullptr;
    for (int slot = 0; slot < protocol::kFeatureSlots; slot++) {
        buffers->keypoints[slot] = static_cast<float *>(allocate_rpc_memory(keypoints * 8));
        buffers->scores[slot] = static_cast<float *>(allocate_rpc_memory(keypoints * 4));
        buffers->descriptors[slot] =
            static_cast<_Float16 *>(allocate_rpc_memory(keypoints * 2 * descriptor_dimensions));
        allocated = allocated && buffers->keypoints[slot] != nullptr && buffers->scores[slot] != nullptr &&
                    buffers->descriptors[slot] != nullptr;
    }
    buffers->match_scores = static_cast<float *>(allocate_rpc_memory(keypoints * 4));
    buffers->matches = static_cast<int32_t *>(allocate_rpc_memory(keypoints * 4));
    return allocated && buffers->match_scores != nullptr && buffers->matches != nullptr;
}

int run_feature_matcher(const Extractor & extractor, int argc, char ** argv) {
    bool adaptive = false;
    const Option options[] = {{"--adaptive", nullptr, &adaptive}};
    if (!parse_command_line(argc, argv, 3, options, 1, "model.hexmodel bundle_dir out_dir [--adaptive]")) return 1;
    const char * model = argv[2];
    const char * output = argv[4];
    LightGlueSettings lightglue{.model = extractor.model,
                                .references = {},
                                .threshold = kMatchThreshold,
                                .depth_confidence = adaptive ? extractor.adaptive_depth_confidence : -1.0f,
                                .width_confidence = adaptive ? extractor.adaptive_width_confidence : -1.0f};

    ModelTable table;
    if (!load_table_of_model(model, &table) || !make_directory(output)) return 1;
    uint64_t begin = 0;
    uint64_t end = 0;
    table.tensor_range(&begin, &end);
    std::vector<int32_t> references;
    if (!plan_references(extractor, table, begin, &references, &lightglue)) return 1;
    PairBundle bundle;
    if (!read_pair_bundle(argv[3], 2, &bundle)) return 1;

    const double load_start = now_microseconds();
    if (!load_into_slot(model, begin, end - begin, kModelSlot)) return 1;
    print_model_load(end - begin, 1, now_microseconds() - load_start);

    size_t largest = 0;
    for (int image = 0; image < 2 * bundle.pairs; image++) {
        const int * size = bundle.image_sizes(image / 2, image % 2);
        const size_t pixels = static_cast<size_t>(size[0]) * size[1];
        largest = pixels > largest ? pixels : largest;
    }
    Buffers buffers{};
    if (!allocate_buffers(largest, kMaximumKeypoints, extractor.descriptor_dimensions, &buffers)) return 1;
    printf("%s: %d pairs, at most %d keypoints, threshold %g%s\n", extractor.name, bundle.pairs, kMaximumKeypoints,
           extractor.default_threshold, adaptive ? ", adaptive LightGlue" : "");

    for (int pair = 0; pair < bundle.pairs; pair++) {
        if (stopped_by_signal()) return 1;
        int keypoints[2];
        for (int slot = 0; slot < 2; slot++) {
            const int * size = bundle.image_sizes(pair, slot);
            keypoints[slot] = extract(extractor, references, buffers, kMaximumKeypoints, extractor.default_threshold,
                                      bundle.image(pair, slot), size[0], size[1], slot, pair, output);
            if (keypoints[slot] < 0) return 1;
        }
        if (!match_pair(lightglue, keypoints, pair, buffers.matches, buffers.match_scores, output)) return 1;
    }
    return 0;
}

}  // namespace

int run_superpoint_lightglue(int argc, char ** argv) {
    return run_feature_matcher(kSuperPoint, argc, argv);
}

int run_xfeat_lightglue(int argc, char ** argv) {
    return run_feature_matcher(kXFeat, argc, argv);
}

int run_aliked_lightglue(int argc, char ** argv) {
    return run_feature_matcher(kAliked, argc, argv);
}

}  // namespace benzene::host
