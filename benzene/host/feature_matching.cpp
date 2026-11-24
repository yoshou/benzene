#include "host/feature_matching.h"

#include <cstdio>
#include <string>

#include "host/fastrpc.h"
#include "host/timing.h"

namespace benzene::host {
namespace {

namespace lightglue = protocol::lightglue;
namespace lightglue_stat = protocol::lightglue::stat;

const char * const kLightGlueParts[lightglue::part::kCount] = {"qkv", "out", "ffn1", "ln", "ffn2"};

double lightglue_layer_microseconds(const int32_t * s) {
    double layers = 0;
    for (int layer = 0; layer < lightglue::kMaximumLayers; layer++) {
        layers += s[lightglue_stat::kFirstLayerMicroseconds + layer];
    }
    return layers;
}

double lightglue_microseconds(const int32_t * s) {
    return lightglue_layer_microseconds(s) + s[lightglue_stat::kSetupMicroseconds] +
           s[lightglue_stat::kFinalProjectionMicroseconds] + s[lightglue_stat::kAssignmentMicroseconds];
}

void print_lightglue(int pair, const int32_t * s, bool adaptive, double call_milliseconds) {
    const double layers = lightglue_layer_microseconds(s);
    const int layers_run = s[lightglue_stat::kLayersRun] > 0 ? s[lightglue_stat::kLayersRun] : 1;
    std::string tokens;
    for (int layer = 0; adaptive && layer < layers_run; layer++) {
        tokens += (layer != 0 ? " " : " tokens ") + std::to_string(s[lightglue_stat::kFirstLayerTokens + layer]);
    }

    const auto ms = [&](int statistic) { return s[statistic] / 1e3; };
    const auto per_layer = [&](int statistic) { return s[statistic] / 1e3 / layers_run; };
    printf(
        "pair %d: lightglue %.2f ms (call %.2f) | %d matches | setup %.2f layers %.2f (%d run, %.2f per layer: qkv "
        "%.2f attention %.2f out+FFN %.2f HVX %.2f%s) final %.2f assignment %.2f\n",
        pair, lightglue_microseconds(s) / 1e3, call_milliseconds, s[lightglue_stat::kMatches],
        ms(lightglue_stat::kSetupMicroseconds), layers / 1e3, layers_run, layers / 1e3 / layers_run,
        per_layer(lightglue_stat::kQkvGemmMicroseconds), per_layer(lightglue_stat::kAttentionMicroseconds),
        per_layer(lightglue_stat::kOutputAndFfnMicroseconds), per_layer(lightglue_stat::kHvxPassesMicroseconds),
        tokens.c_str(), ms(lightglue_stat::kFinalProjectionMicroseconds), ms(lightglue_stat::kAssignmentMicroseconds));
}

bool write_matches(const char * directory, int pair, int keypoints, const int32_t * matches, const float * scores) {
    const std::string path = join_path(directory, (std::to_string(pair) + ".lg").c_str());
    const File file = open_file(path.c_str(), "wb");
    const auto count = static_cast<size_t>(keypoints);
    const int32_t header = keypoints;
    const bool written = file && fwrite(&header, 4, 1, file.get()) == 1 &&
                         fwrite(matches, 4, count, file.get()) == count &&
                         fwrite(scores, 4, count, file.get()) == count;
    if (!written) fprintf(stderr, "write %s failed\n", path.c_str());
    return written;
}

}  // namespace

bool read_pair_bundle(const char * directory, int sizes_per_image, PairBundle * bundle) {
    const std::string meta_path = join_path(directory, "meta.txt");
    const File meta = open_file(meta_path.c_str(), "r");
    if (!meta || fscanf(meta.get(), "%d", &bundle->pairs) != 1 || bundle->pairs < 1) {
        fprintf(stderr, "bad %s\n", meta_path.c_str());
        return false;
    }

    bundle->sizes_per_image = sizes_per_image;
    bundle->sizes.resize(static_cast<size_t>(2 * bundle->pairs) * sizes_per_image);
    bundle->offsets.resize(static_cast<size_t>(2 * bundle->pairs));
    size_t total = 0;
    for (int image = 0; image < 2 * bundle->pairs; image++) {
        bundle->offsets[image] = total;
        for (int plane = 0; plane < sizes_per_image / 2; plane++) {
            int * size = bundle->sizes.data() + static_cast<size_t>(image) * sizes_per_image + 2 * plane;
            if (fscanf(meta.get(), "%d %d", &size[0], &size[1]) != 2 || size[0] < 1 || size[1] < 1) {
                fprintf(stderr, "bad %s\n", meta_path.c_str());
                return false;
            }
            total += static_cast<size_t>(size[0]) * size[1];
        }
    }

    const std::string images_path = join_path(directory, "images.u8");
    size_t image_bytes = 0;
    bundle->images = load_file(images_path.c_str(), &image_bytes);
    if (!bundle->images || image_bytes != total) {
        fprintf(stderr, "bad %s\n", images_path.c_str());
        return false;
    }
    return true;
}

bool plan_lightglue_references(const ModelTable & table, int slot, uint64_t base, lightglue::Model model,
                               std::vector<int32_t> * references) {
    const int layers = lightglue::layers(model);
    const bool token_confidences = lightglue::token_confidences(model);
    references->assign(static_cast<size_t>(lightglue::references(model)), 0);
    (*references)[0] = slot;
    const auto set = [&](int index, const char * name) {
        const TensorRecord * record = table.find(name);
        if (record == nullptr) {
            fprintf(stderr, "missing tensor %s\n", name);
            return false;
        }
        (*references)[static_cast<size_t>(index)] = static_cast<int32_t>(record->offset - base);
        return true;
    };

    for (int layer = 0; layer < layers; layer++) {
        for (int block = 0; block < 2; block++) {
            for (int part = 0; part < lightglue::part::kCount; part++) {
                char name[48];
                snprintf(name, sizeof name, "lg.%d.%s.%s", layer, block != 0 ? "cross" : "self", kLightGlueParts[part]);
                if (!set(lightglue::block_part_reference(layer, block, part), name)) return false;
            }
        }
    }

    static const char * const kHeads[3] = {"final", "match", "conf"};
    for (int layer = 0; layer < layers; layer++) {
        const int index[3] = {lightglue::final_projection_reference(layers, layer),
                              lightglue::matchability_reference(layers, layer),
                              lightglue::token_confidence_reference(layers, layer)};
        const int heads = layer + 1 < layers && token_confidences ? 3 : 2;
        for (int head = 0; head < heads; head++) {
            char name[48];
            snprintf(name, sizeof name, "lg.%d.%s", layer, kHeads[head]);
            if (!set(index[head], name)) return false;
        }
    }

    if (!set(lightglue::positional_encoding_reference(layers), "lg.posenc")) return false;
    return !lightglue::input_projection(model) || set(lightglue::input_projection_reference(layers), "lg.input");
}

bool write_features(const char * directory, const char * extension, int pair, int image, int keypoints,
                    const float * positions, const float * scores, const void * descriptors, size_t descriptor_bytes) {
    const std::string name = std::to_string(pair) + "." + extension + std::to_string(image);
    const std::string path = join_path(directory, name.c_str());
    const File file = open_file(path.c_str(), "wb");
    const auto count = static_cast<size_t>(keypoints);
    const int32_t header = keypoints;
    const bool written = file && fwrite(&header, 4, 1, file.get()) == 1 &&
                         fwrite(positions, 8, count, file.get()) == count &&
                         fwrite(scores, 4, count, file.get()) == count &&
                         fwrite(descriptors, descriptor_bytes, count, file.get()) == count;
    if (!written) fprintf(stderr, "write %s failed\n", path.c_str());
    return written;
}

bool match_pair(const LightGlueSettings & settings, const int keypoints[2], int pair, int32_t * matches,
                float * match_scores, const char * directory) {
    for (int k = 0; k < keypoints[0]; k++) {
        matches[k] = -1;
        match_scores[k] = 0.0f;
    }

    if (keypoints[0] > 0 && keypoints[1] > 0) {
        int32_t statistics[lightglue_stat::kCount] = {};
        const double start = now_microseconds();
        const int error = benzene_lightglue(session(), static_cast<int32_t>(settings.model), settings.references.data(),
                                            static_cast<int>(settings.references.size()), 0, 1, settings.threshold,
                                            settings.depth_confidence, settings.width_confidence, matches, keypoints[0],
                                            match_scores, keypoints[0], statistics, lightglue_stat::kCount);
        const double end = now_microseconds();
        if (!rpc_succeeded(error, "lightglue (pair %d)", pair)) return false;
        print_lightglue(pair, statistics, settings.depth_confidence > 0 || settings.width_confidence > 0,
                        (end - start) / 1e3);
    }
    return write_matches(directory, pair, keypoints[0], matches, match_scores);
}

}  // namespace benzene::host
