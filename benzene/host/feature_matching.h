#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "host/files.h"
#include "host/model_file.h"
#include "benzene/protocol.h"

namespace benzene::host {

// A bundle of image pairs (python/benzene/models/lightglue/bundle.py): meta.txt holds the pair count, then per image
// `sizes_per_image` sizes (the height and width of each of its planes); images.u8 holds the images' planes back to
// back.
struct PairBundle {
    int pairs = 0;
    int sizes_per_image = 0;
    std::vector<int> sizes;
    std::vector<size_t> offsets;
    MallocBuffer<uint8_t> images;

    const int * image_sizes(int pair, int image) const {
        return sizes.data() + static_cast<size_t>(2 * pair + image) * sizes_per_image;
    }
    const uint8_t * image(int pair, int image) const { return images.get() + offsets[2 * pair + image]; }
};

// False and a message when the bundle is missing or inconsistent.
bool read_pair_bundle(const char * directory, int sizes_per_image, PairBundle * bundle);

// The protocol::lightglue::references(model) of a model file mapped into slot `slot` from file offset `base`; false and
// a message when a tensor is missing.
bool plan_lightglue_references(const ModelTable & table, int slot, uint64_t base, protocol::lightglue::Model model,
                               std::vector<int32_t> * references);

// <directory>/<pair>.<extension><image>: int32 keypoints, positions [keypoints][2] f32, scores [keypoints] f32, then
// the descriptors, descriptor_bytes each.
bool write_features(const char * directory, const char * extension, int pair, int image, int keypoints,
                    const float * positions, const float * scores, const void * descriptors, size_t descriptor_bytes);

struct LightGlueSettings {
    protocol::lightglue::Model model;
    std::vector<int32_t> references;
    float threshold;
    float depth_confidence;  // <= 0: off
    float width_confidence;  // <= 0: off
};

// LightGlue between the features in feature slots 0 and 1 (keypoints[0] and keypoints[1] keypoints): prints the
// call's line and writes <directory>/<pair>.lg: int32 keypoints[0], matches [keypoints[0]] int32 (-1: none), match
// scores [keypoints[0]] f32. matches and match_scores: rpcmem buffers for keypoints[0] values.
bool match_pair(const LightGlueSettings & settings, const int keypoints[2], int pair, int32_t * matches,
                float * match_scores, const char * directory);

}  // namespace benzene::host
