#pragma once

#include <cstdint>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"
#include "benzene/protocol.h"

namespace benzene::models::aliked {

struct Arguments {
    const int32_t * references = nullptr;  // protocol::aliked::kReferences: model slot, tensor offsets
    const uint8_t * image = nullptr;       // gray [height][width]
    int height = 0;
    int width = 0;
    int max_keypoints = 0;
    float threshold = 0.0f;
    int slot = 0;                      // the feature slot written (for LightGlue)
    float * keypoints = nullptr;       // [k][2] (x, y)
    float * scores = nullptr;          // [k]
    _Float16 * descriptors = nullptr;  // [k][protocol::aliked::kDescriptorDimensions]
    int32_t * statistics = nullptr;    // protocol::aliked::stat
};

// ALIKED's features of the image, into feature slot `slot` and the outputs. One call at a time (a second caller gets
// kBadParameter).
// - The image: height and width multiples of 32, at least 64; width at most 2048, height at most 4096.
// - The keypoints: at most max_keypoints of the score map's NMS maxima above threshold, by descending score (ties in
//   raster order), refined by DKD's soft-argmax.
runtime::Status run(runtime::Session & session, const Arguments & arguments);

}  // namespace benzene::models::aliked
