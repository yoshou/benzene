#pragma once

#include <cstdint>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"
#include "benzene/protocol.h"

namespace benzene::models::superpoint {

struct Arguments {
    const int32_t * references = nullptr;  // protocol::superpoint::kReferences: model slot, weight stream offsets
    const uint8_t * image = nullptr;       // gray [height][width]
    int height = 0;
    int width = 0;
    int max_keypoints = 0;
    float threshold = 0.0f;
    int slot = 0;                      // the feature slot written (for LightGlue)
    float * keypoints = nullptr;       // [k][2] (x, y)
    float * scores = nullptr;          // [k]
    _Float16 * descriptors = nullptr;  // [k][protocol::superpoint::kDescriptorDimensions]
    int32_t * statistics = nullptr;    // protocol::superpoint::stat
};

// SuperPoint's features of the image as LightGlue's superpoint.py computes them (nms_radius 4, remove_borders 4), into
// feature slot `slot` and the outputs. One call at a time (a second caller gets kBadParameter).
// - The image: height and width multiples of 8, at least 16 (kNoMemory when its maps do not fit VTCM).
// - The keypoints: at most max_keypoints of the highest scores above threshold (ties in raster order), in raster order
//   as superpoint.py returns them.
runtime::Status run(runtime::Session & session, const Arguments & arguments);

}  // namespace benzene::models::superpoint
