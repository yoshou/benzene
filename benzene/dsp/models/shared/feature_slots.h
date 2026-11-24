#pragma once

#include "dsp/runtime/heap.h"
#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"

namespace benzene::models {

// An extractor's features of one image, kept on the DSP for LightGlue: protocol::kFeatureSlots slots per session.
struct Features {
    int count;
    int height;
    int width;
    int dimensions;
    int capacity;
    int descriptor_capacity;                   // halfwords
    runtime::HeapArray<float> keypoints;       // [capacity][2] x, y
    runtime::HeapArray<_Float16> descriptors;  // [count][dimensions]
};

// The session's slot `slot`; nullptr for a slot outside [0, protocol::kFeatureSlots) or when out of memory.
Features * feature_slot(runtime::Session & session, int slot);

// Room for `keypoints` keypoints of `dimensions` descriptor dimensions in `features` (it keeps larger buffers).
runtime::Status reserve_features(Features * features, int keypoints, int dimensions);

}  // namespace benzene::models
