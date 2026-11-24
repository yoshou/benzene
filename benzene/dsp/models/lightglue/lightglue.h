#pragma once

#include <cstdint>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"
#include "benzene/protocol.h"

namespace benzene::models::lightglue {

struct Arguments {
    protocol::lightglue::Model model = protocol::lightglue::Model::kSuperPoint;
    const int32_t * references = nullptr;  // protocol::lightglue::references(model) entries
    int first_slot = 0;
    int second_slot = 0;
    float threshold = 0.0f;
    float depth_confidence = 0.0f;
    float width_confidence = 0.0f;
    int32_t * matches = nullptr;
    float * match_scores = nullptr;
    int32_t * statistics = nullptr;  // protocol::lightglue::stat
};

// Matches the features in slots first_slot and second_slot as cvg/LightGlue does, in fp16. One call at a time (a
// second caller gets kBadParameter). Per keypoint of the first:
// - matches: the second's keypoint index where the max is mutual and exp(max score) > threshold, else -1;
// - match_scores: exp(max score) where mutual, else 0.
// depth_confidence / width_confidence > 0: LightGlue's adaptive inference (Error::kBadParameter for a model without
// token confidences). An unknown model: Error::kBadParameter.
runtime::Status run(runtime::Session & session, const Arguments & arguments);

}  // namespace benzene::models::lightglue
