#pragma once

#include <cstdint>

#include "dsp/kernels/gemm_w8a8/gemm_w8a8.h"
#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"

namespace benzene::models {

// A layer's blob (a region of a mapped model file) and the byte offsets of its parts, in the order of the protocol's
// *_part enums.
struct LayerWeights {
    const uint8_t * blob;
    const int32_t * offsets;

    template <typename T>
    const T * part(int which) const {
        return reinterpret_cast<const T *>(blob + offsets[which]);
    }

    kernels::gemm_w8a8::Weights gemm(int stream_part, int scales_part) const {
        return {part<uint8_t>(stream_part), part<float>(scales_part)};
    }
};

// The weights of a runtime call's layer: reference = {model slot, blob offset in the slot, part offsets ...};
// kBadParameter when the slot is not mapped or the offset is outside it.
inline runtime::Result<LayerWeights> layer_in_model(const runtime::Session & session, const int32_t * reference) {
    const auto slot = static_cast<uint32_t>(reference[0]);
    if (!session.model_mapped(static_cast<int>(slot)) ||
        static_cast<uint32_t>(reference[1]) >= session.model_bytes(static_cast<int>(slot))) {
        return std::unexpected(runtime::Error::kBadParameter);
    }
    return LayerWeights{session.model_base(static_cast<int>(slot)) + reference[1], reference + 2};
}

}  // namespace benzene::models
