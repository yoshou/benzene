#pragma once

#include <cstdint>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"

namespace benzene::runtime {

// The int8 scale of a row whose largest magnitude is the fp16 value amax: inverse = 127 / amax rounded to fp16
// (inverse_bits), scale = 1 / inverse exactly; both 0 for amax 0. Codes are 128 + round(x * inverse).
struct Int8Scale {
    float scale;
    uint16_t inverse_bits;
    uint16_t padding;
};

// Builds the session's table of the scales of all 32768 non-negative fp16 magnitudes (Session::open calls it).
Status build_int8_scales(Session & session);

// The session's table: entry amax_bits (the bits of a non-negative fp16 magnitude: < 0x8000) is its scale.
const Int8Scale * int8_scales(Session & session);

}  // namespace benzene::runtime
