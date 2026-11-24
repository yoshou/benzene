#pragma once

#include <cstdint>

#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/vector.h"
#include "dsp/vector_math/functions.h"

namespace benzene::hvx {

static inline float horizontal_sum(Vector<float> v) {
    return extract_lane(vector_math::sum_to_all_lanes(v), 0);
}

// The max over the 32 lanes of non-negative f32 values (compared as integers).
static inline float horizontal_max_nonnegative(Vector<float> v) {
    return extract_lane(reinterpret<float>(vector_math::max_to_all_lanes(reinterpret<int32_t>(v))), 0);
}

static inline bool any_lane(Mask lanes) {
    const Vector<uint32_t> words = reinterpret<uint32_t>(pattern_where<uint8_t>(lanes, 0x01010101));
    return extract_word(vector_math::or_to_all_lanes(words), 0) != 0;
}

// The max of a vector of fp16 magnitudes (bits 0 .. 0x7fff: they compare as int16), as fp16 bits.
static inline uint16_t max_f16_magnitude_bits(Vector<int16_t> magnitudes) {
    return static_cast<uint16_t>(extract_word(vector_math::max_to_all_lanes(magnitudes), 0));
}

// Lanes 0 .. 3 = lane 0 of a .. d; the lanes after them are not meaningful.
template <class Lane>
static inline Vector<Lane> collect_lane0_of_four(Vector<Lane> a, Vector<Lane> b, Vector<Lane> c, Vector<Lane> d) {
    static_assert(sizeof(Lane) == 4, "4-byte lanes");
    Vector<Lane> result = select(Mask::first_bytes(4), a, rotate_right_bytes(b, 124));
    result = select(Mask::first_bytes(8), result, rotate_right_bytes(c, 120));
    return select(Mask::first_bytes(12), result, rotate_right_bytes(d, 116));
}

static inline void max_f16_magnitude_bits_of_four(Vector<int16_t> m0, Vector<int16_t> m1, Vector<int16_t> m2,
                                                  Vector<int16_t> m3, uint16_t maxima[4]) {
    alignas(128) uint32_t words[32];
    store(words, collect_lane0_of_four(reinterpret<uint32_t>(vector_math::max_to_all_lanes(m0)),
                                       reinterpret<uint32_t>(vector_math::max_to_all_lanes(m1)),
                                       reinterpret<uint32_t>(vector_math::max_to_all_lanes(m2)),
                                       reinterpret<uint32_t>(vector_math::max_to_all_lanes(m3))));
    for (int row = 0; row < 4; row++) maxima[row] = static_cast<uint16_t>(words[row]);
}

}  // namespace benzene::hvx
