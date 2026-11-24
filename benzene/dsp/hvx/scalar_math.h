#pragma once

#include <bit>
#include <cstdint>

namespace benzene::hvx {

template <class T>
constexpr T round_up(T value, T multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

// 1 / sqrt(x) for a normal x > 0: bit-trick seed and two Newton steps (relative error < 5e-6).
static inline float reciprocal_square_root(float x) {
    const uint32_t bits = 0x5f375a86u - (std::bit_cast<uint32_t>(x) >> 1);
    const float half_x = 0.5f * x;
    float y = std::bit_cast<float>(bits);
    y = y * (1.5f - half_x * y * y);
    return y * (1.5f - half_x * y * y);
}

}  // namespace benzene::hvx
