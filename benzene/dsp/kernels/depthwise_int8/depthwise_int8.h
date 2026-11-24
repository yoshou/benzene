#pragma once

#include <cstdint>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"

namespace benzene::kernels::depthwise_int8 {

// - Input: a uint8 NHWC map [height][width][channels], channels 32, 64 or a multiple of 128. With 128 or more a
//   pixel's codes may sit in a wider row: input_stride bytes per pixel (else input_stride is channels). The padding
//   pixels (2 on each side) read zero_points[c]. width >= 2.
// - Weights: int8 [25][channels], tap ky * 5 + kx.
// - Per channel c the exact sum S = initial[c] + sum_t w[t][c] q[t], where initial = -zero_point sum_t w[t][c] (for
//   x = s (q - z)) plus any integer bias.
// - Output: the code q' = trunc(clamp(fp32(fp32(S) output_multipliers[c]) + output_biases[c], 0, 255.5)) into the uint8
// NHWC map
//   output (the same size).
struct Arguments {
    const uint8_t * input = nullptr;
    int height = 0;
    int width = 0;
    int channels = 0;
    const uint8_t * zero_points = nullptr;  // [channels]
    const int8_t * weights = nullptr;
    const int32_t * initial = nullptr;           // [channels]
    const float * output_multipliers = nullptr;  // [channels]
    const float * output_biases = nullptr;       // [channels]
    uint8_t * output = nullptr;
    int input_stride = 0;
};

// The depthwise convolution above. The DMA reads the input through L2. One call at a time (a second caller gets
// kBadParameter).
runtime::Status run(runtime::Session & session, const Arguments & arguments);

}  // namespace benzene::kernels::depthwise_int8
