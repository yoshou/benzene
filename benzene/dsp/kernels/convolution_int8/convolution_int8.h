#pragma once

#include <cstddef>
#include <cstdint>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"

namespace benzene::kernels::convolution_int8 {

// The SELU requantization's per-column constants: Sp, Bp, St, Bt, Dn, Q0 .. Q4 (vector_math::SeluConstants).
constexpr int kSeluConstantCount = 10;

enum class OutputForm : uint8_t {
    kUint8Maps,
    kRowsF32,
};

enum class Requantization : uint8_t {
    kLinear,
    kSelu,
};

// - Input: a uint8 NHWC map [height][width][channels], channels 1, 32, 64 or a multiple of 128.
// - Convolution: 3x3 pad-1 (taps 9; stride 1, or stride 2 for 32 / 64 channels and an even height and width: the
//   output is height / 2 x width / 2) or 1x1 stride 1. With pool, height and width even.
// - Padding pixels read zero_point (1 channel, or 32 channels stride 1; else 0). With `padding` (3x3 stride 1, 32, 64
//   or 128 channels, no pool) they read padding[c] in channel c.
// - Weights (python/benzene/kernels/convolution_int8.py convolution_int8_stream): weight_stream_bytes(K, N) bytes, K = taps *
//   channels in tap-major order (channels 1: K = 32, k = tap), N a multiple of 32, at most 512 (kSelu: 128). Per
//   32-column block:
//   - K / 32 int8 tiles of 2 KB: byte (k / 4) * 128 + c * 4 + k % 4, then 1 KB of zeros;
//   - 4 slice output blocks of 256 B (hmx::store_int32_as_uint8_slices), the bias words -sum_k W[c][k] zero point(k);
//   - 1 KB of output constants: kUint8Maps m and b (4 vectors each, lane i of vector j: column 8j + i / 4), kRowsF32
//     32 column scales and 32 biases.
// - Output kUint8Maps: q = trunc(clamp(fp32(sum) m + b, 0, 255.5)); |fp32(sum) m + b| must stay below 2^30. Channel n
//   goes to uint8_outputs[n / uint8_output_channels] + pixel * uint8_output_channels + n % uint8_output_channels
//   (N = 32: one map of 32 channels, or with kSelu the first 32 of a map of 64; 64; or multiples of 128). With `pool`
//   the pixels are those of the height / 2 x width / 2 max-pooled map. The maps are 128-byte aligned and need room for
//   8 more pixels (whole groups are written).
// - Output kRowsF32: fp32 rows rows_f32 [pixels][N], 128-byte aligned (sum * column scale + bias; with
//   rows_f32_columns = 8 and N = 32 only the first 8 columns, rows of 8).
// - Requantization kLinear: output_multipliers / output_biases (N each, nullptr: the stream's) replace m and b for one
//   call.
// - Requantization kSelu: selu_constants (fp32 [kSeluConstantCount][N], column order) -
//   vector_math::selu_requantize of fp32(sum); no pool.
struct Arguments {
    const uint8_t * input = nullptr;
    int height = 0;
    int width = 0;
    int channels = 0;
    int taps = 0;
    int stride = 1;
    int zero_point = 0;
    const uint8_t * padding = nullptr;
    bool pool = false;
    const uint8_t * weights = nullptr;
    int N = 0;
    uint8_t * uint8_outputs[4] = {};
    int uint8_output_channels = 0;
    const float * output_multipliers = nullptr;
    const float * output_biases = nullptr;
    const float * selu_constants = nullptr;
    float * rows_f32 = nullptr;
    int rows_f32_columns = 0;  // 0: N
    OutputForm output_form = OutputForm::kUint8Maps;
    Requantization requantization = Requantization::kLinear;
};

// The convolution above. The DMA reads the weights past the caches (memory the DSP has not written, as a mapped model
// file, or flushed) and the input through L2. One call at a time (a second caller gets kBadParameter).
runtime::Status run(runtime::Session & session, const Arguments & arguments);

size_t weight_stream_bytes(int K, int N);

}  // namespace benzene::kernels::convolution_int8
