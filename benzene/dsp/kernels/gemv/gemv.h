#pragma once

#include <cstdint>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"

namespace benzene::kernels::gemv {

constexpr int kMaximumK = 9216;

enum class WeightFormat : uint8_t {
    // The W8A8 GEMM's stream and column scales (gemm_w8a8::Weights). K a multiple of 32.
    kInt8,
    // 4-bit block-scaled (python/benzene/kernels/gemv.py gemv_q4_stream), 18 K bytes per 32-row group:
    // - per 32-k block, 512 B of codes: code + 8 in nibbles at byte (kk / 8) * 128 + r * 4 + kk % 4, in the low
    //   nibble for kk % 8 < 4 and in the high nibble for kk % 8 >= 4;
    // - then K / 64 vectors of fp16 block scales (lane 2r + (b & 1) of vector b / 2).
    // K a multiple of 64.
    kQ4_0,
};

struct Arguments {
    WeightFormat format = WeightFormat::kInt8;
    int K = 0;
    int N = 0;                              // a multiple of 32
    const float * x = nullptr;              // [K], 128-byte aligned
    const uint8_t * weights = nullptr;      // 4096-byte aligned, clean in DDR (the DMA bypasses the caches)
    const float * column_scales = nullptr;  // kInt8: [N], 128-byte aligned
    float * y = nullptr;                    // [N], 128-byte aligned
};

// y = W x. The DMA reads the weights past the caches. One call at a time (a second caller gets kBadParameter).
runtime::Status run(runtime::Session & session, const Arguments & arguments);

}  // namespace benzene::kernels::gemv
