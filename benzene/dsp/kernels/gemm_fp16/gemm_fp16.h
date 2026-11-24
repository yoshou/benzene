#pragma once

#include <cstdint>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"

namespace benzene::kernels::gemm_fp16 {

enum class OutputForm : uint8_t {
    kRowsF16,  // fp16 rows in column groups (f16_groups), with residuals and ReLU
    kRowsF32,  // fp32 rows (rows_f32)
};

struct Arguments {
    // Input: fp16 rows (row_stride halfwords apart), K / taps at most 2048 channels.
    // - taps > 1: GEMM row r = taps segments of K / taps channels from rows input_map[r * taps + t] (-1: zeros). With
    //   no map and taps 9: the 3x3 stride-1 pad-1 convolution over the (M / tap_width) x tap_width grid.
    // - taps 1: row input_map[r] (no map: r).
    // - relu_input: max(x, 0).
    const _Float16 * input_rows = nullptr;
    int row_stride = 0;
    int taps = 1;
    bool relu_input = false;
    // width (pixels) of the 3x3 input grid, 0 unknown; with taps 9 and a map, the map must then be the 3x3 stride-1 one
    int tap_width = 0;
    const int * input_map = nullptr;
    // The weight stream (python/benzene/kernels/gemm_fp16.py gemm_fp16_stream), per 32-column block: K / 32 fp16 weight
    // tiles (flash attention layout), then a 2 KB store block whose first vector holds per column the word scale
    // 1.0 | (bias << 16) (out = acc * scale + bias, rounded once to fp16).
    const uint8_t * weights = nullptr;
    int M = 0;
    int K = 0;
    int N = 0;
    // Output kRowsF16: fp16 rows.
    // - Column n goes to f16_groups[n / f16_group_width] + (output_map ? output_map[r] : r) * f16_row_stride +
    //   n % f16_group_width.
    // - The value plus residual (max(residual, 0) with relu_residual) plus second_residual (rows as the output), then
    //   max(., 0) with relu_output. The residuals need column-block pairs: N and f16_group_width multiples of 64.
    // Output kRowsF32: fp32 rows rows_f32 [M][N] (rows output_map[r] when there is a map).
    OutputForm output_form = OutputForm::kRowsF16;
    _Float16 * f16_groups[16] = {};
    int f16_group_width = 0;
    int f16_row_stride = 0;
    const int * output_map = nullptr;
    const _Float16 * residual = nullptr;
    const _Float16 * second_residual = nullptr;
    bool relu_residual = false;
    bool relu_output = false;
    float * rows_f32 = nullptr;
    // the caller keeps everything the DMA reads (the weights; the staged input rows of a 3x3 convolution) clean in DDR
    bool skip_cache_flush = false;
    bool input_in_vtcm = false;  // input_rows are in VTCM (not prefetched)
};

// Y = X W^T + b with fp16 activations and weights; rows 128-byte aligned. The DMA reads the weights and the input rows
// past the caches: this call flushes the data cache unless skip_cache_flush. One call at a time (a second caller gets
// kBadParameter).
runtime::Status run(runtime::Session & session, const Arguments & arguments);

}  // namespace benzene::kernels::gemm_fp16
