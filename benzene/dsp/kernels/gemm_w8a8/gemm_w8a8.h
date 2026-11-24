#pragma once

#include <cstdint>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"

namespace benzene::kernels::gemm_w8a8 {

enum class InputForm : uint8_t {
    kRowsF32,            // fp32 rows, quantized per row (scale = amax / 127) by the kernel
    kRowsF16Scaled,      // fp16 rows (through a row map) times a per-column scale, quantized with fp16 inverse scales
    kPrequantizedTiles,  // activation tiles a producer pass wrote, with their row scales
    kHandoverTiles,      // the fp16 tiles of a previous GEMM's kHandoverTiles output, with its row scales
};

enum class OutputForm : uint8_t {
    kRowsF32Activation,  // Y = activation(v + bias) * scale (activation kNone or kGeluErf)
    kRowsF32Accumulate,  // Y += v + bias (residual stream)
    kRowsF16,            // fp16 v (no bias) into four column groups (RoPE, KV cache)
    kHandoverTiles,      // activation(v + bias) * scale (relu^2, GELU) as fp16 tiles + row scales: the next input
    kAttentionTiles72,   // (v + bias) as the flash attention's Q^T / K / V^T tiles of 72-dimensional heads
    kAttentionTiles64,   // the same for 64-dimensional heads, optionally with QK LayerNorm and 2D RoPE
};

// The activation of kRowsF32Activation and kHandoverTiles (kNone for the other output forms).
enum class Activation : uint8_t { kNone, kReluSquared, kGeluTanh, kGeluErf };

// fp16 hand-over between two GEMMs (output kHandoverTiles of the first, input kHandoverTiles of the second).
// - tiles: for every 64-row block its N / 32 tiles of 64 rows x 32 columns (4 KB), back to back; rows past M are zero.
//   Row quad i of a tile is two vectors of 64 halfwords: lane 2c + p of the first is column c of row 4i + p, lane
//   2c + p of the second is column c of row 4i + 2 + p.
// - Per row (rows padded to the 64-multiple): inverse_bits = the fp16 bits of 127 / amax rounded to fp16,
//   row_scales = 1 / inverse exactly.
// - The codes are 128 + the fp16 product x * inverse, rounded to nearest.
struct HandoverTiles {
    _Float16 * tiles = nullptr;
    float * row_scales = nullptr;
    uint16_t * inverse_bits = nullptr;
};

// The weight stream (gemm_w8a8_stream in python/benzene/kernels/gemm_w8a8.py), 4096-byte aligned and flushed to DDR: the
// weight DMA bypasses the caches. Per 32-column block:
// - K / 32 weight tiles of 1 KB: byte (k / 4) * 128 + c * 4 + k % 4 of k tile kt holds W[c][32 kt + k];
// - then 4 output blocks of 256 B, one per accumulator slice: 32 scale words, the fp16-coded 512, 2, 2^-7, 2^-14 (a
//   uint8 store keeps floor(sum * scale / 512) mod 256: bits 0-7, 8-15, 16-23, 23-30 of the sum), then 32 bias words
//   = -128 * colsum(W), which remove the +128 activation offset.
struct Weights {
    const uint8_t * stream = nullptr;
    const float * column_scales = nullptr;  // [N]
};

struct Input {
    InputForm form = InputForm::kRowsF32;
    const float * rows_f32 = nullptr;  // kRowsF32: [M][K]
    // kRowsF16Scaled: GEMM row r = row row_map[r] (no map: r), row_stride values apart, times column_scale[k]; K and
    // row_stride multiples of 64
    const _Float16 * rows_f16 = nullptr;
    const int * row_map = nullptr;
    const float * column_scale = nullptr;
    int row_stride = 0;
    // kPrequantizedTiles: per 64-row block its K / 32 activation tiles of 2 KB, back to back, written by the CPU side
    // (flushed here). Byte (r / 4) * 128 + k * 4 + r % 4 of k tile kt holds the code (128 + int8) of row r, column
    // 32 kt + k.
    const uint8_t * tiles = nullptr;
    const float * tile_row_scales = nullptr;
    const HandoverTiles * handover = nullptr;  // kHandoverTiles: K = the producer's N
};

struct Output {
    OutputForm form = OutputForm::kRowsF32Activation;
    Activation activation = Activation::kNone;
    float * rows_f32 = nullptr;    // kRowsF32*: Y [M][N]
    const float * bias = nullptr;  // per output column (optional; not kRowsF16), added before the activation
    // per output column (kRowsF32Activation: optional, kHandoverTiles: needed), applied after the activation
    const float * column_scale = nullptr;
    // kRowsF16: column n -> f16_groups[g = n / f16_group_width], row r of f16_group_strides[g] values, column
    // n % f16_group_width
    _Float16 * f16_groups[4] = {};
    int f16_group_width = 0;
    int f16_group_strides[4] = {};
    HandoverTiles * handover = nullptr;  // kHandoverTiles
    // kRowsF16 RoPE (bit g of rope_groups: group g). Its heads (128 columns) hold dims [0..31, 64..95, 32..63,
    // 96..127], so 64-column pair p of a head rotates dims (i, i + 64), i = 32 p + lane, by angle i of GEMM row r:
    // rope_cos / rope_sin [r * rope_stride + i], the cos and sin of a row within its rope_stride values.
    int rope_groups = 0;
    const float * rope_cos = nullptr;
    const float * rope_sin = nullptr;
    int rope_stride = 0;
    // kAttentionTiles72: (v + bias) as the flash attention's fp16 tiles of 72-dim heads. GEMM rows are token positions.
    // - Column block cb is group cb / attention_group_blocks (Q, K, V). Within a group (nh =
    //   attention_group_blocks * 4 / 9 heads), block c < 2 nh is d-tile c % 2 of head c / 2, and block 2 nh + m holds
    //   dims 64..71 of heads 4m .. 4m+3 (d-tile 2, zero beyond).
    // - Q^T tiles attention_q [head][32-block][3], K tiles attention_k [head][key_tile_stride][3], V^T tiles
    //   attention_v [head][3][key_tile_stride].
    // kAttentionTiles64 (column-block pairs): 64-dim heads as the same tiles.
    // - Block cb is group cb / attention_group_blocks, head (cb % attention_group_blocks) / 2, d-tile cb % 2.
    // - Q^T attention_q [head][key_tile_stride][2], K attention_k [head][key_tile_stride][2], V^T attention_v
    //   [head][2][key_tile_stride].
    uint8_t * attention_q = nullptr;
    uint8_t * attention_k = nullptr;
    uint8_t * attention_v = nullptr;
    int attention_group_blocks = 0;
    int attention_key_tile_stride = 0;
    // kAttentionTiles64, Q and K (groups 0, 1); M a multiple of 32:
    // - optionally a LayerNorm over the head's 64 dims (per group gamma[64], then beta[64]);
    // - then 2D RoPE by rope_table, pairs (i, i + 16) within each half: dims 0..31 rotate by the position
    //   rope_positions[2 r] of GEMM row r, dims 32..63 by rope_positions[2 r + 1]. Per position the table holds two
    //   vectors: cos(p f_(lane % 16)), then -sin / +sin(p f_(lane % 16)) for lanes < 16 / >= 16.
    const float * qk_norm_parameters = nullptr;
    float qk_norm_epsilon = 0.0f;
    const int16_t * rope_positions = nullptr;
    const float * rope_table = nullptr;
    int rope_table_positions = 0;
    // kRowsF16 with a KV cache (column-block pairs): groups 2 / 3 (K / V, heads of cache_head_dimensions dims) go to
    // the key tiles key_cache [head][cache_key_tile_stride][dims / 32] (after RoPE when rope bit 2) and the V^T tiles
    // value_cache [head][dims / 32][cache_key_tile_stride] (flash attention layouts). GEMM row r is key position r;
    // the rows from M to the tile's end are zero.
    uint8_t * key_cache = nullptr;
    uint8_t * value_cache = nullptr;
    int cache_key_tile_stride = 0;
    int cache_head_dimensions = 0;
};

struct Arguments {
    int M = 0;
    int K = 0;
    int N = 0;
    Weights weights{};
    Input input{};
    Output output{};
};

// Y [M][N] = X [M][K] W [N][K]^T in the forms the arguments name (kBadParameter when a form's operands are missing);
// rows / tiles 128-byte aligned. The DMA reads the weight stream past the caches (the caller keeps it clean in DDR)
// and the tiles: this call flushes the data cache for them. One call at a time (a second caller gets kBadParameter).
runtime::Status run(runtime::Session & session, const Arguments & arguments);

}  // namespace benzene::kernels::gemm_w8a8
