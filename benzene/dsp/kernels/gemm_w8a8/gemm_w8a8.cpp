#include "dsp/kernels/gemm_w8a8/gemm_w8a8.h"

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

#include "dsp/hmx/instructions.h"
#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/prefetch.h"
#include "dsp/hvx/quantization.h"
#include "dsp/hvx/reductions.h"
#include "dsp/hvx/vector.h"
#include "dsp/runtime/cache.h"
#include "dsp/runtime/dma_ring.h"
#include "dsp/runtime/heap.h"
#include "dsp/runtime/int8_scale_table.h"
#include "dsp/runtime/status.h"
#include "dsp/runtime/synchronization.h"
#include "dsp/runtime/thread_pool.h"
#include "dsp/vector_math/functions.h"

namespace benzene::kernels::gemm_w8a8 {
namespace {

using hmx::kTileBytes;
using hvx::Mask;
using hvx::Vector;
using hvx::VectorPair;

// A staging slot holds a tile's int32 sums as four uint8 slices of 2 KB: bits 0..7, 8..15, 16..23 and 23..30 of the
// sums. Byte (r / 4) * 128 + c * 4 + r % 4 of a slice is row r, column c.
constexpr int kStagingSlotBytes = 8192;
constexpr int kGatherSlotsPerThread = 32;  // one per row pair of a tile
constexpr int kGeluTableEntries = 32768;
constexpr int kThreadScratchVectors = 64;
constexpr int kStagingSlotVectors = kStagingSlotBytes / hvx::kVectorBytes;
constexpr int kMaximumStagingSlots = 32;
constexpr int kHandoverTilesPerItem = 8;
constexpr size_t kHandoverTileHalfwords = 64 * 32;
constexpr size_t kHandoverTileBytes = kHandoverTileHalfwords * sizeof(_Float16);
constexpr size_t kHandoverStagingBytes = kHandoverTilesPerItem * kHandoverTileBytes;
constexpr int kDeepWeightMinimumKTiles = 32;
// A column block of the weight stream: its k tiles of int8 weights (32 x 32), then its four output blocks.
constexpr int kWeightTileBytes = 1024;
constexpr int kOutputBlocksBytes = 1024;

// The GELU stages' VTCM region. values[b] = T(x) = GELU(-x) for the fp16 magnitude x with bits b (0 from 16 on), so
// that GELU(x) = T(|x|) + max(x, 0); a gather finds T(|x|) at byte offset bits(x) << 1 (the shift drops the sign).
struct GeluTable {
    _Float16 values[kGeluTableEntries];
    Vector<_Float16> gather_slots[runtime::kHvxThreads * kGatherSlotsPerThread];
};
constexpr size_t kGeluTableBytes = sizeof(GeluTable);

union ThreadScratch {
    Vector<float> f32[runtime::kHvxThreads * kThreadScratchVectors];
    Vector<_Float16> f16[runtime::kHvxThreads * kThreadScratchVectors];
};

struct OutputContext {
    int M;
    int N;
    int column_blocks;
    const float * row_scales;
    const float * column_scales;
    const Output * output;
    float * rows;
    ThreadScratch * thread_scratch;
    GeluTable * gelu_table;
    Vector<int16_t> * row_maxima;
    int row_quads;
    const runtime::Int8Scale * int8_scales;
};

// A 64-row x 32-column tile, or two column-adjacent tiles for the column-pair outputs (else slots[1] is nullptr).
struct OutputItem {
    const Vector<uint8_t> * slots[2];
    int column_block;
    int first_row;
    int rows;
    int thread;
};

using OutputStage = void (*)(const OutputContext & context, const OutputItem & item);

struct W8a8GemmJob {
    runtime::Session * session;
    OutputStage output_stage;
    OutputContext output_context;

    const float * rope_source;
    float * rope_vtcm;
    uint32_t rope_bytes;

    runtime::DmaRing * driver_ring;
    runtime::DmaRing * handover_rings[runtime::kHvxThreads];
    Vector<_Float16> * handover_staging;
    int prequantized_through_cache;
    int M;
    int K;
    int N;
    int k_tiles;
    int column_blocks;
    int chunk_rows;  // a multiple of 64
    int weight_chunk_column_blocks;
    int staging_slots;
    size_t column_block_bytes;

    const uint8_t * weight_stream;
    const float * column_scales;
    float * row_scales;  // [M rounded up to 64]
    const runtime::Int8Scale * int8_scales;
    uint8_t * activations;
    uint8_t * activation_buffers[2];
    // chunk c + 1 is quantized while chunk c is on HMX (-1 before plan_layout: when M takes several chunks)
    int pipelined;
    int column_pairs;
    // tile t = (cb / g) * g * row_blocks + rb * g + cb % g (g = column_group): groups of g column blocks, row block by
    // row block within a group
    int column_group;
    int prefer_row_groups;
    // weight.b = mxmem():deep: a column block's 1 KB rows (its weight tiles, then its output blocks) lie 2 KB apart in
    // weight_buffers (the KB between is not read); column blocks 2i and 2i + 1 share a region of two stream blocks, at
    // even / odd KB
    int deep_weights;
    // the chunk being quantized
    int quantized_first_row;
    int quantized_row_blocks;
    int quantized_buffer;
    uint8_t * weight_buffers[2];
    Vector<uint8_t> * staging;
    Input input;
    Vector<_Float16> * f16_input_scratch;
    Vector<_Float16> * f16_column_scale;
    Output output;
    GeluTable * gelu_table;
    Vector<int16_t> * row_maxima;
    int row_quads;
    size_t vtcm_reserved;
    // the chunk on HMX
    int chunk_first_row;
    int chunk_row_blocks;

    alignas(128) std::atomic<int> produced;   // tiles of the chunk HMX has stored
    alignas(128) std::atomic<int> next_item;  // output items claimed
    std::atomic<int> next_quad;               // input work items claimed
    // slot_done[t % staging_slots] = t + 1 once tile t is written out: HMX may then store tile t + staging_slots
    alignas(128) std::atomic<int> slot_done[kMaximumStagingSlots];
    alignas(128) runtime::SpinBarrier barrier;
};

inline void tile_position(const W8a8GemmJob & job, int tile, int * column_block, int * row_block) {
    const int group = job.column_group;
    const int group_index = tile / (group * job.chunk_row_blocks);
    const int within = tile % (group * job.chunk_row_blocks);
    *column_block = group_index * group + within % group;
    *row_block = within / group;
}

constexpr bool activation_is_gelu(Activation activation) {
    return activation == Activation::kGeluTanh || activation == Activation::kGeluErf;
}

// int32 sum x column scale x row scale, in this order.
[[gnu::always_inline]] inline Vector<float> dequantize(Vector<int32_t> sums, Vector<float> column_scale,
                                                       float row_scale) {
    const Vector<float> row = Vector<float>::splat(row_scale);
    return to_float(sums) * column_scale * row;
}

[[gnu::always_inline]] inline Vector<float> dequantize_with_bias(Vector<int32_t> sums, Vector<float> column_scale,
                                                                 float row_scale, Vector<float> bias) {
    return dequantize(sums, column_scale, row_scale) + bias;
}

// 32 rows as 16 fp16 vectors (vector k = [row 2k | row 2k + 1]) -> line i = dimension pair i of the 32 rows (halfword
// 2r + d % 2 = row r). Four deals (inverse perfect shuffles) transpose the 32 x 16-word matrix: word index r * 16 + i
// rotated right by 4 -> i * 32 + r.
[[gnu::always_inline]] inline void transpose_row_pairs_to_lines(Vector<_Float16> (&vectors)[16]) {
#pragma clang loop unroll(full)
    for (int stage = 0; stage < 4; stage++) {
        Vector<_Float16> dealt[16];
#pragma clang loop unroll(full)
        for (int k = 0; k < 8; k++) {
            const hvx::VectorPair<_Float16> pair = deinterleave_chunks<4>(vectors[2 * k + 1], vectors[2 * k]);
            dealt[k] = pair.low();
            dealt[k + 8] = pair.high();
        }

#pragma clang loop unroll(full)
        for (int k = 0; k < 16; k++) vectors[k] = dealt[k];
    }
}

[[gnu::always_inline]] inline Vector<float> column_vector(const float * values, int column_block) {
    return values != nullptr ? hvx::load(values + column_block * 32) : Vector<float>::zero();
}

template <Activation kActivation>
[[gnu::always_inline]] inline Vector<float> activate_and_scale(Vector<float> x, Vector<float> scale) {
    if constexpr (kActivation == Activation::kReluSquared) {
        const Vector<float> positive_part = max(x, Vector<float>::zero());
        return positive_part * positive_part * scale;
    } else if constexpr (kActivation == Activation::kGeluErf) {
        return vector_math::gelu_erf(x) * scale;
    } else {
        return x * scale;
    }
}

[[gnu::always_inline]] inline void rotate_pair(Vector<float> a, Vector<float> b, Vector<float> cosine,
                                               Vector<float> sine, Vector<float> & rotated_a,
                                               Vector<float> & rotated_b) {
    rotated_a = a * cosine - b * sine;
    rotated_b = b * cosine + a * sine;
}

// Lanes i and i + 16 of a half head rotate together: the 64-byte rotation swaps them.
[[gnu::always_inline]] inline Vector<float> rotate_half_head(Vector<float> x, Vector<float> cosine,
                                                             Vector<float> signed_sine) {
    return x * cosine + rotate_right_bytes(x, 64) * signed_sine;
}

[[gnu::always_inline]] inline void layer_norm_head64(Vector<float> & first_half, Vector<float> & second_half,
                                                     const float * parameters, Vector<float> epsilon) {
    const float one_64th = 1.0f / 64;
    const Vector<float> mean = vector_math::sum_to_all_lanes(first_half + second_half) * one_64th;
    const Vector<float> squares = first_half * first_half + second_half * second_half;
    const Vector<float> mean_square = vector_math::sum_to_all_lanes(squares) * one_64th;
    const Vector<float> variance = max(mean_square - mean * mean, Vector<float>::zero());
    const Vector<float> inverse_deviation = vector_math::reciprocal_square_root(variance + epsilon);
    first_half = (first_half - mean) * inverse_deviation * hvx::load(parameters) + hvx::load(parameters + 64);
    second_half = (second_half - mean) * inverse_deviation * hvx::load(parameters + 32) + hvx::load(parameters + 96);
}

struct RowStage {
    Vector<float> bias;
    Vector<float> scale;
    _Float16 * f16_columns;
    Mask f16_half;
    int f16_stride;
};

template <OutputForm kOutput, Activation kActivation>
[[gnu::always_inline]] inline void write_output_row(const RowStage & stage, Vector<float> value, float * destination,
                                                    int row) {
    if constexpr (kOutput == OutputForm::kRowsF32Activation) {
        const Vector<float> biased = value + stage.bias;
        hvx::store(destination, activate_and_scale<kActivation>(biased, stage.scale));
    } else if constexpr (kOutput == OutputForm::kRowsF32Accumulate) {
        hvx::store(destination, value + stage.bias + hvx::load(destination));
    } else {
        // both halves hold the 32 values: the half mask stores them at their place in the aligned 128-byte line
        const Vector<_Float16> f16 = to_float16(value, value);
        _Float16 * line = stage.f16_columns + static_cast<size_t>(row) * stage.f16_stride;
        store_masked(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(line) & ~uintptr_t{127}), stage.f16_half,
                     f16);
    }
}

template <OutputForm kOutput, Activation kActivation>
[[gnu::always_inline]] inline void write_scaled_row(const RowStage & stage, Vector<int32_t> sums,
                                                    Vector<float> column_scale, const float * __restrict row_scales,
                                                    float * __restrict destination, int row_stride, int first_row,
                                                    int row) {
    write_output_row<kOutput, kActivation>(stage, dequantize(sums, column_scale, row_scales[row]),
                                           destination + static_cast<size_t>(row) * row_stride, first_row + row);
}

// Returns the rows done (pairs only).
[[gnu::always_inline]] inline int gelu_rows_through_table(const OutputContext & context, const OutputItem & item,
                                                          const RowStage & stage, Vector<float> * __restrict scaled,
                                                          float * __restrict destination, int row_stride) {
    Vector<_Float16> * gathered =
        context.gelu_table->gather_slots + static_cast<size_t>(item.thread) * kGatherSlotsPerThread;
    const uint32_t table = reinterpret_cast<uint32_t>(context.gelu_table->values);
    const Vector<float> zero = Vector<float>::zero();
    const int pairs = item.rows / 2;

    for (int pair = 0; pair < pairs; pair++) {
        const Vector<float> first = scaled[2 * pair] + stage.bias;
        const Vector<float> second = scaled[2 * pair + 1] + stage.bias;
        const Vector<_Float16> f16 = to_float16_interleaved(first, second);
        gather_halfwords(gathered + pair, table, 65535, reinterpret<int16_t>(f16) << 1);
        scaled[2 * pair] = max(first, zero);
        scaled[2 * pair + 1] = max(second, zero);
    }

    for (int pair = 0; pair < pairs; pair++) {
        const VectorPair<float> table_values = to_float32_deinterleaved(gathered[pair]);
        hvx::store(destination + static_cast<size_t>(2 * pair) * row_stride,
                   (table_values.low() + scaled[2 * pair]) * stage.scale);
        hvx::store(destination + static_cast<size_t>(2 * pair + 1) * row_stride,
                   (table_values.high() + scaled[2 * pair + 1]) * stage.scale);
    }
    return pairs * 2;
}

template <OutputForm kOutput, Activation kActivation>
void write_rows_tile(const OutputContext & context, const OutputItem & item) {
    const Output & output = *context.output;
    const Vector<uint8_t> * __restrict slot = item.slots[0];
    const int column_block = item.column_block;
    const int first_row = item.first_row;
    const int rows = item.rows;
    const int row_stride = context.column_blocks * 32;
    float * __restrict destination = context.rows + static_cast<size_t>(first_row) * context.N + column_block * 32;
    const float * __restrict row_scales = context.row_scales + first_row;
    const Vector<float> column_scale = hvx::load(context.column_scales + column_block * 32);

    RowStage stage;
    stage.bias = column_vector(output.bias, column_block);
    stage.scale = output.column_scale != nullptr ? hvx::load(output.column_scale + column_block * 32)
                                                 : Vector<float>::splat(1.0f);
    stage.f16_columns = nullptr;
    stage.f16_half = Mask::first_bytes(64);
    stage.f16_stride = 0;
    if constexpr (kOutput == OutputForm::kRowsF16) {
        const int group = column_block * 32 / output.f16_group_width;
        const int first_column = column_block * 32 % output.f16_group_width;
        stage.f16_columns = output.f16_groups[group] + first_column;
        stage.f16_stride = output.f16_group_strides[group];
        if (first_column % 64 != 0) stage.f16_half = !stage.f16_half;
    }

    if constexpr (kOutput == OutputForm::kRowsF32Activation || kOutput == OutputForm::kRowsF32Accumulate) {
        Vector<float> * __restrict scaled =
            context.thread_scratch->f32 + static_cast<size_t>(item.thread) * kThreadScratchVectors;
        if constexpr (kOutput == OutputForm::kRowsF32Accumulate) {
            hvx::prefetch_to_l2(destination, 128, static_cast<uint32_t>(row_stride) * 4, static_cast<uint32_t>(rows));
        }

        for (int quad = 0; quad < 16; quad++) {
            Vector<int32_t> sums[4];
            hmx::load_int32_row_quad(slot, quad, sums);
            scaled[quad * 4 + 0] = dequantize(sums[0], column_scale, row_scales[quad * 4 + 0]);
            scaled[quad * 4 + 1] = dequantize(sums[1], column_scale, row_scales[quad * 4 + 1]);
            scaled[quad * 4 + 2] = dequantize(sums[2], column_scale, row_scales[quad * 4 + 2]);
            scaled[quad * 4 + 3] = dequantize(sums[3], column_scale, row_scales[quad * 4 + 3]);
        }

        int row = 0;
        if constexpr (kOutput == OutputForm::kRowsF32Activation && activation_is_gelu(kActivation)) {
            row = gelu_rows_through_table(context, item, stage, scaled, destination, row_stride);
        }
        for (const int end = row + (rows - row) / 4 * 4; row < end; row += 4) {
            write_output_row<kOutput, kActivation>(
                stage, scaled[row], destination + static_cast<size_t>(row) * row_stride, first_row + row);
            write_output_row<kOutput, kActivation>(
                stage, scaled[row + 1], destination + static_cast<size_t>(row + 1) * row_stride, first_row + row + 1);
            write_output_row<kOutput, kActivation>(
                stage, scaled[row + 2], destination + static_cast<size_t>(row + 2) * row_stride, first_row + row + 2);
            write_output_row<kOutput, kActivation>(
                stage, scaled[row + 3], destination + static_cast<size_t>(row + 3) * row_stride, first_row + row + 3);
        }
        for (; row < rows; row++) {
            write_output_row<kOutput, kActivation>(
                stage, scaled[row], destination + static_cast<size_t>(row) * row_stride, first_row + row);
        }
    } else if (rows == 64) {
#pragma clang loop unroll_count(2)
        for (int quad = 0; quad < 16; quad++) {
            Vector<int32_t> sums[4];
            hmx::load_int32_row_quad(slot, quad, sums);
            write_scaled_row<kOutput, kActivation>(stage, sums[0], column_scale, row_scales, destination, row_stride,
                                                   first_row, quad * 4 + 0);
            write_scaled_row<kOutput, kActivation>(stage, sums[1], column_scale, row_scales, destination, row_stride,
                                                   first_row, quad * 4 + 1);
            write_scaled_row<kOutput, kActivation>(stage, sums[2], column_scale, row_scales, destination, row_stride,
                                                   first_row, quad * 4 + 2);
            write_scaled_row<kOutput, kActivation>(stage, sums[3], column_scale, row_scales, destination, row_stride,
                                                   first_row, quad * 4 + 3);
        }
    } else {
        for (int quad = 0; quad < 16; quad++) {
            Vector<int32_t> sums[4];
            hmx::load_int32_row_quad(slot, quad, sums);
            for (int k = 0; k < 4; k++) {
                if (quad * 4 + k < rows) {
                    write_scaled_row<kOutput, kActivation>(stage, sums[k], column_scale, row_scales, destination,
                                                           row_stride, first_row, quad * 4 + k);
                }
            }
        }
    }
}

void write_rows_f16_tile(const OutputContext & context, const OutputItem & item) {
    write_rows_tile<OutputForm::kRowsF16, Activation::kNone>(context, item);
}

OutputStage find_row_output_stage(OutputForm form, Activation activation) {
    if (form == OutputForm::kRowsF32Accumulate && activation == Activation::kNone) {
        return write_rows_tile<OutputForm::kRowsF32Accumulate, Activation::kNone>;
    }
    if (form == OutputForm::kRowsF32Activation && activation == Activation::kNone) {
        return write_rows_tile<OutputForm::kRowsF32Activation, Activation::kNone>;
    }
    if (form == OutputForm::kRowsF32Activation && activation == Activation::kGeluErf) {
        return write_rows_tile<OutputForm::kRowsF32Activation, Activation::kGeluErf>;
    }
    return nullptr;
}

constexpr int kCacheKeyGroup = 2;
constexpr int kCacheValueGroup = 3;

[[gnu::always_inline]] inline void store_transposed_rows(const Vector<uint8_t> * __restrict slot, int block,
                                                         Vector<float> column_scale,
                                                         const float * __restrict row_scales, int rows,
                                                         Vector<_Float16> * __restrict tile) {
    Vector<_Float16> vectors[16];
    if (block * 32 + 32 <= rows) {
#pragma clang loop unroll(full)
        for (int quad = 0; quad < 8; quad++) {
            Vector<int32_t> sums[4];
            hmx::load_int32_row_quad(slot, block * 8 + quad, sums);
            const int row = block * 32 + quad * 4;
            vectors[2 * quad] = to_float16(dequantize(sums[0], column_scale, row_scales[row]),
                                           dequantize(sums[1], column_scale, row_scales[row + 1]));
            vectors[2 * quad + 1] = to_float16(dequantize(sums[2], column_scale, row_scales[row + 2]),
                                               dequantize(sums[3], column_scale, row_scales[row + 3]));
        }
    } else {  // padding rows zero: their row scales may not be written
        for (int quad = 0; quad < 8; quad++) {
            Vector<int32_t> sums[4];
            Vector<float> values[4];
            hmx::load_int32_row_quad(slot, block * 8 + quad, sums);
            const int row = block * 32 + quad * 4;
            for (int k = 0; k < 4; k++) {
                values[k] =
                    row + k < rows ? dequantize(sums[k], column_scale, row_scales[row + k]) : Vector<float>::zero();
            }
            vectors[2 * quad] = to_float16(values[0], values[1]);
            vectors[2 * quad + 1] = to_float16(values[2], values[3]);
        }
    }

    transpose_row_pairs_to_lines(vectors);
#pragma clang loop unroll(full)
    for (int k = 0; k < 16; k++) tile[k].bits = vectors[k].bits;
}

inline Vector<_Float16> * key_cache_tile(const Output & output, int head, int key_tile, int d_tile) {
    const int d_tiles = output.cache_head_dimensions / 32;
    const size_t index = (static_cast<size_t>(head) * output.cache_key_tile_stride + key_tile) * d_tiles + d_tile;
    return reinterpret_cast<Vector<_Float16> *>(output.key_cache + index * kTileBytes);
}

inline Vector<_Float16> * value_cache_tile(const Output & output, int head, int key_tile, int d_tile) {
    const int d_tiles = output.cache_head_dimensions / 32;
    const size_t index = (static_cast<size_t>(head) * d_tiles + d_tile) * output.cache_key_tile_stride + key_tile;
    return reinterpret_cast<Vector<_Float16> *>(output.value_cache + index * kTileBytes);
}

struct PairStage {
    const Vector<uint8_t> * first_slot;
    const Vector<uint8_t> * second_slot;
    const float * row_scales;
    Vector<float> first_column_scale;
    Vector<float> second_column_scale;
    _Float16 * columns;
    int stride;
    int first_row;
    const float * cosines;
    const float * sines;
    int rope_stride;
};

inline _Float16 * output_line(const PairStage & pair, int row) {
    return pair.columns + static_cast<size_t>(pair.first_row + row) * pair.stride;
}

[[gnu::always_inline]] inline void rotated_pair(const PairStage & pair, Vector<int32_t> first_sums,
                                                Vector<int32_t> second_sums, int row, Vector<float> & first,
                                                Vector<float> & second) {
    const Vector<float> a = dequantize(first_sums, pair.first_column_scale, pair.row_scales[row]);
    const Vector<float> b = dequantize(second_sums, pair.second_column_scale, pair.row_scales[row]);
    const Vector<float> cosine = hvx::load(pair.cosines + static_cast<size_t>(row) * pair.rope_stride);
    const Vector<float> sine = hvx::load(pair.sines + static_cast<size_t>(row) * pair.rope_stride);
    rotate_pair(a, b, cosine, sine, first, second);
}

[[gnu::always_inline]] inline void write_rotated_row(const PairStage & pair, Vector<int32_t> first_sums,
                                                     Vector<int32_t> second_sums, int row) {
    Vector<float> first;
    Vector<float> second;
    rotated_pair(pair, first_sums, second_sums, row, first, second);
    hvx::vector_at<_Float16>(output_line(pair, row)) = to_float16(first, second);
}

[[gnu::always_inline]] inline void write_plain_row(const PairStage & pair, Vector<int32_t> first_sums,
                                                   Vector<int32_t> second_sums, int row) {
    const float row_scale = pair.row_scales[row];
    hvx::vector_at<_Float16>(output_line(pair, row)) =
        to_float16(dequantize(first_sums, pair.first_column_scale, row_scale),
                   dequantize(second_sums, pair.second_column_scale, row_scale));
}

[[gnu::always_inline]] inline void write_rotated_key_tiles(const PairStage & pair, const Output & output, int rows,
                                                           int head, int key_tile, int d_tile) {
    for (int block = 0; block < 2; block++) {
        Vector<_Float16> * first_tile = key_cache_tile(output, head, key_tile + block, d_tile);
        Vector<_Float16> * second_tile = key_cache_tile(output, head, key_tile + block, d_tile + 1);
        for (int quad = 0; quad < 8; quad++) {
            Vector<int32_t> first_sums[4];
            Vector<int32_t> second_sums[4];
            Vector<float> first[4];
            Vector<float> second[4];
            hmx::load_int32_row_quad(pair.first_slot, block * 8 + quad, first_sums);
            hmx::load_int32_row_quad(pair.second_slot, block * 8 + quad, second_sums);
            const int row = block * 32 + quad * 4;

            if (row + 4 <= rows) {
                rotated_pair(pair, first_sums[0], second_sums[0], row, first[0], second[0]);
                rotated_pair(pair, first_sums[1], second_sums[1], row + 1, first[1], second[1]);
                rotated_pair(pair, first_sums[2], second_sums[2], row + 2, first[2], second[2]);
                rotated_pair(pair, first_sums[3], second_sums[3], row + 3, first[3], second[3]);
            } else {
                for (int k = 0; k < 4; k++) {
                    first[k] = Vector<float>::zero();
                    second[k] = Vector<float>::zero();
                    if (row + k < rows) rotated_pair(pair, first_sums[k], second_sums[k], row + k, first[k], second[k]);
                }
            }

            first_tile[2 * quad] = to_float16_interleaved(first[0], first[1]);
            first_tile[2 * quad + 1] = to_float16_interleaved(first[2], first[3]);
            second_tile[2 * quad] = to_float16_interleaved(second[0], second[1]);
            second_tile[2 * quad + 1] = to_float16_interleaved(second[2], second[3]);
        }
    }
}

template <bool kRotate>
[[gnu::always_inline]] inline void write_pair_rows(const PairStage & pair, int rows) {
    if (rows == 64) {
#pragma clang loop unroll_count(2)
        for (int quad = 0; quad < 16; quad++) {
            Vector<int32_t> first_sums[4];
            Vector<int32_t> second_sums[4];
            hmx::load_int32_row_quad(pair.first_slot, quad, first_sums);
            hmx::load_int32_row_quad(pair.second_slot, quad, second_sums);
            if constexpr (kRotate) {
                write_rotated_row(pair, first_sums[0], second_sums[0], quad * 4);
                write_rotated_row(pair, first_sums[1], second_sums[1], quad * 4 + 1);
                write_rotated_row(pair, first_sums[2], second_sums[2], quad * 4 + 2);
                write_rotated_row(pair, first_sums[3], second_sums[3], quad * 4 + 3);
            } else {
                write_plain_row(pair, first_sums[0], second_sums[0], quad * 4);
                write_plain_row(pair, first_sums[1], second_sums[1], quad * 4 + 1);
                write_plain_row(pair, first_sums[2], second_sums[2], quad * 4 + 2);
                write_plain_row(pair, first_sums[3], second_sums[3], quad * 4 + 3);
            }
        }
        return;
    }

    for (int quad = 0; quad * 4 < rows; quad++) {
        Vector<int32_t> first_sums[4];
        Vector<int32_t> second_sums[4];
        hmx::load_int32_row_quad(pair.first_slot, quad, first_sums);
        hmx::load_int32_row_quad(pair.second_slot, quad, second_sums);
        for (int k = 0; k < 4 && quad * 4 + k < rows; k++) {
            if constexpr (kRotate) {
                write_rotated_row(pair, first_sums[k], second_sums[k], quad * 4 + k);
            } else {
                write_plain_row(pair, first_sums[k], second_sums[k], quad * 4 + k);
            }
        }
    }
}

void write_rows_f16_pair(const OutputContext & context, const OutputItem & item) {
    if (item.slots[1] == nullptr) {
        write_rows_f16_tile(context, item);
        return;
    }

    const Output & output = *context.output;
    const int column_block = item.column_block;
    const int rows = item.rows;
    const int group = column_block * 32 / output.f16_group_width;
    const int first_column = column_block * 32 % output.f16_group_width;

    PairStage pair;
    pair.first_slot = item.slots[0];
    pair.second_slot = item.slots[1];
    pair.row_scales = context.row_scales + item.first_row;
    pair.first_column_scale = hvx::load(context.column_scales + column_block * 32);
    pair.second_column_scale = hvx::load(context.column_scales + column_block * 32 + 32);
    pair.columns = output.f16_groups[group] + first_column;
    pair.stride = output.f16_group_strides[group];
    pair.first_row = item.first_row;

    const bool to_cache = output.key_cache != nullptr && group >= kCacheKeyGroup;
    const int cache_head = to_cache ? first_column / output.cache_head_dimensions : 0;
    const int cache_d_tile = to_cache ? first_column % output.cache_head_dimensions / 32 : 0;
    const int first_key_tile = item.first_row / 32;

    if (to_cache && group == kCacheValueGroup) {
        for (int block = 0; block < 2; block++) {
            store_transposed_rows(pair.first_slot, block, pair.first_column_scale, pair.row_scales, rows,
                                  value_cache_tile(output, cache_head, first_key_tile + block, cache_d_tile));
            store_transposed_rows(pair.second_slot, block, pair.second_column_scale, pair.row_scales, rows,
                                  value_cache_tile(output, cache_head, first_key_tile + block, cache_d_tile + 1));
        }
    } else if ((output.rope_groups >> group) & 1) {
        pair.rope_stride = output.rope_stride;
        const size_t angle_offset = static_cast<size_t>(item.first_row) * pair.rope_stride + (first_column % 128) / 2;
        pair.cosines = output.rope_cos + angle_offset;
        pair.sines = output.rope_sin + angle_offset;
        hvx::prefetch_to_l2(output.rope_cos + static_cast<size_t>(item.first_row) * pair.rope_stride,
                            static_cast<uint32_t>(pair.rope_stride) * 4, static_cast<uint32_t>(pair.rope_stride) * 4,
                            static_cast<uint32_t>(rows));
        if (to_cache) {
            write_rotated_key_tiles(pair, output, rows, cache_head, first_key_tile, cache_d_tile);
        } else {
            write_pair_rows<true>(pair, rows);
        }
    } else {
        write_pair_rows<false>(pair, rows);
    }
}

constexpr int kQueryGroup = 0;
constexpr int kKeyGroup = 1;
constexpr int kValueGroup = 2;

inline Vector<_Float16> * attention72_tile(const Output & output, int group, int head, int d_tile, int block) {
    const int stride = output.attention_key_tile_stride;
    if (group == kQueryGroup) {
        return reinterpret_cast<Vector<_Float16> *>(
            output.attention_q + ((static_cast<size_t>(head) * stride + block) * 3 + d_tile) * kTileBytes);
    }
    if (group == kKeyGroup) {
        return reinterpret_cast<Vector<_Float16> *>(
            output.attention_k + ((static_cast<size_t>(head) * stride + block) * 3 + d_tile) * kTileBytes);
    }
    return reinterpret_cast<Vector<_Float16> *>(
        output.attention_v + ((static_cast<size_t>(head) * 3 + d_tile) * stride + block) * kTileBytes);
}

void write_attention72_tile(const OutputContext & context, const OutputItem & item) {
    const Output & output = *context.output;
    const Vector<uint8_t> * __restrict slot = item.slots[0];
    const int column_block = item.column_block;
    const int first_row = item.first_row;
    const int rows = item.rows;
    const float * __restrict row_scales = context.row_scales + first_row;
    const Vector<float> column_scale = hvx::load(context.column_scales + column_block * 32);
    const Vector<float> bias = column_vector(output.bias, column_block);

    const int group_blocks = output.attention_group_blocks;
    const int heads = group_blocks * 4 / 9;
    const int group = column_block / group_blocks;
    const int block_in_group = column_block % group_blocks;
    const bool shared_block = block_in_group >= 2 * heads;
    const int head = shared_block ? (block_in_group - 2 * heads) * 4 : block_in_group / 2;
    const int d_tile = shared_block ? 2 : block_in_group % 2;

    for (int block = 0; block * 32 < rows; block++) {
        const int token_block = first_row / 32 + block;
        Vector<_Float16> vectors[16];
#pragma clang loop unroll(full)
        for (int quad = 0; quad < 8; quad++) {
            Vector<int32_t> sums[4];
            hmx::load_int32_row_quad(slot, block * 8 + quad, sums);
            const int row = block * 32 + quad * 4;
            const Vector<float> q0 = dequantize_with_bias(sums[0], column_scale, row_scales[row], bias);
            const Vector<float> q1 = dequantize_with_bias(sums[1], column_scale, row_scales[row + 1], bias);
            const Vector<float> q2 = dequantize_with_bias(sums[2], column_scale, row_scales[row + 2], bias);
            const Vector<float> q3 = dequantize_with_bias(sums[3], column_scale, row_scales[row + 3], bias);
            if (group == kKeyGroup) {
                vectors[2 * quad] = to_float16_interleaved(q0, q1);
                vectors[2 * quad + 1] = to_float16_interleaved(q2, q3);
            } else {
                vectors[2 * quad] = to_float16(q0, q1);
                vectors[2 * quad + 1] = to_float16(q2, q3);
            }
        }
        if (group != kKeyGroup) transpose_row_pairs_to_lines(vectors);

        if (!shared_block) {
            Vector<_Float16> * tile = attention72_tile(output, group, head, d_tile, token_block);
#pragma clang loop unroll(full)
            for (int k = 0; k < 16; k++) tile[k] = vectors[k];
        } else if (group == kKeyGroup) {
            const Mask first_32_bytes = Mask::first_bytes(32);
            for (int i = 0; i < 4; i++) {
                Vector<_Float16> * tile = attention72_tile(output, kKeyGroup, head + i, 2, token_block);
#pragma clang loop unroll(full)
                for (int k = 0; k < 16; k++) {
                    tile[k] = select(first_32_bytes, rotate_right_bytes(vectors[k], 32 * i), Vector<_Float16>::zero());
                }
            }
        } else {
            for (int i = 0; i < 4; i++) {
                Vector<_Float16> * tile = attention72_tile(output, group, head + i, 2, token_block);
                tile[0] = vectors[4 * i];
                tile[1] = vectors[4 * i + 1];
                tile[2] = vectors[4 * i + 2];
                tile[3] = vectors[4 * i + 3];
#pragma clang loop unroll(full)
                for (int k = 4; k < 16; k++) tile[k] = Vector<_Float16>::zero();
            }
        }
    }
}

void write_attention64_pair(const OutputContext & context, const OutputItem & item) {
    const Output & output = *context.output;
    const Vector<uint8_t> * __restrict first_slot = item.slots[0];
    const Vector<uint8_t> * __restrict second_slot = item.slots[1];
    const int column_block = item.column_block;
    const int first_row = item.first_row;
    const int rows = item.rows;
    const float * __restrict row_scales = context.row_scales + first_row;
    const Vector<float> first_column_scale = hvx::load(context.column_scales + column_block * 32);
    const Vector<float> second_column_scale = hvx::load(context.column_scales + column_block * 32 + 32);
    const Vector<float> first_bias = column_vector(output.bias, column_block);
    const Vector<float> second_bias = column_vector(output.bias, column_block + 1);

    const int group_blocks = output.attention_group_blocks;
    const int group = column_block / group_blocks;
    const int head = (column_block % group_blocks) / 2;
    const bool normalize = output.qk_norm_parameters != nullptr && group < kValueGroup;
    const bool rotate = output.rope_table != nullptr && group < kValueGroup;
    const float * norm_parameters = normalize ? output.qk_norm_parameters + group * 128 : nullptr;
    const Vector<float> epsilon = Vector<float>::splat(output.qk_norm_epsilon);
    const float * rope_table = output.rope_table;
    const int16_t * __restrict positions = output.rope_positions;
    Vector<float> * __restrict scratch =
        context.thread_scratch->f32 + static_cast<size_t>(item.thread) * kThreadScratchVectors;

    for (int block = 0; block * 32 < rows; block++) {
        const int token_block = first_row / 32 + block;
        for (int quad = 0; quad < 8; quad++) {
            Vector<int32_t> first_sums[4];
            Vector<int32_t> second_sums[4];
            hmx::load_int32_row_quad(first_slot, block * 8 + quad, first_sums);
            hmx::load_int32_row_quad(second_slot, block * 8 + quad, second_sums);
#pragma clang loop unroll(full)
            for (int k = 0; k < 4; k++) {
                const int row = block * 32 + quad * 4 + k;
                Vector<float> first_half =
                    dequantize_with_bias(first_sums[k], first_column_scale, row_scales[row], first_bias);
                Vector<float> second_half =
                    dequantize_with_bias(second_sums[k], second_column_scale, row_scales[row], second_bias);
                if (normalize) layer_norm_head64(first_half, second_half, norm_parameters, epsilon);
                if (rotate) {
                    const float * row_angles = rope_table + 64 * positions[2 * (first_row + row)];
                    const float * column_angles = rope_table + 64 * positions[2 * (first_row + row) + 1];
                    first_half = rotate_half_head(first_half, hvx::load(row_angles), hvx::load(row_angles + 32));
                    second_half =
                        rotate_half_head(second_half, hvx::load(column_angles), hvx::load(column_angles + 32));
                }
                scratch[2 * (quad * 4 + k)] = first_half;
                scratch[2 * (quad * 4 + k) + 1] = second_half;
            }
        }

        for (int d_tile = 0; d_tile < 2; d_tile++) {
            Vector<_Float16> vectors[16];
#pragma clang loop unroll(full)
            for (int i = 0; i < 16; i++) {
                const Vector<float> even_row = scratch[4 * i + d_tile];
                const Vector<float> odd_row = scratch[4 * i + 2 + d_tile];
                vectors[i] =
                    group == kKeyGroup ? to_float16_interleaved(even_row, odd_row) : to_float16(even_row, odd_row);
            }
            if (group != kKeyGroup) transpose_row_pairs_to_lines(vectors);

            const int stride = output.attention_key_tile_stride;
            uint8_t * tile_bytes;
            if (group == kQueryGroup) {
                tile_bytes =
                    output.attention_q + ((static_cast<size_t>(head) * stride + token_block) * 2 + d_tile) * kTileBytes;
            } else if (group == kKeyGroup) {
                tile_bytes =
                    output.attention_k + ((static_cast<size_t>(head) * stride + token_block) * 2 + d_tile) * kTileBytes;
            } else {
                tile_bytes =
                    output.attention_v + ((static_cast<size_t>(head) * 2 + d_tile) * stride + token_block) * kTileBytes;
            }
            auto * tile = reinterpret_cast<Vector<_Float16> *>(tile_bytes);
#pragma clang loop unroll(full)
            for (int k = 0; k < 16; k++) tile[k] = vectors[k];
        }
    }
}

// Stores row quad `quad` of the tile (rows >= `rows` zeroed) and folds its magnitudes into maxima[quad], the running
// max over the tiles: lane 2c + p of its low half for row 4 quad + p, of its high half for row 4 quad + 2 + p (each
// over columns c and c + 16).
[[gnu::always_inline]] inline void store_handover_quad(_Float16 * __restrict tile, Vector<int16_t> * __restrict maxima,
                                                       int quad, int rows, Vector<_Float16> first_rows,
                                                       Vector<_Float16> second_rows) {
    const Vector<_Float16> zero = Vector<_Float16>::zero();
    if (4 * quad + 4 > rows) {
        const int valid = rows - 4 * quad;
        if (valid < 2) first_rows = reinterpret<_Float16>(reinterpret<uint32_t>(first_rows) & 0x0000FFFFu);
        second_rows = valid < 3 ? zero : reinterpret<_Float16>(reinterpret<uint32_t>(second_rows) & 0x0000FFFFu);
    }

    hvx::store(tile + 128 * quad, first_rows);
    hvx::store(tile + 128 * quad + 64, second_rows);

    const Vector<int16_t> magnitude_mask = Vector<int16_t>::splat(0x7fff);
    const Vector<int16_t> first_magnitudes = reinterpret<int16_t>(first_rows) & magnitude_mask;
    const Vector<int16_t> second_magnitudes = reinterpret<int16_t>(second_rows) & magnitude_mask;
    const Vector<int16_t> first_folded = max(first_magnitudes, rotate_right_bytes(first_magnitudes, 64));
    const Vector<int16_t> second_folded = max(second_magnitudes, rotate_right_bytes(second_magnitudes, 64));
    maxima[quad] = max(maxima[quad], select(Mask::first_bytes(64), first_folded, second_folded));
}

template <Activation kActivation>
void write_handover_tile(const OutputContext & context, const OutputItem & item) {
    constexpr bool kGelu = activation_is_gelu(kActivation);
    const Output & output = *context.output;
    const Vector<uint8_t> * __restrict slot = item.slots[0];
    const int column_block = item.column_block;
    const int first_row = item.first_row;
    const int rows = item.rows;
    const int quads = (rows + 3) / 4;
    const size_t tile_index = static_cast<size_t>(first_row / 64) * context.column_blocks + column_block;
    _Float16 * __restrict tile = output.handover->tiles + tile_index * kHandoverTileHalfwords;
    Vector<int16_t> * __restrict maxima =
        context.row_maxima + static_cast<size_t>(item.thread) * context.row_quads + first_row / 4;
    const float * __restrict row_scales = context.row_scales + first_row;
    Vector<_Float16> * __restrict gathered =
        kGelu ? context.gelu_table->gather_slots + static_cast<size_t>(item.thread) * kGatherSlotsPerThread : nullptr;
    Vector<_Float16> * __restrict positive =
        context.thread_scratch->f16 + static_cast<size_t>(item.thread) * kThreadScratchVectors;
    const Vector<_Float16> zero = Vector<_Float16>::zero();
    const Vector<float> column_scale = hvx::load(context.column_scales + column_block * 32);
    const Vector<float> bias = column_vector(output.bias, column_block);
    const Vector<float> output_scale = hvx::load(output.column_scale + column_block * 32);
    const Vector<_Float16> output_scale_f16 = to_float16_interleaved(output_scale, output_scale);
    const uint32_t table = reinterpret_cast<uint32_t>(context.gelu_table->values);

    for (int quad = 0; quad < quads; quad++) {
        Vector<int32_t> sums[4];
        hmx::load_int32_row_quad(slot, quad, sums);
        const Vector<float> y0 = dequantize_with_bias(sums[0], column_scale, row_scales[4 * quad], bias);
        const Vector<float> y1 = dequantize_with_bias(sums[1], column_scale, row_scales[4 * quad + 1], bias);
        const Vector<float> y2 = dequantize_with_bias(sums[2], column_scale, row_scales[4 * quad + 2], bias);
        const Vector<float> y3 = dequantize_with_bias(sums[3], column_scale, row_scales[4 * quad + 3], bias);

        if constexpr (kActivation == Activation::kReluSquared) {
            const Vector<float> r0 = activate_and_scale<kActivation>(y0, output_scale);
            const Vector<float> r1 = activate_and_scale<kActivation>(y1, output_scale);
            const Vector<float> r2 = activate_and_scale<kActivation>(y2, output_scale);
            const Vector<float> r3 = activate_and_scale<kActivation>(y3, output_scale);
            store_handover_quad(tile, maxima, quad, rows, to_float16_interleaved(r0, r1),
                                to_float16_interleaved(r2, r3));
        } else {
            const Vector<_Float16> first_rows = to_float16_interleaved(y0, y1);
            const Vector<_Float16> second_rows = to_float16_interleaved(y2, y3);
            gather_halfwords(gathered + 2 * quad, table, 65535, reinterpret<int16_t>(first_rows) << 1);
            gather_halfwords(gathered + 2 * quad + 1, table, 65535, reinterpret<int16_t>(second_rows) << 1);
            positive[2 * quad] = max(first_rows, zero);
            positive[2 * quad + 1] = max(second_rows, zero);
        }
    }

    if constexpr (kGelu) {
        for (int quad = 0; quad < quads; quad++) {
            const Vector<_Float16> first_rows = (gathered[2 * quad] + positive[2 * quad]) * output_scale_f16;
            const Vector<_Float16> second_rows = (gathered[2 * quad + 1] + positive[2 * quad + 1]) * output_scale_f16;
            store_handover_quad(tile, maxima, quad, rows, first_rows, second_rows);
        }
    }

    for (int quad = quads; quad < 16; quad++) {
        hvx::store(tile + 128 * quad, zero);
        hvx::store(tile + 128 * quad + 64, zero);
    }
}

OutputStage find_handover_output_stage(Activation activation) {
    switch (activation) {
        case Activation::kNone:
            return nullptr;
        case Activation::kReluSquared:
            return write_handover_tile<Activation::kReluSquared>;
        case Activation::kGeluTanh:  // both GELU forms run through the GELU table (the table differs)
        case Activation::kGeluErf:
            return write_handover_tile<Activation::kGeluTanh>;
    }
    return nullptr;
}

void finish_handover_row_scales(const OutputContext & context, int thread) {
    HandoverTiles & handover = *context.output->handover;

    for (int quad = thread; quad < context.row_quads; quad += runtime::kHvxThreads) {
        Vector<int16_t> maximum = context.row_maxima[quad];
        for (int worker = 1; worker < runtime::kHvxThreads; worker++) {
            maximum = max(maximum, context.row_maxima[static_cast<size_t>(worker) * context.row_quads + quad]);
        }

        // lanes 0, 1 / 32, 33: the maxima of rows 4 quad, 4 quad + 1 / 4 quad + 2, 4 quad + 3
        for (int bytes = 4; bytes <= 32; bytes <<= 1) maximum = max(maximum, rotate_right_bytes(maximum, bytes));
        alignas(128) uint32_t words[32];
        hvx::vector_at<int16_t>(words) = maximum;

        for (int row = 0; row < 4; row++) {
            const auto amax_bits = static_cast<uint16_t>(words[row / 2 * 16] >> (16 * (row % 2)));
            const runtime::Int8Scale scale = context.int8_scales[amax_bits];
            handover.inverse_bits[quad * 4 + row] = scale.inverse_bits;
            handover.row_scales[quad * 4 + row] = scale.scale;
        }
    }
}

// The max of non-negative f32 lanes, compared as integers.
float max_nonnegative_lane(Vector<int32_t> v) {
    alignas(128) int32_t lanes[32];
    hvx::vector_at<int32_t>(lanes) = v;
    int32_t maximum = 0;
    for (int i = 0; i < 32; i++) maximum = lanes[i] > maximum ? lanes[i] : maximum;
    return std::bit_cast<float>(maximum);
}

uint8_t * activation_line(W8a8GemmJob & job, int row_quad) {
    const int row_block = row_quad / 16;
    const int line = row_quad % 16;
    return job.activation_buffers[job.quantized_buffer] + static_cast<size_t>(row_block) * job.k_tiles * kTileBytes +
           line * 128;
}

// fp16 rows: one 2D fetch from the first mapped row (the map is ascending, gaps only between images).
void prefetch_next_quad(const W8a8GemmJob & job, int row_quad) {
    const bool f16_rows = job.input.rows_f16 != nullptr;
    const int next_row = job.quantized_first_row + (row_quad + 1) * 4;
    if (row_quad + 1 >= job.quantized_row_blocks * 16 || next_row >= job.M) return;

    const int rows = job.M - next_row < 4 ? job.M - next_row : 4;
    const auto K = static_cast<uint32_t>(job.K);
    if (f16_rows) {
        const int source_row = job.input.row_map != nullptr ? job.input.row_map[next_row] : next_row;
        const _Float16 * row = job.input.rows_f16 + static_cast<size_t>(source_row) * job.input.row_stride;
        hvx::prefetch_to_l2(row, K * 2, static_cast<uint32_t>(job.input.row_stride) * 2, static_cast<uint32_t>(rows));
    } else {
        hvx::prefetch_to_l2(job.input.rows_f32 + static_cast<size_t>(next_row) * job.K, K * 4, K * 4,
                            static_cast<uint32_t>(rows));
    }
}

void quantize_f16_row_quad(W8a8GemmJob & job, int row_quad, int thread) {
    const int K = job.K;
    const int vectors_per_row = K / 64;
    const int first_row = job.quantized_first_row;
    const Vector<int16_t> magnitude_mask = Vector<int16_t>::splat(0x7fff);
    const Vector<_Float16> * rows[4];
    uint16_t inverse_bits[4];
    uint16_t amax_bits[4];
    Vector<int16_t> maxima[4];

    for (int k = 0; k < 4; k++) {
        const int row = first_row + row_quad * 4 + k;
        rows[k] = nullptr;
        maxima[k] = Vector<int16_t>::zero();
        if (row >= job.M) continue;

        const int source_row = job.input.row_map != nullptr ? job.input.row_map[row] : row;
        const _Float16 * source = job.input.rows_f16 + static_cast<size_t>(source_row) * job.input.row_stride;
        Vector<_Float16> * scaled = job.f16_input_scratch + (static_cast<size_t>(thread) * 4 + k) * vectors_per_row;
        Vector<int16_t> maximum = Vector<int16_t>::zero();
        for (int vector = 0; vector < vectors_per_row; vector++) {
            const Vector<_Float16> value = hvx::load(source + 64 * vector) * job.f16_column_scale[vector];
            scaled[vector] = value;
            maximum = max(maximum, reinterpret<int16_t>(value) & magnitude_mask);
        }
        maxima[k] = maximum;
        rows[k] = scaled;
    }

    hvx::max_f16_magnitude_bits_of_four(maxima[0], maxima[1], maxima[2], maxima[3], amax_bits);
    for (int k = 0; k < 4; k++) {
        const runtime::Int8Scale scale = job.int8_scales[amax_bits[k]];
        inverse_bits[k] = scale.inverse_bits;
        job.row_scales[first_row + row_quad * 4 + k] = scale.scale;
    }

    hvx::quantize_four_rows_f16(rows, inverse_bits, K, activation_line(job, row_quad));
}

inline Vector<float> scaled_row_values(const float * row, Vector<float> inverse, int k_tile, Vector<float> zero) {
    return row != nullptr ? hvx::load(row + k_tile * 32) * inverse : zero;
}

void quantize_f32_row_quad(W8a8GemmJob & job, int row_quad) {
    const int K = job.K;
    const int first_row = job.quantized_first_row;
    const Vector<int32_t> magnitude_mask = Vector<int32_t>::splat(0x7fffffff);
    const Vector<float> zero = Vector<float>::zero();
    const float * rows[4];
    Vector<float> inverse[4];

    for (int k = 0; k < 4; k++) {
        const int row = first_row + row_quad * 4 + k;
        rows[k] = nullptr;
        float amax = 0.0f;
        if (row < job.M) {
            rows[k] = job.input.rows_f32 + static_cast<size_t>(row) * K;
            Vector<int32_t> maximum = Vector<int32_t>::zero();
            for (int column = 0; column < K; column += 32) {
                const Vector<float> values = hvx::load(rows[k] + column);
                maximum = max(maximum, reinterpret<int32_t>(values) & magnitude_mask);
            }
            amax = max_nonnegative_lane(maximum);
        }
        const float scale = amax / 127.0f;
        inverse[k] = Vector<float>::splat(scale > 0.0f ? 1.0f / scale : 0.0f);
        job.row_scales[first_row + row_quad * 4 + k] = scale;
    }

    uint8_t * line = activation_line(job, row_quad);
    for (int k_tile = 0; k_tile < job.k_tiles; k_tile++) {
        hvx::vector_at<uint8_t>(line + static_cast<size_t>(k_tile) * kTileBytes) = hvx::int8_activation_line(
            scaled_row_values(rows[0], inverse[0], k_tile, zero), scaled_row_values(rows[1], inverse[1], k_tile, zero),
            scaled_row_values(rows[2], inverse[2], k_tile, zero), scaled_row_values(rows[3], inverse[3], k_tile, zero));
    }
}

struct HandoverItem {
    int row_block;
    int first_k_tile;
    int k_tiles;
    int global_row_block;
};

HandoverItem handover_item(const W8a8GemmJob & job, int item) {
    const int groups = (job.k_tiles + kHandoverTilesPerItem - 1) / kHandoverTilesPerItem;
    HandoverItem result;
    result.row_block = item / groups;
    result.first_k_tile = item % groups * kHandoverTilesPerItem;
    result.k_tiles = job.k_tiles - result.first_k_tile < kHandoverTilesPerItem ? job.k_tiles - result.first_k_tile
                                                                               : kHandoverTilesPerItem;
    result.global_row_block = job.quantized_first_row / 64 + result.row_block;
    return result;
}

Vector<_Float16> * handover_staging_buffer(const W8a8GemmJob & job, int thread, int buffer) {
    return job.handover_staging +
           static_cast<size_t>(thread * 2 + buffer) * (kHandoverStagingBytes / hvx::kVectorBytes);
}

void fetch_handover_item(W8a8GemmJob & job, int item, int thread, int buffer) {
    const HandoverItem tiles = handover_item(job, item);
    const _Float16 * source =
        job.input.handover->tiles +
        (static_cast<size_t>(tiles.global_row_block) * job.k_tiles + tiles.first_k_tile) * kHandoverTileHalfwords;
    const auto bytes = static_cast<uint32_t>(tiles.k_tiles * kHandoverTileBytes);
    job.handover_rings[thread]->push(handover_staging_buffer(job, thread, buffer), source, bytes, 1, bytes, bytes);
}

void convert_handover_item(W8a8GemmJob & job, int item, int thread, int buffer) {
    const HandoverItem tiles = handover_item(job, item);
    const Vector<_Float16> * source = handover_staging_buffer(job, thread, buffer);
    uint8_t * destination = job.activation_buffers[job.quantized_buffer] +
                            (static_cast<size_t>(tiles.row_block) * job.k_tiles + tiles.first_k_tile) * kTileBytes;
    const uint16_t * inverse_bits = job.input.handover->inverse_bits + static_cast<size_t>(tiles.global_row_block) * 64;
    const Vector<uint8_t> offset_128 = Vector<uint8_t>::splat(0x80);
    const int k_tiles = tiles.k_tiles;

    for (int quad = 0; quad < 16; quad++) {
        const Vector<_Float16> first_inverse = reinterpret<_Float16>(Vector<uint32_t>::splat(
            inverse_bits[4 * quad] | (static_cast<uint32_t>(inverse_bits[4 * quad + 1]) << 16)));
        const Vector<_Float16> second_inverse = reinterpret<_Float16>(Vector<uint32_t>::splat(
            inverse_bits[4 * quad + 2] | (static_cast<uint32_t>(inverse_bits[4 * quad + 3]) << 16)));

        int k = 0;
        for (; k + 2 <= k_tiles; k += 2) {
            const Vector<_Float16> * x = source + static_cast<size_t>(k) * 32 + 2 * quad;
            const Vector<_Float16> a0 = x[0] * first_inverse;
            const Vector<_Float16> a1 = x[1] * second_inverse;
            const Vector<_Float16> b0 = x[32] * first_inverse;
            const Vector<_Float16> b1 = x[33] * second_inverse;
            hvx::vector_at<uint8_t>(destination + static_cast<size_t>(k) * kTileBytes + quad * 128) =
                reinterpret<uint8_t>(round_to_int8_interleaved(a0, a1)) ^ offset_128;
            hvx::vector_at<uint8_t>(destination + static_cast<size_t>(k + 1) * kTileBytes + quad * 128) =
                reinterpret<uint8_t>(round_to_int8_interleaved(b0, b1)) ^ offset_128;
        }
        for (; k < k_tiles; k++) {
            const Vector<_Float16> * x = source + static_cast<size_t>(k) * 32 + 2 * quad;
            const Vector<int8_t> codes = round_to_int8_interleaved(x[0] * first_inverse, x[1] * second_inverse);
            hvx::vector_at<uint8_t>(destination + static_cast<size_t>(k) * kTileBytes + quad * 128) =
                reinterpret<uint8_t>(codes) ^ offset_128;
        }
    }
}

void quantize_row_quad(W8a8GemmJob & job, int row_quad, int thread) {
    prefetch_next_quad(job, row_quad);
    if (job.input.rows_f16 != nullptr) {
        quantize_f16_row_quad(job, row_quad, thread);
        return;
    }
    quantize_f32_row_quad(job, row_quad);
}

int input_items(const W8a8GemmJob & job) {
    if (job.input.handover != nullptr) {
        return job.quantized_row_blocks * ((job.k_tiles + kHandoverTilesPerItem - 1) / kHandoverTilesPerItem);
    }
    return job.quantized_row_blocks * 16;
}

void run_input_stage(W8a8GemmJob & job, int thread) {
    const int items = input_items(job);

    if (job.input.handover == nullptr) {
        while (true) {
            const int row_quad = job.next_quad.fetch_add(1);
            if (row_quad >= items) return;
            quantize_row_quad(job, row_quad, thread);
        }
    }

    int pending = -1;
    for (int fetched = 0;; fetched++) {
        int item = job.next_quad.fetch_add(1);
        if (item >= items) item = -1;
        if (item >= 0) fetch_handover_item(job, item, thread, fetched & 1);
        if (pending >= 0) {
            job.handover_rings[thread]->pop();
            convert_handover_item(job, pending, thread, (fetched - 1) & 1);
        }
        if (item < 0) break;
        pending = item;
    }
}

int chunk_row_blocks(const W8a8GemmJob & job, int first_row) {
    const int rows = job.M - first_row < job.chunk_rows ? job.M - first_row : job.chunk_rows;
    return (rows + 63) / 64;
}

void push_weight_chunk(const W8a8GemmJob & job, runtime::DmaRing & ring, int chunk, int buffer) {
    const int first_block = chunk * job.weight_chunk_column_blocks;
    const int remaining = job.column_blocks - first_block;
    const int blocks = remaining < job.weight_chunk_column_blocks ? remaining : job.weight_chunk_column_blocks;
    const size_t block_bytes = job.column_block_bytes;

    if (job.deep_weights) {
        for (int i = 0; i < blocks; i++) {
            uint8_t * destination = job.weight_buffers[buffer] + static_cast<size_t>(i / 2) * 2 * block_bytes +
                                    static_cast<size_t>(i & 1) * kWeightTileBytes;
            const uint8_t * source = job.weight_stream + static_cast<size_t>(first_block + i) * block_bytes;
            ring.push(destination, source, kWeightTileBytes, static_cast<uint32_t>(job.k_tiles) + 1,
                      2 * kWeightTileBytes, kWeightTileBytes);
        }
        return;
    }

    const auto bytes = static_cast<uint32_t>(static_cast<size_t>(blocks) * block_bytes);
    ring.push(job.weight_buffers[buffer], job.weight_stream + static_cast<size_t>(first_block) * block_bytes, bytes, 1,
              bytes, bytes);
}

void run_hmx_chunk(W8a8GemmJob & job, runtime::DmaRing & ring) {
    const int k_tiles = job.k_tiles;
    const int row_blocks = job.chunk_row_blocks;
    const int chunk_blocks = job.weight_chunk_column_blocks;
    const int weight_chunks = (job.column_blocks + chunk_blocks - 1) / chunk_blocks;
    const size_t block_bytes = job.column_block_bytes;
    const int tiles_per_load = hmx::tiles_per_load(k_tiles);
    const bool deep = job.deep_weights;
    const int slots = job.staging_slots;
    const int group = job.column_group;

    push_weight_chunk(job, ring, 0, 0);
    for (int chunk = 0; chunk < weight_chunks; chunk++) {
        if (chunk + 1 < weight_chunks) push_weight_chunk(job, ring, chunk + 1, (chunk + 1) & 1);
        const int remaining = job.column_blocks - chunk * chunk_blocks;
        const int blocks = remaining < chunk_blocks ? remaining : chunk_blocks;
        for (int i = deep ? blocks : 1; i > 0; i--) ring.pop();

        for (int group_first = 0; group_first < blocks; group_first += group) {
            for (int group_tile = 0; group_tile < row_blocks * group; group_tile++) {
                const int row_block = group_tile / group;
                const int block_in_chunk = group_first + group_tile % group;
                const int column_block = chunk * chunk_blocks + block_in_chunk;
                const uint8_t * weights =
                    deep ? job.weight_buffers[chunk & 1] + static_cast<size_t>(block_in_chunk / 2) * 2 * block_bytes +
                               static_cast<size_t>(block_in_chunk & 1) * kWeightTileBytes
                         : job.weight_buffers[chunk & 1] + static_cast<size_t>(block_in_chunk) * block_bytes;
                const uint8_t * output_blocks =
                    weights + static_cast<size_t>(k_tiles) * (deep ? 2 * kWeightTileBytes : kWeightTileBytes);
                const int tile = (column_block / group) * group * row_blocks + group_tile;
                const int slot = tile % slots;
                while (tile >= slots && job.slot_done[slot].load() < tile - slots + 1) {}
                Vector<uint8_t> * staging = job.staging + static_cast<size_t>(slot) * kStagingSlotVectors;
                const uint8_t * activations = job.activations + static_cast<size_t>(row_block) * k_tiles * kTileBytes;

                hmx::clear_integer_accumulator();
                for (int first = 0; first < k_tiles; first += tiles_per_load) {
                    const int tiles = k_tiles - first < tiles_per_load ? k_tiles - first : tiles_per_load;
                    const uint8_t * activation_tiles = activations + static_cast<size_t>(first) * kTileBytes;
                    if (deep) {
                        hmx::multiply_uint8_int8_deep_weights(
                            activation_tiles, weights + static_cast<size_t>(first) * 2 * kWeightTileBytes,
                            tiles * kTileBytes - 1);
                    } else {
                        hmx::multiply_uint8_int8(activation_tiles,
                                                 weights + static_cast<size_t>(first) * kWeightTileBytes,
                                                 tiles * kTileBytes - 1);
                    }
                }
                hmx::store_int32_as_uint8_slices(staging, output_blocks);
                job.produced.store(tile + 1);
            }
        }
    }
}

// The DMA reads the table through the cache: the CPU wrote it.
void copy_rope_table(const W8a8GemmJob & job, runtime::DmaRing & ring) {
    if (job.rope_bytes == 0) return;
    ring.set_bypass(runtime::DmaBypass::kDestination);
    ring.push(job.rope_vtcm, job.rope_source, job.rope_bytes, 1, job.rope_bytes, job.rope_bytes);
    ring.set_bypass(runtime::DmaBypass::kBoth);
    ring.pop();
}

void fetch_prequantized_tiles(const W8a8GemmJob & job, runtime::DmaRing & ring, int first_row) {
    const auto bytes =
        static_cast<uint32_t>(static_cast<size_t>(chunk_row_blocks(job, first_row)) * job.k_tiles * kTileBytes);
    if (job.prequantized_through_cache) ring.set_bypass(runtime::DmaBypass::kDestination);
    ring.push(job.activation_buffers[0],
              job.input.tiles + static_cast<size_t>(first_row / 64) * job.k_tiles * kTileBytes, bytes, 1, bytes, bytes);
    ring.set_bypass(runtime::DmaBypass::kBoth);
    ring.pop();
}

void begin_chunk(W8a8GemmJob & job, int chunk, int chunks, int first_row) {
    job.chunk_first_row = first_row;
    job.chunk_row_blocks = chunk_row_blocks(job, first_row);
    job.activations = job.input.tiles != nullptr ? job.activation_buffers[0] : job.activation_buffers[chunk & 1];
    job.produced.store(0);
    job.next_item.store(0);
#pragma clang loop vectorize(disable)
    for (int slot = 0; slot < job.staging_slots; slot++) job.slot_done[slot].store(0);

    job.quantized_row_blocks = 0;
    if (job.pipelined && chunk + 1 < chunks) {
        job.quantized_first_row = first_row + job.chunk_rows;
        job.quantized_row_blocks = chunk_row_blocks(job, first_row + job.chunk_rows);
        job.quantized_buffer = (chunk + 1) & 1;
    }
    job.next_quad.store(0);
}

void begin_input_stage(W8a8GemmJob & job, int chunk, int first_row) {
    job.quantized_first_row = first_row;
    job.quantized_row_blocks = chunk_row_blocks(job, first_row);
    job.quantized_buffer = chunk & 1;
    job.next_quad.store(0);
}

// The HMX driver is a fifth thread that must never execute an HVX instruction (the four HVX contexts belong to the
// workers): no memset, no vectorized loops.
[[gnu::noinline]] void gemm_driver_thread(void * job_pointer) {
    W8a8GemmJob & job = *static_cast<W8a8GemmJob *>(job_pointer);
    runtime::DmaRing & ring = *job.driver_ring;
    ring.initialize_with_scalar_stores(runtime::DmaBypass::kBoth);
    copy_rope_table(job, ring);
    hmx::lock(*job.session);
    job.barrier.wait();

    const int chunks = (job.M + job.chunk_rows - 1) / job.chunk_rows;
    for (int chunk = 0; chunk < chunks; chunk++) {
        const int first_row = chunk * job.chunk_rows;
        if (job.input.tiles != nullptr) {
            fetch_prequantized_tiles(job, ring, first_row);
            job.barrier.wait();
        } else if (!job.pipelined || chunk == 0) {
            begin_input_stage(job, chunk, first_row);
            job.barrier.wait();
            job.barrier.wait();  // (the workers quantize)
        }

        begin_chunk(job, chunk, chunks, first_row);
        job.barrier.wait();
        run_hmx_chunk(job, ring);
        job.barrier.wait();
    }

    hmx::unlock(*job.session);
    ring.drain();
}

void write_output_item(W8a8GemmJob & job, int tile, int thread) {
    const int slots = job.staging_slots;
    int column_block;
    int row_block;
    tile_position(job, tile, &column_block, &row_block);

    OutputItem item;
    item.slots[0] = job.staging + static_cast<size_t>(tile % slots) * kStagingSlotVectors;
    item.slots[1] =
        job.column_pairs ? job.staging + static_cast<size_t>((tile + 1) % slots) * kStagingSlotVectors : nullptr;
    item.column_block = column_block;
    item.first_row = job.chunk_first_row + row_block * 64;
    item.rows = job.M - item.first_row < 64 ? job.M - item.first_row : 64;
    item.thread = thread;
    job.output_stage(job.output_context, item);

    for (int k = 0; k <= job.column_pairs; k++) job.slot_done[(tile + k) % slots].store(tile + k + 1);
}

void take_chunk_items(W8a8GemmJob & job, int thread) {
    const int total = job.column_blocks * job.chunk_row_blocks;
    const int input_total = input_items(job);

    while (true) {
        int item = job.next_item.load();
        if (item < total && job.produced.load() > item + job.column_pairs) {
            if (job.next_item.compare_exchange_strong(item, item + 1 + job.column_pairs)) {
                write_output_item(job, item, thread);
            }
            continue;
        }

        if (job.next_quad.load() < input_total) {
            const int row_quad = job.next_quad.fetch_add(1);
            if (row_quad < input_total) {
                quantize_row_quad(job, row_quad, thread);  // (pipelined chunks: fp32 / fp16 input rows only)
            }
            continue;
        }

        if (item >= total) return;
    }
}

void gemm_worker_thread(void * job_pointer, int thread) {
    W8a8GemmJob & job = *static_cast<W8a8GemmJob *>(job_pointer);
    if (job.row_maxima != nullptr) {
        for (int quad = 0; quad < job.row_quads; quad++) {
            job.row_maxima[static_cast<size_t>(thread) * job.row_quads + quad] = Vector<int16_t>::zero();
        }
    }
    if (job.input.handover != nullptr) job.handover_rings[thread]->initialize(runtime::DmaBypass::kBoth);
    job.barrier.wait();

    const int chunks = (job.M + job.chunk_rows - 1) / job.chunk_rows;
    for (int chunk = 0; chunk < chunks; chunk++) {
        if (job.input.tiles != nullptr) {
            job.barrier.wait();
        } else if (!job.pipelined || chunk == 0) {
            job.barrier.wait();
            run_input_stage(job, thread);
            job.barrier.wait();
        }

        job.barrier.wait();
        take_chunk_items(job, thread);
        job.barrier.wait();
    }

    if (job.row_maxima != nullptr) finish_handover_row_scales(job.output_context, thread);
}

runtime::Status run_job(W8a8GemmJob & job) {
    job.barrier.reset(runtime::kHvxThreads + 1);
    return runtime::run_parallel_with_driver(*job.session, runtime::kHvxThreads, gemm_worker_thread, gemm_driver_thread,
                                             &job);
}

static_assert(std::is_trivially_destructible_v<W8a8GemmJob>, "freed without a destructor call");

runtime::HeapObject<W8a8GemmJob> allocate_job() {
    runtime::HeapObject<W8a8GemmJob> job = runtime::allocate_object<W8a8GemmJob>(128);
    if (job) new (job.get()) W8a8GemmJob();
    return job;
}

void reset_job(W8a8GemmJob & job) {
    new (&job) W8a8GemmJob();
}

constexpr size_t kRopeTableVtcmLimit = 512 << 10;
constexpr size_t kPrequantizedThroughCacheBytes = 5 << 20;

struct Workspace {
    runtime::HeapObject<W8a8GemmJob> job;
    runtime::HeapObject<runtime::DmaRing> driver_ring;
    runtime::HeapObject<runtime::DmaRing> handover_rings[runtime::kHvxThreads];
    runtime::HeapObject<ThreadScratch> thread_scratch;
    runtime::GrowingArray<float> row_scales;
    runtime::GrowingArray<Vector<_Float16>> f16_column_scale;
    runtime::HeapArray<Vector<_Float16>> gelu_tables[2];
    std::atomic<int> in_use;
};

runtime::Status allocate_workspace(Workspace & workspace) {
    if (!workspace.job) workspace.job = allocate_job();
    if (!workspace.driver_ring) workspace.driver_ring = runtime::allocate_object<runtime::DmaRing>(64);
    if (!workspace.thread_scratch) workspace.thread_scratch = runtime::allocate_object<ThreadScratch>(128);
    const size_t ring_bytes = (sizeof(runtime::DmaRing) + 255) & ~static_cast<size_t>(255);
    for (auto & ring : workspace.handover_rings) {
        if (!ring) ring = runtime::allocate_object<runtime::DmaRing>(256, ring_bytes);
    }

    bool complete = workspace.job && workspace.driver_ring && workspace.thread_scratch;
    for (const auto & ring : workspace.handover_rings) complete = complete && ring;
    if (!complete) return std::unexpected(runtime::Error::kNoMemory);
    return {};
}

bool arguments_supported(const Arguments & arguments) {
    const Input & input = arguments.input;
    const Output & output = arguments.output;
    if (arguments.M < 1 || arguments.K < 1 || arguments.K % 32 != 0 || arguments.N < 1 || arguments.N % 32 != 0 ||
        arguments.weights.stream == nullptr || arguments.weights.column_scales == nullptr) {
        return false;
    }

    bool valid = true;
    switch (input.form) {
        case InputForm::kRowsF32:
            valid = input.rows_f32 != nullptr;
            break;
        case InputForm::kRowsF16Scaled:
            valid = input.rows_f16 != nullptr && input.column_scale != nullptr;
            break;
        case InputForm::kPrequantizedTiles:
            valid = input.tiles != nullptr && input.tile_row_scales != nullptr;
            break;
        case InputForm::kHandoverTiles:
            valid = input.handover != nullptr;
            break;
    }

    const int inputs = (input.rows_f32 != nullptr) + (input.rows_f16 != nullptr) + (input.tiles != nullptr) +
                       (input.handover != nullptr);
    valid = valid && inputs == 1;

    switch (output.form) {
        case OutputForm::kRowsF32Activation:
        case OutputForm::kRowsF32Accumulate:
            valid = valid && output.rows_f32 != nullptr;
            break;
        case OutputForm::kRowsF16:
            valid = valid && output.f16_groups[0] != nullptr && output.f16_group_width > 0 && output.bias == nullptr;
            break;
        case OutputForm::kHandoverTiles:
            valid = valid && output.handover != nullptr && output.column_scale != nullptr;
            break;
        case OutputForm::kAttentionTiles72:
        case OutputForm::kAttentionTiles64:
            valid = valid && output.attention_q != nullptr && output.attention_k != nullptr &&
                    output.attention_v != nullptr;
            break;
    }
    return valid;
}

bool output_uses_column_pairs(OutputForm form) {
    return form == OutputForm::kRowsF16 || form == OutputForm::kAttentionTiles64;
}

bool output_prefers_row_groups(OutputForm form) {
    return form == OutputForm::kRowsF32Accumulate || form == OutputForm::kRowsF16;
}

constexpr int kGeluTableVectors = kGeluTableEntries / 64;
constexpr int kFirstMagnitudeOf16 = 0x4C00;  // fp16 bits of 16.0

runtime::HeapArray<Vector<_Float16>> build_gelu_table(Activation activation) {
    runtime::HeapArray<Vector<_Float16>> table = runtime::allocate_array<Vector<_Float16>>(kGeluTableVectors);
    if (!table) return table;

    alignas(128) int16_t lane_indices[64];
    for (int i = 0; i < 64; i++) lane_indices[i] = static_cast<int16_t>(i);
    const Vector<int16_t> lanes = hvx::load(lane_indices);
    const bool tanh_form = activation == Activation::kGeluTanh;

    for (int vector = 0; vector < kGeluTableVectors; vector++) {
        if (vector * 64 >= kFirstMagnitudeOf16) {
            table[vector] = Vector<_Float16>::zero();
            continue;
        }

        const Vector<_Float16> negative_values = reinterpret<_Float16>(
            (lanes + Vector<int16_t>::splat(vector * 64)) | Vector<int16_t>::splat(static_cast<int16_t>(0x8000)));
        const VectorPair<float> values = to_float32(negative_values);
        const Vector<float> low =
            tanh_form ? vector_math::gelu_tanh(values.low()) : vector_math::gelu_erf(values.low());
        const Vector<float> high =
            tanh_form ? vector_math::gelu_tanh(values.high()) : vector_math::gelu_erf(values.high());
        table[vector] = to_float16(low, high);
    }
    return table;
}

runtime::Status copy_gelu_table(Workspace & workspace, Activation activation, _Float16 * vtcm_table) {
    runtime::HeapArray<Vector<_Float16>> & table = workspace.gelu_tables[activation == Activation::kGeluTanh ? 0 : 1];
    if (!table) table = build_gelu_table(activation);
    if (!table) return std::unexpected(runtime::Error::kNoMemory);
    for (int vector = 0; vector < kGeluTableVectors; vector++) hvx::store(vtcm_table + 64 * vector, table[vector]);
    return {};
}

constexpr int kStagingSlots = 16;
constexpr size_t kDefaultWeightChunkBytes = 256 << 10;
constexpr size_t kRowGroupWeightBytes = 512 << 10;
constexpr size_t kPairGroupWeightBytes = 640 << 10;

size_t align_128(size_t bytes) {
    return (bytes + 127) & ~static_cast<size_t>(127);
}

size_t align_4096(size_t bytes) {
    return (bytes + 4095) & ~static_cast<size_t>(4095);
}

// Column-block groups never straddle a weight chunk: a group of several blocks is the weight chunk.
void choose_column_groups(W8a8GemmJob & job) {
    job.column_group = 1;

    if (job.column_pairs) {
        int group = job.prefer_row_groups ? static_cast<int>(kPairGroupWeightBytes / job.column_block_bytes) & ~1 : 2;
        if (group > job.column_blocks) group = job.column_blocks & ~1;
        while (group > 2 && job.column_blocks % group != 0) group -= 2;
        if (group < 2 || job.column_blocks % group != 0) {
            job.column_pairs = 0;
        } else {
            job.column_group = group;
            job.weight_chunk_column_blocks = group;
        }
    } else if (job.prefer_row_groups) {
        int group = static_cast<int>(kRowGroupWeightBytes / job.column_block_bytes);
        if (group > job.column_blocks) group = job.column_blocks;
        while (group > 1 && job.column_blocks % group != 0) group--;
        if (group > 1) {
            job.column_group = group;
            job.weight_chunk_column_blocks = group;
        }
    }
}

// VTCM at the top for the call: fp16 input scratch rows (or the hand-over staging), then the row maxima or the RoPE
// table, then the GELU table.
struct ReservedVtcm {
    size_t input_scratch;
    size_t rope_table;  // 0: the table stays in DDR
    size_t total;
};

ReservedVtcm reserve_vtcm(const W8a8GemmJob & job, bool gelu, bool handover_output) {
    const Output & output = job.output;

    size_t rope_bytes = 0;
    if (output.form == OutputForm::kRowsF16 && output.rope_groups != 0 && output.rope_cos != nullptr &&
        output.rope_sin > output.rope_cos) {
        rope_bytes = static_cast<size_t>(job.M) * output.rope_stride * 4;
    } else if (output.form == OutputForm::kAttentionTiles64 && output.rope_table != nullptr) {
        rope_bytes = static_cast<size_t>(output.rope_table_positions) * 256;
    }
    ReservedVtcm reserved;
    reserved.rope_table = rope_bytes != 0 && rope_bytes <= kRopeTableVtcmLimit ? align_128(rope_bytes) : 0;

    if (job.input.rows_f16 != nullptr) {
        reserved.input_scratch = align_128(static_cast<size_t>(runtime::kHvxThreads) * 4 * job.K * 2);
    } else if (job.input.handover != nullptr) {
        reserved.input_scratch = static_cast<size_t>(4) * 2 * kHandoverStagingBytes;
    } else {
        reserved.input_scratch = 0;
    }

    const size_t row_maxima_bytes = handover_output ? static_cast<size_t>(4) * job.row_quads * 128 : 0;
    reserved.total = (gelu ? kGeluTableBytes : 0) + row_maxima_bytes + reserved.rope_table + reserved.input_scratch;
    return reserved;
}

bool output_layout_supported(const W8a8GemmJob & job, const ReservedVtcm & reserved) {
    const Output & output = job.output;

    if (output.form == OutputForm::kRowsF16 && output.rope_groups != 0 &&
        (!job.column_pairs || output.f16_group_width % 128 != 0 || output.rope_cos == nullptr ||
         output.rope_sin == nullptr)) {
        return false;
    }

    if (output.form == OutputForm::kAttentionTiles64 &&
        (!job.column_pairs || job.M % 32 != 0 || output.attention_group_blocks % 2 != 0 ||
         job.N != 3 * 32 * output.attention_group_blocks ||
         (output.rope_table != nullptr && (output.rope_positions == nullptr || reserved.rope_table == 0)))) {
        return false;
    }

    if (job.input.rows_f16 != nullptr &&
        (job.K % 64 != 0 || job.input.row_stride % 64 != 0 || job.input.column_scale == nullptr)) {
        return false;
    }

    if (output.form == OutputForm::kRowsF16) {
        if (output.f16_group_width % 64 != 0 || job.N > 4 * output.f16_group_width) return false;
        for (int group = 0; group * output.f16_group_width < job.N; group++) {
            if (output.f16_groups[group] == nullptr || output.f16_group_strides[group] < output.f16_group_width) {
                return false;
            }
        }
    }
    return true;
}

runtime::Status plan_layout(W8a8GemmJob & job) {
    job.k_tiles = job.K / 32;
    job.column_blocks = job.N / 32;
    job.column_block_bytes = static_cast<size_t>(job.k_tiles) * kWeightTileBytes + kOutputBlocksBytes;
    job.staging_slots = kStagingSlots;
    job.deep_weights = job.k_tiles >= kDeepWeightMinimumKTiles;
    const int column_blocks_per_chunk = static_cast<int>(kDefaultWeightChunkBytes / job.column_block_bytes);
    job.weight_chunk_column_blocks = column_blocks_per_chunk > 0 ? column_blocks_per_chunk : 1;
    if (job.weight_chunk_column_blocks > job.column_blocks) job.weight_chunk_column_blocks = job.column_blocks;
    choose_column_groups(job);

    const size_t staging_bytes = static_cast<size_t>(job.staging_slots) * kStagingSlotBytes;
    const int buffer_column_blocks =
        job.deep_weights ? (job.weight_chunk_column_blocks + 1) & ~1 : job.weight_chunk_column_blocks;
    const size_t weight_buffer_bytes = align_4096(static_cast<size_t>(buffer_column_blocks) * job.column_block_bytes);
    if (job.vtcm_reserved > job.session->vtcm_bytes()) return std::unexpected(runtime::Error::kNoMemory);
    const size_t vtcm_usable = job.session->vtcm_bytes() - job.vtcm_reserved;
    const size_t row_block_bytes = static_cast<size_t>(64) * job.K;
    if (staging_bytes + 2 * weight_buffer_bytes + row_block_bytes > vtcm_usable) {
        return std::unexpected(runtime::Error::kNoMemory);
    }

    const size_t activation_bytes = vtcm_usable - staging_bytes - 2 * weight_buffer_bytes;
    const int padded_rows = (job.M + 63) / 64 * 64;
    const int rows_one_buffer = static_cast<int>(activation_bytes / row_block_bytes) * 64;
    const int rows_two_buffers = static_cast<int>(activation_bytes / 2 / row_block_bytes) * 64;
    if (job.pipelined < 0) job.pipelined = padded_rows > rows_one_buffer;
    if (rows_two_buffers < 64) job.pipelined = 0;
    job.chunk_rows = job.pipelined ? rows_two_buffers : rows_one_buffer;
    if (job.chunk_rows > padded_rows) job.chunk_rows = padded_rows;

    uint8_t * vtcm = job.session->vtcm();
    job.staging = reinterpret_cast<Vector<uint8_t> *>(vtcm);
    job.weight_buffers[0] = vtcm + staging_bytes;
    job.weight_buffers[1] = job.weight_buffers[0] + weight_buffer_bytes;
    job.activation_buffers[0] = job.weight_buffers[1] + weight_buffer_bytes;
    job.activations = job.activation_buffers[0];
    job.activation_buffers[1] = job.pipelined ? job.activation_buffers[0] + static_cast<size_t>(job.chunk_rows) * job.K
                                              : job.activation_buffers[0];
    return {};
}

OutputStage find_output_stage(OutputForm form, Activation activation) {
    switch (form) {
        case OutputForm::kRowsF32Activation:
        case OutputForm::kRowsF32Accumulate:
            return find_row_output_stage(form, activation);
        case OutputForm::kRowsF16:
            return activation == Activation::kNone ? write_rows_f16_pair : nullptr;
        case OutputForm::kHandoverTiles:
            return find_handover_output_stage(activation);
        case OutputForm::kAttentionTiles72:
            return activation == Activation::kNone ? write_attention72_tile : nullptr;
        case OutputForm::kAttentionTiles64:
            return activation == Activation::kNone ? write_attention64_pair : nullptr;
    }
    return nullptr;
}

void prepare_output_context(W8a8GemmJob & job) {
    OutputContext & context = job.output_context;
    context.M = job.M;
    context.N = job.N;
    context.column_blocks = job.column_blocks;
    context.row_scales = job.row_scales;
    context.column_scales = job.column_scales;
    context.output = &job.output;
    context.rows = job.output.rows_f32;
    context.thread_scratch = nullptr;
    context.gelu_table = job.gelu_table;
    context.row_maxima = job.row_maxima;
    context.row_quads = job.row_quads;
    context.int8_scales = job.int8_scales;
}

}  // namespace

runtime::Status run(runtime::Session & session, const Arguments & arguments) {
    using runtime::Error;
    const Input & input = arguments.input;
    const Output & output = arguments.output;
    if (!arguments_supported(arguments)) return std::unexpected(Error::kBadParameter);
    const OutputStage output_stage = find_output_stage(output.form, output.activation);
    if (output_stage == nullptr) return std::unexpected(Error::kBadParameter);

    Workspace * kept = session.state<Workspace>();
    if (kept == nullptr) return std::unexpected(Error::kNoMemory);
    Workspace & workspace = *kept;
    const runtime::ExclusiveUse use(workspace.in_use);
    if (!use.taken()) return std::unexpected(Error::kBadParameter);
    if (const runtime::Status allocated = allocate_workspace(workspace); !allocated) return allocated;

    W8a8GemmJob & job = *workspace.job;
    reset_job(job);
    job.driver_ring = workspace.driver_ring.get();
    for (int thread = 0; thread < runtime::kHvxThreads; thread++)
        job.handover_rings[thread] = workspace.handover_rings[thread].get();
    job.session = &session;
    job.int8_scales = runtime::int8_scales(session);
    job.output_stage = output_stage;
    job.M = arguments.M;
    job.K = arguments.K;
    job.N = arguments.N;
    job.pipelined = -1;
    job.input = input;
    job.output = output;

    const bool handover_output = output.form == OutputForm::kHandoverTiles;
    const bool gelu =
        (output.form == OutputForm::kRowsF32Activation || handover_output) && activation_is_gelu(output.activation);
    if (handover_output) job.row_quads = (job.M + 63) / 64 * 16;
    const ReservedVtcm reserved = reserve_vtcm(job, gelu, handover_output);
    job.vtcm_reserved = reserved.total;
    uint8_t * reserved_base = session.vtcm() + session.vtcm_bytes() - reserved.total;

    const bool tiled_input = input.tiles != nullptr || input.handover != nullptr;
    if (tiled_input) job.pipelined = 0;
    job.column_pairs = output_uses_column_pairs(output.form);
    job.prefer_row_groups = output_prefers_row_groups(output.form);
    if (const runtime::Status planned = plan_layout(job); !planned) return planned;
    if (!output_layout_supported(job, reserved)) return std::unexpected(Error::kBadParameter);

    if (handover_output) {
        job.row_maxima = reinterpret_cast<Vector<int16_t> *>(reserved_base + reserved.input_scratch);
    }
    if (reserved.rope_table != 0) {
        auto * rope_vtcm = reinterpret_cast<float *>(reserved_base + reserved.input_scratch);
        job.rope_vtcm = rope_vtcm;
        job.rope_bytes = static_cast<uint32_t>(reserved.rope_table);
        if (output.form == OutputForm::kAttentionTiles64) {
            job.rope_source = output.rope_table;
            job.output.rope_table = rope_vtcm;
        } else {
            job.rope_source = output.rope_cos;
            job.output.rope_sin = rope_vtcm + (output.rope_sin - output.rope_cos);
            job.output.rope_cos = rope_vtcm;
        }
    }
    if (gelu) {
        job.gelu_table = reinterpret_cast<GeluTable *>(session.vtcm() + session.vtcm_bytes() - kGeluTableBytes);
        if (const runtime::Status copied = copy_gelu_table(workspace, output.activation, job.gelu_table->values);
            !copied) {
            return copied;
        }
    }

    float * row_scales = nullptr;
    if (!tiled_input) {
        row_scales = workspace.row_scales.reserve(static_cast<size_t>((job.M + 63) / 64 * 64));
        if (row_scales == nullptr) return std::unexpected(Error::kNoMemory);
    }
    if (input.rows_f16 != nullptr) {
        job.f16_input_scratch = reinterpret_cast<Vector<_Float16> *>(reserved_base);
        job.f16_column_scale = workspace.f16_column_scale.reserve(static_cast<size_t>(job.K) / 64);
        if (job.f16_column_scale == nullptr) return std::unexpected(Error::kNoMemory);
        for (int k = 0; k < job.K / 64; k++) {
            job.f16_column_scale[k] =
                to_float16(hvx::load(input.column_scale + 64 * k), hvx::load(input.column_scale + 64 * k + 32));
        }
    }

    job.weight_stream = arguments.weights.stream;
    job.column_scales = arguments.weights.column_scales;
    job.row_scales = input.tiles != nullptr      ? const_cast<float *>(input.tile_row_scales)
                     : input.handover != nullptr ? input.handover->row_scales
                                                 : row_scales;
    // the tile DMA reads DDR: flush, or for small inputs read them through the cache
    job.prequantized_through_cache =
        input.tiles != nullptr && static_cast<size_t>(job.M) * job.K <= kPrequantizedThroughCacheBytes;
    if ((input.tiles != nullptr && !job.prequantized_through_cache) || input.handover != nullptr) {
        runtime::flush_data_cache();
    }
    if (input.handover != nullptr) {
        job.handover_staging = reinterpret_cast<Vector<_Float16> *>(reserved_base);
    }

    prepare_output_context(job);
    job.output_context.thread_scratch = workspace.thread_scratch.get();
    return run_job(job);
}

}  // namespace benzene::kernels::gemm_w8a8
