#include "dsp/kernels/flash_attention/flash_attention.h"

#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>

#include "dsp/hmx/instructions.h"
#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/transpose.h"
#include "dsp/hvx/vector.h"
#include "dsp/runtime/cache.h"
#include "dsp/runtime/dma_ring.h"
#include "dsp/runtime/heap.h"
#include "dsp/runtime/status.h"
#include "dsp/runtime/synchronization.h"
#include "dsp/runtime/thread_pool.h"
#include "dsp/vector_math/functions.h"
#include "dsp/vector_math/softmax.h"

namespace benzene::kernels::flash_attention {
namespace {

using hmx::kTileBytes;
using hvx::Mask;
using hvx::Vector;
using hvx::VectorPair;

constexpr int kMaximumStreams = 32;
constexpr int kMaximumBlockTiles = 16;  // at most 32: one HMX load pair takes a KV block's tiles
constexpr int kMaximumQueryBlocks = 512;
constexpr int kMaximumSplits = 4;
constexpr int kTileVectors = kTileBytes / hvx::kVectorBytes;
constexpr size_t kWorkerScratchBytes = 24 * 1024;
constexpr int kOutputBuffersInFlight = 4;
// KV packing: the slack after each of a thread's K / V row buffers (zeroed, read by the last head's second vector)
constexpr size_t kPackingSlackBytes = 256;

// The life of a stream:
// 1. an empty stream takes the next item (kPreparing);
// 2. the driver computes S^T (kReadyForScores -> kSoftmax);
// 3. a worker runs the softmax (kSoftmaxBusy -> kReadyForOutput);
// 4. the driver updates O^T and starts the next KV block (kSoftmax) or finishes (kFinish);
// 5. a worker writes the O rows (kFinishBusy) and takes the next item (or kNoItem: no items left).
enum StreamState : int {
    kEmpty = 0,
    kPreparing,
    kReadyForScores,
    kSoftmax,
    kSoftmaxBusy,
    kReadyForOutput,
    kFinish,
    kFinishBusy,
    kNoItem,
};

// A split-K range's partial result (VTCM): its O rows (query q: rows[2 q], rows[2 q + 1]), m and l.
struct PartialResult {
    Vector<_Float16> rows[64];
    Vector<_Float16> running_max;
    Vector<float> running_sum;
};

// A stream computes its item (a 32-query block of a query head) transposed, so that every per-query quantity is a
// per-column quantity of the HMX output tiles, where the store's per-column fp16 scale and bias apply it
// (out = acc * scale + bias):
//   S^T = K Q^T          stored with scale log2(e) / sqrt(D) and bias -m
//   P^T = exp2(S^T - s)  on HVX in place, s = m_new - m_old >= 0; per-query block sums in f32
//   O^T = O^T diag(a l_old) + V^T P^T, stored with scale 1 / l_new: O^T stays normalized (no fp16 overflow),
//                        a = exp2(-s), l_new = a l_old + sum(P)
// Vector i of an S^T tile holds keys 2i, 2i + 1 for all 32 queries (lanes 2q, 2q + 1), so every per-query reduction is
// lane-wise across vectors.
struct alignas(128) Stream {
    // m: the max so far rounded up to an integer (log2 units), fp16, query q in lanes 2q and 2q + 1
    Vector<_Float16> running_max;
    Vector<float> running_sum;  // l: f32, lane q
    Vector<uint32_t> * query_tiles;
    Vector<uint32_t> * output_tiles;
    Vector<_Float16> * score_tiles;
    Vector<uint16_t> * rescale_tile;
    Vector<uint32_t> * score_store_block;
    Vector<uint32_t> * output_store_block;
    int item;
    int query_head;
    int query_block;
    int block;
    int blocks;
    int key_tile_end;
    int key_tile_start;
    int partial_slot;
    std::atomic<int> state;
};

struct alignas(128) OutputDmas {
    uint32_t pushed;
};

struct AttentionJob {
    runtime::Session * session;
    int queries;
    int keys;
    int query_heads;
    int kv_heads;
    int head_dimensions;
    int d_tiles;
    int group;  // query heads per KV head
    int causal;
    int block_tiles;
    int stream_count;
    int query_blocks;
    int key_tiles;
    int items;
    _Float16 * q;  // element (head h, token t) at h * head_stride + t * token_stride
    _Float16 * o;
    _Float16 * k;  // DDR, [token][kv head][D]
    _Float16 * v;
    Vector<uint32_t> * packed_keys;
    Vector<uint32_t> * packed_values;
    Vector<uint32_t> * vtcm_keys[2];
    Vector<uint32_t> * vtcm_values[2];
    Vector<uint32_t> * vtcm_queries[2];
    Vector<uint32_t> * output_buffers[runtime::kHvxThreads][kOutputBuffersInFlight];
    Vector<uint32_t> * scratch[8];
    runtime::DmaRing * rings[8];
    Vector<int16_t> * key_minus_query;  // 16 vectors: key - query within a tile (S^T lane layout)
    Vector<int16_t> * key_index;        // 16 vectors: key index within a tile
    Vector<uint16_t> * diagonal_masks;
    uint16_t score_scale_bits;  // log2(e) / sqrt(D) as fp16 bits
    int head;
    struct alignas(128) ItemCounters {
        std::atomic<int> next;
        std::atomic<int> done;
    } item_counters;
    OutputDmas output_dmas[8];
    Stream streams[kMaximumStreams];
    alignas(128) runtime::SpinBarrier barrier;
    // key tiles per KV head in the packed buffers (>= key_tiles)
    alignas(128) int cache_capacity;
    int token_stride;
    int head_stride;
    // Q^T weight tiles [head][query block][d tiles], nullptr: from Q rows
    const Vector<uint32_t> * prepacked_queries;
    // token-major Q / K / V / O rows in VTCM, and the packed K / V^T tiles of every head too: no DMA and no cache flush
    // anywhere in the call, one item queue over all heads (group 1)
    int rows_in_vtcm;
    // segments: query block qb sees key tiles [segment_first_tile[qb], + segment_tiles[qb]) of which the first
    // segment_keys[qb] keys are valid; 0 = off
    int segment_count;
    int16_t segment_first_tile[kMaximumQueryBlocks];
    int16_t segment_tiles[kMaximumQueryBlocks];
    int16_t segment_keys[kMaximumQueryBlocks];
    // split-K (one query block, not causal: decode): every query head's keys in split_count ranges of split_tiles
    // tiles, one item each; the items leave O rows, m and l in `partials`, merged after the head
    int split_count;
    int split_tiles;
    PartialResult * partials;
};

// Scores masked out become -65504 (finite); the softmax exp makes them exactly 0.
constexpr int kMaskedScoreBits = 0xFBFF;

// VTCM is written by vector stores only: memset / scalar stores there are slow.
template <class Lane>
inline void zero_vectors(Vector<Lane> * destination, size_t count) {
    for (size_t i = 0; i < count; i++) destination[i] = Vector<Lane>::zero();
}

inline const Vector<uint32_t> * kv_tiles(const AttentionJob & job, const Stream & stream, bool values) {
    if (job.rows_in_vtcm) {
        return (values ? job.packed_values : job.packed_keys) +
               static_cast<size_t>(stream.query_head) * job.cache_capacity * job.d_tiles * kTileVectors;
    }
    return values ? job.vtcm_values[job.head & 1] : job.vtcm_keys[job.head & 1];
}

inline int block_range(const AttentionJob & job, const Stream & stream, int * first_tile) {
    if (job.segment_count != 0) {
        *first_tile = job.segment_first_tile[stream.query_block] + stream.block * job.block_tiles;
        const int remaining = stream.key_tile_end - stream.block * job.block_tiles;
        return remaining < job.block_tiles ? remaining : job.block_tiles;
    }
    *first_tile = stream.key_tile_start + stream.block * job.block_tiles;
    return stream.key_tile_end - *first_tile < job.block_tiles ? stream.key_tile_end - *first_tile : job.block_tiles;
}

inline int valid_keys(const AttentionJob & job, const Stream & stream, int key_tile) {
    if (job.segment_count != 0) {
        return job.segment_keys[stream.query_block] - (key_tile - job.segment_first_tile[stream.query_block]) * 32;
    }
    return job.keys - key_tile * 32;
}

[[gnu::noinline]] void compute_scores(AttentionJob & job, Stream & stream) {
    int first_tile;
    const int tiles = block_range(job, stream, &first_tile);
    const Vector<uint32_t> * keys =
        kv_tiles(job, stream, false) + static_cast<size_t>(first_tile) * job.d_tiles * kTileVectors;
    const auto range = static_cast<uint32_t>(job.d_tiles * kTileBytes - 1);

    hmx::load_output_block(stream.score_store_block);
    for (int tile = 0; tile < tiles; tile++) {
        // Redundant: the HMX work before every S^T tile ends with a non-retain store, which leaves the accumulator zero
        // (the driver clears it once after taking HMX). Its wait paces HMX for the softmax workers; only with rows in
        // VTCM do they have the slack to drop it.
        if (!job.rows_in_vtcm) hmx::clear_fp16_accumulator();
        hmx::multiply_fp16(keys + static_cast<size_t>(tile) * job.d_tiles * kTileVectors, stream.query_tiles, range);
        hmx::store_fp16(stream.score_tiles + static_cast<size_t>(tile) * kTileVectors);
    }
}

[[gnu::noinline]] void update_output(AttentionJob & job, Stream & stream) {
    int first_tile;
    const int tiles = block_range(job, stream, &first_tile);
    const Vector<uint32_t> * value_tiles = kv_tiles(job, stream, true);

    hmx::load_output_block(stream.output_store_block);
    for (int d_tile = 0; d_tile < job.d_tiles; d_tile++) {
        Vector<uint32_t> * output = stream.output_tiles + static_cast<size_t>(d_tile) * kTileVectors;
        hmx::clear_fp16_accumulator();
        if (stream.block > 0) hmx::multiply_fp16(output, stream.rescale_tile, kTileBytes - 1);
        hmx::multiply_fp16(value_tiles + (static_cast<size_t>(d_tile) * job.key_tiles + first_tile) * kTileVectors,
                           stream.score_tiles, static_cast<uint32_t>(tiles) * kTileBytes - 1);
        hmx::store_fp16(output);
    }
}

// The fp16 value of an integer (int16 lanes), rounded up to a representable one. Integers are exact in fp16 only below
// 2048 in magnitude (then multiples of 2, 4, ..): the nearest one, and one step up (bits + 1, for negative values
// bits - 1) where that rounded down.
inline Vector<_Float16> integer_to_f16_rounded_up(Vector<int16_t> value) {
    const Vector<_Float16> rounded = to_float16_on_shift_unit(value);
    const Mask below = value > truncate_to_int16(rounded);
    const Mask negative = Vector<int16_t>::zero() > reinterpret<int16_t>(rounded);
    const Vector<int16_t> step = select(negative, Vector<int16_t>::splat(-1), Vector<int16_t>::splat(1));
    return select(below, reinterpret<_Float16>(reinterpret<int16_t>(rounded) + step), rounded);
}

// Returns the block's sum per query (f32, lane q).
[[gnu::noinline]] Vector<float> exponentiate_block(Vector<_Float16> * __restrict scores, int tiles,
                                                   Vector<_Float16> low_clamp, Vector<int16_t> exponent_bias) {
    const Vector<float> zero = Vector<float>::zero();
    Vector<float> sums[4] = {zero, zero, zero, zero};
    vector_math::softmax_exp2(scores, tiles, low_clamp, exponent_bias, sums);
    return (sums[0] + sums[1]) + (sums[2] + sums[3]);
}

[[gnu::noinline]] void softmax_block(AttentionJob * job_pointer, Stream * stream_pointer) {
    AttentionJob & job = *job_pointer;
    Stream & stream = *stream_pointer;
    int first_tile;
    const int tiles = block_range(job, stream, &first_tile);
    Vector<_Float16> * scores = stream.score_tiles;
    const Vector<_Float16> masked = Vector<_Float16>::splat_bits(kMaskedScoreBits);
    const Vector<float> zero = Vector<float>::zero();

    const int first_query = stream.query_block * 32;
    Vector<_Float16> max0 = masked;
    Vector<_Float16> max1 = masked;
    Vector<_Float16> max2 = masked;
    Vector<_Float16> max3 = masked;
    for (int tile = 0; tile < tiles; tile++) {
        const int key_tile = first_tile + tile;
        const int threshold = first_query - key_tile * 32;  // causal: masked if (key - query) within the tile > it
        const int valid = valid_keys(job, stream, key_tile);
        Vector<_Float16> * v = scores + tile * 16;
        if (!job.causal && valid < 32) {
            const int full = valid / 2;
            int i = 0;
            for (; i + 1 < full; i += 2) {
                max0 = max(max0, v[i]);
                max1 = max(max1, v[i + 1]);
            }
            if (i < full) {
                max2 = max(max2, v[i]);
                i++;
            }
            if (valid & 1) {
                const Vector<_Float16> x = select(job.key_index[i] > Vector<int16_t>::splat(valid - 1), masked, v[i]);
                v[i] = x;
                max3 = max(max3, x);
                i++;
            }
            for (; i < 16; i++) v[i] = masked;
        } else if ((job.causal && threshold < 31) || valid < 32) {
            const Vector<int16_t> causal_limit = Vector<int16_t>::splat(job.causal ? threshold : 32767);
            const Vector<int16_t> key_limit = Vector<int16_t>::splat(valid - 1);
            for (int i = 0; i < 16; i++) {
                const Mask outside = (job.key_minus_query[i] > causal_limit) | (job.key_index[i] > key_limit);
                const Vector<_Float16> x = select(outside, masked, v[i]);
                v[i] = x;
                max0 = max(max0, x);
            }
        } else {
            for (int i = 0; i < 16; i += 4) {
                max0 = max(max0, v[i]);
                max1 = max(max1, v[i + 1]);
                max2 = max(max2, v[i + 2]);
                max3 = max(max3, v[i + 3]);
            }
        }
    }

    const Vector<_Float16> maximum = max(max(max0, max1), max(max2, max3));
    const Vector<_Float16> query_max =
        max(interleave_even_lanes(maximum, maximum), interleave_odd_lanes(maximum, maximum));
    const Vector<_Float16> shift = stream.block == 0 ? query_max : max(query_max, Vector<_Float16>::zero());

    // The shift applied is the integer I = ceil(shift) (both lanes of a query): the running max m stays an fp16 integer
    // and the exp subtracts it on the exponent. Computed as trunc toward zero, + 1 for positive non-integers (fp16 bits
    // of non-negative values compare like the values).
    const Vector<int16_t> truncated = truncate_to_int16(shift);
    const Mask round_up = (reinterpret<int16_t>(shift) > reinterpret<int16_t>(to_float16_on_shift_unit(truncated))) &
                          (reinterpret<int16_t>(shift) > -1);
    // Saturating: no wrap at 32767. The new running max m_old + I must be an fp16 integer (it is the next S^T store's
    // bias), and the exp's clamp at least I - 26 (its exponent arithmetic). Beyond 2048 both are rounded up to
    // representable values and I follows the max.
    const Vector<int16_t> smallest_shift = add_saturate(truncated, pattern_where<int16_t>(round_up, 0x00010001));
    const Vector<int16_t> old_max = truncate_to_int16(stream.running_max);
    const Vector<_Float16> new_max = integer_to_f16_rounded_up(add_saturate(old_max, smallest_shift));
    const Vector<int16_t> integer_shift = subtract_saturate(truncate_to_int16(new_max), old_max);

    // -I as f32 in lane q: a = exp2(-I) rescales the earlier blocks
    const Vector<float> negative_shift =
        to_float(Vector<int32_t>::zero() - ((reinterpret<int32_t>(integer_shift) << 16) >> 16));
    const Vector<float> block_sum = exponentiate_block(
        scores, tiles, integer_to_f16_rounded_up(subtract_saturate(integer_shift, Vector<int16_t>::splat(26))),
        subtract_saturate(integer_shift, Vector<int16_t>::splat(12)));

    Vector<float> rescaled_sum = zero;
    Vector<float> new_sum = block_sum;
    if (stream.block > 0) {
        const Vector<float> rescale = vector_math::exp2_softmax(negative_shift);
        rescaled_sum = rescale * stream.running_sum;
        new_sum = rescaled_sum + block_sum;
        // diag(a * l_old) as a weight tile: byte (q / 2) * 128 + q * 4 + (q % 2) * 2, i.e. line i holds q = 2i in
        // halfword lane 4i and q = 2i + 1 in lane 4i + 3, the same lanes as in the pair layout
        const Vector<uint16_t> diagonal = reinterpret<uint16_t>(to_float16_interleaved(rescaled_sum, rescaled_sum));
        Vector<uint16_t> * rescale_lines = stream.rescale_tile;
        for (int i = 0; i < 16; i++) rescale_lines[i] = diagonal & job.diagonal_masks[i];
    }

    const Vector<float> inverse_sum = vector_math::reciprocal(new_sum);
    // O^T store: scale 1 / l, bias 0
    *stream.output_store_block = reinterpret<uint32_t>(to_float16_interleaved(inverse_sum, inverse_sum)) & 0x0000FFFFu;
    // next S^T store: scale c, bias -m_new
    const Vector<uint32_t> negative_max = (reinterpret<uint32_t>(new_max) & 0xFFFF0000u) ^ 0x80000000u;
    *stream.score_store_block = negative_max | job.score_scale_bits;
    stream.running_max = new_max;
    stream.running_sum = new_sum;
}

[[gnu::noinline]] void transpose_rows(const _Float16 * rows, size_t stride, int row_count,
                                      Vector<uint32_t> * first_tile, Vector<uint32_t> * second_tile,
                                      Vector<uint32_t> * scratch0, Vector<uint32_t> * scratch1) {
    hvx::transpose_words_32x32(rows, stride, row_count, first_tile, second_tile, scratch0, scratch1);
}

inline void reset_stream_state(AttentionJob & job, Stream & stream) {
    stream.running_max = Vector<_Float16>::zero();
    stream.running_sum = Vector<float>::zero();
    *stream.score_store_block = Vector<uint32_t>::splat(job.score_scale_bits);  // first block: scale c, bias 0
}

inline void prepare_item(AttentionJob & job, Stream & stream, int item, Vector<uint32_t> * scratch,
                         runtime::DmaRing & ring, int thread) {
    const int split = job.split_count > 1 ? item % job.split_count : 0;
    const int item_in_head = job.split_count > 1 ? item / job.split_count : item;
    const int group_member = item_in_head % job.group;
    const int query_index = job.rows_in_vtcm ? item_in_head % job.query_blocks : item_in_head / job.group;
    const int query_block = job.causal ? job.query_blocks - 1 - query_index : query_index;  // causal: longest first

    stream.item = item;
    stream.query_head = (job.rows_in_vtcm ? item_in_head / job.query_blocks : job.head) * job.group + group_member;
    stream.query_block = query_block;
    stream.block = 0;
    stream.key_tile_start = 0;
    stream.partial_slot = item;
    if (job.split_count > 1) {
        stream.key_tile_start = split * job.split_tiles;
        const int end = stream.key_tile_start + job.split_tiles;
        stream.key_tile_end = end < job.key_tiles ? end : job.key_tiles;
    } else if (job.segment_count != 0) {
        stream.key_tile_end = job.segment_tiles[query_block];
    } else if (job.causal) {
        const int last = query_block + 1;
        stream.key_tile_end = last < job.key_tiles ? last : job.key_tiles;
    } else {
        stream.key_tile_end = job.key_tiles;
    }
    stream.blocks = (stream.key_tile_end - stream.key_tile_start + job.block_tiles - 1) / job.block_tiles;

    if (job.prepacked_queries != nullptr) {
        stream.query_tiles =
            job.vtcm_queries[job.head & 1] + static_cast<size_t>(query_block) * job.d_tiles * kTileVectors;
        reset_stream_state(job, stream);
        return;
    }

    // landing rows of 256 B, zero beyond D since the setup
    Vector<uint32_t> * landing = scratch;
    Vector<uint32_t> * scratch_a = landing + 64;
    Vector<uint32_t> * scratch_b = scratch_a + 64;
    const int rows = job.queries - query_block * 32 < 32 ? job.queries - query_block * 32 : 32;
    const _Float16 * first_row = job.q + static_cast<size_t>(stream.query_head) * job.head_stride +
                                 static_cast<size_t>(query_block) * 32 * job.token_stride;
    if (job.rows_in_vtcm) {
        Vector<uint32_t> * query_tiles = stream.query_tiles;
        for (int vector = 0; vector < (job.d_tiles + 1) / 2; vector++) {
            transpose_rows(first_row + vector * 64, static_cast<size_t>(job.token_stride) * 2, rows,
                           query_tiles + 32 * vector, query_tiles + 32 * vector + 16, scratch_b, scratch_a);
        }
        reset_stream_state(job, stream);
        return;
    }

    if (rows < 32) zero_vectors(landing + 2 * rows, static_cast<size_t>(32 - rows) * 2);
    ring.push(landing, first_row, static_cast<uint32_t>(job.head_dimensions) * 2, static_cast<uint32_t>(rows), 256,
              static_cast<uint32_t>(job.token_stride) * 2);
    ring.pop();
    // vector k = d pair k for queries 0 .. 31: Q^T weight tile k / 16, line k % 16
    hvx::transpose_words_32x64(landing, scratch_b, scratch_a);
    Vector<uint32_t> * query_tiles = stream.query_tiles;
    for (int k = 0; k < job.d_tiles * 16; k++) query_tiles[k] = scratch_b[k];
    reset_stream_state(job, stream);
}

inline void finish_item(AttentionJob & job, Stream & stream, Vector<uint32_t> * scratch, runtime::DmaRing & ring,
                        int thread) {
    Vector<uint32_t> * scratch_a = scratch + 64;
    Vector<uint32_t> * rows_buffer = scratch_a + 64;
    const int rows = job.queries - stream.query_block * 32 < 32 ? job.queries - stream.query_block * 32 : 32;
    _Float16 * first_row = job.o + static_cast<size_t>(stream.query_head) * job.head_stride +
                           static_cast<size_t>(stream.query_block) * 32 * job.token_stride;

    // The worker's ring carries only these writes, each from its own buffer: the oldest of kOutputBuffersInFlight is
    // this buffer's previous one.
    if (job.prepacked_queries != nullptr) {
        if (ring.depth() == kOutputBuffersInFlight) ring.pop();
        rows_buffer = job.output_buffers[thread][job.output_dmas[thread].pushed++ % kOutputBuffersInFlight];
        hvx::untranspose_words_64x32(stream.output_tiles, rows_buffer, scratch_a);
        ring.push(first_row, rows_buffer, static_cast<uint32_t>(job.head_dimensions) * 2, static_cast<uint32_t>(rows),
                  static_cast<uint32_t>(job.token_stride) * 2, 256);
        return;
    }

    if (job.rows_in_vtcm) {
        for (int vector = 0; vector < (job.d_tiles + 1) / 2; vector++) {
            hvx::untranspose_words_32x32(stream.output_tiles + 32 * vector, first_row + vector * 64,
                                         static_cast<size_t>(job.token_stride) * 2, rows, rows_buffer, scratch_a);
        }
        return;
    }

    // reads 64 vectors from O^T: beyond d_tiles * 16 they belong to the next buffer and only feed d >= 32 d_tiles
    hvx::untranspose_words_64x32(stream.output_tiles, rows_buffer, scratch_a);
    if (job.split_count > 1) {
        PartialResult * partial = job.partials + stream.partial_slot;
        for (int q = 0; q < 2 * rows; q++) partial->rows[q] = reinterpret<_Float16>(rows_buffer[q]);
        partial->running_max = stream.running_max;
        partial->running_sum = stream.running_sum;
        return;
    }

    ring.push(first_row, rows_buffer, static_cast<uint32_t>(job.head_dimensions) * 2, static_cast<uint32_t>(rows),
              static_cast<uint32_t>(job.token_stride) * 2, 256);
    ring.pop();
}

// Split-K merge of the current head: O = sum_s l_s 2^m_s O_s / sum_s l_s 2^m_s per query row.
[[gnu::noinline]] void merge_partials(AttentionJob & job) {
    const int rows = job.queries < 32 ? job.queries : 32;
    for (int member = 0; member < job.group; member++) {
        const int query_head = job.head * job.group + member;
        for (int q = 0; q < rows; q++) {
            float weights[kMaximumSplits];
            float maximum = -1e30f;
            float denominator = 0.0f;
            for (int split = 0; split < job.split_count; split++) {
                const PartialResult * partial = job.partials + member * job.split_count + split;
                const auto max_bits = static_cast<uint16_t>(extract_word(partial->running_max, q));
                weights[split] = static_cast<float>(std::bit_cast<_Float16>(max_bits));
                maximum = weights[split] > maximum ? weights[split] : maximum;
            }

            float sums[kMaximumSplits];
            for (int split = 0; split < job.split_count; split++) {
                const PartialResult * partial = job.partials + member * job.split_count + split;
                sums[split] = std::bit_cast<float>(extract_word(partial->running_sum, q));
                weights[split] = sums[split] * exp2f(weights[split] - maximum);
                denominator += weights[split];
            }

            const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
            Vector<float> accumulators[4] = {Vector<float>::zero(), Vector<float>::zero(), Vector<float>::zero(),
                                             Vector<float>::zero()};
            for (int split = 0; split < job.split_count; split++) {
                const PartialResult * partial = job.partials + member * job.split_count + split;
                const Vector<float> weight = Vector<float>::splat(weights[split] * inverse);
                for (int half = 0; half < 2; half++) {
                    const VectorPair<float> values = to_float32(partial->rows[2 * q + half]);
                    accumulators[2 * half] = accumulators[2 * half] + values.low() * weight;
                    accumulators[2 * half + 1] = accumulators[2 * half + 1] + values.high() * weight;
                }
            }

            _Float16 * output =
                job.o + static_cast<size_t>(query_head) * job.head_stride + static_cast<size_t>(q) * job.token_stride;
            hvx::store(output, to_float16(accumulators[0], accumulators[1]));
            if (job.head_dimensions > 64) hvx::store(output + 64, to_float16(accumulators[2], accumulators[3]));
        }
    }
}

[[gnu::always_inline]] inline void drive_hmx(AttentionJob & job) {
    while (job.item_counters.done.load() < job.items) {
        bool worked = false;
        for (int index = 0; index < job.stream_count; index++) {
            Stream & stream = job.streams[index];
            const int state = stream.state.load();
            if (state == kReadyForOutput) {
                update_output(job, stream);
                stream.block++;
                if (stream.block < stream.blocks) {
                    compute_scores(job, stream);
                    stream.state.store(kSoftmax);
                } else {
                    stream.state.store(kFinish);
                }
                worked = true;
            } else if (state == kReadyForScores) {
                compute_scores(job, stream);
                stream.state.store(kSoftmax);
                worked = true;
            }
        }

        if (worked) continue;
        for (volatile int k = 0; k < 16; k = k + 1) {}
    }
}

[[gnu::noinline]] void claim_next_item(AttentionJob & job, Stream & stream, Vector<uint32_t> * scratch, int thread) {
    const int item = job.item_counters.next.fetch_add(1);
    if (item < job.items) {
        prepare_item(job, stream, item, scratch, *job.rings[thread], thread);
        stream.state.store(kReadyForScores);
    } else {
        stream.state.store(kNoItem);
    }
}

[[gnu::noinline]] void run_worker(AttentionJob & job, int thread) {
    Vector<uint32_t> * scratch = job.scratch[thread];
    while (job.item_counters.done.load() < job.items) {
        for (int k = 0; k < job.stream_count; k++) {
            Stream & stream = job.streams[(k + thread) % job.stream_count];
            int state = stream.state.load();
            if (state == kSoftmax) {
                if (stream.state.compare_exchange_strong(state, kSoftmaxBusy)) {
                    softmax_block(&job, &stream);
                    stream.state.store(kReadyForOutput);
                }
            } else if (state == kFinish) {
                if (stream.state.compare_exchange_strong(state, kFinishBusy)) {
                    finish_item(job, stream, scratch, *job.rings[thread], thread);
                    job.item_counters.done.fetch_add(1);
                    claim_next_item(job, stream, scratch, thread);
                }
            } else if (state == kEmpty) {
                if (stream.state.compare_exchange_strong(state, kPreparing))
                    claim_next_item(job, stream, scratch, thread);
            }
        }
    }
}

[[gnu::noinline]] void push_head(const AttentionJob & job, runtime::DmaRing & ring, int head) {
    const size_t head_vectors = static_cast<size_t>(job.cache_capacity) * job.d_tiles * kTileVectors;
    const auto key_bytes = static_cast<uint32_t>(static_cast<size_t>(job.key_tiles) * job.d_tiles * kTileBytes);
    const auto value_row = static_cast<uint32_t>(job.key_tiles * kTileBytes);

    ring.push(job.vtcm_keys[head & 1], job.packed_keys + static_cast<size_t>(head) * head_vectors, key_bytes, 1,
              key_bytes, key_bytes);
    ring.push(job.vtcm_values[head & 1], job.packed_values + static_cast<size_t>(head) * head_vectors, value_row,
              static_cast<uint32_t>(job.d_tiles), value_row, static_cast<uint32_t>(job.cache_capacity * kTileBytes));
    if (job.prepacked_queries != nullptr) {
        const size_t query_vectors = static_cast<size_t>(job.query_blocks) * job.d_tiles * kTileVectors;
        const auto query_bytes = static_cast<uint32_t>(query_vectors * hvx::kVectorBytes);
        ring.push(job.vtcm_queries[head & 1], job.prepacked_queries + static_cast<size_t>(head) * query_vectors,
                  query_bytes, 1, query_bytes, query_bytes);
    }
}

void begin_head(AttentionJob & job, runtime::DmaRing & ring, int head) {
    if (head == 0) push_head(job, ring, 0);
    ring.pop();
    ring.pop();
    if (job.prepacked_queries != nullptr) ring.pop();
    if (head + 1 < job.kv_heads) push_head(job, ring, head + 1);
}

void reset_items(AttentionJob & job, int head) {
    job.head = head;
    job.item_counters.next.store(0);
    job.item_counters.done.store(0);
#pragma clang loop vectorize(disable)
    for (int index = 0; index < job.stream_count; index++) job.streams[index].state.store(kEmpty);
}

// The HMX driver is a fifth thread that must never execute an HVX instruction: the four HVX contexts belong to the
// softmax workers, and a fifth HVX user waits forever. So no memset (lowered to vector stores), and no inlining with
// the workers' code (hoisted vector constants).
[[gnu::noinline]] void attention_driver_thread(void * job_pointer) {
    AttentionJob & job = *static_cast<AttentionJob *>(job_pointer);
    constexpr int kDriverThread = runtime::kHvxThreads;
    runtime::DmaRing * ring = job.rings[kDriverThread];
    ring->initialize_with_scalar_stores(runtime::DmaBypass::kBoth);
    hmx::lock(*job.session);
    hmx::clear_fp16_accumulator();  // compute_scores relies on a zero accumulator with rows in VTCM
    job.barrier.wait();

    const int heads = job.rows_in_vtcm ? 1 : job.kv_heads;
    for (int head = 0; head < heads; head++) {
        if (!job.rows_in_vtcm) begin_head(job, *ring, head);
        reset_items(job, head);
        job.barrier.wait();
        drive_hmx(job);
        job.barrier.wait();
        job.barrier.wait();  // (split-K merge by worker 0)
    }

    hmx::unlock(*job.session);
    ring->drain();
}

void attention_worker_thread(void * job_pointer, int thread) {
    AttentionJob & job = *static_cast<AttentionJob *>(job_pointer);
    runtime::DmaRing * ring = job.rings[thread];
    ring->initialize(runtime::DmaBypass::kBoth);
    ring->set_bypass(runtime::DmaBypass::kSource);  // O rows leave through L2: no invalidation afterwards
    job.barrier.wait();

    const int heads = job.rows_in_vtcm ? 1 : job.kv_heads;
    for (int head = 0; head < heads; head++) {
        job.barrier.wait();
        run_worker(job, thread);
        job.barrier.wait();
        if (thread == 0 && job.split_count > 1) merge_partials(job);
        job.barrier.wait();
    }

    ring->drain();
}

runtime::Status run_job(AttentionJob & job) {
    job.barrier.reset(runtime::kHvxThreads + 1);
    return runtime::run_parallel_with_driver(*job.session, runtime::kHvxThreads, attention_worker_thread,
                                             attention_driver_thread, &job);
}

size_t packing_scratch_bytes(const AttentionJob & job) {
    const size_t row_bytes = static_cast<size_t>(job.kv_heads) * job.head_dimensions * 2;
    return (2 * (32 * row_bytes + kPackingSlackBytes) + 3 * 64 * 128 + 127) & ~static_cast<size_t>(127);
}

struct PackingJob {
    AttentionJob * job;
    // per thread: K rows, V rows (32 x row bytes + slack each), row gather + 2 transpose buffers
    uint8_t * scratch[runtime::kHvxThreads];
    runtime::DmaRing * rings[runtime::kHvxThreads];
    size_t row_bytes;
};

// The 32 key rows of head `head` from the landing buffer: row k = vectors 2k, 2k + 1 (the second zero for D <= 64).
// Lanes d >= D of a row hold the next row's values or zeros (finite): they only meet the zero padding of Q^T (K) or
// produce O^T rows d >= D that are never written out (V).
inline void gather_head_rows(const AttentionJob & job, const uint8_t * landing, size_t row_bytes, int head,
                             Vector<uint32_t> * rows) {
    for (int k = 0; k < 32; k++) {
        const uint8_t * row =
            landing + static_cast<size_t>(k) * row_bytes + static_cast<size_t>(head) * job.head_dimensions * 2;
        rows[2 * k] = hvx::load_unaligned<uint32_t>(row);
        rows[2 * k + 1] =
            job.head_dimensions > 64 ? hvx::load_unaligned<uint32_t>(row + 128) : Vector<uint32_t>::zero();
    }
}

void pack_key_tile(void * context, int key_tile, int thread) {
    auto * packing = static_cast<PackingJob *>(context);
    AttentionJob & job = *packing->job;
    const int d_tiles = job.d_tiles;
    const int rows = job.keys - key_tile * 32 < 32 ? job.keys - key_tile * 32 : 32;
    const size_t row_bytes = packing->row_bytes;
    const size_t buffer_bytes = 32 * row_bytes + kPackingSlackBytes;
    uint8_t * key_rows = packing->scratch[thread];
    uint8_t * value_rows = key_rows + buffer_bytes;
    auto * head_rows = reinterpret_cast<Vector<uint32_t> *>(value_rows + buffer_bytes);
    Vector<uint32_t> * scratch_a = head_rows + 64;
    Vector<uint32_t> * scratch_b = scratch_a + 64;
    runtime::DmaRing & ring = *packing->rings[thread];
    const size_t first_value = static_cast<size_t>(key_tile) * 32 * job.kv_heads * job.head_dimensions;
    const auto bytes = static_cast<uint32_t>(rows * row_bytes);

    ring.push(key_rows, job.k + first_value, bytes, 1, bytes, bytes);
    ring.push(value_rows, job.v + first_value, bytes, 1, bytes, bytes);
    // keys past the end and the slack: zeros
    for (int buffer = 0; buffer < 2; buffer++) {
        uint8_t * zeros = (buffer != 0 ? value_rows : key_rows) + static_cast<size_t>(rows) * row_bytes;
        for (size_t i = 0; i < ((32 - rows) * row_bytes + kPackingSlackBytes) / 128; i++) {
            hvx::store(zeros + 128 * i, Vector<uint8_t>::zero());
        }
    }

    ring.pop();
    for (int head = 0; head < job.kv_heads; head++) {
        gather_head_rows(job, key_rows, row_bytes, head, head_rows);
        Vector<uint32_t> * key_tiles = job.packed_keys + (static_cast<size_t>(head) * job.cache_capacity * d_tiles +
                                                          static_cast<size_t>(key_tile) * d_tiles) *
                                                             kTileVectors;
        for (int line = 0; line < 16; line++) {
            for (int half = 0; half < 2; half++) {
                const VectorPair<uint32_t> pair =
                    interleave_chunks<2>(head_rows[2 * (2 * line + 1) + half], head_rows[2 * (2 * line) + half]);
                if (2 * half < d_tiles) key_tiles[(2 * half) * 16 + line] = pair.low();
                if (2 * half + 1 < d_tiles) key_tiles[(2 * half + 1) * 16 + line] = pair.high();
            }
        }
    }

    ring.pop();
    for (int head = 0; head < job.kv_heads; head++) {
        gather_head_rows(job, value_rows, row_bytes, head, head_rows);
        // vector k: d pair k for the 32 keys = line k % 16 of V^T tile k / 16
        hvx::transpose_words_32x64(head_rows, scratch_b, scratch_a);
        Vector<uint32_t> * head_values =
            job.packed_values + static_cast<size_t>(head) * job.cache_capacity * d_tiles * kTileVectors;
        for (int d_tile = 0; d_tile < d_tiles; d_tile++) {
            Vector<uint32_t> * tile =
                head_values + (static_cast<size_t>(d_tile) * job.cache_capacity + key_tile) * kTileVectors;
            for (int line = 0; line < 16; line++) tile[line] = scratch_b[d_tile * 16 + line];
        }
    }
}

void pack_key_tile_in_vtcm(void * context, int key_tile, int thread) {
    AttentionJob & job = *static_cast<AttentionJob *>(context);
    const int rows = job.keys - key_tile * 32 < 32 ? job.keys - key_tile * 32 : 32;
    const size_t row_stride = job.token_stride;  // halfwords
    const size_t head_vectors = static_cast<size_t>(job.cache_capacity) * job.d_tiles * kTileVectors;
    Vector<uint32_t> * discarded_tile = job.scratch[thread];
    Vector<uint32_t> * scratch_a = job.scratch[thread] + 64;
    Vector<uint32_t> * scratch_b = scratch_a + 64;
    const Vector<uint32_t> zero = Vector<uint32_t>::zero();

    for (int head = 0; head < job.kv_heads; head++) {
        const size_t first_value =
            static_cast<size_t>(key_tile) * 32 * job.token_stride + static_cast<size_t>(head) * job.head_stride;
        Vector<uint32_t> * key_tiles = job.packed_keys + static_cast<size_t>(head) * head_vectors +
                                       static_cast<size_t>(key_tile) * job.d_tiles * kTileVectors;
        Vector<uint32_t> * value_tiles = job.packed_values + static_cast<size_t>(head) * head_vectors;
        for (int vector = 0; vector < (job.d_tiles + 1) / 2; vector++) {
            const _Float16 * key_rows = job.k + first_value + vector * 64;
            const bool second_kept = 2 * vector + 1 < job.d_tiles;
            for (int line = 0; line < 16; line++) {
                const Vector<uint32_t> even_row =
                    2 * line < rows ? hvx::load<uint32_t>(key_rows + (2 * line) * row_stride) : zero;
                const Vector<uint32_t> odd_row =
                    2 * line + 1 < rows ? hvx::load<uint32_t>(key_rows + (2 * line + 1) * row_stride) : zero;
                const VectorPair<uint32_t> pair = interleave_chunks<2>(odd_row, even_row);
                key_tiles[(2 * vector) * 16 + line] = pair.low();
                if (second_kept) key_tiles[(2 * vector + 1) * 16 + line] = pair.high();
            }

            Vector<uint32_t> * first_value_tile =
                value_tiles + (static_cast<size_t>(2 * vector) * job.cache_capacity + key_tile) * kTileVectors;
            Vector<uint32_t> * second_value_tile =
                second_kept
                    ? value_tiles + (static_cast<size_t>(2 * vector + 1) * job.cache_capacity + key_tile) * kTileVectors
                    : discarded_tile;
            transpose_rows(job.v + first_value + vector * 64, row_stride * 2, rows, first_value_tile, second_value_tile,
                           scratch_b, scratch_a);
        }
    }
}

// The per-thread scratch sits at the top of VTCM, above what plan_attention carves.
runtime::Status pack_kv_tiles(AttentionJob & job) {
    PackingJob packing{};
    packing.job = &job;
    packing.row_bytes = static_cast<size_t>(job.kv_heads) * job.head_dimensions * 2;
    const size_t scratch_bytes = packing_scratch_bytes(job);
    uint8_t * const top = job.session->vtcm() + job.session->vtcm_bytes() - runtime::kHvxThreads * scratch_bytes;

    runtime::HeapObject<runtime::DmaRing> rings[runtime::kHvxThreads];
    for (int thread = 0; thread < runtime::kHvxThreads; thread++) {
        packing.scratch[thread] = top + static_cast<size_t>(thread) * scratch_bytes;
        rings[thread] = runtime::allocate_object<runtime::DmaRing>(64);
        if (!rings[thread]) return std::unexpected(runtime::Error::kNoMemory);
        rings[thread]->initialize(runtime::DmaBypass::kBoth);
        packing.rings[thread] = rings[thread].get();
    }

    // the K / V rows come from HVX stores: written back for the DMA, which reads DDR
    runtime::flush_data_cache();
    return runtime::parallel_for(*job.session, runtime::kHvxThreads, job.key_tiles, pack_key_tile, &packing);
}

runtime::Status pack_kv_tiles_from_vtcm_rows(AttentionJob & job) {
    return runtime::parallel_for(*job.session, runtime::kHvxThreads, job.key_tiles, pack_key_tile_in_vtcm, &job);
}

constexpr size_t kTileHalfwords = kTileBytes / sizeof(_Float16);
constexpr size_t kOutputBufferBytes = 8192;
constexpr size_t kMaskTableBytes = 16 * hvx::kVectorBytes;

struct alignas(128) MaskTables {
    int16_t key_minus_query[16][64];
    int16_t key_index[16][64];
    uint16_t diagonal_masks[16][64];
    bool built;
};

struct Workspace {
    runtime::HeapObject<AttentionJob> job;
    runtime::HeapObject<runtime::DmaRing> rings[runtime::kHvxThreads + 1];  // one per worker, then the driver's
    runtime::HeapArray<Vector<uint32_t>> packed_keys;
    runtime::HeapArray<Vector<uint32_t>> packed_values;
    size_t packed_bytes = 0;
    MaskTables mask_tables;
    std::atomic<int> in_use;
};

void build_mask_tables(MaskTables & mask_tables) {
    memset(mask_tables.diagonal_masks, 0, sizeof(mask_tables.diagonal_masks));

    for (int line = 0; line < 16; line++) {
        uint16_t * diagonal = mask_tables.diagonal_masks[line];
        diagonal[4 * line] = 0xFFFF;
        diagonal[4 * line + 3] = 0xFFFF;

        int16_t * key_minus_query = mask_tables.key_minus_query[line];
        int16_t * key_index = mask_tables.key_index[line];
        for (int lane = 0; lane < 64; lane++) {
            key_index[lane] = static_cast<int16_t>(2 * line + (lane & 1));
            key_minus_query[lane] = static_cast<int16_t>(key_index[lane] - (lane >> 1));
        }
    }
    mask_tables.built = true;
}

void copy_mask_tables(MaskTables & mask_tables, AttentionJob & job) {
    if (!mask_tables.built) build_mask_tables(mask_tables);
    for (int line = 0; line < 16; line++) {
        job.key_minus_query[line] = hvx::load(mask_tables.key_minus_query[line]);
        job.key_index[line] = hvx::load(mask_tables.key_index[line]);
        job.diagonal_masks[line] = hvx::load(mask_tables.diagonal_masks[line]);
    }
}

runtime::Status use_kept_kv_buffers(Workspace & workspace, AttentionJob & job, size_t bytes) {
    if (bytes > workspace.packed_bytes) {
        workspace.packed_keys.reset();  // both freed before the larger pair is allocated
        workspace.packed_values.reset();
        workspace.packed_keys = runtime::allocate_array<Vector<uint32_t>>(bytes / hvx::kVectorBytes, 4096);
        workspace.packed_values = runtime::allocate_array<Vector<uint32_t>>(bytes / hvx::kVectorBytes, 4096);
        workspace.packed_bytes = workspace.packed_keys && workspace.packed_values ? bytes : 0;
    }

    job.packed_keys = workspace.packed_keys.get();
    job.packed_values = workspace.packed_values.get();
    if (workspace.packed_bytes == 0) return std::unexpected(runtime::Error::kNoMemory);
    return {};
}

size_t packed_kv_bytes(const AttentionJob & job) {
    return static_cast<size_t>(job.kv_heads) * job.key_tiles * job.d_tiles * kTileBytes;
}

bool arguments_supported(const Arguments & a) {
    const bool segments = a.segments != nullptr;
    if (segments && (a.keys != a.queries || a.kv_heads != a.query_heads || a.causal || a.queries % 32 != 0 ||
                     a.queries / 32 > kMaximumQueryBlocks || a.segments->count < 1 ||
                     a.segments->first_tokens == nullptr || a.segments->token_counts == nullptr)) {
        return false;
    }

    if (a.o == nullptr) return false;
    switch (a.source) {
        case Source::kRows:
            return a.q != nullptr && a.k != nullptr && a.v != nullptr &&
                   (a.tiles.key_tiles != nullptr) == (a.tiles.value_tiles != nullptr);
        case Source::kKeyValueCache:
            return !segments && a.q != nullptr && a.tiles.key_tiles != nullptr && a.tiles.value_tiles != nullptr;
        case Source::kTiles:
            return segments && a.query_tiles != nullptr && a.tiles.key_tiles != nullptr &&
                   a.tiles.value_tiles != nullptr;
        case Source::kRowsInVtcm:
            return segments && a.q != nullptr && a.k != nullptr && a.v != nullptr;
    }
    return false;
}

runtime::Status plan_segments(AttentionJob & job, const Segments & segments) {
    job.segment_count = segments.count;
    const int query_blocks = job.queries / 32;
    memset(job.segment_tiles, 0, sizeof(job.segment_tiles[0]) * static_cast<size_t>(query_blocks));
    int longest = 1;

    for (int segment = 0; segment < segments.count; segment++) {
        const int first_block = segments.first_tokens[segment] / 32;
        const int blocks = (segments.token_counts[segment] + 31) / 32;
        const int key_segment_valid =
            segments.key_segments == nullptr ||
            (segments.key_segments[segment] >= 0 && segments.key_segments[segment] < segments.count);
        if (segments.first_tokens[segment] % 32 != 0 || segments.token_counts[segment] < 1 ||
            segments.first_tokens[segment] + blocks * 32 > job.queries || !key_segment_valid) {
            return std::unexpected(runtime::Error::kBadParameter);
        }

        const int key_segment = segments.key_segments != nullptr ? segments.key_segments[segment] : segment;
        const int key_first_tile = segments.first_tokens[key_segment] / 32;
        const int key_tiles_of_segment = (segments.token_counts[key_segment] + 31) / 32;
        for (int block = first_block; block < first_block + blocks; block++) {
            job.segment_first_tile[block] = static_cast<int16_t>(key_first_tile);
            job.segment_tiles[block] = static_cast<int16_t>(key_tiles_of_segment);
            job.segment_keys[block] = static_cast<int16_t>(segments.token_counts[key_segment]);
        }
        longest = key_tiles_of_segment > longest ? key_tiles_of_segment : longest;
    }

    const int ranges = (longest + kMaximumBlockTiles - 1) / kMaximumBlockTiles;
    job.block_tiles = (longest + ranges - 1) / ranges;

    for (int block = 0; block < query_blocks; block++) {
        if (job.segment_tiles[block] == 0) return std::unexpected(runtime::Error::kBadParameter);
    }
    return {};
}

runtime::Status configure_job(AttentionJob & job, runtime::Session & session, const Arguments & a) {
    job.session = &session;
    job.queries = a.queries;
    job.keys = a.keys;
    job.query_heads = a.query_heads;
    job.kv_heads = a.kv_heads;
    job.head_dimensions = a.head_dimensions;
    job.causal = a.causal;
    job.stream_count = 12;
    job.token_stride = a.query_heads * a.head_dimensions;
    job.head_stride = a.head_dimensions;
    job.q = a.q;
    job.k = a.k;
    job.v = a.v;
    job.o = a.o;

    if (a.segments != nullptr) {
        job.rows_in_vtcm = a.source == Source::kRowsInVtcm;
        if (job.rows_in_vtcm) {
            job.token_stride = a.row_layout.token_stride;
            job.head_stride = a.row_layout.head_stride;
        }
        job.prepacked_queries = static_cast<const Vector<uint32_t> *>(a.query_tiles);
        return plan_segments(job, *a.segments);
    }
    if (a.source == Source::kRows) {
        job.block_tiles = 8;
        return {};
    }

    job.block_tiles = a.queries > 32 ? 16 : 8;
    job.cache_capacity = a.tiles.capacity;

    if (a.queries == 1 && !a.causal && a.query_heads > a.kv_heads) {
        // decode: a KV head's G query heads as the G queries of one block (query v of head h = query head h G + v)
        job.queries = a.query_heads / a.kv_heads;
        job.query_heads = a.kv_heads;
        job.token_stride = a.head_dimensions;
        job.head_stride = a.query_heads / a.kv_heads * a.head_dimensions;
    }
    if (job.queries <= 32 && !a.causal) {
        const int key_tiles = (a.keys + 31) / 32;
        job.split_count = key_tiles < kMaximumSplits ? key_tiles : kMaximumSplits;
    }
    return {};
}

template <class T>
T * take_vtcm(uint8_t *& vtcm, size_t bytes) {
    T * region = reinterpret_cast<T *>(vtcm);
    vtcm += bytes;
    return region;
}

AttentionJob * prepare_job(Workspace & workspace) {
    if (!workspace.job) {
        static_assert(std::is_trivially_destructible_v<AttentionJob>);  // freed without a destructor call
        workspace.job = runtime::allocate_object<AttentionJob>(128);
        if (workspace.job) new (workspace.job.get()) AttentionJob();
    }
    for (auto & ring : workspace.rings) {
        if (!ring) {
            ring = runtime::allocate_object<runtime::DmaRing>(
                256, (sizeof(runtime::DmaRing) + 255) & ~static_cast<size_t>(255));
        }
    }

    bool complete = static_cast<bool>(workspace.job);
    for (const auto & ring : workspace.rings) complete = complete && ring;
    if (!complete) return nullptr;

    AttentionJob & job = *workspace.job;
    // only the fields outside the stream and segment arrays are cleared (those are set before use)
    memset(static_cast<void *>(&job), 0, offsetof(AttentionJob, streams));
    memset(static_cast<void *>(&job.barrier), 0,
           offsetof(AttentionJob, segment_first_tile) - offsetof(AttentionJob, barrier));
    job.split_count = 0;
    job.split_tiles = 0;
    job.partials = nullptr;
    for (int thread = 0; thread < 5; thread++) job.rings[thread] = workspace.rings[thread].get();
    return &job;
}

runtime::Status plan_attention(AttentionJob & job) {
    if (job.queries < 1 || job.keys < 1 || job.kv_heads < 1 || job.query_heads % job.kv_heads != 0 ||
        job.head_dimensions < 1 || job.head_dimensions > 128 || job.head_dimensions % 8 != 0) {
        return std::unexpected(runtime::Error::kBadParameter);
    }

    job.group = job.query_heads / job.kv_heads;
    job.d_tiles = (job.head_dimensions + 31) / 32;
    if (job.token_stride == 0) {
        job.token_stride = job.head_dimensions;
        job.head_stride = job.queries * job.head_dimensions;
    }
    job.query_blocks = (job.queries + 31) / 32;
    job.key_tiles = (job.keys + 31) / 32;
    if (job.cache_capacity < job.key_tiles) job.cache_capacity = job.key_tiles;
    job.items = job.group * job.query_blocks * (job.split_count > 1 ? job.split_count : 1);
    if (job.split_count > 1) job.split_tiles = (job.key_tiles + job.split_count - 1) / job.split_count;
    const _Float16 score_scale =
        static_cast<_Float16>(1.4426950408889634 / sqrt(static_cast<double>(job.head_dimensions)));
    job.score_scale_bits = std::bit_cast<uint16_t>(score_scale);

    const size_t head_tile_bytes = static_cast<size_t>(job.key_tiles) * job.d_tiles * kTileBytes;
    // With rows in VTCM, Q^T and O^T have an even number of tiles: a row vector's transpose is a tile pair.
    const int stream_d_tiles = job.rows_in_vtcm ? (job.d_tiles + 1) / 2 * 2 : job.d_tiles;
    const size_t stream_bytes = (static_cast<size_t>(stream_d_tiles) * 2 + job.block_tiles + 2) * kTileBytes;
    const size_t prepacked_bytes =
        job.prepacked_queries != nullptr ? static_cast<size_t>(job.query_blocks) * job.d_tiles * kTileBytes : 0;
    const size_t kv_bytes =
        job.rows_in_vtcm ? 2 * job.kv_heads * head_tile_bytes : 4 * head_tile_bytes + 4 * packing_scratch_bytes(job);
    const size_t partial_bytes =
        job.split_count > 1 ? static_cast<size_t>(job.group) * job.split_count * sizeof(PartialResult) : 0;
    const size_t output_buffer_bytes =
        job.prepacked_queries != nullptr ? 4 * kOutputBuffersInFlight * kOutputBufferBytes : 0;
    const size_t needed = kv_bytes + job.stream_count * stream_bytes + 4 * kWorkerScratchBytes + 3 * kMaskTableBytes +
                          partial_bytes + 2 * prepacked_bytes + output_buffer_bytes;

    if (job.prepacked_queries != nullptr && job.group != 1) return std::unexpected(runtime::Error::kBadParameter);
    if (job.rows_in_vtcm &&
        (job.head_dimensions % 32 != 0 || job.head_stride % 64 != 0 || job.head_stride < job.head_dimensions ||
         job.token_stride < job.kv_heads * job.head_stride || job.prepacked_queries != nullptr || job.group != 1 ||
         job.split_count > 1 || job.cache_capacity != job.key_tiles)) {
        return std::unexpected(runtime::Error::kBadParameter);
    }
    if (needed > job.session->vtcm_bytes()) return std::unexpected(runtime::Error::kNoMemory);

    uint8_t * vtcm = job.session->vtcm();
    if (job.rows_in_vtcm) {
        job.packed_keys = take_vtcm<Vector<uint32_t>>(vtcm, job.kv_heads * head_tile_bytes);
        job.packed_values = take_vtcm<Vector<uint32_t>>(vtcm, job.kv_heads * head_tile_bytes);
    } else {
        for (int buffer = 0; buffer < 2; buffer++) {
            job.vtcm_keys[buffer] = take_vtcm<Vector<uint32_t>>(vtcm, head_tile_bytes);
            job.vtcm_values[buffer] = take_vtcm<Vector<uint32_t>>(vtcm, head_tile_bytes);
        }
    }

    for (int index = 0; index < job.stream_count; index++) {
        Stream & stream = job.streams[index];
        stream.query_tiles = take_vtcm<Vector<uint32_t>>(vtcm, static_cast<size_t>(stream_d_tiles) * kTileBytes);
        stream.output_tiles = take_vtcm<Vector<uint32_t>>(vtcm, static_cast<size_t>(stream_d_tiles) * kTileBytes);
        if (stream_d_tiles > job.d_tiles) {  // HMX writes d_tiles O^T tiles: the rows' padding columns stay zero
            zero_vectors(stream.output_tiles + static_cast<size_t>(job.d_tiles) * kTileVectors,
                         static_cast<size_t>(stream_d_tiles - job.d_tiles) * kTileVectors);
        }
        stream.score_tiles = take_vtcm<Vector<_Float16>>(vtcm, static_cast<size_t>(job.block_tiles) * kTileBytes);
        stream.rescale_tile = take_vtcm<Vector<uint16_t>>(vtcm, kTileBytes);
        stream.score_store_block = take_vtcm<Vector<uint32_t>>(vtcm, kTileBytes);
        stream.output_store_block = stream.score_store_block + 2;
        zero_vectors(stream.rescale_tile, kTileVectors);
        zero_vectors(stream.score_store_block, 4);
    }

    for (int thread = 0; thread < 4; thread++) {
        job.scratch[thread] = take_vtcm<Vector<uint32_t>>(vtcm, kWorkerScratchBytes);
        zero_vectors(job.scratch[thread], kWorkerScratchBytes / hvx::kVectorBytes);
    }

    if (job.split_count > 1) job.partials = take_vtcm<PartialResult>(vtcm, partial_bytes);
    if (job.prepacked_queries != nullptr) {
        job.vtcm_queries[0] = take_vtcm<Vector<uint32_t>>(vtcm, prepacked_bytes);
        job.vtcm_queries[1] = take_vtcm<Vector<uint32_t>>(vtcm, prepacked_bytes);
        for (int thread = 0; thread < 4; thread++) {
            for (int buffer = 0; buffer < kOutputBuffersInFlight; buffer++) {
                job.output_buffers[thread][buffer] = take_vtcm<Vector<uint32_t>>(vtcm, kOutputBufferBytes);
            }
        }
    }

    job.key_minus_query = take_vtcm<Vector<int16_t>>(vtcm, kMaskTableBytes);
    job.key_index = take_vtcm<Vector<int16_t>>(vtcm, kMaskTableBytes);
    job.diagonal_masks = take_vtcm<Vector<uint16_t>>(vtcm, kMaskTableBytes);
    return {};
}

}  // namespace

runtime::Status run(runtime::Session & session, const Arguments & arguments) {
    using runtime::Error;
    if (!arguments_supported(arguments)) return std::unexpected(Error::kBadParameter);

    Workspace * kept = session.state<Workspace>();
    if (kept == nullptr) return std::unexpected(Error::kNoMemory);
    Workspace & workspace = *kept;
    const runtime::ExclusiveUse use(workspace.in_use);
    if (!use.taken()) return std::unexpected(Error::kBadParameter);
    AttentionJob * job = prepare_job(workspace);
    if (job == nullptr) return std::unexpected(Error::kNoMemory);

    if (const runtime::Status configured = configure_job(*job, session, arguments); !configured) return configured;
    if (const runtime::Status planned = plan_attention(*job); !planned) return planned;
    if (job->cache_capacity != arguments.tiles.capacity && arguments.source == Source::kKeyValueCache) {
        return std::unexpected(Error::kBadParameter);  // the cache is too small
    }
    copy_mask_tables(workspace.mask_tables, *job);
    if (job->rows_in_vtcm) job->items = arguments.query_heads * job->query_blocks;

    if (arguments.source == Source::kRows && arguments.tiles.key_tiles == nullptr) {
        if (const runtime::Status buffers = use_kept_kv_buffers(workspace, *job, packed_kv_bytes(*job)); !buffers) {
            return buffers;
        }
    } else if (!job->rows_in_vtcm) {
        job->packed_keys = static_cast<Vector<uint32_t> *>(arguments.tiles.key_tiles);
        job->packed_values = static_cast<Vector<uint32_t> *>(arguments.tiles.value_tiles);
    }
    if (arguments.source == Source::kRows || arguments.source == Source::kRowsInVtcm) {
        const runtime::Status packed = job->rows_in_vtcm ? pack_kv_tiles_from_vtcm_rows(*job) : pack_kv_tiles(*job);
        if (!packed) return packed;
    }

    if (!job->rows_in_vtcm) {  // the DMA reads DDR
        runtime::flush_data_cache();
    }
    return run_job(*job);
}

void append_to_cache(const _Float16 * key, const _Float16 * value, int position, int kv_heads, int head_dimensions,
                     const KeyValueTiles & cache) {
    const int d_tiles = (head_dimensions + 31) / 32;
    const int key_tile = position / 32;
    const int row = position % 32;

    for (int head = 0; head < kv_heads; head++) {
        const size_t head_offset = static_cast<size_t>(head) * cache.capacity * d_tiles * kTileHalfwords;
        _Float16 * key_tiles = static_cast<_Float16 *>(cache.key_tiles) + head_offset;
        _Float16 * value_tiles = static_cast<_Float16 *>(cache.value_tiles) + head_offset;
        const _Float16 * key_row = key + static_cast<size_t>(head) * head_dimensions;
        const _Float16 * value_row = value + static_cast<size_t>(head) * head_dimensions;
        for (int d = 0; d < head_dimensions; d++) {
            const int d_tile = d / 32;
            const int d_in_tile = d % 32;
            key_tiles[(static_cast<size_t>(key_tile) * d_tiles + d_tile) * kTileHalfwords + (row / 2) * 64 +
                      d_in_tile * 2 + (row & 1)] = key_row[d];
            value_tiles[(static_cast<size_t>(d_tile) * cache.capacity + key_tile) * kTileHalfwords +
                        (d_in_tile / 2) * 64 + row * 2 + (d_in_tile & 1)] = value_row[d];
        }
    }
}

}  // namespace benzene::kernels::flash_attention
