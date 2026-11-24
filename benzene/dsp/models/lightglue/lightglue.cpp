#include "dsp/models/lightglue/lightglue.h"

#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>

#include "dsp/hmx/instructions.h"
#include "dsp/hvx/dot_products.h"
#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/prefetch.h"
#include "dsp/hvx/reductions.h"
#include "dsp/hvx/scalar_math.h"
#include "dsp/hvx/transpose.h"
#include "dsp/hvx/vector.h"
#include "dsp/kernels/flash_attention/flash_attention.h"
#include "dsp/kernels/gemm_fp16/gemm_fp16.h"
#include "dsp/models/shared/feature_slots.h"
#include "dsp/models/shared/layer_weights.h"
#include "dsp/runtime/dma_ring.h"
#include "dsp/runtime/heap.h"
#include "dsp/runtime/session.h"
#include "dsp/runtime/stage_timer.h"
#include "dsp/runtime/status.h"
#include "dsp/runtime/synchronization.h"
#include "dsp/runtime/thread_pool.h"
#include "dsp/vector_math/functions.h"
#include "benzene/protocol.h"

namespace benzene::models::lightglue {
namespace {

namespace part = protocol::lightglue::part;
namespace stat = protocol::lightglue::stat;
namespace gemm_fp16 = kernels::gemm_fp16;
using hvx::Mask;
using hvx::Vector;
using hvx::VectorPair;
using Model = protocol::lightglue::Model;

// The most token rows: n0p + n1p, each image's tokens rounded up to 32.
constexpr int kMaximumTokens = 4096;

template <int DimensionsValue, int HeadsValue, int LayersValue, int InputDimensionsValue>
struct LightGlueShape {
    static constexpr int kDimensions = DimensionsValue;  // D
    static constexpr int kHeads = HeadsValue;
    static constexpr int kHeadDimensions = DimensionsValue / HeadsValue;
    static constexpr int kLayers = LayersValue;
    static constexpr int kInputDimensions = InputDimensionsValue;
    static constexpr bool kInputProjection = InputDimensionsValue != DimensionsValue;
    // Halfwords of a head's columns in a q / k / v / o row, of a q / k / v / o row and of an x row: whole HVX vectors,
    // the columns beyond the dimensions zero.
    static constexpr int kHeadStride = hvx::round_up(kHeadDimensions, 64);
    static constexpr int kAttentionRowStride = kHeads * kHeadStride;
    static constexpr int kTokenRowStride = hvx::round_up(DimensionsValue, 64);
    static constexpr int kTiles = DimensionsValue / 32;  // HMX tiles of a D-wide matrix
    // Row vectors of a D-wide matrix; with an odd kTiles, the buffers have an extra tile, kept zero.
    static constexpr int kTilePairs = (kTiles + 1) / 2;
    static constexpr int kFfnTiles = 2 * kTiles;
    static_assert(DimensionsValue % 32 == 0 && kHeadDimensions % 32 == 0 && InputDimensionsValue % 64 == 0);
    static_assert(kHeads == 1 || kHeadDimensions == 64, "several heads: one vector each");
};

using SuperPointLightGlue = LightGlueShape<256, 4, protocol::lightglue::layers(Model::kSuperPoint),
                                           protocol::superpoint::kDescriptorDimensions>;
using XFeatLightGlue =
    LightGlueShape<96, 1, protocol::lightglue::layers(Model::kXFeat), protocol::xfeat::kDescriptorDimensions>;
using AlikedLightGlue =
    LightGlueShape<256, 4, protocol::lightglue::layers(Model::kAliked), protocol::aliked::kDescriptorDimensions>;

template <protocol::lightglue::Model kModel>
struct Variant;

struct LightGlueConventions {
    // The slots' keypoints are in pixels, normalized as cvg/LightGlue does: (kp - size / 2) / (max(size) / 2); else
    // they come normalized per axis ([-1, 1]).
    static constexpr bool kPixelKeypoints = true;
    // The assignment scores add logsigmoid(x wm + bm) of both tokens; else the dual softmax alone.
    static constexpr bool kMatchability = true;
    // > 0: the attention calls whose bound on |q . k| exceeds it run in f32 on HVX instead of on HMX.
    static constexpr float kPreciseScoreBound = 0.0f;
};

template <>
struct Variant<protocol::lightglue::Model::kSuperPoint> : LightGlueConventions {
    using Shape = SuperPointLightGlue;
};

template <>
struct Variant<protocol::lightglue::Model::kXFeat> : LightGlueConventions {
    using Shape = XFeatLightGlue;
};

template <>
struct Variant<protocol::lightglue::Model::kAliked> : LightGlueConventions {
    using Shape = AlikedLightGlue;
};

// LoMa-B: its matchability head is unused; its cross blocks 4 and 5 are nearly hard argmaxes, with |q . k| beyond the
// precision of the HMX fp16 accumulator (the other calls stay below the bound).
template <>
struct Variant<protocol::lightglue::Model::kLoma> {
    using Shape = SuperPointLightGlue;
    static constexpr bool kPixelKeypoints = false;
    static constexpr bool kMatchability = false;
    static constexpr float kPreciseScoreBound = 8192.0f;
};

// LightGlue's LearnableFourierPositionalEncoding, computed as LightGlue does in fp16: table rows
// [rows][2][Shape::kHeadStride], cos | sin, zero beyond the head's d dimensions and in padding rows. Token r:
// - its keypoint normalized as normalize_keypoints ((kp - size / 2) / (max(size) / 2), f32) and rounded to fp16;
// - angle p = fp16(Wr[p] . kp) (Wr in fp16; the two exact products summed in f32);
// - cos / sin rounded to fp16 and duplicated over the pair's lanes 2p, 2p + 1.
struct RotaryTableJob {
    _Float16 * table;
    const float * keypoints[2];  // per image [n][2]
    const float * frequencies;   // Wr [d / 2][2], fp16 values
    int first_image_tokens;      // the images' rows [0, n0) and [n0p, n0p + n1)
    int second_image_tokens;
    int second_image_first_row;
    int rows;
    float half_size[2][2];   // per image: size / 2 (x, y)
    float half_max_size[2];  // per image: max(size) / 2
};

// q, k of the token rows rotated as apply_cached_rotary_emb in fp16 (every head by the same table row).
struct RotationJob {
    _Float16 * queries;  // rows of Shape::kAttentionRowStride
    _Float16 * keys;
    const _Float16 * table;  // RotaryTableJob rows
    int rows;
    const int * table_rows;  // the table row of each token row
};

constexpr int kRotationRowsPerItem = 8;

// Attention over heads of 64 dimensions; as kernels::flash_attention::Segments, the queries of segment i attend to the
// keys of segment key_segments[i].
struct PreciseAttentionArguments {
    const _Float16 * queries;  // token rows [tokens][row_stride], head h's dimensions from halfword 64 h
    const _Float16 * keys;
    const _Float16 * values;
    _Float16 * outputs;  // may be `keys` (the keys are read before any output row is written)
    int row_stride;      // halfwords (128-byte multiples)
    int heads;
    int segment_count;         // 1 or 2
    const int * first_tokens;  // multiples of 32
    const int * token_counts;  // the rows after a segment's tokens, up to a multiple of 32, are zero
    const int * key_segments;  // nullptr: its own
};

struct BlockPipelineArguments {
    _Float16 * attention_output;  // O rows [rows][Shape::kAttentionRowStride]
    _Float16 * token_rows;        // x rows [rows][Shape::kTokenRowStride]
    int rows;                     // a multiple of 32, at most kMaximumTokens
    // fp16 GEMM weight streams (kernels::gemm_fp16 layout): out projection (K = the O row width, N = D), FFN1 (K 2D,
    // N 2D), FFN2 (K 2D, N D), and the next block's projections (K = the x row width, N = 32 projection_blocks) or
    // nullptr
    const uint8_t * weights[4];
    const float * layer_norm;  // gamma [2D] | beta [2D]
    // the projections' column blocks (3D / 32 self, 2D / 32 cross), into projection_rows[block / (D / 32)] (rows of
    // Shape::kAttentionRowStride)
    int projection_blocks;
    _Float16 * projection_rows[3];
    int first_image_tokens;  // the images' rows [0, n0) and [n0p, n0p + n1), the rest padding
    int second_image_first_row;
    int second_image_tokens;
};

// The heads' parameters in one buffer: the token confidence's weight [D] (zero up to kTokenRowStride) and its bias at
// kTokenRowStride, the matchability's the same from matchability_parameters<Shape>() (a vector boundary).
template <class Shape>
constexpr int matchability_parameters() {
    return hvx::round_up(Shape::kTokenRowStride + 1, 32);
}
template <class Shape>
constexpr int head_parameter_floats() {
    return 2 * matchability_parameters<Shape>();
}

// z = fp16(x . w + b) per token row for the token confidence and / or the matchability.
struct HeadJob {
    const _Float16 * token_rows;  // x rows (Shape::kTokenRowStride)
    const float * parameters;
    float * confidence_logits;
    float * matchability_logits;
    int confidence;  // the heads to evaluate
    int matchability;
};

// Rows r < rows of each array from row source_rows[r] >= r (so in place), or zero where it is -1.
struct MoveJob {
    _Float16 * arrays[4];  // x (Shape::kTokenRowStride), Q, K, V (Shape::kAttentionRowStride); nullptr: not moved
    const int * source_rows;
    int rows;
};

// ls = logsigmoid(z), z = x . wm + bm, per token row (padding rows 0).
struct MatchabilityJob {
    const _Float16 * token_rows;  // x rows [N][Shape::kTokenRowStride]
    const float * parameters;     // matchability weight [D] (zero up to kTokenRowStride), bias at kTokenRowStride
    float * log_sigmoids;
    int first_image_tokens;  // the images' rows [0, n0) and [n0p, n0p + n1)
    int second_image_first_row;
    int second_image_tokens;
};

// The similarity GEMM's K in tiles: the projected descriptors' whole rows (zero beyond D).
template <class Shape>
constexpr int similarity_k_tiles() {
    return Shape::kTokenRowStride / 32;
}

// Image 1's projected descriptors d1^T as the similarity GEMM's weight stream (kernels::gemm_fp16 layout): per
// 32-key block similarity_k_tiles() tiles (dimension pairs x keys) + an output block (scale 1, bias 0).
struct SimilarityWeightsJob {
    const _Float16 * second_image_descriptors;  // [n1p][Shape::kTokenRowStride]
    hvx::Vector<uint32_t> * stream;
};

// LightGlue's log_assignment: scores = 2 sim - lse_rows - lse_cols + ls0 + ls1.
struct AssignmentJob {
    const float * similarity;  // [n0p][stride]
    int first_image_tokens;
    int second_image_tokens;
    int stride;  // n1p
    float * row_lse;
    float * column_lse;
    const float * first_log_sigmoids;  // ls0
    float * row_max_scores;
    int * row_matches;
    int * column_matches;
    float * column_terms;          // ls1 - lse_cols per column
    float * row_terms;             // ls0 - lse_rows per row
    const int32_t * lane_indices;  // 0 .. 31
};

// LightGlue's filter_matches for rows 32 item ..: mutual maxima with exp(max) > threshold; outputs by keypoint index.
struct MatchFilterJob {
    const int * row_matches;
    const int * column_matches;
    const int * first_keypoints;  // the keypoint index of each token
    const int * second_keypoints;
    const float * row_max_scores;
    float threshold;
    int first_image_tokens;
    int32_t * matches;
    float * match_scores;
    std::atomic<int> match_count;
};

// sin and cos of f32 lanes (|x| < 1e4):
// - k = the nearest integer of x 2 / pi (t = x 2 / pi + 1.5 * 2^23 rounds it; t's low bits are k's);
// - r = x - k pi / 2 in three parts (Cody-Waite);
// - Cephes sinf / cosf polynomials on [-pi/4, pi/4], swapped and negated by the quadrant k & 3.
inline void sine_and_cosine(Vector<float> x, Vector<float> * sine, Vector<float> * cosine) {
    const Vector<float> rounding = Vector<float>::splat(12582912.0f);
    const Vector<float> t = x * 0.63661977236758134f + rounding;
    const Vector<float> k = t - rounding;
    Vector<float> r = x - k * 1.5703125f;
    r = r - k * 4.837512969970703125e-4f;
    r = r - k * 7.54978995489188216e-8f;

    const Vector<float> z = r * r;
    Vector<float> sine_polynomial = z * -1.9515295891e-4f + 8.3321608736e-3f;
    sine_polynomial = sine_polynomial * z + -1.6666654611e-1f;
    const Vector<float> sine_r = sine_polynomial * z * r + r;

    Vector<float> cosine_polynomial = z * 2.443315711809948e-5f + -1.388731625493765e-3f;
    cosine_polynomial = cosine_polynomial * z + 4.166664568298827e-2f;
    const Vector<float> cosine_r = Vector<float>::splat(1.0f) - z * 0.5f + cosine_polynomial * z * z;

    const Mask swap = (reinterpret<int32_t>(t) & 1) == Vector<int32_t>::splat(1);
    const Vector<int32_t> sine_sign = (reinterpret<int32_t>(t) & 2) << 30;
    const Vector<int32_t> cosine_sign = ((reinterpret<int32_t>(t) + 1) & 2) << 30;
    *sine = select(swap, cosine_r, sine_r) ^ reinterpret<float>(sine_sign);
    *cosine = select(swap, sine_r, cosine_r) ^ reinterpret<float>(cosine_sign);
}

// q, k rotated as apply_cached_rotary_emb in fp16: t c + rotate_half(t) s, each product and the sum rounded.
// rotate_half: lanes 2p, 2p + 1 <- -t[2p + 1], t[2p], i.e. the halfwords of each word swapped, the low one negated.
template <class Shape>
inline void rotate_row(const RotationJob * job, int row) {
    constexpr int kVectors = Shape::kHeadStride / 64;
    const Vector<_Float16> * table_row = hvx::vectors(job->table) + 2 * kVectors * job->table_rows[row];
    Vector<_Float16> cosine[kVectors];
    Vector<_Float16> sine[kVectors];
    for (int v = 0; v < kVectors; v++) {
        cosine[v] = table_row[v];
        sine[v] = table_row[kVectors + v];
    }

    const Vector<uint32_t> negate_low = Vector<uint32_t>::splat(0x8000);
    for (int w = 0; w < 2; w++) {
        auto * values =
            hvx::vectors((w != 0 ? job->keys : job->queries) + static_cast<size_t>(row) * Shape::kAttentionRowStride);
        for (int head = 0; head < Shape::kHeads; head++) {
            for (int v = 0; v < kVectors; v++) {
                const Vector<_Float16> t = values[head * kVectors + v];
                const Vector<_Float16> rotated = reinterpret<_Float16>(
                    ((reinterpret<uint32_t>(t) >> 16) | (reinterpret<uint32_t>(t) << 16)) ^ negate_low);
                values[head * kVectors + v] = t * cosine[v] + rotated * sine[v];
            }
        }
    }
}

template <class Shape>
void compute_rotary_rows(void * context, int item, int) {
    constexpr int kPairs = Shape::kHeadDimensions / 2;
    constexpr int kAngleVectors = (kPairs + 31) / 32;
    constexpr int kVectors = Shape::kHeadStride / 64;
    const auto * job = static_cast<const RotaryTableJob *>(context);

    alignas(128) float first_frequencies[kAngleVectors][32];
    alignas(128) float second_frequencies[kAngleVectors][32];
    for (int p = 0; p < 32 * kAngleVectors; p++) {
        first_frequencies[p / 32][p % 32] = p < kPairs ? job->frequencies[2 * p] : 0.0f;
        second_frequencies[p / 32][p % 32] = p < kPairs ? job->frequencies[2 * p + 1] : 0.0f;
    }

    Vector<float> first_weights[kAngleVectors];
    Vector<float> second_weights[kAngleVectors];
    for (int a = 0; a < kAngleVectors; a++) {
        first_weights[a] = hvx::load(first_frequencies[a]);
        second_weights[a] = hvx::load(second_frequencies[a]);
    }

    for (int row = item * 32; row < item * 32 + 32 && row < job->rows; row++) {
        auto * out = hvx::vectors(job->table + static_cast<size_t>(row) * 2 * Shape::kHeadStride);
        const int image = row >= job->second_image_first_row;
        const int index = image != 0 ? row - job->second_image_first_row : row;
        if (index >= (image != 0 ? job->second_image_tokens : job->first_image_tokens)) {
            for (int v = 0; v < 2 * kVectors; v++) out[v] = Vector<_Float16>::zero();
            continue;
        }

        const float * keypoint = job->keypoints[image] + 2 * static_cast<size_t>(index);
        const auto x0 = static_cast<float>(
            static_cast<_Float16>((keypoint[0] - job->half_size[image][0]) / job->half_max_size[image]));
        const auto x1 = static_cast<float>(
            static_cast<_Float16>((keypoint[1] - job->half_size[image][1]) / job->half_max_size[image]));

        for (int a = 0; a < kAngleVectors; a++) {
            const Vector<float> angle_f32 = first_weights[a] * x0 + second_weights[a] * x1;
            // the angle rounded to fp16
            const Vector<float> angle = to_float32_deinterleaved(to_float16_interleaved(angle_f32, angle_f32)).low();
            Vector<float> sine;
            Vector<float> cosine;
            sine_and_cosine(angle, &sine, &cosine);
            if (32 * (a + 1) > kPairs) {
                const Mask pairs = Mask::first_bytes(4 * (kPairs - 32 * a));
                sine = select(pairs, sine, Vector<float>::zero());
                cosine = select(pairs, cosine, Vector<float>::zero());
            }
            out[a] = to_float16_interleaved(cosine, cosine);
            out[kVectors + a] = to_float16_interleaved(sine, sine);
        }
    }
}

template <class Shape>
void rotate_rows(void * context, int item, int) {
    constexpr size_t kTableRowHalfwords = 2 * Shape::kHeadStride;
    const auto * job = static_cast<const RotationJob *>(context);

    if ((item + 4) * kRotationRowsPerItem < job->rows) {
        const uint32_t bytes = (kRotationRowsPerItem + 4) * kTableRowHalfwords * 2;
        hvx::prefetch_to_l2(
            job->table + static_cast<size_t>(job->table_rows[(item + 4) * kRotationRowsPerItem]) * kTableRowHalfwords,
            bytes, bytes, 1);
    }

    for (int row = item * kRotationRowsPerItem; row < (item + 1) * kRotationRowsPerItem && row < job->rows; row++) {
        rotate_row<Shape>(job, row);
    }
}

constexpr int kPreciseHeadDimensions = 64;
constexpr int kMaximumHeads = 4;
constexpr int kChunkBlocks = 8;
constexpr float kScoreScale = 1.4426950408889634f / 8.0f;  // log2(e) / sqrt(64)

struct PreciseAttentionWorkspace {
    runtime::HeapArray<float> scores;  // per thread [2][kMaximumTokens + 32]
};

constexpr size_t kScoreFloats = kMaximumTokens + 32;  // room for a zero vector after an odd vector count

runtime::Status allocate_precise_attention_workspace(PreciseAttentionWorkspace & workspace) {
    if (workspace.scores) return {};
    workspace.scores = runtime::allocate_array<float>(static_cast<size_t>(runtime::kHvxThreads) * 2 * kScoreFloats);
    if (!workspace.scores) return std::unexpected(runtime::Error::kNoMemory);
    return {};
}

inline Vector<_Float16> splat_word(Vector<_Float16> v, int word) {
    return reinterpret<_Float16>(Vector<int32_t>::splat(extract_word(v, word)));
}

inline int key_segment(const PreciseAttentionArguments & arguments, int segment) {
    return arguments.key_segments != nullptr ? arguments.key_segments[segment] : segment;
}

inline int segment_blocks(const PreciseAttentionArguments & arguments, int segment) {
    return (arguments.token_counts[segment] + 31) / 32;
}

template <class T>
inline T * head_columns(T * rows, int row_stride, int row, int head) {
    return rows + static_cast<size_t>(row) * row_stride + head * kPreciseHeadDimensions;
}

struct PrepareJob {
    const PreciseAttentionArguments * arguments;
    int blocks;
    // [heads][blocks][32]: vector j = dimensions 2j, 2j + 1 of the block's 32 keys (word lane i: key i)
    Vector<uint32_t> * key_pairs;
    // [heads][blocks * 16][2]: per pair of keys m, word lane d = (v[2m][d], v[2m + 1][d]) for d = 0 .. 31, 32 .. 63
    Vector<_Float16> * value_pairs;
};

void prepare_block(void * context, int item, int) {
    const auto * job = static_cast<const PrepareJob *>(context);
    const PreciseAttentionArguments & arguments = *job->arguments;
    const int head = item / job->blocks;
    const int block = item % job->blocks;
    const size_t first_vector = (static_cast<size_t>(head) * job->blocks + block) * 32;

    const size_t row_bytes = static_cast<size_t>(arguments.row_stride) * 2;
    const _Float16 * keys =
        arguments.keys + static_cast<size_t>(block) * 32 * arguments.row_stride + head * kPreciseHeadDimensions;
    Vector<uint32_t> * key_pairs = job->key_pairs + first_vector;
    Vector<uint32_t> scratch[2][32];
    hvx::transpose_words_32x32(keys, row_bytes, 32, key_pairs, key_pairs + 16, scratch[0], scratch[1]);

    Vector<_Float16> * value_pairs = job->value_pairs + first_vector;
    for (int m = 0; m < 16; m++) {
        const _Float16 * even_row = arguments.values +
                                    (static_cast<size_t>(block) * 32 + 2 * m) * arguments.row_stride +
                                    head * kPreciseHeadDimensions;
        const auto * even = hvx::vectors(even_row);
        const auto * odd = hvx::vectors(even_row + arguments.row_stride);
        const VectorPair<_Float16> interleaved = interleave(*odd, *even);
        value_pairs[2 * m] = interleaved.low();
        value_pairs[2 * m + 1] = interleaved.high();
    }
}

struct QueryJob {
    const PreciseAttentionArguments * arguments;
    int query_blocks[2];
    int blocks;
    const Vector<uint32_t> * key_pairs;
    const Vector<_Float16> * value_pairs;
    float * scores;
};

void score_pair(HVX_Vector q0, HVX_Vector q1, const Vector<uint32_t> * key_pairs, int keys, float * s0, float * s1) {
    const int blocks = (keys + 31) / 32;

    for (int chunk = 0; chunk < blocks; chunk += kChunkBlocks) {
        const int count = blocks - chunk < kChunkBlocks ? blocks - chunk : kChunkBlocks;
        Vector<float> a0[kChunkBlocks];
        Vector<float> a1[kChunkBlocks];
        for (int b = 0; b < kChunkBlocks; b++) a0[b] = a1[b] = Vector<float>::zero();
        const Vector<uint32_t> * pairs = key_pairs + static_cast<size_t>(chunk) * 32;

        if (count == kChunkBlocks) {
            for (int j = 0; j < 32; j++) {
                const Vector<_Float16> w0 = splat_word(Vector<_Float16>{q0}, j);
                const Vector<_Float16> w1 = splat_word(Vector<_Float16>{q1}, j);
#pragma clang loop unroll(full)
                for (int b = 0; b < kChunkBlocks; b++) {
                    const Vector<_Float16> k = reinterpret<_Float16>(pairs[b * 32 + j]);
                    a0[b] = dot_product_accumulate(a0[b], k, w0);
                    a1[b] = dot_product_accumulate(a1[b], k, w1);
                }
            }
        } else {
            for (int j = 0; j < 32; j++) {
                const Vector<_Float16> w0 = splat_word(Vector<_Float16>{q0}, j);
                const Vector<_Float16> w1 = splat_word(Vector<_Float16>{q1}, j);
                for (int b = 0; b < count; b++) {
                    const Vector<_Float16> k = reinterpret<_Float16>(pairs[b * 32 + j]);
                    a0[b] = dot_product_accumulate(a0[b], k, w0);
                    a1[b] = dot_product_accumulate(a1[b], k, w1);
                }
            }
        }

        for (int b = 0; b < count; b++) {
            hvx::vectors(s0)[chunk + b] = a0[b];
            hvx::vectors(s1)[chunk + b] = a1[b];
        }
    }
}

// The probabilities' numerators e = exp2(s log2(e) / 8 - max) of `keys` scores in place (zero past the keys up to an
// even vector count); returns sum e.
float exponentials(float * s, int keys) {
    const int vectors = (keys + 31) / 32;
    auto * v = hvx::vectors(s);
    const Vector<float> scale = Vector<float>::splat(kScoreScale);
    const Vector<float> lowest = Vector<float>::splat(-3.0e38f);
    const Mask last_lanes = Mask::first_bytes_up_to_128(4 * (keys - 32 * (vectors - 1)));
    Vector<float> maximum = lowest;

    for (int i = 0; i < vectors; i++) {
        Vector<float> x = v[i] * scale;
        if (i + 1 == vectors) x = select(last_lanes, x, lowest);
        v[i] = x;
        maximum = max(maximum, x);
    }
    maximum = vector_math::max_to_all_lanes(maximum);
    Vector<float> sum = Vector<float>::zero();

    // exp2 clamps its argument at -100 (fraction 0, polynomial 1.0): exactly 2^-100 there, so a batch with every lane
    // at or below -100 takes that value without the polynomial.
    const Vector<float> clamp = Vector<float>::splat(-100.0f);
    const Vector<float> at_clamp = Vector<float>::splat(0x1p-100f);
    int i = 0;
    for (; i + 8 <= vectors; i += 8) {
        Vector<float> e[8];
        Mask above = clamp > clamp;  // no lane set
#pragma clang loop unroll(full)
        for (int k = 0; k < 8; k++) {
            e[k] = v[i + k] - maximum;
            above = above | (e[k] > clamp);
        }

        if (any_lane(above)) {
            vector_math::exp2_batch<8>(e);
        } else {
#pragma clang loop unroll(full)
            for (int k = 0; k < 8; k++) e[k] = at_clamp;
        }

#pragma clang loop unroll(full)
        for (int k = 0; k < 8; k++) {
            if (i + k + 1 == vectors) e[k] = select(last_lanes, e[k], Vector<float>::zero());
            v[i + k] = e[k];
            sum = sum + e[k];
        }
    }
    for (; i < vectors; i++) {
        Vector<float> e = vector_math::exp2(v[i] - maximum);
        if (i + 1 == vectors) e = select(last_lanes, e, Vector<float>::zero());
        v[i] = e;
        sum = sum + e;
    }

    if (vectors % 2 != 0) v[vectors] = Vector<float>::zero();
    return horizontal_sum(sum);
}

void attend_block(void * context, int item, int thread) {
    const auto * job = static_cast<const QueryJob *>(context);
    const PreciseAttentionArguments & arguments = *job->arguments;
    const int blocks_per_head = job->query_blocks[0] + (arguments.segment_count > 1 ? job->query_blocks[1] : 0);
    const int head = item / blocks_per_head;
    int block = item % blocks_per_head;
    const int segment = block < job->query_blocks[0] ? 0 : 1;
    if (segment == 1) block -= job->query_blocks[0];

    const int keys_segment = key_segment(arguments, segment);
    const int keys = arguments.token_counts[keys_segment];
    const size_t first_block = static_cast<size_t>(head) * job->blocks + arguments.first_tokens[keys_segment] / 32;
    const Vector<uint32_t> * key_pairs = job->key_pairs + first_block * 32;
    const Vector<_Float16> * value_pairs = job->value_pairs + first_block * 32;
    float * scores = job->scores + static_cast<size_t>(thread) * 2 * kScoreFloats;
    const int first_query = block * 32;
    const int queries =
        arguments.token_counts[segment] - first_query < 32 ? arguments.token_counts[segment] - first_query : 32;

    for (int pair = 0; pair < 32; pair += 2) {
        const int token0 = arguments.first_tokens[segment] + first_query + pair;
        _Float16 * out[2] = {head_columns(arguments.outputs, arguments.row_stride, token0, head),
                             head_columns(arguments.outputs, arguments.row_stride, token0 + 1, head)};
        if (pair >= queries) {
            hvx::vector_at<_Float16>(out[0]) = Vector<_Float16>::zero();
            hvx::vector_at<_Float16>(out[1]) = Vector<_Float16>::zero();
            continue;
        }

        const Vector<_Float16> q0 =
            hvx::load<_Float16>(head_columns(arguments.queries, arguments.row_stride, token0, head));
        const Vector<_Float16> q1 =
            hvx::load<_Float16>(head_columns(arguments.queries, arguments.row_stride, token0 + 1, head));
        float * s[2] = {scores, scores + kScoreFloats};
        score_pair(q0.bits, q1.bits, key_pairs, keys, s[0], s[1]);
        const float sums[2] = {exponentials(s[0], keys), exponentials(s[1], keys)};

        Vector<float> o[2][2] = {{Vector<float>::zero(), Vector<float>::zero()},
                                 {Vector<float>::zero(), Vector<float>::zero()}};
        const auto * e0 = hvx::vectors(s[0]);
        const auto * e1 = hvx::vectors(s[1]);
        for (int group = 0; group < (keys + 63) / 64; group++) {
            const Vector<_Float16> p0 = to_float16(e0[2 * group], e0[2 * group + 1]);
            const Vector<_Float16> p1 = to_float16(e1[2 * group], e1[2 * group + 1]);
            if (!any_lane(reinterpret<uint16_t>(p0 | p1) > Vector<uint16_t>::zero())) continue;

            const Vector<_Float16> * pairs = value_pairs + static_cast<size_t>(group) * 64;
            for (int m = 0; m < 32; m++) {
                const int w0 = extract_word(p0, m);
                const int w1 = extract_word(p1, m);
                if ((w0 | w1) == 0) continue;
                const Vector<_Float16> s0 = reinterpret<_Float16>(Vector<int32_t>::splat(w0));
                const Vector<_Float16> s1 = reinterpret<_Float16>(Vector<int32_t>::splat(w1));
                o[0][0] = dot_product_accumulate(o[0][0], pairs[2 * m], s0);
                o[0][1] = dot_product_accumulate(o[0][1], pairs[2 * m + 1], s0);
                o[1][0] = dot_product_accumulate(o[1][0], pairs[2 * m], s1);
                o[1][1] = dot_product_accumulate(o[1][1], pairs[2 * m + 1], s1);
            }
        }

        for (int q = 0; q < 2; q++) {
            if (pair + q >= queries) {
                hvx::vector_at<_Float16>(out[q]) = Vector<_Float16>::zero();
                continue;
            }

            const Vector<float> inverse = Vector<float>::splat(1.0f / sums[q]);
            hvx::vector_at<_Float16>(out[q]) = to_float16(o[q][0] * inverse, o[q][1] * inverse);
        }
    }
}

struct BoundJob {
    const PreciseAttentionArguments * arguments;
    int blocks;
    int block_starts[3];
    // per thread and side (queries, keys): max |row|^2 at segment * kMaximumHeads + head, 128 bytes apart
    float maxima[runtime::kHvxThreads][2][32];
};

// max |row|^2 of 32 rows for one head. The deal + add reduction leaves the rows in any lane order: only the maximum is
// taken.
inline float block_maximum(const _Float16 * rows, int row_stride, int head) {
    Vector<float> v[32];
#pragma clang loop unroll(full)
    for (int r = 0; r < 32; r++) {
        const Vector<_Float16> row =
            hvx::load<_Float16>(rows + static_cast<size_t>(r) * row_stride + head * kPreciseHeadDimensions);
        v[r] = dot_product_accumulate(Vector<float>::zero(), row, row);
    }

#pragma clang loop unroll(full)
    for (int count = 32; count > 1; count /= 2) {
#pragma clang loop unroll(full)
        for (int i = 0; i < count / 2; i++) {
            const VectorPair<float> dealt = deinterleave(v[2 * i + 1], v[2 * i]);
            v[i] = dealt.low() + dealt.high();
        }
    }

    Vector<float> m = v[0];
    for (int shift = 16; shift >= 1; shift /= 2) m = max(m, rotate_right_bytes(m, 4 * shift));
    alignas(128) float lanes[32];
    store(lanes, m);
    return lanes[0];
}

void block_norms(void * context, int item, int thread) {
    auto * job = static_cast<BoundJob *>(context);
    const PreciseAttentionArguments & arguments = *job->arguments;
    const int segment = item < job->block_starts[1] ? 0 : 1;
    const size_t first_row =
        static_cast<size_t>(arguments.first_tokens[segment]) + 32 * (item - job->block_starts[segment]);

    for (int side = 0; side < 2; side++) {
        const _Float16 * rows = (side == 0 ? arguments.queries : arguments.keys) + first_row * arguments.row_stride;
        for (int head = 0; head < arguments.heads; head++) {
            float & maximum = job->maxima[thread][side][segment * kMaximumHeads + head];
            const float squared = block_maximum(rows, arguments.row_stride, head);
            maximum = squared > maximum ? squared : maximum;
        }
    }
}

runtime::Result<float> attention_score_bound(runtime::Session & session, const PreciseAttentionArguments & arguments) {
    BoundJob job{};
    job.arguments = &arguments;
    for (int segment = 0; segment < arguments.segment_count; segment++) {
        job.block_starts[segment + 1] = job.block_starts[segment] + segment_blocks(arguments, segment);
    }
    job.blocks = job.block_starts[arguments.segment_count];
    const runtime::Status status = runtime::parallel_for(session, runtime::kHvxThreads, job.blocks, block_norms, &job);
    if (!status) return std::unexpected(status.error());

    float bound = 0.0f;
    for (int segment = 0; segment < arguments.segment_count; segment++) {
        const int keys = key_segment(arguments, segment);
        for (int head = 0; head < arguments.heads; head++) {
            float queries = 0.0f;
            float key_rows = 0.0f;
            for (int t = 0; t < runtime::kHvxThreads; t++) {
                const float q = job.maxima[t][0][segment * kMaximumHeads + head];
                const float k = job.maxima[t][1][keys * kMaximumHeads + head];
                queries = q > queries ? q : queries;
                key_rows = k > key_rows ? k : key_rows;
            }
            const float b = __builtin_sqrtf(queries) * __builtin_sqrtf(key_rows);
            bound = b > bound ? b : bound;
        }
    }
    return bound;
}

runtime::Status run_precise_attention(runtime::Session & session, const PreciseAttentionArguments & arguments) {
    if (arguments.heads < 1 || arguments.heads > kMaximumHeads || arguments.segment_count < 1 ||
        arguments.segment_count > 2 || arguments.row_stride % 64 != 0) {
        return std::unexpected(runtime::Error::kBadParameter);
    }
    int end = 0;
    for (int segment = 0; segment < arguments.segment_count; segment++) {
        if (arguments.first_tokens[segment] % 32 != 0 || arguments.token_counts[segment] < 1) {
            return std::unexpected(runtime::Error::kBadParameter);
        }
        const int segment_end = arguments.first_tokens[segment] + segment_blocks(arguments, segment) * 32;
        end = segment_end > end ? segment_end : end;
    }
    if (end > kMaximumTokens) return std::unexpected(runtime::Error::kBadParameter);

    PreciseAttentionWorkspace * workspace = session.state<PreciseAttentionWorkspace>();
    if (workspace == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    if (const runtime::Status allocated = allocate_precise_attention_workspace(*workspace); !allocated) {
        return allocated;
    }

    const int blocks = end / 32;
    const size_t pair_bytes = static_cast<size_t>(arguments.heads) * blocks * 32 * 128;
    if (2 * pair_bytes > session.vtcm_bytes()) return std::unexpected(runtime::Error::kNoMemory);
    auto * key_pairs = reinterpret_cast<Vector<uint32_t> *>(session.vtcm());
    auto * value_pairs = reinterpret_cast<Vector<_Float16> *>(session.vtcm() + pair_bytes);

    PrepareJob prepare{&arguments, blocks, key_pairs, value_pairs};
    const runtime::Status prepared =
        runtime::parallel_for(session, runtime::kHvxThreads, arguments.heads * blocks, prepare_block, &prepare);
    if (!prepared) return prepared;

    QueryJob job{
        .arguments = &arguments,
        .query_blocks = {segment_blocks(arguments, 0), arguments.segment_count > 1 ? segment_blocks(arguments, 1) : 0},
        .blocks = blocks,
        .key_pairs = key_pairs,
        .value_pairs = value_pairs,
        .scores = workspace->scores.get()};
    return runtime::parallel_for(session, runtime::kHvxThreads,
                                 arguments.heads * (job.query_blocks[0] + job.query_blocks[1]), attend_block, &job);
}

// An fp16 tile, activation or output alike: 16 vectors, line i = 32 columns of rows 2i, 2i + 1 interleaved.
constexpr size_t kTileBytes = 2048;
constexpr size_t kTileVectors = kTileBytes / hvx::kVectorBytes;
constexpr int kActivationRing = 4;
constexpr int kFfn1OutputRing = 8;
constexpr int kFfn2InputRing = 4;
constexpr int kFfn2OutputRing = 4;
constexpr int kProjectionRing = 4;
constexpr int kMaximumRowBlocks = kMaximumTokens / 32;
constexpr int kMaximumLayerNormPairs = SuperPointLightGlue::kFfnTiles / 2;
constexpr int kConvertedRowVectors = 16 * kMaximumLayerNormPairs;
constexpr int kRowStatisticVectors = 16;

template <class Shape>
struct PipelineTiles {
    static constexpr int kOutputInput = Shape::kAttentionRowStride / 32;
    static constexpr int kFfn = Shape::kFfnTiles;
    static constexpr int kFfnPairs = kFfn / 2;
    static constexpr int kFfn2Output = 2 * Shape::kTilePairs;  // Shape::kTiles, plus a zero tile if odd
    static constexpr int kProjectionInput = Shape::kTokenRowStride / 32;
    static constexpr int kProjectionOutput = 3 * Shape::kTiles;
    static constexpr int kStreamKTiles[4] = {kOutputInput, kFfn, kFfn, kProjectionInput};
    static constexpr int kStreamColumnBlocks[3] = {Shape::kTiles, kFfn, Shape::kTiles};
    static_assert(kFfn % 2 == 0 && kOutputInput % 2 == 0 && kProjectionInput % 2 == 0);
};

struct BlockPipelineJob {
    runtime::Session * session;
    BlockPipelineArguments arguments;
    int row_blocks;
    uint8_t * weights[4];
    Vector<_Float16> * out_projection_inputs;
    Vector<_Float16> * ffn1_inputs;  // cat(x, m): x's k tiles, then the out projection's output tiles
    Vector<_Float16> * ffn1_outputs;
    Vector<_Float16> * ffn2_inputs;
    Vector<_Float16> * ffn2_outputs;
    Vector<_Float16> * projection_inputs;
    Vector<_Float16> * projection_outputs;
    // per worker, layer_norm_quarter's rows and statistics: in cached DDR, as its VTCM loads would stall behind HMX's
    // tile streams
    Vector<float> * converted_rows;
    Vector<float> * row_statistics;
    runtime::DmaRing * driver_ring;
    // The stages of a row block, in order: gather_row_block, the driver's out projection + FFN1, layer_norm_and_gelu
    // (4 quarters), the driver's FFN2, add_residual, and with weights[3] the driver's projections and
    // write_projection_rows. A stage takes row block rb once the stage before has finished it and the stage reading
    // its output ring entry has finished rb - (that ring's entries).
    alignas(128) std::atomic<int> next_gather;
    alignas(128) std::atomic<int> next_layer_norm;
    alignas(128) std::atomic<int> next_residual;
    alignas(128) std::atomic<int> ffn1_done;
    alignas(128) std::atomic<int> ffn2_done;
    alignas(128) std::atomic<int> projections_done;
    alignas(128) std::atomic<int> next_projection_rows;
    alignas(128) std::atomic<int> gathered[kMaximumRowBlocks];
    alignas(128) std::atomic<int> layer_norm_quarters[kMaximumRowBlocks];
    alignas(128) std::atomic<int> residual_done[kMaximumRowBlocks];
    alignas(128) std::atomic<int> projection_rows_done[kMaximumRowBlocks];
};

inline Vector<_Float16> * ring_entry(Vector<_Float16> * ring, int entries, int entry_tiles, int row_block) {
    return ring + static_cast<size_t>(row_block % entries) * entry_tiles * kTileVectors;
}

inline Vector<_Float16> * row_block_start(_Float16 * rows, int row_stride, int row_block) {
    return hvx::vectors(rows + static_cast<size_t>(row_block) * 32 * row_stride);
}

// With an odd Shape::kTiles, the x rows' zero columns also fill ffn1_inputs' tile Shape::kTiles, which the out
// projection then overwrites.
template <class Shape>
void gather_row_block(const BlockPipelineJob * job, int row_block) {
    using Tiles = PipelineTiles<Shape>;
    constexpr int kOutputVectors = Shape::kAttentionRowStride / 64;
    constexpr int kTokenVectors = Shape::kTokenRowStride / 64;
    static_assert(kOutputVectors == kTokenVectors, "one loop over both row kinds");
    Vector<_Float16> * out_projection_inputs =
        ring_entry(job->out_projection_inputs, kActivationRing, Tiles::kOutputInput, row_block);
    Vector<_Float16> * ffn1_inputs = ring_entry(job->ffn1_inputs, kActivationRing, Tiles::kFfn, row_block);
    const auto * o = row_block_start(job->arguments.attention_output, Shape::kAttentionRowStride, row_block);
    const auto * x = row_block_start(job->arguments.token_rows, Shape::kTokenRowStride, row_block);

    for (int i = 0; i < 16; i++) {
        for (int c = 0; c < kOutputVectors; c++) {
            const VectorPair<_Float16> o_line =
                interleave(o[(2 * i + 1) * kOutputVectors + c], o[2 * i * kOutputVectors + c]);
            const VectorPair<_Float16> x_line =
                interleave(x[(2 * i + 1) * kTokenVectors + c], x[2 * i * kTokenVectors + c]);
            out_projection_inputs[2 * c * 16 + i] = o_line.low();
            out_projection_inputs[(2 * c + 1) * 16 + i] = o_line.high();
            ffn1_inputs[2 * c * 16 + i] = x_line.low();
            ffn1_inputs[(2 * c + 1) * 16 + i] = x_line.high();
        }
    }
}

// Exact GELU as vector_math::gelu_erf, of 4 f32 vectors, in two halves for layer_norm_quarter's software pipeline:
//   first   z = |x| / sqrt 2, r = 1 / (1 + 0.3275911 z)
//   second  Q / 2 = p e / 2 (p: Abramowitz-Stegun 7.1.26 in r, e = 2^(-z^2 log2 e) as vector_math::exp2_softmax),
//           x (x < 0 ? Q / 2 : 1 - Q / 2)
[[gnu::always_inline]] inline void gelu_erf_first_half(const Vector<float> x[4], Vector<float> z[4],
                                                       Vector<float> r[4]) {
#pragma clang loop unroll(full)
    for (int k = 0; k < 4; k++) z[k] = absolute(x[k]) * 0.70710678118654752f;
    Vector<float> denominator[4];
#pragma clang loop unroll(full)
    for (int k = 0; k < 4; k++) denominator[k] = z[k] * 0.3275911f + 1.0f;
    vector_math::reciprocal_batch<4>(denominator, r);
}

[[gnu::always_inline]] inline void gelu_erf_second_half(Vector<float> x[4], const Vector<float> z[4],
                                                        const Vector<float> r[4]) {
    const Vector<float> sign_bit = Vector<float>::splat_bits(0x80000000u);
    Vector<float> p[4];
    Vector<float> y[4];
    Vector<int32_t> n[4];
    Vector<float> power[4];

    // p / 2: the polynomial with halved coefficients, every step's value halved exactly (scaling by 2^-1 commutes
    // with the rounding, and no step comes near the subnormal range), so (p / 2) e = fl(fl(p e) / 2).
#pragma clang loop unroll(full)
    for (int k = 0; k < 4; k++) p[k] = r[k] * (0.5f * 1.061405429f) + 0.5f * -1.453152027f;
    constexpr float kHalvedCoefficients[3] = {0.5f * 1.421413741f, 0.5f * -0.284496736f, 0.5f * 0.254829592f};
#pragma clang loop unroll(full)
    for (int c = 0; c < 3; c++) {
#pragma clang loop unroll(full)
        for (int k = 0; k < 4; k++) p[k] = p[k] * r[k] + kHalvedCoefficients[c];
    }
#pragma clang loop unroll(full)
    for (int k = 0; k < 4; k++) p[k] = p[k] * r[k];

    // e = 2^y = 2^n p(y - n), y = -z^2 log2 e clamped at -26, n = trunc(y): 2^n added to the exponent bits
#pragma clang loop unroll(full)
    for (int k = 0; k < 4; k++) y[k] = max((z[k] * z[k] * 1.4426950408889634f) ^ sign_bit, -26.0f);
#pragma clang loop unroll(full)
    for (int k = 0; k < 4; k++) n[k] = truncate_to_int32(y[k]);
#pragma clang loop unroll(full)
    for (int k = 0; k < 4; k++) y[k] = y[k] - to_float(n[k]);
#pragma clang loop unroll(full)
    for (int k = 0; k < 4; k++) power[k] = y[k] * 0.039662234485149384f + 0.23124286532402039f;
#pragma clang loop unroll(full)
    for (int k = 0; k < 4; k++) power[k] = power[k] * y[k] + 0.6916316747665405f;
#pragma clang loop unroll(full)
    for (int k = 0; k < 4; k++) power[k] = power[k] * y[k] + 1.000005841255188f;
#pragma clang loop unroll(full)
    for (int k = 0; k < 4; k++) power[k] = reinterpret<float>(reinterpret<int32_t>(power[k]) + (n[k] << 23));

#pragma clang loop unroll(full)
    for (int k = 0; k < 4; k++) {
        const Vector<float> half_q = p[k] * power[k];
        x[k] = x[k] * select(Vector<int32_t>::zero() > reinterpret<int32_t>(x[k]), half_q, 1.0f + (half_q ^ sign_bit));
    }
}

// LayerNorm(2D, eps 1e-5) and exact GELU of rows 8q .. 8q + 7 of a row block: lines 4q .. 4q + 3 of FFN1's output
// tiles into the same lines of FFN2's activation tiles. The deinterleaving f16 -> f32 conversion splits a line into
// its two rows, the interleaving f32 -> f16 conversion joins them back. The lines from valid_lines on hold padding
// rows only and are written as zeros: nothing of them reaches a real row, and zero keeps them finite.
template <int Pairs>
[[gnu::always_inline]] inline void layer_norm_quarter(const Vector<_Float16> * __restrict outputs,
                                                      Vector<_Float16> * __restrict activations,
                                                      Vector<float> * __restrict statistics,
                                                      Vector<float> * __restrict rows,
                                                      const Vector<float> * __restrict gamma_beta, int valid_lines) {
    const Vector<float> zero = Vector<float>::zero();
    Vector<float> sums[8];
    Vector<float> means[8];
#pragma clang loop unroll(full)
    for (int r = 0; r < 8; r++) sums[r] = zero;

    for (int k = 0; k < Pairs; k++) {
#pragma clang loop unroll(full)
        for (int line = 0; line < 4; line++) {
            const VectorPair<float> first = to_float32_deinterleaved(outputs[32 * k + line]);
            const VectorPair<float> second = to_float32_deinterleaved(outputs[32 * k + 16 + line]);
            Vector<float> * x = rows + 4 * (Pairs * line + k);
            x[0].bits = first.low().bits;
            x[1].bits = first.high().bits;
            x[2].bits = second.low().bits;
            x[3].bits = second.high().bits;
            sums[2 * line] = sums[2 * line] + (first.low() + second.low());
            sums[2 * line + 1] = sums[2 * line + 1] + (first.high() + second.high());
        }
    }
    vector_math::sum_to_all_lanes_batch<8>(sums);
#pragma clang loop unroll(full)
    for (int r = 0; r < 8; r++) {
        means[r] = sums[r] * (1.0f / (64 * Pairs));
        sums[r] = zero;
    }

    for (int k = 0; k < Pairs; k++) {
#pragma clang loop unroll(full)
        for (int line = 0; line < 4; line++) {
            const Vector<float> * x = rows + 4 * (Pairs * line + k);
            const Vector<float> d0 = x[0] - means[2 * line];
            const Vector<float> d1 = x[2] - means[2 * line];
            const Vector<float> e0 = x[1] - means[2 * line + 1];
            const Vector<float> e1 = x[3] - means[2 * line + 1];
            sums[2 * line] = sums[2 * line] + d0 * d0 + d1 * d1;
            sums[2 * line + 1] = sums[2 * line + 1] + e0 * e0 + e1 * e1;
        }
    }
    vector_math::sum_to_all_lanes_batch<8>(sums);
#pragma clang loop unroll(full)
    for (int r = 0; r < 8; r++) sums[r] = sums[r] * (1.0f / (64 * Pairs)) + 1e-5f;
    vector_math::precise_reciprocal_square_root_batch<8>(sums);

#pragma clang loop unroll(full)
    for (int line = 0; line < 4; line++) {
        statistics[4 * line] = means[2 * line];
        statistics[4 * line + 1] = means[2 * line + 1];
        statistics[4 * line + 2] = sums[2 * line];
        statistics[4 * line + 3] = sums[2 * line + 1];
    }

    const Vector<float> * __restrict beta = gamma_beta + 2 * Pairs;
    for (int line = valid_lines; line < 4; line++) {
        for (int tile = 0; tile < 2 * Pairs; tile++) activations[16 * tile + line] = reinterpret<_Float16>(zero);
    }

    const int groups = Pairs * valid_lines;
    if (!groups) return;

    Vector<float> previous[4];
    Vector<float> previous_z[4];
    Vector<float> previous_r[4];
    {
        const Vector<float> * x = rows;
        previous[0] = (x[0] - statistics[0]) * statistics[2] * gamma_beta[0] + beta[0];
        previous[1] = (x[1] - statistics[1]) * statistics[3] * gamma_beta[0] + beta[0];
        previous[2] = (x[2] - statistics[0]) * statistics[2] * gamma_beta[1] + beta[1];
        previous[3] = (x[3] - statistics[1]) * statistics[3] * gamma_beta[1] + beta[1];
    }
    gelu_erf_first_half(previous, previous_z, previous_r);

    for (int group = 1; group < groups; group++) {
        Vector<float> current[4];
        Vector<float> current_z[4];
        Vector<float> current_r[4];
        {
            const int line = group / Pairs;
            const int tile = 2 * (group % Pairs);
            const Vector<float> mean0 = statistics[4 * line];
            const Vector<float> mean1 = statistics[4 * line + 1];
            const Vector<float> inverse0 = statistics[4 * line + 2];
            const Vector<float> inverse1 = statistics[4 * line + 3];
            const Vector<float> * x = rows + 4 * group;
            current[0] = (x[0] - mean0) * inverse0 * gamma_beta[tile] + beta[tile];
            current[1] = (x[1] - mean1) * inverse1 * gamma_beta[tile] + beta[tile];
            current[2] = (x[2] - mean0) * inverse0 * gamma_beta[tile + 1] + beta[tile + 1];
            current[3] = (x[3] - mean1) * inverse1 * gamma_beta[tile + 1] + beta[tile + 1];
        }

        gelu_erf_first_half(current, current_z, current_r);
        gelu_erf_second_half(previous, previous_z, previous_r);

        {
            const int line = (group - 1) / Pairs;
            const int tile = 2 * ((group - 1) % Pairs);
            activations[16 * tile + line] = to_float16_interleaved(previous[0], previous[1]);
            activations[16 * tile + 16 + line] = to_float16_interleaved(previous[2], previous[3]);
        }

#pragma clang loop unroll(full)
        for (int k = 0; k < 4; k++) {
            previous[k] = current[k];
            previous_z[k] = current_z[k];
            previous_r[k] = current_r[k];
        }
    }

    gelu_erf_second_half(previous, previous_z, previous_r);
    const int line = (groups - 1) / Pairs;
    const int tile = 2 * ((groups - 1) % Pairs);
    activations[16 * tile + line] = to_float16_interleaved(previous[0], previous[1]);
    activations[16 * tile + 16 + line] = to_float16_interleaved(previous[2], previous[3]);
}

template <class Shape>
void layer_norm_and_gelu(const BlockPipelineJob * job, int row_block, int quarter, int thread) {
    using Tiles = PipelineTiles<Shape>;

    // n0p is a multiple of 32, so a quarter's padding rows are its last ones; a line with a real row is valid.
    int valid_lines = 0;
    while (valid_lines < 4) {
        const int row = 32 * row_block + 8 * quarter + 2 * valid_lines;
        if ((row >= job->arguments.first_image_tokens && row < job->arguments.second_image_first_row) ||
            row >= job->arguments.second_image_first_row + job->arguments.second_image_tokens) {
            break;
        }
        valid_lines++;
    }

    layer_norm_quarter<Tiles::kFfnPairs>(
        ring_entry(job->ffn1_outputs, kFfn1OutputRing, Tiles::kFfn, row_block) + 4 * quarter,
        ring_entry(job->ffn2_inputs, kFfn2InputRing, Tiles::kFfn, row_block) + 4 * quarter,
        job->row_statistics + static_cast<size_t>(thread) * kRowStatisticVectors,
        job->converted_rows + static_cast<size_t>(thread) * kConvertedRowVectors,
        hvx::vectors(job->arguments.layer_norm), valid_lines);
}

template <class Shape>
void add_residual(const BlockPipelineJob * job, int row_block) {
    using Tiles = PipelineTiles<Shape>;
    constexpr int kTokenVectors = Shape::kTokenRowStride / 64;
    const Vector<_Float16> * outputs = ring_entry(job->ffn2_outputs, kFfn2OutputRing, Tiles::kFfn2Output, row_block);
    auto * x = row_block_start(job->arguments.token_rows, Shape::kTokenRowStride, row_block);
    Vector<_Float16> * projection_inputs =
        job->arguments.weights[3] != nullptr
            ? ring_entry(job->projection_inputs, kProjectionRing, Tiles::kProjectionInput, row_block)
            : nullptr;

    for (int p = 0; p < Shape::kTilePairs; p++) {
        for (int i = 0; i < 16; i++) {
            const VectorPair<_Float16> rows = deinterleave(outputs[(2 * p + 1) * 16 + i], outputs[2 * p * 16 + i]);
            Vector<_Float16> * x0 = x + static_cast<size_t>(2 * i) * kTokenVectors + p;
            Vector<_Float16> * x1 = x0 + kTokenVectors;
            const Vector<_Float16> y0 = rows.low() + *x0;
            const Vector<_Float16> y1 = rows.high() + *x1;
            *x0 = y0;
            *x1 = y1;
            if (projection_inputs) {
                const VectorPair<_Float16> line = interleave(y1, y0);
                projection_inputs[2 * p * 16 + i] = line.low();
                projection_inputs[(2 * p + 1) * 16 + i] = line.high();
            }
        }
    }
}

template <class Shape>
void write_projection_rows(const BlockPipelineJob * job, int row_block) {
    using Tiles = PipelineTiles<Shape>;
    constexpr int kRowVectors = Shape::kAttentionRowStride / 64;
    const Vector<_Float16> * outputs =
        ring_entry(job->projection_outputs, kProjectionRing, Tiles::kProjectionOutput, row_block);
    const int groups = job->arguments.projection_blocks / Shape::kTiles;

    for (int g = 0; g < groups; g++) {
        const Vector<_Float16> * group_outputs = outputs + static_cast<size_t>(g) * Shape::kTiles * 16;
        auto * group_rows = row_block_start(job->arguments.projection_rows[g], Shape::kAttentionRowStride, row_block);

        for (int p = 0; p < Shape::kTiles / 2; p++) {
            Vector<_Float16> * out = group_rows + p;
            for (int i = 0; i < 16; i++) {
                const VectorPair<_Float16> rows =
                    deinterleave(group_outputs[(2 * p + 1) * 16 + i], group_outputs[2 * p * 16 + i]);
                out[(2 * i) * kRowVectors] = rows.low();
                out[(2 * i + 1) * kRowVectors] = rows.high();
            }
        }

        if constexpr (Shape::kTiles % 2 != 0) {  // an odd last tile: masked stores keep the zero padding
            const Mask first_half = Mask::first_bytes(64);
            Vector<_Float16> * out = group_rows + Shape::kTiles / 2;
            for (int i = 0; i < 16; i++) {
                const Vector<_Float16> line = group_outputs[(Shape::kTiles - 1) * 16 + i];
                const VectorPair<_Float16> rows = deinterleave(line, line);
                store_masked(out + (2 * i) * kRowVectors, first_half, rows.low());
                store_masked(out + (2 * i + 1) * kRowVectors, first_half, rows.high());
            }
        }
    }
}

inline void multiply_tile(const Vector<_Float16> * activations, const uint8_t * weights, int k_tiles,
                          Vector<_Float16> * output) {
    hmx::clear_fp16_accumulator();
    hmx::multiply_fp16(activations, weights, static_cast<uint32_t>(k_tiles) * kTileBytes - 1);
    hmx::load_output_block(weights + static_cast<size_t>(k_tiles) * kTileBytes);
    hmx::store_fp16(output);
}

// The fifth thread: no HVX instruction (noinline, scalar code only).
template <class Shape>
[[gnu::noinline]] void block_pipeline_driver_thread(void * job_pointer) {
    using Tiles = PipelineTiles<Shape>;
    auto * job = static_cast<BlockPipelineJob *>(job_pointer);
    job->driver_ring->initialize_with_scalar_stores(runtime::DmaBypass::kBoth);
    hmx::lock(*job->session);
    const int streams = job->arguments.weights[3] != nullptr ? 4 : 3;

    for (int s = 0; s < streams; s++) {
        const uint32_t bytes =
            static_cast<uint32_t>(s < 3 ? Tiles::kStreamColumnBlocks[s] : job->arguments.projection_blocks) *
            static_cast<uint32_t>(Tiles::kStreamKTiles[s] + 1) * kTileBytes;
        job->driver_ring->push(job->weights[s], job->arguments.weights[s], bytes, 1, bytes, bytes);
    }

    const size_t output_block_bytes = static_cast<size_t>(Tiles::kOutputInput + 1) * kTileBytes;
    const size_t ffn_block_bytes = static_cast<size_t>(Tiles::kFfn + 1) * kTileBytes;
    const size_t projection_block_bytes = static_cast<size_t>(Tiles::kProjectionInput + 1) * kTileBytes;
    int next_ffn1 = 0;
    int next_ffn2 = 0;
    int next_projection = streams == 4 ? 0 : job->row_blocks;
    int popped = 0;  // weight DMAs waited for: stream s before its first GEMM

    while (next_ffn2 < job->row_blocks || next_projection < job->row_blocks) {
        if (next_ffn2 < next_ffn1 && job->layer_norm_quarters[next_ffn2].load() == 4 &&
            (next_ffn2 < kFfn2OutputRing || job->residual_done[next_ffn2 - kFfn2OutputRing].load())) {
            for (; popped < 3; popped++) job->driver_ring->pop();
            const Vector<_Float16> * inputs = ring_entry(job->ffn2_inputs, kFfn2InputRing, Tiles::kFfn, next_ffn2);
            Vector<_Float16> * outputs = ring_entry(job->ffn2_outputs, kFfn2OutputRing, Tiles::kFfn2Output, next_ffn2);
            for (int cb = 0; cb < Shape::kTiles; cb++) {
                multiply_tile(inputs, job->weights[2] + cb * ffn_block_bytes, Tiles::kFfn,
                              outputs + static_cast<size_t>(cb) * kTileVectors);
            }
            job->ffn2_done.store(++next_ffn2);
            continue;
        }

        if (next_projection < next_ffn2 && job->residual_done[next_projection].load() &&
            (next_projection < kProjectionRing ||
             job->projection_rows_done[next_projection - kProjectionRing].load())) {
            for (; popped < 4; popped++) job->driver_ring->pop();
            const Vector<_Float16> * inputs =
                ring_entry(job->projection_inputs, kProjectionRing, Tiles::kProjectionInput, next_projection);
            Vector<_Float16> * outputs =
                ring_entry(job->projection_outputs, kProjectionRing, Tiles::kProjectionOutput, next_projection);
            for (int cb = 0; cb < job->arguments.projection_blocks; cb++) {
                multiply_tile(inputs, job->weights[3] + cb * projection_block_bytes, Tiles::kProjectionInput,
                              outputs + static_cast<size_t>(cb) * kTileVectors);
            }
            job->projections_done.store(++next_projection);
            continue;
        }

        if (next_ffn1 < job->row_blocks && job->gathered[next_ffn1].load() &&
            (next_ffn1 < kFfn1OutputRing || job->layer_norm_quarters[next_ffn1 - kFfn1OutputRing].load() == 4)) {
            for (; popped < 2; popped++) job->driver_ring->pop();
            const Vector<_Float16> * out_projection_inputs =
                ring_entry(job->out_projection_inputs, kActivationRing, Tiles::kOutputInput, next_ffn1);
            Vector<_Float16> * ffn1_inputs = ring_entry(job->ffn1_inputs, kActivationRing, Tiles::kFfn, next_ffn1);
            Vector<_Float16> * outputs = ring_entry(job->ffn1_outputs, kFfn1OutputRing, Tiles::kFfn, next_ffn1);
            for (int cb = 0; cb < Shape::kTiles; cb++) {
                multiply_tile(out_projection_inputs, job->weights[0] + cb * output_block_bytes, Tiles::kOutputInput,
                              ffn1_inputs + static_cast<size_t>(Shape::kTiles + cb) * kTileVectors);
            }
            for (int cb = 0; cb < Tiles::kFfn; cb++) {
                multiply_tile(ffn1_inputs, job->weights[1] + cb * ffn_block_bytes, Tiles::kFfn,
                              outputs + static_cast<size_t>(cb) * kTileVectors);
            }
            job->ffn1_done.store(++next_ffn1);
            continue;
        }
    }

    for (; popped < streams; popped++) job->driver_ring->pop();
    hmx::unlock(*job->session);
}

template <class Shape>
void block_pipeline_worker_thread(void * job_pointer, int thread) {
    auto * job = static_cast<BlockPipelineJob *>(job_pointer);
    const int row_blocks = job->row_blocks;
    const int projections = job->arguments.weights[3] != nullptr;

    while (true) {
        if (projections) {
            int row_block = job->next_projection_rows.load();
            if (row_block >= row_blocks) break;
            if (job->projections_done.load() > row_block) {
                if (job->next_projection_rows.compare_exchange_strong(row_block, row_block + 1)) {
                    write_projection_rows<Shape>(job, row_block);
                    job->projection_rows_done[row_block].store(1);
                }
                continue;
            }
        }

        int residual = job->next_residual.load();
        if (!projections && residual >= row_blocks) break;
        if (residual < row_blocks && job->ffn2_done.load() > residual &&
            (!projections || job->projections_done.load() > residual - kProjectionRing)) {
            if (job->next_residual.compare_exchange_strong(residual, residual + 1)) {
                add_residual<Shape>(job, residual);
                job->residual_done[residual].store(1);
            }
            continue;
        }

        int quarter = job->next_layer_norm.load();
        if (quarter < 4 * row_blocks && job->ffn1_done.load() > quarter / 4 &&
            job->ffn2_done.load() > quarter / 4 - kFfn2InputRing) {
            if (job->next_layer_norm.compare_exchange_strong(quarter, quarter + 1)) {
                layer_norm_and_gelu<Shape>(job, quarter / 4, quarter % 4, thread);
                job->layer_norm_quarters[quarter / 4].fetch_add(1);
            }
            continue;
        }

        int gather = job->next_gather.load();
        if (gather < row_blocks && job->ffn1_done.load() > gather - kActivationRing) {
            if (job->next_gather.compare_exchange_strong(gather, gather + 1)) {
                gather_row_block<Shape>(job, gather);
                job->gathered[gather].store(1);
            }
            continue;
        }
    }
}

struct BlockPipelineWorkspace {
    runtime::HeapObject<BlockPipelineJob> job;
    runtime::HeapObject<runtime::DmaRing> driver_ring;
    runtime::HeapArray<Vector<float>> layer_norm_data;
};

template <class Shape>
runtime::Status run_block_pipeline(runtime::Session & session, const BlockPipelineArguments & arguments) {
    using Tiles = PipelineTiles<Shape>;
    if (arguments.rows % 32 != 0 || arguments.rows > kMaximumTokens) {
        return std::unexpected(runtime::Error::kBadParameter);
    }
    if (arguments.weights[3] != nullptr && arguments.projection_blocks != 2 * Shape::kTiles &&
        arguments.projection_blocks != 3 * Shape::kTiles) {
        return std::unexpected(runtime::Error::kBadParameter);
    }

    BlockPipelineWorkspace * kept = session.state<BlockPipelineWorkspace>();
    if (kept == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    BlockPipelineWorkspace & workspace = *kept;
    if (!workspace.job) {
        workspace.job = runtime::allocate_object<BlockPipelineJob>(128);
        workspace.driver_ring = runtime::allocate_object<runtime::DmaRing>(64);
        workspace.layer_norm_data = runtime::allocate_array<Vector<float>>(runtime::kHvxThreads * kConvertedRowVectors +
                                                                           runtime::kHvxThreads * kRowStatisticVectors);
    }
    if (!workspace.job || !workspace.driver_ring || !workspace.layer_norm_data) {
        return std::unexpected(runtime::Error::kNoMemory);
    }

    static_assert(std::is_trivially_destructible_v<BlockPipelineJob>);  // freed without a destructor call
    BlockPipelineJob * job = new (workspace.job.get()) BlockPipelineJob();
    job->session = &session;
    job->arguments = arguments;
    job->row_blocks = arguments.rows / 32;
    job->driver_ring = workspace.driver_ring.get();

    uint8_t * vtcm = session.vtcm();
    for (int s = 0; s < 3; s++) {
        job->weights[s] = vtcm;
        vtcm += static_cast<size_t>(Tiles::kStreamColumnBlocks[s]) * (Tiles::kStreamKTiles[s] + 1) * kTileBytes;
    }
    job->weights[3] = vtcm;
    vtcm += arguments.weights[3] != nullptr
                ? static_cast<size_t>(arguments.projection_blocks) * (Tiles::kProjectionInput + 1) * kTileBytes
                : 0;

    auto * tiles = reinterpret_cast<Vector<_Float16> *>(vtcm);
    job->out_projection_inputs = tiles;
    tiles += static_cast<size_t>(kActivationRing) * Tiles::kOutputInput * kTileVectors;
    job->ffn1_inputs = tiles;
    tiles += static_cast<size_t>(kActivationRing) * Tiles::kFfn * kTileVectors;
    job->ffn1_outputs = tiles;
    tiles += static_cast<size_t>(kFfn1OutputRing) * Tiles::kFfn * kTileVectors;
    job->ffn2_inputs = tiles;
    tiles += static_cast<size_t>(kFfn2InputRing) * Tiles::kFfn * kTileVectors;
    job->ffn2_outputs = tiles;
    tiles += static_cast<size_t>(kFfn2OutputRing) * Tiles::kFfn2Output * kTileVectors;
    job->projection_inputs = tiles;
    tiles += arguments.weights[3] != nullptr
                 ? static_cast<size_t>(kProjectionRing) * Tiles::kProjectionInput * kTileVectors
                 : 0;
    job->projection_outputs = tiles;
    tiles += arguments.weights[3] != nullptr
                 ? static_cast<size_t>(kProjectionRing) * Tiles::kProjectionOutput * kTileVectors
                 : 0;
    const size_t ring_bytes = static_cast<size_t>(tiles - job->out_projection_inputs) * hvx::kVectorBytes;
    if (static_cast<size_t>(vtcm - session.vtcm()) + ring_bytes > session.vtcm_bytes()) {
        return std::unexpected(runtime::Error::kNoMemory);
    }

    if constexpr (Tiles::kFfn2Output > Shape::kTiles) {
        for (int entry = 0; entry < kFfn2OutputRing; entry++) {
            Vector<_Float16> * tile =
                job->ffn2_outputs + (static_cast<size_t>(entry) * Tiles::kFfn2Output + Shape::kTiles) * kTileVectors;
            for (int line = 0; line < 16; line++) tile[line] = Vector<_Float16>::zero();
        }
    }

    job->converted_rows = workspace.layer_norm_data.get();
    job->row_statistics = job->converted_rows + runtime::kHvxThreads * kConvertedRowVectors;
    return runtime::run_parallel_with_driver(session, runtime::kHvxThreads, block_pipeline_worker_thread<Shape>,
                                             block_pipeline_driver_thread<Shape>, job);
}

// The finite fp16 values in increasing order, k in [-0x7BFF, 0x7BFF].
float ordered_f16(int k) {
    const auto bits = static_cast<uint16_t>(k >= 0 ? k : (0x8000 | -k));
    return static_cast<float>(std::bit_cast<_Float16>(bits));
}

// z < -88: this expf returns a finite wrong value instead of overflowing.
float sigmoid_f16(float z) {
    return z < -88.0f ? 0.0f : static_cast<float>(static_cast<_Float16>(1.0f / (1.0f + expf(-z))));
}

template <class Shape>
void evaluate_heads(void * context, int item, int) {
    constexpr int kVectors = Shape::kTokenRowStride / 64;
    constexpr int kMatchability = matchability_parameters<Shape>();
    const auto * job = static_cast<const HeadJob *>(context);
    const auto * confidence_weights = hvx::vectors(job->parameters);
    const auto * matchability_weights = hvx::vectors(job->parameters + kMatchability);
    Vector<float> confidence_sums = Vector<float>::zero();
    Vector<float> matchability_sums = Vector<float>::zero();

    for (int first_row = 32 * item; first_row < 32 * item + 32; first_row += 4) {
        Vector<float> confidence[4];
        Vector<float> matchability[4];
#pragma clang loop unroll(full)
        for (int q = 0; q < 4; q++) {
            confidence[q] = Vector<float>::zero();
            matchability[q] = Vector<float>::zero();
        }

#pragma clang loop unroll(full)
        for (int k = 0; k < kVectors; k++) {
#pragma clang loop unroll(full)
            for (int q = 0; q < 4; q++) {
                const VectorPair<float> x = to_float32(
                    hvx::vectors(job->token_rows + static_cast<size_t>(first_row + q) * Shape::kTokenRowStride)[k]);
                confidence[q] =
                    confidence[q] + (x.low() * confidence_weights[2 * k] + x.high() * confidence_weights[2 * k + 1]);
                matchability[q] = matchability[q] +
                                  (x.low() * matchability_weights[2 * k] + x.high() * matchability_weights[2 * k + 1]);
            }
        }

        vector_math::sum_to_all_lanes_batch<4>(confidence);
        vector_math::sum_to_all_lanes_batch<4>(matchability);
#pragma clang loop unroll(full)
        for (int q = 0; q < 4; q++) {  // after the 32 rows, lane i: row 32 item + i
            confidence_sums = slide_down(confidence[q], confidence_sums, 4);
            matchability_sums = slide_down(matchability[q], matchability_sums, 4);
        }
    }

    if (job->confidence) {
        confidence_sums = confidence_sums + job->parameters[Shape::kTokenRowStride];
        store(job->confidence_logits + 32 * item,
              to_float32_deinterleaved(to_float16_interleaved(confidence_sums, confidence_sums)).low());
    }
    if (job->matchability) {
        matchability_sums = matchability_sums + job->parameters[kMatchability + Shape::kTokenRowStride];
        store(job->matchability_logits + 32 * item,
              to_float32_deinterleaved(to_float16_interleaved(matchability_sums, matchability_sums)).low());
    }
}

// sigmoid_f16 is monotone over the fp16 values, so for every finite fp16 z:
//   sigmoid(z) >= t <=> z >= sigmoid_threshold_logit(t, 0)
//   sigmoid(z) > t  <=> z >= sigmoid_threshold_logit(t, 1)
[[gnu::noinline]] float sigmoid_threshold_logit(float threshold, int strict) {
    int low = -0x7BFF;
    int high = 0x7BFF + 1;

    while (low < high) {
        const int middle = low + (high - low) / 2;
        const float sigmoid = sigmoid_f16(ordered_f16(middle));
        if (strict ? sigmoid > threshold : sigmoid >= threshold) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    return low > 0x7BFF ? INFINITY : ordered_f16(low);
}

float confidence_threshold(int layer, int layers) {
    return static_cast<float>(static_cast<_Float16>(static_cast<float>(0.8 + 0.1 * exp(-4.0 * layer / layers))));
}

int keep_tokens(int * table_rows, int * keypoint_indices, int * source_rows, const float * confidence_logits,
                const float * matchability_logits, float confidence_limit, float matchability_limit, int confidence,
                int first_row, int count, int first_new_row) {
    int kept = 0;
    for (int i = 0; i < count; i++) {
        const int row = first_row + i;
        if (!(matchability_logits[row] >= matchability_limit ||
              (confidence && confidence_logits[row] < confidence_limit))) {
            continue;
        }

        source_rows[first_new_row + kept] = row;
        table_rows[first_new_row + kept] = table_rows[row];
        keypoint_indices[kept++] = keypoint_indices[i];
    }
    return kept;
}

template <class Shape>
void move_rows(void * context, int item, int) {
    const auto * job = static_cast<const MoveJob *>(context);
    if (job->arrays[item] == nullptr) return;

    const size_t stride = item != 0 ? Shape::kAttentionRowStride : Shape::kTokenRowStride;
    const int vectors = static_cast<int>(stride / 64);

    for (int row = 0; row < job->rows; row++) {
        const int source = job->source_rows[row];
        if (source == row) continue;

        auto * destination = hvx::vectors(job->arrays[item] + static_cast<size_t>(row) * stride);
        if (source < 0) {
            for (int k = 0; k < vectors; k++) destination[k] = Vector<_Float16>::zero();
        } else {
            const auto * x = hvx::vectors(job->arrays[item] + static_cast<size_t>(source) * stride);
            for (int k = 0; k < vectors; k++) destination[k] = x[k];
        }
    }
}

constexpr uint32_t kMinusInfinityBits = 0xff800000u;
constexpr float kLog2e = 1.4426950408889634f;

float log_sigmoid(float z) {
    return z >= 0 ? -log1pf(expf(-z)) : z - log1pf(expf(z));
}

inline float horizontal_max_f32(Vector<float> v) {
    alignas(128) float lanes[32];
    store(lanes, vector_math::max_to_all_lanes(v));
    return lanes[0];
}

inline const Vector<float> * similarity_row(const AssignmentJob * job, int row) {
    return hvx::vectors(job->similarity + static_cast<size_t>(row) * job->stride);
}

[[gnu::always_inline]] inline void row_argmax_step(const Vector<float> * const * rows,
                                                   const Vector<float> * column_terms, Vector<int32_t> lane,
                                                   Mask last_lanes, Vector<float> minus_infinity, int v, bool last,
                                                   Vector<float> * best, Vector<int32_t> * argument) {
    const Vector<int32_t> index = lane + 32 * v;
#pragma clang loop unroll(full)
    for (int q = 0; q < 2; q++) {
        Vector<float> x = rows[q][v] + rows[q][v] + column_terms[v];
        if (last) x = select(last_lanes, x, minus_infinity);
        const Mask greater = x > best[q];
        best[q] = select(greater, x, best[q]);
        argument[q] = select(greater, index, argument[q]);
    }
}

template <class Shape>
void compute_matchability(void * context, int item, int) {
    constexpr int kVectors = Shape::kTokenRowStride / 64;
    const auto * job = static_cast<const MatchabilityJob *>(context);
    const auto * weights = hvx::vectors(job->parameters);
    Vector<float> z = Vector<float>::zero();

    for (int first_row = 32 * item; first_row < 32 * item + 32; first_row += 4) {
        Vector<float> sums[4];
#pragma clang loop unroll(full)
        for (int q = 0; q < 4; q++) sums[q] = Vector<float>::zero();

#pragma clang loop unroll(full)
        for (int k = 0; k < kVectors; k++) {
#pragma clang loop unroll(full)
            for (int q = 0; q < 4; q++) {
                const VectorPair<float> x = to_float32(
                    hvx::vectors(job->token_rows + static_cast<size_t>(first_row + q) * Shape::kTokenRowStride)[k]);
                sums[q] = sums[q] + (x.low() * weights[2 * k] + x.high() * weights[2 * k + 1]);
            }
        }

        vector_math::sum_to_all_lanes_batch<4>(sums);
#pragma clang loop unroll(full)
        for (int q = 0; q < 4; q++) z = slide_down(sums[q], z, 4);  // after the 32 rows, lane i: row 32 item + i
    }

    alignas(128) float logits[32];
    store(logits, z + job->parameters[Shape::kTokenRowStride]);
    for (int i = 0; i < 32; i++) {
        const int row = 32 * item + i;
        const bool token =
            row < job->first_image_tokens ||
            (row >= job->second_image_first_row && row < job->second_image_first_row + job->second_image_tokens);
        job->log_sigmoids[row] = token ? log_sigmoid(logits[i]) : 0.0f;
    }
}

// Tile t, line p, word c = dimension pair 16 t + p of key c: a 32 x 32 word transpose per vector of the rows (32
// dimension pairs).
template <class Shape>
void write_similarity_weights(void * context, int item, int) {
    constexpr int kTiles = similarity_k_tiles<Shape>();
    const auto * job = static_cast<const SimilarityWeightsJob *>(context);
    Vector<uint32_t> * tiles = job->stream + static_cast<size_t>(item) * (kTiles + 1) * 16;
    Vector<uint32_t> scratch_a[32];
    Vector<uint32_t> scratch_b[32];
    const _Float16 * rows = job->second_image_descriptors + static_cast<size_t>(32 * item) * Shape::kTokenRowStride;

    for (int v = 0; v < kTiles / 2; v++) {
        hvx::transpose_words_32x32(rows + v * 64, Shape::kTokenRowStride * 2, 32, tiles + v * 32, tiles + v * 32 + 16,
                                   scratch_b, scratch_a);
    }

    tiles[16 * kTiles] = Vector<uint32_t>::splat(0x3C00);  // output block: scale fp16 1.0, bias 0
    for (int k = 1; k < 16; k++) tiles[16 * kTiles + k] = Vector<uint32_t>::zero();
}

void row_log_sum_exp(void * context, int item, int) {
    const auto * job = static_cast<const AssignmentJob *>(context);
    const int first_row = 2 * item;
    const int rows_here = job->first_image_tokens - first_row < 2 ? 1 : 2;
    const int vectors = (job->second_image_tokens + 31) / 32;
    const Vector<float> * rows[2] = {similarity_row(job, first_row), similarity_row(job, first_row + rows_here - 1)};
    const Vector<float> minus_infinity = Vector<float>::splat_bits(kMinusInfinityBits);
    const Vector<float> log2e = Vector<float>::splat(kLog2e);
    const Mask last_lanes = Mask::first_bytes_up_to_128(4 * (job->second_image_tokens - 32 * (vectors - 1)));
    Vector<float> maxima[2] = {minus_infinity, minus_infinity};
    Vector<float> maximum_vectors[2];
    Vector<float> sums[2] = {Vector<float>::zero(), Vector<float>::zero()};

    for (int v = 0; v < vectors; v++) {
#pragma clang loop unroll(full)
        for (int q = 0; q < 2; q++) {
            maxima[q] = max(maxima[q], v + 1 < vectors ? rows[q][v] : select(last_lanes, rows[q][v], minus_infinity));
        }
    }

    float maximum[2];
#pragma clang loop unroll(full)
    for (int q = 0; q < 2; q++) {
        maximum[q] = horizontal_max_f32(maxima[q]);
        maximum_vectors[q] = Vector<float>::splat(maximum[q]);
    }

    int v = 0;
    for (; v + 4 < vectors; v += 4) {  // `<`: the masked last vector is left to the tail
        Vector<float> e[8];
#pragma clang loop unroll(full)
        for (int q = 0; q < 2; q++) {
#pragma clang loop unroll(full)
            for (int k = 0; k < 4; k++) e[4 * q + k] = (rows[q][v + k] - maximum_vectors[q]) * log2e;
        }
        vector_math::exp2_batch<8>(e);
#pragma clang loop unroll(full)
        for (int k = 0; k < 4; k++) {
#pragma clang loop unroll(full)
            for (int q = 0; q < 2; q++) sums[q] = sums[q] + e[4 * q + k];
        }
    }
    {
        Vector<float> e[8];
#pragma clang loop unroll(full)
        for (int q = 0; q < 2; q++) {
#pragma clang loop unroll(full)
            for (int k = 0; k < 4; k++) {
                e[4 * q + k] = ((v + k < vectors ? rows[q][v + k] : maximum_vectors[q]) - maximum_vectors[q]) * log2e;
            }
        }
        vector_math::exp2_batch<8>(e);
        for (int k = 0; k < 4 && v + k < vectors; k++) {
            for (int q = 0; q < 2; q++) {
                sums[q] = sums[q] + (v + k + 1 < vectors ? e[4 * q + k]
                                                         : select(last_lanes, e[4 * q + k], Vector<float>::zero()));
            }
        }
    }

    for (int q = 0; q < rows_here; q++) job->row_lse[first_row + q] = maximum[q] + logf(horizontal_sum(sums[q]));
}

void column_log_sum_exp(void * context, int item, int) {
    const auto * job = static_cast<const AssignmentJob *>(context);
    const int column_vectors = job->stride / 32;
    const int count = column_vectors - 2 * item < 2 ? 1 : 2;
    const int columns[2] = {2 * item, 2 * item + count - 1};
    const Vector<float> minus_infinity = Vector<float>::splat_bits(kMinusInfinityBits);
    const Vector<float> log2e = Vector<float>::splat(kLog2e);
    Vector<float> maxima[2] = {minus_infinity, minus_infinity};
    Vector<float> sums[2] = {Vector<float>::zero(), Vector<float>::zero()};

    for (int i = 0; i < job->first_image_tokens; i++) {
        const auto * row = hvx::vectors(job->similarity + static_cast<size_t>(i) * job->stride);
        maxima[0] = max(maxima[0], row[columns[0]]);
        maxima[1] = max(maxima[1], row[columns[1]]);
    }

    int i = 0;
    for (; i + 4 <= job->first_image_tokens; i += 4) {
        Vector<float> e[8];
#pragma clang loop unroll(full)
        for (int q = 0; q < 2; q++) {
#pragma clang loop unroll(full)
            for (int k = 0; k < 4; k++) {
                e[4 * q + k] =
                    (hvx::vectors(job->similarity + static_cast<size_t>(i + k) * job->stride)[columns[q]] - maxima[q]) *
                    log2e;
            }
        }
        vector_math::exp2_batch<8>(e);
#pragma clang loop unroll(full)
        for (int k = 0; k < 4; k++) {
            sums[0] = sums[0] + e[k];
            sums[1] = sums[1] + e[4 + k];
        }
    }
    if (i < job->first_image_tokens) {
        Vector<float> e[8];
#pragma clang loop unroll(full)
        for (int q = 0; q < 2; q++) {
#pragma clang loop unroll(full)
            for (int k = 0; k < 4; k++) {
                e[4 * q + k] =
                    ((i + k < job->first_image_tokens
                          ? hvx::vectors(job->similarity + static_cast<size_t>(i + k) * job->stride)[columns[q]]
                          : maxima[q]) -
                     maxima[q]) *
                    log2e;
            }
        }
        vector_math::exp2_batch<8>(e);
        for (int k = 0; k < 4 && i + k < job->first_image_tokens; k++) {
            sums[0] = sums[0] + e[k];
            sums[1] = sums[1] + e[4 + k];
        }
    }

    for (int q = 0; q < count; q++) {
        alignas(128) float lane_maxima[32];
        alignas(128) float lane_sums[32];
        store(lane_maxima, maxima[q]);
        store(lane_sums, sums[q]);
        for (int k = 0; k < 32 && 32 * columns[q] + k < job->second_image_tokens; k++) {
            job->column_lse[32 * columns[q] + k] = lane_maxima[k] + logf(lane_sums[k]);
        }
    }
}

// The argmax over a row of 2 sim + column terms (the row's own terms do not change it), the first among equal maxima.
void row_argmax(void * context, int item, int) {
    const auto * job = static_cast<const AssignmentJob *>(context);
    const int first_row = 2 * item;
    const int rows_here = job->first_image_tokens - first_row < 2 ? 1 : 2;
    const int vectors = (job->second_image_tokens + 31) / 32;
    const Vector<float> * rows[2] = {similarity_row(job, first_row), similarity_row(job, first_row + rows_here - 1)};
    const auto * column_terms = hvx::vectors(job->column_terms);
    const Mask last_lanes = Mask::first_bytes_up_to_128(4 * (job->second_image_tokens - 32 * (vectors - 1)));
    const Vector<float> minus_infinity = Vector<float>::splat_bits(kMinusInfinityBits);
    const Vector<int32_t> lane = hvx::load(job->lane_indices);
    Vector<float> best[2] = {minus_infinity, minus_infinity};
    Vector<int32_t> argument[2] = {Vector<int32_t>::zero(), Vector<int32_t>::zero()};

    for (int v = 0; v + 1 < vectors; v++) {
        row_argmax_step(rows, column_terms, lane, last_lanes, minus_infinity, v, false, best, argument);
    }
    row_argmax_step(rows, column_terms, lane, last_lanes, minus_infinity, vectors - 1, true, best, argument);

    for (int q = 0; q < rows_here; q++) {
        alignas(128) float best_values[32];
        alignas(128) int32_t best_arguments[32];
        store(best_values, best[q]);
        store(best_arguments, argument[q]);

        int k = 0;
        for (int l = 1; l < 32; l++) {
            if (best_values[l] > best_values[k] ||
                (best_values[l] == best_values[k] && best_arguments[l] < best_arguments[k])) {
                k = l;
            }
        }

        job->row_matches[first_row + q] = best_arguments[k];
        job->row_max_scores[first_row + q] =
            best_values[k] - job->row_lse[first_row + q] + job->first_log_sigmoids[first_row + q];
    }
}

// The argmax over a column of 2 sim + row terms (the column's own terms do not change it), the first among equal
// maxima.
void column_argmax(void * context, int item, int) {
    const auto * job = static_cast<const AssignmentJob *>(context);
    Vector<float> best = Vector<float>::splat_bits(kMinusInfinityBits);
    Vector<int32_t> argument = Vector<int32_t>::zero();

    for (int i = 0; i < job->first_image_tokens; i++) {
        const auto * row = hvx::vectors(job->similarity + static_cast<size_t>(i) * job->stride);
        const Vector<float> x = row[item] + row[item] + job->row_terms[i];
        const Mask greater = x > best;
        best = select(greater, x, best);
        argument = select(greater, Vector<int32_t>::splat(i), argument);
    }

    alignas(128) int32_t best_arguments[32];
    store(best_arguments, argument);
    for (int k = 0; k < 32 && 32 * item + k < job->second_image_tokens; k++) {
        job->column_matches[32 * item + k] = best_arguments[k];
    }
}

void filter_matches(void * context, int item, int) {
    auto * job = static_cast<MatchFilterJob *>(context);
    int matches = 0;

    for (int i = 32 * item; i < 32 * item + 32 && i < job->first_image_tokens; i++) {
        const int k = job->row_matches[i];
        const int mutual = job->column_matches[k] == i;
        const float score = mutual ? expf(job->row_max_scores[i]) : 0.0f;
        job->match_scores[job->first_keypoints[i]] = score;
        if (mutual && score > job->threshold) {
            job->matches[job->first_keypoints[i]] = job->second_keypoints[k];
            matches++;
        }
    }

    job->match_count.fetch_add(matches);
}

alignas(128) constexpr int32_t kLaneIndices[32] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                                   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

template <class V>
runtime::Result<bool> runs_precisely(runtime::Session & session, const PreciseAttentionArguments & arguments) {
    if constexpr (V::kPreciseScoreBound > 0.0f) {
        const runtime::Result<float> bound = attention_score_bound(session, arguments);
        if (!bound) return std::unexpected(bound.error());
        return *bound > V::kPreciseScoreBound;
    } else {
        return false;
    }
}

constexpr int kMaximumTokenRowStride = SuperPointLightGlue::kTokenRowStride;
constexpr int kMaximumTableRowHalfwords = 2 * 128;
constexpr int kParameterFloats = 4 * SuperPointLightGlue::kDimensions + 128;
static_assert(kMaximumTokenRowStride >= XFeatLightGlue::kTokenRowStride &&
              kMaximumTableRowHalfwords >= 2 * XFeatLightGlue::kHeadStride &&
              kMaximumTableRowHalfwords >= 2 * SuperPointLightGlue::kHeadStride &&
              kParameterFloats >= head_parameter_floats<SuperPointLightGlue>());

struct Workspace {
    int capacity;
    runtime::HeapArray<_Float16> projected_descriptors;
    runtime::HeapArray<_Float16> rotary_table;
    runtime::HeapArray<float> log_sigmoids;
    runtime::HeapArray<int> table_rows;
    runtime::HeapArray<int> source_rows;
    runtime::HeapArray<float> row_lse;
    runtime::HeapArray<float> column_lse;
    runtime::HeapArray<int> row_matches;
    runtime::HeapArray<int> column_matches;
    runtime::HeapArray<float> row_max_scores;
    runtime::HeapArray<float> column_terms;
    runtime::HeapArray<float> row_terms;
    // a LayerNorm's gamma | beta or the heads' weights (matchability_parameters layout), copied out of the mapped
    // model: HVX loads from it are slow
    runtime::HeapArray<float> parameters;
    runtime::HeapArray<int> first_keypoints;
    runtime::HeapArray<int> second_keypoints;
    runtime::HeapArray<float> confidence_logits;
    runtime::HeapArray<float> matchability_logits;
    runtime::HeapArray<_Float16> ddr_token_rows;
    runtime::HeapArray<_Float16> ddr_attention_rows;
    runtime::HeapArray<uint8_t> packed_keys;
    runtime::HeapArray<uint8_t> packed_values;
    runtime::HeapArray<float> ddr_similarity;  // [n0p][n1p], n0p + n1p <= kMaximumTokens
};

runtime::Status allocate_workspace(Workspace & workspace) {
    if (workspace.capacity != 0) return {};

    const size_t rows = kMaximumTokens;
    bool allocated = runtime::allocate(workspace.projected_descriptors, rows * kMaximumTokenRowStride);
    allocated &= runtime::allocate(workspace.rotary_table, (rows + 1) * kMaximumTableRowHalfwords);
    allocated &= runtime::allocate(workspace.log_sigmoids, rows);
    allocated &= runtime::allocate(workspace.table_rows, rows);
    allocated &= runtime::allocate(workspace.source_rows, rows);
    allocated &= runtime::allocate(workspace.row_lse, rows);
    allocated &= runtime::allocate(workspace.column_lse, rows);
    allocated &= runtime::allocate(workspace.row_matches, rows);
    allocated &= runtime::allocate(workspace.column_matches, rows);
    allocated &= runtime::allocate(workspace.row_max_scores, rows);
    allocated &= runtime::allocate(workspace.column_terms, rows);
    allocated &= runtime::allocate(workspace.row_terms, rows);
    allocated &= runtime::allocate(workspace.parameters, kParameterFloats);
    allocated &= runtime::allocate(workspace.first_keypoints, rows);
    allocated &= runtime::allocate(workspace.second_keypoints, rows);
    allocated &= runtime::allocate(workspace.confidence_logits, rows);
    allocated &= runtime::allocate(workspace.matchability_logits, rows);
    if (!allocated) {
        workspace = Workspace{};
        return std::unexpected(runtime::Error::kNoMemory);
    }

    workspace.capacity = kMaximumTokens;
    return {};
}

constexpr size_t kPackedTileBytes = static_cast<size_t>(SuperPointLightGlue::kHeads) * (kMaximumTokens / 32) *
                                    (SuperPointLightGlue::kHeadDimensions / 32) * 2048;

runtime::Status allocate_ddr_rows(Workspace & workspace) {
    if (workspace.ddr_token_rows.get()) return {};

    const size_t rows = kMaximumTokens;
    const size_t row_values = kMaximumTokenRowStride;
    bool allocated = runtime::allocate(workspace.ddr_token_rows, rows * row_values);
    allocated &= runtime::allocate(workspace.ddr_attention_rows, 3 * rows * row_values);
    allocated &= runtime::allocate(workspace.packed_keys, kPackedTileBytes, 4096);
    allocated &= runtime::allocate(workspace.packed_values, kPackedTileBytes, 4096);
    allocated &= runtime::allocate(workspace.ddr_similarity, rows / 2 * (rows / 2));
    if (!allocated) {
        workspace.ddr_token_rows.reset();
        workspace.ddr_attention_rows.reset();
        workspace.packed_keys.reset();
        workspace.packed_values.reset();
        workspace.ddr_similarity.reset();
        return std::unexpected(runtime::Error::kNoMemory);
    }

    return {};
}

// block: 0 self ([q|k|v]), 1 cross ([qk|v]: LightGlue's CrossBlock shares qk between queries and keys)
inline const uint8_t * block_part(const LayerWeights & model, int layer, int block, int which) {
    return model.part<uint8_t>(protocol::lightglue::block_part_reference(layer, block, which));
}

struct DescriptorRowsJob {
    _Float16 * rows;
    int row_stride;
    int input_dimensions;
    const _Float16 * first_descriptors;
    const _Float16 * second_descriptors;
    int first_image_tokens;
    int second_image_tokens;
    int second_image_first_row;
};

inline const _Float16 * descriptor_of_row(const DescriptorRowsJob * job, int row) {
    const auto dimensions = static_cast<size_t>(job->input_dimensions);
    if (row < job->first_image_tokens) return job->first_descriptors + static_cast<size_t>(row) * dimensions;
    if (row >= job->second_image_first_row && row < job->second_image_first_row + job->second_image_tokens) {
        return job->second_descriptors + static_cast<size_t>(row - job->second_image_first_row) * dimensions;
    }
    return nullptr;
}

void write_descriptor_rows(void * context, int item, int) {
    const auto * job = static_cast<const DescriptorRowsJob *>(context);
    const int vectors = job->input_dimensions / 64;
    const uint32_t item_bytes = static_cast<uint32_t>(32 * 2 * job->input_dimensions);
    const _Float16 * const ahead = descriptor_of_row(job, (item + 2) * 32);
    if (ahead) hvx::prefetch_to_l2(ahead, item_bytes, item_bytes, 1);

    for (int row = item * 32; row < item * 32 + 32; row++) {
        auto * destination = hvx::vectors(job->rows + static_cast<size_t>(row) * job->row_stride);
        const _Float16 * source = descriptor_of_row(job, row);
        for (int k = 0; k < vectors; k++) destination[k] = source ? hvx::vectors(source)[k] : Vector<_Float16>::zero();
    }
}

void zero_rows(_Float16 * rows, int row_stride, int first, int end, int first_vector, int vectors) {
    for (int row = first; row < end; row++) {
        for (int k = first_vector; k < vectors; k++) {
            hvx::vectors(rows + static_cast<size_t>(row) * row_stride)[k] = Vector<_Float16>::zero();
        }
    }
}

template <class Shape>
void copy_head_parameters(float * destination, const uint8_t * source) {
    memcpy(destination, source, Shape::kDimensions * 4);
    for (int k = Shape::kDimensions; k < Shape::kTokenRowStride; k++) destination[k] = 0.0f;
    memcpy(destination + Shape::kTokenRowStride, source + Shape::kDimensions * 4, 4);
}

// One call at a time: the calls share the session's state.
struct CallGuard {
    std::atomic<int> in_use;
};

template <protocol::lightglue::Model kModel>
runtime::Status run_variant(runtime::Session & session, const Arguments & arguments) {
    using Shape = typename Variant<kModel>::Shape;
    const float depth_confidence = arguments.depth_confidence;
    const float width_confidence = arguments.width_confidence;
    if (!protocol::lightglue::token_confidences(kModel) && (depth_confidence > 0 || width_confidence > 0)) {
        return std::unexpected(runtime::Error::kBadParameter);
    }

    constexpr int kTokenVectors = Shape::kTokenRowStride / 64;
    constexpr int kAttentionVectors = Shape::kAttentionRowStride / 64;
    const Features * first = feature_slot(session, arguments.first_slot);
    const Features * second = feature_slot(session, arguments.second_slot);
    const int model_slot = arguments.references[0];
    if (!session.model_mapped(model_slot) || first == nullptr || second == nullptr || first->count < 1 ||
        second->count < 1 || first->dimensions != Shape::kInputDimensions ||
        second->dimensions != Shape::kInputDimensions) {
        return std::unexpected(runtime::Error::kBadParameter);
    }

    // Image 0's tokens are rows [0, n0), image 1's [n0p, n0p + n1) (n0p, n1p: n0, n1 rounded up to 32), N rows in all;
    // pruning lowers the counts. The padding rows start at zero and stay finite.
    int n0 = first->count;
    int n1 = second->count;
    const LayerWeights weights{session.model_base(model_slot), arguments.references};
    const int first_keypoint_count = n0;
    const int second_keypoint_count = n1;
    int n0p = hvx::round_up(n0, 32);
    int n1p = hvx::round_up(n1, 32);
    int N = n0p + n1p;
    runtime::StageTimer timer(arguments.statistics);

    // VTCM, top down: x, then Q | K | V; after the layers, the similarity and its weight stream in the same region
    // (sized before pruning, which only makes the similarity smaller). The kernels get the rest.
    const size_t token_bytes = static_cast<size_t>(N) * Shape::kTokenRowStride * 2;
    const size_t qkv_bytes = static_cast<size_t>(3) * N * Shape::kAttentionRowStride * 2;
    const auto similarity_weight_bytes = [](int n1_padded) {
        return static_cast<size_t>(n1_padded / 32) * (similarity_k_tiles<Shape>() + 1) * 2048;
    };
    const size_t similarity_bytes = static_cast<size_t>(n0p) * n1p * 4 + similarity_weight_bytes(n1p);
    const size_t region_bytes =
        (token_bytes + qkv_bytes > similarity_bytes ? token_bytes + qkv_bytes : similarity_bytes);
    // The rows stay in VTCM if the kernels keep 2 MB; in DDR, the attention needs the head stride d.
    const bool rows_in_vtcm = region_bytes + (static_cast<size_t>(2) << 20) <= session.vtcm_bytes();
    if (N > kMaximumTokens || (!rows_in_vtcm && Shape::kHeadStride != Shape::kHeadDimensions)) {
        return std::unexpected(runtime::Error::kBadParameter);
    }

    CallGuard * guard = session.state<CallGuard>();
    if (guard == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(guard->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);
    Workspace * kept = session.state<Workspace>();
    if (kept == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    Workspace & workspace = *kept;
    runtime::Status status = allocate_workspace(workspace);
    if (!status) return status;
    if (!rows_in_vtcm) {
        status = allocate_ddr_rows(workspace);
        if (!status) return status;
    }

    uint8_t * const vtcm_end = session.vtcm() + session.vtcm_bytes();
    _Float16 * X = workspace.ddr_token_rows.get();
    _Float16 * Q = workspace.ddr_attention_rows.get();
    if (rows_in_vtcm) {
        X = reinterpret_cast<_Float16 *>(vtcm_end - token_bytes);
        Q = X - qkv_bytes / sizeof(_Float16);
    }
    _Float16 * K = Q + static_cast<size_t>(N) * Shape::kAttentionRowStride;
    _Float16 * V = K + static_cast<size_t>(N) * Shape::kAttentionRowStride;
    _Float16 * const O = K;  // K's rows: packed into tiles by then (self) or unused (cross)
    runtime::Session kernel_session =
        session.with_vtcm_bytes(static_cast<uint32_t>(session.vtcm_bytes() - (rows_in_vtcm ? region_bytes : 0)));

    DescriptorRowsJob descriptor_job{Shape::kInputProjection ? Q : X,
                                     Shape::kInputProjection ? Shape::kInputDimensions : Shape::kTokenRowStride,
                                     Shape::kInputDimensions,
                                     first->descriptors.get(),
                                     second->descriptors.get(),
                                     n0,
                                     n1,
                                     n0p};
    status = runtime::parallel_for(session, runtime::kHvxThreads, N / 32, write_descriptor_rows, &descriptor_job);
    if (!status) return status;
    if constexpr (Shape::kInputProjection) {
        zero_rows(X, Shape::kTokenRowStride, 0, N, Shape::kDimensions / 64, kTokenVectors);  // columns >= D stay zero
        gemm_fp16::Arguments projection;
        projection.input_rows = Q;
        projection.row_stride = Shape::kInputDimensions;
        projection.input_in_vtcm = rows_in_vtcm;
        projection.weights = weights.part<uint8_t>(protocol::lightglue::input_projection_reference(Shape::kLayers));
        projection.M = N;
        projection.K = Shape::kInputDimensions;
        projection.N = Shape::kDimensions;
        projection.f16_groups[0] = X;
        projection.f16_group_width = Shape::kDimensions;
        projection.f16_row_stride = Shape::kTokenRowStride;
        projection.skip_cache_flush = true;
        status = kernels::gemm_fp16::run(kernel_session, projection);
        if (!status) return status;
        zero_rows(X, Shape::kTokenRowStride, n0, n0p, 0, kTokenVectors);  // the projection put its bias there
        zero_rows(X, Shape::kTokenRowStride, n0p + n1, N, 0, kTokenVectors);
    }

    float frequencies[Shape::kHeadDimensions];
    memcpy(frequencies, weights.part<uint8_t>(protocol::lightglue::positional_encoding_reference(Shape::kLayers)),
           sizeof frequencies);
    RotaryTableJob table_job{workspace.rotary_table.get(),
                             {first->keypoints.get(), second->keypoints.get()},
                             frequencies,
                             n0,
                             n1,
                             n0p,
                             N,
                             {},
                             {}};
    for (int image = 0; image < 2; image++) {
        const int height = image != 0 ? second->height : first->height;
        const int width = image != 0 ? second->width : first->width;
        const int longer = width > height ? width : height;
        table_job.half_size[image][0] = Variant<kModel>::kPixelKeypoints ? 0.5f * static_cast<float>(width) : 0.0f;
        table_job.half_size[image][1] = Variant<kModel>::kPixelKeypoints ? 0.5f * static_cast<float>(height) : 0.0f;
        table_job.half_max_size[image] = Variant<kModel>::kPixelKeypoints ? 0.5f * static_cast<float>(longer) : 1.0f;
    }
    status = runtime::parallel_for(session, runtime::kHvxThreads, N / 32, compute_rotary_rows<Shape>, &table_job);
    if (!status) return status;
    const int zero_table_row = N;
    memset(workspace.rotary_table.get() + static_cast<size_t>(zero_table_row) * 2 * Shape::kHeadStride, 0,
           4 * Shape::kHeadStride);

    for (int i = 0; i < N; i++) workspace.table_rows[i] = i;
    for (int i = 0; i < first_keypoint_count; i++) workspace.first_keypoints[i] = i;
    for (int i = 0; i < second_keypoint_count; i++) workspace.second_keypoints[i] = i;

    timer.lap(stat::kSetupMicroseconds);

    int last_layer = Shape::kLayers - 1;
    int projections_ready = 0;
    for (int layer = 0; layer < Shape::kLayers; layer++) {
        const int first_tokens[2] = {0, n0p};
        const int token_counts[2] = {n0, n1};
        arguments.statistics[stat::kFirstLayerTokens + layer] = n0 + n1;

        for (int block = 0; block < 2; block++) {
            if (!projections_ready) {
                gemm_fp16::Arguments projection;
                projection.input_rows = X;
                projection.row_stride = Shape::kTokenRowStride;
                projection.input_in_vtcm = rows_in_vtcm;
                projection.weights = block_part(weights, layer, block, part::kQkv);
                projection.M = N;
                projection.K = Shape::kTokenRowStride;  // whole rows: the export pads K with zero weights
                projection.N = (block != 0 ? 2 : 3) * Shape::kDimensions;
                projection.f16_groups[0] = Q;
                projection.f16_groups[1] = block != 0 ? V : K;
                projection.f16_groups[2] = V;
                projection.f16_group_width = Shape::kDimensions;
                projection.f16_row_stride = Shape::kAttentionRowStride;
                projection.skip_cache_flush = true;
                runtime::StageTimer qkv_timer(arguments.statistics);
                status = kernels::gemm_fp16::run(kernel_session, projection);
                qkv_timer.lap(stat::kQkvGemmMicroseconds);
                if (!status) return status;
            }

            runtime::StageTimer attention_timer(arguments.statistics);
            for (int k = 0; k < 3; k++) {
                _Float16 * rows = k == 0 ? Q : k == 1 ? K : V;
                zero_rows(rows, Shape::kAttentionRowStride, n0, n0p, 0, kAttentionVectors);
                zero_rows(rows, Shape::kAttentionRowStride, n0p + n1, N, 0, kAttentionVectors);
            }

            const auto attend = [&](_Float16 * keys, const kernels::flash_attention::Segments & segments) {
                namespace attention = kernels::flash_attention;
                attention::Arguments call{
                    .source = rows_in_vtcm ? attention::Source::kRowsInVtcm : attention::Source::kRows,
                    .queries = N,
                    .keys = N,
                    .query_heads = Shape::kHeads,
                    .kv_heads = Shape::kHeads,
                    .head_dimensions = Shape::kHeadDimensions,
                    .segments = &segments,
                    .q = Q,
                    .k = keys,
                    .v = V,
                    .o = O};
                if (rows_in_vtcm) {
                    call.row_layout = {Shape::kAttentionRowStride, Shape::kHeadStride};
                } else {
                    call.tiles = {workspace.packed_keys.get(), workspace.packed_values.get(), N / 32};
                }
                return attention::run(kernel_session, call);
            };

            if (block == 0) {
                RotationJob rotation_job{Q, K, workspace.rotary_table.get(), N, workspace.table_rows.get()};
                status = runtime::parallel_for(session, runtime::kHvxThreads,
                                               (N + kRotationRowsPerItem - 1) / kRotationRowsPerItem,
                                               rotate_rows<Shape>, &rotation_job);
                attention_timer.lap(stat::kHvxPassesMicroseconds);
                if (!status) return status;

                const kernels::flash_attention::Segments segments{2, first_tokens, token_counts, nullptr};
                const PreciseAttentionArguments precise{
                    Q, K, V, O, Shape::kAttentionRowStride, Shape::kHeads, 2, first_tokens, token_counts, nullptr};
                const runtime::Result<bool> precisely = runs_precisely<Variant<kModel>>(session, precise);
                if (!precisely) return std::unexpected(precisely.error());
                status = *precisely ? run_precise_attention(kernel_session, precise) : attend(K, segments);
            } else {
                attention_timer.lap(stat::kHvxPassesMicroseconds);
                const int key_segments[2] = {1, 0};
                const kernels::flash_attention::Segments segments{2, first_tokens, token_counts, key_segments};
                const PreciseAttentionArguments precise{
                    Q, Q, V, O, Shape::kAttentionRowStride, Shape::kHeads, 2, first_tokens, token_counts, key_segments};
                const runtime::Result<bool> precisely = runs_precisely<Variant<kModel>>(session, precise);
                if (!precisely) return std::unexpected(precisely.error());
                status = *precisely ? run_precise_attention(kernel_session, precise) : attend(Q, segments);
            }
            attention_timer.lap(stat::kAttentionMicroseconds);
            if (!status) return status;

            // After a cross block, the next layer's projections run before the stop / pruning decision: the pruning
            // moves their rows, a stop drops them.
            const int next_projections = block == 0 || layer + 1 < Shape::kLayers;
            memcpy(workspace.parameters.get(), block_part(weights, layer, block, part::kLayerNorm),
                   4 * Shape::kDimensions * 4);
            BlockPipelineArguments pipeline;
            pipeline.attention_output = O;
            pipeline.token_rows = X;
            pipeline.rows = N;
            pipeline.weights[0] = block_part(weights, layer, block, part::kOutput);
            pipeline.weights[1] = block_part(weights, layer, block, part::kFfn1);
            pipeline.weights[2] = block_part(weights, layer, block, part::kFfn2);
            pipeline.weights[3] =
                next_projections ? block_part(weights, layer + block, block ^ 1, part::kQkv) : nullptr;
            pipeline.layer_norm = workspace.parameters.get();
            pipeline.projection_blocks = (block != 0 ? 3 : 2) * Shape::kTiles;
            pipeline.projection_rows[0] = Q;
            pipeline.projection_rows[1] = block != 0 ? K : V;
            pipeline.projection_rows[2] = V;
            pipeline.first_image_tokens = n0;
            pipeline.second_image_first_row = n0p;
            pipeline.second_image_tokens = n1;
            runtime::StageTimer pipeline_timer(arguments.statistics);
            status = run_block_pipeline<Shape>(kernel_session, pipeline);
            pipeline_timer.lap(stat::kOutputAndFfnMicroseconds);
            if (!status) return status;
            projections_ready = next_projections;
        }

        if (layer + 1 < Shape::kLayers && (depth_confidence > 0 || width_confidence > 0)) {
            const float layer_threshold = confidence_threshold(layer, Shape::kLayers);
            if (depth_confidence > 0) {
                copy_head_parameters<Shape>(
                    workspace.parameters.get(),
                    weights.part<uint8_t>(protocol::lightglue::token_confidence_reference(Shape::kLayers, layer)));
            }
            if (width_confidence > 0) {
                copy_head_parameters<Shape>(
                    workspace.parameters.get() + matchability_parameters<Shape>(),
                    weights.part<uint8_t>(protocol::lightglue::matchability_reference(Shape::kLayers, layer)));
            }
            HeadJob head_job{X,
                             workspace.parameters.get(),
                             workspace.confidence_logits.get(),
                             workspace.matchability_logits.get(),
                             depth_confidence > 0,
                             width_confidence > 0};
            status = runtime::parallel_for(session, runtime::kHvxThreads, N / 32, evaluate_heads<Shape>, &head_job);
            if (!status) return status;

            if (depth_confidence > 0) {  // as LightGlue: pruned keypoints count as confident
                const float low_limit = sigmoid_threshold_logit(layer_threshold, 0);
                int low = 0;
                for (int i = 0; i < n0; i++) low += workspace.confidence_logits[i] < low_limit;
                for (int i = 0; i < n1; i++) low += workspace.confidence_logits[n0p + i] < low_limit;
                if (1.0f - static_cast<float>(low) / static_cast<float>(first_keypoint_count + second_keypoint_count) >
                    depth_confidence) {
                    last_layer = layer;
                }
            }

            if (last_layer != layer && width_confidence > 0) {  // at any token count, as LightGlue on the CPU
                const float confidence_limit = sigmoid_threshold_logit(layer_threshold, 1);
                const float matchability_limit =
                    sigmoid_threshold_logit(static_cast<float>(static_cast<_Float16>(1.0f - width_confidence)), 1);
                const int kept0 = keep_tokens(workspace.table_rows.get(), workspace.first_keypoints.get(),
                                              workspace.source_rows.get(), workspace.confidence_logits.get(),
                                              workspace.matchability_logits.get(), confidence_limit, matchability_limit,
                                              depth_confidence > 0, 0, n0, 0);
                const int kept0_padded = hvx::round_up(kept0, 32);
                const int kept1 = keep_tokens(workspace.table_rows.get(), workspace.second_keypoints.get(),
                                              workspace.source_rows.get(), workspace.confidence_logits.get(),
                                              workspace.matchability_logits.get(), confidence_limit, matchability_limit,
                                              depth_confidence > 0, n0p, n1, kept0_padded);
                if (kept0 < n0 || kept1 < n1) {
                    n0 = kept0;
                    n1 = kept1;
                    n0p = kept0_padded;
                    n1p = hvx::round_up(kept1, 32);
                    N = n0p + n1p;

                    for (int row = n0; row < n0p; row++) {
                        workspace.source_rows[row] = -1;
                        workspace.table_rows[row] = zero_table_row;
                    }
                    for (int row = n0p + n1; row < N; row++) {
                        workspace.source_rows[row] = -1;
                        workspace.table_rows[row] = zero_table_row;
                    }

                    MoveJob move_job{{X, projections_ready ? Q : nullptr, projections_ready ? K : nullptr,
                                      projections_ready ? V : nullptr},
                                     workspace.source_rows.get(),
                                     N};
                    status = runtime::parallel_for(session, runtime::kHvxThreads, 4, move_rows<Shape>, &move_job);
                    if (!status) return status;
                }
            }
        }

        timer.lap(stat::kFirstLayerMicroseconds + layer);
        if (last_layer == layer || n0 == 0 || n1 == 0) break;
    }

    arguments.statistics[stat::kLayersRun] = last_layer + 1;

    for (int i = 0; i < first_keypoint_count; i++) {
        arguments.matches[i] = -1;
        arguments.match_scores[i] = 0.0f;
    }
    if (n0 == 0 || n1 == 0) return {};

    // d = x Wf^T + bf, Wf and bf scaled by D^-0.25 at export (LightGlue scales both sides)
    if constexpr (Shape::kTokenRowStride > Shape::kDimensions) {
        zero_rows(workspace.projected_descriptors.get(), Shape::kTokenRowStride, 0, N, Shape::kDimensions / 64,
                  kTokenVectors);
    }
    gemm_fp16::Arguments final_projection;
    final_projection.input_rows = X;
    final_projection.row_stride = Shape::kTokenRowStride;
    final_projection.input_in_vtcm = rows_in_vtcm;
    final_projection.weights =
        weights.part<uint8_t>(protocol::lightglue::final_projection_reference(Shape::kLayers, last_layer));
    final_projection.M = N;
    final_projection.K = Shape::kTokenRowStride;
    final_projection.N = Shape::kDimensions;
    final_projection.f16_groups[0] = workspace.projected_descriptors.get();
    final_projection.f16_group_width = Shape::kDimensions;
    final_projection.f16_row_stride = Shape::kTokenRowStride;
    final_projection.skip_cache_flush = true;
    status = kernels::gemm_fp16::run(kernel_session, final_projection);
    if (!status) return status;

    if constexpr (Variant<kModel>::kMatchability) {
        copy_head_parameters<Shape>(
            workspace.parameters.get(),
            weights.part<uint8_t>(protocol::lightglue::matchability_reference(Shape::kLayers, last_layer)));
        MatchabilityJob matchability_job{X, workspace.parameters.get(), workspace.log_sigmoids.get(), n0, n0p, n1};
        status = runtime::parallel_for(session, runtime::kHvxThreads, N / 32, compute_matchability<Shape>,
                                       &matchability_job);
        if (!status) return status;
    } else {
        memset(workspace.log_sigmoids.get(), 0, static_cast<size_t>(N) * 4);
    }

    const size_t weight_bytes = similarity_weight_bytes(n1p);
    uint8_t * const similarity_weights = vtcm_end - weight_bytes;
    float * similarity = workspace.ddr_similarity.get();
    runtime::Session similarity_session = session.with_vtcm_bytes(
        static_cast<uint32_t>(session.vtcm_bytes() - (rows_in_vtcm ? region_bytes : weight_bytes)));
    if (rows_in_vtcm) {
        similarity = reinterpret_cast<float *>(vtcm_end - region_bytes);
        if (static_cast<size_t>(n0p) * n1p * 4 + weight_bytes > region_bytes) {
            return std::unexpected(runtime::Error::kNoMemory);
        }
    }

    SimilarityWeightsJob weights_job{
        workspace.projected_descriptors.get() + static_cast<size_t>(n0p) * Shape::kTokenRowStride,
        reinterpret_cast<hvx::Vector<uint32_t> *>(similarity_weights)};
    status =
        runtime::parallel_for(session, runtime::kHvxThreads, n1p / 32, write_similarity_weights<Shape>, &weights_job);
    if (!status) return status;
    gemm_fp16::Arguments similarity_gemm;
    similarity_gemm.input_rows = workspace.projected_descriptors.get();
    similarity_gemm.row_stride = Shape::kTokenRowStride;
    similarity_gemm.weights = similarity_weights;
    similarity_gemm.M = n0p;
    similarity_gemm.K = Shape::kTokenRowStride;
    similarity_gemm.N = n1p;
    similarity_gemm.output_form = kernels::gemm_fp16::OutputForm::kRowsF32;
    similarity_gemm.rows_f32 = similarity;
    similarity_gemm.skip_cache_flush = true;
    status = kernels::gemm_fp16::run(similarity_session, similarity_gemm);
    if (!status) return status;

    timer.lap(stat::kFinalProjectionMicroseconds);

    AssignmentJob assignment_job{similarity,
                                 n0,
                                 n1,
                                 n1p,
                                 workspace.row_lse.get(),
                                 workspace.column_lse.get(),
                                 workspace.log_sigmoids.get(),
                                 workspace.row_max_scores.get(),
                                 workspace.row_matches.get(),
                                 workspace.column_matches.get(),
                                 workspace.column_terms.get(),
                                 workspace.row_terms.get(),
                                 kLaneIndices};
    status = runtime::parallel_for(session, runtime::kHvxThreads, (n0 + 1) / 2, row_log_sum_exp, &assignment_job);
    if (!status) return status;
    status =
        runtime::parallel_for(session, runtime::kHvxThreads, (n1p / 32 + 1) / 2, column_log_sum_exp, &assignment_job);
    if (!status) return status;

    for (int k = 0; k < n1p; k++) {
        workspace.column_terms[k] = k < n1 ? workspace.log_sigmoids[n0p + k] - workspace.column_lse[k] : 0.0f;
    }
    for (int i = 0; i < n0; i++) workspace.row_terms[i] = workspace.log_sigmoids[i] - workspace.row_lse[i];

    status = runtime::parallel_for(session, runtime::kHvxThreads, (n0 + 1) / 2, row_argmax, &assignment_job);
    if (!status) return status;
    status = runtime::parallel_for(session, runtime::kHvxThreads, n1p / 32, column_argmax, &assignment_job);
    if (!status) return status;

    MatchFilterJob filter_job{workspace.row_matches.get(),
                              workspace.column_matches.get(),
                              workspace.first_keypoints.get(),
                              workspace.second_keypoints.get(),
                              workspace.row_max_scores.get(),
                              arguments.threshold,
                              n0,
                              arguments.matches,
                              arguments.match_scores,
                              0};
    status = runtime::parallel_for(session, runtime::kHvxThreads, (n0 + 31) / 32, filter_matches, &filter_job);
    if (!status) return status;
    arguments.statistics[stat::kMatches] = filter_job.match_count.load();
    timer.lap(stat::kAssignmentMicroseconds);
    return {};
}

}  // namespace

runtime::Status run(runtime::Session & session, const Arguments & arguments) {
    switch (arguments.model) {
        case Model::kSuperPoint:
            return run_variant<Model::kSuperPoint>(session, arguments);
        case Model::kXFeat:
            return run_variant<Model::kXFeat>(session, arguments);
        case Model::kAliked:
            return run_variant<Model::kAliked>(session, arguments);
        case Model::kLoma:
            return run_variant<Model::kLoma>(session, arguments);
    }
    return std::unexpected(runtime::Error::kBadParameter);
}

}  // namespace benzene::models::lightglue
