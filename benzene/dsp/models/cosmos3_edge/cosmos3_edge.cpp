#include "dsp/models/cosmos3_edge/cosmos3_edge.h"

#include <atomic>
#include <cmath>
#include <cstring>

#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/quantization.h"
#include "dsp/hvx/reductions.h"
#include "dsp/hvx/scalar_math.h"
#include "dsp/hvx/vector.h"
#include "dsp/kernels/flash_attention/flash_attention.h"
#include "dsp/kernels/gemm_w8a8/gemm_w8a8.h"
#include "dsp/kernels/gemv/gemv.h"
#include "dsp/models/shared/layer_weights.h"
#include "dsp/runtime/cache.h"
#include "dsp/runtime/dma_ring.h"
#include "dsp/runtime/heap.h"
#include "dsp/runtime/int8_scale_table.h"
#include "dsp/runtime/stage_timer.h"
#include "dsp/runtime/status.h"
#include "dsp/runtime/synchronization.h"
#include "dsp/runtime/thread_pool.h"
#include "dsp/vector_math/functions.h"
#include "benzene/protocol.h"

namespace benzene::models::cosmos3_edge {
namespace {

namespace attention = kernels::flash_attention;
namespace gemm = kernels::gemm_w8a8;
namespace layer_part = protocol::cosmos3_edge::llm_layer_part;
namespace tail_part = protocol::cosmos3_edge::vision_tail_part;
namespace vision_part = protocol::cosmos3_edge::vision_layer_part;
namespace head_reference = protocol::cosmos3_edge::llm_head_reference;
namespace block_stat = protocol::block_stat;
namespace decode_stat = protocol::cosmos3_edge::llm_decode_stat;
namespace prefill_stat = protocol::cosmos3_edge::llm_prefill_stat;
namespace vision_stat = protocol::cosmos3_edge::vision_stat;
using gemm::Activation;
using gemm::InputForm;
using gemm::OutputForm;
using hvx::Mask;
using hvx::Vector;
using hvx::VectorPair;
using protocol::cosmos3_edge::kLlmHidden;
using protocol::cosmos3_edge::kLlmLayer4BitReferences;
using protocol::cosmos3_edge::kLlmLayerReferences;
using protocol::cosmos3_edge::kRopeValuesPerToken;
using protocol::cosmos3_edge::kVisionHidden;
using protocol::cosmos3_edge::kVisionLayerReferences;

constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 8;
constexpr int kHeadDimensions = 128;
constexpr int kQueryWidth = kQueryHeads * kHeadDimensions;
constexpr int kKvWidth = kKvHeads * kHeadDimensions;
constexpr int kQkvColumns = kQueryWidth + 2 * kKvWidth;
constexpr int kLlmMlpHidden = 9216;

struct DecoderWorkspace {
    int tokens;
    int hidden;
    int mlp_hidden;
    int capacity;
    runtime::HeapArray<float> x;  // the residual stream [tokens][hidden]
    runtime::HeapArray<float> rope_angles;
    runtime::HeapArray<_Float16> queries;  // [tokens][16][128]
    runtime::HeapArray<_Float16> attention_output;
    runtime::HeapArray<_Float16> keys;  // [tokens][8][128]
    runtime::HeapArray<_Float16> values;
    runtime::HeapArray<uint8_t> input_tiles;
    runtime::HeapArray<float> input_row_scales;
    kernels::gemm_w8a8::HandoverTiles handover;
    runtime::HeapArray<float> decode_mlp_vector;
    runtime::HeapArray<float> decode_qkv_vector;
    runtime::HeapArray<_Float16> handover_tiles;
    runtime::HeapArray<float> handover_row_scales;
    runtime::HeapArray<uint16_t> handover_inverse_bits;
};

enum class WorkspaceKind : int { kPrefill = 0, kDecode = 1 };

constexpr int kMlpChunkRows = 2048;

struct NormJob {
    int tokens;
    int hidden;
    const float * x;
    const float * gamma;
    uint8_t * input_tiles;
    float * input_row_scales;
    Vector<float> * scratch;
    runtime::DmaRing * rings[runtime::kHvxThreads];
    const runtime::Int8Scale * int8_scales;
};

inline uint8_t * activation_line(const NormJob * job, int quad, int vectors) {
    return job->input_tiles + static_cast<size_t>(quad / 16) * vectors * 2048 + (quad % 16) * 128;
}

void normalize_quad(NormJob * job, int quad, const float * quad_rows, Vector<_Float16> * scratch) {
    const int vectors = job->hidden / 32;
    const auto * gamma = hvx::vectors(job->gamma);
    const Vector<int16_t> magnitude_mask = Vector<int16_t>::splat(0x7fff);
    const Vector<_Float16> * normalized[4];
    uint16_t inverse_bits[4] = {0, 0, 0, 0};
    const float inverse_hidden = 1.0f / static_cast<float>(job->hidden);

    if (quad * 4 + 4 <= job->tokens) {
        const Vector<float> * x[4];
        Vector<float> even_sums[4];
        Vector<float> odd_sums[4];
        Vector<float> inverse_rms[4];
        Vector<int16_t> maxima[4];

#pragma clang loop unroll(full)
        for (int t = 0; t < 4; t++) {
            x[t] = hvx::vectors(quad_rows + static_cast<size_t>(t) * job->hidden);
            even_sums[t] = odd_sums[t] = Vector<float>::zero();
        }
        for (int i = 0; i < vectors; i += 2) {
#pragma clang loop unroll(full)
            for (int t = 0; t < 4; t++) {
                even_sums[t] = even_sums[t] + x[t][i] * x[t][i];
                odd_sums[t] = odd_sums[t] + x[t][i + 1] * x[t][i + 1];
            }
        }

        alignas(128) float sums[32];
        store(sums, hvx::collect_lane0_of_four(vector_math::sum_to_all_lanes(even_sums[0] + odd_sums[0]),
                                               vector_math::sum_to_all_lanes(even_sums[1] + odd_sums[1]),
                                               vector_math::sum_to_all_lanes(even_sums[2] + odd_sums[2]),
                                               vector_math::sum_to_all_lanes(even_sums[3] + odd_sums[3])));
#pragma clang loop unroll(full)
        for (int t = 0; t < 4; t++) {
            inverse_rms[t] = Vector<float>::splat(hvx::reciprocal_square_root(sums[t] * inverse_hidden + 1e-5f));
        }

        for (int t0 = 0; t0 < 4; t0 += 2) {
            const Vector<float> * xa = x[t0];
            const Vector<float> * xb = x[t0 + 1];
            const Vector<float> ra = inverse_rms[t0];
            const Vector<float> rb = inverse_rms[t0 + 1];
            Vector<_Float16> * out_a = scratch + static_cast<size_t>(t0) * (vectors / 2);
            Vector<_Float16> * out_b = out_a + vectors / 2;
            Vector<int16_t> max_a = Vector<int16_t>::zero();
            Vector<int16_t> max_b = Vector<int16_t>::zero();
            for (int i = 0; i < vectors / 2; i++) {
                const Vector<_Float16> half_a =
                    to_float16(xa[2 * i] * ra * gamma[2 * i], xa[2 * i + 1] * ra * gamma[2 * i + 1]);
                const Vector<_Float16> half_b =
                    to_float16(xb[2 * i] * rb * gamma[2 * i], xb[2 * i + 1] * rb * gamma[2 * i + 1]);
                out_a[i] = half_a;
                out_b[i] = half_b;
                max_a = max(max_a, reinterpret<int16_t>(half_a) & magnitude_mask);
                max_b = max(max_b, reinterpret<int16_t>(half_b) & magnitude_mask);
            }
            maxima[t0] = max_a;
            maxima[t0 + 1] = max_b;
            normalized[t0] = out_a;
            normalized[t0 + 1] = out_b;
        }

        uint16_t amax_bits[4];
        hvx::max_f16_magnitude_bits_of_four(maxima[0], maxima[1], maxima[2], maxima[3], amax_bits);
        for (int t = 0; t < 4; t++) {
            const runtime::Int8Scale scale = job->int8_scales[amax_bits[t]];
            inverse_bits[t] = scale.inverse_bits;
            job->input_row_scales[quad * 4 + t] = scale.scale;
        }
        hvx::quantize_four_rows_f16(normalized, inverse_bits, job->hidden, activation_line(job, quad, vectors));
        return;
    }

    for (int t = 0; t < 4; t++) {
        const int token = quad * 4 + t;
        normalized[t] = nullptr;
        if (token >= job->tokens) continue;

        const auto * x = hvx::vectors(quad_rows + static_cast<size_t>(t) * job->hidden);
        Vector<float> even_sum = Vector<float>::zero();
        Vector<float> odd_sum = Vector<float>::zero();
        for (int i = 0; i < vectors; i += 2) {
            even_sum = even_sum + x[i] * x[i];
            odd_sum = odd_sum + x[i + 1] * x[i + 1];
        }

        const float sum = hvx::horizontal_sum(even_sum + odd_sum);
        const Vector<float> inverse_rms =
            Vector<float>::splat(hvx::reciprocal_square_root(sum * inverse_hidden + 1e-5f));

        Vector<_Float16> * out = scratch + static_cast<size_t>(t) * (vectors / 2);
        Vector<int16_t> maximum = Vector<int16_t>::zero();
        for (int i = 0; i < vectors / 2; i++) {
            const Vector<_Float16> half =
                to_float16(x[2 * i] * inverse_rms * gamma[2 * i], x[2 * i + 1] * inverse_rms * gamma[2 * i + 1]);
            out[i] = half;
            maximum = max(maximum, reinterpret<int16_t>(half) & magnitude_mask);
        }

        const runtime::Int8Scale scale = job->int8_scales[hvx::max_f16_magnitude_bits(maximum)];
        inverse_bits[t] = scale.inverse_bits;
        job->input_row_scales[token] = scale.scale;
        normalized[t] = out;
    }

    hvx::quantize_four_rows_f16(normalized, inverse_bits, job->hidden, activation_line(job, quad, vectors));
}

inline void push_quad(const NormJob * job, runtime::DmaRing * ring, uint8_t * buffers, size_t quad_bytes, int quad) {
    const int rows = job->tokens - quad * 4 < 4 ? job->tokens - quad * 4 : 4;
    const uint32_t bytes = static_cast<uint32_t>(rows) * static_cast<uint32_t>(job->hidden) * 4;
    ring->push(buffers + static_cast<size_t>(quad & 1) * quad_bytes,
               job->x + static_cast<size_t>(quad) * 4 * job->hidden, bytes, 1, bytes, bytes);
}

void normalize_thread(void * job_pointer, int thread) {
    auto * job = static_cast<NormJob *>(job_pointer);
    const int quads = (job->tokens + 3) / 4;
    const int first_quad = static_cast<int>(static_cast<int64_t>(quads) * thread / runtime::kHvxThreads);
    const int end_quad = static_cast<int>(static_cast<int64_t>(quads) * (thread + 1) / runtime::kHvxThreads);
    const size_t quad_bytes = static_cast<size_t>(4) * job->hidden * 4;
    uint8_t * buffers = reinterpret_cast<uint8_t *>(job->scratch) + static_cast<size_t>(thread) * 3 * quad_bytes;
    runtime::DmaRing * ring = job->rings[thread];

    if (first_quad < end_quad) push_quad(job, ring, buffers, quad_bytes, first_quad);
    for (int quad = first_quad; quad < end_quad; quad++) {
        if (quad + 1 < end_quad) push_quad(job, ring, buffers, quad_bytes, quad + 1);
        ring->pop();
        normalize_quad(job, quad, reinterpret_cast<const float *>(buffers + static_cast<size_t>(quad & 1) * quad_bytes),
                       reinterpret_cast<Vector<_Float16> *>(buffers + 2 * quad_bytes));
    }
}

struct DecoderState {
    DecoderWorkspace workspaces[2];
    runtime::HeapObject<runtime::DmaRing> norm_rings[runtime::kHvxThreads];
};

// The DMA of a large pass bypasses the cache: x, written by HVX stores, is flushed first.
runtime::Status normalize_and_quantize(runtime::Session & session, NormJob * job) {
    DecoderState * state = session.state<DecoderState>();
    if (state == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const bool through_cache = static_cast<size_t>(job->tokens) * job->hidden * 4 <= (static_cast<size_t>(5) << 20);

    for (int thread = 0; thread < runtime::kHvxThreads; thread++) {
        runtime::HeapObject<runtime::DmaRing> & ring = state->norm_rings[thread];
        if (!ring) {
            ring = runtime::allocate_object<runtime::DmaRing>(
                256, (sizeof(runtime::DmaRing) + 255) & ~static_cast<size_t>(255));
            if (!ring) return std::unexpected(runtime::Error::kNoMemory);
        }
        ring->initialize(through_cache ? runtime::DmaBypass::kDestination : runtime::DmaBypass::kBoth);
        job->rings[thread] = ring.get();
    }

    job->int8_scales = runtime::int8_scales(session);
    if (!through_cache) runtime::flush_data_cache();
    return runtime::run_parallel(session, runtime::kHvxThreads, normalize_thread, job);
}

runtime::Status allocate_workspace(DecoderWorkspace * w, int tokens, int hidden, int mlp_hidden) {
    *w = DecoderWorkspace{};
    w->tokens = tokens;
    w->hidden = hidden;
    w->mlp_hidden = mlp_hidden;
    w->capacity = tokens;
    const size_t rows64 = static_cast<size_t>(tokens + 63) / 64 * 64;
    const size_t chunk_rows64 = static_cast<size_t>((tokens < kMlpChunkRows ? tokens : kMlpChunkRows) + 63) / 64 * 64;

    w->x = runtime::allocate_array<float>(static_cast<size_t>(tokens) * hidden);
    const bool decode = tokens == 1;
    if (decode) {
        w->decode_mlp_vector =
            runtime::allocate_array<float>(static_cast<size_t>(mlp_hidden > hidden ? mlp_hidden : hidden) + 32);
        w->decode_qkv_vector = runtime::allocate_array<float>(kQkvColumns);
    }
    w->rope_angles = runtime::allocate_array<float>(static_cast<size_t>(tokens) * kRopeValuesPerToken);
    w->queries = runtime::allocate_array<_Float16>(static_cast<size_t>(tokens) * kQueryWidth);
    w->attention_output = runtime::allocate_array<_Float16>(static_cast<size_t>(tokens) * kQueryWidth);
    w->keys = runtime::allocate_array<_Float16>(static_cast<size_t>(tokens) * kKvWidth);
    w->values = runtime::allocate_array<_Float16>(static_cast<size_t>(tokens) * kKvWidth);
    w->input_tiles = runtime::allocate_array<uint8_t>(rows64 * hidden, 4096);
    w->input_row_scales = runtime::allocate_array<float>(rows64);
    w->handover_tiles = runtime::allocate_array<_Float16>(chunk_rows64 * mlp_hidden);
    w->handover_row_scales = runtime::allocate_array<float>(chunk_rows64);
    w->handover_inverse_bits = runtime::allocate_array<uint16_t>(chunk_rows64);
    w->handover = {w->handover_tiles.get(), w->handover_row_scales.get(), w->handover_inverse_bits.get()};

    if (!w->x || (decode && (!w->decode_mlp_vector || !w->decode_qkv_vector)) || !w->rope_angles || !w->queries ||
        !w->attention_output || !w->keys || !w->values || !w->input_tiles || !w->input_row_scales ||
        !w->handover_tiles || !w->handover_row_scales || !w->handover_inverse_bits) {
        *w = DecoderWorkspace{};
        return std::unexpected(runtime::Error::kNoMemory);
    }
    return {};
}

runtime::Result<DecoderWorkspace *> decoder_workspace(runtime::Session & session, WorkspaceKind kind, int tokens,
                                                      int hidden, int mlp_hidden) {
    DecoderState * state = session.state<DecoderState>();
    if (state == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    DecoderWorkspace * w = &state->workspaces[static_cast<int>(kind)];
    if (!w->x || w->hidden != hidden || w->mlp_hidden != mlp_hidden || w->capacity < tokens) {
        const runtime::Status allocated = allocate_workspace(w, tokens, hidden, mlp_hidden);
        if (!allocated) return std::unexpected(allocated.error());
    }
    w->tokens = tokens;
    return w;
}

void copy_rope_angles(float * destination, const float * rope, int tokens) {
    constexpr int kHalf = kRopeValuesPerToken / 2;
    for (int token = 0; token < tokens; token++) {
        float * row = destination + static_cast<size_t>(token) * kRopeValuesPerToken;
        memcpy(row, rope + static_cast<size_t>(token) * kHalf, kHalf * 4);
        memcpy(row + kHalf, rope + static_cast<size_t>(tokens + token) * kHalf, kHalf * 4);
    }
}

runtime::Status run_decoder_layer(runtime::Session & session, const LayerWeights & layer, DecoderWorkspace * w,
                                  const kernels::flash_attention::KeyValueTiles * cache, int32_t * statistics) {
    runtime::StageTimer timer(statistics);
    NormJob norm{.tokens = w->tokens,
                 .hidden = w->hidden,
                 .x = w->x.get(),
                 .gamma = layer.part<float>(layer_part::kInputNormGamma),
                 .input_tiles = w->input_tiles.get(),
                 .input_row_scales = w->input_row_scales.get(),
                 .scratch = reinterpret_cast<Vector<float> *>(session.vtcm())};
    runtime::Status status = normalize_and_quantize(session, &norm);
    if (!status) return status;
    timer.lap(block_stat::kNorm1Microseconds);

    gemm::Input normalized;
    normalized.form = InputForm::kPrequantizedTiles;
    normalized.tiles = w->input_tiles.get();
    normalized.tile_row_scales = w->input_row_scales.get();
    const bool to_cache = cache != nullptr;
    _Float16 * keys = w->keys.get();
    _Float16 * values = w->values.get();
    {
        gemm::Output output;
        output.form = OutputForm::kRowsF16;
        output.f16_groups[0] = w->queries.get();
        output.f16_groups[1] = w->queries.get() + kQueryWidth / 2;
        output.f16_groups[2] = keys;
        output.f16_groups[3] = values;
        output.f16_group_width = kQueryWidth / 2;
        output.f16_group_strides[0] = kQueryWidth;
        output.f16_group_strides[1] = kQueryWidth;
        output.f16_group_strides[2] = kKvWidth;
        output.f16_group_strides[3] = kKvWidth;
        output.rope_groups = 0x7;  // Q and K
        output.rope_cos = w->rope_angles.get();
        output.rope_sin = w->rope_angles.get() + kRopeValuesPerToken / 2;
        output.rope_stride = kRopeValuesPerToken;
        output.key_cache = to_cache ? static_cast<uint8_t *>(cache->key_tiles) : nullptr;
        output.value_cache = to_cache ? static_cast<uint8_t *>(cache->value_tiles) : nullptr;
        output.cache_key_tile_stride = to_cache ? cache->capacity : 0;
        output.cache_head_dimensions = kHeadDimensions;
        status = gemm::run(session, {.M = w->tokens,
                                     .K = w->hidden,
                                     .N = kQkvColumns,
                                     .weights = layer.gemm(layer_part::kQkvWeights, layer_part::kQkvScales),
                                     .input = normalized,
                                     .output = output});
        if (!status) return status;
    }
    timer.lap(block_stat::kQkvMicroseconds);

    {
        attention::Arguments call{.source = to_cache ? attention::Source::kKeyValueCache : attention::Source::kRows,
                                  .queries = w->tokens,
                                  .keys = w->tokens,
                                  .query_heads = kQueryHeads,
                                  .kv_heads = kKvHeads,
                                  .head_dimensions = kHeadDimensions,
                                  .causal = true,
                                  .q = w->queries.get(),
                                  .o = w->attention_output.get()};
        if (to_cache) {
            call.tiles = *cache;
        } else {
            call.k = keys;
            call.v = values;
        }
        status = attention::run(session, call);
        if (!status) return status;
    }
    timer.lap(block_stat::kAttentionMicroseconds);

    {
        gemm::Input input;
        input.form = InputForm::kRowsF16Scaled;
        input.rows_f16 = w->attention_output.get();
        input.column_scale = layer.part<float>(layer_part::kOutputInverseScale);
        input.row_stride = kQueryWidth;
        gemm::Output output;
        output.form = OutputForm::kRowsF32Accumulate;
        output.rows_f32 = w->x.get();
        status = gemm::run(session, {.M = w->tokens,
                                     .K = kQueryWidth,
                                     .N = w->hidden,
                                     .weights = layer.gemm(layer_part::kOutputWeights, layer_part::kOutputScales),
                                     .input = input,
                                     .output = output});
        if (!status) return status;
    }
    timer.lap(block_stat::kOutputMicroseconds);

    norm.gamma = layer.part<float>(layer_part::kPostAttentionNormGamma);
    status = normalize_and_quantize(session, &norm);
    if (!status) return status;
    timer.lap(block_stat::kNorm2Microseconds);

    for (int first_row = 0; first_row < w->tokens; first_row += kMlpChunkRows) {
        const int rows = w->tokens - first_row < kMlpChunkRows ? w->tokens - first_row : kMlpChunkRows;

        {
            gemm::Input input;
            input.form = InputForm::kPrequantizedTiles;
            input.tiles = w->input_tiles.get() + static_cast<size_t>(first_row / 64) * (w->hidden / 32) * 2048;
            input.tile_row_scales = w->input_row_scales.get() + first_row;
            gemm::Output output;
            output.form = OutputForm::kHandoverTiles;
            output.activation = Activation::kReluSquared;
            output.column_scale = layer.part<float>(layer_part::kFc2InverseScale);
            output.handover = &w->handover;
            status = gemm::run(session, {.M = rows,
                                         .K = w->hidden,
                                         .N = w->mlp_hidden,
                                         .weights = layer.gemm(layer_part::kFc1Weights, layer_part::kFc1Scales),
                                         .input = input,
                                         .output = output});
            if (!status) return status;
        }
        timer.lap(block_stat::kFc1Microseconds);

        {
            gemm::Input input;
            input.form = InputForm::kHandoverTiles;
            input.handover = &w->handover;
            gemm::Output output;
            output.form = OutputForm::kRowsF32Accumulate;
            output.rows_f32 = w->x.get() + static_cast<size_t>(first_row) * w->hidden;
            status = gemm::run(session, {.M = rows,
                                         .K = w->mlp_hidden,
                                         .N = w->hidden,
                                         .weights = layer.gemm(layer_part::kFc2Weights, layer_part::kFc2Scales),
                                         .input = input,
                                         .output = output});
            if (!status) return status;
        }
        timer.lap(block_stat::kFc2Microseconds);
    }
    return {};
}

constexpr int kMaximumKvCacheLayers = 64;
struct KvCache {
    runtime::HeapArray<uint8_t> key_tiles[kMaximumKvCacheLayers];
    runtime::HeapArray<uint8_t> value_tiles[kMaximumKvCacheLayers];
    int layers;
    int key_tile_capacity;
    int keys;
};

// One call at a time: the calls share the session's state (the KV cache and the decoder workspaces).
struct LanguageModelGuard {
    std::atomic<int> in_use;
};

// One call at a time: the calls share the session's state (the vision workspace).
struct VisionGuard {
    std::atomic<int> in_use;
};

struct LanguageModelState {
    KvCache kv_cache;
    runtime::GrowingArray<float> logits_buffer;
};

attention::KeyValueTiles kv_cache_tiles(const KvCache & kv_cache, int layer) {
    return {kv_cache.key_tiles[layer].get(), kv_cache.value_tiles[layer].get(), kv_cache.key_tile_capacity};
}

void normalize_row(const float * x, const float * gamma, float * y) {
    const auto * in = hvx::vectors(x);
    const auto * gamma_vectors = hvx::vectors(gamma);
    auto * out = hvx::vectors(y);

    Vector<float> even_sum = Vector<float>::zero();
    Vector<float> odd_sum = Vector<float>::zero();
    for (int i = 0; i < kLlmHidden / 32; i += 2) {
        even_sum = even_sum + in[i] * in[i];
        odd_sum = odd_sum + in[i + 1] * in[i + 1];
    }

    const float sum = hvx::horizontal_sum(even_sum + odd_sum);
    const Vector<float> inverse_rms = Vector<float>::splat(1.0f / sqrtf(sum / kLlmHidden + 1e-5f));
    for (int i = 0; i < kLlmHidden / 32; i++) out[i] = in[i] * inverse_rms * gamma_vectors[i];
}

void add_row(float * x, const float * addend) {
    auto * sum = hvx::vectors(x);
    const auto * in = hvx::vectors(addend);
    for (int i = 0; i < kLlmHidden / 32; i++) sum[i] = sum[i] + in[i];
}

void relu_squared_row(float * h, const float * inverse_scales) {
    auto * values = hvx::vectors(h);
    const auto * inverse = hvx::vectors(inverse_scales);
    for (int i = 0; i < kLlmMlpHidden / 32; i++) {
        const Vector<float> positive = max(values[i], Vector<float>::zero());
        values[i] = positive * positive * inverse[i];
    }
}

// rotate_half RoPE: pair (i, i + 64) rotates by angle i. The export stores each Q / K head's dims, in and out, as
// [0..31, 64..95, 32..63, 96..127].
inline void rotate_head(const Vector<float> * source, Vector<_Float16> * destination, Vector<float> cos0,
                        Vector<float> cos1, Vector<float> sin0, Vector<float> sin1) {
    const Vector<float> a0 = source[0];
    const Vector<float> b0 = source[1];
    const Vector<float> a1 = source[2];
    const Vector<float> b1 = source[3];
    destination[0] = to_float16(a0 * cos0 - b0 * sin0, b0 * cos0 + a0 * sin0);
    destination[1] = to_float16(a1 * cos1 - b1 * sin1, b1 * cos1 + a1 * sin1);
}

void split_qkv_row(const float * qkv, const float * rope_angles, _Float16 * queries, _Float16 * keys,
                   _Float16 * values) {
    const auto * row = hvx::vectors(qkv);
    const auto * cosines = hvx::vectors(rope_angles);
    const auto * sines = hvx::vectors(rope_angles + kRopeValuesPerToken / 2);
    const Vector<float> cos0 = cosines[0];
    const Vector<float> cos1 = cosines[1];
    const Vector<float> sin0 = sines[0];
    const Vector<float> sin1 = sines[1];

    for (int head = 0; head < kQueryHeads; head++) {
        rotate_head(row + head * 4, hvx::vectors(queries + head * kHeadDimensions), cos0, cos1, sin0, sin1);
    }

    for (int head = 0; head < kKvHeads; head++) {
        rotate_head(row + kQueryWidth / 32 + head * 4, hvx::vectors(keys + head * kHeadDimensions), cos0, cos1, sin0,
                    sin1);
        const Vector<float> * value = row + (kQueryWidth + kKvWidth) / 32 + head * 4;
        auto * destination = hvx::vectors(values + head * kHeadDimensions);
        destination[0] = to_float16(value[0], value[1]);
        destination[1] = to_float16(value[2], value[3]);
    }
}

void attention_output_row(const _Float16 * attention_output, const float * inverse_scales, float * row) {
    const auto * inverse = hvx::vectors(inverse_scales);
    auto * out = hvx::vectors(row);
    for (int head = 0; head < kQueryHeads; head++) {
        const auto * output = hvx::vectors(attention_output + head * kHeadDimensions);
        for (int half = 0; half < 2; half++) {
            const VectorPair<float> values = to_float32(output[half]);
            const int k = head * 4 + half * 2;
            out[k] = values.low() * inverse[k];
            out[k + 1] = values.high() * inverse[k + 1];
        }
    }
}

enum Linear : int { kQkv = 0, kOutputProjection = 1, kFc1 = 2, kFc2 = 3 };

struct DecodeLayer {
    LayerWeights weights;
    const uint8_t * blob_4bit;
    const int32_t * stream_offsets_4bit;

    runtime::Status gemv(runtime::Session & session, int stream_part, int scales_part, Linear linear, const float * x,
                         float * y, int K, int N) const {
        using kernels::gemv::WeightFormat;
        if (blob_4bit != nullptr) {
            return kernels::gemv::run(session, {.format = WeightFormat::kQ4_0,
                                                .K = K,
                                                .N = N,
                                                .x = x,
                                                .weights = blob_4bit + stream_offsets_4bit[linear],
                                                .y = y});
        }

        return kernels::gemv::run(session, {.format = WeightFormat::kInt8,
                                            .K = K,
                                            .N = N,
                                            .x = x,
                                            .weights = weights.part<uint8_t>(stream_part),
                                            .column_scales = weights.part<float>(scales_part),
                                            .y = y});
    }
};

runtime::Status decode(runtime::Session & session, LanguageModelState & state, const int32_t * references, int layers,
                       const int32_t * head, const int32_t * references_4bit, int layers_4bit, DecoderWorkspace * w,
                       float * logits, int columns, int32_t * statistics) {
    runtime::StageTimer timer(statistics);
    KvCache & kv_cache = state.kv_cache;
    runtime::GrowingArray<float> & logits_buffer = state.logits_buffer;
    const int position = kv_cache.keys;
    float * mlp_vector = w->decode_mlp_vector.get();
    float * qkv_vector = w->decode_qkv_vector.get();
    runtime::Status status;

    for (int layer_index = 0; layer_index < layers; layer_index++) {
        const int32_t * reference = references + layer_index * kLlmLayerReferences;
        const int32_t * reference_4bit =
            references_4bit != nullptr ? references_4bit + layer_index * kLlmLayer4BitReferences : nullptr;
        const DecodeLayer layer{
            {session.model_base(reference[0]) + reference[1], reference + 2},
            reference_4bit != nullptr ? session.model_base(reference_4bit[0]) + reference_4bit[1] : nullptr,
            reference_4bit != nullptr ? reference_4bit + 2 : nullptr};
        const attention::KeyValueTiles cache = kv_cache_tiles(kv_cache, layer_index);

        normalize_row(w->x.get(), layer.weights.part<float>(layer_part::kInputNormGamma), mlp_vector);
        status = layer.gemv(session, layer_part::kQkvWeights, layer_part::kQkvScales, kQkv, mlp_vector, qkv_vector,
                            kLlmHidden, kQkvColumns);
        if (!status) return status;
        timer.lap(decode_stat::kNormAndQkvMicroseconds);

        split_qkv_row(qkv_vector, w->rope_angles.get(), w->queries.get(), w->keys.get(), w->values.get());
        attention::append_to_cache(w->keys.get(), w->values.get(), position, kKvHeads, kHeadDimensions, cache);
        timer.lap(decode_stat::kRopeAndAppendMicroseconds);

        status = attention::run(session, {.source = attention::Source::kKeyValueCache,
                                          .queries = 1,
                                          .keys = position + 1,
                                          .query_heads = kQueryHeads,
                                          .kv_heads = kKvHeads,
                                          .head_dimensions = kHeadDimensions,
                                          .q = w->queries.get(),
                                          .o = w->attention_output.get(),
                                          .tiles = cache});
        if (!status) return status;
        timer.lap(decode_stat::kAttentionMicroseconds);

        attention_output_row(w->attention_output.get(), layer.weights.part<float>(layer_part::kOutputInverseScale),
                             qkv_vector);
        status = layer.gemv(session, layer_part::kOutputWeights, layer_part::kOutputScales, kOutputProjection,
                            qkv_vector, mlp_vector, kQueryWidth, kLlmHidden);
        if (!status) return status;
        add_row(w->x.get(), mlp_vector);
        timer.lap(decode_stat::kOutputMicroseconds);

        normalize_row(w->x.get(), layer.weights.part<float>(layer_part::kPostAttentionNormGamma), qkv_vector);
        status = layer.gemv(session, layer_part::kFc1Weights, layer_part::kFc1Scales, kFc1, qkv_vector, mlp_vector,
                            kLlmHidden, kLlmMlpHidden);
        if (!status) return status;
        relu_squared_row(mlp_vector, layer.weights.part<float>(layer_part::kFc2InverseScale));
        timer.lap(decode_stat::kNorm2AndFc1Microseconds);

        status = layer.gemv(session, layer_part::kFc2Weights, layer_part::kFc2Scales, kFc2, mlp_vector, qkv_vector,
                            kLlmMlpHidden, kLlmHidden);
        if (!status) return status;
        add_row(w->x.get(), qkv_vector);
        timer.lap(decode_stat::kFc2Microseconds);
    }

    normalize_row(w->x.get(),
                  reinterpret_cast<const float *>(session.model_base(head[head_reference::kNormSlot]) +
                                                  head[head_reference::kNormOffset]),
                  mlp_vector);
    if (references_4bit != nullptr) {
        const int32_t * lm_head = references_4bit + layers_4bit * kLlmLayer4BitReferences;
        status = kernels::gemv::run(session, {.format = kernels::gemv::WeightFormat::kQ4_0,
                                              .K = kLlmHidden,
                                              .N = columns,
                                              .x = mlp_vector,
                                              .weights = session.model_base(lm_head[0]) + lm_head[1],
                                              .y = logits_buffer.get()});
    } else {
        status = kernels::gemv::run(
            session,
            {.format = kernels::gemv::WeightFormat::kInt8,
             .K = kLlmHidden,
             .N = columns,
             .x = mlp_vector,
             .weights = session.model_base(head[head_reference::kLmHeadSlot]) + head[head_reference::kLmHeadOffset],
             .column_scales = reinterpret_cast<const float *>(session.model_base(head[head_reference::kScalesSlot]) +
                                                              head[head_reference::kScalesOffset]),
             .y = logits_buffer.get()});
    }
    timer.lap(decode_stat::kHeadMicroseconds);
    if (!status) return status;

    memcpy(logits, logits_buffer.get(), static_cast<size_t>(columns) * 4);
    if (layers > 0) kv_cache.keys = position + 1;
    return {};
}

constexpr int kMaximumImages = 64;
constexpr int kVisionHeads = 16;
constexpr int kVisionHeadDimensions = 72;
constexpr int kVisionMlpHidden = 4320;  // 4304 zero-padded
constexpr int kVisionQkvColumns = 3 * kVisionHidden;
constexpr int kGroupPatches = 2048;

struct NormVtcm {
    struct Thread {
        Vector<float> quads[2][4 * kVisionHidden / 32];
        Vector<_Float16> normalized[4 * kVisionHidden / 32];
    } threads[runtime::kHvxThreads];
    Vector<float> gamma[kVisionHidden / 32];
};

struct VisionWorkspace {
    int patches;
    int images;
    int padded_rows;  // the padded layout: each image starts on a 32-row boundary, padding rows after it
    int image_first_rows[kMaximumImages + 1];
    int image_patches[kMaximumImages];
    runtime::HeapArray<int> padded_row_of_patch;
    runtime::HeapArray<int> patch_of_padded_row;  // -1: a padding row
    runtime::HeapArray<float> x;
    kernels::gemm_w8a8::HandoverTiles handover;
    runtime::HeapArray<_Float16> attention_output;
    runtime::HeapArray<uint8_t> query_tiles;
    runtime::HeapArray<uint8_t> key_tiles;
    runtime::HeapArray<uint8_t> value_tiles;
    runtime::HeapArray<uint8_t> input_tiles;
    runtime::HeapArray<float> input_row_scales;
    runtime::HeapObject<runtime::DmaRing> rings[runtime::kHvxThreads];
    runtime::HeapArray<_Float16> handover_tiles;
    runtime::HeapArray<float> handover_row_scales;
    runtime::HeapArray<uint16_t> handover_inverse_bits;
};

struct VisionNormJob {
    int patches;
    int padded_rows;
    int padded_order;
    const int * patch_of_padded_row;
    const float * x;
    const float * beta;
    const Vector<float> * gamma_vtcm;
    uint8_t * input_tiles;
    float * input_row_scales;
    NormVtcm * vtcm;
    const runtime::HeapObject<runtime::DmaRing> * rings;
    const runtime::Int8Scale * int8_scales;
};

inline uint8_t * vision_activation_line(const VisionNormJob * job, int quad) {
    return job->input_tiles + static_cast<size_t>(quad / 16) * (kVisionHidden / 32) * 2048 + (quad % 16) * 128;
}

inline Vector<float> layer_norm_value(Vector<float> x, Vector<float> negative_mean, Vector<float> inverse_std,
                                      Vector<float> gamma, Vector<float> beta) {
    return (x + negative_mean) * inverse_std * gamma + beta;
}

void vision_normalize_quad(const VisionNormJob * job, int quad, int rows, const Vector<float> * quad_rows,
                           Vector<_Float16> * scratch) {
    const Vector<float> * gamma = job->gamma_vtcm;
    const auto * beta = hvx::vectors(job->beta);
    const Vector<int16_t> magnitude_mask = Vector<int16_t>::splat(0x7fff);
    const Vector<_Float16> * normalized[4];
    uint16_t inverse_bits[4];

    if (rows == 4) {
        Vector<float> sums[4];
        Vector<float> squares[4];
        Vector<float> inverse_std[4];
        Vector<float> negative_mean[4];
        Vector<int16_t> maxima[4];

#pragma clang loop unroll(full)
        for (int t = 0; t < 4; t++) sums[t] = squares[t] = Vector<float>::zero();
        for (int i = 0; i < kVisionHidden / 32; i++) {
#pragma clang loop unroll(full)
            for (int t = 0; t < 4; t++) {
                const Vector<float> x = quad_rows[static_cast<size_t>(t) * (kVisionHidden / 32) + i];
                sums[t] = sums[t] + x;
                squares[t] = squares[t] + x * x;
            }
        }

        alignas(128) float moments[32];
        {
            const Vector<float> first = hvx::collect_lane0_of_four(
                vector_math::sum_to_all_lanes(sums[0]), vector_math::sum_to_all_lanes(sums[1]),
                vector_math::sum_to_all_lanes(sums[2]), vector_math::sum_to_all_lanes(sums[3]));
            const Vector<float> second = hvx::collect_lane0_of_four(
                vector_math::sum_to_all_lanes(squares[0]), vector_math::sum_to_all_lanes(squares[1]),
                vector_math::sum_to_all_lanes(squares[2]), vector_math::sum_to_all_lanes(squares[3]));
            store(moments, select(Mask::first_bytes(16), first, rotate_right_bytes(second, 112)));
        }

#pragma clang loop unroll(full)
        for (int t = 0; t < 4; t++) {
            const float mean = moments[t] * (1.0f / kVisionHidden);
            const float variance = moments[4 + t] * (1.0f / kVisionHidden) - mean * mean;
            inverse_std[t] = Vector<float>::splat(hvx::reciprocal_square_root((variance > 0 ? variance : 0) + 1e-6f));
            negative_mean[t] = Vector<float>::splat(-mean);
        }

        for (int t0 = 0; t0 < 4; t0 += 2) {
            const Vector<float> * xa = quad_rows + static_cast<size_t>(t0) * (kVisionHidden / 32);
            const Vector<float> * xb = xa + kVisionHidden / 32;
            const Vector<float> mean_a = negative_mean[t0];
            const Vector<float> inverse_a = inverse_std[t0];
            const Vector<float> mean_b = negative_mean[t0 + 1];
            const Vector<float> inverse_b = inverse_std[t0 + 1];
            Vector<int16_t> max_a = Vector<int16_t>::zero();
            Vector<int16_t> max_b = Vector<int16_t>::zero();
            Vector<_Float16> * out_a = scratch + static_cast<size_t>(t0) * (kVisionHidden / 64);
            Vector<_Float16> * out_b = out_a + kVisionHidden / 64;
            for (int i = 0; i < kVisionHidden / 64; i++) {
                const Vector<_Float16> half_a =
                    to_float16(layer_norm_value(xa[2 * i], mean_a, inverse_a, gamma[2 * i], beta[2 * i]),
                               layer_norm_value(xa[2 * i + 1], mean_a, inverse_a, gamma[2 * i + 1], beta[2 * i + 1]));
                const Vector<_Float16> half_b =
                    to_float16(layer_norm_value(xb[2 * i], mean_b, inverse_b, gamma[2 * i], beta[2 * i]),
                               layer_norm_value(xb[2 * i + 1], mean_b, inverse_b, gamma[2 * i + 1], beta[2 * i + 1]));
                out_a[i] = half_a;
                out_b[i] = half_b;
                max_a = max(max_a, reinterpret<int16_t>(half_a) & magnitude_mask);
                max_b = max(max_b, reinterpret<int16_t>(half_b) & magnitude_mask);
            }
            maxima[t0] = max_a;
            maxima[t0 + 1] = max_b;
        }

        uint16_t amax_bits[4];
        hvx::max_f16_magnitude_bits_of_four(maxima[0], maxima[1], maxima[2], maxima[3], amax_bits);
        for (int t = 0; t < 4; t++) {
            const runtime::Int8Scale scale = job->int8_scales[amax_bits[t]];
            inverse_bits[t] = scale.inverse_bits;
            job->input_row_scales[quad * 4 + t] = scale.scale;
            normalized[t] = scratch + static_cast<size_t>(t) * (kVisionHidden / 64);
        }
    } else {
        for (int t = 0; t < 4; t++) {
            const int row = quad * 4 + t;
            normalized[t] = nullptr;
            inverse_bits[t] = 0;
            if (t >= rows) {
                job->input_row_scales[row] = 0.0f;
                continue;
            }

            const Vector<float> * x = quad_rows + static_cast<size_t>(t) * (kVisionHidden / 32);
            Vector<float> sum = Vector<float>::zero();
            Vector<float> square = Vector<float>::zero();
            for (int i = 0; i < kVisionHidden / 32; i++) {
                sum = sum + x[i];
                square = square + x[i] * x[i];
            }

            const float mean = hvx::horizontal_sum(sum) * (1.0f / kVisionHidden);
            const float variance = hvx::horizontal_sum(square) * (1.0f / kVisionHidden) - mean * mean;
            const Vector<float> inverse_std =
                Vector<float>::splat(hvx::reciprocal_square_root((variance > 0 ? variance : 0) + 1e-6f));
            const Vector<float> negative_mean = Vector<float>::splat(-mean);

            Vector<_Float16> * out = scratch + static_cast<size_t>(t) * (kVisionHidden / 64);
            Vector<int16_t> maximum = Vector<int16_t>::zero();
            for (int i = 0; i < kVisionHidden / 64; i++) {
                const Vector<_Float16> half = to_float16(
                    layer_norm_value(x[2 * i], negative_mean, inverse_std, gamma[2 * i], beta[2 * i]),
                    layer_norm_value(x[2 * i + 1], negative_mean, inverse_std, gamma[2 * i + 1], beta[2 * i + 1]));
                out[i] = half;
                maximum = max(maximum, reinterpret<int16_t>(half) & magnitude_mask);
            }

            const runtime::Int8Scale scale = job->int8_scales[hvx::max_f16_magnitude_bits(maximum)];
            inverse_bits[t] = scale.inverse_bits;
            job->input_row_scales[row] = scale.scale;
            normalized[t] = out;
        }
    }

    hvx::quantize_four_rows_f16(normalized, inverse_bits, kVisionHidden, vision_activation_line(job, quad));
}

inline int quad_rows(const VisionNormJob * job, int quad, int * first_patch) {
    if (!job->padded_order) {
        *first_patch = quad * 4;
        return job->patches - quad * 4 < 4 ? job->patches - quad * 4 : 4;
    }
    int rows = 0;
    *first_patch = job->patch_of_padded_row[quad * 4];
    while (rows < 4 && job->patch_of_padded_row[quad * 4 + rows] >= 0) rows++;
    return rows;
}

inline void vision_push_quad(const VisionNormJob * job, runtime::DmaRing * ring, NormVtcm::Thread * buffers, int quad) {
    int first_patch;
    const int rows = quad_rows(job, quad, &first_patch);
    if (rows != 0) {
        const uint32_t bytes = static_cast<uint32_t>(rows) * kVisionHidden * 4;
        ring->push(buffers->quads[quad & 1], job->x + static_cast<size_t>(first_patch) * kVisionHidden, bytes, 1, bytes,
                   bytes);
    }
}

void vision_normalize_thread(void * job_pointer, int thread) {
    const auto * job = static_cast<const VisionNormJob *>(job_pointer);
    const int quads = (job->padded_order ? job->padded_rows : job->patches + 3) / 4;
    const int first_quad = quads * thread / runtime::kHvxThreads;
    const int end_quad = quads * (thread + 1) / runtime::kHvxThreads;
    NormVtcm::Thread * buffers = &job->vtcm->threads[thread];
    runtime::DmaRing * ring = job->rings[thread].get();

    if (first_quad < end_quad) vision_push_quad(job, ring, buffers, first_quad);
    for (int quad = first_quad; quad < end_quad; quad++) {
        if (quad + 1 < end_quad) vision_push_quad(job, ring, buffers, quad + 1);
        int first_patch;
        const int rows = quad_rows(job, quad, &first_patch);
        if (rows != 0) ring->pop();
        vision_normalize_quad(job, quad, rows, buffers->quads[quad & 1], buffers->normalized);
    }
}

runtime::Status vision_normalize_and_quantize(runtime::Session & session, VisionWorkspace * w, const float * gamma,
                                              const float * beta, bool padded_order) {
    for (const runtime::HeapObject<runtime::DmaRing> & ring : w->rings) ring->initialize(runtime::DmaBypass::kBoth);
    auto * vtcm = reinterpret_cast<NormVtcm *>(session.vtcm());
    for (int i = 0; i < kVisionHidden / 32; i++) vtcm->gamma[i] = hvx::vectors(gamma)[i];
    // HVX stores leave x in the cache; the DMA reads DDR
    runtime::flush_data_cache();

    VisionNormJob job{.patches = w->patches,
                      .padded_rows = w->padded_rows,
                      .padded_order = padded_order ? 1 : 0,
                      .patch_of_padded_row = w->patch_of_padded_row.get(),
                      .x = w->x.get(),
                      .beta = beta,
                      .gamma_vtcm = vtcm->gamma,
                      .input_tiles = w->input_tiles.get(),
                      .input_row_scales = w->input_row_scales.get(),
                      .vtcm = vtcm,
                      .rings = w->rings,
                      .int8_scales = runtime::int8_scales(session)};
    return runtime::run_parallel(session, runtime::kHvxThreads, vision_normalize_thread, &job);
}

runtime::Status allocate_vision_workspace(VisionWorkspace * w, int patches, int images) {
    *w = VisionWorkspace{};
    const int padded_rows = (patches + 31 * images + 31) / 32 * 32;
    const size_t tile_bytes =
        static_cast<size_t>(kVisionHeads) * (padded_rows / 32) * ((kVisionHeadDimensions + 31) / 32) * 2048;
    const size_t patches64 = static_cast<size_t>(patches + 63) / 64 * 64;
    const size_t padded_rows64 = static_cast<size_t>(padded_rows + 63) / 64 * 64;

    w->x = runtime::allocate_array<float>(static_cast<size_t>(patches) * kVisionHidden);
    w->handover_tiles = runtime::allocate_array<_Float16>(patches64 * kVisionMlpHidden);
    w->handover_row_scales = runtime::allocate_array<float>(patches64);
    w->handover_inverse_bits = runtime::allocate_array<uint16_t>(patches64);
    w->handover = {w->handover_tiles.get(), w->handover_row_scales.get(), w->handover_inverse_bits.get()};
    w->attention_output = runtime::allocate_array<_Float16>(static_cast<size_t>(padded_rows) * kVisionHidden);
    w->query_tiles = runtime::allocate_array<uint8_t>(tile_bytes, 4096);
    w->key_tiles = runtime::allocate_array<uint8_t>(tile_bytes, 4096);
    w->value_tiles = runtime::allocate_array<uint8_t>(tile_bytes, 4096);
    w->padded_row_of_patch = runtime::allocate_array<int>(patches);
    w->patch_of_padded_row = runtime::allocate_array<int>(padded_rows);
    bool allocated = true;
    for (runtime::HeapObject<runtime::DmaRing> & ring : w->rings) {
        ring = runtime::allocate_object<runtime::DmaRing>(64);
        allocated = allocated && ring;
    }
    w->input_tiles = runtime::allocate_array<uint8_t>(padded_rows64 * kVisionHidden, 4096);
    w->input_row_scales = runtime::allocate_array<float>(padded_rows64);

    if (!allocated || !w->x || !w->handover_tiles || !w->handover_row_scales || !w->handover_inverse_bits ||
        !w->attention_output || !w->query_tiles || !w->key_tiles || !w->value_tiles || !w->padded_row_of_patch ||
        !w->patch_of_padded_row || !w->input_tiles || !w->input_row_scales) {
        *w = VisionWorkspace{};
        return std::unexpected(runtime::Error::kNoMemory);
    }
    return {};
}

runtime::Status set_image_layout(VisionWorkspace * w, int patches, int images, const int32_t * image_offsets) {
    if (images < 1 || images > kMaximumImages) return std::unexpected(runtime::Error::kBadParameter);

    w->patches = patches;
    w->images = images;
    w->image_first_rows[0] = 0;
    for (int image = 0; image < images; image++) {
        w->image_patches[image] = image_offsets[image + 1] - image_offsets[image];
        w->image_first_rows[image + 1] = w->image_first_rows[image] + (w->image_patches[image] + 31) / 32 * 32;
        if (w->image_patches[image] < 1) return std::unexpected(runtime::Error::kBadParameter);

        for (int patch = image_offsets[image]; patch < image_offsets[image + 1]; patch++) {
            w->padded_row_of_patch[patch] = w->image_first_rows[image] + patch - image_offsets[image];
        }
        for (int row = w->image_first_rows[image]; row < w->image_first_rows[image + 1]; row++) {
            const int index = row - w->image_first_rows[image];
            w->patch_of_padded_row[row] = index < w->image_patches[image] ? image_offsets[image] + index : -1;
        }
    }
    w->padded_rows = w->image_first_rows[images];
    return {};
}

runtime::Status run_vision_layer(runtime::Session & session, const LayerWeights & layer, VisionWorkspace * w,
                                 int32_t * statistics) {
    runtime::StageTimer timer(statistics);
    gemm::Input normalized;
    normalized.tiles = w->input_tiles.get();
    normalized.tile_row_scales = w->input_row_scales.get();

    runtime::Status status = vision_normalize_and_quantize(session, w, layer.part<float>(vision_part::kNorm1Gamma),
                                                           layer.part<float>(vision_part::kNorm1Beta), true);
    if (!status) return status;
    timer.lap(block_stat::kNorm1Microseconds);

    {
        gemm::Output output;
        output.bias = layer.part<float>(vision_part::kQkvBias);
        output.attention_q = w->query_tiles.get();
        output.attention_k = w->key_tiles.get();
        output.attention_v = w->value_tiles.get();
        output.attention_group_blocks = kVisionHidden / 32;
        output.attention_key_tile_stride = w->padded_rows / 32;
        normalized.form = InputForm::kPrequantizedTiles;
        output.form = OutputForm::kAttentionTiles72;
        status = gemm::run(session, {.M = w->padded_rows,
                                     .K = kVisionHidden,
                                     .N = kVisionQkvColumns,
                                     .weights = layer.gemm(vision_part::kQkvWeights, vision_part::kQkvScales),
                                     .input = normalized,
                                     .output = output});
        if (!status) return status;
    }
    timer.lap(block_stat::kQkvMicroseconds);

    {
        const attention::Segments images{w->images, w->image_first_rows, w->image_patches, nullptr};
        status = attention::run(session, {.source = attention::Source::kTiles,
                                          .queries = w->padded_rows,
                                          .keys = w->padded_rows,
                                          .query_heads = kVisionHeads,
                                          .kv_heads = kVisionHeads,
                                          .head_dimensions = kVisionHeadDimensions,
                                          .segments = &images,
                                          .o = w->attention_output.get(),
                                          .tiles = {w->key_tiles.get(), w->value_tiles.get(), w->padded_rows / 32},
                                          .query_tiles = w->query_tiles.get()});
        if (!status) return status;
    }
    timer.lap(block_stat::kAttentionMicroseconds);

    {
        gemm::Input input;
        input.rows_f16 = w->attention_output.get();
        input.row_map = w->padded_row_of_patch.get();
        input.column_scale = layer.part<float>(vision_part::kOutputInverseScale);
        input.row_stride = kVisionHidden;
        gemm::Output output;
        output.rows_f32 = w->x.get();
        output.bias = layer.part<float>(vision_part::kOutputBias);
        input.form = InputForm::kRowsF16Scaled;
        output.form = OutputForm::kRowsF32Accumulate;
        status = gemm::run(session, {.M = w->patches,
                                     .K = kVisionHidden,
                                     .N = kVisionHidden,
                                     .weights = layer.gemm(vision_part::kOutputWeights, vision_part::kOutputScales),
                                     .input = input,
                                     .output = output});
        if (!status) return status;
    }
    timer.lap(block_stat::kOutputMicroseconds);

    status = vision_normalize_and_quantize(session, w, layer.part<float>(vision_part::kNorm2Gamma),
                                           layer.part<float>(vision_part::kNorm2Beta), false);
    if (!status) return status;
    timer.lap(block_stat::kNorm2Microseconds);

    {
        gemm::Output output;
        output.bias = layer.part<float>(vision_part::kFc1Bias);
        output.column_scale = layer.part<float>(vision_part::kFc2InverseScale);
        output.handover = &w->handover;
        normalized.form = InputForm::kPrequantizedTiles;
        output.form = OutputForm::kHandoverTiles;
        output.activation = Activation::kGeluTanh;
        status = gemm::run(session, {.M = w->patches,
                                     .K = kVisionHidden,
                                     .N = kVisionMlpHidden,
                                     .weights = layer.gemm(vision_part::kFc1Weights, vision_part::kFc1Scales),
                                     .input = normalized,
                                     .output = output});
        if (!status) return status;
    }
    timer.lap(block_stat::kFc1Microseconds);

    {
        gemm::Input input;
        input.handover = &w->handover;
        gemm::Output output;
        output.rows_f32 = w->x.get();
        output.bias = layer.part<float>(vision_part::kFc2Bias);
        input.form = InputForm::kHandoverTiles;
        output.form = OutputForm::kRowsF32Accumulate;
        status = gemm::run(session, {.M = w->patches,
                                     .K = kVisionMlpHidden,
                                     .N = kVisionHidden,
                                     .weights = layer.gemm(vision_part::kFc2Weights, vision_part::kFc2Scales),
                                     .input = input,
                                     .output = output});
        if (!status) return status;
    }
    timer.lap(block_stat::kFc2Microseconds);
    statistics[vision_stat::kLayersMicroseconds] += timer.elapsed();
    return {};
}

constexpr int kPatchWidth = 1152;
constexpr int kTokenWidth = 4 * kPatchWidth;
constexpr int kProjectorHidden = 11520;
constexpr int kOutputWidth = 2048;

struct TailJob {
    int patches;
    const float * x;
    const float * post_norm_gamma;
    const float * post_norm_beta;
    const float * merger_norm_gamma;
    const float * merger_norm_beta;
    uint8_t * input_tiles;
    float * input_row_scales;
    Vector<float> * scratch;
};

struct TailWorkspace {
    runtime::HeapArray<uint8_t> input_tiles;
    runtime::HeapArray<float> input_row_scales;
    kernels::gemm_w8a8::HandoverTiles handover;
    int capacity;
    runtime::HeapArray<_Float16> handover_tiles;
    runtime::HeapArray<float> handover_row_scales;
    runtime::HeapArray<uint16_t> handover_inverse_bits;
};

inline void layer_norm_row(const Vector<float> * x, Vector<float> * y, const Vector<float> * gamma,
                           const Vector<float> * beta) {
    Vector<float> sum = Vector<float>::zero();
    Vector<float> squares = Vector<float>::zero();
    for (int i = 0; i < kPatchWidth / 32; i++) {
        sum = sum + x[i];
        squares = squares + x[i] * x[i];
    }

    const float mean = hvx::horizontal_sum(sum) / kPatchWidth;
    const float variance = hvx::horizontal_sum(squares) / kPatchWidth - mean * mean;
    const Vector<float> inverse_std = Vector<float>::splat(1.0f / sqrtf((variance > 0 ? variance : 0) + 1e-6f));
    const Vector<float> negative_mean = Vector<float>::splat(-mean);
    for (int i = 0; i < kPatchWidth / 32; i++) y[i] = (x[i] + negative_mean) * inverse_std * gamma[i] + beta[i];
}

void normalize_token_quad(void * context, int quad, int thread) {
    const auto * job = static_cast<const TailJob *>(context);
    Vector<float> * scratch = job->scratch + static_cast<size_t>(thread) * 4 * (kTokenWidth / 32);
    const Vector<float> * rows[4];

    for (int t = 0; t < 4; t++) {
        const int token = quad * 4 + t;
        rows[t] = nullptr;
        if (token >= job->patches / 4) continue;

        Vector<float> * row = scratch + static_cast<size_t>(t) * (kTokenWidth / 32);
        for (int slot = 0; slot < 4; slot++) {
            Vector<float> * normalized = row + slot * (kPatchWidth / 32);
            layer_norm_row(hvx::vectors(job->x + static_cast<size_t>(token * 4 + slot) * kPatchWidth), normalized,
                           hvx::vectors(job->post_norm_gamma), hvx::vectors(job->post_norm_beta));
            layer_norm_row(normalized, normalized, hvx::vectors(job->merger_norm_gamma) + slot * (kPatchWidth / 32),
                           hvx::vectors(job->merger_norm_beta) + slot * (kPatchWidth / 32));
        }
        rows[t] = row;
    }

    float scales[4];
    hvx::quantize_four_rows_f32(
        rows, kTokenWidth,
        job->input_tiles + static_cast<size_t>(quad / 16) * (kTokenWidth / 32) * 2048 + (quad % 16) * 128, scales);
    for (int t = 0; t < 4 && quad * 4 + t < job->patches / 4; t++) job->input_row_scales[quad * 4 + t] = scales[t];
}

runtime::Status run_vision_tail(runtime::Session & session, const LayerWeights & layer, const float * x, int patches,
                                float * y, int32_t * statistics) {
    if (patches < 4 || patches % 4 != 0) return std::unexpected(runtime::Error::kBadParameter);
    TailWorkspace * kept = session.state<TailWorkspace>();
    if (kept == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    TailWorkspace & workspace = *kept;
    const int tokens = patches / 4;
    const int padded_tokens = (tokens + 63) / 64 * 64;

    if (workspace.capacity < padded_tokens) {
        workspace = TailWorkspace{};
        workspace.input_tiles =
            runtime::allocate_array<uint8_t>(static_cast<size_t>(padded_tokens) * kTokenWidth, 4096);
        workspace.input_row_scales = runtime::allocate_array<float>(padded_tokens);
        workspace.handover_tiles =
            runtime::allocate_array<_Float16>(static_cast<size_t>(padded_tokens) * kProjectorHidden);
        workspace.handover_row_scales = runtime::allocate_array<float>(padded_tokens);
        workspace.handover_inverse_bits = runtime::allocate_array<uint16_t>(padded_tokens);
        workspace.handover = {workspace.handover_tiles.get(), workspace.handover_row_scales.get(),
                              workspace.handover_inverse_bits.get()};
        if (!workspace.input_tiles || !workspace.input_row_scales || !workspace.handover_tiles ||
            !workspace.handover_row_scales || !workspace.handover_inverse_bits) {
            workspace = TailWorkspace{};
            return std::unexpected(runtime::Error::kNoMemory);
        }
        workspace.capacity = padded_tokens;
    }

    runtime::StageTimer timer(statistics);
    TailJob job{.patches = patches,
                .x = x,
                .post_norm_gamma = layer.part<float>(tail_part::kPostNormGamma),
                .post_norm_beta = layer.part<float>(tail_part::kPostNormBeta),
                .merger_norm_gamma = layer.part<float>(tail_part::kMergerNormGamma),
                .merger_norm_beta = layer.part<float>(tail_part::kMergerNormBeta),
                .input_tiles = workspace.input_tiles.get(),
                .input_row_scales = workspace.input_row_scales.get(),
                .scratch = reinterpret_cast<Vector<float> *>(session.vtcm())};
    runtime::Status status =
        runtime::parallel_for(session, runtime::kHvxThreads, (tokens + 3) / 4, normalize_token_quad, &job);
    if (!status) return status;

    {
        gemm::Input input;
        input.tiles = workspace.input_tiles.get();
        input.tile_row_scales = workspace.input_row_scales.get();
        gemm::Output output;
        output.bias = layer.part<float>(tail_part::kFc1Bias);
        output.column_scale = layer.part<float>(tail_part::kFc2InverseScale);
        output.handover = &workspace.handover;
        input.form = InputForm::kPrequantizedTiles;
        output.form = OutputForm::kHandoverTiles;
        output.activation = Activation::kGeluErf;
        status = gemm::run(session, {.M = tokens,
                                     .K = kTokenWidth,
                                     .N = kProjectorHidden,
                                     .weights = layer.gemm(tail_part::kFc1Weights, tail_part::kFc1Scales),
                                     .input = input,
                                     .output = output});
        if (!status) return status;
    }

    {
        gemm::Input input;
        input.handover = &workspace.handover;
        gemm::Output output;
        output.rows_f32 = y;
        output.bias = layer.part<float>(tail_part::kFc2Bias);
        input.form = InputForm::kHandoverTiles;
        output.form = OutputForm::kRowsF32Activation;
        status = gemm::run(session, {.M = tokens,
                                     .K = kProjectorHidden,
                                     .N = kOutputWidth,
                                     .weights = layer.gemm(tail_part::kFc2Weights, tail_part::kFc2Scales),
                                     .input = input,
                                     .output = output});
        if (!status) return status;
    }

    timer.lap(vision_stat::kTailMicroseconds);
    return {};
}

}  // namespace

runtime::Status run_prefill(runtime::Session & session, const PrefillArguments & arguments) {
    const int layers = arguments.layers;
    if (layers < 1 || arguments.tokens < 1) {
        return std::unexpected(runtime::Error::kBadParameter);
    }
    for (int layer_index = 0; layer_index < layers; layer_index++) {
        if (!layer_in_model(session, arguments.references + layer_index * kLlmLayerReferences))
            return std::unexpected(runtime::Error::kBadParameter);
    }

    LanguageModelGuard * guard = session.state<LanguageModelGuard>();
    if (guard == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(guard->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);
    LanguageModelState * state = session.state<LanguageModelState>();
    if (state == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    KvCache & kv_cache = state->kv_cache;
    if (kv_cache.layers > 0 && arguments.tokens > kv_cache.key_tile_capacity * 32) {
        return std::unexpected(runtime::Error::kBadParameter);
    }
    const runtime::Result<DecoderWorkspace *> workspace =
        decoder_workspace(session, WorkspaceKind::kPrefill, arguments.tokens, kLlmHidden, kLlmMlpHidden);
    if (!workspace) return std::unexpected(workspace.error());
    DecoderWorkspace * w = *workspace;

    runtime::Status status =
        runtime::copy_memory(session, w->x.get(), arguments.x, static_cast<size_t>(arguments.tokens) * kLlmHidden * 4);
    if (!status) return status;
    copy_rope_angles(w->rope_angles.get(), arguments.rope, arguments.tokens);

    runtime::StageTimer timer(arguments.statistics);
    for (int layer_index = 0; layer_index < layers; layer_index++) {
        const runtime::Result<LayerWeights> layer =
            layer_in_model(session, arguments.references + layer_index * kLlmLayerReferences);
        if (!layer) return std::unexpected(layer.error());
        const bool cached = layer_index < kv_cache.layers;
        const attention::KeyValueTiles cache =
            cached ? kv_cache_tiles(kv_cache, layer_index) : attention::KeyValueTiles{};
        status = run_decoder_layer(session, *layer, w, cached ? &cache : nullptr, arguments.statistics);
        if (!status) return status;
    }
    timer.lap(prefill_stat::kLayersMicroseconds);

    status = runtime::copy_memory(
        session, arguments.y,
        arguments.all_rows ? w->x.get() : w->x.get() + static_cast<size_t>(arguments.tokens - 1) * kLlmHidden,
        static_cast<size_t>(arguments.all_rows ? arguments.tokens : 1) * kLlmHidden * 4);
    if (!status) return status;
    if (kv_cache.layers > 0) kv_cache.keys = arguments.tokens;
    return {};
}

runtime::Status run_decode_step(runtime::Session & session, const DecodeStepArguments & arguments) {
    LanguageModelGuard * guard = session.state<LanguageModelGuard>();
    if (guard == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(guard->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);
    LanguageModelState * state = session.state<LanguageModelState>();
    if (state == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const KvCache & kv_cache = state->kv_cache;
    const int layers = arguments.layers;
    if (layers < 0 || (layers > 0 && (layers > kv_cache.layers || kv_cache.keys >= kv_cache.key_tile_capacity * 32))) {
        return std::unexpected(runtime::Error::kBadParameter);
    }
    for (int layer_index = 0; layer_index < layers; layer_index++) {
        if (!layer_in_model(session, arguments.references + layer_index * kLlmLayerReferences))
            return std::unexpected(runtime::Error::kBadParameter);
    }

    const bool decode_4bit = arguments.references_4bit != nullptr;
    const int layers_4bit = decode_4bit ? arguments.layers_4bit : 0;
    if (decode_4bit) {
        if (layers_4bit < layers) return std::unexpected(runtime::Error::kBadParameter);
        for (int i = 0; i <= layers_4bit; i++) {
            if (!session.model_mapped(arguments.references_4bit[i * kLlmLayer4BitReferences])) {
                return std::unexpected(runtime::Error::kBadParameter);
            }
        }
    }
    const int32_t * head = arguments.head;
    if (!session.model_mapped(head[head_reference::kNormSlot]) ||
        (!decode_4bit && (!session.model_mapped(head[head_reference::kLmHeadSlot]) ||
                          !session.model_mapped(head[head_reference::kScalesSlot])))) {
        return std::unexpected(runtime::Error::kBadParameter);
    }

    const int columns = head[head_reference::kColumns];
    const runtime::Result<DecoderWorkspace *> workspace =
        decoder_workspace(session, WorkspaceKind::kDecode, 1, kLlmHidden, kLlmMlpHidden);
    if (!workspace || state->logits_buffer.reserve(columns) == nullptr) {
        return std::unexpected(runtime::Error::kNoMemory);
    }
    DecoderWorkspace * w = *workspace;

    memcpy(w->x.get(), arguments.x, kLlmHidden * 4);
    memcpy(w->rope_angles.get(), arguments.rope, kRopeValuesPerToken * 4);
    return decode(session, *state, arguments.references, layers, arguments.head, arguments.references_4bit, layers_4bit,
                  w, arguments.logits, columns, arguments.statistics);
}

runtime::Status run_vision_encoder(runtime::Session & session, const VisionEncoderArguments & arguments) {
    const runtime::Result<LayerWeights> tail_weights = layer_in_model(session, arguments.tail);
    if (!tail_weights) return std::unexpected(tail_weights.error());
    for (int layer_index = 0; layer_index < arguments.layers; layer_index++) {
        if (!layer_in_model(session, arguments.references + layer_index * kVisionLayerReferences))
            return std::unexpected(runtime::Error::kBadParameter);
    }

    VisionGuard * guard = session.state<VisionGuard>();
    if (guard == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(guard->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);
    VisionWorkspace * workspace = session.state<VisionWorkspace>();
    if (workspace == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    runtime::StageTimer workspace_timer(arguments.statistics);
    runtime::Status status;
    if (!workspace->x) status = allocate_vision_workspace(workspace, kGroupPatches, kMaximumImages);
    workspace_timer.lap(vision_stat::kWorkspaceMicroseconds);
    if (!status) return status;

    int32_t group_offsets[kMaximumImages + 1];
    for (int first_image = 0; first_image < arguments.images;) {
        runtime::StageTimer setup_timer(arguments.statistics);
        int end_image = first_image + 1;
        while (end_image < arguments.images &&
               arguments.image_offsets[end_image + 1] - arguments.image_offsets[first_image] <= kGroupPatches &&
               end_image - first_image < kMaximumImages) {
            end_image++;
        }
        const int patches = arguments.image_offsets[end_image] - arguments.image_offsets[first_image];
        if (patches > kGroupPatches || patches % 4 != 0) return std::unexpected(runtime::Error::kBadParameter);

        for (int image = first_image; image <= end_image; image++) {
            group_offsets[image - first_image] = arguments.image_offsets[image] - arguments.image_offsets[first_image];
        }
        status = set_image_layout(workspace, patches, end_image - first_image, group_offsets);
        if (!status) return status;
        status = runtime::copy_memory(
            session, workspace->x.get(),
            arguments.x + static_cast<size_t>(arguments.image_offsets[first_image]) * kVisionHidden,
            static_cast<size_t>(patches) * kVisionHidden * 4);
        if (!status) return status;
        setup_timer.lap(vision_stat::kGroupSetupMicroseconds);

        for (int layer_index = 0; layer_index < arguments.layers; layer_index++) {
            const runtime::Result<LayerWeights> layer =
                layer_in_model(session, arguments.references + layer_index * kVisionLayerReferences);
            if (!layer) return std::unexpected(layer.error());
            status = run_vision_layer(session, *layer, workspace, arguments.statistics);
            if (!status) return status;
        }
        status = run_vision_tail(session, *tail_weights, workspace->x.get(), patches,
                                 arguments.y + static_cast<size_t>(arguments.image_offsets[first_image] / 4) * 2048,
                                 arguments.statistics);
        if (!status) return status;
        first_image = end_image;
    }
    return {};
}

runtime::Status initialize_kv_cache(runtime::Session & session, int layers, int capacity) {
    LanguageModelGuard * guard = session.state<LanguageModelGuard>();
    if (guard == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(guard->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);
    LanguageModelState * state = session.state<LanguageModelState>();
    if (state == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    KvCache & kv_cache = state->kv_cache;
    kv_cache = KvCache{};
    if (capacity <= 0 || layers <= 0) return {};
    if (layers > kMaximumKvCacheLayers) return std::unexpected(runtime::Error::kBadParameter);

    const int key_tiles = (capacity + 31) / 32;
    const size_t bytes = static_cast<size_t>(kKvHeads) * key_tiles * (kHeadDimensions / 32) * 2048;
    for (int layer = 0; layer < layers; layer++) {
        kv_cache.key_tiles[layer] = runtime::allocate_array<uint8_t>(bytes, 4096);
        kv_cache.value_tiles[layer] = runtime::allocate_array<uint8_t>(bytes, 4096);
        kv_cache.layers = layer + 1;
        if (!kv_cache.key_tiles[layer] || !kv_cache.value_tiles[layer]) {
            kv_cache = KvCache{};
            return std::unexpected(runtime::Error::kNoMemory);
        }
    }
    kv_cache.key_tile_capacity = key_tiles;
    return {};
}

runtime::Status prepare_workspaces(runtime::Session & session, int prefill_tokens) {
    LanguageModelGuard * guard = session.state<LanguageModelGuard>();
    if (guard == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(guard->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);
    if (prefill_tokens > 0) {
        const runtime::Result<DecoderWorkspace *> prefill_workspace =
            decoder_workspace(session, WorkspaceKind::kPrefill, prefill_tokens, kLlmHidden, kLlmMlpHidden);
        if (!prefill_workspace) return std::unexpected(prefill_workspace.error());
    }
    const runtime::Result<DecoderWorkspace *> decode_workspace =
        decoder_workspace(session, WorkspaceKind::kDecode, 1, kLlmHidden, kLlmMlpHidden);
    if (!decode_workspace) return std::unexpected(decode_workspace.error());
    return {};
}

}  // namespace benzene::models::cosmos3_edge
