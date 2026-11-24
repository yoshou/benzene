#include "dsp/models/shared/vision_transformer.h"

#include <cmath>

#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/quantization.h"
#include "dsp/hvx/reductions.h"
#include "dsp/hvx/scalar_math.h"
#include "dsp/hvx/vector.h"
#include "dsp/kernels/flash_attention/flash_attention.h"
#include "dsp/kernels/gemm_fp16/gemm_fp16.h"
#include "dsp/runtime/cache.h"
#include "dsp/runtime/int8_scale_table.h"
#include "dsp/runtime/stage_timer.h"
#include "dsp/runtime/thread_pool.h"
#include "dsp/vector_math/functions.h"
#include "benzene/protocol.h"

namespace benzene::models::vision_transformer {

struct NormVtcm {
    struct Thread {
        hvx::Vector<float> quads[2][4 * kHidden / 32];
        hvx::Vector<_Float16> normalized[4 * kHidden / 32];
    } threads[runtime::kHvxThreads];
    hvx::Vector<float> gamma[kHidden / 32];
};
namespace {

namespace block_part = protocol::vision_transformer::block_part;
namespace block_stat = protocol::block_stat;
namespace gemm = kernels::gemm_w8a8;
using gemm::Activation;
using gemm::InputForm;
using gemm::OutputForm;
using hvx::Mask;
using hvx::Vector;
using protocol::vision_transformer::BlockKind;

constexpr int kQkvColumns = 3 * kHidden;
constexpr int kRopeTablePositions = 128;
constexpr int kRopeTableValues = 64;

void normalize_quad(Workspace * w, int quad, int rows, const Vector<float> * quad_rows, Vector<_Float16> * scratch) {
    const Vector<float> * gamma = w->norm_gamma_vtcm;
    const auto * beta = hvx::vectors(w->norm_beta);
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
        for (int t = 0; t < 4; t++) {
            sums[t] = Vector<float>::zero();
            squares[t] = Vector<float>::zero();
        }
        for (int i = 0; i < kHidden / 32; i++) {
#pragma clang loop unroll(full)
            for (int t = 0; t < 4; t++) {
                const Vector<float> x = quad_rows[static_cast<size_t>(t) * (kHidden / 32) + i];
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
            const float mean = moments[t] * (1.0f / kHidden);
            const float variance = moments[4 + t] * (1.0f / kHidden) - mean * mean;
            inverse_std[t] =
                Vector<float>::splat(hvx::reciprocal_square_root((variance > 0 ? variance : 0) + w->norm_epsilon));
            negative_mean[t] = Vector<float>::splat(-mean);
        }

        for (int t0 = 0; t0 < 4; t0 += 2) {
            const Vector<float> * xa = quad_rows + static_cast<size_t>(t0) * (kHidden / 32);
            const Vector<float> * xb = xa + kHidden / 32;
            const Vector<float> mean_a = negative_mean[t0];
            const Vector<float> inverse_a = inverse_std[t0];
            const Vector<float> mean_b = negative_mean[t0 + 1];
            const Vector<float> inverse_b = inverse_std[t0 + 1];
            Vector<int16_t> max_a = Vector<int16_t>::zero();
            Vector<int16_t> max_b = Vector<int16_t>::zero();
            Vector<_Float16> * out_a = scratch + static_cast<size_t>(t0) * (kHidden / 64);
            Vector<_Float16> * out_b = out_a + kHidden / 64;
            for (int i = 0; i < kHidden / 64; i++) {
                const Vector<_Float16> half_a =
                    to_float16((xa[2 * i] + mean_a) * inverse_a * gamma[2 * i] + beta[2 * i],
                               (xa[2 * i + 1] + mean_a) * inverse_a * gamma[2 * i + 1] + beta[2 * i + 1]);
                const Vector<_Float16> half_b =
                    to_float16((xb[2 * i] + mean_b) * inverse_b * gamma[2 * i] + beta[2 * i],
                               (xb[2 * i + 1] + mean_b) * inverse_b * gamma[2 * i + 1] + beta[2 * i + 1]);
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
            const runtime::Int8Scale scale = w->int8_scales[amax_bits[t]];
            inverse_bits[t] = scale.inverse_bits;
            w->input_row_scales[quad * 4 + t] = scale.scale;
            normalized[t] = scratch + static_cast<size_t>(t) * (kHidden / 64);
        }
    } else {
        for (int t = 0; t < 4; t++) {
            normalized[t] = nullptr;
            inverse_bits[t] = 0;
            if (t >= rows) {
                w->input_row_scales[quad * 4 + t] = 0.0f;
                continue;
            }

            const Vector<float> * x = quad_rows + static_cast<size_t>(t) * (kHidden / 32);
            Vector<float> sum = Vector<float>::zero();
            Vector<float> square = Vector<float>::zero();
            for (int i = 0; i < kHidden / 32; i++) {
                sum = sum + x[i];
                square = square + x[i] * x[i];
            }

            const float mean = hvx::horizontal_sum(sum) * (1.0f / kHidden);
            const float variance = hvx::horizontal_sum(square) * (1.0f / kHidden) - mean * mean;
            const Vector<float> inverse =
                Vector<float>::splat(hvx::reciprocal_square_root((variance > 0 ? variance : 0) + w->norm_epsilon));
            const Vector<float> negative = Vector<float>::splat(-mean);

            Vector<_Float16> * out = scratch + static_cast<size_t>(t) * (kHidden / 64);
            Vector<int16_t> maximum = Vector<int16_t>::zero();
            for (int i = 0; i < kHidden / 64; i++) {
                const Vector<_Float16> half =
                    to_float16((x[2 * i] + negative) * inverse * gamma[2 * i] + beta[2 * i],
                               (x[2 * i + 1] + negative) * inverse * gamma[2 * i + 1] + beta[2 * i + 1]);
                out[i] = half;
                maximum = max(maximum, reinterpret<int16_t>(half) & magnitude_mask);
            }

            const runtime::Int8Scale scale = w->int8_scales[hvx::max_f16_magnitude_bits(maximum)];
            inverse_bits[t] = scale.inverse_bits;
            w->input_row_scales[quad * 4 + t] = scale.scale;
            normalized[t] = out;
        }
    }

    hvx::quantize_four_rows_f16(
        normalized, inverse_bits, kHidden,
        w->input_tiles.get() + static_cast<size_t>(quad / 16) * (kHidden / 32) * 2048 + (quad % 16) * 128);
}

inline int quad_rows(const Workspace * w, int quad, int * first_token) {
    if (w->norm_rows == nullptr) {
        *first_token = quad * 4;
        const int rows = w->tokens - quad * 4;
        return rows < 0 ? 0 : (rows < 4 ? rows : 4);
    }
    int rows = 0;
    *first_token = w->norm_rows[quad * 4];
    while (rows < 4 && w->norm_rows[quad * 4 + rows] >= 0) rows++;
    return rows;
}

inline void push_quad(const Workspace * w, runtime::DmaRing * ring, NormVtcm::Thread * buffers, int quad) {
    int first_token;
    const int rows = quad_rows(w, quad, &first_token);
    if (rows) {
        const auto bytes = static_cast<uint32_t>(rows) * kHidden * 4;
        ring->push(buffers->quads[quad & 1], w->x.get() + static_cast<size_t>(first_token) * kHidden, bytes, 1, bytes,
                   bytes);
    }
}

void normalize_thread(void * job_pointer, int thread) {
    auto * w = static_cast<Workspace *>(job_pointer);
    const int quads = w->norm_layout_rows / 4;
    const int first_quad = quads * thread / runtime::kHvxThreads;
    const int end_quad = quads * (thread + 1) / runtime::kHvxThreads;
    NormVtcm::Thread * buffers = &w->norm_vtcm->threads[thread];
    runtime::DmaRing * ring = w->rings[thread].get();

    if (first_quad < end_quad) push_quad(w, ring, buffers, first_quad);
    for (int quad = first_quad; quad < end_quad; quad++) {
        if (quad + 1 < end_quad) push_quad(w, ring, buffers, quad + 1);
        int first_token;
        const int rows = quad_rows(w, quad, &first_token);
        if (rows) ring->pop();
        normalize_quad(w, quad, rows, buffers->quads[quad & 1], buffers->normalized);
    }
}

runtime::Status normalize(runtime::Session & session, Workspace * w, const float * gamma, const float * beta,
                          float epsilon, const int * rows, int layout_rows) {
    w->norm_beta = beta;
    w->norm_epsilon = epsilon;
    w->norm_rows = rows;
    w->norm_layout_rows = layout_rows;
    w->norm_vtcm = reinterpret_cast<NormVtcm *>(session.vtcm());
    w->int8_scales = runtime::int8_scales(session);

    for (int thread = 0; thread < runtime::kHvxThreads; thread++)
        w->rings[thread]->initialize(runtime::DmaBypass::kBoth);
    for (int i = 0; i < kHidden / 32; i++) w->norm_vtcm->gamma[i] = hvx::vectors(gamma)[i];
    w->norm_gamma_vtcm = w->norm_vtcm->gamma;

    // HVX stores leave x in the cache; the DMA reads DDR
    runtime::flush_data_cache();
    return runtime::run_parallel(session, runtime::kHvxThreads, normalize_thread, w);
}

struct PatchNormJob {
    Workspace * workspace;
    const float * gamma;
    const float * beta;
};

void normalize_patch_token(void * context, int token, int) {
    const auto * job = static_cast<const PatchNormJob *>(context);
    const Workspace * w = job->workspace;
    if (token % w->tokens_per_frame < w->special_tokens) return;

    auto * x = hvx::vectors(w->x.get() + static_cast<size_t>(token) * kHidden);
    Vector<float> sum = Vector<float>::zero();
    Vector<float> square = Vector<float>::zero();
    for (int k = 0; k < kHidden / 32; k++) {
        sum = sum + x[k];
        square = square + x[k] * x[k];
    }

    const float mean = hvx::horizontal_sum(sum) * (1.0f / kHidden);
    const float variance = hvx::horizontal_sum(square) * (1.0f / kHidden) - mean * mean;
    const Vector<float> inverse =
        Vector<float>::splat(hvx::reciprocal_square_root((variance > 0 ? variance : 0) + 1e-6f));
    const Vector<float> negative = Vector<float>::splat(-mean);

    const auto * gamma = hvx::vectors(job->gamma);
    const auto * beta = hvx::vectors(job->beta);
    for (int k = 0; k < kHidden / 32; k++) x[k] = (x[k] + negative) * inverse * gamma[k] + beta[k];
}

struct EmbeddingJob {
    float * x;
    const float * positions;
    const float * special_tokens;
    int tokens_per_frame;
    int special_token_count;
};

void finish_input_token(void * context, int token, int) {
    const auto * job = static_cast<const EmbeddingJob *>(context);
    const int index = token % job->tokens_per_frame;
    auto * x = hvx::vectors(job->x + static_cast<size_t>(token) * kHidden);
    if (index < job->special_token_count) {
        const auto * source = hvx::vectors(job->special_tokens + static_cast<size_t>(index) * kHidden);
        for (int k = 0; k < kHidden / 32; k++) x[k] = source[k];
        return;
    }

    const auto * position =
        hvx::vectors(job->positions + static_cast<size_t>(index - job->special_token_count) * kHidden);
    for (int k = 0; k < kHidden / 32; k++) x[k] = x[k] + position[k];
}

void set_rope_position(runtime::HeapArray<int16_t> & positions, int row, int token, int special_tokens,
                       int patch_columns) {
    const int patch = token - special_tokens;
    positions[2 * row] = static_cast<int16_t>(patch < 0 ? 0 : patch / patch_columns + 1);
    positions[2 * row + 1] = static_cast<int16_t>(patch < 0 ? 0 : patch % patch_columns + 1);
}

}  // namespace

runtime::Status allocate_workspace(Workspace * w, int tokens, int rows) {
    if (w->token_capacity >= tokens && w->row_capacity >= rows) return {};

    *w = Workspace{};
    const size_t rows64 = static_cast<size_t>(rows + 63) / 64 * 64;
    const size_t tile_bytes = static_cast<size_t>(kHeads) * (rows / 32) * (kHeadDimensions / 32) * 2048;

    w->x = runtime::allocate_array<float>(static_cast<size_t>(tokens) * kHidden);
    w->input_tiles = runtime::allocate_array<uint8_t>(rows64 * kHidden, 4096);
    w->input_row_scales = runtime::allocate_array<float>(rows64);
    w->query_tiles = runtime::allocate_array<uint8_t>(tile_bytes, 4096);
    w->key_tiles = runtime::allocate_array<uint8_t>(tile_bytes, 4096);
    w->value_tiles = runtime::allocate_array<uint8_t>(tile_bytes, 4096);
    w->attention_output = runtime::allocate_array<_Float16>(static_cast<size_t>(rows) * kHidden);
    w->handover_tiles = runtime::allocate_array<_Float16>(rows64 * kMlpHidden);
    w->handover_row_scales = runtime::allocate_array<float>(rows64);
    w->handover_inverse_bits = runtime::allocate_array<uint16_t>(rows64);
    w->handover = {w->handover_tiles.get(), w->handover_row_scales.get(), w->handover_inverse_bits.get()};
    w->frame_rows = runtime::allocate_array<int>(rows);
    w->global_rows = runtime::allocate_array<int>(rows);
    w->token_frame_rows = runtime::allocate_array<int>(tokens);
    w->frame_positions = runtime::allocate_array<int16_t>(static_cast<size_t>(rows) * 2);
    w->global_positions = runtime::allocate_array<int16_t>(static_cast<size_t>(rows) * 2);
    w->rope_table = runtime::allocate_array<float>(static_cast<size_t>(kRopeTablePositions) * kRopeTableValues);

    bool allocated = w->x && w->input_tiles && w->input_row_scales && w->query_tiles && w->key_tiles &&
                     w->value_tiles && w->attention_output && w->handover_tiles && w->handover_row_scales &&
                     w->handover_inverse_bits && w->frame_rows && w->global_rows && w->token_frame_rows &&
                     w->frame_positions && w->global_positions && w->rope_table;
    for (runtime::HeapObject<runtime::DmaRing> & ring : w->rings) {
        ring = runtime::allocate_object<runtime::DmaRing>(64);
        allocated = allocated && ring;
    }
    if (!allocated) {
        *w = Workspace{};
        return std::unexpected(runtime::Error::kNoMemory);
    }

    w->token_capacity = tokens;
    w->row_capacity = rows;
    return {};
}

runtime::Status set_geometry(Workspace * w, int frames, int tokens_per_frame, int patch_rows, int patch_columns,
                             int special_tokens) {
    if (frames < 1 || frames > kMaximumFrames || special_tokens < 0 ||
        tokens_per_frame != special_tokens + patch_rows * patch_columns || patch_rows < 1 || patch_columns < 1 ||
        patch_rows > 126 || patch_columns > 126) {
        return std::unexpected(runtime::Error::kBadParameter);
    }

    w->special_tokens = special_tokens;
    w->frames = frames;
    w->tokens_per_frame = tokens_per_frame;
    w->patch_rows = patch_rows;
    w->patch_columns = patch_columns;
    w->tokens = frames * tokens_per_frame;
    w->padded_frame_tokens = (tokens_per_frame + 31) / 32 * 32;
    w->frame_layout_rows = frames * w->padded_frame_tokens;
    w->global_layout_rows = (w->tokens + 31) / 32 * 32;

    for (int row = 0; row < w->global_layout_rows; row++) w->global_rows[row] = row < w->tokens ? row : -1;
    for (int frame = 0; frame < frames; frame++) {
        w->frame_first_tokens[frame] = frame * w->padded_frame_tokens;
        w->frame_token_counts[frame] = tokens_per_frame;
        for (int row = 0; row < w->padded_frame_tokens; row++) {
            w->frame_rows[frame * w->padded_frame_tokens + row] =
                row < tokens_per_frame ? frame * tokens_per_frame + row : -1;
        }
        for (int token = 0; token < tokens_per_frame; token++) {
            w->token_frame_rows[frame * tokens_per_frame + token] = frame * w->padded_frame_tokens + token;
        }
    }

    for (int row = 0; row < w->frame_layout_rows; row++) {
        const int token = w->frame_rows[row] < 0 ? 0 : w->frame_rows[row] % tokens_per_frame;
        set_rope_position(w->frame_positions, row, token, special_tokens, patch_columns);
    }
    for (int row = 0; row < w->global_layout_rows; row++) {
        const int token = row < w->tokens ? row % tokens_per_frame : 0;
        set_rope_position(w->global_positions, row, token, special_tokens, patch_columns);
    }

    w->rope_positions = (patch_rows > patch_columns ? patch_rows : patch_columns) + 1;
    for (int position = 0; position < w->rope_positions; position++) {
        float * cosines = w->rope_table.get() + static_cast<size_t>(position) * kRopeTableValues;
        float * sines = cosines + 32;
        for (int i = 0; i < 16; i++) {
            const float frequency = 1.0f / powf(100.0f, static_cast<float>(2 * i) / 32.0f);
            const float angle = static_cast<float>(position) * frequency;
            cosines[i] = cosf(angle);
            cosines[i + 16] = cosines[i];
            sines[i] = -sinf(angle);
            sines[i + 16] = sinf(angle);
        }
    }
    return {};
}

runtime::Status run_block(runtime::Session & session, const LayerWeights & layer, BlockKind kind, Workspace * w,
                          int32_t * statistics) {
    const bool global = kind == BlockKind::kGlobal;
    const bool aggregator = kind != BlockKind::kDinov2;
    const int layout_rows = global ? w->global_layout_rows : w->frame_layout_rows;
    const float epsilon = aggregator ? 1e-5f : 1e-6f;
    runtime::StageTimer timer(statistics);
    kernels::gemm_w8a8::Input prequantized;
    prequantized.tiles = w->input_tiles.get();
    prequantized.tile_row_scales = w->input_row_scales.get();

    runtime::Status status =
        normalize(session, w, layer.part<float>(block_part::kNorm1Gamma), layer.part<float>(block_part::kNorm1Beta),
                  epsilon, global ? w->global_rows.get() : w->frame_rows.get(), layout_rows);
    if (!status) return status;
    timer.lap(block_stat::kNorm1Microseconds);

    {
        kernels::gemm_w8a8::Output output;
        output.bias = layer.part<float>(block_part::kQkvBias);
        output.attention_q = w->query_tiles.get();
        output.attention_k = w->key_tiles.get();
        output.attention_v = w->value_tiles.get();
        output.attention_group_blocks = kHidden / 32;
        output.attention_key_tile_stride = layout_rows / 32;
        output.qk_norm_parameters = aggregator ? layer.part<float>(block_part::kQkNorm) : nullptr;
        output.qk_norm_epsilon = 1e-5f;
        output.rope_positions = aggregator ? (global ? w->global_positions.get() : w->frame_positions.get()) : nullptr;
        output.rope_table = aggregator ? w->rope_table.get() : nullptr;
        output.rope_table_positions = w->rope_positions;
        prequantized.form = InputForm::kPrequantizedTiles;
        output.form = OutputForm::kAttentionTiles64;
        status = gemm::run(session, {.M = layout_rows,
                                     .K = kHidden,
                                     .N = kQkvColumns,
                                     .weights = layer.gemm(block_part::kQkvWeights, block_part::kQkvScales),
                                     .input = prequantized,
                                     .output = output});
        if (!status) return status;
    }
    timer.lap(block_stat::kQkvMicroseconds);

    {
        const int global_first_token = 0;
        const int global_tokens = w->tokens;
        const kernels::flash_attention::Segments segments{global ? 1 : w->frames,
                                                          global ? &global_first_token : w->frame_first_tokens,
                                                          global ? &global_tokens : w->frame_token_counts, nullptr};
        namespace attention = kernels::flash_attention;
        status = attention::run(session, {.source = attention::Source::kTiles,
                                          .queries = layout_rows,
                                          .keys = layout_rows,
                                          .query_heads = kHeads,
                                          .kv_heads = kHeads,
                                          .head_dimensions = kHeadDimensions,
                                          .segments = &segments,
                                          .o = w->attention_output.get(),
                                          .tiles = {w->key_tiles.get(), w->value_tiles.get(), layout_rows / 32},
                                          .query_tiles = w->query_tiles.get()});
        if (!status) return status;
    }
    timer.lap(block_stat::kAttentionMicroseconds);

    {
        kernels::gemm_w8a8::Input input;
        input.rows_f16 = w->attention_output.get();
        input.row_map = global ? nullptr : w->token_frame_rows.get();
        input.column_scale = layer.part<float>(block_part::kOutputInverseScale);
        input.row_stride = kHidden;
        kernels::gemm_w8a8::Output output;
        output.rows_f32 = w->x.get();
        output.bias = layer.part<float>(block_part::kOutputBias);
        input.form = InputForm::kRowsF16Scaled;
        output.form = OutputForm::kRowsF32Accumulate;
        status = gemm::run(session, {.M = w->tokens,
                                     .K = kHidden,
                                     .N = kHidden,
                                     .weights = layer.gemm(block_part::kOutputWeights, block_part::kOutputScales),
                                     .input = input,
                                     .output = output});
        if (!status) return status;
    }
    timer.lap(block_stat::kOutputMicroseconds);

    status = normalize(session, w, layer.part<float>(block_part::kNorm2Gamma),
                       layer.part<float>(block_part::kNorm2Beta), epsilon, nullptr, (w->tokens + 3) / 4 * 4);
    if (!status) return status;
    timer.lap(block_stat::kNorm2Microseconds);

    {
        kernels::gemm_w8a8::Output output;
        output.bias = layer.part<float>(block_part::kFc1Bias);
        output.column_scale = layer.part<float>(block_part::kFc2InverseScale);
        output.handover = &w->handover;
        prequantized.form = InputForm::kPrequantizedTiles;
        output.form = OutputForm::kHandoverTiles;
        output.activation = Activation::kGeluErf;
        status = gemm::run(session, {.M = w->tokens,
                                     .K = kHidden,
                                     .N = kMlpHidden,
                                     .weights = layer.gemm(block_part::kFc1Weights, block_part::kFc1Scales),
                                     .input = prequantized,
                                     .output = output});
        if (!status) return status;
    }
    timer.lap(block_stat::kFc1Microseconds);

    {
        kernels::gemm_w8a8::Input input;
        input.handover = &w->handover;
        kernels::gemm_w8a8::Output output;
        output.rows_f32 = w->x.get();
        output.bias = layer.part<float>(block_part::kFc2Bias);
        input.form = InputForm::kHandoverTiles;
        output.form = OutputForm::kRowsF32Accumulate;
        status = gemm::run(session, {.M = w->tokens,
                                     .K = kMlpHidden,
                                     .N = kHidden,
                                     .weights = layer.gemm(block_part::kFc2Weights, block_part::kFc2Scales),
                                     .input = input,
                                     .output = output});
        if (!status) return status;
    }
    timer.lap(block_stat::kFc2Microseconds);
    return {};
}

runtime::Status embed(runtime::Session & session, Workspace * w, const uint8_t * embedding_weights,
                      const _Float16 * patches, int patch_inputs, const float * positions,
                      const float * special_tokens) {
    const int patch_count = w->patch_rows * w->patch_columns;
    const runtime::HeapArray<int> output_rows =
        runtime::allocate_array<int>(static_cast<size_t>(w->frames) * patch_count);
    if (!output_rows) return std::unexpected(runtime::Error::kNoMemory);
    for (int frame = 0; frame < w->frames; frame++) {
        for (int i = 0; i < patch_count; i++)
            output_rows[frame * patch_count + i] = frame * w->tokens_per_frame + w->special_tokens + i;
    }

    kernels::gemm_fp16::Arguments embedding;
    embedding.input_rows = patches;
    embedding.row_stride = patch_inputs;
    embedding.taps = 1;
    embedding.weights = embedding_weights;
    embedding.M = w->frames * patch_count;
    embedding.K = patch_inputs;
    embedding.N = kHidden;
    embedding.output_map = output_rows.get();
    embedding.output_form = kernels::gemm_fp16::OutputForm::kRowsF32;
    embedding.rows_f32 = w->x.get();
    if (const runtime::Status embedded = kernels::gemm_fp16::run(session, embedding); !embedded) return embedded;

    EmbeddingJob embedding_job{w->x.get(), positions, special_tokens, w->tokens_per_frame, w->special_tokens};
    return runtime::parallel_for(session, runtime::kHvxThreads, w->tokens, finish_input_token, &embedding_job);
}

runtime::Status normalize_patch_tokens(runtime::Session & session, Workspace * w, const float * gamma,
                                       const float * beta) {
    PatchNormJob job{w, gamma, beta};
    return runtime::parallel_for(session, runtime::kHvxThreads, w->tokens, normalize_patch_token, &job);
}

}  // namespace benzene::models::vision_transformer
