#include "dsp/models/shared/dpt_head.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "dsp/hvx/prefetch.h"
#include "dsp/hvx/reductions.h"
#include "dsp/hvx/scalar_math.h"
#include "dsp/hvx/vector.h"
#include "dsp/runtime/stage_timer.h"
#include "dsp/runtime/thread_pool.h"

namespace benzene::models::dpt_head {
namespace {

namespace convolution = protocol::dpt_head::convolution;
namespace convolution_part = protocol::dpt_head::convolution_part;
namespace dpt_stat = protocol::dpt_head::stat;
namespace gemm_fp16 = kernels::gemm_fp16;
using hvx::Vector;
using hvx::VectorPair;

constexpr int kProjectChannels[4] = {256, 512, 1024, 1024};

// The session's workspace (Session::state's key).
struct SessionWorkspace {
    Workspace workspace;
};

runtime::HeapArray<int> make_tap_map(int input_rows, int input_columns, int stride, int output_rows,
                                     int output_columns) {
    runtime::HeapArray<int> map = runtime::allocate_array<int>(static_cast<size_t>(output_rows) * output_columns * 9);
    if (!map) return map;

    for (int oy = 0; oy < output_rows; oy++) {
        for (int ox = 0; ox < output_columns; ox++) {
            for (int tap = 0; tap < 9; tap++) {
                const int iy = oy * stride - 1 + tap / 3;
                const int ix = ox * stride - 1 + tap % 3;
                map[(static_cast<size_t>(oy) * output_columns + ox) * 9 + tap] =
                    (iy >= 0 && iy < input_rows && ix >= 0 && ix < input_columns) ? iy * input_columns + ix : -1;
            }
        }
    }
    return map;
}

struct NormJob {
    const _Float16 * source;
    const float * gamma;
    const float * beta;
    _Float16 * destination;
    int first_row;
};

void normalize_row(void * context, int row, int) {
    const auto * job = static_cast<const NormJob *>(context);
    const auto * x = hvx::vectors(job->source + static_cast<size_t>(job->first_row + row) * 2048);
    hvx::prefetch_to_l2(x + 32, 4096, 4096, 1);

    Vector<float> sum = Vector<float>::zero();
    Vector<float> square = Vector<float>::zero();
    for (int k = 0; k < 32; k++) {
        const VectorPair<float> values = to_float32(x[k]);
        sum = sum + values.low() + values.high();
        square = square + values.low() * values.low() + values.high() * values.high();
    }

    const float mean = hvx::horizontal_sum(sum) * (1.0f / 2048);
    const float variance = hvx::horizontal_sum(square) * (1.0f / 2048) - mean * mean;
    const Vector<float> inverse =
        Vector<float>::splat(hvx::reciprocal_square_root((variance > 0 ? variance : 0) + 1e-5f));
    const Vector<float> negative = Vector<float>::splat(-mean);

    const auto * gamma = hvx::vectors(job->gamma);
    const auto * beta = hvx::vectors(job->beta);
    auto * out = hvx::vectors(job->destination + static_cast<size_t>(row) * 2048);
    for (int k = 0; k < 32; k++) {
        const VectorPair<float> values = to_float32(x[k]);
        out[k] = to_float16((values.low() + negative) * inverse * gamma[2 * k] + beta[2 * k],
                            (values.high() + negative) * inverse * gamma[2 * k + 1] + beta[2 * k + 1]);
    }
}

struct ResampleJob {
    const _Float16 * source;
    _Float16 * destination;
    int input_rows;
    int input_columns;
    int output_rows;
    int output_columns;
    int channels;
    const float * column_positions;  // final resampling: + position embedding (channels < 64 by column, >= 64 by row)
    const float * row_positions;
    bool accumulate;  // + the destination's values (before the position embedding)
};

// Bilinear, align_corners=True.
void resample_row(void * context, int oy, int) {
    const auto * job = static_cast<const ResampleJob *>(context);
    const float fy = job->output_rows > 1 ? static_cast<float>(oy) * static_cast<float>(job->input_rows - 1) /
                                                static_cast<float>(job->output_rows - 1)
                                          : 0.0f;
    const int y0 = static_cast<int>(fy);
    const int y1 = y0 + 1 < job->input_rows ? y0 + 1 : y0;
    const float ly = fy - static_cast<float>(y0);
    const int vectors = job->channels / 64;
    hvx::prefetch_to_l2(job->source + static_cast<size_t>(y0) * job->input_columns * job->channels,
                        static_cast<uint32_t>(job->input_columns) * job->channels * 2,
                        static_cast<uint32_t>(y1 - y0) * job->input_columns * job->channels * 2, 2);

    for (int ox = 0; ox < job->output_columns; ox++) {
        const float fx = job->output_columns > 1 ? static_cast<float>(ox) * static_cast<float>(job->input_columns - 1) /
                                                       static_cast<float>(job->output_columns - 1)
                                                 : 0.0f;
        const int x0 = static_cast<int>(fx);
        const int x1 = x0 + 1 < job->input_columns ? x0 + 1 : x0;
        const float lx = fx - static_cast<float>(x0);
        const Vector<float> w00 = Vector<float>::splat((1 - ly) * (1 - lx));
        const Vector<float> w01 = Vector<float>::splat((1 - ly) * lx);
        const Vector<float> w10 = Vector<float>::splat(ly * (1 - lx));
        const Vector<float> w11 = Vector<float>::splat(ly * lx);

        const auto * a =
            hvx::vectors(job->source + (static_cast<size_t>(y0) * job->input_columns + x0) * job->channels);
        const auto * b =
            hvx::vectors(job->source + (static_cast<size_t>(y0) * job->input_columns + x1) * job->channels);
        const auto * c =
            hvx::vectors(job->source + (static_cast<size_t>(y1) * job->input_columns + x0) * job->channels);
        const auto * d =
            hvx::vectors(job->source + (static_cast<size_t>(y1) * job->input_columns + x1) * job->channels);
        auto * out =
            hvx::vectors(job->destination + (static_cast<size_t>(oy) * job->output_columns + ox) * job->channels);

        for (int k = 0; k < vectors; k++) {
            const VectorPair<float> ea = to_float32(a[k]);
            const VectorPair<float> eb = to_float32(b[k]);
            const VectorPair<float> ec = to_float32(c[k]);
            const VectorPair<float> ed = to_float32(d[k]);
            Vector<float> low = (ea.low() * w00 + eb.low() * w01) + (ec.low() * w10 + ed.low() * w11);
            Vector<float> high = (ea.high() * w00 + eb.high() * w01) + (ec.high() * w10 + ed.high() * w11);
            if (job->accumulate) {
                const VectorPair<float> existing = to_float32(out[k]);
                low = low + existing.low();
                high = high + existing.high();
            }
            if (job->column_positions) {
                const float * table = k == 0 ? job->column_positions + static_cast<size_t>(ox) * 64
                                             : job->row_positions + static_cast<size_t>(oy) * 64;
                low = low + hvx::vectors(table)[0];
                high = high + hvx::vectors(table)[1];
            }
            out[k] = to_float16(low, high);
        }
    }
}

runtime::Status resample(runtime::Session & session, const _Float16 * source, int input_rows, int input_columns,
                         _Float16 * destination, int output_rows, int output_columns, int channels,
                         const float * column_positions, const float * row_positions, bool accumulate = false) {
    ResampleJob job{source,         destination, input_rows,       input_columns, output_rows,
                    output_columns, channels,    column_positions, row_positions, accumulate};
    return runtime::parallel_for(session, runtime::kHvxThreads, output_rows, resample_row, &job);
}

// conv2(relu(conv1(relu(x)))) + relu(x) [+ skip]: DPT's inplace ReLU also rectifies the residual.
runtime::Status residual_conv_unit(runtime::Session & session, const HeadWeights & head, int conv1, const _Float16 * x,
                                   int pixels, int tap_width, const _Float16 * skip, _Float16 * temporary,
                                   _Float16 * out) {
    const runtime::Status first =
        convolve(session, head, conv1, x, kFeatures, nullptr, 9, 1, pixels, 9 * kFeatures, kFeatures,
                 f16_output(temporary, kFeatures, nullptr, 0, nullptr, 1), tap_width);
    if (!first) return first;
    return convolve(session, head, conv1 + 1, temporary, kFeatures, nullptr, 9, 0, pixels, 9 * kFeatures, kFeatures,
                    f16_output(out, kFeatures, x, 1, skip, 0), tap_width);
}

runtime::Status allocate_workspace(Workspace * w, int patch_rows, int patch_columns, int patch_size,
                                   const float * position_tables) {
    *w = Workspace{};
    w->patch_rows = patch_rows;
    w->patch_columns = patch_columns;
    w->height = patch_rows * patch_size;
    w->width = patch_columns * patch_size;
    w->stride2_rows = (patch_rows - 1) / 2 + 1;
    w->stride2_columns = (patch_columns - 1) / 2 + 1;
    const size_t patches = static_cast<size_t>(patch_rows) * patch_columns;
    const size_t H = w->height;
    const size_t W = w->width;

    bool allocated = runtime::allocate(w->normalized, patches * 2048);
    for (int k = 0; k < 4; k++) allocated &= runtime::allocate(w->project_positions[k], patches * kProjectChannels[k]);
    for (int k = 0; k < 4; k++) allocated &= runtime::allocate(w->projected[k], patches * kProjectChannels[k]);
    allocated &= runtime::allocate(w->layer1, 16 * patches * 256);
    allocated &= runtime::allocate(w->layer2, 4 * patches * 512);
    allocated &= runtime::allocate(w->layer4, static_cast<size_t>(w->stride2_rows) * w->stride2_columns * 1024);
    allocated &= runtime::allocate(w->fusion_inputs[0], 16 * patches * kFeatures);
    allocated &= runtime::allocate(w->fusion_inputs[1], 4 * patches * kFeatures);
    allocated &= runtime::allocate(w->fusion_inputs[2], patches * kFeatures);
    allocated &=
        runtime::allocate(w->fusion_inputs[3], static_cast<size_t>(w->stride2_rows) * w->stride2_columns * kFeatures);
    allocated &= runtime::allocate(w->temporary, 16 * patches * kFeatures);
    allocated &= runtime::allocate(w->refined, 16 * patches * kFeatures);
    allocated &= runtime::allocate(w->fused, 16 * patches * kFeatures);
    allocated &= runtime::allocate(w->upsampled, 64 * patches * kFeatures);
    allocated &= runtime::allocate(w->fusion_output, 64 * patches * kFeatures);
    allocated &= runtime::allocate(w->head_features, 64 * patches * kHeadFeatures);
    allocated &= runtime::allocate(w->image_features, H * W * kHeadFeatures);
    allocated &= runtime::allocate(w->column_positions, W * 64);
    allocated &= runtime::allocate(w->row_positions, H * 64);
    allocated &= runtime::allocate(w->transposed_rows[0], patches);
    allocated &= runtime::allocate(w->transposed_rows[1], patches);
    w->stride2_taps = make_tap_map(patch_rows, patch_columns, 2, w->stride2_rows, w->stride2_columns);
    if (!allocated || !w->stride2_taps) {
        *w = Workspace{};
        return std::unexpected(runtime::Error::kNoMemory);
    }

    for (int i = 0; i < patch_rows; i++) {  // transposed convolutions: input pixel -> its tap (0, 0) output row
        for (int j = 0; j < patch_columns; j++) {
            w->transposed_rows[0][i * patch_columns + j] = 4 * i * 4 * patch_columns + 4 * j;
            w->transposed_rows[1][i * patch_columns + j] = 2 * i * 2 * patch_columns + 2 * j;
        }
    }

    const float * table = position_tables;
    for (int k = 0; k < 4; k++) {
        const int C = kProjectChannels[k];
        const float * columns = table;
        const float * rows = table + static_cast<size_t>(patch_columns) * C / 2;
        for (int y = 0; y < patch_rows; y++) {
            for (int x = 0; x < patch_columns; x++) {
                _Float16 * destination =
                    w->project_positions[k].get() + (static_cast<size_t>(y) * patch_columns + x) * C;
                for (int c = 0; c < C / 2; c++) {
                    destination[c] = static_cast<_Float16>(columns[static_cast<size_t>(x) * C / 2 + c]);
                    destination[C / 2 + c] = static_cast<_Float16>(rows[static_cast<size_t>(y) * C / 2 + c]);
                }
            }
        }
        table += static_cast<size_t>(patch_columns + patch_rows) * C / 2;
    }
    memcpy(w->column_positions.get(), table, W * 64 * 4);
    memcpy(w->row_positions.get(), table + W * 64, H * 64 * 4);
    return {};
}

}  // namespace

runtime::Status load_head(const runtime::Session & session, const int32_t * references, HeadWeights * head) {
    for (int i = 0; i <= convolution::kCount; i++) {
        const runtime::Result<LayerWeights> part =
            layer_in_model(session, references + i * protocol::blob_references(convolution_part::kCount));
        if (!part) return std::unexpected(part.error());
        head->parts[i] = *part;
    }
    return {};
}

int position_table_values(int patch_rows, int patch_columns, int patch_size) {
    return (patch_columns + patch_rows) * (128 + 256 + 512 + 512) +
           (patch_size * patch_columns + patch_size * patch_rows) * 64;
}

runtime::Result<Workspace *> session_workspace(runtime::Session & session, int patch_rows, int patch_columns,
                                               int patch_size, const float * position_tables) {
    SessionWorkspace * state = session.state<SessionWorkspace>();
    if (state == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    Workspace * w = &state->workspace;
    if (!w->normalized || w->patch_rows != patch_rows || w->patch_columns != patch_columns ||
        w->height != patch_rows * patch_size || w->width != patch_columns * patch_size) {
        const runtime::Status allocated = allocate_workspace(w, patch_rows, patch_columns, patch_size, position_tables);
        if (!allocated) return std::unexpected(allocated.error());
    }
    return w;
}

runtime::Status convolve(runtime::Session & session, const uint8_t * weights, const _Float16 * input, int row_stride,
                         const int * input_map, int taps, bool relu_input, int M, int K, int N,
                         gemm_fp16::Arguments output, int tap_width) {
    output.input_rows = input;
    output.row_stride = row_stride;
    output.input_map = input_map;
    output.taps = taps;
    output.relu_input = relu_input;
    output.weights = weights;
    output.tap_width = tap_width;
    output.M = M;
    output.K = K;
    output.N = N;
    return kernels::gemm_fp16::run(session, output);
}

runtime::Status convolve(runtime::Session & session, const HeadWeights & head, int which, const _Float16 * input,
                         int row_stride, const int * input_map, int taps, bool relu_input, int M, int K, int N,
                         gemm_fp16::Arguments output, int tap_width) {
    return convolve(session, head.parts[which].part<uint8_t>(convolution_part::kWeights), input, row_stride, input_map,
                    taps, relu_input, M, K, N, output, tap_width);
}

gemm_fp16::Arguments f16_output(_Float16 * out, int channels, const _Float16 * residual, bool relu_residual,
                                const _Float16 * second_residual, bool relu) {
    gemm_fp16::Arguments output;
    output.f16_groups[0] = out;
    output.f16_group_width = channels;
    output.residual = residual;
    output.relu_residual = relu_residual;
    output.second_residual = second_residual;
    output.relu_output = relu;
    return output;
}

runtime::Status run_trunk(runtime::Session & session, const HeadWeights & head, Workspace * w,
                          const _Float16 * const kept[kKeptOutputs], int first_token, const float * norm_gamma,
                          const float * norm_beta, int32_t * statistics) {
    const int patches = w->patch_rows * w->patch_columns;
    const int stride2_pixels = w->stride2_rows * w->stride2_columns;
    runtime::Status status;
    runtime::StageTimer timer(statistics);

    for (int k = 0; k < kKeptOutputs; k++) {
        NormJob norm_job{kept[k], norm_gamma, norm_beta, w->normalized.get(), first_token};
        status = runtime::parallel_for(session, runtime::kHvxThreads, patches, normalize_row, &norm_job);
        if (!status) return status;
        timer.lap(dpt_stat::kElementwiseMicroseconds);
        status = convolve(
            session, head, convolution::kProject0 + k, w->normalized.get(), 2048, nullptr, 1, 0, patches, 2048,
            kProjectChannels[k],
            f16_output(w->projected[k].get(), kProjectChannels[k], nullptr, 0, w->project_positions[k].get(), 0), 0);
        if (!status) return status;
        timer.lap(dpt_stat::kProjectsMicroseconds);
    }

    {
        gemm_fp16::Arguments resize0;
        resize0.f16_group_width = 256;
        resize0.output_map = w->transposed_rows[0].get();
        for (int t = 0; t < 16; t++)
            resize0.f16_groups[t] = w->layer1.get() + (static_cast<size_t>(t / 4) * 4 * w->patch_columns + t % 4) * 256;
        gemm_fp16::Arguments resize1;
        resize1.f16_group_width = 512;
        resize1.output_map = w->transposed_rows[1].get();
        for (int t = 0; t < 4; t++)
            resize1.f16_groups[t] = w->layer2.get() + (static_cast<size_t>(t / 2) * 2 * w->patch_columns + t % 2) * 512;
        status = convolve(session, head, convolution::kResize0, w->projected[0].get(), 256, nullptr, 1, 0, patches, 256,
                          16 * 256, resize0, 0);
        if (!status) return status;
        status = convolve(session, head, convolution::kResize1, w->projected[1].get(), 512, nullptr, 1, 0, patches, 512,
                          4 * 512, resize1, 0);
        if (!status) return status;
        status =
            convolve(session, head, convolution::kResize3, w->projected[3].get(), 1024, w->stride2_taps.get(), 9, 0,
                     stride2_pixels, 9 * 1024, 1024, f16_output(w->layer4.get(), 1024, nullptr, 0, nullptr, 0), 0);
        if (!status) return status;
    }
    timer.lap(dpt_stat::kProjectsMicroseconds);

    status = convolve(session, head, convolution::kLayer1Rn, w->layer1.get(), 256, nullptr, 9, 0, 16 * patches, 9 * 256,
                      kFeatures, f16_output(w->fusion_inputs[0].get(), kFeatures, nullptr, 0, nullptr, 0),
                      4 * w->patch_columns);
    if (!status) return status;
    status = convolve(session, head, convolution::kLayer2Rn, w->layer2.get(), 512, nullptr, 9, 0, 4 * patches, 9 * 512,
                      kFeatures, f16_output(w->fusion_inputs[1].get(), kFeatures, nullptr, 0, nullptr, 0),
                      2 * w->patch_columns);
    if (!status) return status;
    status =
        convolve(session, head, convolution::kLayer3Rn, w->projected[2].get(), 1024, nullptr, 9, 0, patches, 9 * 1024,
                 kFeatures, f16_output(w->fusion_inputs[2].get(), kFeatures, nullptr, 0, nullptr, 0), w->patch_columns);
    if (!status) return status;
    status = convolve(session, head, convolution::kLayer4Rn, w->layer4.get(), 1024, nullptr, 9, 0, stride2_pixels,
                      9 * 1024, kFeatures, f16_output(w->fusion_inputs[3].get(), kFeatures, nullptr, 0, nullptr, 0),
                      w->stride2_columns);
    if (!status) return status;
    timer.lap(dpt_stat::kLayerRnMicroseconds);

    status = residual_conv_unit(session, head, convolution::kRefine4Unit2Conv1, w->fusion_inputs[3].get(),
                                stride2_pixels, w->stride2_columns, nullptr, w->temporary.get(), w->fused.get());
    if (!status) return status;
    status = resample(session, w->fused.get(), w->stride2_rows, w->stride2_columns, w->upsampled.get(), w->patch_rows,
                      w->patch_columns, kFeatures, nullptr, nullptr);
    if (!status) return status;
    status = convolve(session, head, convolution::kRefine4Output, w->upsampled.get(), kFeatures, nullptr, 1, 0, patches,
                      kFeatures, kFeatures, f16_output(w->fusion_output.get(), kFeatures, nullptr, 0, nullptr, 0), 0);
    if (!status) return status;

    constexpr int kRefineFirstConvolution[3] = {convolution::kRefine3Unit1Conv1, convolution::kRefine2Unit1Conv1,
                                                convolution::kRefine1Unit1Conv1};
    const _Float16 * layer_inputs[3] = {w->fusion_inputs[2].get(), w->fusion_inputs[1].get(),
                                        w->fusion_inputs[0].get()};
    constexpr int kScales[3] = {1, 2, 4};
    for (int level = 0; level < 3; level++) {
        const int rows = w->patch_rows * kScales[level];
        const int columns = w->patch_columns * kScales[level];
        const int pixels = rows * columns;
        const int output_rows = 2 * rows;
        const int output_columns = 2 * columns;
        const int first = kRefineFirstConvolution[level];

        status = residual_conv_unit(session, head, first, layer_inputs[level], pixels, columns, w->fusion_output.get(),
                                    w->temporary.get(), w->fused.get());
        if (!status) return status;

        status = residual_conv_unit(session, head, first + 2, w->fused.get(), pixels, columns, nullptr,
                                    w->temporary.get(), w->refined.get());
        if (!status) return status;

        status = resample(session, w->refined.get(), rows, columns, w->upsampled.get(), output_rows, output_columns,
                          kFeatures, nullptr, nullptr);
        if (!status) return status;
        status = convolve(session, head, first + 4, w->upsampled.get(), kFeatures, nullptr, 1, 0,
                          output_rows * output_columns, kFeatures, kFeatures,
                          f16_output(w->fusion_output.get(), kFeatures, nullptr, 0, nullptr, 0), 0);
        if (!status) return status;
    }
    timer.lap(dpt_stat::kRefineNetsMicroseconds);

    status = convolve(session, head, convolution::kOutputConv1, w->fusion_output.get(), kFeatures, nullptr, 9, 0,
                      64 * patches, 9 * kFeatures, kHeadFeatures,
                      f16_output(w->head_features.get(), kHeadFeatures, nullptr, 0, nullptr, 0), 8 * w->patch_columns);
    if (!status) return status;
    timer.lap(dpt_stat::kOutputConvolutionsMicroseconds);
    return {};
}

runtime::Status upsample_to_image(runtime::Session & session, Workspace * w, bool accumulate) {
    return resample(session, w->head_features.get(), 8 * w->patch_rows, 8 * w->patch_columns, w->image_features.get(),
                    w->height, w->width, kHeadFeatures, w->column_positions.get(), w->row_positions.get(), accumulate);
}

}  // namespace benzene::models::dpt_head
