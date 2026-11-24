#include "dsp/models/vggt/vggt.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/prefetch.h"
#include "dsp/hvx/reductions.h"
#include "dsp/hvx/scalar_math.h"
#include "dsp/hvx/vector.h"
#include "dsp/kernels/gemm_fp16/gemm_fp16.h"
#include "dsp/kernels/gemm_w8a8/gemm_w8a8.h"
#include "dsp/models/shared/layer_weights.h"
#include "dsp/models/shared/vision_transformer.h"
#include "dsp/runtime/heap.h"
#include "dsp/runtime/stage_timer.h"
#include "dsp/runtime/status.h"
#include "dsp/runtime/synchronization.h"
#include "dsp/runtime/thread_pool.h"
#include "dsp/vector_math/functions.h"
#include "benzene/protocol.h"

namespace benzene::models::vggt {
namespace {

namespace block_part = protocol::vision_transformer::block_part;
namespace camera_part = protocol::vggt::camera_part;
namespace convolution = protocol::vggt::dpt_convolution;
namespace gemm = kernels::gemm_w8a8;
namespace gemm_fp16 = kernels::gemm_fp16;
namespace tail_part = protocol::vggt::tail_part;
namespace camera_stat = protocol::vggt::camera_stat;
namespace dpt_stat = protocol::vggt::dpt_stat;
namespace transformer_stat = protocol::vggt::transformer_stat;
using gemm::Activation;
using gemm::OutputForm;
using hvx::Mask;
using hvx::Vector;
using hvx::VectorPair;
using protocol::vggt::kPatchSize;
using protocol::vggt::kPoseValues;
using protocol::vggt::kSpecialTokens;
using protocol::vision_transformer::BlockKind;
using vision_transformer::kHidden;
using vision_transformer::kMaximumFrames;

constexpr int kKeptPairs[protocol::vggt::kKeptOutputs] = {4, 11, 17, 23};
constexpr int kKeepItemRows = 16;

// One call at a time: the calls share the session's state (the heads read the transformer's kept outputs).
struct CallGuard {
    std::atomic<int> in_use;
};

struct Workspace {
    vision_transformer::Workspace blocks;
    runtime::HeapArray<_Float16> kept[protocol::vggt::kKeptOutputs];
    int kept_capacity;
};

struct TransformerGeometry {
    int frames;
    int tokens_per_frame;
    int patch_rows;
    int patch_columns;
};

runtime::Status allocate_transformer_workspace(Workspace * w, int tokens, int rows) {
    const runtime::Status blocks = vision_transformer::allocate_workspace(&w->blocks, tokens, rows);
    if (!blocks || w->kept_capacity >= tokens) return blocks;

    for (runtime::HeapArray<_Float16> & kept : w->kept) kept.reset();  // (freed before the larger ones are allocated)
    bool allocated = true;
    for (runtime::HeapArray<_Float16> & kept : w->kept) {
        kept = runtime::allocate_array<_Float16>(static_cast<size_t>(tokens) * 2 * kHidden);
        allocated = allocated && kept;
    }
    if (!allocated) {
        *w = Workspace{};
        return std::unexpected(runtime::Error::kNoMemory);
    }

    w->kept_capacity = tokens;
    return {};
}

BlockKind block_kind(const int32_t * references, int block) {
    return static_cast<BlockKind>(references[block * protocol::vggt::kBlockReferences + 2]);
}

struct SpecialTokenJob {
    const vision_transformer::Workspace * workspace;
    const float * camera_tokens;
    const float * register_tokens;
};

void replace_special_token(void * context, int token, int) {
    const auto * job = static_cast<const SpecialTokenJob *>(context);
    const vision_transformer::Workspace * w = job->workspace;
    const int frame = token / w->tokens_per_frame;
    const int index = token % w->tokens_per_frame;
    if (index >= kSpecialTokens) return;

    const int set = frame > 0;
    auto * x = hvx::vectors(w->x.get() + static_cast<size_t>(token) * kHidden);
    const auto * source =
        hvx::vectors(index == 0 ? job->camera_tokens + static_cast<size_t>(set) * kHidden
                                : job->register_tokens + (static_cast<size_t>(set) * 4 + index - 1) * kHidden);
    for (int k = 0; k < kHidden / 32; k++) x[k] = source[k];
}

struct KeepJob {
    Workspace * workspace;
    int k;
    int half;
};

void keep_rows(void * context, int item, int) {
    const auto * job = static_cast<const KeepJob *>(context);
    const vision_transformer::Workspace * w = &job->workspace->blocks;
    const int first = item * kKeepItemRows;

    for (int row = first; row < first + kKeepItemRows && row < w->tokens; row++) {
        const auto * x = hvx::vectors(w->x.get() + static_cast<size_t>(row) * kHidden);
        auto * out = hvx::vectors(job->workspace->kept[job->k].get() + static_cast<size_t>(row) * 2 * kHidden +
                                  static_cast<size_t>(job->half) * kHidden);
        for (int k = 0; k < kHidden / 64; k++) out[k] = to_float16(x[2 * k], x[2 * k + 1]);
    }
}

const _Float16 * kept_output(runtime::Session & session, int k, TransformerGeometry * geometry) {
    const Workspace * workspace = session.state<Workspace>();
    if (workspace == nullptr || k < 0 || k >= protocol::vggt::kKeptOutputs || !workspace->kept[k] ||
        workspace->blocks.frames == 0) {
        return nullptr;
    }
    geometry->frames = workspace->blocks.frames;
    geometry->tokens_per_frame = workspace->blocks.tokens_per_frame;
    geometry->patch_rows = workspace->blocks.patch_rows;
    geometry->patch_columns = workspace->blocks.patch_columns;
    return workspace->kept[k].get();
}

constexpr int kCameraHidden = 2048;
constexpr int kCameraMlpHidden = 8192;
constexpr int kCameraHeads = 16;
constexpr int kCameraHeadDimensions = 128;
constexpr int kTrunkBlocks = protocol::vggt::kCameraTrunkBlocks;
constexpr int kTrunkBlockReferences = protocol::blob_references(block_part::kCount);
constexpr int kBranchHidden = 1024;

struct CameraWorkspace {
    int frames;
    runtime::HeapArray<float> normalized_tokens;
    runtime::HeapArray<float> x;
    runtime::HeapArray<float> hidden;
    runtime::HeapArray<float> qkv;
    runtime::HeapArray<float> attention;
    runtime::HeapArray<float> modulation;
    runtime::HeapArray<float> scratch;
    runtime::HeapArray<float> embedding;
};

void layer_norm(float * y, const float * x, int n, const float * gamma, const float * beta, float epsilon) {
    double sum = 0;
    double square = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i];
        square += static_cast<double>(x[i]) * x[i];
    }

    const double mean = sum / n;
    const double variance = square / n - mean * mean;
    const auto m = static_cast<float>(mean);
    const float r = 1.0f / sqrtf(static_cast<float>(variance > 0 ? variance : 0) + epsilon);
    for (int i = 0; i < n; i++) y[i] = (x[i] - m) * r * (gamma ? gamma[i] : 1.0f) + (beta ? beta[i] : 0.0f);
}

void layer_norm_rows(float * y, const float * x, int rows, const float * gamma, const float * beta, float epsilon) {
    for (int s = 0; s < rows; s++) {
        layer_norm(y + static_cast<size_t>(s) * kCameraHidden, x + static_cast<size_t>(s) * kCameraHidden,
                   kCameraHidden, gamma, beta, epsilon);
    }
}

runtime::Status run_w8a8(runtime::Session & session, OutputForm form, Activation activation, const float * x, int M,
                         int K, int N, const gemm::Weights & weights, const float * bias,
                         const float * output_column_scale, float * y) {
    gemm::Arguments call{.M = M, .K = K, .N = N, .weights = weights};
    call.input.form = gemm::InputForm::kRowsF32;
    call.input.rows_f32 = x;
    call.output.form = form;
    call.output.activation = activation;
    call.output.rows_f32 = y;
    call.output.bias = bias;
    call.output.column_scale = output_column_scale;
    return gemm::run(session, call);
}

runtime::Status run_trunk_block(runtime::Session & session, const LayerWeights & layer, CameraWorkspace * w) {
    const int frames = w->frames;

    layer_norm_rows(w->scratch.get(), w->x.get(), frames, layer.part<float>(block_part::kNorm1Gamma),
                    layer.part<float>(block_part::kNorm1Beta), 1e-5f);
    runtime::Status status =
        run_w8a8(session, OutputForm::kRowsF32Activation, Activation::kNone, w->scratch.get(), frames, kCameraHidden,
                 3 * kCameraHidden, layer.gemm(block_part::kQkvWeights, block_part::kQkvScales),
                 layer.part<float>(block_part::kQkvBias), nullptr, w->qkv.get());
    if (!status) return status;

    const auto * inverse_output_scale = layer.part<float>(block_part::kOutputInverseScale);
    for (int head = 0; head < kCameraHeads; head++) {
        for (int i = 0; i < frames; i++) {
            const float * q = w->qkv.get() + static_cast<size_t>(i) * 3 * kCameraHidden + head * kCameraHeadDimensions;
            float probabilities[kMaximumFrames];
            float maximum = -INFINITY;
            float sum = 0.0f;

            for (int k = 0; k < frames; k++) {
                const float * key = w->qkv.get() + static_cast<size_t>(k) * 3 * kCameraHidden + kCameraHidden +
                                    head * kCameraHeadDimensions;
                float dot = 0.0f;
                for (int e = 0; e < kCameraHeadDimensions; e++) dot += q[e] * key[e];
                probabilities[k] = dot * 0.08838834764831845f;  // 1 / sqrt(128)
                maximum = probabilities[k] > maximum ? probabilities[k] : maximum;
            }

            for (int k = 0; k < frames; k++) {
                probabilities[k] = expf(probabilities[k] - maximum);
                sum += probabilities[k];
            }

            float * out = w->attention.get() + static_cast<size_t>(i) * kCameraHidden + head * kCameraHeadDimensions;
            for (int e = 0; e < kCameraHeadDimensions; e++) {
                float accumulator = 0.0f;
                for (int k = 0; k < frames; k++) {
                    accumulator += probabilities[k] * w->qkv[static_cast<size_t>(k) * 3 * kCameraHidden +
                                                             2 * kCameraHidden + head * kCameraHeadDimensions + e];
                }
                out[e] = accumulator / sum * inverse_output_scale[head * kCameraHeadDimensions + e];
            }
        }
    }

    status = run_w8a8(session, OutputForm::kRowsF32Accumulate, Activation::kNone, w->attention.get(), frames,
                      kCameraHidden, kCameraHidden, layer.gemm(block_part::kOutputWeights, block_part::kOutputScales),
                      layer.part<float>(block_part::kOutputBias), nullptr, w->x.get());
    if (!status) return status;

    layer_norm_rows(w->scratch.get(), w->x.get(), frames, layer.part<float>(block_part::kNorm2Gamma),
                    layer.part<float>(block_part::kNorm2Beta), 1e-5f);
    status = run_w8a8(session, OutputForm::kRowsF32Activation, Activation::kGeluErf, w->scratch.get(), frames,
                      kCameraHidden, kCameraMlpHidden, layer.gemm(block_part::kFc1Weights, block_part::kFc1Scales),
                      layer.part<float>(block_part::kFc1Bias), layer.part<float>(block_part::kFc2InverseScale),
                      w->hidden.get());
    if (!status) return status;
    return run_w8a8(session, OutputForm::kRowsF32Accumulate, Activation::kNone, w->hidden.get(), frames,
                    kCameraMlpHidden, kCameraHidden, layer.gemm(block_part::kFc2Weights, block_part::kFc2Scales),
                    layer.part<float>(block_part::kFc2Bias), nullptr, w->x.get());
}

runtime::Status run_head(runtime::Session & session, const LayerWeights blocks[kTrunkBlocks], const LayerWeights & head,
                         const float * tokens, int frames, float * pose, int32_t * statistics) {
    runtime::StageTimer timer(statistics);
    CameraWorkspace w{};
    w.frames = frames;
    const auto rows = static_cast<size_t>(frames);
    w.normalized_tokens = runtime::allocate_array<float>(rows * kCameraHidden);
    w.x = runtime::allocate_array<float>(rows * kCameraHidden);
    w.scratch = runtime::allocate_array<float>(rows * kCameraHidden);
    w.attention = runtime::allocate_array<float>(rows * kCameraHidden);
    w.embedding = runtime::allocate_array<float>(rows * kCameraHidden);
    w.hidden = runtime::allocate_array<float>(rows * kCameraMlpHidden);
    w.qkv = runtime::allocate_array<float>(rows * 3 * kCameraHidden);
    w.modulation = runtime::allocate_array<float>(rows * 3 * kCameraHidden);
    if (!w.normalized_tokens || !w.x || !w.scratch || !w.attention || !w.embedding || !w.hidden || !w.qkv ||
        !w.modulation) {
        return std::unexpected(runtime::Error::kNoMemory);
    }

    const auto * embed_weights = head.part<float>(camera_part::kPoseEmbedWeights);
    const auto * embed_bias = head.part<float>(camera_part::kPoseEmbedBias);
    const auto * modulation_inverse = head.part<float>(camera_part::kModulationInverseScale);
    const auto * branch2_weights = head.part<float>(camera_part::kPoseBranch2Weights);
    const auto * branch2_bias = head.part<float>(camera_part::kPoseBranch2Bias);
    float prediction[kMaximumFrames][kPoseValues];
    layer_norm_rows(w.normalized_tokens.get(), tokens, frames, head.part<float>(camera_part::kTokenNormGamma),
                    head.part<float>(camera_part::kTokenNormBeta), 1e-5f);

    for (int iteration = 0; iteration < 4; iteration++) {
        for (int s = 0; s < frames; s++) {
            const float * previous = iteration == 0 ? head.part<float>(camera_part::kEmptyPoseTokens) : prediction[s];
            float * e = w.embedding.get() + static_cast<size_t>(s) * kCameraHidden;
            for (int i = 0; i < kCameraHidden; i++) {
                float a = embed_bias[i];
                for (int k = 0; k < kPoseValues; k++) a += embed_weights[i * kPoseValues + k] * previous[k];
                e[i] = a / (1.0f + expf(-a)) * modulation_inverse[i];
            }
        }
        runtime::Status status = run_w8a8(session, OutputForm::kRowsF32Activation, Activation::kNone, w.embedding.get(),
                                          frames, kCameraHidden, 3 * kCameraHidden,
                                          head.gemm(camera_part::kModulationWeights, camera_part::kModulationScales),
                                          head.part<float>(camera_part::kModulationBias), nullptr, w.modulation.get());
        if (!status) return status;

        for (int s = 0; s < frames; s++) {  // x = gate * (LN(pt) * (1 + scale) + shift) + pt
            const float * m = w.modulation.get() + static_cast<size_t>(s) * 3 * kCameraHidden;
            const float * pt = w.normalized_tokens.get() + static_cast<size_t>(s) * kCameraHidden;
            float * x = w.x.get() + static_cast<size_t>(s) * kCameraHidden;
            float * t = w.scratch.get() + static_cast<size_t>(s) * kCameraHidden;
            layer_norm(t, pt, kCameraHidden, nullptr, nullptr, 1e-6f);
            for (int i = 0; i < kCameraHidden; i++)
                x[i] = m[2 * kCameraHidden + i] * (t[i] * (1.0f + m[kCameraHidden + i]) + m[i]) + pt[i];
        }

        for (int block = 0; block < kTrunkBlocks; block++) {
            status = run_trunk_block(session, blocks[block], &w);
            if (!status) return status;
        }

        layer_norm_rows(w.scratch.get(), w.x.get(), frames, head.part<float>(camera_part::kTrunkNormGamma),
                        head.part<float>(camera_part::kTrunkNormBeta), 1e-5f);
        status = run_w8a8(session, OutputForm::kRowsF32Activation, Activation::kGeluErf, w.scratch.get(), frames,
                          kCameraHidden, kBranchHidden,
                          head.gemm(camera_part::kPoseBranch1Weights, camera_part::kPoseBranch1Scales),
                          head.part<float>(camera_part::kPoseBranch1Bias), nullptr, w.hidden.get());
        if (!status) return status;

        for (int s = 0; s < frames; s++) {
            const float * h = w.hidden.get() + static_cast<size_t>(s) * kBranchHidden;
            for (int k = 0; k < kPoseValues; k++) {
                float a = branch2_bias[k];
                for (int i = 0; i < kBranchHidden; i++) a += branch2_weights[k * kBranchHidden + i] * h[i];
                prediction[s][k] = iteration == 0 ? a : prediction[s][k] + a;
            }
            float * out = pose + (static_cast<size_t>(iteration) * frames + s) * kPoseValues;
            for (int k = 0; k < kPoseValues; k++)
                out[k] = (k >= 7 && prediction[s][k] < 0.0f) ? 0.0f : prediction[s][k];  // FoV: relu
        }
    }

    timer.lap(camera_stat::kMicroseconds);
    return {};
}

constexpr int kFeatures = 256;
constexpr int kProjectChannels[4] = {256, 512, 1024, 1024};

struct DptWorkspace {
    int patch_rows;
    int patch_columns;
    int height;
    int width;
    int stride2_rows;
    int stride2_columns;
    runtime::HeapArray<_Float16> normalized;
    runtime::HeapArray<_Float16> project_positions[4];
    runtime::HeapArray<_Float16> projected[4];
    runtime::HeapArray<_Float16> layer1;
    runtime::HeapArray<_Float16> layer2;
    runtime::HeapArray<_Float16> layer4;
    runtime::HeapArray<_Float16> fusion_inputs[4];
    runtime::HeapArray<_Float16> temporary;
    runtime::HeapArray<_Float16> refined;
    runtime::HeapArray<_Float16> fused;
    runtime::HeapArray<_Float16> upsampled;
    runtime::HeapArray<_Float16> fusion_output;
    runtime::HeapArray<_Float16> head_features;
    runtime::HeapArray<_Float16> image_features;
    runtime::HeapArray<float> last_hidden_rows;
    runtime::HeapArray<float> column_positions;
    runtime::HeapArray<float> row_positions;
    runtime::HeapArray<int> stride2_taps;
    runtime::HeapArray<int> transposed_rows[2];
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

int position_table_values(int patch_rows, int patch_columns) {
    return (patch_columns + patch_rows) * (128 + 256 + 512 + 512) +
           (kPatchSize * patch_columns + kPatchSize * patch_rows) * 64;
}

runtime::Status allocate_workspace(DptWorkspace * w, int patch_rows, int patch_columns, const float * position_tables) {
    if (w->normalized && w->patch_rows == patch_rows && w->patch_columns == patch_columns) return {};

    *w = DptWorkspace{};
    w->patch_rows = patch_rows;
    w->patch_columns = patch_columns;
    w->height = patch_rows * kPatchSize;
    w->width = patch_columns * kPatchSize;
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
    allocated &= runtime::allocate(w->head_features, 64 * patches * 128);
    allocated &= runtime::allocate(w->image_features, H * W * 128);
    allocated &= runtime::allocate(w->last_hidden_rows, H * W * 32);
    allocated &= runtime::allocate(w->column_positions, W * 64);
    allocated &= runtime::allocate(w->row_positions, H * 64);
    allocated &= runtime::allocate(w->transposed_rows[0], patches);
    allocated &= runtime::allocate(w->transposed_rows[1], patches);
    w->stride2_taps = make_tap_map(patch_rows, patch_columns, 2, w->stride2_rows, w->stride2_columns);
    if (!allocated || !w->stride2_taps) {
        *w = DptWorkspace{};
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
                         const float * column_positions, const float * row_positions) {
    ResampleJob job{source,         destination, input_rows,       input_columns, output_rows,
                    output_columns, channels,    column_positions, row_positions};
    return runtime::parallel_for(session, runtime::kHvxThreads, output_rows, resample_row, &job);
}

struct LastConvolutionJob {
    const float * hidden_rows;
    const float * weights;
    const float * bias;
    float * out;
    int channels;
    int pixels;
};

void last_convolution(void * context, int item, int) {
    const auto * job = static_cast<const LastConvolutionJob *>(context);
    const Vector<float> log2e = Vector<float>::splat(1.4426950408889634f);
    const Vector<float> one = Vector<float>::splat(1.0f);
    const Vector<float> sign_bit = Vector<float>::splat_bits(0x80000000u);
    const Vector<float> magnitude_mask = Vector<float>::splat_bits(0x7fffffffu);
    const Mask last_lane = Mask::first_bytes(124);
    alignas(128) float results[4][32];
    const int first_pixel = item * 32;
    const int pixels = job->pixels - first_pixel < 32 ? job->pixels - first_pixel : 32;
    hvx::prefetch_to_l2(job->hidden_rows + static_cast<size_t>(first_pixel) * 32, 4096, 4096, 1);

    for (int o = 0; o < job->channels; o++) {
        const Vector<float> weights = hvx::load(job->weights + o * 32);
        Vector<float> sums = Vector<float>::zero();
        // pixel p's dot product (in every lane) shifts in at lane 31: after the 32 pixels lane p holds pixel p
        for (int p = 0; p < 32; p++) {
            const Vector<float> sum = vector_math::sum_to_all_lanes(
                hvx::load(job->hidden_rows + static_cast<size_t>(first_pixel + (p < pixels ? p : 0)) * 32) * weights);
            sums = select(last_lane, rotate_right_bytes(sums, 4), sum);
        }

        Vector<float> y = sums + job->bias[o];
        if (o == job->channels - 1 || job->channels == 2) {  // confidence: 1 + exp; depth: exp
            const Vector<float> e = vector_math::exp2(y * log2e);
            y = o == job->channels - 1 ? e + one : e;
        } else {  // points: inv_log = sign(y) (exp |y| - 1)
            const Vector<float> e = vector_math::exp2((y & magnitude_mask) * log2e) - one;
            y = e | (y & sign_bit);
        }
        store(results[o], y);
    }

    for (int p = 0; p < pixels; p++) {
        for (int o = 0; o < job->channels; o++)
            job->out[static_cast<size_t>(first_pixel + p) * job->channels + o] = results[o][p];
    }
}

// The DPT head's convolutions, then its other parts (dpt_head_part) at kCount.
struct HeadWeights {
    LayerWeights parts[convolution::kCount + 1];
};

runtime::Status convolve(runtime::Session & session, const HeadWeights * head, int which, const _Float16 * input,
                         int row_stride, const int * input_map, int taps, bool relu_input, int M, int K, int N,
                         gemm_fp16::Arguments output, int tap_width) {
    output.input_rows = input;
    output.row_stride = row_stride;
    output.input_map = input_map;
    output.taps = taps;
    output.relu_input = relu_input;
    output.weights = head->parts[which].part<uint8_t>(protocol::vggt::dpt_convolution_part::kWeights);
    output.tap_width = tap_width;
    output.M = M;
    output.K = K;
    output.N = N;
    return kernels::gemm_fp16::run(session, output);
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

// conv2(relu(conv1(relu(x)))) + relu(x) [+ skip]: DPT's inplace ReLU also rectifies the residual.
runtime::Status residual_conv_unit(runtime::Session & session, const HeadWeights * head, int conv1, const _Float16 * x,
                                   int pixels, int tap_width, const _Float16 * skip, _Float16 * temporary,
                                   _Float16 * out) {
    const runtime::Status first =
        convolve(session, head, conv1, x, kFeatures, nullptr, 9, 1, pixels, 9 * kFeatures, kFeatures,
                 f16_output(temporary, kFeatures, nullptr, 0, nullptr, 1), tap_width);
    if (!first) return first;
    return convolve(session, head, conv1 + 1, temporary, kFeatures, nullptr, 9, 0, pixels, 9 * kFeatures, kFeatures,
                    f16_output(out, kFeatures, x, 1, skip, 0), tap_width);
}

runtime::Status run_frame(runtime::Session & session, const HeadWeights * head, DptWorkspace * w,
                          const _Float16 * const kept[4], int frame, int tokens_per_frame, int channels, float * out,
                          int32_t * statistics) {
    const int patches = w->patch_rows * w->patch_columns;
    const int stride2_pixels = w->stride2_rows * w->stride2_columns;
    const int image_pixels = w->height * w->width;
    const auto * norm_gamma = head->parts[convolution::kCount].part<float>(protocol::vggt::dpt_head_part::kNormGamma);
    const auto * norm_beta = head->parts[convolution::kCount].part<float>(protocol::vggt::dpt_head_part::kNormBeta);
    runtime::Status status;
    runtime::StageTimer timer(statistics);

    for (int k = 0; k < 4; k++) {
        NormJob norm_job{kept[k], norm_gamma, norm_beta, w->normalized.get(),
                         frame * tokens_per_frame + kSpecialTokens};
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
                      64 * patches, 9 * kFeatures, 128, f16_output(w->head_features.get(), 128, nullptr, 0, nullptr, 0),
                      8 * w->patch_columns);
    if (!status) return status;
    timer.lap(dpt_stat::kOutputConvolutionsMicroseconds);
    status = resample(session, w->head_features.get(), 8 * w->patch_rows, 8 * w->patch_columns, w->image_features.get(),
                      w->height, w->width, 128, w->column_positions.get(), w->row_positions.get());
    if (!status) return status;
    timer.lap(dpt_stat::kElementwiseMicroseconds);
    gemm_fp16::Arguments output;
    output.output_form = kernels::gemm_fp16::OutputForm::kRowsF32;
    output.rows_f32 = w->last_hidden_rows.get();
    output.relu_output = true;
    status = convolve(session, head, convolution::kOutputConv2, w->image_features.get(), 128, nullptr, 9, 0,
                      image_pixels, 9 * 128, 32, output, w->width);
    if (!status) return status;
    timer.lap(dpt_stat::kOutputConvolutionsMicroseconds);

    LastConvolutionJob last_job{
        w->last_hidden_rows.get(),
        head->parts[convolution::kCount].part<float>(protocol::vggt::dpt_head_part::kLastConvWeights),
        head->parts[convolution::kCount].part<float>(protocol::vggt::dpt_head_part::kLastConvBias),
        out,
        channels,
        image_pixels};
    status =
        runtime::parallel_for(session, runtime::kHvxThreads, (image_pixels + 31) / 32, last_convolution, &last_job);
    if (!status) return status;
    timer.lap(dpt_stat::kElementwiseMicroseconds);
    return {};
}

}  // namespace

runtime::Status run_transformer(runtime::Session & session, const TransformerArguments & arguments) {
    runtime::StageTimer call_timer(arguments.statistics);
    for (int block = 0; block < arguments.blocks; block++) {
        const int slot = arguments.references[block * protocol::vggt::kBlockReferences];
        if (!session.model_mapped(slot)) return std::unexpected(runtime::Error::kBadParameter);
    }
    if (!session.model_mapped(arguments.tail[0])) return std::unexpected(runtime::Error::kBadParameter);

    const int tokens = arguments.frames * arguments.tokens_per_frame;
    const int frame_layout_rows = arguments.frames * ((arguments.tokens_per_frame + 31) / 32 * 32);
    const int global_layout_rows = (tokens + 31) / 32 * 32;
    CallGuard * guard = session.state<CallGuard>();
    if (guard == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(guard->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);
    runtime::StageTimer workspace_timer(arguments.statistics);
    Workspace * kept = session.state<Workspace>();
    if (kept == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    Workspace & workspace = *kept;
    runtime::Status status = allocate_transformer_workspace(
        &workspace, tokens, frame_layout_rows > global_layout_rows ? frame_layout_rows : global_layout_rows);
    if (!status) return status;
    status = vision_transformer::set_geometry(&workspace.blocks, arguments.frames, arguments.tokens_per_frame,
                                              arguments.patch_rows, arguments.patch_columns, kSpecialTokens);
    if (!status) return status;
    vision_transformer::Workspace * w = &workspace.blocks;
    workspace_timer.lap(transformer_stat::kWorkspaceMicroseconds);

    const LayerWeights tail_weights{session.model_base(arguments.tail[0]) + arguments.tail[1], arguments.tail + 2};
    {
        runtime::StageTimer embedding_timer(arguments.statistics);
        status = vision_transformer::embed(session, w, tail_weights.part<uint8_t>(tail_part::kPatchEmbedding),
                                           arguments.patches, protocol::vggt::kPatchInputs, arguments.positions,
                                           arguments.special_tokens);
        embedding_timer.lap(transformer_stat::kEmbeddingMicroseconds);
        if (!status) return status;
    }

    int aggregator_blocks = 0;
    for (int block = 0; block < arguments.blocks; block++) {
        const int32_t * block_references = arguments.references + block * protocol::vggt::kBlockReferences;
        const BlockKind kind = block_kind(arguments.references, block);
        if (kind != BlockKind::kDinov2 && block > 0 &&
            block_kind(arguments.references, block - 1) == BlockKind::kDinov2) {
            runtime::StageTimer output_timer(arguments.statistics);
            status = vision_transformer::normalize_patch_tokens(session, w,
                                                                tail_weights.part<float>(tail_part::kDinov2NormGamma),
                                                                tail_weights.part<float>(tail_part::kDinov2NormBeta));
            if (!status) return status;
            SpecialTokenJob special_job{w, tail_weights.part<float>(tail_part::kCameraTokens),
                                        tail_weights.part<float>(tail_part::kRegisterTokens)};
            status = runtime::parallel_for(session, runtime::kHvxThreads, tokens, replace_special_token, &special_job);
            if (!status) return status;
            output_timer.lap(transformer_stat::kDinov2OutputMicroseconds);
        }

        const LayerWeights layer{session.model_base(block_references[0]) + block_references[1], block_references + 3};
        status = vision_transformer::run_block(session, layer, kind, w, arguments.statistics);
        if (!status) return status;
        if (kind != BlockKind::kDinov2) {
            const int pair = aggregator_blocks / 2;
            for (int k = 0; k < protocol::vggt::kKeptOutputs; k++) {
                if (kKeptPairs[k] == pair) {
                    runtime::StageTimer keep_timer(arguments.statistics);
                    KeepJob keep_job{&workspace, k, kind == BlockKind::kGlobal};
                    status = runtime::parallel_for(session, runtime::kHvxThreads,
                                                   (tokens + kKeepItemRows - 1) / kKeepItemRows, keep_rows, &keep_job);
                    if (!status) return status;
                    keep_timer.lap(transformer_stat::kKeptOutputsMicroseconds);
                }
            }
            aggregator_blocks++;
        }
    }

    if (arguments.kept_outputs != nullptr) {
        runtime::StageTimer copy_timer(arguments.statistics);
        for (int k = 0; k < protocol::vggt::kKeptOutputs; k++) {
            status =
                runtime::copy_memory(session, arguments.kept_outputs + static_cast<size_t>(k) * tokens * 2 * kHidden,
                                     workspace.kept[k].get(), static_cast<size_t>(tokens) * 2 * kHidden * 2);
            if (!status) return status;
        }
        copy_timer.lap(transformer_stat::kKeptOutputsMicroseconds);
    }
    call_timer.lap(transformer_stat::kTotalMicroseconds);
    return {};
}

KeptGeometry kept_geometry(runtime::Session & session) {
    const Workspace * workspace = session.state<Workspace>();
    if (workspace == nullptr || !workspace->kept[0]) return {0, 0, 0};
    return {workspace->blocks.frames, workspace->blocks.patch_rows * kPatchSize,
            workspace->blocks.patch_columns * kPatchSize};
}

runtime::Status run_camera_head(runtime::Session & session, const CameraHeadArguments & arguments) {
    CallGuard * guard = session.state<CallGuard>();
    if (guard == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(guard->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);
    TransformerGeometry geometry;
    const _Float16 * kept = kept_output(session, 3, &geometry);
    if (kept == nullptr) return std::unexpected(runtime::Error::kBadParameter);
    if (geometry.frames < 1 || geometry.frames > kMaximumFrames) return std::unexpected(runtime::Error::kBadParameter);
    LayerWeights blocks[kTrunkBlocks];
    for (int b = 0; b < kTrunkBlocks; b++) {
        const runtime::Result<LayerWeights> block =
            layer_in_model(session, arguments.references + b * kTrunkBlockReferences);
        if (!block) return std::unexpected(block.error());
        blocks[b] = *block;
    }
    const runtime::Result<LayerWeights> head =
        layer_in_model(session, arguments.references + kTrunkBlocks * kTrunkBlockReferences);
    if (!head) return std::unexpected(head.error());

    const runtime::HeapArray<float> tokens =
        runtime::allocate_array<float>(static_cast<size_t>(geometry.frames) * kCameraHidden);
    if (!tokens) return std::unexpected(runtime::Error::kNoMemory);
    for (int frame = 0; frame < geometry.frames; frame++) {
        for (int i = 0; i < kCameraHidden; i++) {
            tokens[static_cast<size_t>(frame) * kCameraHidden + i] =
                static_cast<float>(kept[static_cast<size_t>(frame) * geometry.tokens_per_frame * kCameraHidden + i]);
        }
    }
    return run_head(session, blocks, *head, tokens.get(), geometry.frames, arguments.pose, arguments.statistics);
}

runtime::Status run_dpt_head(runtime::Session & session, const DptHeadArguments & arguments) {
    CallGuard * guard = session.state<CallGuard>();
    if (guard == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(guard->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);
    runtime::StageTimer call_timer(arguments.statistics);
    TransformerGeometry geometry;
    const _Float16 * kept[4];
    for (int k = 0; k < 4; k++) {
        kept[k] = kept_output(session, k, &geometry);
        if (kept[k] == nullptr) return std::unexpected(runtime::Error::kBadParameter);
    }
    if (arguments.channels != 2 && arguments.channels != 4) return std::unexpected(runtime::Error::kBadParameter);
    HeadWeights head{};
    for (int i = 0; i <= convolution::kCount; i++) {
        const runtime::Result<LayerWeights> part =
            layer_in_model(session, arguments.references +
                                        i * protocol::blob_references(protocol::vggt::dpt_convolution_part::kCount));
        if (!part) return std::unexpected(part.error());
        head.parts[i] = *part;
    }
    if (arguments.position_values != position_table_values(geometry.patch_rows, geometry.patch_columns)) {
        return std::unexpected(runtime::Error::kBadParameter);
    }

    DptWorkspace * kept_workspace = session.state<DptWorkspace>();
    if (kept_workspace == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    DptWorkspace & workspace = *kept_workspace;
    runtime::StageTimer workspace_timer(arguments.statistics);
    const runtime::Status allocated =
        allocate_workspace(&workspace, geometry.patch_rows, geometry.patch_columns, arguments.position_tables);
    workspace_timer.lap(dpt_stat::kWorkspaceMicroseconds);
    if (!allocated) return allocated;

    for (int frame = 0; frame < geometry.frames; frame++) {
        const runtime::Status ran = run_frame(
            session, &head, &workspace, kept, frame, geometry.tokens_per_frame, arguments.channels,
            arguments.out + static_cast<size_t>(frame) * workspace.height * workspace.width * arguments.channels,
            arguments.statistics);
        if (!ran) return ran;
    }
    call_timer.lap(dpt_stat::kTotalMicroseconds);
    return {};
}

}  // namespace benzene::models::vggt
