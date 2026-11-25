#include "dsp/models/vggt/vggt.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/prefetch.h"
#include "dsp/hvx/vector.h"
#include "dsp/kernels/gemm_fp16/gemm_fp16.h"
#include "dsp/kernels/gemm_w8a8/gemm_w8a8.h"
#include "dsp/models/shared/dpt_head.h"
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
namespace convolution = protocol::dpt_head::convolution;
namespace gemm = kernels::gemm_w8a8;
namespace gemm_fp16 = kernels::gemm_fp16;
namespace tail_part = protocol::vggt::tail_part;
namespace camera_stat = protocol::vggt::camera_stat;
namespace dpt_head_part = protocol::vggt::dpt_head_part;
namespace dpt_stat = protocol::dpt_head::stat;
namespace transformer_stat = protocol::vggt::transformer_stat;
using gemm::Activation;
using gemm::OutputForm;
using hvx::Mask;
using hvx::Vector;
using protocol::vggt::kPatchSize;
using protocol::vggt::kPoseValues;
using protocol::vggt::kSpecialTokens;
using protocol::vision_transformer::BlockKind;
using vision_transformer::kHidden;
using vision_transformer::kMaximumFrames;

constexpr int kKeptPairs[protocol::vggt::kKeptOutputs] = {4, 11, 17, 23};
constexpr int kKeepItemRows = 16;
static_assert(protocol::vggt::kKeptOutputs == dpt_head::kKeptOutputs);

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

// The depth and point heads' own buffer: output_conv2.0's output, the last convolution's input [pixels][32] (fp32).
struct DptHeadWorkspace {
    size_t pixels;
    runtime::HeapArray<float> hidden_rows;
};

runtime::Status allocate_dpt_head_workspace(DptHeadWorkspace * w, size_t pixels) {
    if (w->hidden_rows && w->pixels == pixels) return {};

    *w = DptHeadWorkspace{};
    if (!runtime::allocate(w->hidden_rows, pixels * 32)) return std::unexpected(runtime::Error::kNoMemory);
    w->pixels = pixels;
    return {};
}

runtime::Status run_frame(runtime::Session & session, const dpt_head::HeadWeights & head, dpt_head::Workspace * w,
                          float * hidden_rows, const KeptOutputs & kept, int frame, int channels, float * out,
                          int32_t * statistics) {
    const LayerWeights & misc = head.parts[convolution::kCount];
    runtime::Status status = dpt_head::run_trunk(
        session, head, w, kept.outputs, frame * kept.tokens_per_frame + kSpecialTokens,
        misc.part<float>(dpt_head_part::kNormGamma), misc.part<float>(dpt_head_part::kNormBeta), statistics);
    if (!status) return status;
    runtime::StageTimer timer(statistics);
    status = dpt_head::upsample_to_image(session, w, false);
    if (!status) return status;
    timer.lap(dpt_stat::kElementwiseMicroseconds);
    gemm_fp16::Arguments output;
    output.output_form = kernels::gemm_fp16::OutputForm::kRowsF32;
    output.rows_f32 = hidden_rows;
    output.relu_output = true;
    const int image_pixels = w->height * w->width;
    status =
        dpt_head::convolve(session, head, convolution::kOutputConv2, w->image_features.get(), dpt_head::kHeadFeatures,
                           nullptr, 9, 0, image_pixels, 9 * dpt_head::kHeadFeatures, 32, output, w->width);
    if (!status) return status;
    timer.lap(dpt_stat::kOutputConvolutionsMicroseconds);

    LastConvolutionJob last_job{hidden_rows,
                                misc.part<float>(dpt_head_part::kLastConvWeights),
                                misc.part<float>(dpt_head_part::kLastConvBias),
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
    const KeptOutputs kept = kept_outputs(session);
    if (kept.outputs[0] == nullptr) return std::unexpected(runtime::Error::kBadParameter);
    if (arguments.channels != 2 && arguments.channels != 4) return std::unexpected(runtime::Error::kBadParameter);
    dpt_head::HeadWeights head{};
    if (const runtime::Status loaded = dpt_head::load_head(session, arguments.references, &head); !loaded) {
        return loaded;
    }
    if (arguments.position_values != dpt_head::position_table_values(kept.patch_rows, kept.patch_columns, kPatchSize)) {
        return std::unexpected(runtime::Error::kBadParameter);
    }

    DptHeadWorkspace * own = session.state<DptHeadWorkspace>();
    if (own == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    runtime::StageTimer workspace_timer(arguments.statistics);
    const runtime::Result<dpt_head::Workspace *> workspace = dpt_head::session_workspace(
        session, kept.patch_rows, kept.patch_columns, kPatchSize, arguments.position_tables);
    if (!workspace) return std::unexpected(workspace.error());
    dpt_head::Workspace * w = *workspace;
    const runtime::Status allocated = allocate_dpt_head_workspace(own, static_cast<size_t>(w->height) * w->width);
    workspace_timer.lap(dpt_stat::kWorkspaceMicroseconds);
    if (!allocated) return allocated;

    for (int frame = 0; frame < kept.frames; frame++) {
        const runtime::Status ran =
            run_frame(session, head, w, own->hidden_rows.get(), kept, frame, arguments.channels,
                      arguments.out + static_cast<size_t>(frame) * w->height * w->width * arguments.channels,
                      arguments.statistics);
        if (!ran) return ran;
    }
    call_timer.lap(dpt_stat::kTotalMicroseconds);
    return {};
}

KeptOutputs kept_outputs(runtime::Session & session) {
    KeptOutputs kept{};
    TransformerGeometry geometry{};
    for (int k = 0; k < protocol::vggt::kKeptOutputs; k++) kept.outputs[k] = kept_output(session, k, &geometry);
    if (kept.outputs[0] == nullptr) return KeptOutputs{};
    kept.frames = geometry.frames;
    kept.tokens_per_frame = geometry.tokens_per_frame;
    kept.patch_rows = geometry.patch_rows;
    kept.patch_columns = geometry.patch_columns;
    return kept;
}

std::atomic<int> * call_guard(runtime::Session & session) {
    CallGuard * guard = session.state<CallGuard>();
    return guard != nullptr ? &guard->in_use : nullptr;
}

}  // namespace benzene::models::vggt
