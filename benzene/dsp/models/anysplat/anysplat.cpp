#include "dsp/models/anysplat/anysplat.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "dsp/models/shared/dpt_head.h"
#include "dsp/models/shared/layer_weights.h"
#include "dsp/models/vggt/vggt.h"
#include "dsp/runtime/cache.h"
#include "dsp/runtime/heap.h"
#include "dsp/runtime/stage_timer.h"
#include "dsp/runtime/synchronization.h"
#include "dsp/runtime/thread_pool.h"
#include "benzene/protocol.h"

namespace benzene::models::anysplat {
namespace {

namespace convolution = protocol::dpt_head::convolution;
namespace dpt_stat = protocol::dpt_head::stat;
namespace gaussian_stat = protocol::anysplat::gaussian_stat;
namespace head_part = protocol::anysplat::gaussian_head_part;
using protocol::anysplat::kGaussianOutputChannels;
using protocol::anysplat::kMergerInputs;
using protocol::anysplat::kMergerKernel;
using protocol::anysplat::kMergerStrip;
using protocol::vggt::kPatchSize;
using protocol::vggt::kSpecialTokens;

constexpr int kHalf = kMergerKernel / 2;

struct GaussianWorkspace {
    int height;
    int width;
    runtime::HeapArray<_Float16> hidden;  // output_conv2.0's output [pixels][dpt_head::kHeadFeatures]
    runtime::HeapArray<_Float16> strips;  // the merger's strips [pixels][kMergerStrip] (zeros past 3 x 7)
    runtime::HeapArray<int> taps;         // per pixel and window row: the strip of that image row (-1: outside)
    _Float16 pixel_values[256];           // u / 255 of every pixel value u
};

runtime::Status allocate_gaussian_workspace(GaussianWorkspace * g, int height, int width) {
    if (g->hidden && g->height == height && g->width == width) return {};

    *g = GaussianWorkspace{};
    const size_t pixels = static_cast<size_t>(height) * width;
    bool allocated = runtime::allocate(g->hidden, pixels * dpt_head::kHeadFeatures);
    allocated &= runtime::allocate(g->strips, pixels * kMergerStrip);
    allocated &= runtime::allocate(g->taps, pixels * kMergerKernel);
    if (!allocated) {
        *g = GaussianWorkspace{};
        return std::unexpected(runtime::Error::kNoMemory);
    }
    g->height = height;
    g->width = width;
    memset(g->strips.get(), 0, pixels * kMergerStrip * sizeof(_Float16));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int ky = 0; ky < kMergerKernel; ky++) {
                const int source = y + ky - kHalf;
                g->taps[(static_cast<size_t>(y) * width + x) * kMergerKernel + ky] =
                    source >= 0 && source < height ? source * width + x : -1;
            }
        }
    }
    for (int u = 0; u < 256; u++) g->pixel_values[u] = static_cast<_Float16>(static_cast<float>(u) / 255.0f);
    return {};
}

struct StripJob {
    const uint8_t * image;  // [3][height][width]
    const _Float16 * pixel_values;
    _Float16 * strips;
    int height;
    int width;
};

// The strips of image row y: per pixel the 7 values around it in the row of each channel, (channel, column) order as
// input_merger.0's weight per window row, zeros outside the image.
void make_strips(void * context, int y, int) {
    const auto * job = static_cast<const StripJob *>(context);
    const int W = job->width;
    const _Float16 zero = static_cast<_Float16>(0.0f);
    _Float16 * strip = job->strips + static_cast<size_t>(y) * W * kMergerStrip;
    for (int x = 0; x < W; x++, strip += kMergerStrip) {
        for (int c = 0; c < 3; c++) {
            const uint8_t * row = job->image + (static_cast<size_t>(c) * job->height + y) * W;
            _Float16 * values = strip + c * kMergerKernel;
            if (x >= kHalf && x + kHalf < W) {
                for (int kx = 0; kx < kMergerKernel; kx++) values[kx] = job->pixel_values[row[x + kx - kHalf]];
            } else {
                for (int kx = 0; kx < kMergerKernel; kx++) {
                    const int source = x + kx - kHalf;
                    values[kx] = source >= 0 && source < W ? job->pixel_values[row[source]] : zero;
                }
            }
        }
    }
}

}  // namespace

runtime::Status run_gaussian_head(runtime::Session & session, const GaussianHeadArguments & arguments) {
    std::atomic<int> * guard = vggt::call_guard(session);
    if (guard == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(*guard);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);
    runtime::StageTimer call_timer(arguments.statistics);
    const vggt::KeptOutputs kept = vggt::kept_outputs(session);
    if (kept.outputs[0] == nullptr || arguments.frame < 0 || arguments.frame >= kept.frames ||
        arguments.position_values != dpt_head::position_table_values(kept.patch_rows, kept.patch_columns, kPatchSize) ||
        reinterpret_cast<uintptr_t>(arguments.out) % 128 != 0) {
        return std::unexpected(runtime::Error::kBadParameter);
    }
    dpt_head::HeadWeights head{};
    if (const runtime::Status loaded = dpt_head::load_head(session, arguments.references, &head); !loaded) {
        return loaded;
    }
    const LayerWeights & own = head.parts[convolution::kCount];

    GaussianWorkspace * g = session.state<GaussianWorkspace>();
    if (g == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    runtime::StageTimer workspace_timer(arguments.statistics);
    const runtime::Result<dpt_head::Workspace *> workspace = dpt_head::session_workspace(
        session, kept.patch_rows, kept.patch_columns, kPatchSize, arguments.position_tables);
    if (!workspace) return std::unexpected(workspace.error());
    dpt_head::Workspace * w = *workspace;
    runtime::Status status = allocate_gaussian_workspace(g, w->height, w->width);
    workspace_timer.lap(dpt_stat::kWorkspaceMicroseconds);
    if (!status) return status;

    status = dpt_head::run_trunk(
        session, head, w, kept.outputs, arguments.frame * kept.tokens_per_frame + kSpecialTokens,
        own.part<float>(head_part::kNormGamma), own.part<float>(head_part::kNormBeta), arguments.statistics);
    if (!status) return status;

    runtime::StageTimer timer(arguments.statistics);
    const int H = w->height;
    const int W = w->width;
    runtime::invalidate_host_input(arguments.image, static_cast<size_t>(3) * H * W);
    StripJob strip_job{arguments.image, g->pixel_values, g->strips.get(), H, W};
    status = runtime::parallel_for(session, runtime::kHvxThreads, H, make_strips, &strip_job);
    if (!status) return status;
    status = dpt_head::convolve(
        session, own.part<uint8_t>(head_part::kMergerWeights), g->strips.get(), kMergerStrip, g->taps.get(),
        kMergerKernel, false, H * W, kMergerInputs, dpt_head::kHeadFeatures,
        dpt_head::f16_output(w->image_features.get(), dpt_head::kHeadFeatures, nullptr, false, nullptr, true), 0);
    if (!status) return status;
    timer.lap(gaussian_stat::kMergerMicroseconds);
    status = dpt_head::upsample_to_image(session, w, true);
    if (!status) return status;
    timer.lap(dpt_stat::kElementwiseMicroseconds);
    status = dpt_head::convolve(
        session, head, convolution::kOutputConv2, w->image_features.get(), dpt_head::kHeadFeatures, nullptr, 9, false,
        H * W, 9 * dpt_head::kHeadFeatures, dpt_head::kHeadFeatures,
        dpt_head::f16_output(g->hidden.get(), dpt_head::kHeadFeatures, nullptr, false, nullptr, true), W);
    if (!status) return status;
    status = dpt_head::convolve(
        session, own.part<uint8_t>(head_part::kOutputWeights), g->hidden.get(), dpt_head::kHeadFeatures, nullptr, 1,
        false, H * W, dpt_head::kHeadFeatures, kGaussianOutputChannels,
        dpt_head::f16_output(arguments.out, kGaussianOutputChannels, nullptr, false, nullptr, false), 0);
    if (!status) return status;
    timer.lap(dpt_stat::kOutputConvolutionsMicroseconds);
    call_timer.lap(dpt_stat::kTotalMicroseconds);
    return {};
}

}  // namespace benzene::models::anysplat
