#pragma once

#include <cstdint>

#include "dsp/kernels/gemm_fp16/gemm_fp16.h"
#include "dsp/models/shared/layer_weights.h"
#include "dsp/runtime/heap.h"
#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"
#include "benzene/protocol.h"

namespace benzene::models::dpt_head {

// The DPT head of VGGT's architecture (protocol::dpt_head) on a transformer's kKeptOutputs kept outputs (fp16 rows of
// 2048, frame | global block outputs), for the heads built on it (VGGT's depth and point heads, AnySplat's Gaussian
// head): the trunk from the kept outputs to output_conv1, the upsampling of its features to the image with the UV
// position embedding, and the convolutions as fp16 GEMMs. The heads' own parts are the callers'.

constexpr int kKeptOutputs = 4;
constexpr int kFeatures = 256;      // the fusion's channels
constexpr int kHeadFeatures = 128;  // output_conv1's

// A head's convolutions in protocol::dpt_head::convolution order, then its own blob at kCount.
struct HeadWeights {
    LayerWeights parts[protocol::dpt_head::convolution::kCount + 1];
};

// references: per convolution {slot, offset, convolution_part parts}, then {slot, offset, the own blob's parts}.
runtime::Status load_head(const runtime::Session & session, const int32_t * references, HeadWeights * head);

// The buffers for a patch grid.
struct Workspace {
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
    runtime::HeapArray<_Float16> head_features;  // output_conv1's output [8 patch_rows][8 patch_columns][kHeadFeatures]
    runtime::HeapArray<_Float16> image_features;  // [height][width][kHeadFeatures]
    runtime::HeapArray<float> column_positions;
    runtime::HeapArray<float> row_positions;
    runtime::HeapArray<int> stride2_taps;
    runtime::HeapArray<int> transposed_rows[2];
};

// The values of the UV position tables of a patch grid with patch_size pixels per patch
// (dsp/models/vggt/vggt.h DptHeadArguments::position_tables).
int position_table_values(int patch_rows, int patch_columns, int patch_size);

// The session's workspace for the patch grid, with the position tables: one for all the heads (they run one at a
// time), allocated again when the grid changes; kNoMemory when out of memory.
runtime::Result<Workspace *> session_workspace(runtime::Session & session, int patch_rows, int patch_columns,
                                               int patch_size, const float * position_tables);

// The trunk on one frame's patch tokens, the patch_rows * patch_columns rows from first_token of each kept output:
// norm, projects, resizes, layer_rn, refinenets, output_conv1 -> w->head_features. Stage times in
// protocol::dpt_head::stat.
runtime::Status run_trunk(runtime::Session & session, const HeadWeights & head, Workspace * w,
                          const _Float16 * const kept[kKeptOutputs], int first_token, const float * norm_gamma,
                          const float * norm_beta, int32_t * statistics);

// w->head_features bilinearly (align_corners) to the image, plus the UV position embedding (channels < 64 by column,
// the others by row), into w->image_features; accumulate: added to what w->image_features holds.
runtime::Status upsample_to_image(runtime::Session & session, Workspace * w, bool accumulate);

// One convolution as the fp16 GEMM: `output` with its output fields set (f16_output, or the fp32 rows).
runtime::Status convolve(runtime::Session & session, const uint8_t * weights, const _Float16 * input, int row_stride,
                         const int * input_map, int taps, bool relu_input, int M, int K, int N,
                         kernels::gemm_fp16::Arguments output, int tap_width);
runtime::Status convolve(runtime::Session & session, const HeadWeights & head, int which, const _Float16 * input,
                         int row_stride, const int * input_map, int taps, bool relu_input, int M, int K, int N,
                         kernels::gemm_fp16::Arguments output, int tap_width);

// fp16 rows [M][channels] (+ residual, relu'd when relu_residual; + second_residual; then relu).
kernels::gemm_fp16::Arguments f16_output(_Float16 * out, int channels, const _Float16 * residual, bool relu_residual,
                                         const _Float16 * second_residual, bool relu);

}  // namespace benzene::models::dpt_head
