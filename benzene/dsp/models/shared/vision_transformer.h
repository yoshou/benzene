#pragma once

#include <cstdint>

#include "dsp/hvx/vector.h"
#include "dsp/kernels/gemm_w8a8/gemm_w8a8.h"
#include "dsp/models/shared/layer_weights.h"
#include "dsp/runtime/dma_ring.h"
#include "dsp/runtime/heap.h"
#include "dsp/runtime/int8_scale_table.h"
#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"
#include "dsp/runtime/thread_pool.h"
#include "benzene/protocol.h"

namespace benzene::models::vision_transformer {

constexpr int kHidden = 1024;
constexpr int kHeads = 16;
constexpr int kHeadDimensions = 64;
constexpr int kMlpHidden = 4096;
constexpr int kMaximumFrames = 64;

struct NormVtcm;

// The blocks' state; callers read the geometry and the residual stream x.
struct Workspace {
    int frames;
    int tokens_per_frame;
    int special_tokens;  // per frame, before the patch tokens
    int patch_rows;
    int patch_columns;
    int tokens;
    int padded_frame_tokens;
    int frame_layout_rows;
    int global_layout_rows;
    int token_capacity;
    int row_capacity;
    runtime::HeapArray<float> x;  // the residual stream [tokens][1024], frame-major
    runtime::HeapArray<uint8_t> input_tiles;
    runtime::HeapArray<float> input_row_scales;
    runtime::HeapArray<uint8_t> query_tiles;
    runtime::HeapArray<uint8_t> key_tiles;
    runtime::HeapArray<uint8_t> value_tiles;
    runtime::HeapArray<_Float16> attention_output;
    kernels::gemm_w8a8::HandoverTiles handover;
    runtime::HeapArray<int> frame_rows;   // frame layout row -> token (-1: padding)
    runtime::HeapArray<int> global_rows;  // global layout row -> token (-1: padding)
    runtime::HeapArray<int> token_frame_rows;
    runtime::HeapArray<int16_t> frame_positions;  // RoPE positions (row, column) per layout row
    runtime::HeapArray<int16_t> global_positions;
    runtime::HeapArray<float> rope_table;  // per position: cos lanes, -sin / +sin lanes
    int rope_positions;
    int frame_first_tokens[kMaximumFrames];  // the frame segments of the frame layout
    int frame_token_counts[kMaximumFrames];
    runtime::HeapObject<runtime::DmaRing> rings[runtime::kHvxThreads];
    const float * norm_beta;
    float norm_epsilon;
    const int * norm_rows;
    int norm_layout_rows;
    NormVtcm * norm_vtcm;
    const hvx::Vector<float> * norm_gamma_vtcm;
    const runtime::Int8Scale * int8_scales;
    runtime::HeapArray<_Float16> handover_tiles;
    runtime::HeapArray<float> handover_row_scales;
    runtime::HeapArray<uint16_t> handover_inverse_bits;
};

// Buffers for `tokens` tokens and `rows` layout rows (rows >= both layouts' rows, a multiple of 32).
runtime::Status allocate_workspace(Workspace * w, int tokens, int rows);

// The frame layout puts every frame on a 32-row boundary (padded_frame_tokens rows each); the global layout keeps
// token order (rows tokens .. global_layout_rows: padding).
runtime::Status set_geometry(Workspace * w, int frames, int tokens_per_frame, int patch_rows, int patch_columns,
                             int special_tokens);

// One block on w->x in place; adds its stage times to statistics (protocol::block_stat).
runtime::Status run_block(runtime::Session & session, const LayerWeights & layer,
                          protocol::vision_transformer::BlockKind kind, Workspace * w, int32_t * statistics);

// Fills w->x: per frame the special tokens (special_tokens [w->special_tokens][1024], position included), then the
// patch tokens: patches [frames * patch_rows * patch_columns][patch_inputs] (fp16) through the fp16 GEMM weight stream
// `embedding`, plus positions [patch_rows * patch_columns][1024].
runtime::Status embed(runtime::Session & session, Workspace * w, const uint8_t * embedding, const _Float16 * patches,
                      int patch_inputs, const float * positions, const float * special_tokens);

// The patch tokens normalized in place (LayerNorm eps 1e-6, f32 throughout); the special tokens unchanged.
runtime::Status normalize_patch_tokens(runtime::Session & session, Workspace * w, const float * gamma,
                                       const float * beta);

}  // namespace benzene::models::vision_transformer
