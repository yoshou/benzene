#pragma once

#include <cstdint>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"

namespace benzene::models::vggt {

// The references, sizes and statistics are protocol::vggt's. The calls run one at a time (a second caller gets
// kBadParameter).

struct TransformerArguments {
    const int32_t * references = nullptr;  // kBlockReferences per block
    int blocks = 0;
    const int32_t * tail = nullptr;          // kTailReferences
    const _Float16 * patches = nullptr;      // [frames * patch_rows * patch_columns][kPatchInputs]
    const float * positions = nullptr;       // position embeddings [patch_rows * patch_columns][1024]
    const float * special_tokens = nullptr;  // [5][1024]
    int frames = 0;
    int tokens_per_frame = 0;
    int patch_rows = 0;
    int patch_columns = 0;
    _Float16 * kept_outputs = nullptr;  // fp16 [kKeptOutputs][frames * tokens_per_frame][kKeptOutputWidth], or nullptr
    int32_t * statistics = nullptr;     // transformer_stat
};

// The transformer on mapped weights: the DINOv2 input from the patches, position embeddings and special tokens, through
// the blocks of references.
// - The outputs of the frame / global pairs 4, 11, 17 and 23 stay on the DSP for the heads, and are copied to
//   kept_outputs unless it is nullptr.
runtime::Status run_transformer(runtime::Session & session, const TransformerArguments & arguments);

// What the last run_transformer kept for the heads: its frames (0: none) and image size (the patch grid x kPatchSize).
struct KeptGeometry {
    int frames;
    int height;
    int width;
};
KeptGeometry kept_geometry(runtime::Session & session);

struct CameraHeadArguments {
    const int32_t * references = nullptr;  // kCameraReferences
    // [kCameraIterations][frames][kPoseValues]: every iteration's activated encoding; the last one is VGGT's pose_enc
    float * pose = nullptr;
    int32_t * statistics = nullptr;  // camera_stat
};

// On the camera tokens of the last run_transformer (kept output 3 = frame | global block 23, token 0 of each frame).
runtime::Status run_camera_head(runtime::Session & session, const CameraHeadArguments & arguments);

struct DptHeadArguments {
    const int32_t * references = nullptr;  // kDptReferences
    // VGGT's UV position embeddings times 0.1, per grid a column table, then a row table: for project k (C = 256, 512,
    // 1024, 1024) [patch columns][C / 2], [patch rows][C / 2]; then for the image [W][64], [H][64]
    const float * position_tables = nullptr;
    int position_values = 0;
    int channels = 0;                // 2 (the depth head) or 4 (the point head); the last channel is the confidence
    float * out = nullptr;           // [frames][H][W][channels]
    int32_t * statistics = nullptr;  // dpt_stat
};

// One head over all frames of the last run_transformer, on its kept outputs (patch tokens of frame / global blocks 4,
// 11, 17, 23).
runtime::Status run_dpt_head(runtime::Session & session, const DptHeadArguments & arguments);

}  // namespace benzene::models::vggt
