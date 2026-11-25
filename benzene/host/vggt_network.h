#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "host/files.h"
#include "host/model_file.h"
#include "benzene/protocol.h"

namespace benzene::host {

// VGGT's network as the vggt and anysplat commands run it: the input bundle, the model file's VGGT tensors (vggt.*,
// python/benzene/models/vggt/export_hexmodel.py), their weight ranges, DINOv2's inputs, the DPT heads' UV position
// tables, and the transformer, camera head and DPT head calls.

constexpr int kVggtBlocks = 72;

// A bundle (python/benzene/models/vggt/prepare_bundle.py): meta.txt "frames height width", images.u8 the frames'
// planes [frames][3][height][width].
struct VggtBundle {
    int frames = 0;
    int height = 0;
    int width = 0;
    MallocBuffer<uint8_t> images;
};

// False and a message when the bundle is missing or inconsistent.
bool read_vggt_bundle(const char * directory, VggtBundle * bundle);

// The weight ranges (slot: the index) and the transformer's and camera head's references.
struct VggtWeights {
    std::vector<WeightRange> ranges;
    int32_t blocks[kVggtBlocks * protocol::vggt::kBlockReferences];
    int32_t tail[protocol::vggt::kTailReferences];
    int32_t camera[protocol::vggt::kCameraReferences];
};

// The blocks in ranges of at most 240 MiB (the tail with the last), then the camera head in a range; false and a
// message when a tensor is missing.
bool plan_vggt_weights(const ModelTable & table, const char * model, VggtWeights * weights);

// A head of the DPT head's architecture: its dpt_head::convolution::kCount convolutions "<prefix>.<i>", then its own
// blob "<prefix>.<own>" of own_parts parts. Its references: per convolution {slot, offset, convolution_part parts},
// then {slot, offset, own parts}.
struct DptHeadTensors {
    const char * prefix;
    const char * own;
    int own_parts;
    int32_t * references;
};

// The heads in one more range; false and a message when a tensor is missing.
bool plan_dpt_heads(const ModelTable & table, const char * model, const DptHeadTensors * heads, int count,
                    VggtWeights * weights);

// Reads the ranges on threads while DINOv2's inputs are made (patches: a new rpcmem buffer [frames * patches]
// [kPatchInputs]; grid_positions [patches][kHidden]; special [kSpecialTokens][kHidden]), then maps the ranges; false
// and a message on failure.
bool load_vggt_weights(const char * model, const ModelTable & table, const VggtBundle & bundle, VggtWeights * weights,
                       _Float16 ** patches, std::vector<float> * grid_positions, float * special);

// The DPT heads' UV position tables (dsp/models/vggt/vggt.h DptHeadArguments::position_tables).
std::vector<float> dpt_uv_tables(int patch_rows, int patch_columns, int height, int width);

// The calls; each prints its times (`name` the command's) and returns false with a message on failure. kept: the
// transformer's kept outputs (kept_values) or nullptr; pose [kCameraIterations][frames][kPoseValues]; out
// [frames][height][width][channels] (rpcmem).
bool run_vggt_transformer(const char * name, const VggtWeights & weights, const VggtBundle & bundle,
                          const _Float16 * patches, const std::vector<float> & grid_positions, const float * special,
                          _Float16 * kept, size_t kept_values);
bool run_vggt_camera_head(const VggtWeights & weights, int frames, std::vector<float> * pose);
bool run_vggt_dpt_head(const char * head, const int32_t * references, int channels, const std::vector<float> & uv,
                       float * out, size_t values);

}  // namespace benzene::host
