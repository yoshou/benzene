#pragma once

#include <cstdint>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"

namespace benzene::models::loma {

// The references, sizes and statistics are protocol::loma's.
struct Arguments {
    const int32_t * references = nullptr;      // kReferences
    const uint8_t * detector_image = nullptr;  // [detector_height][detector_width]
    int detector_height = 0;
    int detector_width = 0;
    int detector_zero_point = 0;                 // the code of the first convolution's padding pixels
    const uint8_t * descriptor_image = nullptr;  // [descriptor_height][descriptor_width]
    int descriptor_height = 0;
    int descriptor_width = 0;
    const _Float16 * patches = nullptr;  // [rows * columns][kDinov2PatchInputs]
    const float * positions = nullptr;   // [rows * columns][kDinov2Hidden]
    int count = 0;                       // keypoints
    int slot = 0;                        // the feature slot written (for LightGlue)
    float * keypoints = nullptr;         // [count][2] (x, y) normalized to [-1, 1]
    float * scores = nullptr;            // [count]
    float * descriptors = nullptr;       // fp32 [count][kDescriptorDimensions]
    int32_t * statistics = nullptr;      // stat
};

// LoMa's features of one image, into feature slot `slot` (normalized positions, fp16 descriptions) for the matcher and
// the outputs (fp32 descriptions: the precision the descriptor computes). One call at a time (a second caller gets
// kBadParameter).
// - The images: the detector image, gray codes at the detector's size (height and width multiples of 8); the
//   descriptor image, gray codes at the descriptor's size (multiples of kPatchSize).
// - The keypoints: `count` by descending probability after the softmax and a 3x3 NMS (ties in raster order), at their
//   sub-pixel positions normalized to [-1, 1] (pixel centres at -1 + (2 i + 1) / size); scores their probabilities.
// - The descriptions: from the descriptor image with DINOv2's input from the host for its rows x columns patches
//   (rows = descriptor_height / kPatchSize, columns = descriptor_width / kPatchSize): patches, gray / 255
//   (zero-padded), and positions, the position embedding resized to the patch grid (bicubic as DINOv2's
//   interpolate_pos_encoding); not bit-exact (DINOv2's attention and GEMM stages are fp16).
runtime::Status run(runtime::Session & session, const Arguments & arguments);

}  // namespace benzene::models::loma
