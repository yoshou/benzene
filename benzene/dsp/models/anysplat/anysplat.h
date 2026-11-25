#pragma once

#include <cstdint>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"

namespace benzene::models::anysplat {

// The references, sizes and statistics are protocol::anysplat's. AnySplat's transformer, camera head and depth head
// are VGGT's (dsp/models/vggt/vggt.h); this is its Gaussian head. It runs one at a time with the vggt calls (a second
// caller gets kBadParameter).
struct GaussianHeadArguments {
    const int32_t * references = nullptr;     // kGaussianReferences
    const float * position_tables = nullptr;  // the UV position tables (vggt::DptHeadArguments::position_tables)
    int position_values = 0;
    int frame = 0;                    // a frame of the last vggt::run_transformer
    const uint8_t * image = nullptr;  // its pixels [3][height][width]
    _Float16 * out = nullptr;         // [height][width][kGaussianOutputChannels], 128-byte aligned
    int32_t * statistics = nullptr;   // gaussian_stat
};

// The Gaussian head of one frame on the kept outputs of the last vggt::run_transformer: the DPT trunk
// (dsp/models/shared/dpt_head.h), the image merger (per pixel a strip of the 7 values around it in its row of each
// channel, then a GEMM with a tap per window row, ReLU) plus the trunk's features upsampled to the image with the UV
// position embedding, output_conv2.0 (3x3, ReLU) and the output convolution: the raw values of every pixel's Gaussian.
// A frame per call (vggt::run_dpt_head runs every frame): a call takes one frame's image and fills one frame's output,
// 37 MiB at 448 x 448.
runtime::Status run_gaussian_head(runtime::Session & session, const GaussianHeadArguments & arguments);

}  // namespace benzene::models::anysplat
