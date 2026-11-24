#pragma once

#include <cstdint>

// What the host and the DSP agree on beyond benzene.idl: the layout of the references (where a call finds its weights
// in the mapped model files), the statistics a call returns, and the sizes both sides use. One namespace per model.
// A call's statistics are zeroed before the call; the DSP adds the times (microseconds) and sets the counts.
namespace benzene::protocol {

// Model files the host maps into a session: slots 0 .. kModelSlots - 1.
constexpr int kModelSlots = 16;

// Features kept on the DSP between an extractor's call and LightGlue's.
constexpr int kFeatureSlots = 2;

// A layer's weights are a blob: parts at 4096-aligned byte offsets. A reference to a blob is {model slot, the blob's
// byte offset in the slot, the parts' byte offsets in the blob in the order of the model's *_part enum}.
constexpr int blob_references(int parts) {
    return 2 + parts;
}

// The stages of a transformer block, microseconds summed over the blocks of a call.
namespace block_stat {
enum : int {
    kNorm1Microseconds = 0,
    kQkvMicroseconds,
    kAttentionMicroseconds,
    kOutputMicroseconds,  // the attention's output projection and the residual
    kNorm2Microseconds,
    kFc1Microseconds,  // with the activation
    kFc2Microseconds,  // with the residual
    kCount
};
}  // namespace block_stat

namespace cosmos3_edge {

constexpr int kLlmHidden = 2048;
constexpr int kVisionHidden = 1152;
constexpr int kRopeValuesPerToken = 128;  // 64 cosines, then 64 sines
constexpr int kPatchesPerVisionToken = 4;
constexpr int kMaximumVisionImages = 63;  // per vision encoder call

// Language model decoder layer
namespace llm_layer_part {
enum : int {
    kQkvWeights = 0,  // fused q | k | v W8A8 stream (4096 columns)
    kQkvScales,       // fp32 column scales
    kOutputWeights,   // o_proj
    kOutputScales,
    kFc1Weights,  // mlp.fc1 (9216 columns)
    kFc1Scales,
    kFc2Weights,  // mlp.fc2
    kFc2Scales,
    kInputNormGamma,          // input_layernorm.weight * inverse scale of qkv, fp32 [2048]
    kOutputInverseScale,      // 1 / scale of o_proj, fp32 [2048]
    kPostAttentionNormGamma,  // post_attention_layernorm.weight * inverse scale of fc1, fp32 [2048]
    kFc2InverseScale,         // 1 / scale of fc2, fp32 [9216]
    kCount
};
}  // namespace llm_layer_part

constexpr int kLlmLayerReferences = blob_references(llm_layer_part::kCount);

// A layer's 4-bit decode weights: gemv Q4_0 streams.
namespace llm_layer_4bit_part {
enum : int { kQkv = 0, kOutput, kFc1, kFc2, kCount };
}  // namespace llm_layer_4bit_part

constexpr int kLlmLayer4BitReferences = blob_references(llm_layer_4bit_part::kCount);

// The decode step's head: the final norm and the lm_head (int8 GEMV stream and its column scales).
namespace llm_head_reference {
enum : int {
    kNormSlot = 0,
    kNormOffset,  // gamma = norm weight * inverse scale of the lm_head
    kLmHeadSlot,
    kLmHeadOffset,
    kScalesSlot,
    kScalesOffset,
    kColumns,  // the vocabulary
    kCount
};
}  // namespace llm_head_reference

// Vision encoder layer
namespace vision_layer_part {
enum : int {
    kQkvWeights = 0,  // fused q | k | v (3456 columns)
    kQkvScales,       // fp32 column scales
    kQkvBias,         // fp32 bias
    kOutputWeights,   // out_proj
    kOutputScales,
    kOutputBias,
    kFc1Weights,  // mlp.fc1, 4304 columns zero-padded to 4320
    kFc1Scales,
    kFc1Bias,
    kFc2Weights,  // mlp.fc2, 4320 inputs (zero-padded)
    kFc2Scales,
    kFc2Bias,
    kNorm1Gamma,          // layer_norm1 weight * inverse scale of qkv, fp32 [1152]
    kNorm1Beta,           // layer_norm1 bias * inverse scale of qkv
    kOutputInverseScale,  // 1 / scale of out_proj [1152]
    kNorm2Gamma,          // layer_norm2 weight * inverse scale of fc1
    kNorm2Beta,
    kFc2InverseScale,  // 1 / scale of fc2 [4320] (0 in the padding)
    kCount
};
}  // namespace vision_layer_part

constexpr int kVisionLayerReferences = blob_references(vision_layer_part::kCount);

// Vision tail: post_layernorm and projector
namespace vision_tail_part {
enum : int {
    kPostNormGamma = 0,  // vision post_layernorm weight [1152]
    kPostNormBeta,       // vision post_layernorm bias [1152]
    kMergerNormGamma,    // merger norm weight tiled 4 times, times the inverse scale of fc1 [4608]
    kMergerNormBeta,
    kFc1Weights,  // linear_fc1 4608 -> 11520
    kFc1Scales,
    kFc1Bias,
    kFc2InverseScale,  // 1 / scale of fc2 [11520]
    kFc2Weights,       // linear_fc2 11520 -> 2048
    kFc2Scales,
    kFc2Bias,
    kCount
};
}  // namespace vision_tail_part

constexpr int kVisionTailReferences = blob_references(vision_tail_part::kCount);

// After the block stages (block_stat)
namespace llm_prefill_stat {
enum : int { kLayersMicroseconds = block_stat::kCount, kCount };
}  // namespace llm_prefill_stat

namespace llm_decode_stat {
enum : int {
    kNormAndQkvMicroseconds = 0,  // summed over the layers
    kRopeAndAppendMicroseconds,   // RoPE and the KV cache append
    kAttentionMicroseconds,
    kOutputMicroseconds,
    kNorm2AndFc1Microseconds,
    kFc2Microseconds,
    kHeadMicroseconds,  // the final norm and the lm_head
    kCount
};
}  // namespace llm_decode_stat

// After the block stages (block_stat)
namespace vision_stat {
enum : int {
    kLayersMicroseconds = block_stat::kCount,
    kTailMicroseconds,
    kWorkspaceMicroseconds,
    kGroupSetupMicroseconds,  // the image groups' layouts and inputs
    kCount
};
}  // namespace vision_stat

}  // namespace cosmos3_edge

// The transformer blocks of DINOv2 ViT-L (VGGT's aggregator and DINOv2 encoder, LoMa's DINOv2) as
// dsp/models/shared/vision_transformer runs them.
namespace vision_transformer {

namespace block_part {
enum : int {
    kQkvWeights = 0,  // fused qkv (3072 columns: q, k, v, each 16 heads x 64)
    kQkvScales,       // fp32 column scales
    kQkvBias,
    kOutputWeights,  // attn.proj; scales and bias times LayerScale ls1
    kOutputScales,
    kOutputBias,
    kFc1Weights,  // mlp.fc1 (4096 columns)
    kFc1Scales,
    kFc1Bias,
    kFc2Weights,  // mlp.fc2; scales and bias times ls2
    kFc2Scales,
    kFc2Bias,
    kNorm1Gamma,  // norm1 weight * inverse scale of qkv [1024]
    kNorm1Beta,
    kOutputInverseScale,  // 1 / scale of proj [1024]
    kNorm2Gamma,          // norm2 weight * inverse scale of fc1
    kNorm2Beta,
    kFc2InverseScale,  // 1 / scale of fc2 [4096]
    kQkNorm,           // q_norm weight, bias, k_norm weight, bias [4][64] (zeros in DINOv2 blocks)
    kCount
};
}  // namespace block_part

enum class BlockKind : int32_t { kDinov2 = 0, kFrame = 1, kGlobal = 2 };

}  // namespace vision_transformer

namespace vggt {

constexpr int kHidden = 1024;
constexpr int kPatchSize = 14;
constexpr int kPatchInputs = 640;       // per patch: 3 x 14 x 14 values, zero-padded
constexpr int kSpecialTokens = 5;       // per frame: the camera token (cls + pos[0]) and the 4 register tokens
constexpr int kKeptOutputs = 4;         // frame / global blocks 4, 11, 17, 23
constexpr int kKeptOutputWidth = 2048;  // a kept output: frame | global block outputs
constexpr int kPoseValues = 9;          // VGGT's pose encoding
constexpr int kCameraIterations = 4;    // the camera head's refinement iterations

constexpr int kBlockReferences = 3 + vision_transformer::block_part::kCount;  // slot, offset, kind, parts

// DINOv2 final norm weight / bias [1024], aggregator camera tokens [2][1024], register tokens [2][4][1024], the DINOv2
// patch embedding as an fp16 GEMM weight stream (K = kPatchInputs)
namespace tail_part {
enum : int { kDinov2NormGamma = 0, kDinov2NormBeta, kCameraTokens, kRegisterTokens, kPatchEmbedding, kCount };
}  // namespace tail_part

constexpr int kTailReferences = blob_references(tail_part::kCount);

// The camera head; its trunk blocks use vision_transformer::block_part with 2048 / 8192 dimensions.
namespace camera_part {
enum : int {
    kTokenNormGamma = 0,  // token_norm weight [2048]
    kTokenNormBeta,       // token_norm bias [2048]
    kPoseEmbedWeights,    // embed_pose [2048][9] (fp32)
    kPoseEmbedBias,
    kEmptyPoseTokens,    // empty_pose_tokens [9]
    kModulationWeights,  // poseLN_modulation.1 (6144 columns: shift, scale, gate)
    kModulationScales,
    kModulationBias,
    kModulationInverseScale,  // its 1 / scale [2048] (times SiLU(embedding) before the GEMM)
    kTrunkNormGamma,          // trunk_norm weight * inverse scale of pose_branch.fc1
    kTrunkNormBeta,
    kPoseBranch1Weights,  // pose_branch.fc1 (1024 columns)
    kPoseBranch1Scales,
    kPoseBranch1Bias,
    kPoseBranch2Weights,  // pose_branch.fc2 [9][1024] (fp32)
    kPoseBranch2Bias,
    kCount
};
}  // namespace camera_part

constexpr int kCameraTrunkBlocks = 4;
// Per trunk block {slot, offset, block_part parts}, then {slot, offset, camera_part parts}.
constexpr int kCameraReferences =
    kCameraTrunkBlocks * blob_references(vision_transformer::block_part::kCount) + blob_references(camera_part::kCount);

// The DPT head's convolutions in run order, one model-file tensor each.
namespace dpt_convolution {
enum : int {
    kProject0 = 0,  // projects[k]: 1x1, 2048 -> 256 / 512 / 1024 / 1024
    kProject1,
    kProject2,
    kProject3,
    kResize0,   // resize: ConvTranspose 4x4 stride 4 (256)
    kResize1,   // ConvTranspose 2x2 stride 2 (512)
    kResize3,   // conv 3x3 stride 2
    kLayer1Rn,  // scratch.layer<k>_rn: 3x3 -> 256, no bias
    kLayer2Rn,
    kLayer3Rn,
    kLayer4Rn,
    kRefine4Unit2Conv1,  // refinenet4: resConfUnit2 conv1 / conv2, out_conv (1x1)
    kRefine4Unit2Conv2,
    kRefine4Output,
    kRefine3Unit1Conv1,  // refinenet3: resConfUnit1, resConfUnit2, out_conv
    kRefine3Unit1Conv2,
    kRefine3Unit2Conv1,
    kRefine3Unit2Conv2,
    kRefine3Output,
    kRefine2Unit1Conv1,
    kRefine2Unit1Conv2,
    kRefine2Unit2Conv1,
    kRefine2Unit2Conv2,
    kRefine2Output,
    kRefine1Unit1Conv1,
    kRefine1Unit1Conv2,
    kRefine1Unit2Conv1,
    kRefine1Unit2Conv2,
    kRefine1Output,
    kOutputConv1,  // output_conv1 (3x3 256 -> 128)
    kOutputConv2,  // output_conv2.0 (3x3 -> 32)
    kCount
};
}  // namespace dpt_convolution

// A convolution's gemm_fp16 weight stream: K = taps x channels, tap-major; a transposed convolution has its columns in
// (tap, channel) order; the bias is in the store blocks.
namespace dpt_convolution_part {
enum : int { kWeights = 0, kCount };
}  // namespace dpt_convolution_part

// norm weight / bias [2048], output_conv2.2 weight [channels][32] and bias [channels] (fp32)
namespace dpt_head_part {
enum : int { kNormGamma = 0, kNormBeta, kLastConvWeights, kLastConvBias, kCount };
}  // namespace dpt_head_part

// Per convolution {slot, offset, dpt_convolution_part parts}, then {slot, offset, dpt_head_part parts}.
constexpr int kDptReferences =
    dpt_convolution::kCount * blob_references(dpt_convolution_part::kCount) + blob_references(dpt_head_part::kCount);

// After the block stages (block_stat)
namespace transformer_stat {
enum : int {
    kDinov2OutputMicroseconds = block_stat::kCount,  // the final norm of the DINOv2 blocks
    kKeptOutputsMicroseconds,
    kWorkspaceMicroseconds,
    kTotalMicroseconds,
    kEmbeddingMicroseconds,  // the DINOv2 input tokens
    kCount
};
}  // namespace transformer_stat

namespace camera_stat {
enum : int { kMicroseconds = 0, kCount };
}  // namespace camera_stat

namespace dpt_stat {
enum : int {
    kProjectsMicroseconds = 0,  // the projects and the resizes
    kLayerRnMicroseconds,
    kRefineNetsMicroseconds,
    kOutputConvolutionsMicroseconds,
    kElementwiseMicroseconds,
    kTotalMicroseconds,
    kWorkspaceMicroseconds,  // the workspace and the position tables
    kCount
};
}  // namespace dpt_stat

}  // namespace vggt

namespace superpoint {

constexpr int kDescriptorDimensions = 256;

// The convolution weight streams in this order.
namespace convolution {
enum : int {
    kConv1a = 0,
    kConv1b,
    kConv2a,
    kConv2b,
    kConv3a,
    kConv3b,
    kConv4a,
    kConv4b,
    kDetectorAndDescriptorA,  // convPa and convDa fused (one 3x3 convolution with both outputs)
    kDetectorB,               // convPb
    kDescriptorB,             // convDb
    kCount
};
}  // namespace convolution

constexpr int kReferences = 1 + convolution::kCount;  // the model slot, the streams' offsets

namespace stat {
enum : int {
    kKeypoints = 0,
    kCandidates = 1,
    kFirstConvolutionMicroseconds = 2,  // conv1a .. conv4b: indices 2 .. 9
    kHeadsMicroseconds = 10,
    kSoftmaxMicroseconds = 11,
    kNmsMicroseconds = 12,
    kKeypointSelectionMicroseconds = 13,
    kDescriptorsMicroseconds = 14,
    kCount = 15
};
}  // namespace stat

}  // namespace superpoint

namespace xfeat {

constexpr int kDescriptorDimensions = 64;

// The tensors in this order. block1's layers are HVX tensors (layouts in dsp/models/xfeat/xfeat.cpp Block1Arguments),
// the others convolution_int8 weight streams.
namespace tensor {
enum : int {
    kBlock1Layer0 = 0,  // 1 -> 4, raw pixels (the image normalization folded in per image)
    kBlock1Layer1,      // 4 -> 8, stride 2
    kBlock1Layer2,      // 8 -> 8
    kBlock1Layer3,      // 8 -> 24, stride 2, plus the skip branch: block2's signed input
    kBlock2Layer0,
    kBlock2Layer1,
    kBlock3Layer0,
    kBlock3Layer1,
    kBlock3Layer2,
    kBlock4Layer0,
    kBlock4Layer1,
    kBlock4Layer2,
    kBlock5Layer0,
    kBlock5Layer1,
    kBlock5Layer2,
    kBlock5Layer3,
    kFusionLayer0,
    kFusionLayer1,
    kFusionLayer2,  // the descriptor map (fp32)
    kHeatmapLayer0,
    kHeatmapLayer1,
    kHeatmapLayer2,   // the reliability logit (fp32)
    kKeypointLayer0,  // over the raw 8 x 8 cells (normalization folded in per image)
    kKeypointLayer1,
    kKeypointLayer2,
    kKeypointLayer3,  // the 65 keypoint logits (fp32)
    // fp32 [4][64]: the fusion's x3, x4, x5 scales relative to its input scale, then the inverse input scale of the
    // heatmap head
    kFusionScales,
    kCount
};
}  // namespace tensor

constexpr int kReferences = 1 + tensor::kCount;  // the model slot, the tensors' offsets

namespace stat {
enum : int {
    kKeypoints = 0,
    kCandidates = 1,
    kImageMeanBits = 2,  // the normalization's fp32 mean and 1 / std as bits
    kImageInverseStdBits = 3,
    kBlock1Microseconds = 4,    // the image statistics and block1 (HVX)
    kBackboneMicroseconds = 5,  // block2 .. block5
    kFusionMicroseconds = 6,    // the fusion's sum and convolutions
    kHeadsMicroseconds = 7,
    kHeatmapMicroseconds = 8,  // softmax, depth to space, reliability
    kNmsMicroseconds = 9,
    kKeypointSelectionMicroseconds = 10,
    kDescriptorsMicroseconds = 11,
    kReliabilityMicroseconds = 12,        // part of kHeatmapMicroseconds
    kCellNormalizationMicroseconds = 13,  // part of kDescriptorsMicroseconds: the descriptor map's cells normalized
    kCount = 14
};
}  // namespace stat

}  // namespace xfeat

namespace aliked {

constexpr int kDescriptorDimensions = 128;

// The tensors in this order.
// - A convolution_int8 layer has its weight stream (K tap-major: k = tap * C + channel) and an auxiliary tensor: a
//   SELU layer's fp32 constants [convolution_int8::kSeluConstantCount][N] (column order) and, for a 3x3 layer, its
//   input's zero points [C] (uint8, the padding codes), after the constants at a 128-byte boundary.
// - An HVX layer (block1.conv1, the score head's 3x3 layers) has one tensor (layouts in dsp/models/aliked/aliked.cpp).
namespace tensor {
enum : int {
    kBlock1Conv1 = 0,  // HVX: 1 -> 16, raw pixels, int16 weights
    kBlock1Conv2,      // 16 -> 16 (input and output padded to 32 channels)
    kBlock1Conv2Aux,
    kBlock2Conv1,  // the 2x2 average of x1 -> 32
    kBlock2Conv1Aux,
    kBlock2Conv2,  // 3x3 over [block2.mid | x1.pool] (64 channels): conv2 and the downsample (centre tap)
    kBlock2Conv2Aux,
    kBlock3Conv1Offset,  // 3x3 -> 18 offsets (fp32 rows of 32)
    kBlock3Conv1OffsetAux,
    kBlock3Conv1,  // 1x1 over the deformable columns (K = 9 x 32, padded to 384)
    kBlock3Conv1Aux,
    kBlock3Conv2Offset,
    kBlock3Conv2OffsetAux,
    kBlock3Conv2,  // 1x1 over [deformable columns (9 x 64) | x2.pool (32)] (K padded to 640)
    kBlock3Conv2Aux,
    kBlock4Conv1Offset,
    kBlock4Conv1OffsetAux,
    kBlock4Conv1,  // K = 9 x 64 (padded to 640)
    kBlock4Conv1Aux,
    kBlock4Conv2Offset,
    kBlock4Conv2OffsetAux,
    kBlock4Conv2,  // K = 9 x 128 + 64 (padded to 1280)
    kBlock4Conv2Aux,
    kConv1,  // 1x1 x1 (32 channels) -> agg1 (32)
    kConv1Aux,
    kConv2,
    kConv2Aux,
    kConv3,
    kConv3Aux,
    kConv4,
    kConv4Aux,
    kScoreHead0Part1,  // 1x1 agg_k -> 8 (fp32 rows; 32 columns)
    kScoreHead0Part2,
    kScoreHead0Part3,
    kScoreHead0Part4,
    kScoreHead2,         // HVX: 3x3 8 -> 4
    kScoreHead4,         // HVX: 3x3 4 -> 4
    kScoreHead6,         // HVX: 3x3 4 -> 1, the logit
    kDescriptorOffset0,  // 1x1 over the 3x3 patches (K = 9 x 128) -> 32
    kDescriptorOffset0Aux,
    kDescriptorOffset2,   // 1x1 32 -> 32 offsets (fp32)
    kDescriptorFeatures,  // sf_conv: 1x1 over the 16 samples' codes (128) -> 128
    kDescriptorFeaturesAux,
    kDescriptorAggregation,  // 1x1 over the 16 samples' features (K = 16 x 128) -> 128 (fp32)
    kStageConstants,         // fp32 vectors of the HVX stages (dsp/models/aliked/aliked.cpp StageConstants)
    kCount
};
}  // namespace tensor

constexpr int kReferences = 1 + tensor::kCount;  // the model slot, the tensors' offsets

// The stages (2 .. 8) sum to the call; the later indices are parts of a stage.
namespace stat {
enum : int {
    kKeypoints = 0,
    kCandidates = 1,              // NMS maxima above the threshold
    kBlock1Microseconds = 2,      // block1 (conv1 on HVX, conv2), conv1 (agg1), score_head.0's part 1
    kBlock2Microseconds = 3,      // the 2x2 average, block2, conv2 (agg2), part 2
    kDeformableMicroseconds = 4,  // the 4x4 averages, blocks 3 and 4, conv3 / conv4, parts 3 / 4
    kScoreHeadMicroseconds = 5,   // the upsampling, the score rows (SELU, 3x3 layers, sigmoid)
    kNmsMicroseconds = 6,
    kKeypointSelectionMicroseconds = 7,  // the candidates, the selection and the soft-argmax
    kDescriptorsMicroseconds = 8,        // the feature map samples and SDDH
    kBlock1Conv1Microseconds = 9,        // (of block1) the HVX layer
    kBlock1Conv2Microseconds = 10,
    kConv1Microseconds = 11,
    kPart1Microseconds = 12,
    kBlock2Conv1Microseconds = 13,  // (of block2)
    kBlock2Conv2Microseconds = 14,
    kBlock3Microseconds = 15,                  // (of blocks 3, 4) the two deformable layers
    kDeformableSamplingMicroseconds = 16,      // their four samplings
    kScoreUpsamplingMicroseconds = 17,         // (of the score head)
    kDescriptorPatchesMicroseconds = 18,       // (of the descriptors) the feature map at the patches, their codes
    kDescriptorSamplesMicroseconds = 19,       // the samples' codes
    kDescriptorConvolutionsMicroseconds = 20,  // the four convolutions
    kCount = 21
};
}  // namespace stat

}  // namespace aliked

namespace lightglue {

enum class Model : int32_t {
    kSuperPoint = 0,  // 9 layers of 256 dimensions, 4 heads
    kXFeat = 1,       // LighterGlue: 6 layers of 96 dimensions, 1 head, descriptors of 64 projected in
    kAliked = 2,      // 9 layers of 256 dimensions, 4 heads, descriptors of 128 projected in
    // LoMa-B's matcher: 9 layers of 256 dimensions, 4 heads, no adaptive inference (no token confidences), keypoints
    // normalized per axis, the assignment without matchability
    kLoma = 3,
};

constexpr bool known_model(int32_t model) {
    return model >= static_cast<int32_t>(Model::kSuperPoint) && model <= static_cast<int32_t>(Model::kLoma);
}

constexpr int layers(Model model) {
    return model == Model::kXFeat ? 6 : 9;
}
constexpr bool input_projection(Model model) {
    return model == Model::kXFeat || model == Model::kAliked;
}
// The weights have token confidences (adaptive inference: depth / width confidence) and their references are set.
constexpr bool token_confidences(Model model) {
    return model != Model::kLoma;
}
constexpr int kMaximumLayers = 9;

namespace part {
enum : int { kQkv = 0, kOutput, kFfn1, kLayerNorm, kFfn2, kCount };
}  // namespace part

// The references: the model slot; per layer and block (self, cross) the part offsets; per layer its assignment head's
// final projection, matchability and (but for the last layer) token confidence; the positional encoding's Wr; for a
// model whose descriptors are projected in, the input projection.
constexpr int block_part_reference(int layer, int block, int which) {
    return 1 + (layer * 2 + block) * part::kCount + which;
}
constexpr int final_projection_reference(int layers, int layer) {
    return 1 + layers * 2 * part::kCount + layer;
}
constexpr int matchability_reference(int layers, int layer) {
    return final_projection_reference(layers, layer) + layers;
}
constexpr int token_confidence_reference(int layers, int layer) {
    return final_projection_reference(layers, layer) + 2 * layers;
}
constexpr int positional_encoding_reference(int layers) {
    return token_confidence_reference(layers, layers - 1);
}
constexpr int input_projection_reference(int layers) {
    return positional_encoding_reference(layers) + 1;
}
constexpr int references(int layers, bool input_projection) {
    return positional_encoding_reference(layers) + 1 + (input_projection ? 1 : 0);
}
constexpr int references(Model model) {
    return references(layers(model), input_projection(model));
}

namespace stat {
enum : int {
    kMatches = 0,
    kSetupMicroseconds = 1,
    kFirstLayerMicroseconds = 2,        // layers: indices 2 .. 10
    kFinalProjectionMicroseconds = 11,  // final projections + similarity
    kAssignmentMicroseconds = 12,
    kQkvGemmMicroseconds = 13,  // summed over the layers
    kAttentionMicroseconds = 14,
    kHvxPassesMicroseconds = 15,
    kLayersRun = 16,
    kFirstLayerTokens = 17,  // tokens (both images) entering each layer: indices 17 .. 25
    kOutputAndFfnMicroseconds = 26,
    kCount = 27
};
}  // namespace stat

}  // namespace lightglue

// The references: per tensor {slot, offset of its blob in the slot, its part offsets} (kReferenceStride int32 each), at
// the indices below. The detector is DaD (VGG-11 encoder, refiners with out.logit), the descriptor DeDoDe-G (VGG-19
// encoder, refiners with out.deltas, DINOv2 blocks in vision_transformer::block_part); the matcher's references are
// LightGlue's.
namespace loma {

constexpr int kReferenceStride = blob_references(vision_transformer::block_part::kCount);  // the most parts: a block
constexpr int kDetectorEncoderLayers = 6;
constexpr int kDetectorScales = 4;
constexpr int kDetectorHiddenBlocks = 3;
constexpr int kDescriptorEncoderLayers = 12;
constexpr int kDescriptorScales = 5;
constexpr int kDescriptorHiddenBlocks = 5;
constexpr int kDinov2Blocks = 24;
constexpr int kDescriptorDimensions = 256;
constexpr int kPatchSize = 14;
constexpr int kDinov2PatchInputs = 256;  // 14 x 14 gray pixels per patch, zero-padded (gemm_fp16: K % 64 == 0)
constexpr int kDinov2Hidden = 1024;

constexpr int refiner_tensors(int hidden_blocks, bool with_context) {
    return 2 + 2 * hidden_blocks + (with_context ? 1 : 0) + 1;
}
namespace refiner_part {  // relative to the refiner's first tensor
constexpr int kBlock1Conv1 = 0;
constexpr int kBlock1Conv2 = 1;
constexpr int depthwise(int block) {
    return 2 + 2 * block;
}
constexpr int pointwise(int block) {
    return 3 + 2 * block;
}
constexpr int context(int hidden_blocks) {
    return 2 + 2 * hidden_blocks;
}
constexpr int deltas(int hidden_blocks, bool with_context) {
    return 2 + 2 * hidden_blocks + (with_context ? 1 : 0);
}
}  // namespace refiner_part

constexpr int detector_encoder(int layer) {
    return layer;
}
constexpr int detector_refiner(int scale_index) {  // 0: scale 8 .. 3: scale 1; every refiner but the last has a context
    return kDetectorEncoderLayers + scale_index * refiner_tensors(kDetectorHiddenBlocks, true);
}
constexpr int detector_context_quantizer(int scale_index) {  // the context entering refiner scale_index (1 .. 3)
    return detector_refiner(kDetectorScales - 1) + refiner_tensors(kDetectorHiddenBlocks, false) + scale_index - 1;
}
constexpr int kDetectorTensors = detector_context_quantizer(kDetectorScales);

constexpr int descriptor_encoder(int layer) {
    return kDetectorTensors + layer;
}
constexpr int descriptor_refiner(int scale_index) {  // 0: scale 14 .. 4: scale 1
    return descriptor_encoder(kDescriptorEncoderLayers) + scale_index * refiner_tensors(kDescriptorHiddenBlocks, true);
}
constexpr int descriptor_context_quantizer(int scale_index) {  // 1 .. 4
    return descriptor_refiner(kDescriptorScales - 1) + refiner_tensors(kDescriptorHiddenBlocks, false) + scale_index -
           1;
}
constexpr int kDinov2OutputQuantizer = descriptor_context_quantizer(kDescriptorScales);
constexpr int dinov2_block(int block) {
    return kDinov2OutputQuantizer + 1 + block;
}
constexpr int kDinov2Tail = dinov2_block(kDinov2Blocks);
constexpr int kTensors = kDinov2Tail + 1;
constexpr int kReferences = kTensors * kReferenceStride;

namespace convolution_part {
enum : int { kWeights = 0, kCount };
}  // namespace convolution_part
namespace depthwise_part {
enum : int { kWeights = 0, kInitial, kMultipliers, kBiases, kZeroPoints, kCount };
}  // namespace depthwise_part
namespace quantizer_part {
enum : int { kReciprocalScales = 0, kOffsets, kCount };
}  // namespace quantizer_part
namespace dinov2_tail_part {
enum : int { kNormGamma = 0, kNormBeta, kClassToken, kPatchEmbedding, kCount };
}  // namespace dinov2_tail_part

namespace stat {
enum : int {
    kKeypoints = 0,
    kDetectorEncoderMicroseconds = 1,  // the VGG-11 convolutions and pools
    kDetectorBlock1Microseconds = 2,   // the refiners' parts, summed over the scales
    kDetectorHiddenMicroseconds = 3,
    kDetectorOutMicroseconds = 4,
    kDetectorContextMicroseconds = 5,  // contexts into the next scale, logits
    kSoftmaxMicroseconds = 6,          // softmax and NMS
    kSelectionMicroseconds = 7,
    kPositionMicroseconds = 8,
    kDescriptorEncoderMicroseconds = 9,  // the VGG-19 convolutions and pools
    kDinov2Microseconds = 10,
    kDescriptorBlock1Microseconds = 11,
    kDescriptorHiddenMicroseconds = 12,
    kDescriptorOutMicroseconds = 13,
    kDescriptorContextMicroseconds = 14,  // contexts, the dense description sums
    kSparseMicroseconds = 15,             // the deltas at the samples' pixels, the descriptions at the keypoints
    kTotalMicroseconds = 16,
    kDetectorDepthwiseMicroseconds = 17,  // part of kDetectorHiddenMicroseconds: the depthwise layers
    kDescriptorDepthwiseMicroseconds = 18,
    kDinov2BlockMicroseconds = 19,  // DINOv2's blocks: block_stat::kCount stages
    kCount = kDinov2BlockMicroseconds + block_stat::kCount
};
}  // namespace stat

}  // namespace loma

}  // namespace benzene::protocol
