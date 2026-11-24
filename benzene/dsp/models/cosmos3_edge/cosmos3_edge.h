#pragma once

#include <cstdint>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"
#include "benzene/protocol.h"

namespace benzene::models::cosmos3_edge {

// The references, sizes and statistics are protocol::cosmos3_edge's. The language model calls (run_prefill,
// run_decode_step, initialize_kv_cache, prepare_workspaces) run one at a time, and so do the run_vision_encoder calls
// (a second caller gets kBadParameter).
struct PrefillArguments {
    const int32_t * references = nullptr;  // kLlmLayerReferences per layer
    int layers = 0;
    const float * x = nullptr;     // the embedded prompt [tokens][2048]
    const float * rope = nullptr;  // cosines [tokens][64], then sines [tokens][64]
    int tokens = 0;
    bool all_rows = false;
    float * y = nullptr;             // all hidden states [tokens][2048], or (all_rows false) the last row only
    int32_t * statistics = nullptr;  // llm_prefill_stat
};

// The prefill: x through the layers of references into y. The layers within the KV cache's layers fill it with keys
// [0, tokens).
runtime::Status run_prefill(runtime::Session & session, const PrefillArguments & arguments);

struct DecodeStepArguments {
    const int32_t * references = nullptr;  // kLlmLayerReferences per layer
    int layers = 0;
    const int32_t * head = nullptr;  // llm_head_reference
    // Unless nullptr, the 4-bit weights replace the int8 GEMM streams and the head's stream / scales:
    // kLlmLayer4BitReferences per layer (layers_4bit >= layers), then {slot, offset} of the lm_head stream.
    const int32_t * references_4bit = nullptr;
    int layers_4bit = 0;
    const float * x = nullptr;       // the new token's embedding [2048]
    const float * rope = nullptr;    // cosines [64], then sines [64]
    float * logits = nullptr;        // [head[llm_head_reference::kColumns]]
    int32_t * statistics = nullptr;  // llm_decode_stat
};

// One decode step: x at the KV cache's next key position through the layers of references (each appends its key /
// value to the KV cache), then the final norm (norm gamma = weight * inv_s(lm_head)) and the lm_head GEMV into logits.
// No layers: only the head (on x).
runtime::Status run_decode_step(runtime::Session & session, const DecodeStepArguments & arguments);

struct VisionEncoderArguments {
    const int32_t * references = nullptr;  // kVisionLayerReferences per layer
    int layers = 0;
    const int32_t * tail = nullptr;           // kVisionTailReferences
    const float * x = nullptr;                // [patches][1152] patch + position embeddings
    const int32_t * image_offsets = nullptr;  // image i is patches [image_offsets[i], image_offsets[i + 1])
    int images = 0;
    float * y = nullptr;             // [patches / 4][2048]
    int32_t * statistics = nullptr;  // vision_stat
};

// The encoder: x through the layers, then the tail, which merges each 2 x 2 block of 4 consecutive patches into a token
// of y. Images run in groups of whole images of up to 2048 patches (no attention across images).
runtime::Status run_vision_encoder(runtime::Session & session, const VisionEncoderArguments & arguments);

// Allocates the session's KV cache, empty, for `layers` layers of up to `capacity` keys (capacity 0: frees it).
runtime::Status initialize_kv_cache(runtime::Session & session, int layers, int capacity);

// Allocates the prefill (prefill_tokens > 0) and decode workspaces ahead of their first use, also while
// run_vision_encoder runs.
runtime::Status prepare_workspaces(runtime::Session & session, int prefill_tokens);

}  // namespace benzene::models::cosmos3_edge
