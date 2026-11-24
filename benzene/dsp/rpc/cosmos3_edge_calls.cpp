#include <cstdint>

#include <AEEStdErr.h>

#include "dsp/models/cosmos3_edge/cosmos3_edge.h"
#include "dsp/rpc/rpc_helpers.h"
#include "benzene/protocol.h"

namespace cosmos3_edge = benzene::models::cosmos3_edge;
namespace protocol = benzene::protocol::cosmos3_edge;
using benzene::rpc::clear_statistics;
using benzene::rpc::session_from_handle;
using benzene::runtime::aee_code;
using benzene::runtime::Session;
using protocol::kLlmHidden;
using protocol::kRopeValuesPerToken;

AEEResult benzene_cosmos3_edge_llm_prefill(remote_handle64 handle, const int32_t * references, int references_length,
                                           const float * x, int x_length, const float * rope, int rope_length,
                                           uint32_t tokens, float * y, int y_length, int32_t * statistics,
                                           int statistics_length) {
    Session * session = session_from_handle(handle);
    const auto token_count = static_cast<int>(tokens);
    if (session == nullptr || references_length % protocol::kLlmLayerReferences != 0 ||
        x_length != token_count * kLlmHidden || (y_length != x_length && y_length != kLlmHidden) ||
        rope_length != token_count * kRopeValuesPerToken || statistics_length < protocol::llm_prefill_stat::kCount) {
        return AEE_EBADPARM;
    }

    clear_statistics(statistics, statistics_length);
    const cosmos3_edge::PrefillArguments arguments{.references = references,
                                                   .layers = references_length / protocol::kLlmLayerReferences,
                                                   .x = x,
                                                   .rope = rope,
                                                   .tokens = token_count,
                                                   .all_rows = y_length == x_length,
                                                   .y = y,
                                                   .statistics = statistics};
    return aee_code(cosmos3_edge::run_prefill(*session, arguments));
}

AEEResult benzene_cosmos3_edge_llm_decode_step(remote_handle64 handle, const int32_t * references,
                                               int references_length, const int32_t * head, int head_length,
                                               const int32_t * references_4bit, int references_4bit_length,
                                               const float * x, int x_length, const float * rope, int rope_length,
                                               float * logits, int logits_length, int32_t * statistics,
                                               int statistics_length) {
    Session * session = session_from_handle(handle);
    const int layers_4bit = (references_4bit_length - 2) / protocol::kLlmLayer4BitReferences;
    if (session == nullptr || references_length % protocol::kLlmLayerReferences != 0 ||
        (references_4bit_length != 0 &&
         references_4bit_length != layers_4bit * protocol::kLlmLayer4BitReferences + 2) ||
        head_length != protocol::llm_head_reference::kCount || x_length != kLlmHidden ||
        rope_length != kRopeValuesPerToken || logits_length != head[protocol::llm_head_reference::kColumns] ||
        statistics_length < protocol::llm_decode_stat::kCount) {
        return AEE_EBADPARM;
    }

    clear_statistics(statistics, statistics_length);
    const cosmos3_edge::DecodeStepArguments arguments{
        .references = references,
        .layers = references_length / protocol::kLlmLayerReferences,
        .head = head,
        .references_4bit = references_4bit_length != 0 ? references_4bit : nullptr,
        .layers_4bit = references_4bit_length != 0 ? layers_4bit : 0,
        .x = x,
        .rope = rope,
        .logits = logits,
        .statistics = statistics};
    return aee_code(cosmos3_edge::run_decode_step(*session, arguments));
}

AEEResult benzene_cosmos3_edge_vision_encoder(remote_handle64 handle, const int32_t * references, int references_length,
                                              const int32_t * tail, int tail_length, const float * x, int x_length,
                                              const int32_t * image_offsets, int image_offsets_length, float * y,
                                              int y_length, int32_t * statistics, int statistics_length) {
    Session * session = session_from_handle(handle);
    const int images = image_offsets_length - 1;
    if (session == nullptr || references_length % protocol::kVisionLayerReferences != 0 ||
        tail_length != protocol::kVisionTailReferences || images < 1 || images > protocol::kMaximumVisionImages ||
        x_length != image_offsets[images] * protocol::kVisionHidden ||
        y_length != image_offsets[images] / protocol::kPatchesPerVisionToken * kLlmHidden ||
        statistics_length < protocol::vision_stat::kCount) {
        return AEE_EBADPARM;
    }

    clear_statistics(statistics, statistics_length);
    const cosmos3_edge::VisionEncoderArguments arguments{.references = references,
                                                         .layers = references_length / protocol::kVisionLayerReferences,
                                                         .tail = tail,
                                                         .x = x,
                                                         .image_offsets = image_offsets,
                                                         .images = images,
                                                         .y = y,
                                                         .statistics = statistics};
    return aee_code(cosmos3_edge::run_vision_encoder(*session, arguments));
}

AEEResult benzene_cosmos3_edge_kv_cache(remote_handle64 handle, uint32_t layers, uint32_t capacity) {
    Session * session = session_from_handle(handle);
    if (session == nullptr) return AEE_EBADPARM;
    return aee_code(cosmos3_edge::initialize_kv_cache(*session, static_cast<int>(layers), static_cast<int>(capacity)));
}

AEEResult benzene_cosmos3_edge_prepare_workspaces(remote_handle64 handle, uint32_t prefill_tokens) {
    Session * session = session_from_handle(handle);
    if (session == nullptr) return AEE_EBADPARM;
    return aee_code(cosmos3_edge::prepare_workspaces(*session, static_cast<int>(prefill_tokens)));
}
