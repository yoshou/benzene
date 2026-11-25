#include <cstdint>

#include <AEEStdErr.h>

#include "dsp/models/vggt/vggt.h"
#include "dsp/rpc/rpc_helpers.h"
#include "benzene/protocol.h"

using benzene::rpc::clear_statistics;
using benzene::rpc::session_from_handle;
using benzene::runtime::aee_code;
using benzene::runtime::Session;
namespace protocol = benzene::protocol::vggt;
namespace vggt = benzene::models::vggt;

AEEResult benzene_vggt_transformer(remote_handle64 handle, const int32_t * references, int references_length,
                                   const int32_t * tail, int tail_length, const uint16_t * patches, int patches_length,
                                   const float * positions, int positions_length, const float * special_tokens,
                                   int special_tokens_length, uint32_t frames, uint32_t patch_rows,
                                   uint32_t patch_columns, uint16_t * kept_outputs, int kept_outputs_length,
                                   int32_t * statistics, int statistics_length) {
    Session * session = session_from_handle(handle);
    const auto tokens_per_frame = static_cast<int>(protocol::kSpecialTokens + patch_rows * patch_columns);
    const auto kept_values =
        static_cast<int>(protocol::kKeptOutputs * frames * tokens_per_frame * protocol::kKeptOutputWidth);
    if (session == nullptr || references_length % protocol::kBlockReferences != 0 ||
        tail_length != protocol::kTailReferences ||
        patches_length != static_cast<int>(frames * patch_rows * patch_columns * protocol::kPatchInputs) ||
        positions_length != static_cast<int>(patch_rows * patch_columns * protocol::kHidden) ||
        special_tokens_length != protocol::kSpecialTokens * protocol::kHidden ||
        statistics_length < protocol::transformer_stat::kCount ||
        (kept_outputs_length != 0 && kept_outputs_length != kept_values)) {
        return AEE_EBADPARM;
    }

    clear_statistics(statistics, statistics_length);
    const vggt::TransformerArguments arguments{
        .references = references,
        .blocks = references_length / protocol::kBlockReferences,
        .tail = tail,
        .patches = reinterpret_cast<const _Float16 *>(patches),
        .positions = positions,
        .special_tokens = special_tokens,
        .frames = static_cast<int>(frames),
        .tokens_per_frame = tokens_per_frame,
        .patch_rows = static_cast<int>(patch_rows),
        .patch_columns = static_cast<int>(patch_columns),
        .kept_outputs = kept_outputs_length != 0 ? reinterpret_cast<_Float16 *>(kept_outputs) : nullptr,
        .statistics = statistics,
    };
    return aee_code(vggt::run_transformer(*session, arguments));
}

AEEResult benzene_vggt_camera_head(remote_handle64 handle, const int32_t * references, int references_length,
                                   float * pose, int pose_length, int32_t * statistics, int statistics_length) {
    Session * session = session_from_handle(handle);
    if (session == nullptr || references_length != protocol::kCameraReferences || pose_length == 0 ||
        pose_length != protocol::kCameraIterations * vggt::kept_geometry(*session).frames * protocol::kPoseValues ||
        statistics_length < protocol::camera_stat::kCount) {
        return AEE_EBADPARM;
    }

    clear_statistics(statistics, statistics_length);
    const vggt::CameraHeadArguments arguments{.references = references, .pose = pose, .statistics = statistics};
    return aee_code(vggt::run_camera_head(*session, arguments));
}

AEEResult benzene_vggt_dpt_head(remote_handle64 handle, const int32_t * references, int references_length,
                                uint32_t channels, const float * position_tables, int position_tables_length, float * y,
                                int y_length, int32_t * statistics, int statistics_length) {
    Session * session = session_from_handle(handle);
    if (session == nullptr || references_length != protocol::kDptReferences ||
        statistics_length < benzene::protocol::dpt_head::stat::kCount) {
        return AEE_EBADPARM;
    }
    const vggt::KeptGeometry geometry = vggt::kept_geometry(*session);
    if (geometry.frames == 0 ||
        y_length != geometry.frames * geometry.height * geometry.width * static_cast<int>(channels)) {
        return AEE_EBADPARM;
    }

    clear_statistics(statistics, statistics_length);
    const vggt::DptHeadArguments arguments{
        .references = references,
        .position_tables = position_tables,
        .position_values = position_tables_length,
        .channels = static_cast<int>(channels),
        .out = y,
        .statistics = statistics,
    };
    return aee_code(vggt::run_dpt_head(*session, arguments));
}
