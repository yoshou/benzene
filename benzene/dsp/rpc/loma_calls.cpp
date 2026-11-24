#include <cstdint>

#include <AEEStdErr.h>

#include "dsp/models/loma/loma.h"
#include "dsp/rpc/rpc_helpers.h"
#include "benzene/protocol.h"

using benzene::rpc::clear_statistics;
using benzene::rpc::session_from_handle;
using benzene::runtime::aee_code;
using benzene::runtime::Session;
namespace loma = benzene::models::loma;
namespace protocol = benzene::protocol::loma;

AEEResult benzene_loma(remote_handle64 handle, const int32_t * references, int references_length,
                       const uint8_t * detector_image, int detector_image_length, uint32_t detector_height,
                       uint32_t detector_width, uint32_t detector_zero_point, const uint8_t * descriptor_image,
                       int descriptor_image_length, uint32_t descriptor_height, uint32_t descriptor_width,
                       const uint16_t * patches, int patches_length, const float * positions, int positions_length,
                       uint32_t count, uint32_t feature_slot, float * keypoints, int keypoints_length, float * scores,
                       int scores_length, float * descriptors, int descriptors_length, int32_t * statistics,
                       int statistics_length) {
    Session * session = session_from_handle(handle);
    const int patch_count =
        static_cast<int>((descriptor_height / protocol::kPatchSize) * (descriptor_width / protocol::kPatchSize));
    if (session == nullptr || references_length != protocol::kReferences ||
        detector_image_length != static_cast<int>(detector_height * detector_width) ||
        descriptor_image_length != static_cast<int>(descriptor_height * descriptor_width) ||
        patches_length != patch_count * protocol::kDinov2PatchInputs ||
        positions_length != patch_count * protocol::kDinov2Hidden || keypoints_length != static_cast<int>(2 * count) ||
        scores_length != static_cast<int>(count) ||
        descriptors_length != static_cast<int>(count) * protocol::kDescriptorDimensions ||
        statistics_length < protocol::stat::kCount) {
        return AEE_EBADPARM;
    }

    clear_statistics(statistics, statistics_length);
    const loma::Arguments arguments{.references = references,
                                    .detector_image = detector_image,
                                    .detector_height = static_cast<int>(detector_height),
                                    .detector_width = static_cast<int>(detector_width),
                                    .detector_zero_point = static_cast<int>(detector_zero_point),
                                    .descriptor_image = descriptor_image,
                                    .descriptor_height = static_cast<int>(descriptor_height),
                                    .descriptor_width = static_cast<int>(descriptor_width),
                                    .patches = reinterpret_cast<const _Float16 *>(patches),
                                    .positions = positions,
                                    .count = static_cast<int>(count),
                                    .slot = static_cast<int>(feature_slot),
                                    .keypoints = keypoints,
                                    .scores = scores,
                                    .descriptors = descriptors,
                                    .statistics = statistics};
    return aee_code(loma::run(*session, arguments));
}
