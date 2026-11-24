#include <cstdint>

#include <AEEStdErr.h>

#include "dsp/models/aliked/aliked.h"
#include "dsp/models/superpoint/superpoint.h"
#include "dsp/models/xfeat/xfeat.h"
#include "dsp/rpc/rpc_helpers.h"
#include "benzene/protocol.h"

namespace {

using benzene::rpc::clear_statistics;
using benzene::rpc::session_from_handle;
using benzene::runtime::aee_code;
using benzene::runtime::Session;

// The three extractors' calls: the same arguments, the model's sizes.
template <class Arguments, benzene::runtime::Status (*kRun)(Session &, const Arguments &), int kReferences,
          int kDescriptorDimensions, int kStatistics>
AEEResult extract(remote_handle64 handle, const int32_t * references, int references_length, const uint8_t * image,
                  int image_length, uint32_t height, uint32_t width, uint32_t max_keypoints, float threshold,
                  uint32_t feature_slot, float * keypoints, int keypoints_length, float * scores, int scores_length,
                  uint16_t * descriptors, int descriptors_length, int32_t * statistics, int statistics_length) {
    Session * session = session_from_handle(handle);
    if (session == nullptr || references_length != kReferences || image_length != static_cast<int>(height * width) ||
        keypoints_length != static_cast<int>(2 * max_keypoints) || scores_length != static_cast<int>(max_keypoints) ||
        descriptors_length != static_cast<int>(kDescriptorDimensions * max_keypoints) ||
        statistics_length < kStatistics) {
        return AEE_EBADPARM;
    }

    clear_statistics(statistics, statistics_length);
    const Arguments arguments{.references = references,
                              .image = image,
                              .height = static_cast<int>(height),
                              .width = static_cast<int>(width),
                              .max_keypoints = static_cast<int>(max_keypoints),
                              .threshold = threshold,
                              .slot = static_cast<int>(feature_slot),
                              .keypoints = keypoints,
                              .scores = scores,
                              .descriptors = reinterpret_cast<_Float16 *>(descriptors),
                              .statistics = statistics};
    return aee_code(kRun(*session, arguments));
}

}  // namespace

namespace protocol = benzene::protocol;
namespace models = benzene::models;

AEEResult benzene_superpoint(remote_handle64 handle, const int32_t * references, int references_length,
                             const uint8_t * image, int image_length, uint32_t height, uint32_t width,
                             uint32_t max_keypoints, float threshold, uint32_t feature_slot, float * keypoints,
                             int keypoints_length, float * scores, int scores_length, uint16_t * descriptors,
                             int descriptors_length, int32_t * statistics, int statistics_length) {
    return extract<models::superpoint::Arguments, models::superpoint::run, protocol::superpoint::kReferences,
                   protocol::superpoint::kDescriptorDimensions, protocol::superpoint::stat::kCount>(
        handle, references, references_length, image, image_length, height, width, max_keypoints, threshold,
        feature_slot, keypoints, keypoints_length, scores, scores_length, descriptors, descriptors_length, statistics,
        statistics_length);
}

AEEResult benzene_xfeat(remote_handle64 handle, const int32_t * references, int references_length,
                        const uint8_t * image, int image_length, uint32_t height, uint32_t width,
                        uint32_t max_keypoints, float threshold, uint32_t feature_slot, float * keypoints,
                        int keypoints_length, float * scores, int scores_length, uint16_t * descriptors,
                        int descriptors_length, int32_t * statistics, int statistics_length) {
    return extract<models::xfeat::Arguments, models::xfeat::run, protocol::xfeat::kReferences,
                   protocol::xfeat::kDescriptorDimensions, protocol::xfeat::stat::kCount>(
        handle, references, references_length, image, image_length, height, width, max_keypoints, threshold,
        feature_slot, keypoints, keypoints_length, scores, scores_length, descriptors, descriptors_length, statistics,
        statistics_length);
}

AEEResult benzene_aliked(remote_handle64 handle, const int32_t * references, int references_length,
                         const uint8_t * image, int image_length, uint32_t height, uint32_t width,
                         uint32_t max_keypoints, float threshold, uint32_t feature_slot, float * keypoints,
                         int keypoints_length, float * scores, int scores_length, uint16_t * descriptors,
                         int descriptors_length, int32_t * statistics, int statistics_length) {
    return extract<models::aliked::Arguments, models::aliked::run, protocol::aliked::kReferences,
                   protocol::aliked::kDescriptorDimensions, protocol::aliked::stat::kCount>(
        handle, references, references_length, image, image_length, height, width, max_keypoints, threshold,
        feature_slot, keypoints, keypoints_length, scores, scores_length, descriptors, descriptors_length, statistics,
        statistics_length);
}
