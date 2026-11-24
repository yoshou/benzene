#include "dsp/models/shared/feature_slots.h"

#include "benzene/protocol.h"

namespace benzene::models {
namespace {

struct FeatureSlots {
    Features slots[protocol::kFeatureSlots];
};

}  // namespace

Features * feature_slot(runtime::Session & session, int slot) {
    if (slot < 0 || slot >= protocol::kFeatureSlots) return nullptr;
    FeatureSlots * slots = session.state<FeatureSlots>();
    return slots != nullptr ? &slots->slots[slot] : nullptr;
}

runtime::Status reserve_features(Features * features, int keypoints, int dimensions) {
    features->dimensions = dimensions;
    const int halfwords = keypoints * dimensions;
    if (features->capacity >= keypoints && features->descriptor_capacity >= halfwords) return {};

    features->keypoints.reset();  // both freed before the larger ones are allocated
    features->descriptors.reset();
    features->keypoints = runtime::allocate_array<float>(static_cast<size_t>(keypoints) * 2);
    features->descriptors = runtime::allocate_array<_Float16>(halfwords);
    const bool allocated = features->keypoints && features->descriptors;
    features->capacity = allocated ? keypoints : 0;
    features->descriptor_capacity = allocated ? halfwords : 0;
    if (!allocated) return std::unexpected(runtime::Error::kNoMemory);
    return {};
}

}  // namespace benzene::models
