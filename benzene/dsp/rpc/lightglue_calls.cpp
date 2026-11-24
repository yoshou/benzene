#include <cstdint>

#include <AEEStdErr.h>

#include "dsp/models/lightglue/lightglue.h"
#include "dsp/models/shared/feature_slots.h"
#include "dsp/rpc/rpc_helpers.h"
#include "benzene/protocol.h"

using benzene::rpc::clear_statistics;
using benzene::rpc::session_from_handle;
using benzene::runtime::aee_code;
using benzene::runtime::Session;
namespace lightglue = benzene::models::lightglue;
namespace protocol = benzene::protocol::lightglue;

AEEResult benzene_lightglue(remote_handle64 handle, int32_t model, const int32_t * references, int references_length,
                            uint32_t first_slot, uint32_t second_slot, float threshold, float depth_confidence,
                            float width_confidence, int32_t * matches, int matches_length, float * match_scores,
                            int match_scores_length, int32_t * statistics, int statistics_length) {
    Session * session = session_from_handle(handle);
    if (session == nullptr || !protocol::known_model(model) ||
        references_length != protocol::references(static_cast<protocol::Model>(model)) ||
        statistics_length < protocol::stat::kCount) {
        return AEE_EBADPARM;
    }
    const benzene::models::Features * first = benzene::models::feature_slot(*session, static_cast<int>(first_slot));
    if (first == nullptr || matches_length != first->count || match_scores_length != first->count) {
        return AEE_EBADPARM;
    }

    clear_statistics(statistics, statistics_length);
    const lightglue::Arguments arguments{.model = static_cast<protocol::Model>(model),
                                         .references = references,
                                         .first_slot = static_cast<int>(first_slot),
                                         .second_slot = static_cast<int>(second_slot),
                                         .threshold = threshold,
                                         .depth_confidence = depth_confidence,
                                         .width_confidence = width_confidence,
                                         .matches = matches,
                                         .match_scores = match_scores,
                                         .statistics = statistics};
    return aee_code(lightglue::run(*session, arguments));
}
