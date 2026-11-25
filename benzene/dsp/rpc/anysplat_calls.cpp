#include <cstdint>

#include <AEEStdErr.h>

#include "dsp/models/anysplat/anysplat.h"
#include "dsp/models/vggt/vggt.h"
#include "dsp/rpc/rpc_helpers.h"
#include "benzene/protocol.h"

using benzene::rpc::clear_statistics;
using benzene::rpc::session_from_handle;
using benzene::runtime::aee_code;
using benzene::runtime::Session;
namespace protocol = benzene::protocol::anysplat;
namespace anysplat = benzene::models::anysplat;
namespace vggt = benzene::models::vggt;

AEEResult benzene_anysplat_gaussian_head(remote_handle64 handle, const int32_t * references, int references_length,
                                         const float * position_tables, int position_tables_length, uint32_t frame,
                                         const uint8_t * image, int image_length, uint16_t * y, int y_length,
                                         int32_t * statistics, int statistics_length) {
    Session * session = session_from_handle(handle);
    if (session == nullptr || references_length != protocol::kGaussianReferences ||
        statistics_length < protocol::gaussian_stat::kCount) {
        return AEE_EBADPARM;
    }
    const vggt::KeptGeometry geometry = vggt::kept_geometry(*session);
    const int pixels = geometry.height * geometry.width;
    if (geometry.frames == 0 || frame >= static_cast<uint32_t>(geometry.frames) || image_length != 3 * pixels ||
        y_length != pixels * protocol::kGaussianOutputChannels) {
        return AEE_EBADPARM;
    }

    clear_statistics(statistics, statistics_length);
    const anysplat::GaussianHeadArguments arguments{
        .references = references,
        .position_tables = position_tables,
        .position_values = position_tables_length,
        .frame = static_cast<int>(frame),
        .image = image,
        .out = reinterpret_cast<_Float16 *>(y),
        .statistics = statistics,
    };
    return aee_code(anysplat::run_gaussian_head(*session, arguments));
}
