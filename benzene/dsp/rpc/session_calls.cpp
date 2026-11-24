#include <cstdint>

#include <AEEStdErr.h>
#include <HAP_perf.h>
#include <qurt.h>

#include "dsp/rpc/rpc_helpers.h"

using benzene::rpc::session_from_handle;
using benzene::runtime::aee_code;
using benzene::runtime::Session;

namespace {

constexpr uint64_t kClockMeasurementMicroseconds = 20000;

}  // namespace

AEEResult benzene_open(const char *, remote_handle64 * handle) {
    benzene::runtime::Result<Session::Owner> session = Session::open();
    if (!session) return aee_code(session.error());
    // the handle owns the session until benzene_close
    *handle = static_cast<remote_handle64>(reinterpret_cast<uintptr_t>(session->release()));
    return AEE_SUCCESS;
}

AEEResult benzene_close(remote_handle64 handle) {
    const Session::Owner session(session_from_handle(handle));
    if (!session) return AEE_EBADPARM;
    return AEE_SUCCESS;
}

AEEResult benzene_info(remote_handle64 handle, uint32_t * architecture, uint32_t * vtcm_bytes, uint32_t * hvx_units,
                       uint32_t * hmx_units, uint32_t * pcycle_khz) {
    const Session * session = session_from_handle(handle);
    if (session == nullptr) return AEE_EBADPARM;
    *architecture = __HEXAGON_ARCH__;
    *vtcm_bytes = session->vtcm_bytes();
    *hvx_units = (static_cast<uint32_t>(qurt_hvx_get_units()) >> 8) & 0xff;  // 128-byte contexts
    *hmx_units = 1;

    const uint64_t start_microseconds = HAP_perf_get_time_us();
    const uint64_t start_pcycles = HAP_perf_get_pcycles();
    while (HAP_perf_get_time_us() - start_microseconds < kClockMeasurementMicroseconds) {}
    const uint64_t end_microseconds = HAP_perf_get_time_us();
    const uint64_t end_pcycles = HAP_perf_get_pcycles();
    *pcycle_khz = static_cast<uint32_t>((end_pcycles - start_pcycles) * 1000 / (end_microseconds - start_microseconds));
    return AEE_SUCCESS;
}

AEEResult benzene_map_model(remote_handle64 handle, int32_t file_descriptor, uint32_t bytes, uint32_t slot) {
    Session * session = session_from_handle(handle);
    if (session == nullptr) return AEE_EBADPARM;
    return aee_code(session->map_model(static_cast<int>(slot), file_descriptor, bytes));
}

AEEResult benzene_unmap_model(remote_handle64 handle, uint32_t slot) {
    Session * session = session_from_handle(handle);
    if (session == nullptr) return AEE_EBADPARM;
    return aee_code(session->unmap_model(static_cast<int>(slot)));
}
