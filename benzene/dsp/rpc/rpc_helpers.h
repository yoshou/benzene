#pragma once

#include <cstdint>
#include <cstring>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"
#include "benzene.h"

namespace benzene::rpc {

// The remote handle is the session pointer (benzene_open).
inline runtime::Session * session_from_handle(remote_handle64 handle) {
    return reinterpret_cast<runtime::Session *>(static_cast<uintptr_t>(handle));
}

inline void clear_statistics(int32_t * statistics, int count) {
    memset(statistics, 0, static_cast<size_t>(count) * sizeof(int32_t));
}

}  // namespace benzene::rpc
