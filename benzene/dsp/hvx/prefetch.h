#pragma once

#include <cstdint>

#include <hexagon_protos.h>

namespace benzene::hvx {

// Asynchronous: a new l2fetch on the same thread replaces the one still running.
inline void prefetch_to_l2(const void * address, uint32_t width, uint32_t stride, uint32_t rows) {
    const uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, rows));
    Q6_l2fetch_AP(const_cast<void *>(address), control);
}

inline void prefetch_to_l1(const void * address) {
    Q6_dcfetch_A(const_cast<void *>(address));
}

}  // namespace benzene::hvx
