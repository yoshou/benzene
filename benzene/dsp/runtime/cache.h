#pragma once

#include <cstddef>

#include <qurt_memory.h>

namespace benzene::runtime {

// Before a call reads an input the host wrote: the host reuses its rpcmem buffers, and lines of an earlier call's
// contents may still be in the DSP's data cache.
inline void invalidate_host_input(const void * buffer, size_t bytes) {
    qurt_mem_cache_clean(reinterpret_cast<qurt_addr_t>(buffer), static_cast<qurt_size_t>(bytes),
                         QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
}

// Writes every dirty line back to DDR, for DMA that bypasses the caches.
inline void flush_data_cache() {
    qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_ALL, QURT_MEM_DCACHE);
}

}  // namespace benzene::runtime
