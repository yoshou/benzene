#pragma once

#include <cstddef>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"

namespace benzene::runtime {

constexpr int kHvxThreads = 4;

// A job's per-thread function: thread 0 is the calling thread.
using ThreadFunction = void (*)(void * job, int thread);
// Must never execute an HVX instruction: while the four workers hold the HVX contexts, that blocks the whole process.
using DriverFunction = void (*)(void * job);
using ItemFunction = void (*)(void * context, int item, int thread);

// Runs worker(job, i) on threads i = 0 .. thread_count - 1 (at most 8) and returns after all of them finished. Up to
// four threads run on the calling thread and the session's persistent workers (started at the first call, joined when
// the session closes); more threads, or a nested or concurrent call, run on threads created for the call.
Status run_parallel(Session & session, int thread_count, ThreadFunction worker, void * job);

// run_parallel with driver(job) running on one more thread at the same time. One call at a time: kBadParameter while
// another runs.
Status run_parallel_with_driver(Session & session, int thread_count, ThreadFunction worker, DriverFunction driver,
                                void * job);

// function(context, item, thread) for every item in [0, item_count), items taken dynamically by the threads.
Status parallel_for(Session & session, int thread_count, int item_count, ItemFunction function, void * context);

// libc memcpy below 1 MiB, else copy_memory_parallel.
Status copy_memory(Session & session, void * destination, const void * source, size_t bytes);
Status copy_memory_parallel(Session & session, void * destination, const void * source, size_t bytes);

}  // namespace benzene::runtime
