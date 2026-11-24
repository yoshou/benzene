#include "dsp/runtime/thread_pool.h"

#include <atomic>
#include <cstdint>
#include <cstring>

#include <qurt.h>
#include <qurt_futex.h>

#include "dsp/hvx/prefetch.h"
#include "dsp/hvx/vector.h"
#include "dsp/runtime/heap.h"

namespace benzene::runtime {
namespace {

constexpr int kMaximumSpawnedThreads = 8;
constexpr int kStackBytes = 65536;  // fused GEMM output stages spill HVX registers
constexpr int kStackAlignment = 128;
constexpr int kPersistentWorkers = kHvxThreads - 1;

constexpr int kIdleSpinIterations = 4000;

HeapArray<uint8_t> allocate_stack() {
    return allocate_array<uint8_t>(kStackBytes, kStackAlignment);
}

Status create_thread(qurt_thread_t * thread, uint8_t * stack, const char * name, void (*entry)(void *),
                     void * argument) {
    qurt_thread_attr_t attributes;
    qurt_thread_attr_init(&attributes);
    qurt_thread_attr_set_stack_addr(&attributes, stack);
    qurt_thread_attr_set_stack_size(&attributes, kStackBytes);
    qurt_thread_attr_set_priority(&attributes, qurt_thread_get_priority(qurt_thread_get_id()));
    qurt_thread_attr_set_name(&attributes, const_cast<char *>(name));
    if (qurt_thread_create(thread, &attributes, entry, argument) != QURT_EOK) return std::unexpected(Error::kFailed);
    return {};
}

// Spins, then sleeps on the futex counted in `sleepers`: the publisher wakes the futex only when sleepers != 0.
int wait_for_next_generation(std::atomic<int> & generation, std::atomic<int> & sleepers, int seen) {
    int spins = 0;
    while (true) {
        const int current = generation.load();
        if (current != seen) return current;

        spins++;
        if (spins > kIdleSpinIterations) {
            sleepers.fetch_add(1);  // before the futex's own check of generation: no lost wake
            qurt_futex_wait(&generation, seen);
            sleepers.fetch_sub(1);
            spins = 0;
        }
    }
}

struct SpawnedThread {
    ThreadFunction worker;
    void * job;
    int thread_index;
    qurt_thread_t handle;
    HeapArray<uint8_t> stack;
};

void spawned_thread_main(void * argument) {
    auto * thread = static_cast<SpawnedThread *>(argument);
    thread->worker(thread->job, thread->thread_index);
    qurt_thread_exit(0);
}

Status run_on_spawned_threads(int thread_count, ThreadFunction worker, void * job) {
    SpawnedThread threads[kMaximumSpawnedThreads];
    if (thread_count < 1) thread_count = 1;
    if (thread_count > kMaximumSpawnedThreads) thread_count = kMaximumSpawnedThreads;

    for (int index = 1; index < thread_count; index++) {
        SpawnedThread & thread = threads[index];
        thread = SpawnedThread{worker, job, index, 0, allocate_stack()};
        const Status created =
            thread.stack ? create_thread(&thread.handle, thread.stack.get(), "benzene", spawned_thread_main, &thread)
                         : std::unexpected(Error::kNoMemory);
        if (!created) {  // the threads created so far keep running on their stacks: those are left allocated
            for (int started = 1; started < index; started++) static_cast<void>(threads[started].stack.release());
            return std::unexpected(Error::kFailed);
        }
    }

    worker(job, 0);
    for (int index = 1; index < thread_count; index++) {
        int status;
        qurt_thread_join(threads[index].handle, &status);
    }
    return {};
}

struct WorkerPool;

struct PoolWorker {
    WorkerPool * pool;
    int thread_index;
};

// Thread indices 1 .. 3 (the caller is 0); a job is published by incrementing `generation`.
struct WorkerPool {
    qurt_thread_t threads[kPersistentWorkers];
    HeapArray<uint8_t> stacks[kPersistentWorkers];
    PoolWorker workers[kPersistentWorkers];
    ThreadFunction worker;
    void * job;
    int thread_count;
    int quit;
    int started;
    std::atomic<int> generation;
    std::atomic<int> finished;
    std::atomic<int> busy;
    std::atomic<int> sleepers;
};

void pool_worker_main(void * argument) {
    const auto * pool_worker = static_cast<const PoolWorker *>(argument);
    WorkerPool & pool = *pool_worker->pool;
    const int thread_index = pool_worker->thread_index;
    int seen = 0;
    while (true) {
        seen = wait_for_next_generation(pool.generation, pool.sleepers, seen);
        if (pool.quit) break;
        if (thread_index < pool.thread_count) pool.worker(pool.job, thread_index);
        pool.finished.fetch_add(1);
    }
    qurt_thread_exit(0);
}

Status start_pool(WorkerPool & pool) {
    for (int index = 0; index < kPersistentWorkers; index++) {
        pool.stacks[index] = allocate_stack();
        if (!pool.stacks[index]) return std::unexpected(Error::kNoMemory);
        pool.workers[index] = {&pool, index + 1};
        const Status created = create_thread(&pool.threads[index], pool.stacks[index].get(), "benzene_worker",
                                             pool_worker_main, &pool.workers[index]);
        if (!created) return created;
        pool.started = index + 1;
    }
    return {};
}

void stop_pool(WorkerPool & pool) {
    if (pool.started == 0) return;
    pool.quit = 1;
    pool.generation.fetch_add(1);
    qurt_futex_wake(&pool.generation, kPersistentWorkers);
    for (int index = 0; index < pool.started; index++) {
        int status;
        qurt_thread_join(pool.threads[index], &status);
    }
}

struct DriverThread {
    qurt_thread_t thread;
    HeapArray<uint8_t> stack;
    DriverFunction driver;
    void * job;
    int quit;
    int started;
    std::atomic<int> generation;
    std::atomic<int> finished;
    std::atomic<int> busy;
    std::atomic<int> sleepers;
};

void driver_thread_main(void * argument) {
    DriverThread & driver_thread = *static_cast<DriverThread *>(argument);
    int seen = 0;
    while (true) {
        seen = wait_for_next_generation(driver_thread.generation, driver_thread.sleepers, seen);
        if (driver_thread.quit) break;
        driver_thread.driver(driver_thread.job);
        driver_thread.finished.fetch_add(1);
    }
    qurt_thread_exit(0);
}

Status start_driver_thread(DriverThread & driver_thread) {
    driver_thread.stack = allocate_stack();
    if (!driver_thread.stack) return std::unexpected(Error::kNoMemory);
    const Status created = create_thread(&driver_thread.thread, driver_thread.stack.get(), "benzene_driver",
                                         driver_thread_main, &driver_thread);
    if (!created) {
        driver_thread.stack.reset();
        return created;
    }
    driver_thread.started = 1;
    return {};
}

void stop_driver_thread(DriverThread & driver_thread) {
    if (driver_thread.started == 0) return;
    driver_thread.quit = 1;
    driver_thread.generation.fetch_add(1);
    qurt_futex_wake(&driver_thread.generation, 1);
    int status;
    qurt_thread_join(driver_thread.thread, &status);
}

struct ThreadPool {
    WorkerPool pool;
    DriverThread driver_thread;

    ~ThreadPool() {
        stop_pool(pool);
        stop_driver_thread(driver_thread);
    }
};

struct ParallelLoop {
    ItemFunction function;
    void * context;
    int item_count;
    std::atomic<int> next_item;
};

void parallel_loop_thread(void * job, int thread) {
    auto * loop = static_cast<ParallelLoop *>(job);
    while (true) {
        const int item = loop->next_item.fetch_add(1);
        if (item >= loop->item_count) return;
        loop->function(loop->context, item, thread);
    }
}

constexpr size_t kCopyChunkBytes = 65536;
constexpr size_t kParallelCopyThreshold = size_t{1} << 20;
constexpr size_t kPrefetchRowBytes = 4096;

struct CopyJob {
    uint8_t * destination;
    const uint8_t * source;
    size_t bytes;
};

void copy_chunk(void * context, int chunk, int) {
    const auto * copy = static_cast<const CopyJob *>(context);
    const size_t offset = static_cast<size_t>(chunk) * kCopyChunkBytes;
    const size_t remaining = copy->bytes - offset;
    const size_t bytes = remaining < kCopyChunkBytes ? remaining : kCopyChunkBytes;
    const size_t vectors = bytes / 128;

    hvx::prefetch_to_l2(copy->source + offset, kPrefetchRowBytes, kPrefetchRowBytes,
                        static_cast<uint32_t>((bytes + kPrefetchRowBytes - 1) / kPrefetchRowBytes));

    uint8_t * destination = copy->destination + offset;
    const uint8_t * source = copy->source + offset;
    for (size_t k = 0; k < vectors; k++) store_unaligned(destination + 128 * k, hvx::load_unaligned(source + 128 * k));

    const size_t tail_offset = offset + vectors * 128;
    if (bytes % 128 != 0) memcpy(copy->destination + tail_offset, copy->source + tail_offset, bytes % 128);
}

}  // namespace

Status run_parallel(Session & session, int thread_count, ThreadFunction worker, void * job) {
    if (thread_count < 1) thread_count = 1;
    if (thread_count == 1) {
        worker(job, 0);
        return {};
    }

    ThreadPool * threads = session.state<ThreadPool>();
    int idle = 0;
    if (threads == nullptr || thread_count > kHvxThreads || !threads->pool.busy.compare_exchange_strong(idle, 1)) {
        return run_on_spawned_threads(thread_count, worker, job);
    }

    WorkerPool & pool = threads->pool;
    if (pool.started < kPersistentWorkers && (pool.started != 0 || !start_pool(pool))) {
        pool.busy.store(0);
        return run_on_spawned_threads(thread_count, worker, job);
    }

    pool.worker = worker;
    pool.job = job;
    pool.thread_count = thread_count;
    pool.finished.store(0);
    pool.generation.fetch_add(1);  // publishes worker / job / thread_count (seq_cst)
    if (pool.sleepers.load() != 0) qurt_futex_wake(&pool.generation, kPersistentWorkers);

    worker(job, 0);
    while (pool.finished.load() < kPersistentWorkers) {}
    pool.busy.store(0);
    return {};
}

Status run_parallel_with_driver(Session & session, int thread_count, ThreadFunction worker, DriverFunction driver,
                                void * job) {
    ThreadPool * threads = session.state<ThreadPool>();
    if (threads == nullptr) return std::unexpected(Error::kNoMemory);

    DriverThread & driver_thread = threads->driver_thread;
    int idle = 0;
    if (!driver_thread.busy.compare_exchange_strong(idle, 1)) return std::unexpected(Error::kBadParameter);
    if (driver_thread.started == 0 && !start_driver_thread(driver_thread)) {
        driver_thread.busy.store(0);
        return std::unexpected(Error::kFailed);
    }

    driver_thread.driver = driver;
    driver_thread.job = job;
    driver_thread.finished.store(0);
    driver_thread.generation.fetch_add(1);  // publishes driver / job
    if (driver_thread.sleepers.load() != 0) qurt_futex_wake(&driver_thread.generation, 1);

    const Status status = run_parallel(session, thread_count, worker, job);
    while (driver_thread.finished.load() == 0) {}
    driver_thread.busy.store(0);
    return status;
}

Status parallel_for(Session & session, int thread_count, int item_count, ItemFunction function, void * context) {
    ParallelLoop loop{function, context, item_count, 0};
    return run_parallel(session, thread_count, parallel_loop_thread, &loop);
}

Status copy_memory(Session & session, void * destination, const void * source, size_t bytes) {
    if (bytes < kParallelCopyThreshold) {
        memcpy(destination, source, bytes);
        return {};
    }
    return copy_memory_parallel(session, destination, source, bytes);
}

Status copy_memory_parallel(Session & session, void * destination, const void * source, size_t bytes) {
    CopyJob copy{static_cast<uint8_t *>(destination), static_cast<const uint8_t *>(source), bytes};
    const int chunks = static_cast<int>((bytes + kCopyChunkBytes - 1) / kCopyChunkBytes);
    return parallel_for(session, kHvxThreads, chunks, copy_chunk, &copy);
}

}  // namespace benzene::runtime
