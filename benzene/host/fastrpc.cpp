#include "host/fastrpc.h"

#include <dlfcn.h>
#include <linux/dma-buf.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>

#include <atomic>
#include <climits>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "host/files.h"
#include "remote.h"

namespace benzene::host {
namespace {

remote_handle64 session_handle = 0;
std::atomic<bool> stop_signalled{false};

// libcdsprpc's functions, loaded at run time by open_session.
struct LibraryFunctions {
    int (*handle_open)(const char * name, remote_handle64 * handle);
    int (*handle_invoke)(remote_handle64 handle, uint32_t scalars, remote_arg * arguments);
    int (*handle_close)(remote_handle64 handle);
    int (*session_control)(uint32_t request, void * data, uint32_t length);
    void * (*allocate)(int heap, uint32_t flags, int bytes);
    int (*to_descriptor)(void * buffer);
    int (*map)(int domain, int descriptor, void * address, int offset, size_t bytes, int flags);
    int (*unmap)(int domain, int descriptor, void * address, size_t bytes);
};
LibraryFunctions library;

template <typename Function>
bool load_function(void * handle, const char * name, Function * function) {
    *function = reinterpret_cast<Function>(dlsym(handle, name));
    if (*function == nullptr) fprintf(stderr, "libcdsprpc: missing %s\n", name);
    return *function != nullptr;
}

bool load_library() {
    void * handle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) handle = dlopen("libcdsprpc.so.1", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        fprintf(stderr, "dlopen libcdsprpc: %s\n", dlerror());
        return false;
    }
    return load_function(handle, "remote_handle64_open", &library.handle_open) &&
           load_function(handle, "remote_handle64_invoke", &library.handle_invoke) &&
           load_function(handle, "remote_handle64_close", &library.handle_close) &&
           load_function(handle, "remote_session_control", &library.session_control) &&
           load_function(handle, "rpcmem_alloc", &library.allocate) &&
           load_function(handle, "rpcmem_to_fd", &library.to_descriptor) &&
           load_function(handle, "fastrpc_mmap", &library.map) &&
           load_function(handle, "fastrpc_munmap", &library.unmap);
}

constexpr int kSystemHeap = 25;  // RPCMEM_HEAP_ID_SYSTEM
constexpr uint32_t kCachedFlag = 1;

void pin_to_big_cores() {
    constexpr int kMaximumCpus = 64;
    int capacity[kMaximumCpus];
    int count = 0;
    int maximum = 0;
    for (; count < kMaximumCpus; count++) {
        char path[80];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpu_capacity", count);
        const File file = open_file(path, "r");
        if (!file) break;
        if (fscanf(file.get(), "%d", &capacity[count]) != 1) capacity[count] = 0;
        maximum = capacity[count] > maximum ? capacity[count] : maximum;
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    int big = 0;
    for (int cpu = 0; cpu < count; cpu++) {
        if (capacity[cpu] * 2 > maximum) {
            CPU_SET(cpu, &set);
            big++;
        }
    }
    if (big > 0 && big < count) sched_setaffinity(0, sizeof set, &set);
}

}  // namespace

void install_signal_guard() {
    signal(SIGPIPE, SIG_IGN);

    sigset_t signals;
    sigemptyset(&signals);
    for (const int s : {SIGINT, SIGTERM, SIGHUP, SIGQUIT}) sigaddset(&signals, s);
    pthread_sigmask(SIG_BLOCK, &signals, nullptr);

    std::thread([signals] {
        for (;;) {
            int received = 0;
            if (sigwait(&signals, &received) != 0) continue;
            if (!stop_signalled.exchange(true)) {
                fprintf(stderr, "benzene: signal %d - stopping after the current step\n", received);
            } else {
                fprintf(stderr,
                        "benzene: signal %d - still finishing the current step (SIGKILL would leave this session's DSP "
                        "mappings in place until the board reboots)\n",
                        received);
            }
        }
    }).detach();
}

bool stop_requested() {
    return stop_signalled.load();
}

bool stopped_by_signal() {
    if (!stop_requested()) return false;
    fprintf(stderr, "stopped by a signal\n");
    return true;
}

bool rpc_succeeded(int error, const char * call, ...) {
    if (error == 0) return true;
    va_list arguments;
    va_start(arguments, call);
    vfprintf(stderr, call, arguments);
    va_end(arguments);
    fprintf(stderr, " error 0x%x\n", error);
    return false;
}

bool open_session() {
    pin_to_big_cores();
    if (!load_library()) return false;

    remote_rpc_control_unsigned_module unsigned_module{};
    unsigned_module.domain = CDSP_DOMAIN_ID;
    unsigned_module.enable = 1;
    if (!rpc_succeeded(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &unsigned_module, sizeof unsigned_module),
                       "unsigned PD request")) {
        return false;
    }

    // The DSP side runs GEMM thread 0 on the FastRPC thread, which needs a large stack.
    remote_rpc_thread_params thread_parameters{};
    thread_parameters.domain = CDSP_DOMAIN_ID;
    thread_parameters.prio = -1;
    thread_parameters.stack_size = 256 * 1024;
    if (!rpc_succeeded(remote_session_control(FASTRPC_THREAD_PARAMS, &thread_parameters, sizeof thread_parameters),
                       "thread parameters request")) {
        return false;
    }

    char uri[256];
    snprintf(uri, sizeof uri, "%s%s", benzene_URI, CDSP_DOMAIN);
    return rpc_succeeded(benzene_open(uri, &session_handle), "benzene_open(%s)", uri);
}

void close_session() {
    if (session_handle != 0) benzene_close(session_handle);
    session_handle = 0;
}

remote_handle64 session() {
    return session_handle;
}

void * allocate_rpc_memory(uint64_t bytes) {
    void * buffer = bytes <= INT_MAX ? library.allocate(kSystemHeap, kCachedFlag, static_cast<int>(bytes)) : nullptr;
    if (buffer == nullptr) fprintf(stderr, "rpcmem_alloc %llu failed\n", static_cast<unsigned long long>(bytes));
    return buffer;
}

void * load_file_into_rpc_memory(const char * path, size_t * bytes) {
    const File file = open_file(path, "rb");
    if (!file || !file_size(file.get(), bytes)) {
        fprintf(stderr, "read %s failed\n", path);
        return nullptr;
    }
    void * buffer = allocate_rpc_memory(*bytes != 0 ? *bytes : 128);
    if (buffer != nullptr && fread(buffer, 1, *bytes, file.get()) != *bytes) {
        fprintf(stderr, "read %s failed\n", path);
        return nullptr;
    }
    return buffer;
}

bool fill_from_file(uint8_t * buffer, const char * path, uint64_t offset, uint64_t bytes, bool mapped) {
    const int descriptor = mapped ? library.to_descriptor(buffer) : -1;
    dma_buf_sync sync{DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE};
    if (descriptor >= 0 && ioctl(descriptor, DMA_BUF_IOCTL_SYNC, &sync) != 0) {
        fprintf(stderr, "dma-buf sync start failed\n");
        return false;
    }

    const File file = open_file(path, "rb");
    const bool read = file && fseeko(file.get(), static_cast<off_t>(offset), SEEK_SET) == 0 &&
                      fread(buffer, 1, bytes, file.get()) == bytes;
    if (!read) fprintf(stderr, "read %s failed\n", path);

    sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    if (descriptor >= 0 && ioctl(descriptor, DMA_BUF_IOCTL_SYNC, &sync) != 0) {
        fprintf(stderr, "dma-buf sync end failed\n");
        return false;
    }
    return read;
}

bool map_into_slot(uint8_t * buffer, uint64_t bytes, int slot) {
    if (buffer == nullptr) return false;
    if (bytes > UINT32_MAX) {
        fprintf(stderr, "map slot %d: %llu bytes do not fit the DSP's address space\n", slot,
                static_cast<unsigned long long>(bytes));
        return false;
    }

    const int descriptor = library.to_descriptor(buffer);
    return rpc_succeeded(library.map(CDSP_DOMAIN_ID, descriptor, buffer, 0, bytes, FASTRPC_MAP_FD),
                         "fastrpc_mmap (slot %d)", slot) &&
           rpc_succeeded(
               benzene_map_model(session_handle, descriptor, static_cast<uint32_t>(bytes), static_cast<uint32_t>(slot)),
               "map_model (slot %d)", slot);
}

bool unmap_slot(uint8_t * buffer, uint64_t bytes, int slot) {
    return rpc_succeeded(benzene_unmap_model(session_handle, static_cast<uint32_t>(slot)), "unmap_model (slot %d)",
                         slot) &&
           rpc_succeeded(library.unmap(CDSP_DOMAIN_ID, library.to_descriptor(buffer), buffer, bytes),
                         "fastrpc_munmap (slot %d)", slot);
}

bool load_into_slot(const char * path, uint64_t offset, uint64_t bytes, int slot) {
    auto * buffer = static_cast<uint8_t *>(allocate_rpc_memory(bytes));
    return buffer != nullptr && fill_from_file(buffer, path, offset, bytes, false) &&
           map_into_slot(buffer, bytes, slot);
}

}  // namespace benzene::host

// Called by the generated stub (benzene_stub.c); forwarded to the libcdsprpc loaded at run time.
extern "C" {

int remote_handle64_open(const char * name, remote_handle64 * handle) {
    return benzene::host::library.handle_open(name, handle);
}

int remote_handle64_invoke(remote_handle64 handle, uint32_t scalars, remote_arg * arguments) {
    return benzene::host::library.handle_invoke(handle, scalars, arguments);
}

int remote_handle64_close(remote_handle64 handle) {
    return benzene::host::library.handle_close(handle);
}

int remote_session_control(uint32_t request, void * data, uint32_t length) {
    return benzene::host::library.session_control(request, data, length);
}

}  // extern "C"
