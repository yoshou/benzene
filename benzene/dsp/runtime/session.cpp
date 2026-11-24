#include "dsp/runtime/session.h"

#include <atomic>
#include <cstring>
#include <new>
#include <type_traits>

#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_power.h>

#include "dsp/runtime/heap.h"
#include "dsp/runtime/int8_scale_table.h"

namespace benzene::runtime {
namespace {

constexpr int kAcquireTimeoutMicroseconds = 1000000;

Status pin_power_to_maximum(void * client) {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_apptype;
    request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    if (const Status status = status_of_code(HAP_power_set(client, &request)); !status) return status;

    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_DCVS_v3;
    request.dcvs_v3.set_dcvs_enable = TRUE;
    request.dcvs_v3.dcvs_enable = FALSE;
    request.dcvs_v3.set_bus_params = TRUE;
    request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.set_core_params = TRUE;
    request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.set_sleep_disable = TRUE;
    request.dcvs_v3.sleep_disable = TRUE;
    if (const Status status = status_of_code(HAP_power_set(client, &request)); !status) return status;

    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = TRUE;
    if (const Status status = status_of_code(HAP_power_set(client, &request)); !status) return status;

    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HMX_v2;
    request.hmx_v2.set_power = TRUE;
    request.hmx_v2.power_up = TRUE;
    request.hmx_v2.set_clock = TRUE;
    request.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_MAX;
    request.hmx_v2.min_corner = HAP_DCVS_EXP_VCORNER_MAX;
    request.hmx_v2.max_corner = HAP_DCVS_EXP_VCORNER_MAX;
    request.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
    return status_of_code(HAP_power_set(client, &request));
}

}  // namespace

// Append-only. An entry is written before `count` publishes it (release), so lookups read the published entries
// without the lock. `adding` serializes the callers that make an object.
struct Session::States {
    static constexpr int kCapacity = 64;
    struct Entry {
        const void * key;
        void * object;
        void (*destroy)(void * object);
    };
    Entry entries[kCapacity];
    std::atomic<int> count;
    std::atomic<int> adding;
};

static_assert(std::is_trivially_destructible_v<Session>, "freed without a destructor call");

void * Session::state(const void * key, size_t bytes, size_t alignment, void (*make)(void *),
                      void (*destroy)(void *)) const {
    States & states = *states_;
    const int published = states.count.load(std::memory_order_acquire);
    for (int i = 0; i < published; i++) {
        if (states.entries[i].key == key) return states.entries[i].object;
    }

    int idle = 0;
    while (!states.adding.compare_exchange_weak(idle, 1)) idle = 0;
    const int count = states.count.load(std::memory_order_relaxed);
    void * object = nullptr;
    for (int i = published; i < count && object == nullptr; i++) {
        if (states.entries[i].key == key) object = states.entries[i].object;
    }

    if (object == nullptr && count < States::kCapacity) {
        object = allocate_array<uint8_t>(bytes, alignment < 8 ? 8 : alignment).release();
        if (object != nullptr) {
            make(object);
            states.entries[count] = {key, object, destroy};
            states.count.store(count + 1, std::memory_order_release);
        }
    }
    states.adding.store(0);
    return object;
}

Session Session::with_vtcm_bytes(uint32_t bytes) const {
    Session limited(*this);
    limited.vtcm_bytes_ = bytes < vtcm_bytes_ ? bytes : vtcm_bytes_;
    return limited;
}

Result<Session::Owner> Session::open() {
    static_assert(std::is_trivially_destructible_v<States>, "freed without a destructor call");
    HeapObject<Session> memory = allocate_object<Session>();
    if (!memory) return std::unexpected(Error::kNoMemory);
    HeapObject<States> states = allocate_object<States>();
    if (!states) return std::unexpected(Error::kNoMemory);
    Owner created(new (memory.release()) Session());
    created->states_ = new (states.release()) States();

    if (const Status built = build_int8_scales(*created); !built) return std::unexpected(built.error());

    if (const Status powered = pin_power_to_maximum(created.get()); !powered) {
        FARF(ERROR, "benzene: power setup failed 0x%x", aee_code(powered));
        return std::unexpected(powered.error());
    }

    unsigned int vtcm_bytes = 8u << 20;
    unsigned int available_bytes = 0;
    compute_res_vtcm_page_t total_layout;
    compute_res_vtcm_page_t available_layout;
    HAP_compute_res_query_VTCM(0, &vtcm_bytes, &total_layout, &available_bytes, &available_layout);

    compute_res_attr_t attributes;
    HAP_compute_res_attr_init(&attributes);
    HAP_compute_res_attr_set_serialize(&attributes, 0);
    HAP_compute_res_attr_set_vtcm_param_v2(&attributes, vtcm_bytes, vtcm_bytes, vtcm_bytes);
    HAP_compute_res_attr_set_hmx_param(&attributes, 1);
    created->compute_resource_ = HAP_compute_res_acquire(&attributes, kAcquireTimeoutMicroseconds);
    void * vtcm = nullptr;
    if (created->compute_resource_ == 0 || HAP_compute_res_attr_get_vtcm_ptr_v2(&attributes, &vtcm, &vtcm_bytes) != 0) {
        FARF(ERROR, "benzene: VTCM/HMX acquire failed");
        return std::unexpected(Error::kNoMemory);
    }

    created->vtcm_ = static_cast<uint8_t *>(vtcm);
    created->vtcm_bytes_ = vtcm_bytes;
    return created;
}

void Session::Closer::operator()(Session * session) const {
    if (States * states = session->states_; states != nullptr) {
        for (int i = states->count.load() - 1; i >= 0; i--) {
            states->entries[i].destroy(states->entries[i].object);
            FreeDeleter{}(states->entries[i].object);
        }
        FreeDeleter{}(states);
    }
    if (session->compute_resource_ != 0) HAP_compute_res_release(session->compute_resource_);
    FreeDeleter{}(session);
}

Status Session::map_model(int slot, int file_descriptor, uint32_t bytes) {
    if (slot < 0 || slot >= kModelSlots || model_mapped(slot)) return std::unexpected(Error::kBadParameter);
    void * mapped = HAP_mmap2(nullptr, bytes, HAP_PROT_READ | HAP_PROT_WRITE, 0, file_descriptor, 0);
    if (mapped == nullptr || mapped == reinterpret_cast<void *>(-1)) return std::unexpected(Error::kNoMemory);
    model_base_[slot] = static_cast<uint8_t *>(mapped);
    model_bytes_[slot] = bytes;
    return {};
}

Status Session::unmap_model(int slot) {
    if (!model_mapped(slot)) return std::unexpected(Error::kBadParameter);
    const int error = HAP_munmap2(model_base_[slot], model_bytes_[slot]);
    model_base_[slot] = nullptr;
    model_bytes_[slot] = 0;
    if (error != 0) return std::unexpected(Error::kFailed);
    return {};
}

}  // namespace benzene::runtime
