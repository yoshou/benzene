#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#include "dsp/runtime/status.h"
#include "benzene/protocol.h"

namespace benzene::runtime {

constexpr int kModelSlots = protocol::kModelSlots;

// One FastRPC session (benzene_open .. benzene_close): the compute resource that holds all of VTCM together with HMX,
// the model files the host mapped (runtime slots) and the state kept between calls (state()).
class Session {
  public:
    // Destroys the state (the worker threads stop), releases the compute resource and frees the session.
    struct Closer {
        void operator()(Session * session) const;
    };
    using Owner = std::unique_ptr<Session, Closer>;

    // Pins power to the maximum corners, acquires VTCM + HMX and builds the int8 scale table.
    static Result<Owner> open();

    uint32_t compute_resource() const { return compute_resource_; }
    uint8_t * vtcm() const { return vtcm_; }
    uint32_t vtcm_bytes() const { return vtcm_bytes_; }

    // Maps an rpcmem buffer (registered by the host with fastrpc_mmap) into the free slot `slot`.
    Status map_model(int slot, int file_descriptor, uint32_t bytes);
    Status unmap_model(int slot);
    bool model_mapped(int slot) const { return slot >= 0 && slot < kModelSlots && model_base_[slot] != nullptr; }
    uint8_t * model_base(int slot) const { return model_base_[slot]; }
    uint32_t model_bytes(int slot) const { return model_bytes_[slot]; }

    // A copy for kernel calls that may use only the first `bytes` of VTCM (the caller keeps its own data above them).
    // It shares the compute resource, the model mappings and the state; it is not for close().
    Session with_vtcm_bytes(uint32_t bytes) const;

    // The session's object of type T (nullptr when out of memory), value-initialized at the first call and destroyed
    // when the session closes, in the reverse order of the first calls. Concurrent calls get the same object. One
    // object per type: T is a type of the caller's own.
    template <class T>
    T * state() const {
        return static_cast<T *>(state(&kStateKey<T>, sizeof(T), alignof(T), make_state<T>, destroy_state<T>));
    }

  private:
    struct States;

    template <class T>
    static constexpr char kStateKey = 0;  // its address identifies T
    template <class T>
    static void make_state(void * memory) {
        new (memory) T();
    }
    template <class T>
    static void destroy_state(void * object) {
        static_cast<T *>(object)->~T();
    }
    void * state(const void * key, size_t bytes, size_t alignment, void (*make)(void *), void (*destroy)(void *)) const;

    Session() = default;
    Session(const Session & other) = default;

    uint32_t compute_resource_ = 0;
    uint8_t * vtcm_ = nullptr;
    uint32_t vtcm_bytes_ = 0;
    uint8_t * model_base_[kModelSlots] = {};
    uint32_t model_bytes_[kModelSlots] = {};
    States * states_ = nullptr;
};

}  // namespace benzene::runtime
