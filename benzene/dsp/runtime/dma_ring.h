// The Hexagon DMA engine's 2D descriptors in a ring per thread, submitted with release + dmlink.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace benzene::runtime {

struct DmaDescriptor {
    void * next;
    uint32_t destination_stride : 24;
    uint32_t descriptor_size : 2;  // 1: 2D
    uint32_t destination_compressed : 1;
    uint32_t source_compressed : 1;
    uint32_t destination_bypass : 1;
    uint32_t source_bypass : 1;
    uint32_t order : 1;
    uint32_t done : 1;  // set by the engine on completion
    uint32_t source;
    uint32_t destination;
    uint32_t descriptor_type : 8;  // 9: 2D with 24-bit row size and strides (v75 .. v79)
    uint32_t reserved0 : 24;
    uint32_t row_bytes : 24;
    uint32_t rows_low : 8;
    uint32_t rows_high : 8;
    uint32_t source_stride : 24;
    uint32_t offset : 24;
    uint32_t reserved1 : 8;
};

// Which side of a transfer bypasses L2: right when the caller keeps DDR and the cache coherent itself.
enum class DmaBypass : uint32_t { kSource = 1, kDestination = 2, kBoth = 3 };

// One ring per issuing thread.
class alignas(64) DmaRing {
  public:
    static constexpr uint32_t kCapacity = 64;  // a power of two

    void initialize(DmaBypass bypass) {
        memset(this, 0, sizeof(*this));
        tail_ = &descriptors_[kCapacity - 1];  // never submitted; the first dmlink starts the engine
        bypass_ = static_cast<uint32_t>(bypass);
    }

    // initialize() for a thread that must not execute HVX: a memset of the ring compiles to vector stores.
    void initialize_with_scalar_stores(DmaBypass bypass) {
        volatile uint32_t * words = reinterpret_cast<volatile uint32_t *>(this);
#pragma clang loop vectorize(disable)
        for (size_t i = 0; i < sizeof(*this) / 4; i++) words[i] = 0;
        tail_ = &descriptors_[kCapacity - 1];
        bypass_ = static_cast<uint32_t>(bypass);
    }

    void set_bypass(DmaBypass bypass) { bypass_ = static_cast<uint32_t>(bypass); }

    uint32_t depth() const { return pushed_ - popped_; }

    // The caller keeps depth() < kCapacity.
    void push(void * destination, const void * source, uint32_t row_bytes, uint32_t rows, uint32_t destination_stride,
              uint32_t source_stride) {
        DmaDescriptor * descriptor = &descriptors_[pushed_ & (kCapacity - 1)];
        descriptor->next = nullptr;
        descriptor->destination_stride = destination_stride;
        descriptor->descriptor_size = 1;
        descriptor->destination_compressed = 0;
        descriptor->source_compressed = 0;
        descriptor->destination_bypass = (bypass_ >> 1) & 1;
        descriptor->source_bypass = bypass_ & 1;
        descriptor->order = 0;
        descriptor->done = 0;
        descriptor->source = reinterpret_cast<uint32_t>(source);
        descriptor->destination = reinterpret_cast<uint32_t>(destination);
        descriptor->descriptor_type = 9;
        descriptor->reserved0 = 0;
        descriptor->row_bytes = row_bytes;
        descriptor->rows_low = rows & 0xff;
        descriptor->rows_high = rows >> 8;
        descriptor->source_stride = source_stride;
        descriptor->offset = 0;
        descriptor->reserved1 = 0;

        asm volatile("release(%0):at" ::"r"(descriptor));
        asm volatile("dmlink(%0, %1)" ::"r"(tail_), "r"(descriptor));
        tail_ = descriptor;
        pushed_++;
    }

    void push_contiguous(void * destination, const void * source, uint32_t bytes) {
        push(destination, source, bytes, 1, bytes, bytes);
    }

    // Waits for the oldest outstanding descriptor.
    void pop() {
        volatile DmaDescriptor * descriptor = &descriptors_[popped_ & (kCapacity - 1)];
        while (!descriptor->done) {
            uint32_t state;
            asm volatile("%0 = dmpoll" : "=r"(state)::"memory");
            (void)state;
        }
        popped_++;
    }

    void drain() {
        while (popped_ != pushed_) pop();
    }

  private:
    DmaDescriptor descriptors_[kCapacity];
    DmaDescriptor * tail_;
    uint32_t pushed_;
    uint32_t popped_;
    uint32_t bypass_;
};

}  // namespace benzene::runtime
