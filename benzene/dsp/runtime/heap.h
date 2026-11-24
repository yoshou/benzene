#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>

namespace benzene::runtime {

// memalign blocks, freed with free(): the DSP process has no operator new / delete.
struct FreeDeleter {
    void operator()(void * memory) const { free(memory); }
};

template <class T>
using HeapArray = std::unique_ptr<T[], FreeDeleter>;

template <class T>
using HeapObject = std::unique_ptr<T, FreeDeleter>;

// `count` Ts at an `alignment`-byte boundary (empty when out of memory). The memory is not initialized and no
// constructor runs: T is an implicit-lifetime type (arithmetic types, vectors, structs of them).
template <class T>
HeapArray<T> allocate_array(size_t count, size_t alignment = 128) {
    return HeapArray<T>(static_cast<T *>(memalign(alignment, count * sizeof(T))));
}

// allocate_array into `buffer`; false when out of memory.
template <class T>
bool allocate(HeapArray<T> & buffer, size_t count, size_t alignment = 128) {
    buffer = allocate_array<T>(count, alignment);
    return static_cast<bool>(buffer);
}

// Room for one T (`bytes`, at least sizeof(T)) at an `alignment`-byte boundary (empty when out of memory). Not
// initialized: a T with a constructor is placement-new'd into it.
template <class T>
HeapObject<T> allocate_object(size_t alignment = alignof(T), size_t bytes = sizeof(T)) {
    return HeapObject<T>(static_cast<T *>(memalign(alignment, bytes)));
}

// A heap array that grows to the largest count asked for; growing does not keep the contents.
template <class T>
class GrowingArray {
  public:
    // At least `count` elements (nullptr when out of memory).
    T * reserve(size_t count, size_t alignment = 128) {
        if (capacity_ < count) {
            // freed first: with both blocks live, the DSP heap grows through the host
            data_.reset();
            data_ = allocate_array<T>(count, alignment);
            capacity_ = data_ ? count : 0;
        }
        return data_.get();
    }
    T * get() const { return data_.get(); }
    void reset() {
        data_.reset();
        capacity_ = 0;
    }

  private:
    HeapArray<T> data_;
    size_t capacity_ = 0;
};

}  // namespace benzene::runtime
