#include "dsp/runtime/int8_scale_table.h"

#include <bit>
#include <utility>

#include "dsp/runtime/heap.h"

namespace benzene::runtime {
namespace {

constexpr int kMagnitudes = 32768;
constexpr float kLargestFp16 = 65504.0f;

Int8Scale compute_int8_scale(uint16_t amax_bits) {
    const auto amax = std::bit_cast<_Float16>(amax_bits);
    Int8Scale result = {0.0f, 0, 0};
    if (static_cast<float>(amax) > 0.0f) {
        const float inverse = 127.0f / static_cast<float>(amax);
        const _Float16 inverse_fp16 = static_cast<_Float16>(inverse < kLargestFp16 ? inverse : kLargestFp16);
        result.inverse_bits = std::bit_cast<uint16_t>(inverse_fp16);
        result.scale = 1.0f / static_cast<float>(inverse_fp16);
    }
    return result;
}

struct Int8Scales {
    HeapArray<Int8Scale> table;
};

}  // namespace

Status build_int8_scales(Session & session) {
    Int8Scales * scales = session.state<Int8Scales>();
    if (scales == nullptr) return std::unexpected(Error::kNoMemory);
    if (scales->table) return {};
    HeapArray<Int8Scale> table = allocate_array<Int8Scale>(kMagnitudes);
    if (!table) return std::unexpected(Error::kNoMemory);
    for (int bits = 0; bits < kMagnitudes; bits++) table[bits] = compute_int8_scale(static_cast<uint16_t>(bits));
    scales->table = std::move(table);
    return {};
}

const Int8Scale * int8_scales(Session & session) {
    const Int8Scales * scales = session.state<Int8Scales>();
    return scales != nullptr ? scales->table.get() : nullptr;
}

}  // namespace benzene::runtime
