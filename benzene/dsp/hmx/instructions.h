#pragma once

#include <cstddef>
#include <cstdint>

#include <HAP_compute_res.h>

#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/vector.h"
#include "dsp/runtime/session.h"

namespace benzene::hmx {

// A tile: 32 x 32 fp16, or 64 rows x 32 uint8 activations.
constexpr size_t kTileBytes = 2048;

// One multiply covers at most 32 k tiles: the tiles per multiply of a run of k_tiles, spread evenly.
constexpr int tiles_per_load(int k_tiles) {
    constexpr int kMaximumTilesPerLoad = 32;
    const int loads = (k_tiles + kMaximumTilesPerLoad - 1) / kMaximumTilesPerLoad;
    return (k_tiles + loads - 1) / loads;
}

inline void lock(const runtime::Session & session) {
    HAP_compute_res_hmx_lock(session.compute_resource());
}
inline void unlock(const runtime::Session & session) {
    HAP_compute_res_hmx_unlock(session.compute_resource());
}

// HMX instructions are queued: the issuing thread blocks only when the queue is full, at mxclracc (which waits for the
// queued work) or at a scalar store after an HMX store (ordered behind it).
inline void clear_fp16_accumulator() {
    asm volatile("mxclracc.hf\n");
}
inline void clear_integer_accumulator() {
    asm volatile("mxclracc\n");
}

// Takes HMX for the calling thread and zeroes its accumulators.
// A store without :retain also leaves the accumulator zero.
inline void lock_and_clear(const runtime::Session & session) {
    lock(session);
    clear_fp16_accumulator();
    clear_integer_accumulator();
}

// The output block register ("bias"): per column the output scale and bias of the next store.
inline void load_output_block(const void * block) {
    asm volatile("bias = mxmem2(%0)\n" ::"r"(block) : "memory");
}

// Accumulates fp16 activation tiles (:deep) times fp16 weight tiles; `range` = bytes of the activation run - 1.
inline void multiply_fp16(const void * activations, const void * weights, uint32_t range) {
    asm volatile("{\n activation.hf = mxmem(%0, %2):deep\n weight.hf = mxmem(%1, %2)\n}\n" ::"r"(activations),
                 "r"(weights), "r"(range)
                 : "memory");
}

// Accumulates uint8 activation tiles (:deep) times int8 weight tiles.
inline void multiply_uint8_int8(const void * activations, const void * weights, uint32_t range) {
    asm volatile("{\n activation.ub = mxmem(%0, %2):deep\n weight.b = mxmem(%1, %2)\n}\n" ::"r"(activations),
                 "r"(weights), "r"(range)
                 : "memory");
}

// The same with the weights also read :deep (weight tiles laid out as the activation tiles' depth pairs).
inline void multiply_uint8_int8_deep_weights(const void * activations, const void * weights, uint32_t range) {
    asm volatile("{\n activation.ub = mxmem(%0, %2):deep\n weight.b = mxmem(%1, %2):deep\n}\n" ::"r"(activations),
                 "r"(weights), "r"(range)
                 : "memory");
}

inline void store_fp16(void * output) {
    asm volatile("mxmem(%0, %1):after.hf = acc\n" ::"r"(output), "r"(0) : "memory");
}

inline void store_uint8_and_retain(void * output) {
    asm volatile("mxmem(%0, %1):after:retain.ub = acc\n" ::"r"(output), "r"(0) : "memory");
}

inline void store_uint8(void * output) {
    asm volatile("mxmem(%0, %1):after.ub = acc\n" ::"r"(output), "r"(0) : "memory");
}

// The int32 accumulator of a W8A8 tile (64 x 32) as four uint8 slices, 2048 bytes apart. The four 256-byte output
// blocks at `output_blocks` hold the scales 512, 2, 2^-7 and 2^-14 (bits 0-7, 8-15, 16-23 and 23-30 of acc + bias),
// with the bias that removes the +128 activation offset. The last store leaves the accumulator zero.
inline void store_int32_as_uint8_slices(void * output, const uint8_t * output_blocks) {
    uint8_t * const slices = static_cast<uint8_t *>(output);
    load_output_block(output_blocks);
    store_uint8_and_retain(slices);
    load_output_block(output_blocks + 256);
    store_uint8_and_retain(slices + 2048);
    load_output_block(output_blocks + 512);
    store_uint8_and_retain(slices + 4096);
    load_output_block(output_blocks + 768);
    store_uint8(slices + 6144);
}

// The int32 sums of rows 4 quad .. 4 quad + 3 of a tile stored by store_int32_as_uint8_slices, one row per vector.
// The fourth slice (bits 23..30) as a signed byte averaged with 0 is bits 24..31 (|sum| < 2^30).
template <class Byte>
[[gnu::always_inline]] inline void load_int32_row_quad(const hvx::Vector<Byte> * __restrict slices, int quad,
                                                       hvx::Vector<int32_t> rows[4]) {
    using hvx::Vector;
    static_assert(sizeof(Byte) == 1);
    constexpr int kSliceVectors = 2048 / hvx::kVectorBytes;
    const Vector<uint8_t> slice0 = hvx::reinterpret<uint8_t>(slices[quad]);
    const Vector<uint8_t> slice1 = hvx::reinterpret<uint8_t>((slices + kSliceVectors)[quad]);
    const Vector<uint8_t> slice2 = hvx::reinterpret<uint8_t>((slices + 2 * kSliceVectors)[quad]);
    const Vector<uint8_t> slice3 = hvx::reinterpret<uint8_t>(
        average(hvx::reinterpret<int8_t>((slices + 3 * kSliceVectors)[quad]), Vector<int8_t>::zero()));

    const Vector<uint16_t> low_even_rows = hvx::reinterpret<uint16_t>(interleave_even_lanes(slice1, slice0));
    const Vector<uint16_t> low_odd_rows = hvx::reinterpret<uint16_t>(interleave_odd_lanes(slice1, slice0));
    const Vector<uint16_t> high_even_rows = hvx::reinterpret<uint16_t>(interleave_even_lanes(slice3, slice2));
    const Vector<uint16_t> high_odd_rows = hvx::reinterpret<uint16_t>(interleave_odd_lanes(slice3, slice2));

    rows[0] = hvx::reinterpret<int32_t>(interleave_even_lanes(high_even_rows, low_even_rows));
    rows[1] = hvx::reinterpret<int32_t>(interleave_even_lanes(high_odd_rows, low_odd_rows));
    rows[2] = hvx::reinterpret<int32_t>(interleave_odd_lanes(high_even_rows, low_even_rows));
    rows[3] = hvx::reinterpret<int32_t>(interleave_odd_lanes(high_odd_rows, low_odd_rows));
}

}  // namespace benzene::hmx
