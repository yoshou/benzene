#include "dsp/kernels/depthwise_int8/depthwise_int8.h"

#include <atomic>
#include <cstring>

#include "dsp/hvx/dot_products.h"
#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/vector.h"
#include "dsp/runtime/dma_ring.h"
#include "dsp/runtime/heap.h"
#include "dsp/runtime/synchronization.h"
#include "dsp/runtime/thread_pool.h"

namespace benzene::kernels::depthwise_int8 {
namespace {

using hvx::Mask;
using hvx::Vector;
using hvx::VectorPair;

constexpr int kTaps = 25;
constexpr int kPhases = 4;
constexpr int kWeightsPerPhase = 10;
constexpr int kPixelsPerIteration = 4;

struct GroupConstants {
    Vector<int32_t> initial;
    Vector<float> multipliers;
    Vector<float> biases;
    Vector<uint8_t> pad;
};

struct DepthwiseJob {
    Arguments a;
    int groups;
    int padded_width;
    int buffer_pixels;
    int blocks;
    size_t buffer_vectors;
    size_t staging_row_bytes;
    const Vector<int8_t> * weights;
    const GroupConstants * constants;
    Vector<uint8_t> * buffers;
    uint8_t * staging;
    runtime::DmaRing * rings;
    Vector<uint8_t> pad_codes;
};

// Output rows go in blocks of four, y0 .. y0 + 3, from the row groups A = rows y0 - 2 .. y0 + 1 and B = rows
// y0 + 2 .. y0 + 5. Row y0 + j (phase j) sums, per column kx, a vrmpy of A by WA[j][kx] (tap (a - j, kx) at A's row
// a >= j) and one of B by WB[j][kx] (tap (b + 4 - j, kx) at B's row b <= j), zero elsewhere.
void prepare_constants(const Arguments & a, const int8_t * weights, int groups, Vector<int8_t> * weight_vectors,
                       GroupConstants * constants) {
    const Vector<int8_t> zero = Vector<int8_t>::zero();
    for (int g = 0; g < groups; g++) {
        Vector<int8_t> taps[kTaps];
        for (int t = 0; t < kTaps; t++) {
            taps[t] = hvx::load_unaligned(weights + static_cast<size_t>(t) * a.channels + 32 * g);
        }

        for (int j = 0; j < kPhases; j++) {
            for (int kx = 0; kx < 5; kx++) {
                Vector<int8_t> rows_a[4];
                Vector<int8_t> rows_b[4];
                for (int r = 0; r < 4; r++) {
                    rows_a[r] = r >= j ? taps[(r - j) * 5 + kx] : zero;
                    rows_b[r] = r <= j ? taps[(r + 4 - j) * 5 + kx] : zero;
                }

                Vector<int8_t> words[4];
                Vector<int8_t> * out = weight_vectors + ((static_cast<size_t>(g) * kPhases + j) * 2) * 5;
                interleave_four(rows_a[0], rows_a[1], rows_a[2], rows_a[3], words);
                out[kx] = words[0];
                interleave_four(rows_b[0], rows_b[1], rows_b[2], rows_b[3], words);
                out[5 + kx] = words[0];
            }
        }

        GroupConstants & c = constants[g];
        c.initial = hvx::load_unaligned(a.initial + 32 * g);
        c.multipliers = hvx::load_unaligned(a.output_multipliers + 32 * g);
        c.biases = hvx::load_unaligned(a.output_biases + 32 * g);

        alignas(128) uint8_t bytes[128];
        for (int i = 0; i < 32; i++) {
            const uint8_t z = a.zero_points[32 * g + i];
            bytes[4 * i] = bytes[4 * i + 1] = bytes[4 * i + 2] = bytes[4 * i + 3] = z;
        }
        c.pad = hvx::load(bytes);
    }
}

int stage_rows(const DepthwiseJob & job, runtime::DmaRing & ring, uint8_t * staging, int first_row) {
    const Arguments & a = job.a;
    int queued = 0;
    for (int r = 0; r < 4; r++) {
        const int y = first_row + r;
        if (y < 0 || y >= a.height) continue;

        const uint8_t * source = a.input + static_cast<size_t>(y) * a.width * a.input_stride;
        uint8_t * destination = staging + static_cast<size_t>(r) * job.staging_row_bytes;
        if (a.input_stride == a.channels) {
            ring.push_contiguous(destination, source, static_cast<uint32_t>(a.width) * a.channels);
        } else {
            ring.push(destination, source, static_cast<uint32_t>(a.channels), static_cast<uint32_t>(a.width),
                      static_cast<uint32_t>(a.channels), static_cast<uint32_t>(a.input_stride));
        }
        queued++;
    }
    return queued;
}

// The row group buffer of input rows first_row .. first_row + 3: per channel group g a row of buffer_pixels vectors,
// pixel p (-2 .. padded_width + 1) at p + 2, word i = the four rows' codes of channel 32 g + i (interleave_rows).
// Pixels and rows outside the image hold the zero points.
void build_row_group(const DepthwiseJob & job, Vector<uint8_t> * buffer, int first_row, const uint8_t * staging) {
    const Arguments & a = job.a;
    const int groups = job.groups;
    const size_t pitch = static_cast<size_t>(job.buffer_pixels);
    bool inside[4];
    for (int r = 0; r < 4; r++) inside[r] = first_row + r >= 0 && first_row + r < a.height;

    for (int g = 0; g < groups; g++) {
        Vector<uint8_t> * row = buffer + static_cast<size_t>(g) * pitch;
        row[0] = row[1] = job.constants[g].pad;
        for (int p = a.width + 2; p < job.buffer_pixels; p++) row[p] = job.constants[g].pad;
    }

    if (a.channels >= 128) {
        const int chunks = a.channels / 128;
        for (int p = 0; p < a.width; p++) {
            for (int k = 0; k < chunks; k++) {
                Vector<uint8_t> v[4];
                for (int r = 0; r < 4; r++) {
                    v[r] = inside[r] ? hvx::load(staging + r * job.staging_row_bytes +
                                                 static_cast<size_t>(p) * a.channels + 128 * k)
                                     : hvx::load_unaligned(a.zero_points + 128 * k);
                }

                Vector<uint8_t> words[4];
                interleave_four(v[0], v[1], v[2], v[3], words);
                for (int r = 0; r < 4; r++) buffer[static_cast<size_t>(4 * k + r) * pitch + 2 + p] = words[r];
            }
        }
        return;
    }

    const size_t row_bytes = static_cast<size_t>(a.width) * a.channels;
    const size_t chunks = (row_bytes + 127) / 128;
    for (size_t k = 0; k < chunks; k++) {
        const size_t valid = row_bytes - k * 128 < 128 ? row_bytes - k * 128 : 128;
        const Mask staged = Mask::first_bytes(static_cast<int>(valid));  // (valid < 128 only)
        Vector<uint8_t> v[4];
        for (int r = 0; r < 4; r++) {
            v[r] = job.pad_codes;
            if (inside[r]) {
                const Vector<uint8_t> codes = hvx::load(staging + r * job.staging_row_bytes + k * 128);
                v[r] = valid == 128 ? codes : select(staged, codes, job.pad_codes);
            }
        }

        Vector<uint8_t> words[4];
        interleave_four(v[0], v[1], v[2], v[3], words);
        if (a.channels == 32) {
            for (int r = 0; r < 4; r++) buffer[2 + 4 * k + r] = words[r];
        } else {
            for (int r = 0; r < 4; r++) buffer[static_cast<size_t>(r % 2) * pitch + 2 + 2 * k + r / 2] = words[r];
        }
    }
}

inline void group_sums(const DepthwiseJob & job, const Vector<uint8_t> * buffer_a, const Vector<uint8_t> * buffer_b,
                       int g, int j, int x, Vector<int32_t> * sums) {
    const Vector<int32_t> initial = job.constants[g].initial;
#pragma clang loop unroll(full)
    for (int p = 0; p < kPixelsPerIteration; p++) sums[p] = initial;

#pragma clang loop unroll(full)
    for (int side = 0; side < 2; side++) {
        const Vector<uint8_t> * column =
            (side == 0 ? buffer_a : buffer_b) + static_cast<size_t>(g) * job.buffer_pixels + x;
        Vector<uint8_t> window[kPixelsPerIteration + 4];
#pragma clang loop unroll(full)
        for (int k = 0; k < kPixelsPerIteration + 4; k++) window[k] = column[k];

        const Vector<int8_t> * weights = job.weights + ((static_cast<size_t>(g) * kPhases + j) * 2 + side) * 5;
#pragma clang loop unroll(full)
        for (int kx = 0; kx < 5; kx++) {
            const Vector<int8_t> w = weights[kx];
#pragma clang loop unroll(full)
            for (int p = 0; p < kPixelsPerIteration; p++) sums[p] = dot_product_accumulate(sums[p], window[p + kx], w);
        }
    }
}

inline void requantize(const GroupConstants & c, Vector<int32_t> * sums) {
    const Vector<int32_t> zero = Vector<int32_t>::zero();
    const Vector<int32_t> top = Vector<int32_t>::splat(255);
#pragma clang loop unroll(full)
    for (int p = 0; p < kPixelsPerIteration; p++) {
        const Vector<float> value = to_float(sums[p]) * c.multipliers + c.biases;
        sums[p] = min(max(truncate_to_int32(value), zero), top);
    }
}

// 128 codes from four word vectors of codes: a's 32, then b's, c's, d's.
inline Vector<uint8_t> pack_codes(Vector<int32_t> a, Vector<int32_t> b, Vector<int32_t> c, Vector<int32_t> d) {
    return reinterpret<uint8_t>(pack_low_halves(pack_low_halves(d, c), pack_low_halves(b, a)));
}

inline void store_part(uint8_t * destination, Vector<uint8_t> codes, size_t bytes) {
    alignas(128) uint8_t staging[128];
    store(staging, codes);
    memcpy(destination, staging, bytes);
}

template <int kGroups>
inline void store_iteration(const DepthwiseJob & job, uint8_t * output_row, int x, int valid,
                            Vector<int32_t> (*sums)[4]) {
#pragma clang loop unroll(full)
    for (int g = 0; g < kGroups; g++) requantize(job.constants[g], sums[g]);

    if constexpr (kGroups == 1) {
        const Vector<uint8_t> codes = pack_codes(sums[0][0], sums[0][1], sums[0][2], sums[0][3]);
        uint8_t * destination = output_row + static_cast<size_t>(x) * 32;
        if (valid == kPixelsPerIteration) {
            store_unaligned(destination, codes);
        } else {
            store_part(destination, codes, static_cast<size_t>(valid) * 32);
        }
    } else {
        const Vector<uint8_t> first = pack_codes(sums[0][0], sums[1][0], sums[0][1], sums[1][1]);
        const Vector<uint8_t> second = pack_codes(sums[0][2], sums[1][2], sums[0][3], sums[1][3]);
        uint8_t * destination = output_row + static_cast<size_t>(x) * 64;
        if (valid == kPixelsPerIteration) {
            store_unaligned(destination, first);
            store_unaligned(destination + 128, second);
        } else {
            store_part(destination, first, static_cast<size_t>(valid < 2 ? valid : 2) * 64);
            if (valid > 2) store_part(destination + 128, second, static_cast<size_t>(valid - 2) * 64);
        }
    }
}

template <int kGroups>
void convolve_narrow_row(const DepthwiseJob & job, const Vector<uint8_t> * buffer_a, const Vector<uint8_t> * buffer_b,
                         int y, int j) {
    const Arguments & a = job.a;
    uint8_t * output_row = a.output + static_cast<size_t>(y) * a.width * a.channels;

    Vector<int32_t> previous[kGroups][4];
#pragma clang loop unroll(full)
    for (int g = 0; g < kGroups; g++) group_sums(job, buffer_a, buffer_b, g, j, 0, previous[g]);

    int x = kPixelsPerIteration;
    for (; x < a.width; x += kPixelsPerIteration) {
        Vector<int32_t> current[kGroups][4];
#pragma clang loop unroll(full)
        for (int g = 0; g < kGroups; g++) group_sums(job, buffer_a, buffer_b, g, j, x, current[g]);
        store_iteration<kGroups>(job, output_row, x - kPixelsPerIteration, kPixelsPerIteration, previous);
#pragma clang loop unroll(full)
        for (int g = 0; g < kGroups; g++) {
#pragma clang loop unroll(full)
            for (int p = 0; p < 4; p++) previous[g][p] = current[g][p];
        }
    }

    const int last = x - kPixelsPerIteration;
    store_iteration<kGroups>(job, output_row, last, a.width - last, previous);
}

// Per group the codes of pixels x .. x + 3 (32 each); a 4 x 4 transpose of 32-byte chunks gives each pixel's groups
// 4 quad .. 4 quad + 3.
void convolve_wide_row(const DepthwiseJob & job, const Vector<uint8_t> * buffer_a, const Vector<uint8_t> * buffer_b,
                       int y, int j, int quad) {
    const Arguments & a = job.a;
    uint8_t * output_row = a.output + static_cast<size_t>(y) * a.width * a.channels;

    for (int x = 0; x < a.width; x += kPixelsPerIteration) {
        const int valid = a.width - x < kPixelsPerIteration ? a.width - x : kPixelsPerIteration;
        Vector<uint8_t> by_group[4];
#pragma clang loop unroll(full)
        for (int q = 0; q < 4; q++) {
            Vector<int32_t> s[kPixelsPerIteration];
            group_sums(job, buffer_a, buffer_b, 4 * quad + q, j, x, s);
            requantize(job.constants[4 * quad + q], s);
            by_group[q] = pack_codes(s[0], s[1], s[2], s[3]);
        }

        const VectorPair<uint8_t> g01 = interleave_chunks<32>(by_group[1], by_group[0]);
        const VectorPair<uint8_t> g23 = interleave_chunks<32>(by_group[3], by_group[2]);
        const VectorPair<uint8_t> first = interleave_chunks<64>(g23.low(), g01.low());
        const VectorPair<uint8_t> second = interleave_chunks<64>(g23.high(), g01.high());
        const Vector<uint8_t> codes[kPixelsPerIteration] = {first.low(), first.high(), second.low(), second.high()};

        uint8_t * destination = output_row + static_cast<size_t>(x) * a.channels + 128 * quad;
        if (valid == kPixelsPerIteration) {
#pragma clang loop unroll(full)
            for (int p = 0; p < kPixelsPerIteration; p++) {
                store_unaligned(destination + static_cast<size_t>(p) * a.channels, codes[p]);
            }
        } else {
            for (int p = 0; p < valid; p++) {
                store_unaligned(destination + static_cast<size_t>(p) * a.channels, codes[p]);
            }
        }
    }
}

void band_thread(void * job_pointer, int thread) {
    const auto & job = *static_cast<const DepthwiseJob *>(job_pointer);
    const Arguments & a = job.a;
    const int first_block = job.blocks * thread / runtime::kHvxThreads;
    const int end_block = job.blocks * (thread + 1) / runtime::kHvxThreads;
    if (first_block >= end_block) return;

    Vector<uint8_t> * buffers[2] = {job.buffers + static_cast<size_t>(thread) * 2 * job.buffer_vectors,
                                    job.buffers + (static_cast<size_t>(thread) * 2 + 1) * job.buffer_vectors};
    uint8_t * staging[2] = {job.staging + static_cast<size_t>(thread) * 2 * 4 * job.staging_row_bytes,
                            job.staging + (static_cast<size_t>(thread) * 2 + 1) * 4 * job.staging_row_bytes};
    runtime::DmaRing & ring = job.rings[thread];
    // The source through L2: rows an earlier kernel wrote are seen without a cache flush.
    ring.initialize(runtime::DmaBypass::kDestination);

    int a_index = 0;
    int next = 1;
    int y0 = first_block * kPhases;
    for (int i = stage_rows(job, ring, staging[0], y0 - 2); i > 0; i--) ring.pop();
    build_row_group(job, buffers[a_index], y0 - 2, staging[0]);
    int queued = stage_rows(job, ring, staging[next], y0 + 2);

    for (int block = first_block; block < end_block; block++) {
        y0 = block * kPhases;
        for (; queued > 0; queued--) ring.pop();
        build_row_group(job, buffers[1 - a_index], y0 + 2, staging[next]);
        next = 1 - next;
        if (block + 1 < end_block) queued = stage_rows(job, ring, staging[next], y0 + 6);

        const Vector<uint8_t> * buffer_a = buffers[a_index];
        const Vector<uint8_t> * buffer_b = buffers[1 - a_index];
        for (int j = 0; j < kPhases && y0 + j < a.height; j++) {
            if (a.channels == 32) {
                convolve_narrow_row<1>(job, buffer_a, buffer_b, y0 + j, j);
            } else if (a.channels == 64) {
                convolve_narrow_row<2>(job, buffer_a, buffer_b, y0 + j, j);
            } else {
                for (int quad = 0; quad < job.groups / 4; quad++) {
                    convolve_wide_row(job, buffer_a, buffer_b, y0 + j, j, quad);
                }
            }
        }
        a_index = 1 - a_index;  // this block's B is the next block's A
    }

    ring.drain();
}

struct Workspace {
    runtime::HeapArray<runtime::DmaRing> rings;
    std::atomic<int> in_use;
};

bool arguments_supported(const Arguments & a) {
    if (a.input == nullptr || a.zero_points == nullptr || a.weights == nullptr || a.initial == nullptr ||
        a.output_multipliers == nullptr || a.output_biases == nullptr || a.output == nullptr) {
        return false;
    }
    return a.height >= 1 && a.width >= 2 && a.channels >= 1 &&
           (a.channels == 32 || a.channels == 64 || a.channels % 128 == 0) && a.input_stride >= a.channels &&
           (a.channels >= 128 || a.input_stride == a.channels);
}

}  // namespace

runtime::Status run(runtime::Session & session, const Arguments & arguments) {
    if (!arguments_supported(arguments)) return std::unexpected(runtime::Error::kBadParameter);

    Workspace * workspace = session.state<Workspace>();
    if (workspace == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(workspace->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);

    DepthwiseJob job{};
    job.a = arguments;
    job.groups = arguments.channels / 32;
    job.padded_width = (arguments.width + kPixelsPerIteration - 1) / kPixelsPerIteration * kPixelsPerIteration;
    job.buffer_pixels = job.padded_width + 4;
    job.blocks = (arguments.height + kPhases - 1) / kPhases;
    job.buffer_vectors = static_cast<size_t>(job.buffer_pixels) * job.groups;
    job.staging_row_bytes = (static_cast<size_t>(arguments.width) * arguments.channels + 127) / 128 * 128;

    const size_t weight_vectors = static_cast<size_t>(job.groups) * kPhases * kWeightsPerPhase;
    const size_t constant_vectors = static_cast<size_t>(job.groups) * sizeof(GroupConstants) / 128;
    const size_t buffer_vectors = static_cast<size_t>(runtime::kHvxThreads) * 2 * job.buffer_vectors;
    const size_t staging_bytes = static_cast<size_t>(runtime::kHvxThreads) * 2 * 4 * job.staging_row_bytes;
    if ((weight_vectors + constant_vectors + buffer_vectors) * 128 + staging_bytes > session.vtcm_bytes()) {
        return std::unexpected(runtime::Error::kNoMemory);
    }

    auto * vtcm = hvx::vectors(session.vtcm());
    auto * weights = reinterpret_cast<Vector<int8_t> *>(vtcm);
    auto * constants = reinterpret_cast<GroupConstants *>(vtcm + weight_vectors);
    job.weights = weights;
    job.constants = constants;
    job.buffers = vtcm + weight_vectors + constant_vectors;
    job.staging = reinterpret_cast<uint8_t *>(job.buffers + buffer_vectors);

    // A copy with a vector of room after it: prepare_constants loads whole vectors.
    const size_t weight_bytes = static_cast<size_t>(kTaps) * arguments.channels;
    {
        const runtime::HeapArray<int8_t> weight_copy = runtime::allocate_array<int8_t>(weight_bytes + 128);
        if (!weight_copy) return std::unexpected(runtime::Error::kNoMemory);
        memcpy(weight_copy.get(), arguments.weights, weight_bytes);
        prepare_constants(arguments, weight_copy.get(), job.groups, weights, constants);
    }

    if (arguments.channels < 128) {
        alignas(128) uint8_t bytes[128];
        for (int i = 0; i < 128; i++) bytes[i] = arguments.zero_points[i % arguments.channels];
        job.pad_codes = hvx::load(bytes);
    }

    if (!workspace->rings) workspace->rings = runtime::allocate_array<runtime::DmaRing>(runtime::kHvxThreads, 64);
    if (!workspace->rings) return std::unexpected(runtime::Error::kNoMemory);
    job.rings = workspace->rings.get();
    return runtime::run_parallel(session, runtime::kHvxThreads, band_thread, &job);
}

}  // namespace benzene::kernels::depthwise_int8
