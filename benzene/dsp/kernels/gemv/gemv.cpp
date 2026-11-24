#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>

#include "dsp/hvx/dot_products.h"
#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/vector.h"
#include "dsp/kernels/gemv/gemv.h"
#include "dsp/runtime/dma_ring.h"
#include "dsp/runtime/synchronization.h"
#include "dsp/runtime/thread_pool.h"
#include "dsp/vector_math/functions.h"

namespace benzene::kernels::gemv {
namespace {

using hvx::Vector;
using hvx::VectorPair;

constexpr int kDmaStreams = 2;

struct Workspace {
    alignas(128) int32_t token_code_words[(kMaximumK + 128) / 4];
    alignas(128) float padded_input[kMaximumK + 128];
    runtime::DmaRing rings[kDmaStreams];
    alignas(128) int32_t block_corrections[kMaximumK / 32 + 32];
    std::atomic<int> in_use;
};

constexpr size_t kTileBytes = 1024;
constexpr size_t kOutputBlockBytes = 1024;
constexpr size_t kChunkBytes = 64 << 10;

// Quantizes x per token (K a multiple of 128): scale = amax / 127 (returned), code = trunc(x / scale + 128.5) - 128
// clamped to [-127, 127] in fp32 (the W8A8 GEMM rounds through fp16 and to nearest even), byte r of code_words[j] =
// code 4j + r.
float quantize_token(const float * x, int K, int32_t * code_words, int32_t * sum) {
    const auto * values = hvx::vectors(x);
    const Vector<int32_t> magnitude_mask = Vector<int32_t>::splat(0x7fffffff);
    const Vector<float> offset = Vector<float>::splat(128.5f);
    const Vector<int32_t> one = Vector<int32_t>::splat(1);
    const Vector<int32_t> top = Vector<int32_t>::splat(255);

    Vector<int32_t> maximum = Vector<int32_t>::zero();
    for (int k = 0; k < K / 32; k++) maximum = max(maximum, reinterpret<int32_t>(values[k]) & magnitude_mask);
    maximum = vector_math::max_to_all_lanes(maximum);
    const auto amax = std::bit_cast<float>(extract_word(maximum, 0));
    const float scale = amax / 127.0f;
    const Vector<float> inverse = Vector<float>::splat(scale > 0.0f ? 1.0f / scale : 0.0f);

    Vector<int32_t> code_sum = Vector<int32_t>::zero();
    Vector<int32_t> * code_vectors = hvx::vectors(code_words);
    for (int k = 0; k < K / 128; k++) {
        const VectorPair<float> even_odd_01 = deinterleave(values[4 * k + 1], values[4 * k]);
        const VectorPair<float> even_odd_23 = deinterleave(values[4 * k + 3], values[4 * k + 2]);
        const VectorPair<float> residues_02 = deinterleave(even_odd_23.low(), even_odd_01.low());
        const VectorPair<float> residues_13 = deinterleave(even_odd_23.high(), even_odd_01.high());

        Vector<int32_t> offset_codes[4];
        const Vector<float> residues[4] = {residues_02.low(), residues_13.low(), residues_02.high(),
                                           residues_13.high()};
        for (int r = 0; r < 4; r++) {
            const Vector<int32_t> truncated = truncate_to_int32(residues[r] * inverse + offset);
            offset_codes[r] = max(min(truncated, top), one);
            code_sum = code_sum + offset_codes[r];
        }

        const Vector<int32_t> packed =
            (offset_codes[0] | (offset_codes[1] << 8)) | ((offset_codes[2] << 16) | (offset_codes[3] << 24));
        code_vectors[k] = packed ^ static_cast<int32_t>(0x80808080);  // code + 128 -> code
    }

    code_sum = vector_math::sum_to_all_lanes(code_sum);
    *sum = extract_word(code_sum, 0) - 128 * K;
    return scale;
}

// vrmpy multiplies unsigned bytes: the weights go in as w ^ 0x80 = w + 128, and correction = -128 sum(q).
inline Vector<int32_t> column_block_sums(const Vector<uint8_t> * tiles, const int32_t * code_words, int K,
                                         int32_t correction) {
    const Vector<uint8_t> flip = reinterpret<uint8_t>(Vector<int32_t>::splat(static_cast<int>(0x80808080)));
    Vector<int32_t> sum0 = Vector<int32_t>::splat(correction);
    Vector<int32_t> sum1 = Vector<int32_t>::zero();
    Vector<int32_t> sum2 = sum1;
    Vector<int32_t> sum3 = sum1;

    for (int j = 0; j < K / 4; j += 4) {
        sum0 = dot_product_accumulate_scalar(sum0, tiles[j] ^ flip, code_words[j]);
        sum1 = dot_product_accumulate_scalar(sum1, tiles[j + 1] ^ flip, code_words[j + 1]);
        sum2 = dot_product_accumulate_scalar(sum2, tiles[j + 2] ^ flip, code_words[j + 2]);
        sum3 = dot_product_accumulate_scalar(sum3, tiles[j + 3] ^ flip, code_words[j + 3]);
    }
    return (sum0 + sum1) + (sum2 + sum3);
}

inline size_t vtcm_per_thread(const runtime::Session & session, int thread_count) {
    return (session.vtcm_bytes() / thread_count) & ~static_cast<size_t>(4095);
}

// At most kCapacity - 1: a DmaRing keeps depth() < kCapacity.
inline int ring_slots(size_t vtcm_bytes, size_t slot_bytes) {
    const int fitting = static_cast<int>(vtcm_bytes / slot_bytes);
    const int most = static_cast<int>(runtime::DmaRing::kCapacity) - 1;
    return fitting < most ? fitting : most;
}

struct Int8GemvJob {
    runtime::Session * session;
    runtime::DmaRing * rings;
    const int32_t * code_words;
    int32_t correction;
    float token_scale;
    const uint8_t * stream;
    const float * column_scales;
    float * y;
    int K;
    int N;
    int k_tiles;
    int column_blocks;
    int thread_count;
    int chunk_blocks;
    size_t block_bytes;
    size_t slot_bytes;
};

[[gnu::noinline]] void int8_gemv_thread(void * job_pointer, int thread) {
    const auto * job = static_cast<const Int8GemvJob *>(job_pointer);
    const int first_block = job->column_blocks * thread / job->thread_count;
    const int end_block = job->column_blocks * (thread + 1) / job->thread_count;
    runtime::DmaRing * ring = &job->rings[thread];
    const float token_scale = job->token_scale;
    const int32_t * code_words = job->code_words;
    const int32_t correction = job->correction;
    ring->initialize(runtime::DmaBypass::kBoth);

    const size_t thread_vtcm_bytes = vtcm_per_thread(*job->session, job->thread_count);
    uint8_t * vtcm = job->session->vtcm() + thread_vtcm_bytes * thread;
    const int slots = ring_slots(thread_vtcm_bytes, job->slot_bytes);
    const int chunks = (end_block - first_block + job->chunk_blocks - 1) / job->chunk_blocks;
    const size_t tile_bytes = static_cast<size_t>(job->k_tiles) * kTileBytes;

    int prefetched = 0;
    for (int chunk = 0; chunk < chunks; chunk++) {
        while (prefetched < chunks && prefetched < chunk + slots) {
            const int block = first_block + prefetched * job->chunk_blocks;
            const int blocks = end_block - block < job->chunk_blocks ? end_block - block : job->chunk_blocks;
            ring->push(vtcm + static_cast<size_t>(prefetched % slots) * job->slot_bytes,
                       job->stream + static_cast<size_t>(block) * job->block_bytes, static_cast<uint32_t>(tile_bytes),
                       static_cast<uint32_t>(blocks), static_cast<uint32_t>(tile_bytes),
                       static_cast<uint32_t>(job->block_bytes));
            prefetched++;
        }

        ring->pop();
        const int block = first_block + chunk * job->chunk_blocks;
        const int blocks = end_block - block < job->chunk_blocks ? end_block - block : job->chunk_blocks;
        const uint8_t * tiles = vtcm + static_cast<size_t>(chunk % slots) * job->slot_bytes;
        for (int b = 0; b < blocks; b++) {
            const Vector<int32_t> sums = column_block_sums(hvx::vectors(tiles + static_cast<size_t>(b) * tile_bytes),
                                                           code_words, job->K, correction);
            const auto * scales = hvx::vectors(job->column_scales + static_cast<size_t>(block + b) * 32);
            hvx::vector_at<float>(job->y + static_cast<size_t>(block + b) * 32) =
                to_float(sums) * *scales * token_scale;
        }
    }

    ring->drain();
}

// correction[b] = -8 sum(q) over 32-k block b: the nibbles hold w + 8 (K a multiple of 128).
void compute_block_corrections(const int32_t * code_words, int K, int32_t * corrections) {
    const Vector<int32_t> * code_vectors = hvx::vectors(code_words);
    const Vector<uint8_t> flip = reinterpret<uint8_t>(Vector<int32_t>::splat(static_cast<int>(0x80808080)));

    for (int k = 0; k < K / 128; k++) {
        Vector<uint32_t> sums = dot_product_scalar(reinterpret<uint8_t>(code_vectors[k]) ^ flip, 0x01010101);
        sums = sums + rotate_right_bytes(sums, 4);
        sums = sums + rotate_right_bytes(sums, 8);
        sums = sums + rotate_right_bytes(sums, 16);

        int32_t lanes[32];
        memcpy(lanes, &sums, sizeof(lanes));
        for (int b = 0; b < 4; b++) corrections[4 * k + b] = -8 * (lanes[8 * b] - 128 * 32);
    }
}

[[gnu::always_inline]] inline Vector<int32_t> q4_block_sums(const Vector<uint8_t> * codes, const int32_t * code_words,
                                                            int32_t correction, Vector<uint8_t> low_nibbles) {
    Vector<int32_t> sum0 = Vector<int32_t>::splat(correction);
    Vector<int32_t> sum1 = Vector<int32_t>::zero();
#pragma clang loop unroll(full)
    for (int h = 0; h < 4; h++) {
        sum0 = dot_product_accumulate_scalar(sum0, codes[h] & low_nibbles, code_words[2 * h]);
        sum1 = dot_product_accumulate_scalar(sum1, codes[h] >> 4, code_words[2 * h + 1]);
    }
    return sum0 + sum1;
}

struct Q4GemvJob {
    runtime::Session * session;
    runtime::DmaRing * rings;
    const int32_t * code_words;
    const int32_t * corrections;
    float token_scale;
    const uint8_t * stream;
    float * y;
    int K;
    int row_groups;
    int thread_count;
    int chunk_groups;
    size_t group_bytes;
    size_t slot_bytes;
};

[[gnu::noinline]] void q4_gemv_thread(void * job_pointer, int thread) {
    const auto * job = static_cast<const Q4GemvJob *>(job_pointer);
    const int first_group = job->row_groups * thread / job->thread_count;
    const int end_group = job->row_groups * (thread + 1) / job->thread_count;
    const int K = job->K;
    runtime::DmaRing * ring = &job->rings[thread];
    const int32_t * code_words = job->code_words;
    const int32_t * corrections = job->corrections;
    ring->initialize(runtime::DmaBypass::kBoth);

    const size_t thread_vtcm_bytes = vtcm_per_thread(*job->session, job->thread_count);
    uint8_t * vtcm = job->session->vtcm() + thread_vtcm_bytes * thread;
    const int slots = ring_slots(thread_vtcm_bytes, job->slot_bytes);
    const int chunks = (end_group - first_group + job->chunk_groups - 1) / job->chunk_groups;
    const Vector<uint8_t> low_nibbles = {Vector<uint8_t>::splat(0x0F).bits};
    const Vector<float> token_scale = {Vector<float>::splat(job->token_scale).bits};

    int prefetched = 0;
    for (int chunk = 0; chunk < chunks; chunk++) {
        while (prefetched < chunks && prefetched < chunk + slots) {
            const int group = first_group + prefetched * job->chunk_groups;
            const int groups = end_group - group < job->chunk_groups ? end_group - group : job->chunk_groups;
            const auto bytes = static_cast<uint32_t>(static_cast<size_t>(groups) * job->group_bytes);
            ring->push(vtcm + static_cast<size_t>(prefetched % slots) * job->slot_bytes,
                       job->stream + static_cast<size_t>(group) * job->group_bytes, bytes, 1, bytes, bytes);
            prefetched++;
        }

        ring->pop();
        const int group = first_group + chunk * job->chunk_groups;
        const int groups = end_group - group < job->chunk_groups ? end_group - group : job->chunk_groups;
        const uint8_t * source = vtcm + static_cast<size_t>(chunk % slots) * job->slot_bytes;
        for (int g = 0; g < groups; g++) {
            const auto * codes = hvx::vectors(source + static_cast<size_t>(g) * job->group_bytes);
            const auto * scales = reinterpret_cast<const Vector<_Float16> *>(
                source + static_cast<size_t>(g) * job->group_bytes + static_cast<size_t>(K) * 16);
            Vector<float> accumulator0 = Vector<float>::zero();
            Vector<float> accumulator1 = Vector<float>::zero();
            for (int pair = 0; pair < K / 64; pair++) {
                const Vector<int32_t> sums0 =
                    q4_block_sums(codes + pair * 8, code_words + pair * 16, corrections[2 * pair], low_nibbles);
                const Vector<int32_t> sums1 = q4_block_sums(codes + pair * 8 + 4, code_words + pair * 16 + 8,
                                                            corrections[2 * pair + 1], low_nibbles);
                const VectorPair<float> block_scales = to_float32_deinterleaved(scales[pair]);
                accumulator0 = accumulator0 + to_float(sums0) * block_scales.low();
                accumulator1 = accumulator1 + to_float(sums1) * block_scales.high();
            }

            hvx::vector_at<float>(job->y + static_cast<size_t>(group + g) * 32) =
                (accumulator0 + accumulator1) * token_scale;
        }
    }

    ring->drain();
}

const float * padded(Workspace & workspace, const float * x, int K, int padded_K) {
    if (K % 128 == 0) return x;
    memcpy(workspace.padded_input, x, static_cast<size_t>(K) * 4);
    memset(workspace.padded_input + K, 0, static_cast<size_t>(padded_K - K) * 4);
    return workspace.padded_input;
}

runtime::Status run_int8(Workspace & workspace, runtime::Session & session, const Arguments & arguments) {
    const int padded_K = (arguments.K + 127) / 128 * 128;
    const float * x = padded(workspace, arguments.x, arguments.K, padded_K);

    Int8GemvJob job{};
    job.session = &session;
    job.rings = workspace.rings;
    job.code_words = workspace.token_code_words;
    job.stream = arguments.weights;
    job.column_scales = arguments.column_scales;
    job.y = arguments.y;
    job.K = arguments.K;
    job.N = arguments.N;
    job.k_tiles = arguments.K / 32;
    job.column_blocks = arguments.N / 32;
    job.thread_count = kDmaStreams;

    const size_t tile_bytes = static_cast<size_t>(job.k_tiles) * kTileBytes;
    job.block_bytes = tile_bytes + kOutputBlockBytes;
    const int blocks_per_chunk = static_cast<int>(kChunkBytes / tile_bytes);
    job.chunk_blocks = blocks_per_chunk > 0 ? blocks_per_chunk : 1;
    job.slot_bytes = static_cast<size_t>(job.chunk_blocks) * tile_bytes;
    if (ring_slots(vtcm_per_thread(session, job.thread_count), job.slot_bytes) < 1) {
        return std::unexpected(runtime::Error::kNoMemory);
    }

    int32_t sum;
    job.token_scale = quantize_token(x, padded_K, workspace.token_code_words, &sum);
    job.correction = -128 * sum;
    return runtime::run_parallel(session, job.thread_count, int8_gemv_thread, &job);
}

runtime::Status run_q4(Workspace & workspace, runtime::Session & session, const Arguments & arguments) {
    const int padded_K = (arguments.K + 127) / 128 * 128;
    const float * x = padded(workspace, arguments.x, arguments.K, padded_K);

    Q4GemvJob job{};
    job.session = &session;
    job.rings = workspace.rings;
    job.code_words = workspace.token_code_words;
    job.corrections = workspace.block_corrections;
    job.stream = arguments.weights;
    job.y = arguments.y;
    job.K = arguments.K;
    job.row_groups = arguments.N / 32;
    job.thread_count = kDmaStreams;

    job.group_bytes = static_cast<size_t>(arguments.K) * 18;
    const int groups_per_chunk = static_cast<int>(kChunkBytes / job.group_bytes);
    job.chunk_groups = groups_per_chunk > 0 ? groups_per_chunk : 1;
    job.slot_bytes = static_cast<size_t>(job.chunk_groups) * job.group_bytes;
    if (ring_slots(vtcm_per_thread(session, job.thread_count), job.slot_bytes) < 1) {
        return std::unexpected(runtime::Error::kNoMemory);
    }

    int32_t sum;
    job.token_scale = quantize_token(x, padded_K, workspace.token_code_words, &sum);
    compute_block_corrections(workspace.token_code_words, padded_K, workspace.block_corrections);
    return runtime::run_parallel(session, job.thread_count, q4_gemv_thread, &job);
}

bool arguments_supported(const Arguments & a) {
    const bool int8 = a.format == WeightFormat::kInt8;
    return a.x != nullptr && a.weights != nullptr && a.y != nullptr && (!int8 || a.column_scales != nullptr) &&
           a.K >= 1 && a.K <= kMaximumK && a.K % (int8 ? 32 : 64) == 0 && a.N >= 1 && a.N % 32 == 0;
}

}  // namespace

runtime::Status run(runtime::Session & session, const Arguments & arguments) {
    if (!arguments_supported(arguments)) return std::unexpected(runtime::Error::kBadParameter);

    Workspace * workspace = session.state<Workspace>();
    if (workspace == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(workspace->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);

    return arguments.format == WeightFormat::kInt8 ? run_int8(*workspace, session, arguments)
                                                   : run_q4(*workspace, session, arguments);
}

}  // namespace benzene::kernels::gemv
