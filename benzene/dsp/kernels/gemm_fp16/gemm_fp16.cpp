#include "dsp/kernels/gemm_fp16/gemm_fp16.h"

#include <atomic>
#include <new>
#include <type_traits>

#include "dsp/hmx/instructions.h"
#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/prefetch.h"
#include "dsp/hvx/vector.h"
#include "dsp/runtime/cache.h"
#include "dsp/runtime/dma_ring.h"
#include "dsp/runtime/heap.h"
#include "dsp/runtime/synchronization.h"
#include "dsp/runtime/thread_pool.h"

namespace benzene::kernels::gemm_fp16 {
namespace {

using hmx::kTileBytes;
using hvx::Mask;
using hvx::Vector;
using hvx::VectorPair;

constexpr int kOutputSlots = 16;
// fp16 tiles of 32 x 32: activation / output byte (r / 2) * 128 + c * 4 + (r % 2) * 2, weight byte (k / 2) * 128 +
// c * 4 + (k % 2) * 2
constexpr size_t kTileVectors = kTileBytes / hvx::kVectorBytes;
constexpr int kPipelinedChunks = 4;
constexpr size_t kResidentWeightBytes = 3 << 20;
constexpr size_t kWeightChunkBytes = 1 << 20;
constexpr int kMaximumChannels = 2048;

struct Fp16GemmJob {
    runtime::Session * session;
    Arguments arguments;
    int k_tiles;
    int column_blocks;
    int chunk_row_blocks;
    int chunks;
    int weight_chunk_blocks;
    int weight_chunks;
    int column_pairs;
    int resident_weights;
    size_t column_block_bytes;
    Vector<_Float16> * activation_buffers[2];
    uint8_t * weight_buffers[2];
    Vector<_Float16> * slots;
    runtime::DmaRing * driver_ring;
    // 3x3 stride-1 convolutions with a known map width: per M chunk, the contiguous span of input rows it reads (its
    // outputs' rows +- one image row and one pixel) comes into a VTCM staging buffer by one DMA, issued a chunk ahead
    // by worker 0.
    int staged;
    _Float16 * staging_buffers[2];
    int staging_first_row[2];
    runtime::DmaRing * staging_ring;
    alignas(128) std::atomic<int> staged_chunks;  // chunks whose staged rows have arrived
    alignas(128) std::atomic<int> produced;       // tiles of the current M chunk finished by HMX
    alignas(128) std::atomic<int> next_item;      // output items claimed
    alignas(128) std::atomic<int> next_gather;    // row blocks of the next M chunk claimed for gathering
    alignas(128) std::atomic<int> slot_done[kOutputSlots];
    alignas(128) runtime::SpinBarrier barrier;
};

// The row blocks of M chunk `chunk`, and its first row.
inline int chunk_rows(const Fp16GemmJob * job, int chunk, int * first_row) {
    *first_row = chunk * job->chunk_row_blocks * 32;
    const int left = (job->arguments.M - *first_row + 31) / 32;
    return left < job->chunk_row_blocks ? left : job->chunk_row_blocks;
}

// The input row of GEMM row `row`, tap `tap` (nullptr: zeros).
inline const _Float16 * source_row(const Fp16GemmJob * job, int row, int tap, const _Float16 * staging,
                                   int staging_first_row) {
    const Arguments & a = job->arguments;
    if (row >= a.M) return nullptr;

    int source;
    if (a.taps > 1 && a.input_map == nullptr) {
        const int width = a.tap_width;
        const int y = row / width + tap / 3 - 1;
        const int x = row % width + tap % 3 - 1;
        source = y >= 0 && y < a.M / width && x >= 0 && x < width ? y * width + x : -1;
    } else if (a.taps > 1) {
        source = a.input_map[static_cast<size_t>(row) * a.taps + tap];
    } else {
        source = a.input_map != nullptr ? a.input_map[row] : row;
    }
    if (source < 0) return nullptr;

    if (staging != nullptr) {
        return staging + static_cast<size_t>(source - staging_first_row) * a.row_stride;
    }
    return a.input_rows + static_cast<size_t>(source) * a.row_stride;
}

inline void staging_span(const Fp16GemmJob * job, int chunk, int * first, int * end) {
    const Arguments & a = job->arguments;
    const int first_output = chunk * job->chunk_row_blocks * 32;
    const int chunk_end = first_output + job->chunk_row_blocks * 32;
    const int end_output = chunk_end < a.M ? chunk_end : a.M;
    *first = first_output - a.tap_width - 1 > 0 ? first_output - a.tap_width - 1 : 0;
    *end = end_output + a.tap_width + 1 < a.M ? end_output + a.tap_width + 1 : a.M;
}

[[gnu::noinline]] void push_staging(Fp16GemmJob * job, int chunk) {
    int first;
    int end;
    staging_span(job, chunk, &first, &end);
    job->staging_first_row[chunk & 1] = first;
    const auto bytes = static_cast<uint32_t>((end - first) * job->arguments.row_stride * 2);
    job->staging_ring->push(job->staging_buffers[chunk & 1],
                            job->arguments.input_rows + static_cast<size_t>(first) * job->arguments.row_stride, bytes,
                            1, bytes, bytes);
}

[[gnu::noinline]] void prefetch_row_block(const Fp16GemmJob * job, int first_row, int row_blocks, int row_block) {
    const Arguments & a = job->arguments;
    const int block_first = first_row + row_block * 32;
    if (row_block >= row_blocks || block_first >= a.M || a.input_in_vtcm) return;

    const int taps = a.taps > 1 ? a.taps : 1;
    const int block_last = block_first + 31 < a.M ? block_first + 31 : a.M - 1;
    if (taps == 9 && a.tap_width > 0) {
        const int center = a.input_map != nullptr ? a.input_map[static_cast<size_t>(block_first) * 9 + 4] : block_first;
        if (center < 0) return;
        const int width = a.tap_width;
        const int y = center / width;
        const int x = center % width;
        const int first_y = y > 0 ? y - 1 : 0;
        const int first_x = x > 0 ? x - 1 : 0;
        const int pixels = first_x + 34 > width ? width - first_x : 34;
        hvx::prefetch_to_l2(a.input_rows + (static_cast<size_t>(first_y) * width + first_x) * a.row_stride,
                            static_cast<uint32_t>(pixels) * a.row_stride * 2,
                            static_cast<uint32_t>(width) * a.row_stride * 2, 3);
    } else if (taps == 1) {
        const int first_source = a.input_map != nullptr ? a.input_map[block_first] : block_first;
        const int last_source = a.input_map != nullptr ? a.input_map[block_last] : block_last;
        if (first_source < 0 || last_source < first_source || last_source - first_source > 64) return;
        const auto bytes = static_cast<uint32_t>(last_source - first_source + 1) * a.row_stride * 2;
        hvx::prefetch_to_l2(a.input_rows + static_cast<size_t>(first_source) * a.row_stride,
                            bytes < 32768 ? bytes : 32768, bytes, 1);
    }
}

alignas(128) const Vector<_Float16> zero_row[kMaximumChannels / 64] = {};

[[gnu::always_inline]] inline void interleave_strided_rows(const Vector<_Float16> * column, int stride_vectors,
                                                           bool relu, Vector<_Float16> * first_tile,
                                                           Vector<_Float16> * second_tile) {
    if (relu) {
        const Vector<_Float16> zero = Vector<_Float16>::zero();
#pragma clang loop unroll_count(4)
        for (int i = 0; i < 16; i++) {
            const VectorPair<_Float16> pair =
                interleave(max(column[(2 * i + 1) * stride_vectors], zero), max(column[2 * i * stride_vectors], zero));
            first_tile[i] = pair.low();
            second_tile[i] = pair.high();
        }
    } else {
#pragma clang loop unroll_count(4)
        for (int i = 0; i < 16; i++) {
            const VectorPair<_Float16> pair =
                interleave(column[(2 * i + 1) * stride_vectors], column[2 * i * stride_vectors]);
            first_tile[i] = pair.low();
            second_tile[i] = pair.high();
        }
    }
}

[[gnu::always_inline]] inline void interleave_listed_rows(const Vector<_Float16> * const * sources, int vector,
                                                          bool relu, Vector<_Float16> * first_tile,
                                                          Vector<_Float16> * second_tile) {
    if (relu) {
        const Vector<_Float16> zero = Vector<_Float16>::zero();
#pragma clang loop unroll_count(4)
        for (int i = 0; i < 16; i++) {
            const VectorPair<_Float16> pair =
                interleave(max(sources[2 * i + 1][vector], zero), max(sources[2 * i][vector], zero));
            first_tile[i] = pair.low();
            second_tile[i] = pair.high();
        }
    } else {
#pragma clang loop unroll_count(4)
        for (int i = 0; i < 16; i++) {
            const VectorPair<_Float16> pair = interleave(sources[2 * i + 1][vector], sources[2 * i][vector]);
            first_tile[i] = pair.low();
            second_tile[i] = pair.high();
        }
    }
}

[[gnu::noinline]] void gather_row_block(const Fp16GemmJob * job, Vector<_Float16> * activations, int first_row,
                                        int row_block, const _Float16 * staging, int staging_first_row) {
    const Arguments & a = job->arguments;
    const int taps = a.taps > 1 ? a.taps : 1;
    const int channels = a.K / taps;
    const int channel_vectors = channels / 64;
    const int block_first = first_row + row_block * 32;

    // Staged 3x3: a block of 32 outputs inside one image row, away from the borders, reads tap t's rows as 32
    // consecutive staged rows.
    if (staging != nullptr) {
        const int width = a.tap_width;
        const int y = block_first / width;
        const int x = block_first % width;
        const int height = a.M / width;
        if (x >= 1 && x + 33 <= width && y >= 1 && y + 2 <= height) {
            const int stride_vectors = a.row_stride / 64;
            for (int tap = 0; tap < 9; tap++) {
                const Vector<_Float16> * rows =
                    hvx::vectors(staging) +
                    static_cast<size_t>(block_first + (tap / 3 - 1) * width + tap % 3 - 1 - staging_first_row) *
                        stride_vectors;
                Vector<_Float16> * tiles = activations + (static_cast<size_t>(row_block) * job->k_tiles +
                                                          static_cast<size_t>(tap) * (channels / 32)) *
                                                             kTileVectors;
                for (int vector = 0; vector < channel_vectors; vector++) {
                    Vector<_Float16> * first_tile = tiles + static_cast<size_t>(2 * vector) * kTileVectors;
                    Vector<_Float16> * second_tile = first_tile + kTileVectors;
                    interleave_strided_rows(rows + vector, stride_vectors, a.relu_input != 0, first_tile, second_tile);
                }
            }
            return;
        }
    }

    const Vector<_Float16> * sources[32];
    int16_t row_y[32];
    int16_t row_x[32];
    const bool arithmetic = taps == 9 && a.input_map == nullptr;
    const int width = a.tap_width;
    const int height = arithmetic ? a.M / width : 0;
    if (arithmetic) {
        int y = block_first / width;
        int x = block_first % width;
        for (int row = 0; row < 32; row++) {
            row_y[row] = static_cast<int16_t>(block_first + row < a.M ? y : -2);
            row_x[row] = static_cast<int16_t>(x);
            x++;
            if (x == width) {
                x = 0;
                y++;
            }
        }
    }

    for (int tap = 0; tap < taps; tap++) {
        if (arithmetic) {
            const int dy = tap / 3 - 1;
            const int dx = tap % 3 - 1;
            for (int row = 0; row < 32; row++) {
                const int y = row_y[row] + dy;
                const int x = row_x[row] + dx;
                const int source = y >= 0 && y < height && x >= 0 && x < width && row_y[row] >= 0 ? y * width + x : -1;
                sources[row] =
                    source < 0
                        ? zero_row
                        : hvx::vectors(staging != nullptr
                                           ? staging + static_cast<size_t>(source - staging_first_row) * a.row_stride
                                           : a.input_rows + static_cast<size_t>(source) * a.row_stride);
            }
        } else {
            for (int row = 0; row < 32; row++) {
                const _Float16 * pointer = source_row(job, block_first + row, tap, staging, staging_first_row);
                sources[row] = pointer != nullptr ? hvx::vectors(pointer) : zero_row;
            }
        }

        Vector<_Float16> * tiles =
            activations +
            (static_cast<size_t>(row_block) * job->k_tiles + static_cast<size_t>(tap) * (channels / 32)) * kTileVectors;
        for (int vector = 0; vector < channel_vectors; vector++) {
            Vector<_Float16> * first_tile = tiles + static_cast<size_t>(2 * vector) * kTileVectors;
            Vector<_Float16> * second_tile = first_tile + kTileVectors;
            interleave_listed_rows(sources, vector, a.relu_input != 0, first_tile, second_tile);
        }
    }
}

inline void tile_position(const Fp16GemmJob * job, int row_blocks, int tile, int * column_block, int * row_block) {
    if (job->column_pairs) {
        const int pair = tile / (2 * row_blocks);
        const int within = tile % (2 * row_blocks);
        *column_block = 2 * pair + (within & 1);
        *row_block = within / 2;
        return;
    }
    *column_block = tile / row_blocks;
    *row_block = tile % row_blocks;
}

void write_output_item(const Fp16GemmJob * job, int first_row, int row_blocks, int tile, int first_tile) {
    int column_block;
    int row_block;
    tile_position(job, row_blocks, tile, &column_block, &row_block);
    const Arguments & a = job->arguments;
    const Vector<_Float16> * first =
        job->slots + static_cast<size_t>((first_tile + tile) % kOutputSlots) * kTileVectors;
    const Vector<_Float16> * second =
        job->slots + static_cast<size_t>((first_tile + tile + 1) % kOutputSlots) * kTileVectors;
    const int block_first = first_row + row_block * 32;
    const int rows = a.M - block_first < 32 ? a.M - block_first : 32;
    const Vector<_Float16> zero = Vector<_Float16>::zero();

    if (a.rows_f32 != nullptr) {
        for (int i = 0; i < 16; i++) {
            const VectorPair<_Float16> dealt = deinterleave(first[i], first[i]);
            for (int p = 0; p < 2; p++) {
                const int row = 2 * i + p;
                if (row >= rows) break;
                Vector<float> values = to_float32(p != 0 ? dealt.high() : dealt.low()).low();
                if (a.relu_output) values = max(values, Vector<float>::zero());
                const size_t output_row = a.output_map != nullptr ? static_cast<size_t>(a.output_map[block_first + row])
                                                                  : static_cast<size_t>(block_first + row);
                hvx::vector_at<float>(a.rows_f32 + output_row * a.N + column_block * 32) = values;
            }
        }
        return;
    }

    const int group = column_block * 32 / a.f16_group_width;
    const int first_column = column_block * 32 % a.f16_group_width;
    if (job->column_pairs) {
        for (int i = 0; i < 16; i++) {
            const VectorPair<_Float16> dealt = deinterleave(second[i], first[i]);
            for (int p = 0; p < 2; p++) {
                const int row = 2 * i + p;
                if (row >= rows) break;

                const size_t output_row = a.output_map != nullptr ? static_cast<size_t>(a.output_map[block_first + row])
                                                                  : static_cast<size_t>(block_first + row);
                Vector<_Float16> values = p != 0 ? dealt.high() : dealt.low();
                if (a.residual != nullptr) {
                    Vector<_Float16> residual =
                        hvx::load<_Float16>(a.residual + output_row * a.f16_row_stride + first_column);
                    if (a.relu_residual) residual = max(residual, zero);
                    values = values + residual;
                }
                if (a.second_residual != nullptr) {
                    values =
                        values + hvx::load<_Float16>(a.second_residual + output_row * a.f16_row_stride + first_column);
                }
                if (a.relu_output) values = max(values, zero);
                hvx::vector_at<_Float16>(a.f16_groups[group] + output_row * a.f16_row_stride + first_column) = values;
            }
        }
        return;
    }

    const Mask first_half = Mask::first_bytes(64);
    for (int i = 0; i < 16; i++) {
        const VectorPair<_Float16> dealt = deinterleave(first[i], first[i]);
        for (int p = 0; p < 2; p++) {
            const int row = 2 * i + p;
            if (row >= rows) break;
            const size_t output_row = a.output_map != nullptr ? static_cast<size_t>(a.output_map[block_first + row])
                                                              : static_cast<size_t>(block_first + row);
            Vector<_Float16> values = p != 0 ? dealt.high() : dealt.low();
            if (a.relu_output) values = max(values, zero);
            _Float16 * output = a.f16_groups[group] + output_row * a.f16_row_stride + first_column;
            const int offset = static_cast<int>(reinterpret_cast<uintptr_t>(output) & 127);
            store_masked(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(output) & ~uintptr_t{127}),
                         offset != 0 ? !first_half : first_half,
                         offset != 0 ? rotate_right_bytes(values, 128 - offset) : values);
        }
    }
}

[[gnu::noinline]] void run_hmx_chunk(Fp16GemmJob * job, const Vector<_Float16> * activations, int row_blocks,
                                     int first_tile, bool first_chunk) {
    const int tiles_per_load = hmx::tiles_per_load(job->k_tiles);
    const size_t block_bytes = job->column_block_bytes;

    int tile = 0;
    for (int weight_chunk = 0; weight_chunk < job->weight_chunks; weight_chunk++) {
        const int chunk_first_block = weight_chunk * job->weight_chunk_blocks;
        const int remaining = job->column_blocks - chunk_first_block;
        const int blocks = remaining < job->weight_chunk_blocks ? remaining : job->weight_chunk_blocks;

        if (!job->resident_weights || first_chunk) {
            if (weight_chunk == 0) {
                const auto bytes = static_cast<uint32_t>(blocks * block_bytes);
                job->driver_ring->push(job->weight_buffers[0],
                                       job->arguments.weights + static_cast<size_t>(chunk_first_block) * block_bytes,
                                       bytes, 1, bytes, bytes);
            }
            if (weight_chunk + 1 < job->weight_chunks) {
                const int next_first_block = chunk_first_block + blocks;
                const int next_remaining = job->column_blocks - next_first_block;
                const int next_blocks =
                    next_remaining < job->weight_chunk_blocks ? next_remaining : job->weight_chunk_blocks;
                const auto bytes = static_cast<uint32_t>(next_blocks * block_bytes);
                job->driver_ring->push(job->weight_buffers[(weight_chunk + 1) & 1],
                                       job->arguments.weights + static_cast<size_t>(next_first_block) * block_bytes,
                                       bytes, 1, bytes, bytes);
            }
            job->driver_ring->pop();
        }

        const uint8_t * weights = job->weight_buffers[weight_chunk & 1];
        for (int k = 0; k < blocks * row_blocks; k++, tile++) {
            int column_block;
            int row_block;
            tile_position(job, row_blocks, tile, &column_block, &row_block);
            const int global_tile = first_tile + tile;
            const int slot = global_tile % kOutputSlots;
            while (global_tile >= kOutputSlots && job->slot_done[slot].load() < global_tile - kOutputSlots + 1) {}
            const uint8_t * block_weights =
                weights + static_cast<size_t>(column_block - chunk_first_block) * block_bytes;
            const Vector<_Float16> * block_activations =
                activations + static_cast<size_t>(row_block) * job->k_tiles * kTileVectors;

            hmx::clear_fp16_accumulator();
            for (int first = 0; first < job->k_tiles; first += tiles_per_load) {
                const int tiles = job->k_tiles - first < tiles_per_load ? job->k_tiles - first : tiles_per_load;
                hmx::multiply_fp16(block_activations + static_cast<size_t>(first) * kTileVectors,
                                   block_weights + static_cast<size_t>(first) * kTileBytes,
                                   static_cast<uint32_t>(tiles) * kTileBytes - 1);
            }
            hmx::load_output_block(block_weights + static_cast<size_t>(job->k_tiles) * kTileBytes);
            hmx::store_fp16(job->slots + static_cast<size_t>(slot) * kTileVectors);
            job->produced.store(tile + 1);
        }
    }
}

// The HMX driver is a fifth thread, and all code it runs must be free of HVX (the four HVX contexts belong to the
// workers): noinline, scalar only.
[[gnu::noinline]] void fp16_gemm_driver_thread(void * job_pointer) {
    auto * job = static_cast<Fp16GemmJob *>(job_pointer);
    job->driver_ring->initialize_with_scalar_stores(runtime::DmaBypass::kBoth);
    hmx::lock(*job->session);
    job->barrier.wait();
    job->barrier.wait();  // (the workers gather chunk 0)

    int first_tile = 0;
    for (int chunk = 0; chunk < job->chunks; chunk++) {
        int first_row;
        const int row_blocks = chunk_rows(job, chunk, &first_row);
        job->produced.store(0);
        job->next_item.store(0);
        job->next_gather.store(0);
        job->barrier.wait();
        run_hmx_chunk(job, job->activation_buffers[chunk & 1], row_blocks, first_tile, chunk == 0);
        job->barrier.wait();  // (its outputs written, chunk + 1 gathered)
        first_tile += job->column_blocks * row_blocks;
    }

    hmx::unlock(*job->session);
}

void fp16_gemm_worker_thread(void * job_pointer, int thread) {
    auto * job = static_cast<Fp16GemmJob *>(job_pointer);
    if (job->staged && thread == 0) {
        job->staging_ring->initialize(runtime::DmaBypass::kBoth);
        push_staging(job, 0);
        if (job->chunks > 1) push_staging(job, 1);
        job->staging_ring->pop();
        job->staged_chunks.store(1);
    }
    job->barrier.wait();
    while (job->staged && job->staged_chunks.load() < 1) {}

    {
        int first_row;
        const int row_blocks = chunk_rows(job, 0, &first_row);
        const int first_block = row_blocks * thread / runtime::kHvxThreads;
        const int end_block = row_blocks * (thread + 1) / runtime::kHvxThreads;
        const _Float16 * staging = job->staged ? job->staging_buffers[0] : nullptr;
        if (staging == nullptr) prefetch_row_block(job, first_row, row_blocks, first_block);
        for (int row_block = first_block; row_block < end_block; row_block++) {
            if (staging == nullptr && row_block + 1 < end_block)
                prefetch_row_block(job, first_row, row_blocks, row_block + 1);
            gather_row_block(job, job->activation_buffers[0], first_row, row_block, staging, job->staging_first_row[0]);
        }
    }
    job->barrier.wait();

    const int step = job->column_pairs ? 2 : 1;
    int first_tile = 0;
    for (int chunk = 0; chunk < job->chunks; chunk++) {
        int first_row;
        int next_first_row = 0;
        const int row_blocks = chunk_rows(job, chunk, &first_row);
        const int tiles = job->column_blocks * row_blocks;
        const int next_row_blocks = chunk + 1 < job->chunks ? chunk_rows(job, chunk + 1, &next_first_row) : 0;
        Vector<_Float16> * next_activations = job->activation_buffers[(chunk + 1) & 1];
        const _Float16 * next_staging = job->staged ? job->staging_buffers[(chunk + 1) & 1] : nullptr;

        job->barrier.wait();
        if (job->staged && thread == 0 && next_row_blocks != 0) {
            job->staging_ring->pop();
            job->staged_chunks.store(chunk + 2);
        }

        bool gathered = next_row_blocks == 0;
        while (true) {
            int item = job->next_item.load();
            if (item < tiles && job->produced.load() >= item + step) {
                if (job->next_item.compare_exchange_strong(item, item + step)) {
                    write_output_item(job, first_row, row_blocks, item, first_tile);
                    for (int k = 0; k < step; k++) {
                        job->slot_done[(first_tile + item + k) % kOutputSlots].store(first_tile + item + k + 1);
                    }
                }
                continue;
            }

            if (!gathered && (next_staging == nullptr || job->staged_chunks.load() >= chunk + 2)) {
                const int row_block = job->next_gather.fetch_add(1);
                if (row_block < next_row_blocks) {
                    if (next_staging == nullptr)
                        prefetch_row_block(job, next_first_row, next_row_blocks, row_block + 4);
                    gather_row_block(job, next_activations, next_first_row, row_block, next_staging,
                                     job->staging_first_row[(chunk + 1) & 1]);
                    continue;
                }
                gathered = true;
            }

            if (item >= tiles) break;
        }

        job->barrier.wait();
        // staging buffer chunk & 1 is free now
        if (job->staged && thread == 0 && chunk + 2 < job->chunks) push_staging(job, chunk + 2);
        first_tile += tiles;
    }

    if (job->staged && thread == 0) job->staging_ring->drain();
}

struct Workspace {
    runtime::HeapObject<Fp16GemmJob> job;
    runtime::HeapObject<runtime::DmaRing> driver_ring;
    runtime::HeapObject<runtime::DmaRing> staging_ring;
    std::atomic<int> in_use;
};

bool arguments_supported(const Arguments & a) {
    const bool f32_output = a.output_form == OutputForm::kRowsF32;
    if (a.input_rows == nullptr || a.weights == nullptr) return false;
    if (f32_output ? (a.rows_f32 == nullptr || a.f16_groups[0] != nullptr)
                   : (a.f16_groups[0] == nullptr || a.rows_f32 != nullptr)) {
        return false;
    }

    if (a.N % 32 != 0 || a.M < 1 || a.K < 1 || a.N < 1) return false;
    if (a.K / (a.taps > 1 ? a.taps : 1) > kMaximumChannels) return false;
    const bool column_pairs = !f32_output && a.N % 64 == 0 && a.f16_group_width % 64 == 0;
    if ((a.residual != nullptr || a.second_residual != nullptr) && !column_pairs) return false;
    if (a.taps > 1) {
        const bool grid_known = a.input_map != nullptr || (a.taps == 9 && a.tap_width >= 1 && a.M % a.tap_width == 0);
        if (!grid_known || a.K % (64 * a.taps) != 0) return false;
    }
    if (a.taps <= 1 && a.K % 64 != 0) return false;
    if (a.row_stride % 64 != 0) return false;
    if (!f32_output && (a.f16_group_width == 0 || a.f16_group_width % 32 != 0 || a.N > 16 * a.f16_group_width)) {
        return false;
    }
    return true;
}

runtime::Status plan_layout(Fp16GemmJob * job, runtime::DmaRing * staging_ring) {
    const Arguments & a = job->arguments;
    const size_t vtcm_bytes = job->session->vtcm_bytes();
    const size_t slot_bytes = static_cast<size_t>(kOutputSlots) * kTileBytes;

    job->resident_weights = static_cast<size_t>(job->column_blocks) * job->column_block_bytes <= kResidentWeightBytes;
    job->weight_chunk_blocks =
        job->resident_weights ? job->column_blocks : static_cast<int>(kWeightChunkBytes / job->column_block_bytes);
    if (job->column_pairs) job->weight_chunk_blocks &= ~1;
    const int minimum_chunk_blocks = job->column_pairs ? 2 : 1;
    if (job->weight_chunk_blocks < minimum_chunk_blocks) job->weight_chunk_blocks = minimum_chunk_blocks;
    job->weight_chunks = (job->column_blocks + job->weight_chunk_blocks - 1) / job->weight_chunk_blocks;
    const size_t weight_buffer_bytes = static_cast<size_t>(job->weight_chunk_blocks) * job->column_block_bytes;
    const size_t row_block_bytes = static_cast<size_t>(job->k_tiles) * kTileBytes;
    const size_t used = slot_bytes + (job->resident_weights ? 1 : 2) * weight_buffer_bytes;
    if (used + row_block_bytes > vtcm_bytes) return std::unexpected(runtime::Error::kNoMemory);

    uint8_t * vtcm = job->session->vtcm();
    job->slots = reinterpret_cast<Vector<_Float16> *>(vtcm);
    job->weight_buffers[0] = vtcm + slot_bytes;
    job->weight_buffers[1] = job->weight_buffers[0] + (job->resident_weights ? 0 : weight_buffer_bytes);

    job->staged = a.taps == 9 && a.tap_width > 0 && staging_ring != nullptr;
    const size_t pixel_bytes = static_cast<size_t>(a.row_stride) * 2;
    const size_t staging_fixed = job->staged ? 2 * (2 * static_cast<size_t>(a.tap_width) + 2) * pixel_bytes : 0;
    const size_t per_row_block = job->staged ? 2 * (row_block_bytes + 32 * pixel_bytes) : 2 * row_block_bytes;
    const int row_blocks = (a.M + 31) / 32;
    const int fitting_one_buffer = job->staged ? 0 : static_cast<int>((vtcm_bytes - used) / row_block_bytes);

    if (fitting_one_buffer >= row_blocks) {
        job->chunk_row_blocks =
            row_blocks >= 2 * kPipelinedChunks ? (row_blocks + kPipelinedChunks - 1) / kPipelinedChunks : row_blocks;
    } else {
        job->chunk_row_blocks = vtcm_bytes > used + staging_fixed
                                    ? static_cast<int>((vtcm_bytes - used - staging_fixed) / per_row_block)
                                    : 0;
    }
    if (job->chunk_row_blocks > row_blocks) job->chunk_row_blocks = row_blocks;
    if (job->chunk_row_blocks < 1) return std::unexpected(runtime::Error::kNoMemory);
    job->chunks = (row_blocks + job->chunk_row_blocks - 1) / job->chunk_row_blocks;

    const size_t chunk_bytes = static_cast<size_t>(job->chunk_row_blocks) * row_block_bytes;
    job->activation_buffers[0] = reinterpret_cast<Vector<_Float16> *>(vtcm + used);
    job->activation_buffers[1] = job->activation_buffers[0] + chunk_bytes / hvx::kVectorBytes;
    if (job->staged) {
        const size_t staging_bytes =
            (static_cast<size_t>(job->chunk_row_blocks) * 32 + 2 * static_cast<size_t>(a.tap_width) + 2) * pixel_bytes;
        job->staging_buffers[0] = reinterpret_cast<_Float16 *>(vtcm + used + 2 * chunk_bytes);
        job->staging_buffers[1] = job->staging_buffers[0] + ((staging_bytes + 127) & ~static_cast<size_t>(127)) / 2;
        job->staging_ring = staging_ring;
    }
    return {};
}

runtime::Status run_in_workspace(Workspace & workspace, runtime::Session & session, const Arguments & arguments) {
    if (!workspace.job) workspace.job = runtime::allocate_object<Fp16GemmJob>(128);
    if (!workspace.driver_ring) workspace.driver_ring = runtime::allocate_object<runtime::DmaRing>(64);
    if (!workspace.staging_ring) workspace.staging_ring = runtime::allocate_object<runtime::DmaRing>(64);
    if (!workspace.job || !workspace.driver_ring) return std::unexpected(runtime::Error::kNoMemory);

    static_assert(std::is_trivially_destructible_v<Fp16GemmJob>);  // freed without a destructor call
    Fp16GemmJob * job = new (workspace.job.get()) Fp16GemmJob();
    job->session = &session;
    job->arguments = arguments;
    job->driver_ring = workspace.driver_ring.get();
    if (job->arguments.f16_row_stride == 0) job->arguments.f16_row_stride = job->arguments.f16_group_width;
    job->k_tiles = arguments.K / 32;
    job->column_blocks = arguments.N / 32;
    job->column_block_bytes = static_cast<size_t>(job->k_tiles + 1) * kTileBytes;
    job->column_pairs = arguments.output_form == OutputForm::kRowsF16 && job->column_blocks % 2 == 0 &&
                        arguments.f16_group_width % 64 == 0;
    if (const runtime::Status planned = plan_layout(job, workspace.staging_ring.get()); !planned) return planned;

    job->barrier.reset(runtime::kHvxThreads + 1);
    // the DMA bypasses the caches
    if (!arguments.skip_cache_flush) runtime::flush_data_cache();
    return runtime::run_parallel_with_driver(session, runtime::kHvxThreads, fp16_gemm_worker_thread,
                                             fp16_gemm_driver_thread, job);
}

}  // namespace

runtime::Status run(runtime::Session & session, const Arguments & arguments) {
    if (!arguments_supported(arguments)) return std::unexpected(runtime::Error::kBadParameter);
    Workspace * workspace = session.state<Workspace>();
    if (workspace == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(workspace->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);
    return run_in_workspace(*workspace, session, arguments);
}

}  // namespace benzene::kernels::gemm_fp16
