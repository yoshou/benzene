#include "dsp/kernels/convolution_int8/convolution_int8.h"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>

#include <hexagon_types.h>

#include "dsp/hmx/instructions.h"
#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/vector.h"
#include "dsp/runtime/dma_ring.h"
#include "dsp/runtime/heap.h"
#include "dsp/runtime/synchronization.h"
#include "dsp/runtime/thread_pool.h"
#include "dsp/vector_math/functions.h"

namespace benzene::kernels::convolution_int8 {
namespace {

using hmx::kTileBytes;
using hvx::Mask;
using hvx::Vector;
using hvx::VectorPair;

constexpr int kOutputSlots = 32;
constexpr size_t kSlotBytes = 8192;
// An activation tile: 64 rows x 32 k, byte (r / 4) * 128 + k * 4 + r % 4, so a line (vector) holds a quad of rows.
constexpr size_t kSliceStoreBytes = 1024;
constexpr int kMaximumChunks = 1024;
constexpr int kMaximumColumnBlocks = 16;
constexpr int kMaximumSeluColumnBlocks = 4;
constexpr int kMinimumChunks = 4;
constexpr size_t kSlotVectors = kSlotBytes / hvx::kVectorBytes;
constexpr size_t kTileVectors = kTileBytes / hvx::kVectorBytes;
constexpr int kHotLoopAlignment = 32;

struct ConvolutionJob {
    runtime::Session * session;
    Arguments arguments;
    int output_height;
    int output_width;
    int M;
    int K;
    int k_tiles;
    int column_blocks;
    int group;  // column blocks per output item
    int f32_output;
    int chunk_row_blocks;
    int chunks;
    int pooled_width;
    size_t column_block_bytes;
    // GEMM rows are the output pixels in raster order, or with `pool` row 4q + e is pixel (2Y + e / 2, 2X + e % 2) of
    // pooled pixel q = (Y, X).
    Vector<uint8_t> * activation_buffers[2];
    uint8_t * weights;
    Vector<int8_t> * slots;
    uint8_t * staging_buffers[2];
    int staging_first_pixel[2];
    runtime::DmaRing * driver_ring;
    runtime::DmaRing * staging_ring;
    Vector<uint8_t> pad_pixels;
    Vector<uint8_t> pad_lines[4];
    alignas(128) uint8_t pad_row[256];
    vector_math::SeluConstants selu[kMaximumSeluColumnBlocks];
    // The threads order their work by these counters alone (tiles and items numbered call-wide): worker 0 stages
    // chunk c into staging buffer c & 1 once chunk c - 2 is gathered; chunk c is gathered into activation buffer
    // c & 1 once it is staged and HMX is done with chunk c - 2; HMX runs chunk c once it is gathered, and a tile once
    // its slot's previous tile is written out; the workers write out produced items first and gather in between.
    alignas(128) std::atomic<int> staged_chunks;
    alignas(128) std::atomic<int> produced;  // tiles finished by HMX
    alignas(128) std::atomic<int> next_item;
    alignas(128) std::atomic<int> next_gather;
    alignas(128) std::atomic<int> hmx_chunks_done;
    alignas(128) std::atomic<int> slot_done[kOutputSlots];  // the slot's last tile written out, + 1
    alignas(128) std::atomic<int> gathered_row_blocks[kMaximumChunks];
};

inline int chunk_rows(const ConvolutionJob * job, int chunk, int * first_row) {
    *first_row = chunk * job->chunk_row_blocks * 64;
    const int left = (job->M - *first_row + 63) / 64;
    return left < job->chunk_row_blocks ? left : job->chunk_row_blocks;
}

inline int row_block_quads(const ConvolutionJob * job, int block_first) {
    return job->M - block_first >= 64 ? 16 : (job->M - block_first + 3) / 4;
}

void staging_span(const ConvolutionJob * job, int chunk, int * first, int * end) {
    const Arguments * a = &job->arguments;
    int first_row;
    const int row_blocks = chunk_rows(job, chunk, &first_row);
    const int last_row = (first_row + row_blocks * 64 < job->M ? first_row + row_blocks * 64 : job->M) - 1;

    int first_y;
    int last_y;
    if (a->pool) {
        first_y = 2 * ((first_row / 4) / job->pooled_width);
        last_y = 2 * ((last_row / 4) / job->pooled_width) + 1;
    } else {
        first_y = first_row / job->output_width * a->stride;
        last_y = last_row / job->output_width * a->stride;
    }

    if (a->taps == 9) {
        first_y--;
        last_y++;
    }
    if (first_y < 0) first_y = 0;
    if (last_y > a->height - 1) last_y = a->height - 1;

    *first = first_y * a->width;
    *end = (last_y + 1) * a->width;
}

[[gnu::noinline]] void push_staging(ConvolutionJob * job, int chunk) {
    int first;
    int end;
    staging_span(job, chunk, &first, &end);
    job->staging_first_pixel[chunk & 1] = first;
    const auto bytes = static_cast<uint32_t>(end - first) * static_cast<uint32_t>(job->arguments.channels);
    job->staging_ring->push(job->staging_buffers[chunk & 1],
                            job->arguments.input + static_cast<size_t>(first) * job->arguments.channels, bytes, 1,
                            bytes, bytes);
}

// C = 32 / 64 layers: pixel p of a quad is GEMM row pixel_row_64(p), and row p holds pixel pixel_row_64(p).
constexpr int pixel_row_64(int pixel) {
    return pixel == 1 ? 2 : pixel == 2 ? 1 : pixel;
}

// first = e0 | e1, second = e2 | e3 (64-byte pixels) -> the lines of channels 0 .. 31, 32 .. 63, pixel_row_64 order.
inline void transpose_quad_64(Vector<uint8_t> first, Vector<uint8_t> second, Vector<uint8_t> * low_line,
                              Vector<uint8_t> * high_line) {
    const VectorPair<uint8_t> bytes = interleave(second, first);
    const VectorPair<uint8_t> halves = interleave_chunks<2>(bytes.high(), bytes.low());
    *low_line = halves.low();
    *high_line = halves.high();
}

// Rows outside the image read zero_row (or pad_row) with a step of 0: no branch in the gather loops.
alignas(128) const uint8_t zero_row[256] = {};

// The image row of GEMM rows past the map's end: outside the image at every tap.
constexpr int kRowPastTheEnd = -100;

inline Vector<uint8_t> pixel_vector(const uint8_t * base, int width, int height, int channels, int y, int x, int half,
                                    Vector<uint8_t> pad) {
    if (y < 0 || y >= height || x < 0 || x >= width) return pad;
    const uint8_t * pixel = base + (static_cast<size_t>(y) * width + x) * channels + half * 128;
    return channels == 64 ? hvx::load_unaligned(pixel) : hvx::load(pixel);
}

inline void quad_pixel(int pool, int y, int x, int quad, int row, int dy, int dx, int * pixel_y, int * pixel_x) {
    if (pool) {
        *pixel_y = y + row / 2 + dy;
        *pixel_x = x + 2 * quad + row % 2 + dx;
    } else {
        *pixel_y = y + dy;
        *pixel_x = x + 4 * quad + row + dx;
    }
}

inline Vector<uint8_t> image_tap(const uint8_t * base, int width, int height, int y, int x, int dx, int quads,
                                 Vector<uint8_t> pad) {
    if (y < 0 || y >= height) return pad;
    const uint8_t * row = base + static_cast<size_t>(y) * width + x;
    if (dx < 0 && x == 0) {
        return slide_up(hvx::load_unaligned(row), pad, 1);
    }
    const Vector<uint8_t> pixels = hvx::load_unaligned(row + dx);
    return (dx > 0 && x + 4 * quads == width) ? select(Mask::first_bytes(4 * quads - 1), pixels, pad) : pixels;
}

// C = 1 (k = tap, k >= 9 zero): n <= 16 raster quads from pixel (y, x), by a word transpose of the 9 tap vectors
// (word q of tap t: quad q's pixels at tap t).
[[gnu::noinline]] void gather_image_run(Vector<uint8_t> * __restrict tiles, const uint8_t * base, int width, int height,
                                        int y, int x, int first_line, int quads, HVX_Vector pad_bits) {
    const Vector<uint8_t> pad{pad_bits};
    const Vector<uint8_t> zero = Vector<uint8_t>::zero();

    Vector<uint8_t> taps[10];
    taps[0] = image_tap(base, width, height, y - 1, x, -1, quads, pad);
    taps[1] = image_tap(base, width, height, y - 1, x, 0, quads, pad);
    taps[2] = image_tap(base, width, height, y - 1, x, 1, quads, pad);
    taps[3] = image_tap(base, width, height, y, x, -1, quads, pad);
    taps[4] = image_tap(base, width, height, y, x, 0, quads, pad);
    taps[5] = image_tap(base, width, height, y, x, 1, quads, pad);
    taps[6] = image_tap(base, width, height, y + 1, x, -1, quads, pad);
    taps[7] = image_tap(base, width, height, y + 1, x, 0, quads, pad);
    taps[8] = image_tap(base, width, height, y + 1, x, 1, quads, pad);
    taps[9] = zero;

    Vector<uint8_t> pairs[5];
    for (int k = 0; k < 5; k++) pairs[k] = interleave_chunks<4>(taps[2 * k + 1], taps[2 * k]).low();

    Vector<uint8_t> quads_low[3];
    Vector<uint8_t> quads_high[3];
    for (int k = 0; k < 3; k++) {
        const VectorPair<uint8_t> shuffled = interleave_chunks<8>(k == 2 ? zero : pairs[2 * k + 1], pairs[2 * k]);
        quads_low[k] = shuffled.low();
        quads_high[k] = shuffled.high();
    }

    Vector<uint8_t> words[4][2];
    for (int k = 0; k < 2; k++) {
        const Vector<uint8_t> upper_low = k != 0 ? zero : quads_low[1];
        const Vector<uint8_t> upper_high = k != 0 ? zero : quads_high[1];
        const VectorPair<uint8_t> low = interleave_chunks<16>(upper_low, quads_low[2 * k]);
        const VectorPair<uint8_t> high = interleave_chunks<16>(upper_high, quads_high[2 * k]);
        words[0][k] = low.low();
        words[1][k] = low.high();
        words[2][k] = high.low();
        words[3][k] = high.high();
    }

    const Mask first_64_bytes = Mask::first_bytes(64);
    Vector<uint8_t> * __restrict lines = tiles + first_line;
    for (int g = 0; g < 4; g++) {
        const VectorPair<uint8_t> shuffled = interleave_chunks<32>(words[g][1], words[g][0]);
        if (4 * g < quads) lines[4 * g] = select(first_64_bytes, shuffled.low(), zero);
        if (4 * g + 1 < quads) lines[4 * g + 1] = select(first_64_bytes, rotate_right_bytes(shuffled.low(), 64), zero);
        if (4 * g + 2 < quads) lines[4 * g + 2] = select(first_64_bytes, shuffled.high(), zero);
        if (4 * g + 3 < quads) lines[4 * g + 3] = select(first_64_bytes, rotate_right_bytes(shuffled.high(), 64), zero);
    }
}

inline VectorPair<uint8_t> interleave_pixel_pairs(const uint8_t * p, const uint8_t * q) {
    return interleave(hvx::load(q), hvx::load(p));
}

inline void store_tap_lines(Vector<uint8_t> * const lines[6], int quad, VectorPair<uint8_t> left,
                            VectorPair<uint8_t> center, VectorPair<uint8_t> right) {
    lines[0][quad] = left.low();
    lines[1][quad] = left.high();
    lines[2][quad] = center.low();
    lines[3][quad] = center.high();
    lines[4][quad] = right.low();
    lines[5][quad] = right.high();
}

// C = 64, 3x3, an even map width, from aligned pixel pairs: per tap row the pairs are interleaved once, and each
// tap's line is the halfword interleave (first, second) of two of them, in the pixel_row_64 order.
// - pool: S(X) = pixel pair X (pixels 2X, 2X + 1) of rows A = y + dy and B = A + 1 interleaved: low = column 2X
//   (A, B), high = column 2X + 1. Window X's tap dx:
//     -1 = (high S(X - 1), low S(X)),  0 = (low S(X), high S(X)),  +1 = (high S(X), low S(X + 1)).
// - raster: T(k) = the pixel pairs k and k + 1 of row y + dy interleaved: low = pixels 2k, 2k + 2, high = 2k + 1,
//   2k + 3. The tap dx of the quad at pixel 2m:
//     -1 = (high T(m - 1), low T(m)),  0 = (low T(m), high T(m)),  +1 = (high T(m), low T(m + 1)).
void gather_run_64(const ConvolutionJob * job, Vector<uint8_t> * __restrict tiles, const uint8_t * base, int first_line,
                   int quads, int y, int x) {
    const Arguments * a = &job->arguments;
    const int width = a->width;
    const int height = a->height;
    const int pool = a->pool;
    const int row_pairs = width / 2;
    const Vector<uint8_t> zero = Vector<uint8_t>::zero();
    const VectorPair<uint8_t> zero_pair = make_pair(zero, zero);

    for (int dy = -1; dy <= 1; dy++) {
        Vector<uint8_t> * lines[6];
        for (int k = 0; k < 6; k++) {
            lines[k] = (tiles + static_cast<size_t>(2 * (3 * (dy + 1) + k / 2) + k % 2) * kTileVectors) + first_line;
        }

        if (pool) {
            const int row_a = y + dy;
            const bool a_inside = row_a >= 0 && row_a < height;
            const bool b_inside = row_a + 1 >= 0 && row_a + 1 < height;
            const int first_pair = x / 2;
            const uint8_t * row_a_pixels = a_inside ? base + static_cast<size_t>(row_a) * width * 64 : zero_row;
            const uint8_t * row_b_pixels = b_inside ? base + static_cast<size_t>(row_a + 1) * width * 64 : zero_row;
            const size_t step_a = a_inside ? 128 : 0;
            const size_t step_b = b_inside ? 128 : 0;
            VectorPair<uint8_t> previous = first_pair > 0
                                               ? interleave_pixel_pairs(row_a_pixels + (first_pair - 1) * step_a,
                                                                        row_b_pixels + (first_pair - 1) * step_b)
                                               : zero_pair;
            VectorPair<uint8_t> current =
                interleave_pixel_pairs(row_a_pixels + first_pair * step_a, row_b_pixels + first_pair * step_b);

            int quad = 0;
            [[clang::code_align(kHotLoopAlignment)]] for (; quad + 1 < quads; quad++) {
                const VectorPair<uint8_t> next = interleave_pixel_pairs(
                    row_a_pixels + (first_pair + quad + 1) * step_a, row_b_pixels + (first_pair + quad + 1) * step_b);
                const VectorPair<uint8_t> left = interleave_chunks<2>(current.low(), previous.high());
                const VectorPair<uint8_t> center = interleave_chunks<2>(current.high(), current.low());
                const VectorPair<uint8_t> right = interleave_chunks<2>(next.low(), current.high());
                store_tap_lines(lines, quad, left, center, right);
                previous = current;
                current = next;
            }

            const VectorPair<uint8_t> next =
                first_pair + quad + 1 < row_pairs
                    ? interleave_pixel_pairs(row_a_pixels + (first_pair + quad + 1) * step_a,
                                             row_b_pixels + (first_pair + quad + 1) * step_b)
                    : zero_pair;
            const VectorPair<uint8_t> left = interleave_chunks<2>(current.low(), previous.high());
            const VectorPair<uint8_t> center = interleave_chunks<2>(current.high(), current.low());
            const VectorPair<uint8_t> right = interleave_chunks<2>(next.low(), current.high());
            store_tap_lines(lines, quad, left, center, right);
        } else {
            const int row = y + dy;
            const bool inside = row >= 0 && row < height;
            const int first_pair = x / 2;
            const uint8_t * row_pixels = inside ? base + static_cast<size_t>(row) * width * 64 : job->pad_row;
            const size_t step = inside ? 128 : 0;
            const Vector<uint8_t> pad = job->pad_pixels;
            Vector<uint8_t> pair_before = first_pair > 0 ? hvx::load(row_pixels + (first_pair - 1) * step) : pad;
            Vector<uint8_t> pair0 = hvx::load(row_pixels + first_pair * step);
            VectorPair<uint8_t> previous = interleave(pair0, pair_before);

            int quad = 0;
            [[clang::code_align(kHotLoopAlignment)]] for (; quad + 1 < quads; quad++) {
                const Vector<uint8_t> pair1 = hvx::load(row_pixels + (first_pair + 2 * quad + 1) * step);
                const Vector<uint8_t> pair2 = hvx::load(row_pixels + (first_pair + 2 * quad + 2) * step);
                const VectorPair<uint8_t> current = interleave(pair1, pair0);
                const VectorPair<uint8_t> next = interleave(pair2, pair1);
                const VectorPair<uint8_t> left = interleave_chunks<2>(current.low(), previous.high());
                const VectorPair<uint8_t> center = interleave_chunks<2>(current.high(), current.low());
                const VectorPair<uint8_t> right = interleave_chunks<2>(next.low(), current.high());
                store_tap_lines(lines, quad, left, center, right);
                previous = next;
                pair0 = pair2;
            }

            const int pair1_index = first_pair + 2 * quad + 1;
            const int pair2_index = pair1_index + 1;
            const Vector<uint8_t> pair1 = pair1_index < row_pairs ? hvx::load(row_pixels + pair1_index * step) : pad;
            const Vector<uint8_t> pair2 = pair2_index < row_pairs ? hvx::load(row_pixels + pair2_index * step) : pad;
            const VectorPair<uint8_t> current = interleave(pair1, pair0);
            const VectorPair<uint8_t> next = interleave(pair2, pair1);
            const VectorPair<uint8_t> left = interleave_chunks<2>(current.low(), previous.high());
            const VectorPair<uint8_t> center = interleave_chunks<2>(current.high(), current.low());
            const VectorPair<uint8_t> right = interleave_chunks<2>(next.low(), current.high());
            store_tap_lines(lines, quad, left, center, right);
        }
    }
}

[[gnu::noinline]] void gather_run(const ConvolutionJob * job, Vector<uint8_t> * __restrict tiles, const uint8_t * base,
                                  int first_line, int quads, int y, int x) {
    const Arguments * a = &job->arguments;
    const int channels = a->channels;
    const int width = a->width;
    const int height = a->height;
    const int taps = a->taps;
    const int pool = a->pool;
    const Mask first_64_bytes = Mask::first_bytes(64);

    if (channels == 1) {
        gather_image_run(tiles, base, width, height, y, x, first_line, quads, job->pad_pixels.bits);
        return;
    }
    if (channels == 64 && taps == 9 && width % 2 == 0) {
        gather_run_64(job, tiles, base, first_line, quads, y, x);
        return;
    }

    const int span = pool ? 2 * quads : 4 * quads;
    for (int tap = 0; tap < taps; tap++) {
        const int dy = taps == 9 ? tap / 3 - 1 : 0;
        const int dx = taps == 9 ? tap % 3 - 1 : 0;
        const int row_a = y + dy;
        const int row_b = pool ? y + 1 + dy : row_a;
        const bool a_inside = row_a >= 0 && row_a < height;
        const bool b_inside = row_b >= 0 && row_b < height;
        const int first_inside = x + dx < 0;
        const int end_inside = x + span - 1 + dx > width - 1 ? quads - 1 : quads;
        const uint8_t * pixels_a = base + (static_cast<ptrdiff_t>(row_a) * width + x + dx) * channels;
        const uint8_t * pixels_b = base + (static_cast<ptrdiff_t>(row_b) * width + x + dx) * channels;

        if (channels == 64) {
            Vector<uint8_t> * low_lines = (tiles + static_cast<size_t>(2 * tap) * kTileVectors) + first_line;
            Vector<uint8_t> * high_lines = low_lines + kTileVectors;
            if (!a_inside && !b_inside) {
                for (int quad = 0; quad < quads; quad++) {
                    low_lines[quad] = job->pad_lines[0];
                    high_lines[quad] = job->pad_lines[1];
                }
                continue;
            }

            const size_t quad_step = pool ? 128 : 256;
            if (!pool) pixels_b = pixels_a + 128;
            const size_t step_a = a_inside ? quad_step : 0;
            const size_t step_b = b_inside ? quad_step : 0;
            const uint8_t * source_a = a_inside ? pixels_a + first_inside * quad_step : zero_row;
            const uint8_t * source_b = b_inside ? pixels_b + first_inside * quad_step : zero_row;
            Vector<uint8_t> * low_output = low_lines + first_inside;
            Vector<uint8_t> * high_output = high_lines + first_inside;

            int quad = first_inside;
            [[clang::code_align(kHotLoopAlignment)]] for (; quad + 4 <= end_inside; quad += 4) {
                VectorPair<uint8_t> interleaved[4];
                for (int k = 0; k < 4; k++) {
                    interleaved[k] = interleave(hvx::load_unaligned(source_b + k * step_b),
                                                hvx::load_unaligned(source_a + k * step_a));
                }
                for (int k = 0; k < 4; k++) {
                    interleaved[k] = interleave_chunks<2>(interleaved[k].high(), interleaved[k].low());
                }
                for (int k = 0; k < 4; k++) {
                    low_output[k] = interleaved[k].low();
                    high_output[k] = interleaved[k].high();
                }

                source_a += 4 * step_a;
                source_b += 4 * step_b;
                low_output += 4;
                high_output += 4;
            }

            for (; quad < end_inside; quad++) {
                Vector<uint8_t> low_line;
                Vector<uint8_t> high_line;
                transpose_quad_64(hvx::load_unaligned(source_a), hvx::load_unaligned(source_b), &low_line, &high_line);
                *low_output = low_line;
                *high_output = high_line;
                source_a += step_a;
                source_b += step_b;
                low_output++;
                high_output++;
            }

            for (int edge = 0; edge < quads; edge += quads - 1 > 0 ? quads - 1 : 1) {
                if (edge >= first_inside && edge < end_inside) continue;

                Vector<uint8_t> pixels[4];
                for (int row = 0; row < 4; row++) {
                    int pixel_y;
                    int pixel_x;
                    quad_pixel(pool, y, x, edge, row, dy, dx, &pixel_y, &pixel_x);
                    pixels[row] = pixel_vector(base, width, height, channels, pixel_y, pixel_x, 0, job->pad_pixels);
                }

                Vector<uint8_t> low_line;
                Vector<uint8_t> high_line;
                transpose_quad_64(select(first_64_bytes, pixels[0], rotate_right_bytes(pixels[1], 64)),
                                  select(first_64_bytes, pixels[2], rotate_right_bytes(pixels[3], 64)), &low_line,
                                  &high_line);
                low_lines[edge] = low_line;
                high_lines[edge] = high_line;
            }
            continue;
        }

        const size_t quad_step = (pool ? 2 : 4) * static_cast<size_t>(channels);
        for (int half = 0; half < channels / 128; half++) {
            Vector<uint8_t> * half_tiles =
                tiles + (static_cast<size_t>(tap) * (channels / 32) + 4 * half) * kTileVectors;
            Vector<uint8_t> * lines[4];
            for (int g = 0; g < 4; g++) lines[g] = (half_tiles + static_cast<size_t>(g) * kTileVectors) + first_line;
            if (!a_inside && !b_inside) {
                for (int quad = 0; quad < quads; quad++) {
                    for (int g = 0; g < 4; g++) lines[g][quad] = job->pad_lines[g];
                }
                continue;
            }

            const uint8_t * a0 = pixels_a + half * 128;
            const uint8_t * a1 = pixels_a + channels + half * 128;
            const uint8_t * b0 = pool ? pixels_b + half * 128 : pixels_a + 2 * channels + half * 128;
            const uint8_t * b1 = pool ? pixels_b + channels + half * 128 : pixels_a + 3 * channels + half * 128;
            if (!a_inside) {
                a0 = zero_row;
                a1 = zero_row;
            }
            if (!b_inside) {
                b0 = zero_row;
                b1 = zero_row;
            }
            const size_t step_a = a_inside ? quad_step : 0;
            const size_t step_b = b_inside ? quad_step : 0;

#pragma clang loop unroll_count(2)
            [[clang::code_align(kHotLoopAlignment)]] for (int quad = first_inside; quad < end_inside; quad++) {
                Vector<uint8_t> quad_lines[4];
                interleave_four(hvx::load(a0 + quad * step_a), hvx::load(a1 + quad * step_a),
                                hvx::load(b0 + quad * step_b), hvx::load(b1 + quad * step_b), quad_lines);
                for (int g = 0; g < 4; g++) lines[g][quad] = quad_lines[g];
            }

            for (int edge = 0; edge < quads; edge += quads - 1 > 0 ? quads - 1 : 1) {
                if (edge >= first_inside && edge < end_inside) continue;

                Vector<uint8_t> pixels[4];
                Vector<uint8_t> quad_lines[4];
                for (int row = 0; row < 4; row++) {
                    int pixel_y;
                    int pixel_x;
                    quad_pixel(pool, y, x, edge, row, dy, dx, &pixel_y, &pixel_x);
                    pixels[row] = pixel_vector(base, width, height, channels, pixel_y, pixel_x, half, job->pad_pixels);
                }
                interleave_four(pixels[0], pixels[1], pixels[2], pixels[3], quad_lines);
                for (int g = 0; g < 4; g++) lines[g][edge] = quad_lines[g];
            }
        }
    }
}

[[gnu::noinline]] void gather_quad_by_pixels(const ConvolutionJob * job, Vector<uint8_t> * __restrict tiles,
                                             const uint8_t * base, int quad, int first_row) {
    const Arguments * a = &job->arguments;
    const int channels = a->channels;
    const int width = a->width;
    const int height = a->height;
    const int taps = a->taps;

    int pixel_y[4];
    int pixel_x[4];
    for (int row = 0; row < 4; row++) {
        const int pixel = first_row + 4 * quad + row;
        pixel_y[row] = pixel < job->M ? pixel / width : kRowPastTheEnd;
        pixel_x[row] = pixel % width;
    }

    const Vector<uint8_t> zero = Vector<uint8_t>::zero();
    const Mask first_64_bytes = Mask::first_bytes(64);
    if (channels == 1) {
        Vector<uint8_t> line = zero;
        const auto pad = static_cast<uint32_t>(a->zero_point);
        for (int tap = 0; tap < 9; tap++) {
            uint32_t word = 0;
            for (int row = 0; row < 4; row++) {
                const int y = pixel_y[row] + tap / 3 - 1;
                const int x = pixel_x[row] + tap % 3 - 1;
                const bool inside = y >= 0 && y < height && x >= 0 && x < width;
                word |= (inside ? static_cast<uint32_t>(base[static_cast<size_t>(y) * width + x]) : pad) << (8 * row);
            }
            const Mask tap_word = and_not(Mask::first_bytes_up_to_128(4 * tap + 4), Mask::first_bytes(4 * tap));
            line = select(tap_word, reinterpret<uint8_t>(Vector<uint32_t>::splat(word)), line);
        }
        tiles[quad] = line;
        return;
    }

    for (int tap = 0; tap < taps; tap++) {
        const int dy = taps == 9 ? tap / 3 - 1 : 0;
        const int dx = taps == 9 ? tap % 3 - 1 : 0;
        for (int half = 0; half < (channels == 64 ? 1 : channels / 128); half++) {
            Vector<uint8_t> pixels[4];
            for (int row = 0; row < 4; row++) {
                pixels[row] = pixel_vector(base, width, height, channels, pixel_y[row] + dy, pixel_x[row] + dx, half,
                                           job->pad_pixels);
            }

            if (channels == 64) {
                Vector<uint8_t> low_line;
                Vector<uint8_t> high_line;
                transpose_quad_64(select(first_64_bytes, pixels[0], rotate_right_bytes(pixels[1], 64)),
                                  select(first_64_bytes, pixels[2], rotate_right_bytes(pixels[3], 64)), &low_line,
                                  &high_line);
                (tiles + static_cast<size_t>(2 * tap) * kTileVectors)[quad] = low_line;
                (tiles + static_cast<size_t>(2 * tap + 1) * kTileVectors)[quad] = high_line;
            } else {
                Vector<uint8_t> quad_lines[4];
                interleave_four(pixels[0], pixels[1], pixels[2], pixels[3], quad_lines);
                Vector<uint8_t> * half_tiles =
                    tiles + (static_cast<size_t>(tap) * (channels / 32) + 4 * half) * kTileVectors;
                for (int g = 0; g < 4; g++) {
                    (half_tiles + static_cast<size_t>(g) * kTileVectors)[quad] = quad_lines[g];
                }
            }
        }
    }
}

// e0 | e1 | e2 | e3 (pixels of 32 bytes) -> their line (k = channel), rows in the pixel_row_64 order.
inline Vector<uint8_t> transpose_quad_32(Vector<uint8_t> pixels) {
    return reinterpret<uint8_t>(interleave_halves(reinterpret<uint16_t>(interleave_halves(pixels))));
}

void gather_run_32(const ConvolutionJob * job, Vector<uint8_t> * __restrict tiles, const uint8_t * base, int first_line,
                   int quads, int y, int x) {
    const int width = job->arguments.width;
    const int height = job->arguments.height;
    const int taps = job->arguments.taps;
    const Vector<uint8_t> pad = job->pad_pixels;
    const Vector<uint8_t> pad_line = job->pad_lines[0];
    const Mask first_pixel = Mask::first_bytes(32);
    const Mask last_pixel = !Mask::first_bytes(96);

    for (int tap = 0; tap < taps; tap++) {
        const int dy = taps == 9 ? tap / 3 - 1 : 0;
        const int dx = taps == 9 ? tap % 3 - 1 : 0;
        Vector<uint8_t> * lines = (tiles + static_cast<size_t>(tap) * kTileVectors) + first_line;
        const int row = y + dy;
        if (row < 0 || row >= height) {
            for (int quad = 0; quad < quads; quad++) lines[quad] = pad_line;
            continue;
        }

        const uint8_t * pixels = base + (static_cast<ptrdiff_t>(row) * width + x + dx) * 32;
        [[clang::code_align(kHotLoopAlignment)]] for (int quad = 0; quad < quads; quad++) {
            Vector<uint8_t> four = hvx::load_unaligned(pixels + quad * 128);
            if (quad == 0 && x + dx < 0) four = select(first_pixel, pad, four);
            if (quad == quads - 1 && x + 4 * quads + dx > width) four = select(last_pixel, pad, four);
            lines[quad] = transpose_quad_32(four);
        }
    }
}

inline Vector<uint8_t> pixel_or_pad(const ConvolutionJob * job, const uint8_t * base, int y, int x, int half) {
    const Arguments * a = &job->arguments;
    if (y < 0 || y >= a->height || x < 0 || x >= a->width) return job->pad_pixels;
    return hvx::load_unaligned(base + (static_cast<size_t>(y) * a->width + x) * a->channels + half * 128);
}

[[gnu::noinline]] void gather_quad_strided(const ConvolutionJob * job, Vector<uint8_t> * __restrict tiles,
                                           const uint8_t * base, int quad, int first_row) {
    const Arguments * a = &job->arguments;
    const int channels = a->channels;
    const int taps = a->taps;
    const int stride = a->stride;

    int pixel_y[4];
    int pixel_x[4];
    for (int row = 0; row < 4; row++) {
        const int pixel = first_row + 4 * quad + row;
        pixel_y[row] = pixel < job->M ? pixel / job->output_width * stride : kRowPastTheEnd;
        pixel_x[row] = pixel % job->output_width * stride;
    }

    const Mask first_32 = Mask::first_bytes(32);
    const Mask first_64 = Mask::first_bytes(64);
    const Mask first_96 = Mask::first_bytes(96);
    [[clang::code_align(kHotLoopAlignment)]] for (int tap = 0; tap < taps; tap++) {
        const int dy = taps == 9 ? tap / 3 - 1 : 0;
        const int dx = taps == 9 ? tap % 3 - 1 : 0;
        if (channels == 32) {
            Vector<uint8_t> pixels[4];
            for (int row = 0; row < 4; row++)
                pixels[row] = pixel_or_pad(job, base, pixel_y[row] + dy, pixel_x[row] + dx, 0);
            Vector<uint8_t> four = select(first_32, pixels[0], rotate_right_bytes(pixels[1], 96));
            four = select(first_64, four, rotate_right_bytes(pixels[2], 64));
            four = select(first_96, four, rotate_right_bytes(pixels[3], 32));
            (tiles + static_cast<size_t>(tap) * kTileVectors)[quad] = transpose_quad_32(four);
            continue;
        }

        for (int half = 0; half < (channels == 64 ? 1 : channels / 128); half++) {
            Vector<uint8_t> pixels[4];
            for (int row = 0; row < 4; row++) {
                pixels[row] = pixel_or_pad(job, base, pixel_y[row] + dy, pixel_x[row] + dx, half);
            }

            if (channels == 64) {
                Vector<uint8_t> low_line;
                Vector<uint8_t> high_line;
                transpose_quad_64(select(first_64, pixels[0], rotate_right_bytes(pixels[1], 64)),
                                  select(first_64, pixels[2], rotate_right_bytes(pixels[3], 64)), &low_line,
                                  &high_line);
                (tiles + static_cast<size_t>(2 * tap) * kTileVectors)[quad] = low_line;
                (tiles + static_cast<size_t>(2 * tap + 1) * kTileVectors)[quad] = high_line;
            } else {
                Vector<uint8_t> quad_lines[4];
                interleave_four(pixels[0], pixels[1], pixels[2], pixels[3], quad_lines);
                Vector<uint8_t> * half_tiles =
                    tiles + (static_cast<size_t>(tap) * (channels / 32) + 4 * half) * kTileVectors;
                for (int g = 0; g < 4; g++) {
                    (half_tiles + static_cast<size_t>(g) * kTileVectors)[quad] = quad_lines[g];
                }
            }
        }
    }
}

void gather_row_block(const ConvolutionJob * job, Vector<uint8_t> * activations, int first_row, int row_block,
                      const uint8_t * staging, int staging_first_pixel) {
    const Arguments * a = &job->arguments;
    const int width = a->width;
    const int block_first = first_row + row_block * 64;
    const int quads = row_block_quads(job, block_first);
    Vector<uint8_t> * tiles = activations + static_cast<size_t>(row_block) * job->k_tiles * kTileVectors;
    const uint8_t * base = staging - static_cast<size_t>(staging_first_pixel) * a->channels;

    if (a->stride == 2) {
        for (int quad = 0; quad < quads; quad++) gather_quad_strided(job, tiles, base, quad, block_first);
        return;
    }

    int y;
    int x;
    if (a->pool) {
        const int pooled = block_first / 4;
        y = 2 * (pooled / job->pooled_width);
        x = 2 * (pooled % job->pooled_width);
    } else {
        y = block_first / width;
        x = block_first % width;
    }

    for (int quad = 0; quad < quads;) {
        if (!a->pool && (x + 4 > width || block_first + 4 * quad + 4 > job->M)) {
            if (a->channels == 32) {
                gather_quad_strided(job, tiles, base, quad, block_first);
            } else {
                gather_quad_by_pixels(job, tiles, base, quad, block_first);
            }
            quad++;
            x += 4 - width;
            y++;
            continue;
        }

        const int left_in_row = (width - x) / (a->pool ? 2 : 4);
        const int whole = a->pool ? quads - quad : (job->M - block_first - 4 * quad) / 4;
        int run = left_in_row < whole ? left_in_row : whole;
        if (run > quads - quad) run = quads - quad;
        if (a->channels == 32) {
            gather_run_32(job, tiles, base, quad, run, y, x);
        } else {
            gather_run(job, tiles, base, quad, run, y, x);
        }

        quad += run;
        if (a->pool) {
            x = 0;
            y += 2;
        } else {
            x += 4 * run;
            if (x == width) {
                x = 0;
                y++;
            }
        }
    }
}

// The stream's tile-line word order (lane i of vector k: column 8k + i / 4) -> lane c = column c.
inline Vector<float> plain_column_order(const Vector<float> * values) {
    const Vector<float> words01 = deinterleave(values[1], values[0]).low();
    const Vector<float> words23 = deinterleave(values[3], values[2]).low();
    return deinterleave(words23, words01).low();
}

void load_output_constants(const ConvolutionJob * job, Vector<float> * multipliers, Vector<float> * biases) {
    if (job->arguments.output_multipliers != nullptr) {
        for (int column_block = 0; column_block < job->column_blocks; column_block++) {
            multipliers[column_block] = hvx::load_unaligned(job->arguments.output_multipliers + 32 * column_block);
            biases[column_block] = hvx::load_unaligned(job->arguments.output_biases + 32 * column_block);
        }
        return;
    }

    for (int column_block = 0; column_block < job->column_blocks; column_block++) {
        const auto * constants = reinterpret_cast<const Vector<float> *>(
            job->weights + static_cast<size_t>(column_block) * job->column_block_bytes +
            static_cast<size_t>(job->k_tiles) * kTileBytes + kSliceStoreBytes);
        if (job->f32_output) {
            multipliers[column_block] = constants[0];
            biases[column_block] = constants[1];
        } else {
            multipliers[column_block] = plain_column_order(constants);
            biases[column_block] = plain_column_order(constants + 4);
        }
    }
}

// The max-pool is taken on the sums: the requantization is monotone.
inline Vector<int32_t> window_maximum(const Vector<int8_t> * __restrict slot, int quad) {
    Vector<int32_t> rows[4];
    hmx::load_int32_row_quad(slot, quad, rows);
    return max(max(rows[0], rows[1]), max(rows[2], rows[3]));
}

// Group g of an item: x[k] = sums of column block k % G, and the pixels of the group's two code vectors (->):
//   raster, G = 1   quads 2g, 2g + 1: x[4q + p] = pixel p of quad 2g + q    -> pixels 0 .. 3, 4 .. 7
//   raster, G = 2   quad g: x[2p + c] = pixel p                             -> pixels 0, 1 and 2, 3
//   raster, G = 4   quad g: x[4q + c] = pixel 2 half + q                    -> pixels 2 half and 2 half + 1
//   pool, G = 2     quads 4g ..: x[2p + c] = window max of quad 4g + p      -> pooled pixels 4g, +1 and +2, +3
//   pool, G = 4     quads 2g, 2g + 1: x[4q + c] = window max of quad 2g + q -> pooled pixels 2g and 2g + 1
[[gnu::always_inline]] inline void load_group(const Vector<int8_t> * const * slots, int group_blocks, int pool,
                                              int row_order_64, int g, int half, Vector<int32_t> * x) {
    if (!pool && group_blocks == 1) {
#pragma clang loop unroll(full)
        for (int q = 0; q < 2; q++) {
            Vector<int32_t> rows[4];
            hmx::load_int32_row_quad(slots[0], 2 * g + q, rows);
            x[4 * q] = rows[0];
            x[4 * q + 1].bits = rows[row_order_64 ? 2 : 1].bits;
            x[4 * q + 2].bits = rows[row_order_64 ? 1 : 2].bits;
            x[4 * q + 3] = rows[3];
        }
    } else if (!pool && group_blocks == 2) {
#pragma clang loop unroll(full)
        for (int c = 0; c < 2; c++) {
            Vector<int32_t> rows[4];
            hmx::load_int32_row_quad(slots[c], g, rows);
            x[c] = rows[0];
            x[2 + c].bits = rows[row_order_64 ? 2 : 1].bits;
            x[4 + c].bits = rows[row_order_64 ? 1 : 2].bits;
            x[6 + c] = rows[3];
        }
    } else if (!pool) {
#pragma clang loop unroll(full)
        for (int c = 0; c < 4; c++) {
            Vector<int32_t> rows[4];
            hmx::load_int32_row_quad(slots[c], g, rows);
            x[c].bits = rows[half ? (row_order_64 ? 1 : 2) : 0].bits;
            x[4 + c].bits = rows[half ? 3 : (row_order_64 ? 2 : 1)].bits;
        }
    } else if (group_blocks == 2) {
#pragma clang loop unroll(full)
        for (int k = 0; k < 8; k++) x[k] = window_maximum(slots[k % 2], 4 * g + k / 2);
    } else {
#pragma clang loop unroll(full)
        for (int k = 0; k < 8; k++) x[k] = window_maximum(slots[k % 4], 2 * g + k / 4);
    }
}

struct RequantizationPipeline {
    Vector<float> products[8];
    Vector<int32_t> integers[8];
};

// Step g of a two-step pipeline: converts and multiplies group g (x), adds and truncates group g - 1, and stores the
// two code vectors of group g - 2 to first_output and second_output.
[[gnu::always_inline]] inline void requantization_step(RequantizationPipeline * __restrict pipeline,
                                                       const Vector<int32_t> * __restrict x,
                                                       const Vector<float> * __restrict multipliers,
                                                       const Vector<float> * __restrict biases, int group_blocks,
                                                       uint8_t * first_output, uint8_t * second_output) {
    Vector<float> converted[8];
    Vector<float> biased[8];
    Vector<int16_t> halfwords[4];

#pragma clang loop unroll(full)
    for (int k = 0; k < 8; k++) {
        converted[k] = to_float(x[k]);
        biased[k] = pipeline->products[k] + biases[k % group_blocks];
    }

#pragma clang loop unroll(full)
    for (int k = 0; k < 8; k++) {
        if (!(k & 1)) halfwords[k / 2] = pack_to_int16_saturating(pipeline->integers[k + 1], pipeline->integers[k]);
        pipeline->products[k] = converted[k] * multipliers[k % group_blocks];
        pipeline->integers[k] = truncate_to_int32(biased[k]);
    }

    store(first_output, pack_to_uint8_saturating(halfwords[1], halfwords[0]));
    store(second_output, pack_to_uint8_saturating(halfwords[3], halfwords[2]));
}

// Group g's two code vectors go to output + g group_step and + second_offset; whole groups, as the maps have room.
[[gnu::always_inline]] inline void write_uint8_item(const Vector<int8_t> * const * slots, int group_blocks, int pool,
                                                    int row_order_64, int quads, const Vector<float> * multipliers,
                                                    const Vector<float> * biases, uint8_t * output, size_t group_step,
                                                    size_t second_offset) {
    const int groups_per_step = (!pool && group_blocks == 4) ? 2 : 1;
    const int steps = !pool               ? (group_blocks == 1 ? (quads + 1) / 2 : quads)
                      : group_blocks == 2 ? (quads + 3) / 4
                                          : (quads + 1) / 2;
    const int groups = steps * groups_per_step;
    const int warmup_steps = groups_per_step == 1 ? 2 : 1;
    alignas(128) uint8_t scratch[256];

    RequantizationPipeline pipeline;
#pragma clang loop unroll(full)
    for (int k = 0; k < 8; k++) {
        pipeline.products[k] = Vector<float>::zero();
        pipeline.integers[k] = Vector<int32_t>::zero();
    }

    for (int step = 0; step < warmup_steps; step++) {
#pragma clang loop unroll(full)
        for (int half = 0; half < groups_per_step; half++) {
            Vector<int32_t> x[8];
            if (step < steps) {
                load_group(slots, group_blocks, pool, row_order_64, step, half, x);
            } else {
#pragma clang loop unroll(full)
                for (int k = 0; k < 8; k++) x[k] = Vector<int32_t>::zero();
            }
            requantization_step(&pipeline, x, multipliers, biases, group_blocks, scratch, scratch + 128);
        }
    }
    [[clang::code_align(kHotLoopAlignment)]] for (int step = warmup_steps; step < steps; step++) {
#pragma clang loop unroll(full)
        for (int half = 0; half < groups_per_step; half++) {
            Vector<int32_t> x[8];
            load_group(slots, group_blocks, pool, row_order_64, step, half, x);
            uint8_t * destination = output + static_cast<size_t>(step * groups_per_step + half - 2) * group_step;
            requantization_step(&pipeline, x, multipliers, biases, group_blocks, destination,
                                destination + second_offset);
        }
    }
    for (int group = (steps > warmup_steps ? steps : warmup_steps) * groups_per_step; group < groups + 2; group++) {
        Vector<int32_t> x[8];
#pragma clang loop unroll(full)
        for (int k = 0; k < 8; k++) x[k] = Vector<int32_t>::zero();
        uint8_t * destination = output + static_cast<size_t>(group - 2) * group_step;
        requantization_step(&pipeline, x, multipliers, biases, group_blocks, destination, destination + second_offset);
    }
}

inline Vector<float> compact_eight_columns(Vector<float> p0, Vector<float> p1, Vector<float> p2, Vector<float> p3) {
    const Vector<float> pairs01 = interleave_chunks<32>(p1, p0).low();
    const Vector<float> pairs23 = interleave_chunks<32>(p3, p2).low();
    return interleave_chunks<64>(pairs23, pairs01).low();
}

inline void write_f32_item(const Vector<int8_t> * __restrict slot, int quads, int row_order_64, HVX_Vector scale_bits,
                           HVX_Vector bias_bits, float * rows, size_t N, int first_row, int M, int eight_columns) {
    const Vector<float> scale{scale_bits};
    const Vector<float> bias{bias_bits};
    const int groups = (quads + 1) / 2;
    Vector<float> products[8];
#pragma clang loop unroll(full)
    for (int k = 0; k < 8; k++) products[k] = Vector<float>::zero();

    [[clang::code_align(kHotLoopAlignment)]] for (int g = 0; g < groups + 1; g++) {
        Vector<int32_t> x[8];
        Vector<float> converted[8];
        Vector<float> biased[8];
        if (g < groups) {
            hmx::load_int32_row_quad(slot, 2 * g, x);
            hmx::load_int32_row_quad(slot, 2 * g + 1, x + 4);
        } else {
#pragma clang loop unroll(full)
            for (int k = 0; k < 8; k++) x[k] = Vector<int32_t>::zero();
        }

#pragma clang loop unroll(full)
        for (int k = 0; k < 8; k++) {
            converted[k] = to_float(x[k]);
            biased[k] = products[k] + bias;
        }

        if (eight_columns) {
#pragma clang loop unroll(full)
            for (int q = 0; q < 2; q++) {
                const int quad_first = first_row + 4 * (2 * (g - 1) + q);
                const auto row = [&](int p) { return biased[4 * q + (row_order_64 ? pixel_row_64(p) : p)]; };
                if (g >= 1 && quad_first < M) {
                    store(rows + static_cast<size_t>(quad_first) * 8,
                          compact_eight_columns(row(0), row(1), row(2), row(3)));
                }
            }
        }

#pragma clang loop unroll(full)
        for (int k = 0; k < 8; k++) {
            products[k] = converted[k] * scale;
            const int pixel = first_row + 4 * (2 * (g - 1) + k / 4) + (row_order_64 ? pixel_row_64(k % 4) : k % 4);
            if (!eight_columns && g >= 1 && pixel < M) {
                store(rows + static_cast<size_t>(pixel) * N, biased[k]);
            }
        }
    }
}

void write_selu_item(const Vector<int8_t> * const * slots, int group_blocks, int row_order_64, int quads,
                     const vector_math::SeluConstants * constants, uint8_t * output, size_t group_step,
                     size_t second_offset, int wide) {
    const int groups_per_step = group_blocks == 4 ? 2 : 1;
    const int steps = group_blocks == 1 ? (quads + 1) / 2 : quads;
    const Mask first_halves = Mask::first_bytes(32) | and_not(Mask::first_bytes(96), Mask::first_bytes(64));

    [[clang::code_align(kHotLoopAlignment)]] for (int step = 0; step < steps; step++) {
        for (int half = 0; half < groups_per_step; half++) {
            Vector<int32_t> x[8];
            load_group(slots, group_blocks, 0, row_order_64, step, half, x);

            Vector<int32_t> integers[8];
#pragma clang loop unroll(full)
            for (int k = 0; k < 8; k++) {
                const Vector<float> y = vector_math::selu_requantize(to_float(x[k]), constants[k % group_blocks]);
                integers[k] = truncate_to_int32(y);
            }

            Vector<int16_t> halfwords[4];
#pragma clang loop unroll(full)
            for (int k = 0; k < 4; k++) halfwords[k] = pack_to_int16_saturating(integers[2 * k + 1], integers[2 * k]);
            const Vector<uint8_t> first = pack_to_uint8_saturating(halfwords[1], halfwords[0]);
            const Vector<uint8_t> second = pack_to_uint8_saturating(halfwords[3], halfwords[2]);

            uint8_t * destination = output + static_cast<size_t>(step * groups_per_step + half) * group_step;
            if (!wide) {
                store(destination, first);
                store(destination + second_offset, second);
            } else {
                const VectorPair<uint8_t> low = interleave_chunks<32>(first, first);
                const VectorPair<uint8_t> high = interleave_chunks<32>(second, second);
                store_masked(destination, first_halves, low.low());
                store_masked(destination + 128, first_halves, low.high());
                store_masked(destination + second_offset, first_halves, high.low());
                store_masked(destination + second_offset + 128, first_halves, high.high());
            }
        }
    }
}

// Item `item` holds the tiles first_tile + item G + c (c < G), numbered in run_hmx_chunk's order.
void write_output_item(const ConvolutionJob * job, int first_row, int row_blocks, int item, int first_tile,
                       const Vector<float> * multipliers, const Vector<float> * biases) {
    const Arguments * a = &job->arguments;
    const int group = job->group;
    const int row_block = item % row_blocks;
    const int first_block = group * (item / row_blocks);
    const int block_first = first_row + row_block * 64;
    const int quads = row_block_quads(job, block_first);
    const Vector<int8_t> * slots[4];
    for (int c = 0; c < group; c++) {
        slots[c] = job->slots + static_cast<size_t>((first_tile + item * group + c) % kOutputSlots) * kSlotVectors;
    }
    const int row_order_64 = a->channels == 64 || a->channels == 32;

    if (job->f32_output) {
        const int eight_columns = a->rows_f32_columns == 8;
        write_f32_item(slots[0], quads, row_order_64, multipliers[first_block].bits, biases[first_block].bits,
                       a->rows_f32 + first_block * (eight_columns ? 8 : 32), static_cast<size_t>(a->N), block_first,
                       job->M, eight_columns);
        return;
    }

    const int first_channel = first_block * 32;
    const int map = first_channel / a->uint8_output_channels;
    const int channel_offset = first_channel % a->uint8_output_channels;
    const auto map_channels = static_cast<size_t>(a->uint8_output_channels);
    const size_t first_pooled = static_cast<size_t>(block_first) / 4;
    uint8_t * output = a->uint8_outputs[map] + channel_offset;

    if (a->selu_constants != nullptr) {
        const vector_math::SeluConstants * constants = job->selu + first_block;
        if (group == 1) {
            write_selu_item(slots, 1, row_order_64, quads, constants,
                            output + static_cast<size_t>(block_first) * map_channels, 8 * map_channels,
                            4 * map_channels, map_channels == 64);
        } else if (group == 2) {
            write_selu_item(slots, 2, row_order_64, quads, constants, output + static_cast<size_t>(block_first) * 64,
                            256, 128, 0);
        } else {
            write_selu_item(slots, 4, row_order_64, quads, constants,
                            output + static_cast<size_t>(block_first) * map_channels, 2 * map_channels, map_channels,
                            0);
        }
        return;
    }

    const Vector<float> * block_multipliers = multipliers + first_block;
    const Vector<float> * block_biases = biases + first_block;
    if (!a->pool && group == 1) {
        if (row_order_64) {
            write_uint8_item(slots, 1, 0, 1, quads, block_multipliers, block_biases,
                             output + static_cast<size_t>(block_first) * 32, 256, 128);
        } else {
            write_uint8_item(slots, 1, 0, 0, quads, block_multipliers, block_biases,
                             output + static_cast<size_t>(block_first) * 32, 256, 128);
        }
    } else if (!a->pool) {
        if (group == 2 && row_order_64) {
            write_uint8_item(slots, 2, 0, 1, quads, block_multipliers, block_biases,
                             output + static_cast<size_t>(block_first) * 64, 256, 128);
        } else if (group == 2) {
            write_uint8_item(slots, 2, 0, 0, quads, block_multipliers, block_biases,
                             output + static_cast<size_t>(block_first) * 64, 256, 128);
        } else if (row_order_64) {
            write_uint8_item(slots, 4, 0, 1, quads, block_multipliers, block_biases,
                             output + static_cast<size_t>(block_first) * map_channels, 2 * map_channels, map_channels);
        } else {
            write_uint8_item(slots, 4, 0, 0, quads, block_multipliers, block_biases,
                             output + static_cast<size_t>(block_first) * map_channels, 2 * map_channels, map_channels);
        }
    } else if (group == 2) {
        write_uint8_item(slots, 2, 1, 0, quads, block_multipliers, block_biases, output + first_pooled * 64, 256, 128);
    } else {
        write_uint8_item(slots, 4, 1, 0, quads, block_multipliers, block_biases, output + first_pooled * map_channels,
                         2 * map_channels, map_channels);
    }
}

// The HMX driver thread has no HVX context: no HVX instruction here (noinline, no memset / memcpy).
[[gnu::noinline]] void run_hmx_chunk(ConvolutionJob * job, const Vector<uint8_t> * activations, int row_blocks,
                                     int first_tile) {
    const int k_tiles = job->k_tiles;
    const int tiles_per_load = hmx::tiles_per_load(k_tiles);
    const int group = job->group;
    const int groups = job->column_blocks / group;

    int tile = 0;
    for (int g = 0; g < groups; g++) {
        for (int row_block = 0; row_block < row_blocks; row_block++) {
            for (int c = 0; c < group; c++, tile++) {
                const int global_tile = first_tile + tile;
                const int slot = global_tile % kOutputSlots;
                const int column_block = g * group + c;
                while (global_tile >= kOutputSlots && job->slot_done[slot].load() < global_tile - kOutputSlots + 1) {}

                const uint8_t * weights = job->weights + static_cast<size_t>(column_block) * job->column_block_bytes;
                const Vector<uint8_t> * block_activations =
                    activations + static_cast<size_t>(row_block) * k_tiles * kTileVectors;
                const uint8_t * output_blocks = weights + static_cast<size_t>(k_tiles) * kTileBytes;
                Vector<int8_t> * staging = job->slots + static_cast<size_t>(slot) * kSlotVectors;
                // No clear: the previous tile's last store (without :retain) left the accumulator zero.
                for (int first = 0; first < k_tiles; first += tiles_per_load) {
                    const int tiles = k_tiles - first < tiles_per_load ? k_tiles - first : tiles_per_load;
                    hmx::multiply_uint8_int8_deep_weights(block_activations + static_cast<size_t>(first) * kTileVectors,
                                                          weights + static_cast<size_t>(first) * kTileBytes,
                                                          tiles * kTileBytes - 1);
                }
                hmx::store_int32_as_uint8_slices(staging, output_blocks);
                job->produced.store(first_tile + tile + 1);
            }
        }
    }
}

[[gnu::noinline]] void convolution_driver_thread(void * job_pointer) {
    auto * job = static_cast<ConvolutionJob *>(job_pointer);
    job->driver_ring->initialize_with_scalar_stores(runtime::DmaBypass::kBoth);
    hmx::lock_and_clear(*job->session);

    const auto weight_bytes = static_cast<uint32_t>(static_cast<size_t>(job->column_blocks) * job->column_block_bytes);
    job->driver_ring->push(job->weights, job->arguments.weights, weight_bytes, 1, weight_bytes, weight_bytes);
    job->driver_ring->pop();

    int first_tile = 0;
    for (int chunk = 0; chunk < job->chunks; chunk++) {
        int first_row;
        const int row_blocks = chunk_rows(job, chunk, &first_row);
        while (job->gathered_row_blocks[chunk].load() < row_blocks) {}
        run_hmx_chunk(job, job->activation_buffers[chunk & 1], row_blocks, first_tile);
        job->hmx_chunks_done.store(chunk + 1);
        first_tile += job->column_blocks * row_blocks;
    }

    hmx::unlock(*job->session);
}

[[gnu::noinline]] void convolution_worker_thread(void * job_pointer, int thread) {
    auto * job = static_cast<ConvolutionJob *>(job_pointer);
    const int row_blocks_total = (job->M + 63) / 64;
    const int items_per_row_block = job->column_blocks / job->group;
    const int items_per_chunk = items_per_row_block * job->chunk_row_blocks;
    const int items = items_per_row_block * row_blocks_total;
    int pushed_chunks = 0;
    // Loaded at the first item: the weights are in VTCM once HMX has produced a tile.
    Vector<float> multipliers[kMaximumColumnBlocks];
    Vector<float> biases[kMaximumColumnBlocks];
    int constants_loaded = 0;

    if (thread == 0) {
        // The source through L2: rows an earlier kernel wrote are seen without a cache flush.
        job->staging_ring->initialize(runtime::DmaBypass::kDestination);
        for (; pushed_chunks < 2 && pushed_chunks < job->chunks; pushed_chunks++) push_staging(job, pushed_chunks);
    }

    while (true) {
        int item = job->next_item.load();
        if (item < items && job->produced.load() >= (item + 1) * job->group) {
            if (job->next_item.compare_exchange_strong(item, item + 1)) {
                int first_row;
                const int chunk = item / items_per_chunk;
                const int row_blocks = chunk_rows(job, chunk, &first_row);
                if (!constants_loaded) {
                    load_output_constants(job, multipliers, biases);
                    constants_loaded = 1;
                }
                write_output_item(job, first_row, row_blocks, item - chunk * items_per_chunk,
                                  chunk * job->column_blocks * job->chunk_row_blocks, multipliers, biases);
                for (int c = 0; c < job->group; c++) {
                    job->slot_done[(item * job->group + c) % kOutputSlots].store(item * job->group + c + 1);
                }
            }
            continue;
        }

        int gather = job->next_gather.load();
        if (thread == 0) {
            const int staged = job->staged_chunks.load();
            if (staged < pushed_chunks && gather >= staged * job->chunk_row_blocks) {
                job->staging_ring->pop();
                job->staged_chunks.store(staged + 1);
                continue;
            }

            if (pushed_chunks < job->chunks) {
                int first_row;
                const int row_blocks = chunk_rows(job, pushed_chunks - 2, &first_row);
                if (job->gathered_row_blocks[pushed_chunks - 2].load() >= row_blocks) {
                    push_staging(job, pushed_chunks);
                    pushed_chunks++;
                    continue;
                }
            }
        }

        if (gather < row_blocks_total) {
            const int chunk = gather / job->chunk_row_blocks;
            if (job->staged_chunks.load() > chunk && job->hmx_chunks_done.load() >= chunk - 1) {
                if (job->next_gather.compare_exchange_strong(gather, gather + 1)) {
                    int first_row;
                    chunk_rows(job, chunk, &first_row);
                    gather_row_block(job, job->activation_buffers[chunk & 1], first_row,
                                     gather - chunk * job->chunk_row_blocks, job->staging_buffers[chunk & 1],
                                     job->staging_first_pixel[chunk & 1]);
                    job->gathered_row_blocks[chunk].fetch_add(1);
                }
                continue;
            }
        }

        if (item >= items && gather >= row_blocks_total) break;
    }

    if (thread == 0) job->staging_ring->drain();
}

struct Workspace {
    runtime::HeapObject<ConvolutionJob> job;
    runtime::HeapObject<runtime::DmaRing> driver_ring;
    runtime::HeapObject<runtime::DmaRing> staging_ring;
    std::atomic<int> in_use;
};

bool arguments_supported(const Arguments & a) {
    const bool f32_output = a.output_form == OutputForm::kRowsF32;
    const bool selu = a.requantization == Requantization::kSelu;

    if (a.input == nullptr || a.weights == nullptr) return false;
    if (f32_output ? (a.rows_f32 == nullptr || a.uint8_outputs[0] != nullptr || selu)
                   : (a.uint8_outputs[0] == nullptr || a.rows_f32 != nullptr)) {
        return false;
    }
    if (selu != (a.selu_constants != nullptr)) return false;

    if (a.N < 1 || a.N % 32 != 0 || a.N > 32 * kMaximumColumnBlocks || a.height < 1 || a.width < 2 || a.channels < 1) {
        return false;
    }
    if ((a.pool && a.width % 2 != 0) || (a.taps != 1 && a.taps != 9)) return false;
    if (a.channels != 1 && a.channels != 32 && a.channels != 64 && a.channels % 128 != 0) return false;
    if (a.channels == 1 && (a.taps != 9 || a.pool)) return false;
    if (a.stride != 1 && (a.stride != 2 || a.taps != 9 || a.pool || (a.channels != 32 && a.channels != 64) ||
                          a.height % 2 != 0 || a.width % 2 != 0)) {
        return false;
    }

    if (a.zero_point < 0 || a.zero_point > 255 ||
        (a.zero_point != 0 && a.channels != 1 &&
         (a.zero_point != 128 || (a.taps == 9 && (a.channels != 32 || a.stride != 1))))) {
        return false;
    }
    if (a.padding != nullptr && (a.taps != 9 || a.stride != 1 || a.pool || a.zero_point != 0 ||
                                 (a.channels != 32 && a.channels != 64 && a.channels != 128))) {
        return false;
    }

    if (a.pool && (f32_output || a.height % 2 != 0)) return false;
    if (f32_output && a.rows_f32_columns != 0 && (a.rows_f32_columns != 8 || a.N != 32)) return false;
    if (!f32_output) {
        const bool layout_supported =
            a.N == 32
                ? ((a.uint8_output_channels == 32 || (a.uint8_output_channels == 64 && a.selu_constants != nullptr)) &&
                   !a.pool)
            : a.N == 64 ? a.uint8_output_channels == 64
                        : (a.N % 128 == 0 && a.uint8_output_channels % 128 == 0 && a.N <= 4 * a.uint8_output_channels);
        if (!layout_supported) return false;
        if ((a.output_multipliers == nullptr) != (a.output_biases == nullptr)) return false;
        if (a.selu_constants != nullptr &&
            (a.pool || a.output_multipliers != nullptr || a.N > 32 * kMaximumSeluColumnBlocks)) {
            return false;
        }
    }
    return true;
}

void prepare_padding_and_constants(ConvolutionJob * job) {
    const Arguments & a = job->arguments;
    alignas(128) uint8_t pattern[128];
    for (int i = 0; i < 128; i++) {
        pattern[i] = a.padding != nullptr ? a.padding[i % a.channels] : static_cast<uint8_t>(a.zero_point);
    }

    const Vector<uint8_t> pixels = hvx::load(pattern);
    job->pad_pixels = pixels;
    memcpy(job->pad_row, pattern, 128);
    memcpy(job->pad_row + 128, pattern, 128);
    if (a.channels == 32) {
        job->pad_lines[0] = transpose_quad_32(pixels);
    } else if (a.channels == 64) {
        transpose_quad_64(pixels, pixels, &job->pad_lines[0], &job->pad_lines[1]);
    } else {
        interleave_four(pixels, pixels, pixels, pixels, job->pad_lines);
    }

    if (a.selu_constants != nullptr) {
        const auto lanes = [&](int constant, int block) {
            return hvx::load_unaligned(a.selu_constants + static_cast<size_t>(constant) * a.N + 32 * block);
        };
        for (int block = 0; block < a.N / 32; block++) {
            vector_math::SeluConstants & c = job->selu[block];
            c.positive_scale = lanes(0, block);
            c.positive_offset = lanes(1, block);
            c.exponent_scale = lanes(2, block);
            c.exponent_offset = lanes(3, block);
            c.negative_offset = lanes(4, block);
            for (int i = 0; i < kSeluConstantCount - 5; i++) c.exponential[i] = lanes(5 + i, block);
        }
    }
}

// VTCM: the output slots, the weights, two activation buffers, then two staging buffers, so that every HMX operand
// starts on a 2 KB boundary (HMX reads a misaligned one from the boundary below).
runtime::Status plan_layout(ConvolutionJob * job) {
    const Arguments & a = job->arguments;
    const size_t vtcm_bytes = job->session->vtcm_bytes();
    const size_t slot_bytes = static_cast<size_t>(kOutputSlots) * kSlotBytes;
    const size_t weight_bytes = static_cast<size_t>(job->column_blocks) * job->column_block_bytes;
    const size_t row_block_bytes = static_cast<size_t>(job->k_tiles) * kTileBytes;
    const size_t used = slot_bytes + weight_bytes;
    const int row_blocks = (job->M + 63) / 64;
    if (used + 2 * row_block_bytes > vtcm_bytes) return std::unexpected(runtime::Error::kNoMemory);

    uint8_t * vtcm = job->session->vtcm();
    job->slots = reinterpret_cast<Vector<int8_t> *>(vtcm);
    job->weights = vtcm + slot_bytes;

    int chunk_blocks =
        row_blocks >= 2 * kMinimumChunks ? (row_blocks + kMinimumChunks - 1) / kMinimumChunks : row_blocks;
    while (true) {
        if (chunk_blocks < 1) return std::unexpected(runtime::Error::kNoMemory);

        job->chunk_row_blocks = chunk_blocks;
        job->chunks = (row_blocks + chunk_blocks - 1) / chunk_blocks;
        size_t largest_staging = 0;
        for (int chunk = 0; chunk < job->chunks; chunk++) {
            int first;
            int end;
            staging_span(job, chunk, &first, &end);
            // + 256: the gathers load past the last staged pixel
            const size_t bytes =
                ((static_cast<size_t>(end - first) * a.channels + 127) & ~static_cast<size_t>(127)) + 256;
            if (bytes > largest_staging) largest_staging = bytes;
        }

        const int buffers = job->chunks > 1 ? 2 : 1;
        if (used + buffers * (static_cast<size_t>(chunk_blocks) * row_block_bytes + largest_staging) <= vtcm_bytes) {
            const size_t chunk_bytes = static_cast<size_t>(chunk_blocks) * row_block_bytes;
            job->activation_buffers[0] = reinterpret_cast<Vector<uint8_t> *>(vtcm + used);
            job->activation_buffers[1] = job->activation_buffers[0] + chunk_bytes / hvx::kVectorBytes;
            job->staging_buffers[0] = vtcm + used + buffers * chunk_bytes;
            job->staging_buffers[1] = job->staging_buffers[0] + largest_staging;
            break;
        }
        chunk_blocks = chunk_blocks > 8 ? chunk_blocks * 7 / 8 : chunk_blocks - 1;
    }

    if (job->chunks > kMaximumChunks) {
        return std::unexpected(runtime::Error::kNoMemory);
    }
    return {};
}

runtime::Status run_in_workspace(Workspace & workspace, runtime::Session & session, const Arguments & arguments, int K,
                                 bool f32_output) {
    if (!workspace.job) workspace.job = runtime::allocate_object<ConvolutionJob>(128);
    if (!workspace.driver_ring) workspace.driver_ring = runtime::allocate_object<runtime::DmaRing>(64);
    if (!workspace.staging_ring) workspace.staging_ring = runtime::allocate_object<runtime::DmaRing>(64);
    if (!workspace.job || !workspace.driver_ring || !workspace.staging_ring) {
        return std::unexpected(runtime::Error::kNoMemory);
    }

    static_assert(std::is_trivially_destructible_v<ConvolutionJob>);
    ConvolutionJob * job = new (workspace.job.get()) ConvolutionJob();
    job->session = &session;
    job->arguments = arguments;
    job->driver_ring = workspace.driver_ring.get();
    job->staging_ring = workspace.staging_ring.get();
    job->output_height = arguments.height / arguments.stride;
    job->output_width = arguments.width / arguments.stride;
    job->M = job->output_height * job->output_width;
    job->K = K;
    job->k_tiles = (K + 31) / 32;
    job->column_blocks = arguments.N / 32;
    job->f32_output = f32_output;
    job->pooled_width = arguments.width / 2;
    job->group = (f32_output || arguments.N == 32) ? 1 : arguments.N == 64 ? 2 : 4;
    job->column_block_bytes = static_cast<size_t>(job->k_tiles + 1) * kTileBytes;
    prepare_padding_and_constants(job);
    if (const runtime::Status planned = plan_layout(job); !planned) return planned;

    return runtime::run_parallel_with_driver(session, runtime::kHvxThreads, convolution_worker_thread,
                                             convolution_driver_thread, job);
}

}  // namespace

size_t weight_stream_bytes(int K, int N) {
    return static_cast<size_t>(N / 32) * (static_cast<size_t>((K + 31) / 32) + 1) * kTileBytes;
}

runtime::Status run(runtime::Session & session, const Arguments & arguments) {
    if (!arguments_supported(arguments)) return std::unexpected(runtime::Error::kBadParameter);

    Workspace * workspace = session.state<Workspace>();
    if (workspace == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(workspace->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);

    const int K = arguments.channels == 1 ? 32 : arguments.channels * arguments.taps;
    return run_in_workspace(*workspace, session, arguments, K, arguments.output_form == OutputForm::kRowsF32);
}

}  // namespace benzene::kernels::convolution_int8
