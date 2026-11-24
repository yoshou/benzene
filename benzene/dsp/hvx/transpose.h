#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/vector.h"

namespace benzene::hvx {
namespace transpose_network {

using Words = Vector<uint32_t>;
using StagePairs = std::make_index_sequence<16>;

// Stage s (0 .. 4) pairs vectors 16 >> s apart.
constexpr int low_vector(int stage, int pair) {
    const int distance = 16 >> stage;
    const int groups = 16 / distance;
    return pair / groups + 2 * distance * (pair % groups);
}

constexpr int high_vector(int stage, int pair) {
    return low_vector(stage, pair) + (16 >> stage);
}

// In place: `low` gets the low half of the word interleave, `high` the high half.
[[gnu::always_inline]] static inline void shuffle_words(Words & high, Words & low) {
    asm("vshuff(%0,%1,%2)" : "+v"(high.bits), "+v"(low.bits) : "r"(-4));
}

[[gnu::always_inline]] static inline void deal_words(Words & high, Words & low) {
    asm("vdeal(%0,%1,%2)" : "+v"(high.bits), "+v"(low.bits) : "r"(-4));
}

template <int Stage, size_t... Pair>
[[gnu::always_inline]] static inline void shuffle_stage(Words (&vectors)[32], std::index_sequence<Pair...>) {
    (shuffle_words(vectors[high_vector(Stage, Pair)], vectors[low_vector(Stage, Pair)]), ...);
}

template <int Stage, size_t... Pair>
[[gnu::always_inline]] static inline void deal_stage(Words (&vectors)[32], std::index_sequence<Pair...>) {
    (deal_words(vectors[high_vector(Stage, Pair)], vectors[low_vector(Stage, Pair)]), ...);
}

template <size_t... Row>
[[gnu::always_inline]] static inline void load_rows(Words (&vectors)[32], const uint8_t * source, size_t stride,
                                                    std::index_sequence<Row...>) {
    ((vectors[Row] = load<uint32_t>(source + Row * stride)), ...);
}

template <size_t... Column>
[[gnu::always_inline]] static inline void store_two_tiles(const Words (&vectors)[32], Words * first_tile,
                                                          Words * second_tile, std::index_sequence<Column...>) {
    ((first_tile[Column] = vectors[Column]), ...);
    ((second_tile[Column] = vectors[16 + Column]), ...);
}

template <size_t... Column>
[[gnu::always_inline]] static inline void load_columns(Words (&vectors)[32], const Words * columns,
                                                       std::index_sequence<Column...>) {
    ((vectors[Column] = columns[Column]), ...);
}

template <size_t... Row>
[[gnu::always_inline]] static inline void store_rows(const Words (&vectors)[32], uint8_t * destination, size_t stride,
                                                     std::index_sequence<Row...>) {
    ((store(destination + Row * stride, vectors[Row])), ...);
}

}  // namespace transpose_network

// 32 rows of 32 words (`stride` bytes apart) -> word column k as vector k: first_tile[0 .. 15], second_tile[0 .. 15].
// A perfect shuffle of the two halves rotates the 10-bit word index left by one bit; five of them transpose.
static inline void transpose_words_32x32_in_registers(const void * source, size_t stride, Vector<uint32_t> * first_tile,
                                                      Vector<uint32_t> * second_tile) {
    using namespace transpose_network;
    Words vectors[32];
    load_rows(vectors, static_cast<const uint8_t *>(source), stride, std::make_index_sequence<32>{});
    shuffle_stage<0>(vectors, StagePairs{});
    shuffle_stage<1>(vectors, StagePairs{});
    shuffle_stage<2>(vectors, StagePairs{});
    shuffle_stage<3>(vectors, StagePairs{});
    shuffle_stage<4>(vectors, StagePairs{});
    store_two_tiles(vectors, first_tile, second_tile, std::make_index_sequence<16>{});
}

// The inverse: 32 word-column vectors -> 32 rows, `stride` bytes apart.
static inline void untranspose_words_32x32_in_registers(const Vector<uint32_t> * columns, void * destination,
                                                        size_t stride) {
    using namespace transpose_network;
    Words vectors[32];
    load_columns(vectors, columns, std::make_index_sequence<32>{});
    deal_stage<4>(vectors, StagePairs{});
    deal_stage<3>(vectors, StagePairs{});
    deal_stage<2>(vectors, StagePairs{});
    deal_stage<1>(vectors, StagePairs{});
    deal_stage<0>(vectors, StagePairs{});
    store_rows(vectors, static_cast<uint8_t *>(destination), stride, std::make_index_sequence<32>{});
}

// transpose_words_32x32_in_registers with rows >= row_count read as zero; scratch0, scratch1: 32 vectors each.
static inline void transpose_words_32x32(const void * source, size_t stride, int row_count,
                                         Vector<uint32_t> * first_tile, Vector<uint32_t> * second_tile,
                                         Vector<uint32_t> * scratch0, Vector<uint32_t> * scratch1) {
    if (row_count == 32) {
        transpose_words_32x32_in_registers(source, stride, first_tile, second_tile);
        return;
    }

    const auto * rows = static_cast<const uint8_t *>(source);
    const Vector<uint32_t> zero_row = Vector<uint32_t>::zero();
    for (int v = 0; v < 16; v++) {
        const Vector<uint32_t> low = v < row_count ? load<uint32_t>(rows + v * stride) : zero_row;
        const Vector<uint32_t> high = v + 16 < row_count ? load<uint32_t>(rows + (v + 16) * stride) : zero_row;
        const VectorPair<uint32_t> shuffled = interleave(high, low);
        scratch0[2 * v] = shuffled.low();
        scratch0[2 * v + 1] = shuffled.high();
    }

    for (int stage = 0; stage < 3; stage++) {
        const Vector<uint32_t> * input = stage == 1 ? scratch1 : scratch0;
        Vector<uint32_t> * output = stage == 1 ? scratch0 : scratch1;
        for (int v = 0; v < 16; v++) {
            const VectorPair<uint32_t> shuffled = interleave(input[v + 16], input[v]);
            output[2 * v] = shuffled.low();
            output[2 * v + 1] = shuffled.high();
        }
    }

    for (int v = 0; v < 16; v++) {
        const VectorPair<uint32_t> shuffled = interleave(scratch1[v + 16], scratch1[v]);
        Vector<uint32_t> * output = v < 8 ? first_tile + 2 * v : second_tile + 2 * (v - 8);
        output[0] = shuffled.low();
        output[1] = shuffled.high();
    }
}

// The inverse: 32 word-column vectors -> rows 0 .. row_count - 1, `stride` bytes apart.
static inline void untranspose_words_32x32(const Vector<uint32_t> * columns, void * destination, size_t stride,
                                           int row_count, Vector<uint32_t> * scratch0, Vector<uint32_t> * scratch1) {
    if (row_count == 32) {
        untranspose_words_32x32_in_registers(columns, destination, stride);
        return;
    }

    for (int v = 0; v < 16; v++) {
        const VectorPair<uint32_t> dealt = deinterleave(columns[2 * v + 1], columns[2 * v]);
        scratch0[v] = dealt.low();
        scratch0[v + 16] = dealt.high();
    }

    for (int stage = 0; stage < 3; stage++) {
        const Vector<uint32_t> * input = stage == 1 ? scratch1 : scratch0;
        Vector<uint32_t> * output = stage == 1 ? scratch0 : scratch1;
        for (int v = 0; v < 16; v++) {
            const VectorPair<uint32_t> dealt = deinterleave(input[2 * v + 1], input[2 * v]);
            output[v] = dealt.low();
            output[v + 16] = dealt.high();
        }
    }

    auto * rows = static_cast<uint8_t *>(destination);
    for (int v = 0; v < 16; v++) {
        const VectorPair<uint32_t> dealt = deinterleave(scratch1[2 * v + 1], scratch1[2 * v]);
        if (v < row_count) store(rows + v * stride, dealt.low());
        if (v + 16 < row_count) store(rows + (v + 16) * stride, dealt.high());
    }
}

// 32 rows of 64 words (row r = vectors 2r, 2r + 1) -> 64 vectors of 32 words (vector k = word column k) in `result`;
// `input` is only read, `scratch` holds 64 vectors. As one array of 2048 words, a perfect shuffle of its halves
// (vshuff of vectors v and v + 32) rotates the 11-bit index left by one; five of them map (r, c) -> (c, r).
static inline void transpose_words_32x64(const Vector<uint32_t> * input, Vector<uint32_t> * result,
                                         Vector<uint32_t> * scratch) {
    const Vector<uint32_t> * source = input;
    Vector<uint32_t> * destination = result;
    for (int stage = 0; stage < 5; stage++) {
        for (int v = 0; v < 32; v++) {
            const HVX_VectorPair shuffled = Q6_W_vshuff_VVR(source[v + 32].bits, source[v].bits, -4);
            destination[2 * v].bits = Q6_V_lo_W(shuffled);
            destination[2 * v + 1].bits = Q6_V_hi_W(shuffled);
        }
        source = destination;
        destination = destination == result ? scratch : result;
    }
}

// The inverse: 64 vectors of 32 words -> 32 rows of 64 words (row r = result[2r], result[2r + 1]).
static inline void untranspose_words_64x32(const Vector<uint32_t> * input, Vector<uint32_t> * result,
                                           Vector<uint32_t> * scratch) {
    const Vector<uint32_t> * source = input;
    Vector<uint32_t> * destination = result;
    for (int stage = 0; stage < 5; stage++) {
        for (int v = 0; v < 32; v++) {
            const HVX_VectorPair dealt = Q6_W_vdeal_VVR(source[2 * v + 1].bits, source[2 * v].bits, -4);
            destination[v].bits = Q6_V_lo_W(dealt);
            destination[v + 32].bits = Q6_V_hi_W(dealt);
        }
        source = destination;
        destination = destination == result ? scratch : result;
    }
}

}  // namespace benzene::hvx
