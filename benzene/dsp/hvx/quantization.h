#pragma once

#include <cstddef>
#include <cstdint>

#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/reductions.h"
#include "dsp/hvx/vector.h"

namespace benzene::hvx {

constexpr int kActivationTileBytes = 2048;  // 64 rows x 32 depth positions

// Four scaled f32 rows of 32 columns -> one activation line of the GEMM: byte 4c + r = 128 + round(row_r[c]), rounded
// to nearest even and saturated (+ 128 as an xor of the sign bits).
static inline Vector<uint8_t> int8_activation_line(Vector<float> row0, Vector<float> row1, Vector<float> row2,
                                                   Vector<float> row3) {
    const Vector<int8_t> codes =
        round_to_int8_interleaved(to_float16_interleaved(row0, row1), to_float16_interleaved(row2, row3));
    return reinterpret<uint8_t>(codes) ^ 0x80;
}

static inline Vector<float> scaled_f32_row(const Vector<float> * const rows[4], const Vector<float> inverse[4], int row,
                                           int k_tile) {
    return rows[row] != nullptr ? rows[row][k_tile] * inverse[row] : Vector<float>::zero();
}

// rows[0 .. 3] (K floats each, K a multiple of 32; nullptr = a padding row) quantized per row: scale = amax / 127,
// code = 128 + round(y / scale) in [1, 255]. Word k of destination + kt * 2048 gets the codes of the four rows at depth
// k of k tile kt (byte r = row r); scales[r] = the scale of row r.
static inline void quantize_four_rows_f32(const Vector<float> * const rows[4], int K, uint8_t * destination,
                                          float * scales) {
    Vector<float> inverse[4];
    for (int row = 0; row < 4; row++) {
        float amax = 0.0f;
        if (rows[row] != nullptr) {
            Vector<int32_t> maximum = Vector<int32_t>::zero();  // magnitudes compare as integers
            for (int k = 0; k < K / 32; k++) maximum = max(maximum, reinterpret<int32_t>(absolute(rows[row][k])));
            amax = horizontal_max_nonnegative(reinterpret<float>(maximum));
        }
        scales[row] = amax / 127.0f;
        inverse[row] = Vector<float>::splat(scales[row] > 0.0f ? 1.0f / scales[row] : 0.0f);
    }

    for (int k_tile = 0; k_tile < K / 32; k_tile++) {
        vector_at<uint8_t>(destination + static_cast<size_t>(k_tile) * kActivationTileBytes) =
            int8_activation_line(scaled_f32_row(rows, inverse, 0, k_tile), scaled_f32_row(rows, inverse, 1, k_tile),
                                 scaled_f32_row(rows, inverse, 2, k_tile), scaled_f32_row(rows, inverse, 3, k_tile));
    }
}

static inline Vector<_Float16> scaled_f16_row(const Vector<_Float16> * const rows[4], const Vector<_Float16> inverse[4],
                                              int row, int step) {
    return rows[row] != nullptr ? rows[row][step] * inverse[row] : Vector<_Float16>::zero();
}

// The scaled fp16 rows z0 .. z3 of one step (64 depth positions) -> the activation lines of k tiles 2 step and
// 2 step + 1. Rows (0, 2) and (1, 3) are converted together, so that the even / odd bytes of the two results are the
// four rows of the even / odd depth positions.
static inline void store_int8_line_pair(Vector<_Float16> z0, Vector<_Float16> z1, Vector<_Float16> z2,
                                        Vector<_Float16> z3, Vector<uint8_t> offset_128, uint8_t * destination,
                                        int step) {
    const Vector<uint8_t> rows_0_and_2 = reinterpret<uint8_t>(round_to_int8_interleaved(z0, z2));
    const Vector<uint8_t> rows_1_and_3 = reinterpret<uint8_t>(round_to_int8_interleaved(z1, z3));
    const VectorPair<uint8_t> lines = interleave_chunks<4>(interleave_odd_lanes(rows_1_and_3, rows_0_and_2),
                                                           interleave_even_lanes(rows_1_and_3, rows_0_and_2));
    vector_at<uint8_t>(destination + static_cast<size_t>(2 * step) * kActivationTileBytes) = lines.low() ^ offset_128;
    vector_at<uint8_t>(destination + static_cast<size_t>(2 * step + 1) * kActivationTileBytes) =
        lines.high() ^ offset_128;
}

// fp16 rows rows[0 .. 3] (K / 64 vectors each, nullptr = a padding row) quantized with the fp16 inverse scales
// inverse_bits[] into one line of the GEMM's activation tiles (destination + kt * 2048 for k tile kt). The codes are
// 128 + round(y * inverse); |y| <= amax and inverse <= 127 / amax (1 + 2^-11) keep them in [1, 255].
[[gnu::always_inline]] static inline void quantize_four_rows_f16(const Vector<_Float16> * const rows[4],
                                                                 const uint16_t inverse_bits[4], int K,
                                                                 uint8_t * destination) {
    const Vector<uint8_t> offset_128 = Vector<uint8_t>::splat(0x80);
    Vector<_Float16> inverse[4];
    for (int row = 0; row < 4; row++) inverse[row] = Vector<_Float16>::splat_bits(inverse_bits[row]);
    const int steps = K / 64;

    if (rows[0] == nullptr || rows[1] == nullptr || rows[2] == nullptr || rows[3] == nullptr) {
        for (int step = 0; step < steps; step++) {
            store_int8_line_pair(scaled_f16_row(rows, inverse, 0, step), scaled_f16_row(rows, inverse, 1, step),
                                 scaled_f16_row(rows, inverse, 2, step), scaled_f16_row(rows, inverse, 3, step),
                                 offset_128, destination, step);
        }
        return;
    }

    for (int pair = 0; pair < steps / 2; pair++) {  // two steps interleaved: hides the product latency
        const int step = 2 * pair;
        const Vector<_Float16> a0 = rows[0][step] * inverse[0];
        const Vector<_Float16> a1 = rows[1][step] * inverse[1];
        const Vector<_Float16> a2 = rows[2][step] * inverse[2];
        const Vector<_Float16> a3 = rows[3][step] * inverse[3];
        const Vector<_Float16> b0 = rows[0][step + 1] * inverse[0];
        const Vector<_Float16> b1 = rows[1][step + 1] * inverse[1];
        const Vector<_Float16> b2 = rows[2][step + 1] * inverse[2];
        const Vector<_Float16> b3 = rows[3][step + 1] * inverse[3];

        store_int8_line_pair(a0, a1, a2, a3, offset_128, destination, step);
        store_int8_line_pair(b0, b1, b2, b3, offset_128, destination, step + 1);
    }

    if (steps % 2 != 0) {
        const int step = steps - 1;
        store_int8_line_pair(rows[0][step] * inverse[0], rows[1][step] * inverse[1], rows[2][step] * inverse[2],
                             rows[3][step] * inverse[3], offset_128, destination, step);
    }
}

}  // namespace benzene::hvx
