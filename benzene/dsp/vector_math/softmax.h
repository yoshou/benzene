#pragma once

#include <cstdint>

#include "dsp/hvx/dot_products.h"
#include "dsp/hvx/vector.h"

namespace benzene::vector_math {

using hvx::Vector;

// P = 2^(x - I) in place over `tiles` tiles of 16 fp16 vectors (64 scores each).
// - low_clamp = I - 26 as fp16 (the clamp of x).
// - exponent_bias = I - 12 (int16 lanes; both lanes of a query hold its I).
// - sums: four f32 accumulators of the unrounded results over the lane pairs (lane q = query q).
// Per vector:
// 1. n = rint(x), d = x - n in [-0.5, 0.5] exactly.
// 2. 2^d by a cubic in fp16.
// 3. exponent + n - I + 12 on the bits (the field stays >= 0).
// 4. x 2^-12 by the fp16 product. It denormalizes correctly (results below 2^-14) and rounds the clamp value 2^-26 to
//    an exact 0, so masked scores (-65504) need no zeroing afterwards.
BENZENE_VECTOR_FUNCTION void softmax_exp2(Vector<_Float16> * __restrict scores, int tiles, Vector<_Float16> low_clamp,
                                          Vector<int16_t> exponent_bias, Vector<float> sums[4]) {
    constexpr int kBatch = 8;  // vectors through each step together: covers the instruction latency
    constexpr int kTileVectors = 16;
    const auto two_to_minus_12 = Vector<_Float16>::splat_bits(0x0C00);
    const auto coefficient0 = Vector<_Float16>::splat_bits(0x3C00);
    const auto coefficient1 = Vector<_Float16>::splat_bits(0x398C);
    const auto coefficient2 = Vector<_Float16>::splat_bits(0x33BF);
    const auto coefficient3 = Vector<_Float16>::splat_bits(0x2B1A);
    Vector<float> sum0 = sums[0];
    Vector<float> sum1 = sums[1];
    Vector<float> sum2 = sums[2];
    Vector<float> sum3 = sums[3];

    for (int tile = 0; tile < tiles; tile++) {
        Vector<_Float16> * vectors = scores + tile * kTileVectors;
        for (int first = 0; first < kTileVectors; first += kBatch) {
            Vector<_Float16> x[kBatch];
            Vector<int16_t> n[kBatch];
            Vector<_Float16> d[kBatch];
            Vector<_Float16> q[kBatch];

            for (int k = 0; k < kBatch; k++) x[k] = max(vectors[first + k], low_clamp);
            for (int k = 0; k < kBatch; k++) n[k] = round_to_int16(x[k]);
            for (int k = 0; k < kBatch; k++) d[k] = to_float16(n[k]);
            for (int k = 0; k < kBatch; k++) d[k] = x[k] - d[k];
            for (int k = 0; k < kBatch; k++) n[k] = n[k] - exponent_bias;

            for (int k = 0; k < kBatch; k++) q[k] = coefficient3 * d[k];
            for (int k = 0; k < kBatch; k++) q[k] = q[k] + coefficient2;
            for (int k = 0; k < kBatch; k++) q[k] = q[k] * d[k];
            for (int k = 0; k < kBatch; k++) q[k] = q[k] + coefficient1;
            for (int k = 0; k < kBatch; k++) q[k] = q[k] * d[k];
            for (int k = 0; k < kBatch; k++) q[k] = q[k] + coefficient0;

            // the exponent field + n: the f16 bits plus n << 10
            for (int k = 0; k < kBatch; k++) {
                q[k] = reinterpret<_Float16>(add_shifted_left(reinterpret<int16_t>(q[k]), n[k], 10));
            }

            for (int k = 0; k < kBatch; k += 4) {
                sum0 = dot_product_accumulate(sum0, q[k], two_to_minus_12);
                sum1 = dot_product_accumulate(sum1, q[k + 1], two_to_minus_12);
                sum2 = dot_product_accumulate(sum2, q[k + 2], two_to_minus_12);
                sum3 = dot_product_accumulate(sum3, q[k + 3], two_to_minus_12);
            }

            for (int k = 0; k < kBatch; k++) vectors[first + k] = q[k] * two_to_minus_12;
        }
    }

    sums[0] = sum0;
    sums[1] = sum1;
    sums[2] = sum2;
    sums[3] = sum3;
}

}  // namespace benzene::vector_math
