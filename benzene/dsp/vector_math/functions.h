#pragma once

#include <cstdint>

#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/vector.h"

namespace benzene::vector_math {

using hvx::Vector;

// x 2^n: n << 23 added to the bits of x (no range check).
BENZENE_VECTOR_FUNCTION Vector<float> add_to_exponent(Vector<float> x, Vector<int32_t> n) {
    return reinterpret<float>(reinterpret<int32_t>(x) + (n << 23));
}

BENZENE_VECTOR_FUNCTION auto sign_bit_set(Vector<float> x) {
    return Vector<int32_t>::zero() > reinterpret<int32_t>(x);
}
BENZENE_VECTOR_FUNCTION auto positive(Vector<float> x) {
    return reinterpret<int32_t>(x) > Vector<int32_t>::zero();
}

BENZENE_VECTOR_FUNCTION Vector<float> minus_absolute(Vector<float> x) {
    return reinterpret<float>(reinterpret<uint32_t>(x) | 0x80000000u);
}

// 2^y for y <= ~0.05 (softmax exponents), y clamped to >= -26 so that the exponent add stays normal: 2^n p(f) with
// n = trunc(y), f in (-1, 0.05] and p a degree-3 fit of 2^f (max relative error 9.1e-5).
BENZENE_VECTOR_FUNCTION Vector<float> exp2_softmax(Vector<float> y) {
    y = max(y, -26.0f);
    const Vector<int32_t> integer_part = truncate_to_int32(y);
    const Vector<float> fraction = y - to_float(integer_part);
    Vector<float> power = fraction * 0.039662234485149384f + 0.23124286532402039f;
    power = power * fraction + 0.6916316747665405f;
    power = power * fraction + 1.000005841255188f;
    return add_to_exponent(power, integer_part);
}

// 2^y with y clamped to [-100, 100]: 2^n p(f) with n = trunc(y), f in (-1, 1) and p the degree-7 Taylor polynomial
// of 2^f in f ln 2 (relative error < 2e-7).
BENZENE_VECTOR_FUNCTION Vector<float> exp2(Vector<float> y) {
    constexpr float kTaylorCoefficients[8] = {1.0f,      1.0f,       0.5f,       1.0f / 6,
                                              1.0f / 24, 1.0f / 120, 1.0f / 720, 1.0f / 5040};
    y = min(max(y, -100.0f), 100.0f);
    const Vector<int32_t> integer_part = truncate_to_int32(y);
    const Vector<float> fraction = (y - to_float(integer_part)) * 0.69314718055994531f;
    Vector<float> power = Vector<float>::splat(kTaylorCoefficients[7]);
    for (int k = 6; k >= 0; k--) power = power * fraction + kTaylorCoefficients[k];
    return add_to_exponent(power, integer_part);
}

// exp2 of values[0 .. Count) in place (Count <= 8): the same operations, stage by stage over the vectors.
template <int Count>
BENZENE_VECTOR_FUNCTION void exp2_batch(Vector<float> * values) {
    constexpr float kTaylorCoefficients[7] = {1.0f, 1.0f, 0.5f, 1.0f / 6, 1.0f / 24, 1.0f / 120, 1.0f / 720};
    Vector<int32_t> integer_part[8];
    Vector<float> fraction[8];
    Vector<float> power[8];

#pragma clang loop unroll(full)
    for (int k = 0; k < Count; k++) values[k] = min(max(values[k], -100.0f), 100.0f);

#pragma clang loop unroll(full)
    for (int k = 0; k < Count; k++) integer_part[k] = truncate_to_int32(values[k]);
#pragma clang loop unroll(full)
    for (int k = 0; k < Count; k++) fraction[k] = to_float(integer_part[k]);
#pragma clang loop unroll(full)
    for (int k = 0; k < Count; k++) fraction[k] = values[k] - fraction[k];
#pragma clang loop unroll(full)
    for (int k = 0; k < Count; k++) fraction[k] = fraction[k] * 0.69314718055994531f;

#pragma clang loop unroll(full)
    for (int k = 0; k < Count; k++) power[k] = Vector<float>::splat(1.0f / 5040);
#pragma clang loop unroll(full)
    for (int term = 6; term >= 0; term--) {
#pragma clang loop unroll(full)
        for (int k = 0; k < Count; k++) power[k] = power[k] * fraction[k];
#pragma clang loop unroll(full)
        for (int k = 0; k < Count; k++) power[k] = power[k] + kTaylorCoefficients[term];
    }

#pragma clang loop unroll(full)
    for (int k = 0; k < Count; k++) values[k] = add_to_exponent(power[k], integer_part[k]);
}

// 1 / x for x in ~[1, 2^20]: bit-trick seed and 3 Newton steps.
BENZENE_VECTOR_FUNCTION Vector<float> reciprocal(Vector<float> x) {
    Vector<float> estimate = reinterpret<float>(0x7EF311C3 - reinterpret<int32_t>(x));
    for (int step = 0; step < 3; step++) {
        const Vector<float> correction = 2.0f - x * estimate;
        estimate = estimate * correction;
    }
    return estimate;
}

// reciprocal of x[0 .. Count) into result[0 .. Count): the same operations, step by step over the vectors.
template <int Count>
BENZENE_VECTOR_FUNCTION void reciprocal_batch(const Vector<float> * x, Vector<float> * result) {
#pragma clang loop unroll(full)
    for (int k = 0; k < Count; k++) result[k] = reinterpret<float>(0x7EF311C3 - reinterpret<int32_t>(x[k]));
#pragma clang loop unroll(full)
    for (int step = 0; step < 3; step++) {
#pragma clang loop unroll(full)
        for (int k = 0; k < Count; k++) result[k] = result[k] * (2.0f - x[k] * result[k]);
    }
}

// gelu_tanh(x) = x * sigmoid(z), z = 2 sqrt(2 / pi) (x + 0.044715 x^3). With t = 2^(-|z| log2 e) in (0, 1]:
// sigmoid(|z|) = r = 1 / (1 + t) (degree-5 fit on [0, 1], max relative error 5.1e-5), sigmoid(-|z|) = t r.
BENZENE_VECTOR_FUNCTION Vector<float> gelu_tanh(Vector<float> x) {
    const Vector<float> z_log2e = x * (x * x * 0.1029432396f + 2.302208198f);
    const Vector<float> t = exp2_softmax(minus_absolute(z_log2e));
    Vector<float> r = t * -0.10448431223630905f + 0.4179374575614929f;
    r = r * t + -0.7705726027488708f;
    r = r * t + 0.9534206390380859f;
    r = r * t + -0.9962756633758545f;
    r = r * t + 0.9999489784240723f;
    return x * select(sign_bit_set(z_log2e), r * t, r);
}

// Exact (erf) GELU of f32 x: x * (x >= 0 ? 1 - Q / 2 : Q / 2), Q = erfc(|x| / sqrt 2) by Abramowitz-Stegun 7.1.26
// (|error| < 1.5e-7).
BENZENE_VECTOR_FUNCTION Vector<float> gelu_erf(Vector<float> x) {
    const Vector<float> z = absolute(x) * 0.70710678118654752f;
    const Vector<float> t = reciprocal(z * 0.3275911f + 1.0f);
    Vector<float> p = t * 1.061405429f + -1.453152027f;
    p = p * t + 1.421413741f;
    p = p * t + -0.284496736f;
    p = p * t + 0.254829592f;
    p = p * t;

    const Vector<float> gaussian = exp2_softmax(-(z * z * 1.4426950408889634f));
    const Vector<float> half_q = p * gaussian * 0.5f;
    const Vector<float> one_minus_half_q = 1.0f + -half_q;
    return x * select(sign_bit_set(x), half_q, one_minus_half_q);
}

// The sum, the max and the bitwise or of all lanes, in every lane.
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> sum_to_all_lanes(Vector<Lane> v) {
    for (int bytes = 64; bytes >= static_cast<int>(sizeof(Lane)); bytes >>= 1) v = v + rotate_right_bytes(v, bytes);
    return v;
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> max_to_all_lanes(Vector<Lane> v) {
    for (int bytes = 64; bytes >= static_cast<int>(sizeof(Lane)); bytes >>= 1) v = max(v, rotate_right_bytes(v, bytes));
    return v;
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> or_to_all_lanes(Vector<Lane> v) {
    for (int bytes = 64; bytes >= static_cast<int>(sizeof(Lane)); bytes >>= 1) v = v | rotate_right_bytes(v, bytes);
    return v;
}

// sum_to_all_lanes of values[0 .. Count) in place, step by step over the vectors.
template <int Count, class Lane>
BENZENE_VECTOR_FUNCTION void sum_to_all_lanes_batch(Vector<Lane> * values) {
#pragma clang loop unroll(full)
    for (int bytes = 64; bytes >= static_cast<int>(sizeof(Lane)); bytes >>= 1) {
#pragma clang loop unroll(full)
        for (int k = 0; k < Count; k++) values[k] = values[k] + rotate_right_bytes(values[k], bytes);
    }
}

// 1 / sqrt(x) for normal x > 0 in every lane: bit-trick seed and 2 Newton steps (relative error < 5e-6).
BENZENE_VECTOR_FUNCTION Vector<float> reciprocal_square_root(Vector<float> x) {
    Vector<float> y = reinterpret<float>(0x5f375a86u - (reinterpret<uint32_t>(x) >> 1));
    const Vector<float> half_x = x * 0.5f;
    y = y * (1.5f - half_x * (y * y));
    return y * (1.5f - half_x * (y * y));
}

// 1 / sqrt(x) for normal x > 0 in every lane to about f32 precision: bit-trick seed 0x5f3759df and 3 Newton steps
// (python/benzene/models/aliked_lightglue/device_model.py inverse_norm follows it).
BENZENE_VECTOR_FUNCTION Vector<float> precise_reciprocal_square_root(Vector<float> x) {
    Vector<float> y = reinterpret<float>(Vector<uint32_t>::splat(0x5f3759dfu) - (reinterpret<uint32_t>(x) >> 1));
    const Vector<float> half_x = x * 0.5f;
    for (int step = 0; step < 3; step++) y = y * (1.5f - half_x * (y * y));
    return y;
}

// precise_reciprocal_square_root of values[0 .. Count) in place, the same operations step by step over the vectors.
template <int Count>
BENZENE_VECTOR_FUNCTION void precise_reciprocal_square_root_batch(Vector<float> * values) {
    Vector<float> half_x[Count];
#pragma clang loop unroll(full)
    for (int k = 0; k < Count; k++) {
        const Vector<float> x = values[k];
        values[k] = reinterpret<float>(Vector<uint32_t>::splat(0x5f3759dfu) - (reinterpret<uint32_t>(x) >> 1));
        half_x[k] = x * 0.5f;
    }

#pragma clang loop unroll(full)
    for (int step = 0; step < 3; step++) {
#pragma clang loop unroll(full)
        for (int k = 0; k < Count; k++) values[k] = values[k] * (1.5f - half_x[k] * (values[k] * values[k]));
    }
}

// The constants of a SELU requantization (python/benzene/models/aliked_lightglue/device_model.py selu_codes), per lane
// (output channel) of a layer with pre-activation v = a S + B and output codes x = s (q - z):
//   positive_scale   = S lambda / s
//   positive_offset  = B lambda / s + z + 0.5
//   exponent_scale   = S log2 e
//   exponent_offset  = B log2 e
//   negative_offset  = z + 0.5 - A
//   exponential[i]   = A c_i
// with A = lambda alpha / s and c_i the degree-4 fit of 2^f on [-0.5, 0.5] (max relative error 2.7e-6).
struct SeluConstants {
    Vector<float> positive_scale;
    Vector<float> positive_offset;
    Vector<float> exponent_scale;
    Vector<float> exponent_offset;
    Vector<float> negative_offset;
    Vector<float> exponential[5];
};

// SELU(v) / s + z + 0.5 of the f32 accumulators a, clamped to [0, 255.5] (truncated, it is the output code). With
// t = v log2 e clamped to [-30, 0], the negative branch is A 2^n p(f) + negative_offset, n = round(t), f = t - n.
BENZENE_VECTOR_FUNCTION Vector<float> selu_requantize(Vector<float> a, const SeluConstants & constants) {
    constexpr float kRounding = 12582912.0f;  // 1.5 * 2^23: adding it rounds |t| < 2^22 to an integer
    const Vector<float> rounding = Vector<float>::splat(kRounding);

    const Vector<float> positive_branch = a * constants.positive_scale + constants.positive_offset;

    const Vector<float> t = a * constants.exponent_scale + constants.exponent_offset;
    const Vector<float> clamped = min(max(t, -30.0f), Vector<float>::zero());
    const Vector<float> rounded = clamped + rounding;  // n in the low mantissa bits
    const Vector<float> fraction = clamped - (rounded - rounding);

    Vector<float> power = fraction * constants.exponential[4] + constants.exponential[3];
    power = power * fraction + constants.exponential[2];
    power = power * fraction + constants.exponential[1];
    power = power * fraction + constants.exponential[0];
    const Vector<float> negative_branch =
        add_to_exponent(power, reinterpret<int32_t>(rounded)) + constants.negative_offset;

    const Vector<float> y = select(positive(t), positive_branch, negative_branch);
    return min(max(y, Vector<float>::zero()), 255.5f);
}

}  // namespace benzene::vector_math
