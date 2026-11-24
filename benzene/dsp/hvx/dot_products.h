#pragma once

#include "dsp/hvx/vector.h"

namespace benzene::hvx {

// 32-bit lane i += the dot product of bytes 4i .. 4i + 3 of a and b (exact).
BENZENE_VECTOR_FUNCTION Vector<int32_t> dot_product_accumulate(Vector<int32_t> accumulator, Vector<uint8_t> a,
                                                               Vector<int8_t> b) {
    return {Q6_Vw_vrmpyacc_VwVubVb(accumulator.bits, a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<int32_t> dot_product_accumulate(Vector<int32_t> accumulator, Vector<int8_t> a,
                                                               Vector<int8_t> b) {
    return {Q6_Vw_vrmpyacc_VwVbVb(accumulator.bits, a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<uint32_t> dot_product_accumulate(Vector<uint32_t> accumulator, Vector<uint8_t> a,
                                                                Vector<uint8_t> b) {
    return {Q6_Vuw_vrmpyacc_VuwVubVub(accumulator.bits, a.bits, b.bits)};
}

// The same with one set of four weights for every lane: byte k of `weights` (a 32-bit word) weighs byte 4i + k.
BENZENE_VECTOR_FUNCTION Vector<int32_t> dot_product_accumulate_scalar(Vector<int32_t> accumulator, Vector<uint8_t> a,
                                                                      int32_t weights) {
    return {Q6_Vw_vrmpyacc_VwVubRb(accumulator.bits, a.bits, weights)};
}
BENZENE_VECTOR_FUNCTION Vector<uint32_t> dot_product_accumulate_scalar(Vector<uint32_t> accumulator, Vector<uint8_t> a,
                                                                       uint32_t weights) {
    return {Q6_Vuw_vrmpyacc_VuwVubRub(accumulator.bits, a.bits, static_cast<int32_t>(weights))};
}
BENZENE_VECTOR_FUNCTION Vector<uint32_t> dot_product_scalar(Vector<uint8_t> a, uint32_t weights) {
    return {Q6_Vuw_vrmpy_VubRub(a.bits, static_cast<int32_t>(weights))};
}

// f32 lane i += a[2i] b[2i] + a[2i + 1] b[2i + 1] (f16 products).
BENZENE_VECTOR_FUNCTION Vector<float> dot_product_accumulate(Vector<float> accumulator, Vector<_Float16> a,
                                                             Vector<_Float16> b) {
    return {Q6_Vsf_vdmpyacc_VsfVhfVhf(accumulator.bits, a.bits, b.bits)};
}

// Widening products accumulated into a pair: the low vector's lane i gets the product of lanes 2i, the high vector's
// lane i that of lanes 2i + 1. With a 32-bit `factors` word: even lanes times its low halfword, odd lanes times its
// high halfword.
BENZENE_VECTOR_FUNCTION VectorPair<int32_t> multiply_accumulate_widening(VectorPair<int32_t> accumulator,
                                                                         Vector<int16_t> a, Vector<int16_t> b) {
    return {Q6_Ww_vmpyacc_WwVhVh(accumulator.bits, a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION VectorPair<int32_t> multiply_accumulate_widening(VectorPair<int32_t> accumulator,
                                                                         Vector<int16_t> a, int32_t factors) {
    return {Q6_Ww_vmpyacc_WwVhRh(accumulator.bits, a.bits, factors)};
}

}  // namespace benzene::hvx
