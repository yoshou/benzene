#pragma once

#include "dsp/hvx/vector.h"

namespace benzene::hvx {

// Word lane i holds 1 << i: the lane masks of a 32-lane word vector.
struct WordLaneBits {
    uint32_t bits[32];
};
constexpr WordLaneBits make_word_lane_bits() {
    WordLaneBits lane_bits{};
    for (int lane = 0; lane < 32; lane++) lane_bits.bits[lane] = 1u << lane;
    return lane_bits;
}
alignas(128) static constexpr WordLaneBits kWordLaneBits = make_word_lane_bits();

// Byte i of the result is byte (i + bytes) % 128 of v.
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> rotate_right_bytes(Vector<Lane> v, int bytes) {
    return {Q6_V_vror_VR(v.bits, bytes)};
}

// Two vectors are given high first, as the hardware writes a register pair, and stand for the 256 bytes low | high.

// The 128 bytes at byte offset `bytes` (0 .. 127) of low | high.
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> slide_down(Vector<Lane> high, Vector<Lane> low, int bytes) {
    return {Q6_V_valign_VVR(high.bits, low.bits, bytes)};
}

// The 128 bytes at byte offset 128 - `bytes` of low | high.
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> slide_up(Vector<Lane> high, Vector<Lane> low, int bytes) {
    return {Q6_V_vlalign_VVR(high.bits, low.bits, bytes)};
}

// Chunks of ChunkBytes bytes of `low` and `high` alternating (low's chunk 0, high's chunk 0, low's chunk 1, ..): the
// first halves of both into the pair's low vector, the second halves into its high vector.
template <int ChunkBytes, class Lane>
BENZENE_VECTOR_FUNCTION VectorPair<Lane> interleave_chunks(Vector<Lane> high, Vector<Lane> low) {
    static_assert(ChunkBytes >= 1 && ChunkBytes <= 64 && (ChunkBytes & (ChunkBytes - 1)) == 0);
    return {Q6_W_vshuff_VVR(high.bits, low.bits, -ChunkBytes)};
}
// interleave_chunks of whole lanes: lane q of `low` to lane 2q, lane q of `high` to lane 2q + 1 of the pair.
template <class Lane>
BENZENE_VECTOR_FUNCTION VectorPair<Lane> interleave(Vector<Lane> high, Vector<Lane> low) {
    return interleave_chunks<static_cast<int>(sizeof(Lane))>(high, low);
}

// The inverse: of the 256 bytes low | high as chunks of ChunkBytes bytes, the even chunks into the pair's low vector,
// the odd chunks into its high vector (in order).
template <int ChunkBytes, class Lane>
BENZENE_VECTOR_FUNCTION VectorPair<Lane> deinterleave_chunks(Vector<Lane> high, Vector<Lane> low) {
    static_assert(ChunkBytes >= 1 && ChunkBytes <= 64 && (ChunkBytes & (ChunkBytes - 1)) == 0);
    return {Q6_W_vdeal_VVR(high.bits, low.bits, -ChunkBytes)};
}
template <class Lane>
BENZENE_VECTOR_FUNCTION VectorPair<Lane> deinterleave(Vector<Lane> high, Vector<Lane> low) {
    return deinterleave_chunks<static_cast<int>(sizeof(Lane))>(high, low);
}

// The even lanes of `low` and `high` alternating: low's lane 0, high's lane 0, low's lane 2, high's lane 2, ..;
// interleave_odd_lanes the same with the odd lanes.
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> interleave_even_lanes(Vector<Lane> high, Vector<Lane> low) {
    if constexpr (sizeof(Lane) == 2) {
        return {Q6_Vh_vshuffe_VhVh(high.bits, low.bits)};
    } else {
        static_assert(sizeof(Lane) == 1, "1- or 2-byte lanes");
        return {Q6_Vb_vshuffe_VbVb(high.bits, low.bits)};
    }
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> interleave_odd_lanes(Vector<Lane> high, Vector<Lane> low) {
    if constexpr (sizeof(Lane) == 2) {
        return {Q6_Vh_vshuffo_VhVh(high.bits, low.bits)};
    } else {
        static_assert(sizeof(Lane) == 1, "1- or 2-byte lanes");
        return {Q6_Vb_vshuffo_VbVb(high.bits, low.bits)};
    }
}

// The two halves of one vector alternating: lane 0, lane n/2, lane 1, lane n/2 + 1, .. (n lanes).
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> interleave_halves(Vector<Lane> v) {
    if constexpr (sizeof(Lane) == 2) {
        return {Q6_Vh_vshuff_Vh(v.bits)};
    } else {
        static_assert(sizeof(Lane) == 1, "1- or 2-byte lanes");
        return {Q6_Vb_vshuff_Vb(v.bits)};
    }
}

// Byte 4i + r of out[n] = byte 32n + i of v_r (byte lanes): the four vectors interleaved lane by lane.
template <class Lane>
inline void interleave_four(Vector<Lane> v0, Vector<Lane> v1, Vector<Lane> v2, Vector<Lane> v3, Vector<Lane> * out) {
    const VectorPair<Lane> p01 = interleave(v1, v0);
    const VectorPair<Lane> p23 = interleave(v3, v2);
    const VectorPair<Lane> low = interleave_chunks<2>(p23.low(), p01.low());
    const VectorPair<Lane> high = interleave_chunks<2>(p23.high(), p01.high());
    out[0] = low.low();
    out[1] = low.high();
    out[2] = high.low();
    out[3] = high.high();
}

// Packs into lanes of half the width: the lanes of `low`, then those of `high`.
BENZENE_VECTOR_FUNCTION Vector<int16_t> pack_to_int16_saturating(Vector<int32_t> high, Vector<int32_t> low) {
    return {Q6_Vh_vpack_VwVw_sat(high.bits, low.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<uint8_t> pack_to_uint8_saturating(Vector<int16_t> high, Vector<int16_t> low) {
    return {Q6_Vub_vpack_VhVh_sat(high.bits, low.bits)};
}
template <class Lane>
BENZENE_VECTOR_FUNCTION auto pack_low_halves(Vector<Lane> high, Vector<Lane> low) {
    if constexpr (sizeof(Lane) == 4) {
        return Vector<std::conditional_t<std::is_signed_v<Lane>, int16_t, uint16_t>>{
            Q6_Vh_vpacke_VwVw(high.bits, low.bits)};
    } else {
        static_assert(sizeof(Lane) == 2, "2- or 4-byte lanes");
        return Vector<std::conditional_t<std::is_signed_v<Lane>, int8_t, uint8_t>>{
            Q6_Vb_vpacke_VhVh(high.bits, low.bits)};
    }
}

// Byte 4i + k = lane i of vk, saturated to 0 .. 255.
BENZENE_VECTOR_FUNCTION Vector<uint8_t> pack_to_uint8_interleaved(Vector<int32_t> v0, Vector<int32_t> v1,
                                                                  Vector<int32_t> v2, Vector<int32_t> v3) {
    const Vector<int16_t> even = pack_to_int16_saturating(v2, v0);
    const Vector<int16_t> odd = pack_to_int16_saturating(v3, v1);
    return reinterpret<uint8_t>(
        interleave_halves(reinterpret<uint16_t>(interleave_halves(pack_to_uint8_saturating(odd, even)))));
}

// Zero-extended, in lane order: the pair's low vector gets the first half of the lanes, its high vector the second.
BENZENE_VECTOR_FUNCTION VectorPair<uint16_t> widen(Vector<uint8_t> v) {
    return {Q6_Wuh_vunpack_Vub(v.bits)};
}
BENZENE_VECTOR_FUNCTION VectorPair<uint32_t> widen(Vector<uint16_t> v) {
    return {Q6_Wuw_vunpack_Vuh(v.bits)};
}

}  // namespace benzene::hvx
