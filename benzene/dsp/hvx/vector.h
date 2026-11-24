#pragma once

#include <cstdint>
#include <type_traits>

#include <hexagon_protos.h>
#include <hexagon_types.h>

// Always inlined: a call that is not inlined passes a vector struct in memory (out-of-line functions take and return
// HVX_Vector).
#define BENZENE_VECTOR_FUNCTION [[gnu::always_inline]] inline

namespace benzene::hvx {

constexpr int kVectorBytes = 128;

// A 128-byte HVX register as lanes of one type. Expressions are evaluated as written: each IEEE f32 / f16 operation
// rounds on its own, and nothing is reassociated or fused.
template <class Lane>
struct Vector {
    static_assert(sizeof(Lane) == 1 || sizeof(Lane) == 2 || sizeof(Lane) == 4, "lanes of 1, 2 or 4 bytes");
    using Bits =
        std::conditional_t<sizeof(Lane) == 4, uint32_t, std::conditional_t<sizeof(Lane) == 2, uint16_t, uint8_t>>;

    HVX_Vector bits;

    // Through the register: the implicit assignment is a 128-byte memcpy. The copy constructor stays trivial (a
    // user-provided one changes how vectors are passed).
    [[gnu::always_inline]] Vector & operator=(Vector other) {
        bits = other.bits;
        return *this;
    }

    BENZENE_VECTOR_FUNCTION static Vector zero() { return {Q6_V_vzero()}; }

    // The lane value, or for 8- and 16-bit integer lanes an int whose low bits are used (no conversion instruction).
    using SplatValue = std::conditional_t<std::is_integral_v<Lane> && (sizeof(Lane) < 4), int32_t, Lane>;

    BENZENE_VECTOR_FUNCTION static Vector splat(SplatValue value) {
        if constexpr (std::is_same_v<Lane, float> || std::is_same_v<Lane, _Float16>) {
            return splat_bits(__builtin_bit_cast(Bits, value));
        } else if constexpr (sizeof(Lane) == 4) {
            return {Q6_V_vsplat_R(value)};
        } else if constexpr (sizeof(Lane) == 2) {
            return {Q6_Vh_vsplat_R(value)};
        } else {
            return {Q6_Vb_vsplat_R(value)};
        }
    }

    BENZENE_VECTOR_FUNCTION static Vector splat_bits(Bits bits) {
        if constexpr (sizeof(Lane) == 4) {
            return {Q6_V_vsplat_R(static_cast<int32_t>(bits))};
        } else if constexpr (sizeof(Lane) == 2) {
            return {Q6_Vh_vsplat_R(bits)};
        } else {
            return {Q6_Vb_vsplat_R(bits)};
        }
    }
};

template <class Lane>
struct VectorPair {
    HVX_VectorPair bits;

    [[gnu::always_inline]] VectorPair & operator=(VectorPair other) {
        bits = other.bits;
        return *this;
    }

    BENZENE_VECTOR_FUNCTION Vector<Lane> low() const { return {Q6_V_lo_W(bits)}; }
    BENZENE_VECTOR_FUNCTION Vector<Lane> high() const { return {Q6_V_hi_W(bits)}; }
};

// High first, as the hardware writes a pair.
template <class Lane>
BENZENE_VECTOR_FUNCTION VectorPair<Lane> make_pair(Vector<Lane> high, Vector<Lane> low) {
    return {Q6_W_vcombine_VV(high.bits, low.bits)};
}

// A predicate over the 128 bytes of a vector: comparisons set all bytes of the lanes where they hold.
struct Mask {
    HVX_VectorPred bits;

    [[gnu::always_inline]] Mask & operator=(Mask other) {
        bits = other.bits;
        return *this;
    }

    // Bytes 0 .. count - 1, count in 0 .. 127 (128 gives no bytes).
    BENZENE_VECTOR_FUNCTION static Mask first_bytes(int count) { return {Q6_Q_vsetq_R(count)}; }
    // Bytes 0 .. count - 1, count in 1 .. 128 (128 gives every byte).
    BENZENE_VECTOR_FUNCTION static Mask first_bytes_up_to_128(int count) { return {Q6_Q_vsetq2_R(count)}; }
    BENZENE_VECTOR_FUNCTION static Mask uniform(bool value) {
        return {Q6_Q_vand_VR(Q6_V_vsplat_R(value ? -1 : 0), -1)};
    }
};

BENZENE_VECTOR_FUNCTION Mask operator&(Mask a, Mask b) {
    return {Q6_Q_and_QQ(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Mask operator|(Mask a, Mask b) {
    return {Q6_Q_or_QQ(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Mask operator!(Mask a) {
    return {Q6_Q_not_Q(a.bits)};
}
BENZENE_VECTOR_FUNCTION Mask and_not(Mask a, Mask b) {
    return {Q6_Q_and_QQn(a.bits, b.bits)};
}

template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> select(Mask mask, Vector<Lane> if_true, Vector<Lane> if_false) {
    return {Q6_V_vmux_QVV(mask.bits, if_true.bits, if_false.bits)};
}

// A vector of the 32-bit pattern's bytes (byte i: pattern byte i % 4) where the mask is set, zero elsewhere.
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> pattern_where(Mask mask, uint32_t pattern) {
    return {Q6_V_vand_QR(mask.bits, static_cast<int32_t>(pattern))};
}

template <class To, class From>
BENZENE_VECTOR_FUNCTION Vector<To> reinterpret(Vector<From> v) {
    return {v.bits};
}
template <class To, class From>
BENZENE_VECTOR_FUNCTION VectorPair<To> reinterpret(VectorPair<From> pair) {
    return {pair.bits};
}

// load / store need a 128-byte aligned address, load_unaligned / store_unaligned take any address.
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> load(const Lane * address) {
    return {*reinterpret_cast<const HVX_Vector *>(address)};
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> load(const void * address) {
    return {*reinterpret_cast<const HVX_Vector *>(address)};
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> load_unaligned(const Lane * address) {
    return {*reinterpret_cast<const HVX_UVector *>(address)};
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> load_unaligned(const void * address) {
    return {*reinterpret_cast<const HVX_UVector *>(address)};
}
template <class Lane>
BENZENE_VECTOR_FUNCTION void store(Lane * address, Vector<Lane> v) {
    *reinterpret_cast<HVX_Vector *>(address) = v.bits;
}
template <class Lane>
BENZENE_VECTOR_FUNCTION void store(void * address, Vector<Lane> v) {
    *reinterpret_cast<HVX_Vector *>(address) = v.bits;
}
template <class Lane>
BENZENE_VECTOR_FUNCTION void store_unaligned(Lane * address, Vector<Lane> v) {
    *reinterpret_cast<HVX_UVector *>(address) = v.bits;
}
template <class Lane>
BENZENE_VECTOR_FUNCTION void store_unaligned(void * address, Vector<Lane> v) {
    *reinterpret_cast<HVX_UVector *>(address) = v.bits;
}
// A 128-byte aligned array as its vectors: vectors(p)[k] = elements [n k, n (k + 1)) for n = 128 / sizeof(Lane).
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> * vectors(Lane * elements) {
    return reinterpret_cast<Vector<Lane> *>(elements);
}
template <class Lane>
BENZENE_VECTOR_FUNCTION const Vector<Lane> * vectors(const Lane * elements) {
    return reinterpret_cast<const Vector<Lane> *>(elements);
}
// The vector at a 128-byte aligned address, as an lvalue.
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> & vector_at(void * address) {
    return *reinterpret_cast<Vector<Lane> *>(address);
}
template <class Lane>
BENZENE_VECTOR_FUNCTION const Vector<Lane> & vector_at(const void * address) {
    return *reinterpret_cast<const Vector<Lane> *>(address);
}
// The bytes of v where the mask is set (address 128-byte aligned).
template <class Lane>
BENZENE_VECTOR_FUNCTION void store_masked(void * address, Mask mask, Vector<Lane> v) {
    Q6_vmem_QRIV(mask.bits, reinterpret_cast<HVX_Vector *>(address), v.bits);
}

// vgather: the 64 halfwords at the byte offsets `offsets` (even, <= region_bytes_minus_one) of the VTCM region at
// address `region` into the VTCM vector `destination` (asynchronous: a later load of destination waits for it).
template <class Lane>
BENZENE_VECTOR_FUNCTION void gather_halfwords(Vector<Lane> * destination, uint32_t region,
                                              int32_t region_bytes_minus_one, Vector<int16_t> offsets) {
    static_assert(sizeof(Lane) == 2, "2-byte lanes");
    Q6_vgather_ARMVh(&destination->bits, static_cast<int32_t>(region), region_bytes_minus_one, offsets.bits);
}

template <class Lane>
BENZENE_VECTOR_FUNCTION int32_t extract_word(Vector<Lane> v, int word) {
    return Q6_R_vextract_VR(v.bits, 4 * word);
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Lane extract_lane(Vector<Lane> v, int lane) {
    static_assert(sizeof(Lane) == 4, "4-byte lanes");
    return __builtin_bit_cast(Lane, Q6_R_vextract_VR(v.bits, 4 * lane));
}

template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> operator&(Vector<Lane> a, Vector<Lane> b) {
    return {Q6_V_vand_VV(a.bits, b.bits)};
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> operator|(Vector<Lane> a, Vector<Lane> b) {
    return {Q6_V_vor_VV(a.bits, b.bits)};
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> operator^(Vector<Lane> a, Vector<Lane> b) {
    return {Q6_V_vxor_VV(a.bits, b.bits)};
}

BENZENE_VECTOR_FUNCTION Vector<float> operator+(Vector<float> a, Vector<float> b) {
    return {Q6_Vsf_vadd_VsfVsf(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<float> operator-(Vector<float> a, Vector<float> b) {
    return {Q6_Vsf_vsub_VsfVsf(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<float> operator*(Vector<float> a, Vector<float> b) {
    return {Q6_Vsf_vmpy_VsfVsf(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<float> min(Vector<float> a, Vector<float> b) {
    return {Q6_Vsf_vmin_VsfVsf(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<float> max(Vector<float> a, Vector<float> b) {
    return {Q6_Vsf_vmax_VsfVsf(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Mask operator>(Vector<float> a, Vector<float> b) {
    return {Q6_Q_vcmp_gt_VsfVsf(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<float> operator-(Vector<float> x) {
    return {Q6_V_vxor_VV(x.bits, Q6_V_vsplat_R(static_cast<int32_t>(0x80000000)))};
}
BENZENE_VECTOR_FUNCTION Vector<float> absolute(Vector<float> x) {
    return {Q6_V_vand_VV(x.bits, Q6_V_vsplat_R(0x7fffffff))};
}

BENZENE_VECTOR_FUNCTION Vector<_Float16> operator+(Vector<_Float16> a, Vector<_Float16> b) {
    return {Q6_Vhf_vadd_VhfVhf(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<_Float16> operator-(Vector<_Float16> a, Vector<_Float16> b) {
    return {Q6_Vhf_vsub_VhfVhf(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<_Float16> operator*(Vector<_Float16> a, Vector<_Float16> b) {
    return {Q6_Vhf_vmpy_VhfVhf(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<_Float16> max(Vector<_Float16> a, Vector<_Float16> b) {
    return {Q6_Vhf_vmax_VhfVhf(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<_Float16> min(Vector<_Float16> a, Vector<_Float16> b) {
    return {Q6_Vhf_vmin_VhfVhf(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Mask operator>(Vector<_Float16> a, Vector<_Float16> b) {
    return {Q6_Q_vcmp_gt_VhfVhf(a.bits, b.bits)};
}

BENZENE_VECTOR_FUNCTION Vector<int32_t> operator+(Vector<int32_t> a, Vector<int32_t> b) {
    return {Q6_Vw_vadd_VwVw(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<int32_t> operator-(Vector<int32_t> a, Vector<int32_t> b) {
    return {Q6_Vw_vsub_VwVw(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<int32_t> operator<<(Vector<int32_t> v, int bits) {
    return {Q6_Vw_vasl_VwR(v.bits, bits)};
}
BENZENE_VECTOR_FUNCTION Vector<int32_t> operator>>(Vector<int32_t> v, int bits) {
    return {Q6_Vw_vasr_VwR(v.bits, bits)};
}
BENZENE_VECTOR_FUNCTION Vector<int32_t> min(Vector<int32_t> a, Vector<int32_t> b) {
    return {Q6_Vw_vmin_VwVw(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<int32_t> max(Vector<int32_t> a, Vector<int32_t> b) {
    return {Q6_Vw_vmax_VwVw(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Mask operator>(Vector<int32_t> a, Vector<int32_t> b) {
    return {Q6_Q_vcmp_gt_VwVw(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Mask operator==(Vector<int32_t> a, Vector<int32_t> b) {
    return {Q6_Q_vcmp_eq_VwVw(a.bits, b.bits)};
}
// The low 32 bits of the products: even lanes times the low halfword of `factors`, odd lanes times its high halfword.
BENZENE_VECTOR_FUNCTION Vector<int32_t> multiply_by_halfwords(Vector<int32_t> v, int32_t factors) {
    return {Q6_Vw_vmpyi_VwRh(v.bits, factors)};
}

BENZENE_VECTOR_FUNCTION Vector<uint32_t> operator+(Vector<uint32_t> a, Vector<uint32_t> b) {
    return {Q6_Vw_vadd_VwVw(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<uint32_t> operator-(Vector<uint32_t> a, Vector<uint32_t> b) {
    return {Q6_Vw_vsub_VwVw(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<uint32_t> operator<<(Vector<uint32_t> v, int bits) {
    return {Q6_Vw_vasl_VwR(v.bits, bits)};
}
BENZENE_VECTOR_FUNCTION Vector<uint32_t> operator>>(Vector<uint32_t> v, int bits) {
    return {Q6_Vuw_vlsr_VuwR(v.bits, bits)};
}

BENZENE_VECTOR_FUNCTION Vector<int16_t> operator+(Vector<int16_t> a, Vector<int16_t> b) {
    return {Q6_Vh_vadd_VhVh(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<int16_t> operator-(Vector<int16_t> a, Vector<int16_t> b) {
    return {Q6_Vh_vsub_VhVh(a.bits, b.bits)};
}
// the low 16 bits of the products
BENZENE_VECTOR_FUNCTION Vector<int16_t> operator*(Vector<int16_t> a, Vector<int16_t> b) {
    return {Q6_Vh_vmpyi_VhVh(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<int16_t> operator<<(Vector<int16_t> v, int bits) {
    return {Q6_Vh_vasl_VhR(v.bits, bits)};
}
BENZENE_VECTOR_FUNCTION Vector<int16_t> max(Vector<int16_t> a, Vector<int16_t> b) {
    return {Q6_Vh_vmax_VhVh(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Mask operator>(Vector<int16_t> a, Vector<int16_t> b) {
    return {Q6_Q_vcmp_gt_VhVh(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<int16_t> add_saturate(Vector<int16_t> a, Vector<int16_t> b) {
    return {Q6_Vh_vadd_VhVh_sat(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<int16_t> subtract_saturate(Vector<int16_t> a, Vector<int16_t> b) {
    return {Q6_Vh_vsub_VhVh_sat(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<int16_t> add_shifted_left(Vector<int16_t> accumulator, Vector<int16_t> v, int bits) {
    return {Q6_Vh_vaslacc_VhVhR(accumulator.bits, v.bits, bits)};
}

BENZENE_VECTOR_FUNCTION Mask operator>(Vector<uint16_t> a, Vector<uint16_t> b) {
    return {Q6_Q_vcmp_gt_VuhVuh(a.bits, b.bits)};
}

BENZENE_VECTOR_FUNCTION Vector<uint8_t> max(Vector<uint8_t> a, Vector<uint8_t> b) {
    return {Q6_Vub_vmax_VubVub(a.bits, b.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<uint8_t> operator>>(Vector<uint8_t> v, int bits) {
    return {Q6_Vub_vlsr_VubR(v.bits, bits)};
}
// floor((a + b) / 2)
BENZENE_VECTOR_FUNCTION Vector<int8_t> average(Vector<int8_t> a, Vector<int8_t> b) {
    return {Q6_Vb_vavg_VbVb(a.bits, b.bits)};
}

template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> operator+(Vector<Lane> a, std::type_identity_t<Lane> b) {
    return a + Vector<Lane>::splat(b);
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> operator+(std::type_identity_t<Lane> a, Vector<Lane> b) {
    return Vector<Lane>::splat(a) + b;
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> operator-(Vector<Lane> a, std::type_identity_t<Lane> b) {
    return a - Vector<Lane>::splat(b);
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> operator-(std::type_identity_t<Lane> a, Vector<Lane> b) {
    return Vector<Lane>::splat(a) - b;
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> operator*(Vector<Lane> a, std::type_identity_t<Lane> b) {
    return a * Vector<Lane>::splat(b);
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> operator*(std::type_identity_t<Lane> a, Vector<Lane> b) {
    return Vector<Lane>::splat(a) * b;
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> operator&(Vector<Lane> a, std::type_identity_t<Lane> b) {
    return a & Vector<Lane>::splat(b);
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> operator|(Vector<Lane> a, std::type_identity_t<Lane> b) {
    return a | Vector<Lane>::splat(b);
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> operator^(Vector<Lane> a, std::type_identity_t<Lane> b) {
    return a ^ Vector<Lane>::splat(b);
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> min(Vector<Lane> a, std::type_identity_t<Lane> b) {
    return min(a, Vector<Lane>::splat(b));
}
template <class Lane>
BENZENE_VECTOR_FUNCTION Vector<Lane> max(Vector<Lane> a, std::type_identity_t<Lane> b) {
    return max(a, Vector<Lane>::splat(b));
}
template <class Lane>
BENZENE_VECTOR_FUNCTION auto operator>(Vector<Lane> a, std::type_identity_t<Lane> b) {
    return a > Vector<Lane>::splat(b);
}
template <class Lane>
BENZENE_VECTOR_FUNCTION auto operator>(std::type_identity_t<Lane> a, Vector<Lane> b) {
    return Vector<Lane>::splat(a) > b;
}

// f32 -> int32 rounded toward zero; int32 -> f32 (rounded to nearest even)
BENZENE_VECTOR_FUNCTION Vector<int32_t> truncate_to_int32(Vector<float> v) {
    return {Q6_Vw_equals_Vsf(v.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<float> to_float(Vector<int32_t> v) {
    return {Q6_Vsf_equals_Vw(v.bits)};
}
// f16 -> int16, saturating: rounded to nearest even / toward zero
BENZENE_VECTOR_FUNCTION Vector<int16_t> round_to_int16(Vector<_Float16> v) {
    return {Q6_Vh_vcvt_Vhf(v.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<int16_t> truncate_to_int16(Vector<_Float16> v) {
    return {Q6_Vh_equals_Vhf(v.bits)};
}
// The same values: to_float16 issues on a multiply unit, to_float16_on_shift_unit on the shift unit.
BENZENE_VECTOR_FUNCTION Vector<_Float16> to_float16(Vector<int16_t> v) {
    return {Q6_Vhf_vcvt_Vh(v.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<_Float16> to_float16_on_shift_unit(Vector<int16_t> v) {
    return {Q6_Vhf_equals_Vh(v.bits)};
}

// Lane q of even_lanes to lane 2q, lane q of odd_lanes to lane 2q + 1.
BENZENE_VECTOR_FUNCTION Vector<_Float16> to_float16_interleaved(Vector<float> even_lanes, Vector<float> odd_lanes) {
    return {Q6_Vhf_vcvt_VsfVsf(even_lanes.bits, odd_lanes.bits)};
}
BENZENE_VECTOR_FUNCTION Vector<_Float16> to_float16(Vector<float> first_half, Vector<float> second_half) {
    const HVX_VectorPair even_and_odd = Q6_W_vdeal_VVR(second_half.bits, first_half.bits, -4);
    return {Q6_Vhf_vcvt_VsfVsf(Q6_V_lo_W(even_and_odd), Q6_V_hi_W(even_and_odd))};
}
// Lane 2q to lane q of the low vector, lane 2q + 1 to lane q of the high one.
BENZENE_VECTOR_FUNCTION VectorPair<float> to_float32_deinterleaved(Vector<_Float16> v) {
    return {Q6_Wsf_vcvt_Vhf(v.bits)};
}
// In lane order: low = lanes 0 .. 31, high = lanes 32 .. 63.
BENZENE_VECTOR_FUNCTION VectorPair<float> to_float32(Vector<_Float16> v) {
    const HVX_VectorPair even_and_odd = Q6_Wsf_vcvt_Vhf(v.bits);
    return {Q6_W_vshuff_VVR(Q6_V_hi_W(even_and_odd), Q6_V_lo_W(even_and_odd), -4)};
}
// Two f16 vectors -> int8 (rounded to nearest even, saturating): halfword 2k of the result holds lanes 2k, 2k + 1 of
// `first`, halfword 2k + 1 those of `second`.
BENZENE_VECTOR_FUNCTION Vector<int8_t> round_to_int8_interleaved(Vector<_Float16> first, Vector<_Float16> second) {
    return {Q6_Vb_vcvt_VhfVhf(first.bits, second.bits)};
}

}  // namespace benzene::hvx
