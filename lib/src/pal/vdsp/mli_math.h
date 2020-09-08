/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _VDSP_MLI_MATH_H_
#define _VDSP_MLI_MATH_H_

#include <limits>
#include <arc_vector.h>
#include "arc_vector_ext.h"

//=========================================================================
//
// Definitions
//
//=========================================================================
typedef int32_t mli_acc32_t;
typedef int64_t mli_acc40_t;

template <typename T, typename shift_T>
MLI_FORCE_INLINE T mli_math_asl_fx(T x, shift_T nbits);

template <typename T>
MLI_FORCE_INLINE T mli_math_limit_fx(T sign) {
    return sign < (T)0 ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
}

template <typename T, typename shift_T>
MLI_FORCE_INLINE T mli_math_asr_fx(T x, shift_T nbits) {
    if (nbits > (sizeof(T) * 8 - 1))
        return x < (T)0 ? -1 : 0;
    if (nbits < 0)
        return mli_math_asl_fx<T, int>(x, (-nbits));
    return x >> nbits;
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_asl_fx(vNx4short_t x, int nbits);

template <>
MLI_FORCE_INLINE vNx2short_t mli_math_asr_fx(vNx2short_t x, int nbits) {
    return x >> nbits;
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_asr_fx(vNx4short_t x, int nbits) {
    vNx4short_t r;
    r.lo = mli_math_asr_fx(x.lo, nbits);
    r.hi = mli_math_asr_fx(x.hi, nbits);
    return r;
}

template <>
MLI_FORCE_INLINE vNx2short_t mli_math_asr_fx(vNx2short_t x, vNx2short_t nbits) {
    return x >> nbits;
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_asr_fx(vNx4short_t x, vNx4short_t nbits) {
    vNx4short_t r;
    r.lo = mli_math_asr_fx(x.lo, nbits.lo);
    r.hi = mli_math_asr_fx(x.hi, nbits.hi);
    return r;
}

MLI_FORCE_INLINE vNint_t mli_math_asr_fx(vNint_t x, int nbits) {
    return x >> nbits;
}

template <>
MLI_FORCE_INLINE vNx2int_t mli_math_asr_fx(vNx2int_t x, int nbits) {
    vNx2int_t r;
    r.lo = mli_math_asr_fx(x.lo, nbits);
    r.hi = mli_math_asr_fx(x.hi, nbits);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_asr_fx(vNx4int_t x, int nbits) {
    vNx4int_t r;
    r.lo = mli_math_asr_fx(x.lo, nbits);
    r.hi = mli_math_asr_fx(x.hi, nbits);
    return r;
}

MLI_FORCE_INLINE vNint_t mli_math_asr_fx(vNint_t x, vNint_t nbits) {
    return x >> nbits;
}

template <>
MLI_FORCE_INLINE vNx2int_t mli_math_asr_fx(vNx2int_t x, vNx2int_t nbits) {
    vNx2int_t r;
    r.lo = mli_math_asr_fx(x.lo, nbits.lo);
    r.hi = mli_math_asr_fx(x.hi, nbits.hi);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_asr_fx(vNx4int_t x, vNx4int_t nbits) {
    vNx4int_t r;
    r.lo = mli_math_asr_fx(x.lo, nbits.lo);
    r.hi = mli_math_asr_fx(x.hi, nbits.hi);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_asr_fx(vNx4accshort_t x, vNx4short_t nbits) {
    return __vacc_concat(vvcasrm(__vacc_lo(x), nbits.lo), vvcasrm(__vacc_hi(x), nbits.hi));
}

template <typename T, typename shift_T>
MLI_FORCE_INLINE T mli_math_asl_fx(T x, shift_T nbits) {
    shift_T inp_size = sizeof(T) * 8;
    T hi = 0;

    if (nbits < 0)
        return mli_math_asr_fx<T>(x, (-nbits));

    if (nbits > (inp_size - 1))
        return x != (T)0 ? mli_math_limit_fx<T>(x) : 0;

    hi = x >> ((inp_size - 1) - nbits);
    if (hi == (T)0 || hi == (T)-1)
        return x << nbits;

    return mli_math_limit_fx<T>(hi);
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_asl_fx(vNx4short_t x, int nbits) {
    if (nbits < 0)
        return mli_math_asr_fx<vNx4short_t>(x, (-nbits));

    return vvslm_sat(x, nbits);
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_asl_fx(vNx4accshort_t x, vNx4short_t nbits) {
    return __vacc_concat(vvcslm(__vacc_lo(x), nbits.lo), vvcslm(__vacc_hi(x), nbits.hi));
}

template <typename T, typename shift_T>
MLI_FORCE_INLINE T mli_math_asr_rnd_fx(T x, shift_T nbits) {
    T r = 0;
    T last_deleted_mask = (T)1 << (nbits-1);

    if (nbits < 0)
        return mli_math_asl_fx<T, shift_T>(x, (-nbits));
    if (nbits == 0)
        return x;

    if (nbits > (sizeof(T) * 8 - 1))
        return 0;

    r = mli_math_asr_fx<T>(x, nbits);
    // Rounding up:
    // if the most significant deleted bit is 1, add 1 to the remaining bits.
#ifdef ROUND_UP
        if ((last_deleted_mask & x) != (T)0)
            return mli_math_add_fx<T>(r, 1);
#endif


    // Convergent: rounding with half-way value rounded to even value.
    // If the most significant deleted bit is 1, and 
    // either the least significant of the remaining bits
    // or at least one other deleted bit is 1, add 1 to the remaining bits.
#ifdef ROUND_CONVERGENT
    if (((x & last_deleted_mask) != (T)0) && 
            (((r & (T)1) != (T)0) ||  ((x & (last_deleted_mask-(T)1))!= (T)0))) {
        return mli_math_add_fx<T>(r, 1);
    }
#endif

    return r;
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_mac_fx(vNx4accshort_t acc, vNx4char_t L, vNx4char_t R);

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_asr_rnd_fx(vNx4accshort_t x, vNx4short_t nbits) {
    vNx4accshort_t r;
#ifdef ROUND_UP
    // adding 1 << (nbits-1)
    // when nbits >= 8, 1 << nbits would result in overflow. that is why nbits is divided by 2 and multiplied
    r = mli_math_mac_fx(x, to_vNx4char_t(1 << ((nbits - 1)/2)), to_vNx4char_t(1 << (nbits/2)));
#endif
#ifdef ROUND_CONVERGENT
#error "Convergent rounding not supported"
#endif
    r = mli_math_asr_fx(r, nbits);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_asr_rnd_fx(vNx4short_t x, vNx4short_t nbits) {
    vNx4short_t r;
#ifdef ROUND_UP
    r = x + (1 << (nbits - 1));
#endif
#ifdef ROUND_CONVERGENT
#error "Convergent rounding not supported"
#endif
    r = mli_math_asr_fx(r, nbits);

    return r;
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_asr_rnd_fx(vNx4int_t x, vNx4int_t nbits) {
    vNx4int_t r;
#ifdef ROUND_UP
    r = x + (1 << (nbits - 1));
#endif
#ifdef ROUND_CONVERGENT
#error "Convergent rounding not supported"
#endif
    r = mli_math_asr_fx(r, nbits);

    return r;
}

template <typename T>
MLI_FORCE_INLINE T mli_math_asl_rnd_fx(T x, int nbits) {
    return mli_math_asr_rnd_fx<T, int>(x, -nbits);
}

template <typename T>
MLI_FORCE_INLINE T mli_math_neg_fx(T x) {
    return x == std::numeric_limits<T>::lowest() ? std::numeric_limits<T>::max() : -x;
}

// nbits is nr of bits that are clipped.
template <typename T>
MLI_FORCE_INLINE T mli_math_sat_fx(T x, unsigned nbits) {
    return mli_math_asr_fx<T>(mli_math_asl_fx<T, int>(x, nbits), nbits);
}

template <typename T>
MLI_FORCE_INLINE T mli_math_abs_fx(T x) {    
    return x >= (T)0 ? x : mli_math_neg_fx(x);
}

template <typename T, typename o_T>
MLI_FORCE_INLINE o_T mli_math_norm_fx(T x) {
    o_T inp_size = sizeof(T) * 8;
    T hi = x < (T)0 ? (T)-1 : (T)0;
    o_T r = 0;

    if (x == (T)0)
        return inp_size - 1;

    while ((x >> r) != hi)
        r++;
    return (inp_size - 1) - r;
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_norm_fx(vNx4int_t x) {
    vNx4int_t r;
    r.lo.lo = vvnorm(x.lo.lo);
    r.lo.hi = vvnorm(x.lo.hi);
    r.hi.lo = vvnorm(x.hi.lo);
    r.hi.hi = vvnorm(x.hi.hi);
    return r;
}

// Addition of two fx operands with saturation
//========================================================================
template <typename T>
MLI_FORCE_INLINE T mli_math_add_fx(T L, T R) {
    if ((R > 0) && (L > std::numeric_limits<T>::max() - R)) // `L + R` would overflow
        return std::numeric_limits<T>::max();
    if ((R < 0) && (L < std::numeric_limits<T>::lowest() - R)) // `L + R` would underflow
        return std::numeric_limits<T>::lowest();
    return L + R;
}

// Subtraction of two fx operands with saturation
//========================================================================
template <typename T>
MLI_FORCE_INLINE T mli_math_sub_fx(T L, T R) {
    if ((R < 0) && (L > std::numeric_limits<T>::max() + R)) // `L - R` would overflow
        return std::numeric_limits<T>::max();
    if ((R > 0) && (L < std::numeric_limits<T>::lowest() + R)) // `L - R` would underflow
        return std::numeric_limits<T>::lowest();
    return L - R;
}

template<typename io_T, typename lr_T>
MLI_FORCE_INLINE io_T mli_math_bound_range_fx(io_T in, lr_T L, lr_T R) {
    io_T out;
    out = mli_math_max_fx(in, L);
    out = mli_math_min_fx(out, R);
    return out;
}

// Cast scalar to/from void pointer
//========================================================================
template < typename out_T > 
MLI_FORCE_INLINE out_T mli_math_cast_ptr_to_scalar_fx(void *src) {
    // REMARK: Need to check from C/Cpp standard point of view
    return static_cast < out_T > ((intptr_t) (src));
}

template < typename in_T > 
MLI_FORCE_INLINE void *mli_math_cast_scalar_to_ptr_fx(in_T src) {
    // REMARK: Need to check from C/Cpp standard point of view
    intptr_t out_upcast = src;
    return static_cast < void *>((intptr_t *) out_upcast);
}

// Comparators
//========================================================================
template < typename io_T > 
MLI_FORCE_INLINE bool mli_prv_less_than_1(io_T value, uint8_t frac_bits) {
    if (frac_bits > sizeof(io_T) * 8 - 1)
        return true;

    io_T unit = (io_T) 1 << frac_bits;
    return (value < unit);
}

// Multipliers used to apply scale factors in asymetric data types
//========================================================================
template <typename acc_T, bool asym_data>
MLI_FORCE_INLINE mli_acc32_t mli_math_scale_mul(mli_acc32_t accu, int32_t mul) {
    if (asym_data) {
        return accu = mli_math_mul_fx_high<int32_t, int32_t>(accu, mul);
    }else{
        return accu;
    }
}

template <typename b_T, typename acc_T, bool asym_data>
MLI_FORCE_INLINE acc_T mli_math_init_accu(b_T bias, int32_t bias_mul, int bias_shift) {
    acc_T accu = mli_math_mul_fx<b_T, acc_T>(bias, asym_data ? 2 : 1); // extra factor of 2 needed because scale mul cannot multiply by 1.
    accu = mli_math_acc_ashift_fx(accu, -bias_shift);
    accu = mli_math_scale_mul<acc_T, asym_data>(accu, bias_mul);
    return accu;
}

//TODO: REWRITE ACC INIT ACCORDING TO VDSP
// template <typename b_T, typename acc_T, bool asym_data>
// static inline int32_t mli_math_init_accu(int32_t bias, int32_t bias_mul, int bias_shift) {
//     int32_t accu = bias;
//     _setacc(accu, 1);
//     // for an int32_t bias type and int32_t accumulator, there is no requantization of the bias.
//     // in the mli_check function it has been checked that the bias scale and shift match the accumulator scale and shift.
//     //accu = mli_math_acc_ashift_fx(accu, bias_shift);
//     //accu = mli_math_scale_mul<acc_T, asym_data>(accu, bias_mul);
//     return accu;
// }

// Cast value to output type (including accumulator type)
//========================================================================
template <>
MLI_FORCE_INLINE int8_t mli_math_cast_fx(int16_t in_val, int shift_right) {
    return (int8_t)mli_math_sat_fx<int16_t>(mli_math_asr_rnd_fx<int16_t, int>(in_val, shift_right), 8);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(int8_t in_val, int shift_right) {
    return (int16_t)mli_math_asr_rnd_fx<int16_t, int>((int16_t)in_val, shift_right);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(int16_t in_val, int shift_right) {
    return (int16_t)mli_math_asr_rnd_fx<int16_t, int>(in_val, shift_right);
}

template <>
MLI_FORCE_INLINE int8_t mli_math_cast_fx(mli_acc32_t in_val, int shift_right) {
    int32_t temp = (int32_t)mli_math_asr_rnd_fx<mli_acc32_t, int>((int32_t)in_val, shift_right);
    return (int8_t)mli_math_sat_fx<int32_t>(temp, 24);
}


template <>
MLI_FORCE_INLINE int8_t mli_math_cast_fx(mli_acc32_t in_val) {
    int32_t temp = (int32_t)mli_math_asr_rnd_fx<mli_acc32_t, int>((int32_t)in_val, 24);
    return (int8_t)mli_math_sat_fx<int32_t>(temp, 24);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(mli_acc32_t in_val, int shift_right) {
    int32_t temp = (int32_t)mli_math_asr_rnd_fx<mli_acc32_t, int>(in_val, shift_right);
    return (int16_t)mli_math_sat_fx<int32_t>(temp, 16);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(mli_acc32_t in_val) {
    int32_t temp = (int32_t)mli_math_asr_rnd_fx<mli_acc32_t, int>(in_val, 16);
    return (int16_t)mli_math_sat_fx<int32_t>(temp, 16);
}

template <>
MLI_FORCE_INLINE int32_t mli_math_cast_fx(mli_acc32_t in_val, int shift_right) {
    return (int32_t)mli_math_asr_rnd_fx<mli_acc32_t, int>(in_val, shift_right);
}

template <>
MLI_FORCE_INLINE mli_acc40_t mli_math_cast_fx(int16_t in_val, int shift_right) {
    return (int32_t)mli_math_asr_rnd_fx<mli_acc40_t, int>((mli_acc40_t)in_val, shift_right);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(int64_t in_val, int shift_right) {
    int64_t temp = mli_math_asr_rnd_fx<int64_t, int>(in_val, shift_right);
    return (int16_t)mli_math_sat_fx<int64_t>(temp, 48);
}

template <>
MLI_FORCE_INLINE int32_t mli_math_cast_fx(int64_t in_val, int shift_right) {
    in_val = mli_math_asr_rnd_fx<int64_t, int>(in_val, shift_right);
    return (int32_t)mli_math_sat_fx<int64_t>(in_val, 32);
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_cast_fx(vNx4char_t in_val) {
    return to_vNx4short_t(vvcmpy(in_val, 1));
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_cast_fx(vNx4short_t in_val) {
    return to_vNx4int_t(in_val);
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_cast_fx(vNx4int_t in_val) {
    return to_vNx4short_t(in_val);
}

template<>
MLI_FORCE_INLINE vNx4short_t mli_math_cast_fx(vNx4short_t in_val, int shift_right) {
    /* Shift, round and Sat */
    vNx4int_t acc;
    acc.lo = to_vNx2int_t(in_val.lo);
    acc.hi = to_vNx2int_t(in_val.hi);
    if (shift_right > 0) {
        int round = 0;
#ifdef ROUND_UP
        // Rounding up:
        round = 1 << (shift_right - 1);
#else
        #error Rounding mode not supported
#endif
        acc = (acc + round) >> shift_right;
    } else {
        acc = (acc << (-shift_right));
    }
    acc = mli_math_bound_range_fx(acc, INT16_MIN, INT16_MAX);
    return to_vNx4short_t(acc);
}

template<>
MLI_FORCE_INLINE vNx4char_t mli_math_cast_fx(vNx4short_t in_val, int shift_right) {
    /* Shift, round and Sat */
    vNx4int_t acc;
    acc.lo = to_vNx2int_t(in_val.lo);
    acc.hi = to_vNx2int_t(in_val.hi);
    if (shift_right > 0) {
        int round = 0;
#ifdef ROUND_UP
        // Rounding up:
        round = 1 << (shift_right - 1);
#else
        #error Rounding mode not supported
#endif
        acc = (acc + round) >> shift_right;
    } else {
        acc = (acc << (-shift_right));
    }
    acc = mli_math_bound_range_fx(acc, INT8_MIN, INT8_MAX);
    return to_vNx4char_t(acc);
}
// Cast accum to output type
//========================================================================
template <>
MLI_FORCE_INLINE int8_t mli_math_acc_cast_fx(mli_acc32_t acc, int shift_right) {
    int32_t temp = (int32_t) mli_math_asr_rnd_fx<mli_acc32_t, int>(acc, shift_right);
    temp = mli_math_asl_fx<int32_t, int>(temp, 24);
    return (int8_t) (temp >> 24);
}

template <> 
MLI_FORCE_INLINE int16_t mli_math_acc_cast_fx(mli_acc32_t acc, int shift_right) {
    int32_t temp = (int32_t) mli_math_asr_rnd_fx<mli_acc32_t, int>(acc, shift_right);
    temp = mli_math_asl_fx<mli_acc32_t, int>(temp, 16);
    return (int16_t) mli_math_sat_fx<mli_acc32_t>(mli_math_asr_fx<mli_acc32_t>(temp, 16), 16);
}

template <>
MLI_FORCE_INLINE vNx2short_t mli_math_acc_cast_fx(vNx2accshort_t acc) {
    return to_vNx2short_t(acc);
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_acc_cast_fx(vNx4accshort_t acc) {
    return to_vNx4short_t(acc);
}

template <>
MLI_FORCE_INLINE vNint_t mli_math_acc_cast_fx(vNaccint_t acc) {
    return to_vNint_t(acc);
}

template <>
MLI_FORCE_INLINE vNx2int_t mli_math_acc_cast_fx(vNx2accint_t acc) {
    return to_vNx2int_t(acc);
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_acc_cast_fx(vNx4accint_t acc) {
    return to_vNx4int_t(acc);
}

template<>
MLI_FORCE_INLINE vNx4short_t mli_math_acc_cast_fx(vNx4accint_t acc, int shift_right) {
    vNx4int_t acc_int = mli_math_acc_cast_fx<vNx4int_t>(acc);
    /* Shift, round and Sat */
    if (shift_right > 0) {
        int round = 0;
#ifdef ROUND_UP
        // Rounding up:
        round = 1 << (shift_right - 1);
#else
        #error Rounding mode not supported
#endif
        acc_int = (acc_int + round) >> shift_right;
    } else {
        acc_int = (acc_int << (-shift_right));
    }
    acc_int = mli_math_bound_range_fx(acc_int, INT16_MIN, INT16_MAX);
    return mli_math_cast_fx<vNx4int_t, vNx4short_t>(acc_int);
}

template<>
MLI_FORCE_INLINE vNx4short_t mli_math_acc_cast_fx(vNx4accshort_t acc, int shift_right) {
    int ctrlword = SAT|SIGNED|TARGET_SZ_16|SHIFT(shift_right);
    vNx4short_t accu_result;
    accu_result.lo = to_vNx2short_t(vvconvert(__vacc_lo(acc), ctrlword));
    accu_result.hi = to_vNx2short_t(vvconvert(__vacc_hi(acc), ctrlword));

    return accu_result;
}

// Addition/subtraction of two operands
template <typename l_T, typename r_T>
MLI_FORCE_INLINE l_T mli_math_add(l_T L, r_T R) {
    return L + R;
}

template <typename l_T, typename r_T>
MLI_FORCE_INLINE l_T mli_math_sub(l_T L, r_T R) {
    return L - R;
}

template <> 
MLI_FORCE_INLINE vNx4short_t mli_math_add_fx(vNx4short_t L, vNx4short_t R) {
    return vvadd_sat(L, R);
}
template <> 
MLI_FORCE_INLINE vNx4short_t mli_math_sub_fx(vNx4short_t L, vNx4short_t R) {
    return vvsub_sat(L, R);
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_add(vNx4accshort_t L, vNx4short_t R) {
    return __vacc_concat(vvcadd(__vacc_lo(L), R.lo,(int16_t)0), vvcadd(__vacc_hi(L), R.hi,(int16_t)0));
}

// Maximum of two fx operands
//========================================================================
template <typename io_T>
MLI_FORCE_INLINE io_T mli_math_max_fx(io_T L, io_T R) {
    return (L > R) ? L : R;
}

template <typename l_T, typename r_T>
MLI_FORCE_INLINE l_T mli_math_max_fx(l_T L, r_T R) {
    return (L > R) ? L : R;
}

// Minimum of two fx operands
//========================================================================
template <typename io_T>
MLI_FORCE_INLINE io_T mli_math_min_fx(io_T L, io_T R) {
     return (L < R) ? L : R;
}

template <typename l_T, typename r_T>
MLI_FORCE_INLINE l_T mli_math_min_fx(l_T L, r_T R) {
    return (L < R) ? L : R;
}

// Multiply two operands
//========================================================================
template <typename in_T, typename acc_T>
MLI_FORCE_INLINE acc_T mli_math_mul_fx(in_T L, in_T R) {
    return L * R;
}

template <>
MLI_FORCE_INLINE mli_acc32_t mli_math_mul_fx_high(int32_t L, int32_t R) {
    // this function takes the MSB part of the result. (L * R) >> 31
    // in optimized code check if mpyfr instruction is used here.
    return (mli_acc32_t)(((int64_t)L * (int64_t)R + (int64_t)(1<<30)) >> 31);
}
//--------------
// mul hi short
//--------------
MLI_FORCE_INLINE vNx2short_t mli_math_mul_fx_high(vNx2short_t L, int16_t R) {
    vNx2short_t r;
    r = vvmpy_hi(L, R);
    return r;
}

MLI_FORCE_INLINE vNx4short_t mli_math_mul_fx_high(vNx4short_t L, int16_t R) {
    vNx4short_t r;
    r.lo = mli_math_mul_fx_high(L.lo, R);
    r.hi = mli_math_mul_fx_high(L.hi, R);
    return r;
}

MLI_FORCE_INLINE vNx2short_t mli_math_mul_fx_high(vNx2short_t L, vNx2short_t R) {
    vNx2short_t r;
    r = vvmpy_hi(L, R);
    return r;
}

MLI_FORCE_INLINE vNx4short_t mli_math_mul_fx_high(vNx4short_t L, vNx4short_t R) {
    vNx4short_t r;
    r.lo = mli_math_mul_fx_high(L.lo, R.lo);
    r.hi = mli_math_mul_fx_high(L.hi, R.hi);
    return r;
}
//--------------
// mul hi int
//--------------
MLI_FORCE_INLINE vNint_t mli_math_mul_fx_high(vNint_t L, int32_t R) {
    vNint_t r;
    r = vvmpy_hi(L, R);
    return r;
}

MLI_FORCE_INLINE vNx2int_t mli_math_mul_fx_high(vNx2int_t L, int32_t R) {
    vNx2int_t r;
    r.lo = mli_math_mul_fx_high(L.lo, R);
    r.hi = mli_math_mul_fx_high(L.hi, R);
    return r;
}

MLI_FORCE_INLINE vNx4int_t mli_math_mul_fx_high(vNx4int_t L, int32_t R) {
    vNx4int_t r;
    r.lo = mli_math_mul_fx_high(L.lo, R);
    r.hi = mli_math_mul_fx_high(L.hi, R);
    return r;
}

MLI_FORCE_INLINE vNint_t mli_math_mul_fx_high(vNint_t L, vNint_t R) {
    vNint_t r;
    r = vvmpy_hi(L, R);
    return r;
}

MLI_FORCE_INLINE vNx2int_t mli_math_mul_fx_high(vNx2int_t L, vNx2int_t R) {
    vNx2int_t r;
    r.lo = mli_math_mul_fx_high(L.lo, R.lo);
    r.hi = mli_math_mul_fx_high(L.hi, R.hi);
    return r;
}

MLI_FORCE_INLINE vNx4int_t mli_math_mul_fx_high(vNx4int_t L, vNx4int_t R) {
    vNx4int_t r;
    r.lo = mli_math_mul_fx_high(L.lo, R.lo);
    r.hi = mli_math_mul_fx_high(L.hi, R.hi);
    return r;
}

template <>
MLI_FORCE_INLINE int64_t mli_math_mul_fx(int32_t L, int32_t R) {
    // Result of multiplication is fractional number (shifted left by 1)
    // To return correct result we shift it right afterward
    int64_t acc = (int64_t)L * (int64_t)R;
    return (int64_t) acc;
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_mul_fx(vNx4short_t L, vNx4short_t R) {
    return vvcmpy(L, R);
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_mul_fx(int8_t L, int8_t R) {
    return vvcmpy((vNx4char_t)L, R);
}

// Multiply-and-accumulate operands
//========================================================================
template <>
MLI_FORCE_INLINE mli_acc32_t mli_math_mac_fx(mli_acc32_t acc, int8_t L, int8_t R) {
    return acc + (mli_acc32_t) (L * R);
}

template <>
MLI_FORCE_INLINE mli_acc32_t mli_math_mac_fx(mli_acc32_t acc, int16_t L, int16_t R) {
    return acc + (mli_acc32_t) (L * R);
}

template <>
MLI_FORCE_INLINE mli_acc32_t mli_math_mac_fx(mli_acc32_t acc, int16_t L, int8_t R) {
    return acc + (mli_acc32_t)(L * (int16_t)R);
}

template <>
MLI_FORCE_INLINE mli_acc40_t mli_math_mac_fx(mli_acc40_t acc, int16_t L, int16_t R) {
    return acc + (mli_acc40_t) ((int32_t)L * (int32_t)R);
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_mac_fx(vNx4accshort_t acc, vNx4char_t L, int8_t R) {
    return vvcmac(acc, L, R);
}
template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_mac_fx(vNx4accshort_t acc, vNx4char_t L, vNx4char_t R) {
    return vvcmac(acc, L, R);
}

// Accumulator shift
//========================================================================

template <>
MLI_FORCE_INLINE mli_acc32_t mli_math_acc_ashift_fx(mli_acc32_t acc, int shift_right) {
    return mli_math_asr_rnd_fx<mli_acc32_t, int>(acc, shift_right);
}

#endif // _VDSP_MLI_MATH_H_
