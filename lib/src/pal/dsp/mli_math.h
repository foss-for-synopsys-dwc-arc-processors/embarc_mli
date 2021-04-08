/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _DSP_MLI_MATH_H_
#define _DSP_MLI_MATH_H_

#if defined(__FXAPI__)
#include "fxarc.h"
#else
#error "ARC FX Library (FXAPI) is required dependency"
#endif

#include <limits>
#include <type_traits>
#include "mli_debug.h"
#include "mli_math_macros.h"
#include "mli_private_types.h"

#ifdef _ARC
#include <arc/arc_intrinsics.h>
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")

typedef accum40_t mli_acc40_t;
typedef int32_t   mli_acc32_t;
//typedef signed char v2i8_t __attribute__((__vector_size__(2)));

//=========================================================================
//
// Definitions
//
//=========================================================================

template <typename T>
MLI_FORCE_INLINE T mli_math_asl_fx(T x, int nbits);
template <typename T>
MLI_FORCE_INLINE T mli_math_asr_fx(T x, int nbits);

template <>
MLI_FORCE_INLINE int32_t mli_math_asl_fx(int32_t x, int nbits) {
    return fx_asl_q31(x, nbits);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_asl_fx(int16_t x, int nbits) {
    return fx_asl_q15(x, nbits);
}

template <>
MLI_FORCE_INLINE int32_t mli_math_asr_fx(int32_t acc, int shift_right) {
    return fx_asr_q31(acc, shift_right);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_asr_fx(int16_t acc, int shift_right) {
    return fx_asr_q15(acc, shift_right);
}

template <typename T>
MLI_FORCE_INLINE T mli_math_limit_fx(T sign) {
    return sign < (T)0 ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
}

template <typename T>
MLI_FORCE_INLINE T mli_math_sat_fx(T x, unsigned nbits);

template <>
MLI_FORCE_INLINE int32_t mli_math_sat_fx(int32_t x, unsigned nbits) {
    return fx_sat_q31(x, nbits);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_sat_fx(int16_t x, unsigned nbits) {
    return fx_sat_q15(x, nbits);
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
MLI_FORCE_INLINE int32_t mli_math_norm_fx(int64_t x) {
    if ((x <= std::numeric_limits<int32_t>::max()) &&
        (x >= std::numeric_limits<int32_t>::min())) {
        return (32 + _norm((int32_t) x ));
    } else {
        return _norm((int32_t) (x >> 32));
    }
}

template <>
MLI_FORCE_INLINE int32_t mli_math_norm_fx(int32_t x) {
    return _norm(x);
}

template <>
MLI_FORCE_INLINE int32_t mli_math_norm_fx(int16_t x) {
    return _normh(x);
}

template <>
MLI_FORCE_INLINE int mli_math_norm_fx(mli_acc40_t acc) {
    return fx_norm_a40(acc) + 1;
}

template<typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_math_norm_cast_fx(in_T val , int *norm_shift) {
    int cast_shift = (sizeof(in_T) - sizeof(out_T)) * 8;
    int norm = mli_math_norm_fx<in_T, in_T>(val);
    *norm_shift = cast_shift - norm;
    return mli_math_cast_fx<in_T, out_T>(val, *norm_shift);
}

// Addition of two fx operands with saturation
//========================================================================
template <> MLI_FORCE_INLINE int8_t mli_math_add_fx(int8_t L, int8_t R) {
    return (int8_t)fx_sat_q15(L + R, 8);
}

template <> MLI_FORCE_INLINE int16_t mli_math_add_fx(int16_t L, int16_t R) {
    return fx_add_q15(L, R);   // not cast operands intentionally (rise compile warnings)
}

template <>
MLI_FORCE_INLINE int32_t mli_math_add_fx(int32_t L, int32_t R) {
    return (int32_t)fx_add_q31(L, R);
}
template <>
MLI_FORCE_INLINE mli_acc40_t mli_math_add_fx(mli_acc40_t L, mli_acc40_t R) {
    return fx_add_a40(L, R);
}

template <>
MLI_FORCE_INLINE v2q15_t mli_math_add_fx(v2q15_t L, v2q15_t R) {
    return fx_add_v2q15(L, R);
}

// Subtraction of two fx operands with saturation
//========================================================================
template <> MLI_FORCE_INLINE int8_t mli_math_sub_fx(int8_t L, int8_t R) {
    return (int8_t)fx_sat_q15(L - R, 8);
}

template <> MLI_FORCE_INLINE int16_t mli_math_sub_fx(int16_t L, int16_t R) {
    return fx_sub_q15(L, R);   // not cast operands intentionally (rise compile warnings)
}

template <> MLI_FORCE_INLINE int32_t mli_math_sub_fx(int32_t L, int32_t R) {
    return (int32_t)fx_sub_q31(L, R);
}

template <> MLI_FORCE_INLINE v2q15_t mli_math_sub_fx(v2q15_t L, v2q15_t R) {
    return fx_sub_v2q15(L, R);
}

// Maximum of two fx operands
//========================================================================
template < typename io_T > 
MLI_FORCE_INLINE io_T mli_math_max_fx(io_T L, io_T R) {
    return MAX(L, R);
}

template <typename l_T, typename r_T>
MLI_FORCE_INLINE l_T mli_math_max_fx(l_T L, r_T R) {
    return MAX(L, R);
}

template <> 
MLI_FORCE_INLINE v2q15_t mli_math_max_fx(v2q15_t L, v2q15_t R) {
    return fx_max_v2q15(L, R);
}

template <typename l_T, typename r_T> 
MLI_FORCE_INLINE v2q15_t mli_math_max_fx(v2q15_t L, r_T R) {
    return fx_max_v2q15(L, fx_replic_v2q15(R));
}

// Minimum of two fx operands
//========================================================================
template < typename io_T > 
MLI_FORCE_INLINE io_T mli_math_min_fx(io_T L, io_T R) {
    return MIN(L, R);
}

template <typename l_T, typename r_T>
MLI_FORCE_INLINE l_T mli_math_min_fx(l_T L, r_T R) {
    return (L < R) ? L : R;
}

template <> 
MLI_FORCE_INLINE v2q15_t mli_math_min_fx(v2q15_t L, v2q15_t R) {
    return fx_min_v2q15(L, R);
}

template <typename l_T, typename r_T> 
MLI_FORCE_INLINE v2q15_t mli_math_min_fx(v2q15_t L, r_T R) {
    return fx_min_v2q15(L, fx_replic_v2q15(R));
}

// Multiply two operands
//========================================================================
template <> MLI_FORCE_INLINE mli_acc32_t mli_math_mul_fx(int8_t L, int8_t R) {
    return (mli_acc32_t) (L * R);
}

template <> MLI_FORCE_INLINE mli_acc32_t mli_math_mul_fx(int16_t L, int16_t R) {
    return (mli_acc32_t) (L * R);
}

template <> MLI_FORCE_INLINE mli_acc40_t mli_math_mul_fx(int16_t L, int16_t R) {
    return fx_a40_mpy_nf_q15(L, R);
}

template <> MLI_FORCE_INLINE int64_t mli_math_mul_fx(int16_t L, int16_t R) {
    return (int64_t) (L * R);
}

template <> MLI_FORCE_INLINE mli_acc32_t mli_math_mul_fx_high(int32_t L, int32_t R) {
    // this function takes the MSB part of the result. (L * R) >> 31
    // in optimized code check if mpyfr instruction is used here.
    return (mli_acc32_t)fx_q31_cast_rnd_a72(fx_a72_mpy_q31(L, R));
}

template <> MLI_FORCE_INLINE int16_t mli_math_mul_fx_high(int16_t L, int16_t R) {
    return (int16_t)fx_q15_cast_rnd_a40(fx_a40_mpy_q15(L, R));
}

template <>
MLI_FORCE_INLINE int64_t mli_math_mul_fx(int32_t L, int32_t R) {
    // Result of multiplication is fractional number (shifted left by 1)
    // To return correct result we shift it right afterward
    return (int64_t )(fx_q63_cast_a72(fx_a72_mpy_q31(L, R)) >> 1);
}

template <>
MLI_FORCE_INLINE accum72_t mli_math_mul_fx(int32_t L, int32_t R) {
    return fx_a72_mpy_q31(L, R);
}

template <>
MLI_FORCE_INLINE v2accum40_t mli_math_mul_fx(v2q15_t L, v2q15_t R) {
    return fx_v2a40_mpy_v2q15(L, R);
}

// Multiply-and-accumulate operands
//========================================================================
template <> MLI_FORCE_INLINE mli_acc32_t mli_math_mac_fx(mli_acc32_t acc, int8_t L, int8_t R) {
    return acc + (mli_acc32_t) (L * R);
}

template <> MLI_FORCE_INLINE mli_acc32_t mli_math_mac_fx(mli_acc32_t acc, int16_t L, int16_t R) {
    return acc + (mli_acc32_t) (L * R);
}

template <> MLI_FORCE_INLINE mli_acc32_t mli_math_mac_fx(mli_acc32_t acc, int16_t L, int8_t R) {
    return acc + (mli_acc32_t)(L * (int16_t)R);
}

template <> MLI_FORCE_INLINE mli_acc32_t mli_math_mac_fx(mli_acc32_t acc, int8_t L, int16_t R) {
    return acc + (mli_acc32_t)((int16_t)L * R);
}

template <> MLI_FORCE_INLINE mli_acc40_t mli_math_mac_fx(mli_acc40_t acc, int16_t L, int16_t R) {
    return fx_a40_mac_nf_q15(acc, L, R);
}

template <> MLI_FORCE_INLINE mli_acc40_t mli_math_mac_fx(mli_acc40_t acc, int16_t L, int8_t R) {
    return fx_a40_mac_nf_q15(acc, L, (int16_t) R);
}

template <> MLI_FORCE_INLINE mli_acc40_t mli_math_mac_fx(mli_acc40_t acc, int8_t L, int8_t R) {
    return fx_a40_mac_nf_q15(acc, (int16_t)L, (int16_t)R);
}

template <> MLI_FORCE_INLINE int64_t mli_math_mac_fx(int64_t acc, int16_t L, int16_t R) {
    return acc + (int64_t) (L * R);
}

static MLI_FORCE_INLINE mli_acc40_t mli_math_mac_fx(mli_acc40_t acc, v2q15_t in, v2q15_t k) {
    return fx_a40_dmac_v2q15(acc, in, k);
}

static MLI_FORCE_INLINE void mli_math_mac_fx_vec2(__v2i32_t * accu, v2q15_t in, v2q15_t k) { //mli_math_mac_fx_vec2 , acc by value
    *accu += __builtin_convertvector(in, __v2i32_t) * __builtin_convertvector(k, __v2i32_t);
}

static MLI_FORCE_INLINE void mli_math_mac_fx_vec2(v2accum40_t * accu, v2q15_t in, v2q15_t k) {//mli_math_mac_fx_vec2
    *accu = fx_v2a40_mac_nf_v2q15(*accu, in, k);
}

// Accumulator shift
//========================================================================

template <> MLI_FORCE_INLINE mli_acc32_t mli_math_acc_ashift_fx(mli_acc32_t acc, int shift_right) {
    return fx_asr_rnd_q31(acc, shift_right);
}

template <> MLI_FORCE_INLINE mli_acc40_t mli_math_acc_ashift_fx(mli_acc40_t acc, int shift_right) {
    return fx_asr_a40(acc, shift_right);
}

template <> MLI_FORCE_INLINE v2q15_t mli_math_acc_ashift_fx(v2q15_t acc, int shift_right) {
    return fx_asr_v2q15_n(acc, shift_right);
}

// Cast accum to output type
//========================================================================
template <> MLI_FORCE_INLINE int8_t mli_math_acc_cast_fx(mli_acc32_t acc, int shift_right) {
    int32_t temp = (int32_t) fx_asr_rnd_q31(acc, shift_right);
    temp = fx_asl_q31(temp, 24);
    return (int8_t) fx_q7_cast_q31(temp);
}

template <> MLI_FORCE_INLINE int8_t mli_math_acc_cast_fx(mli_acc40_t acc, int shift_right) {
    return fx_q7_cast_nf_asl_rnd_a40(acc, 24 - shift_right);
}

template <> MLI_FORCE_INLINE int16_t mli_math_acc_cast_fx(mli_acc40_t acc, int shift_right) {
    return fx_q15_cast_nf_asl_rnd_a40(acc, 16 - shift_right);
}

template <> MLI_FORCE_INLINE int16_t mli_math_acc_cast_fx(mli_acc32_t acc, int shift_right) {
    int32_t temp = (int32_t) fx_asr_rnd_q31(acc, shift_right);
    temp = fx_asl_q31(temp, 16);
    return (int16_t) fx_q15_cast_q31(temp);
}

template <> MLI_FORCE_INLINE v2q15_t mli_math_acc_cast_fx(v2accum40_t acc, int shift_right) {
    return fx_v2q15_cast_nf_asl_rnd_v2a40(acc, 16 - shift_right);
}

template <> MLI_FORCE_INLINE mli_acc32_t mli_math_acc_cast_fx(mli_acc32_t acc, int shift_right) {
    return (int32_t) fx_asr_rnd_q31(acc, shift_right);
}

/*
*   Vectorized version of fx_q7_cast_rnd_q15() with Q7 saturation after rounding
*/
static MLI_FORCE_INLINE v2q15_t mli_prv_v2q7_cast_rnd_v2q15(v2q15_t x) {
    return fx_sat_v2q15_n(fx_asr_rnd_v2q15_n(x, 8), 8);
}

// Cast scalar to/from void pointer
//========================================================================
template < typename out_T > MLI_FORCE_INLINE out_T mli_math_cast_ptr_to_scalar_fx(void *src) {
    // REMARK: Need to check from C/Cpp standard point of view
    return static_cast < out_T > ((intptr_t) (src));
}

template < typename in_T > MLI_FORCE_INLINE void *mli_math_cast_scalar_to_ptr_fx(in_T src) {
    // REMARK: Need to check from C/Cpp standard point of view
    intptr_t out_upcast = src;
    return static_cast < void *>((intptr_t *) out_upcast);
}

// Comparators
//========================================================================
template < typename io_T > 
static MLI_FORCE_INLINE bool mli_prv_less_than_1(io_T value, uint8_t frac_bits) {
    if (frac_bits >= sizeof(io_T) * 8 - 1)
        return true;

    io_T unit = (io_T) 1 << frac_bits;
    return (value < unit);
}

template <typename in_T>
MLI_FORCE_INLINE in_T mli_math_asr_rnd_fx(in_T x, int nbits);

template <>
MLI_FORCE_INLINE int16_t mli_math_asr_rnd_fx(int16_t x, int nbits) {
    return fx_asr_rnd_q15(x, nbits);
}

template <>
MLI_FORCE_INLINE int32_t mli_math_asr_rnd_fx(int32_t x, int nbits) {
    return fx_asr_rnd_q31(x, nbits);
}

template <>
MLI_FORCE_INLINE int64_t mli_math_asr_rnd_fx(int64_t x, int nbits) {
    return fx_asr_rnd_q63(x, nbits);
}

// Arithmetic shift (right is default, left on the negative val)
//========================================================================

template <>
MLI_FORCE_INLINE int8_t mli_math_ashift_right_fx(int8_t in_val, int shift_right) {
    int16_t shifted_in_val = mli_math_asr_rnd_fx<int16_t>((int16_t)in_val, shift_right);
    return (int8_t)mli_math_sat_fx<int16_t>(shifted_in_val, 8);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_ashift_right_fx(int16_t in_val, int shift_right) {
    return mli_math_asr_rnd_fx<int16_t>(in_val, shift_right);
}

template <>
MLI_FORCE_INLINE int32_t mli_math_ashift_right_fx(int32_t in_val, int shift_right) {
    return mli_math_asr_rnd_fx<int32_t>(in_val, shift_right);
}

template <typename in_T, typename acc_T>
MLI_FORCE_INLINE acc_T mli_math_mul_fx_high(in_T L, in_T R);

// Multiply and cast float to accum
//========================================================================

MLI_FORCE_INLINE int32_t mli_math_float_scale(float value, float scale) {
    /* This function is taken from reference mli_math as is */
    const float round_val = value > 0 ? 0.5f : -0.5f;
    return (int32_t)(value * scale + round_val);
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

template <typename acc_T, bool asym_data>
MLI_FORCE_INLINE mli_acc40_t mli_math_scale_mul(mli_acc40_t accu, int32_t mul) {
    if (asym_data) {
        MLI_ASSERT(0); // asymetric data not supported with 40bit accumulator
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

template <typename b_T, typename acc_T, bool asym_data>
MLI_FORCE_INLINE int32_t mli_math_init_accu(int32_t bias, int32_t bias_mul, int bias_shift) {
    int32_t accu = bias;
    _setacc(accu, 1);
    // for an int32_t bias type and int32_t accumulator, there is no requantization of the bias.
    // in the mli_check function it has been checked that the bias scale and shift match the accumulator scale and shift.
    //accu = mli_math_acc_ashift_fx(accu, bias_shift);
    //accu = mli_math_scale_mul<acc_T, asym_data>(accu, bias_mul);
    return accu;
}

// Cast value to output type (including accumulator type)
//========================================================================
template <>
MLI_FORCE_INLINE int8_t mli_math_cast_fx(int8_t in_val, int shift_right) {
    return (int8_t)fx_sat_q15(fx_asr_rnd_q15((int16_t)in_val, shift_right), 8);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(int8_t in_val, int shift_right) {
    return (int16_t)fx_asr_rnd_q15((int16_t)in_val, shift_right);
}

template <>
MLI_FORCE_INLINE int8_t mli_math_cast_fx(int16_t in_val, int shift_right) {
    return (int8_t)fx_sat_q15(fx_asr_rnd_q15(in_val, shift_right), 8);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(int16_t in_val, int shift_right) {
    return (int16_t)fx_asr_rnd_q15(in_val, shift_right);
}

template <>
MLI_FORCE_INLINE int8_t mli_math_cast_fx(mli_acc32_t in_val, int shift_right) {
    int32_t temp = (int32_t)fx_asr_rnd_q31((int32_t)in_val, shift_right);
    temp = fx_asl_q31(temp, 24);
    return (int8_t)fx_q7_cast_q31(temp);
}

template <>
MLI_FORCE_INLINE int32_t mli_math_cast_fx(mli_acc32_t in_val, int shift_right) {
    return (int32_t)fx_asr_rnd_q31(in_val, shift_right);
}



template <>
MLI_FORCE_INLINE mli_acc40_t mli_math_cast_fx(int16_t in_val, int shift_right) {
    return fx_asr_a40(fx_a40_cast_q15(in_val), shift_right + 15);
}


template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(mli_acc40_t in_val, int shift_right) {
    return fx_q15_cast_nf_asl_rnd_a40(in_val, 16 - shift_right);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(mli_acc40_t in_val) {
    return fx_q15_cast_nf_asl_rnd_a40(in_val, 16);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(mli_acc32_t in_val, int shift_right) {
    int32_t temp = (int32_t)fx_asr_rnd_q31(in_val, shift_right);
    temp = fx_asl_q31(temp, 16);
    return (int16_t)fx_q15_cast_q31(temp);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(mli_acc32_t in_val) {
	int32_t temp = fx_asl_q31(in_val, 16);
    return (int16_t)fx_q15_cast_q31(temp);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(int64_t in_val, int shift_right) {
    int32_t temp = (int32_t)fx_asr_rnd_q63(in_val, shift_right);
    temp = fx_asl_q31(temp, 16);
    return (int16_t)fx_q15_cast_q31(temp);
}

template <>
MLI_FORCE_INLINE int8_t mli_math_cast_fx(int64_t in_val, int shift_right) {
    int64_t temp = (int64_t)fx_asr_rnd_q63((int64_t)in_val, shift_right);
    temp = fx_asl_q63(temp, 56);
    return (int8_t)fx_q7_cast_q63(temp);
}

template <>
MLI_FORCE_INLINE int32_t mli_math_cast_fx(mli_acc40_t in_val, int shift_right) {
    return (int32_t)fx_q31_cast_nf_asl_rnd_a40(in_val, 32 - shift_right);
}

template <>
MLI_FORCE_INLINE int32_t mli_math_cast_fx(int64_t in_val, int shift_right) {
    in_val = fx_asr_rnd_q63(in_val, shift_right);
    in_val = fx_asl_q63(in_val, 32);
    return fx_q31_cast_q63(in_val);
}

template <>
MLI_FORCE_INLINE int32_t mli_math_cast_fx(accum72_t in_val, int shift_right) {
    return fx_q31_cast_nf_asl_rnd_a72(in_val, 64 - sizeof(int32_t) * 8 - shift_right);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(accum72_t in_val, int shift_right) {
    return fx_q15_cast_nf_asl_rnd_a72(in_val, 64 - sizeof(int16_t) * 8 - shift_right);
}

template <>
MLI_FORCE_INLINE float mli_math_cast_fx(int32_t in_val, int shift_right) {
    /* This function is taken from reference mli_math as is */
    return (float)in_val / (float)(1 << shift_right);
}

template<typename io_T, typename l_T, typename r_T>
MLI_FORCE_INLINE io_T mli_math_bound_range_fx(io_T in, l_T L, r_T R) {
    io_T out;
    out = mli_math_max_fx(in, L);
    out = mli_math_min_fx(out, R);
    return out;
}

// Number of lanes in a vector
//========================================================================
template <typename T>
MLI_FORCE_INLINE int get_number_lanes() {
    int lanes = 0;
    if (  std::is_same<T, int8_t>::value
       || std::is_same<T, int16_t>::value
       || std::is_same<T, int32_t>::value
       || std::is_same<T, mli_acc40_t>::value
       || std::is_same<T, int64_t>::value
       || std::is_same<T, uint8_t>::value
       || std::is_same<T, uint16_t>::value
       || std::is_same<T, uint32_t>::value
       || std::is_same<T, uint64_t>::value) {
        lanes = 1;
    }

    if (  std::is_same<T, v2q15_t>::value
       || std::is_same<T, v2accum32_t>::value
       || std::is_same<T, __v2i32_t>::value
       || std::is_same<T, v2accum40_t>::value) {
        lanes = 2;
    }

    MLI_ASSERT(lanes > 0);
    return lanes;
}

template <typename T>
MLI_FORCE_INLINE int get_number_lanes(T dummy) {
    return get_number_lanes<T>();
}

#pragma MLI_CODE_SECTION_END()


#endif // _DSP_MLI_MATH_H_
