/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_MATH_H_
#define _MLI_MATH_H_

#if defined(__FXAPI__)
#include "fxarc.h"
#else
#error "ARC FX Library (FXAPI) is required dependency"
#endif

#include "mli_debug.h"
#include "mli_math_macros.h"
#include "mli_private_types.h"

#ifdef _ARC
#include <arc/arc_intrinsics.h>
#endif

#pragma Code(".mli_lib")

//=========================================================================
//
// Declaration
//
//=========================================================================

template < typename io_T > inline io_T mli_math_add_fx(io_T L, io_T R);
template < typename io_T > inline io_T mli_math_sub_fx(io_T L, io_T R);
template < typename io_T > inline io_T mli_math_max_fx(io_T L, io_T R);
template < typename io_T > inline io_T mli_math_min_fx(io_T L, io_T R);
template < typename in_T, typename acc_T > inline acc_T mli_math_mul_fx(in_T L, in_T R);
template < typename in_T, typename acc_T > inline acc_T mli_math_mul_fx_high(in_T L, in_T R);
template < typename l_T, typename r_T, typename acc_T > inline acc_T mli_math_mac_fx(acc_T acc, l_T L, r_T R);
template < typename out_T, typename acc_T > inline out_T mli_math_acc_cast_fx(acc_T acc, int shift_right);
template < typename acc_T > inline acc_T mli_math_acc_ashift_fx(acc_T acc, int shift_right);
template < typename out_T > inline out_T mli_math_cast_ptr_to_scalar_fx(void *src);
template < typename in_T > inline void *mli_math_cast_scalar_to_ptr_fx(in_T src);

template <typename in_T, typename out_T> inline out_T mli_math_cast_fx(in_T in_val, int shift_right);
//=========================================================================
//
// Definitions
//
//=========================================================================

// Addition of two fx operands with saturation
//========================================================================
template <> inline int8_t mli_math_add_fx(int8_t L, int8_t R) {
    return (int8_t)fx_sat_q15(L + R, 8);
}

template <> inline int16_t mli_math_add_fx(int16_t L, int16_t R) {
    return fx_add_q15(L, R);   // not cast operands intentionally (rise compile warnings)
}

template <>
inline int32_t mli_math_add_fx(int32_t L, int32_t R) {
    return (int32_t)fx_add_q31(L, R);
}
template <>
inline mli_acc40_t mli_math_add_fx(mli_acc40_t L, mli_acc40_t R) {
    return fx_add_a40(L, R);
}
// Subtraction of two fx operands with saturation
//========================================================================
template <> inline int8_t mli_math_sub_fx(int8_t L, int8_t R) {
    return (int8_t)fx_sat_q15(L - R, 8);
}

template <> inline int16_t mli_math_sub_fx(int16_t L, int16_t R) {
    return fx_sub_q15(L, R);   // not cast operands intentionally (rise compile warnings)
}

// Maximum of two fx operands
//========================================================================
template < typename io_T > inline io_T mli_math_max_fx(io_T L, io_T R) {
    return MAX(L, R);
}

// Minimum of two fx operands
//========================================================================
template < typename io_T > inline io_T mli_math_min_fx(io_T L, io_T R) {
    return MIN(L, R);
}

// Multiply two operands
//========================================================================
template <> inline mli_acc32_t mli_math_mul_fx(int8_t L, int8_t R) {
    return (mli_acc32_t) (L * R);
}

template <> inline mli_acc32_t mli_math_mul_fx(int16_t L, int16_t R) {
    return (mli_acc32_t) (L * R);
}

template <> inline mli_acc40_t mli_math_mul_fx(int16_t L, int16_t R) {
    return fx_a40_mpy_nf_q15(L, R);
}

template <> inline mli_acc32_t mli_math_mul_fx_high(int32_t L, int32_t R) {
    // this function takes the MSB part of the result. (L * R) >> 31
    // in optimized code check if mpyfr instruction is used here.
    return (mli_acc32_t)fx_q31_cast_rnd_a72(fx_a72_mpy_q31(L, R));
}

template <>
inline int64_t mli_math_mul_fx(int32_t L, int32_t R) {
    // Result of multiplication is fractional number (shifted left by 1)
    // To return correct result we shift it right afterward
    return (int64_t )(fx_q63_cast_a72(fx_a72_mpy_q31(L, R)) >> 1);
}

// Multiply-and-accumulate operands
//========================================================================
template <> inline mli_acc32_t mli_math_mac_fx(mli_acc32_t acc, int8_t L, int8_t R) {
    return acc + (mli_acc32_t) (L * R);
}

template <> inline mli_acc32_t mli_math_mac_fx(mli_acc32_t acc, int16_t L, int16_t R) {
    return acc + (mli_acc32_t) (L * R);
}

template <> inline mli_acc32_t mli_math_mac_fx(mli_acc32_t acc, int16_t L, int8_t R) {
    return acc + (mli_acc32_t)(L * (int16_t)R);
}

template <> inline mli_acc40_t mli_math_mac_fx(mli_acc40_t acc, int16_t L, int16_t R) {
    return fx_a40_mac_nf_q15(acc, L, R);
}

template <> inline mli_acc40_t mli_math_mac_fx(mli_acc40_t acc, int16_t L, int8_t R) {
    return fx_a40_mac_nf_q15(acc, L, (int16_t) R);
}

template <> inline mli_acc40_t mli_math_mac_fx(mli_acc40_t acc, int8_t L, int8_t R) {
    return fx_a40_mac_nf_q15(acc, (int16_t)L, (int16_t)R);
}

static inline void __attribute__ ((always_inline)) mli_math_mac_fx_vec2(__v2i32_t * accu, v2q15_t in, v2q15_t k) { //mli_math_mac_fx_vec2 , acc by value
    *accu += __builtin_convertvector(in, __v2i32_t) * __builtin_convertvector(k, __v2i32_t);
}

static inline void __attribute__ ((always_inline)) mli_math_mac_fx_vec2(v2accum40_t * accu, v2q15_t in, v2q15_t k) {//mli_math_mac_fx_vec2
    *accu = fx_v2a40_mac_nf_v2q15(*accu, in, k);
}

// Accumulator shift
//========================================================================

template <> inline mli_acc32_t mli_math_acc_ashift_fx(mli_acc32_t acc, int shift_right) {
    return fx_asr_rnd_q31(acc, shift_right);
}

template <> inline mli_acc40_t mli_math_acc_ashift_fx(mli_acc40_t acc, int shift_right) {
    return fx_asr_a40(acc, shift_right);
}

// Cast accum to output type
//========================================================================
template <> inline int8_t mli_math_acc_cast_fx(mli_acc32_t acc, int shift_right) {
    int32_t temp = (int32_t) fx_asr_rnd_q31(acc, shift_right);
    temp = fx_asl_q31(temp, 24);
    return (int8_t) fx_q7_cast_q31(temp);
}

template <> inline int8_t mli_math_acc_cast_fx(mli_acc40_t acc, int shift_right) {
    return fx_q7_cast_nf_asl_rnd_a40(acc, 24 - shift_right);
}

template <> inline int16_t mli_math_acc_cast_fx(mli_acc40_t acc, int shift_right) {
    return fx_q15_cast_nf_asl_rnd_a40(acc, 16 - shift_right);
}

template <> inline int16_t mli_math_acc_cast_fx(mli_acc32_t acc, int shift_right) {
    int32_t temp = (int32_t) fx_asr_rnd_q31(acc, shift_right);
    temp = fx_asl_q31(temp, 16);
    return (int16_t) fx_q15_cast_q31(temp);
}

/*
*   Vectorized version of fx_q7_cast_rnd_q15() with Q7 saturation after rounding
*/
static v2q15_t __attribute__ ((always_inline)) mli_prv_v2q7_cast_rnd_v2q15(v2q15_t x) {
    return fx_sat_v2q15_n(fx_asr_rnd_v2q15_n(x, 8), 8);
}

// Cast scalar to/from void pointer
//========================================================================
template < typename out_T > inline out_T mli_math_cast_ptr_to_scalar_fx(void *src) {
    // REMARK: Need to check from C/Cpp standard point of view
    return static_cast < out_T > ((intptr_t) (src));
}

template < typename in_T > inline void *mli_math_cast_scalar_to_ptr_fx(in_T src) {
    // REMARK: Need to check from C/Cpp standard point of view
    intptr_t out_upcast = src;
    return static_cast < void *>((intptr_t *) out_upcast);
}

// Comparators
//========================================================================
template < typename io_T > 
static inline bool __attribute__ ((always_inline)) mli_prv_less_than_1(io_T value, uint8_t frac_bits) {
    if (frac_bits > sizeof(io_T) * 8 - 1)
        return true;

    io_T unit = (io_T) 1 << frac_bits;
    return (value < unit);
}

template <typename in_T, typename acc_T>
inline acc_T mli_math_mul_fx_high(in_T L, in_T R);

// Multipliers used to apply scale factors in asymetric data types
//========================================================================
template <typename acc_T, bool asym_data>
static inline mli_acc32_t mli_math_scale_mul(mli_acc32_t accu, int32_t mul) {
    if (asym_data) {
        return accu = mli_math_mul_fx_high<int32_t, int32_t>(accu, mul);
    }else{
        return accu;
    }
}

template <typename acc_T, bool asym_data>
static inline mli_acc40_t mli_math_scale_mul(mli_acc40_t accu, int32_t mul) {
    if (asym_data) {
        MLI_ASSERT(0); // asymetric data not supported with 40bit accumulator
    }else{
        return accu;
    }
}

template <typename b_T, typename acc_T, bool asym_data>
static inline acc_T mli_math_init_accu(b_T bias, int32_t bias_mul, int bias_shift) {
    acc_T accu = mli_math_mul_fx<b_T, acc_T>(bias, asym_data ? 2 : 1); // extra factor of 2 needed because scale mul cannot multiply by 1.
    accu = mli_math_acc_ashift_fx(accu, -bias_shift);
    accu = mli_math_scale_mul<acc_T, asym_data>(accu, bias_mul);
    return accu;
}

template <typename b_T, typename acc_T, bool asym_data>
static inline int32_t mli_math_init_accu(int32_t bias, int32_t bias_mul, int bias_shift) {
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
inline int16_t mli_math_cast_fx(int8_t in_val, int shift_right) {
    return (int16_t)fx_asr_rnd_q15((int16_t)in_val, shift_right);
}

template <>
inline int8_t mli_math_cast_fx(int16_t in_val, int shift_right) {
    return (int8_t)fx_sat_q15(fx_asr_rnd_q15(in_val, shift_right), 8);
}

template <>
inline int16_t mli_math_cast_fx(int16_t in_val, int shift_right) {
    return (int16_t)fx_asr_rnd_q15(in_val, shift_right);
}

template <>
inline int8_t mli_math_cast_fx(mli_acc32_t in_val, int shift_right) {
    int32_t temp = (int32_t)fx_asr_rnd_q31((int32_t)in_val, shift_right);
    temp = fx_asl_q31(temp, 24);
    return (int8_t)fx_q7_cast_q31(temp);
}

template <>
inline int32_t mli_math_cast_fx(mli_acc32_t in_val, int shift_right) {
    return (int32_t)fx_asr_rnd_q31(in_val, shift_right);
}



template <>
inline mli_acc40_t mli_math_cast_fx(int16_t in_val, int shift_right) {
    return fx_asr_a40(fx_a40_cast_q15(in_val), shift_right + 15);
}


template <>
inline int16_t mli_math_cast_fx(mli_acc40_t in_val, int shift_right) {
    return fx_q15_cast_nf_asl_rnd_a40(in_val, 16 - shift_right);
}

template <>
inline int16_t mli_math_cast_fx(mli_acc32_t in_val, int shift_right) {
    int32_t temp = (int32_t)fx_asr_rnd_q31(in_val, shift_right);
    temp = fx_asl_q31(temp, 16);
    return (int16_t)fx_q15_cast_q31(temp);
}

template <>
inline int16_t mli_math_cast_fx(int64_t in_val, int shift_right) {
    int32_t temp = (int32_t)fx_asr_rnd_q63(in_val, shift_right);
    temp = fx_asl_q31(temp, 16);
    return (int16_t)fx_q15_cast_q31(temp);
}

template <>
inline int32_t mli_math_cast_fx(mli_acc40_t in_val, int shift_right) {
    return (int32_t)fx_q31_cast_nf_asl_rnd_a40(in_val, 32 - shift_right);
}

#pragma Code()


#endif // _MLI_MATH_H_
