/*
 *  Copyright (c) 2019, Synopsys, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1) Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 
 * 2)  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3) Neither the name of the <ORGANIZATION> nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _MLI_MATH_H_
#define _MLI_MATH_H_

#if defined(__FXAPI__)
#include "fxarc.h"
#else
#error "ARC FX Library (FXAPI) is required dependency"
#endif

#include "mli_math_macros.h"
#include "mli_private_types.h"


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
template < typename l_T, typename r_T, typename acc_T > inline acc_T mli_math_mac_fx(acc_T acc, l_T L, r_T R);
template < typename out_T, typename acc_T > inline out_T mli_math_acc_cast_fx(acc_T acc, int shift_right);
template < typename acc_T > inline acc_T mli_math_acc_ashift_fx(acc_T acc, int shift_right);
template < typename out_T > inline out_T mli_math_cast_ptr_to_scalar_fx(void *src);
template < typename in_T > inline void *mli_math_cast_scalar_to_ptr_fx(in_T src);

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

// Multiply-and-accumulate operands
//========================================================================
template <> inline mli_acc32_t mli_math_mac_fx(mli_acc32_t acc, int8_t L, int8_t R) {
    return acc + (mli_acc32_t) (L * R);
}

template <> inline mli_acc32_t mli_math_mac_fx(mli_acc32_t acc, int16_t L, int16_t R) {
    return acc + (mli_acc32_t) (L * R);
}

template <> inline mli_acc40_t mli_math_mac_fx(mli_acc40_t acc, int16_t L, int16_t R) {
    return fx_a40_mac_nf_q15(acc, L, R);
}

template <> inline mli_acc40_t mli_math_mac_fx(mli_acc40_t acc, int16_t L, int8_t R) {
    return fx_a40_mac_nf_q15(acc, L, (int16_t) R);
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

#pragma Code()


#endif // _MLI_MATH_H_
