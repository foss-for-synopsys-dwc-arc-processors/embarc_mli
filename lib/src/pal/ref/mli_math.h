/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _REF_MLI_MATH_H_
#define _REF_MLI_MATH_H_

////////////////////////////////////////////////////////////////////
// TODO: TO BE UPDATED
////////////////////////////////////////////////////////////////////

typedef int32_t   mli_acc32_t;

// Cast accum to output type
//========================================================================
template <>
MLI_FORCE_INLINE int8_t mli_math_acc_cast_fx(mli_acc32_t acc, int shift_right) {
    int32_t temp = (int32_t) mli_math_asr_rnd_fx<mli_acc32_t>(acc, shift_right);
    temp = mli_math_asl_fx<int32_t>(temp, 24);
    return (int8_t) (temp >> 24);
}

template <> 
MLI_FORCE_INLINE int16_t mli_math_acc_cast_fx(mli_acc32_t acc, int shift_right) {
    int32_t temp = (int32_t) mli_math_asr_rnd_fx<mli_acc32_t>(acc, shift_right);
    temp = mli_math_asl_fx<mli_acc32_t>(temp, 16);
    return (int16_t) mli_math_sat_fx<mli_acc32_t>(mli_math_asr_fx<mli_acc32_t>(temp, 16), 16);
}

// Addition of two fx operands with saturation
//========================================================================
template <typename io_T>
MLI_FORCE_INLINE io_T mli_math_add_fx(io_T L, io_T R) {
    return L + R;
}

template <> 
MLI_FORCE_INLINE int8_t mli_math_add_fx(int8_t L, int8_t R) {
    return (int8_t)mli_math_sat_fx<int16_t>(L + R, 8);
}

template <> 
MLI_FORCE_INLINE int16_t mli_math_add_fx(int16_t L, int16_t R) {
    return (int16_t)mli_math_sat_fx<int32_t>(L + R, 16);
}

template <> 
MLI_FORCE_INLINE int32_t mli_math_add_fx(int32_t L, int32_t R) {
    return (int32_t)mli_math_sat_fx<int64_t>(L + R, 32);
}

// Subtraction of two fx operands with saturation
//========================================================================
template <typename io_T>
MLI_FORCE_INLINE io_T mli_math_sub_fx(io_T L, io_T R) {
    return L - R;
}

template <> 
MLI_FORCE_INLINE int8_t mli_math_sub_fx(int8_t L, int8_t R) {
    return (int8_t)mli_math_sat_fx<int16_t>(L - R, 8);
}

template <> 
MLI_FORCE_INLINE int16_t mli_math_sub_fx(int16_t L, int16_t R) {
    return (int16_t)mli_math_sat_fx<int32_t>(L - R, 16);
}

template <> 
MLI_FORCE_INLINE int32_t mli_math_sub_fx(int32_t L, int32_t R) {
    return (int32_t)mli_math_sat_fx<int64_t>(L - R, 32);
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

template <>
MLI_FORCE_INLINE int64_t mli_math_mul_fx(int32_t L, int32_t R) {
    // Result of multiplication is fractional number (shifted left by 1)
    // To return correct result we shift it right afterward
    int64_t acc = (int64_t)L * (int64_t)R;
    return (int64_t) acc;
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

#endif // _REF_MLI_MATH_H_
