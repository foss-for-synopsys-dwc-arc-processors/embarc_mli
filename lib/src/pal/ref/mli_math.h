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

#include <limits>
#include <type_traits>
#include "mli_debug.h"

typedef int32_t mli_acc32_t;
typedef int64_t mli_acc40_t;

template <typename T>
MLI_FORCE_INLINE T mli_math_asl_fx(T x, int nbits);

template <typename T>
MLI_FORCE_INLINE T mli_math_limit_fx(T sign) {
    return sign < (T)0 ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
}

template <typename T>
MLI_FORCE_INLINE T mli_math_asr_pos_nbits_fx(T x, int nbits)
{
    // This function is internal to this file
    // and is required to avoid compiler issues.
    if (nbits > (sizeof(T) * 8 - 1))
        return x < (T)0 ? -1 : 0;
    return x >> nbits;
}

template <typename T>
MLI_FORCE_INLINE T mli_math_asr_fx(T x, int nbits)
{
    int nbits_max = sizeof(T) * 8 - 1;
    int nbits_min = 0;
    if (nbits > nbits_max)
        return x < (T)0 ? -1 : 0;
    if (nbits < nbits_min)
        return mli_math_asl_fx<T>(x, (-nbits));
    return x >> nbits;
}

template <typename T>
MLI_FORCE_INLINE T mli_math_asl_fx(T x, int nbits)
{
    int inp_size = sizeof(T) * 8;
    T hi = 0;

    if (nbits < 0)
        return mli_math_asr_pos_nbits_fx<T>(x, (-nbits));

    if (nbits > (inp_size - 1))
        return x != (T)0 ? mli_math_limit_fx<T>(x) : 0;

    hi = x >> ((inp_size - 1) - nbits);
    if (hi == (T)0 || hi == (T)-1)
        return x << nbits;

    return mli_math_limit_fx<T>(hi);
}

template <typename T>
MLI_FORCE_INLINE T mli_math_asr_rnd_fx(T x, int nbits)
{
    using unsigned_T = typename std::make_unsigned<T>::type;
    T r = 0;
    unsigned_T one = 1u;

    if (nbits < 0)
        return mli_math_asl_fx<T>(x, (-nbits));
    if (nbits == 0)
        return x;

    if (nbits > (sizeof(T) * 8 - 1))
        return 0;

    // Rounding up:
    // if the most significant deleted bit is 1, add 1 to the remaining bits.
#ifdef ROUND_UP
    T round = (T)((one << nbits) >> 1);
    r = mli_math_add_fx<T>(x, round);
    r = mli_math_asr_fx<T>(r, nbits);
#endif

    // Convergent: rounding with half-way value rounded to even value.
    // If the most significant deleted bit is 1, and 
    // either the least significant of the remaining bits
    // or at least one other deleted bit is 1, add 1 to the remaining bits.
#ifdef ROUND_CONVERGENT
    r = mli_math_asr_fx<T>(x, nbits);
    T last_deleted_mask = (T)((one << nbits) >> 1);
    if (((x & last_deleted_mask) != (T)0) && 
            (((r & (T)1) != (T)0) ||  ((x & (last_deleted_mask-(T)1))!= (T)0))) {
        return mli_math_add_fx<T>(r, 1);
    }
#endif

    return r;
}

template <typename T>
MLI_FORCE_INLINE T mli_math_asl_rnd_fx(T x, int nbits)
{
    return mli_math_asr_rnd_fx<T>(x, -nbits);
}

template <typename T>
MLI_FORCE_INLINE T mli_math_neg_fx(T x) {
    return x == std::numeric_limits<T>::lowest() ? std::numeric_limits<T>::max() : -x;
}

template <typename T>
MLI_FORCE_INLINE T mli_math_sat_fx(T x, unsigned nbits)
{
    return mli_math_asr_fx<T>(mli_math_asl_fx<T>(x, nbits), nbits);
}

template <typename T>
MLI_FORCE_INLINE T mli_math_abs_fx(T x)
{    
    return x >= (T)0 ? x : mli_math_neg_fx(x);
}

template <typename T, typename o_T>
MLI_FORCE_INLINE o_T mli_math_norm_fx(T x)
{
    o_T inp_size = sizeof(T) * 8;
    T hi = x < (T)0 ? (T)-1 : (T)0;
    o_T r = 0;

    if (x == (T)0)
        return inp_size - 1;

    while ((x >> r) != hi)
        r++;
    return (inp_size - 1) - r;
}

template<typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_math_norm_cast_fx(in_T val , int32_t *norm_shift) {
    int32_t cast_shift = (sizeof(in_T) - sizeof(out_T)) * 8;
    int32_t norm = mli_math_norm_fx<in_T, int32_t>(val);
    *norm_shift = cast_shift - norm;
    return mli_math_cast_fx<in_T, out_T>(val, *norm_shift);
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

// Comparators
//========================================================================
template < typename io_T > 
MLI_FORCE_INLINE bool mli_prv_less_than_1(io_T value, uint8_t frac_bits) {
    if (frac_bits >= sizeof(io_T) * 8 - 1)
        return true;

    io_T unit = (io_T) 1 << frac_bits;
    return (value < unit);
}

// Accumulator shift
//========================================================================

template <>
MLI_FORCE_INLINE mli_acc32_t mli_math_acc_ashift_fx(mli_acc32_t acc, int shift_right) {
    return mli_math_asr_rnd_fx<mli_acc32_t>(acc, shift_right);
}

template <>
MLI_FORCE_INLINE mli_acc40_t mli_math_acc_ashift_fx(mli_acc40_t acc, int shift_right) {
    return mli_math_asr_rnd_fx<mli_acc40_t>(acc, shift_right);
}

// Arithmetic shift (right is default, left on the negative val)
//========================================================================

template <>
MLI_FORCE_INLINE int8_t mli_math_ashift_right_fx(int8_t in_val, int shift_right) {
    int8_t shifted_in_val = mli_math_asr_rnd_fx<int8_t>((int8_t)in_val, shift_right);
    return shifted_in_val;
}

template <>
MLI_FORCE_INLINE int16_t mli_math_ashift_right_fx(int16_t in_val, int shift_right) {
    return mli_math_asr_rnd_fx<int16_t>(in_val, shift_right);
}

template <>
MLI_FORCE_INLINE int32_t mli_math_ashift_right_fx(int32_t in_val, int shift_right) {
    return mli_math_asr_rnd_fx<int32_t>(in_val, shift_right);
}

// Multiply and cast float to accum
//========================================================================

MLI_FORCE_INLINE int32_t mli_math_float_scale(float value, float scale) {
    const float round_val = value > 0 ? 0.5f : -0.5f;
    return (int32_t)(value * scale + round_val);
}

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

template <> MLI_FORCE_INLINE mli_acc32_t mli_math_acc_cast_fx(mli_acc32_t acc, int shift_right) {
    return (int32_t) mli_math_asr_rnd_fx<mli_acc32_t>(acc, shift_right);
}

// Cast value to output type (including accumulator type)
//========================================================================
template <>
MLI_FORCE_INLINE int8_t mli_math_cast_fx(int16_t in_val, int shift_right) {
    return (int8_t)mli_math_sat_fx<int16_t>(mli_math_asr_rnd_fx<int16_t>(in_val, shift_right), 8);
}

template <>
MLI_FORCE_INLINE int8_t mli_math_cast_fx(int8_t in_val, int shift_right) {
    return (int8_t)mli_math_asr_rnd_fx<int8_t>((int8_t)in_val, shift_right);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(int8_t in_val, int shift_right) {
    return (int16_t)mli_math_asr_rnd_fx<int16_t>((int16_t)in_val, shift_right);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(int16_t in_val, int shift_right) {
    return (int16_t)mli_math_asr_rnd_fx<int16_t>(in_val, shift_right);
}

template <>
MLI_FORCE_INLINE int8_t mli_math_cast_fx(mli_acc32_t in_val, int shift_right) {
    int32_t temp = (int32_t)mli_math_asr_rnd_fx<mli_acc32_t>((int32_t)in_val, shift_right);
    return (int8_t)mli_math_sat_fx<int32_t>(temp, 24);
}


template <>
MLI_FORCE_INLINE int8_t mli_math_cast_fx(mli_acc32_t in_val) {
    int32_t temp = (int32_t)mli_math_asr_rnd_fx<mli_acc32_t>((int32_t)in_val, 24);
    return (int8_t)mli_math_sat_fx<int32_t>(temp, 24);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(mli_acc32_t in_val, int shift_right) {
    int32_t temp = (int32_t)mli_math_asr_rnd_fx<mli_acc32_t>(in_val, shift_right);
    return (int16_t)mli_math_sat_fx<int32_t>(temp, 16);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(mli_acc32_t in_val) {
    return (int16_t)mli_math_sat_fx<int32_t>(in_val, 16);
}

template <>
MLI_FORCE_INLINE int32_t mli_math_cast_fx(mli_acc32_t in_val, int shift_right) {
    return (int32_t)mli_math_asr_rnd_fx<mli_acc32_t>(in_val, shift_right);
}

template <>
MLI_FORCE_INLINE mli_acc40_t mli_math_cast_fx(int16_t in_val, int shift_right) {
    return (int32_t)mli_math_asr_rnd_fx<mli_acc40_t>((mli_acc40_t)in_val, shift_right);
}

template <>
MLI_FORCE_INLINE int8_t mli_math_cast_fx(int64_t in_val, int shift_right) {
    int64_t temp = (int64_t)mli_math_asr_rnd_fx<int64_t>((int64_t)in_val, shift_right);
    return (int8_t)mli_math_sat_fx<int64_t>(temp, 56);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(int64_t in_val, int shift_right) {
    int64_t temp = (int64_t)mli_math_asr_rnd_fx<int64_t>(in_val, shift_right);
    return (int16_t)mli_math_sat_fx<int64_t>(temp, 48);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(int64_t in_val) {
    return (int16_t)mli_math_sat_fx<int64_t>(in_val, 48);
}

template <>
MLI_FORCE_INLINE int32_t mli_math_cast_fx(int64_t in_val, int shift_right) {
    in_val = mli_math_asr_rnd_fx<int64_t>(in_val, shift_right);
    return (int32_t)mli_math_sat_fx<int64_t>(in_val, 32);
}

template <>
MLI_FORCE_INLINE float mli_math_cast_fx(int32_t in_val, int shift_right) {
    return (float)in_val / (float)(1 << shift_right);
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

template<typename io_T, typename lr_T>
MLI_FORCE_INLINE io_T mli_math_bound_range_fx(io_T in, lr_T L, lr_T R) {
    io_T out;
    out = mli_math_max_fx(in, L);
    out = mli_math_min_fx(out, R);
    return out;
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
MLI_FORCE_INLINE mli_acc32_t mli_math_mac_fx(mli_acc32_t acc, int8_t L, int16_t R) {
    return acc + (mli_acc32_t)((int16_t)L * R);
}

template <>
MLI_FORCE_INLINE mli_acc40_t mli_math_mac_fx(mli_acc40_t acc, int16_t L, int16_t R) {
    return acc + (mli_acc40_t) ((int32_t)L * (int32_t)R);
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
    MLI_ASSERT(lanes > 0);
    return lanes;
}

template <typename T>
MLI_FORCE_INLINE int get_number_lanes(T dummy) {
    return get_number_lanes<T>();
}

#endif // _REF_MLI_MATH_H_
