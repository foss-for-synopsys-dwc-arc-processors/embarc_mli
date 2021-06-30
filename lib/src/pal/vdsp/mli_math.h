/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _VDSP_MLI_MATH_H_
#define _VDSP_MLI_MATH_H_

#include <arc_vector.h>
#include <arc/arc_intrinsics.h>
#include <type_traits>
#include <limits>
#include "arc_vector_ext.h"
#include "mli_debug.h"


//=========================================================================
//
// Definitions
//
//=========================================================================
typedef int32_t mli_acc32_t;
typedef int64_t mli_acc40_t;

// Number of lanes in a vector
//========================================================================
template <typename T>
constexpr int get_number_lanes() {
    return 
     (  std::is_same<T, int8_t>::value
       || std::is_same<T, int16_t>::value
       || std::is_same<T, int32_t>::value
       || std::is_same<T, mli_acc40_t>::value
       || std::is_same<T, int64_t>::value
       || std::is_same<T, uint8_t>::value
       || std::is_same<T, uint16_t>::value
       || std::is_same<T, uint32_t>::value
       || std::is_same<T, uint64_t>::value) ?
        1
    :
     (  std::is_same<T, vNx4char_t>::value
       || std::is_same<T, vNx4short_t>::value
       || std::is_same<T, vNx4int_t>::value
       || std::is_same<T, vNx4accshort_t>::value
       || std::is_same<T, vNx4accint_t>::value) ?
         _VDSP_NUM_8BIT_LANES
    :
     (  std::is_same<T, vNx2short_t>::value
       || std::is_same<T, vNx2int_t>::value
       || std::is_same<T, vNx2accshort_t>::value
       || std::is_same<T, vNx2accint_t>::value) ?
         _VDSP_NUM_16BIT_LANES
    :
     (  std::is_same<T, vNint_t>::value) ?
        _VDSP_NUM_32BIT_LANES
    : 0;
}

template <typename T>
MLI_FORCE_INLINE int get_number_lanes(T dummy) {
    return get_number_lanes<T>();
}

template <typename T, typename shift_T>
MLI_FORCE_INLINE T mli_math_asl_fx(T x, shift_T nbits);

template <typename T>
MLI_FORCE_INLINE T mli_math_limit_fx(T sign) {
    return sign < (T)0 ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
}

template <typename T, typename shift_T>
MLI_FORCE_INLINE T mli_math_asr_fx(T x, shift_T nbits) {
    shift_T nbits_max = sizeof(T) * 8 - 1;
    shift_T nbits_min = 0;
    if (nbits > nbits_max)
        return x < (T)0 ? -1 : 0;
    if (nbits < nbits_min)
        return mli_math_asl_fx<T, shift_T>(x, (-nbits));
    return x >> nbits;
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
MLI_FORCE_INLINE vNx4char_t mli_math_add_fx(vNx4char_t L, vNx4char_t R) {
    return vvadd_sat(L, R);
}

template <>
MLI_FORCE_INLINE vNx2short_t mli_math_add_fx(vNx2short_t L, vNx2short_t R) {
    return vvadd_sat(L, R);
}


template <>
MLI_FORCE_INLINE vNx4short_t mli_math_add_fx(vNx4short_t L, vNx4short_t R) {
    return vvadd_sat(L, R);
}

template <>
MLI_FORCE_INLINE vNx2int_t mli_math_add_fx(vNx2int_t L, vNx2int_t R) {
    return vvadd_sat(L, R);
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_add_fx(vNx4int_t L, vNx4int_t R) {
    return vvadd_sat(L, R);
}

template <> 
MLI_FORCE_INLINE vNx4short_t mli_math_sub_fx(vNx4short_t L, vNx4short_t R) {
    return vvsub_sat(L, R);
}

template <> 
MLI_FORCE_INLINE vNx4char_t mli_math_sub_fx(vNx4char_t L, vNx4char_t R) {
    return vvsub_sat(L, R);
}

template <> 
MLI_FORCE_INLINE vNx2short_t mli_math_sub_fx(vNx2short_t L, vNx2short_t R) {
    return vvsub_sat(L, R);
}

template <>
MLI_FORCE_INLINE vNx4accchar_t mli_math_add(vNx4accchar_t L, vNx4char_t R) {
    return vvcadd(L, R,(int8_t)0);
}

template <>
MLI_FORCE_INLINE vNx4accchar_t mli_math_add(vNx4accchar_t L, vNx4accchar_t R) {
    return vvcaddacc(L, R);
}

template <>
MLI_FORCE_INLINE vNx2accshort_t mli_math_add(vNx2accshort_t L, vNx2short_t R) {
    return vvcadd(L, R,(int16_t)0);
}

template <>
MLI_FORCE_INLINE vNx2accshort_t mli_math_add(vNx2accshort_t L, vNx2accshort_t R) {
    return vvcaddacc(L, R);
}
template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_add(vNx4accshort_t L, vNx4short_t R) {
    return __vacc_concat(vvcadd(__vacc_lo(L), R.lo,(int16_t)0), vvcadd(__vacc_hi(L), R.hi,(int16_t)0));
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_add(vNx4accshort_t L, vNx4accshort_t R) {
    return __vacc_concat(vvcaddacc(__vacc_lo(L), __vacc_lo(R)), vvcaddacc(__vacc_hi(L), __vacc_hi(R)));
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_add(vNx2accint_t L, vNx2int_t R) {
    return __vacc_concat(vvcadd(__vacc_lo(L), R.lo,(int32_t)0), vvcadd(__vacc_hi(L), R.hi,(int32_t)0));
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_add(vNx4accint_t L, vNx4int_t R) {
    vNx4accint_t r;
    r.lo = mli_math_add(L.lo, R.lo);
    r.hi = mli_math_add(L.hi, R.hi);
    return r;
}

template <>
MLI_FORCE_INLINE vNaccint_t mli_math_add(vNaccint_t L, vNaccint_t R) {
    return vvcaddacc(L, R);
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_add(vNx2accint_t L, vNx2accint_t R) {
    return __vacc_concat(vvcaddacc(__vacc_lo(L), __vacc_lo(R)), vvcaddacc(__vacc_hi(L), __vacc_hi(R)));
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_add(vNx4accint_t L, vNx4accint_t R) {
    vNx4accint_t r;
    r.lo = mli_math_add(L.lo, R.lo);
    r.hi = mli_math_add(L.hi, R.hi);
    return r;
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

template <>
MLI_FORCE_INLINE vNx4char_t mli_math_asr_fx(vNx4char_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    return (x >> nbits);
}

template <>
MLI_FORCE_INLINE vNx2short_t mli_math_asr_fx(vNx2short_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    return (x >> nbits);
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_asr_fx(vNx4short_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    vNx4short_t r;

    r.lo = mli_math_asr_fx(x.lo, nbits);
    r.hi = mli_math_asr_fx(x.hi, nbits);
    return r;
}

template <>
MLI_FORCE_INLINE vNx2short_t mli_math_asr_fx(vNx2short_t x, vNx2short_t nbits) {
    return (x >> nbits);
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_asr_fx(vNx4short_t x, vNx4short_t nbits) {
    vNx4short_t r;
    r.lo = mli_math_asr_fx(x.lo, nbits.lo);
    r.hi = mli_math_asr_fx(x.hi, nbits.hi);
    return r;
}

MLI_FORCE_INLINE vNint_t mli_math_asr_fx(vNint_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    return (x >> nbits);
}

template <>
MLI_FORCE_INLINE vNx2int_t mli_math_asr_fx(vNx2int_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    vNx2int_t r;

    r.lo = mli_math_asr_fx(x.lo, nbits);
    r.hi = mli_math_asr_fx(x.hi, nbits);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_asr_fx(vNx4int_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    vNx4int_t r;

    r.lo = mli_math_asr_fx(x.lo, nbits);
    r.hi = mli_math_asr_fx(x.hi, nbits);
    return r;
}

MLI_FORCE_INLINE vNint_t mli_math_asr_fx(vNint_t x, vNint_t nbits) {
    return (x >> nbits);
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
MLI_FORCE_INLINE vNx4accchar_t mli_math_asr_fx(vNx4accchar_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    return vvcasrm(x, nbits);
}

template <>
MLI_FORCE_INLINE vNx4accchar_t mli_math_asr_fx(vNx4accchar_t x, vNx4char_t nbits) {
    return vvcasrm(x, nbits);
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_asr_fx(vNx4accshort_t x, int nbits) {
    return __vacc_concat(vvcasrm(__vacc_lo(x), nbits), vvcasrm(__vacc_hi(x), nbits));
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_asr_fx(vNx4accshort_t x, vNx4short_t nbits) {
    return __vacc_concat(vvcasrm(__vacc_lo(x), nbits.lo), vvcasrm(__vacc_hi(x), nbits.hi));
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_asr_fx(vNx2accint_t x, vNx2int_t nbits) {
    return __vacc_concat(vvcasrm(__vacc_lo(x), nbits.lo), vvcasrm(__vacc_hi(x), nbits.hi));
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_asr_fx(vNx4accint_t x, vNx4int_t nbits) {
    vNx4accint_t r;
    r.lo = mli_math_asr_fx(x.lo, nbits.lo);
    r.hi = mli_math_asr_fx(x.hi, nbits.hi);
    return r;
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_asr_fx(vNx2accint_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    vNint_t nbits_v = nbits;

    return __vacc_concat(vvcasrm(__vacc_lo(x), nbits_v), vvcasrm(__vacc_hi(x), nbits_v));
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_asr_fx(vNx4accint_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    vNx4accint_t r;

    r.lo = mli_math_asr_fx(x.lo, nbits);
    r.hi = mli_math_asr_fx(x.hi, nbits);
    return r;
}

template <typename T, typename shift_T>
MLI_FORCE_INLINE T mli_math_asl_fx(T x, shift_T nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    shift_T inp_size = sizeof(T) * 8;
    T hi = 0;

    if (nbits > (inp_size - 1))
        return x != (T)0 ? mli_math_limit_fx<T>(x) : 0;

    hi = x >> ((inp_size - 1) - nbits);
    if (hi == (T)0 || hi == (T)-1)
        return x << nbits;

    return mli_math_limit_fx<T>(hi);
}

template <>
MLI_FORCE_INLINE vNx4char_t mli_math_asl_fx(vNx4char_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    return vvslm_sat(x, (vNx4char_t) nbits);
}

MLI_FORCE_INLINE vNx2short_t mli_math_asl_fx(vNx2short_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    return vvslm_sat(x, (vNx2short_t)nbits);
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_asl_fx(vNx4short_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    return vvslm_sat(x, nbits);
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_asl_fx(vNx4int_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    return vvslm_sat(x, nbits);
}

template <>
MLI_FORCE_INLINE vNx2int_t mli_math_asl_fx(vNx2int_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    return vvslm_sat(x, nbits);
}

template <>
MLI_FORCE_INLINE vNint_t mli_math_asl_fx(vNint_t x, vNint_t nbits) {
    return vvslm_sat(x, nbits);
}

template <>
MLI_FORCE_INLINE vNx2int_t mli_math_asl_fx(vNx2int_t x, vNx2int_t nbits) {
    vNx2int_t r;
    r.lo = mli_math_asl_fx(x.lo, nbits.lo);
    r.hi = mli_math_asl_fx(x.hi, nbits.hi);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_asl_fx(vNx4int_t x, vNx4int_t nbits) {
    vNx4int_t r;
    r.lo = mli_math_asl_fx(x.lo, nbits.lo);
    r.hi = mli_math_asl_fx(x.hi, nbits.hi);
    return r;
}

template <>
MLI_FORCE_INLINE vNx2short_t mli_math_asl_fx(vNx2short_t x, vNx2short_t nbits) {
    return vvslm_sat(x, nbits);
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_asl_fx(vNx4short_t x, vNx4short_t nbits) {
    vNx4short_t r;
    r.lo = mli_math_asl_fx(x.lo, nbits.lo);
    r.hi = mli_math_asl_fx(x.hi, nbits.hi);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4accchar_t mli_math_asl_fx(vNx4accchar_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    return vvcslm(x, (vNx4char_t) nbits);
}

template <>
MLI_FORCE_INLINE vNx4accchar_t mli_math_asl_fx(vNx4accchar_t x, vNx4char_t nbits) {
    return vvcslm(x, nbits);
}

template <>
MLI_FORCE_INLINE vNx2accshort_t mli_math_asl_fx(vNx2accshort_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    return vvcslm(x, (vNx2short_t) nbits);
}


template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_asl_fx(vNx4accshort_t x, vNx4short_t nbits) {
    return __vacc_concat(vvcslm(__vacc_lo(x), nbits.lo), vvcslm(__vacc_hi(x), nbits.hi));
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_asl_fx(vNx4accshort_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    vNx2short_t nbits_v = nbits;

    return __vacc_concat(vvcslm(__vacc_lo(x), nbits_v), vvcslm(__vacc_hi(x), nbits_v));
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_asl_fx(vNx2accint_t x, vNx2int_t nbits) {
    return __vacc_concat(vvcslm(__vacc_lo(x), nbits.lo), vvcslm(__vacc_hi(x), nbits.hi));
}
template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_asl_fx(vNx2accint_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    vNint_t nbits_v = nbits;

    return __vacc_concat(vvcslm(__vacc_lo(x), nbits_v), vvcslm(__vacc_hi(x), nbits_v));
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_asl_fx(vNx4accint_t x, vNx4int_t nbits) {
    vNx4accint_t r;
    r.lo = mli_math_asl_fx(x.lo, nbits.lo);
    r.hi = mli_math_asl_fx(x.hi, nbits.hi);
    return r;
}
template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_asl_fx(vNx4accint_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    vNx4accint_t r;

    r.lo = mli_math_asl_fx(x.lo, nbits);
    r.hi = mli_math_asl_fx(x.hi, nbits);
    return r;
}

template <typename T, typename shift_T>
MLI_FORCE_INLINE T mli_math_asr_rnd_fx(T x, shift_T nbits) {
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

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_asr_rnd_fx(vNx4accshort_t x, vNx4short_t nbits) {
    vNx4accshort_t r;
#ifdef ROUND_UP
    // adding 1 << (nbits-1)
    // shift twice to prevent negative shift if nbits = 0
    r = mli_math_add<vNx4accshort_t, vNx4short_t>(x, (vNx4short_t)(((vNx4ushort_t)1 << nbits) >> 1));
#endif
#ifdef ROUND_CONVERGENT
#error "Convergent rounding not supported"
#endif
    r = mli_math_asr_fx(r, nbits);
    return r;
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_asr_rnd_fx(vNx2accint_t x, vNx2int_t nbits) {
    vNx2accint_t r;
#ifdef ROUND_UP
    // adding 1 << (nbits-1)
    // shift twice to prevent negative shift if nbits = 0
    r = mli_math_add<vNx2accint_t, vNx2int_t>(x, (vNx2int_t)(((vNx2uint_t)1 << nbits) >> 1));
#endif
#ifdef ROUND_CONVERGENT
#error "Convergent rounding not supported"
#endif
    r = mli_math_asr_fx(r, nbits);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_asr_rnd_fx(vNx4accint_t x, vNx4int_t nbits) {
    vNx4accint_t r;
#ifdef ROUND_UP
    // adding 1 << (nbits-1)
    // shift twice to prevent negative shift if nbits = 0
    r = mli_math_add<vNx4accint_t, vNx4int_t>(x, (vNx4int_t)(((vNx4uint_t)1 << nbits) >> 1));
#endif
#ifdef ROUND_CONVERGENT
#error "Convergent rounding not supported"
#endif
    r = mli_math_asr_fx(r, nbits);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_asr_rnd_fx(vNx4accshort_t x, int nbits) {
    vNx4accshort_t r;
#ifdef ROUND_UP
    // adding 1 << (nbits-1)
    // shift twice to prevent negative shift if nbits = 0
    r = mli_math_add<vNx4accshort_t, vNx4short_t>(x, (vNx4short_t)(((vNx4ushort_t)1 << nbits) >> 1));
#endif
#ifdef ROUND_CONVERGENT
#error "Convergent rounding not supported"
#endif
    r = mli_math_asr_fx(r, nbits);
    return r;
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_asr_rnd_fx(vNx2accint_t x, int nbits) {
    vNx2accint_t r;
#ifdef ROUND_UP
    // adding 1 << (nbits-1)
    // shift twice to prevent negative shift if nbits = 0
    r = mli_math_add<vNx2accint_t, vNx2int_t>(x, (vNx2int_t)(((vNx2uint_t)1 << nbits) >> 1));
#endif
#ifdef ROUND_CONVERGENT
#error "Convergent rounding not supported"
#endif
    r = mli_math_asr_fx(r, nbits);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_asr_rnd_fx(vNx4accint_t x, int nbits) {
    vNx4accint_t r;
#ifdef ROUND_UP
    // adding 1 << (nbits-1)
    // shift twice to prevent negative shift if nbits = 0
    r = mli_math_add<vNx4accint_t, vNx4int_t>(x, (vNx4int_t)(((vNx4uint_t)1 << nbits) >> 1));
#endif
#ifdef ROUND_CONVERGENT
#error "Convergent rounding not supported"
#endif
    r = mli_math_asr_fx(r, nbits);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4char_t mli_math_asr_rnd_fx(vNx4char_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    vNx4char_t r;

#ifdef ROUND_UP
    vNx4char_t round = (vNx4char_t)(((vNx4uchar_t)1 << nbits) >> 1);
    r = mli_math_add_fx(x, round);
#endif
#ifdef ROUND_CONVERGENT
#error "Convergent rounding not supported"
#endif
    r = mli_math_asr_fx(r, nbits);
    return r;
 }

template <>
MLI_FORCE_INLINE vNx2short_t mli_math_asr_rnd_fx(vNx2short_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    vNx2short_t r;

#ifdef ROUND_UP
    vNx2short_t round = (vNx2short_t)(((vNx2ushort_t)1 << nbits) >> 1);
    r = mli_math_add_fx(x, round);
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
    // shift twice to prevent negative shift if nbits = 0
    r = mli_math_add_fx(x, (vNx4short_t)(((vNx4ushort_t)1 << nbits) >> 1));
#endif
#ifdef ROUND_CONVERGENT
#error "Convergent rounding not supported"
#endif
    r = mli_math_asr_fx(r, nbits);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_asr_rnd_fx(vNx4short_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    vNx4short_t r;

#ifdef ROUND_UP
    // shift twice to prevent negative shift if nbits = 0
    r = mli_math_add_fx(x, (vNx4short_t) (((vNx4ushort_t)1 << nbits) >> 1));
#endif
#ifdef ROUND_CONVERGENT
#error "Convergent rounding not supported"
#endif

    r = mli_math_asr_fx(r, nbits);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_asr_rnd_fx(vNx4int_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    vNx4int_t r;

#ifdef ROUND_UP
    r = mli_math_add_fx(x, (vNx4int_t) (((vNx4uint_t)1 << nbits) >> 1));
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
    // shift twice to prevent negative shift if nbits = 0
    r = mli_math_add_fx(x, (vNx4int_t)(((vNx4uint_t)1 << nbits) >> 1));
#endif
#ifdef ROUND_CONVERGENT
#error "Convergent rounding not supported"
#endif

    r = mli_math_asr_fx(r, nbits);
    return r;
}

template <>
MLI_FORCE_INLINE vNx2int_t mli_math_asr_rnd_fx(vNx2int_t x, int nbits) {
    MLI_EXTRA_ASSERT(nbits >= 0);
    vNx2int_t r;

#ifdef ROUND_UP
    // shift twice to prevent negative shift if nbits = 0
    r = mli_math_add_fx(x, (vNx2int_t)(((vNx2uint_t)1 << nbits) >> 1));
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

template<>
MLI_FORCE_INLINE vNx2short_t mli_math_abs_fx(vNx2short_t x) {
    return vvabs_sat(x);
}

template<>
MLI_FORCE_INLINE vNx4short_t mli_math_abs_fx(vNx4short_t x) {
    vNx4short_t res;
    res.lo = mli_math_abs_fx(x.lo);
    res.hi = mli_math_abs_fx(x.hi);
    return res;
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

// todo: remove bias related things from mli math
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

template <typename in_T, typename acc_T>
MLI_FORCE_INLINE acc_T mli_math_init_accu(in_T val);

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_init_accu(int16_t val) {
    return __vacc_concat(vvcadd_init((vNx2short_t)val,(int16_t)0), vvcadd_init((vNx2short_t)0,(int16_t)val));
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_init_accu(vNx4short_t val) {
    return __vacc_concat(vvcadd_init(val.lo,(int16_t)0), vvcadd_init(val.hi,(int16_t)0));
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_init_accu(int32_t val) {
    vNx2accint_t acc;
    acc = __vacc_concat(vvcadd_init((vNint_t)val,(int32_t)0), vvcadd_init((vNint_t)val,(int32_t)0));
    return acc;
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_init_accu(int32_t val) {
    vNx4accint_t acc;
    acc.lo = __vacc_concat(vvcadd_init((vNint_t)val,(int32_t)0), vvcadd_init((vNint_t)val,(int32_t)0));
    acc.hi = __vacc_concat(vvcadd_init((vNint_t)val,(int32_t)0), vvcadd_init((vNint_t)val,(int32_t)0));
    return acc;
}

// Cast value to output type (including accumulator type)
//========================================================================
template <>
MLI_FORCE_INLINE int8_t mli_math_cast_fx(int16_t in_val, int shift_right) {
    return (int8_t)mli_math_sat_fx<int16_t>(mli_math_asr_rnd_fx<int16_t, int>(in_val, shift_right), 8);
}

template <>
MLI_FORCE_INLINE int8_t mli_math_cast_fx(int8_t in_val, int shift_right) {
    return (int8_t)mli_math_asr_rnd_fx<int8_t, int>((int8_t)in_val, shift_right);
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
    return (int16_t)mli_math_sat_fx<int32_t>(in_val, 16);
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
MLI_FORCE_INLINE int8_t mli_math_cast_fx(int64_t in_val, int shift_right) {
    /* This function is taken from reference mli_math as is */
    int64_t temp = (int64_t)mli_math_asr_rnd_fx<int64_t>((int64_t)in_val, shift_right);
    return (int8_t)mli_math_sat_fx<int64_t>(temp, 56);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(int64_t in_val, int shift_right) {
    int64_t temp = mli_math_asr_rnd_fx<int64_t, int>(in_val, shift_right);
    return (int16_t)mli_math_sat_fx<int64_t>(temp, 48);
}

template <>
MLI_FORCE_INLINE int16_t mli_math_cast_fx(int64_t in_val) {
    return (int16_t)mli_math_sat_fx<int64_t>(in_val, 48);
}

template <>
MLI_FORCE_INLINE int32_t mli_math_cast_fx(int64_t in_val, int shift_right) {
    in_val = mli_math_asr_rnd_fx<int64_t, int>(in_val, shift_right);
    return (int32_t)mli_math_sat_fx<int64_t>(in_val, 32);
}

template <>
MLI_FORCE_INLINE float mli_math_cast_fx(int32_t in_val, int shift_right) {
    /* This function is taken from reference mli_math as is */
    return (float)in_val / (float)(1 << shift_right);
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_cast_fx(vNx4char_t in_val) {
    return to_vNx4short_t(vvcmpy(in_val, 1));
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_cast_fx(vNx4char_t in_val) {
    return to_vNx4int_t(in_val);
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_cast_fx(vNx4short_t in_val) {
    return to_vNx4int_t(in_val);
}

template <>
MLI_FORCE_INLINE vNx4char_t mli_math_cast_fx(vNx4short_t in_val) {
    in_val = mli_math_bound_range_fx(in_val, INT8_MIN, INT8_MAX);
    return to_vNx4char_t(in_val);
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_cast_fx(vNx4int_t in_val) {
    in_val = mli_math_bound_range_fx(in_val, INT16_MIN, INT16_MAX);
    return to_vNx4short_t(in_val);
}

template <>
MLI_FORCE_INLINE vNx2short_t mli_math_cast_fx(vNx2int_t in_val) {
    in_val = mli_math_bound_range_fx(in_val, INT16_MIN, INT16_MAX);
    return to_vNx2short_t(in_val);
}

template <>
MLI_FORCE_INLINE vNx4char_t mli_math_cast_fx(vNx4int_t in_val) {
    return to_vNx4char_t(mli_math_bound_range_fx(in_val, INT8_MIN, INT8_MAX));
}

template <>
MLI_FORCE_INLINE vNx4char_t mli_math_cast_fx(vNx4int_t in_val, int shift_right) {
    vNx4int_t r;
#ifdef ROUND_UP
    vNx4int_t round = (vNx4int_t)(((vNx4uint_t)1 << shift_right) >> 1);
    r = mli_math_add_fx(in_val, round);
#else
    #error Rounding mode not supported
#endif

    r = mli_math_asr_fx(r, shift_right);
    return to_vNx4char_t(mli_math_bound_range_fx(r, INT8_MIN, INT8_MAX));
}

template <>
MLI_FORCE_INLINE vNx2short_t mli_math_cast_fx(vNx2int_t in_val, int shift_right) {
    vNx2int_t r;
#ifdef ROUND_UP
    vNx2int_t round = (vNx2int_t)(((vNx2uint_t)1 << shift_right) >> 1);
    r = mli_math_add_fx(in_val, round);
#else
        #error Rounding mode not supported
#endif
    r = mli_math_asr_fx(r, shift_right);
    r = mli_math_bound_range_fx(r, INT16_MIN, INT16_MAX);

    return to_vNx2short_t(r);
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_cast_fx(vNx4int_t in_val, int shift_right) {
    vNx4short_t r;
    r.lo = mli_math_cast_fx<vNx2int_t, vNx2short_t>(in_val.lo, shift_right);
    r.hi = mli_math_cast_fx<vNx2int_t, vNx2short_t>(in_val.hi, shift_right);
    return r;
}

template<>
MLI_FORCE_INLINE vNx4short_t mli_math_cast_fx(vNx4short_t in_val, int shift_right) {
    /* Shift, round and Sat */
    MLI_EXTRA_ASSERT(shift_right >= 0);
    vNx4short_t acc;

#ifdef ROUND_UP
    // Rounding up:
    vNx4short_t round = (vNx4short_t)(((vNx4ushort_t)1 << shift_right) >> 1);
    acc = mli_math_add_fx(in_val, round);
#else
    #error Rounding mode not supported
#endif

    acc = mli_math_asr_fx(acc, shift_right);
    return acc;
}

template<>
MLI_FORCE_INLINE vNx4char_t mli_math_cast_fx<vNx4short_t, vNx4char_t, false >(vNx4short_t in_val, int shift_right) {
    MLI_EXTRA_ASSERT(shift_right >= 0);
    vNx4short_t acc = in_val;
    acc = mli_math_asr_fx(acc, shift_right);
    acc = mli_math_bound_range_fx(acc, INT8_MIN, INT8_MAX);
    return to_vNx4char_t(acc);
}

template<>
MLI_FORCE_INLINE vNx4char_t mli_math_cast_fx(vNx4short_t in_val, int shift_right) {
    MLI_EXTRA_ASSERT(shift_right >= 0);
    vNx4short_t acc;

#ifdef ROUND_UP
    // Rounding up
    vNx4short_t round = (vNx4short_t)(((vNx4ushort_t)1 << shift_right) >> 1);
    acc = mli_math_add_fx(in_val, round);
#else
    #error Rounding mode not supported
#endif

    acc = mli_math_asr_fx(acc, shift_right);
    acc = mli_math_bound_range_fx(acc, INT8_MIN, INT8_MAX);
    return to_vNx4char_t(acc);
}

// Multiply and cast float to accum
//========================================================================

MLI_FORCE_INLINE int32_t mli_math_float_scale(float value, float scale) {
    /* This function is taken from reference mli_math as is */
    const float round_val = value > 0 ? 0.5f : -0.5f;
    return (int32_t)(value * scale + round_val);
}

// Cast accum to output type
//========================================================================
MLI_FORCE_INLINE vNx4char_t mli_math_acc_cast(vNx4accchar_t acc) {
    return to_vNx4char_t(acc);
}

MLI_FORCE_INLINE vNx2short_t mli_math_acc_cast(vNx2accshort_t acc) {
    return to_vNx2short_t(acc);
}

MLI_FORCE_INLINE vNx4short_t mli_math_acc_cast(vNx4accshort_t acc) {
    return to_vNx4short_t(acc);
}

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
MLI_FORCE_INLINE vNx4char_t mli_math_acc_cast_fx(vNx4accchar_t acc) {
    int ctrlword = SAT|SIGNED|TARGET_SZ_8|SHIFT(0);
    vNx4char_t accu_result;
    accu_result = to_vNx4char_t(vvconvert(acc, ctrlword));

    return accu_result;
}

template <>
MLI_FORCE_INLINE vNx2short_t mli_math_acc_cast_fx(vNx2accshort_t acc) {
    int ctrlword = SAT|SIGNED|TARGET_SZ_16|SHIFT(0);
    vNx2short_t accu_result;
    accu_result = to_vNx2short_t(vvconvert(acc, ctrlword));

    return accu_result;
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_acc_cast_fx(vNx4accshort_t acc) {
    int ctrlword = SAT|SIGNED|TARGET_SZ_16|SHIFT(0);
    vNx4short_t accu_result;
    accu_result.lo = to_vNx2short_t(vvconvert(__vacc_lo(acc), ctrlword));
    accu_result.hi = to_vNx2short_t(vvconvert(__vacc_hi(acc), ctrlword));

    return accu_result;
}

template <>
MLI_FORCE_INLINE vNint_t mli_math_acc_cast_fx(vNaccint_t acc) {
    int ctrlword = SAT|SIGNED|TARGET_SZ_32|SHIFT(0);
    vNint_t accu_result;
    accu_result = to_vNint_t(vvconvert(acc, ctrlword));

    return accu_result;
}

template <>
MLI_FORCE_INLINE vNx2int_t mli_math_acc_cast_fx(vNx2accint_t acc) {
    int ctrlword = SAT|SIGNED|TARGET_SZ_32|SHIFT(0);
    vNx2int_t accu_result;
    accu_result.lo = to_vNint_t(vvconvert(__vacc_lo(acc), ctrlword));
    accu_result.hi = to_vNint_t(vvconvert(__vacc_hi(acc), ctrlword));

    return accu_result;
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_acc_cast_fx(vNx4accint_t acc) {
    int ctrlword = SAT|SIGNED|TARGET_SZ_32|SHIFT(0);
    vNx4int_t accu_result;
    accu_result.lo.lo = to_vNint_t(vvconvert(__vacc_lo(acc.lo), ctrlword));
    accu_result.lo.hi = to_vNint_t(vvconvert(__vacc_hi(acc.lo), ctrlword));
    accu_result.hi.lo = to_vNint_t(vvconvert(__vacc_lo(acc.hi), ctrlword));
    accu_result.hi.hi = to_vNint_t(vvconvert(__vacc_hi(acc.hi), ctrlword));

    return accu_result;
}

template <>
MLI_FORCE_INLINE vNx2short_t mli_math_acc_cast_fx(vNx2accint_t acc) {
    int ctrlword = SAT|SIGNED|TARGET_SZ_16|SHIFT(0);
    vNx2int_t accu_result;
    accu_result.lo = to_vNint_t(vvconvert(__vacc_lo(acc), ctrlword));
    accu_result.hi = to_vNint_t(vvconvert(__vacc_hi(acc), ctrlword));

    return to_vNx2short_t(accu_result);
}

template <>
MLI_FORCE_INLINE vNx4short_t mli_math_acc_cast_fx(vNx4accint_t acc) {
    int ctrlword = SAT|SIGNED|TARGET_SZ_16|SHIFT(0);
    vNx4int_t accu_result;
    accu_result.lo.lo = to_vNint_t(vvconvert(__vacc_lo(acc.lo), ctrlword));
    accu_result.lo.hi = to_vNint_t(vvconvert(__vacc_hi(acc.lo), ctrlword));
    accu_result.hi.lo = to_vNint_t(vvconvert(__vacc_lo(acc.hi), ctrlword));
    accu_result.hi.hi = to_vNint_t(vvconvert(__vacc_hi(acc.hi), ctrlword));

    return to_vNx4short_t(accu_result);
}

template <>
MLI_FORCE_INLINE vNx4char_t mli_math_acc_cast_fx(vNx4accint_t acc) {
    int ctrlword = SAT|SIGNED|TARGET_SZ_8|SHIFT(0);
    vNx4int_t accu_result;
    accu_result.lo.lo = to_vNint_t(vvconvert(__vacc_lo(acc.lo), ctrlword));
    accu_result.lo.hi = to_vNint_t(vvconvert(__vacc_hi(acc.lo), ctrlword));
    accu_result.hi.lo = to_vNint_t(vvconvert(__vacc_lo(acc.hi), ctrlword));
    accu_result.hi.hi = to_vNint_t(vvconvert(__vacc_hi(acc.hi), ctrlword));

    return to_vNx4char_t(accu_result);
}

template <>
MLI_FORCE_INLINE vNx4char_t mli_math_acc_cast_fx(vNx4accshort_t acc) {
    int ctrlword = SAT|SIGNED|TARGET_SZ_8|SHIFT(0);
    vNx4short_t accu_result;
    accu_result.lo = to_vNx2short_t(vvconvert(__vacc_lo(acc), ctrlword));
    accu_result.hi = to_vNx2short_t(vvconvert(__vacc_hi(acc), ctrlword));

    return to_vNx4char_t(accu_result);
}

template<>
MLI_FORCE_INLINE vNx4short_t mli_math_acc_cast_fx(vNx4accint_t acc, int shift_right) {
    MLI_EXTRA_ASSERT(shift_right >= 0);

#ifdef ROUND_UP
    vNx4int_t round = (vNx4int_t)(((vNx4uint_t)1 << shift_right) >> 1);
    acc = mli_math_add<vNx4accint_t, vNx4int_t>(acc, round);
#else
    #error Rounding mode not supported
#endif

    int ctrlword = SAT|SIGNED|TARGET_SZ_16|SHIFT(shift_right);
    vNx4int_t accu_result;
    accu_result.lo.lo = to_vNint_t(vvconvert(__vacc_lo(acc.lo), ctrlword));
    accu_result.lo.hi = to_vNint_t(vvconvert(__vacc_hi(acc.lo), ctrlword));
    accu_result.hi.lo = to_vNint_t(vvconvert(__vacc_lo(acc.hi), ctrlword));
    accu_result.hi.hi = to_vNint_t(vvconvert(__vacc_hi(acc.hi), ctrlword));

    return to_vNx4short_t(accu_result);
}

template<>
MLI_FORCE_INLINE vNx4char_t mli_math_acc_cast_fx(vNx4accchar_t acc, int shift_right) {
    MLI_EXTRA_ASSERT(shift_right >= 0);

#ifdef ROUND_UP
    vNx4char_t round = (vNx4char_t)(((vNx4uchar_t)1 << shift_right) >> 1);
    acc = mli_math_add<vNx4accchar_t, vNx4char_t>(acc, round);
#else
    #error Rounding mode not supported
#endif

    int ctrlword = SAT|SIGNED|TARGET_SZ_8|SHIFT(shift_right);
    vNx4char_t accu_result;
    accu_result = to_vNx4char_t(vvconvert(acc, ctrlword));

    return accu_result;
}

template<>
MLI_FORCE_INLINE vNx2short_t mli_math_acc_cast_fx(vNx2accshort_t acc, int shift_right) {
    MLI_EXTRA_ASSERT(shift_right >= 0);

#ifdef ROUND_UP
    vNx2short_t round = (vNx2short_t)(((vNx2ushort_t)1 << shift_right) >> 1);
    acc = mli_math_add<vNx2accshort_t, vNx2short_t>(acc, round);
#else
    #error Rounding mode not supported
#endif

    int ctrlword = SAT|SIGNED|TARGET_SZ_16|SHIFT(shift_right);
    vNx2short_t accu_result;
    accu_result = to_vNx2short_t(vvconvert(acc, ctrlword));

    return accu_result;
}

template<>
MLI_FORCE_INLINE vNx4short_t mli_math_acc_cast_fx(vNx4accshort_t acc, int shift_right) {
    MLI_EXTRA_ASSERT(shift_right >= 0);

#ifdef ROUND_UP
    vNx4short_t round = (vNx4short_t)(((vNx4ushort_t)1 << shift_right) >> 1);
    acc = mli_math_add<vNx4accshort_t, vNx4short_t>(acc, round);
#else
    #error Rounding mode not supported
#endif

    int ctrlword = SAT|SIGNED|TARGET_SZ_16|SHIFT(shift_right);
    vNx4short_t accu_result;
    accu_result.lo = to_vNx2short_t(vvconvert(__vacc_lo(acc), ctrlword));
    accu_result.hi = to_vNx2short_t(vvconvert(__vacc_hi(acc), ctrlword));

    return accu_result;
}

template<>
MLI_FORCE_INLINE vNx2int_t mli_math_acc_cast_fx(vNx2accint_t acc, int shift_right) {
    MLI_EXTRA_ASSERT(shift_right >= 0);

#ifdef ROUND_UP
    vNx2int_t round = (vNx2int_t)(((vNx2uint_t)1 << shift_right) >> 1);
    acc = mli_math_add<vNx2accint_t, vNx2int_t>(acc, round);
#else
    #error Rounding mode not supported
#endif

    int ctrlword = SAT|SIGNED|TARGET_SZ_32|SHIFT(shift_right);
    vNx2int_t accu_result;
    accu_result.lo = to_vNint_t(vvconvert(__vacc_lo(acc), ctrlword));
    accu_result.hi = to_vNint_t(vvconvert(__vacc_hi(acc), ctrlword));

    return accu_result;
}

template<>
MLI_FORCE_INLINE vNx2short_t mli_math_acc_cast_fx(vNx2accint_t acc, int shift_right) {
    MLI_EXTRA_ASSERT(shift_right >= 0);

#ifdef ROUND_UP
    vNx2int_t round = (vNx2int_t)(((vNx2uint_t)1 << shift_right) >> 1);
    acc = mli_math_add<vNx2accint_t, vNx2int_t>(acc, round);
#else
    #error Rounding mode not supported
#endif

    int ctrlword = SAT|SIGNED|TARGET_SZ_16|SHIFT(shift_right);
    vNx2int_t accu_result;
    accu_result.lo = to_vNint_t(vvconvert(__vacc_lo(acc), ctrlword));
    accu_result.hi = to_vNint_t(vvconvert(__vacc_hi(acc), ctrlword));

    return to_vNx2short_t(accu_result);

}

template<>
MLI_FORCE_INLINE vNx4int_t mli_math_acc_cast_fx(vNx4accint_t acc, int shift_right) {
    MLI_EXTRA_ASSERT(shift_right >= 0);

#ifdef ROUND_UP
    vNx4int_t round = (vNx4int_t)(((vNx4uint_t)1 << shift_right) >> 1);
    acc = mli_math_add<vNx4accint_t, vNx4int_t>(acc, round);
#else
    #error Rounding mode not supported
#endif

    int ctrlword = SAT|SIGNED|TARGET_SZ_32|SHIFT(shift_right);
    vNx4int_t accu_result;
    accu_result.lo.lo = to_vNint_t(vvconvert(__vacc_lo(acc.lo), ctrlword));
    accu_result.lo.hi = to_vNint_t(vvconvert(__vacc_hi(acc.lo), ctrlword));
    accu_result.hi.lo = to_vNint_t(vvconvert(__vacc_lo(acc.hi), ctrlword));
    accu_result.hi.hi = to_vNint_t(vvconvert(__vacc_hi(acc.hi), ctrlword));

    return accu_result;
}

template<>
MLI_FORCE_INLINE vNx4char_t mli_math_acc_cast_fx(vNx4accint_t acc, int shift_right) {
    MLI_EXTRA_ASSERT(shift_right >= 0);

#ifdef ROUND_UP
    vNx4int_t round = (vNx4int_t)(((vNx4uint_t)1 << shift_right) >> 1);
    acc = mli_math_add<vNx4accint_t, vNx4int_t>(acc, round);
#else
    #error Rounding mode not supported
#endif

    int ctrlword = SAT|SIGNED|TARGET_SZ_8|SHIFT(shift_right);
    vNx4int_t accu_result;
    accu_result.lo.lo = to_vNint_t(vvconvert(__vacc_lo(acc.lo), ctrlword));
    accu_result.lo.hi = to_vNint_t(vvconvert(__vacc_hi(acc.lo), ctrlword));

    accu_result.hi.lo = to_vNint_t(vvconvert(__vacc_lo(acc.hi), ctrlword));
    accu_result.hi.hi = to_vNint_t(vvconvert(__vacc_hi(acc.hi), ctrlword));

    return to_vNx4char_t(accu_result);
}

template<>
MLI_FORCE_INLINE vNx4short_t mli_math_acc_cast_fx<vNx4short_t, vNx4accshort_t,/*round = */ false>(
        vNx4accshort_t acc, int shift_right) {
    MLI_EXTRA_ASSERT(shift_right >= 0);

    int ctrlword = SAT|SIGNED|TARGET_SZ_16|SHIFT(shift_right);
    vNx4short_t accu_result;
    accu_result.lo = to_vNx2short_t(vvconvert(__vacc_lo(acc), ctrlword));
    accu_result.hi = to_vNx2short_t(vvconvert(__vacc_hi(acc), ctrlword));

    return accu_result;
}


template<>
MLI_FORCE_INLINE vNx4char_t mli_math_acc_cast_fx<vNx4char_t, vNx4accint_t,/*round = */ false>(
        vNx4accint_t acc, int shift_right) {
    int ctrlword = SAT|SIGNED|TARGET_SZ_8|SHIFT(shift_right);
    vNx4int_t accu_result;

    accu_result.lo.lo = to_vNint_t(vvconvert(__vacc_lo(acc.lo), ctrlword));
    accu_result.lo.hi = to_vNint_t(vvconvert(__vacc_hi(acc.lo), ctrlword));

    accu_result.hi.lo = to_vNint_t(vvconvert(__vacc_lo(acc.hi), ctrlword));
    accu_result.hi.hi = to_vNint_t(vvconvert(__vacc_hi(acc.hi), ctrlword));

    return to_vNx4char_t(accu_result);
}

template<>
MLI_FORCE_INLINE vNx4int_t mli_math_acc_cast_fx<vNx4int_t, vNx4accint_t,/*round = */ false>(vNx4accint_t acc, int shift_right) {
    MLI_EXTRA_ASSERT(shift_right >= 0);

    int ctrlword = SAT|SIGNED|TARGET_SZ_32|SHIFT(shift_right);
    vNx4int_t accu_result;
    accu_result.lo.lo = to_vNint_t(vvconvert(__vacc_lo(acc.lo), ctrlword));
    accu_result.lo.hi = to_vNint_t(vvconvert(__vacc_hi(acc.lo), ctrlword));
    accu_result.hi.lo = to_vNint_t(vvconvert(__vacc_lo(acc.hi), ctrlword));
    accu_result.hi.hi = to_vNint_t(vvconvert(__vacc_hi(acc.hi), ctrlword));

    return accu_result;
}

template<>
MLI_FORCE_INLINE vNx4char_t mli_math_acc_cast_fx<vNx4char_t, vNx4accshort_t,/*round = */ false>(
        vNx4accshort_t acc, int shift_right) {
    MLI_EXTRA_ASSERT(shift_right >= 0);

    int ctrlword = SAT|SIGNED|TARGET_SZ_8|SHIFT(shift_right);
    vNx4short_t accu_result;
    accu_result.lo = to_vNx2short_t(vvconvert(__vacc_lo(acc), ctrlword));
    accu_result.hi = to_vNx2short_t(vvconvert(__vacc_hi(acc), ctrlword));

    return to_vNx4char_t(accu_result);
}

template<>
MLI_FORCE_INLINE vNx2short_t mli_math_acc_cast_fx<vNx2short_t, vNx2accint_t,/*round = */ false>(
        vNx2accint_t acc, int shift_right) {
    MLI_EXTRA_ASSERT(shift_right >= 0);

    int ctrlword = SAT|SIGNED|TARGET_SZ_16|SHIFT(shift_right);
    vNx2int_t accu_result;
    accu_result.lo = to_vNint_t(vvconvert(__vacc_lo(acc), ctrlword));
    accu_result.hi = to_vNint_t(vvconvert(__vacc_hi(acc), ctrlword));

    return to_vNx2short_t(accu_result);
}

template<>
MLI_FORCE_INLINE vNx4char_t mli_math_acc_cast_fx(vNx4accshort_t acc, int shift_right) {
    MLI_EXTRA_ASSERT(shift_right >= 0);

#ifdef ROUND_UP
    vNx4short_t round = (vNx4short_t)(((vNx4ushort_t)1 << shift_right) >> 1);
    acc = mli_math_add<vNx4accshort_t, vNx4short_t>(acc, round);
#else
    #error Rounding mode not supported
#endif

    int ctrlword = SAT|SIGNED|TARGET_SZ_8|SHIFT(shift_right);
    vNx4short_t accu_result;
    accu_result.lo = to_vNx2short_t(vvconvert(__vacc_lo(acc), ctrlword));
    accu_result.hi = to_vNx2short_t(vvconvert(__vacc_hi(acc), ctrlword));

    return to_vNx4char_t(accu_result);
}

// Norm
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
MLI_FORCE_INLINE vNx4short_t mli_math_norm_fx(vNx4short_t x) {
    vNx4short_t r;
    r.lo = vvnorm(x.lo);
    r.hi = vvnorm(x.hi);
    return r;
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

template <>
MLI_FORCE_INLINE vNx2int_t mli_math_norm_fx(vNx2accint_t x) {
    vNx2int_t r;
    r.lo = vvcnorm (__vacc_lo(x));
    r.hi = vvcnorm (__vacc_hi(x));
    return r;
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_norm_fx(vNx4accint_t x) {
    vNx4int_t r;
    r.lo = mli_math_norm_fx<vNx2accint_t, vNx2int_t>(x.lo);
    r.hi = mli_math_norm_fx<vNx2accint_t, vNx2int_t>(x.hi);
    return r;
}

template<typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_math_norm_cast_fx(in_T val , int32_t *norm_shift) {
    int32_t cast_shift = (sizeof(in_T) - sizeof(out_T)) * 8;
    int32_t norm = mli_math_norm_fx<in_T, int32_t>(val);
    *norm_shift = cast_shift - norm;
    return mli_math_cast_fx<in_T, out_T>(val, *norm_shift);
}

template<bool left_shift = true>
MLI_FORCE_INLINE vNx4short_t mli_math_norm_cast_fx(vNx4int_t val , vNx4int_t *norm_shift) {
    int cast_shift = 16; // Casting from int to short.
    vNx4int_t norm_val = cast_shift - mli_math_norm_fx<vNx4int_t, vNx4int_t>(val);
    vNx4int_t shift_right = mli_math_max_fx(norm_val, 0);
    val = mli_math_asr_rnd_fx(val , shift_right);
    if (left_shift) {
        vNx4int_t shift_left = mli_math_max_fx(-norm_val, 0);
        val = mli_math_asl_fx(val, shift_left);
        *norm_shift = norm_val;
    } else {
        *norm_shift = shift_right;
    }
    return mli_math_cast_fx<vNx4int_t, vNx4short_t>(val);
}

//This function works only on acc_T which supported by vvc4add and vvc4pack
template<typename acc_T, typename vec_T, typename out_T>
MLI_FORCE_INLINE out_T mli_math_intra_sum(acc_T L) {
    int acc_len = get_number_lanes<vec_T>();
    MLI_ASSERT(acc_len >= 4);
    while (acc_len > 8){
        L = vvc4add(L);
        L = vvc4pack(L);
        acc_len >>= 2;
    }

    L = vvc4add(L);
    acc_len >>= 2;
    if (acc_len == 2){
        L = vvc4pack(L);
        L = vvc2add(L);
    }
    vec_T vec = mli_math_acc_cast_fx<vec_T, acc_T>(L);
    return (out_T) vec[0];
}

MLI_FORCE_INLINE int32_t mli_math_intra_sum(vNx2accint_t L) {
    int32_t sum_acc = mli_math_intra_sum<vNaccint_t, vNint_t, int32_t>(vvcaddacc(__vacc_lo(L), __vacc_hi(L)));
    return sum_acc;
}

MLI_FORCE_INLINE int32_t mli_math_intra_sum(vNx4accint_t L) {
    int32_t sum_acc = mli_math_intra_sum(mli_math_add(L.lo, L.hi));
    return sum_acc;
}

MLI_FORCE_INLINE int32_t mli_math_intra_sum(vNx4accshort_t L) {
    int32_t sum_acc = mli_math_intra_sum<vNx2accshort_t, vNx2short_t, int32_t>(mli_math_add(__vacc_lo(L), __vacc_hi(L)));
    return sum_acc;
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

MLI_FORCE_INLINE vNx4accchar_t mli_math_max_fx(vNx4accchar_t L, vNx4char_t R) {
    return vvcmax(L, R);
}

MLI_FORCE_INLINE vNx2accshort_t mli_math_max_fx(vNx2accshort_t L, vNx2short_t R) {
    return vvcmax(L, R);
}

//This function works only on acc_T which supported by vvc2max and vvceven
template<typename acc_T, typename vec_T, typename out_T>
MLI_FORCE_INLINE out_T mli_math_intra_max(acc_T L) {
    int acc_len = get_number_lanes<vec_T>();
    while (acc_len > 2){
        L = vvc2max(L);
        L = vvceven(L);
        acc_len >>= 1;
    }
    L = vvc2max(L);
    vec_T vec = mli_math_acc_cast_fx<vec_T, acc_T>(L);
    return (out_T) vec[0];
}

MLI_FORCE_INLINE int8_t mli_math_intra_max(vNx4char_t L) {
    vNx4accchar_t acc = vvcadd_init(L, (vNx4char_t) 0);
    int8_t max_val = mli_math_intra_max<vNx4accchar_t, vNx4char_t, int8_t>(acc);
    return max_val;
}

MLI_FORCE_INLINE int16_t mli_math_intra_max(vNx4short_t L) {
    vNx2short_t half_acc = mli_math_max_fx(L.lo, L.hi);
    vNx2accshort_t acc = vvcadd_init(half_acc, (vNx2short_t) 0);
    int16_t max_val = mli_math_intra_max<vNx2accshort_t, vNx2short_t, int16_t>(acc);
    return max_val;
}

MLI_FORCE_INLINE int16_t mli_math_intra_max(vNx2short_t L) {
    vNx2accshort_t acc = vvcadd_init(L, (vNx2short_t) 0);
    int16_t max_val = mli_math_intra_max<vNx2accshort_t, vNx2short_t, int16_t>(acc);
    return max_val;
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

//This function works only on acc_T which supported by vvc2max and vvceven
template<typename acc_T, typename vec_T, typename out_T>
MLI_FORCE_INLINE out_T mli_math_intra_min(acc_T L) {
    int acc_len = get_number_lanes<vec_T>();
    while (acc_len > 2){
        L = vvc2min(L);
        L = vvceven(L);
        acc_len >>= 1;
    }
    L = vvc2min(L);
    vec_T vec = mli_math_acc_cast_fx<vec_T, acc_T>(L);
    return (out_T) vec[0];
}

MLI_FORCE_INLINE int8_t mli_math_intra_min(vNx4char_t L) {
    vNx4accchar_t acc = vvcadd_init(L, (vNx4char_t) 0);
    int8_t min_val = mli_math_intra_min<vNx4accchar_t, vNx4char_t, int8_t>(acc);
    return min_val;
}

MLI_FORCE_INLINE int16_t mli_math_intra_min(vNx4short_t L) {
    vNx2short_t half_acc = mli_math_min_fx(L.lo, L.hi);
    vNx2accshort_t acc = vvcadd_init(half_acc, (vNx2short_t) 0);
    int16_t min_val = mli_math_intra_min<vNx2accshort_t, vNx2short_t, int16_t>(acc);
    return min_val;
}

MLI_FORCE_INLINE int16_t mli_math_intra_min(vNx2short_t L) {
    vNx2accshort_t acc = vvcadd_init(L, (vNx2short_t) 0);
    int16_t min_val = mli_math_intra_min<vNx2accshort_t, vNx2short_t, int16_t>(acc);
    return min_val;
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


MLI_FORCE_INLINE vNaccint_t mli_math_mul_fx_low(vNint_t L, vNint_t R) {
    vNaccint_t r;
    r = vvcmpy_lo(L, R);
    return r;
}

MLI_FORCE_INLINE vNx2accint_t mli_math_mul_fx_low(vNx2int_t L, vNx2int_t R) {
    vNx2accint_t r;
    r = __vacc_concat(vvcmpy_lo(L.lo, R.lo), vvcmpy_lo(L.hi, R.hi));
    return r;
}

MLI_FORCE_INLINE vNx4accint_t mli_math_mul_fx_low(vNx4int_t L, vNx4int_t R) {
    vNx4accint_t r;
    r.lo = mli_math_mul_fx_low(L.lo, R.lo);
    r.hi = mli_math_mul_fx_low(L.hi, R.hi);
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
MLI_FORCE_INLINE vNx4accint_t mli_math_mul_fx(int16_t L, int16_t R) {
    return vvcmpy((vNx4short_t)L, R);
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_mul_fx(vNx4short_t L, vNx4short_t R) {
    return vvcmpy(L, R);
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_mul_fx(int8_t L, int8_t R) {
    return vvcmpy((vNx4char_t)L, R);
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_mul_fx(int16_t L, int16_t R) {
    return vvcmpy((vNx2short_t)L, R);
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_mul_fx(vNx2short_t L, vNx2short_t R) {
    return vvcmpy(L, R);
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_mul_fx(vNx4char_t L, vNx4char_t R) {
    return vvcmpy(L, R);
}

template <>
MLI_FORCE_INLINE vNx4int_t mli_math_mul_fx(vNx4short_t L, vNx4short_t R) {
    return to_vNx4int_t(vvcmpy(L, R));
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

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_mac_fx(vNx4accshort_t acc, vNx4char_t L, int8_t R) {
    return vvcmac(acc, L, R);
}
template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_mac_fx(vNx4accshort_t acc, vNx4char_t L, vNx4char_t R) {
    return vvcmac(acc, L, R);
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_mac_fx(vNx2accint_t acc, vNx2short_t L, int16_t R) {
    return vvcmac(acc, L, R);
}
template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_mac_fx(vNx2accint_t acc, vNx2short_t L, vNx2short_t R) {
    return vvcmac(acc, L, R);
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_mac_fx(vNx4accint_t acc, vNx4short_t L, int16_t R) {
    vNx4accint_t r;
    r.lo = mli_math_mac_fx(acc.lo, L.lo, R);
    r.hi = mli_math_mac_fx(acc.hi, L.hi, R);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_mac_fx(vNx4accint_t acc, vNx4short_t L, vNx4short_t R) {
    vNx4accint_t r;
    r.lo = mli_math_mac_fx(acc.lo, L.lo, R.lo);
    r.hi = mli_math_mac_fx(acc.hi, L.hi, R.hi);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_mac_fx(vNx4accint_t acc, vNx4char_t L, int16_t R) {
    vNx4accint_t r;
    vNx4short_t Lshort = to_vNx4short_t(L);
    r.lo = mli_math_mac_fx(acc.lo, Lshort.lo, R);
    r.hi = mli_math_mac_fx(acc.hi, Lshort.hi, R);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_mac_fx(vNx4accint_t acc, vNx4char_t L, vNx4short_t R) {
    vNx4accint_t r;
    vNx4short_t Lshort = to_vNx4short_t(L);
    r.lo = mli_math_mac_fx(acc.lo, Lshort.lo, R.lo);
    r.hi = mli_math_mac_fx(acc.hi, Lshort.hi, R.hi);
    return r;
}

template <typename l_T, typename r_T, typename acc_T>
MLI_FORCE_INLINE acc_T mli_math_mac_fx_low(acc_T acc, l_T L, r_T R);

template <>
MLI_FORCE_INLINE vNaccint_t mli_math_mac_fx_low(vNaccint_t acc, vNint_t L, vNint_t R) {
    vNaccint_t r;
    r = vvcmac_lo(acc, L, R);
    return r;
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_mac_fx_low(vNx2accint_t acc, vNx2int_t L, vNx2int_t R) {
    vNx2accint_t r;
    r = __vacc_concat(vvcmac_lo(__vacc_lo(acc), L.lo, R.lo), vvcmac_lo(__vacc_hi(acc), L.hi, R.hi));
    return r;
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_mac_fx_low(vNx4accint_t acc, vNx4int_t L, vNx4int_t R) {
    vNx4accint_t r;
    r.lo = mli_math_mac_fx_low(acc.lo, L.lo, R.lo);
    r.hi = mli_math_mac_fx_low(acc.hi, L.hi, R.hi);
    return r;
}

template <typename l_T, typename r_T, typename acc_T>
MLI_FORCE_INLINE acc_T mli_math_mac_su_fx(acc_T acc, l_T L, r_T R);

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_mac_su_fx(vNx2accint_t acc, vNx2short_t L, uint16_t R) {
    return vvcmac_su(acc, L, R);
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_mac_su_fx(vNx4accshort_t acc, vNx4char_t L, uint8_t R) {
    return vvcmac_su(acc, L, R);
}

template <typename l_T, typename r_T, typename acc_T>
MLI_FORCE_INLINE acc_T mli_math_msub_fx(acc_T acc, l_T L, r_T R);

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_msub_fx(vNx4accshort_t acc, vNx4char_t L, int8_t R) {
    return vvcmsub(acc, L, R);
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_msub_fx(vNx2accint_t acc, vNx2short_t L, int16_t R) {
    return vvcmsub(acc, L, R);
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_msub_fx(vNx4accint_t acc, vNx4char_t L, int16_t R) {
    vNx4accint_t r;
    vNx4short_t l_short = to_vNx4short_t(L);
    r.lo = mli_math_msub_fx(acc.lo, l_short.lo, R);
    r.hi = mli_math_msub_fx(acc.hi, l_short.hi, R);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_msub_fx(vNx4accint_t acc, vNx4short_t L, int16_t R) {
    vNx4accint_t r;
    r.lo = mli_math_msub_fx(acc.lo, L.lo, R);
    r.hi = mli_math_msub_fx(acc.hi, L.hi, R);
    return r;
}


template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_msub_fx(vNx4accshort_t acc, vNx4char_t L, vNx4char_t R) {
    return vvcmsub(acc, L, R);
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_msub_fx(vNx2accint_t acc, vNx2short_t L, vNx2short_t R) {
    return vvcmsub(acc, L, R);
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_msub_fx(vNx4accint_t acc, vNx4short_t L, vNx4short_t R) {
    vNx4accint_t r;
    r.lo = mli_math_msub_fx(acc.lo, L.lo, R.lo);
    r.hi = mli_math_msub_fx(acc.hi, L.hi, R.hi);
    return r;
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_math_msub_fx(vNx4accint_t acc, vNx4char_t L, vNx4short_t R) {
    vNx4accint_t r;
    vNx4short_t l_short = to_vNx4short_t(L);
    r.lo = mli_math_msub_fx(acc.lo, l_short.lo, R.lo);
    r.hi = mli_math_msub_fx(acc.hi, l_short.hi, R.hi);
    return r;
}

template <typename l_T, typename r_T, typename acc_T> 
MLI_FORCE_INLINE acc_T mli_math_msub_su_fx(acc_T acc, l_T L, r_T R);

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_msub_su_fx(vNx2accint_t acc, vNx2short_t L, uint16_t R) {
    return vvcmsub_su(acc, L, R);
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_msub_su_fx(vNx4accshort_t acc, vNx4char_t L, uint8_t R) {
    return vvcmsub_su(acc, L, R);
}


// Accumulator shift
//========================================================================

template <>
MLI_FORCE_INLINE mli_acc32_t mli_math_acc_ashift_fx(mli_acc32_t acc, int shift_right) {
    return mli_math_asr_rnd_fx<mli_acc32_t, int>(acc, shift_right);
}

template <>
MLI_FORCE_INLINE mli_acc40_t mli_math_acc_ashift_fx(mli_acc40_t acc, int shift_right) {
    return mli_math_asr_rnd_fx<mli_acc40_t, int>(acc, shift_right);
}

typedef struct {
    pvNx2 lo;
    pvNx2 hi;
} grp_pvNx2_t;

MLI_FORCE_INLINE grp_pvNx2_t init_predicate_grp(int remaining_part_tmp) {
    pvNx2 predicate_lo = to_pvNx2(vvci_h() < remaining_part_tmp);
    pvNx2 predicate_hi = to_pvNx2(vvci_h() < mli_math_max_fx(remaining_part_tmp - _VDSP_NUM_16BIT_LANES, 0));
    grp_pvNx2_t r;
    r.lo = predicate_lo;
    r.hi = predicate_hi;
    return r;
}

MLI_FORCE_INLINE grp_pvNx2_t init_predicate_grp(vNx4short_t in) {
    grp_pvNx2_t r;
    r.lo = to_pvNx2(in.lo);
    r.hi = to_pvNx2(in.hi);
    return r;
}

MLI_FORCE_INLINE pvNx4 init_predicate(int limit, vNx4char_t in) {
    return to_pvNx4(vvci_b() < limit);
}

MLI_FORCE_INLINE pvNx2 init_predicate(int limit, vNx2short_t in) {
    return to_pvNx2(vvci_h() < limit);
}

MLI_FORCE_INLINE pvNx4 init_predicate(vNx4char_t in) {
    return to_pvNx4(in);
}

MLI_FORCE_INLINE pvNx2 init_predicate(vNx2short_t in) {
    return to_pvNx2(in);
}

MLI_FORCE_INLINE pvNx4 init_predicate(vNx4short_t in) {
    return to_pvNx4(to_vNx4char_t(in));
}

MLI_FORCE_INLINE int32_t get_predicate_count(pvNx4 p) {
    return vvpnumset(p);
}

MLI_FORCE_INLINE int32_t move_predicate_lo_to_scalar(pvNx4 p) {
    return vvpmovps_0(p);
}

MLI_FORCE_INLINE int32_t move_predicate_hi_to_scalar(pvNx4 p) {
    return vvpmovps_1(p);
}

MLI_FORCE_INLINE vNuint_t mli_math_trailing_zeros(vNuint_t in) {
    return vvnumtz(in);
}

template<typename vec_T, typename pred_T>
MLI_FORCE_INLINE vec_T mli_math_select_fx(pred_T predicate, vec_T L, vec_T R) {
    return (vec_T) vvsel(predicate, L, R);
}

template<>
MLI_FORCE_INLINE vNx4short_t mli_math_select_fx(grp_pvNx2_t predicate, vNx4short_t L, vNx4short_t R) {
    vNx4short_t res;
    res.lo = mli_math_select_fx<vNx2short_t, pvNx2>(predicate.lo, L.lo, R.lo);
    res.hi = mli_math_select_fx<vNx2short_t, pvNx2>(predicate.hi, L.hi, R.hi);
    return res;
}

template <typename in_T, typename acc_T>
MLI_FORCE_INLINE acc_T mli_math_init_accu_sub(in_T L, in_T R) {
    acc_T acc = vvcsub_init(L, R);
    return acc;
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_init_accu_sub(vNx4short_t L, vNx4short_t R) {
    vNx4accshort_t acc;
    acc = __vacc_concat(vvcsub_init(L.lo, R.lo), vvcsub_init(L.hi, R.hi));
    return acc;
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_init_accu_sub(vNx2int_t L, vNx2int_t R) {
    vNx2accint_t acc;
    acc = __vacc_concat(vvcsub_init(L.lo, R.lo), vvcsub_init(L.hi, R.hi));
    return acc;
}

template <typename in_T, typename acc_T>
MLI_FORCE_INLINE acc_T mli_math_init_accu_add(in_T L, in_T R) {
    acc_T acc = vvcadd_init(L, R);
    return acc;
}

template <>
MLI_FORCE_INLINE vNx4accshort_t mli_math_init_accu_add(vNx4short_t L, vNx4short_t R) {
    vNx4accshort_t acc;
    acc = __vacc_concat(vvcadd_init(L.lo, R.lo), vvcadd_init(L.hi, R.hi));
    return acc;
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_math_init_accu_add(vNx2int_t L, vNx2int_t R) {
    vNx2accint_t acc;
    acc = __vacc_concat(vvcadd_init(L.lo, R.lo), vvcadd_init(L.hi, R.hi));
    return acc;
}

#endif // _VDSP_MLI_MATH_H_
