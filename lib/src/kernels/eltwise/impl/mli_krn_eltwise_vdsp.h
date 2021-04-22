/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_ELTWISE_ADD_VDSP_H_
#define _MLI_KRN_ELTWISE_ADD_VDSP_H_

#include "mli_krn_eltwise_decl.h"

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_math.h"
#include "arc_vector.h"

const int unroll_factor[2][5] = {
{
/* ELTWISE_ADD_NO_CONVERT = */ 1,
/* ELTWISE_SUB_NO_CONVERT = */ 1,
/* ELTWISE_MUL_NO_CONVERT = */ 4,
/* ELTWISE_MAX_NO_CONVERT = */ 4,
/* ELTWISE_MIN_NO_CONVERT = */ 4
} ,
{
/* ELTWISE_ADD_CONVERT = */ 1,
/* ELTWISE_SUB_CONVERT = */ 1,
/* ELTWISE_MUL_CONVERT = */ 4,
/* ELTWISE_MAX_CONVERT = */ 4,
/* ELTWISE_MIN_CONVERT = */ 4
}
};

namespace mli {
namespace krn {
namespace vdsp {

//======================================================
//
//======================================================

template <>
MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_ADD, false>(
        const vNx2short_t op1,
        const vNx2short_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNx2short_t res;
    int shift_right = MAX(post_op_shift, 0);
    int shift_left = MAX(-post_op_shift, 0);

    #ifdef ROUND_UP
        int32_t accu_init = (1 << shift_right) >> 1;
        vNx2accint_t accu = mli_math_init_accu<int32_t, vNx2accint_t>(accu_init);
    #else
        #error Rounding mode not supported
    #endif

    accu = mli_math_mac_su_fx(accu, op1, (uint16_t) (1u << -pre_op_shift1));
    accu = mli_math_mac_su_fx(accu, op2, (uint16_t) (1u << -pre_op_shift2));
    res = mli_math_acc_cast_fx<vNx2short_t, vNx2accint_t, /*round = */false>(accu, shift_right);

    res = mli_math_asl_fx(res, shift_left);
    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_ADD, false>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNx4char_t res;
    int shift_right = MAX(post_op_shift, 0);
    int shift_left = MAX(-post_op_shift, 0);

    #ifdef ROUND_UP
        int16_t accu_init = (1 << shift_right) >> 1;
        vNx4accshort_t accu = mli_math_init_accu<int16_t, vNx4accshort_t>(accu_init);
    #else
        #error Rounding mode not supported
    #endif

    accu = mli_math_mac_su_fx(accu, op1, (uint8_t) (1u << -pre_op_shift1));
    accu = mli_math_mac_su_fx(accu, op2, (uint8_t) (1u << -pre_op_shift2));
    res = mli_math_acc_cast_fx<vNx4char_t, vNx4accshort_t, false>(accu, shift_right);

    res = mli_math_asl_fx(res, shift_left);
    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_ADD, true>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNx4char_t res;

    /* initialize the accumulator with the input offsets, the output offset, and the rounding value.*/
    int32_t acc_init =  -in_offset1 * scale_factor1 - in_offset2 * scale_factor2;
    acc_init += (out_offset << post_op_shift);
#ifdef ROUND_UP
    acc_init += ((1 << post_op_shift) >> 1); /* rounding half up */
#else
    #error Rounding mode not supported
#endif

    vNx4accint_t acc = mli_math_init_accu<int32_t, vNx4accint_t>(acc_init);
    acc = mli_math_mac_fx(acc, to_vNx4short_t(op1), scale_factor1);
    acc = mli_math_mac_fx(acc, to_vNx4short_t(op2), scale_factor2);
    /* rounding value is already added as part of the acc_init. therefore it shouldn't
       be added again inside the cast functin.*/
    res = mli_math_acc_cast_fx<vNx4char_t, vNx4accint_t, /*round = */false> (acc, post_op_shift);
    return res;
}

template <>
MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_SUB, false>(
        const vNx2short_t op1,
        const vNx2short_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNx2short_t res;
    int shift_left = MAX(-post_op_shift, 0);
    int shift_right = MAX(post_op_shift, 0);

    #ifdef ROUND_UP
        int32_t accu_init = (1 << shift_right) >> 1;
        vNx2accint_t accu = mli_math_init_accu<int32_t, vNx2accint_t>(accu_init);
    #else
        #error Rounding mode not supported
    #endif

    accu = mli_math_mac_su_fx(accu, op1, (uint16_t) (1u << -pre_op_shift1));
    accu = mli_math_msub_su_fx(accu, op2, (uint16_t) (1u << -pre_op_shift2));
    res = mli_math_acc_cast_fx<vNx2short_t, vNx2accint_t, /*round = */false>(accu, shift_right);

    res = mli_math_asl_fx(res, shift_left);
    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_SUB, false>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNx4char_t res;
    int shift_left = MAX(-post_op_shift, 0);
    int shift_right = MAX(post_op_shift, 0);

    #ifdef ROUND_UP
        int16_t accu_init = (1 << shift_right) >> 1;
        vNx4accshort_t accu = mli_math_init_accu<int16_t, vNx4accshort_t>(accu_init);
    #else
        #error Rounding mode not supported
    #endif

    accu = mli_math_mac_su_fx(accu, op1, (uint8_t) (1u << -pre_op_shift1));
    accu = mli_math_msub_su_fx(accu, op2, (uint8_t) (1u << -pre_op_shift2));
    res = mli_math_acc_cast_fx<vNx4char_t, vNx4accshort_t, false>(accu, shift_right);

    res = mli_math_asl_fx(res, shift_left);
    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_SUB, true>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNx4char_t res;
    int32_t acc_init = in_offset2 * scale_factor2 - in_offset1 * scale_factor1;
    acc_init += (out_offset << post_op_shift);
    #ifdef ROUND_UP
        acc_init += ((1 << post_op_shift) >> 1); /* rounding half up */
    #else
        #error Rounding mode not supported
    #endif
    vNx4accint_t acc = mli_math_init_accu<int32_t, vNx4accint_t>(acc_init);

    acc = mli_math_mac_fx(acc, to_vNx4short_t(op1), scale_factor1);
    acc = mli_math_msub_fx(acc, to_vNx4short_t(op2), scale_factor2);
    /* rounding value is already added as part of the acc_init. therefore it shouldn't
       be added again inside the cast functin.*/
    res = mli_math_acc_cast_fx<vNx4char_t, vNx4accint_t, /*round = */false> (acc, post_op_shift);

    return res;
}

template <>
MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_MUL, false>(
        const vNx2short_t op1,
        const vNx2short_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNx2short_t res;
    int shift_left = MAX(-post_op_shift, 0);
    int shift_right = MAX(post_op_shift, 0);
    #ifdef ROUND_UP
        int32_t acc_init = ((1 << shift_right) >> 1);
    #else
        #error Rounding mode not supported
    #endif
    vNx2accint_t acc = mli_math_init_accu<int32_t, vNx2accint_t>(acc_init);
    acc = mli_math_mac_fx(acc, op1, op2);
    res = mli_math_acc_cast_fx<vNx2short_t, vNx2accint_t, false>(acc, shift_right);
    res = mli_math_asl_fx(res, shift_left);
    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MUL, false>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNx4char_t res;
    int shift_left = MAX(-post_op_shift, 0);
    int shift_right = MAX(post_op_shift, 0);
#ifdef ROUND_UP
    int16_t acc_init = ((1 << shift_right) >> 1);
#else
    #error Rounding mode not supported
#endif
    vNx4accshort_t acc = mli_math_init_accu<int16_t, vNx4accshort_t>(acc_init);
    acc = mli_math_mac_fx(acc, op1, op2);
    res = mli_math_acc_cast_fx<vNx4char_t, vNx4accshort_t, false> (acc, shift_right);
    res = mli_math_asl_fx(res, shift_left);

    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MUL, true>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNx4char_t res;
    const int headroom = 3;
    const int hi_comp = 16;
    const int acc_len = 32;
    const int out_len = 8;
    const int target_out_shift = acc_len - out_len - headroom;
    const int preshift = mli_math_min_fx(mli_math_max_fx(post_op_shift - target_out_shift, 0), headroom);
    const int shift = post_op_shift - hi_comp - preshift;
    const int shift_left = mli_math_max_fx(1 - shift, 0);
    const int shift_right = mli_math_max_fx(shift, 1);

#if defined(__Xvec_guard_bit_option) && __Xvec_guard_bit_option != 0
    /*
     *  res = ((op1 - in_offset1) * (op2 - in_offset2) * scale_factor1 >> post_op_shift) + out_offset
     *  acc_init = in_offset1 * in_offset2
     *  term1  = op1 * op2 * scale_factor1
     *  term2 = - op2 * in_offset1 * scale_factor1
     *  term3 = - op1 * in_offset2 * scale_factor1
     *  acc = (term1 + term2 + term3) * scale_factor >> post_op_shift + out_offset
     *
     */

    int16_t acc_init = in_offset1 * in_offset2;
#ifdef ROUND_UP
    acc_init += ((1 << preshift) >> 1); /* rounding half up */
#else
    #error Rounding mode not supported
#endif
    vNx4accshort_t acc16 = mli_math_init_accu<int16_t, vNx4accshort_t>(acc_init);
    acc16 = mli_math_mac_fx(acc16, op1, op2);
    acc16 = mli_math_msub_fx(acc16, op2, (vNx4char_t)(int8_t)in_offset1);
    acc16 = mli_math_msub_fx(acc16, op1, (vNx4char_t)(int8_t)in_offset2);

    /*
     * If we preshift we can continue the operations in 16 bits. Only 8 bits are needs from the
     * mul_hi output. with headroom of 3 bits.
     */

    vNx4short_t vacc16 = mli_math_acc_cast_fx<vNx4short_t, vNx4accshort_t, false>(acc16, preshift);
#else

    vNx4short_t op1_offset = to_vNx4short_t(op1) - in_offset1;
    vNx4short_t op2_offset = to_vNx4short_t(op2) - in_offset2;
    vNx4int_t acc32 = mli_math_mul_fx<vNx4short_t, vNx4int_t>(op1_offset, op2_offset);

    /*
     * If we preshift we can continue the operations in 16 bits. Only 8 bits are needs from the
     * mul_hi output. with headroom of 3 bits.
     */

    vNx4short_t vacc16 = mli_math_cast_fx<vNx4int_t, vNx4short_t>(acc32, preshift);
#endif

    vacc16 = mli_math_asl_fx(vacc16, shift_left);
    vNx4short_t accu_scaled = mli_math_mul_fx_high(vacc16, scale_factor1);
    accu_scaled = mli_math_asr_rnd_fx(accu_scaled, shift_right);
    accu_scaled = mli_math_add_fx(accu_scaled, (vNx4short_t) out_offset);
    res = mli_math_cast_fx<vNx4short_t, vNx4char_t>(accu_scaled);

    return res;
}

template <>
MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_MAX, false>(
        const vNx2short_t op1,
        const vNx2short_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNx2short_t res;
    res = mli_math_max_fx(op1, op2);
    if (post_op_shift > 0) {
        res = mli_math_asr_rnd_fx(res, post_op_shift);
    } else {
        res = mli_math_asl_fx(res, -post_op_shift);
    }

    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MAX, false>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNx4char_t res ;

    res = mli_math_max_fx(op1, op2);
    if (post_op_shift > 0) {
        res = mli_math_asr_rnd_fx(res, post_op_shift);
    } else {
        res = mli_math_asl_fx(res, -post_op_shift);
    }
    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MAX, true>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNx4char_t res;
    constexpr int mul_hi_shift = 16;
    int shift = post_op_shift - mul_hi_shift;
    int shift_left = mli_math_max_fx(1 - shift, 0);
    int shift_right = mli_math_max_fx(shift, 1);
    // As shift is limited by 23 the shift_right is limited by 7 so we can pre_shift left the out_offset
    int16_t offset = out_offset << shift_right;
#ifdef ROUND_UP
    offset += ((1 << shift_right) >> 1);
#else
    #error Rounding mode not supported
#endif
    vNx4short_t max = to_vNx4short_t(mli_math_max_fx(op1, op2));
    max = mli_math_sub_fx(max, (vNx4short_t)in_offset1);
    max = mli_math_asl_fx(max, shift_left);
    vNx4short_t max_scaled = mli_math_mul_fx_high(max, scale_factor1);
    max_scaled = mli_math_add_fx(max_scaled, (vNx4short_t) offset);
    res = mli_math_cast_fx<vNx4short_t, vNx4char_t, false>(max_scaled, shift_right);
    return res;
}

template <>
MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_MIN, false>(
        const vNx2short_t op1,
        const vNx2short_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
     vNx2short_t res;

     res = mli_math_min_fx(op1, op2);
     if (post_op_shift > 0) {
         res = mli_math_asr_rnd_fx(res, post_op_shift);
     } else {
         res = mli_math_asl_fx(res, -post_op_shift);
     }
     return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MIN, false>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNx4char_t res ;

    res = mli_math_min_fx(op1, op2);
    if (post_op_shift > 0) {
        res = mli_math_asr_rnd_fx(res, post_op_shift);
    } else {
        res = mli_math_asl_fx(res, -post_op_shift);
    }
    return res;


}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MIN, true>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNx4char_t res;
    constexpr int mul_hi_shift = 16;
    int shift = post_op_shift - mul_hi_shift;
    int shift_left = mli_math_max_fx(1 - shift, 0);
    int shift_right = mli_math_max_fx(shift, 1);
    // As shift is limited by 23 the shift_right is limited by 7 so we can pre_shift left the out_offset
    int16_t offset = out_offset << shift_right;
#ifdef ROUND_UP
    offset += ((1 << shift_right) >> 1);
#else
    #error Rounding mode not supported
#endif
    vNx4short_t max = to_vNx4short_t(mli_math_min_fx(op1, op2));
    max = mli_math_sub_fx(max, (vNx4short_t)in_offset1);
    max = mli_math_asl_fx(max, shift_left);
    vNx4short_t max_scaled = mli_math_mul_fx_high(max, scale_factor1);
    max_scaled = mli_math_add_fx(max_scaled, (vNx4short_t) offset);
    res = mli_math_cast_fx<vNx4short_t, vNx4char_t, false>(max_scaled, shift_right);
    return res;

}


template <typename io_T, mli_eltwise_type func_type, bool convert>
void eltwise_innerloop(
        const MLI_PTR(io_T) __restrict op1_ptr,
        const MLI_PTR(io_T) __restrict op2_ptr,
        MLI_PTR(io_T) __restrict out_ptr,
        int idx1,
        int idx2,
        int idx_out,
        const int count,
        io_T op1_s,
        io_T op2_s,
        const bool scalar_op1,
        const bool scalar_op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale1,
        const int16_t scale2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(op1_ptr);
    int num_lanes = get_number_lanes(input);
    int remaining_part = count & (num_lanes - 1);
    decltype(input) op1_scalar = op1_s;
    decltype(input) op2_scalar = op2_s;
    const int convert_int = static_cast<int>(convert);
    const int func_int = static_cast<int>(func_type);

    if (remaining_part) {
        auto val1 = (scalar_op1) ? op1_scalar : mli_prv_load_1vec(op1_ptr + idx1);
        auto val2 = (scalar_op2) ? op2_scalar : mli_prv_load_1vec(op2_ptr + idx2);
        auto res = mli::krn::eltwise_perform_operation<decltype(input), decltype(input), func_type, convert>(
                                                       val1, val2, in_offset1, in_offset2, out_offset, scale1,
                                                       scale2, pre_op_shift1, pre_op_shift2, post_op_shift);
        mli_prv_store_n_samples(&out_ptr[idx_out], res, remaining_part);
        idx1 += remaining_part;
        idx2 += remaining_part;
        idx_out += remaining_part;
    }

#pragma clang loop unroll_count(unroll_factor[convert_int][func_int])
    for (int pos = 0; pos < (count - remaining_part); pos+=num_lanes) {
        auto val1 = (scalar_op1) ? op1_scalar : mli_prv_load_1vec(op1_ptr + idx1);
        auto val2 = (scalar_op2) ? op2_scalar : mli_prv_load_1vec(op2_ptr + idx2);
        auto res = mli::krn::eltwise_perform_operation<decltype(input), decltype(input), func_type, convert>(
                                                       val1, val2, in_offset1, in_offset2, out_offset, scale1,
                                                       scale2, pre_op_shift1, pre_op_shift2, post_op_shift);
        mli_prv_store_n_samples(&out_ptr[idx_out], res);
        idx1 += num_lanes;
        idx2 += num_lanes;
        idx_out += num_lanes;
    }
}

template<>
MLI_FORCE_INLINE void eltwise_innerloop<int16_t, ELTWISE_MAX, false>(
        const MLI_PTR(int16_t) __restrict op1_ptr,
        const MLI_PTR(int16_t) __restrict op2_ptr,
        MLI_PTR(int16_t) __restrict out_ptr,
        int idx1,
        int idx2,
        int idx_out,
        const int count,
        int16_t op1_s,
        int16_t op2_s,
        const bool scalar_op1,
        const bool scalar_op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale1,
        const int16_t scale2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(op1_ptr);
    int num_lanes = get_number_lanes(input);
    int remaining_part = count & (num_lanes - 1);
    decltype(input) op1_scalar = op1_s;
    decltype(input) op2_scalar = op2_s;
    const int convert_int = static_cast<int>(false);
    const int func_int = static_cast<int>(ELTWISE_MAX);

    if (remaining_part) {
        auto val1 = (scalar_op1) ? op1_scalar : mli_prv_load_1vec(op1_ptr + idx1);
        auto val2 = (scalar_op2) ? op2_scalar : mli_prv_load_1vec(op2_ptr + idx2);
        auto res = mli::krn::eltwise_perform_operation<decltype(input), decltype(input), ELTWISE_MAX, false>(
                                                       val1, val2, in_offset1, in_offset2, out_offset, scale1,
                                                       scale2, pre_op_shift1, pre_op_shift2, post_op_shift);
        mli_prv_store_n_samples(&out_ptr[idx_out], res, remaining_part);
        idx1 += remaining_part;
        idx2 += remaining_part;
        idx_out += remaining_part;
    }

#pragma clang loop unroll_count(unroll_factor[convert_int][func_int])
    for (int pos = 0; pos < (count - remaining_part); pos+=num_lanes) {
        auto val1 = (scalar_op1) ? op1_scalar : mli_prv_load_1vec(op1_ptr + idx1);
        auto val2 = (scalar_op2) ? op2_scalar : mli_prv_load_1vec(op2_ptr + idx2);
        auto res = mli::krn::eltwise_perform_operation<decltype(input), decltype(input), ELTWISE_MAX, false>(
                                                       val1, val2, in_offset1, in_offset2, out_offset, scale1,
                                                       scale2, pre_op_shift1, pre_op_shift2, post_op_shift);
        mli_prv_store_n_samples(&out_ptr[idx_out], res);
        idx1 += num_lanes;
        idx2 += num_lanes;
        idx_out += num_lanes;
    }
}

template<>
MLI_FORCE_INLINE void eltwise_innerloop<int16_t, ELTWISE_MIN, false>(
        const MLI_PTR(int16_t) __restrict op1_ptr,
        const MLI_PTR(int16_t) __restrict op2_ptr,
        MLI_PTR(int16_t) __restrict out_ptr,
        int idx1,
        int idx2,
        int idx_out,
        const int count,
        int16_t op1_s,
        int16_t op2_s,
        const bool scalar_op1,
        const bool scalar_op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale1,
        const int16_t scale2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(op1_ptr);
    int num_lanes = get_number_lanes(input);
    int remaining_part = count & (num_lanes - 1);
    decltype(input) op1_scalar = op1_s;
    decltype(input) op2_scalar = op2_s;
    const int convert_int = static_cast<int>(false);
    const int func_int = static_cast<int>(ELTWISE_MIN);

    if (remaining_part) {
        auto val1 = (scalar_op1) ? op1_scalar : mli_prv_load_1vec(op1_ptr + idx1);
        auto val2 = (scalar_op2) ? op2_scalar : mli_prv_load_1vec(op2_ptr + idx2);
        auto res = mli::krn::eltwise_perform_operation<decltype(input), decltype(input), ELTWISE_MIN, false>(
                                                       val1, val2, in_offset1, in_offset2, out_offset, scale1,
                                                       scale2, pre_op_shift1, pre_op_shift2, post_op_shift);
        mli_prv_store_n_samples(&out_ptr[idx_out], res, remaining_part);
        idx1 += remaining_part;
        idx2 += remaining_part;
        idx_out += remaining_part;
    }

#pragma clang loop unroll_count(unroll_factor[convert_int][func_int])
    for (int pos = 0; pos < (count - remaining_part); pos+=num_lanes) {
        auto val1 = (scalar_op1) ? op1_scalar : mli_prv_load_1vec(op1_ptr + idx1);
        auto val2 = (scalar_op2) ? op2_scalar : mli_prv_load_1vec(op2_ptr + idx2);
        auto res = mli::krn::eltwise_perform_operation<decltype(input), decltype(input), ELTWISE_MIN, false>(
                                                       val1, val2, in_offset1, in_offset2, out_offset, scale1,
                                                       scale2, pre_op_shift1, pre_op_shift2, post_op_shift);
        mli_prv_store_n_samples(&out_ptr[idx_out], res);
        idx1 += num_lanes;
        idx2 += num_lanes;
        idx_out += num_lanes;
    }
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_ELTWISE_ADD_VDSP_H_
