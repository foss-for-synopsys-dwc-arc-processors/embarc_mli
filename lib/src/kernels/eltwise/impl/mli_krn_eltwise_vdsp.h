/*
* Copyright 2019-2022, Synopsys, Inc.
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
#include "mli_mem_info.h"
#include "arc_vector.h"

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

#ifdef ROUND_MODE_UP
    uint32_t one = 1u;
    int32_t accu_init = (one << shift_right) >> 1;
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

#ifdef ROUND_MODE_UP
    uint16_t one = 1u;
    int16_t accu_init = (one << shift_right) >> 1;
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
    constexpr int mul_hi_shift = 16;
    int shift1 = pre_op_shift1 - mul_hi_shift;
    int shift_right1 = MAX(shift1, 1);
    int shift_left1 = MAX(1 - shift1, 0);
    int shift2 = pre_op_shift2 - mul_hi_shift;
    int shift_right2 = MAX(shift2, 1);
    int shift_left2 = MAX(1 - shift2, 0);
    int out_offset1 = out_offset << shift_right1;
#ifdef ROUND_MODE_UP
    uint32_t one = 1u;
    out_offset1 += ((one << shift_right1) >> 1);
#else
    #error Rounding mode not supported
#endif
    vNx4short_t oper1 = mli_math_sub_fx(mli_math_cast_fx<vNx4char_t, vNx4short_t>(op1), (vNx4short_t)in_offset1);
    vNx4short_t oper2 = mli_math_sub_fx(mli_math_cast_fx<vNx4char_t, vNx4short_t>(op2), (vNx4short_t)in_offset2);
    oper1 = mli_math_asl_fx(oper1, shift_left1);
    oper2 = mli_math_asl_fx(oper2, shift_left2);
    vNx4short_t oper1_scaled = mli_math_mul_fx_high(oper1, scale_factor1);
    vNx4short_t oper2_scaled = mli_math_mul_fx_high(oper2, scale_factor2);
    oper1_scaled = mli_math_add_fx(oper1_scaled, (vNx4short_t)out_offset1);
    oper1_scaled = mli_math_asr_fx(oper1_scaled, shift_right1);
    oper2_scaled = mli_math_asr_rnd_fx(oper2_scaled, shift_right2);
    vNx4short_t add_res = mli_math_add_fx(oper1_scaled, oper2_scaled);
    res = mli_math_cast_fx<vNx4short_t, vNx4char_t>(add_res);
    return res;
}

template <>
MLI_FORCE_INLINE vNint_t eltwise_perform_operation<vNint_t, vNint_t, ELTWISE_ADD, false>(
        const vNint_t op1,
        const vNint_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNint_t res;
    int shift_right = MAX(post_op_shift, 0);
    int shift_left = MAX(-post_op_shift, 0);

#ifdef ROUND_MODE_UP
    uint32_t one = 1u;
    int32_t accu_init = (one << shift_right) >> 1;
    vNaccint_t accu = mli_math_init_accu<int32_t, vNaccint_t>(accu_init);
#else
    #error Rounding mode not supported
#endif

    MLI_ASSERT(pre_op_shift1 == 0);
    MLI_ASSERT(pre_op_shift2 == 0);
    MLI_ASSERT(shift_left  == 0);
    MLI_ASSERT(shift_right == 0);

    accu = mli_math_add(accu, op1);
    accu = mli_math_add(accu, op2);
    res = to_vNint_t(accu);

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

#ifdef ROUND_MODE_UP
    uint32_t one = 1u;
    int32_t accu_init = (one << shift_right) >> 1;
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

#ifdef ROUND_MODE_UP
    uint16_t one = 1u;
    int16_t accu_init = (one << shift_right) >> 1;
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
    constexpr int mul_hi_shift = 16;
    int shift1 = pre_op_shift1 - mul_hi_shift;
    int shift_right1 = MAX(shift1, 1);
    int shift_left1 = MAX(1 - shift1, 0);
    int shift2 = pre_op_shift2 - mul_hi_shift;
    int shift_right2 = MAX(shift2, 1);
    int shift_left2 = MAX(1 - shift2, 0);
    int out_offset1 = out_offset << shift_right1;
#ifdef ROUND_MODE_UP
    uint32_t one = 1u;
    out_offset1 += ((one << shift_right1) >> 1);
#else
    #error Rounding mode not supported
#endif
    vNx4short_t oper1 = mli_math_sub_fx(mli_math_cast_fx<vNx4char_t, vNx4short_t>(op1), (vNx4short_t)in_offset1);
    vNx4short_t oper2 = mli_math_sub_fx(mli_math_cast_fx<vNx4char_t, vNx4short_t>(op2), (vNx4short_t)in_offset2);
    oper1 = mli_math_asl_fx(oper1, shift_left1);
    oper2 = mli_math_asl_fx(oper2, shift_left2);
    vNx4short_t oper1_scaled = mli_math_mul_fx_high(oper1, scale_factor1);
    vNx4short_t oper2_scaled = mli_math_mul_fx_high(oper2, scale_factor2);
    oper1_scaled = mli_math_add_fx(oper1_scaled, (vNx4short_t)out_offset1);
    oper1_scaled = mli_math_asr_fx(oper1_scaled, shift_right1);
    oper2_scaled = mli_math_asr_rnd_fx(oper2_scaled, shift_right2);
    vNx4short_t sub_res = mli_math_sub_fx(oper1_scaled, oper2_scaled);
    res = mli_math_cast_fx<vNx4short_t, vNx4char_t>(sub_res);
    return res;
}

template <>
MLI_FORCE_INLINE vNint_t eltwise_perform_operation<vNint_t, vNint_t, ELTWISE_SUB, false>(
        const vNint_t op1,
        const vNint_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    vNint_t res;
    int shift_right = MAX(post_op_shift, 0);
    int shift_left = MAX(-post_op_shift, 0);

#ifdef ROUND_MODE_UP
    uint32_t one = 1u;
    int32_t accu_init = (one << shift_right) >> 1;
    vNaccint_t accu = mli_math_init_accu<int32_t, vNaccint_t>(accu_init);
#else
    #error Rounding mode not supported
#endif

    MLI_ASSERT(pre_op_shift1 == 0);
    MLI_ASSERT(pre_op_shift2 == 0);
    MLI_ASSERT(shift_left  == 0);
    MLI_ASSERT(shift_right == 0);

    vNaccint_t accu_sub = mli_math_init_accu_sub<vNint_t, vNaccint_t>(op1, op2);
    accu = mli_math_add(accu, accu_sub);
    res = to_vNint_t(accu);

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
#ifdef ROUND_MODE_UP
    uint32_t one = 1u;
    int32_t acc_init = ((one << shift_right) >> 1);
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
#ifdef ROUND_MODE_UP
    uint16_t one = 1u;
    int16_t acc_init = ((one << shift_right) >> 1);
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
#ifdef ROUND_MODE_UP
    uint16_t one = 1u;
    acc_init += ((one << preshift) >> 1); /* rounding half up */
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

    vNx4short_t op1_offset = mli_math_cast_fx<vNx4char_t, vNx4short_t>(op1) - in_offset1;
    vNx4short_t op2_offset = mli_math_cast_fx<vNx4char_t, vNx4short_t>(op2) - in_offset2;
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
MLI_FORCE_INLINE vNx4int_t eltwise_perform_operation<vNx4char_t, vNx4int_t, ELTWISE_MUL, false>(
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
    vNx4int_t res;
    int shift_left = MAX(-post_op_shift, 0);
    int shift_right = MAX(post_op_shift, 0);
#ifdef ROUND_MODE_UP
    uint16_t one = 1u;
    int16_t acc_init = ((one << shift_right) >> 1);
#else
    #error Rounding mode not supported
#endif
    vNx4accint_t acc32 = mli_math_init_accu<int32_t, vNx4accint_t>((int32_t)acc_init);
    vNx4short_t op1_prom = mli_math_cast_fx<vNx4char_t, vNx4short_t>(op1);
    vNx4short_t op2_prom = mli_math_cast_fx<vNx4char_t, vNx4short_t>(op2);
    acc32 = mli_math_mac_fx(acc32, op1_prom, op2_prom);
    res = mli_math_acc_cast_fx<vNx4int_t, vNx4accint_t, false> (acc32, shift_right);
    res = mli_math_asl_fx(res, shift_left);

    return res;
}

template <>
MLI_FORCE_INLINE vNx2int_t eltwise_perform_operation<vNx2short_t, vNx2int_t, ELTWISE_MUL, false>(
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
    vNx2int_t res;
    int shift_left = MAX(-post_op_shift, 0);
    int shift_right = MAX(post_op_shift, 0);
#ifdef ROUND_MODE_UP
    uint16_t one = 1u;
    int16_t acc_init = ((one << shift_right) >> 1);
#else
    #error Rounding mode not supported
#endif
    vNx2accint_t acc32 = mli_math_init_accu<int32_t, vNx2accint_t>((int32_t)acc_init);
    acc32 = mli_math_mac_fx(acc32, op1, op2);
    res = mli_math_acc_cast_fx<vNx2int_t, vNx2accint_t, false> (acc32, shift_right);
    res = mli_math_asl_fx(res, shift_left);

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
#ifdef ROUND_MODE_UP
    uint16_t one = 1u;
    offset += ((one << shift_right) >> 1);
#else
    #error Rounding mode not supported
#endif
    vNx4short_t max = mli_math_cast_fx<vNx4char_t, vNx4short_t>(mli_math_max_fx(op1, op2));
    max = mli_math_sub_fx(max, (vNx4short_t)in_offset1);
    max = mli_math_asl_fx(max, shift_left);
    vNx4short_t max_scaled = mli_math_mul_fx_high(max, scale_factor1);
    max_scaled = mli_math_add_fx(max_scaled, (vNx4short_t) offset);
    res = mli_math_cast_fx<vNx4short_t, vNx4char_t, false>(max_scaled, shift_right);
    return res;
}

template <>
MLI_FORCE_INLINE vNint_t eltwise_perform_operation<vNint_t, vNint_t, ELTWISE_MAX, false>(
        const vNint_t op1,
        const vNint_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    MLI_ASSERT(pre_op_shift1 == 0);
    MLI_ASSERT(pre_op_shift2 == 0);
    MLI_ASSERT(post_op_shift  == 0);

    vNint_t res;
    res = mli_math_max_fx(op1, op2);

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
#ifdef ROUND_MODE_UP
    uint16_t one = 1u;
    offset += ((one << shift_right) >> 1);
#else
    #error Rounding mode not supported
#endif
    vNx4short_t max = mli_math_cast_fx<vNx4char_t, vNx4short_t>(mli_math_min_fx(op1, op2));
    max = mli_math_sub_fx(max, (vNx4short_t)in_offset1);
    max = mli_math_asl_fx(max, shift_left);
    vNx4short_t max_scaled = mli_math_mul_fx_high(max, scale_factor1);
    max_scaled = mli_math_add_fx(max_scaled, (vNx4short_t) offset);
    res = mli_math_cast_fx<vNx4short_t, vNx4char_t, false>(max_scaled, shift_right);
    return res;

}

template <>
MLI_FORCE_INLINE vNint_t eltwise_perform_operation<vNint_t, vNint_t, ELTWISE_MIN, false>(
        const vNint_t op1,
        const vNint_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    MLI_ASSERT(pre_op_shift1 == 0);
    MLI_ASSERT(pre_op_shift2 == 0);
    MLI_ASSERT(post_op_shift  == 0);

    vNint_t res;
    res = mli_math_min_fx(op1, op2);

    return res;
}


template <typename i_T, typename o_T, mli_eltwise_type func_type, bool convert = false>
MLI_FORCE_INLINE void eltwise_innerloop(
        const MLI_PTR(i_T) __restrict op1_ptr,
        const MLI_PTR(i_T) __restrict op2_ptr,
        MLI_PTR(o_T) __restrict out_ptr,
        int idx1,
        int idx2,
        int idx_out,
        const int count,
        i_T op1_s,
        i_T op2_s,
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

    // The input dtype is deduced by `load_1vec` function
    using IDtype = decltype(input);
    // Then, get the output dtype by type triats. Pseudocode:
    //   if (IDtype == vNx4char_t && o_T == int32_t) {
    //       ODtype = vNx4int_t;
    //   } else if (IDType == vNx2short_t && o_T == int32_t) {
    //       ODtype = vNx2int_t;
    //   } else {
    //       ODtype = IDtype;
    //   }
    using ODtype = typename std::conditional<
        std::is_same<IDtype, vNx4char_t>::value & std::is_same<o_T, int32_t>::value,
            vNx4int_t, typename std::conditional<
                std::is_same<IDtype, vNx2short_t>::value & std::is_same<o_T, int32_t>::value,
                vNx2int_t, IDtype> ::type > ::type;

    IDtype op1_scalar = op1_s;
    IDtype op2_scalar = op2_s;

    if (remaining_part) {
        auto val1 = (scalar_op1) ? op1_scalar : mli_prv_load_1vec(op1_ptr + idx1);
        auto val2 = (scalar_op2) ? op2_scalar : mli_prv_load_1vec(op2_ptr + idx2);
        auto res = mli::krn::eltwise_perform_operation<IDtype, ODtype, func_type, convert>(
                                                       val1, val2, in_offset1, in_offset2, out_offset, scale1,
                                                       scale2, pre_op_shift1, pre_op_shift2, post_op_shift);
        mli_prv_store_n_samples(&out_ptr[idx_out], res, remaining_part);
        idx1 += remaining_part;
        idx2 += remaining_part;
        idx_out += remaining_part;
    }

#pragma clang loop unroll_count(4)
    for (int pos = 0; pos < (count - remaining_part); pos+=num_lanes) {
        auto val1 = (scalar_op1) ? op1_scalar : mli_prv_load_1vec(op1_ptr + idx1);
        auto val2 = (scalar_op2) ? op2_scalar : mli_prv_load_1vec(op2_ptr + idx2);
        auto res = mli::krn::eltwise_perform_operation<IDtype, ODtype, func_type, convert>(
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
