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

    vNx2accint_t tmp = mli_math_mul_fx<vNx2short_t, vNx2accint_t>(op1, op2);
    res = mli_math_acc_cast_fx<vNx2short_t, vNx2accint_t>(tmp, shift_right);
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

    vNx4accshort_t tmp = mli_math_mul_fx<vNx4char_t, vNx4accshort_t>(op1, op2);
    res = mli_math_acc_cast_fx<vNx4char_t, vNx4accshort_t> (tmp, shift_right);
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
    MLI_ASSERT(post_op_shift > 3);
#if defined(__Xvec_guard_bit_option) && __Xvec_guard_bit_option != 0
    const int preshift_sf = 1;
#else
    const int preshift_sf = 3;
#endif

    const int mask = (1 << preshift_sf) - 1;
    vNx4char_t res;
    vNx4short_t op1_offset = to_vNx4short_t(op1) - in_offset1;
    vNx4short_t op2_offset = to_vNx4short_t(op2) - in_offset2;

    /*
     * Each operand is 9 bit. The first multiplier output is 18 bit. After scaling with positive 15 bit scale_factor,
     * The second multiplier output is 32 bits. A headroom of 3 is sufficient to add the offset, round and compensate.
     *
     * Note: Minimum shift value is 15
     */

    vNx4accint_t acc = mli_math_mul_fx<vNx4short_t, vNx4accint_t>(op1_offset, op2_offset);
    vNx4int_t temp1 = mli_math_acc_cast_fx<vNx4int_t, vNx4accint_t>(acc);
    vNx4int_t temp2 = (scale_factor1 & mask);
    vNx4int_t offset = out_offset << (post_op_shift - preshift_sf);
    acc = mli_math_mul_fx_low(temp1, temp2);
    acc = mli_math_asr_fx(acc, preshift_sf);
    acc = mli_math_add(acc, offset);
    temp2 = (scale_factor1 >> preshift_sf);
    acc = mli_math_mac_fx_low(acc, temp1, temp2);
    res = mli_math_acc_cast_fx<vNx4char_t, vNx4accint_t>(acc, post_op_shift - preshift_sf);
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
    int shift_right = MAX(post_op_shift,0);
    int shift_left = MAX(-post_op_shift,0);

    res = mli_math_max_fx(op1, op2);
    res = mli_math_asl_fx(res, shift_left);
    res = mli_math_asr_rnd_fx(res, shift_right);

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
    int shift_right = MAX(post_op_shift,0);
    int shift_left = MAX(-post_op_shift,0);

    res = mli_math_max_fx(op1, op2);
    res = mli_math_asl_fx(res, shift_left);
    res = mli_math_asr_rnd_fx(res, shift_right);

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

    vNx4short_t max = to_vNx4short_t(mli_math_max_fx(op1, op2));
    max = mli_math_sub_fx(max, (vNx4short_t)in_offset1);
    vNx4accint_t acc = mli_math_mul_fx<vNx4short_t, vNx4accint_t>(max, scale_factor1);
    vNx4int_t temp = mli_math_acc_cast_fx<vNx4int_t, vNx4accint_t>(acc);
    temp = mli_math_asr_rnd_fx(temp, post_op_shift);
    temp = mli_math_add_fx(temp, (vNx4int_t)out_offset);
    res = mli_math_cast_fx<vNx4int_t, vNx4char_t>(temp);

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

     int shift_right = MAX(post_op_shift,0);
     int shift_left = MAX(-post_op_shift,0);
     res = mli_math_min_fx(op1, op2);
     res = mli_math_asl_fx(res, shift_left);
     res = mli_math_asr_rnd_fx(res, shift_right);

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

    int shift_right = MAX(post_op_shift,0);
    int shift_left = MAX(-post_op_shift,0);
    res = mli_math_min_fx(op1, op2);
    res = mli_math_asl_fx(res, shift_left);
    res = mli_math_asr_rnd_fx(res, shift_right);

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
    vNx4short_t min = to_vNx4short_t(mli_math_min_fx(op1, op2));
    min = mli_math_sub_fx(min, (vNx4short_t)in_offset1);
    vNx4accint_t acc = mli_math_mul_fx<vNx4short_t, vNx4accint_t>(min, scale_factor1);
    vNx4int_t temp = mli_math_acc_cast_fx<vNx4int_t, vNx4accint_t>(acc);
    temp = mli_math_asr_rnd_fx(temp, post_op_shift);
    temp = mli_math_add_fx(temp, (vNx4int_t)out_offset);
    res = mli_math_cast_fx<vNx4int_t, vNx4char_t>(temp);

    return res;
}


template <typename io_T, mli_eltwise_type func_type, bool convert>
MLI_FORCE_INLINE void eltwise_innerloop(
        const MLI_PTR(io_T) op1_ptr,
        const MLI_PTR(io_T) op2_ptr,
        MLI_PTR(io_T) out_ptr,
        int idx1,
        int idx2,
        int idx_out,
        const int count,
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

    if (remaining_part) {
        auto val1 = (scalar_op1) ? op1_ptr[0] : mli_prv_load_1vec(op1_ptr + idx1);
        auto val2 = (scalar_op2) ? op2_ptr[0] : mli_prv_load_1vec(op2_ptr + idx2);
        auto res = mli::krn::eltwise_perform_operation<decltype(val1), decltype(val1), func_type, convert>(
                                                       val1, val2, in_offset1, in_offset2, out_offset, scale1,
                                                       scale2, pre_op_shift1, pre_op_shift2, post_op_shift);
        mli_prv_store_n_samples(&out_ptr[idx_out], res, remaining_part);
        idx1 += remaining_part;
        idx2 += remaining_part;
        idx_out += remaining_part;
    }

    for (int pos = remaining_part; pos < count; pos+=num_lanes) {
        auto val1 = mli_prv_load_1vec(op1_ptr + idx1);
        auto val2 = mli_prv_load_1vec(op2_ptr + idx2);
        val1 = (scalar_op1) ? op1_ptr[0] : val1;
        val2 = (scalar_op2) ? op2_ptr[0] : val2;
        auto res = mli::krn::eltwise_perform_operation<decltype(val1), decltype(val1), func_type, convert>(
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
