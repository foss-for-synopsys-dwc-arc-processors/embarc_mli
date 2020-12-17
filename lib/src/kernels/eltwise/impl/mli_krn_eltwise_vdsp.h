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
template <typename in_T, typename out_T, mli_eltwise_type func_type, bool convert>
static MLI_FORCE_INLINE out_T eltwise_perform_operation(
        const in_T op1,
        const in_T op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    return mli::krn::ref::eltwise_perform_operation<in_T, out_T, func_type, convert>
        (op1, op2, in_offset, out_offset, scale_factor, shift, reverse_sub);
}

#if defined(__Xvec_guard_bit_option) && (__Xvec_guard_bit_option != 0)
template <>
MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_ADD, false>(
        const vNx2short_t op1,
        const vNx2short_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx2short_t res;
    int shift_right = MAX(shift, 0);
    int shift_left = MAX(-shift, 0);

    vNx2accshort_t tmp = mli_math_init_accu_add<vNx2short_t, vNx2accshort_t>(op1, op2);
    res = mli_math_acc_cast_fx<vNx2short_t, vNx2accshort_t> (tmp, shift_right);
    res = mli_math_asl_fx(res, shift_left);
    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_ADD, false>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx4char_t res;
    int shift_right = MAX(shift, 0);
    int shift_left = MAX(-shift, 0);

    vNx4accchar_t tmp = mli_math_init_accu_add<vNx4char_t, vNx4accchar_t>(op1, op2);
    res = mli_math_acc_cast_fx<vNx4char_t, vNx4accchar_t> (tmp, shift_right);
    res = mli_math_asl_fx(res, shift_left);
    return res;
}

#else
template <>
MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_ADD, false>(
        const vNx2short_t op1,
        const vNx2short_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx2short_t res;
    int shift_right = MAX(shift, 0);
    int shift_left = MAX(-shift, 0);
    int preshift = (shift_right > 0)? 1: 0;

    res = mli_math_add_fx((op1 >> preshift), (op2 >> preshift));
    /* Compensate preshift bit loss. */
    if (shift_right == 1) {
        /* rounding up */
        res += ((op1 | op2) & preshift);
    } else {
        res += ((op1 & op2) & preshift);
    }

    res = mli_math_asr_rnd_fx(res, shift_right - preshift);
    res = mli_math_asl_fx(res, shift_left);
    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_ADD, false>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx4char_t res;
    int shift_right = MAX(shift, 0);
    int shift_left = MAX(-shift, 0);
    int preshift = (shift_right > 0)? 1: 0;

    res = mli_math_add_fx((op1 >> preshift), (op2 >> preshift));
    /* Compensate preshift bit loss. */
    if (shift_right == 1) {
        /* rounding app */
        res += ((op1 | op2) & preshift);
    } else {
        res += ((op1 & op2) & preshift);
    }

    res = mli_math_asr_rnd_fx(res, shift_right - preshift);
    res = mli_math_asl_fx(res, shift_left);
    return res;
}
#endif

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_ADD, true>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx4char_t res;

    /* initialize the accumulator with the input offsets, the output offset, and the rounding value.*/
    int32_t acc_init = -2 * in_offset * scale_factor;
    acc_init += (out_offset << shift);
#ifdef ROUND_UP
    acc_init += ((1 << shift) >> 1); /* rounding half up */
#else
    #error Rounding mode not supported
#endif

    vNx4accint_t acc = mli_math_init_accu<int32_t, vNx4accint_t>(acc_init);
    acc = mli_math_mac_fx(acc, to_vNx4short_t(op1), scale_factor);
    acc = mli_math_mac_fx(acc, to_vNx4short_t(op2), scale_factor);
    /* rounding value is already added as part of the acc_init. therefore it shouldn't
       be added again inside the cast functin.*/
    res = mli_math_acc_cast_fx<vNx4char_t, vNx4accint_t, /*round = */false> (acc, shift);
    return res;
}

template <>
MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_MUL, false>(
        const vNx2short_t op1,
        const vNx2short_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx2short_t res;
    int shift_left = MAX(-shift, 0);
    int shift_right = MAX(shift, 0);

    vNx2accint_t tmp = mli_math_mul_fx<vNx2short_t, vNx2accint_t>(op1, op2);
    vNx2int_t res_32 = mli_math_acc_cast_fx<vNx2int_t, vNx2accint_t>(tmp);
    res_32 = mli_math_asl_fx(res_32, shift_left);
    res = mli_math_cast_fx<vNx2int_t, vNx2short_t>(res_32, shift_right);

    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MUL, false>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx4char_t res;
    int shift_left = MAX(-shift, 0);
    int shift_right = MAX(shift, 0);

    vNx4accshort_t tmp = mli_math_mul_fx<vNx4char_t, vNx4accshort_t>(op1, op2);
    vNx4short_t res_32 = mli_math_acc_cast_fx<vNx4short_t, vNx4accshort_t> (tmp);
    res_32 = mli_math_asl_fx(res_32, shift_left);
    res = mli_math_cast_fx<vNx4short_t, vNx4char_t>(res_32, shift_right);

    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MUL, true>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {

    MLI_ASSERT(shift > 3);
    int preshift = 3;
    int preshift_mask = ((1 << preshift) - 1);

    vNx4char_t res;
    vNx4short_t op1_offset = to_vNx4short_t(op1) - in_offset;
    vNx4short_t op2_offset = to_vNx4short_t(op2) - in_offset;
    vNx4accint_t acc = mli_math_mul_fx<vNx4short_t, vNx4accint_t>(op1_offset, op2_offset);
    vNx4int_t temp = mli_math_acc_cast_fx<vNx4int_t, vNx4accint_t>(acc);

    /*
     * Each operand is 9 bit. The multiplier needs 18 bit. After scaling, 34 bits are needed.
     * By preshifting the multiplier output by 3 bits (2 bits to fit in 16 bits and 1 bit for
     * preshift compensation). We can continue the computation in 16 bits.
     *
     * Note: Minimum shift value is 15
     */
    vNx4short_t temp_16 = to_vNx4short_t(temp >> preshift);
    acc = mli_math_mul_fx<vNx4short_t, vNx4accint_t>(temp_16, scale_factor);
    vNx4int_t comp = (((temp & preshift_mask) * scale_factor) >> preshift);
    acc = mli_math_add(acc, comp);
    vNx4short_t res16 = mli_math_acc_cast_fx<vNx4short_t, vNx4accint_t>(acc, shift - preshift);
    res16 = mli_math_add_fx(res16, vNx4short_t(out_offset));
    res = mli_math_cast_fx<vNx4short_t, vNx4char_t>(res16);

    return res;
}


#if defined(__Xvec_guard_bit_option) && __Xvec_guard_bit_option != 0

template <>
MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_SUB, false>(
        const vNx2short_t op1,
        const vNx2short_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx2short_t res;
    vNx2short_t sub1 = reverse_sub? op2 : op1;
    vNx2short_t sub2 = reverse_sub? op1 : op2;
    int shift_left = MAX(-shift, 0);
    int shift_right = MAX(shift, 0);

    vNx2accshort_t acc = mli_math_init_accu_sub<vNx2short_t, vNx2accshort_t>(sub1, sub2);
    res = mli_math_acc_cast_fx<vNx2short_t, vNx2accshort_t> (acc, shift_right);
    res = mli_math_asl_fx(res, shift_left);

    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_SUB, false>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx4char_t res;
    vNx4char_t sub1 = reverse_sub? op2 : op1;
    vNx4char_t sub2 = reverse_sub? op1 : op2;
    int shift_left = MAX(-shift, 0);
    int shift_right = MAX(shift, 0);

    vNx4accchar_t acc = mli_math_init_accu_sub<vNx4char_t, vNx4accchar_t>(sub1, sub2);
    res = mli_math_acc_cast_fx<vNx4char_t, vNx4accchar_t> (acc, shift_right);
    res = mli_math_asl_fx(res, shift_left);

    return res;
}

#else

template <>
MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_SUB, false>(
        const vNx2short_t op1,
        const vNx2short_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx2short_t res;
    vNx2short_t sub1 = reverse_sub? op2 : op1;
    vNx2short_t sub2 = reverse_sub? op1 : op2;
    int shift_left = MAX(-shift, 0);
    int shift_right = MAX(shift, 0);
    int preshift = (shift_right > 0)? 1: 0;

    res = mli_math_sub_fx((sub1 >> preshift), (sub2 >> preshift));
    /* Compensate preshift bit loss. */
    if (shift_right == 1) {
        /* rounding up */
        res += (sub1 & ~sub2 & preshift);
    } else {
        res -= (~sub1 & sub2 & preshift);
    }

    res = mli_math_asr_rnd_fx(res, shift_right - preshift);
    res = mli_math_asl_fx(res, shift_left);

    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_SUB, false>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx4char_t res;
    vNx4char_t sub1 = reverse_sub? op2 : op1;
    vNx4char_t sub2 = reverse_sub? op1 : op2;
    int shift_left = MAX(-shift, 0);
    int shift_right = MAX(shift, 0);
    int preshift = (shift_right > 0)? 1: 0;

    res = mli_math_sub_fx((sub1 >> preshift), (sub2 >> preshift));
    /* Compensate preshift bit loss. */
    if (shift_right == 1) {
        /* rounding up */
        res += (sub1 & ~sub2 & preshift);
    } else {
        res -= (~sub1 & sub2 & preshift);
    }

    res = mli_math_asr_rnd_fx(res, shift_right - preshift);
    res = mli_math_asl_fx(res, shift_left);

    return res;
}

#endif

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_SUB, true>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx4char_t res;
    vNx4char_t sub1 = reverse_sub? op2 : op1;
    vNx4char_t sub2 = reverse_sub? op1 : op2;

    /* initialize the accumulator with the output offset, and the rounding value.*/
    int32_t acc_init = (out_offset << shift);
#ifdef ROUND_UP
    acc_init += ((1 << shift) >> 1); /* rounding half up */
#else
    #error Rounding mode not supported
#endif
    vNx4accint_t acc = mli_math_init_accu<int32_t, vNx4accint_t>(acc_init);
    vNx4short_t sub_result = mli_math_sub_fx(to_vNx4short_t(sub1), to_vNx4short_t(sub2));
    acc = mli_math_mac_fx(acc, sub_result, scale_factor);
    /* rounding value is already added as part of the acc_init. therefore it shouldn't
       be added again inside the cast functin.*/
    res = mli_math_acc_cast_fx<vNx4char_t, vNx4accint_t, /*round = */false> (acc, shift);

    return res;
}

template <>
MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_MAX, false>(
        const vNx2short_t op1,
        const vNx2short_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx2short_t res;

    int shift_right = MAX(shift,0);
    int shift_left = MAX(-shift,0);
    res = mli_math_max_fx(op1, op2);
    res = mli_math_asl_fx(res, shift_left);
    res = mli_math_asr_rnd_fx(res, shift_right);

    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MAX, false>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx4char_t res ;

    int shift_right = MAX(shift,0);
    int shift_left = MAX(-shift,0);
    res = mli_math_max_fx(op1, op2);
    res = mli_math_asl_fx(res, shift_left);
    res = mli_math_asr_rnd_fx(res, shift_right);

    return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MAX, true>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx4char_t res;
    vNx4short_t max = to_vNx4short_t(mli_math_max_fx(op1, op2));
    max = mli_math_sub_fx(max, (vNx4short_t)in_offset);
    vNx4accint_t acc = mli_math_mul_fx<vNx4short_t, vNx4accint_t>(max, scale_factor);
    vNx4int_t temp = mli_math_acc_cast_fx<vNx4int_t, vNx4accint_t>(acc);
    temp = mli_math_asr_rnd_fx(temp, shift);
    temp = mli_math_add_fx(temp, (vNx4int_t)out_offset);
    res = mli_math_cast_fx<vNx4int_t, vNx4char_t>(temp);
    return res;
}

template <>
MLI_FORCE_INLINE vNx2short_t eltwise_perform_operation<vNx2short_t, vNx2short_t, ELTWISE_MIN, false>(
        const vNx2short_t op1,
        const vNx2short_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
     vNx2short_t res;

     int shift_right = MAX(shift,0);
     int shift_left = MAX(-shift,0);
     res = mli_math_min_fx(op1, op2);
     res = mli_math_asl_fx(res, shift_left);
     res = mli_math_asr_rnd_fx(res, shift_right);

     return res;
}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MIN, false>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx4char_t res ;

    int shift_right = MAX(shift,0);
    int shift_left = MAX(-shift,0);
    res = mli_math_min_fx(op1, op2);
    res = mli_math_asl_fx(res, shift_left);
    res = mli_math_asr_rnd_fx(res, shift_right);

    return res;


}

template <>
MLI_FORCE_INLINE vNx4char_t eltwise_perform_operation<vNx4char_t, vNx4char_t, ELTWISE_MIN, true>(
        const vNx4char_t op1,
        const vNx4char_t op2,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale_factor,
        const int shift,
        bool reverse_sub) {
    vNx4char_t res;
    vNx4short_t min = to_vNx4short_t(mli_math_min_fx(op1, op2));
    min = mli_math_sub_fx(min, (vNx4short_t)in_offset);
    vNx4accint_t acc = mli_math_mul_fx<vNx4short_t, vNx4accint_t>(min, scale_factor);
    vNx4int_t temp = mli_math_acc_cast_fx<vNx4int_t, vNx4accint_t>(acc);
    temp = mli_math_asr_rnd_fx(temp, shift);
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
        const bool scalar_op,
        const int16_t in_offset,
        const int16_t out_offset,
        const int16_t scale,
        const int shift,
        const bool reverse_sub) {
    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(op1_ptr);
    int num_lanes = get_number_lanes(input);
    int remaining_part = count & (num_lanes - 1);


    if (remaining_part) {
        auto val1 = mli_prv_load_1vec(op1_ptr + idx1);
        auto val2 = mli_prv_load_1vec(op2_ptr + idx2);
        val2 = (scalar_op) ? op2_ptr[0] : val2;
        auto res = mli::krn::eltwise_perform_operation<decltype(val1), decltype(val1), func_type, convert>(
                                                       val1, val2, in_offset, out_offset, scale, shift, reverse_sub);
        mli_prv_store_n_samples(&out_ptr[idx_out], res, remaining_part);
        idx1 += remaining_part;
        idx2 += remaining_part;
        idx_out += remaining_part;
    }

    for (int pos = remaining_part; pos < count; pos+=num_lanes) {
        auto val1 = mli_prv_load_1vec(op1_ptr + idx1);
        auto val2 = mli_prv_load_1vec(op2_ptr + idx2);
        /* op1_ptr is always vector, op2_ptr can be scalar or vector. */
        val2 = (scalar_op) ? op2_ptr[0] : val2;
        auto res = mli::krn::eltwise_perform_operation<decltype(val1), decltype(val1), func_type, convert>(
                val1, val2, in_offset, out_offset, scale, shift, reverse_sub);
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
