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

    vNx2accshort_t tmp = vvcadd_init(op1, op2);

    res = mli_math_acc_cast_fx<vNx2short_t, vNx2accshort_t> (tmp, shift);

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

    vNx4accchar_t tmp = vvcadd_init(op1, op2);

    res = mli_math_acc_cast_fx<vNx4char_t, vNx4accchar_t> (tmp, shift);

    return res;
}

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
    acc_init += out_offset << shift;
#ifdef ROUND_UP
    acc_init += ((shift > 0) ? (1 << (shift-1)) : 0); /* rounding half up */
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
    /* for now only eltwise_add is vectorized. when all functions are vectorized this bool can be removed. */
    bool vectorized = (func_type == ELTWISE_ADD);

    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(op1_ptr);
    int num_lanes = vectorized ? get_number_lanes(input) : 1;
    int remaining_part = count & (num_lanes - 1);

    if (vectorized) {
        if (remaining_part) {
            auto val1 = mli_prv_load_1vec(op1_ptr + idx1);
            auto val2 = mli_prv_load_1vec(op2_ptr + idx2);
            val2 = (scalar_op) ? op2_ptr[0] : val2;
            auto res = mli::krn::eltwise_perform_operation<decltype(val1), decltype(val1), ELTWISE_ADD, convert>(
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
            auto res = mli::krn::eltwise_perform_operation<decltype(val1), decltype(val1), ELTWISE_ADD, convert>(
                    val1, val2, in_offset, out_offset, scale, shift, reverse_sub);
            mli_prv_store_n_samples(&out_ptr[idx_out], res);
            idx1 += num_lanes;
            idx2 += num_lanes;
            idx_out += num_lanes;
        }
    } else {
        mli::krn::ref::eltwise_innerloop<io_T, func_type, convert>(op1_ptr, op2_ptr, out_ptr, idx1, idx2, idx_out, count, scalar_op, in_offset, out_offset, scale, shift, reverse_sub);
    }
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_ELTWISE_ADD_VDSP_H_
