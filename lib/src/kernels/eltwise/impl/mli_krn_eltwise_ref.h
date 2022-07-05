/*
 * Copyright 2019-2021, Synopsys, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-3-Clause license found in
 * the LICENSE file in the root directory of this source tree.
 *
 */

#ifndef _MLI_KRN_ELTWISE_ADD_REF_H_
#define _MLI_KRN_ELTWISE_ADD_REF_H_

#include "mli_krn_eltwise_decl.h"
#include "mli_prv_tensor.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_prv_tensor.h"
#include "mli_prv_dsp.h"
#include "mli_math.h"
#include "mli_mem_info.h"


#define MUL_MAX_SHIFT 31
/*
 * For max/min shifting more than 23 is not needed
 * as the scaled result ((max - in_offset) * scale) will be limited by 24 bits including the sign bit.
 */
#define MAX_MIN_UPPER_LIMIT_SHIFT 23

namespace mli {
namespace krn {
namespace ref {

//======================================================
//
//======================================================

template <typename in_T, typename out_T, mli_eltwise_type func_type, bool convert>
out_T eltwise_perform_operation(
        const in_T op1,
        const in_T op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    out_T res = 0;
    typedef typename std::conditional<convert, int32_t, in_T>::type op_T;
    typedef typename std::conditional<(func_type == ELTWISE_MAX) ||
            (func_type == ELTWISE_MIN), op_T, int64_t>::type accu_T;
    accu_T acc;
    int32_t input1, input2;

    if (convert) {

        input1 = mli_math_sub_fx<int16_t> (op1, in_offset1);
        input2 = mli_math_sub_fx<int16_t> (op2, in_offset2);
        input1 = mli_math_mul_fx<int16_t, int32_t>((int16_t) input1, scale_factor1);
        input2 = mli_math_mul_fx<int16_t, int32_t>((int16_t) input2, scale_factor2);
        if (func_type == ELTWISE_ADD || func_type == ELTWISE_SUB) {
            input1 = mli_math_ashift_right_fx<int32_t>(input1, pre_op_shift1);
            input2 = mli_math_ashift_right_fx<int32_t>(input2, pre_op_shift2);
        }
    } else {
        input1 = mli_math_ashift_right_fx<int32_t>(op1, pre_op_shift1);
        input2 = mli_math_ashift_right_fx<int32_t>(op2, pre_op_shift2);
    }


    switch (func_type) {

    case ELTWISE_ADD:
        acc = mli_math_add_fx<int32_t> (input1, input2);
        break;

    case ELTWISE_SUB:
        acc = mli_math_sub_fx<int32_t> (input1, input2);
        break;

    case ELTWISE_MUL:
        acc = (accu_T)(mli_math_mul_fx<int32_t, int64_t> (input1, input2));
        break;

    case ELTWISE_MAX:
        acc = mli_math_max_fx(input1, input2);
        break;

    case ELTWISE_MIN:
        acc = mli_math_min_fx(input1, input2);
        break;

    default:
        MLI_ASSERT(0);
        break;
    }

    if (convert) {
        int16_t tmp16 = mli_math_cast_fx<accu_T, int16_t>(acc, post_op_shift);
        tmp16 = mli_math_add_fx<int16_t>(tmp16, out_offset);
        res = mli_math_cast_fx<int16_t, int8_t>(tmp16, 0);
    } else {
        res = mli_math_cast_fx<accu_T, out_T> (acc, post_op_shift);
    }

    return res;
}


template <>
MLI_FORCE_INLINE int8_t eltwise_perform_operation<int8_t, int8_t, ELTWISE_MUL, true>(
        const int8_t op1,
        const int8_t op2,
        const int16_t in_offset1,
        const int16_t in_offset2,
        const int16_t out_offset,
        const int16_t scale_factor1,
        const int16_t scale_factor2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift) {
    int8_t res = 0;
    int32_t acc;
    int32_t input1, input2;

    input1 = mli_math_sub_fx<int16_t> (op1, in_offset1);
    input2 = mli_math_sub_fx<int16_t> (op2, in_offset2);

    acc = (int32_t)mli_math_mul_fx<int32_t, int64_t> (input1, input2);
    const int headroom = 3;
    const int acc_len = 32;
    const int out_len = 8;
    const int target_out_shift = acc_len - out_len - headroom;
    const int preshift = mli_math_min_fx(mli_math_max_fx(post_op_shift - target_out_shift, 0), headroom);
    const int shift = post_op_shift - preshift;
    int16_t acc_result = mli_math_cast_fx<int32_t, int16_t>(acc, preshift);
    int32_t acc_scaled = mli_math_mul_fx<int16_t, int32_t> (acc_result, scale_factor1);
    int16_t tmp16 = mli_math_cast_fx<int32_t, int16_t>(acc_scaled, shift);
    tmp16 = mli_math_add_fx<int16_t>(tmp16, out_offset);
    res = mli_math_cast_fx<int16_t, int8_t>(tmp16, 0);

    return res;
}

template <typename i_T, typename o_T, mli_eltwise_type func_type, bool convert>
void eltwise_innerloop(
        const MLI_PTR(i_T) __restrict  op1_ptr,
        const MLI_PTR(i_T) __restrict op2_ptr,
        MLI_PTR(o_T) __restrict out_ptr,
        int idx1,
        int idx2,
        int idx_out,
        const int count,
        const i_T op1_s,
        const i_T op2_s,
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
    for (int pos = 0; pos < count; pos++) {
        /* op1_ptr is always vector, op2_ptr can be scalar or vector.*/
        i_T val1 = (scalar_op1)? op1_s : op1_ptr[idx1];
        i_T val2 = (scalar_op2)? op2_s : op2_ptr[idx2];
        o_T res = mli::krn::eltwise_perform_operation<i_T, o_T, func_type, convert>(
                val1, val2, in_offset1, in_offset2, out_offset, scale1, scale2, pre_op_shift1, pre_op_shift2, post_op_shift);
        out_ptr[idx_out] = res;
        idx1++;
        idx2++;
        idx_out++;
    }
}

template <typename i_T, typename o_T, mli_eltwise_type func_type, bool convert>
void eltwise_op_basic(
        const generic_tensor_private_t<MLI_PTR(i_T)> * __restrict in1,
        const generic_tensor_private_t<MLI_PTR(i_T)> * __restrict in2,
        generic_tensor_private_t<MLI_OUT_PTR(o_T)> * __restrict out,
        const i_T op1_s,
        const i_T op2_s,
        const bool scalar_op1,
        const bool scalar_op2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift,
        const int scale16_1,
        const int scale16_2,
        const int in_offset1,
        const int in_offset2,
        const int out_offset) {

    MLI_PRINTF_FUNC();
    int *shape = (scalar_op2)? ((int *) in1->shape) : ((int *) in2->shape);
    for (int pos0 = 0; pos0 < shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < shape[2]; pos2++) {
                int pos3 = 0;
                int idx1 = POS(in1, pos0, pos1, pos2, pos3);
                int idx2 = POS(in2, pos0, pos1, pos2, pos3);
                int idx = POS(out, pos0, pos1, pos2, pos3);
                mli::krn::eltwise_innerloop<i_T, o_T, func_type, convert>(
                        in1->ptr, in2->ptr, out->ptr, idx1, idx2, idx, shape[3],
                        op1_s, op2_s, scalar_op1, scalar_op2, in_offset1, in_offset2,
                        out_offset, scale16_1, scale16_2, pre_op_shift1, pre_op_shift2, post_op_shift);
            } /* pos1 */
        } /* pos2 */
    } /* pos3 */
}

template <typename i_T, typename o_T, mli_eltwise_type func_type, bool convert>
void eltwise_op_basic(
        const mli_tensor * __restrict in1,
        const mli_tensor * __restrict in2,
        mli_tensor * __restrict out,
        const int *shape,
        const i_T op1_s,
        const i_T op2_s,
        const bool scalar_op1,
        const bool scalar_op2,
        const int pre_op_shift1,
        const int pre_op_shift2,
        const int post_op_shift,
        const int scale16_1,
        const int scale16_2,
        const int in_offset1,
        const int in_offset2,
        const int out_offset) {

    MLI_PRINTF_FUNC();
    MLI_PTR(i_T) in1_ptr = nullptr;
    MLI_PTR(i_T) in2_ptr = nullptr;
    if (!scalar_op2)
        in2_ptr = mli_prv_tensor_data_ptr<MLI_PTR(i_T)>(in2);

    if (!scalar_op1)
        in1_ptr = mli_prv_tensor_data_ptr<MLI_PTR(i_T)>(in1);
    auto out_ptr = mli_prv_tensor_data_ptr<MLI_PTR(o_T)>(out);

    mli::krn::eltwise_innerloop<i_T, o_T, func_type, convert>(
            in1_ptr, in2_ptr, out_ptr, 0, 0, 0, shape[0],
            op1_s, op2_s, scalar_op1, scalar_op2, in_offset1, in_offset2,
            out_offset, scale16_1, scale16_2, pre_op_shift1, pre_op_shift2, post_op_shift);
}

//======================================================
//
//======================================================
struct convert_params {
    bool scalar_op1 = 0;
    bool scalar_op2 = 0;
    int pre_op_shift1 = 0;
    int pre_op_shift2 = 0;
    int post_op_shift = 0;
    int16_t scale16_1 = 1;
    int16_t scale16_2 = 1;
    int16_t in_offset1 = 0;
    int16_t in_offset2 = 0;
    int16_t out_offset = 0;
};

template <typename io_T, mli_eltwise_type func_type, bool convert>
void calc_convert_params(const mli_tensor* in1, const mli_tensor* in2, const mli_tensor* out, convert_params& params) {
    int32_t scale_factor1 = 0, scale_factor2 = 0;
    int16_t scale_1 = 1, scale_2 = 1, scale_out = 1,
            shift1 = 0, shift2 = 0, shift_out = 0;

    if (convert) {
        params.in_offset1 = in1->el_params.sa.zero_point.mem.i16;
        params.in_offset2 = in2->el_params.sa.zero_point.mem.i16;
        params.out_offset = out->el_params.sa.zero_point.mem.i16;
        scale_1 = in1->el_params.sa.scale.mem.i16;
        scale_2 = in2->el_params.sa.scale.mem.i16;
        scale_out = out->el_params.sa.scale.mem.i16;
        shift1 = in1->el_params.sa.scale_frac_bits.mem.i8;
        shift2 = in2->el_params.sa.scale_frac_bits.mem.i8;
        shift_out = out->el_params.sa.scale_frac_bits.mem.i8;
        if (func_type == ELTWISE_MAX || func_type == ELTWISE_MIN) {
            int shift;
            int32_t scale_factor = mli_math_norm_cast_fx<int32_t, int32_t>((int32_t)scale_1, &shift);
            scale_factor = scale_factor / scale_out;
            params.post_op_shift = shift1 - shift_out - shift;
            params.scale16_1 = mli_math_norm_cast_fx<int32_t, int16_t>(scale_factor, &shift);
            params.post_op_shift -= shift;
            shift = MAX(params.post_op_shift - MAX_MIN_UPPER_LIMIT_SHIFT, 0) + MIN(MUL_MAX_SHIFT + params.post_op_shift, 0);
            params.scale16_1 = mli_math_asr_rnd_fx<int16_t>(params.scale16_1, shift);
            params.post_op_shift -= shift;
            params.scale16_2 = params.scale16_1;
        } else if (func_type == ELTWISE_MUL) {
            int shift;
            scale_factor1 = scale_1 * scale_2;
            scale_factor1 = mli_math_norm_cast_fx<int32_t, int32_t>(scale_factor1, &shift);
            scale_factor1 = (scale_factor1 / scale_out);
            params.post_op_shift = shift1 + shift2 - shift_out - shift;
            params.scale16_1 = mli_math_norm_cast_fx<int32_t, int16_t>(scale_factor1, &shift);
            params.post_op_shift -= shift;
            shift = MAX(params.post_op_shift - MUL_MAX_SHIFT, 0) + MIN(MUL_MAX_SHIFT + params.post_op_shift, 0);
            params.scale16_1 = mli_math_asr_rnd_fx<int16_t>(params.scale16_1, shift);
            params.post_op_shift -= shift;
        } else {
            int norm_shift1, norm_shift2;
            scale_factor1 = mli_math_norm_cast_fx<int32_t, int32_t>((int32_t)scale_1, &norm_shift1);
            scale_factor2 = mli_math_norm_cast_fx<int32_t, int32_t>((int32_t)scale_2, &norm_shift2);
            scale_factor1 /= scale_out;
            scale_factor2 /= scale_out;
            params.pre_op_shift1 = -norm_shift1 + shift1 - shift_out;
            params.pre_op_shift2 = -norm_shift2 + shift2 - shift_out;
            params.scale16_1 = mli_math_norm_cast_fx<int32_t, int16_t>(scale_factor1, &norm_shift1);
            params.scale16_2 = mli_math_norm_cast_fx<int32_t, int16_t>(scale_factor2, &norm_shift2);
            params.pre_op_shift1 -= norm_shift1;
            params.pre_op_shift2 -= norm_shift2;
            shift1 = MAX(params.pre_op_shift1 - MAX_MIN_UPPER_LIMIT_SHIFT, 0) + MIN(MUL_MAX_SHIFT + params.pre_op_shift1, 0);
            shift2 = MAX(params.pre_op_shift2 - MAX_MIN_UPPER_LIMIT_SHIFT, 0) + MIN(MUL_MAX_SHIFT + params.pre_op_shift2, 0);
            params.scale16_1 = mli_math_asr_rnd_fx<int16_t>(params.scale16_1, shift1);
            params.scale16_2 = mli_math_asr_rnd_fx<int16_t>(params.scale16_2, shift2);
            params.pre_op_shift1 -= shift1;
            params.pre_op_shift2 -= shift2;
        }
    } else {
        constexpr int byte_size = 8;
        /*
         * max_shift will be determined according to the size of the out register to avoid
         * overflow in the rounding value.
         */
        int max_shift = sizeof(io_T) * byte_size;
        if (func_type == ELTWISE_MUL) {
            max_shift = 2 * max_shift - 1;
            params.post_op_shift = mli_prv_calc_shift(in1, in2, out);
        } else if (func_type == ELTWISE_MIN || func_type == ELTWISE_MAX) {
            max_shift = max_shift - 1;
            params.post_op_shift = in1->el_params.fx.frac_bits - out->el_params.fx.frac_bits;
        } else {
            max_shift = 2 * max_shift - 1;
            params.pre_op_shift1 = MIN(in1->el_params.fx.frac_bits -  in2->el_params.fx.frac_bits, 0);
            params.pre_op_shift2 = MIN(in2->el_params.fx.frac_bits -  in1->el_params.fx.frac_bits, 0);
            params.post_op_shift = MAX(in1->el_params.fx.frac_bits, in2->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;
            MLI_EXTRA_ASSERT(params.pre_op_shift1 > -max_shift);
            MLI_EXTRA_ASSERT(params.pre_op_shift2 > -max_shift);
        }
        params.post_op_shift = MIN(params.post_op_shift, max_shift);
    }
}

template <typename i_T, typename o_T, mli_eltwise_type func_type, bool convert, bool no_scalar , bool no_out_update,  bool shape_1d >
void eltwise_prepare_and_run(
        const mli_tensor * in1,
        const mli_tensor * in2,
        mli_tensor * out) {

    MLI_PRINTF_FUNC();
    mli_prv_fx_init_dsp_ctrl();

    // initial values for conversion
    convert_params params = {
        .scalar_op1 = 0, .scalar_op2 = 0,
        .pre_op_shift1 = 0, .pre_op_shift2 = 0, .post_op_shift = 0,
        .scale16_1 = 1, .scale16_2 = 1,
        .in_offset1 = 0, .in_offset2 = 0, .out_offset = 0};

    if constexpr(std::is_same<i_T, o_T>::value) {
        calc_convert_params<i_T, func_type, convert>(in1, in2, out, params);
    } else {
        // TODO: uncomment when ACC type enums have been merged
        // MLI_ASSERT(out->el_type == MLI_EL_ACC_BASE || out->el_type == MLI_EL_ACC_WIDE);
    }

    if (no_scalar){
        // assumption that always 0
        params.scalar_op1 = 0;
        params.scalar_op2 = 0;
    } else {
        /* Extract general parameters for function */
        params.scalar_op1 = (mli_prv_count_elem_num(in1) == 1);
        params.scalar_op2 = (mli_prv_count_elem_num(in2) == 1);
    }

    /* Extract in/out as scalar values */
    i_T in1_scalar = mli_prv_tensor_data_val<i_T>(in1);
    i_T in2_scalar = mli_prv_tensor_data_val<i_T>(in2);

    int flatten_count = 0;

    if (shape_1d) {
        //assumption that the condition ((in1->mem_stride[0] == in1->shape[1]) &&
        //                               (in2->mem_stride[0] == in1->shape[1]) &&
        //                               (out->mem_stride[0] == in1->shape[1])) is always true
        if (no_scalar){
           flatten_count = in1->shape[in1->rank - 1];
        } else {
           flatten_count = 0;
           if (params.scalar_op1 && !params.scalar_op2) {
                flatten_count = in2->shape[in2->rank - 1];
            } else if (!params.scalar_op1) {
              flatten_count = in1->shape[in1->rank - 1];
            } else {
              flatten_count = 1;
            }
        }

        MLI_PTR(i_T) in1_ptr = nullptr;
        MLI_PTR(i_T) in2_ptr = nullptr;
        if (!params.scalar_op2)
            in2_ptr = mli_prv_tensor_data_ptr<MLI_PTR(i_T)>(in2);

        if (!params.scalar_op1)
            in1_ptr = mli_prv_tensor_data_ptr<MLI_PTR(i_T)>(in1);

        auto out_ptr = mli_prv_tensor_data_ptr<MLI_PTR(o_T)>(out);

        mli::krn::eltwise_innerloop<i_T, o_T, func_type, convert>(
                    in1_ptr, in2_ptr , out_ptr, 0, 0, 0, flatten_count,
                    in1_scalar, in2_scalar, params.scalar_op1, params.scalar_op2,
                    params.in_offset1, params.in_offset2, params.out_offset,
                    params.scale16_1, params.scale16_2,
                    params.pre_op_shift1, params.pre_op_shift2, params.post_op_shift);

        return;
    } else {
        flatten_count = 0;
        if (params.scalar_op1 && !params.scalar_op2) {
            flatten_count = mli_prv_squash_tensor_to_one_dim(in2, out);
        } else if (!params.scalar_op1 && params.scalar_op2) {
            flatten_count = mli_prv_squash_tensor_to_one_dim(in1, out);
        } else if (!params.scalar_op1 && !params.scalar_op2) {
            flatten_count = mli_prv_squash_tensor_to_one_dim(in1, in2, out);
        } else {
            flatten_count = 1;
        }

        if (flatten_count) {
            mli::krn::eltwise_op_basic<i_T, o_T, func_type, convert>(in1, in2, out, &flatten_count,
                                    in1_scalar, in2_scalar, params.scalar_op1, params.scalar_op2,
                                    params.pre_op_shift1, params.pre_op_shift2, params.post_op_shift,
                                    params.scale16_1, params.scale16_2,
                                    params.in_offset1, params.in_offset2, params.out_offset);
            return;
        }
    }

    auto out_prv = mli_prv_get_generic_tensor<MLI_PTR(o_T)>(out);
    generic_tensor_private_t<MLI_PTR(i_T)>  in1_prv;
    generic_tensor_private_t<MLI_PTR(i_T)>  in2_prv;
    if (params.scalar_op1 && !params.scalar_op2) {
        in2_prv = mli_prv_get_generic_tensor<MLI_PTR(i_T)>(in2);
        mli_prv_squash_generic_tensor<MLI_PTR(i_T), MLI_PTR(o_T)>(&in2_prv, &out_prv);
    } else if (!params.scalar_op1 && params.scalar_op2) {
        in1_prv = mli_prv_get_generic_tensor<MLI_PTR(i_T)>(in1);
        mli_prv_squash_generic_tensor<MLI_PTR(i_T), MLI_PTR(o_T)>(&in1_prv, &out_prv);
    } else if (!params.scalar_op1 && !params.scalar_op2) {
        in2_prv = mli_prv_get_generic_tensor<MLI_PTR(i_T)>(in2);
        in1_prv = mli_prv_get_generic_tensor<MLI_PTR(i_T)>(in1);
        mli_prv_squash_generic_tensor<MLI_PTR(i_T), MLI_PTR(o_T)>(&in1_prv, &in2_prv, &out_prv);
    }

    mli::krn::eltwise_op_basic<i_T, o_T, func_type, convert>(&in1_prv, &in2_prv, &out_prv,
                                        in1_scalar, in2_scalar, params.scalar_op1, params.scalar_op2,
                                        params.pre_op_shift1, params.pre_op_shift2, params.post_op_shift,
                                        params.scale16_1, params.scale16_2,
                                        params.in_offset1, params.in_offset2, params.out_offset);

}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_ELTWISE_ADD_REF_H_
