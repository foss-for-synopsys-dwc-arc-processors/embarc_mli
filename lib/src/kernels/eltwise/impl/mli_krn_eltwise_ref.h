/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_ELTWISE_ADD_REF_H_
#define _MLI_KRN_ELTWISE_ADD_REF_H_

#include "mli_krn_eltwise_decl.h"

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_prv_tensor.h"
#include "mli_prv_dsp.h"
#include "mli_math.h"

#define IN_SCALE_SHIFT 16

namespace mli {
namespace krn {
namespace ref {

//======================================================
//
//======================================================

template <typename in_T, typename out_T, mli_eltwise_type func_type, bool convert>
static MLI_FORCE_INLINE out_T eltwise_perform_operation(
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
        acc = mli_math_mul_fx<int32_t, int64_t> (input1, input2);
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
    for (int pos = 0; pos < count; pos++) {
        /* op1_ptr is always vector, op2_ptr can be scalar or vector.*/
        io_T val1 = scalar_op1? op1_ptr[0] : op1_ptr[idx1];
        io_T val2 = scalar_op2? op2_ptr[0] : op2_ptr[idx2];
        io_T res = mli::krn::eltwise_perform_operation<io_T, io_T, func_type, convert>(
                val1, val2, in_offset1, in_offset2, out_offset, scale1, scale2, pre_op_shift1, pre_op_shift2, post_op_shift);
        out_ptr[idx_out] = res;
        idx1++;
        idx2++;
        idx_out++;
    }
}

template <typename io_T, mli_eltwise_type func_type, bool convert>
static MLI_FORCE_INLINE void eltwise_op_basic(
        const generic_tensor_private_t<MLI_PTR(io_T)> *in1,
        const generic_tensor_private_t<MLI_PTR(io_T)> *in2,
        generic_tensor_private_t<MLI_OUT_PTR(io_T)> *out,
        const int in1_size,
        const int in2_size,
        const int pre_op_shift1,
        const int pre_op_shift2,
        int post_op_shift,
        const struct s8asym_quant_params *in_quant_params1,
        const struct s8asym_quant_params *in_quant_params2,
        const struct s8asym_quant_params *out_quant_params) {

    MLI_PRINTF_FUNC();

    MLI_PTR(io_T) op1_ptr;
    MLI_PTR(io_T) op2_ptr;
    int *shape;
    bool scalar_op1 = (in1_size == 1);
    bool scalar_op2 = (in2_size == 1);
    const generic_tensor_private_t<MLI_PTR(io_T)> *op1 = in1;
    const generic_tensor_private_t<MLI_PTR(io_T)> *op2 = in2;
    shape = (scalar_op2)? ((int *) op1->shape) : ((int *) op2->shape);
    op1_ptr = op1->ptr;
    op2_ptr = op2->ptr;

    int32_t in_scale_fx1 = 0, in_scale_fx2 = 0, out_scale_fx = 0,
            scale_factor1 = 0, scale_factor2 = 0;
    int16_t scale16_1 = 1, scale16_2 = 1;
    int16_t in_offset1 = 0, in_offset2 = 0, out_offset = 0;
    const int kMaxFracBitsFx16 = (sizeof(int16_t) * 8) - 1;
    const int frac_bits_fx16 = kMaxFracBitsFx16;
    /* For SA8 conversion:
     *  out_fx = scale_in * ADD,SUB,MAX,MIN,MUL[(sa8_in1 - off_in1), (sa8_in2 - off_in2)]
     *
     *  out_sa8 = (out_fx / scale_out) + off_out
     *                            ---------- 1 ---------    -------- 2 -------  ---------- 3 ---------     ------- 4 -------    --- 5 ---
     *  out_sa8 = ADD,SUB,MAX,MIN[[scale_in1 / scale_out] * (sa8_in1 - off_in1), [scale_in2 / scale_out] * (sa8_in2 - off_in2)] + [off_out]
     *
     *            ----------------- 1 ---------------   -------------------- 2 --------------------   --- 3 ---
     *  out_sa8 = [(scale_in / scale_out) * scale_in] * MUL[(sa8_in1 - off_in1), (sa8_in2 - off_in2)] + [off_out]
     */
    if (convert) {
        in_offset1 = in_quant_params1->offset;
        in_offset2 = in_quant_params2->offset;
        out_offset = out_quant_params->offset;
        if (func_type == ELTWISE_MUL) {
            in_scale_fx1 = mli_math_asr_rnd_fx<int32_t>(in_quant_params1->scale,
                                                          (int32_t) in_quant_params1->shift - frac_bits_fx16);
            in_scale_fx2 = mli_math_asr_rnd_fx<int32_t>(in_quant_params2->scale,
                                                          (int32_t) in_quant_params2->shift - frac_bits_fx16);
            out_scale_fx = mli_math_asr_rnd_fx<int32_t>(out_quant_params->scale,
                                                           (int32_t) out_quant_params->shift - frac_bits_fx16);
            scale_factor1 = mli_math_asr_rnd_fx<int32_t>(in_scale_fx1, -IN_SCALE_SHIFT);
            scale_factor1 = (scale_factor1 / out_scale_fx) * in_scale_fx2;
            post_op_shift = IN_SCALE_SHIFT + frac_bits_fx16;
            int norm = (scale_factor1 != 0) ? mli_math_norm_fx<int32_t, int>(scale_factor1) : 0;
            int shift = MAX((IN_SCALE_SHIFT - norm), 0);
            scale16_1 = mli_math_cast_fx<int32_t, int16_t>(scale_factor1, shift);
            post_op_shift -= shift;
        } else {
            in_scale_fx1 = mli_math_asr_rnd_fx<int32_t>(in_quant_params1->scale,
                                                           (int32_t) in_quant_params1->shift - frac_bits_fx16);
            in_scale_fx2 = mli_math_asr_rnd_fx<int32_t>(in_quant_params2->scale,
                                                           (int32_t) in_quant_params2->shift - frac_bits_fx16);
            out_scale_fx = mli_math_asr_rnd_fx<int32_t>(out_quant_params->scale,
                                                           (int32_t) out_quant_params->shift - frac_bits_fx16);
            scale_factor1 = mli_math_asr_rnd_fx<int32_t>(in_scale_fx1, -IN_SCALE_SHIFT);
            scale_factor2 = mli_math_asr_rnd_fx<int32_t>(in_scale_fx2, -IN_SCALE_SHIFT);
            scale_factor1 /= out_scale_fx;
            scale_factor2 /= out_scale_fx;
            post_op_shift = IN_SCALE_SHIFT;
            int norm1 = (scale_factor1 != 0) ? mli_math_norm_fx<int32_t, int>(scale_factor1) : 0;
            int norm2 = (scale_factor2 != 0) ? mli_math_norm_fx<int32_t, int>(scale_factor2) : 0;
            int shift = MAX(IN_SCALE_SHIFT - MIN(norm1, norm2), 0);
            scale16_1 = mli_math_cast_fx<int32_t, int16_t>(scale_factor1, shift);
            scale16_2 = mli_math_cast_fx<int32_t, int16_t>(scale_factor2, shift);
            post_op_shift -= shift;
        }
    }

    for (int pos0 = 0; pos0 < shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < shape[2]; pos2++) {
                int pos3 = 0;
                int idx1 = POS(op1, pos0, pos1, pos2, pos3);
                int idx2 = POS(op2, pos0, pos1, pos2, pos3);
                int idx = POS(out, pos0, pos1, pos2, pos3);

                mli::krn::eltwise_innerloop<io_T, func_type, convert>(
                        op1_ptr, op2_ptr, out->ptr, idx1, idx2, idx, shape[3], scalar_op1, scalar_op2, in_offset1,
                        in_offset2, out_offset, scale16_1, scale16_2, pre_op_shift1, pre_op_shift2, post_op_shift);
            } /* pos1 */
        } /* pos2 */
    } /* pos3 */
}

//======================================================
//
//======================================================
template <typename io_T, mli_eltwise_type func_type, bool convert>
static MLI_FORCE_INLINE void eltwise_prepare_and_run(
        const mli_tensor *in1,
        const mli_tensor *in2,
        mli_tensor *out) {

    MLI_PRINTF_FUNC();

    mli_prv_fx_init_dsp_ctrl();

    struct s8asym_quant_params in_quant_params1;
    struct s8asym_quant_params in_quant_params2;
    struct s8asym_quant_params out_quant_params;

    if (convert) {
        in_quant_params1.offset = in1->el_params.sa.zero_point.mem.i16;
        in_quant_params1.scale = in1->el_params.sa.scale.mem.i16;
        in_quant_params1.shift = in1->el_params.sa.scale_frac_bits.mem.i8;
        in_quant_params2.offset = in2->el_params.sa.zero_point.mem.i16;
        in_quant_params2.scale = in2->el_params.sa.scale.mem.i16;
        in_quant_params2.shift = in2->el_params.sa.scale_frac_bits.mem.i8;
        out_quant_params.offset = out->el_params.sa.zero_point.mem.i16;
        out_quant_params.scale = out->el_params.sa.scale.mem.i16;
        out_quant_params.shift = out->el_params.sa.scale_frac_bits.mem.i8;
    }

    /* Extract general parameters for function */
    uint32_t in1_sz = mli_prv_count_elem_num(in1);
    uint32_t in2_sz = mli_prv_count_elem_num(in2);

    /* Extract in/out pointers to mem */
    MLI_PTR(io_T) in1_ptr = (MLI_PTR(io_T))(in1->data.mem.void_p);
    MLI_PTR(io_T) in2_ptr = (MLI_PTR(io_T))(in2->data.mem.void_p);
    MLI_OUT_PTR(io_T) out_ptr = (MLI_OUT_PTR(io_T))(out->data.mem.void_p);

    /* Extract in/out as scalar values */
    io_T in1_scalar = (io_T)(in1->data.mem.i32);
    io_T in2_scalar = (io_T)(in2->data.mem.i32);

    /* Fill output tensor parameters
    //======================================
    */
    const unsigned *shape_ptr = (in1_sz > in2_sz)? in1->shape: in2->shape;
    int rank = (in1_sz > in2_sz)? (int)in1->rank: (int)in2->rank;
    out->rank = rank;
    for (int k = 0; k < rank; k++)
        out->shape[k] = shape_ptr[k];

    auto in1_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in1);
    auto in2_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in2);
    auto out_prv =  mli_prv_get_generic_tensor<MLI_OUT_PTR(io_T)>(out);
    int pre_op_shift1 = 0, pre_op_shift2 = 0, post_op_shift = 0;
    if (func_type == ELTWISE_MUL) {
        post_op_shift = mli_prv_calc_shift(in1, in2, out);
    } else {
        pre_op_shift1 = MIN(in1->el_params.fx.frac_bits -  in2->el_params.fx.frac_bits, 0);
        pre_op_shift2 = MIN(in2->el_params.fx.frac_bits -  in1->el_params.fx.frac_bits, 0);
        post_op_shift = MAX(in1->el_params.fx.frac_bits, in2->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;
    }

    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&in1_prv );
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&in2_prv );
    mli_prv_reorder_generic_tensor<MLI_OUT_PTR(io_T)>(&out_prv);

    in1_prv.ptr = (in1->rank != 0)? in1_ptr: (MLI_PTR(io_T)) &in1_scalar;
    in2_prv.ptr = (in2->rank != 0)? in2_ptr: (MLI_PTR(io_T)) &in2_scalar;
    out_prv.ptr = out_ptr;

    mli::krn::eltwise_op_basic<io_T, func_type, convert>(&in1_prv, &in2_prv, &out_prv,
                                                         in1_sz, in2_sz, pre_op_shift1, pre_op_shift2,
                                                         post_op_shift, &in_quant_params1, &in_quant_params2, &out_quant_params);

}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_ELTWISE_ADD_REF_H_
