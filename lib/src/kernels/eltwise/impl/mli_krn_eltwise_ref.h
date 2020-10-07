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
        bool reverse_sub) {
    in_T sub_op1, sub_op2;
    out_T res;

    switch (func_type) {

    case ELTWISE_ADD:
        res = mli_math_add_fx<out_T> (op1, op2);
        break;

    case ELTWISE_SUB:
        sub_op1 = (reverse_sub) ? op2 : op1;
        sub_op2 = (reverse_sub) ? op1 : op2;
        res = mli_math_sub_fx<out_T> (sub_op1, sub_op2);
        break;

    case ELTWISE_MUL:
        res = mli_math_mul_fx<in_T, out_T> (op1, op2);
        break;

    case ELTWISE_MAX:
        res = mli_math_max_fx(op1, op2);
        break;

    case ELTWISE_MIN:
        res = mli_math_min_fx(op1, op2);
        break;

    default:
        MLI_ASSERT(0);
        break;
    }

    return res;
}

template <typename io_T, mli_eltwise_type func_type, bool convert>
static MLI_FORCE_INLINE void eltwise_op_basic(
        const generic_tensor_private_t<MLI_PTR(io_T)> *in1,
        const generic_tensor_private_t<MLI_PTR(io_T)> *in2,
        generic_tensor_private_t<MLI_OUT_PTR(io_T)> *out,
        const int op1_size,
        const int op2_size,
        const int mul_out_shift,
        const struct s8asym_quant_params *in_quant_params,
        const struct s8asym_quant_params *out_quant_params) {

    MLI_PRINTF_FUNC();

    MLI_PTR(io_T) op1_ptr;
    MLI_PTR(io_T) op2_ptr;
    int *shape;
    bool scalar_op = (op1_size == 1 || op2_size == 1);
    const generic_tensor_private_t<MLI_PTR(io_T)> *vec = (op1_size > op2_size) ? in1 : in2;
    bool reverse_sub = false;

    // Simple broadcast (vector on scalar)
    //==============================================
    if (scalar_op) {
        shape = (int *) vec->shape;
        op1_ptr = vec->ptr;
        op2_ptr = (op1_size > op2_size)? (in2->ptr) : (in1->ptr);
        reverse_sub = (func_type == ELTWISE_SUB && op1_ptr == in1->ptr) ? false : true;
    }
    // Elemetnwise between tensors of the same shape
    //==============================================
    else {
        shape = (int *) in1->shape;
        op1_ptr = in1->ptr;
        op2_ptr = in2->ptr;
    }

    int32_t in_scale_fx = 0, out_scale_fx = 0, in_scale_fx_shifted = 0, scale_factor = 0;
    int16_t in_offset = 0, out_offset = 0;
    const int kMaxFracBitsFx16 = (sizeof(int16_t) * 8) - 1;
    const int frac_bits_fx16 = kMaxFracBitsFx16;
    int shift_back = 0;

    /* For SA8 conversion:
     *  out_fx = scale_in * [(sa8_in1 - off_in) + (sa8_in2 - off_in)]
     *  out_sa8 = (out_fx / scale_out) + off_out
     *
     *            --------- 1 ----------   -------------------------- 2 --------------------------   --- 3 ---
     *  out_sa8 = [scale_in / scale_out] * ADD,SUB,MAX,MIN[(sa8_in1 - off_in), (sa8_in2 - off_in)] + [off_out]
     *
     *            ----------------- 1 ---------------   -------------------- 2 --------------------   --- 3 ---
     *  out_sa8 = [(scale_in / scale_out) * scale_in] * MUL[(sa8_in1 - off_in), (sa8_in2 - off_in)] + [off_out]
     */
    if (convert) {
        in_scale_fx = mli_math_acc_ashift_fx<int32_t>(in_quant_params->scale,
                                                      (int32_t) in_quant_params->shift - frac_bits_fx16);
        out_scale_fx = mli_math_acc_ashift_fx<int32_t>(out_quant_params->scale,
                                                       (int32_t) out_quant_params->shift - frac_bits_fx16);

        in_offset = in_quant_params->offset;
        out_offset = out_quant_params->offset;

        in_scale_fx_shifted = mli_math_acc_ashift_fx<int32_t>(in_scale_fx, -IN_SCALE_SHIFT);
        scale_factor = (func_type == ELTWISE_MUL) ? (in_scale_fx_shifted / out_scale_fx) * in_scale_fx :
                                                    (in_scale_fx_shifted / out_scale_fx);

        shift_back = (func_type == ELTWISE_MUL) ? IN_SCALE_SHIFT + frac_bits_fx16 : IN_SCALE_SHIFT;
    }

    for (int pos0 = 0; pos0 < shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < shape[2]; pos2++) {
                for (int pos3 = 0; pos3 < shape[3]; pos3++) {

                    int idx1 = scalar_op ? POS(vec, pos0, pos1, pos2, pos3) : POS(in1, pos0, pos1, pos2, pos3);
                    int idx2 = scalar_op ? 0 : POS(in2, pos0, pos1, pos2, pos3);
                    int idx = POS(out, pos0, pos1, pos2, pos3);

                    io_T op1 = op1_ptr[idx1];
                    io_T op2 = op2_ptr[idx2];

                    if (convert) {
                        /* convert to FX16 */
                        int16_t input1 = mli_math_sub_fx<int16_t> (op1, in_offset);
                        int16_t input2 = mli_math_sub_fx<int16_t> (op2, in_offset);

                        int32_t res = eltwise_perform_operation<int16_t, int32_t, func_type, true>(
                                input1, input2, reverse_sub);

                        /* convert to SA8 */
                        int64_t acc64 = mli_math_mul_fx<int32_t, int64_t> (res, scale_factor);
                        int16_t acc = mli_math_cast_fx<int64_t, int16_t>(acc64, shift_back);
                        acc = mli_math_add_fx<int16_t>(acc, out_offset);
                        out->ptr[idx] = mli_math_cast_fx<int16_t, int8_t>(acc, 0);

                    } else {
                        int32_t res = eltwise_perform_operation<io_T, int32_t, func_type, false>(
                                op1, op2, reverse_sub);
                        /* Adjusting output fractional bits */
                        out->ptr[idx] = mli_math_acc_cast_fx<io_T, int32_t> (res, mul_out_shift);;
                    }
                }
            }
        }
    } //for loop

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

    struct s8asym_quant_params in_quant_params;
    struct s8asym_quant_params out_quant_params;

    if (convert) {
        in_quant_params.offset = in1->el_params.sa.zero_point.mem.i16;
        in_quant_params.scale = in1->el_params.sa.scale.mem.i16;
        in_quant_params.shift = in1->el_params.sa.scale_frac_bits.mem.i8;
        out_quant_params.offset = out->el_params.sa.zero_point.mem.i16;
        out_quant_params.scale = out->el_params.sa.scale.mem.i16;
        out_quant_params.shift = out->el_params.sa.scale_frac_bits.mem.i8;
    }

    // Extract general parameters for function
    uint32_t in1_sz = mli_prv_count_elem_num(in1);
    uint32_t in2_sz = mli_prv_count_elem_num(in2);

    // Extract in/out pointers to mem
    MLI_PTR(io_T) in1_ptr = (MLI_PTR(io_T))(in1->data.mem.void_p);
    MLI_PTR(io_T) in2_ptr = (MLI_PTR(io_T))(in2->data.mem.void_p);
    MLI_OUT_PTR(io_T) out_ptr = (MLI_OUT_PTR(io_T))(out->data.mem.void_p);

    // Extract in/out as scalar values
    io_T in1_scalar = (io_T)(in1->data.mem.i32);
    io_T in2_scalar = (io_T)(in2->data.mem.i32);

    // Calc outshift for MUL operation
    const int out_shift = (func_type == ELTWISE_MUL)? mli_prv_calc_shift(in1, in2, out):
                                                      in1->el_params.fx.frac_bits - out->el_params.fx.frac_bits;

    // Fill output tensor parameters
    //======================================
    const unsigned *shape_ptr = (in1_sz > in2_sz)? in1->shape: in2->shape;
    int rank = (in1_sz > in2_sz)? (int)in1->rank: (int)in2->rank;
    out->rank = rank;
    for (int k = 0; k < rank; k++)
        out->shape[k] = shape_ptr[k];

    auto in1_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in1);
    auto in2_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in2);
    auto out_prv =  mli_prv_get_generic_tensor<MLI_OUT_PTR(io_T)>(out);

    in1_prv.ptr = (in1->rank != 0)? in1_ptr: (MLI_PTR(io_T)) &in1_scalar;
    in2_prv.ptr = (in2->rank != 0)? in2_ptr: (MLI_PTR(io_T)) &in2_scalar;
    out_prv.ptr = out_ptr;

    mli::krn::eltwise_op_basic<io_T, func_type, convert>(
            &in1_prv, &in2_prv, &out_prv,
            in1_sz, in2_sz, out_shift, &in_quant_params, &out_quant_params);

    out->el_type = in1->el_type;
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_ELTWISE_ADD_REF_H_
