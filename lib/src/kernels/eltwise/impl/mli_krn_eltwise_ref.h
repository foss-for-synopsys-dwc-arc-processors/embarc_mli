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

namespace mli {
namespace krn {
namespace ref {

//======================================================
//
//======================================================
template <typename io_T, mli_eltwise_type func_type>
static MLI_FORCE_INLINE void eltwise_op_basic_fx(
        const io_T* op1,
        const io_T* op2,
              io_T* out,

        const int op1_size,
        const int op2_size,
        const int mul_out_shift) {

    MLI_PRINTF_FUNC();

    // Simple broadcast (vector on scalar)
    //==============================================
    if (op1_size == 1 || op2_size == 1) {
        const io_T broadcast_val = (op1_size > op2_size)? (*op2): (*op1);
        const io_T *vec = (op1_size > op2_size)? op1: op2;
        const int out_size = MAX(op1_size, op2_size);

        if (func_type == ELTWISE_ADD) {
            for (int idx = 0; idx < out_size; idx++)
                out[idx] = mli_math_add_fx(vec[idx], broadcast_val);
        } else if (func_type == ELTWISE_SUB) {
            if (vec == op1) {
                // Vector minus scalar
                for (int idx = 0; idx < out_size; idx++)
                    out[idx] = mli_math_sub_fx(vec[idx], broadcast_val);
            } else {
                // Scalar minus Vector
                for (int idx = 0; idx < out_size; idx++)
                    out[idx] = mli_math_sub_fx(broadcast_val, vec[idx]);
            }
        } else if (func_type == ELTWISE_MUL) {
            for (int idx = 0; idx < out_size; idx++) {
                mli_acc32_t acc = mli_math_mul_fx<io_T, mli_acc32_t> (vec[idx], broadcast_val);
                out[idx] = mli_math_acc_cast_fx<io_T, mli_acc32_t> (acc, mul_out_shift);
            }
        } else if (func_type == ELTWISE_MAX) {
            for (int idx = 0; idx < out_size; idx++)
                out[idx] = mli_math_max_fx(vec[idx], broadcast_val);
        } else if (func_type == ELTWISE_MIN) {
            for (int idx = 0; idx < out_size; idx++)
                out[idx] = mli_math_min_fx(vec[idx], broadcast_val);
        }
    }

    // Elemetnwise between tensors of the same shape
    //==============================================
    else {
        if (func_type == ELTWISE_ADD) {
            for (int idx = 0; idx < op1_size; idx++)
                out[idx] = mli_math_add_fx(op1[idx], op2[idx]);
        } else if (func_type == ELTWISE_SUB) {
            for (int idx = 0; idx < op1_size; idx++)
                out[idx] = mli_math_sub_fx(op1[idx], op2[idx]);
        } else if (func_type == ELTWISE_MUL) {
            for (int idx = 0; idx < op1_size; idx++) {
                mli_acc32_t acc = mli_math_mul_fx<io_T, mli_acc32_t>(op1[idx], op2[idx]);
                out[idx] = mli_math_acc_cast_fx<io_T, mli_acc32_t>(acc, mul_out_shift);
            }
        } else if (func_type == ELTWISE_MAX) {
            for (int idx = 0; idx < op1_size; idx++)
                out[idx] = mli_math_max_fx(op1[idx], op2[idx]);
        } else if (func_type == ELTWISE_MIN) {
            for (int idx = 0; idx < op1_size; idx++)
                out[idx] = mli_math_min_fx(op1[idx], op2[idx]);
        }
    }
}

//======================================================
//
//======================================================
template <typename io_T, mli_eltwise_type func_type>
static MLI_FORCE_INLINE void eltwise_prepare_and_run_fx(
        const mli_tensor *in1,
        const mli_tensor *in2,
        mli_tensor *out) {

    MLI_PRINTF_FUNC();

    mli_prv_fx_init_dsp_ctrl();

    // Extract general parameters for function
    uint32_t in1_sz = mli_prv_count_elem_num(in1);
    uint32_t in2_sz = mli_prv_count_elem_num(in2);

    // Extract in/out pointers to mem
    const io_T *in1_ptr = static_cast<io_T *>(in1->data.mem.void_p);
    const io_T *in2_ptr = static_cast<io_T *>(in2->data.mem.void_p);
    io_T *out_ptr       = static_cast<io_T *>(out->data.mem.void_p);

    // Extract in/out as scalar values
    const io_T in1_scalar = (io_T)((intptr_t)(in1->data.mem.void_p));
    const io_T in2_scalar = (io_T)((intptr_t)(in2->data.mem.void_p));
    io_T out_scalar = 0;

    // Calc outshift for MUL operation
    const int out_shift = (func_type == ELTWISE_MUL)? mli_prv_calc_shift(in1, in2, out): 0;

    mli::eltwise_op_basic_fx<io_T, func_type>(
            (in1->rank != 0)? in1_ptr: &in1_scalar,
            (in2->rank != 0)? in2_ptr: &in2_scalar,
            (out->data.capacity > 0)? out_ptr: &out_scalar,
            in1_sz, in2_sz, out_shift);

    // Fill output tensor parameters
    //======================================
    if (out->data.capacity == 0) {
        // In case we calculated 1 scalar value
        out->data.mem.void_p = mli_math_cast_scalar_to_ptr_fx<io_T>(out_scalar);
        out->rank = 0;
    } else {
        const unsigned *shape_ptr = (in1_sz > in2_sz)? in1->shape: in2->shape;
        int rank       = (in1_sz > in2_sz)? (int)in1->rank: (int)in2->rank;
        for (int k = 0; k < rank; k++)
            out->shape[k] = shape_ptr[k];
        out->rank = rank;
    }
    if (func_type != ELTWISE_MUL)
        out->el_params.fx.frac_bits = in1->el_params.fx.frac_bits;
    out->el_type = in1->el_type;
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_ELTWISE_ADD_REF_H_