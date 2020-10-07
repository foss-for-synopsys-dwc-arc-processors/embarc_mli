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
template <typename io_T>
static MLI_FORCE_INLINE void eltwise_op_add_fx (
        const io_T* op1,
        const io_T* op2,
        io_T* out,
        const int op1_size,
        const int op2_size) {

    MLI_PRINTF_FUNC();

    // Simple broadcast (vector on scalar)
    //==============================================
    if (op1_size == 1 || op2_size == 1) {
        const io_T broadcast_val = (op1_size > op2_size) ? (*(const io_T *)op2) : (*(const io_T *)op1);
        const MLI_PTR(io_T) vec = (op1_size > op2_size) ? (MLI_PTR(io_T))op1 : (MLI_PTR(io_T))op2;
        const int out_size = (op1_size > op2_size) ? op1_size : op2_size; // TODO TODO MAX(...)

        for (int idx = 0; idx < out_size; idx++) {
            out[idx] = mli_math_add_fx<io_T>(vec[idx], broadcast_val);
        }

    }
    // Elemetnwise between tensors of the same shape
    //==============================================
    else {
        MLI_ASSERT(op1_size == op2_size);

        for (int idx = 0; idx < op1_size; idx++) {
            out[idx] = mli_math_add_fx<io_T>(op1[idx], op2[idx]);
        }
    }
}

template <typename io_T, mli_eltwise_type func_type>
static MLI_FORCE_INLINE void eltwise_op_basic(
        const io_T* op1,
        const io_T* op2,
              io_T* out,

        const int op1_size,
        const int op2_size,
        const int mul_out_shift) {

    MLI_PRINTF_FUNC();

    if (func_type == ELTWISE_ADD) {
        mli::krn::vdsp::eltwise_op_add_fx(op1, op2, out, op1_size, op2_size);
    } else {
        mli::krn::ref::eltwise_op_basic<io_T, func_type>(op1, op2, out, op1_size, op2_size, mul_out_shift);
    }
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_ELTWISE_ADD_VDSP_H_
