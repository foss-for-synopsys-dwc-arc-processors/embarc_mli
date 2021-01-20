/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_RNN_DENSE_OP_DECL_REF_H_
#define _MLI_KRN_RNN_DENSE_OP_DECL_REF_H_

#include "mli_config.h"
#include "mli_prv_layout.h"
#include "mli_prv_quant.h"
#include "mli_types.h"

namespace mli {
namespace krn {
////////////////////////////////////////////////////////////////////////////////
// Functions (in *_ref/*_dsp/*vdsp) that can be called from outside their own
// file must be declared here. This includes all overloads. For example, if we
// have: io_T f(io_T a) and int8_t f(int8_t a), then both must be declared.
// Not doing so, can cause the compiler to use the wrong overload.
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// REF
////////////////////////////////////////////////////////////////////////////////
namespace ref {
        
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
static MLI_FORCE_INLINE void rnn_dense_op(
        const MLI_PTR(io_T) __restrict * inputs,
        const MLI_PTR(w_T) __restrict * weights,
        const MLI_PTR(b_T) __restrict bias,
        MLI_CONV_OUT_PTR(io_T) __restrict out,
        const int inputs_num,
        const int * in_elements,
        const int out_elements,
        const int * w_ch_out_mem_strides,
        quant_T * in_to_out_quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit);

template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
static MLI_FORCE_INLINE void rnn_dense_op_stacked(
        const MLI_PTR (io_T) * inputs_ptr,
        const mli_tensor ** weights,
        const mli_tensor * bias,
        const int gates_num,
        const int inputs_num,
        const int * inputs_elements,
        quant_T * in_to_out_quant_params,
        const int * w_ch_out_mem_strides,
        mli_tensor * out);

} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {

template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
static inline void rnn_dense_op(
        const MLI_PTR(io_T) __restrict * inputs,
        const MLI_PTR(w_T) __restrict * weights,
        const MLI_PTR(b_T) __restrict bias,
        MLI_CONV_OUT_PTR(io_T) __restrict out,
        const int inputs_num,
        const int * in_elements,
        const int out_elements,
        const int * w_ch_out_mem_strides,
        quant_T * in_to_out_quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit);

} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {
        
} // namespace vdsp

} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RNN_DENSE_OP_DECL_REF_H_
