/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_RNN_DENSE_OP_REF_H_
#define _MLI_KRN_RNN_DENSE_OP_REF_H_

#include <type_traits>

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_math.h"
#include "mli_mem_info.h"
#include "mli_types.h"
#include "mli_prv_dsp.h"
#include "mli_krn_dotprod.h"
#include "mli_prv_tensor.h"

namespace mli {
namespace krn {
namespace ref {

static inline void adjust_weights_dim_for_rnn_dense(fx_quant_specific_params* params) {
	return;
}

static inline void adjust_weights_dim_for_rnn_dense(s8asym_quant_specific_params* params) {
	params->weight_dim = -1;
}

static inline void adjust_weights_scale_for_rnn_dense(
    fx_quant_specific_params* params, 
    fx_quant_specific_params* initial_params) {
	return;
}

static inline void adjust_weights_scale_for_rnn_dense(
    s8asym_quant_specific_params* params, 
    s8asym_quant_specific_params* initial_params) {
	if (initial_params->weight_dim != -1) {
        params->weight_scales++;
        params->weight_shifts++;
    }
}

static inline void adjust_weights_scale_back_for_rnn_dense(
    fx_quant_specific_params* params, 
    fx_quant_specific_params* initial_params, 
    int gates) {
	return;
}

static inline void adjust_weights_scale_back_for_rnn_dense(
    s8asym_quant_specific_params* params, 
    s8asym_quant_specific_params* initial_params, 
    int gates) {
	if(initial_params->weight_dim != -1) {
        params->weight_scales -= gates;
        params->weight_shifts -= gates;
    }
}

template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
static inline void rnn_dense_op_stacked(
        const MLI_PTR (io_T) * inputs_ptr,
        const mli_tensor ** weights,
        const mli_tensor * bias,
        const int gates_num,
        const int inputs_num,
        const int * inputs_elements,
        quant_T * in_to_out_quant_params,
        const int * w_ch_out_mem_strides,
        const int * w_gate_mem_strides,
        mli_tensor * out) {

    constexpr bool asym = std::is_same<quant_T, s8asym_quant_specific_params>::value;

    mli_relu_cfg relu_none = {MLI_RELU_NONE};
    mli_minmax_t val_limit = mli_prv_get_relu_limits<io_T, asym>(&relu_none, out);

    const MLI_PTR (w_T) weights_ptr[MLI_RNN_MAX_INPUT];
    quant_T initial_params[MLI_RNN_MAX_INPUT];
    uint32_t weights_shift[MLI_RNN_MAX_INPUT];

    int out_elements = mli_prv_count_elem_num_part(bias, 1);

    for(int idx = 0; idx < inputs_num; ++idx) {
        weights_ptr[idx] = mli_prv_tensor_data_ptr<MLI_PTR (w_T)>(weights[idx]);
        weights_shift[idx] = w_gate_mem_strides[idx];
        initial_params[idx] = in_to_out_quant_params[idx];
        adjust_weights_dim_for_rnn_dense(&in_to_out_quant_params[idx]);
    }

    const MLI_PTR (b_T) bias_ptr = mli_prv_tensor_data_ptr<MLI_PTR (b_T)>(bias);
    MLI_CONV_OUT_PTR (io_T) dense_out_ptr = mli_prv_tensor_data_ptr<MLI_CONV_OUT_PTR (io_T)>(out);

    for (int gate = 0; gate < gates_num; ++gate) {
        mli::krn::ref::rnn_dense_op<io_T, w_T, b_T, acc_T, quant_T>(
            inputs_ptr, weights_ptr, bias_ptr, dense_out_ptr, inputs_num, inputs_elements,
            out_elements, w_ch_out_mem_strides, in_to_out_quant_params, 
            (io_T)val_limit.min, (io_T)val_limit.max);

        for (int weight_idx = 0; weight_idx < inputs_num; ++weight_idx) {
            weights_ptr[weight_idx] += weights_shift[weight_idx];
            adjust_weights_scale_for_rnn_dense(&in_to_out_quant_params[weight_idx], &initial_params[weight_idx]);
        }
        
        bias_ptr += out_elements;
        dense_out_ptr += out_elements;
    }

    for (int weight_idx = 0; weight_idx < inputs_num; ++weight_idx) {
        weights_ptr[weight_idx] -= gates_num * weights_shift[weight_idx];
        adjust_weights_scale_back_for_rnn_dense(&in_to_out_quant_params[weight_idx], &initial_params[weight_idx], gates_num);
    }

    bias_ptr -= gates_num * out_elements;
    dense_out_ptr -= gates_num * out_elements;
}

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
        const io_T val_max_limit) {

    acc_T other_additives[MLI_RNN_MAX_INPUT];

    for (int idx = 0; idx < inputs_num; idx++) {
        other_additives[idx] = mli_math_mul_fx<io_T, acc_T>(0, 0);
        other_additives[idx] = mli::krn::in_additive(inputs[idx], other_additives[idx], &in_to_out_quant_params[idx], 
                                in_elements[idx], /* col_step= */ 1, /* row_step= */ 1, /* ch_step= */ 1);
        other_additives[idx] = mli::krn::zp_additive(&in_to_out_quant_params[idx], other_additives[idx], 
                                in_elements[idx]);
    }

    for (int o_idx = 0; o_idx < out_elements; o_idx++) {

        acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
        acc_T acc_ir = mli_math_mul_fx<io_T, acc_T>(0, 0);
        acc_T acc_res_ir = mli_math_mul_fx<io_T, acc_T>(0, 0);

        accu = mli::krn::bias_additive(&bias[o_idx], accu, &in_to_out_quant_params[0]);

        for(int idx = 0; idx < inputs_num; idx++) {
            mli::krn::ref::adjust_quant_params(&in_to_out_quant_params[idx], /* krn_idx= */ 0);

            accu = dotprod1D(inputs[idx], &weights[idx][o_idx], accu, in_elements[idx], 
                         1, w_ch_out_mem_strides[idx]);

            accu = mli::krn::ref::weights_additive(&weights[idx][o_idx], accu, &in_to_out_quant_params[idx],
                    in_elements[idx], /* height= */ 1, /* ch= */ 1, w_ch_out_mem_strides[idx], 
                    /* row_step= */ 1, /* ch_step= */ 1);
            accu = mli_math_add_fx(accu, other_additives[idx]);

            acc_ir = mli::krn::ir_rnn_result_requantize<acc_T>(accu, &in_to_out_quant_params[idx]);
            acc_res_ir = mli_math_add_fx(acc_res_ir, acc_ir);
            accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
        }

        out[o_idx] = mli::krn::ir_result_cast_relu_store<io_T, acc_T, quant_T>(acc_res_ir,
        		&in_to_out_quant_params[inputs_num - 1], val_min_limit, val_max_limit);
    }
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RNN_DENSE_OP_REF_H_
