/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_RNN_DENSE_OP_VDSP_H_
#define _MLI_KRN_RNN_DENSE_OP_VDSP_H_

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_math.h"
#include "mli_mem_info.h"
#include "mli_types.h"
#include "mli_prv_dsp.h"
#include "mli_krn_dotprod.h"

namespace mli {
namespace krn {
namespace vdsp {

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
        rnn_dense_op<io_T, w_T, b_T, acc_T, quant_T>(
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

MLI_FORCE_INLINE vNx4int_t mli_math_add_accus(vNx4int_t L, vNx4int_t R) {
    return mli_math_add_fx(L, R);
}

MLI_FORCE_INLINE vNx2accint_t mli_math_add_accus(vNx2accint_t L, vNx2accint_t R) {
	return mli_math_add(L, R);
}

MLI_FORCE_INLINE vNx4accint_t mli_math_add_accus(vNx4accint_t L, vNx4accint_t R) {
	return mli_math_add(L, R);
}

MLI_FORCE_INLINE vNx4accshort_t mli_math_add_accus(vNx4accshort_t L, vNx4accshort_t R) {
#if (__Xvec_guard_bit_option == 0)
	vNx4short_t L_short = mli_math_acc_cast_fx<vNx4short_t, vNx4accshort_t>(L);
	vNx4short_t R_short = mli_math_acc_cast_fx<vNx4short_t, vNx4accshort_t>(R);

	vNx4short_t res = mli_math_add_fx<vNx4short_t>(L_short, R_short);

	return mli_math_init_accu_add<vNx4short_t, vNx4accshort_t>(res, (vNx4short_t)0);
#else
	return mli_math_add(L, R);
#endif
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
    typedef typename std::conditional<std::is_same<acc_T, vNx4accshort_t>::value, vNx4int_t, acc_T>::type ir_T;
    int num_lanes = get_number_lanes<acc_T>();

    for (int o_idx = 0; o_idx < out_elements; o_idx += num_lanes) {
        int remaining_ch = out_elements - o_idx;
        int current_chs = MIN(remaining_ch, num_lanes); // number of channels computed in this loop iteration

        acc_T accu = mli_prv_init_accu<acc_T>();
        ir_T acc_ir = mli_prv_init_accu<ir_T>();
        ir_T acc_res_ir = mli_prv_init_accu<ir_T>();

        auto output_params = adjust_quant_params_v(&in_to_out_quant_params[0], 0);
        accu = mli::krn::bias_additive(&bias[o_idx], accu, &output_params, /* add_preshift_rnd */ false);

        for(int idx = 0; idx < inputs_num; idx++) {

            accu = dotprod_inputzp_1D_v(inputs[idx], &weights[idx][o_idx], accu, in_elements[idx],
                    1, w_ch_out_mem_strides[idx], &in_to_out_quant_params[idx]);

            /* TODO: can be optimized using adjust_quant_params_v, and also optimize ir_rnn_result_requantize function */
            mli::krn::ref::adjust_quant_params(&in_to_out_quant_params[idx], o_idx);
            acc_ir = mli::krn::ir_rnn_result_requantize<acc_T, ir_T>(accu, &in_to_out_quant_params[idx]);

            acc_res_ir = mli_math_add_accus(acc_res_ir, acc_ir);
            accu = mli_prv_init_accu<acc_T>();
        }

        // Cast result to output type with scaling
        mli::krn::ir_result_cast_relu_store_v(&out[o_idx], acc_res_ir, &in_to_out_quant_params[0],
                                val_min_limit, val_max_limit, current_chs);
    }
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RNN_DENSE_OP_VDSP_H_
