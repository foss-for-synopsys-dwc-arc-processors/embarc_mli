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
#include "mli_types.h"
#include "mli_prv_dsp.h"
#include "mli_krn_dotprod.h"

namespace mli {
namespace krn {
namespace ref {

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
        mli_tensor * out) {

    constexpr bool asym = std::is_same<quant_T, s8asym_quant_specific_params>::value;

    mli_relu_cfg relu_none = {MLI_RELU_NONE};
    mli_minmax_t val_limit = mli_prv_get_relu_limits<io_T, asym>(&relu_none, out);

    const MLI_PTR (w_T) weights_ptr[MLI_RNN_MAX_INPUT];
    uint32_t weights_shift[MLI_RNN_MAX_INPUT];

    const int16_t * weights_scales[MLI_RNN_MAX_INPUT];
    const int8_t * weights_scale_frac_bits[MLI_RNN_MAX_INPUT];

    int out_elements = mli_prv_count_elem_num_part(bias, 1);

    for(int idx = 0; idx < inputs_num; ++idx) {
        weights_ptr[idx] = (const MLI_PTR (w_T)) weights[idx]->data.mem.void_p;
        weights_shift[idx] = mli_prv_count_elem_num_part(weights[idx], 1);

        weights_scales[idx] = weights[idx]->el_params.sa.scale.mem.pi16;
        weights_scale_frac_bits[idx] = weights[idx]->el_params.sa.scale_frac_bits.mem.pi8;
    }

    const MLI_PTR (b_T) bias_ptr = (const MLI_PTR (b_T)) bias->data.mem.void_p;
    MLI_CONV_OUT_PTR (io_T) dense_out_ptr = (MLI_CONV_OUT_PTR (io_T)) out->data.mem.void_p;

    for (int gate = 0; gate < gates_num; ++gate) {
        mli::krn::rnn_dense_op<io_T, w_T, b_T, acc_T, quant_T>(
            inputs_ptr, weights_ptr, bias_ptr, dense_out_ptr, inputs_num, inputs_elements,
            out_elements, w_ch_out_mem_strides, in_to_out_quant_params, 
            (io_T)val_limit.min, (io_T)val_limit.max);

        for (int weight_idx = 0; weight_idx < inputs_num; ++weight_idx)
            weights_ptr[weight_idx] += weights_shift[weight_idx];
        
        bias_ptr += out_elements;
        dense_out_ptr += out_elements;

        if (asym) {
            for (int weight_idx = 0; weight_idx < inputs_num; ++weight_idx) {
                weights_scales[weight_idx]++;
                weights_scale_frac_bits[weight_idx]++;
            }      
        }
    }

    for (int weight_idx = 0; weight_idx < inputs_num; ++weight_idx)
        weights_ptr[weight_idx] -= gates_num * weights_shift[weight_idx];

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
        io_T out_val = 0; 
        acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
        acc_T prev_step = mli_math_mul_fx<io_T, acc_T>(0, 0);
        accu = mli::krn::bias_additive(&bias[o_idx], accu, &in_to_out_quant_params[0]);

        for(int idx = 0; idx < inputs_num; idx++) {
            mli::krn::adjust_quant_params(&in_to_out_quant_params[idx], /* krn_idx= */ 0);

            accu = dotprod1D(inputs[idx], &weights[idx][o_idx], accu, in_elements[idx], 
                         1, w_ch_out_mem_strides[idx]);

            accu = mli::krn::weights_additive(&weights[idx][o_idx], accu, &in_to_out_quant_params[idx],
                    in_elements[idx], /* height= */ 1, /* ch= */ 1, w_ch_out_mem_strides[idx], 
                    /* row_step= */ 1, /* ch_step= */ 1);
            accu = mli_math_add_fx(accu, other_additives[idx]);
            accu = mli_math_add_fx(accu, prev_step);

            if(inputs_num - idx != 1) {
                prev_step = mli::krn::ir_rnn_result_requantize(accu, &in_to_out_quant_params[idx], 
                                &in_to_out_quant_params[idx+1], /* krn_idx= */ 0);
                accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
            } else {
                out_val = mli::krn::result_cast<io_T, acc_T, quant_T>(accu, &in_to_out_quant_params[idx]);
            }
        }

        out_val = MIN(out_val, val_max_limit);
        out_val = MAX(out_val, val_min_limit);
        out[o_idx] = out_val;
    }
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RNN_DENSE_OP_REF_H_
