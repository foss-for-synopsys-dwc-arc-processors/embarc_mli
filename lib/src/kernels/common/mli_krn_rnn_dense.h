/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_RNN_DENSE_H_
#define _MLI_KRN_RNN_DENSE_H_

#include "mli_api.h"
#include "mli_math.h"
#include "mli_private_types.h"
#include "mli_prv_quant.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

#include "mli_krn_rnn_dense_op.h"

namespace mli {
namespace krn {

#pragma MLI_CODE_SECTION_START(".mli_lib")

//========================================================================================
// Common routine for pre-calculation of various basic rnn cell parameters and running it.
//========================================================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void rnn_dense_prepare_and_run(
        const mli_tensor **inputs,
        const mli_tensor **weights,
        const mli_tensor *bias,
        const mli_rnn_dense_cfg *cfg,
        mli_tensor *out) {

    const int inputs_num = cfg->inputs_num;

    int inputs_elements[MLI_RNN_MAX_INPUT];
    const MLI_PTR (io_T) inputs_ptr[MLI_RNN_MAX_INPUT];
    const MLI_PTR (w_T) weights_ptr[MLI_RNN_MAX_INPUT];
    const MLI_PTR (b_T) bias_ptr = mli_prv_tensor_data_ptr<MLI_PTR (b_T)>(bias);

    for(int idx = 0; idx < inputs_num; ++idx) {
        inputs_elements[idx] = static_cast<int>(mli_prv_count_elem_num(inputs[idx]));
        inputs_ptr[idx] = mli_prv_tensor_data_ptr<MLI_PTR (io_T)>(inputs[idx]);
        weights_ptr[idx] = mli_prv_tensor_data_ptr<MLI_PTR (w_T)>(weights[idx]);
    }

    const int out_elements = static_cast<int>(mli_prv_count_elem_num(bias));

    constexpr bool asym = std::is_same<quant_T, s8asym_quant_specific_params>::value;
    mli_relu_cfg relu_none = {MLI_RELU_NONE};
    mli_minmax_t val_limit = mli_prv_get_relu_limits<io_T, asym>(&relu_none, out);

    MLI_CONV_OUT_PTR (io_T) out_ptr = mli_prv_tensor_data_ptr<MLI_CONV_OUT_PTR (io_T)>(out);
    quant_T in_to_out_params[MLI_RNN_MAX_INPUT];

    int input_idx = 0;
    for(; input_idx < inputs_num; input_idx++) {
        define_quant_params(inputs[input_idx], weights[input_idx], bias, out, &in_to_out_params[input_idx]);
    }

    int w_ch_out_mem_stride_from_tensors[MLI_RNN_MAX_INPUT];
    int w_ch_out_mem_strides[MLI_RNN_MAX_INPUT];

    for(int idx = 0; idx < inputs_num; ++idx) {
        w_ch_out_mem_stride_from_tensors[idx] =  weights[idx]->mem_stride[0];
        w_ch_out_mem_strides[idx] = (w_ch_out_mem_stride_from_tensors[idx] != 0) ?
            w_ch_out_mem_stride_from_tensors[idx] : weights[idx]->shape[1];
    }

    // Applying Dense
    //=======================================
    mli::krn::rnn_dense_op<io_T, w_T, b_T, acc_T, quant_T>(
        inputs_ptr, weights_ptr, bias_ptr, out_ptr, inputs_num, inputs_elements,
        out_elements, w_ch_out_mem_strides, in_to_out_params, 
        (io_T)val_limit.min, (io_T)val_limit.max);

    out->rank = bias->rank;
    for (uint32_t k = 0; k < bias->rank; k++)
        out->shape[k] = bias->shape[k];

}

#pragma MLI_CODE_SECTION_END()
} // namespace krn
} // namespace mli

#endif  //_MLI_KRN_RNN_DENSE_H_
