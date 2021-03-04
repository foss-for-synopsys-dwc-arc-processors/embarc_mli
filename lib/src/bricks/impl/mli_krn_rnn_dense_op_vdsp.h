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
#include "mli_types.h"
#include "mli_prv_dsp.h"
#include "mli_krn_dotprod.h"

namespace mli {
namespace krn {
namespace vdsp {

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

    int num_lanes = get_number_lanes<acc_T>();
    for (int o_idx = 0; o_idx < out_elements; o_idx += num_lanes) {
        int remaining_ch = out_elements - o_idx;
        int current_chs = MIN(remaining_ch, num_lanes); // number of channels computed in this loop iteration

        acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
        acc_T prev_step = mli_math_mul_fx<io_T, acc_T>(0, 0);

        auto output_params = adjust_quant_params_v(&in_to_out_quant_params[0], 0);
        accu = mli::krn::bias_additive(&bias[o_idx], accu, &output_params);

        for(int idx = 0; idx < inputs_num; idx++) {

            output_params = adjust_quant_params_v(&in_to_out_quant_params[idx], 0);
            accu = dotprod_inputzp_1D_v(inputs[idx], &weights[idx][o_idx], accu, in_elements[idx],
                    1, w_ch_out_mem_strides[idx], &in_to_out_quant_params[idx]);
            accu = mli_math_add(accu, prev_step);

            if(inputs_num - idx != 1) {
                mli::krn::ref::adjust_quant_params(&in_to_out_quant_params[idx], o_idx);
                prev_step = mli::krn::ir_rnn_result_requantize(accu, &in_to_out_quant_params[idx],
                                &in_to_out_quant_params[idx + 1], /* krn_idx= */ 0);
                accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
            } else {
                // Cast result to output type with scaling
                mli::krn::result_cast_relu_store_v(&out[o_idx], accu, &output_params, val_min_limit, val_max_limit, current_chs);
            }
        }

    }
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RNN_DENSE_OP_VDSP_H_
