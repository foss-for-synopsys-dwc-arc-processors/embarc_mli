/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_RNN_DENSE_OP_DSP_H_
#define _MLI_KRN_RNN_DENSE_OP_DSP_H_

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_math.h"
#include "mli_mem_info.h"
#include "mli_types.h"
#include "mli_prv_dsp.h"

namespace mli {
namespace krn {
namespace dsp {

// template <typename io_T, typename w_T, typename acc_T, typename quant_T>
// static MLI_FORCE_INLINE void rnn_dense_op(
//         const MLI_PTR (io_T) __restrict in,
//         const MLI_PTR (io_T) __restrict state,
//         const MLI_PTR (w_T) __restrict weights,
//         const MLI_PTR (w_T) __restrict biases,
//         MLI_CONV_OUT_PTR (io_T) __restrict out,
//         const int inp_size,
//         const int s_size,
//         const int ch_out,
//         quant_T in_to_out_quant_params,
//         quant_T in_to_state_quant_params)
// {

//     if (_Rarely (inp_size < 8 && s_size < 8)) {
//         for (int i = 0; i < ch_out; i++) {
//             const MLI_PTR (io_T) __restrict in_ptr = in;
//             const MLI_PTR (io_T) __restrict s_ptr = state;

//             auto ip_out = mli_prv_init_accu_with_bias(in_ptr, *biases++, bias_shift);

//             for (int j = 0; j < inp_size; j++) {
//                 mli_prv_load_mac(&ip_out, in_ptr++, weights++);
//             }

//             ip_out = mli_prv_ashift_accu(ip_out, in_to_state_fraq_dif);
        
//             for (int k = 0; k < s_size; k++) {
//                 mli_prv_load_mac(&ip_out, s_ptr++, weights++);
//             }
        
//             mli_prv_clip_and_store_output(out++, &ip_out, out_shift);
//         }
//     } else if ((inp_size%4 == 0) && (s_size%4 == 0)) {
//         for (int i = 0; i < ch_out; i++) {
//             const MLI_PTR (io_T) __restrict in_ptr = in;
//             const MLI_PTR (io_T) __restrict s_ptr = state;

//             auto ip_out = mli_prv_init_accu_with_bias(in_ptr, *biases++, bias_shift);

// LOOP_PIPELINE_ENABLE
// LOOP_PIPELINE_ENABLE_BACKTRACKING
//             for (int jj = 0; jj < (inp_size/4); jj++) {
//                 mli_prv_load_mac_vec4(&ip_out, in_ptr, weights);
//                 in_ptr += 4;
//                 weights += 4;
//             }

//             ip_out = mli_prv_ashift_accu(ip_out, in_to_state_fraq_dif);

// LOOP_PIPELINE_ENABLE
// LOOP_PIPELINE_ENABLE_BACKTRACKING
//             for (int kk = 0; kk < (s_size/4); kk++) {
//                 mli_prv_load_mac_vec4(&ip_out, s_ptr, weights);
//                 s_ptr += 4;
//                 weights += 4;
//             }
//             s_ptr -= s_size;

//             mli_prv_clip_and_store_output(out++, &ip_out, out_shift);
//         }
//     } else {
//         for (int i = 0; i < ch_out; i++) {
//             const MLI_PTR (io_T) __restrict in_ptr = in;
//             const MLI_PTR (io_T) __restrict s_ptr = state;

//             auto ip_out = mli_prv_init_accu_with_bias(in_ptr, *biases++, bias_shift);

//             for (int j = 0; j < (inp_size&3); j++) {
//                 mli_prv_load_mac(&ip_out, in_ptr++, weights++);
//             }

// LOOP_PIPELINE_ENABLE
// LOOP_PIPELINE_ENABLE_BACKTRACKING
//             for (int jj = 0; jj < (inp_size/4); jj++) {
//                 mli_prv_load_mac_vec4(&ip_out, in_ptr, weights);
//                 in_ptr += 4;
//                 weights += 4;
//             }

//             ip_out = mli_prv_ashift_accu(ip_out, in_to_state_fraq_dif);

//             for (int k = 0; k < (s_size&3); k++) {
//                 mli_prv_load_mac(&ip_out, s_ptr++, weights++);
//             }

// LOOP_PIPELINE_ENABLE
// LOOP_PIPELINE_ENABLE_BACKTRACKING
//             for (int kk = 0; kk < (s_size/4); kk++) {
//                 mli_prv_load_mac_vec4(&ip_out, s_ptr, weights);
//                 s_ptr += 4;
//                 weights += 4;
//             }

//             mli_prv_clip_and_store_output(out++, &ip_out, out_shift);
//         }
//     }
// }

} // namespace dsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RNN_DENSE_OP_DSP_H_
