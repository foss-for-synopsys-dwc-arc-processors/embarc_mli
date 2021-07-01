/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_FULLY_CONNECTED_REF_H_
#define _MLI_KRN_FULLY_CONNECTED_REF_H_

#include "mli_api.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"
#include "mli_math.h"
#include "mli_mem_info.h"
#include "mli_private_types.h"
#include "mli_types.h"
#include "mli_krn_dotprod.h"

namespace mli {
namespace krn {
namespace ref {

#pragma MLI_CODE_SECTION_START(".mli_lib")

//========================================================
// Unified IP (Inner Product) template
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T, bool no_zp>
MLI_FORCE_INLINE void inner_product(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict weights,
        const MLI_PTR(b_T)  __restrict biases,
        MLI_CONV_OUT_PTR(io_T) __restrict out,
        const int in_elements,
        const int out_elements,
        const int w_ch_out_mem_stride,
        quant_T quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit) {
    // Unified Inner Product for both quantization scheme:  MLI_FX (symmetric data, scales are power of two)
    // and s8asym (assymetric data, scales of any value)
    // Calculation implies dotproduct and bias add:
    //            out_val = sum_i(x_r * w_r) + b_r
    //
    // Considering assymetric types(x_r = (x - x_zp) and w_r = (w - w_zp) + b_r
    //                    out_val = sum_i((x-x_zp)*(w-w_zp)) + b_r
    //
    // when we will open brackets:
    //      out_val = sum(x*w) - sum_i(w*x_zp) - sum_i(x*w_zp) + sum_i(w_zp*x_zp) + b_r
    // where:
    //      sum(x*w)       - generic dotproduct which can't be avoided for any type
    //      -sum_i(w*x_zp) - weights_additive. 
    //                       Allways Zero for FX and can be reused in output channel calculations for s8asym
    //      -sum_i(x*w_zp) - in_additive
    //                       Allways Zero for both FX and TF_s8asym assuming symmetric weights (w_zp == 0)
    //     sum_i(w_zp*x_zp)- zp_additive
    //                       Allways Zero for both FX and TF_s8asym assuming symmetric weights (w_zp == 0)
    //      b_r             - bias_additive
    //                        (must be of the same type as accumulator, that may require bias re-quantization)
    //============================================
    acc_T other_additives = mli_math_mul_fx<io_T, acc_T>(0, 0);
    other_additives  = mli::krn::in_additive(in, other_additives, &quant_params, in_elements, 1, 1, 1);
    other_additives  = mli::krn::zp_additive(&quant_params, other_additives, in_elements);
    
    for (int o_idx = 0; o_idx < out_elements; o_idx++) {
        mli::krn::adjust_quant_params(&quant_params, o_idx);
        acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
        accu = dotprod1D(in, &weights[o_idx], accu, in_elements, 
                         1, w_ch_out_mem_stride);
        accu = mli::krn::weights_additive(&weights[o_idx], accu, &quant_params,
                                in_elements, 1, 1, w_ch_out_mem_stride, 1, 1);
        accu = mli_math_add_fx(accu, other_additives);
        accu = mli::krn::bias_additive(&biases[o_idx], accu, &quant_params);

        // Cast result to output type with scaling
        io_T out_val = mli::krn::result_cast<io_T, acc_T, quant_T>(accu, &quant_params);
        out_val = MIN(out_val, val_max_limit);
        out_val = MAX(out_val, val_min_limit);
        out[o_idx] = out_val;
    }
}

//========================================================================================
// Common routin for pre-calculation of various fully connected parameters and running it.
//========================================================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T, bool is_bias_ext>
MLI_FORCE_INLINE void fully_connected_prepare_and_run(
        const mli_tensor *in,
        const mli_tensor *weights,
        const mli_tensor *bias,
        const mli_fully_connected_cfg *cfg,
        mli_tensor *out) {
    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(io_T) in_ptr = mli_prv_tensor_data_ptr<MLI_PTR(io_T)>(in);
    const MLI_PTR(w_T) w_ptr = mli_prv_tensor_data_ptr<MLI_PTR(w_T)>(weights);
    const MLI_PTR(b_T) b_ptr = mli_prv_tensor_data_ptr<MLI_PTR(b_T)>(bias);
    MLI_CONV_OUT_PTR(io_T) out_ptr = mli_prv_tensor_data_ptr<MLI_CONV_OUT_PTR(io_T)>(out);

    const int ch_out = weights->shape[1];
    const int in_sz = mli_prv_count_elem_num(in);

    out->el_type = in->el_type;

    constexpr bool asym = std::is_same<quant_T, s8asym_quant_specific_params>::value;
    mli_minmax_t val_limit = mli_prv_get_relu_limits<io_T, asym>(&cfg->relu, out);

    // fill output tensor parameters
    out->shape[0] = ch_out;
    out->rank = 1;

    // Define quantization specific params
    quant_T params;
    define_quant_params(in, weights, bias, out, &params);
    
    // Various additives might be merged into bias in advance for sa data type.
    // In this case we assign 0 to input zero point to have no effect on final result
    // as bias already "biased" to take it into account. 
    if (is_bias_ext)
        quant_params_set_in_zeropoint(&params, 0);
   
   // Define memory stride
    const int w_ch_out_mem_stride_from_tensor = weights->mem_stride[0];
    const int w_ch_out_mem_stride = (w_ch_out_mem_stride_from_tensor != 0) ?
            w_ch_out_mem_stride_from_tensor : ch_out;

    // Run basic calculation
    //=======================================================================
    mli::krn::inner_product<io_T, w_T, b_T, acc_T, quant_T, is_bias_ext>(
            in_ptr, w_ptr, b_ptr, out_ptr, in_sz, ch_out, w_ch_out_mem_stride, /* cent_area, */ params, (io_T)val_limit.min, (io_T)val_limit.max);
}
#pragma MLI_CODE_SECTION_END()
} // namespace ref
} // namespace krn
} // namespace mli

#endif  //_MLI_KRN_FULLY_CONNECTED_REF_H_
