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
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
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
    // in FC we iterate over all elements as it linear array for this reason inner_increment == 1 and 
    // equal for input and wights tensor during output point calculation
    const int inner_increment = 1;

    // Also we look at input kernel as linear array height and ch lenghts == 1
    const int ch_len = 1;
    const int height_len = 1;

    acc_T other_additives = mli_math_mul_fx<io_T, acc_T>(0, 0);
    other_additives  = mli::krn::in_additive(in, other_additives, &quant_params, in_elements, height_len,
            inner_increment, inner_increment);
    other_additives  = mli::krn::zp_additive(&quant_params, other_additives, in_elements);
    

    for (int o_idx = 0; o_idx < out_elements; o_idx++) {
        mli::krn::adjust_quant_params(&quant_params, o_idx);
        acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
        accu = dotprod1D(in, weights, accu, in_elements, 
                         inner_increment, inner_increment);
        accu = mli::krn::weights_additive(weights, accu, &quant_params,
                                in_elements, height_len, ch_len, w_ch_out_mem_stride, inner_increment, inner_increment);
        accu = mli_math_add_fx(accu, other_additives);
        accu = mli::krn::bias_additive(&biases[o_idx], accu, &quant_params);
        
        // Cast result to output type with scaling
        io_T out_val = mli::krn::result_cast<io_T, acc_T, quant_T>(accu, &quant_params);
        out_val = MIN(out_val, val_max_limit);
        out_val = MAX(out_val, val_min_limit);
        out[o_idx] = out_val;
        weights += w_ch_out_mem_stride;
    }
}

//========================================================================================
// Common routin for pre-calculation of various fully connected parameters and running it.
//========================================================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void fully_connected_prepare_and_run(
        const mli_tensor *in,
        const mli_tensor *weights,
        const mli_tensor *bias,
        const mli_fully_connected_cfg *cfg,
        mli_tensor *out) {
    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(io_T) in_ptr = (MLI_PTR(io_T))(in->data.mem.void_p);
    const MLI_PTR(w_T) w_ptr = (MLI_PTR(w_T))(weights->data.mem.void_p);
    const MLI_PTR(b_T) b_ptr = (MLI_PTR(b_T))(bias->data.mem.void_p);
    MLI_CONV_OUT_PTR(io_T) out_ptr = (MLI_CONV_OUT_PTR(io_T))(out->data.mem.void_p);

    const int ch_out = weights->shape[0];
    const int in_sz = mli_prv_count_elem_num(in);

    out->el_type = in->el_type;
    mli_minmax_t val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // fill output tensor parameters
    out->shape[0] = ch_out;
    out->rank = 1;

    // Define quantization specific params
    quant_T params;
    define_quant_params(in, weights, bias, out, &params);
   
   // Define memory stride
    const int w_ch_out_mem_stride_from_tensor = weights->mem_stride[0];
    const int w_ch_out_mem_stride = w_ch_out_mem_stride_from_tensor ?
            w_ch_out_mem_stride_from_tensor : ch_out;

    // Run basic calculation
    //=======================================================================
    mli::krn::inner_product<io_T, w_T, b_T, acc_T, quant_T>(
                                in_ptr, w_ptr, b_ptr, out_ptr, in_sz, ch_out, w_ch_out_mem_stride, 
                                /* cent_area, */ params, (io_T)val_limit.min, (io_T)val_limit.max);
}
#pragma MLI_CODE_SECTION_END()
} // namespace ref
} // namespace krn
} // namespace mli

#endif  //_MLI_KRN_FULLY_CONNECTED_REF_H_
