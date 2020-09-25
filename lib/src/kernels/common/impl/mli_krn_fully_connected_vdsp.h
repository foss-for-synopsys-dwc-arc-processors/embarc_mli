/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_FULLY_CONNECTED_VDSP_H_
#define _MLI_KRN_FULLY_CONNECTED_VDSP_H_

#include "mli_api.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"
#include "mli_math.h"
#include "mli_private_types.h"
#include "mli_types.h"
#include "mli_krn_dotprod.h"

namespace mli {
namespace krn {
namespace vdsp {

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
    MLI_ASSERT(quant_params_get_weigths_zeropoint(&quant_params) == 0);
    int num_lanes = get_number_lanes<acc_T>();
    for (int o_idx = 0; o_idx < out_elements; o_idx += num_lanes) {
        int remaining_ch = out_elements - o_idx;
        int current_chs = MIN(remaining_ch, num_lanes); // nr channels computed in this loop iteration
        auto output_params = adjust_quant_params_v(&quant_params, o_idx);

        acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
        accu = mli::krn::bias_additive(&biases[o_idx], accu, &quant_params); // bias has to be first in optimized code.

        accu = dotprod_inputzp_1D_v(in, &weights[o_idx], accu, in_elements, 1, w_ch_out_mem_stride, &quant_params);

        // Cast result to output type with scaling
        mli::krn::result_cast_relu_store_v(&out[o_idx], accu, &output_params, val_min_limit, val_max_limit, current_chs);
    }
}

#pragma MLI_CODE_SECTION_END()
} // namespace vdsp
} // namespace krn
} // namespace mli

#endif  //_MLI_KRN_FULLY_CONNECTED_VDSP_H_
