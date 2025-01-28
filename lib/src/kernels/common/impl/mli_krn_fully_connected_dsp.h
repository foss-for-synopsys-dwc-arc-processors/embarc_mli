/*
* Copyright 2019-2025, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_FULLY_CONNECTED_DSP_H_
#define _MLI_KRN_FULLY_CONNECTED_DSP_H_

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_mem_info.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"
#include "math.h"
#include "mli_prv_quant.h"

namespace mli {
namespace krn {
namespace dsp {

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
#ifdef __Xdsp_wide
    if (quant_params_get_weigths_zeropoint(&quant_params) == 0 && (out_elements & 1) == 0) {
        int32_t bias_e = 1 << (((const fx_quant_specific_params *)(void*)&quant_params)->bias_shift);
        if(std::is_same<acc_T, mli_acc40_t>::value && std::is_same<io_T, int16_t>::value && std::is_same<w_T, int16_t>::value && bias_e < ((1 << 15) - 1)) {
            v2q15_t vval_max_limit = fx_replic_v2q15(val_max_limit);
            v2q15_t vval_min_limit = fx_replic_v2q15(val_min_limit);

            int remaining_part = in_elements & 3;
            for (int o_idx = 0; o_idx < out_elements; o_idx += 2) {
                v2i16_t vbias_e = fx_create_v2i16(bias_e, bias_e);
                int32_t out_shift = ((const fx_quant_specific_params *)(void *)&quant_params)->out_shift - 16;
                v2accum40_t accu = fx_v2a40_mpy_v2q15(*(const v2q15_t *__restrict)(void *)&biases[o_idx], vbias_e);

                for (int i = 0; i < in_elements - remaining_part; i += 4) {
                    accu = fx_v2a40_mac_v2q15(accu, fx_replic_v2q15(in[i + 0]), *(const v2q15_t *__restrict)(void *)&weights[o_idx + (i + 0) * w_ch_out_mem_stride]);
                    accu = fx_v2a40_mac_v2q15(accu, fx_replic_v2q15(in[i + 1]), *(const v2q15_t *__restrict)(void *)&weights[o_idx + (i + 1) * w_ch_out_mem_stride]);
                    accu = fx_v2a40_mac_v2q15(accu, fx_replic_v2q15(in[i + 2]), *(const v2q15_t *__restrict)(void *)&weights[o_idx + (i + 2) * w_ch_out_mem_stride]);
                    accu = fx_v2a40_mac_v2q15(accu, fx_replic_v2q15(in[i + 3]), *(const v2q15_t *__restrict)(void *)&weights[o_idx + (i + 3) * w_ch_out_mem_stride]);
                }
                
                for (int r = (in_elements - remaining_part); r < in_elements; r++) {
                    accu = fx_v2a40_mac_v2q15(accu, fx_replic_v2q15(in[r]), *(const v2q15_t *__restrict)(void *)&weights[o_idx + r * w_ch_out_mem_stride]);
                }

                // Cast result to output type, apply built-in ReLU Applying and write result
                v2q15_t out_val = fx_v2q15_cast_nf_asr_rnd_v2a40(accu, out_shift);
                out_val = fx_min_v2q15(out_val, vval_max_limit);
                out_val = fx_max_v2q15(out_val, vval_min_limit);
                    
                v2q15_t *__restrict out_ptr = (v2q15_t *__restrict)(void *)&out[o_idx];
                *out_ptr = out_val;
            }
        } else {
            mli::krn::ref::inner_product<io_T, w_T, b_T, acc_T, quant_T, no_zp>(in, 
                                                                                weights, 
                                                                                biases, 
                                                                                out,
                                                                                in_elements, 
                                                                                out_elements,
                                                                                w_ch_out_mem_stride,
                                                                                quant_params,
                                                                                val_min_limit,
                                                                                val_max_limit);
        }
	}
	else
#endif // #ifdef __Xdsp_wide
	{
        mli::krn::ref::inner_product<io_T, w_T, b_T, acc_T, quant_T, no_zp>(in, 
                                                                            weights, 
                                                                            biases, 
                                                                            out,
                                                                            in_elements, 
                                                                            out_elements,
                                                                            w_ch_out_mem_stride,
                                                                            quant_params,
                                                                            val_min_limit,
                                                                            val_max_limit);
	}

}

#pragma MLI_CODE_SECTION_END()

} // namespace dsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_FULLY_CONNECTED_DSP_H_
