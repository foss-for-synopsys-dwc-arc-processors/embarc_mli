/*
//
// CONFIDENTIAL AND PROPRIETARY INFORMATION
//
// Copyright (c) 2018 Synopsys, Inc. All rights reserved.
// This software and documentation contain confidential and
// proprietary information that is the property of
// Synopsys, Inc. The software and documentation are
// furnished under a license agreement and may be used
// or copied only in accordance with the terms of the license
// agreement. No part of the software and documentation
// may be reproduced, transmitted, or translated, in any
// form or by any means, electronic, mechanical, manual,
// optical, or otherwise, without prior written permission
// of Synopsys, Inc., or as expressly provided by the license agreement.
// Reverse engineering is prohibited, and reproduction,
// disclosure or use without specific written authorization
// of Synopsys Inc. is strictly forbidden.
//
//
*/
#ifndef _MLI_KRN_CONV2D_HWC_H_
#define _MLI_KRN_CONV2D_HWC_H_

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "mli_api.h"
#include "mli_krn_dotprod_chw.h"
#include "mli_krn_reduce_sum2d.h"
#include "mli_math.h"
#include "mli_prv_aux_calc.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"
#include "mli_types.h"

//========================================================
// Depthwise convolution 2D template
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T>
static void depthwise_convolution2D_hwc(
        const io_T* __restrict in_ftrs,
        const w_T*  __restrict weights,
        const b_T*  __restrict biases,
              io_T* __restrict out_ftrs,

        const rect_t * const perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,

        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left
        ) {

    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;
    const int in_col_step = mli_prv_column_step<LAYOUT_HWC>(in_height, in_width, in_ch, /*filters =*/ 1);
    const int in_row_step = mli_prv_row_step<LAYOUT_HWC>(in_height, in_width, in_ch, /*filters =*/ 1);
    const int krn_col_step = mli_prv_column_step<LAYOUT_HWC>(kernel_height, kernel_width, /*channels =*/ 1, out_ch);
    const int krn_row_step = mli_prv_row_step<LAYOUT_HWC>(kernel_height, kernel_width, /*channels =*/ 1, out_ch);
    const int ch_mul = out_ch / in_ch;

    for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
        for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
            // Define area of input and filter for convolution
            // comp - compensation values for valid area definition
            mli_compensations comp = mli_prv_valid_area_compensations(
                    H_idx, W_idx, in_height, in_width, kernel_height, kernel_width, 
                    stride_height, stride_width, padding_left, padding_top);

            const int rows = kernel_height - comp.top - comp.bottom;
            const int clmns = kernel_width - comp.right - comp.left;
            const int h_idx_in = (H_idx * stride_height - padding_top + comp.top);
            const int w_idx_in = (W_idx * stride_width - padding_left + comp.left);

            // Next loops is subject for vectorization.
            // Cases with channel multiplier (rare) and without might be vectorized slightly different.
            // without channel multiplier - similar to pooling
            // with channel multiplier - similar to convolution with HWCN layout for weights
            for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++) {
                const io_T *in_ptr = in_ftrs; 
                in_ptr += mli_prv_calc_index<LAYOUT_HWC>(in_height, in_width, in_ch, /*filters =*/ 1,
                                                                  h_idx_in, w_idx_in, in_ch_idx);

                acc_T other_additives = mli_math_mul_fx<io_T, acc_T>(0, 0);
                other_additives  = zp_additive(&quant_params, other_additives, clmns * rows);
                other_additives  = in_additive(in_ptr, other_additives, &quant_params, 
                                               clmns, rows, in_col_step, in_row_step);
                for (int ch_mult_idx = 0; ch_mult_idx < ch_mul; ch_mult_idx++) {
                    const int out_ch_idx = in_ch_idx * ch_mul + ch_mult_idx;
                    const w_T *w_ptr = weights + mli_prv_calc_index<LAYOUT_HWC>(
                            kernel_height, kernel_width, /*channels =*/ 1, 
                            out_ch, comp.top, comp.left, 0, out_ch_idx);
                    adjust_quant_params(&quant_params, out_ch_idx);

                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
                    accu = dotprod2D(in_ptr, w_ptr, accu, clmns, rows,
                                        in_col_step, in_row_step, krn_col_step, krn_row_step);
                    accu = weights_additive(w_ptr, accu, &quant_params, clmns, rows, krn_col_step, krn_row_step);
                    accu = bias_additive(biases[out_ch_idx], accu, &quant_params);
                    accu = mli_math_add_fx(accu, other_additives);
                    
                    // Cast result to output type
                    io_T out_val = result_cast<io_T, acc_T, s8asym_quant_specific_params>(accu, &quant_params);

                    // built-in ReLU Applying and result writing 
                    out_val = MIN(out_val, val_max_limit);
                    out_val = MAX(out_val, val_min_limit);
                    io_T* out_ptr = out_ftrs;
                    out_ptr += mli_prv_calc_index<LAYOUT_HWC>(out_height, out_width, out_ch, /*filters =*/ 1,
                                                                       H_idx, W_idx, out_ch_idx);
                    *out_ptr = out_val;
                } // for ch_mult_idx 
            } // for in_ch_idx
        } // for W_idx
    } // for H_idx
}




#endif // _MLI_KRN_CONV2D_HWC_H_
