/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_GROUP_CONV2D_VDSP_H_
#define _MLI_KRN_GROUP_CONV2D_VDSP_H_

#include "mli_api.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"
#include "mli_math.h"
#include "mli_mem_info.h"
#include "mli_private_types.h"
#include "mli_types.h"
#include "mli_krn_dotprod.h"
#include "mli_prv_layout.h"
#include "mli_krn_convolution.h"

namespace mli {
namespace krn {
namespace vdsp {

#pragma MLI_CODE_SECTION_START(".mli_lib")

MLI_FORCE_INLINE s8asym_quant_specific_params quant_params_offset(s8asym_quant_specific_params &params,
		int offset) {
    s8asym_quant_specific_params params_prv = params;
    if (params.weight_dim < 0) {
    	params_prv.weight_scales = &params.weight_scales[0];
    	params_prv.weight_shifts = &params.weight_shifts[0];
    } else {
    	params_prv.weight_scales = params.weight_scales + offset;
    	params_prv.weight_shifts = params.weight_shifts + offset;
    }

    return params_prv;
}

MLI_FORCE_INLINE fx_quant_specific_params quant_params_offset(fx_quant_specific_params &in,
		int offset) {
    // No need to adjust something during calculations for MLI_FX specific quantization
    return in;
}

//========================================================
// Group Convolution 2D without padding
//========================================================
template <typename i_T, typename w_T, typename o_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void group_convolution2D_nopad(
        const tensor_private_t<MLI_PTR(i_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &weights,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(o_T)> &out,
        const rect_t &perception_area,
        quant_T quant_params,
        const o_T val_min_limit,
        const o_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    const int group_count = in.ch / weights.in_ch;
    const int filters_per_group = weights.out_ch / group_count;

    tensor_private_t<MLI_CONV_OUT_PTR(o_T)> out_prv = out;
    tensor_private_t<MLI_CONV_OUT_PTR(i_T)> in_prv = in;
    conv2d_weights_tensor_private_t<MLI_PTR(w_T)> weights_prv = weights;
    const MLI_PTR(b_T)  __restrict biases_prv = biases;
    out_prv.ch = filters_per_group;
    in_prv.ch = weights.in_ch;

    for (int M = 0; M < group_count; M++) {

        int group_offset = M * filters_per_group;
        in_prv.ptr = in.ptr + in.ch_mem_stride * weights.in_ch * M;
        out_prv.ptr = out.ptr + out.ch_mem_stride * group_offset;
        weights_prv.ptr = weights.ptr + weights.out_ch_mem_stride * group_offset;
        biases_prv = biases + group_offset;
        quant_T quant_params_prv = quant_params_offset(quant_params, group_offset);

        mli::krn::vdsp::convolution2D_nopad<i_T, w_T, o_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
                in_prv, weights_prv, biases_prv, out_prv, perception_area, quant_params_prv,
                val_min_limit, val_max_limit,
                stride_height, stride_width,
                dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);

    } // for M (group_count)
}

//========================================================
// Group Convolution 2D with padding
//========================================================
template <typename i_T, typename w_T, typename o_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void group_convolution2D_pad(
        const tensor_private_t<MLI_PTR(i_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &weights,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(o_T)> &out,
        const rect_t &perception_area,
        quant_T quant_params,
        const o_T val_min_limit,
        const o_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    const int group_count = in.ch / weights.in_ch;
    const int filters_per_group = weights.out_ch / group_count;

    tensor_private_t<MLI_CONV_OUT_PTR(o_T)> out_prv = out;
    tensor_private_t<MLI_CONV_OUT_PTR(i_T)> in_prv = in;
    conv2d_weights_tensor_private_t<MLI_PTR(w_T)> weights_prv = weights;
    const MLI_PTR(b_T)  __restrict biases_prv = biases;
    out_prv.ch = filters_per_group;
    in_prv.ch = weights.in_ch;

    for (int M = 0; M < group_count; M++) {

        int group_offset = M * filters_per_group;
        in_prv.ptr = in.ptr + in.ch_mem_stride * weights.in_ch * M;
        out_prv.ptr = out.ptr + out.ch_mem_stride * group_offset;
        weights_prv.ptr = weights.ptr + weights.out_ch_mem_stride * group_offset;
        biases_prv = biases + group_offset;
        quant_T quant_params_prv = quant_params_offset(quant_params, group_offset);

        mli::krn::vdsp::convolution2D_pad<i_T, w_T, o_T, b_T, acc_T, quant_T>(
                in_prv, weights_prv, biases_prv, out_prv, perception_area, quant_params_prv,
                val_min_limit, val_max_limit,
                stride_height, stride_width,
                dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);
    }
}

//========================================================
// Group Convolution 2D
//========================================================
template <typename i_T, typename w_T, typename o_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void group_convolution2D(
        const tensor_private_t<MLI_PTR(i_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &weights,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(o_T)> &out,
        const rect_t &perception_area,
        quant_T quant_params,
        const o_T val_min_limit,
        const o_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    MLI_ASSERT(quant_params_get_weigths_zeropoint(&quant_params) == 0); /* this optimized implementation assumes no zero offset for weights */

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    rect_t perception_area_nopad;
    perception_area_nopad.row_beg = CEIL_DIV(padding_top, stride_height);
    perception_area_nopad.row_end = out.height - CEIL_DIV(padding_bot, stride_height);
    perception_area_nopad.clmn_beg = CEIL_DIV(padding_left, stride_width);
    perception_area_nopad.clmn_end = out.width - CEIL_DIV(padding_right, stride_width);

    if ((perception_area_nopad.row_end - perception_area_nopad.row_beg > 0)
        && (perception_area_nopad.clmn_end - perception_area_nopad.clmn_beg > 0)){
        group_convolution2D_nopad<i_T, w_T, o_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
                in, weights, biases, out, perception_area_nopad, quant_params,
                val_min_limit, val_max_limit,
                stride_height, stride_width,
                dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);
    }

    // Phase 2: Process border part with more complex algorithm
    // (usually significantly smaller part of computations)
    //=======================================================================
    if (padding_top || padding_left || padding_bot || padding_right) {
        for(int i = 0; i < 4; i ++) {
            rect_t area;
            auto in_ = in;
            auto w_ = weights;
            auto out_ = out;
            int p_top = padding_top;
            int p_left = padding_left;
            int p_bot = padding_bot;
            int p_right = padding_right;
            int stride_w = stride_width;
            int stride_h = stride_height;
            int dilation_w = dilation_width;
            int dilation_h = dilation_height;

            if (i == 0) {
                if (padding_top) {
                    area.row_beg = 0;
                    area.row_end = CEIL_DIV(padding_top, stride_height);
                    area.clmn_beg = 0;
                    area.clmn_end = out.width - CEIL_DIV(padding_right, stride_width);
                    // compensate for cases where kernel size is larger than input size
                    area.clmn_end = MAX(CEIL_DIV(padding_left, stride_width), (int)area.clmn_end);
                } else {
                    continue;
                }
            }
            if (i == 1) {
                if (padding_right) {
                    int row_beg = 0;
                    int row_end = out.height - CEIL_DIV(padding_bot, stride_height);
                    int clmn_beg = out.width - CEIL_DIV(padding_right, stride_width);
                    int clmn_end = out.width;
                    // rotate the right padding area to the top
                    area.clmn_beg = row_beg;
                    area.clmn_end = row_end;
                    area.row_beg = out.width - clmn_end;
                    area.row_end = out.width - clmn_beg;
                    in_ = mli_prv_rotate_tensor_private<1>(in);
                    out_ = mli_prv_rotate_tensor_private<1>(out);
                    w_ = mli_prv_rotate_weights_tensor_private<1>(weights);
                    p_top = padding_right;
                    p_bot = padding_left;
                    p_left = padding_top;
                    p_right = padding_bot;
                    stride_w = stride_height;
                    stride_h = stride_width;
                    dilation_w = dilation_height;
                    dilation_h = dilation_width;
                } else {
                    continue;
                }
            }
            if (i == 2) {
                if (padding_bot) {
                    int row_beg = out.height - CEIL_DIV(padding_bot, stride_height);
                    int row_end = out.height;
                    int clmn_beg = CEIL_DIV(padding_left, stride_width);
                    int clmn_end = out.width;
                    // rotate the bottom padding area to the top
                    area.clmn_beg = out.width - clmn_end;
                    area.clmn_end = out.width - clmn_beg;
                    area.row_beg = out.height - row_end;
                    area.row_end = out.height - row_beg;
                    in_ = mli_prv_rotate_tensor_private<2>(in);
                    out_ = mli_prv_rotate_tensor_private<2>(out);
                    w_ = mli_prv_rotate_weights_tensor_private<2>(weights);
                    p_top = padding_bot;
                    p_bot = padding_top;
                    p_left = padding_right;
                    p_right = padding_left;
                } else {
                    continue;
                }
            }
            if (i == 3) {
                if (padding_left) {
                    int row_beg = CEIL_DIV(padding_top, stride_height);
                    int row_end = out.height;
                    int clmn_beg = 0;
                    int clmn_end = CEIL_DIV(padding_left, stride_width);
                    // rotate the left padding area to the top
                    area.clmn_beg = out.height - row_end;
                    area.clmn_end = out.height - row_beg;
                    area.row_beg = clmn_beg;
                    area.row_end = clmn_end;
                    in_ = mli_prv_rotate_tensor_private<3>(in);
                    out_ = mli_prv_rotate_tensor_private<3>(out);
                    w_ = mli_prv_rotate_weights_tensor_private<3>(weights);
                    p_top = padding_left;
                    p_bot = padding_right;
                    p_left = padding_bot;
                    p_right = padding_top;
                    stride_w = stride_height;
                    stride_h = stride_width;
                    dilation_w = dilation_height;
                    dilation_h = dilation_width;
                } else {
                    continue;
                }
            }
            group_convolution2D_pad<i_T, w_T, o_T, b_T, acc_T, quant_T>(
                    in_, w_, biases, out_, area, quant_params,
                    val_min_limit, val_max_limit,
                    stride_h, stride_w,
                    dilation_h, dilation_w,
                    p_top, p_left,
                    p_bot, p_right);
        }
    }
}

#pragma MLI_CODE_SECTION_END()
} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_GROUP_CONV2D_VDSP_H_
