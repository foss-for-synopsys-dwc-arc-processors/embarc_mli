/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_GROUP_CONV2D_REF_H_
#define _MLI_KRN_GROUP_CONV2D_REF_H_

#include "mli_api.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"
#include "mli_math.h"
#include "mli_private_types.h"
#include "mli_types.h"
#include "mli_krn_dotprod.h"
#include "mli_prv_layout.h"
#include "mli_krn_convolution.h"

namespace mli {
namespace krn {
namespace ref {

#pragma MLI_CODE_SECTION_START(".mli_lib")

//========================================================
// Unified Group convolution 2D template
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void group_convolution2D(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &weights,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        quant_T quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    const int row_begin = perception_area.row_beg;
    const int row_end = perception_area.row_end;
    const int clmn_begin = perception_area.clmn_beg;
    const int clmn_end = perception_area.clmn_end;

    const int group_count = in.ch / weights.in_ch;
    const int filters_per_group = weights.out_ch / group_count;

    for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
        for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
            // Define area of input and filter for convolution
            // comp - compensation values for valid area definition
            const mli_compensations comp = mli_prv_valid_area_compensations(
                    H_idx, W_idx, in.height, in.width,
                    weights.kernel_height, weights.kernel_width,
                    stride_height, stride_width, padding_left, padding_top,
                    dilation_height, dilation_width);

            const int rows = weights.kernel_height - comp.kernel_top - comp.kernel_bottom;
            const int clmns = weights.kernel_width - comp.kernel_right - comp.kernel_left;
            const int h_idx_in = (H_idx * stride_height - padding_top + comp.in_top);
            const int w_idx_in = (W_idx * stride_width - padding_left + comp.in_left);
            for (int out_ch_idx = 0; out_ch_idx < out.ch; out_ch_idx++) {
                MLI_CONV_OUT_PTR(io_T) out_ptr = out.ptr
                        + out.row_mem_stride * H_idx
                        + out.col_mem_stride * W_idx
                        + out.ch_mem_stride * out_ch_idx;
                        
                const MLI_PTR(io_T) in_ptr = in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in
                        + in.ch_mem_stride * weights.in_ch 
                        * (out_ch_idx / filters_per_group);

                const MLI_PTR(w_T) w_ptr = weights.ptr
                        + weights.row_mem_stride * comp.kernel_top
                        + weights.col_mem_stride * comp.kernel_left
                        + weights.out_ch_mem_stride * out_ch_idx;

                mli::krn::adjust_quant_params(&quant_params, out_ch_idx);

                acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
                mli::krn::dotprod3D(in_ptr, w_ptr, clmns, rows, weights.in_ch,
                          in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                          weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                          &accu);

                accu = mli::krn::weights_additive(w_ptr, accu, &quant_params, clmns, rows, weights.in_ch,
                            weights.col_mem_stride,
                            weights.row_mem_stride,
                            weights.in_ch_mem_stride);

                accu = mli::krn::in_additive(in_ptr , accu, &quant_params, clmns, rows, weights.in_ch,
                                       in.col_mem_stride, in.row_mem_stride, in.ch_mem_stride);
                accu = mli::krn::zp_additive(&quant_params, accu , clmns * rows);

                accu = mli::krn::bias_additive(&biases[out_ch_idx], accu, &quant_params);

                // Cast result to output type, apply built-in ReLU Applying and write result
                io_T out_val = mli::krn::result_cast<io_T, acc_T, quant_T>(accu, &quant_params);
                out_val = MIN(out_val, val_max_limit);
                out_val = MAX(out_val, val_min_limit);
                *out_ptr = out_val;
            } // for out_ch_idx
        } // for W_idx
    } // for H_idx
}

//====================================================================================
// Common routin for pre-calculation of various convolution parameters and running it.
//====================================================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T,
          mli_layout_type data_layout, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void group_conv2d_prepare_and_run(
        const mli_tensor *in,
        const mli_tensor *weights,
        const mli_tensor *bias,
        const mli_conv2d_cfg *cfg,
        mli_tensor *out) {
    mli_prv_fx_init_dsp_ctrl();
    const uint8_t stride_width = cfg->stride_width;
    const uint8_t stride_height = cfg->stride_height;
    const bool    no_pad = (fix_kernel_height == 1) && (fix_kernel_width == 1);
    int padding_top = no_pad ? 0 : cfg->padding_top;
    int padding_bot = no_pad ? 0 : cfg->padding_bottom;
    int padding_left = no_pad ? 0 : cfg->padding_left;
    int padding_right = no_pad ? 0 : cfg->padding_right;

    // Define output val limits (may affect built in ReLU)
    out->el_type = in->el_type;
    mli_minmax_t val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    const MLI_PTR(b_T) bs = (MLI_PTR(b_T))(bias->data.mem.void_p);

    auto in_prv = (data_layout == LAYOUT_HWC || data_layout == LAYOUT_HWCN || data_layout == LAYOUT_HW1N) ?
            mli_prv_get_tensor_hwc<MLI_PTR(io_T)>(in)
            : mli_prv_get_tensor_chw<MLI_PTR(io_T)>(in);

    conv2d_weights_tensor_private_t<MLI_PTR(w_T)> weights_prv;
    int out_ch;
    if (data_layout == LAYOUT_HWC) {
        weights_prv = mli_prv_get_conv2d_weights_tensor_nhwc<MLI_PTR(w_T)>(weights);
        out_ch = weights_prv.out_ch;
    } else if (data_layout == LAYOUT_HWCN) {
        weights_prv = mli_prv_get_conv2d_weights_tensor_hwcn<MLI_PTR(w_T)>(weights, 0, fix_kernel_width, fix_kernel_height);
        out_ch = weights_prv.out_ch;
    } else if ( data_layout == LAYOUT_HW1N) {
        weights_prv = mli_prv_get_conv2d_weights_tensor_hw1n<MLI_PTR(w_T)>(weights, fix_kernel_width, fix_kernel_height);
        out_ch = weights_prv.out_ch;
    } else {
        // LAYOUT_CHW
        weights_prv= mli_prv_get_conv2d_weights_tensor_nchw<MLI_PTR(w_T)>(weights);
        out_ch = weights_prv.out_ch;
    }

    // fill the rest output tensor parameters
    int dilation_width = (cfg->dilation_width > 0) ? cfg->dilation_width : 1;
    int dilation_height = (cfg->dilation_height > 0) ? cfg->dilation_height : 1;
    int effective_kernel_width = (weights_prv.kernel_width - 1) * dilation_width + 1;
    int effective_kernel_height = (weights_prv.kernel_height - 1) * dilation_height + 1;
    const int out_width  = CEIL_DIV(in_prv.width + padding_left + padding_right - effective_kernel_width + 1,
                                    stride_width);
    const int out_height = CEIL_DIV(in_prv.height + padding_top + padding_bot - effective_kernel_height + 1,
                                    stride_height);
    // Adjust the padding at the bottom and at the right in case too much padding was provided
    // (this can happen when stride > 1)
    // in case not all input samples can be used, adjust the width and height.
    padding_right = (out_width * stride_width + effective_kernel_width - stride_width) - in_prv.width - padding_left;
    padding_bot = (out_height * stride_height + effective_kernel_height - stride_height) - in_prv.height - padding_top;
    if (padding_right < 0) {
        in_prv.width += padding_right;
        padding_right = 0;
    }
    if (padding_bot < 0) {
        in_prv.height += padding_bot;
        padding_bot = 0;
    }

    out->rank = in->rank;
    if (data_layout == LAYOUT_HWC || data_layout == LAYOUT_HWCN || data_layout == LAYOUT_HW1N) {
        out->shape[FMAP_H_DIM_HWC] = out_height;
        out->shape[FMAP_W_DIM_HWC] = out_width;
        out->shape[FMAP_C_DIM_HWC] = out_ch;
    } else {
        out->shape[FMAP_H_DIM_CHW] = out_height;
        out->shape[FMAP_W_DIM_CHW] = out_width;
        out->shape[FMAP_C_DIM_CHW] = out_ch;
    }
    auto out_prv = (data_layout == LAYOUT_HWC || data_layout == LAYOUT_HWCN || data_layout == LAYOUT_HW1N) ?
            mli_prv_get_tensor_hwc<MLI_CONV_OUT_PTR(io_T)>(out)
            : mli_prv_get_tensor_chw<MLI_CONV_OUT_PTR(io_T)>(out);

    // Define quantization specific params
    quant_T params;
    define_quant_params(in, weights, bias, out, &params);

    rect_t cent_area;
    cent_area.row_beg = 0; cent_area.row_end = out_height;
    cent_area.clmn_beg = 0; cent_area.clmn_end = out_width;

    // Reuse all optimizations for convolution2d and depthwise_conv2d for particular cases of group_convolution2d
    if (in_prv.ch == weights_prv.in_ch) {
        mli::krn::convolution2D<io_T, w_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
                in_prv, weights_prv, bs, out_prv, cent_area, params,
                (io_T)val_limit.min, (io_T)val_limit.max,
                stride_height, stride_width, dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);
    } else if (weights_prv.in_ch == 1) {
        depthwise_convolution2D_wrapper<io_T, w_T, b_T, acc_T, quant_T>(
                in_prv.ptr, weights_prv.ptr, out_prv.ptr,
                in_prv, weights_prv, bs, out_prv, cent_area, params,
                (io_T)val_limit.min, (io_T)val_limit.max,
                stride_height, stride_width, dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);
    } else {
        mli::krn::group_convolution2D<io_T, w_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
                in_prv, weights_prv, bs, out_prv, cent_area, params,
                (io_T)val_limit.min, (io_T)val_limit.max,
                stride_height, stride_width, dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);
    }
    
}
#pragma MLI_CODE_SECTION_END()
} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_GROUP_CONV2D_REF_H_
