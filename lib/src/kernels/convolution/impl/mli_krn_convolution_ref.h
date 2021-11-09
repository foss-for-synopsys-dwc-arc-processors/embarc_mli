/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_CONVOLUTION_REF_H_
#define _MLI_KRN_CONVOLUTION_REF_H_

#include "mli_api.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"
#include "mli_math.h"
#include "mli_mem_info.h"
#include "mli_private_types.h"
#include "mli_types.h"
#include "mli_krn_dotprod.h"
#include "mli_prv_layout.h"

namespace mli {
namespace krn {
namespace ref {

#pragma MLI_CODE_SECTION_START(".mli_lib")

//========================================================
// Unified Generic Convolution 2D template
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void convolution2D(
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
    // Unified Generic convolution for all layouts (CHW/HWC/HWCN) and quantization scheme:  
    // MLI_FX (symmetric data, scales are power of two) and s8asym (assymetric data, scales of any value)
    // For each output point Calculation implies dotproduct and bias add:
    //            out_val = sum_i(x_r * w_r) + b_r
    //
    // Considering assymetric types(x_r = (x - x_zp) and w_r = (w - w_zp)
    //                    out_val = sum_i((x-x_zp)*(w-w_zp)) + b_r
    //
    // after opening brackets:
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
    //
    // IMPORTANT NOTE: For border areas with padding, weights/input/zp can be reused only in case of explicitly padded values. 
    //                 In other case, these additives must be calculatid for valid area of dotproduct only.
    //================================================================================================
    const int row_begin = perception_area.row_beg;
    const int row_end = perception_area.row_end;
    const int clmn_begin = perception_area.clmn_beg;
    const int clmn_end = perception_area.clmn_end;

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
                        + in.col_mem_stride * w_idx_in;

                const MLI_PTR(w_T) w_ptr = weights.ptr
                        + weights.row_mem_stride * comp.kernel_top
                        + weights.col_mem_stride * comp.kernel_left
                        + weights.out_ch_mem_stride * out_ch_idx;

                mli::krn::adjust_quant_params(&quant_params, out_ch_idx);

                acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);

                mli::krn::dotprod3D(in_ptr, w_ptr, clmns, rows, in.ch,
                          in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                          weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                          &accu);

		const MLI_PTR(w_T) w_ptr_comp = weights.ptr + weights.out_ch_mem_stride * out_ch_idx;
		accu = mli::krn::ref::weights_sub(w_ptr_comp, accu, &quant_params, weights.kernel_width, weights.kernel_height, in.ch,
			    weights.col_mem_stride,
			    weights.row_mem_stride,
			    weights.in_ch_mem_stride);

                accu = mli::krn::weights_additive(w_ptr, accu, &quant_params, clmns, rows, in.ch,
                                            weights.col_mem_stride,
                                            weights.row_mem_stride,
                                            weights.in_ch_mem_stride);

                accu = mli::krn::in_additive(in_ptr , accu, &quant_params, clmns, rows, in.ch,
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

//========================================================
// Unified Depthwise convolution 2D template
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void depthwise_convolution2D(
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
    // Unified Depthwise convolutions for all layouts (NCHW/HWCN) and quantization schemes:  
    // MLI_FX (symmetric data, scales are power of two) and s8asym (assymetric data, scales of any value)
    // For more info on calculations see generic convolution 2D notes above 
    //================================================================================================
    const int row_begin = perception_area.row_beg;
    const int row_end = perception_area.row_end;
    const int clmn_begin = perception_area.clmn_beg;
    const int clmn_end = perception_area.clmn_end;

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

            for (int in_ch_idx = 0; in_ch_idx < in.ch; in_ch_idx++) {
                const MLI_PTR(io_T) in_ptr = in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in
                        + in.ch_mem_stride * in_ch_idx;

                acc_T other_additives = mli_math_mul_fx<io_T, acc_T>(0, 0);
                other_additives  = mli::krn::zp_additive(&quant_params, other_additives,
                                               clmns * rows);
                other_additives  = mli::krn::in_additive(in_ptr, other_additives, &quant_params,
                                               clmns, rows,
                                               in.col_mem_stride * dilation_width,
                                               in.row_mem_stride * dilation_height);

                const int out_ch_idx = in_ch_idx;
                const MLI_PTR(w_T) w_ptr = weights.ptr
                        + weights.row_mem_stride * comp.kernel_top
                        + weights.col_mem_stride * comp.kernel_left
                        + weights.in_ch_mem_stride * 0
                        + weights.out_ch_mem_stride * out_ch_idx;
                mli::krn::adjust_quant_params(&quant_params, out_ch_idx);

                // Convolution core. Here calculations performes in a unfolded expression way:
                // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                //============================================
                acc_T accu = other_additives;//mli_math_mul_fx<io_T, acc_T>(0, 0);
                accu = mli::krn::dotprod2D(in_ptr, w_ptr, accu, clmns, rows,
                                    in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                                    weights.col_mem_stride,
                                    weights.row_mem_stride);
                accu = mli::krn::ref::weights_additive(w_ptr, accu, &quant_params, clmns, rows,
                                        weights.col_mem_stride,
                                        weights.row_mem_stride);
                accu = mli::krn::bias_additive(&biases[out_ch_idx], accu, &quant_params);
                //accu = mli_math_add(accu, other_additives);

                // Cast result to output type, apply built-in ReLU Applying and write result
                io_T out_val = mli::krn::result_cast<io_T, acc_T, quant_T>(accu, &quant_params);
                out_val = MIN(out_val, val_max_limit);
                out_val = MAX(out_val, val_min_limit);

                MLI_CONV_OUT_PTR(io_T) out_ptr = out.ptr
                        + out.row_mem_stride * H_idx
                        + out.col_mem_stride * W_idx
                        + out.ch_mem_stride * out_ch_idx;
                *out_ptr = out_val;
            } // for in_ch_idx
        } // for W_idx
    } // for H_idx
}

template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void depthwise_convolution2D_wrapper(
        MLI_PTR(io_T) __restrict in_ptr,
        MLI_PTR(w_T) __restrict w_ptr,
        MLI_CONV_OUT_PTR(io_T) __restrict out_ptr,
        tensor_private_t<MLI_PTR(io_T)> &in,
        conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &weights,
        const MLI_PTR(b_T)  __restrict biases,
        tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        quant_T quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    tensor_private_t<MLI_PTR(io_T)> in_ = in;
    conv2d_weights_tensor_private_t<MLI_PTR(w_T)> weights_ = weights;
    tensor_private_t<MLI_CONV_OUT_PTR(io_T)> out_ = out;
    in_.ptr = in_ptr;
    weights_.ptr = w_ptr;
    out_.ptr = out_ptr;

    mli::krn::depthwise_convolution2D<io_T, w_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
            in_, weights_, biases, out_, perception_area, quant_params,
            val_min_limit, val_max_limit,
            stride_height, stride_width, dilation_height, dilation_width,
            padding_top, padding_left,
            padding_bot, padding_right);
}

//====================================================================================
// Common routin for pre-calculation of various convolution parameters and running it.
//====================================================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T,
          mli_layout_type data_layout, mli_conv_type conv_type, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void conv2d_prepare_and_run(
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
    int dilation_width = cfg->dilation_width;
    int dilation_height = cfg->dilation_height;

    // Define output val limits (may affect built in ReLU)
    out->el_type = in->el_type;

    constexpr bool asym = std::is_same<quant_T, s8asym_quant_specific_params>::value;
    mli_minmax_t val_limit = mli_prv_get_relu_limits<io_T, asym>(&cfg->relu, out);

    const MLI_PTR(b_T) bs = mli_prv_tensor_data_ptr<MLI_PTR(b_T)>(bias);

    auto in_prv = (data_layout == LAYOUT_HWC || data_layout == LAYOUT_HWCN || data_layout == LAYOUT_HW1N) ?
            mli_prv_get_tensor_hwc<MLI_PTR(io_T)>(in)
            : mli_prv_get_tensor_chw<MLI_PTR(io_T)>(in);

    conv2d_weights_tensor_private_t<MLI_PTR(w_T)> weights_prv;
    if (data_layout == LAYOUT_HWC) {
        weights_prv = mli_prv_get_conv2d_weights_tensor_nhwc<MLI_PTR(w_T)>(weights);
    } else if (data_layout == LAYOUT_HWCN) {
        weights_prv = mli_prv_get_conv2d_weights_tensor_hwcn<MLI_PTR(w_T)>(weights, 0, fix_kernel_width, fix_kernel_height);
    } else if ( data_layout == LAYOUT_HW1N) {
        weights_prv = mli_prv_get_conv2d_weights_tensor_hw1n<MLI_PTR(w_T)>(weights, fix_kernel_width, fix_kernel_height);
    } else {
        // LAYOUT_CHW
        weights_prv= mli_prv_get_conv2d_weights_tensor_nchw<MLI_PTR(w_T)>(weights);
    }

    auto out_prv = (data_layout == LAYOUT_HWC || data_layout == LAYOUT_HWCN || data_layout == LAYOUT_HW1N) ?
            mli_prv_get_tensor_hwc<MLI_CONV_OUT_PTR(io_T)>(out)
            : mli_prv_get_tensor_chw<MLI_CONV_OUT_PTR(io_T)>(out);

    // Adjust the padding at the bottom and at the right in case too much padding was provided
    // (this can happen when stride > 1)
    // in case not all input samples can be used, adjust the width and height.
    int effective_kernel_width = (weights_prv.kernel_width - 1) * dilation_width + 1;
    int effective_kernel_height = (weights_prv.kernel_height - 1) * dilation_height + 1;
    padding_right = (out_prv.width * stride_width + effective_kernel_width - stride_width) - in_prv.width - padding_left;
    padding_bot = (out_prv.height * stride_height + effective_kernel_height - stride_height) - in_prv.height - padding_top;
    if (padding_right < 0) {
        in_prv.width += padding_right;
        padding_right = 0;
    }
    if (padding_bot < 0) {
        in_prv.height += padding_bot;
        padding_bot = 0;
    }

    // Define quantization specific params
    quant_T params;
    define_quant_params(in, weights, bias, out, &params);

    rect_t cent_area;
    cent_area.row_beg = 0; cent_area.row_end = out_prv.height;
    cent_area.clmn_beg = 0; cent_area.clmn_end = out_prv.width;

    // Applying main convolution core (depends on layout)
    //=======================================================================
    if (conv_type == CONV_GENERAL) {
        mli::krn::convolution2D<io_T, w_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
                in_prv, weights_prv, bs, out_prv, cent_area, params,
                (io_T)val_limit.min, (io_T)val_limit.max,
                stride_height, stride_width, dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);
    } else {
        depthwise_convolution2D_wrapper<io_T, w_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
                in_prv.ptr, weights_prv.ptr, out_prv.ptr,
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

#endif // _MLI_KRN_CONVOLUTION_REF_H_
