/*
* Copyright 2022, Synopsys, Inc.
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
template <typename i_T, typename w_T, typename o_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void convolution2D(
        const tensor_private_t<MLI_PTR(i_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &weights_full,
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
    //                 In other case, these additives must be calculated for valid area of dotproduct only.
    //================================================================================================
    const int row_begin = perception_area.row_beg;
    const int row_end = perception_area.row_end;
    const int clmn_begin = perception_area.clmn_beg;
    const int clmn_end = perception_area.clmn_end;

    // There is no bias in MLI3.0
    const bool has_bias = biases != nullptr;

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
                const MLI_PTR(i_T) in_ptr = in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in;

                const MLI_PTR(w_T) w_ptr = weights.ptr
                        + weights.row_mem_stride * comp.kernel_top
                        + weights.col_mem_stride * comp.kernel_left
                        + weights.out_ch_mem_stride * out_ch_idx;

                acc_T accu = mli_math_mul_fx<i_T, acc_T>(0, 0);

                mli::krn::dotprod3D(in_ptr, w_ptr, clmns, rows, in.ch,
                          in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                          weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                          &accu);

                accu = mli::krn::in_additive(in_ptr , accu, &quant_params, clmns, rows, in.ch,
                                       in.col_mem_stride, in.row_mem_stride, in.ch_mem_stride);

                o_T out_val;

                if (has_bias) {
                    accu = mli::krn::weights_additive(w_ptr, accu, &quant_params, clmns, rows, in.ch,
                                                weights.col_mem_stride,
                                                weights.row_mem_stride,
                                                weights.in_ch_mem_stride);

                    accu = mli::krn::zp_additive(&quant_params, accu , clmns * rows);

                    accu = mli::krn::bias_additive(&biases[out_ch_idx], accu, &quant_params);

                    mli::krn::adjust_quant_params(&quant_params, out_ch_idx);

                    // Cast result to output type, apply built-in ReLU Applying and write result
                    out_val = mli::krn::result_cast<o_T, acc_T, quant_T>(accu, &quant_params);
                    out_val = MIN(out_val, val_max_limit);
                    out_val = MAX(out_val, val_min_limit);
                } else {
                    // full weight area including padded values w.r.t. input
                    const MLI_PTR(w_T) w_ptr_full =
                        weights_full.ptr + weights_full.in_ch_mem_stride * 0 +
                        weights_full.out_ch_mem_stride * out_ch_idx;

                    // Additional calculations for padding areas only:
                    // out_val = 0 - sum_i(w*x_zp) - 0 + sum_i(w_zp*x_zp) + b_r,
                    // out_val += (sum_i(w*x_zp) - sum_i(w_zp*x_zp))
                    //============================================
                    if (rows * clmns != weights_full.kernel_height * weights_full.kernel_width) {
                        // This part emulate dotproduct out of valid area. it adds sum_i(w_full*x_zp) for the whole kernel,
                        // and afterward subtracts sum_i(w_valid*x_zp) part for valid area which we don't need due to
                        // conducted core dotproduct;
                        acc_T zero = mli_math_mul_fx<i_T, acc_T>(0, 0);

                        // out_val = 0 - (-out_val - sum_i(w_full*x_zp))
                        accu = ::mli::krn::ref::weights_additive(
                            w_ptr_full, mli_math_sub_fx<acc_T>(zero, accu),
                            &quant_params, weights_full.kernel_width,
                            weights_full.kernel_height, in.ch,
                            weights_full.col_mem_stride,
                            weights_full.row_mem_stride,
                            weights_full.in_ch_mem_stride);
                        accu = mli_math_sub_fx<acc_T>(zero, accu);

                        // out_val = out_val - sum_i(w_valid*x_zp))
                        accu = ::mli::krn::ref::weights_additive(
                            w_ptr, accu, &quant_params, clmns, rows, in.ch,
                            weights.col_mem_stride, weights.row_mem_stride,
                            weights.in_ch_mem_stride);

                        // This part emulate in_additive out of valid area. it adds sum_i(w_zp*x_zp)
                        // for all points out of valid area (e.g kernel_size - valid_area_size)
                        // out_val = 0 - (-out_val - sum_i(w_zp*x_zp))
                        accu = ::mli::krn::ref::zp_additive(
                            &quant_params, mli_math_sub_fx<acc_T>(zero, accu),
                            (weights_full.kernel_height * weights_full.kernel_width) - (clmns * rows));
                        accu = mli_math_sub_fx<acc_T>(zero, accu);
                    }

                    // Cast result to output type, no shift/runding/relu/saturation.
                    // o_T is expected equal or wider than acc_t
                    out_val = mli::krn::result_cast<o_T, acc_T, quant_T>(accu, &quant_params);
                }

                MLI_CONV_OUT_PTR(o_T) out_ptr = out.ptr
                        + out.row_mem_stride * H_idx
                        + out.col_mem_stride * W_idx
                        + out.ch_mem_stride * out_ch_idx;
                *out_ptr = out_val;
            } // for out_ch_idx
        } // for W_idx
    } // for H_idx
}

//========================================================
// Unified Depthwise convolution 2D template
//========================================================
template <typename i_T, typename w_T, typename o_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void depthwise_convolution2D(
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
    // Unified Depthwise convolutions for all layouts (NCHW/HWCN) and quantization schemes:
    // MLI_FX (symmetric data, scales are power of two) and s8asym (assymetric data, scales of any value)
    // For more info on calculations see generic convolution 2D notes above
    //================================================================================================
    const int row_begin = perception_area.row_beg;
    const int row_end = perception_area.row_end;
    const int clmn_begin = perception_area.clmn_beg;
    const int clmn_end = perception_area.clmn_end;

    // There is no bias in MLI3.0
    const bool has_bias = biases != nullptr;

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
                const MLI_PTR(i_T) in_ptr = in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in
                        + in.ch_mem_stride * in_ch_idx;

                const int out_ch_idx = in_ch_idx;
                const MLI_PTR(w_T) w_ptr = weights.ptr
                        + weights.row_mem_stride * comp.kernel_top
                        + weights.col_mem_stride * comp.kernel_left
                        + weights.in_ch_mem_stride * 0
                        + weights.out_ch_mem_stride * out_ch_idx;

                // Complete Convolution. Here calculations performes in a unfolded expression way.
                // If the calulations do not involve input, they can be compuated offline and folded.
                // out_val  = (x-x_zp)*(w-w_zp)
                //          = sum(x*w) - sum_i(x*w_zp) - sum_i(w*x_zp) + sum_i(w_zp*x_zp) + b_r
                //          = core_val + fold_val
                // core_val = sum(x*w) - sum_i(x*w_zp)
                // fold_val = - sum_i(w*x_zp) + sum_i(w_zp*x_zp) + b_r
                // The core_val will be cacluated at all times, but the fold_val depends on if there is a bias.

                // Convolution core. Do a common 2D Conv and add in additives.
                //============================================
                // core_val = sum(x*w) - sum_i(x*w_zp)
                acc_T accu = mli_math_mul_fx<i_T, acc_T>(0, 0);
                accu = mli::krn::dotprod2D(in_ptr, w_ptr, accu, clmns, rows,
                                           in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                                           weights.col_mem_stride, weights.row_mem_stride);
                accu  = mli::krn::in_additive(in_ptr, accu, &quant_params,
                                              clmns, rows,
                                              in.col_mem_stride * dilation_width,
                                              in.row_mem_stride * dilation_height);
                o_T out_val;

                if (has_bias) {
                    // MLI2.0
                    // Cacluations of other addiives in the order of in_additives, zp_additives and bias
                    // out_val = core_val - sum_i(w*x_zp) + sum_i(w_zp*x_zp) + b_r
                    //============================================
                    accu = mli::krn::ref::weights_additive(w_ptr, accu, &quant_params, clmns, rows,
                                            weights.col_mem_stride,
                                            weights.row_mem_stride);
                    accu = mli::krn::bias_additive(&biases[out_ch_idx], accu, &quant_params);

                    accu  = mli::krn::zp_additive(&quant_params, accu, clmns * rows);

                    mli::krn::adjust_quant_params(&quant_params, out_ch_idx);

                    // Cast result to output type, apply built-in ReLU Applying and write result
                    out_val = mli::krn::result_cast<o_T, acc_T, quant_T>(accu, &quant_params);
                    out_val = MIN(out_val, val_max_limit);
                    out_val = MAX(out_val, val_min_limit);
                } else {
                    // MLI3.0
                    // Core calculations with new behavior is the following: out = sum(in_i * (w - w_zp)).
                    // This not implies that input doesn't have zero point at all. Main expectation is that
                    // input zero point is folded into bias and will be added after convolution.
                    // Complexities for programmable core and for the former approach is that padded values should
                    // EXPLICITLY participate in calculations, while in past we avoided it calculating and re-using
                    // additives. One of the option is modify dotproduct with using zero points and defining which
                    // values are padding in runtime. Another option is to use additives as a way to add "calculations on the padding area".
                    // It isn't straightforward from the first glance.
                    // But implies less code modifications for core calculations.

                    // Additional calculations for padding areas only:
                    // out_val += (sum_i(w*x_zp) - sum_i(w_zp*x_zp))
                    //============================================
                    if(rows * clmns != weights.kernel_height * weights.kernel_width) {
                        // full weight area including padded values w.r.t. input
                        const MLI_PTR(w_T) w_ptr_full = weights.ptr
                                + weights.in_ch_mem_stride * 0
                                + weights.out_ch_mem_stride * out_ch_idx;

                        // This part emulate dotproduct out of valid area. it adds sum_i(w_full*x_zp) for the whole kernel,
                        // and afterward subtracts sum_i(w_valid*x_zp) part for valid area which we don't need due to
                        // conducted core dotproduct;
                        acc_T zero = mli_math_mul_fx<i_T, acc_T>(0, 0);

                        // out_val = 0 - (-out_val - sum_i(w_full*x_zp))
                        accu = ::mli::krn::ref::weights_additive(w_ptr_full, mli_math_sub_fx<acc_T>(zero, accu), &quant_params,
                            weights.kernel_width, weights.kernel_height, weights.col_mem_stride, weights.row_mem_stride);
                        accu = mli_math_sub_fx<acc_T>(zero, accu);

                        // out_val = out_val - sum_i(w_valid*x_zp))
                        accu = ::mli::krn::ref::weights_additive(w_ptr, accu, &quant_params,
                            clmns, rows, weights.col_mem_stride, weights.row_mem_stride);

                        // This part emulate in_additive out of valid area. it adds sum_i(w_zp*x_zp)
                        // for all points out of valid area (e.g kernel_size - valid_area_size)
                        // out_val = 0 - (-out_val - sum_i(w_zp*x_zp))
                        accu = ::mli::krn::ref::zp_additive(&quant_params, mli_math_sub_fx<acc_T>(zero, accu),
                            (weights.kernel_height * weights.kernel_width) - (clmns * rows));
                        accu = mli_math_sub_fx<acc_T>(zero, accu);
                    }

                    // Cast result to output type, no shift/runding/relu/saturation.
                    // o_T is expected equal or wider than acc_t
                    out_val = mli::krn::result_cast<o_T, acc_T, quant_T>(accu, &quant_params);
                }

                MLI_CONV_OUT_PTR(o_T) out_ptr = out.ptr
                        + out.row_mem_stride * H_idx
                        + out.col_mem_stride * W_idx
                        + out.ch_mem_stride * out_ch_idx;
                *out_ptr = out_val;
            } // for in_ch_idx
        } // for W_idx
    } // for H_idx
}

template <typename i_T, typename w_T, typename o_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void depthwise_convolution2D_wrapper(
        MLI_PTR(i_T) __restrict in_ptr,
        MLI_PTR(w_T) __restrict w_ptr,
        MLI_CONV_OUT_PTR(o_T) __restrict out_ptr,
        tensor_private_t<MLI_PTR(i_T)> &in,
        conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &weights,
        const MLI_PTR(b_T)  __restrict biases,
        tensor_private_t<MLI_CONV_OUT_PTR(o_T)> &out,
        const rect_t &perception_area,
        quant_T quant_params,
        const o_T val_min_limit,
        const o_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    tensor_private_t<MLI_PTR(i_T)> in_ = in;
    conv2d_weights_tensor_private_t<MLI_PTR(w_T)> weights_ = weights;
    tensor_private_t<MLI_CONV_OUT_PTR(o_T)> out_ = out;
    in_.ptr = in_ptr;
    weights_.ptr = w_ptr;
    out_.ptr = out_ptr;

    mli::krn::depthwise_convolution2D<i_T, w_T, o_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
            in_, weights_, biases, out_, perception_area, quant_params,
            val_min_limit, val_max_limit,
            stride_height, stride_width, dilation_height, dilation_width,
            padding_top, padding_left,
            padding_bot, padding_right);
}

//====================================================================================
// Common routin for pre-calculation of various convolution parameters and running it.
//====================================================================================
template <typename i_T, typename w_T, typename o_T, typename b_T, typename acc_T, typename quant_T,
          mli_conv_type conv_type, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void conv2d_run(
        tensor_private_t<MLI_PTR(i_T)> &in_prv,
        conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &weights_prv,
        const MLI_PTR(b_T) &bs,
        tensor_private_t<MLI_CONV_OUT_PTR(o_T)> &out_prv,
        mli_minmax_t val_limit,
        const mli_conv2d_cfg *cfg,
        const quant_T& params) {

    const uint8_t stride_width = cfg->stride_width;
    const uint8_t stride_height = cfg->stride_height;
    const bool    no_pad = (fix_kernel_height == 1) && (fix_kernel_width == 1);
    int padding_top = no_pad ? 0 : cfg->padding_top;
    int padding_bot = no_pad ? 0 : cfg->padding_bottom;
    int padding_left = no_pad ? 0 : cfg->padding_left;
    int padding_right = no_pad ? 0 : cfg->padding_right;
    int dilation_width = cfg->dilation_width;
    int dilation_height = cfg->dilation_height;

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
    rect_t cent_area;
    cent_area.row_beg = 0; cent_area.row_end = out_prv.height;
    cent_area.clmn_beg = 0; cent_area.clmn_end = out_prv.width;

    // Applying main convolution core (depends on layout)
    //=======================================================================
    if (conv_type == CONV_GENERAL) {
        mli::krn::convolution2D<i_T, w_T, o_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
                in_prv, weights_prv, weights_prv, bs, out_prv, cent_area, params,
                (o_T)val_limit.min, (o_T)val_limit.max,
                stride_height, stride_width, dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);
    } else {
        depthwise_convolution2D_wrapper<i_T, w_T, o_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
                in_prv.ptr, weights_prv.ptr, out_prv.ptr,
                in_prv, weights_prv, bs, out_prv, cent_area, params,
                (o_T)val_limit.min, (o_T)val_limit.max,
                stride_height, stride_width, dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);
    }
}

template <typename i_T, typename w_T, typename o_T, typename b_T, typename acc_T, typename quant_T,
          mli_layout_type data_layout, mli_conv_type conv_type, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void conv2d_prepare_and_run(
        const mli_tensor *in,
        const mli_tensor *weights,
        const mli_tensor *bias,
        const mli_conv2d_cfg *cfg,
        mli_tensor *out) {
    mli_prv_fx_init_dsp_ctrl();

    constexpr bool asym = std::is_same<quant_T, s8asym_quant_specific_params>::value;
    mli_minmax_t val_limit = mli_prv_get_relu_limits<o_T, asym>(&cfg->relu, out);

    // For MLI3.0, bias will be added in the subsequent operation
    const bool has_bias = bias != nullptr;
    const MLI_PTR(b_T) bs = nullptr;
    if (has_bias) {
        bs = mli_prv_tensor_data_ptr<MLI_PTR(b_T)>(bias);
    }

    auto in_prv = (data_layout == LAYOUT_HWC || data_layout == LAYOUT_HWCN || data_layout == LAYOUT_HW1N) ?
            mli_prv_get_tensor_hwc<MLI_PTR(i_T)>(in)
            : mli_prv_get_tensor_chw<MLI_PTR(i_T)>(in);

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
            mli_prv_get_tensor_hwc<MLI_CONV_OUT_PTR(o_T)>(out)
            : mli_prv_get_tensor_chw<MLI_CONV_OUT_PTR(o_T)>(out);

    quant_T params;
    define_quant_params(in, weights, bias, out, &params);

    conv2d_run<i_T, w_T, o_T, b_T, acc_T, quant_T, conv_type, fix_kernel_width, fix_kernel_height>(
            in_prv, weights_prv, bs, out_prv, val_limit, cfg, params);
}
#pragma MLI_CODE_SECTION_END()
} // namespace ref
} // namespace krn
} // namespace mli

namespace snps_arc::metaware::mli::ref {
#pragma MLI_CODE_SECTION_START(".mli_lib")

template <typename i_T, typename w_T, typename o_T, typename acc_T,
          mli_layout_type data_layout, ::mli::mli_conv_type conv_type,
          unsigned io_rank, unsigned w_rank, typename cfg_T>
MLI_FORCE_INLINE void conv2d_prepare_and_run(
    const QTensor<InternalBuffer, io_rank> &in,
    const QTensor<InternalBuffer, w_rank> &weights,
    Tensor<InternalBuffer, io_rank> &out, const cfg_T &cfg) {

    using b_T = o_T;
    using quant_T = ::mli::krn::int_quant_specific_params;

    // Define quantization specific params
    quant_T params;
    define_quant_params(in, weights, &params);

    MLI_ASSERT(data_layout == LAYOUT_HWC);

    // I/O Tensor -> tensor_private_t
    auto in_prv = mli_prv_get_tensor_hwc<MLI_PTR(i_T)>(in.t);

    conv2d_weights_tensor_private_t<MLI_PTR(w_T)> weights_prv;
    if constexpr (w_rank == 5) {
        MLI_ASSERT(weights.t.get_dim(kKernelGroupDim) == 1);
        weights_prv = mli_prv_get_conv2d_weights_tensor_hwcn<MLI_PTR(w_T)>(weights.t);
    } else if constexpr (w_rank == 3) {
        weights_prv = mli_prv_get_conv2d_weights_tensor_hwc<MLI_PTR(w_T)>(weights.t);
    } else {
        // not supported yet
        MLI_ASSERT(false);
    }

    auto out_prv = mli_prv_get_tensor_hwc<MLI_CONV_OUT_PTR(o_T)>(out);

    // no bias and relu in MLI3.0
    const MLI_PTR(b_T) bs = nullptr;
    mli_minmax_t val_limit = {std::numeric_limits<o_T>::min(), std::numeric_limits<o_T>::max()};

    if constexpr (std::is_same_v<cfg_T, Conv2DConfig>) {
        MLI_ASSERT(cfg.groups == 1);
    }
    mli_conv2d_cfg krn_cfg;
    krn_cfg.stride_height = cfg.stride[0];
    krn_cfg.stride_width = cfg.stride[1];
    krn_cfg.padding_top = cfg.padding_begin[0];
    krn_cfg.padding_left = cfg.padding_begin[1];
    krn_cfg.padding_bottom = cfg.padding_end[0];
    krn_cfg.padding_right = cfg.padding_end[1];
    krn_cfg.dilation_height = cfg.dilation[0];
    krn_cfg.dilation_width = cfg.dilation[1];
    krn_cfg.relu = {MLI_RELU_NONE};

    // Update config and run convolution
    //=======================================================================
    ::mli::krn::ref::conv2d_run<i_T, w_T, o_T, b_T, acc_T, quant_T, conv_type, KRN_SZ_VAR, KRN_SZ_VAR>(
            in_prv, weights_prv, bs, out_prv, val_limit, &krn_cfg, params);
}

#pragma MLI_CODE_SECTION_END()
} // namespace snps_arc::metaware::mli::ref

#endif // _MLI_KRN_CONVOLUTION_REF_H_
