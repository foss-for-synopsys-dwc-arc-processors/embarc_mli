/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_CONVOLUTION_VDSP_H_
#define _MLI_KRN_CONVOLUTION_VDSP_H_

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
namespace vdsp {

#pragma MLI_CODE_SECTION_START(".mli_lib")

//========================================================
// Convolution 2D without padding
//========================================================
template <typename i_T, typename w_T, typename o_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void convolution2D_nopad(
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
    int width = clmn_end - clmn_begin;
    int height = row_end - row_begin;
    constexpr int numaccuregs = sizeof(acc_T) / sizeof(vNint_t);
    // for larger kernel sizes in combination with an unroll of 4, the gather load of the input samples
    // don't fit into one vector anymore. in those cases fall back to unroll 2
    constexpr int unroll = ((numaccuregs > 2) || (fix_kernel_width * fix_kernel_height * 4 > _VDSP_NUM_8BIT_LANES)) ? 2 : 4;
    int out_w_inc = out.col_mem_stride;
    int out_h_inc = out.row_mem_stride - width * out_w_inc;
    int in_w_inc = in.col_mem_stride * stride_width;
    int in_h_inc = in.row_mem_stride * stride_height - width * in_w_inc;
    if ((fix_kernel_width == 1) && (fix_kernel_height == 1)) {
        if ((out_h_inc == 0) && (in_h_inc == 0)) {
            width = width * height;
            height = 1;
        }
    }

    // There is no bias in MLI3.0
    const bool has_bias = biases != nullptr;

    for (int out_ch_idx = 0; out_ch_idx < out.ch; out_ch_idx+= get_number_lanes<acc_T>()) {
        int remaining_ch = out.ch - out_ch_idx;
        int current_ch = MIN(remaining_ch, get_number_lanes<acc_T>()); /* nr channels computed in this loop iteration */
        const MLI_PTR(w_T) w_ptr = weights.ptr
                + weights.out_ch_mem_stride * out_ch_idx;
        const int rows = (fix_kernel_height > 0) ? fix_kernel_height : weights.kernel_height;
        const int clmns = (fix_kernel_width > 0) ? fix_kernel_width : weights.kernel_width;
        __builtin_assume (rows > 0);
        __builtin_assume (clmns > 0);
        __builtin_assume (in.ch > 0);

        auto output_params = adjust_quant_params_v(&quant_params, out_ch_idx);

        acc_T pre_accu = mli_math_mul_fx<i_T, acc_T>(0, 0);

        if (has_bias) {
            pre_accu = mli::krn::bias_additive(&biases[out_ch_idx], pre_accu, &output_params);
            pre_accu = mli::krn::weights_additive(w_ptr, pre_accu, &quant_params, clmns, rows, in.ch,
                                        weights.col_mem_stride,
                                        weights.row_mem_stride,
                                        weights.in_ch_mem_stride);
        } else {
            if(rows * clmns != weights.kernel_height * weights.kernel_width) {
                const MLI_PTR(w_T) w_ptr_full = weights.ptr
                        + weights.in_ch_mem_stride * 0
                        + weights.out_ch_mem_stride * out_ch_idx;

                acc_T zero = mli_math_mul_fx<i_T, acc_T>(0, 0);

                // out_val = 0 - (-out_val - sum_i(w_full*x_zp))
                pre_accu = mli::krn::weights_additive(w_ptr_full, mli_math_sub<acc_T>(zero, pre_accu), &quant_params,
                    weights.kernel_width, weights.kernel_height, in.ch,
                    weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride);
                pre_accu = mli_math_sub<acc_T>(zero, pre_accu);

                // out_val = out_val - sum_i(w_valid*x_zp))
                pre_accu = mli::krn::weights_additive(w_ptr, pre_accu, &quant_params, clmns, rows, in.ch,
                    weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride);
            }
        }

        const int h_idx_in = row_begin * stride_height - padding_top;
        const int w_idx_in = clmn_begin * stride_width - padding_left;
        MLI_CONV_OUT_PTR(o_T) out_ptr = out.ptr
                + out.row_mem_stride * row_begin
                + out.col_mem_stride * clmn_begin
                + out.ch_mem_stride * out_ch_idx;
        const MLI_PTR(i_T) in_ptr = in.ptr
                + in.row_mem_stride * h_idx_in
                + in.col_mem_stride * w_idx_in;

        for (int H_idx = 0; H_idx < height; H_idx++) {
            int W_cnt = 0;

            for (W_cnt = 0; W_cnt <= width - unroll; W_cnt+=unroll) {
                auto accu = init_accu_grp(pre_accu);

                if ((fix_kernel_width == 1) && (fix_kernel_height == 1)) {
                    accu = mli::krn::dotprod1D_v_unroll<unroll>(in_ptr, w_ptr, accu, in.ch, in.ch_mem_stride, in_w_inc, weights.in_ch_mem_stride);
                } else if ((fix_kernel_width > 0) && (fix_kernel_height > 0)) {
                    // unrolled version with fixed kernelsize
                    if (dilation_width == stride_width) {
                        int ext_width = clmns + unroll - 1;
                        int required_loads = ext_width * rows;
                        int unroll_step = 0;
                        int unroll_1 = 1;
                        int kernel_size = 1;
                        accu = mli::krn::dotprod3D_v_unroll<unroll, true>(in_ptr, w_ptr, clmns, rows, in.ch,
                                  in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride, in_w_inc,
                                  weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                                  unroll_step, unroll_1, required_loads, kernel_size, ext_width,
                                  accu);
                    } else {
                        int ext_width = clmns;
                        int required_loads = clmns * rows * unroll;
                        int unroll_step = in_w_inc;
                        int unroll_1 = unroll;
                        int kernel_size = clmns * rows;
                        accu = mli::krn::dotprod3D_v_unroll<unroll, true>(in_ptr, w_ptr, clmns, rows, in.ch,
                                  in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride, in_w_inc,
                                  weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                                  unroll_step, unroll_1, required_loads, kernel_size, ext_width,
                                  accu);
                    }
                } else if (weights.row_mem_stride == clmns * weights.col_mem_stride) {
                    accu = mli::krn::dotprod3D_v_nopad_unroll<unroll>(in_ptr, w_ptr, clmns, rows, in.ch,
                              in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                              weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride, in_w_inc,
                              accu);
                } else {
                    int ext_width = clmns;
                    int required_loads = clmns * rows * unroll;
                    int unroll_step = in_w_inc;
                    int unroll_1 = unroll;
                    int kernel_size = clmns * rows;
                    accu = mli::krn::dotprod3D_v_unroll<unroll, false>(in_ptr, w_ptr, clmns, rows, in.ch,
                              in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride, in_w_inc,
                              weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                              unroll_step, unroll_1, required_loads, kernel_size, ext_width,
                              accu);
                }
                // Cast result to output type, apply built-in ReLU Applying and write result
                mli::krn::result_cast_relu_store_v(out_ptr, accu.accu0, &output_params, val_min_limit, val_max_limit, current_ch);
                out_ptr += out_w_inc;
                if (unroll > 1) {
                    mli::krn::result_cast_relu_store_v(out_ptr, accu.accu1, &output_params, val_min_limit, val_max_limit, current_ch);
                    out_ptr += out_w_inc;
                }
                if (unroll > 2) {
                    mli::krn::result_cast_relu_store_v(out_ptr, accu.accu2, &output_params, val_min_limit, val_max_limit, current_ch);
                    out_ptr += out_w_inc;
                }
                if (unroll > 3) {
                    mli::krn::result_cast_relu_store_v(out_ptr, accu.accu3, &output_params, val_min_limit, val_max_limit, current_ch);
                    out_ptr += out_w_inc;
                }
                in_ptr += in_w_inc * unroll;
            } // for W_cnt

            for (; W_cnt < width; W_cnt++) {

                acc_T accu = pre_accu;

                if ((fix_kernel_width == 1) && (fix_kernel_height == 1)) {
                    accu = mli::krn::dotprod1D_v(in_ptr, w_ptr, accu, in.ch, in.ch_mem_stride, weights.in_ch_mem_stride);
                } else if ((fix_kernel_width > 0) && (fix_kernel_height > 0)) {
                    accu = mli::krn::dotprod3D_v<i_T, w_T, acc_T, /*fixedsize*/true>(in_ptr, w_ptr, clmns, rows, in.ch,
                              in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                              weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                              accu);
                } else if (weights.row_mem_stride == clmns * weights.col_mem_stride) {
                    accu = mli::krn::dotprod3D_v_nopad(in_ptr, w_ptr, clmns, rows, in.ch,
                              in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                              weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                              accu);
                } else {
                    accu = mli::krn::dotprod3D_v<i_T, w_T, acc_T, /*fixedsize*/false>(in_ptr, w_ptr, clmns, rows, in.ch,
                              in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                              weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                              accu);
                }
                // Cast result to output type, apply built-in ReLU Applying and write result
                mli::krn::result_cast_relu_store_v(out_ptr, accu, &output_params, val_min_limit, val_max_limit, current_ch);

                out_ptr += out_w_inc;
                in_ptr += in_w_inc;
            } // for W_idx

            out_ptr += out_h_inc;
            in_ptr += in_h_inc;
        } // for H_idx
    } // for out_ch_idx
}

//========================================================
// Convolution 2D with padding for topleft
//========================================================
template <typename i_T, typename w_T, typename o_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void convolution2D_pad(
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
    MLI_ASSERT(perception_area.row_beg == 0);
    MLI_ASSERT(perception_area.clmn_beg == 0);
    const int row_begin = 0;
    const int row_end = perception_area.row_end;
    const int clmn_begin = 0;
    const int clmn_end = perception_area.clmn_end;

    int width = clmn_end - clmn_begin;
    constexpr int unroll = 2;
    int unroll_width = MAX(width - padding_left, 0) & ~(unroll - 1);
    int remainder_width = width - unroll_width;

    // There is no bias in MLI3.0
    const bool has_bias = biases != nullptr;

    for (int out_ch_idx = 0; out_ch_idx < out.ch; out_ch_idx+= get_number_lanes<acc_T>()) {
        int remaining_ch = out.ch - out_ch_idx;
        int current_ch = MIN(remaining_ch, get_number_lanes<acc_T>()); /* nr channels computed in this loop iteration */
        auto output_params = adjust_quant_params_v(&quant_params, out_ch_idx);

        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            for (int W_idx = 0; W_idx < remainder_width; W_idx++) {
                // Define area of input and filter for convolution
                // comp - compensation values for valid area definition
                mli_compensations comp = mli_prv_valid_area_compensations(
                        H_idx, W_idx, in.height, in.width,
                        weights.kernel_height, weights.kernel_width,
                        stride_height, stride_width, padding_left, padding_top,
                        dilation_height, dilation_width);
                const int rows = weights.kernel_height - comp.kernel_top - comp.kernel_bottom;
                const int clmns = weights.kernel_width - comp.kernel_right - comp.kernel_left;
                const int h_idx_in = (H_idx * stride_height - padding_top + comp.in_top);
                const int w_idx_in = (W_idx * stride_width - padding_left + comp.in_left);

                MLI_CONV_OUT_PTR(o_T) out_ptr = out.ptr
                        + out.row_mem_stride * H_idx
                        + out.col_mem_stride * W_idx
                        + out.ch_mem_stride * out_ch_idx;
                const MLI_PTR(i_T) in_ptr = in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in;

                const MLI_PTR(w_T) w_ptr = weights.ptr
                        + weights.row_mem_stride * comp.kernel_top
                        + weights.col_mem_stride * comp.kernel_left
                        + weights.out_ch_mem_stride * out_ch_idx;

                acc_T accu = mli_math_mul_fx<i_T, acc_T>(0, 0);

                if (has_bias) {
                    accu = mli::krn::bias_additive(&biases[out_ch_idx], accu, &output_params);
                    accu = mli::krn::weights_additive(w_ptr, accu, &quant_params, clmns, rows, in.ch,
                                                weights.col_mem_stride,
                                                weights.row_mem_stride,
                                                weights.in_ch_mem_stride);
                } else {
                    if(rows * clmns != weights.kernel_height * weights.kernel_width) {
                        const MLI_PTR(w_T) w_ptr_full = weights.ptr
                                + weights.in_ch_mem_stride * 0
                                + weights.out_ch_mem_stride * out_ch_idx;

                        acc_T zero = mli_math_mul_fx<i_T, acc_T>(0, 0);

                        // out_val = 0 - (-out_val - sum_i(w_full*x_zp))
                        accu = mli::krn::weights_additive(w_ptr_full, mli_math_sub<acc_T>(zero, accu), &quant_params,
                            weights.kernel_width, weights.kernel_height, in.ch,
                            weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride);
                        accu = mli_math_sub<acc_T>(zero, accu);

                        // out_val = out_val - sum_i(w_valid*x_zp))
                        accu = mli::krn::weights_additive(w_ptr, accu, &quant_params, clmns, rows, in.ch,
                            weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride);
                    }
                }

                accu = mli::krn::vdsp::dotprod3D_v_variable_krn_sz(in_ptr, w_ptr, clmns, rows, in.ch,
                          in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                          weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                          accu);

                // Cast result to output type, apply built-in ReLU Applying and write result
                mli::krn::result_cast_relu_store_v(out_ptr, accu, &output_params, val_min_limit, val_max_limit, current_ch);
            } // for W_idx remainder width

            mli_compensations comp = mli_prv_valid_area_compensations(
                    H_idx, remainder_width, in.height, in.width,
                    weights.kernel_height, weights.kernel_width,
                    stride_height, stride_width, padding_left, padding_top,
                    dilation_height, dilation_width);

            // check the compensations to be zero only in case we use the unrolled loop
            MLI_ASSERT((width == remainder_width) || (comp.kernel_right == 0));
            MLI_ASSERT((width == remainder_width) || (comp.kernel_left == 0));
            MLI_ASSERT((width == remainder_width) || (comp.in_left == 0));
            comp.kernel_right = 0;
            comp.kernel_left = 0;
            comp.in_left = 0;

            const int rows = weights.kernel_height - comp.kernel_top - comp.kernel_bottom;
            const int clmns = weights.kernel_width - comp.kernel_right - comp.kernel_left;
            const int h_idx_in = (H_idx * stride_height - padding_top + comp.in_top);
            const int w_idx_in = (remainder_width * stride_width - padding_left + comp.in_left);
            MLI_CONV_OUT_PTR(o_T) out_ptr = out.ptr
                    + out.row_mem_stride * H_idx
                    + out.col_mem_stride * remainder_width
                    + out.ch_mem_stride * out_ch_idx;
            const MLI_PTR(i_T) in_ptr = in.ptr
                    + in.row_mem_stride * h_idx_in
                    + in.col_mem_stride * w_idx_in;

            const MLI_PTR(w_T) w_ptr = weights.ptr
                    + weights.row_mem_stride * comp.kernel_top
                    + weights.col_mem_stride * comp.kernel_left
                    + weights.out_ch_mem_stride * out_ch_idx;

            acc_T pre_accu = mli_math_mul_fx<i_T, acc_T>(0, 0);

            if (has_bias) {
                pre_accu = mli::krn::bias_additive(&biases[out_ch_idx], pre_accu, &output_params);
                pre_accu = mli::krn::weights_additive(w_ptr, pre_accu, &quant_params, clmns, rows, in.ch,
                                            weights.col_mem_stride,
                                            weights.row_mem_stride,
                                            weights.in_ch_mem_stride);
            } else {
                if(rows * clmns != weights.kernel_height * weights.kernel_width) {
                    const MLI_PTR(w_T) w_ptr_full = weights.ptr
                            + weights.in_ch_mem_stride * 0
                            + weights.out_ch_mem_stride * out_ch_idx;

                    acc_T zero = mli_math_mul_fx<i_T, acc_T>(0, 0);

                    // out_val = 0 - (-out_val - sum_i(w_full*x_zp))
                    pre_accu = mli::krn::weights_additive(w_ptr_full, mli_math_sub<acc_T>(zero, pre_accu), &quant_params,
                        weights.kernel_width, weights.kernel_height, in.ch,
                        weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride);
                    pre_accu = mli_math_sub<acc_T>(zero, pre_accu);

                    // out_val = out_val - sum_i(w_valid*x_zp))
                    pre_accu = mli::krn::weights_additive(w_ptr, pre_accu, &quant_params, clmns, rows, in.ch,
                        weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride);
                }
            }

            for (int W_idx = remainder_width; W_idx < width; W_idx+=unroll) {
                auto accu = init_accu_grp(pre_accu);

                accu = mli::krn::vdsp::dotprod3D_v_variable_krn_sz_unroll<unroll>(in_ptr, w_ptr, clmns, rows, in.ch,
                          in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride, in.col_mem_stride * stride_width,
                          weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                          accu);

                // Cast result to output type, apply built-in ReLU Applying and write result
                mli::krn::result_cast_relu_store_v(out_ptr, accu.accu0, &output_params, val_min_limit, val_max_limit, current_ch);
                out_ptr += out.col_mem_stride;
                mli::krn::result_cast_relu_store_v(out_ptr, accu.accu1, &output_params, val_min_limit, val_max_limit, current_ch);
                if (unroll > 2) {
                    out_ptr += out.col_mem_stride;
                    mli::krn::result_cast_relu_store_v(out_ptr, accu.accu2, &output_params, val_min_limit, val_max_limit, current_ch);
                    out_ptr += out.col_mem_stride;
                    mli::krn::result_cast_relu_store_v(out_ptr, accu.accu3, &output_params, val_min_limit, val_max_limit, current_ch);
                }
                out_ptr += out.col_mem_stride;
                in_ptr += in.col_mem_stride * stride_width * unroll;
            } // for W_idx unrolled loop
        } // for H_idx
    } // for out_ch_idx
}

//========================================================
// Convolution 2D
//========================================================
template <typename i_T, typename w_T, typename o_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void convolution2D(
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
        convolution2D_nopad<i_T, w_T, o_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
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
    if ((fix_kernel_width != 1) || (fix_kernel_height != 1)) {
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

                convolution2D_pad<i_T, w_T, o_T, b_T, acc_T, quant_T>(
                        in_, w_, biases, out_, area, quant_params,
                        val_min_limit, val_max_limit,
                        stride_h, stride_w,
                        dilation_h, dilation_w,
                        p_top, p_left,
                        p_bot, p_right);
            }
        }
    }
}

template <typename i_T, typename w_T, typename o_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void depthwise_convolution2D_nopad(
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
    const int row_begin = perception_area.row_beg;
    const int row_end = perception_area.row_end;
    const int clmn_begin = perception_area.clmn_beg;
    const int clmn_end = perception_area.clmn_end;

    MLI_ASSERT(quant_params_get_weigths_zeropoint(&quant_params) == 0); /* this optimized implementation assumes no zero offset for weights */

    // There is no bias in MLI3.0
    const bool has_bias = biases != nullptr;

    for (int in_ch_idx = 0; in_ch_idx < in.ch; in_ch_idx += get_number_lanes<acc_T>()) {
        const int out_ch_idx = in_ch_idx;
        int remaining_ch = in.ch - in_ch_idx;
        int current_ch = MIN(remaining_ch, get_number_lanes<acc_T>()); /* nr channels computed in this loop iteration */
        const int rows = weights.kernel_height;
        const int clmns = weights.kernel_width;

        const MLI_PTR(w_T) w_ptr = weights.ptr + weights.out_ch_mem_stride * out_ch_idx;
        auto output_params = adjust_quant_params_v(&quant_params, out_ch_idx);

        acc_T base_accu = mli_math_mul_fx<i_T, acc_T>(0, 0);
        if (has_bias) {
            base_accu = mli::krn::bias_additive(&biases[out_ch_idx], base_accu, &output_params);
            base_accu = mli::krn::weights_additive(w_ptr, base_accu, &quant_params, clmns, rows, 1 /*channel*/,
                                    weights.col_mem_stride,
                                    weights.row_mem_stride,
                                    0 /* channel step */);
        }

        const int h_idx_in = (row_begin * stride_height - padding_top);
        const int w_idx_in = (clmn_begin * stride_width - padding_left);
        const MLI_PTR(i_T) __restrict in_ptr = in.ptr
                + in.row_mem_stride * h_idx_in
                + in.col_mem_stride * w_idx_in
                + in.ch_mem_stride * in_ch_idx;

        MLI_CONV_OUT_PTR(o_T) __restrict out_ptr = out.ptr
                + out.row_mem_stride * row_begin
                + out.col_mem_stride * clmn_begin
                + out.ch_mem_stride * out_ch_idx;

        int width = clmn_end - clmn_begin;
        int out_col_inc = out.col_mem_stride;
        int out_row_inc = out.row_mem_stride - out_col_inc * (width);
        int in_col_inc = in.col_mem_stride * stride_width;
        int in_row_inc = in.row_mem_stride * stride_height - (in_col_inc * (width));
        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
#pragma clang loop pipeline(enable)
#pragma clang loop pipeline_options(0x10)
            for (int W_idx = 0; W_idx < width; W_idx++) {
                // Convolution core. Here calculations performes in a unfolded expression way:
                // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                //============================================
                acc_T accu = base_accu;
                if (fix_kernel_height > 4) {
                    // for larger kernelheights we store some of the pointers in a vector to reduce scalar register pressure.
                    accu = mli::krn::dotprod2D_vv_ptrvector(in_ptr, w_ptr, accu, clmns, rows,
                                        in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                                        weights.col_mem_stride,
                                        weights.row_mem_stride);
                } else {
                    accu = mli::krn::vdsp::dotprod2D_vv_unrolled(in_ptr, w_ptr, accu, clmns, rows,
                                        in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                                        weights.col_mem_stride,
                                        weights.row_mem_stride);
                }

                // Cast result to output type, apply built-in ReLU Applying and write result
                mli::krn::result_cast_relu_store_v(out_ptr, accu, &output_params, val_min_limit, val_max_limit, current_ch);
                in_ptr += in_col_inc;
                out_ptr += out_col_inc;
            } // for W_idx
            in_ptr += in_row_inc;
            out_ptr += out_row_inc;
        } // for H_idx
    } // for in_ch_idx
}

template <typename i_T, typename w_T, typename o_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void depthwise_convolution2D_nopad_wunroll(
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
    const int row_begin = perception_area.row_beg;
    const int row_end = perception_area.row_end;
    const int clmn_begin = perception_area.clmn_beg;
    const int clmn_end = perception_area.clmn_end;

    MLI_ASSERT(quant_params_get_weigths_zeropoint(&quant_params) == 0); /* this optimized implementation assumes no zero offset for weights */

    constexpr int numaccuregs = sizeof(acc_T) / sizeof(vNint_t);
    constexpr int unroll = (numaccuregs > 2) ? 2 : 4;

    // There is no bias in MLI3.0
    const bool has_bias = biases != nullptr;

    for (int in_ch_idx = 0; in_ch_idx < in.ch; in_ch_idx += get_number_lanes<acc_T>()) {
        const int out_ch_idx = in_ch_idx;
        int remaining_ch = in.ch - in_ch_idx;
        int current_ch = MIN(remaining_ch, get_number_lanes<acc_T>()); /* nr channels computed in this loop iteration */
        const int rows = weights.kernel_height;
        const int clmns = weights.kernel_width;

        const MLI_PTR(w_T) w_ptr = weights.ptr + weights.out_ch_mem_stride * out_ch_idx;
        auto output_params = adjust_quant_params_v(&quant_params, out_ch_idx);

        acc_T base_accu = mli_math_mul_fx<i_T, acc_T>(0, 0);
        if (has_bias) {
            base_accu = mli::krn::bias_additive(&biases[out_ch_idx], base_accu, &output_params);
            base_accu = mli::krn::weights_additive(w_ptr, base_accu, &quant_params, clmns, rows, 1 /*channel*/,
                                    weights.col_mem_stride,
                                    weights.row_mem_stride,
                                    0 /* channel step */);
        }

        const int h_idx_in = (row_begin * stride_height - padding_top);
        const int w_idx_in = (clmn_begin * stride_width - padding_left);
        const MLI_PTR(i_T) __restrict in_ptr = in.ptr
                + in.row_mem_stride * h_idx_in
                + in.col_mem_stride * w_idx_in
                + in.ch_mem_stride * in_ch_idx;

        MLI_CONV_OUT_PTR(o_T) __restrict out_ptr = out.ptr
                + out.row_mem_stride * row_begin
                + out.col_mem_stride * clmn_begin
                + out.ch_mem_stride * out_ch_idx;

        int width = clmn_end - clmn_begin;
        int out_col_inc = out.col_mem_stride;
        int out_row_inc = out.row_mem_stride - out_col_inc * (width);
        int in_col_inc = in.col_mem_stride * stride_width;
        int in_row_inc = in.row_mem_stride * stride_height - (in_col_inc * (width));

        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            int W_idx = 0;
            for (W_idx = 0; W_idx <= width - unroll; W_idx+=unroll) {
                // Convolution core. Here calculations performes in a unfolded expression way:
                // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                //============================================
                auto accu = init_accu_grp(base_accu);
                accu = mli::krn::vdsp::dotprod2D_vv_wunroll<unroll>(in_ptr, w_ptr, accu, clmns, rows,
                                    in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in_col_inc,
                                    weights.col_mem_stride,
                                    weights.row_mem_stride);

                // Cast result to output type, apply built-in ReLU Applying and write result
                mli::krn::result_cast_relu_store_v(out_ptr, accu.accu0, &output_params, val_min_limit, val_max_limit, current_ch);
                in_ptr += in_col_inc;
                out_ptr += out_col_inc;
                if (unroll > 1) {
                    mli::krn::result_cast_relu_store_v(out_ptr, accu.accu1, &output_params, val_min_limit, val_max_limit, current_ch);
                    in_ptr += in_col_inc;
                    out_ptr += out_col_inc;
                }
                if (unroll > 2) {
                    mli::krn::result_cast_relu_store_v(out_ptr, accu.accu2, &output_params, val_min_limit, val_max_limit, current_ch);
                    in_ptr += in_col_inc;
                    out_ptr += out_col_inc;
                }
                if (unroll > 3) {
                    mli::krn::result_cast_relu_store_v(out_ptr, accu.accu3, &output_params, val_min_limit, val_max_limit, current_ch);
                    in_ptr += in_col_inc;
                    out_ptr += out_col_inc;
                }
            } // for W_idx
            for ( ; W_idx < width; W_idx++) {
                // Convolution core. Here calculations performes in a unfolded expression way:
                // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                //============================================
                acc_T accu = base_accu;
                accu = mli::krn::vdsp::dotprod2D_vv_unrolled(in_ptr, w_ptr, accu, clmns, rows,
                                    in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                                    weights.col_mem_stride,
                                    weights.row_mem_stride);

                // Cast result to output type, apply built-in ReLU Applying and write result
                mli::krn::result_cast_relu_store_v(out_ptr, accu, &output_params, val_min_limit, val_max_limit, current_ch);
                in_ptr += in_col_inc;
                out_ptr += out_col_inc;
            } // for W_idx
            in_ptr += in_row_inc;
            out_ptr += out_row_inc;
        } // for H_idx
    } // for in_ch_idx
}

template <typename i_T, typename w_T, typename o_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void depthwise_convolution2D_pad(
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
    MLI_ASSERT(perception_area.row_beg == 0);
    MLI_ASSERT(perception_area.clmn_beg == 0);
    const int row_begin = 0;
    const int row_end = perception_area.row_end;
    const int clmn_begin = 0;
    const int clmn_end = perception_area.clmn_end;
    int width = clmn_end - clmn_begin;
    constexpr int unroll = 1;
    int unroll_width = MAX(width - padding_left, 0) & ~(unroll - 1);
    int remainder_width = width - unroll_width;

    MLI_ASSERT(quant_params_get_weigths_zeropoint(&quant_params) == 0); /* this optimized implementation assumes no zero offset for weights */

    // There is no bias in MLI3.0
    const bool has_bias = biases != nullptr;

    for (int in_ch_idx = 0; in_ch_idx < in.ch; in_ch_idx += get_number_lanes<acc_T>()) {
        const int out_ch_idx = in_ch_idx;
        int remaining_ch = in.ch - in_ch_idx;
        int current_ch = MIN(remaining_ch, get_number_lanes<acc_T>()); /* nr channels computed in this loop iteration */
        auto output_params = adjust_quant_params_v(&quant_params, out_ch_idx);

        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            for (int W_idx = 0; W_idx < remainder_width; W_idx++) {
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


                const MLI_PTR(i_T) in_ptr = in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in
                        + in.ch_mem_stride * in_ch_idx;
                MLI_CONV_OUT_PTR(o_T) out_ptr = out.ptr
                        + out.row_mem_stride * H_idx
                        + out.col_mem_stride * W_idx
                        + out.ch_mem_stride * out_ch_idx;

                const MLI_PTR(w_T) w_ptr = weights.ptr
                        + weights.row_mem_stride * comp.kernel_top
                        + weights.col_mem_stride * comp.kernel_left
                        + weights.in_ch_mem_stride * 0
                        + weights.out_ch_mem_stride * out_ch_idx;

                // Convolution core. Here calculations performes in a unfolded expression way:
                // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                //============================================
                acc_T accu = mli_math_mul_fx<i_T, acc_T>(0, 0);
                if (has_bias) {
                    accu = mli::krn::bias_additive(&biases[out_ch_idx], accu, &output_params);
                    accu = mli::krn::weights_additive(w_ptr, accu, &quant_params, clmns, rows, 1 /*channel*/,
                                            weights.col_mem_stride,
                                            weights.row_mem_stride,
                                            0 /* channel step */);
                } else {
                    if(rows * clmns != weights.kernel_height * weights.kernel_width) {
                        const MLI_PTR(w_T) w_ptr_full = weights.ptr
                                + weights.in_ch_mem_stride * 0
                                + weights.out_ch_mem_stride * out_ch_idx;

                        // This part emulate dotproduct out of valid area. it adds sum_i(w_full*x_zp) for the whole kernel,
                        // and afterward subtracts sum_i(w_valid*x_zp) part for valid area which we don't need due to
                        // conducted core dotproduct;
                        acc_T zero = mli_math_mul_fx<i_T, acc_T>(0, 0);

                        // out_val = 0 - (-out_val - sum_i(w_full*x_zp))
                        accu = mli::krn::weights_additive(w_ptr_full, mli_math_sub<acc_T>(zero, accu), &quant_params,
                            weights.kernel_width, weights.kernel_height, 1 /* channel */,
                            weights.col_mem_stride, weights.row_mem_stride, 0 /* channel step */);
                        accu = mli_math_sub<acc_T>(zero, accu);

                        // out_val = out_val - sum_i(w_valid*x_zp))
                        accu = mli::krn::weights_additive(w_ptr, accu, &quant_params,
                            clmns, rows, 1 /* channel */, weights.col_mem_stride, weights.row_mem_stride, 0 /* channel step */);
                    }
                }

                accu = mli::krn::dotprod2D_vv(in_ptr, w_ptr, accu, clmns, rows,
                                    in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                                    weights.col_mem_stride,
                                    weights.row_mem_stride);

                // Cast result to output type, apply built-in ReLU Applying and write result
                mli::krn::result_cast_relu_store_v(out_ptr, accu, &output_params, val_min_limit, val_max_limit, current_ch);
            } // for W_idx remainder

            // Define area of input and filter for convolution
            // comp - compensation values for valid area definition
            mli_compensations comp = mli_prv_valid_area_compensations(
                    H_idx, remainder_width, in.height, in.width,
                    weights.kernel_height, weights.kernel_width,
                    stride_height, stride_width, padding_left, padding_top,
                    dilation_height, dilation_width);

            // check the compensations to be zero only in case we use the unrolled loop
            MLI_ASSERT((width == remainder_width) || (comp.kernel_right == 0));
            MLI_ASSERT((width == remainder_width) || (comp.kernel_left == 0));
            MLI_ASSERT((width == remainder_width) || (comp.in_left == 0));
            comp.kernel_right = 0;
            comp.kernel_left = 0;
            comp.in_left = 0;

            const int rows = weights.kernel_height - comp.kernel_top - comp.kernel_bottom;
            const int clmns = weights.kernel_width - comp.kernel_right - comp.kernel_left;
            const int h_idx_in = (H_idx * stride_height - padding_top + comp.in_top);
            const int w_idx_in = (remainder_width * stride_width - padding_left + comp.in_left);


            const MLI_PTR(i_T) in_ptr = in.ptr
                    + in.row_mem_stride * h_idx_in
                    + in.col_mem_stride * w_idx_in
                    + in.ch_mem_stride * in_ch_idx;
            MLI_CONV_OUT_PTR(o_T) out_ptr = out.ptr
                    + out.row_mem_stride * H_idx
                    + out.col_mem_stride * remainder_width
                    + out.ch_mem_stride * out_ch_idx;

            const MLI_PTR(w_T) w_ptr = weights.ptr
                    + weights.row_mem_stride * comp.kernel_top
                    + weights.col_mem_stride * comp.kernel_left
                    + weights.in_ch_mem_stride * 0
                    + weights.out_ch_mem_stride * out_ch_idx;

            // Convolution core. Here calculations performes in a unfolded expression way:
            // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
            //============================================
            acc_T pre_accu = mli_math_mul_fx<i_T, acc_T>(0, 0);
            if (has_bias) {
                pre_accu = mli::krn::bias_additive(&biases[out_ch_idx], pre_accu, &output_params);
                pre_accu = mli::krn::weights_additive(w_ptr, pre_accu, &quant_params, clmns, rows, 1 /*channel*/,
                                        weights.col_mem_stride,
                                        weights.row_mem_stride,
                                        0 /* channel step */);
            } else {
                if(rows * clmns != weights.kernel_height * weights.kernel_width) {
                    const MLI_PTR(w_T) w_ptr_full = weights.ptr
                            + weights.in_ch_mem_stride * 0
                            + weights.out_ch_mem_stride * out_ch_idx;

                    // This part emulate dotproduct out of valid area. it adds sum_i(w_full*x_zp) for the whole kernel,
                    // and afterward subtracts sum_i(w_valid*x_zp) part for valid area which we don't need due to
                    // conducted core dotproduct;
                    acc_T zero = mli_math_mul_fx<i_T, acc_T>(0, 0);

                    // out_val = 0 - (-out_val - sum_i(w_full*x_zp))
                    pre_accu = mli::krn::weights_additive(w_ptr_full, mli_math_sub<acc_T>(zero, pre_accu), &quant_params,
                        weights.kernel_width, weights.kernel_height, 1 /* channel */,
                        weights.col_mem_stride, weights.row_mem_stride, 0 /* channel step */);
                    pre_accu = mli_math_sub<acc_T>(zero, pre_accu);

                    // out_val = out_val - sum_i(w_valid*x_zp))
                    pre_accu = mli::krn::weights_additive(w_ptr, pre_accu, &quant_params,
                        clmns, rows, 1 /* channel */, weights.col_mem_stride, weights.row_mem_stride, 0 /* channel step */);
                }
            }

            for (int W_idx = remainder_width; W_idx < width; W_idx++) {
                acc_T accu = pre_accu;
                accu = mli::krn::dotprod2D_vv(in_ptr, w_ptr, accu, clmns, rows,
                                    in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                                    weights.col_mem_stride,
                                    weights.row_mem_stride);

                // Cast result to output type, apply built-in ReLU Applying and write result
                mli::krn::result_cast_relu_store_v(out_ptr, accu, &output_params, val_min_limit, val_max_limit, current_ch);
                out_ptr += out.col_mem_stride;
                in_ptr += in.col_mem_stride * stride_width * unroll;
            } // for W_idx
        } // for H_idx
    } // for in_ch_idx
}

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
        if ((fix_kernel_width > 0) && (fix_kernel_height > 0)) {
            depthwise_convolution2D_nopad<i_T, w_T, o_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
                    in, weights, biases, out, perception_area_nopad, quant_params,
                    val_min_limit, val_max_limit,
                    stride_height, stride_width,
                    dilation_height, dilation_width,
                    padding_top, padding_left,
                    padding_bot, padding_right);
        } else {
            depthwise_convolution2D_nopad_wunroll<i_T, w_T, o_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
                    in, weights, biases, out, perception_area_nopad, quant_params,
                    val_min_limit, val_max_limit,
                    stride_height, stride_width,
                    dilation_height, dilation_width,
                    padding_top, padding_left,
                    padding_bot, padding_right);
        }
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
            depthwise_convolution2D_pad<i_T, w_T, o_T, b_T, acc_T, quant_T>(
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

#endif // _MLI_KRN_CONVOLUTION_VDSP_H_
