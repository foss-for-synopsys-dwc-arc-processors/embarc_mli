/*
* Copyright 2020, Synopsys, Inc.
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
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void convolution2D_nopad(
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
    int width = clmn_end - clmn_begin;
    constexpr int numaccuregs = sizeof(acc_T) / sizeof(vNint_t);
    constexpr int unroll = numaccuregs > 2 ? 2 : 4;
    int remainder_width = width & (unroll - 1);
    int unroll_width = width - remainder_width;

    for (int out_ch_idx = 0; out_ch_idx < out.ch; out_ch_idx+= get_number_lanes<acc_T>()) {
        int remaining_ch = out.ch - out_ch_idx;
        int current_ch = MIN(remaining_ch, get_number_lanes<acc_T>()); /* nr channels computed in this loop iteration */
        const MLI_PTR(w_T) w_ptr = weights.ptr
                + weights.out_ch_mem_stride * out_ch_idx;
        const int rows = (fix_kernel_height > 0) ? fix_kernel_height : weights.kernel_height;
        const int clmns = (fix_kernel_width > 0) ? fix_kernel_width : weights.kernel_width;

        auto output_params = adjust_quant_params_v(&quant_params, out_ch_idx);

        acc_T pre_accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
        pre_accu = mli::krn::bias_additive(&biases[out_ch_idx], pre_accu, &quant_params);

        pre_accu = mli::krn::weights_additive(w_ptr, pre_accu, &quant_params, clmns, rows, in.ch,
                                    weights.col_mem_stride,
                                    weights.row_mem_stride,
                                    weights.in_ch_mem_stride);

        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            int W_idx = clmn_begin;
            for (int W_cnt = 0; W_cnt < remainder_width; W_cnt++, W_idx++) {

                acc_T accu = pre_accu;

                const int h_idx_in = H_idx * stride_height - padding_top;
                const int w_idx_in = W_idx * stride_width - padding_left;
                MLI_CONV_OUT_PTR(io_T) out_ptr = out.ptr
                        + out.row_mem_stride * H_idx
                        + out.col_mem_stride * W_idx
                        + out.ch_mem_stride * out_ch_idx;
                const MLI_PTR(io_T) in_ptr = in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in;

                if ((fix_kernel_width == 1) && (fix_kernel_height == 1)) {
                    accu = mli::krn::dotprod1D_v(in_ptr, w_ptr, accu, in.ch, in.ch_mem_stride, weights.in_ch_mem_stride);
                } else if ((fix_kernel_width > 0) && (fix_kernel_height > 0)) {
                    accu = mli::krn::dotprod3D_v<io_T, w_T, acc_T, /*fixedsize*/true>(in_ptr, w_ptr, clmns, rows, in.ch,
                              in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                              weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                              accu);
                } else if (weights.row_mem_stride == clmns * weights.col_mem_stride) {
                    accu = mli::krn::dotprod3D_v_nopad(in_ptr, w_ptr, clmns, rows, in.ch,
                              in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                              weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                              accu);
                } else {
                    accu = mli::krn::dotprod3D_v<io_T, w_T, acc_T, /*fixedsize*/false>(in_ptr, w_ptr, clmns, rows, in.ch,
                              in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                              weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                              accu);
                }
                // Cast result to output type, apply built-in ReLU Applying and write result
                mli::krn::result_cast_relu_store_v(out_ptr, accu, &output_params, val_min_limit, val_max_limit, current_ch);
            } // for W_idx
            for (int W_cnt = 0; W_cnt < unroll_width; W_cnt+=unroll, W_idx+=unroll) {
                int in_unroll_inc = stride_width * in.col_mem_stride;
                int out_unroll_inc = out.col_mem_stride;
                auto accu = init_accu_grp(pre_accu);

                const int h_idx_in = H_idx * stride_height - padding_top;
                const int w_idx_in = W_idx * stride_width - padding_left;
                MLI_CONV_OUT_PTR(io_T) out_ptr = out.ptr
                        + out.row_mem_stride * H_idx
                        + out.col_mem_stride * W_idx
                        + out.ch_mem_stride * out_ch_idx;
                const MLI_PTR(io_T) in_ptr = in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in;

                if ((fix_kernel_width == 1) && (fix_kernel_height == 1)) {
                    accu = mli::krn::dotprod1D_v_unroll<unroll>(in_ptr, w_ptr, accu, in.ch, in.ch_mem_stride, in.col_mem_stride, weights.in_ch_mem_stride);
                } else if ((fix_kernel_width > 0) && (fix_kernel_height > 0) && (dilation_width == stride_width)) {
                    // unrolled version with fixed kernelsize
                    accu = mli::krn::dotprod3D_v_unroll<unroll, true>(in_ptr, w_ptr, clmns, rows, in.ch,
                              in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride, in_unroll_inc,
                              weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                              accu);
                } else if (weights.row_mem_stride == clmns * weights.col_mem_stride) {
                    accu = mli::krn::dotprod3D_v_nopad_unroll<unroll>(in_ptr, w_ptr, clmns, rows, in.ch,
                              in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                              weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride, in_unroll_inc,
                              accu);
                } else {
                    accu = mli::krn::dotprod3D_v_unroll<unroll, false>(in_ptr, w_ptr, clmns, rows, in.ch,
                              in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride, in_unroll_inc,
                              weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                              accu);
                }
                // Cast result to output type, apply built-in ReLU Applying and write result
                mli::krn::result_cast_relu_store_v(out_ptr, accu.accu0, &output_params, val_min_limit, val_max_limit, current_ch);
                out_ptr += out_unroll_inc;
                mli::krn::result_cast_relu_store_v(out_ptr, accu.accu1, &output_params, val_min_limit, val_max_limit, current_ch);
                if (unroll > 2) {
                    out_ptr += out_unroll_inc;
                    mli::krn::result_cast_relu_store_v(out_ptr, accu.accu2, &output_params, val_min_limit, val_max_limit, current_ch);
                    out_ptr += out_unroll_inc;
                    mli::krn::result_cast_relu_store_v(out_ptr, accu.accu3, &output_params, val_min_limit, val_max_limit, current_ch);
                }
            } // for W_idx
        } // for H_idx
    } // for out_ch_idx
}

//========================================================
// Convolution 2D with padding
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void convolution2D_pad(
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
            for (int out_ch_idx = 0; out_ch_idx < out.ch; out_ch_idx+= get_number_lanes<acc_T>()) {
                int remaining_ch = out.ch - out_ch_idx;
                int current_ch = MIN(remaining_ch, get_number_lanes<acc_T>()); /* nr channels computed in this loop iteration */
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

                auto output_params = adjust_quant_params_v(&quant_params, out_ch_idx);

                acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
                accu = mli::krn::bias_additive(&biases[out_ch_idx], accu, &quant_params);

                accu = mli::krn::dotprod3D_v(in_ptr, w_ptr, clmns, rows, in.ch,
                          in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                          weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                          accu);

                accu = mli::krn::weights_additive(w_ptr, accu, &quant_params, clmns, rows, in.ch,
                                            weights.col_mem_stride,
                                            weights.row_mem_stride,
                                            weights.in_ch_mem_stride);

                // Cast result to output type, apply built-in ReLU Applying and write result
                mli::krn::result_cast_relu_store_v(out_ptr, accu, &output_params, val_min_limit, val_max_limit, current_ch);
            } // for out_ch_idx
        } // for W_idx
    } // for H_idx
}

//========================================================
// Convolution 2D
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
        convolution2D_nopad<io_T, w_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
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
        rect_t perc_areas[4];
        int areas_num = 0;
        if (padding_top) {
            perc_areas[areas_num].row_beg = 0;
            perc_areas[areas_num].row_end = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = out.width;
        }
        if (padding_bot) {
            perc_areas[areas_num].row_beg = out.height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].row_end = out.height;
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = out.width;
        }
        if (padding_left) {
            perc_areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].row_end = out.height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = CEIL_DIV(padding_left, stride_width);
        }
        if (padding_right) {
            perc_areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].row_end = out.height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = out.width - CEIL_DIV(padding_right, stride_width);
            perc_areas[areas_num++].clmn_end = out.width;
        }
        for(int i = 0; i < areas_num; i ++) {
            convolution2D_pad<io_T, w_T, b_T, acc_T, quant_T>(
                    in, weights, biases, out, perc_areas[i], quant_params,
                    val_min_limit, val_max_limit,
                    stride_height, stride_width,
                    dilation_height, dilation_width,
                    padding_top, padding_left,
                    padding_bot, padding_right);
        }
    }
}

template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void depthwise_convolution2D_nopad(
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
    constexpr int numaccuregs = sizeof(acc_T) / sizeof(vNint_t);
    constexpr int unroll = numaccuregs > 2 ? 1 : 1;

    MLI_ASSERT(quant_params_get_weigths_zeropoint(&quant_params) == 0); /* this optimized implementation assumes no zero offset for weights */

    for (int in_ch_idx = 0; in_ch_idx < in.ch; in_ch_idx += get_number_lanes<acc_T>()) {
        const int out_ch_idx = in_ch_idx;
        int remaining_ch = in.ch - in_ch_idx;
        int current_ch = MIN(remaining_ch, get_number_lanes<acc_T>()); /* nr channels computed in this loop iteration */
        const int rows = weights.kernel_height;
        const int clmns = weights.kernel_width;

        const MLI_PTR(w_T) w_ptr = weights.ptr + weights.out_ch_mem_stride * out_ch_idx;
        auto output_params = adjust_quant_params_v(&quant_params, out_ch_idx);

        acc_T base_accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
        base_accu = mli::krn::bias_additive(&biases[out_ch_idx], base_accu, &quant_params);
        base_accu = mli::krn::weights_additive(w_ptr, base_accu, &quant_params, clmns, rows, 1 /*channel*/,
                                weights.col_mem_stride,
                                weights.row_mem_stride,
                                0 /* channel step */);

        const int h_idx_in = (row_begin * stride_height - padding_top);
        const int w_idx_in = (clmn_begin * stride_width - padding_left);
        const MLI_PTR(io_T) __restrict in_ptr = in.ptr
                + in.row_mem_stride * h_idx_in
                + in.col_mem_stride * w_idx_in
                + in.ch_mem_stride * in_ch_idx;

        MLI_CONV_OUT_PTR(io_T) __restrict out_ptr = out.ptr
                + out.row_mem_stride * row_begin
                + out.col_mem_stride * clmn_begin
                + out.ch_mem_stride * out_ch_idx;

        int out_col_inc = out.col_mem_stride;
        int out_row_inc = out.row_mem_stride - out_col_inc * (clmn_end-clmn_begin);

        if (dilation_width == stride_width) {
            int in_col_inc = in.col_mem_stride * dilation_width;// only valid if stride_width == dilation_width
            int in_row_inc = in.row_mem_stride * stride_height - (in_col_inc * (clmn_end-clmn_begin));
            for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
#pragma clang loop unroll_count(unroll)
                for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {

                    // Convolution core. Here calculations performes in a unfolded expression way:
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    acc_T accu = base_accu;
                    accu = mli::krn::dotprod2D_vv(in_ptr, w_ptr, accu, clmns, rows,
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
        } else {
            // dilation_width != stride_width
            int in_col_inc = in.col_mem_stride * stride_width;
            int in_row_inc = in.row_mem_stride * stride_height - (in_col_inc * (clmn_end-clmn_begin));
            for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
#pragma clang loop unroll_count(unroll)
                for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {

                    // Convolution core. Here calculations performes in a unfolded expression way:
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    acc_T accu = base_accu;
                    accu = mli::krn::dotprod2D_vv(in_ptr, w_ptr, accu, clmns, rows,
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
        } // dilation == stride
    } // for in_ch_idx

}

template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void depthwise_convolution2D_pad(
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

    MLI_ASSERT(quant_params_get_weigths_zeropoint(&quant_params) == 0); /* this optimized implementation assumes no zero offset for weights */

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

            for (int in_ch_idx = 0; in_ch_idx < in.ch; in_ch_idx += get_number_lanes<acc_T>()) {
                const int out_ch_idx = in_ch_idx;
                int remaining_ch = in.ch - in_ch_idx;
                int current_ch = MIN(remaining_ch, get_number_lanes<acc_T>()); /* nr channels computed in this loop iteration */

                const MLI_PTR(io_T) in_ptr = in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in
                        + in.ch_mem_stride * in_ch_idx;
                MLI_CONV_OUT_PTR(io_T) out_ptr = out.ptr
                        + out.row_mem_stride * H_idx
                        + out.col_mem_stride * W_idx
                        + out.ch_mem_stride * out_ch_idx;

                const MLI_PTR(w_T) w_ptr = weights.ptr
                        + weights.row_mem_stride * comp.kernel_top
                        + weights.col_mem_stride * comp.kernel_left
                        + weights.in_ch_mem_stride * 0
                        + weights.out_ch_mem_stride * out_ch_idx;
                auto output_params = adjust_quant_params_v(&quant_params, out_ch_idx);

                // Convolution core. Here calculations performes in a unfolded expression way:
                // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                //============================================
                acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
                accu = mli::krn::bias_additive(&biases[out_ch_idx], accu, &quant_params);
                accu = mli::krn::dotprod2D_vv(in_ptr, w_ptr, accu, clmns, rows,
                                    in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                                    weights.col_mem_stride,
                                    weights.row_mem_stride);
                accu = mli::krn::weights_additive(w_ptr, accu, &quant_params, clmns, rows, 1 /*channel*/,
                                        weights.col_mem_stride,
                                        weights.row_mem_stride,
                                        0 /* channel step */);

                // Cast result to output type, apply built-in ReLU Applying and write result
                mli::krn::result_cast_relu_store_v(out_ptr, accu, &output_params, val_min_limit, val_max_limit, current_ch);
            } // for in_ch_idx
        } // for W_idx
    } // for H_idx
}

template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
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
        depthwise_convolution2D_nopad<io_T, w_T, b_T, acc_T, quant_T>(
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
        rect_t perc_areas[4];
        int areas_num = 0;
        if (padding_top) {
            perc_areas[areas_num].row_beg = 0;
            perc_areas[areas_num].row_end = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = out.width;
        }
        if (padding_bot) {
            perc_areas[areas_num].row_beg = out.height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].row_end = out.height;
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = out.width;
        }
        if (padding_left) {
            perc_areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].row_end = out.height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = CEIL_DIV(padding_left, stride_width);
        }
        if (padding_right) {
            perc_areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].row_end = out.height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = out.width - CEIL_DIV(padding_right, stride_width);
            perc_areas[areas_num++].clmn_end = out.width;
        }
        for(int i = 0; i < areas_num; i ++) {
            depthwise_convolution2D_pad<io_T, w_T, b_T, acc_T, quant_T>(
                    in, weights, biases, out, perc_areas[i], quant_params,
                    val_min_limit, val_max_limit,
                    stride_height, stride_width,
                    dilation_height, dilation_width,
                    padding_top, padding_left,
                    padding_bot, padding_right);
        }
    }
}

#pragma MLI_CODE_SECTION_END()
} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_CONVOLUTION_VDSP_H_
