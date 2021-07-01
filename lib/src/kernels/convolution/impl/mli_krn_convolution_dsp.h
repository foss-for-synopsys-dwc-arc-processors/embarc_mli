/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_CONVOLUTION_DSP_H_
#define _MLI_KRN_CONVOLUTION_DSP_H_

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
namespace dsp {

#pragma MLI_CODE_SECTION_START(".mli_lib")
//========================================================
// Depthwise convolution 2D template
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void depthwise_convolution2D_hwcn_nopad(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &w,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        s8asym_quant_specific_params quant_params,
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
    const int filters = 1;
    const int ch_mul = out.ch / in.ch;
    const int amount_rows = row_end - row_begin;
    const int amount_columns = clmn_end - clmn_begin;
    const int in_compensation_row_loop   = in.row_mem_stride * stride_height * amount_rows;
    const int out_compensation_row_loop  = out.row_mem_stride * amount_rows;
    const int in_compensation_clmn_loop  = in.col_mem_stride * stride_width * amount_columns;
    const int out_compensation_clmn_loop = out.col_mem_stride * amount_columns;
    const int in_increment_clmn_loop     = in.col_mem_stride * stride_width;
    const int in_increment_row_loop      = in.row_mem_stride * stride_height - in_compensation_clmn_loop;
    const int out_increment_row_loop     = out.row_mem_stride  - out_compensation_clmn_loop;
    const int channel_per_loop = 1;
    const int channels_per_loop_v = 2;
    const int out_increment_in_ch_loop = channel_per_loop - out_compensation_row_loop;
    const int out_increment_in_ch_loop_v = channels_per_loop_v - out_compensation_row_loop;

    const MLI_PTR(w_T) __restrict w_ptr = w.ptr;
    int out_ch_idx = 0;

    const MLI_PTR(io_T) __restrict in_ptr = in.ptr
            + in.col_mem_stride * (clmn_begin * stride_width - padding_left) // move to column
            + in.row_mem_stride * (row_begin * stride_height - padding_top); // move to row

    MLI_CONV_OUT_PTR(io_T) __restrict out_ptr = out.ptr
            + out.col_mem_stride * clmn_begin // move to column
            + out.row_mem_stride * row_begin; // move to row

    s8asym_quant_specific_params v2quant_params[] = {quant_params, quant_params};
    for (int in_ch_idx = 0; in_ch_idx < in.ch-1; in_ch_idx += 2) {
        for (int ch_mult_idx = 0; ch_mult_idx < ch_mul; ch_mult_idx++) {
            mli::krn::adjust_quant_params(&v2quant_params[0], out_ch_idx++);
            mli::krn::adjust_quant_params(&v2quant_params[1], out_ch_idx++);

            acc_T bias_add_ch1 = bias_additive(biases++, 0x0, &v2quant_params[0]);
            acc_T bias_add_ch2 = bias_additive(biases++, 0x0, &v2quant_params[1]);

            __v2i32_t v2acc_weights_add = {bias_add_ch1, bias_add_ch2};
            v2acc_weights_add = weights_additive_v(w_ptr, &v2acc_weights_add, &quant_params, w.kernel_width, w.kernel_height,
                    w.col_mem_stride, w.row_mem_stride);
            __builtin_assume(amount_rows > 0);
            for (int H_idx = 0; H_idx < amount_rows; H_idx++) {
                __builtin_assume(amount_columns > 0);

                __v2i32_t v2accu_dotprod = v2acc_weights_add;
                for (int W_idx = 0; W_idx < amount_columns; W_idx++) {
                    dotprod2D_hwc_v(&in_ptr, &w_ptr, &v2accu_dotprod, w.kernel_width, w.kernel_height,
                                    in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                                    w.col_mem_stride, w.row_mem_stride);
                    //compensite increment of input tensor pointer from dotprod2D_hwc_v function
                    in_ptr += in_increment_clmn_loop - w.kernel_height * in.row_mem_stride * dilation_height;
                    //compensite increment of weights pointer from dotprod2D_hwc_v function
                    w_ptr -= w.kernel_height * w.row_mem_stride;

                    // Cast result to output type
                    result_cast_relu_store_v(out_ptr, &v2accu_dotprod, v2quant_params, val_min_limit, val_max_limit);
                    out_ptr += out.col_mem_stride;

                    v2accu_dotprod = v2acc_weights_add;
                } // for W_idx
                in_ptr += in_increment_row_loop;
                out_ptr += out_increment_row_loop;
            } // for H_idx
            in_ptr -= in_compensation_row_loop;
            out_ptr += out_increment_in_ch_loop_v;
            // out_ch_idx++;
            w_ptr += 2;
            // biases++;
        } // for ch_mult_idx
        in_ptr += 2 * filters;
    } // for in_ch_idx

    if (in.ch & 1) {
        for (int ch_mult_idx = 0; ch_mult_idx < ch_mul; ch_mult_idx++) {
            mli::krn::adjust_quant_params(&quant_params, out_ch_idx);

            acc_T global_other_additives = weights_additive(w_ptr, 0x0, &quant_params, w.kernel_width, w.kernel_height,
                    w.col_mem_stride, w.row_mem_stride);
            global_other_additives += bias_additive(biases, 0x0, &quant_params);
            __v2i32_t v2global_other_additives = {global_other_additives, global_other_additives};

            for (int H_idx = 0; H_idx < amount_rows; H_idx++) {
                for (int W_idx = 0; W_idx < amount_columns - 1; W_idx += 2) {
                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    __v2i32_t accu = v2global_other_additives;
                    accu = dotprod2D_inp_width_v(&in_ptr, &w_ptr, &accu, w.kernel_width, w.kernel_height,
                            in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                            w.col_mem_stride, w.row_mem_stride, in_increment_clmn_loop);
                    //compensite increment of input tensor pointer from dotprod2D_hwc_v function
                    in_ptr += 2 * in_increment_clmn_loop - w.kernel_height * in.row_mem_stride * dilation_height;
                    //compensite increment of weights pointer from dotprod2D_hwc_v function
                    w_ptr -= w.kernel_height * w.row_mem_stride;
                    // Cast result to output type
                    result_cast_relu_store_inp_width_v(out_ptr, &accu, &quant_params, val_min_limit, val_max_limit, out.col_mem_stride);
                    out_ptr += 2 * out.col_mem_stride;
                } // for W_idx
                if (amount_columns & 0x1) {
                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    acc_T accu = 0;
                    accu = dotprod2D(&in_ptr, &w_ptr, accu, w.kernel_width, w.kernel_height,
                            in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, w.col_mem_stride, w.row_mem_stride);
                    //compensite increment of input tensor pointer from dotprod2D_hwc_v function
                    in_ptr += in_increment_clmn_loop - w.kernel_height * in.row_mem_stride * dilation_height;
                    //compensite increment of weights pointer from dotprod2D_hwc_v function
                    w_ptr -= w.kernel_height * w.row_mem_stride;
                    accu += global_other_additives;

                    // Cast result to output type
                    result_cast_relu_store(out_ptr, accu, &quant_params, val_min_limit, val_max_limit);
                    out_ptr += out.col_mem_stride;
                }
                in_ptr += in_increment_row_loop;
                out_ptr += out_increment_row_loop;
            } // for H_idx
            in_ptr -= in_compensation_row_loop;
            out_ptr += out_increment_in_ch_loop;
            out_ch_idx++;
            w_ptr++;
            biases++;
        } // for in_ch_idx
        in_ptr += filters;
    } // for ch_mult_idx
}

template <typename io_T, typename w_T, typename b_T, typename acc_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void depthwise_convolution2D_hwcn(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &w,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        s8asym_quant_specific_params quant_params,
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
    const int ch_mul = out.ch / in.ch;
    const int amount_rows = row_end - row_begin;
    const int amount_columns = clmn_end - clmn_begin;
    const int out_compensation_row_loop  = out.row_mem_stride * amount_rows;
    const int out_compensation_clmn_loop = out.col_mem_stride * amount_columns;
    const int out_increment_row_loop     = out.row_mem_stride  - out_compensation_clmn_loop;
    const int channel_per_loop = 1;
    const int channels_per_loop_v = 2;
    const int out_increment_in_ch_loop = channel_per_loop - out_compensation_row_loop;
    const int out_increment_in_ch_loop_v = channels_per_loop_v - out_compensation_row_loop;
    int out_ch_idx = 0;

    MLI_CONV_OUT_PTR(io_T) __restrict out_ptr = out.ptr
            + out.col_mem_stride * clmn_begin // move to columm
            + out.row_mem_stride * row_begin; // move to row

    s8asym_quant_specific_params v2quant_params[] = {quant_params, quant_params};
    for (int in_ch_idx = 0; in_ch_idx < in.ch - 1; in_ch_idx += 2) {
        for (int ch_mult_idx = 0; ch_mult_idx < ch_mul; ch_mult_idx++) {
            mli::krn::adjust_quant_params(&v2quant_params[0], out_ch_idx);
            mli::krn::adjust_quant_params(&v2quant_params[1], out_ch_idx + 1);

            acc_T bias_add_ch1 = bias_additive(biases++, 0x0, &v2quant_params[0]);
            acc_T bias_add_ch2 = bias_additive(biases++, 0x0, &v2quant_params[1]);
            __v2i32_t v2_bias_add = {bias_add_ch1, bias_add_ch2};

            for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    // comp - compensation values for valid area definition
                    const mli_compensations comp = mli_prv_valid_area_compensations(
                        H_idx, W_idx, in.height, in.width,
                        w.kernel_height, w.kernel_width,
                        stride_height, stride_width, padding_left, padding_top,
                        dilation_height, dilation_width);
                    
                    const int rows = w.kernel_height - comp.kernel_top - comp.kernel_bottom;
                    const int clmns = w.kernel_width - comp.kernel_right - comp.kernel_left;
                    const int h_idx_in = (H_idx * stride_height - padding_top + comp.in_top);
                    const int w_idx_in = (W_idx * stride_width - padding_left + comp.in_left);

                    const MLI_PTR(io_T) __restrict in_ptr = in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in
                        + in.ch_mem_stride * in_ch_idx;

                    const MLI_PTR(w_T) __restrict w_ptr = w.ptr
                        + w.row_mem_stride * comp.kernel_top
                        + w.col_mem_stride * comp.kernel_left
                        + w.in_ch_mem_stride * 0
                        + w.out_ch_mem_stride * out_ch_idx;

                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    __v2i32_t v2accu_dotprod = {0, 0};
                    dotprod2D_hwc_v(in_ptr, w_ptr, &v2accu_dotprod, clmns, rows,
                            in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                            w.col_mem_stride, w.row_mem_stride);

                    __v2i32_t v2acc_weights_add = {0, 0};
                    v2acc_weights_add = weights_additive_v(w_ptr, &v2acc_weights_add, &quant_params, clmns, rows,
                            w.col_mem_stride, w.row_mem_stride);

                    v2accu_dotprod += v2acc_weights_add;
                    v2accu_dotprod += v2_bias_add;

                    // Cast result to output type
                    result_cast_relu_store_v(out_ptr, &v2accu_dotprod, v2quant_params, val_min_limit, val_max_limit);

                    out_ptr += out.col_mem_stride;
                } // for W_idx
                out_ptr += out.row_mem_stride - out_compensation_clmn_loop;
            } // for H_idx
            out_ptr += out_increment_in_ch_loop_v;
            out_ch_idx += 2;
        } // for ch_mult_idx
    } // for in_ch_idx

    if (in.ch & 1) {
        for (int ch_mult_idx = 0; ch_mult_idx < ch_mul; ch_mult_idx++) {

            mli::krn::adjust_quant_params(&quant_params, out_ch_idx);
            int32_t bias_add = bias_additive(biases++, 0x0, &quant_params);

            for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    // comp - compensation values for valid area definition
                    const mli_compensations comp = mli_prv_valid_area_compensations(
                        H_idx, W_idx, in.height, in.width,
                        w.kernel_height, w.kernel_width,
                        stride_height, stride_width, padding_left, padding_top,
                        dilation_height, dilation_width);
                    
                    const int rows = w.kernel_height - comp.kernel_top - comp.kernel_bottom;
                    const int clmns = w.kernel_width - comp.kernel_right - comp.kernel_left;
                    const int h_idx_in = (H_idx * stride_height - padding_top + comp.in_top);
                    const int w_idx_in = (W_idx * stride_width - padding_left + comp.in_left);
                    const MLI_PTR(io_T) __restrict in_ptr = in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in
                        + in.ch_mem_stride * (in.ch - 1);

                    const MLI_PTR(w_T) __restrict w_ptr = w.ptr
                        + w.row_mem_stride * comp.kernel_top
                        + w.col_mem_stride * comp.kernel_left
                        + w.in_ch_mem_stride * 0
                        + w.out_ch_mem_stride * out_ch_idx;

                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
                    accu = dotprod2D(in_ptr, w_ptr, accu, clmns, rows,
                            in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                            w.col_mem_stride, w.row_mem_stride);

                    int32_t w_adds = weights_additive(w_ptr, 0x0, &quant_params, clmns, rows,
                            w.col_mem_stride, w.row_mem_stride);

                    accu += w_adds;
                    accu += bias_add;

                    // Cast result to output type
                    io_T out_val = mli::krn::result_cast<io_T, acc_T, s8asym_quant_specific_params>(accu, &quant_params);
                    out_val = MIN(out_val, val_max_limit);
                    out_val = MAX(out_val, val_min_limit);
                    *out_ptr = out_val;

                    out_ptr += out.col_mem_stride;
                } // for W_idx
                out_ptr += out_increment_row_loop;
            } // for H_idx
            out_ptr += out_increment_in_ch_loop;
            out_ch_idx++;
        } // for ch_mult_idx
    }
}

template <typename io_T, typename w_T, typename b_T, typename acc_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void depthwise_convolution2D_hwcn_nopad(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &w,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        fx_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {
    mli::krn::ref::depthwise_convolution2D<io_T, w_T, b_T, acc_T, fx_quant_specific_params, fix_kernel_width, fix_kernel_height>(
                in, w, biases, out, perception_area, quant_params,
                val_min_limit, val_max_limit,
                stride_height, stride_width,
                dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);
}

template <typename io_T, typename w_T, typename b_T, typename acc_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void depthwise_convolution2D_hwcn(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &w,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        fx_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {
    mli::krn::ref::depthwise_convolution2D<io_T, w_T, b_T, acc_T, fx_quant_specific_params, fix_kernel_width, fix_kernel_height>(
                in, w, biases, out, perception_area, quant_params,
                val_min_limit, val_max_limit,
                stride_height, stride_width,
                dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);

}

template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void depthwise_convolution2D(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &w,
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

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    rect_t perception_area_nopad;
    perception_area_nopad.row_beg = CEIL_DIV(padding_top, stride_height);
    perception_area_nopad.row_end = out.height - CEIL_DIV(padding_bot, stride_height);
    perception_area_nopad.clmn_beg = CEIL_DIV(padding_left, stride_width);
    perception_area_nopad.clmn_end = out.width - CEIL_DIV(padding_right, stride_width);
    
    if ((perception_area_nopad.row_end > perception_area_nopad.row_beg)
        && (perception_area_nopad.clmn_end > perception_area_nopad.clmn_beg)){
    depthwise_convolution2D_hwcn_nopad<io_T, w_T, b_T, acc_T, fix_kernel_width, fix_kernel_height>(
                in, w, biases, out, perception_area_nopad, quant_params,
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
            depthwise_convolution2D_hwcn<io_T, w_T, b_T, acc_T, fix_kernel_width, fix_kernel_height>(
                    in, w, biases, out, perc_areas[i], quant_params,
                    val_min_limit, val_max_limit,
                    stride_height, stride_width,
                    dilation_height, dilation_width,
                    padding_top, padding_left,
                    padding_bot, padding_right);
        }
    }
}

#pragma MLI_CODE_SECTION_END()
} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_CONVOLUTION_DSP_H_
