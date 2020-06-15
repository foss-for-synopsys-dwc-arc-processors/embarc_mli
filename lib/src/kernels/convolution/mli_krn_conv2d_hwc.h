/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_CONV2D_HWC_H_
#define _MLI_KRN_CONV2D_HWC_H_

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "mli_api.h"
#include "mli_krn_dotprod.h"
#include "mli_krn_reduce_sum2d.h"
#include "mli_math.h"
#include "mli_private_types.h"
#include "mli_prv_aux_calc.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"
#include "mli_types.h"

//========================================================
// Depthwise convolution 2D template
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T>
static __attribute__ ((always_inline)) void depthwise_convolution2D_hwcn_nopad(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &w,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t * const perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {
    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;
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
            adjust_quant_params(&v2quant_params[0], out_ch_idx++);
            adjust_quant_params(&v2quant_params[1], out_ch_idx++);

            acc_T bias_add_ch1 = bias_additive(*biases++, 0x0, &v2quant_params[0]);
            acc_T bias_add_ch2 = bias_additive(*biases++, 0x0, &v2quant_params[1]);

            __v2i32_t v2acc_weights_add = {bias_add_ch1, bias_add_ch2};
            v2acc_weights_add = weights_additive_v(w_ptr, &v2acc_weights_add, &quant_params, w.kernel_width, w.kernel_height,
                    w.col_mem_stride, w.row_mem_stride);
            __builtin_assume(amount_rows > 0);
            for (int H_idx = 0; H_idx < amount_rows; H_idx++) {
                __builtin_assume(amount_columns > 0);

                __v2i32_t v2accu_dotprod = v2acc_weights_add;
                for (int W_idx = 0; W_idx < amount_columns; W_idx++) {
                    dotprod2D_hwc_v(&in_ptr, &w_ptr, &v2accu_dotprod, w.kernel_width, w.kernel_height,
                                    in.col_mem_stride, in.row_mem_stride, w.col_mem_stride, w.row_mem_stride);
                    //compensite increment of input tensor pointer from dotprod2D_hwc_v function
                    in_ptr += in_increment_clmn_loop - w.kernel_height * in.row_mem_stride;
                    //compensite increment of weights pointer from dotprod2D_hwc_v function
                    w_ptr -= w.kernel_height * w.row_mem_stride;

                    // Cast result to output type
                    mli_prv_clip_relu_store_output_v(out_ptr, &v2accu_dotprod, v2quant_params, val_min_limit, val_max_limit);
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
            adjust_quant_params(&quant_params, out_ch_idx);

            acc_T global_other_additives = weights_additive(w_ptr, 0x0, &quant_params, w.kernel_width, w.kernel_height,
                    w.col_mem_stride, w.row_mem_stride);
            global_other_additives += bias_additive(*biases, 0x0, &quant_params);
            __v2i32_t v2global_other_additives = {global_other_additives, global_other_additives};

            for (int H_idx = 0; H_idx < amount_rows; H_idx++) {
                for (int W_idx = 0; W_idx < amount_columns - 1; W_idx += 2) {
                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    __v2i32_t accu = v2global_other_additives;
                    accu = dotprod2D_inp_width_v(&in_ptr, &w_ptr, &accu, w.kernel_width, w.kernel_height,
                            in.col_mem_stride, in.row_mem_stride, w.col_mem_stride, w.row_mem_stride, in_increment_clmn_loop);
                    //compensite increment of input tensor pointer from dotprod2D_hwc_v function
                    in_ptr += 2 * in_increment_clmn_loop - w.kernel_height * in.row_mem_stride;
                    //compensite increment of weights pointer from dotprod2D_hwc_v function
                    w_ptr -= w.kernel_height * w.row_mem_stride;
                    // Cast result to output type
                    mli_prv_clip_relu_store_output_inp_width_v(out_ptr, &accu, &quant_params, val_min_limit, val_max_limit, out.col_mem_stride);
                    out_ptr += 2 * out.col_mem_stride;
                } // for W_idx
                if (amount_columns & 0x1) {
                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    acc_T accu = 0;
                    accu = dotprod2D(&in_ptr, &w_ptr, accu, w.kernel_width, w.kernel_height,
                            in.col_mem_stride, in.row_mem_stride, w.col_mem_stride, w.row_mem_stride);
                    //compensite increment of input tensor pointer from dotprod2D_hwc_v function
                    in_ptr += in_increment_clmn_loop - w.kernel_height * in.row_mem_stride;
                    //compensite increment of weights pointer from dotprod2D_hwc_v function
                    w_ptr -= w.kernel_height * w.row_mem_stride;
                    accu += global_other_additives;

                    // Cast result to output type
                    mli_prv_clip_relu_store_output(out_ptr, accu, &quant_params, val_min_limit, val_max_limit);
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

template <typename io_T, typename w_T, typename b_T, typename acc_T>
static __attribute__ ((always_inline)) void depthwise_convolution2D_hwcn(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &w,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t * const perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;
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
            adjust_quant_params(&v2quant_params[0], out_ch_idx);
            adjust_quant_params(&v2quant_params[1], out_ch_idx + 1);

            acc_T bias_add_ch1 = bias_additive(*biases++, 0x0, &v2quant_params[0]);
            acc_T bias_add_ch2 = bias_additive(*biases++, 0x0, &v2quant_params[1]);
            __v2i32_t v2_bias_add = {bias_add_ch1, bias_add_ch2};

            for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
                // Define area of input and filter for convolution
                // comp - compensation values for valid area definition
                mli_compensations comp;
                comp.top    = -MIN((H_idx * stride_height)- padding_top, 0);
                comp.bottom = -MIN(in.height - ((H_idx * stride_height)- padding_top + w.kernel_height), 0);
                const int rows = w.kernel_height - comp.top - comp.bottom;
                const int h_idx_in = (H_idx * stride_height - padding_top + comp.top);
                MLI_PTR(io_T) __restrict in_ptr = in.ptr
                        + in.row_mem_stride * h_idx_in // move to row
                        + in_ch_idx;                   // move to channel
                MLI_PTR(w_T) __restrict w_ptr = w.ptr
                        + w.row_mem_stride * comp.top // move to row
                        + out_ch_idx;                 // move to filter

                for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    // comp - compensation values for valid area definition
                    comp.left   = -MIN((W_idx * stride_width)- padding_left, 0);
                    comp.right  = -MIN(in.width - ((W_idx * stride_width)- padding_left + w.kernel_width), 0);
                    const int clmns = w.kernel_width - comp.right - comp.left;
                    const int w_idx_in = (W_idx * stride_width - padding_left + comp.left);

                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    __v2i32_t v2accu_dotprod = {0, 0};
                    dotprod2D_hwc_v(
                            &in_ptr[in.col_mem_stride * w_idx_in],
                            &w_ptr[w.col_mem_stride * comp.left], &v2accu_dotprod, clmns, rows,
                            in.col_mem_stride, in.row_mem_stride, w.col_mem_stride, w.row_mem_stride);

                    __v2i32_t v2acc_weights_add = {0, 0};
                    v2acc_weights_add = weights_additive_v(
                            &w_ptr[w.col_mem_stride * comp.left], &v2acc_weights_add, &quant_params, clmns, rows,
                            w.col_mem_stride, w.row_mem_stride);

                    v2accu_dotprod += v2acc_weights_add;
                    v2accu_dotprod += v2_bias_add;

                    // Cast result to output type
                    mli_prv_clip_relu_store_output_v(out_ptr, &v2accu_dotprod, v2quant_params, val_min_limit, val_max_limit);

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

            adjust_quant_params(&quant_params, out_ch_idx);
            int32_t bias_add = bias_additive(*biases++, 0x0, &quant_params);

            for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
                // Define area of input and filter for convolution
                // comp - compensation values for valid area definition
                mli_compensations comp;
                comp.top    = -MIN((H_idx * stride_height)- padding_top, 0);
                comp.bottom = -MIN(in.height - ((H_idx * stride_height)- padding_top + w.kernel_height), 0);
                const int rows = w.kernel_height - comp.top - comp.bottom;
                const int h_idx_in = (H_idx * stride_height - padding_top + comp.top);
                MLI_PTR(io_T) __restrict in_ptr = in.ptr
                        + in.row_mem_stride * h_idx_in // move to row
                        + in.ch - 1;                   // move to channel
                MLI_PTR(w_T) __restrict w_ptr = w.ptr
                        + w.row_mem_stride * comp.top // move to row
                        + out_ch_idx;                 // move to filter

                for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    // comp - compensation values for valid area definition
                    comp.left  = -MIN((W_idx * stride_width) - padding_left, 0);
                    comp.right = -MIN(in.width - ((W_idx * stride_width) - padding_left + w.kernel_width), 0);
                    const int clmns = w.kernel_width - comp.right - comp.left;
                    const int w_idx_in = (W_idx * stride_width - padding_left + comp.left);

                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
                    accu = dotprod2D(
                            &in_ptr[in.col_mem_stride * w_idx_in],
                            &w_ptr[w.col_mem_stride * comp.left], accu, clmns, rows,
                            in.col_mem_stride, in.row_mem_stride, w.col_mem_stride, w.row_mem_stride);

                    int32_t w_adds = weights_additive(
                            &w_ptr[w.col_mem_stride * comp.left], 0x0, &quant_params, clmns, rows,
                            w.col_mem_stride, w.row_mem_stride);

                    accu += w_adds;
                    accu += bias_add;

                    // Cast result to output type
                    mli_prv_clip_relu_store_output(out_ptr, accu, &quant_params, val_min_limit, val_max_limit);

                    out_ptr += out.col_mem_stride;
                } // for W_idx
                out_ptr += out_increment_row_loop;
            } // for H_idx
            out_ptr += out_increment_in_ch_loop;
            out_ch_idx++;
        } // for ch_mult_idx
    }
}

template <typename io_T, typename w_T, typename b_T, typename acc_T>
static __attribute__ ((always_inline)) void depthwise_convolution2D_hwcn_krnpad(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &w,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t * const perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    rect_t perception_area_nopad;
    perception_area_nopad.row_beg = CEIL_DIV(padding_top, stride_height);
    perception_area_nopad.row_end = out.height - CEIL_DIV(padding_bot, stride_height);
    perception_area_nopad.clmn_beg = CEIL_DIV(padding_left, stride_width);
    perception_area_nopad.clmn_end = out.width - CEIL_DIV(padding_right, stride_width);
    
    if ((perception_area_nopad.row_end - perception_area_nopad.row_beg > 0)
        && (perception_area_nopad.clmn_end - perception_area_nopad.clmn_beg > 0)){
    depthwise_convolution2D_hwcn_nopad<int8_t, int8_t, int32_t, mli_acc32_t>(
                in, w, biases, out, &perception_area_nopad, quant_params,
                val_min_limit, val_max_limit,
                stride_height, stride_width,
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
            depthwise_convolution2D_hwcn<int8_t, int8_t, int32_t, mli_acc32_t>(
                    in, w, biases, out, &perc_areas[i], quant_params,
                    val_min_limit, val_max_limit,
                    stride_height, stride_width,
                    padding_top, padding_left,
                    padding_bot, padding_right);
        }
    }
}

//========================================================
// Convolution 2D template
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T>
static __attribute__ ((always_inline)) void convolution2D_nhwc(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &w,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t * const perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;
    const int amount_rows = row_end - row_begin;
    const int amount_columns = clmn_end - clmn_begin;
    const int out_compensation_row_loop  = out.row_mem_stride * amount_rows;
    const int out_compensation_clmn_loop = out.col_mem_stride * amount_columns;

    MLI_CONV_OUT_PTR(io_T) __restrict out_ptr = out.ptr
            + out.row_mem_stride * row_begin   // setup init coef for moving to row
            + out.col_mem_stride * clmn_begin; // setup init coef for moving to colum

    for (int out_ch_idx = 0; out_ch_idx < out.ch; out_ch_idx++) {
        adjust_quant_params(&quant_params, out_ch_idx);
        const int bias_add = bias_additive(biases[out_ch_idx], 0x0, &quant_params);

        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            mli_compensations comp;
            comp.top    = -MIN((H_idx * stride_height) - padding_top, 0);
            comp.bottom = -MIN(in.height - ((H_idx * stride_height) - padding_top + w.kernel_height), 0);
            const int rows = w.kernel_height - comp.top - comp.bottom;

            int32_t w_adds = 0;

            for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                // Define area of input and filter for convolution
                // comp - compensation values for valid area definition
                comp.left   = -MIN((W_idx * stride_width)- padding_left, 0);
                comp.right  = -MIN(in.width - ((W_idx * stride_width)- padding_left + w.kernel_width), 0);
                const int clmns = w.kernel_width - comp.right - comp.left;

                const MLI_PTR (int8_t) in_ptr = in.ptr                                           // starting point
                        + in.row_mem_stride * (H_idx * stride_height - padding_top + comp.top)   // move to row
                        + in.col_mem_stride * (W_idx * stride_width - padding_left + comp.left); // move to column

                const MLI_PTR (int8_t) w_ptr = w.ptr        // Start point
                        + w.out_ch_mem_stride * out_ch_idx  // move to filter
                        + w.row_mem_stride * comp.top       // move to row
                        + w.col_mem_stride * comp.left;     // move to column

                int8_t init_accum_weights_add_val = 0;
                w_adds = mli_prv_init_accu(init_accum_weights_add_val);
LOOP_PIPELINE_ENABLE
LOOP_PIPELINE_ENABLE_BACKTRACKING
                for (int in_ch_idx = 0; in_ch_idx < in.ch - 1; in_ch_idx += 2) {
                    w_adds = weights_additive_d(
                            &w_ptr[w.col_mem_stride * comp.left + in_ch_idx], &w_adds, &quant_params,
                            clmns, rows, w.col_mem_stride, w.row_mem_stride);
                }
                if (in.ch & 1) {
                    w_adds = weights_additive(
                            &w_ptr[w.col_mem_stride * comp.left + in.ch - 1], w_adds, &quant_params,
                            clmns, rows, w.col_mem_stride, w.row_mem_stride);
                }

                int32_t init_accum_val = w_adds;
                acc_T accu = mli_prv_init_accu(init_accum_val);

                // Convolution core
LOOP_PIPELINE_ENABLE
LOOP_PIPELINE_ENABLE_BACKTRACKING
                for (int in_ch_idx = 0; in_ch_idx < in.ch - 1; in_ch_idx += 2) {
                    dotprod2D_hwc_d(in_ptr, w_ptr, &accu, clmns, rows,
                            in.col_mem_stride, in.row_mem_stride,
                            w.col_mem_stride, w.row_mem_stride);
                    in_ptr += 2;
                    w_ptr += 2;
                }

                if (in.ch & 1) {
                    accu = dotprod2D(in_ptr, w_ptr, accu, clmns, rows,
                            in.col_mem_stride, in.row_mem_stride,
                            w.col_mem_stride, w.row_mem_stride);
                    in_ptr += 1;
                    w_ptr += 1;
                }

                in_ptr -= in.ch;
                w_ptr -= in.ch;

                accu += bias_add;

                // Cast result to output type, apply built-in ReLU Applying and write result
                mli_prv_clip_relu_store_output(out_ptr, accu, &quant_params, val_min_limit, val_max_limit);
                out_ptr += out.col_mem_stride;
            } // for W_idx
            out_ptr += out.row_mem_stride - out_compensation_clmn_loop;
        } // for H_idx
        out_ptr += 1 - out_compensation_row_loop;
    } // for out_ch_idx
}

template <typename io_T, typename w_T, typename b_T, typename acc_T>
static __attribute__ ((always_inline)) void convolution2D_nhwc_nopad(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &w,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t * const perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {
    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;
    const int amount_rows = row_end - row_begin;
    const int amount_columns = clmn_end - clmn_begin;
    const int in_compensation_row_loop = in.row_mem_stride * stride_height * amount_rows;
    const int in_compensation_clmn_loop = in.col_mem_stride * stride_width * amount_columns;
    const int out_compensation_row_loop  = out.row_mem_stride * amount_rows;
    const int out_compensation_clmn_loop = out.col_mem_stride * amount_columns;

    MLI_CONV_OUT_PTR(io_T) __restrict out_ptr = out.ptr
            + out.row_mem_stride * row_begin   // move to row
            + out.col_mem_stride * clmn_begin; // move column

    MLI_PTR(io_T) __restrict in_ptr = in.ptr
            + in.row_mem_stride * (row_begin * stride_height - padding_top)   // move to row
            + in.col_mem_stride * (clmn_begin * stride_width - padding_left); // move to column

    for (int out_ch_idx = 0; out_ch_idx < out.ch; out_ch_idx++) {
        adjust_quant_params(&quant_params, out_ch_idx);
        const int bias_add = bias_additive(biases[out_ch_idx], 0x0, &quant_params);
        int8_t init_accum_val = 0;
        int weights_add = mli_prv_init_accu(init_accum_val);

        MLI_PTR(w_T) __restrict w_ptr = w.ptr +
                w.out_ch_mem_stride * out_ch_idx; // move to filter

        for (int in_ch_idx = 0; in_ch_idx < (in.ch - 1); in_ch_idx += 2) {
            weights_add = weights_additive_d(w_ptr, &weights_add, &quant_params, 
                    w.kernel_width, w.kernel_height, w.col_mem_stride, w.row_mem_stride);
            w_ptr += 2;
        }

        if (in.ch & 1) {
            weights_add = weights_additive(w_ptr, weights_add, &quant_params,
                    w.kernel_width, w.kernel_height, w.col_mem_stride, w.row_mem_stride);
            w_ptr += 1;
        }
        w_ptr -= in.ch;

        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {

            int32_t init_accum_val = weights_add;
            acc_T accu = mli_prv_init_accu(init_accum_val);
            for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
LOOP_PIPELINE_ENABLE
LOOP_PIPELINE_ENABLE_BACKTRACKING
                for (int in_ch_idx = 0; in_ch_idx < (in.ch - 1); in_ch_idx += 2) {
                    dotprod2D_hwc_d(in_ptr, w_ptr, &accu, w.kernel_width, w.kernel_height,
                            in.col_mem_stride, in.row_mem_stride, w.col_mem_stride, w.row_mem_stride);
                    in_ptr += 2;
                    w_ptr += 2;
                }

                if (in.ch & 1) {
                    accu = dotprod2D(in_ptr, w_ptr, accu, w.kernel_width, w.kernel_height,
                            in.col_mem_stride, in.row_mem_stride, w.col_mem_stride, w.row_mem_stride);
                    in_ptr += 1;
                    w_ptr += 1;
                }
                accu += bias_add;
                
                // Cast result to output type, apply built-in ReLU Applying and write result
                mli_prv_clip_relu_store_output(out_ptr, accu, &quant_params, val_min_limit, val_max_limit);

                out_ptr += out.col_mem_stride;
                in_ptr += in.col_mem_stride * stride_width - in.ch;
                w_ptr -= in.ch;

                init_accum_val = weights_add;
                accu = mli_prv_init_accu(init_accum_val);
            } // for W_idx
            out_ptr += out.row_mem_stride - out_compensation_clmn_loop;
            in_ptr += in.row_mem_stride * stride_height - in_compensation_clmn_loop;
        } // for H_idx
        in_ptr -=  in_compensation_row_loop;
        out_ptr += 1 - out_compensation_row_loop;
    } // for out_ch_idx
}

template <typename io_T, typename w_T, typename b_T, typename acc_T>
static __attribute__ ((always_inline)) void convolution2D_nhwc_krnpad(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &w,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t * const perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    rect_t perception_area_nopad;
    perception_area_nopad.row_beg = CEIL_DIV(padding_top, stride_height);
    perception_area_nopad.row_end = out.height - CEIL_DIV(padding_bot, stride_height);
    perception_area_nopad.clmn_beg = CEIL_DIV(padding_left, stride_width);
    perception_area_nopad.clmn_end = out.width - CEIL_DIV(padding_right, stride_width);

    if ((perception_area_nopad.row_end - perception_area_nopad.row_beg > 0)
        && (perception_area_nopad.clmn_end - perception_area_nopad.clmn_beg > 0)){
        convolution2D_nhwc_nopad<int8_t, int8_t, int32_t, mli_acc32_t>(
                in, w, biases, out, &perception_area_nopad, quant_params,
                val_min_limit, val_max_limit,
                stride_height, stride_width,
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
            convolution2D_nhwc<int8_t, int8_t, int32_t, mli_acc32_t>(
                    in, w, biases, out, &perc_areas[i], quant_params,
                    val_min_limit, val_max_limit,
                    stride_height, stride_width,
                    padding_top, padding_left,
                    padding_bot, padding_right);
        }
    }
}

template <typename io_T, typename w_T, typename b_T, typename acc_T>
static __attribute__ ((always_inline)) void pointwise_convolution2D_nhwc_nopad(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &w,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t * const perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;
    const int amount_rows = row_end - row_begin;
    const int amount_columns = clmn_end - clmn_begin;
    const int in_compensation_row_loop   = in.row_mem_stride  * stride_height * amount_rows;
    const int in_compensation_clmn_loop  = in.col_mem_stride  * amount_columns;
    const int out_compensation_row_loop  = out.row_mem_stride * amount_rows;
    const int out_compensation_clmn_loop = out.col_mem_stride * amount_columns;

    MLI_CONV_OUT_PTR(io_T) __restrict out_ptr = out.ptr
            + out.row_mem_stride * row_begin   // move to row
            + out.col_mem_stride * clmn_begin; // move to column
    const MLI_PTR(io_T) __restrict in_ptr = in.ptr
            + in.row_mem_stride * (row_begin * stride_height)  // move to row
            + in.col_mem_stride * (clmn_begin * stride_width); // move to column

    for (int out_ch_idx = 0; out_ch_idx < out.ch; out_ch_idx++) {
        adjust_quant_params(&quant_params, out_ch_idx);
        const int bias_add = bias_additive(biases[out_ch_idx], 0x0, &quant_params);
        const MLI_PTR(w_T) __restrict w_ptr = w.ptr + w.out_ch_mem_stride * out_ch_idx;
        int8_t init_accum_val = 0;
        int weights_add = mli_prv_init_accu(init_accum_val);

        for (int in_ch_idx = 0; in_ch_idx < in.ch-1; in_ch_idx+=2) {
            weights_add = weights_additive_d(w_ptr, &weights_add, &quant_params, 
                    w.kernel_width, w.kernel_height, w.col_mem_stride, w.row_mem_stride);
            w_ptr += 2;
        }

        if (in.ch & 1) {
            weights_add = weights_additive(w_ptr++, weights_add, &quant_params, 
                    w.kernel_width, w.kernel_height, w.col_mem_stride, w.row_mem_stride);
        }
        w_ptr -= in.ch; //=w_ptr.in_ch

        int odd_rest_of_in_ch = (in.ch & 0x3);
        int even_in_ch = in.ch & (~0x3);

        if ((in.ch & 0x3) == 0) {
            for (int H_idx = 0; H_idx < amount_rows; H_idx++) {
#if !defined(_ARCVER_ARCv2HS)
                int32_t init_accum_val = weights_add;
                acc_T accu = mli_prv_init_accu(init_accum_val);
                for (int j = 0; j < (in.ch / 4); j++) {
                    mli_prv_load_mac_vec4(&accu, in_ptr, w_ptr);
                    in_ptr += 4; // chan_mem_stride
                    w_ptr += 4; // chan_mem_stride
                }
                accu += bias_add;

                // Cast result to output type, apply built-in ReLU Applying and write result
                mli_prv_clip_relu_store_output(out_ptr, accu, &quant_params, val_min_limit, val_max_limit);
                out_ptr += out.col_mem_stride;
                in_ptr += in.col_mem_stride * stride_width - in.ch;
                w_ptr -= in.ch; //=w_ptr.in_ch

                for (int W_idx = 1; W_idx < amount_columns; W_idx++) {
                    init_accum_val = weights_add;
                    accu = mli_prv_init_accu(init_accum_val);
#else
                for (int W_idx = 0; W_idx < amount_columns; W_idx++) {
                    int32_t init_accum_val = weights_add;
                    acc_T accu = mli_prv_init_accu(init_accum_val);
#endif

LOOP_PIPELINE_ENABLE
LOOP_PIPELINE_ENABLE_BACKTRACKING
                    for (int j = 0; j < (in.ch / 4); j++) {
                        mli_prv_load_mac_vec4(&accu, in_ptr, w_ptr);
                        in_ptr += 4; // chan_mem_stride
                        w_ptr += 4;  // chan_mem_stride
                    }
                    accu += bias_add;

                    // Cast result to output type, apply built-in ReLU Applying and write result
                    mli_prv_clip_relu_store_output(out_ptr, accu, &quant_params, val_min_limit, val_max_limit);
                    out_ptr += out.col_mem_stride;
                    in_ptr += in.col_mem_stride * stride_width - in.ch;
                    w_ptr -= in.ch; //=w_ptr.in_ch
              } // for W_idx
                out_ptr += out.row_mem_stride - out_compensation_clmn_loop;
                in_ptr += in.row_mem_stride  * stride_height - in_compensation_clmn_loop;
            } // for H_idx
        } else {
            for (int H_idx = 0; H_idx < amount_rows; H_idx++) {
                int32_t init_accum_val = weights_add;
                acc_T accu = mli_prv_init_accu(init_accum_val);
                for (int W_idx = 0; W_idx < amount_columns; W_idx++) {

                    for (int k = 0; k < odd_rest_of_in_ch; k++) {
                        mli_prv_load_mac(&accu, in_ptr++, w_ptr++);
                    }

LOOP_PIPELINE_ENABLE
LOOP_PIPELINE_ENABLE_BACKTRACKING
                    for (int j = 0; j < (even_in_ch / 4); j++) {
                        mli_prv_load_mac_vec4(&accu, in_ptr, w_ptr);
                        in_ptr += 4;
                        w_ptr += 4;
                    }
                    accu += bias_add;

                    // Cast result to output type, apply built-in ReLU Applying and write result
                    mli_prv_clip_relu_store_output(out_ptr, accu, &quant_params, val_min_limit, val_max_limit);
                    out_ptr += out.col_mem_stride;
                    in_ptr += in.col_mem_stride * stride_width - in.ch;
                    w_ptr -= in.ch; //=w_ptr.in_ch

                    init_accum_val = weights_add;
                    accu = mli_prv_init_accu(init_accum_val);
                } // for W_idx
                out_ptr += out.row_mem_stride - out_compensation_clmn_loop;
                in_ptr += in.row_mem_stride  * stride_height - in_compensation_clmn_loop;
            } // for H_idx
        }
        in_ptr -=  in_compensation_row_loop;
        out_ptr += 1 - out_compensation_row_loop;
    } // for out_ch_idx
}

#endif // _MLI_KRN_CONV2D_HWC_H_


