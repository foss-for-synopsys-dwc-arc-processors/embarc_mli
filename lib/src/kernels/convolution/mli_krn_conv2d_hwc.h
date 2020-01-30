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
#include "mli_krn_dotprod.h"
#include "mli_krn_reduce_sum2d.h"
#include "mli_math.h"
#include "mli_prv_aux_calc.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"
#include "mli_types.h"



//========================================================
// Depthwise convolution 2D template
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T>
static __attribute__ ((always_inline)) void depthwise_convolution2D_hwc_nopad(
        const MLI_PTR(io_T) __restrict in_ftrs,
        const MLI_PTR(w_T)  __restrict weights,
        const MLI_PTR(b_T)  __restrict biases,
              MLI_CONV_OUT_PTR(io_T) __restrict out_ftrs,

        const rect_t * const perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,

        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {
    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;
    const int filters = 1;
    const int in_col_step = in_ch * filters;
    const int in_row_step = in_width * in_ch * filters;
    const int krn_col_step = out_ch; /*channels == 1 */
    const int krn_row_step = kernel_width * out_ch; /*channels == 1 */
    const int ch_mul = out_ch / in_ch;

    const int amount_rows = row_end - row_begin;
    const int amount_columns = clmn_end - clmn_begin;
    const int in_compensation_row_loop = in_ch * stride_height * in_width * filters * amount_rows;
    const int out_compensation_row_loop = out_ch * out_width * filters * amount_rows;
    const int in_compensation_clmn_loop = stride_width * filters * in_ch * amount_columns;
    const int out_compensation_clmn_loop = filters * out_ch * amount_columns;
    const int in_increment_clmn_loop = stride_width * filters * in_ch;
    const int out_increment_clmn_loop = filters * out_ch;
    const int in_increment_row_loop = in_ch * stride_height * in_width * filters - in_compensation_clmn_loop;
    const int out_increment_row_loop = out_ch * out_width * filters  - out_compensation_clmn_loop;
    const int channel_per_loop = 1;
    const int channels_per_loop_v = 2;
    const int out_increment_in_ch_loop = channel_per_loop - out_compensation_row_loop;
    const int out_increment_in_ch_loop_v = channels_per_loop_v - out_compensation_row_loop;


    MLI_PTR(io_T) __restrict in_ptr = (MLI_PTR(io_T) __restrict)in_ftrs;
    MLI_CONV_OUT_PTR(io_T) __restrict out_ptr = (MLI_CONV_OUT_PTR(io_T) __restrict)out_ftrs;
    MLI_PTR(w_T) __restrict w_ptr = (MLI_PTR(w_T) __restrict)weights;
    // MLI_PTR(w_T) __restrict w_ptr_local = (MLI_PTR(w_T) __restrict)weights;
    int out_ch_idx = 0;

    in_ptr += in_ch * filters *                     // common coefs
        ((row_begin * stride_height - padding_top) * in_width  +    // setup init coef for moving to row
        (clmn_begin * stride_width - padding_left)) ;                // setup init coef for moving to colum;
    
    out_ptr += out_ch * filters *       // common coefs
            (row_begin * out_width  +   // setup init coef for moving to row
            clmn_begin) ;               // setup init coef for moving to colum;

    s8asym_quant_specific_params v2quant_params[] = {quant_params, quant_params};
    for (int in_ch_idx = 0; in_ch_idx < in_ch-1; in_ch_idx += 2) {
        for (int ch_mult_idx = 0; ch_mult_idx < ch_mul; ch_mult_idx++) {
            adjust_quant_params(&v2quant_params[0], out_ch_idx++);
            adjust_quant_params(&v2quant_params[1], out_ch_idx++);

            acc_T bias_add_ch1 = bias_additive(*biases++, 0x0, &v2quant_params[0]);
            acc_T bias_add_ch2 = bias_additive(*biases++, 0x0, &v2quant_params[1]);

            v2accum40_t v2acc_weights_add = {bias_add_ch1, bias_add_ch2};
            v2acc_weights_add = weights_additive_v(w_ptr, &v2acc_weights_add, &quant_params, kernel_width, kernel_height, 
                    krn_col_step, krn_row_step);
            __v2i32_t v2acc_weights_add_int = {fx_q31_cast_nf_a40(fx_get_v2a40(v2acc_weights_add, 0)),
                                               fx_q31_cast_nf_a40(fx_get_v2a40(v2acc_weights_add, 1))};

            for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                    __v2i32_t v2accu_dotprod = {0, 0};
                    dotprod2D_hwc_v(in_ptr, w_ptr, &v2accu_dotprod, kernel_width, kernel_height,
                                        in_col_step, in_row_step, krn_col_step, krn_row_step);

                    v2accu_dotprod += v2acc_weights_add_int;

                    // Cast result to output type
                    mli_prv_clip_relu_store_output_v(out_ptr, &v2accu_dotprod, v2quant_params, val_min_limit, val_max_limit);

                    in_ptr += in_increment_clmn_loop;
                    out_ptr += out_increment_clmn_loop;
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

    if(in_ch & 1){
        for (int ch_mult_idx = 0; ch_mult_idx < ch_mul; ch_mult_idx++) {
            adjust_quant_params(&quant_params, out_ch_idx);

            acc_T global_other_additives = weights_additive(w_ptr, 0x0, &quant_params, kernel_width, kernel_height, 
                    krn_col_step, krn_row_step);
            global_other_additives += bias_additive(*biases, 0x0, &quant_params);

            for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    acc_T accu = 0;
                    accu = dotprod2D(in_ptr, w_ptr, accu, kernel_width, kernel_height,
                                        in_col_step, in_row_step, krn_col_step, krn_row_step);
                    accu = mli_math_add_fx(accu, global_other_additives);

                    // Cast result to output type
                    mli_prv_clip_relu_store_output(out_ptr, accu, &quant_params, val_min_limit, val_max_limit);

                    in_ptr += in_increment_clmn_loop;
                    out_ptr += out_increment_clmn_loop;
                } // for W_idx
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
static __attribute__ ((always_inline)) void depthwise_convolution2D_hwc(
        const MLI_PTR(io_T) __restrict in_ftrs,
        const MLI_PTR(w_T)  __restrict weights,
        const MLI_PTR(b_T)  __restrict biases,
              MLI_CONV_OUT_PTR(io_T) __restrict out_ftrs,

        const rect_t * const perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,

        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;
    const int in_col_step = mli_prv_column_step<LAYOUT_HWCN>(in_height, in_width, in_ch, /*filters =*/ 1);
    const int in_row_step = mli_prv_row_step<LAYOUT_HWCN>(in_height, in_width, in_ch, /*filters =*/ 1);
    const int krn_col_step = mli_prv_column_step<LAYOUT_HWCN>(kernel_height, kernel_width, /*channels =*/ 1, out_ch);
    const int krn_row_step = mli_prv_row_step<LAYOUT_HWCN>(kernel_height, kernel_width, /*channels =*/ 1, out_ch);
    const int ch_mul = out_ch / in_ch;
    const int filters = 1;

    const int amount_rows = row_end - row_begin;
    const int amount_columns = clmn_end - clmn_begin;
    const int out_compensation_row_loop = out_ch * out_width * filters * amount_rows;
    const int out_compensation_clmn_loop = filters * out_ch * amount_columns;
    const int out_increment_clmn_loop = filters * out_ch;
    const int out_increment_row_loop = out_ch * out_width * filters  - out_compensation_clmn_loop;
    const int channel_per_loop = 1;
    const int channels_per_loop_v = 2;
    const int out_increment_in_ch_loop = channel_per_loop - out_compensation_row_loop;
    const int out_increment_in_ch_loop_v = channels_per_loop_v - out_compensation_row_loop;
    int out_ch_idx = 0;

    MLI_CONV_OUT_PTR(io_T) __restrict out_ptr = (MLI_CONV_OUT_PTR(io_T) __restrict)out_ftrs;
    out_ptr += out_ch * filters *       // common coefs
            (row_begin * out_width  +   // setup init coef for moving to row
            clmn_begin) ;               // setup init coef for moving to colum;

    s8asym_quant_specific_params v2quant_params[] = {quant_params, quant_params};
    for (int in_ch_idx = 0; in_ch_idx < in_ch - 1; in_ch_idx += 2) {
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
                comp.bottom = -MIN(in_height - ((H_idx * stride_height)- padding_top + kernel_height), 0);
                const int rows = kernel_height - comp.top - comp.bottom;
                const int h_idx_in = (H_idx * stride_height - padding_top + comp.top);
                MLI_PTR(io_T) __restrict in_ptr = (MLI_PTR(io_T) __restrict)in_ftrs 
                        + h_idx_in * in_width * in_ch * filters                 // move to row
                        // + w_idx * in_ch * filters                            // move to column
                        + in_ch_idx * filters;                                  // move to channel
                MLI_PTR(w_T) __restrict w_ptr = (MLI_PTR(w_T) __restrict)weights
                        + comp.top * kernel_width * filters * out_ch            // move to row
                        //+ comp.left * filters * out_ch                        // move to column
                        + out_ch_idx;                                           // move to filter

                for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    // comp - compensation values for valid area definition
                    comp.left   = -MIN((W_idx * stride_width)- padding_left, 0);
                    comp.right  = -MIN(in_width - ((W_idx * stride_width)- padding_left + kernel_width), 0);
                    const int clmns = kernel_width - comp.right - comp.left;
                    const int w_idx_in = (W_idx * stride_width - padding_left + comp.left);

                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    __v2i32_t v2accu_dotprod = {0, 0};
                    dotprod2D_hwc_v(&in_ptr[w_idx_in * in_ch * filters], &w_ptr[comp.left * filters * out_ch], &v2accu_dotprod, clmns, rows,
                                        in_col_step, in_row_step, krn_col_step, krn_row_step);

                        __v2i32_t v2acc_weights_add = {0, 0};
                        v2acc_weights_add = weights_additive_v(&w_ptr[comp.left * filters * out_ch], &v2acc_weights_add, &quant_params, clmns, rows, krn_col_step, krn_row_step);

                    v2accu_dotprod += v2acc_weights_add;
                    v2accu_dotprod += v2_bias_add;

                    // Cast result to output type

                    mli_prv_clip_relu_store_output_v(out_ptr, &v2accu_dotprod, v2quant_params, val_min_limit, val_max_limit);

                    out_ptr += filters * out_ch;
                } // for W_idx
                out_ptr += out_ch * out_width * filters - out_compensation_clmn_loop;
            } // for H_idx
            out_ptr += out_increment_in_ch_loop_v;
            out_ch_idx += 2;
        } // for ch_mult_idx
    } // for in_ch_idx

    if (in_ch & 1) {
        for (int ch_mult_idx = 0; ch_mult_idx < ch_mul; ch_mult_idx++) {

            adjust_quant_params(&quant_params, out_ch_idx);
            int32_t bias_add = bias_additive(*biases++, 0x0, &quant_params);

            for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
                // Define area of input and filter for convolution
                // comp - compensation values for valid area definition
                mli_compensations comp;
                comp.top    = -MIN((H_idx * stride_height)- padding_top, 0);
                comp.bottom = -MIN(in_height - ((H_idx * stride_height)- padding_top + kernel_height), 0);
                const int rows = kernel_height - comp.top - comp.bottom;
                const int h_idx_in = (H_idx * stride_height - padding_top + comp.top);
                MLI_PTR(io_T) __restrict in_ptr = (MLI_PTR(io_T) __restrict)in_ftrs 
                        + h_idx_in * in_width * in_ch * filters                 // move to row
                        // + w_idx * in_ch * filters                            // move to column
                        + (in_ch - 1) * filters;                                  // move to channel
                MLI_PTR(w_T) __restrict w_ptr = (MLI_PTR(w_T) __restrict)weights
                        + comp.top * kernel_width * filters * out_ch            // move to row
                        //+ comp.left * filters * out_ch                        // move to column
                        + out_ch_idx;                                           // move to filter

                for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    // comp - compensation values for valid area definition
                    comp.left   = -MIN((W_idx * stride_width)- padding_left, 0);
                    comp.right  = -MIN(in_width - ((W_idx * stride_width)- padding_left + kernel_width), 0);
                    const int clmns = kernel_width - comp.right - comp.left;
                    const int w_idx_in = (W_idx * stride_width - padding_left + comp.left);

                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
                    accu = dotprod2D(&in_ptr[w_idx_in * in_ch * filters], &w_ptr[comp.left * filters * out_ch], accu, clmns, rows,
                                        in_col_step, in_row_step, krn_col_step, krn_row_step);

                    int32_t prev_w_adds = weights_additive(&w_ptr[comp.left * filters * out_ch], 0x0, &quant_params, clmns, rows, krn_col_step, krn_row_step);

                    accu = fx_add_q31(accu, prev_w_adds);
                    accu = fx_add_q31(bias_add, accu);

                    // Cast result to output type
                    mli_prv_clip_relu_store_output(out_ptr, accu, &quant_params, val_min_limit, val_max_limit);

                    out_ptr += out_increment_clmn_loop;
                } // for W_idx
                out_ptr += out_increment_row_loop;
            } // for H_idx
            out_ptr += out_increment_in_ch_loop;
            out_ch_idx++;
        } // for ch_mult_idx
    }
}

template <typename io_T, typename w_T, typename b_T, typename acc_T>
static __attribute__ ((always_inline)) void depthwise_convolution2D_hwc_krnpad(
        const MLI_PTR(io_T) __restrict in_ftrs,
        const MLI_PTR(w_T)  __restrict weights,
        const MLI_PTR(b_T)  __restrict biases,
              MLI_CONV_OUT_PTR(io_T) __restrict out_ftrs,

        const rect_t * const perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,

        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    rect_t perception_area_nopad;
    perception_area_nopad.row_beg = CEIL_DIV(padding_top, stride_height);
    perception_area_nopad.row_end = out_height - CEIL_DIV(padding_bot, stride_height);
    perception_area_nopad.clmn_beg = CEIL_DIV(padding_left, stride_width);
    perception_area_nopad.clmn_end = out_width - CEIL_DIV(padding_right, stride_width);
    
    depthwise_convolution2D_hwc_nopad<int8_t, int8_t, int32_t, mli_acc32_t>(
                in_ftrs, weights, biases, out_ftrs, &perception_area_nopad, quant_params,
                    val_min_limit, val_max_limit,
                in_ch, in_width, in_height,
                out_ch, out_width, out_height,
                kernel_height, kernel_width,
                stride_height, stride_width,
                padding_top, padding_left,
                padding_bot, padding_right);

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
            perc_areas[areas_num++].clmn_end = out_width;
        }
        if (padding_bot) {
            perc_areas[areas_num].row_beg = out_height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].row_end = out_height;
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = out_width;
        }
        if (padding_left) {
            perc_areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].row_end = out_height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = CEIL_DIV(padding_left, stride_width);
        }
        if (padding_right) {
            perc_areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].row_end = out_height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = out_width - CEIL_DIV(padding_right, stride_width);
            perc_areas[areas_num++].clmn_end = out_width;
        }
        for(int i = 0; i < areas_num; i ++) {
            depthwise_convolution2D_hwc<int8_t, int8_t, int32_t, mli_acc32_t>(
                    in_ftrs, weights, biases, out_ftrs, &perc_areas[i], quant_params,
                    val_min_limit, val_max_limit,
                    in_ch, in_width, in_height,
                    out_ch, out_width, out_height,
                    kernel_height, kernel_width,
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
static __attribute__ ((always_inline)) void convolution2D_hwc(
        const MLI_PTR(io_T) __restrict in_ftrs,
        const MLI_PTR(w_T)  __restrict weights,
        const MLI_PTR(b_T)  __restrict biases,
              MLI_CONV_OUT_PTR(io_T) __restrict out_ftrs,

        const rect_t * const perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,

        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;
    const int in_col_step = in_ch;
    const int in_row_step = in_width * in_ch;
    const int krn_col_step = in_ch;
    const int krn_row_step = kernel_width * in_ch;
    const int amount_rows = row_end - row_begin;
    const int amount_columns = clmn_end - clmn_begin;
    const int out_compensation_row_loop = out_ch * out_width * amount_rows;
    const int out_compensation_clmn_loop = out_ch * amount_columns;

    MLI_CONV_OUT_PTR(io_T) __restrict out_ptr = (MLI_CONV_OUT_PTR(io_T) __restrict)out_ftrs;
    out_ptr += out_ch *                 // common coefs
            (row_begin * out_width  +   // setup init coef for moving to row
            clmn_begin);                // setup init coef for moving to colum;

    for (int out_ch_idx = 0; out_ch_idx < out_ch; out_ch_idx++) {

        adjust_quant_params(&quant_params, out_ch_idx);
        const int bias_add = bias_additive(biases[out_ch_idx], 0x0, &quant_params);

        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {

            mli_compensations comp;
            comp.top    = -MIN((H_idx * stride_height)- padding_top, 0);
            comp.bottom = -MIN(in_height - ((H_idx * stride_height)- padding_top + kernel_height), 0);
            const int rows = kernel_height - comp.top - comp.bottom;
            const int h_idx_in = (H_idx * stride_height - padding_top + comp.top);
            MLI_PTR(io_T) __restrict in_ptr = (MLI_PTR(io_T) __restrict)in_ftrs 
                    + h_idx_in * in_width * in_ch;   // move to row
                    // + w_idx_in * in_ch              // move to column
                    // + in_ch_idx;                    // move to channel
            MLI_PTR(w_T) __restrict w_ptr = (MLI_PTR(w_T) __restrict)weights
                    + out_ch_idx * kernel_height * kernel_width * in_ch     // move to filter
                    + comp.top * kernel_width * in_ch;                      // move to row
                    //+   comp.left * in_ch +                               // move to column
                    //+   in_ch_idx;                                        // move to channel

            int32_t w_adds = 0;

            for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                // Define area of input and filter for convolution
                // comp - compensation values for valid area definition
                comp.left   = -MIN((W_idx * stride_width)- padding_left, 0);
                comp.right  = -MIN(in_width - ((W_idx * stride_width)- padding_left + kernel_width), 0);

                const int clmns = kernel_width - comp.right - comp.left;
                const int w_idx_in = (W_idx * stride_width - padding_left + comp.left);

                int8_t init_accum_weights_add_val = 0;
                w_adds = mli_prv_init_accu(init_accum_weights_add_val);
                for (int in_ch_idx = 0; in_ch_idx < in_ch-1; in_ch_idx+=2) {
                    w_adds = weights_additive_d(&w_ptr[comp.left * in_ch + in_ch_idx], &w_adds, &quant_params, 
                                clmns, rows, krn_col_step, krn_row_step);
                }
                if (in_ch & 1)
                {
                    w_adds = weights_additive(&w_ptr[comp.left * in_ch + in_ch-1], w_adds, &quant_params, 
                            clmns, rows, krn_col_step, krn_row_step);
                }

                int32_t init_accum_val = w_adds;
                acc_T accu = mli_prv_init_accu(init_accum_val);
                for (int in_ch_idx = 0; in_ch_idx < in_ch - 1; in_ch_idx+=2) {
                    dotprod2D_hwc_d<io_T, w_T, acc_T>(&in_ptr[w_idx_in * in_ch + in_ch_idx], 
                            &w_ptr[comp.left * in_ch + in_ch_idx], &accu, clmns, rows, in_col_step, in_row_step, 
                            krn_col_step, krn_row_step);
                }

                if (in_ch & 1)
                {
                    accu = dotprod2D(&in_ptr[w_idx_in * in_ch + in_ch - 1], &w_ptr[comp.left * in_ch + in_ch - 1], 
                            accu, clmns, rows, in_col_step, in_row_step, krn_col_step, krn_row_step);
                }
                accu += bias_add;

                // Cast result to output type, apply built-in ReLU Applying and write result
                mli_prv_clip_relu_store_output(out_ptr, accu, &quant_params, val_min_limit, val_max_limit);
                out_ptr += out_ch;
            } // for W_idx
            out_ptr += out_width * out_ch - out_compensation_clmn_loop;
        } // for H_idx
        out_ptr += 1 - out_compensation_row_loop;
    } // for out_ch_idx
}

template <typename io_T, typename w_T, typename b_T, typename acc_T>
static __attribute__ ((always_inline)) void convolution2D_hwc_nopad(
        const MLI_PTR(io_T) __restrict in_ftrs,
        const MLI_PTR(w_T)  __restrict weights,
        const MLI_PTR(b_T)  __restrict biases,
              MLI_CONV_OUT_PTR(io_T) __restrict out_ftrs,

        const rect_t * const perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,

        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;
    const int in_col_step = in_ch;
    const int in_row_step = in_width * in_ch;
    const int krn_col_step = in_ch;
    const int krn_row_step = kernel_width * in_ch;
    const int amount_rows = row_end - row_begin;
    const int amount_columns = clmn_end - clmn_begin;
    const int in_compensation_row_loop = in_ch * stride_height * in_width * amount_rows;
    const int in_compensation_clmn_loop = stride_width * in_ch * amount_columns;
    const int out_compensation_row_loop = out_ch * out_width * amount_rows;
    const int out_compensation_clmn_loop = out_ch * amount_columns;

    MLI_CONV_OUT_PTR(io_T) __restrict out_ptr = (MLI_CONV_OUT_PTR(io_T) __restrict)out_ftrs;
    out_ptr += out_ch *                 // common coefs
            (row_begin * out_width  +   // setup init coef for moving to row
            clmn_begin);                // setup init coef for moving to colum;
    MLI_PTR(io_T) __restrict in_ptr = (MLI_PTR(io_T) __restrict)in_ftrs;
    in_ptr += in_ch * ((row_begin * stride_height - padding_top) * in_width +          // move to row
              (clmn_begin * stride_width - padding_left));                             // move to column

    for (int out_ch_idx = 0; out_ch_idx < out_ch; out_ch_idx++) {
        adjust_quant_params(&quant_params, out_ch_idx);
        const int bias_add = bias_additive(biases[out_ch_idx], 0x0, &quant_params);
        MLI_PTR(w_T) __restrict w_ptr_local = (MLI_PTR(w_T) __restrict)weights + out_ch_idx * kernel_height * kernel_width * in_ch;
        int8_t init_accum_val = 0;
        int weights_add = mli_prv_init_accu(init_accum_val);

        for (int in_ch_idx = 0; in_ch_idx < in_ch-1; in_ch_idx+=2) {
            weights_add = weights_additive_d(w_ptr_local, &weights_add, &quant_params, 
                        kernel_width, kernel_height, krn_col_step, krn_row_step);
            w_ptr_local += 2;
        }

        if (in_ch & 1)
        {
            weights_add = weights_additive(w_ptr_local++, weights_add, &quant_params, 
                    kernel_width, kernel_height, krn_col_step, krn_row_step);
        }

        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                MLI_PTR(w_T) __restrict w_ptr = (MLI_PTR(w_T) __restrict)weights + out_ch_idx * kernel_height 
                        * kernel_width * in_ch;
                int32_t init_accum_val = weights_add;
                acc_T accu = mli_prv_init_accu(init_accum_val);
                for (int in_ch_idx = 0; in_ch_idx < in_ch -1; in_ch_idx+=2) {
                    dotprod2D_hwc_d<io_T, w_T, acc_T>(in_ptr, w_ptr, &accu, kernel_width, kernel_height,
                                        in_col_step, in_row_step, krn_col_step, krn_row_step);
                    in_ptr+= 2;
                    w_ptr += 2;
                }

                if (in_ch & 1) {
                    accu = dotprod2D(in_ptr, w_ptr, accu, kernel_width, kernel_height,
                                        in_col_step, in_row_step, krn_col_step, krn_row_step);
                    in_ptr++;
                    w_ptr++;
                }
                accu += bias_add;
                
                // Cast result to output type, apply built-in ReLU Applying and write result
                mli_prv_clip_relu_store_output(out_ptr, accu, &quant_params, val_min_limit, val_max_limit);
                out_ptr += out_ch;
                in_ptr += in_ch * (stride_width - 1);
            } // for W_idx
            out_ptr += out_width * out_ch - out_compensation_clmn_loop;
            in_ptr += stride_height * in_width * in_ch - in_compensation_clmn_loop;
        } // for H_idx
        in_ptr -=  in_compensation_row_loop;
        out_ptr += 1 - out_compensation_row_loop;
    } // for out_ch_idx
}

template <typename io_T, typename w_T, typename b_T, typename acc_T>
static __attribute__ ((always_inline)) void convolution2D_hwc_krnpad(
        const MLI_PTR(io_T) __restrict in_ftrs,
        const MLI_PTR(w_T)  __restrict weights,
        const MLI_PTR(b_T)  __restrict biases,
              MLI_CONV_OUT_PTR(io_T) __restrict out_ftrs,

        const rect_t * const perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,

        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right ) {

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    rect_t perception_area_nopad;
    perception_area_nopad.row_beg = CEIL_DIV(padding_top, stride_height);
    perception_area_nopad.row_end = out_height - CEIL_DIV(padding_bot, stride_height);
    perception_area_nopad.clmn_beg = CEIL_DIV(padding_left, stride_width);
    perception_area_nopad.clmn_end = out_width - CEIL_DIV(padding_right, stride_width);
    
    convolution2D_hwc_nopad<int8_t, int8_t, int32_t, mli_acc32_t>(
                in_ftrs, weights, biases, out_ftrs, &perception_area_nopad, quant_params,
                val_min_limit, val_max_limit,
                in_ch, in_width, in_height,
                out_ch, out_width, out_height,
                kernel_height, kernel_width,
                stride_height, stride_width,
                padding_top, padding_left,
                padding_bot, padding_right);

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
            perc_areas[areas_num++].clmn_end = out_width;
        }
        if (padding_bot) {
            perc_areas[areas_num].row_beg = out_height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].row_end = out_height;
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = out_width;
        }
        if (padding_left) {
            perc_areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].row_end = out_height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = CEIL_DIV(padding_left, stride_width);
        }
        if (padding_right) {
            perc_areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].row_end = out_height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = out_width - CEIL_DIV(padding_right, stride_width);
            perc_areas[areas_num++].clmn_end = out_width;
        }
        for(int i = 0; i < areas_num; i ++) {
            convolution2D_hwc<int8_t, int8_t, int32_t, mli_acc32_t>(
                    in_ftrs, weights, biases, out_ftrs, &perc_areas[i], quant_params,
                    val_min_limit, val_max_limit,
                    in_ch, in_width, in_height,
                    out_ch, out_width, out_height,
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_top, padding_left,
                    padding_bot, padding_right);
        }
    }
}

template <typename io_T, typename w_T, typename b_T, typename acc_T>
static __attribute__ ((always_inline)) void pointwise_convolution2D_hwc_nopad(
        const MLI_PTR(io_T) __restrict in_ftrs,
        const MLI_PTR(w_T)  __restrict weights,
        const MLI_PTR(b_T)  __restrict biases,
              MLI_CONV_OUT_PTR(io_T) __restrict out_ftrs,

        const rect_t * const perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,

        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;
    const int krn_col_step = in_ch;
    const int krn_row_step = in_ch;
    const int amount_rows = row_end - row_begin;
    const int amount_columns = clmn_end - clmn_begin;
    const int in_compensation_row_loop = in_ch * stride_height * in_width * amount_rows;
    const int in_compensation_clmn_loop = stride_width * in_ch * amount_columns;
    const int out_compensation_row_loop = out_ch * out_width * amount_rows;
    const int out_compensation_clmn_loop = out_ch * amount_columns;

    MLI_CONV_OUT_PTR(io_T) __restrict out_ptr = (MLI_CONV_OUT_PTR(io_T) __restrict)out_ftrs;
    out_ptr += out_ch *                 // common coefs
            (row_begin * out_width  +   // setup init coef for moving to row
            clmn_begin);                // setup init coef for moving to colum;
    MLI_PTR(io_T) __restrict in_ptr = (MLI_PTR(io_T) __restrict)in_ftrs;
    in_ptr += in_ch * (row_begin * stride_height * in_width +          // move to row
              clmn_begin * stride_width);                              // move to column

    for (int out_ch_idx = 0; out_ch_idx < out_ch; out_ch_idx++) {
        adjust_quant_params(&quant_params, out_ch_idx);
        const int bias_add = bias_additive(biases[out_ch_idx], 0x0, &quant_params);
        MLI_PTR(w_T) __restrict w_ptr_local = (MLI_PTR(w_T) __restrict)weights + out_ch_idx * in_ch;
        int8_t init_accum_val = 0;
        int weights_add = mli_prv_init_accu(init_accum_val);

        for (int in_ch_idx = 0; in_ch_idx < in_ch-1; in_ch_idx+=2) {
            weights_add = weights_additive_d(w_ptr_local, &weights_add, &quant_params, 
                        kernel_width, kernel_height, krn_col_step, krn_row_step);
            w_ptr_local += 2;
        }

        if (in_ch & 1)
        {
            weights_add = weights_additive(w_ptr_local++, weights_add, &quant_params, 
                    kernel_width, kernel_height, krn_col_step, krn_row_step);
        }

        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                MLI_PTR(w_T) __restrict w_ptr = (MLI_PTR(w_T) __restrict)weights + out_ch_idx * in_ch;

                int32_t init_accum_val = weights_add;
                acc_T accu = mli_prv_init_accu(init_accum_val);

                for (int in_ch_idx = 0; in_ch_idx < in_ch -1; in_ch_idx+=2) {
                    const MLI_PTR(io_T) __restrict tmp_in_ptr = (const MLI_PTR(io_T) __restrict)in_ptr;
                    const MLI_PTR(w_T) __restrict tmp_w_ptr = (const MLI_PTR(w_T) __restrict)w_ptr;
                    mli_prv_load_mac_vec2(&accu, tmp_in_ptr, tmp_w_ptr);
                    in_ptr+= 2;
                    w_ptr += 2;
                }

                if (in_ch & 1) {
                    accu = mli_math_mac_fx(accu, *in_ptr, *w_ptr);
                    in_ptr++;
                    w_ptr++;
                }
                accu += bias_add;
                
                // Cast result to output type, apply built-in ReLU Applying and write result
                mli_prv_clip_relu_store_output(out_ptr, accu, &quant_params, val_min_limit, val_max_limit);
                out_ptr += out_ch;
                in_ptr += in_ch * (stride_width - 1);
            } // for W_idx
            out_ptr += out_width * out_ch - out_compensation_clmn_loop;
            in_ptr += stride_height * in_width * in_ch - in_compensation_clmn_loop;
        } // for H_idx
        in_ptr -=  in_compensation_row_loop;
        out_ptr += 1 - out_compensation_row_loop;
    } // for out_ch_idx
}

#endif // _MLI_KRN_CONV2D_HWC_H_

