/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_AVEPOOL_hwc_H_
#define _MLI_KRN_AVEPOOL_hwc_H_

#include "mli_krn_reduce_sum2d.h"

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_prv_dsp.h"

/******************************************************************************
 *
 * Version & platform description
 * Targets:
 *
 ******************************************************************************/
template <typename io_T>
static inline void __attribute__((always_inline)) avepool_hwc_nopad(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const MLI_PTR(io_T) __restrict in_ftrs,
        MLI_OUT_PTR(io_T) __restrict out_ftrs,
        const int channels,
        const int in_width,
        const int in_height,
        const int out_width,
        const int out_height,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_left,
        const int padding_right,
        const int padding_bot) {
    (void)padding_right;
    (void)padding_bot;
    
    unsigned int max_kernel_size = kernel_width * kernel_height;
    const int go_over_ch_in_wd = channels * in_width * stride_height;
    const int go_over_ch = channels * stride_width;
    int16_t mul = 0;
    int shift = 0;
    // in case of average pooling, the sum needs to be divided by the kernel size.
    // calculate 1/(rows*clmns) to get a multiplication factor and shift value.
    get_mul_shift_value(max_kernel_size, &mul, &shift);

    for (int ch_idx = 0; ch_idx < (channels - 1); ch_idx += 2) {
        for (int H_idx = 0; H_idx < out_height; H_idx++) {
            for (int W_idx = 0; W_idx < out_width; W_idx++) {
                auto v2acc = reduce_sum2D_hwc_v(in_ftrs, kernel_width, kernel_height, channels, in_width, mul);
                mli_prv_clip_and_store_output_v(out_ftrs, &v2acc, shift);
                in_ftrs += go_over_ch;
                out_ftrs += channels;
            } // for W_idx
            in_ftrs += channels * (in_width * stride_height - stride_width * clmn_end);
        } // for H_idx
        in_ftrs += 2 - go_over_ch_in_wd * row_end;
        out_ftrs += 2 - channels * clmn_end * row_end;
    } // for ch_idx 

    if(channels & 1){
        for (int H_idx = 0; H_idx < out_height; H_idx++) {
            for (int W_idx = 0; W_idx < out_width; W_idx++) {
                auto acc = reduce_sum2D_hwc(in_ftrs, kernel_width, kernel_height, channels, in_width, mul);
                mli_prv_shift_clip_and_store_output(out_ftrs, &acc, shift);
                // Write results
                in_ftrs += channels * stride_width;
                out_ftrs += channels;
            } // for W_idx 
            in_ftrs += channels * (in_width * stride_height - stride_width * clmn_end);
        } // for H_idx
    }
}


template <typename io_T>
static inline void __attribute__((always_inline)) avepool_hwc(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const MLI_PTR(io_T) __restrict in_ftrs,
        MLI_OUT_PTR(io_T) __restrict out_ftrs,
        const int channels,
        const int in_width,
        const int in_height,
        const int out_width,
        const int out_height,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_left,
        const int padding_right,
        const int padding_bot) {
    (void)padding_right;
    (void)padding_bot;
    
    for (int ch_idx = 0; ch_idx < channels; ch_idx++) {
        for (int H_idx = 0; H_idx < out_height; H_idx++) {
            for (int W_idx = 0; W_idx < out_width; W_idx++) {
                // Define area of input and filter for convolution
                // *_comp - compensation values for valid area defining
                int top_comp = -MIN((int)(H_idx * stride_height)- padding_top, 0);
                int left_comp = -MIN((int)(W_idx * stride_width)- padding_left, 0);

                int right_comp = -MIN((int)in_width - ((int32_t)(W_idx * stride_width)- padding_left + kernel_width), 0);
                int bottom_comp = -MIN((int)in_height - ((int32_t)(H_idx * stride_height)- padding_top + kernel_height), 0);

                int rows = kernel_height - top_comp - bottom_comp;
                int clmns = kernel_width - right_comp - left_comp;

                // in case of average pooling, the sum needs to be divided by the kernel size.
                // calculate 1/(rows*clmns) to get a multiplication factor and shift value.
                const int kernel_size = rows * clmns;
                int16_t mul = 0;
                int shift = 0;
                get_mul_shift_value(kernel_size, &mul, &shift);

                MLI_PTR(io_T) in_ptr = (MLI_PTR(io_T))in_ftrs +                                     // starting point
                        channels * in_width * (H_idx * stride_height - padding_top + top_comp) +    // move to row
                        channels * ((W_idx * stride_width) - padding_left + left_comp) +            // move to column
                        ch_idx;                                                                     // move to channel
                mli_acc40_t acc = reduce_sum2D_hwc(in_ptr, clmns, rows, channels, in_width, mul);
                mli_prv_shift_clip_and_store_output(out_ftrs, &acc, shift);
                out_ftrs += channels;
            } // for W_idx 
        } // for H_idx
        out_ftrs += 1 - channels * clmn_end * row_end;
    } // for ch_idx 
}

template <typename io_T>
static inline void __attribute__((always_inline)) avepool_hwc_krnpad(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const MLI_PTR(io_T) __restrict in_ftrs,
        MLI_OUT_PTR(io_T) __restrict out_ftrs,
        const int channels,
        const int in_width,
        const int in_height,
        const int out_width,
        const int out_height,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_left,
        const int padding_right,
        const int padding_bot) {
    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    avepool_hwc_nopad(
            row_beg, row_end, clmn_beg, clmn_end,
            in_ftrs, out_ftrs, channels, in_width,
            in_height, out_width, out_height, kernel_height,
            kernel_width, stride_height, stride_width,
            padding_top, padding_left, padding_right, padding_bot);

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

        avepool_hwc(
            row_beg, row_end,
            clmn_beg, clmn_end,
            in_ftrs, out_ftrs,
            channels, in_width, in_height,
            out_width, out_height,
            kernel_height, kernel_width,
            stride_height, stride_width,
            padding_top, padding_left, padding_right, padding_bot);
    }
}

#endif  //_MLI_KRN_AVEPOOL_hwc_H_
