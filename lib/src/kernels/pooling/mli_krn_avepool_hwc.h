/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_AVEPOOL_HWC_H_
#define _MLI_KRN_AVEPOOL_HWC_H_

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
    
    const unsigned int max_kernel_size = kernel_width * kernel_height;
    const int number_rows = row_end - row_beg;
    const int number_clmns = clmn_end - clmn_beg;
    const int goto_next_row = channels * in_width * stride_height;
    const int goto_next_clmn = channels * stride_width;
    const int compensation_inp_width_lp = channels * stride_width * number_clmns;
    const int compensation_inp_height_lp = goto_next_row  * number_rows;
    const int out_compensation_row_loop = channels * out_width * number_rows;
    const int out_compensation_clmn_loop = channels * number_clmns;
    int16_t mul = 0;

    in_ftrs  += ((row_beg * stride_height - padding_top) * in_width +            // move to row
                 (clmn_beg * stride_width - padding_left)) * channels;           // move to column
    out_ftrs += (row_beg  * out_width +                                          // move to row
                 clmn_beg)  * channels;                                           // move to column
    int shift = 0;
    // in case of average pooling, the sum needs to be divided by the kernel size.
    // calculate 1/(rows*clmns) to get a multiplication factor and shift value.
    get_mul_shift_value(max_kernel_size, &mul, &shift);

    for (int ch_idx = 0; ch_idx < (channels - 1); ch_idx += 2) {
        for (int H_idx = 0; H_idx < number_rows; H_idx++) {
            for (int W_idx = 0; W_idx < number_clmns; W_idx++) {
                auto v2acc = reduce_sum2D_hwc_v(in_ftrs, kernel_width, kernel_height, channels, in_width, mul);
                mli_prv_clip_and_store_output_v(out_ftrs, &v2acc, shift);
                in_ftrs += goto_next_clmn; //go to the next input column
                out_ftrs += channels; //go to the next output column without out stride
            } // for W_idx
            // go to the next row given compensation of previous loops incrementing
            in_ftrs += goto_next_row - compensation_inp_width_lp;
            out_ftrs += out_width * channels - out_compensation_clmn_loop;
        } // for H_idx
        // go to the next channel given compensation of previous loops incrementing 
        in_ftrs  += 2 - compensation_inp_height_lp;
        // go to next channel with compensation of previous loops incrementing
        out_ftrs += 2 - out_compensation_row_loop;
    } // for ch_idx 


    if(channels & 1){
        for (int H_idx = 0; H_idx < number_rows; H_idx++) {
            for (int W_idx = 0; W_idx < number_clmns; W_idx++) {
                auto acc = reduce_sum2D_hwc(in_ftrs, kernel_width, kernel_height, channels, in_width, mul);
                mli_prv_shift_clip_and_store_output(out_ftrs, &acc, shift);
                // Write results
                in_ftrs += goto_next_clmn; //go to the next input column
                out_ftrs += channels; //go to the next output column without out stride
            } // for W_idx 
            // go to the next row given compensation of previous loops incrementing
            out_ftrs += out_width * channels - out_compensation_clmn_loop;
            in_ftrs += goto_next_row - compensation_inp_width_lp;
        } // for H_idx
        out_ftrs += 1 - out_compensation_row_loop;
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
    
    const int number_rows = row_end - row_beg;
    const int number_clmns = clmn_end - clmn_beg;
    const int out_compensation_row_loop = channels * out_width * number_rows;
    const int out_compensation_clmn_loop = channels * number_clmns;

    out_ftrs += (row_beg  * out_width +  // move to row
                 clmn_beg)  * channels;   // move to column

    for (int ch_idx = 0; ch_idx < channels; ch_idx++) {
        for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
            for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
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
                out_ftrs += channels; //go to the next output column without out stride
            } // for W_idx 
            out_ftrs += out_width * channels - out_compensation_clmn_loop;
        } // for H_idx
        // go to next channel with compensation of previous loops incrementing
        out_ftrs += 1 - out_compensation_row_loop;
    } // for ch_idx 
}

template <typename io_T>
static inline void __attribute__((always_inline)) avepool_hwc_krnpad(
        int row_beg,
        int row_end,
        int clmn_beg,
        int clmn_end,
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

    row_beg = CEIL_DIV(padding_top, stride_height);
    row_end = out_height - CEIL_DIV(padding_bot, stride_height);
    clmn_beg = CEIL_DIV(padding_left, stride_width);
    clmn_end = out_width - CEIL_DIV(padding_right, stride_width);

    if ((row_end - row_beg > 0) && (clmn_end - clmn_beg > 0)) {
        avepool_hwc_nopad(
                row_beg, row_end, clmn_beg, clmn_end,
                in_ftrs, out_ftrs, channels, in_width,
                in_height, out_width, out_height, kernel_height,
                kernel_width, stride_height, stride_width,
                padding_top, padding_left, padding_right, padding_bot);
    }

    // Phase 2: Process border part with more complex algorithm
    // (usually significantly smaller part of computations)
    //=======================================================================
    if (padding_top || padding_left || padding_bot || padding_right) {
        rect_t areas[4];
        int areas_num = 0;
        if (padding_top) {
            areas[areas_num].row_beg = 0;
            areas[areas_num].row_end = CEIL_DIV(padding_top, stride_height);
            areas[areas_num].clmn_beg = 0;
            areas[areas_num++].clmn_end = out_width;
        }
        if (padding_bot) {
            areas[areas_num].row_beg = out_height - CEIL_DIV(padding_bot, stride_height);
            areas[areas_num].row_end = out_height;
            areas[areas_num].clmn_beg = 0;
            areas[areas_num++].clmn_end = out_width;
        }
        if (padding_left) {
            areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            areas[areas_num].row_end = out_height - CEIL_DIV(padding_bot, stride_height);
            areas[areas_num].clmn_beg = 0;
            areas[areas_num++].clmn_end = CEIL_DIV(padding_left, stride_width);
        }
        if (padding_right) {
            areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            areas[areas_num].row_end = out_height - CEIL_DIV(padding_bot, stride_height);
            areas[areas_num].clmn_beg = out_width - CEIL_DIV(padding_right, stride_width);
            areas[areas_num++].clmn_end = out_width;
        }

        for (int i = 0; i < areas_num; i++) {
            avepool_hwc(
                    areas[i].row_beg, areas[i].row_end, areas[i].clmn_beg, areas[i].clmn_end,
                    in_ftrs, out_ftrs,
                    channels, in_width, in_height,
                    out_width, out_height,
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_top, padding_left, padding_right, padding_bot);
        }
    }
}

#endif  //_MLI_KRN_AVEPOOL_HWC_H_
