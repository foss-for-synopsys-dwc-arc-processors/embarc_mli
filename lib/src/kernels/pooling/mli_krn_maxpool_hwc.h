/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_MAXPOOL_HWC_H_
#define _MLI_KRN_MAXPOOL_HWC_H_

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math_macros.h"
#include "mli_private_types.h"
#include "mli_prv_dsp.h"
#include "mli_types.h"

#define REDUCE_MAX2D_UNROLL_FACTOR_FOR_WIDTH 7
#define REDUCE_MAX2D_UNROLL_FACTOR_FOR_HEIGHT 7

#ifdef __FXAPI__
#include <fxarc.h>
#endif

/**
 * Function Short Description
 *
 * \param[in]
 * \param[in/out]
 * \param[out]
 * \result
 *
 * Some Details
 */

#pragma Code(".mli_lib")

/******************************************************************************
 *
 * Version & platform description
 * Targets:
 *
 ******************************************************************************/
template <typename io_T>
static inline io_T __attribute__((always_inline)) reduce_max2D_hwc(
        const MLI_PTR(io_T) in,
        const int width,
        const int height,
        const int channels,
        const int in_row_step,
        const bool fixed_size) {
    int row = 0;
    int clmn = 0;
    io_T cur_max = in[channels * clmn++];
    for (; row < height; row++) {
        for (; clmn < width; clmn++) {
            cur_max = MAX(cur_max, in[clmn * channels]);
        }
        clmn = 0;
        in += in_row_step * channels;
    }
    return cur_max;
}

template <typename io_T>
static inline v2q15_t reduce_max2D_hwc_v(
        const MLI_PTR(io_T) in,
        const int width,
        const int height,
        const int channels,
        const int in_row_step,
        const bool fixed_size) {
    v2q15_t cur_max = mli_prv_load_2_samples(in);
    if (width == 1){
        for (int row = 0; row < height; row++) {
            cur_max = fx_max_v2q15(cur_max, mli_prv_load_2_samples(&in[in_row_step * channels * row]));
        }
    } else if (height == 1){
        for (int clmn = 0; clmn < width; clmn++) {
            cur_max = fx_max_v2q15(cur_max, mli_prv_load_2_samples(&in[clmn * channels]));
        }
    } else {
        if (fixed_size && height <= REDUCE_MAX2D_UNROLL_FACTOR_FOR_HEIGHT && height <= REDUCE_MAX2D_UNROLL_FACTOR_FOR_WIDTH) {
#pragma clang loop unroll(full)
            for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
                for (int clmn = 0; clmn < width; clmn++) {
                    cur_max = fx_max_v2q15(cur_max, mli_prv_load_2_samples(&in[clmn * channels]));
                }
                in += in_row_step * channels;
            }
        } else {
            for (int row = 0; row < height; row++) {
                for (int clmn = 0; clmn < width; clmn++) {
                    cur_max = fx_max_v2q15(cur_max, mli_prv_load_2_samples(&in[clmn * channels]));
                }
                in += in_row_step * channels;
            }
        }
    }
    return cur_max;
}

template <typename io_T>
static inline void __attribute__((always_inline)) maxpool_hwc_nopad(
    int row_beg,
    int row_end, 
    int clmn_beg,
    int clmn_end,
    int stride_width,
    int stride_height,
    int padding_top,
    int padding_bot,
    int padding_left,
    int padding_right,
    const MLI_PTR(io_T) in_ftrs,
    MLI_OUT_PTR(io_T) out_ftrs,
    int channels_num,
    int kernel_height,
    int kernel_width,
    int in_height,
    int in_width,
    int out_width,
    int out_height) {

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in_height >= kernel_height && in_width >= kernel_width) {
        for (int ch_idx = 0; ch_idx < channels_num - (channels_num & 1); ch_idx+=2) {
            for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    const MLI_PTR(io_T) in_ptr =
                            in_ftrs +                                                          // starting point
                            channels_num * in_width * (H_idx * stride_height - padding_top) +  // move to row
                            channels_num * (W_idx * stride_width - padding_left) +             // move to column
                            ch_idx;                                                            // move to channel

                    // Core Max
                    v2q15_t max_val = reduce_max2D_hwc_v<io_T>(in_ptr, kernel_width, kernel_height, channels_num, in_width, true);
                    // Write results
                    mli_prv_store_2_samples(&out_ftrs[ch_idx + (H_idx * out_width + W_idx) * channels_num], max_val);
                }
            }
        }
        if (channels_num & 1){
            for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    const MLI_PTR(io_T) in_ptr =
                            in_ftrs +                                                          // starting point
                            channels_num * in_width * (H_idx * stride_height - padding_top) +  // move to row
                            channels_num * (W_idx * stride_width - padding_left) +             // move to column
                            channels_num - 1;                                                  // move to channel

                    // Core Max
                    io_T max_val = reduce_max2D_hwc<io_T>(in_ptr, kernel_width, kernel_height, channels_num, in_width, true);

                    // Write results
                    out_ftrs[channels_num - 1 + (H_idx * out_width + W_idx) * channels_num] = max_val;
                }
            }
        }
    }   
}

template <typename io_T>
static inline void __attribute__((always_inline)) maxpool_hwc_pad(
    int row_beg,
    int row_end, 
    int clmn_beg,
    int clmn_end,
    int stride_width,
    int stride_height,
    int padding_top,
    int padding_bot,
    int padding_left,
    int padding_right,
    const MLI_PTR(io_T) in_ftrs,
    MLI_OUT_PTR(io_T) out_ftrs,
    int channels_num,
    int kernel_height,
    int kernel_width,
    int in_height,
    int in_width,
    int out_width,
    int out_height) {

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    row_beg = CEIL_DIV(padding_top, stride_height);
    row_end = out_height - CEIL_DIV(padding_bot, stride_height);
    clmn_beg = CEIL_DIV(padding_left, stride_width);
    clmn_end = out_width - CEIL_DIV(padding_right, stride_width);

    if ((row_end - row_beg > 0) && (clmn_end - clmn_beg > 0)) {
        maxpool_hwc_nopad<io_T>(
            row_beg, row_end, clmn_beg, clmn_end,
            stride_width, stride_height, padding_top,
            padding_bot, padding_left, padding_right,
            in_ftrs, out_ftrs, channels_num, kernel_height,
            kernel_width, in_height, in_width, out_width,
            out_height);
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

        for (int area_idx = 0; area_idx < areas_num; ++area_idx) {
            for (int ch_idx = 0; ch_idx < channels_num - (channels_num & 1); ch_idx+=2) {
                for (int H_idx = perc_areas[area_idx].row_beg; H_idx < perc_areas[area_idx].row_end; H_idx++) {
                    for (int W_idx = perc_areas[area_idx].clmn_beg; W_idx < perc_areas[area_idx].clmn_end; W_idx++) {
                        // Define area of input and filter for convolution
                        // *_comp - compensation values for valid area defining
                        int top_comp = -MIN((H_idx * stride_height) - padding_top, 0);
                        int left_comp = -MIN((W_idx * stride_width) - padding_left, 0);

                        int right_comp = -MIN(in_width - ((W_idx * stride_width) - padding_left + kernel_width), 0);
                        int bottom_comp = -MIN(in_height - ((H_idx * stride_height) - padding_top + kernel_height), 0);

                        int rows = kernel_height - top_comp - bottom_comp;
                        int clmns = kernel_width - right_comp - left_comp;

                        const MLI_PTR(io_T) in_ptr =
                                in_ftrs +  // starting point
                                channels_num * in_width *
                                (H_idx * stride_height - padding_top + top_comp) +            // move to row
                                channels_num * ((W_idx * stride_width) - padding_left + left_comp) +  // move to column
                                ch_idx;

                        // Core Max
                        v2q15_t max_val = reduce_max2D_hwc_v<io_T>(in_ptr, clmns, rows, channels_num, in_width, false);

                        // Write result
                        mli_prv_store_2_samples(&out_ftrs[ch_idx + (H_idx * out_width + W_idx) * channels_num], max_val);
                    }
                }
            }
            if (channels_num & 1)
            {
                for (int H_idx = perc_areas[area_idx].row_beg; H_idx < perc_areas[area_idx].row_end; H_idx++) {
                    for (int W_idx = perc_areas[area_idx].clmn_beg; W_idx < perc_areas[area_idx].clmn_end; W_idx++) {
                        // Define area of input and filter for convolution
                        // *_comp - compensation values for valid area defining
                        int top_comp = -MIN((H_idx * stride_height) - padding_top, 0);
                        int left_comp = -MIN((W_idx * stride_width) - padding_left, 0);

                        int right_comp = -MIN(in_width - ((W_idx * stride_width) - padding_left + kernel_width), 0);
                        int bottom_comp = -MIN(in_height - ((H_idx * stride_height) - padding_top + kernel_height), 0);

                        int rows = kernel_height - top_comp - bottom_comp;
                        int clmns = kernel_width - right_comp - left_comp;

                        const MLI_PTR(io_T) in_ptr =
                                in_ftrs +  // starting point
                                channels_num * in_width *
                                (H_idx * stride_height - padding_top + top_comp) +            // move to row
                                channels_num * ((W_idx * stride_width) - padding_left + left_comp) +  // move to column
                                channels_num - 1;

                        // Core Max
                        io_T max_val = reduce_max2D_hwc<io_T>(in_ptr, clmns, rows, channels_num, in_width, false);

                        // Write result
                        out_ftrs[channels_num - 1 + (H_idx * out_width + W_idx) * channels_num] = max_val;
                    }
                }
            }
        }
    }
}

#pragma code()

#endif //_MLI_KRN_MAXPOOL_HWC_H_