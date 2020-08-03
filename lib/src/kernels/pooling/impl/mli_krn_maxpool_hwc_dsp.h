/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_MAXPOOL_HWC_DSP_H_
#define _MLI_KRN_MAXPOOL_HWC_DSP_H_

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
        const int col_mem_stride,
        const int row_mem_stride,
        const bool fixed_size) {
    io_T cur_max = in[0];
    for (int row = 0; row < height; row++) {
        for (int clmn = 0; clmn < width; clmn++) {
            cur_max = MAX(cur_max, in[(row_mem_stride * row) + (col_mem_stride * clmn)]);
        }
    }
    return cur_max;
}

template <typename io_T>
static inline v2q15_t reduce_max2D_hwc_v(
        const MLI_PTR(io_T) in,
        const int width,
        const int height,
        const int channels,
        const int col_mem_stride,
        const int row_mem_stride,
        const bool fixed_size) {

    v2q15_t cur_max = mli_prv_load_2_samples(in);
    if (width == 1){
        for (int row = 0; row < height; row++) {
            cur_max = fx_max_v2q15(cur_max, mli_prv_load_2_samples(&in[row*row_mem_stride]));
        }
    } else if (height == 1){
        for (int clmn = 0; clmn < width; clmn++) {
            cur_max = fx_max_v2q15(cur_max, mli_prv_load_2_samples(&in[clmn*col_mem_stride]));
        }
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
        if (fixed_size && height <= REDUCE_MAX2D_UNROLL_FACTOR_FOR_HEIGHT && height <= REDUCE_MAX2D_UNROLL_FACTOR_FOR_WIDTH) {
#pragma clang loop unroll(full)
            for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
                for (int clmn = 0; clmn < width; clmn++) {
                    cur_max = fx_max_v2q15(cur_max, mli_prv_load_2_samples(
                        &in[(row * row_mem_stride) + (clmn * col_mem_stride)]));
                }
            }
        } else {
            for (int row = 0; row < height; row++) {
                for (int clmn = 0; clmn < width; clmn++) {
                    cur_max = fx_max_v2q15(cur_max, mli_prv_load_2_samples(
                        &in[(row * row_mem_stride) +  (clmn * col_mem_stride)]));
                }
            }
        }
#pragma clang diagnostic pop
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
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        int kernel_height,
        int kernel_width) {

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in.height >= kernel_height && in.width >= kernel_width) {
        for (int ch_idx = 0; ch_idx < in.ch - (in.ch & 1); ch_idx+=2) {
            for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    const MLI_PTR(io_T) in_ptr = in.ptr
                            + in.row_mem_stride * (H_idx * stride_height - padding_top) // move to row
                            + in.col_mem_stride * (W_idx * stride_width - padding_left) // move to column
                            + ch_idx;                                                   // move to channel

                    // Core Max
                    v2q15_t max_val = reduce_max2D_hwc_v<io_T>(in_ptr, kernel_width, kernel_height, in.ch,
                        in.col_mem_stride, in.row_mem_stride, true);
                    // Write results
                    mli_prv_store_2_samples(&out.ptr[
                            out.row_mem_stride * H_idx +
                            out.col_mem_stride * W_idx +
                            ch_idx], max_val);
                }
            }
        }
        if (in.ch & 1) {
            for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    const MLI_PTR(io_T) in_ptr = in.ptr
                            + in.row_mem_stride * (H_idx * stride_height - padding_top) // move to row
                            + in.col_mem_stride * (W_idx * stride_width - padding_left) // move to column
                            + in.ch - 1;                                                // move to channel

                    // Core Max
                    io_T max_val = reduce_max2D_hwc<io_T>(in_ptr, kernel_width, kernel_height,
                        in.col_mem_stride, in.row_mem_stride, true);

                    // Write results
                    out.ptr[out.row_mem_stride * H_idx
                            + out.col_mem_stride * W_idx
                            + out.ch - 1] = max_val;
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
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        int kernel_height,
        int kernel_width) {

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    row_beg = CEIL_DIV(padding_top, stride_height);
    row_end = out.height - CEIL_DIV(padding_bot, stride_height);
    clmn_beg = CEIL_DIV(padding_left, stride_width);
    clmn_end = out.width - CEIL_DIV(padding_right, stride_width);

    if ((row_end - row_beg > 0) && (clmn_end - clmn_beg > 0)) {
        maxpool_hwc_nopad<io_T>(
                row_beg, row_end, clmn_beg, clmn_end,
                stride_width, stride_height, padding_top,
                padding_bot, padding_left, padding_right,
                in, out,
                kernel_height, kernel_width);
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

        for (int area_idx = 0; area_idx < areas_num; ++area_idx) {
            for (int ch_idx = 0; ch_idx < in.ch - (in.ch & 1); ch_idx+=2) {
                for (int H_idx = perc_areas[area_idx].row_beg; H_idx < perc_areas[area_idx].row_end; H_idx++) {
                    for (int W_idx = perc_areas[area_idx].clmn_beg; W_idx < perc_areas[area_idx].clmn_end; W_idx++) {
                        // Define area of input and filter for convolution
                        // *_comp - compensation values for valid area defining
                        int top_comp = -MIN((H_idx * stride_height) - padding_top, 0);
                        int left_comp = -MIN((W_idx * stride_width) - padding_left, 0);

                        int right_comp = -MIN(in.width - ((W_idx * stride_width) - padding_left + kernel_width), 0);
                        int bottom_comp = -MIN(in.height - ((H_idx * stride_height) - padding_top + kernel_height), 0);

                        int rows = kernel_height - top_comp - bottom_comp;
                        int clmns = kernel_width - right_comp - left_comp;

                        // Define area of input and filter for convolution
                        const MLI_PTR(io_T) in_ptr = in.ptr
                                + in.row_mem_stride * (H_idx * stride_height - padding_top + top_comp)  // move to row
                                + in.col_mem_stride * (W_idx * stride_width - padding_left + left_comp) // move to column
                                + ch_idx;                                                               // move to channel

                        // Core Max
                        v2q15_t max_val = reduce_max2D_hwc_v<io_T>(
                            in_ptr, clmns, rows, in.ch, in.col_mem_stride, in.row_mem_stride, false);

                        // Write result
                        mli_prv_store_2_samples(&out.ptr[
                                out.row_mem_stride * H_idx
                                + out.col_mem_stride * W_idx
                                + ch_idx], max_val);
                    }
                }
            }
            if (in.ch & 1) {
                for (int H_idx = perc_areas[area_idx].row_beg; H_idx < perc_areas[area_idx].row_end; H_idx++) {
                    for (int W_idx = perc_areas[area_idx].clmn_beg; W_idx < perc_areas[area_idx].clmn_end; W_idx++) {
                        // Define area of input and filter for convolution
                        // *_comp - compensation values for valid area defining
                        int top_comp = -MIN((H_idx * stride_height) - padding_top, 0);
                        int left_comp = -MIN((W_idx * stride_width) - padding_left, 0);

                        int right_comp = -MIN(in.width - ((W_idx * stride_width) - padding_left + kernel_width), 0);
                        int bottom_comp = -MIN(in.height - ((H_idx * stride_height) - padding_top + kernel_height), 0);

                        int rows = kernel_height - top_comp - bottom_comp;
                        int clmns = kernel_width - right_comp - left_comp;

                        const MLI_PTR(io_T) in_ptr = in.ptr
                                + in.row_mem_stride * (H_idx * stride_height - padding_top + top_comp)  // move to row
                                + in.col_mem_stride * (W_idx * stride_width - padding_left + left_comp) // move to column
                                + in.ch -  1;                                                           // move to channel

                        // Core Max
                        io_T max_val = reduce_max2D_hwc<io_T>(
                            in_ptr, clmns, rows, in.col_mem_stride, in.row_mem_stride, false);

                        // Write results
                        out.ptr[out.row_mem_stride * H_idx
                                + out.col_mem_stride * W_idx
                                + out.ch - 1] = max_val;
                    }
                }
            }
        }
    }
}

#pragma code()

#endif //_MLI_KRN_MAXPOOL_HWC_DSP_H_