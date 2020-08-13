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

#include "mli_config.h"
#include "mli_math_macros.h"
#include "mli_private_types.h"
#include "mli_krn_reduce_max2d.h"

////////////////////////////////////////////////////////////////////////////////
// Setting up namespace
////////////////////////////////////////////////////////////////////////////////
//namespace mli { // TODO: callers of below functions expect global namespace
//namespace krn {

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
#define CHANNCEL_LANES 	_VDSP_NUM_8BIT_LANES
#elif !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
#define CHANNCEL_LANES 	2
#else
#define CHANNCEL_LANES 	1
#endif


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

	int remaining_chans = in.ch & (CHANNCEL_LANES - 1);

	MLI_OUT_PTR(io_T) out_ptr;

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in.height >= kernel_height && in.width >= kernel_width) {
        for (int ch_idx = 0; ch_idx < in.ch - remaining_chans; ch_idx += CHANNCEL_LANES) {
            for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    const MLI_PTR(io_T) in_ptr = in.ptr
                            + in.row_mem_stride * (H_idx * stride_height - padding_top) // move to row
                            + in.col_mem_stride * (W_idx * stride_width - padding_left) // move to column
                            + ch_idx;                                                   // move to channel

                    out_ptr = &out.ptr[out.row_mem_stride * H_idx +
			                           out.col_mem_stride * W_idx +
			                           ch_idx];

                    // Core Max
                    reduce_max2D_hwc_v(in_ptr, out_ptr, kernel_width, kernel_height, in.ch,
                        in.col_mem_stride, in.row_mem_stride, true);
                }
            }
        }
        if (remaining_chans) {
            for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    const MLI_PTR(io_T) in_ptr = in.ptr
                            + in.row_mem_stride * (H_idx * stride_height - padding_top) // move to row
                            + in.col_mem_stride * (W_idx * stride_width - padding_left) // move to column
                            + in.ch - (CHANNCEL_LANES - 1);                                                // move to channel

                    out_ptr = &out.ptr[out.row_mem_stride * H_idx +
									  out.col_mem_stride * W_idx +
									  out.ch - (CHANNCEL_LANES - 1)];

                    // Core Max
                    reduce_max2D_hwc(in_ptr, out_ptr, kernel_width, kernel_height, in.ch,
                        in.col_mem_stride, in.row_mem_stride, true);
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

	int remaining_chans = in.ch & (CHANNCEL_LANES - 1);

	MLI_OUT_PTR(io_T) out_ptr;

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    row_beg = CEIL_DIV(padding_top, stride_height);
    row_end = out.height - CEIL_DIV(padding_bot, stride_height);
    clmn_beg = CEIL_DIV(padding_left, stride_width);
    clmn_end = out.width - CEIL_DIV(padding_right, stride_width);

    if ((row_end - row_beg > 0) && (clmn_end - clmn_beg > 0)) {
        maxpool_hwc_nopad(
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
            for (int ch_idx = 0; ch_idx < in.ch - remaining_chans; ch_idx += CHANNCEL_LANES) {
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

                        out_ptr = &out.ptr[out.row_mem_stride * H_idx +
										   out.col_mem_stride * W_idx +
										   ch_idx];

                        // Core Max
                        reduce_max2D_hwc_v(in_ptr, out_ptr, clmns, rows, in.ch,
                        		in.col_mem_stride, in.row_mem_stride, false);
                    }
                }
            }
            if (remaining_chans) {
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
                                + in.ch -  (CHANNCEL_LANES - 1);                                        // move to channel

                        out_ptr = &out.ptr[out.row_mem_stride * H_idx +
										  out.col_mem_stride * W_idx +
										  out.ch - (CHANNCEL_LANES - 1)];

                        // Core Max
                        reduce_max2D_hwc(in_ptr, out_ptr, clmns, rows, in.ch,
                        		in.col_mem_stride, in.row_mem_stride, false);
                    }
                }
            }
        }
    }
}

//} // krn
//} // mli

#endif // _MLI_KRN_MAXPOOL_HWC_H_
