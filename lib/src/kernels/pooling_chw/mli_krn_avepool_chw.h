/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_AVEPOOL_CHW_H_
#define _MLI_KRN_AVEPOOL_CHW_H_

#include "mli_krn_reduce_sum2d.h"

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_private_types.h"
#include "mli_prv_dsp.h"

/******************************************************************************
 *
 * Version & platform description
 * Targets:
 *
 ******************************************************************************/

template <typename io_T>
static inline void __attribute__((always_inline)) avepool_chw_nopad(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
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

    const int kernel_size = kernel_width * kernel_height;
    int16_t mul = 0;
    int shift = 0;
    get_mul_shift_value(kernel_size, &mul, &shift);

    MLI_OUT_PTR(io_T) __restrict out_ptr = out.ptr
            + out.col_mem_stride * clmn_beg
            + out.row_mem_stride * row_beg;
    MLI_PTR(io_T) __restrict in_ptr = in.ptr
            + in.col_mem_stride * (clmn_beg * stride_width - padding_left)
            + in.row_mem_stride * (row_beg * stride_height - padding_top);
    const int delta_W = (clmn_end - clmn_beg);
    const int delta_H = (row_end - row_beg);

    for (int ch_idx = 0; ch_idx < in.ch; ch_idx++) {
        for (int j = 0; j < delta_H; j++) {
            for (int k = 0; k < delta_W; k++) {
                accum40_t accum_40 = fx_create_a40(0x0, 0x0);
                reduce_sum2D_chw(&accum_40, in_ptr, kernel_width, kernel_height, in.row_mem_stride, mul);
                // Write results
                mli_prv_shift_clip_and_store_output(out_ptr, &accum_40, shift);

                in_ptr += in.col_mem_stride * stride_width;
                out_ptr++;
            }  // W_idx
            in_ptr += in.row_mem_stride * stride_height - (in.col_mem_stride * stride_width * delta_W);
            out_ptr += out.row_mem_stride - delta_W;
        }  // H_idx
        in_ptr += in.ch_mem_stride - (in.row_mem_stride * stride_height * delta_H);
        out_ptr += out.ch_mem_stride - (out.row_mem_stride * delta_H);
    }  // ch_idx
}

template <typename io_T>
static inline void __attribute__((always_inline)) avepool_chw(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
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

    MLI_ASSERT(in.col_mem_stride == 1 && out.col_mem_stride == 1);

    MLI_OUT_PTR(io_T) __restrict out_ptr = out.ptr
            + out.col_mem_stride * clmn_beg
            + out.row_mem_stride * row_beg;

    for (int ch_idx = 0; ch_idx < in.ch; ch_idx++) {
        for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
            for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
                // Define area of input and filter for convolution
                // *_comp - compensation values for valid area defining
                int top_comp = MIN((H_idx * stride_height) - padding_top, 0);
                int left_comp = MIN((W_idx * stride_width) - padding_left, 0);

                int right_comp = MIN(in.width - ((W_idx * stride_width) - padding_left + kernel_width), 0);
                int bottom_comp = MIN(in.height - ((H_idx * stride_height) - padding_top + kernel_height), 0);

                int rows = kernel_height + top_comp + bottom_comp;
                int clmns = kernel_width + right_comp + left_comp;

                const int kernel_size = rows * clmns;
                int16_t mul = 0;
                int shift = 0;
                get_mul_shift_value(kernel_size, &mul, &shift);

                const MLI_PTR(io_T) __restrict in_ptr = in.ptr
                        + in.row_mem_stride * (H_idx * stride_height - padding_top - top_comp)  // move to row
                        + in.col_mem_stride * (W_idx * stride_width - padding_left - left_comp) // move to column
                        + in.ch_mem_stride * ch_idx;                                            // move to channels

                accum40_t accum_40 = fx_create_a40(0x0, 0x0);
                reduce_sum2D_chw(&accum_40, in_ptr, clmns, rows, in.row_mem_stride, mul);
                // Write results
                mli_prv_shift_clip_and_store_output(out_ptr, &accum_40, shift);

                out_ptr++;
            }  // W_idx
            out_ptr += out.row_mem_stride - (clmn_end - clmn_beg);
        }  // H_idx
        out_ptr += out.ch_mem_stride - out.row_mem_stride * (row_end - row_beg);
    }  // ch_idx
}

template <typename io_T>
static inline void __attribute__((always_inline)) avepool_chw_krnpad(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_left,
        const int padding_right,
        const int padding_bot) {
#if (_ARCVER >= 0x50)  // Then will choose branch for HS processors
    avepool_chw(
            row_beg, row_end, clmn_beg, clmn_end, in, out,
            kernel_height, kernel_width, stride_height, stride_width, padding_top, padding_left,
            padding_right, padding_bot);
#else
    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in.height >= kernel_height && in.width >= kernel_width) {
        const int row_beg = CEIL_DIV(padding_top, stride_height);
        const int row_end = out.height - CEIL_DIV(padding_bot, stride_height);
        const int clmn_beg = CEIL_DIV(padding_left, stride_width);
        const int clmn_end = out.width - CEIL_DIV(padding_right, stride_width);

        avepool_chw_nopad(
                row_beg, row_end, clmn_beg, clmn_end, in, out,
                kernel_height, kernel_width, stride_height, stride_width, padding_top, padding_left,
                padding_right, padding_bot);
    }
    // Phase 2: Process border part with more complex algorithm
    // (usually significantly smaller part of computations)
    //=======================================================================
    if (padding_top || padding_left || padding_bot || padding_right) {
        rect_t areas[4];
        uint32_t areas_num = 0;
        if (padding_top) {
            areas[areas_num].row_beg = 0;
            areas[areas_num].row_end = CEIL_DIV (padding_top, stride_height);
            areas[areas_num].clmn_beg = 0;
            areas[areas_num++].clmn_end = out.width;
        }
        if (padding_bot) {
            areas[areas_num].row_beg = out.height - CEIL_DIV (padding_bot, stride_height);
            areas[areas_num].row_end = out.height;
            areas[areas_num].clmn_beg = 0;
            areas[areas_num++].clmn_end = out.width;
        }
        if (padding_left) {
            areas[areas_num].row_beg = CEIL_DIV (padding_top, stride_height);
            areas[areas_num].row_end = out.height - CEIL_DIV (padding_bot, stride_height);
            areas[areas_num].clmn_beg = 0;
            areas[areas_num++].clmn_end = CEIL_DIV (padding_left, stride_width);
        }
        if (padding_right) {
            areas[areas_num].row_beg = CEIL_DIV (padding_top, stride_height);
            areas[areas_num].row_end = out.height - CEIL_DIV (padding_bot, stride_height);
            areas[areas_num].clmn_beg = out.width - CEIL_DIV (padding_right, stride_width);
            areas[areas_num++].clmn_end = out.width;
        }
        for (int i = 0; i < areas_num; i++) {
            avepool_chw(
                    areas[i].row_beg, areas[i].row_end, areas[i].clmn_beg, areas[i].clmn_end, in, out,
                    kernel_height, kernel_width, stride_height, stride_width, padding_top,
                    padding_left, padding_right, padding_bot);
        }
    }
#endif
}

#endif  //_MLI_KRN_AVEPOOL_CHW_H_
