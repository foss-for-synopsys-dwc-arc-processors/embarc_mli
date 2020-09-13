/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_MAXPOOL_CHW_H_
#define _MLI_KRN_MAXPOOL_CHW_H_

#ifdef DEBUG_MAXPOOL
#include <stdio.h>
#endif

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_private_types.h"
#include "mli_prv_dsp.h"
#include "mli_prv_load_store.h"

#define REDUCE_MAX2D_UNROLL_FACTOR_FOR_WIDTH 11
#define REDUCE_MAX2D_UNROLL_FACTOR_FOR_HEIGHT 11

#ifdef DEBUG_MAXPOOL
#define MAXPOOL_DBG_PRINT(ch_idx, H_idx, W_idx, max_val) \
    MLI_PRINTF("MLI_MAXPOOL: [%d, %d, %d] max_val = %d\n", ch_idx, H_idx, W_idx, (int)max_val)
#else
#define MAXPOOL_DBG_PRINT(ch_idx, H_idx, W_idx, max_val)
#endif

template <typename io_T>
static MLI_FORCE_INLINE io_T reduce_max2D (
        const MLI_PTR(io_T) __restrict in,
        const int width,
        const int height,
        const int in_row_step,
        const bool fixedsize) {
    q15_t cur_max;

    if (width == 2) {
        v2q15_t v2_cur_max = mli_prv_load_2_samples(in);
        in += in_row_step;
        __builtin_assume(height > 0);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang loop unroll(full)
        for (int row = 0; row < (height - 1); row++) {
            v2_cur_max = fx_max_v2q15(v2_cur_max, mli_prv_load_2_samples(in));
            in += in_row_step;
        }
        cur_max = MAX(v2_cur_max[1], v2_cur_max[0]);
    } else if (width == 1) {
        cur_max = in[0];
        in += in_row_step;
        __builtin_assume(height > 0);
#pragma clang loop unroll(full)
        for (int row = 0; row < (height - 1); row++) {
            cur_max = MAX(cur_max, in[0]);
            in += in_row_step;
        }
    } else if (height == 1) {
        const MLI_PTR(io_T) __restrict in_ptr = in;
        v2q15_t v2_cur_max = mli_prv_load_2_samples(in_ptr);
        in_ptr += 2;
        __builtin_assume(width > 2);
#pragma clang loop unroll(full)
        for (int i = 0; i < (width - 2) / 2; i++) {
            v2_cur_max = fx_max_v2q15(v2_cur_max, mli_prv_load_2_samples(in_ptr));
            in_ptr += 2;
        }
        cur_max = MAX(v2_cur_max[1], v2_cur_max[0]);
        if ((width - 2) & 1) {
            cur_max = MAX(cur_max, in_ptr[0]);
        }
    } else {
        v2q15_t v2_cur_max = {INT16_MIN, INT16_MIN};

        if (fixedsize && height <= REDUCE_MAX2D_UNROLL_FACTOR_FOR_HEIGHT && height <= REDUCE_MAX2D_UNROLL_FACTOR_FOR_WIDTH) {
            if (width & 1) {
                cur_max = INT16_MIN;
                __builtin_assume(height > 0);
#pragma clang loop unroll(full)
                for (int row = 0; row < height; row++) {
                    __builtin_assume(width / 2 > 0);
#pragma clang loop unroll(full)
                    for (int i = 0, clmn = 0; i < width / 2; i++, clmn += 2) {
                        v2_cur_max = fx_max_v2q15(v2_cur_max, mli_prv_load_2_samples(&in[clmn]));
                    }
                    cur_max = MAX(cur_max, in[width - 1]);
                    in += in_row_step;
                }
                cur_max = MAX(cur_max, MAX(v2_cur_max[1], v2_cur_max[0]));
            } else {
                __builtin_assume(height > 0);
#pragma clang loop unroll(full)
                for (int row = 0; row < height; row++) {
                    __builtin_assume(width / 2 > 0);
#pragma clang loop unroll(full)
                    for (int i = 0, clmn = 0; i < width / 2; i++, clmn += 2) {
                        v2_cur_max = fx_max_v2q15(v2_cur_max, mli_prv_load_2_samples(&in[clmn]));
                    }
                    in += in_row_step;
                }
                cur_max = MAX(v2_cur_max[1], v2_cur_max[0]);
            }
        } else {
            if (width & 1) {
                cur_max = INT16_MIN;
                __builtin_assume(height > 0);
                for (int row = 0; row < height; row++) {
                    __builtin_assume(width / 2 > 0);
                    for (int i = 0, clmn = 0; i < width / 2; i++, clmn += 2) {
                        v2_cur_max = fx_max_v2q15(v2_cur_max, mli_prv_load_2_samples(&in[clmn]));
                    }
                    cur_max = MAX(cur_max, in[width - 1]);
                    in += in_row_step;
                }
                cur_max = MAX(cur_max, MAX(v2_cur_max[1], v2_cur_max[0]));
            } else {
                __builtin_assume(height > 0);
                for (int row = 0; row < height; row++) {
                    __builtin_assume(width / 2 > 0);
                    for (int i = 0, clmn = 0; i < width / 2; i++, clmn += 2) {
                        v2_cur_max = fx_max_v2q15(v2_cur_max, mli_prv_load_2_samples(&in[clmn]));
                    }
                    in += in_row_step;
                }
                cur_max = MAX(v2_cur_max[1], v2_cur_max[0]);
            }
        }
    }
#pragma clang diagnostic pop
    return (io_T)cur_max;
}

template <typename io_T>
static MLI_FORCE_INLINE io_T reduce_max2D_small(
        const MLI_PTR(io_T) __restrict in,
        const int width,
        const int height,
        const int in_row_step,
		const int kernel_width) {
    q15_t cur_max = 0;

    if (kernel_width == 3 && width == 3) {
        v2q15_t v2_cur_max = mli_prv_load_2_samples(in);
        cur_max = in[2];
        in += in_row_step;
        __builtin_assume(height > 0);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang loop unroll(full)
        for (int row = 0; row < (height - 1); row++) {
            v2_cur_max = fx_max_v2q15(v2_cur_max, mli_prv_load_2_samples(in));
            cur_max = MAX(cur_max, in[2]);
            in += in_row_step;
        }
        cur_max = MAX(cur_max, MAX(v2_cur_max[1], v2_cur_max[0]));
    } else if (width == 2) {
        v2q15_t v2_cur_max = mli_prv_load_2_samples(in);
        in += in_row_step;
        __builtin_assume(height > 0);
#pragma clang loop unroll(full)
        for (int row = 0; row < (height - 1); row++) {
            v2_cur_max = fx_max_v2q15(v2_cur_max, mli_prv_load_2_samples(in));
            in += in_row_step;
        }
        cur_max = MAX(v2_cur_max[1], v2_cur_max[0]);
    } else if (kernel_width < 3 && width == 1) {
        cur_max = in[0];
        in += in_row_step;
        __builtin_assume(height > 0);
#pragma clang loop unroll(full)
        for (int row = 0; row < (height - 1); row++) {
            cur_max = MAX(cur_max, in[0]);
            in += in_row_step;
        }
    } else {
		MLI_ASSERT(0);
	}
#pragma clang diagnostic pop
    return (io_T)cur_max;
}

template <typename io_T>
static MLI_FORCE_INLINE void maxpool_chw_nopad(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int row_begin,
        const int row_end,
        const int clmn_begin,
        const int clmn_end,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_bot,
        const int padding_left,
        const int padding_right,
        const int fixed_padding) {

    MLI_ASSERT(in.col_mem_stride == 1 && out.col_mem_stride == 1);

    MLI_OUT_PTR(io_T) __restrict out_ptr = out.ptr
            + out.col_mem_stride * clmn_begin
            + out.row_mem_stride * row_begin;
    const MLI_PTR(io_T) __restrict in_ptr = in.ptr
            + in.col_mem_stride * (clmn_begin * stride_width - padding_left)
            + in.row_mem_stride * (row_begin * stride_height - padding_top);

    if (kernel_width < 4 && kernel_height <= REDUCE_MAX2D_UNROLL_FACTOR_FOR_HEIGHT) {
        __builtin_assume(in.ch > 0);
        for (int ch_idx = 0; ch_idx < in.ch; ch_idx++) {
            for (int j = 0; j < (row_end - row_begin); j++) {
LOOP_PIPELINE_ENABLE
LOOP_PIPELINE_ENABLE_BACKTRACKING
#pragma unroll 2
                for (int k = 0; k < (clmn_end - clmn_begin); k++) {
                    // Core Max
                    io_T max_val = reduce_max2D(in_ptr, kernel_width, kernel_height, in.row_mem_stride, true);

                    in_ptr += in.col_mem_stride * stride_width;
                    // Write results
                    *out_ptr++ = max_val;
                    MAXPOOL_DBG_PRINT(ch_idx, j + row_begin, k + clmn_begin, max_val);
                }
                in_ptr += in.row_mem_stride * stride_height - (in.col_mem_stride * stride_width * (clmn_end - clmn_begin));
                out_ptr += out.row_mem_stride - (clmn_end - clmn_begin);
            }
            in_ptr += in.ch_mem_stride - (in.row_mem_stride * stride_height * (row_end - row_begin));
            out_ptr += out.ch_mem_stride -  (out.row_mem_stride * (row_end - row_begin));
        }
    } else {
        __builtin_assume(in.ch > 0);
        for (int ch_idx = 0; ch_idx < in.ch; ch_idx++) {
            for (int j = 0; j < (row_end - row_begin); j++) {
                for (int k = 0; k < (clmn_end - clmn_begin); k++) {
                    // Core Max
                    io_T max_val = reduce_max2D(in_ptr, kernel_width, kernel_height, in.row_mem_stride, true);

                    in_ptr += in.col_mem_stride * stride_width;
                    // Write results
                    *out_ptr++ = max_val;
                    MAXPOOL_DBG_PRINT(ch_idx, j + row_begin, k + clmn_begin, max_val);
                }
                in_ptr += in.row_mem_stride * stride_height - (in.col_mem_stride * stride_width * (clmn_end - clmn_begin));
                out_ptr += out.row_mem_stride - (clmn_end - clmn_begin);
            }
            in_ptr += in.ch_mem_stride - (in.row_mem_stride * stride_height * (row_end - row_begin));
            out_ptr += out.ch_mem_stride -  (out.row_mem_stride * (row_end - row_begin));
        }
    }
}

template <typename io_T>
static MLI_FORCE_INLINE void maxpool_chw(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int row_begin,
        const int row_end,
        const int clmn_begin,
        const int clmn_end,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_bot,
        const int padding_left,
        const int padding_right,
        const int fixed_padding) {

    MLI_ASSERT(in.col_mem_stride == 1 && out.col_mem_stride == 1);

    MLI_OUT_PTR(io_T) __restrict out_ptr = out.ptr
            + out.col_mem_stride * clmn_begin
            + out.row_mem_stride * row_begin;

    for (int ch_idx = 0; ch_idx < in.ch; ch_idx++) {
        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                // Define area of input for maxpooling
                // *_comp - compensation values for valid area defining
                int top_comp = -MIN((int)(H_idx * stride_height) - padding_top, 0);
                int left_comp = -MIN((int)(W_idx * stride_width) - padding_left, 0);

                int right_comp = -MIN((int)in.width - ((int)(W_idx * stride_width) - padding_left + kernel_width), 0);
                int bottom_comp =
                        -MIN((int)in.height - ((int)(H_idx * stride_height) - padding_top + kernel_height), 0);

                if (fixed_padding) {
                    if (padding_left == 0) left_comp = 0;
                    if (padding_right == 0) right_comp = 0;
                    if (padding_top == 0) top_comp = 0;
                    if (padding_bot == 0) bottom_comp = 0;
                }

                int rows = kernel_height - top_comp - bottom_comp;
                int clmns = kernel_width - right_comp - left_comp;

                const MLI_PTR(io_T) __restrict in_ptr = in.ptr
                        + in.ch_mem_stride * ch_idx                                              // move to channels
                        + in.row_mem_stride * (H_idx * stride_height - padding_top + top_comp)   // move to row
                        + in.col_mem_stride * (W_idx * stride_width - padding_left + left_comp); // move to column

                // Core Max
                io_T max_val = reduce_max2D(in_ptr, clmns, rows, in.row_mem_stride, false);

                // Write result
                *out_ptr++ = max_val;
                MAXPOOL_DBG_PRINT(ch_idx, H_idx, W_idx, max_val);
            }
            out_ptr += out.row_mem_stride - (clmn_end - clmn_begin);
        }
        out_ptr += out.ch_mem_stride - out.row_mem_stride * (row_end - row_begin);
    }
}

template <typename io_T>
static MLI_FORCE_INLINE void maxpool_chw_small(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int row_begin,
        const int row_end,
        const int clmn_begin,
        const int clmn_end,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_bot,
        const int padding_left,
        const int padding_right,
        const int fixed_padding) {

    MLI_ASSERT(in.col_mem_stride == 1 && out.col_mem_stride == 1);

    MLI_ASSERT(kernel_width <= 3); // the code in this function assumes small kernel width.
    MLI_OUT_PTR(io_T) __restrict out_ptr = out.ptr
            + out.col_mem_stride * clmn_begin
            + out.row_mem_stride * row_begin;

    for (int ch_idx = 0; ch_idx < in.ch; ch_idx++) {
        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                // Define area of input for maxpooling
                // *_comp - compensation values for valid area defining
                int top_comp = -MIN((int)(H_idx * stride_height) - padding_top, 0);
                // because kernel width is max 3, left_comp and right_comp can be max 1
                int left_comp = (W_idx * stride_width < padding_left) ? 1 : 0;
                int right_comp = (in.width + padding_left - kernel_width < W_idx * stride_width) ? 1 : 0;
                int bottom_comp =
                        -MIN((int)in.height - ((int)(H_idx * stride_height) - padding_top + kernel_height), 0);

                if (fixed_padding) {
                    if (padding_left == 0) left_comp = 0;
                    if (padding_right == 0) right_comp = 0;
                    if (padding_top == 0) top_comp = 0;
                    if (padding_bot == 0) bottom_comp = 0;
                }

                int rows = kernel_height - top_comp - bottom_comp;
                int clmns = kernel_width - right_comp - left_comp;

                const MLI_PTR(io_T) __restrict in_ptr = in.ptr
                        + in.ch_mem_stride * ch_idx                                              // move to channels
                        + in.row_mem_stride * (H_idx * stride_height - padding_top + top_comp)   // move to row
                        + in.col_mem_stride * (W_idx * stride_width - padding_left + left_comp); // move to column

                // Core Max
                io_T max_val = reduce_max2D(in_ptr, clmns, rows, in.row_mem_stride, kernel_width);

                // Write result
                *out_ptr++ = max_val;
                MAXPOOL_DBG_PRINT(ch_idx, H_idx, W_idx, max_val);
            }
            out_ptr += out.row_mem_stride - (clmn_end - clmn_begin);
        }
        out_ptr += out.ch_mem_stride - out.row_mem_stride * (row_end - row_begin);
    }
}

template <typename io_T>
static MLI_FORCE_INLINE void maxpool_chw_pad(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_bot,
        const int padding_left,
        const int padding_right,
        const int fixed_padding) {
    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in.height >= kernel_height && in.width >= kernel_width) {
        const int row_beg = CEIL_DIV(padding_top, stride_height);
        const int row_end = out.height - CEIL_DIV(padding_bot, stride_height);
        const int clmn_beg = CEIL_DIV(padding_left, stride_width);
        const int clmn_end = out.width - CEIL_DIV(padding_right, stride_width);

        maxpool_chw_nopad(
                in, out, row_beg, row_end, clmn_beg, clmn_end,
                kernel_height, kernel_width, stride_height, stride_width, padding_top, padding_bot,
                padding_left, padding_right, fixed_padding);
    }
    // Phase 2: Process border part with more complex algorithm
    // (usually significantly smaller part of computations)
    //=======================================================================
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
        maxpool_chw(
                in, out, areas[i].row_beg, areas[i].row_end, areas[i].clmn_beg, areas[i].clmn_end,
                kernel_height, kernel_width, stride_height, stride_width, padding_top, padding_bot,
                padding_left, padding_right, fixed_padding);
    }
}

template <typename io_T>
static MLI_FORCE_INLINE void maxpool_chw_krnpad_small(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_bot,
        const int padding_left,
        const int padding_right,
        const int fixed_padding) {

    MLI_ASSERT(in.col_mem_stride == 1 && out.col_mem_stride == 1);

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in.height >= kernel_height && in.width >= kernel_width) {
        const int row_beg = CEIL_DIV(padding_top, stride_height);
        const int row_end = out.height - CEIL_DIV(padding_bot, stride_height);
        const int clmn_beg = CEIL_DIV(padding_left, stride_width);
        const int clmn_end = out.width - CEIL_DIV(padding_right, stride_width);

        maxpool_chw_nopad(
                in, out, row_beg, row_end, clmn_beg, clmn_end,
                kernel_height, kernel_width, stride_height, stride_width, padding_top, padding_bot,
                padding_left, padding_right, fixed_padding);
    }
    // Phase 2: Process border part with more complex algorithm
    // (usually significantly smaller part of computations)
    //=======================================================================
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
        maxpool_chw_small(
                in, out, areas[i].row_beg, areas[i].row_end, areas[i].clmn_beg, areas[i].clmn_end,
                kernel_height, kernel_width, stride_height, stride_width, padding_top, padding_bot,
                padding_left, padding_right, fixed_padding);
    }
}

#endif  //_MLI_KRN_MAXPOOL_CHW_H_
