/*
* Copyright 2019, Synopsys, Inc.
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
static inline io_T __attribute__((always_inline)) reduce_max2D (
        const MLI_PTR(io_T) __restrict in, 
        const int width, 
        const int height, 
        const int in_row_step) {
    q15_t cur_max;

    if (width == 2) {
        v2q15_t v2_cur_max = mli_prv_load_2_samples(in);
        in += in_row_step;
        __builtin_assume(height > 0);
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

        if (height <= REDUCE_MAX2D_UNROLL_FACTOR_FOR_HEIGHT && height <= REDUCE_MAX2D_UNROLL_FACTOR_FOR_WIDTH) {
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
    return (io_T)cur_max;
}

template <typename io_T>
static inline void __attribute__((always_inline)) maxpool_chw_nopad(
        const MLI_PTR(io_T) __restrict in_ftrs,
        MLI_OUT_PTR(io_T) __restrict out_ftrs,
        const int row_begin,
        const int row_end,
        const int clmn_begin,
        const int clmn_end,
        const int channels_num,
        const int in_width,
        const int in_height,
        const int out_width,
        const int out_height,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_bot,
        const int padding_left,
        const int padding_right,
        const int fixed_padding) {
    MLI_OUT_PTR(io_T) __restrict out_ptr = out_ftrs + row_begin * out_width + clmn_begin;
    const MLI_PTR(io_T) __restrict in_ptr =
            in_ftrs + (row_begin * stride_height - padding_top) * in_width + (clmn_begin * stride_width - padding_left);
    if (kernel_width < 4 && kernel_height <= REDUCE_MAX2D_UNROLL_FACTOR_FOR_HEIGHT) {
        __builtin_assume(channels_num > 0);
        for (int ch_idx = 0; ch_idx < channels_num; ch_idx++) {
            for (int j = 0; j < (row_end - row_begin); j++) {
#pragma unroll 2
                for (int k = 0; k < (clmn_end - clmn_begin); k++) {
                    // Core Max
                    io_T max_val = reduce_max2D(in_ptr, kernel_width, kernel_height, in_width);

                    in_ptr += stride_width;
                    // Write results
                    *out_ptr++ = max_val;
                    MAXPOOL_DBG_PRINT(ch_idx, j + row_begin, k + clmn_begin, max_val);
                }
                in_ptr += in_width * stride_height + (clmn_begin - clmn_end) * stride_width;
                out_ptr += out_width + clmn_begin - clmn_end;
            }
            in_ptr += in_width * (in_height + (row_begin - row_end) * stride_height);
            out_ptr += out_width * (out_height + row_begin - row_end);
        }
    } else {
        __builtin_assume(channels_num > 0);
        for (int ch_idx = 0; ch_idx < channels_num; ch_idx++) {
            for (int j = 0; j < (row_end - row_begin); j++) {
                for (int k = 0; k < (clmn_end - clmn_begin); k++) {
                    // Core Max
                    io_T max_val = reduce_max2D(in_ptr, kernel_width, kernel_height, in_width);

                    in_ptr += stride_width;
                    // Write results
                    *out_ptr++ = max_val;
                    MAXPOOL_DBG_PRINT(ch_idx, j + row_begin, k + clmn_begin, max_val);
                }
                in_ptr += in_width * stride_height + (clmn_begin - clmn_end) * stride_width;
                out_ptr += out_width + clmn_begin - clmn_end;
            }
            in_ptr += in_width * (in_height + (row_begin - row_end) * stride_height);
            out_ptr += out_width * (out_height + row_begin - row_end);
        }
    }
}

template <typename io_T>
static inline void __attribute__((always_inline)) maxpool_chw(
        const MLI_PTR(io_T) __restrict in_ftrs,
        MLI_OUT_PTR(io_T) __restrict out_ftrs,
        const int row_begin,
        const int row_end,
        const int clmn_begin,
        const int clmn_end,
        const int channels_num,
        const int in_width,
        const int in_height,
        const int out_width,
        const int out_height,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_bot,
        const int padding_left,
        const int padding_right,
        const int fixed_padding) {
    MLI_OUT_PTR(io_T) __restrict out_ptr = out_ftrs + row_begin * out_width + clmn_begin;

    for (int ch_idx = 0; ch_idx < channels_num; ch_idx++) {
        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                // Define area of input for maxpooling
                // *_comp - compensation values for valid area defining
                int top_comp = -MIN((int)(H_idx * stride_height) - padding_top, 0);
                int left_comp = -MIN((int)(W_idx * stride_width) - padding_left, 0);

                int right_comp = -MIN((int)in_width - ((int)(W_idx * stride_width) - padding_left + kernel_width), 0);
                int bottom_comp =
                        -MIN((int)in_height - ((int)(H_idx * stride_height) - padding_top + kernel_height), 0);

                if (fixed_padding) {
                    if (padding_left == 0) left_comp = 0;
                    if (padding_right == 0) right_comp = 0;
                    if (padding_top == 0) top_comp = 0;
                    if (padding_bot == 0) bottom_comp = 0;
                }

                int rows = kernel_height - top_comp - bottom_comp;
                int clmns = kernel_width - right_comp - left_comp;

                const MLI_PTR(io_T) __restrict in_ptr =
                        in_ftrs +                                                      // starting point
                        in_width * in_height * ch_idx +                                // move to channels
                        in_width * (H_idx * stride_height - padding_top + top_comp) +  // move to row
                        (W_idx * stride_width) - padding_left + left_comp;             // move to column

                // Core Max
                io_T max_val = reduce_max2D(in_ptr, clmns, rows, in_width);

                // Write result
                *out_ptr++ = max_val;
                MAXPOOL_DBG_PRINT(ch_idx, H_idx, W_idx, max_val);
            }
            out_ptr += out_width + clmn_begin - clmn_end;
        }
        out_ptr += out_width * (out_height + row_begin - row_end);
    }
}

template <typename io_T>
static inline void __attribute__((always_inline)) maxpool_chw_krnpad(
        const MLI_PTR(io_T) __restrict in_ftrs,
        MLI_OUT_PTR(io_T) __restrict out_ftrs,
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const int channels_num,
        const int in_width,
        const int in_height,
        const int out_width,
        const int out_height,
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
    if (in_height >= kernel_height && in_width >= kernel_width) {
        const int row_beg = CEIL_DIV(padding_top, stride_height);
        const int row_end = out_height - CEIL_DIV(padding_bot, stride_height);
        const int clmn_beg = CEIL_DIV(padding_left, stride_width);
        const int clmn_end = out_width - CEIL_DIV(padding_right, stride_width);

        maxpool_chw_nopad(
                in_ftrs, out_ftrs, row_beg, row_end, clmn_beg, clmn_end, channels_num, in_width, in_height, out_width,
                out_height, kernel_height, kernel_width, stride_height, stride_width, padding_top, padding_bot,
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
        areas[areas_num++].clmn_end = out_width;
    }
    if (padding_bot) {
        areas[areas_num].row_beg = out_height - CEIL_DIV (padding_bot, stride_height);
        areas[areas_num].row_end = out_height;
        areas[areas_num].clmn_beg = 0;
        areas[areas_num++].clmn_end = out_width;
    }
    if (padding_left) {
        areas[areas_num].row_beg = CEIL_DIV (padding_top, stride_height);
        areas[areas_num].row_end = out_height - CEIL_DIV (padding_bot, stride_height);
        areas[areas_num].clmn_beg = 0;
        areas[areas_num++].clmn_end = CEIL_DIV (padding_left, stride_width);
    }
    if (padding_right) {
        areas[areas_num].row_beg = CEIL_DIV (padding_top, stride_height);
        areas[areas_num].row_end = out_height - CEIL_DIV (padding_bot, stride_height);
        areas[areas_num].clmn_beg = out_width - CEIL_DIV (padding_right, stride_width);
        areas[areas_num++].clmn_end = out_width;
    }
    for (int i = 0; i < areas_num; i++) {
        maxpool_chw(
                in_ftrs, out_ftrs, areas[i].row_beg, areas[i].row_end, areas[i].clmn_beg, areas[i].clmn_end, channels_num, in_width, in_height, out_width,
                out_height, kernel_height, kernel_width, stride_height, stride_width, padding_top, padding_bot,
                padding_left, padding_right, fixed_padding);
    }
}

#endif  //_MLI_KRN_MAXPOOL_CHW_H_
