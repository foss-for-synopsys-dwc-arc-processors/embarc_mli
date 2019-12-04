/*
* Copyright 2019, Synopsys, Inc.
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
        const MLI_PTR(io_T) __restrict in_ftrs,
        MLI_OUT_PTR(io_T) __restrict out_ftrs,
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
        const int padding_left,
        const int padding_right,
        const int padding_bot) {
    (void)padding_right;
    (void)padding_bot;

    const int kernel_size = kernel_width * kernel_height;
    int16_t mul = 0;
    int shift = 0;
    get_mul_shift_value(kernel_size, &mul, &shift);

    MLI_OUT_PTR(io_T) __restrict p_out_ftrs = out_ftrs + row_beg * out_width + clmn_beg;
    MLI_PTR(io_T) __restrict in_ptr = (MLI_PTR(io_T))in_ftrs + in_width * (row_beg * stride_height - padding_top) +
            (clmn_beg * stride_width - padding_left);
    const int delta_W = (clmn_end - clmn_beg);
    const int delta_H = (row_end - row_beg);

    for (int ch_idx = 0; ch_idx < channels_num; ch_idx++) {
        for (int j = 0; j < (row_end - row_beg); j++) {
            for (int k = 0; k < (clmn_end - clmn_beg); k++) {
                accum40_t accum_40 = fx_create_a40(0x0, 0x0);
                reduce_sum2D(&accum_40, in_ptr, kernel_width, kernel_height, in_width, mul);
                // Write results
                mli_prv_shift_clip_and_store_output(p_out_ftrs, &accum_40, shift);

                p_out_ftrs++;
                in_ptr += stride_width;
            }  // W_idx
            p_out_ftrs += out_width - delta_W;
            in_ptr += in_width * stride_height - (stride_width * delta_W);
        }  // H_idx
        p_out_ftrs += out_width * (out_height - delta_H);
        in_ptr += in_width * (in_height - stride_height * delta_H);
    }  // ch_idx
}

template <typename io_T>
static inline void __attribute__((always_inline)) avepool_chw_nopad_even(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const MLI_PTR(io_T) __restrict in_ftrs,
        MLI_OUT_PTR(io_T) __restrict out_ftrs,
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
        const int padding_left,
        const int padding_right,
        const int padding_bot) {
    (void)padding_right;
    (void)padding_bot;

    const int kernel_size = kernel_width * kernel_height;
    int16_t mul = 0;
    int shift = 0;
    get_mul_shift_value(kernel_size, &mul, &shift);

    MLI_OUT_PTR(io_T) __restrict p_out_ftrs = out_ftrs + row_beg * out_width + clmn_beg;
    MLI_PTR(io_T) __restrict in_ptr = (MLI_PTR(io_T))in_ftrs + in_width * (row_beg * stride_height - padding_top) +
            (clmn_beg * stride_width - padding_left);
    const int delta_W = (clmn_end - clmn_beg);
    const int delta_H = (row_end - row_beg);

    for (int ch_idx = 0; ch_idx < channels_num; ch_idx++) {
        for (int j = 0; j < (row_end - row_beg); j++) {
            for (int k = 0; k < (clmn_end - clmn_beg); k++) {
                // Core Sum
                accum40_t accum_40 = fx_create_a40(0x0, 0x0);
                reduce_sum2D_even(&accum_40, in_ptr, kernel_width, kernel_height, in_width, mul);
                // Write results
                mli_prv_shift_clip_and_store_output(p_out_ftrs, &accum_40, shift);

                p_out_ftrs++;
                in_ptr += stride_width;
            }  // W_idx
            p_out_ftrs += out_width - delta_W;
            in_ptr += in_width * stride_height - (stride_width * delta_W);
        }  // H_idx
        p_out_ftrs += out_width * (out_height - delta_H);
        in_ptr += in_width * (in_height - stride_height * delta_H);
    }  // ch_idx
}

template <typename io_T>
static inline void __attribute__((always_inline)) avepool_chw(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const MLI_PTR(io_T) __restrict in_ftrs,
        MLI_OUT_PTR(io_T) __restrict out_ftrs,
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
        const int padding_left,
        const int padding_right,
        const int padding_bot) {
    (void)padding_right;
    (void)padding_bot;

    MLI_OUT_PTR(io_T) __restrict out_ptr = out_ftrs + clmn_beg * out_width + clmn_beg;
    for (int ch_idx = 0; ch_idx < channels_num; ch_idx++) {
        for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
            MLI_OUT_PTR(io_T) __restrict p_out_ftrs = (out_ftrs + ch_idx * out_width * out_height + H_idx * out_width);
            for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
                // Define area of input and filter for convolution
                // *_comp - compensation values for valid area defining
                int top_comp = MIN((H_idx * stride_height) - padding_top, 0);
                int left_comp = MIN((W_idx * stride_width) - padding_left, 0);

                int right_comp = MIN(in_width - ((W_idx * stride_width) - padding_left + kernel_width), 0);
                int bottom_comp = MIN(in_height - ((H_idx * stride_height) - padding_top + kernel_height), 0);

                int rows = kernel_height + top_comp + bottom_comp;
                int clmns = kernel_width + right_comp + left_comp;

                const int kernel_size = rows * clmns;
                int16_t mul = 0;
                int shift = 0;
                get_mul_shift_value(kernel_size, &mul, &shift);

                const MLI_PTR(io_T) __restrict in_ptr =
                        in_ftrs +                                                      // starting point
                        in_width * in_height * ch_idx +                                // move to channels
                        in_width * (H_idx * stride_height - padding_top - top_comp) +  // move to row
                        (W_idx * stride_width) - padding_left - left_comp;             // move to column

                accum40_t accum_40 = fx_create_a40(0x0, 0x0);
                reduce_sum2D(&accum_40, in_ptr, clmns, rows, in_width, mul);
                // Write results
                mli_prv_shift_clip_and_store_output(&p_out_ftrs[W_idx], &accum_40, shift);

            }  // W_idx
            out_ptr += out_width + clmn_beg - clmn_end;
        }  // H_idx
        out_ptr += out_width * (out_height + clmn_beg - row_end);
    }  // ch_idx
}

template <typename io_T>
static inline void __attribute__((always_inline)) avepool_chw_k4x4_str1_nopad(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const MLI_PTR(io_T) __restrict in_ftrs,
        MLI_OUT_PTR(io_T) __restrict out_ftrs,
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
        const int padding_left,
        const int padding_right,
        const int padding_bot) {
    (void)padding_right;
    (void)padding_bot;

    MLI_ASSERT(stride_width == 1);
    MLI_ASSERT(stride_height == 1);
    MLI_ASSERT(kernel_width == 4);
    MLI_ASSERT(kernel_height == 4);

    MLI_OUT_PTR(io_T) __restrict p_out_ftrs = out_ftrs + row_beg * out_width + clmn_beg;
    MLI_PTR(io_T) __restrict in_ptr = (MLI_PTR(io_T))in_ftrs + in_width * (row_beg * stride_height - padding_top) +
           (clmn_beg * stride_width - padding_left);
    const int delta_W = (clmn_end - clmn_beg);
    const int delta_H = (row_end - row_beg);

    for (int ch_idx = 0; ch_idx < channels_num; ch_idx++) {
        for (int j = 0; j < (row_end - row_beg); j++) {
            for (int k = 0; k < (clmn_end - clmn_beg); k++) {
                accum40_t accum_40 = fx_create_a40(0x0, 0x0);
                reduce_sum2D_even(&accum_40, (const MLI_PTR(io_T))in_ptr, kernel_width, kernel_height, in_width, 1);
                mli_prv_shift_clip_and_store_output(p_out_ftrs, &accum_40, 4);

                p_out_ftrs++;
                in_ptr += stride_width;
            }  // W_idx
            p_out_ftrs += out_width - delta_W;
            in_ptr += in_width * stride_height - (stride_width * delta_W);
        }  // H_idx
        p_out_ftrs += out_width * (out_height - delta_H);
        in_ptr += in_width * (in_height - stride_height * delta_H);
    }  // ch_idx
}

template <typename io_T>
static inline void __attribute__((always_inline)) avepool_chw_nopad_k2x2(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const MLI_PTR(io_T) __restrict in_ftrs,
        MLI_OUT_PTR(io_T) __restrict out_ftrs,
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
        const int padding_left,
        const int padding_right,
        const int padding_bot) {
    (void)padding_right;
    (void)padding_bot;

    MLI_OUT_PTR(io_T) __restrict p_out_ftrs = out_ftrs + row_beg * out_width + clmn_beg;
    MLI_PTR(io_T) __restrict in_ptr = (MLI_PTR(io_T))in_ftrs + in_width * (row_beg * stride_height - padding_top) +
           (clmn_beg * stride_width - padding_left);
    const int delta_W = (clmn_end - clmn_beg);
    const int delta_H = (row_end - row_beg);

    MLI_ASSERT(kernel_width == 2);
    MLI_ASSERT(kernel_height == 2);

    for (int ch_idx = 0; ch_idx < channels_num; ch_idx++) {
        for (int j = 0; j < (row_end - row_beg); j++) {
LOOP_PIPELINE_ENABLE  
            for (int k = 0; k < (clmn_end - clmn_beg); k++) {
                // Core Sum

                accum40_t accum_40 = fx_create_a40(0x0, 0x0);
                reduce_sum2D_even(&accum_40, in_ptr, kernel_width, kernel_height, in_width, 1);
                mli_prv_shift_clip_and_store_output(p_out_ftrs, &accum_40, 2);

                p_out_ftrs++;
                in_ptr += stride_width;
            }  // W_idx
            p_out_ftrs += out_width - delta_W;
            in_ptr += in_width * stride_height - (stride_width * delta_W);
        }  // H_idx
        p_out_ftrs += out_width * (out_height - delta_H);
        in_ptr += in_width * (in_height - stride_height * delta_H);
    }  // ch_idx
}

template <typename io_T>
static inline void __attribute__((always_inline)) avepool_chw_nopad_k4_Nx2_N_even(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const MLI_PTR(io_T) __restrict in_ftrs,
        MLI_OUT_PTR(io_T) __restrict out_ftrs,
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
        const int padding_left,
        const int padding_right,
        const int padding_bot) {
    (void)padding_right;
    (void)padding_bot;

    const int kernel_size = kernel_height * kernel_width;
    int16_t mul = 0;
    int shift = 0;
    get_mul_shift_value(kernel_size, &mul, &shift);

    MLI_OUT_PTR(io_T) __restrict p_out_ftrs = out_ftrs + row_beg * out_width + clmn_beg;
    MLI_PTR(io_T) __restrict in_ptr = (MLI_PTR(io_T))in_ftrs + in_width * (row_beg * stride_height - padding_top) +
            (clmn_beg * stride_width - padding_left);
    const int delta_W = (clmn_end - clmn_beg);
    const int delta_H = (row_end - row_beg);

    for (int ch_idx = 0; ch_idx < channels_num; ch_idx++) {
        for (int j = 0; j < (row_end - row_beg); j++) {
            for (int k = 0; k < (clmn_end - clmn_beg); k++) {
                // Core Sum
                accum40_t accum_40 = fx_create_a40(0x0, 0x0);
                reduce_sum2D_even(&accum_40, in_ptr, kernel_width, kernel_height, in_width, mul);
                mli_prv_shift_clip_and_store_output(p_out_ftrs, &accum_40, shift);

                p_out_ftrs++;
                in_ptr += stride_width;
            }  // W_idx
            p_out_ftrs += out_width - delta_W;
            in_ptr += in_width * stride_height - (stride_width * delta_W);
        }  // H_idx
        p_out_ftrs += out_width * (out_height - delta_H);
        in_ptr += in_width * (in_height - stride_height * delta_H);
    }  // ch_idx
}

template <typename io_T>
static inline void __attribute__((always_inline)) avepool_chw_krnpad(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const MLI_PTR(io_T) __restrict in_ftrs,
        MLI_OUT_PTR(io_T) __restrict out_ftrs,
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
        const int padding_left,
        const int padding_right,
        const int padding_bot) {
#if (_ARCVER >= 0x50)  // Then will choose branch for HS processors
    avepool_chw(
            row_beg, row_end, clmn_beg, clmn_end, in_ftrs, out_ftrs, channels_num, in_width, in_height, out_width,
            out_height, kernel_height, kernel_width, stride_height, stride_width, padding_top, padding_left,
            padding_right, padding_bot);
#else
    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in_height >= kernel_height && in_width >= kernel_width) {
        const int row_beg = CEIL_DIV(padding_top, stride_height);
        const int row_end = out_height - CEIL_DIV(padding_bot, stride_height);
        const int clmn_beg = CEIL_DIV(padding_left, stride_width);
        const int clmn_end = out_width - CEIL_DIV(padding_right, stride_width);

        avepool_chw_nopad(
                row_beg, row_end, clmn_beg, clmn_end, in_ftrs, out_ftrs, channels_num, in_width, in_height, out_width,
                out_height, kernel_height, kernel_width, stride_height, stride_width, padding_top, padding_left,
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
            avepool_chw(
                    areas[i].row_beg, areas[i].row_end, areas[i].clmn_beg, areas[i].clmn_end, in_ftrs, out_ftrs, channels_num, in_width, in_height,
                    out_width, out_height, kernel_height, kernel_width, stride_height, stride_width, padding_top,
                    padding_left, padding_right, padding_bot);
        }
    }
#endif
}

template <typename io_T>
static inline void __attribute__((always_inline)) avepool_chw_krnpad_k4_Nx2_N_even(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const MLI_PTR(io_T) __restrict in_ftrs,
        MLI_OUT_PTR(io_T) __restrict out_ftrs,
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
        const int padding_left,
        const int padding_right,
        const int padding_bot) {
#if (_ARCVER >= 0x50)  // Then will choose branch for HS processors
    avepool_chw(
            row_beg, row_end, clmn_beg, clmn_end, in_ftrs, out_ftrs, channels_num, in_width, in_height, out_width,
            out_height, kernel_height, kernel_width, stride_height, stride_width, padding_top, padding_left,
            padding_right, padding_bot);
#else
    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in_height >= kernel_height && in_width >= kernel_width) {
        const int row_beg = CEIL_DIV(padding_top, stride_height);
        const int row_end = out_height - CEIL_DIV(padding_bot, stride_height);
        const int clmn_beg = CEIL_DIV(padding_left, stride_width);
        const int clmn_end = out_width - CEIL_DIV(padding_right, stride_width);

        avepool_chw_nopad_k4_Nx2_N_even(
                row_beg, row_end, clmn_beg, clmn_end, in_ftrs, out_ftrs, channels_num, in_width, in_height, out_width,
                out_height, kernel_height, kernel_width, stride_height, stride_width, padding_top, padding_left,
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
            avepool_chw(
                    areas[i].row_beg, areas[i].row_end, areas[i].clmn_beg, areas[i].clmn_end, in_ftrs, out_ftrs, channels_num, in_width, in_height,
                    out_width, out_height, kernel_height, kernel_width, stride_height, stride_width, padding_top,
                    padding_left, padding_right, padding_bot);
        }
    }
#endif
}

#endif  //_MLI_KRN_AVEPOOL_CHW_H_
