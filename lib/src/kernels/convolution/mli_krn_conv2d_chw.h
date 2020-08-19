/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_CONV2D_CHW_H_
#define _MLI_KRN_CONV2D_CHW_H_

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_prv_dsp.h"
#include "mli_krn_dotprod_deprecated.h"

#ifdef DEBUG_CONV2D
#define CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, out_val) \
                MLI_PRINTF("MLI_CONV2D: [%d, %d, %d] out_val = %d\n", out_ch_idx, H_idx, W_idx    , (int)out_val)
#define CONV2D_DBG_PRINT_EXTRA(out_ch_idx, H_idx, W_idx, out_val, rows, clms) \
                MLI_PRINTF("MLI_CONV2D: [%d, %d, %d] out_val = %d (rows %d clms %d)\n", out_ch_idx, H_idx, W_idx, (int)out_val, rows, clms)
#else
#define CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, out_val)
#define CONV2D_DBG_PRINT_EXTRA(out_ch_idx, H_idx, W_idx, out_val, rows, clms)
#endif

/* This define controls the manual loop unrolling for the padding loop.
 * only the values 1 and 2 are supported. value 2 gives better performance,
 * value 1 gives better codesize. */
#define VPAD_UNROLL 2

template < typename io_T, typename w_T >
static void convolution_chw_nopad (
        const MLI_PTR (io_T) __restrict in_ftrs,
        const MLI_PTR (w_T) __restrict weights,
        const MLI_PTR (w_T) __restrict biases,
        MLI_CONV_OUT_PTR (io_T) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_bot, 
        const int padding_left, const int padding_right, 
        const int fixed_padding, const int depthwise);

template < typename io_T, typename w_T >
static void convolution_chw (
        const MLI_PTR (io_T) __restrict in_ftrs,
        const MLI_PTR (w_T) __restrict weights,
        const MLI_PTR (w_T) __restrict biases,
        MLI_CONV_OUT_PTR (io_T) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_bot, 
        const int padding_left, const int padding_right, 
        const int fixed_padding, const int depthwise);

template < typename io_T, typename w_T >
static void conv2d_chw_nopad_k1x1_str1 (
        const MLI_PTR (io_T) __restrict in_ftrs,
        const MLI_PTR (w_T) __restrict weights,
        const MLI_PTR (w_T) __restrict biases,
        MLI_CONV_OUT_PTR (io_T) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_bot,
        const int padding_left, const int padding_right, 
        const int fixed_padding, const int depthwise);

/* function that can do both the borders and the main part */
template < typename io_T, typename w_T >
static void conv2d_chw_str1 (
        const MLI_PTR (io_T) __restrict in_ftrs,
        const MLI_PTR (w_T) __restrict weights,
        const MLI_PTR (w_T) __restrict biases,
        MLI_CONV_OUT_PTR (io_T) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_bot, 
        const int padding_left, const int padding_right, 
        const int fixed_padding, const int depthwise);

/******************************************************************************/

template < typename io_T, typename w_T >
static void __attribute__ ((always_inline)) convolution (
        const MLI_PTR (io_T) __restrict in_ptr,
        const MLI_PTR (w_T) __restrict w_ptr,
        MLI_CONV_OUT_PTR (io_T) __restrict o_ptr,
        const w_T bias,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_width, const int in_height, 
        const int kernel_w, const int kernel_h, 
        const int clmns, const int rows, const int in_ch) {
    auto conv_out = mli_prv_init_accu_with_bias (in_ptr, bias, bias_shift);

    __builtin_assume(in_ch > 0);
    for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++) {
        // Convolution core
        dotprod2D (in_ptr, w_ptr, clmns, rows, in_width, kernel_w, &conv_out);

        // move to next channel
        w_ptr += kernel_w * kernel_h;
        in_ptr += in_width * in_height;
    }
    mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);
}

#ifdef __FXAPI__
static void __attribute__ ((always_inline)) convolution (
        const MLI_PTR (int16_t) __restrict in_ptr,
        const MLI_PTR (int16_t) __restrict w_ptr,
        MLI_CONV_OUT_PTR (int16_t) __restrict o_ptr,
        const int16_t bias,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_width,const int in_height, 
        const int kernel_w, const int kernel_h, 
        const int clmns, const int rows, const int in_ch) {
    auto conv_out = mli_prv_init_accu_with_bias (in_ptr, bias, bias_shift);

    __builtin_assume(in_ch > 0);
    for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++) {
        // Convolution core
        dotprod2D (in_ptr, w_ptr, clmns, rows, in_width, kernel_w, &conv_out);

        // move to next channel
        w_ptr += kernel_w * kernel_h;
        in_ptr += in_width * in_height;
    }

    int16_t out_val = fx_q15_cast_nf_asl_rnd_a40 (conv_out, 16 - out_shift);
    out_val = MIN (out_val, val_max_limit);
    out_val = MAX (out_val, val_min_limit);

    // Write result
    *o_ptr = out_val;
}
#endif //__FXAPI__

template < typename io_T, typename w_T >
static void __attribute__ ((always_inline)) convolution_even (
        const MLI_PTR (io_T) __restrict in_ptr,
        const MLI_PTR (w_T) __restrict w_ptr,
        MLI_CONV_OUT_PTR (io_T) __restrict o_ptr,
        const w_T bias,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_width, const int in_height, 
        const int kernel_w, const int kernel_h, 
        const int clmns, const int rows, const int in_ch) {
    auto conv_out = mli_prv_init_accu_with_bias (in_ptr, bias, bias_shift);

    for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++) {
        // Convolution core
        dotprod2D_unroll2 (in_ptr, w_ptr, clmns, rows, in_width, kernel_w, &conv_out);

        // move to next channel
        w_ptr += kernel_w * kernel_h;
        in_ptr += in_width * in_height;
    }

    mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);
}

template <typename io_T, typename w_T>
static void __attribute__((always_inline)) convolution_odd_even(
        const MLI_PTR(io_T) __restrict in_ptr,
        const MLI_PTR(w_T) __restrict w_ptr,
        MLI_CONV_OUT_PTR(io_T) __restrict o_ptr,
        const w_T bias,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_width, const int in_height,
        const int kernel_w, const int kernel_h,
        int clmns, const int rows, const int in_ch) {
    const MLI_PTR(io_T) __restrict in_ptr1 = in_ptr;
    const MLI_PTR(w_T) __restrict w_ptr1 = w_ptr;

    auto conv_out = mli_prv_init_accu_with_bias (in_ptr, bias, bias_shift);
    __builtin_assume(in_ch > 0);

    if (clmns & 1)
    {
        for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++)
        {
            // Convolution core
            dotprod2D_odd(
                    in_ptr1,
                    w_ptr1,
                    1,
                    rows,
                    in_width,
                    kernel_w,
                    &conv_out);

            // move to next channel
            w_ptr1 += kernel_w * kernel_h;
            in_ptr1 += in_width * in_height;
        }
        clmns--;
        in_ptr++;
        w_ptr++;
    }
    for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++)
    {
        // Convolution core
        dotprod2D_unroll2(
                in_ptr,
                w_ptr,
                clmns,
                rows,
                in_width,
                kernel_w,
                &conv_out);

        // move to next channel
        w_ptr += kernel_w * kernel_h;
        in_ptr += in_width * in_height;
    }

    mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);
}

template < typename io_T, typename w_T >
static void __attribute__ ((always_inline)) convolution_unroll4_plus1 (
        const MLI_PTR (io_T) __restrict in_ptr,
        const MLI_PTR (w_T) __restrict w_ptr,
        MLI_CONV_OUT_PTR (io_T) __restrict o_ptr,
        const w_T bias,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_width,
        const int in_height,
        const int kernel_w, const int kernel_h, 
        const int clmns, const int rows, const int in_ch) {
    auto conv_out = mli_prv_init_accu_with_bias (in_ptr, bias, bias_shift);

    for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++) {
        // Convolution core
        dotprod2D_unroll4_plus1 (in_ptr, w_ptr, clmns, rows, in_width, kernel_w, &conv_out);

        // move to next channel
        w_ptr += kernel_w * kernel_h;
        in_ptr += in_width * in_height;
    }

    mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);
}

template < typename io_T, typename w_T >
static void __attribute__ ((always_inline)) convolution_unroll4_plus2 (
        const MLI_PTR (io_T) __restrict in_ptr,
        const MLI_PTR (w_T) __restrict w_ptr,
        MLI_CONV_OUT_PTR (io_T) __restrict o_ptr,
        const w_T bias,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_width,
        const int in_height,
        const int kernel_w, const int kernel_h, 
        const int clmns, const int rows, const int in_ch) {
    auto conv_out = mli_prv_init_accu_with_bias (in_ptr, bias, bias_shift);

    for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++) {
        // Convolution core
        dotprod2D_unroll4_plus2 (in_ptr, w_ptr, clmns, rows, in_width, kernel_w, &conv_out);

        // move to next channel
        w_ptr += kernel_w * kernel_h;
        in_ptr += in_width * in_height;
    }

    mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);
}

template < typename io_T, typename w_T >
static void __attribute__ ((always_inline)) convolution_unroll4 (
        const MLI_PTR (io_T) __restrict in_ptr,
        const MLI_PTR (w_T) __restrict w_ptr,
        MLI_CONV_OUT_PTR (io_T) __restrict o_ptr,
        const w_T bias,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_width,
        const int in_height,
        const int kernel_w, const int kernel_h, 
        const int clmns, const int rows, const int in_ch) {
    auto conv_out = mli_prv_init_accu_with_bias (in_ptr, bias, bias_shift);

    for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++) {
        // Convolution core
        dotprod2D_mac4 (in_ptr, w_ptr, clmns, rows, in_width, kernel_w, &conv_out);

        // move to next channel
        w_ptr += kernel_w * kernel_h;
        in_ptr += in_width * in_height;
    }

    mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);
}

template < typename io_T, typename w_T >
static void __attribute__ ((always_inline)) convolution_unroll4_plus3 (
        const MLI_PTR (io_T) __restrict in_ptr,
        const MLI_PTR (w_T) __restrict w_ptr,
        MLI_CONV_OUT_PTR (io_T) __restrict o_ptr,
        const w_T bias,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_width, const int in_height,
        const int kernel_w, const int kernel_h, 
        const int clmns, const int rows, const int in_ch) {
    auto conv_out = mli_prv_init_accu_with_bias (in_ptr, bias, bias_shift);

    for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++) {
        // Convolution core
        dotprod2D_unroll4_plus3 (in_ptr, w_ptr, clmns, rows, in_width, kernel_w, &conv_out);

        // move to next channel
        w_ptr += kernel_w * kernel_h;
        in_ptr += in_width * in_height;
    }

    mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);
}

template < typename io_T, typename w_T >
static void __attribute__ ((always_inline)) convolution_v (
        const MLI_PTR (io_T) __restrict in_ptr,
        const MLI_PTR (w_T) __restrict w_ptr,
        MLI_CONV_OUT_PTR (io_T) __restrict o_ptr,
        const w_T bias,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_width, const int in_height, 
        const int kernel_w, const int kernel_h, 
        const int clmns, const int rows, const int in_ch) {
    auto conv_out_v = mli_prv_init_accu_with_bias_v(in_ptr, bias, bias_shift);

    __builtin_assume(in_ch > 0);
    for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++) {
        // Convolution core
        dotprod2D_v (in_ptr, w_ptr, clmns, rows, in_width, kernel_w, &conv_out_v);
        // move to next channel
        w_ptr += kernel_w * kernel_h;
        in_ptr += in_width * in_height;
    }

    mli_prv_clip_relu_store_output_v (o_ptr, &conv_out_v, out_shift, val_min_limit, val_max_limit);
}


template < typename io_T, typename w_T > static void
convolution_chw_nopad (
        const MLI_PTR (io_T) __restrict in_ftrs,
        const MLI_PTR (w_T) __restrict weights,
        const MLI_PTR (w_T) __restrict biases,
        MLI_CONV_OUT_PTR (io_T) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_bot, 
        const int padding_left, const int padding_right, 
        const int fixed_padding, const int depthwise) {
    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;

    for (int out_ch_idx = 0; out_ch_idx < out_ch; out_ch_idx++) {
        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                auto conv_out = mli_prv_init_accu_with_bias (in_ftrs, biases[out_ch_idx], bias_shift);

                for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++) {
                    // Define area of input and filter for convolution
                    const MLI_PTR (io_T) in_ptr = in_ftrs + // starting point
                        in_width * in_height * in_ch_idx +  // move to channels
                        in_width * (H_idx * stride_height - padding_top) +  // move to row
                        (W_idx * stride_width - padding_left);  // move to column

                    const MLI_PTR (w_T) w_ptr = weights +   // Start point
                        out_ch_idx * in_ch * kernel_width * kernel_height + // move to filter
                        in_ch_idx * kernel_width * kernel_height;   // move to channel

                    // Convolution core
                    dotprod2D (in_ptr, w_ptr, kernel_width, kernel_height, in_width, kernel_width, &conv_out);
                }

                MLI_CONV_OUT_PTR(io_T) o_ptr = &out_ftrs[out_ch_idx * out_width * out_height + H_idx * out_width + W_idx];
                mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);
                CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, *o_ptr);
            }
        }
    }
}

template < typename io_T, typename w_T > 
static void convolution_chw (
        const MLI_PTR (io_T) __restrict in_ftrs,
        const MLI_PTR (w_T) __restrict weights,
        const MLI_PTR (w_T) __restrict biases,
        MLI_CONV_OUT_PTR (io_T) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_bot, 
        const int padding_left, const int padding_right, 
        const int fixed_padding, const int depthwise) {
    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;

    for (int out_ch_idx = 0; out_ch_idx < out_ch; out_ch_idx++) {
        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                // Define area of input and filter for convolution
                // *_comp - compensation values for valid area defining
                int32_t top_comp = -MIN ((int32_t) (H_idx * stride_height) - padding_top, 0);
                int32_t left_comp = -MIN ((int32_t) (W_idx * stride_width) - padding_left, 0);

                int32_t right_comp = -MIN ((int32_t) in_width - ((int32_t) (W_idx * stride_width) - padding_left + kernel_width), 0);
                int32_t bottom_comp = -MIN ((int32_t) in_height - ((int32_t) (H_idx * stride_height) - padding_top + kernel_height), 0);

                int32_t rows = kernel_height - top_comp - bottom_comp;
                int32_t clmns = kernel_width - right_comp - left_comp;

                auto conv_out = mli_prv_init_accu_with_bias (in_ftrs, biases[out_ch_idx], bias_shift);

                for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++) {
                    const MLI_PTR (io_T) in_ptr = in_ftrs + // starting point
                        in_width * in_height * in_ch_idx +  // move to channels
                        in_width * (H_idx * stride_height - padding_top + top_comp) +   // move to row
                        (W_idx * stride_width) - padding_left + left_comp;  // move to column

                    const MLI_PTR (w_T) w_ptr = weights +   // Start point
                        out_ch_idx * in_ch * kernel_width * kernel_height + // move to filter
                        in_ch_idx * kernel_width * kernel_height +  // move to channel
                        top_comp * kernel_width +   // move to row
                        left_comp;  // move to column

                    // Convolution core
                    dotprod2D (in_ptr, w_ptr, clmns, rows, in_width, kernel_width, &conv_out);
                }
                MLI_CONV_OUT_PTR(io_T) o_ptr = &out_ftrs[out_ch_idx * out_width * out_height + H_idx * out_width + W_idx];
                mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);
                CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, *o_ptr);
            }
        }
    }
}

template < typename io_T, typename w_T >
static inline void __attribute__ ((always_inline)) conv2d_chw_nopad_k1x1_str1 (
        const MLI_PTR (io_T) __restrict in_ftrs,
        const MLI_PTR (w_T) __restrict weights,
        const MLI_PTR (w_T) __restrict biases,
        MLI_CONV_OUT_PTR (io_T) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_bot,
        const int padding_left, const int padding_right, 
        const int fixed_padding, const int depthwise) {
    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;

    MLI_ASSERT(kernel_height == 1);
    MLI_ASSERT(kernel_width == 1);

    for (int out_ch_idx = 0; out_ch_idx < out_ch; out_ch_idx++) {
        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            int W_idx;
            MLI_CONV_OUT_PTR(io_T) __restrict o_ptr = out_ftrs +
                    out_ch_idx * out_width * out_height +
                    H_idx * out_width + clmn_begin;

            for (W_idx = clmn_begin; W_idx < clmn_end - 1; W_idx += 2) {
                auto conv_out_v = mli_prv_init_accu_with_bias_v(in_ftrs, biases[out_ch_idx], bias_shift);

#if !defined __Xxy
                const MLI_PTR (io_T) in_ptr;
                const MLI_PTR (w_T) w_ptr;
                int in_ch_idx;
                /* for platforms without AGU, this code gives better performance, to be investigated why */
                for (in_ch_idx = 0; in_ch_idx < in_ch - 1; in_ch_idx += 2) {
                    // Define area of input and filter for convolution
                    in_ptr = in_ftrs +  // starting point
                        in_width * in_height * in_ch_idx +  // move to channels
                        in_width * (H_idx * stride_height - padding_top) +  // move to row
                        (W_idx * stride_width - padding_left);  // move to column

                    w_ptr = weights +   // Start point
                        out_ch_idx * in_ch * kernel_width * kernel_height + // move to filter
                        in_ch_idx * kernel_width * kernel_height;   // move to channel

                    // Convolution core
                    dotprod1D_v_unroll2 (in_ptr, w_ptr, 2, in_width * in_height, &conv_out_v);

                }
#else
                int in_ch_idx = 0;
                const MLI_PTR (io_T) in_ptr = in_ftrs + // starting point
                    in_width * in_height * in_ch_idx +  // move to channels
                    in_width * (H_idx * stride_height - padding_top) +  // move to row
                    (W_idx * stride_width - padding_left);  // move to column

                const MLI_PTR (w_T) w_ptr = weights +   // Start point
                    out_ch_idx * in_ch * kernel_width * kernel_height + // move to filter
                    in_ch_idx * kernel_width * kernel_height;   // move to channel

                // Convolution core
                dotprod1D_v_unroll2 (in_ptr, w_ptr, in_ch & ~1, in_width * in_height, &conv_out_v);

                in_ch_idx = in_ch & ~1;
#endif
                if (in_ch & 1) {
                    in_ptr = in_ftrs +  // starting point
                            in_width * in_height * in_ch_idx +  // move to channels
                            in_width * (H_idx * stride_height - padding_top) +  // move to row
                            (W_idx * stride_width - padding_left);  // move to column

                    w_ptr = weights +   // Start point
                            out_ch_idx * in_ch * kernel_width * kernel_height + // move to filter
                            in_ch_idx * kernel_width * kernel_height;   // move to channel

                    // Convolution core
                    dotprod1D_v (in_ptr, w_ptr, 1, 1, &conv_out_v);

                }

                mli_prv_clip_relu_store_output_v (o_ptr, &conv_out_v, out_shift, val_min_limit, val_max_limit);
                CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
                CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx + 1, o_ptr[1]);
                o_ptr += 2;
            }
            /* because the main loop is doing 2 pixels at a time, we need an exception for odd widths */
            if (_Rarely ((clmn_end - clmn_begin) & 1)) {
                const MLI_PTR (io_T) in_ptr;
                const MLI_PTR (w_T) w_ptr;

                in_ptr = in_ftrs +  // starting point
                        in_width * (H_idx * stride_height - padding_top) +  // move to row
                        (W_idx * stride_width - padding_left);  // move to column

                w_ptr = weights +   // Start point
                        out_ch_idx * in_ch * kernel_width * kernel_height;  // move to filter

                convolution (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                        out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_width, kernel_height, kernel_width, kernel_height, in_ch);
            }
        }
    }
}

static inline void __attribute__ ((always_inline)) conv2d_chw_nopad_k1x1_str1 (
        const MLI_PTR (int16_t) __restrict in_ftrs,
        const MLI_PTR (int16_t) __restrict weights,
        const MLI_PTR (int16_t) __restrict biases,
        MLI_CONV_OUT_PTR (int16_t) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_bot,
        const int padding_left, const int padding_right, 
        const int fixed_padding, const int depthwise) {
    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;

    MLI_ASSERT(kernel_height == 1);
    MLI_ASSERT(kernel_width == 1);

    for (int out_ch_idx = 0; out_ch_idx < out_ch; out_ch_idx++) {
        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            int W_idx;
            MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr = out_ftrs +
                    out_ch_idx * out_width * out_height +
                    H_idx * out_width + clmn_begin;

            for (W_idx = clmn_begin; W_idx < clmn_end - 1; W_idx += 2) {
                v2q15_t bias_v = { biases[out_ch_idx], biases[out_ch_idx] };
                v2accum40_t conv_out_v = fx_v2a40_mpy_nf_v2q15 (bias_v, (v2q15_t) 0x00010001);
                conv_out_v = fx_asr_v2a40_n (conv_out_v, -bias_shift);

#if !defined __Xxy
                const MLI_PTR (int16_t) in_ptr;
                const MLI_PTR (int16_t) w_ptr;
                int in_ch_idx;
                /* for platforms without AGU, this code gives better performance, because the unroll2 can load 2 weights in one load */
                for (in_ch_idx = 0; in_ch_idx < in_ch - 1; in_ch_idx += 2) {
                    // Define area of input and filter for convolution
                    in_ptr = in_ftrs +  // starting point
                            in_width * in_height * in_ch_idx +  // move to channels
                            in_width * (H_idx * stride_height - padding_top) +  // move to row
                            (W_idx * stride_width - padding_left);  // move to column

                    w_ptr = weights +   // Start point
                            out_ch_idx * in_ch * kernel_width * kernel_height + // move to filter
                            in_ch_idx * kernel_width * kernel_height;   // move to channel

                    // Convolution core
                    dotprod1D_v_unroll2 (in_ptr, w_ptr, 2, in_width * in_height, &conv_out_v);
                }

                if (in_ch&1){
                    in_ptr = in_ftrs + // starting point
                            in_width * in_height * in_ch_idx + // move to channels
                            in_width * (H_idx * stride_height - padding_top) + // move to row
                            (W_idx * stride_width - padding_left);    // move to column

                    w_ptr = weights + // Start point
                            out_ch_idx * in_ch * kernel_width * kernel_height + // move to filter
                            in_ch_idx * kernel_width * kernel_height;           // move to channel

                    // Convolution core
                    dotprod1D_v(in_ptr, w_ptr, 1, 1, &conv_out_v);
                }
#else
                int in_ch_idx = 0;
                const MLI_PTR (int16_t) in_ptr = in_ftrs +  // starting point
                        in_width * in_height * in_ch_idx +  // move to channels
                        in_width * (H_idx * stride_height - padding_top) +  // move to row
                        (W_idx * stride_width - padding_left);  // move to column

                const MLI_PTR (int16_t) w_ptr = weights +   // Start point
                        out_ch_idx * in_ch * kernel_width * kernel_height + // move to filter
                        in_ch_idx * kernel_width * kernel_height;   // move to channel

                // Convolution core
                dotprod1D_v(in_ptr,
                        w_ptr,
                        in_ch,
                        in_width * in_height,
                        &conv_out_v );

#endif

                mli_prv_clip_relu_store_output_v(o_ptr, &conv_out_v, out_shift, val_min_limit, val_max_limit);
                CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
                CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx + 1, o_ptr[1]);
                o_ptr += 2;
            }
            /* because the main loop is doing 2 pixels at a time, we need an exception for odd widths */
            if (_Rarely ((clmn_end - clmn_begin) & 1)) {
                const MLI_PTR (int16_t) in_ptr;
                const MLI_PTR (int16_t) w_ptr;

                in_ptr = in_ftrs +  // starting point
                        in_width * (H_idx * stride_height - padding_top) +  // move to row
                        (W_idx * stride_width - padding_left);  // move to column

                w_ptr = weights +   // Start point
                        out_ch_idx * in_ch * kernel_width * kernel_height;  // move to filter

                convolution (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                        out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_width, kernel_height, kernel_width, kernel_height, in_ch);
            }
        }
    }
}

template < typename io_T, typename w_T >
static inline void __attribute__ ((always_inline)) conv2d_row_str1 (
        const MLI_PTR (io_T) __restrict in_ftrs,
        const MLI_PTR (w_T) __restrict weights,
        const MLI_PTR (w_T) __restrict biases,
        MLI_CONV_OUT_PTR (io_T) __restrict out_ftrs,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width, const int in_height,
        const int out_ch_idx, const int out_width, const int out_height,
        const int in_ch_start_idx,
        const int kernel_h, const int kernel_w,
        const int stride_height, const int stride_width,
        const int pad_top, const int pad_left, const int pad_right,
        const int H_idx,
        const int left_comp, const int right_comp, const int top_comp, 
        const int clmn_begin, const int clmn_end, 
        const int rows, const int fixed_padding) {
    /* difference between pad_left and left_comp:
     * pad_left is the part of the kernel_width that is left of the current pixel
     * this is a compile time constant in case the kernel size is also fixed.
     * left_comp is how many border pixels we are processing currently.
     * so if we process the complete row pad_left == left_comp
     * same holds for right, top and bottom.
     * when processing the top row, pad_top == top_comp, but with the other
     * rows this is not the case anymore.
     */
    int W_idx = 0;
    // Define area of input and filter for convolution
    const MLI_PTR (io_T) __restrict in_ptr = in_ftrs +  // starting point
            in_width * in_height * in_ch_start_idx +    // move to channels
            in_width * (H_idx * stride_height - pad_top + top_comp) +   // move to row
            (clmn_begin * stride_width - pad_left + left_comp); // move to column

    const MLI_PTR (w_T) __restrict w_ptr = weights +    // Start point
            out_ch_idx * in_ch * kernel_w * kernel_h +  // move to filter
            top_comp * kernel_w;    // move to row

    MLI_CONV_OUT_PTR (io_T) __restrict o_ptr = out_ftrs + out_ch_idx * out_width * out_height + H_idx * out_width + clmn_begin;
    int left_border_size = CEIL_DIV(pad_left, stride_width);
    int right_border_size = CEIL_DIV(pad_right, stride_width);

    MLI_ASSERT(stride_width == 1);
    MLI_ASSERT(H_idx < out_height);

    int cols = out_width - left_border_size - right_border_size;
    int cols_even = cols>>1;
    int odd = cols - cols_even*2;
    int l_comp = pad_left;
    int lr_comp = left_comp;

    /* for large padding sizes and in case the padding is not fixed, we use a loop. */
    bool use_padding_loop = ((fixed_padding == 0) || (pad_left > 2) || (pad_right > 2));
    bool use_padding2 = !use_padding_loop && ((pad_left > 1) || (pad_right > 1));
    bool use_padding1 = !use_padding_loop && ((pad_left > 0) || (pad_right > 0));

    for (int i = 0; i < 2; i++)
    {
        if (use_padding2 && (((left_comp > 1) && (i==0)) || ((right_comp > 1) && (i==1))) ){
            // border processing with kernelsize = kernel width - 2
            convolution (in_ptr, w_ptr + l_comp, o_ptr, biases[out_ch_idx], bias_shift,
                    out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, kernel_w - 2, rows, in_ch);
            CONV2D_DBG_PRINT_EXTRA(out_ch_idx, H_idx, W_idx, o_ptr[0], rows, kernel_w - 2);
            if (i == 0){
                // At the left border l_comp (used as w_ptr offset) needs to be decremented. on the right border it is always zero.
                l_comp--;
                W_idx += 1;
                o_ptr += 1;
            } else {
                // At the right border, the pointers need to be decremented because the order of point that will be computed
                // next sits left of the currently calculated point.
                in_ptr -= 1;
                W_idx -= 1;
                o_ptr -= 1;
            }
        }

        if (use_padding1 && (((left_comp > 0) && (i==0)) || ((right_comp > 0) && (i==1)))){
            // border processing with kernelsize = kernel width - 1
            convolution (in_ptr, w_ptr + l_comp, o_ptr, biases[out_ch_idx], bias_shift,
                    out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, kernel_w - 1, rows, in_ch);
            CONV2D_DBG_PRINT_EXTRA(out_ch_idx, H_idx, W_idx, o_ptr[0], rows, kernel_w - 1);
            W_idx++;
            o_ptr += 1;
            if (i == 0){
                // At the left border l_comp (used as w_ptr offset) needs to be decremented. on the right border it is always zero.
                l_comp--;
            }
        }
        /* for large padding sizes and in case the padding is not fixed, we use a loop. */
        if (use_padding_loop) {
            for (int loop = 0; loop < lr_comp; loop++) {
                int l_comp = left_comp - loop;
                int r_comp = loop + 1;
                int comp = (i == 0) ? l_comp : r_comp;
                int w_ptr_offset = (i == 0) ? l_comp : 0;
                convolution_odd_even (in_ptr, w_ptr + w_ptr_offset, o_ptr, biases[out_ch_idx], bias_shift,
                        out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h,
                        kernel_w - comp, rows, in_ch);
                CONV2D_DBG_PRINT_EXTRA(out_ch_idx, H_idx, W_idx, o_ptr[0], rows, kernel_w - comp);
                W_idx++;
                o_ptr++;
                /* when processing the left border, input ptr is not incremented because the input
                 * pixels for the border processing start at the same point as the first part of the center
                 * processing.
                 */
                 if (i == 1) in_ptr++;
            }
        }

        //if (i == 1) break;

#ifdef __Xdsp_wide
        if (( kernel_w >= 4 ) && (cols > 0)) {
            /* This is the main loop without run-in and run-out effects */
            /* This will calculate one output samples at once using qmac instruction in loops.      */
            if ( (kernel_w & 3) == 0) {
                 __builtin_assume(cols > 0);
                for (int col = 0; col < cols; col++) {
                    convolution_unroll4 (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                        out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, 
                        kernel_w, rows, in_ch);
                    CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
                     W_idx++;
                    o_ptr += 1;
                    in_ptr += 1;
                }
            } else if ( (kernel_w & 3) == 1) {
                 __builtin_assume(cols > 0);
                for (int col = 0; col < cols; col++) {
                    convolution_unroll4_plus1 (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                        out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, 
                        kernel_w, rows, in_ch);
                    CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
                     W_idx++;
                    o_ptr += 1;
                    in_ptr += 1;
                }
            } else if ( (kernel_w & 3) == 2) {
                 __builtin_assume(cols > 0);
                for (int col = 0; col < cols; col++) {
                    convolution_unroll4_plus2 (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                        out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, 
                        kernel_w, rows, in_ch);
                    CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
                     W_idx++;
                    o_ptr += 1;
                    in_ptr += 1;
                }
            } else if ( (kernel_w & 3) == 3) {
                 __builtin_assume(cols > 0);
                for (int col = 0; col < cols; col++) {
                    convolution_unroll4_plus3 (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                        out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, 
                        kernel_w, rows, in_ch);
                    CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
                     W_idx++;
                    o_ptr += 1;
                    in_ptr += 1;
                }
            }
        cols = 0;
        } else  
#endif
       {
           if (cols_even > 0) {
                /* This is the main loop without run-in and run-out effects */
                /* when stride is fixed to one, the vectorized convolution can be used.
                 * This will calculate two output samples at once.
                 * in this case an extra case is needed for odd widths.
                 */
                for (int col = 0; col < cols_even; col++) {
                    __builtin_assume(cols_even > 0);
                    convolution_v (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                            out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h,
                            kernel_w, rows, in_ch);
                    CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
                    CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx + 1, o_ptr[1]);
                    W_idx += 2;
                    o_ptr += 2;
                    in_ptr += 2;
                }
            }
    #if 1 // if disabled, odd sizes are not supported
            if (_Rarely(odd)) {
                //odd
                convolution (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                        out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h,
                        kernel_w, rows, in_ch);
                CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
                W_idx++;
                o_ptr++;
                in_ptr += 1;
            }
    #endif
       }
        if (use_padding2 && (right_comp > 1)){
            // extra increment because the border pixel with kernel_w - 2 is computed before kernel_w - 1
            W_idx += 1;
            o_ptr += 1;
            in_ptr += 1;
        }
        cols_even = 0;
        odd = 0;
        lr_comp = right_comp;
    }
}

/* optimized function that can do both the borders and the main part
 * for multiple kernel sizes and padding sizes. */
template < typename io_T, typename w_T >
static inline void __attribute__ ((always_inline)) conv2d_chw_str1_impl (
        const MLI_PTR (io_T) __restrict in_ftrs,
        const MLI_PTR (w_T) __restrict weights,
        const MLI_PTR (w_T) __restrict biases,
        MLI_CONV_OUT_PTR (io_T) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_bot,
        const int padding_left, const int padding_right, 
        const int fixed_padding, const int depthwise) {
    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;
    int left_comp = -MIN ((clmn_begin * stride_width) - padding_left, 0);
    int right_comp = -MIN (in_width - ((clmn_end * stride_width) - padding_left + kernel_width - 1), 0);
    int top_border_size = CEIL_DIV(padding_top, stride_height);
    int bot_border_size = CEIL_DIV(padding_bot, stride_height);

    MLI_ASSERT(stride_width == 1);
    MLI_ASSERT(stride_height == 1);
    int str_height = 1;

    for (int out_ch_idx = 0; out_ch_idx < out_ch; out_ch_idx++) {
        int in_ch_start_idx = depthwise ? out_ch_idx : 0;
        int in_ch_num = depthwise ? 1 : in_ch;

        int H_idx = row_begin;
        int tb_comp = -MIN((row_begin * str_height) - padding_top, 0);
        int top_comp = tb_comp;
        int lines = MIN(out_height - bot_border_size, row_end) - top_border_size;

        for (int i = 0; i < 2; i++)
        {
            if ((fixed_padding == 0) || (padding_top > VPAD_UNROLL) || (padding_bot > VPAD_UNROLL)) {
                for (int loop = 0; loop < tb_comp; loop++) {
                    int t_comp = tb_comp - loop;
                    int b_comp = loop + 1;
                    int comp = (i == 0) ? t_comp : b_comp;
                    top_comp = (i == 0) ? t_comp : 0;

                    conv2d_row_str1(
                            in_ftrs, weights, biases, out_ftrs, bias_shift, out_shift, val_min_limit, val_max_limit,
                            in_ch_num, in_width, in_height,    out_ch_idx, out_width, out_height, in_ch_start_idx,
                            kernel_height, kernel_width, str_height, stride_width, padding_top, padding_left, padding_right,
                            H_idx, left_comp, right_comp, top_comp, clmn_begin, clmn_end,
                            kernel_height - comp/*rows*/, fixed_padding);
                    H_idx++;

                }
            } else {
#if VPAD_UNROLL > 1
                if (((fixed_padding == 1) && ((padding_top > 1) || (padding_bot > 1))) && (tb_comp > 1)) {
                    if (i == 1) {
                        H_idx++;
                    }
                    conv2d_row_str1(
                            in_ftrs, weights, biases, out_ftrs, bias_shift, out_shift, val_min_limit, val_max_limit,
                            in_ch_num, in_width, in_height,    out_ch_idx, out_width, out_height, in_ch_start_idx,
                            kernel_height, kernel_width, str_height, stride_width, padding_top, padding_left, padding_right,
                            H_idx, left_comp, right_comp, top_comp, clmn_begin, clmn_end,
                            kernel_height - 2/*rows*/, fixed_padding);
                    if (i == 0) {
                        H_idx++;
                        top_comp--;
                    } else {
                        H_idx--;;
                    }
                }
#endif
                if (((fixed_padding == 1) && ((padding_top > 0) || (padding_bot > 0))) && (tb_comp > 0)) {
                    conv2d_row_str1(
                            in_ftrs, weights, biases, out_ftrs, bias_shift, out_shift, val_min_limit, val_max_limit,
                            in_ch_num, in_width, in_height,    out_ch_idx, out_width, out_height, in_ch_start_idx,
                            kernel_height, kernel_width, str_height, stride_width, padding_top, padding_left, padding_right,
                            H_idx, left_comp, right_comp, top_comp, clmn_begin, clmn_end,
                            kernel_height - 1/*rows*/, fixed_padding);
                    H_idx++;
                }
            }
            //if (i==1) break;

            for (int line = 0; line < lines; line++ ){
                conv2d_row_str1(
                        in_ftrs, weights, biases, out_ftrs, bias_shift, out_shift, val_min_limit, val_max_limit,
                        in_ch_num, in_width, in_height,    out_ch_idx, out_width, out_height, in_ch_start_idx,
                        kernel_height, kernel_width, str_height, stride_width, padding_top, padding_left, padding_right,
                        H_idx, left_comp, right_comp, 0/*top_comp*/, clmn_begin, clmn_end,
                        kernel_height/*rows*/, fixed_padding);
                H_idx++;
            }

            tb_comp = -MIN (in_height - ((row_end * stride_height) - 1 - padding_top + kernel_height), 0);
            top_comp = 0;
            lines = 0;
        }
    }
}

template < typename io_T, typename w_T >
static inline void __attribute__ ((always_inline)) conv2d_row_anystride (
        const MLI_PTR (io_T) __restrict in_ftrs,
        const MLI_PTR (w_T) __restrict weights,
        const MLI_PTR (w_T) __restrict biases,
        MLI_CONV_OUT_PTR (io_T) __restrict out_ftrs,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width, const int in_height,
        const int out_ch_idx, const int out_width, const int out_height,
        const int in_ch_start_idx,
        const int kernel_h, const int kernel_w,
        const int stride_height, const int stride_width,
        const int pad_top, const int pad_left, const int pad_right,
        const int H_idx,
        const int left_comp, const int right_comp,
        const int top_comp, const int clmn_begin, const int clmn_end, const int rows) {
    /* difference between pad_left and left_comp:
     * pad_left is the part of the kernel_width that is left of the current pixel
     * this is a compile time constant in case the kernel size is also fixed.
     * left_comp is how many border pixels we are processing currently.
     * so if we process the complete row pad_left == left_comp
     * same holds for right, top and bottom.
     * when processing the top row, pad_top == top_comp, but with the other
     * rows this is not the case anymore.
     */
    int W_idx = 0;
    int W_input_idx = clmn_begin * stride_width;
    int weights_offset = left_comp;
    // Define area of input and filter for convolution
    const MLI_PTR (io_T) __restrict in_ptr = in_ftrs +  // starting point
            in_width * in_height * in_ch_start_idx +    // move to channels
            in_width * (H_idx * stride_height - pad_top + top_comp) +   // move to row
            (clmn_begin * stride_width - pad_left + left_comp); // move to column

    const MLI_PTR (io_T) __restrict in_ptr_start = in_ftrs +    // starting point
            in_width * in_height * in_ch_start_idx +    // move to channels
            in_width * (H_idx * stride_height - pad_top + top_comp);    // move to row

    const MLI_PTR (w_T) __restrict w_ptr = weights +    // Start point
            out_ch_idx * in_ch * kernel_w * kernel_h +  // move to filter
            top_comp * kernel_w;    // move to row

    MLI_CONV_OUT_PTR(io_T) __restrict o_ptr = out_ftrs +
            out_ch_idx * out_width * out_height +
            H_idx * out_width + clmn_begin;

    MLI_ASSERT(H_idx < out_height);

    /* for large kernel sizes use a loop for the first part of the run-in
     * for the rest of the run-in and for small kernel sizes, use the
     * unrolled part below.
     */
    if (pad_left > 0) {
        for (; W_input_idx < left_comp; W_input_idx += stride_width) {
            MLI_EXTRA_ASSERT((in_ptr - in_ptr_start) == MAX (W_idx * stride_width - pad_left, 0));
            MLI_EXTRA_ASSERT(weights_offset == -MIN (W_idx * stride_width - pad_left, 0));

            convolution (in_ptr, w_ptr + weights_offset, o_ptr, biases[out_ch_idx], bias_shift,
                    out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, 
                    kernel_w - weights_offset, rows, in_ch);
            CONV2D_DBG_PRINT_EXTRA(out_ch_idx, H_idx, W_idx, o_ptr[0], rows, kernel_w - weights_offset);
            weights_offset = MAX (weights_offset - stride_width, 0);
            W_idx++;
            o_ptr++;
            /* Input ptr is not incremented because the input
             * pixels for the border processing start at the same point as the first part of the center
             * processing.
             */
        }
    }

    /* this is the main loop without run-in and run-out effects */
    {
        in_ptr += W_idx * stride_width - pad_left;
#ifdef __Xdsp_wide
        if ((kernel_w & 3) == 0) {
            for (; W_input_idx < clmn_end * stride_width - right_comp; W_input_idx += stride_width) {
                MLI_EXTRA_ASSERT((in_ptr - in_ptr_start) == MAX (W_idx * stride_width - pad_left, 0));
                MLI_EXTRA_ASSERT(W_input_idx + kernel_w - pad_left <= in_width);

                convolution_unroll4 (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                        out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, kernel_w, 
                        rows, in_ch);
                CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
                W_idx++;
                o_ptr++;
                in_ptr += stride_width;
            }

        } else 
#endif
        if ((kernel_w & 1) == 0) {
            for (; W_input_idx < clmn_end * stride_width - right_comp; W_input_idx += stride_width) {
                MLI_EXTRA_ASSERT((in_ptr - in_ptr_start) == MAX (W_idx * stride_width - pad_left, 0));
                MLI_EXTRA_ASSERT(W_input_idx + kernel_w - pad_left <= in_width);

                convolution_even (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                        out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, kernel_w, 
                        rows, in_ch);
                CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
                W_idx++;
                o_ptr++;
                in_ptr += stride_width;
            }

        } else if ((kernel_w & 3) == 3) {
            for (; W_input_idx < clmn_end * stride_width - right_comp; W_input_idx += stride_width) {
                MLI_EXTRA_ASSERT((in_ptr - in_ptr_start) == MAX (W_idx * stride_width - pad_left, 0));
                MLI_EXTRA_ASSERT(W_input_idx + kernel_w - pad_left <= in_width);

                convolution_unroll4_plus3 (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                        out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, kernel_w, 
                        rows, in_ch);
                CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
                W_idx++;
                o_ptr++;
                in_ptr += stride_width;
            }

        } else if ((kernel_w & 3) == 1) {
            for (; W_input_idx < clmn_end * stride_width - right_comp; W_input_idx += stride_width) {
                MLI_EXTRA_ASSERT((in_ptr - in_ptr_start) == MAX (W_idx * stride_width - pad_left, 0));
                MLI_EXTRA_ASSERT(W_input_idx + kernel_w - pad_left <= in_width);

                convolution_unroll4_plus1 (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                        out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, kernel_w, 
                        rows, in_ch);
                CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
                W_idx++;
                o_ptr++;
                in_ptr += stride_width;
            }

        } else {
            for (; W_input_idx < clmn_end * stride_width - right_comp; W_input_idx += stride_width) {
                MLI_EXTRA_ASSERT((in_ptr - in_ptr_start) == MAX (W_idx * stride_width - pad_left, 0));
                MLI_EXTRA_ASSERT(W_input_idx + kernel_w - pad_left <= in_width);

                convolution (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                        out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, kernel_w, 
                        rows, in_ch);
                CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);

                W_idx++;
                o_ptr++;
                in_ptr += stride_width;
            }
        }
    }

    if (pad_right > 0) {
        for (; W_input_idx < (clmn_end * stride_width); W_input_idx += stride_width) {
            int clmns = kernel_w - MAX (W_input_idx + kernel_w - pad_left - in_width, 0);
            MLI_EXTRA_ASSERT((in_ptr - in_ptr_start) == MAX (W_idx * stride_width - pad_left, 0));

            convolution (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                    out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, clmns, rows, in_ch);
            CONV2D_DBG_PRINT_EXTRA(out_ch_idx, H_idx, W_idx, o_ptr[0], rows, clmns);
            W_idx++;
            o_ptr++;
            in_ptr += stride_width;
        }
    }

}

/* optimized function that can do both the borders and the main part
 * for multiple kernel sizes and padding sizes. */
template < typename io_T, typename w_T >
static inline void __attribute__ ((always_inline)) conv2d_chw (
        const MLI_PTR (io_T) __restrict in_ftrs,
        const MLI_PTR (w_T) __restrict weights,
        const MLI_PTR (w_T) __restrict biases,
        MLI_CONV_OUT_PTR (io_T) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_bot,
        const int padding_left, const int padding_right, 
        const int fixed_padding, const int depthwise) {
    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;
    int top_comp = -MIN ((row_begin * stride_height) - padding_top, 0);
    int bottom_comp = -MIN (in_height - ((row_end * stride_height) - 1 - padding_top + kernel_height), 0);
    int left_comp = -MIN ((clmn_begin * stride_width) - padding_left, 0);
    int right_comp = -MIN (in_width - ((clmn_end * stride_width) - padding_left + kernel_width - 1), 0);

    for (int out_ch_idx = 0; out_ch_idx < out_ch; out_ch_idx++) {
        int H_idx = row_begin;
        int H_input_idx = row_begin * stride_height;
        int w_row_offset = top_comp;
        int in_ch_start_idx = depthwise ? out_ch_idx : 0;
        int in_ch_num = depthwise ? 1 : in_ch;

        if (padding_top > 0) {
            for (; H_input_idx < top_comp; H_input_idx += stride_height) {
                conv2d_row_anystride (
                        in_ftrs, weights, biases, out_ftrs,
                        bias_shift, out_shift,
                        val_min_limit, val_max_limit,
                        in_ch_num, in_width, in_height,
                        out_ch_idx, out_width, out_height,
                        in_ch_start_idx,
                        kernel_height, kernel_width,
                        stride_height, stride_width,
                        padding_top, padding_left, padding_right,
                        H_idx, left_comp, right_comp, w_row_offset, 
                        clmn_begin, clmn_end, kernel_height - w_row_offset /*rows */ );
                w_row_offset = MAX (w_row_offset - stride_height, 0);
                H_idx++;
            }
        }

        for (; H_input_idx < row_end * stride_height - bottom_comp; H_input_idx += stride_height) {
            conv2d_row_anystride (
                    in_ftrs, weights, biases, out_ftrs,
                    bias_shift, out_shift,
                    val_min_limit, val_max_limit,
                    in_ch_num, in_width, in_height,
                    out_ch_idx, out_width, out_height,
                    in_ch_start_idx,
                    kernel_height, kernel_width,
                    stride_height, stride_width, 
                    padding_top, padding_left, padding_right, H_idx, left_comp, right_comp, 0 /*top_comp */ ,
                    clmn_begin, clmn_end, kernel_height);

            MLI_EXTRA_ASSERT(H_input_idx + kernel_height - padding_top <= in_height);
            H_idx++;

        }
        if (padding_bot > 0) {
            for (; H_input_idx < (row_end * stride_height); H_input_idx += stride_height) {
                int rows = kernel_height - MAX (H_input_idx + kernel_height - padding_top - in_height, 0);

                conv2d_row_anystride (
                        in_ftrs, weights, biases, out_ftrs,
                        bias_shift, out_shift,
                        val_min_limit, val_max_limit,
                        in_ch_num, in_width, in_height,
                        out_ch_idx, out_width, out_height,
                        in_ch_start_idx,
                        kernel_height, kernel_width,
                        stride_height, stride_width, 
                        padding_top, padding_left, padding_right, H_idx, left_comp, right_comp, 0 /*top_comp */ ,
                        clmn_begin, clmn_end, rows);
                H_idx++;
            }
        }

    }
}

template < typename io_T, typename w_T >
static inline void __attribute__ ((always_inline)) conv2d_chw_str1 (
        const MLI_PTR (io_T) __restrict in_ftrs,
        const MLI_PTR (w_T) __restrict weights,
        const MLI_PTR (w_T) __restrict biases,
        MLI_CONV_OUT_PTR (io_T) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_bot,
        const int padding_left, const int padding_right,
        const int fixed_padding, const int depthwise) {

    conv2d_chw_str1_impl(
        in_ftrs, weights, biases, out_ftrs, perception_area,
        bias_shift, out_shift,
        val_min_limit, val_max_limit,
        in_ch, in_width, in_height,
        out_ch, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        fixed_padding, depthwise);
}

#if !defined __Xxy
/* For platforms without AGU, conv2d_chw gives better performance for 8bit,
 * because the _dmachbl and _dmachbm are used, and they have integrated
 * sign extention from 8 to 16bit.
 * For platforms with AGU, the sign extention is done by the AGU
 */
static inline void __attribute__ ((always_inline)) conv2d_chw_str1 (
        const MLI_PTR (int8_t) __restrict in_ftrs,
        const MLI_PTR (int8_t) __restrict weights,
        const MLI_PTR (int8_t) __restrict biases,
        MLI_CONV_OUT_PTR (int8_t) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width, const int in_height,
        const int out_ch, const int out_width, const int out_height,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_bot,
        const int padding_left, const int padding_right,
        const int fixed_padding, const int depthwise) {

    conv2d_chw(
        in_ftrs, weights, biases, out_ftrs, perception_area,
        bias_shift, out_shift,
        val_min_limit, val_max_limit,
        in_ch, in_width, in_height,
        out_ch, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        fixed_padding, depthwise);
}
#endif

#endif // _MLI_KRN_CONV2D_CHW_H_
