/*
* Copyright 2019, Synopsys, Inc.
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
#include "mli_krn_dotprod_chw.h"

#ifdef DEBUG_CONV2D
#define CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, out_val) \
                MLI_PRINTF("MLI_CONV2D: [%d, %d, %d] out_val = %d\n", out_ch_idx, H_idx, W_idx    , (int)out_val)
#define CONV2D_DBG_PRINT_EXTRA(out_ch_idx, H_idx, W_idx, out_val, rows, clms) \
                MLI_PRINTF("MLI_CONV2D: [%d, %d, %d] out_val = %d (rows %d clms %d)\n", out_ch_idx, H_idx, W_idx, (int)out_val, rows, clms)
#else
#define CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, out_val)
#define CONV2D_DBG_PRINT_EXTRA(out_ch_idx, H_idx, W_idx, out_val, rows, clms)
#endif

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
    int32_t conv_out = fx_asr_rnd_q31((int32_t) bias, -bias_shift);
    __v2i32_t conv_out_v = { conv_out, conv_out };

    for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++) {
        // Convolution core
        dotprod2D_v (in_ptr, w_ptr, clmns, rows, in_width, kernel_w, &conv_out_v);
        // move to next channel
        w_ptr += kernel_w * kernel_h;
        in_ptr += in_width * in_height;
    }

    mli_prv_clip_relu_store_output_v (o_ptr, conv_out_v, out_shift, val_min_limit, val_max_limit);
}

#ifdef __FXAPI__
static void __attribute__ ((always_inline)) convolution_v (
        const MLI_PTR (int16_t) __restrict in_ptr,
        const MLI_PTR (int16_t) __restrict w_ptr,
        MLI_CONV_OUT_PTR (int16_t) __restrict o_ptr,
        const int16_t bias,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_width, const int in_height,
        const int kernel_w, const int kernel_h,
        const int clmns, const int rows, const int in_ch) {
    v2q15_t bias_v = { bias, bias };

    v2accum40_t conv_out_v = fx_v2a40_mpy_nf_v2q15 (bias_v, (v2q15_t) 0x00010001);
    conv_out_v = fx_asr_v2a40_n (conv_out_v, -bias_shift);

    for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++) {
        // Convolution core
        dotprod2D_v (in_ptr, w_ptr, clmns, rows, in_width, kernel_w, &conv_out_v);
        // move to next channel
        w_ptr += kernel_w * kernel_h;
        in_ptr += in_width * in_height;
    }

    mli_prv_clip_relu_store_output_v (o_ptr, &conv_out_v, out_shift, val_min_limit, val_max_limit);
}
#endif //__FXAPI__

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

#ifdef __FXAPI__
static void convolution_chw (
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
                    const MLI_PTR (int16_t) in_ptr = in_ftrs +  // starting point
                        in_width * in_height * in_ch_idx +  // move to channels
                        in_width * (H_idx * stride_height - padding_top + top_comp) +   // move to row
                        (W_idx * stride_width) - padding_left + left_comp;  // move to column

                    const MLI_PTR (int16_t) w_ptr = weights +   // Start point
                        out_ch_idx * in_ch * kernel_width * kernel_height + // move to filter
                        in_ch_idx * kernel_width * kernel_height +  // move to channel
                        top_comp * kernel_width +   // move to row
                        left_comp;  // move to column

                    // Convolution core
                    dotprod2D (in_ptr, w_ptr, clmns, rows, in_width, kernel_width, &conv_out);
                }
                MLI_CONV_OUT_PTR(int16_t) o_ptr = &out_ftrs[out_ch_idx * out_width * out_height + H_idx * out_width + W_idx];
                mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);
                CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, *o_ptr);
            }
        }
    }
}
#endif //__FXAPI__

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
                int32_t conv_out = (biases[out_ch_idx] << bias_shift);
                __v2i32_t conv_out_v = { conv_out, conv_out };

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

                mli_prv_clip_relu_store_output_v (o_ptr, conv_out_v, out_shift, val_min_limit, val_max_limit);
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

#ifdef __FXAPI__
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

                mli_prv_clip_relu_store_output_v (o_ptr, &conv_out_v, out_shift, val_min_limit, val_max_limit);
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
#endif

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

    MLI_ASSERT(stride_width == 1);
    MLI_ASSERT(H_idx < out_height);

    /* for large kernel sizes use a loop for the first part of the run-in
     * for the rest of the run-in and for small kernel sizes, use the
     * unrolled part below.
     */
    if ((fixed_padding == 0) || ((fixed_padding == 1) && (pad_left > 2))) {
        for (int l_comp = left_comp; l_comp > 0; l_comp--) {
            convolution_odd_even (in_ptr, w_ptr + l_comp, o_ptr, biases[out_ch_idx], bias_shift,
                    out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, 
                    kernel_w - l_comp, rows, in_ch);
            CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
            W_idx++;
            o_ptr++;
            /* only output ptr is incremented here, input ptr is not incremented because the input
             * pixels for the border processing start at the same point as the first part of the center
             * processing.
             */
        }
    } else {
        /* the extra condition with pad_left is only added because it enables
         * the compiler to remove the complete code block when not needed.
         * pad_left is compiletime constant.
         */
        if ((pad_left > 1) && (left_comp > 1)) {
            int l_comp = 2;
            convolution (in_ptr, w_ptr + l_comp, o_ptr, biases[out_ch_idx], bias_shift,
                    out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, 
                    kernel_w - l_comp, rows, in_ch);
            CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
            W_idx++;
            o_ptr++;
            /* only output ptr is incremented here, input ptr is not incremented because the input
             * pixels for the border processing start at the same point as the first part of the center
             * processing.
             */
        }
        if ((pad_left > 0) && (left_comp > 0)) {
            int l_comp = 1;
            convolution (in_ptr, w_ptr + l_comp, o_ptr, biases[out_ch_idx], bias_shift,
                    out_shift, val_min_limit, val_max_limit, in_width, in_height, 
                    kernel_w, kernel_h, kernel_w - l_comp, rows, in_ch);
            CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
            W_idx++;
            o_ptr++;
            /* only output ptr is incremented here, input ptr is not incremented because the input
             * pixels for the border processing start at the same point as the first part of the center
             * processing.
             */
        }
    }
    /* this is the main loop without run-in and run-out effects */
    /* when stride is fixed to one, the vectorized convolution can be used.
     * This will calculate two output samples at once.
     * in this case an extra case is needed for odd widths.
     */
    for (W_idx = clmn_begin + left_comp; W_idx < clmn_end - right_comp - 1; W_idx += 2) {
        convolution_v (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, 
                kernel_w, rows, in_ch);
        CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
        CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx + 1, o_ptr[1]);
        o_ptr += 2;
        in_ptr += 2;
    }

    /* because the main loop is doing 2 pixels at a time, we need an exception for odd widths */
    if (_Rarely (((clmn_end - right_comp) - (clmn_begin + left_comp)) & 1)) {
        convolution (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift, out_shift, val_min_limit, val_max_limit, 
                in_width, in_height, kernel_w, kernel_h, kernel_w, rows, in_ch);
        CONV2D_DBG_PRINT(out_ch_idx, H_idx, W_idx, o_ptr[0]);
        W_idx++;
        o_ptr += 1;
        in_ptr += 1;
    }

    if ((fixed_padding == 0) || ((fixed_padding == 1) && (pad_right > 2))) {
        /* for large padding sizes and in case the padding is not fixed, we use a loop. */
        for (int r_comp = 1; r_comp <= right_comp; r_comp++) {
            convolution_odd_even (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                    out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, 
                    kernel_w - r_comp, rows, in_ch);
            CONV2D_DBG_PRINT_EXTRA(out_ch_idx, H_idx, W_idx, o_ptr[0], rows, kernel_w - r_comp);
            W_idx++;
            o_ptr++;
            in_ptr++;
        }

    } else {
        if ((pad_right > 0) && (right_comp > 0))
        {
            convolution (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                    out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, kernel_w - 1, rows, in_ch);
            CONV2D_DBG_PRINT_EXTRA(out_ch_idx, H_idx, W_idx, o_ptr[0], rows, kernel_w - 1);
            W_idx++;
            o_ptr += 1;
            in_ptr += 1;
        }

        if ((pad_right > 1) && (right_comp > 1)) {
            convolution (in_ptr, w_ptr, o_ptr, biases[out_ch_idx], bias_shift,
                    out_shift, val_min_limit, val_max_limit, in_width, in_height, kernel_w, kernel_h, kernel_w - 2, rows, in_ch);
            CONV2D_DBG_PRINT_EXTRA(out_ch_idx, H_idx, W_idx, o_ptr[0], rows, kernel_w - 2);
            W_idx++;
            o_ptr += 1;
            in_ptr += 1;
        }
    }

}

/* optimized function that can do both the borders and the main part
 * for multiple kernel sizes and padding sizes. */
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

        for (int H_idx = row_begin; H_idx < row_end; )
        {
            int input_kernel_start = H_idx * str_height - padding_top;
            int kernel_rows = kernel_height + MIN(input_kernel_start, 0) - MAX(input_kernel_start + kernel_height - in_height, 0);
            int lines = (H_idx >= top_border_size && H_idx < (out_height - bot_border_size))? MIN(out_height - bot_border_size, row_end) - H_idx : 1;
            int top_comp = -MIN((H_idx * str_height) - padding_top, 0);

            if (kernel_rows == kernel_height){
                for (int line = 0; line < lines; line++ ){
                    conv2d_row_str1(
                            in_ftrs, weights, biases, out_ftrs, bias_shift, out_shift, val_min_limit, val_max_limit,
                            in_ch_num, in_width, in_height,    out_ch_idx, out_width, out_height, in_ch_start_idx,
                            kernel_height, kernel_width, str_height, stride_width, padding_top, padding_left, padding_right,
                            H_idx, left_comp, right_comp, top_comp, clmn_begin, clmn_end,
                            kernel_height/*rows*/, fixed_padding);
                    H_idx++;
                }
            } else if ((fixed_padding == 1) && (kernel_height <= 5) && ((padding_top > 0) || (padding_bot > 0)) && (kernel_rows == kernel_height - 1)){
                {
                    conv2d_row_str1(
                            in_ftrs, weights, biases, out_ftrs, bias_shift, out_shift, val_min_limit, val_max_limit,
                            in_ch_num, in_width, in_height,    out_ch_idx, out_width, out_height, in_ch_start_idx,
                            kernel_height, kernel_width, str_height, stride_width, padding_top, padding_left, padding_right,
                            H_idx, left_comp, right_comp, top_comp, clmn_begin, clmn_end,
                            kernel_height - 1/*rows*/, fixed_padding);
                    H_idx++;
                }
            } else if ((fixed_padding == 1) && (kernel_height <= 5) && ((padding_top > 1) || (padding_bot > 1)) && (kernel_rows == kernel_height - 2)){
                {
                    conv2d_row_str1(
                            in_ftrs, weights, biases, out_ftrs, bias_shift, out_shift, val_min_limit, val_max_limit,
                            in_ch_num, in_width, in_height,    out_ch_idx, out_width, out_height, in_ch_start_idx,
                            kernel_height, kernel_width, str_height, stride_width, padding_top, padding_left, padding_right,
                            H_idx, left_comp, right_comp, top_comp, clmn_begin, clmn_end,
                            kernel_height - 2/*rows*/, fixed_padding);
                    H_idx++;
                }
            } else if ((fixed_padding == 0) || (padding_top > 2) || (padding_bot > 2)) {
                for (int line = 0; line < lines; line++ ){
                    conv2d_row_str1(
                            in_ftrs, weights, biases, out_ftrs, bias_shift, out_shift, val_min_limit, val_max_limit,
                            in_ch_num, in_width, in_height,    out_ch_idx, out_width, out_height, in_ch_start_idx,
                            kernel_height, kernel_width, str_height, stride_width, padding_top, padding_left, padding_right,
                            H_idx, left_comp, right_comp, top_comp, clmn_begin, clmn_end,
                            kernel_rows/*rows*/, fixed_padding);
                    H_idx++;
                }
            }
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

#endif // _MLI_KRN_CONV2D_CHW_H_
