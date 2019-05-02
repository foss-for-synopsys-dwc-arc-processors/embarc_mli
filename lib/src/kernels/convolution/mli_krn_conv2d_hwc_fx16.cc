/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_prv_tensor.h"
#include "mli_prv_dsp.h"
#include "mli_math_macros.h"
#include "mli_private_types.h"
#include "mli_krn_dotprod_chw.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma Code(".mli_lib")
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

static inline void convolution_hwc_no_pad (
        const MLI_PTR (int16_t) __restrict in_ftrs,
        const MLI_PTR (int16_t) __restrict weights,
        const MLI_PTR (int16_t) __restrict biases,
        MLI_CONV_OUT_PTR (int16_t) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width,
        const int out_ch, const int out_width,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width, 
        const int padding_top, const int padding_left);

static inline void convolution_hwc_no_pad_unroll4 (
        const MLI_PTR (int16_t) __restrict in_ftrs,
        const MLI_PTR (int16_t) __restrict weights,
        const MLI_PTR (int16_t) __restrict biases,
        MLI_CONV_OUT_PTR (int16_t) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width,
        const int out_ch, const int out_width,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width, 
        const int padding_top, const int padding_left);

static void convolution_hwc (
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
        const int out_ch, const int out_width,
        const int kernel_height, const int kernel_width, 
        const int stride_height, const int stride_width, 
        const int padding_top, const int padding_left);

mli_status mli_krn_conv2d_hwc_fx16 (
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwc(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
            return ret;

    mli_prv_fx_init_dsp_ctrl();

    uint8_t stride_width = cfg->stride_width;
    uint8_t stride_height = cfg->stride_height;
    uint8_t padding_top = cfg->padding_top;
    uint8_t padding_bot = cfg->padding_bottom;
    uint8_t padding_left = cfg->padding_left;
    uint8_t padding_right = cfg->padding_right;

    mli_minmax_t val_limit;
    out->el_type = MLI_EL_FX_16;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max (&cfg->relu, out);

    MLI_PTR (int16_t) in_ftrs = (MLI_PTR (int16_t)) in->data;
    MLI_CONV_OUT_PTR (int16_t) out_ftrs = (MLI_CONV_OUT_PTR (int16_t)) out->data;
    MLI_PTR (int16_t) wt = (MLI_PTR (int16_t)) weights->data;
    MLI_PTR (int16_t) bs = (MLI_PTR (int16_t)) bias->data;

    int in_ch = (int) in->shape[2];
    int out_ch = (int) weights->shape[0];

    int kernel_height = (int) weights->shape[1];
    int kernel_width = (int) weights->shape[2];

    int in_height = (int) in->shape[0];
    int in_width = (int) in->shape[1];

    int out_width = (in_width + padding_left + padding_right - kernel_width + 1);
            out_width = (out_width % stride_width != 0) ? (out_width / stride_width + 1) : out_width / stride_width;

    int out_height = (in_height + padding_top + padding_bot - kernel_height + 1);
            out_height = (out_height % stride_height != 0) ? (out_height / stride_height + 1) : out_height / stride_height;

    uint8_t bias_shift = (in->el_params.fx.frac_bits + weights->el_params.fx.frac_bits) - bias->el_params.fx.frac_bits;
    uint8_t out_shift = (in->el_params.fx.frac_bits + weights->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in_height >= kernel_height && in_width >= kernel_width) {
        rect_t cent_area;
            cent_area.row_beg = (padding_top % stride_height != 0) ? (padding_top / stride_height + 1) 
                    : padding_top / stride_height;
            cent_area.row_end = out_height - ((padding_bot % stride_height != 0) ? (padding_bot / stride_height + 1) 
                    : padding_bot / stride_height);
            cent_area.clmn_beg = (padding_left % stride_width != 0) ? (padding_left / stride_width + 1) 
                    : padding_left / stride_width;
            cent_area.clmn_end = out_width - ((padding_right % stride_width != 0) ? (padding_right / stride_width + 1) 
                    : padding_right / stride_width);

            convolution_hwc_no_pad (in_ftrs, wt, bs, out_ftrs, &cent_area, bias_shift, out_shift,
                                    val_limit.min, val_limit.max, in_ch, in_width, out_ch, out_width,
                                    kernel_height, kernel_width, stride_height, stride_width, padding_top, padding_left);
    }
    // Phase 2: Process border part with more complex algorithm
    // (usually significantly smaller part of computations)
    //=======================================================================//
        if (padding_top || padding_left || padding_bot || padding_right) {
        rect_t perc_areas[4];
        int areas_num = 0;
        if (padding_top) {
            perc_areas[areas_num].row_beg = 0;
            perc_areas[areas_num].row_end = CEIL_DIV (padding_top, stride_height);
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = out_width;
        }
        if (padding_bot) {
            perc_areas[areas_num].row_beg = out_height - CEIL_DIV (padding_bot, stride_height);
            perc_areas[areas_num].row_end = out_height;
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = out_width;
        }
        if (padding_left) {
            perc_areas[areas_num].row_beg = CEIL_DIV (padding_top, stride_height);
            perc_areas[areas_num].row_end = out_height - CEIL_DIV (padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = CEIL_DIV (padding_left, stride_width);
        }
        if (padding_right) {
            perc_areas[areas_num].row_beg = CEIL_DIV (padding_top, stride_height);
            perc_areas[areas_num].row_end = out_height - CEIL_DIV (padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = out_width - CEIL_DIV (padding_right, stride_width);
            perc_areas[areas_num++].clmn_end = out_width;
        }
        // Iterating over perception areas and perform general convolution per each
        for (int area_idx = 0; area_idx < areas_num; area_idx++) {
            convolution_hwc (in_ftrs, wt, bs, out_ftrs, &perc_areas[area_idx], bias_shift, out_shift,
                    val_limit.min, val_limit.max, in_ch, in_width, in_height, out_ch, out_width,
                    kernel_height, kernel_width, stride_height, stride_width, padding_top, padding_left);
        }
    }
    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[0] = out_height;
    out->shape[1] = out_width;
    out->shape[2] = out_ch;

    return MLI_STATUS_OK;
}

mli_status mli_krn_conv2d_hwc_fx16_1x1_str1_nopad (
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwc(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    mli_prv_fx_init_dsp_ctrl();

    uint8_t stride_width = cfg->stride_width;
    uint8_t stride_height = cfg->stride_height;
    uint8_t padding_top = cfg->padding_top;
    uint8_t padding_bot = cfg->padding_bottom;
    uint8_t padding_left = cfg->padding_left;
    uint8_t padding_right = cfg->padding_right;

    MLI_ASSERT(stride_width == 1);
    MLI_ASSERT(stride_height == 1);
    MLI_ASSERT(padding_top == 0);
    MLI_ASSERT(padding_bot == 0);
    MLI_ASSERT(padding_left == 0);
    MLI_ASSERT(padding_right == 0);

    stride_width = 1;
    stride_height = 1;
    padding_top = 0;
    padding_bot = 0;
    padding_left = 0;
    padding_right = 0;

    mli_minmax_t val_limit;
    out->el_type = MLI_EL_FX_16;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max (&cfg->relu, out);

    MLI_PTR (int16_t) in_ftrs = (MLI_PTR (int16_t)) in->data;
    MLI_CONV_OUT_PTR (int16_t) out_ftrs = (MLI_CONV_OUT_PTR (int16_t)) out->data;
    MLI_PTR (int16_t) wt = (MLI_PTR (int16_t)) weights->data;
    MLI_PTR (int16_t) bs = (MLI_PTR (int16_t)) bias->data;

    int in_ch = (int) in->shape[2];
    int out_ch = (int) weights->shape[0];

    int kernel_height = (int) weights->shape[1];
    int kernel_width = (int) weights->shape[2];
    MLI_ASSERT(kernel_width == 1);
    MLI_ASSERT(kernel_height == 1);
    kernel_width = 1;
    kernel_height = 1;

    int in_height = (int) in->shape[0];
    int in_width = (int) in->shape[1];

    int out_width = (in_width + padding_left + padding_right - kernel_width + 1);
    int out_height = (in_height + padding_top + padding_bot - kernel_height + 1);

    uint8_t bias_shift = (in->el_params.fx.frac_bits + weights->el_params.fx.frac_bits) - bias->el_params.fx.frac_bits;
    uint8_t out_shift = (in->el_params.fx.frac_bits + weights->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;

    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    convolution_hwc_no_pad_unroll4 (in_ftrs, wt, bs, out_ftrs, &cent_area, bias_shift, out_shift,
            val_limit.min, val_limit.max, in_ch, in_width, out_ch, out_width,
            kernel_height, kernel_width, stride_height, stride_width, padding_top, padding_left);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[0] = out_height;
    out->shape[1] = out_width;
    out->shape[2] = out_ch;

    return MLI_STATUS_OK;
}

/******************************************************************************
 *
 * All platform specific implementations for functions
  *
 ******************************************************************************/
#if (ARC_PLATFORM == V2DSP) || \
	(ARC_PLATFORM == V2DSP_XY) || \
	(ARC_PLATFORM == V2DSP_WIDE)

/******************************************************************************
 *
 * Version & platform description
 * Targets:
 *
 ******************************************************************************/

static inline void convolution_hwc_no_pad (
        const MLI_PTR (int16_t) __restrict in_ftrs,
        const MLI_PTR (int16_t) __restrict weights,
        const MLI_PTR (int16_t) __restrict biases,
        MLI_CONV_OUT_PTR (int16_t) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width,
        const int out_ch, const int out_width,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width, 
        const int padding_top, const int padding_left) {
    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;

    for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
        const MLI_PTR (int16_t) in_ptr = in_ftrs +  // starting point
            in_ch * in_width * (H_idx * stride_height - padding_top) +  // move to row
            in_ch * (clmn_begin * stride_width - padding_left); // move to column
        int W_idx_inc = in_ch * stride_width;

        for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
            MLI_CONV_OUT_PTR (int16_t) o_ptr = out_ftrs + (H_idx * out_width + W_idx) * out_ch;

            const MLI_PTR (int16_t) w_ptr = weights;    // Start point
            int w_ptr_inc = in_ch * kernel_width * kernel_height;   // move to filter
            for (int out_ch_idx = 0; out_ch_idx < out_ch; out_ch_idx++) {

                auto conv_out = mli_prv_init_accu_with_bias (in_ftrs, biases[out_ch_idx], bias_shift);

                // Convolution core
                dotprod2D (in_ptr, w_ptr, kernel_width * in_ch, kernel_height, in_width * in_ch, kernel_width * in_ch, 
                        &conv_out);

                mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);
                o_ptr++;
                w_ptr += w_ptr_inc;
            }               // out_ch_idx
            in_ptr += W_idx_inc;
        }                   // W_idx
    }
}

static inline void convolution_hwc_no_pad_unroll4 (
        const MLI_PTR (int16_t) __restrict in_ftrs,
        const MLI_PTR (int16_t) __restrict weights,
        const MLI_PTR (int16_t) __restrict biases,
        MLI_CONV_OUT_PTR (int16_t) __restrict out_ftrs,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int in_ch, const int in_width,
        const int out_ch, const int out_width,
        const int kernel_height, const int kernel_width,
        const int stride_height, const int stride_width, 
        const int padding_top, const int padding_left) {
    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;

    for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
        const MLI_PTR (int16_t) in_ptr = in_ftrs +  // starting point
                in_ch * in_width * (H_idx * stride_height - padding_top) +  // move to row
                in_ch * (clmn_begin * stride_width - padding_left); // move to column
        int W_idx_inc = in_ch * stride_width;

        for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
            MLI_CONV_OUT_PTR (int16_t) o_ptr = out_ftrs + (H_idx * out_width + W_idx) * out_ch;

            const MLI_PTR (int16_t) w_ptr = weights;    // Start point
            int w_ptr_inc = in_ch * kernel_width * kernel_height;   // move to filter
            for (int out_ch_idx = 0; out_ch_idx < out_ch; out_ch_idx++) {

                auto conv_out = mli_prv_init_accu_with_bias (in_ptr, biases[out_ch_idx], bias_shift);

                int unroll4_mask = 0x3;
                int rounded_count = (kernel_width * in_ch) & ~unroll4_mask;
                int remainder_count = (kernel_width * in_ch) & unroll4_mask;

                // Convolution core
                dotprod2D_unroll4 (in_ptr, w_ptr, rounded_count, kernel_height, in_width * in_ch, kernel_width * in_ch, 
                        &conv_out);

                if (remainder_count) {
                    dotprod2D (in_ptr + rounded_count, w_ptr + rounded_count, remainder_count, kernel_height, 
                            in_width * in_ch, kernel_width * in_ch, &conv_out);
                }

                mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);
                o_ptr++;
                w_ptr += w_ptr_inc;
            }               // out_ch_idx
            in_ptr += W_idx_inc;
        }                   // W_idx
    }
}
static void convolution_hwc (
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
        const int out_ch, const int out_width,
        const int kernel_height, const int kernel_width, 
        const int stride_height, const int stride_width, 
        const int padding_top, const int padding_left) {

    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;

    for (int out_ch_idx = 0; out_ch_idx < out_ch; out_ch_idx++) {
        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                auto conv_out = mli_prv_init_accu_with_bias (in_ftrs, biases[out_ch_idx], bias_shift);

                // Define area of input and filter for convolution
                // *_comp - compensation values for valid area defining
                int top_comp = -MIN ((H_idx * stride_height) - padding_top, 0);
                int left_comp = -MIN ((W_idx * stride_width) - padding_left, 0);

                int right_comp = -MIN (in_width - ((W_idx * stride_width) 
                        - padding_left + kernel_width), 0);
                int bottom_comp = -MIN (in_height - ((H_idx * stride_height) 
                        - padding_top + kernel_height), 0);

                int rows = kernel_height - top_comp - bottom_comp;
                int clmns = kernel_width - right_comp - left_comp;

                const MLI_PTR (int16_t) in_ptr = in_ftrs +  // starting point
                        in_ch * in_width * (H_idx * stride_height - padding_top + top_comp) +   // move to row
                        in_ch * ((W_idx * stride_width) - padding_left + left_comp);    // move to column

                const MLI_PTR (int16_t) w_ptr = weights +   // Start point
                        out_ch_idx * in_ch * kernel_width * kernel_height + // move to filter
                        top_comp * kernel_width * in_ch +   // move to row
                        left_comp * in_ch;  // move to column

                // Convolution core
                dotprod2D (in_ptr, w_ptr, clmns * in_ch, rows, in_width * in_ch,
                         kernel_width * in_ch, &conv_out);

                MLI_CONV_OUT_PTR(int16_t) o_ptr = &out_ftrs[out_ch_idx + (H_idx * out_width + W_idx) * out_ch];
                mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);
            }
        }
    }
}

#else

#error "Target platform is undefined or defined incorrectly"

#endif
#pragma code()
#ifdef __cplusplus
}
#endif
