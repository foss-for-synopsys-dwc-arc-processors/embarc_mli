/*
* Copyright 2019-2020, Synopsys, Inc.
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
#include "mli_krn_dotprod.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma Code(".mli_lib")

static void convolution_hwc_no_pad (
        const tensor_private_t<MLI_PTR(int16_t)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(int16_t)> &w,
        const MLI_PTR (int16_t) __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(int16_t)> &out,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left);

static inline void convolution_hwc_no_pad_unroll4 (
        const tensor_private_t<MLI_PTR(int16_t)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(int16_t)> &w,
        const MLI_PTR (int16_t) __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(int16_t)> &out,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int stride_height, const int stride_width, 
        const int padding_top, const int padding_left);

static void convolution_hwc (
        const tensor_private_t<MLI_PTR(int16_t)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(int16_t)> &w,
        const MLI_PTR (int16_t) __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(int16_t)> &out,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int stride_height, const int stride_width, 
        const int padding_top, const int padding_left);

static mli_status mli_krn_conv2d_hwc_fx16_1x1_str1_nopad (
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        mli_tensor * out);

mli_status mli_krn_conv2d_hwc_fx16 (
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        mli_tensor * out) {

    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwc(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
            return ret;

    const auto in_prv = mli_prv_get_tensor_hwc<MLI_PTR(int16_t), MLI_PTR_IS_XY>(in);
    const auto w = mli_prv_get_conv2d_weights_tensor_nhwc<MLI_PTR(int16_t), MLI_PTR_IS_XY>(weights);
    __builtin_assume(in_prv.ch == w.in_ch);

    mli_prv_fx_init_dsp_ctrl();

    uint8_t stride_width = cfg->stride_width;
    uint8_t stride_height = cfg->stride_height;
    uint8_t padding_top = cfg->padding_top;
    uint8_t padding_bot = cfg->padding_bottom;
    uint8_t padding_left = cfg->padding_left;
    uint8_t padding_right = cfg->padding_right;

    if (w.kernel_height == 1 && w.kernel_width == 1 && cfg->stride_height == 1 && cfg->stride_width == 1 &&
        (padding_top == 0 & padding_bot == 0 & padding_left == 0 & padding_right == 0))
        return mli_krn_conv2d_hwc_fx16_1x1_str1_nopad(in, weights, bias, cfg, out);

    mli_minmax_t val_limit;
    out->el_type = MLI_EL_FX_16;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max (&cfg->relu, out);
    MLI_PTR (int16_t) bs = (MLI_PTR (int16_t)) bias->data;

    int out_width = CEIL_DIV(in_prv.width + padding_left + padding_right - w.kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_prv.height + padding_top + padding_bot - w.kernel_height + 1, stride_height);

    uint8_t bias_shift = (in->el_params.fx.frac_bits + weights->el_params.fx.frac_bits) - bias->el_params.fx.frac_bits;
    uint8_t out_shift = (in->el_params.fx.frac_bits + weights->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[0] = out_height;
    out->shape[1] = out_width;
    out->shape[2] = w.out_ch;

    const auto out_prv = mli_prv_get_tensor_hwc<MLI_CONV_OUT_PTR(int16_t), MLI_CONV_OUT_PTR_IS_XY>(out);

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in_prv.height >= w.kernel_height && in_prv.width >= w.kernel_width) {
        rect_t cent_area;
        cent_area.row_beg = CEIL_DIV(padding_top, stride_height);
        cent_area.row_end = out_height - CEIL_DIV(padding_bot, stride_height);
        cent_area.clmn_beg = CEIL_DIV(padding_left, stride_width);
        cent_area.clmn_end = out_width - CEIL_DIV(padding_right, stride_width);

        convolution_hwc_no_pad (in_prv, w, bs, out_prv, &cent_area, bias_shift, out_shift,
                val_limit.min, val_limit.max,
                stride_height, stride_width, padding_top, padding_left);
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
            convolution_hwc (in_prv, w, bs, out_prv, &perc_areas[area_idx], bias_shift, out_shift,
                    val_limit.min, val_limit.max,
                    stride_height, stride_width, padding_top, padding_left);
        }
    }

    return MLI_STATUS_OK;
}

static mli_status mli_krn_conv2d_hwc_fx16_1x1_str1_nopad (
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwc(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    mli_prv_fx_init_dsp_ctrl();

    auto in_prv = mli_prv_get_tensor_hwc<MLI_PTR(int16_t), MLI_PTR_IS_XY>(in);
    auto w = mli_prv_get_conv2d_weights_tensor_nhwc<MLI_PTR(int16_t), MLI_PTR_IS_XY>(weights);
    __builtin_assume(in_prv.ch == w.in_ch);

    uint8_t stride_width = cfg->stride_width;
    uint8_t stride_height = cfg->stride_height;
    uint8_t padding_top = cfg->padding_top;
    uint8_t padding_bot = cfg->padding_bottom;
    uint8_t padding_left = cfg->padding_left;
    uint8_t padding_right = cfg->padding_right;

    MLI_ASSERT(w.kernel_width == 1);
    MLI_ASSERT(w.kernel_height == 1);
    MLI_ASSERT(stride_width == 1);
    MLI_ASSERT(stride_height == 1);
    MLI_ASSERT(padding_top == 0);
    MLI_ASSERT(padding_bot == 0);
    MLI_ASSERT(padding_left == 0);
    MLI_ASSERT(padding_right == 0);

    w.kernel_width = 1;
    w.kernel_height = 1;
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

    MLI_PTR (int16_t) bs = (MLI_PTR (int16_t)) bias->data;

    int out_width = (in_prv.width + padding_left + padding_right - w.kernel_width + 1);
    int out_height = (in_prv.height + padding_top + padding_bot - w.kernel_height + 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[0] = out_height;
    out->shape[1] = out_width;
    out->shape[2] = w.out_ch;
    const auto out_prv = mli_prv_get_tensor_hwc<MLI_CONV_OUT_PTR(int16_t), MLI_CONV_OUT_PTR_IS_XY>(out);

    uint8_t bias_shift = (in->el_params.fx.frac_bits + weights->el_params.fx.frac_bits) - bias->el_params.fx.frac_bits;
    uint8_t out_shift = (in->el_params.fx.frac_bits + weights->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;

    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    convolution_hwc_no_pad_unroll4 (in_prv, w, bs, out_prv, &cent_area, bias_shift, out_shift,
            val_limit.min, val_limit.max,
            stride_height, stride_width, padding_top, padding_left);

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
__attribute__((always_inline))
static inline void convolution_hwc_no_pad (
        const tensor_private_t<MLI_PTR(int16_t)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(int16_t)> &w,
        const MLI_PTR (int16_t) __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(int16_t)> &out,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left) {
    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;
    const int out_ch_mem_stride = 1;

    for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
        const MLI_PTR (int16_t) in_ptr = in.ptr
                + in.row_mem_stride * (H_idx * stride_height - padding_top)       // move to row
                + in.col_mem_stride * (clmn_begin * stride_width - padding_left); // move to column

        MLI_CONV_OUT_PTR (int16_t) o_ptr = out.ptr
                + out.row_mem_stride  * H_idx
                + out.col_mem_stride * clmn_begin;

        for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
            const MLI_PTR (int16_t) w_ptr = w.ptr; // Start point

            for (int out_ch_idx = 0; out_ch_idx < w.out_ch; out_ch_idx++) {
                auto accu = mli_prv_init_accu_with_bias (in.ptr, biases[out_ch_idx], bias_shift);

                // Convolution core
                for (int in_ch_idx = 0; in_ch_idx < (in.ch - 1); in_ch_idx += 2) {
                    dotprod2D_hwc_d(in_ptr, w_ptr, &accu, w.kernel_width, w.kernel_height,
                            in.col_mem_stride, in.row_mem_stride,
                            w.col_mem_stride, w.row_mem_stride);
                    in_ptr += 2;
                    w_ptr += 2;
                }

                if (in.ch & 1) {
                    accu = dotprod2D(in_ptr, w_ptr, accu, w.kernel_width, w.kernel_height,
                            in.col_mem_stride, in.row_mem_stride,
                            w.col_mem_stride, w.row_mem_stride);
                    in_ptr += 1;
                    w_ptr += 1;
                }
                in_ptr -= in.ch;
                w_ptr -= in.ch;

                mli_prv_clip_relu_store_output (o_ptr, accu, out_shift, val_min_limit, val_max_limit);
                o_ptr += out_ch_mem_stride;
                w_ptr += w.out_ch_mem_stride;
            } // out_ch_idx
            o_ptr += out.col_mem_stride - w.out_ch * out_ch_mem_stride;
            in_ptr += in.col_mem_stride * stride_width;
        } // W_idx
    }
}

__attribute__((always_inline))
static inline void convolution_hwc_no_pad_unroll4 (
        const tensor_private_t<MLI_PTR(int16_t)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(int16_t)> &w,
        const MLI_PTR (int16_t) __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(int16_t)> &out,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int stride_height, const int stride_width,
        const int padding_top, const int padding_left) {
    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;
    const int out_ch_mem_stride = 1;

    for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
        const MLI_PTR (int16_t) in_ptr = in.ptr
                + in.row_mem_stride * (H_idx * stride_height - padding_top)       // move to row
                + in.col_mem_stride * (clmn_begin * stride_width - padding_left); // move to column

        for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
            MLI_CONV_OUT_PTR (int16_t) o_ptr = out.ptr
                    + out.row_mem_stride  * H_idx
                    + out.col_mem_stride * W_idx;

            const MLI_PTR (int16_t) w_ptr = w.ptr; // Start point

            for (int out_ch_idx = 0; out_ch_idx < w.out_ch; out_ch_idx++) {
                auto conv_out = mli_prv_init_accu_with_bias (in_ptr, biases[out_ch_idx], bias_shift);

                int unroll4_mask = 0x3;
                int rounded_count = (w.kernel_width * in.ch) & ~unroll4_mask;
                int remainder_count = (w.kernel_width * in.ch) & unroll4_mask;

                // Convolution core
                dotprod2D_unroll4 (in_ptr, w_ptr, rounded_count, w.kernel_height,
                        in.row_mem_stride, w.row_mem_stride, &conv_out);

                if (remainder_count) {
                    dotprod2D (in_ptr + rounded_count, w_ptr + rounded_count, remainder_count, w.kernel_height,
                            in.row_mem_stride, w.row_mem_stride, &conv_out);
                }

                mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);
                o_ptr += out_ch_mem_stride;
                w_ptr += w.out_ch_mem_stride;
            } // out_ch_idx
            in_ptr += in.col_mem_stride * stride_width;
        } // W_idx
    } // H_idx
}

__attribute__((noinline))
static void convolution_hwc (
        const tensor_private_t<MLI_PTR(int16_t)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(int16_t)> &w,
        const MLI_PTR (int16_t) __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(int16_t)> &out,
        const rect_t * const perception_area,
        const int bias_shift,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int stride_height, const int stride_width, 
        const int padding_top, const int padding_left) {

    const int row_begin = perception_area->row_beg;
    const int row_end = perception_area->row_end;
    const int clmn_begin = perception_area->clmn_beg;
    const int clmn_end = perception_area->clmn_end;

    for (int out_ch_idx = 0; out_ch_idx < w.out_ch; out_ch_idx++) {
        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                auto accu = mli_prv_init_accu_with_bias (in.ptr, biases[out_ch_idx], bias_shift);

                // Define area of input and filter for convolution
                // *_comp - compensation values for valid area defining
                int top_comp = -MIN ((H_idx * stride_height) - padding_top, 0);
                int left_comp = -MIN ((W_idx * stride_width) - padding_left, 0);

                int right_comp = -MIN (in.width - ((W_idx * stride_width)
                        - padding_left + w.kernel_width), 0);
                int bottom_comp = -MIN (in.height - ((H_idx * stride_height)
                        - padding_top + w.kernel_height), 0);

                int rows = w.kernel_height - top_comp - bottom_comp;
                int clmns = w.kernel_width - right_comp - left_comp;

                const MLI_PTR (int16_t) in_ptr = in.ptr
                        + in.col_mem_stride * (W_idx * stride_width - padding_left + left_comp) // move to column
                        + in.row_mem_stride * (H_idx * stride_height - padding_top + top_comp); // move to row

                const MLI_PTR (int16_t) w_ptr = w.ptr
                        + w.col_mem_stride * left_comp      // move to column
                        + w.row_mem_stride * top_comp       // move to row
                        + w.out_ch_mem_stride * out_ch_idx; // move to filter

                // Convolution core
                for (int in_ch_idx = 0; in_ch_idx < in.ch - 1; in_ch_idx += 2) {
                    dotprod2D_hwc_d (in_ptr, w_ptr, &accu, clmns, rows,
                            in.col_mem_stride, in.row_mem_stride, 
                            w.col_mem_stride, w.row_mem_stride);
                    in_ptr += 2;
                    w_ptr += 2;
                }

                if (in.ch & 1) {
                    accu = dotprod2D (in_ptr, w_ptr, accu, clmns, rows,
                            in.col_mem_stride, in.row_mem_stride,
                            w.col_mem_stride, w.row_mem_stride);
                    in_ptr += 1;
                    w_ptr += 1;
                }
                in_ptr -= in.ch;
                w_ptr -= in.ch;

                MLI_CONV_OUT_PTR(int16_t) o_ptr = out.ptr
                        + out.col_mem_stride * W_idx
                        + out.row_mem_stride * H_idx
                        + out_ch_idx;
                mli_prv_clip_relu_store_output (o_ptr, accu, out_shift, val_min_limit, val_max_limit);
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
