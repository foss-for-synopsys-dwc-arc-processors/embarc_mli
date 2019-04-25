/* This file is generated, do not edit!
 * edit following template files instead:
 * filetemplate.txt
 * mli_krn_depthwise_conv2d_func_body.txt
 */
/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_krn_conv2d_chw.h"

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_tensor.h"
#include "mli_private_types.h"



#ifdef __cplusplus
extern "C" {
#endif

#pragma Code(".mli_lib")

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k1x2_ch1_str1_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 1);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 1
    MLI_CHECK_AND_FIX(kernel_width, 1);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_height, 2);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k2x1_ch1_str1_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 1);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_width, 2);
#endif
#if 1
    MLI_CHECK_AND_FIX(kernel_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k2x2_ch1_str1_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 1);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 1);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_width, 2);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_height, 2);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k3x3_ch1_str1_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 1);
    MLI_CHECK_AND_FIX(padding_bot, 1);
    MLI_CHECK_AND_FIX(padding_left, 1);
    MLI_CHECK_AND_FIX(padding_right, 1);
#endif
#if 3
    MLI_CHECK_AND_FIX(kernel_width, 3);
#endif
#if 3
    MLI_CHECK_AND_FIX(kernel_height, 3);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k4x4_ch1_str1_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 1);
    MLI_CHECK_AND_FIX(padding_bot, 2);
    MLI_CHECK_AND_FIX(padding_left, 1);
    MLI_CHECK_AND_FIX(padding_right, 2);
#endif
#if 4
    MLI_CHECK_AND_FIX(kernel_width, 4);
#endif
#if 4
    MLI_CHECK_AND_FIX(kernel_height, 4);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k5x5_ch1_str1_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 2);
    MLI_CHECK_AND_FIX(padding_bot, 2);
    MLI_CHECK_AND_FIX(padding_left, 2);
    MLI_CHECK_AND_FIX(padding_right, 2);
#endif
#if 5
    MLI_CHECK_AND_FIX(kernel_width, 5);
#endif
#if 5
    MLI_CHECK_AND_FIX(kernel_height, 5);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k6x6_ch1_str1_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 2);
    MLI_CHECK_AND_FIX(padding_bot, 3);
    MLI_CHECK_AND_FIX(padding_left, 2);
    MLI_CHECK_AND_FIX(padding_right, 3);
#endif
#if 6
    MLI_CHECK_AND_FIX(kernel_width, 6);
#endif
#if 6
    MLI_CHECK_AND_FIX(kernel_height, 6);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k7x7_ch1_str1_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 3);
    MLI_CHECK_AND_FIX(padding_bot, 3);
    MLI_CHECK_AND_FIX(padding_left, 3);
    MLI_CHECK_AND_FIX(padding_right, 3);
#endif
#if 7
    MLI_CHECK_AND_FIX(kernel_width, 7);
#endif
#if 7
    MLI_CHECK_AND_FIX(kernel_height, 7);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k2x2_str1_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 1);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 1);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_width, 2);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_height, 2);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k3x3_str1_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 1);
    MLI_CHECK_AND_FIX(padding_bot, 1);
    MLI_CHECK_AND_FIX(padding_left, 1);
    MLI_CHECK_AND_FIX(padding_right, 1);
#endif
#if 3
    MLI_CHECK_AND_FIX(kernel_width, 3);
#endif
#if 3
    MLI_CHECK_AND_FIX(kernel_height, 3);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k4x4_str1_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 1);
    MLI_CHECK_AND_FIX(padding_bot, 2);
    MLI_CHECK_AND_FIX(padding_left, 1);
    MLI_CHECK_AND_FIX(padding_right, 2);
#endif
#if 4
    MLI_CHECK_AND_FIX(kernel_width, 4);
#endif
#if 4
    MLI_CHECK_AND_FIX(kernel_height, 4);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k5x5_str1_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 2);
    MLI_CHECK_AND_FIX(padding_bot, 2);
    MLI_CHECK_AND_FIX(padding_left, 2);
    MLI_CHECK_AND_FIX(padding_right, 2);
#endif
#if 5
    MLI_CHECK_AND_FIX(kernel_width, 5);
#endif
#if 5
    MLI_CHECK_AND_FIX(kernel_height, 5);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k6x6_str1_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 2);
    MLI_CHECK_AND_FIX(padding_bot, 3);
    MLI_CHECK_AND_FIX(padding_left, 2);
    MLI_CHECK_AND_FIX(padding_right, 3);
#endif
#if 6
    MLI_CHECK_AND_FIX(kernel_width, 6);
#endif
#if 6
    MLI_CHECK_AND_FIX(kernel_height, 6);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k7x7_str1_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 3);
    MLI_CHECK_AND_FIX(padding_bot, 3);
    MLI_CHECK_AND_FIX(padding_left, 3);
    MLI_CHECK_AND_FIX(padding_right, 3);
#endif
#if 7
    MLI_CHECK_AND_FIX(kernel_width, 7);
#endif
#if 7
    MLI_CHECK_AND_FIX(kernel_height, 7);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_str1_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 0
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 0
    MLI_CHECK_AND_FIX(kernel_width, 0);
#endif
#if 0
    MLI_CHECK_AND_FIX(kernel_height, 0);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k1x2_ch1_str1_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 1
    MLI_CHECK_AND_FIX(kernel_width, 1);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_height, 2);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k2x1_ch1_str1_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_width, 2);
#endif
#if 1
    MLI_CHECK_AND_FIX(kernel_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k2x2_ch1_str1_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_width, 2);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_height, 2);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k3x3_ch1_str1_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 3
    MLI_CHECK_AND_FIX(kernel_width, 3);
#endif
#if 3
    MLI_CHECK_AND_FIX(kernel_height, 3);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k4x4_ch1_str1_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 4
    MLI_CHECK_AND_FIX(kernel_width, 4);
#endif
#if 4
    MLI_CHECK_AND_FIX(kernel_height, 4);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k5x5_ch1_str1_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 5
    MLI_CHECK_AND_FIX(kernel_width, 5);
#endif
#if 5
    MLI_CHECK_AND_FIX(kernel_height, 5);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k6x6_ch1_str1_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 6
    MLI_CHECK_AND_FIX(kernel_width, 6);
#endif
#if 6
    MLI_CHECK_AND_FIX(kernel_height, 6);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k7x7_ch1_str1_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 7
    MLI_CHECK_AND_FIX(kernel_width, 7);
#endif
#if 7
    MLI_CHECK_AND_FIX(kernel_height, 7);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k2x2_str1_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_width, 2);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_height, 2);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k3x3_str1_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 3
    MLI_CHECK_AND_FIX(kernel_width, 3);
#endif
#if 3
    MLI_CHECK_AND_FIX(kernel_height, 3);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k4x4_str1_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 4
    MLI_CHECK_AND_FIX(kernel_width, 4);
#endif
#if 4
    MLI_CHECK_AND_FIX(kernel_height, 4);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k5x5_str1_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 5
    MLI_CHECK_AND_FIX(kernel_width, 5);
#endif
#if 5
    MLI_CHECK_AND_FIX(kernel_height, 5);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k6x6_str1_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 6
    MLI_CHECK_AND_FIX(kernel_width, 6);
#endif
#if 6
    MLI_CHECK_AND_FIX(kernel_height, 6);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k7x7_str1_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 7
    MLI_CHECK_AND_FIX(kernel_width, 7);
#endif
#if 7
    MLI_CHECK_AND_FIX(kernel_height, 7);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_k1xn_str1(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 0
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 1
    MLI_CHECK_AND_FIX(kernel_width, 1);
#endif
#if 0
    MLI_CHECK_AND_FIX(kernel_height, 0);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_knx1_str1(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 0
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 0
    MLI_CHECK_AND_FIX(kernel_width, 0);
#endif
#if 1
    MLI_CHECK_AND_FIX(kernel_height, 1);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_ch1_str1(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 0
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 0
    MLI_CHECK_AND_FIX(kernel_width, 0);
#endif
#if 0
    MLI_CHECK_AND_FIX(kernel_height, 0);
#endif
#if 1
    MLI_CHECK_AND_FIX(channels, 1);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw_str1(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_depthwise_conv2d_chw_fx8w16d_generic(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];

    // assign hard coded values for this variation to some variables
#if 0
    MLI_CHECK_AND_FIX(stride_width, 0);
#endif
#if 0
    MLI_CHECK_AND_FIX(stride_height, 0);
#endif
#if 0
    MLI_CHECK_AND_FIX(padding_top, 0);
    MLI_CHECK_AND_FIX(padding_bot, 0);
    MLI_CHECK_AND_FIX(padding_left, 0);
    MLI_CHECK_AND_FIX(padding_right, 0);
#endif
#if 0
    MLI_CHECK_AND_FIX(kernel_width, 0);
#endif
#if 0
    MLI_CHECK_AND_FIX(kernel_height, 0);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = in->el_type;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    conv2d_chw(
        in_ftrs, wt, bs, out_ftrs, &cent_area,
        bias_shift, out_shift,
        val_limit.min, val_limit.max,
        1, in_width, in_height,
        channels, out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_bot, padding_left, padding_right,
        1, 1);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;

    return MLI_STATUS_OK;
}


mli_status mli_krn_depthwise_conv2d_chw_fx8w16d(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    int stride_w = cfg->stride_width;
    int stride_h = cfg->stride_height;
    int kernel_w = weights->shape[KRNL_W_DIM_CHW];
    int kernel_h = weights->shape[KRNL_H_DIM_CHW];
    int channels = in->shape[FMAP_C_DIM_CHW];
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;

    if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 7) && (kernel_h == 7) && 
            (channels == 1) && 
            (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k7x7_ch1_str1_nopad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 7) && (kernel_h == 7) && 
            (channels == 1) && 
            (padding_top == 3) && (padding_bot == 3) && (padding_left == 3) && (padding_right == 3)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k7x7_ch1_str1_krnpad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 7) && (kernel_h == 7) && 
            (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k7x7_str1_nopad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 7) && (kernel_h == 7) && 
            (padding_top == 3) && (padding_bot == 3) && (padding_left == 3) && (padding_right == 3)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k7x7_str1_krnpad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 6) && (kernel_h == 6) && 
            (channels == 1) && 
            (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k6x6_ch1_str1_nopad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 6) && (kernel_h == 6) && 
            (channels == 1) && 
            (padding_top == 2) && (padding_bot == 3) && (padding_left == 2) && (padding_right == 3)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k6x6_ch1_str1_krnpad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 6) && (kernel_h == 6) && 
            (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k6x6_str1_nopad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 6) && (kernel_h == 6) && 
            (padding_top == 2) && (padding_bot == 3) && (padding_left == 2) && (padding_right == 3)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k6x6_str1_krnpad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 5) && (kernel_h == 5) && 
            (channels == 1) && 
            (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k5x5_ch1_str1_nopad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 5) && (kernel_h == 5) && 
            (channels == 1) && 
            (padding_top == 2) && (padding_bot == 2) && (padding_left == 2) && (padding_right == 2)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k5x5_ch1_str1_krnpad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 5) && (kernel_h == 5) && 
            (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k5x5_str1_nopad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 5) && (kernel_h == 5) && 
            (padding_top == 2) && (padding_bot == 2) && (padding_left == 2) && (padding_right == 2)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k5x5_str1_krnpad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 4) && (kernel_h == 4) && 
            (channels == 1) && 
            (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k4x4_ch1_str1_nopad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 4) && (kernel_h == 4) && 
            (channels == 1) && 
            (padding_top == 1) && (padding_bot == 2) && (padding_left == 1) && (padding_right == 2)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k4x4_ch1_str1_krnpad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 4) && (kernel_h == 4) && 
            (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k4x4_str1_nopad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 4) && (kernel_h == 4) && 
            (padding_top == 1) && (padding_bot == 2) && (padding_left == 1) && (padding_right == 2)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k4x4_str1_krnpad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 3) && (kernel_h == 3) && 
            (channels == 1) && 
            (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k3x3_ch1_str1_nopad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 3) && (kernel_h == 3) && 
            (channels == 1) && 
            (padding_top == 1) && (padding_bot == 1) && (padding_left == 1) && (padding_right == 1)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k3x3_ch1_str1_krnpad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 3) && (kernel_h == 3) && 
            (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k3x3_str1_nopad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 3) && (kernel_h == 3) && 
            (padding_top == 1) && (padding_bot == 1) && (padding_left == 1) && (padding_right == 1)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k3x3_str1_krnpad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 2) && (kernel_h == 2) && 
            (channels == 1) && 
            (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k2x2_ch1_str1_nopad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 2) && (kernel_h == 2) && 
            (channels == 1) && 
            (padding_top == 0) && (padding_bot == 1) && (padding_left == 0) && (padding_right == 1)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k2x2_ch1_str1_krnpad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 2) && (kernel_h == 2) && 
            (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k2x2_str1_nopad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 2) && (kernel_h == 2) && 
            (padding_top == 0) && (padding_bot == 1) && (padding_left == 0) && (padding_right == 1)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k2x2_str1_krnpad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 2) && (kernel_h == 1) && 
            (channels == 1) && 
            (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k2x1_ch1_str1_nopad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 2) && (kernel_h == 1) && 
            (channels == 1) && 
            (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 1)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k2x1_ch1_str1_krnpad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 1) && (kernel_h == 2) && 
            (channels == 1) && 
            (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k1x2_ch1_str1_nopad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && 
            (kernel_w == 1) && (kernel_h == 2) && 
            (channels == 1) && 
            (padding_top == 0) && (padding_bot == 1) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k1x2_ch1_str1_krnpad(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && (kernel_w == 1)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_k1xn_str1(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && (kernel_h == 1)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_knx1_str1(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1) && (channels == 1)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_ch1_str1(in, weights, bias, cfg, out);
    } else if ((stride_w == 1) && (stride_h == 1)) {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_str1_krnpad(in, weights, bias, cfg, out);
    } else {
        return mli_krn_depthwise_conv2d_chw_fx8w16d_generic(in, weights, bias, cfg, out);
    }
}

#pragma code()

#ifdef __cplusplus
    }
#endif
