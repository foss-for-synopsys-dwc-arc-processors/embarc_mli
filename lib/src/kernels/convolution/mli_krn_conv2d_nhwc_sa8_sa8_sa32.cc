/* This file is generated, do not edit!
 * edit following template files instead:
 * filetemplate.txt
 * mli_krn_conv2d_nhwc_func_body.txt
 */
/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_krn_conv2d_hwc.h"

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

mli_status mli_krn_conv2d_nhwc_sa8_sa8_sa32_k3x3_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_nhwc_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    mli_prv_fx_init_dsp_ctrl();

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_HWC];
    int kernel_width = weights->shape[KRNL_W_DIM_HWC];
    int out_ch = weights->shape[KRNL_C_DIM_HWC];
    int in_ch = in->shape[FMAP_C_DIM_HWC];

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
#if 3
    MLI_CHECK_AND_FIX(kernel_width, 3);
#endif
#if 3
    MLI_CHECK_AND_FIX(kernel_height, 3);
#endif
#if 0
    MLI_CHECK_AND_FIX(in_ch, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = MLI_EL_ASYM_I8;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int8_t) in_ftrs = (MLI_PTR(int8_t ))in->data;
    MLI_CONV_OUT_PTR(int8_t) out_ftrs = (MLI_CONV_OUT_PTR(int8_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int32_t) bs = (MLI_PTR(int32_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_HWC];
    int in_width = in->shape[FMAP_W_DIM_HWC];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define quantization specific params
    s8asym_quant_specific_params params;
    define_quant_params(in, weights, bias, out, &params);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    convolution2D_nhwc_krnpad<int8_t, int8_t, int32_t, int32_t>(
            in_ftrs, wt, bs, out_ftrs, &cent_area, params,
            (int8_t)val_limit.min, (int8_t)val_limit.max, in_ch, in_width, in_height, 
            out_ch, out_width, out_height, kernel_height, kernel_width,
            stride_height, stride_width, padding_top, padding_left, padding_bot, padding_right);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_HWC] = out_ch;
    out->shape[FMAP_H_DIM_HWC] = out_height;
    out->shape[FMAP_W_DIM_HWC] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_conv2d_nhwc_sa8_sa8_sa32_k5x5_krnpad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_nhwc_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    mli_prv_fx_init_dsp_ctrl();

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_HWC];
    int kernel_width = weights->shape[KRNL_W_DIM_HWC];
    int out_ch = weights->shape[KRNL_C_DIM_HWC];
    int in_ch = in->shape[FMAP_C_DIM_HWC];

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
#if 5
    MLI_CHECK_AND_FIX(kernel_width, 5);
#endif
#if 5
    MLI_CHECK_AND_FIX(kernel_height, 5);
#endif
#if 0
    MLI_CHECK_AND_FIX(in_ch, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = MLI_EL_ASYM_I8;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int8_t) in_ftrs = (MLI_PTR(int8_t ))in->data;
    MLI_CONV_OUT_PTR(int8_t) out_ftrs = (MLI_CONV_OUT_PTR(int8_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int32_t) bs = (MLI_PTR(int32_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_HWC];
    int in_width = in->shape[FMAP_W_DIM_HWC];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define quantization specific params
    s8asym_quant_specific_params params;
    define_quant_params(in, weights, bias, out, &params);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    convolution2D_nhwc_krnpad<int8_t, int8_t, int32_t, int32_t>(
            in_ftrs, wt, bs, out_ftrs, &cent_area, params,
            (int8_t)val_limit.min, (int8_t)val_limit.max, in_ch, in_width, in_height, 
            out_ch, out_width, out_height, kernel_height, kernel_width,
            stride_height, stride_width, padding_top, padding_left, padding_bot, padding_right);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_HWC] = out_ch;
    out->shape[FMAP_H_DIM_HWC] = out_height;
    out->shape[FMAP_W_DIM_HWC] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_conv2d_nhwc_sa8_sa8_sa32_k3x3_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_nhwc_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    mli_prv_fx_init_dsp_ctrl();

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_HWC];
    int kernel_width = weights->shape[KRNL_W_DIM_HWC];
    int out_ch = weights->shape[KRNL_C_DIM_HWC];
    int in_ch = in->shape[FMAP_C_DIM_HWC];

    // assign hard coded values for this variation to some variables
#if 0
    MLI_CHECK_AND_FIX(stride_width, 0);
#endif
#if 0
    MLI_CHECK_AND_FIX(stride_height, 0);
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
    MLI_CHECK_AND_FIX(in_ch, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = MLI_EL_ASYM_I8;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int8_t) in_ftrs = (MLI_PTR(int8_t ))in->data;
    MLI_CONV_OUT_PTR(int8_t) out_ftrs = (MLI_CONV_OUT_PTR(int8_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int32_t) bs = (MLI_PTR(int32_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_HWC];
    int in_width = in->shape[FMAP_W_DIM_HWC];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define quantization specific params
    s8asym_quant_specific_params params;
    define_quant_params(in, weights, bias, out, &params);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    convolution2D_nhwc_nopad<int8_t, int8_t, int32_t, int32_t>(
            in_ftrs, wt, bs, out_ftrs, &cent_area, params,
            (int8_t)val_limit.min, (int8_t)val_limit.max, in_ch, in_width, in_height, 
            out_ch, out_width, out_height, kernel_height, kernel_width,
            stride_height, stride_width, padding_top, padding_left, padding_bot, padding_right);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_HWC] = out_ch;
    out->shape[FMAP_H_DIM_HWC] = out_height;
    out->shape[FMAP_W_DIM_HWC] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_conv2d_nhwc_sa8_sa8_sa32_k5x5_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_nhwc_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    mli_prv_fx_init_dsp_ctrl();

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_HWC];
    int kernel_width = weights->shape[KRNL_W_DIM_HWC];
    int out_ch = weights->shape[KRNL_C_DIM_HWC];
    int in_ch = in->shape[FMAP_C_DIM_HWC];

    // assign hard coded values for this variation to some variables
#if 0
    MLI_CHECK_AND_FIX(stride_width, 0);
#endif
#if 0
    MLI_CHECK_AND_FIX(stride_height, 0);
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
    MLI_CHECK_AND_FIX(in_ch, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = MLI_EL_ASYM_I8;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int8_t) in_ftrs = (MLI_PTR(int8_t ))in->data;
    MLI_CONV_OUT_PTR(int8_t) out_ftrs = (MLI_CONV_OUT_PTR(int8_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int32_t) bs = (MLI_PTR(int32_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_HWC];
    int in_width = in->shape[FMAP_W_DIM_HWC];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define quantization specific params
    s8asym_quant_specific_params params;
    define_quant_params(in, weights, bias, out, &params);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    convolution2D_nhwc_nopad<int8_t, int8_t, int32_t, int32_t>(
            in_ftrs, wt, bs, out_ftrs, &cent_area, params,
            (int8_t)val_limit.min, (int8_t)val_limit.max, in_ch, in_width, in_height, 
            out_ch, out_width, out_height, kernel_height, kernel_width,
            stride_height, stride_width, padding_top, padding_left, padding_bot, padding_right);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_HWC] = out_ch;
    out->shape[FMAP_H_DIM_HWC] = out_height;
    out->shape[FMAP_W_DIM_HWC] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_conv2d_nhwc_sa8_sa8_sa32_k1x1_nopad(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_nhwc_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    mli_prv_fx_init_dsp_ctrl();

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_HWC];
    int kernel_width = weights->shape[KRNL_W_DIM_HWC];
    int out_ch = weights->shape[KRNL_C_DIM_HWC];
    int in_ch = in->shape[FMAP_C_DIM_HWC];

    // assign hard coded values for this variation to some variables
#if 0
    MLI_CHECK_AND_FIX(stride_width, 0);
#endif
#if 0
    MLI_CHECK_AND_FIX(stride_height, 0);
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
#if 1
    MLI_CHECK_AND_FIX(kernel_height, 1);
#endif
#if 0
    MLI_CHECK_AND_FIX(in_ch, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = MLI_EL_ASYM_I8;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int8_t) in_ftrs = (MLI_PTR(int8_t ))in->data;
    MLI_CONV_OUT_PTR(int8_t) out_ftrs = (MLI_CONV_OUT_PTR(int8_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int32_t) bs = (MLI_PTR(int32_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_HWC];
    int in_width = in->shape[FMAP_W_DIM_HWC];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define quantization specific params
    s8asym_quant_specific_params params;
    define_quant_params(in, weights, bias, out, &params);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    pointwise_convolution2D_nhwc_nopad<int8_t, int8_t, int32_t, int32_t>(
            in_ftrs, wt, bs, out_ftrs, &cent_area, params,
            (int8_t)val_limit.min, (int8_t)val_limit.max, in_ch, in_width, in_height, 
            out_ch, out_width, out_height, kernel_height, kernel_width,
            stride_height, stride_width, padding_top, padding_left, padding_bot, padding_right);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_HWC] = out_ch;
    out->shape[FMAP_H_DIM_HWC] = out_height;
    out->shape[FMAP_W_DIM_HWC] = out_width;

    return MLI_STATUS_OK;
}

mli_status mli_krn_conv2d_nhwc_sa8_sa8_sa32_generic(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_nhwc_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    mli_prv_fx_init_dsp_ctrl();

    // Extract general conv2D parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int kernel_height = weights->shape[KRNL_H_DIM_HWC];
    int kernel_width = weights->shape[KRNL_W_DIM_HWC];
    int out_ch = weights->shape[KRNL_C_DIM_HWC];
    int in_ch = in->shape[FMAP_C_DIM_HWC];

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
    MLI_CHECK_AND_FIX(in_ch, 0);
#endif

    mli_minmax_t val_limit;
    // fill output tensor el_type parameter
    out->el_type = MLI_EL_ASYM_I8;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    // Data pointers
    MLI_PTR(int8_t) in_ftrs = (MLI_PTR(int8_t ))in->data;
    MLI_CONV_OUT_PTR(int8_t) out_ftrs = (MLI_CONV_OUT_PTR(int8_t ))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t ))weights->data;
    MLI_PTR(int32_t) bs = (MLI_PTR(int32_t ))bias->data;

    // Define Data dimensions
    int in_height = in->shape[FMAP_H_DIM_HWC];
    int in_width = in->shape[FMAP_W_DIM_HWC];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Define quantization specific params
    s8asym_quant_specific_params params;
    define_quant_params(in, weights, bias, out, &params);

    //=======================================================================
    rect_t cent_area;
    cent_area.row_beg = 0;
    cent_area.row_end = out_height;
    cent_area.clmn_beg = 0;
    cent_area.clmn_end = out_width;

    convolution2D_nhwc_krnpad<int8_t, int8_t, int32_t, int32_t>(
            in_ftrs, wt, bs, out_ftrs, &cent_area, params,
            (int8_t)val_limit.min, (int8_t)val_limit.max, in_ch, in_width, in_height, 
            out_ch, out_width, out_height, kernel_height, kernel_width,
            stride_height, stride_width, padding_top, padding_left, padding_bot, padding_right);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_HWC] = out_ch;
    out->shape[FMAP_H_DIM_HWC] = out_height;
    out->shape[FMAP_W_DIM_HWC] = out_width;

    return MLI_STATUS_OK;
}


mli_status mli_krn_conv2d_nhwc_sa8_sa8_sa32(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    int stride_w = cfg->stride_width;
    int stride_h = cfg->stride_height;
    int kernel_w = weights->shape[KRNL_W_DIM_HWC];
    int kernel_h = weights->shape[KRNL_H_DIM_HWC];
    int in_ch = in->shape[KRNL_C_DIM_HWC];
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;

    if ((kernel_w == 5) && (kernel_h == 5) && (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_conv2d_nhwc_sa8_sa8_sa32_k5x5_nopad(in, weights, bias, cfg, out);
    } else if ((kernel_w == 5) && (kernel_h == 5) && (padding_top <= 2) && (padding_bot <= 2) && (padding_left <= 2) && (padding_right <= 2)) {
        return mli_krn_conv2d_nhwc_sa8_sa8_sa32_k5x5_krnpad(in, weights, bias, cfg, out);
    } else if ((kernel_w == 3) && (kernel_h == 3) && (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_conv2d_nhwc_sa8_sa8_sa32_k3x3_nopad(in, weights, bias, cfg, out);
    } else if ((kernel_w == 3) && (kernel_h == 3) && (padding_top <= 1) && (padding_bot <= 1) && (padding_left <= 1) && (padding_right <= 1)) {
        return mli_krn_conv2d_nhwc_sa8_sa8_sa32_k3x3_krnpad(in, weights, bias, cfg, out);
    } else if ((kernel_w == 1) && (kernel_h == 1) && (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return mli_krn_conv2d_nhwc_sa8_sa8_sa32_k1x1_nopad(in, weights, bias, cfg, out);
    } else {
        return mli_krn_conv2d_nhwc_sa8_sa8_sa32_generic(in, weights, bias, cfg, out);
    }
}
char * mli_debug_krn_conv2d_nhwc_sa8_sa8_sa32(
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg, 
        mli_tensor * out) {
    int stride_w = cfg->stride_width;
    int stride_h = cfg->stride_height;
    int kernel_w = weights->shape[KRNL_W_DIM_HWC];
    int kernel_h = weights->shape[KRNL_H_DIM_HWC];
    int in_ch = in->shape[KRNL_C_DIM_HWC];
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;

    if ((kernel_w == 5) && (kernel_h == 5) && (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return (char*)"mli_krn_conv2d_nhwc_sa8_sa8_sa32_k5x5_nopad";
    } else if ((kernel_w == 5) && (kernel_h == 5) && (padding_top <= 2) && (padding_bot <= 2) && (padding_left <= 2) && (padding_right <= 2)) {
        return (char*)"mli_krn_conv2d_nhwc_sa8_sa8_sa32_k5x5_krnpad";
    } else if ((kernel_w == 3) && (kernel_h == 3) && (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return (char*)"mli_krn_conv2d_nhwc_sa8_sa8_sa32_k3x3_nopad";
    } else if ((kernel_w == 3) && (kernel_h == 3) && (padding_top <= 1) && (padding_bot <= 1) && (padding_left <= 1) && (padding_right <= 1)) {
        return (char*)"mli_krn_conv2d_nhwc_sa8_sa8_sa32_k3x3_krnpad";
    } else if ((kernel_w == 1) && (kernel_h == 1) && (padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
        return (char*)"mli_krn_conv2d_nhwc_sa8_sa8_sa32_k1x1_nopad";
    } else {
        return (char*)"mli_krn_conv2d_nhwc_sa8_sa8_sa32_generic";
    }
}

#pragma code()

#ifdef __cplusplus
    }
#endif
