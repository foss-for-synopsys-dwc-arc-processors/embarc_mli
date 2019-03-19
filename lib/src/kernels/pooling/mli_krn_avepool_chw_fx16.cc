/* This file is generated, do not edit!
 * edit following template files instead:
 * filetemplate.txt
 * avepool_func_body.txt
 */
/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_krn_avepool_chw.h"

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

mli_status mli_krn_avepool_chw_fx16_k2x2_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_nopad_k2x2(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k4x4_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_k4x4_str1_nopad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k5x5_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k7x7_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k9x9_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 4);
    MLI_CHECK_AND_FIX(padding_bot, 4);
    MLI_CHECK_AND_FIX(padding_left, 4);
    MLI_CHECK_AND_FIX(padding_right, 4);
#endif
#if 9
    MLI_CHECK_AND_FIX(kernel_width, 9);
#endif
#if 9
    MLI_CHECK_AND_FIX(kernel_height, 9);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k4x2_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(padding_left, 1);
    MLI_CHECK_AND_FIX(padding_right, 2);
#endif
#if 4
    MLI_CHECK_AND_FIX(kernel_width, 4);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_height, 2);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k4x4_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k4x6_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(padding_left, 1);
    MLI_CHECK_AND_FIX(padding_right, 2);
#endif
#if 4
    MLI_CHECK_AND_FIX(kernel_width, 4);
#endif
#if 6
    MLI_CHECK_AND_FIX(kernel_height, 6);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k4x8_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 3);
    MLI_CHECK_AND_FIX(padding_bot, 4);
    MLI_CHECK_AND_FIX(padding_left, 1);
    MLI_CHECK_AND_FIX(padding_right, 2);
#endif
#if 4
    MLI_CHECK_AND_FIX(kernel_width, 4);
#endif
#if 8
    MLI_CHECK_AND_FIX(kernel_height, 8);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k6x2_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(padding_left, 2);
    MLI_CHECK_AND_FIX(padding_right, 3);
#endif
#if 6
    MLI_CHECK_AND_FIX(kernel_width, 6);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_height, 2);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k6x4_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(padding_left, 2);
    MLI_CHECK_AND_FIX(padding_right, 3);
#endif
#if 6
    MLI_CHECK_AND_FIX(kernel_width, 6);
#endif
#if 4
    MLI_CHECK_AND_FIX(kernel_height, 4);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k6x6_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k6x8_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 3);
    MLI_CHECK_AND_FIX(padding_bot, 4);
    MLI_CHECK_AND_FIX(padding_left, 2);
    MLI_CHECK_AND_FIX(padding_right, 3);
#endif
#if 6
    MLI_CHECK_AND_FIX(kernel_width, 6);
#endif
#if 8
    MLI_CHECK_AND_FIX(kernel_height, 8);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k8x2_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(padding_left, 3);
    MLI_CHECK_AND_FIX(padding_right, 4);
#endif
#if 8
    MLI_CHECK_AND_FIX(kernel_width, 8);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_height, 2);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k8x4_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(padding_left, 3);
    MLI_CHECK_AND_FIX(padding_right, 4);
#endif
#if 8
    MLI_CHECK_AND_FIX(kernel_width, 8);
#endif
#if 4
    MLI_CHECK_AND_FIX(kernel_height, 4);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k8x6_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(padding_left, 3);
    MLI_CHECK_AND_FIX(padding_right, 4);
#endif
#if 8
    MLI_CHECK_AND_FIX(kernel_width, 8);
#endif
#if 6
    MLI_CHECK_AND_FIX(kernel_height, 6);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k8x8_str1_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
    // assign hard coded values for this variation to some variables
#if 1
    MLI_CHECK_AND_FIX(stride_width, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(stride_height, 1);
#endif
#if 1
    MLI_CHECK_AND_FIX(padding_top, 3);
    MLI_CHECK_AND_FIX(padding_bot, 4);
    MLI_CHECK_AND_FIX(padding_left, 3);
    MLI_CHECK_AND_FIX(padding_right, 4);
#endif
#if 8
    MLI_CHECK_AND_FIX(kernel_width, 8);
#endif
#if 8
    MLI_CHECK_AND_FIX(kernel_height, 8);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k4x2_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 2
    MLI_CHECK_AND_FIX(kernel_height, 2);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_nopad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k6x2_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 2
    MLI_CHECK_AND_FIX(kernel_height, 2);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_nopad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k6x4_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 4
    MLI_CHECK_AND_FIX(kernel_height, 4);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_nopad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k6x6_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_nopad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k6x8_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 8
    MLI_CHECK_AND_FIX(kernel_height, 8);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_nopad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k8x2_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 8
    MLI_CHECK_AND_FIX(kernel_width, 8);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_height, 2);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_nopad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k8x4_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 8
    MLI_CHECK_AND_FIX(kernel_width, 8);
#endif
#if 4
    MLI_CHECK_AND_FIX(kernel_height, 4);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_nopad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k8x6_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 8
    MLI_CHECK_AND_FIX(kernel_width, 8);
#endif
#if 6
    MLI_CHECK_AND_FIX(kernel_height, 6);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_nopad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k8x8_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 8
    MLI_CHECK_AND_FIX(kernel_width, 8);
#endif
#if 8
    MLI_CHECK_AND_FIX(kernel_height, 8);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_nopad_k4_Nx2_N_even(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k3x3_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_nopad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k5x5_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_nopad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k7x7_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_nopad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k9x9_str1_nopad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 9
    MLI_CHECK_AND_FIX(kernel_width, 9);
#endif
#if 9
    MLI_CHECK_AND_FIX(kernel_height, 9);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_nopad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k2x2_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 2
    MLI_CHECK_AND_FIX(kernel_width, 2);
#endif
#if 2
    MLI_CHECK_AND_FIX(kernel_height, 2);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k3x3_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k4x4_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 4
    MLI_CHECK_AND_FIX(kernel_width, 4);
#endif
#if 4
    MLI_CHECK_AND_FIX(kernel_height, 4);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k5x5_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k6x6_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 6
    MLI_CHECK_AND_FIX(kernel_width, 6);
#endif
#if 6
    MLI_CHECK_AND_FIX(kernel_height, 6);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k7x7_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 7
    MLI_CHECK_AND_FIX(kernel_width, 7);
#endif
#if 7
    MLI_CHECK_AND_FIX(kernel_height, 7);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k8x8_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 8
    MLI_CHECK_AND_FIX(kernel_width, 8);
#endif
#if 8
    MLI_CHECK_AND_FIX(kernel_height, 8);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k9x9_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 9
    MLI_CHECK_AND_FIX(kernel_width, 9);
#endif
#if 9
    MLI_CHECK_AND_FIX(kernel_height, 9);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_k10x10_krnpad(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
#if 10
    MLI_CHECK_AND_FIX(kernel_width, 10);
#endif
#if 10
    MLI_CHECK_AND_FIX(kernel_height, 10);
#endif
#if 0
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}

mli_status mli_krn_avepool_chw_fx16_generic(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general avepool parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;
    int channels_num = in->shape[FMAP_C_DIM_CHW];
    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;
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
    MLI_CHECK_AND_FIX(channels_num, 0);
#endif

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t ))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t ))out->data;

    // Define Data dimensions
    const int in_height = in->shape[FMAP_H_DIM_CHW];
    const int in_width = in->shape[FMAP_W_DIM_CHW];

    const int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    avepool_chw_krnpad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels_num, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_C_DIM_CHW] = channels_num;
    out->shape[FMAP_H_DIM_CHW] = out_height;
    out->shape[FMAP_W_DIM_CHW] = out_width;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;

    return MLI_STATUS_OK;
}


mli_status mli_krn_avepool_chw_fx16(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    int stride_w = cfg->stride_width;
    int stride_h = cfg->stride_height;
    int kernel_w = cfg->kernel_width;
    int kernel_h = cfg->kernel_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;

    if (stride_w == 1) {
        if (stride_h == 1) {
            if (kernel_w == 9) {
                if (kernel_h == 9) {
                    if ((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
                        return mli_krn_avepool_chw_fx16_k9x9_str1_nopad(in, cfg, out);
                    } else if ((padding_top == 4) && (padding_bot == 4) && (padding_left == 4) && (padding_right == 4)) {
                        return mli_krn_avepool_chw_fx16_k9x9_str1_krnpad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else if (kernel_w == 8) {
                if (kernel_h == 8) {
                    if ((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
                        return mli_krn_avepool_chw_fx16_k8x8_str1_nopad(in, cfg, out);
                    } else if ((padding_top == 3) && (padding_bot == 4) && (padding_left == 3) && (padding_right == 4)) {
                        return mli_krn_avepool_chw_fx16_k8x8_str1_krnpad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else if (kernel_h == 6) {
                    if ((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
                        return mli_krn_avepool_chw_fx16_k8x6_str1_nopad(in, cfg, out);
                    } else if ((padding_top == 2) && (padding_bot == 3) && (padding_left == 3) && (padding_right == 4)) {
                        return mli_krn_avepool_chw_fx16_k8x6_str1_krnpad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else if (kernel_h == 4) {
                    if ((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
                        return mli_krn_avepool_chw_fx16_k8x4_str1_nopad(in, cfg, out);
                    } else if ((padding_top == 1) && (padding_bot == 2) && (padding_left == 3) && (padding_right == 4)) {
                        return mli_krn_avepool_chw_fx16_k8x4_str1_krnpad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else if (kernel_h == 2) {
                    if ((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
                        return mli_krn_avepool_chw_fx16_k8x2_str1_nopad(in, cfg, out);
                    } else if ((padding_top == 0) && (padding_bot == 1) && (padding_left == 3) && (padding_right == 4)) {
                        return mli_krn_avepool_chw_fx16_k8x2_str1_krnpad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else if (kernel_w == 7) {
                if (kernel_h == 7) {
                    if ((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
                        return mli_krn_avepool_chw_fx16_k7x7_str1_nopad(in, cfg, out);
                    } else if ((padding_top == 3) && (padding_bot == 3) && (padding_left == 3) && (padding_right == 3)) {
                        return mli_krn_avepool_chw_fx16_k7x7_str1_krnpad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else if (kernel_w == 6) {
                if (kernel_h == 8) {
                    if ((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
                        return mli_krn_avepool_chw_fx16_k6x8_str1_nopad(in, cfg, out);
                    } else if ((padding_top == 3) && (padding_bot == 4) && (padding_left == 2) && (padding_right == 3)) {
                        return mli_krn_avepool_chw_fx16_k6x8_str1_krnpad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else if (kernel_h == 6) {
                    if ((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
                        return mli_krn_avepool_chw_fx16_k6x6_str1_nopad(in, cfg, out);
                    } else if ((padding_top == 2) && (padding_bot == 3) && (padding_left == 2) && (padding_right == 3)) {
                        return mli_krn_avepool_chw_fx16_k6x6_str1_krnpad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else if (kernel_h == 4) {
                    if ((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
                        return mli_krn_avepool_chw_fx16_k6x4_str1_nopad(in, cfg, out);
                    } else if ((padding_top == 1) && (padding_bot == 2) && (padding_left == 2) && (padding_right == 3)) {
                        return mli_krn_avepool_chw_fx16_k6x4_str1_krnpad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else if (kernel_h == 2) {
                    if ((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
                        return mli_krn_avepool_chw_fx16_k6x2_str1_nopad(in, cfg, out);
                    } else if ((padding_top == 0) && (padding_bot == 1) && (padding_left == 2) && (padding_right == 3)) {
                        return mli_krn_avepool_chw_fx16_k6x2_str1_krnpad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else if (kernel_w == 5) {
                if (kernel_h == 5) {
                    if ((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
                        return mli_krn_avepool_chw_fx16_k5x5_str1_nopad(in, cfg, out);
                    } else if ((padding_top == 2) && (padding_bot == 2) && (padding_left == 2) && (padding_right == 2)) {
                        return mli_krn_avepool_chw_fx16_k5x5_str1_krnpad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else if (kernel_w == 4) {
                if (kernel_h == 8) {
                    if ((padding_top == 3) && (padding_bot == 4) && (padding_left == 1) && (padding_right == 2)) {
                        return mli_krn_avepool_chw_fx16_k4x8_str1_krnpad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else if (kernel_h == 6) {
                    if ((padding_top == 2) && (padding_bot == 3) && (padding_left == 1) && (padding_right == 2)) {
                        return mli_krn_avepool_chw_fx16_k4x6_str1_krnpad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else if (kernel_h == 4) {
                    if ((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
                        return mli_krn_avepool_chw_fx16_k4x4_str1_nopad(in, cfg, out);
                    } else if ((padding_top == 1) && (padding_bot == 2) && (padding_left == 1) && (padding_right == 2)) {
                        return mli_krn_avepool_chw_fx16_k4x4_str1_krnpad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else if (kernel_h == 2) {
                    if ((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
                        return mli_krn_avepool_chw_fx16_k4x2_str1_nopad(in, cfg, out);
                    } else if ((padding_top == 0) && (padding_bot == 1) && (padding_left == 1) && (padding_right == 2)) {
                        return mli_krn_avepool_chw_fx16_k4x2_str1_krnpad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else if (kernel_w == 3) {
                if (kernel_h == 3) {
                    if ((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
                        return mli_krn_avepool_chw_fx16_k3x3_str1_nopad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else if (kernel_w == 2) {
                if (kernel_h == 2) {
                    if ((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0)) {
                        return mli_krn_avepool_chw_fx16_k2x2_str1_nopad(in, cfg, out);
                    } else {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else {
                return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
            }
        } else {
            return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
        }
    } else {
        {
            if (kernel_w == 10) {
                if (kernel_h == 10) {
                    {
                        return mli_krn_avepool_chw_fx16_k10x10_krnpad(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else if (kernel_w == 9) {
                if (kernel_h == 9) {
                    {
                        return mli_krn_avepool_chw_fx16_k9x9_krnpad(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else if (kernel_w == 8) {
                if (kernel_h == 8) {
                    {
                        return mli_krn_avepool_chw_fx16_k8x8_krnpad(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else if (kernel_w == 7) {
                if (kernel_h == 7) {
                    {
                        return mli_krn_avepool_chw_fx16_k7x7_krnpad(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else if (kernel_w == 6) {
                if (kernel_h == 6) {
                    {
                        return mli_krn_avepool_chw_fx16_k6x6_krnpad(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else if (kernel_w == 5) {
                if (kernel_h == 5) {
                    {
                        return mli_krn_avepool_chw_fx16_k5x5_krnpad(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else if (kernel_w == 4) {
                if (kernel_h == 4) {
                    {
                        return mli_krn_avepool_chw_fx16_k4x4_krnpad(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else if (kernel_w == 3) {
                if (kernel_h == 3) {
                    {
                        return mli_krn_avepool_chw_fx16_k3x3_krnpad(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else if (kernel_w == 2) {
                if (kernel_h == 2) {
                    {
                        return mli_krn_avepool_chw_fx16_k2x2_krnpad(in, cfg, out);
                    }
                } else {
                    return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                }
            } else {
                {
                    {
                        return mli_krn_avepool_chw_fx16_generic(in, cfg, out);
                    }
                }
            }
        }
    }
}

#pragma code()

#ifdef __cplusplus
    }
#endif