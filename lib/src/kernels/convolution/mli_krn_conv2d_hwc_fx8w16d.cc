/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_types.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_prv_tensor.h"
#include "mli_math_macros.h"
#include "mli_private_types.h"
#include "mli_prv_dsp.h"
#include "mli_krn_dotprod_chw.h"

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
#ifdef __cplusplus
extern "C" {
#endif

#pragma Code(".mli_lib")

mli_status mli_krn_conv2d_hwc_fx8w16d (
        const mli_tensor * in, 
        const mli_tensor * weights, 
        const mli_tensor * bias, 
        const mli_conv2d_cfg * cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwc_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    uint8_t stride_width = cfg->stride_width;
    uint8_t stride_height = cfg->stride_height;
    uint8_t padding_top = cfg->padding_top;
    uint8_t padding_bot = cfg->padding_bottom;
    uint8_t padding_left = cfg->padding_left;
    uint8_t padding_right = cfg->padding_right;

    mli_minmax_t val_limit;
    // Define output val limits - we need it in case built-in RELU
    val_limit = mli_prv_get_relu_min_max (&cfg->relu, out);

    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t))in->data;
    MLI_CONV_OUT_PTR(int16_t) out_ftrs = (MLI_CONV_OUT_PTR(int16_t))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t))bias->data;

    uint16_t in_ch = in->shape[2];
    uint16_t out_ch = weights->shape[0];

    uint16_t kernel_height = weights->shape[1];
    uint16_t kernel_width = weights->shape[2];

    uint16_t in_height = in->shape[0];
    uint16_t in_width = in->shape[1];

    uint16_t out_width = (in_width + padding_left + padding_right - kernel_width + 1);
    out_width = (out_width % stride_width != 0) ? (out_width / stride_width + 1) : out_width / stride_width;

    uint16_t out_height = (in_height + padding_top + padding_bot - kernel_height + 1);
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
        for (uint32_t out_ch_idx = 0; out_ch_idx < out_ch; out_ch_idx++) {
            for (uint32_t H_idx = cent_area.row_beg; H_idx < cent_area.row_end; H_idx++) {
                for (uint32_t W_idx = cent_area.clmn_beg; W_idx < cent_area.clmn_end; W_idx++) {
                    auto conv_out = mli_prv_init_accu_with_bias (in_ftrs, bs[out_ch_idx], bias_shift);

                    // Define area of input and filter for convolution

                    MLI_PTR(int16_t) in_ptr = in_ftrs + // starting point
                            in_ch * in_width * (H_idx * stride_height - padding_top) +  // move to row
                            in_ch * (W_idx * stride_width - padding_left);  // move to column

                    MLI_PTR(int8_t) w_ptr = wt +    // Start point
                            out_ch_idx * in_ch * kernel_width * kernel_height;  // move to filter

                    // Convolution core
                    dotprod2D (in_ptr, w_ptr, kernel_width * in_ch, kernel_height, in_width * in_ch, 
                            kernel_width * in_ch, &conv_out);

                    MLI_CONV_OUT_PTR(int16_t) o_ptr = &out_ftrs[out_ch_idx + (H_idx * out_width + W_idx) * out_ch];
                    mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_limit.min, val_limit.max);
                }
            }
        }
    }
    // Phase 2: Process border part with more complex algorithm
    // (usually significantly smaller part of computations)
    //=======================================================================
    if (padding_top || padding_left || padding_bot || padding_right) {
        rect_t perc_areas[4];
        uint32_t areas_num = 0;
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

        for (uint32_t area_idx = 0; area_idx < areas_num; ++area_idx) {
            for (uint32_t out_ch_idx = 0; out_ch_idx < out_ch; out_ch_idx++) {
                for (uint32_t H_idx = perc_areas[area_idx].row_beg; H_idx < perc_areas[area_idx].row_end; H_idx++) {
                    for (uint32_t W_idx = perc_areas[area_idx].clmn_beg; W_idx < perc_areas[area_idx].clmn_end; W_idx++) {
                        auto conv_out = mli_prv_init_accu_with_bias (in_ftrs, bs[out_ch_idx], bias_shift);

                        // Define area of input and filter for convolution
                        // *_comp - compensation values for valid area defining
                        int32_t top_comp = -MIN ((int32_t) (H_idx * stride_height) - padding_top, 0);
                        int32_t left_comp = -MIN ((int32_t) (W_idx * stride_width) - padding_left, 0);

                        int32_t right_comp = -MIN ((int32_t) in_width - ((int32_t) (W_idx * stride_width) 
                                - padding_left + kernel_width), 0);
                        int32_t bottom_comp = -MIN ((int32_t) in_height - ((int32_t) (H_idx * stride_height) 
                                - padding_top + kernel_height), 0);

                        int32_t rows = kernel_height - top_comp - bottom_comp;
                        int32_t clmns = kernel_width - right_comp - left_comp;

                        MLI_PTR(int16_t) in_ptr = in_ftrs + // starting point
                                in_ch * in_width * (H_idx * stride_height - padding_top + top_comp) +   // move to row
                                in_ch * ((W_idx * stride_width) - padding_left + left_comp);    // move to column

                        MLI_PTR(int8_t) w_ptr = wt +    // Start point
                                out_ch_idx * in_ch * kernel_width * kernel_height + // move to filter
                                top_comp * kernel_width * in_ch +   // move to row
                                left_comp * in_ch;  // move to column

                        // Convolution core
                        dotprod2D (in_ptr, w_ptr, clmns * in_ch, rows, in_width * in_ch, kernel_width * in_ch, &conv_out);

                        MLI_CONV_OUT_PTR(int16_t) o_ptr = &out_ftrs[out_ch_idx + (H_idx * out_width + W_idx) * out_ch];
                        mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_limit.min, val_limit.max);
                    }
                }
            }
        }
    }
    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[0] = out_height;
    out->shape[1] = out_width;
    out->shape[2] = out_ch;

    return MLI_STATUS_OK;
}

#pragma code()
#ifdef __cplusplus
}
#endif
