/*
* Copyright 2019-2020, Synopsys, Inc.
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
#include "mli_krn_dotprod_deprecated.h"

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
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t))bias->data.mem.void_p;

    const auto in_prv = mli_prv_get_tensor_hwc<MLI_PTR(int16_t), MLI_PTR_IS_XY>(in);
    const auto w = mli_prv_get_conv2d_weights_tensor_nhwc<MLI_PTR(int8_t), MLI_PTR_IS_XY>(weights);
    __builtin_assume(in_prv.ch == w.in_ch);

    uint16_t out_width = CEIL_DIV (in_prv.width + padding_left + padding_right - w.kernel_width + 1, stride_width);
    uint16_t out_height = CEIL_DIV (in_prv.height + padding_top + padding_bot - w.kernel_height + 1, stride_height);

    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[0] = out_height;
    out->shape[1] = out_width;
    out->shape[2] = w.out_ch;

    const auto out_prv = mli_prv_get_tensor_hwc<MLI_CONV_OUT_PTR(int16_t), MLI_CONV_OUT_PTR_IS_XY>(out);

    uint8_t bias_shift = (in->el_params.fx.frac_bits + weights->el_params.fx.frac_bits) - bias->el_params.fx.frac_bits;
    uint8_t out_shift = (in->el_params.fx.frac_bits + weights->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in_prv.height >= w.kernel_height && in_prv.width >= w.kernel_width) {
        rect_t cent_area;
        cent_area.row_beg = CEIL_DIV (padding_top, stride_height);
        cent_area.row_end = out_height - CEIL_DIV (padding_bot, stride_height);
        cent_area.clmn_beg = CEIL_DIV (padding_left, stride_width);
        cent_area.clmn_end = out_width - CEIL_DIV (padding_right, stride_width);

        for (uint32_t out_ch_idx = 0; out_ch_idx < w.out_ch; out_ch_idx++) {
            for (uint32_t H_idx = cent_area.row_beg; H_idx < cent_area.row_end; H_idx++) {
                for (uint32_t W_idx = cent_area.clmn_beg; W_idx < cent_area.clmn_end; W_idx++) {
                    auto accu = mli_prv_init_accu_with_bias (in_prv.ptr, bs[out_ch_idx], bias_shift);

                    // Define area of input and filter for convolution
                    const MLI_PTR (int16_t) in_ptr = in_prv.ptr +
                            + in_prv.row_mem_stride * (H_idx * stride_height - padding_top)  // move to row
                            + in_prv.col_mem_stride * (W_idx * stride_width - padding_left); // move to column

                    const MLI_PTR(int8_t) w_ptr = w.ptr
                            + w.out_ch_mem_stride * out_ch_idx; // move to filter

                    // Convolution core
                    for (int in_ch_idx = 0; in_ch_idx < in_prv.ch - 1; in_ch_idx += 2) {
                        dotprod2D_hwc_d (in_ptr, w_ptr, &accu, w.kernel_width, w.kernel_height,
                                in_prv.col_mem_stride, in_prv.row_mem_stride,
                                w.col_mem_stride, w.row_mem_stride);
                        in_ptr += 2;
                        w_ptr += 2;
                    }

                    if (in_prv.ch & 1) {
                        accu = dotprod2D (in_ptr, w_ptr, accu, w.kernel_width, w.kernel_height,
                                in_prv.col_mem_stride, in_prv.row_mem_stride,
                                w.col_mem_stride, w.row_mem_stride);
                        in_ptr += 1;
                        w_ptr += 1;
                    }
                    MLI_CONV_OUT_PTR(int16_t) o_ptr = &out_prv.ptr[out_ch_idx +
                        (out_prv.row_mem_stride * H_idx) +
                        (out_prv.col_mem_stride * W_idx)];
                    mli_prv_clip_relu_store_output (o_ptr, accu, out_shift, val_limit.min, val_limit.max);
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
            for (uint32_t out_ch_idx = 0; out_ch_idx < w.out_ch; out_ch_idx++) {
                for (uint32_t H_idx = perc_areas[area_idx].row_beg; H_idx < perc_areas[area_idx].row_end; H_idx++) {
                    for (uint32_t W_idx = perc_areas[area_idx].clmn_beg; W_idx < perc_areas[area_idx].clmn_end; W_idx++) {
                        auto accu = mli_prv_init_accu_with_bias (in_prv.ptr, bs[out_ch_idx], bias_shift);

                        // Define area of input and filter for convolution
                        // *_comp - compensation values for valid area defining
                        int32_t top_comp = -MIN ((int32_t) (H_idx * stride_height) - padding_top, 0);
                        int32_t left_comp = -MIN ((int32_t) (W_idx * stride_width) - padding_left, 0);

                        int32_t right_comp = -MIN ((int32_t) in_prv.width - ((int32_t) (W_idx * stride_width)
                                - padding_left + w.kernel_width), 0);
                        int32_t bottom_comp = -MIN ((int32_t) in_prv.height - ((int32_t) (H_idx * stride_height)
                                - padding_top + w.kernel_height), 0);

                        int32_t rows = w.kernel_height - top_comp - bottom_comp;
                        int32_t clmns = w.kernel_width - right_comp - left_comp;

                        const MLI_PTR (int16_t) in_ptr = in_prv.ptr
                                + in_prv.col_mem_stride * (W_idx * stride_width - padding_left + left_comp) // move to column
                                + in_prv.row_mem_stride * (H_idx * stride_height - padding_top + top_comp); // move to row

                        const MLI_PTR (int8_t) w_ptr = w.ptr
                                + w.col_mem_stride * left_comp      // move to column
                                + w.row_mem_stride * top_comp       // move to row
                                + w.out_ch_mem_stride * out_ch_idx; // move to filter

                        // Convolution core
                        for (int in_ch_idx = 0; in_ch_idx < in_prv.ch - 1; in_ch_idx += 2) {
                            dotprod2D_hwc_d (in_ptr, w_ptr, &accu, clmns, rows,
                                    in_prv.col_mem_stride, in_prv.row_mem_stride,
                                    w.col_mem_stride, w.row_mem_stride);
                            in_ptr += 2;
                            w_ptr += 2;
                        }

                        if (in_prv.ch & 1) {
                            accu = dotprod2D( in_ptr, w_ptr, accu, clmns, rows,
                                    in_prv.col_mem_stride, in_prv.row_mem_stride,
                                    w.col_mem_stride, w.row_mem_stride);
                            in_ptr += 1;
                            w_ptr += 1;
                        }
                        in_ptr -= in_prv.ch;
                        w_ptr -= in_prv.ch;

                        MLI_CONV_OUT_PTR(int16_t) o_ptr = out_prv.ptr
                                + out_prv.col_mem_stride * W_idx
                                + out_prv.row_mem_stride * H_idx
                                + out_prv.ch_mem_stride * out_ch_idx;
                        mli_prv_clip_relu_store_output (o_ptr, accu, out_shift, val_limit.min, val_limit.max);
                    }
                }
            }
        }
    }

    return MLI_STATUS_OK;
}

#pragma code()
#ifdef __cplusplus
}
#endif
