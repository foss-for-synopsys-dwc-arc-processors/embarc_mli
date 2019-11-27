/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <assert.h>
#include <stdio.h>

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_krn_reduce_sum2d.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"

#ifdef __FXAPI__
#include <fxarc.h>
#endif

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

/******************************************************************************
 *
 * Version & platform description
 * Targets:
 *
 ******************************************************************************/

static inline int32_t reduce_sum2D_hwc(MLI_PTR(int8_t) in, uint32_t width, uint32_t height, uint32_t channels, uint32_t in_row_step);

mli_status mli_krn_avepool_hwc_fx8(const mli_tensor* in, const mli_pool_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_hwc_fx8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli_prv_fx_init_dsp_ctrl();

    // Extract general maxpooling parameters
    int32_t stride_width = cfg->stride_width;
    int32_t stride_height = cfg->stride_height;
    int32_t padding_top = cfg->padding_top;
    int32_t padding_bot = cfg->padding_bottom;
    int32_t padding_left = cfg->padding_left;
    int32_t padding_right = cfg->padding_right;

    // Data pointers
    MLI_PTR(int8_t) in_ftrs = (MLI_PTR(int8_t))in->data;
    MLI_OUT_PTR(int8_t) out_ftrs = (MLI_OUT_PTR(int8_t))out->data;

    // Define Data dimensions
    int32_t channels_num = in->shape[FMAP_C_DIM_HWC];

    int32_t kernel_height = cfg->kernel_height;
    int32_t kernel_width = cfg->kernel_width;

    int32_t in_height = in->shape[FMAP_H_DIM_HWC];
    int32_t in_width = in->shape[FMAP_W_DIM_HWC];

    const int32_t out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int32_t out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in_height >= kernel_height && in_width >= kernel_width) {
        const int32_t row_beg = CEIL_DIV(padding_top, stride_height);
        const int32_t row_end = out_height - CEIL_DIV(padding_bot, stride_height);
        const int32_t clmn_beg = CEIL_DIV(padding_left, stride_width);
        const int32_t clmn_end = out_width - CEIL_DIV(padding_right, stride_width);

        const int32_t kernel_size = kernel_width * kernel_height;
        int16_t mul = 0;
        int shift = 0;
        get_mul_shift_value(kernel_size, &mul, &shift);
        for (int ch_idx = 0; ch_idx < channels_num; ch_idx++) {
            for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    MLI_PTR(int8_t)
                    in_ptr = in_ftrs +                                                          // starting point
                             channels_num * in_width * (H_idx * stride_height - padding_top) +  // move to row
                             channels_num * (W_idx * stride_width - padding_left) +             // move to column
                             ch_idx;                                                            // move to channel

                    // Core Sum
                    accum40_t accu = reduce_sum2D_hwc(in_ptr, kernel_width, kernel_height, channels_num, in_width, mul);

                    MLI_OUT_PTR(int8_t)
                    p_out_ftrs = (MLI_OUT_PTR(int8_t))(out_ftrs + ch_idx + (H_idx * out_width + W_idx) * channels_num);
                    mli_prv_shift_clip_and_store_output(p_out_ftrs, &accu, shift);
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
            perc_areas[areas_num].row_end = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = out_width;
        }
        if (padding_bot) {
            perc_areas[areas_num].row_beg = out_height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].row_end = out_height;
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = out_width;
        }
        if (padding_left) {
            perc_areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].row_end = out_height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = CEIL_DIV(padding_left, stride_width);
        }
        if (padding_right) {
            perc_areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].row_end = out_height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = out_width - CEIL_DIV(padding_right, stride_width);
            perc_areas[areas_num++].clmn_end = out_width;
        }

        for (int area_idx = 0; area_idx < areas_num; ++area_idx) {
            for (int ch_idx = 0; ch_idx < channels_num; ch_idx++) {
                for (int H_idx = perc_areas[area_idx].row_beg; H_idx < perc_areas[area_idx].row_end; H_idx++) {
                    for (int W_idx = perc_areas[area_idx].clmn_beg; W_idx < perc_areas[area_idx].clmn_end; W_idx++) {
                        // Define area of input and filter for convolution
                        // *_comp - compensation values for valid area defining
                        int32_t top_comp = -MIN((int32_t)(H_idx * stride_height) - padding_top, 0);
                        int32_t left_comp = -MIN((int32_t)(W_idx * stride_width) - padding_left, 0);

                        int32_t right_comp = -MIN(
                                (int32_t)in_width - ((int32_t)(W_idx * stride_width) - padding_left + kernel_width), 0);
                        int32_t bottom_comp = -MIN(
                                (int32_t)in_height - ((int32_t)(H_idx * stride_height) - padding_top + kernel_height),
                                0);

                        int32_t rows = kernel_height - top_comp - bottom_comp;
                        int32_t clmns = kernel_width - right_comp - left_comp;
                        unsigned int max_kernel_size = kernel_width * kernel_height;
                        int kernel_size = rows * clmns;
                        int16_t mul = 0;
                        int shift = 0;
                        get_mul_shift_value(kernel_size, &mul, &shift);

                        MLI_PTR(int8_t)
                        in_ptr = in_ftrs +  // starting point
                                 channels_num * in_width *
                                         (H_idx * stride_height - padding_top + top_comp) +            // move to row
                                 channels_num * ((W_idx * stride_width) - padding_left + left_comp) +  // move to column
                                 ch_idx;

                        // Core Sum
                        accum40_t accu = reduce_sum2D_hwc(in_ptr, clmns, rows, channels_num, in_width, mul);

                        // Write result
                        MLI_OUT_PTR(int8_t)
                        p_out_ftrs = (MLI_OUT_PTR(int8_t))(out_ftrs + ch_idx + (H_idx * out_width + W_idx) * channels_num);
                        mli_prv_shift_clip_and_store_output(p_out_ftrs, &accu, shift);
                    }
                }
            }
        }
    }
    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_H_DIM_HWC] = out_height;
    out->shape[FMAP_W_DIM_HWC] = out_width;
    out->shape[FMAP_C_DIM_HWC] = channels_num;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;
    out->el_type = in->el_type;

    return MLI_STATUS_OK;
}

#pragma code()

#ifdef __cplusplus
}
#endif
