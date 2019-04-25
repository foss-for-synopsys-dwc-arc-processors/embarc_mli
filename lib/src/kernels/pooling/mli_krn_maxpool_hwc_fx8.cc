/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math_macros.h"
#include "mli_private_types.h"
#include "mli_prv_dsp.h"

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

static inline int8_t reduce_max2D_hwc(
        const MLI_PTR(int8_t) in,
        const int width,
        const int height,
        const int channels,
        const int in_row_step);

mli_status mli_krn_maxpool_hwc_fx8(const mli_tensor* in, const mli_pool_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_maxpool_hwc_fx8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli_prv_fx_init_dsp_ctrl();

    // Extract general maxpooling parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;

    // Data pointers
    const MLI_PTR(int8_t) in_ftrs = (const MLI_PTR(int8_t))in->data;
    MLI_OUT_PTR(int8_t) out_ftrs = (MLI_OUT_PTR(int8_t))out->data;

    // Define Data dimensions
    int channels_num = (int)in->shape[2];

    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;

    int in_height = (int)in->shape[0];
    int in_width = (int)in->shape[1];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in_height >= kernel_height && in_width >= kernel_width) {
        int row_beg = CEIL_DIV(padding_top, stride_height);
        int row_end = out_height - CEIL_DIV(padding_bot, stride_height);
        int clmn_beg = CEIL_DIV(padding_left, stride_width);
        int clmn_end = out_width - CEIL_DIV(padding_right, stride_width);

        for (int ch_idx = 0; ch_idx < channels_num; ch_idx++) {
            for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    const MLI_PTR(int8_t) in_ptr =
                            in_ftrs +                                                          // starting point
                            channels_num * in_width * (H_idx * stride_height - padding_top) +  // move to row
                            channels_num * (W_idx * stride_width - padding_left) +             // move to column
                            ch_idx;                                                            // move to channel

                    // Core Max
                    int8_t max_val = reduce_max2D_hwc(in_ptr, kernel_width, kernel_height, channels_num, in_width);

                    out_ftrs[ch_idx + (H_idx * out_width + W_idx) * channels_num] = max_val;
                }
            }
        }
    }
    // Phase 2: Process border part with more complex algorithm
    // (usually significantly smaller part of computations)
    //=======================================================================
    if (padding_top || padding_left || padding_bot || padding_right) {
        rect_t perc_areas[4];
        int areas_num = 0;
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
                        int top_comp = -MIN((H_idx * stride_height) - padding_top, 0);
                        int left_comp = -MIN((W_idx * stride_width) - padding_left, 0);

                        int right_comp = -MIN(in_width - ((W_idx * stride_width) - padding_left + kernel_width), 0);
                        int bottom_comp = -MIN(in_height - ((H_idx * stride_height) - padding_top + kernel_height), 0);

                        int rows = kernel_height - top_comp - bottom_comp;
                        int clmns = kernel_width - right_comp - left_comp;

                        const MLI_PTR(int8_t) in_ptr =
                                in_ftrs +  // starting point
                                channels_num * in_width *
                                (H_idx * stride_height - padding_top + top_comp) +            // move to row
                                channels_num * ((W_idx * stride_width) - padding_left + left_comp) +  // move to column
                                ch_idx;

                        // Core Max
                        int8_t max_val = reduce_max2D_hwc(in_ptr, clmns, rows, channels_num, in_width);

                        // Write result
                        out_ftrs[ch_idx + (H_idx * out_width + W_idx) * channels_num] = max_val;
                    }
                }
            }
        }
    }
    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[0] = (unsigned)out_height;
    out->shape[1] = (unsigned)out_width;
    out->shape[2] = (unsigned)channels_num;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;
    out->el_type = in->el_type;

    return MLI_STATUS_OK;
}

static inline int8_t reduce_max2D_hwc(
        const MLI_PTR(int8_t) in,
        const int width,
        const int height,
        const int channels,
        const int in_row_step) {
    int8_t cur_max = INT8_MIN;
    for (int row = 0; row < (int)height; row++) {
        for (int clmn = 0; clmn < (int)width; clmn++) {
            cur_max = MAX(cur_max, in[clmn * channels]);
        }
        in += in_row_step * channels;
    }
    return cur_max;
}

#pragma code()
#ifdef __cplusplus
}
#endif
