/*
 *  Copyright (c) 2019, Synopsys, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1) Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 
 * 2)  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3) Neither the name of the <ORGANIZATION> nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "mli_types.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_prv_tensor.h"
#include "mli_math_macros.h"
#include "mli_private_types.h"
#include "mli_prv_dsp.h"

  /**
   * @brief
   * @param[in]
   * @param[in]
   * @param[out]
   * @return
   *
   * @details
   *
   *
   *
   */

#ifdef __cplusplus
extern "C" {
#endif

static inline int32_t dotprod2D (int16_t * in, int8_t * krn, uint32_t width, uint32_t height, uint32_t in_row_step, uint32_t kern_row_step);

mli_status
mli_krn_depthwise_conv2d_chw_fx8w16d (const mli_tensor * in, const mli_tensor * weights, const mli_tensor * bias, const mli_conv2d_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // Extract general conv2D parameters
    uint8_t stride_width = cfg->stride_width;
    uint8_t stride_height = cfg->stride_height;
    uint8_t padding_top = cfg->padding_top;
    uint8_t padding_bot = cfg->padding_bottom;
    uint8_t padding_left = cfg->padding_left;
    uint8_t padding_right = cfg->padding_right;

    int16_t val_min_limit;
    int16_t val_max_limit;

    // Define output val limits - we need it in case built-in RELU
    switch (cfg->relu.type) {
    case MLI_RELU_GEN:
        val_min_limit = 0;
        val_max_limit = INT16_MAX;
        break;
    case MLI_RELU_6:
        val_min_limit = 0;
        val_max_limit = MIN (6 << (int) out->el_params.fx.frac_bits, INT16_MAX);
        break;
    case MLI_RELU_1:
        if (out->el_params.fx.frac_bits >= sizeof (int16_t)) {
            val_min_limit = -(1 << out->el_params.fx.frac_bits);
            val_max_limit = (1 << out->el_params.fx.frac_bits);
            break;
        }
        // Else:For RELU_1 we use default branch (equal NO_ReLU)
    default:
        val_min_limit = INT16_MIN;
        val_max_limit = INT16_MAX;
    }

    // Data pointers
    MLI_PTR(int16_t) in_ftrs = (MLI_PTR(int16_t))in->data;
    MLI_PTR(int16_t) out_ftrs = (MLI_PTR(int16_t))out->data;
    MLI_PTR(int8_t) wt = (MLI_PTR(int8_t))weights->data;
    MLI_PTR(int8_t) bs = (MLI_PTR(int8_t))bias->data;

    // Define Data dimensions
    uint16_t channels = in->shape[0];

    uint16_t kernel_height = weights->shape[2];
    uint16_t kernel_width = weights->shape[3];

    uint16_t in_height = in->shape[1];
    uint16_t in_width = in->shape[2];

    uint16_t out_width = (in_width + padding_left + padding_right - kernel_width + 1);
    out_width = (out_width % stride_width != 0) ? (out_width / stride_width + 1) : out_width / stride_width;

    uint16_t out_height = (in_height + padding_top + padding_bot - kernel_height + 1);
    out_height = (out_height % stride_height != 0) ? (out_height / stride_height + 1) : out_height / stride_height;

    // Define shift values
    uint8_t bias_shift = (in->el_params.fx.frac_bits + weights->el_params.fx.frac_bits) - bias->el_params.fx.frac_bits;
    uint8_t out_shift = (in->el_params.fx.frac_bits + weights->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in_height >= kernel_height && in_width >= kernel_width) {
        rect_t cent_area;
        cent_area.row_beg = (padding_top % stride_height != 0) ? (padding_top / stride_height + 1) : padding_top / stride_height;
        cent_area.row_end = out_height - ((padding_bot % stride_height != 0) ? (padding_bot / stride_height + 1) : padding_bot / stride_height);
        cent_area.clmn_beg = (padding_left % stride_width != 0) ? (padding_left / stride_width + 1) : padding_left / stride_width;
        cent_area.clmn_end = out_width - ((padding_right % stride_width != 0) ? (padding_right / stride_width + 1) : padding_right / stride_width);
        for (uint32_t ch_idx = 0; ch_idx < channels; ch_idx++) {
            for (uint32_t H_idx = cent_area.row_beg; H_idx < cent_area.row_end; H_idx++) {
                for (uint32_t W_idx = cent_area.clmn_beg; W_idx < cent_area.clmn_end; W_idx++) {
                    int32_t conv_out = (bs[ch_idx] << bias_shift);
                    // Define area of input and filter for convolution
                    int16_t *in_ptr = in_ftrs + // starting point
                        in_width * in_height * ch_idx + // move to channels
                        in_width * (H_idx * stride_height - padding_top) +  // move to row
                        (W_idx * stride_width - padding_left);  // move to column

                    int8_t *w_ptr = wt +    // Start point
                        ch_idx * kernel_width * kernel_height;  // move to filter

                    // Convolution core
                    conv_out += dotprod2D (in_ptr, w_ptr, kernel_width, kernel_height, in_width, kernel_width);

                    // Write results
                    MLI_PTR(int16_t) o_ptr = &out_ftrs[ch_idx * out_width * out_height + H_idx * out_width + W_idx];
                    mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);
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
            for (uint32_t ch_idx = 0; ch_idx < channels; ch_idx++) {
                for (uint32_t H_idx = perc_areas[area_idx].row_beg; H_idx < perc_areas[area_idx].row_end; H_idx++) {
                    for (uint32_t W_idx = perc_areas[area_idx].clmn_beg; W_idx < perc_areas[area_idx].clmn_end; W_idx++) {
                        // Define area of input and filter for convolution
                        // *_comp - compensation values for valid area defining
                        int32_t top_comp = -MIN ((int32_t) (H_idx * stride_height) - padding_top, 0);
                        int32_t left_comp = -MIN ((int32_t) (W_idx * stride_width) - padding_left, 0);

                        int32_t right_comp = -MIN ((int32_t) in_width - ((int32_t) (W_idx * stride_width) - padding_left + kernel_width), 0);
                        int32_t bottom_comp = -MIN ((int32_t) in_height - ((int32_t) (H_idx * stride_height) - padding_top + kernel_height), 0);

                        int32_t rows = kernel_height - top_comp - bottom_comp;
                        int32_t clmns = kernel_width - right_comp - left_comp;

                        int32_t conv_out = (bs[ch_idx] << bias_shift);

                        int16_t *in_ptr = in_ftrs + // starting point
                            in_width * in_height * ch_idx + // move to channels
                            in_width * (H_idx * stride_height - padding_top + top_comp) +   // move to row
                            (W_idx * stride_width) - padding_left + left_comp;  // move to column

                        int8_t *w_ptr = wt +    // Start point
                            ch_idx * kernel_width * kernel_height + // move to filter
                            top_comp * kernel_width +   // move to row
                            left_comp;  // move to column

                        // Convolution core
                        conv_out += dotprod2D (in_ptr, w_ptr, clmns, rows, in_width, kernel_width);

                        // Write result
                        MLI_PTR(int16_t) o_ptr = &out_ftrs[ch_idx * out_width * out_height + H_idx * out_width + W_idx];
                        mli_prv_clip_relu_store_output (o_ptr, conv_out, out_shift, val_min_limit, val_max_limit);

                    }
                }
            }
        }

    }
    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[0] = channels;
    out->shape[1] = out_height;
    out->shape[2] = out_width;

    return MLI_STATUS_OK;
}

static inline int32_t
dotprod2D (int16_t * in, int8_t * krn, uint32_t width, uint32_t height, uint32_t in_row_step, uint32_t kern_row_step) {
    int32_t accu = 0;
    in_row_step -= width;
    kern_row_step -= width;
    for (uint32_t row = 0; row < height; row++) {
        for (uint32_t clmn = 0; clmn < width; clmn++) {
            accu += (*in++) * (*krn++);
        }
        in += in_row_step;
        krn += kern_row_step;
    }
    return accu;
}
#pragma code()
#ifdef __cplusplus
}
#endif
