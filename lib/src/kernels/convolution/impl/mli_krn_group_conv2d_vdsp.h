/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_GROUP_CONV2D_VDSP_H_
#define _MLI_KRN_GROUP_CONV2D_VDSP_H_

#include "mli_api.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"
#include "mli_math.h"
#include "mli_private_types.h"
#include "mli_types.h"
#include "mli_krn_dotprod.h"
#include "mli_prv_layout.h"
#include "mli_krn_convolution.h"

namespace mli {
namespace krn {
namespace vdsp {

#pragma MLI_CODE_SECTION_START(".mli_lib")

MLI_FORCE_INLINE s8asym_quant_specific_params quant_params_offset(s8asym_quant_specific_params &params,
		int offset) {
    s8asym_quant_specific_params params_prv = params;
    params_prv.weight_scales = params.weight_scales + offset;
    params_prv.weight_shifts = params.weight_shifts + offset;
    return params_prv;
}

MLI_FORCE_INLINE fx_quant_specific_params quant_params_offset(fx_quant_specific_params &in,
		int offset) {
    // No need to adjust something during calculations for MLI_FX specific quantization
    return in;
}

//========================================================
// Group Convolution 2D without padding
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void group_convolution2D_nopad(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &weights,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        quant_T quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    const int group_count = in.ch / weights.in_ch;
    const int filters_per_group = weights.out_ch / group_count;

    tensor_private_t<MLI_CONV_OUT_PTR(io_T)> out_prv = out;
    tensor_private_t<MLI_CONV_OUT_PTR(io_T)> in_prv = in;
    conv2d_weights_tensor_private_t<MLI_PTR(w_T)> weights_prv = weights;
    const MLI_PTR(b_T)  __restrict biases_prv = biases;
    out_prv.ch = filters_per_group;
    in_prv.ch = weights.in_ch;

    for (int M = 0; M < group_count; M++) {

        int group_offset = M * filters_per_group;
        in_prv.ptr = in.ptr + in.ch_mem_stride * weights.in_ch * M;
        out_prv.ptr = out.ptr + out.ch_mem_stride * group_offset;
        weights_prv.ptr = weights.ptr + weights.out_ch_mem_stride * group_offset;
        biases_prv = biases + group_offset;
        quant_T quant_params_prv = quant_params_offset(quant_params, group_offset);

        mli::krn::vdsp::convolution2D_nopad<io_T, w_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
                in_prv, weights_prv, biases_prv, out_prv, perception_area, quant_params_prv,
                val_min_limit, val_max_limit,
                stride_height, stride_width,
                dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);

    } // for M (group_count)
}

//========================================================
// Group Convolution 2D with padding
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void group_convolution2D_pad(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &weights,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        quant_T quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    const int group_count = in.ch / weights.in_ch;
    const int filters_per_group = weights.out_ch / group_count;

    tensor_private_t<MLI_CONV_OUT_PTR(io_T)> out_prv = out;
    tensor_private_t<MLI_CONV_OUT_PTR(io_T)> in_prv = in;
    conv2d_weights_tensor_private_t<MLI_PTR(w_T)> weights_prv = weights;
    const MLI_PTR(b_T)  __restrict biases_prv = biases;
    out_prv.ch = filters_per_group;
    in_prv.ch = weights.in_ch;

    for (int M = 0; M < group_count; M++) {

        int group_offset = M * filters_per_group;
        in_prv.ptr = in.ptr + in.ch_mem_stride * weights.in_ch * M;
        out_prv.ptr = out.ptr + out.ch_mem_stride * group_offset;
        weights_prv.ptr = weights.ptr + weights.out_ch_mem_stride * group_offset;
        biases_prv = biases + group_offset;
        quant_T quant_params_prv = quant_params_offset(quant_params, group_offset);

        mli::krn::vdsp::convolution2D_pad<io_T, w_T, b_T, acc_T, quant_T>(
                in_prv, weights_prv, biases_prv, out_prv, perception_area, quant_params_prv,
                val_min_limit, val_max_limit,
                stride_height, stride_width,
                dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);
    }
}

//========================================================
// Group Convolution 2D
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void group_convolution2D(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &weights,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        quant_T quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    MLI_ASSERT(quant_params_get_weigths_zeropoint(&quant_params) == 0); /* this optimized implementation assumes no zero offset for weights */

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    rect_t perception_area_nopad;
    perception_area_nopad.row_beg = CEIL_DIV(padding_top, stride_height);
    perception_area_nopad.row_end = out.height - CEIL_DIV(padding_bot, stride_height);
    perception_area_nopad.clmn_beg = CEIL_DIV(padding_left, stride_width);
    perception_area_nopad.clmn_end = out.width - CEIL_DIV(padding_right, stride_width);

    if ((perception_area_nopad.row_end - perception_area_nopad.row_beg > 0)
        && (perception_area_nopad.clmn_end - perception_area_nopad.clmn_beg > 0)){
        group_convolution2D_nopad<io_T, w_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
                in, weights, biases, out, perception_area_nopad, quant_params,
                val_min_limit, val_max_limit,
                stride_height, stride_width,
                dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);
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
            perc_areas[areas_num++].clmn_end = out.width;
        }
        if (padding_bot) {
            perc_areas[areas_num].row_beg = out.height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].row_end = out.height;
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = out.width;
        }
        if (padding_left) {
            perc_areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].row_end = out.height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = CEIL_DIV(padding_left, stride_width);
        }
        if (padding_right) {
            perc_areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].row_end = out.height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = out.width - CEIL_DIV(padding_right, stride_width);
            perc_areas[areas_num++].clmn_end = out.width;
        }
        for(int i = 0; i < areas_num; i ++) {
            group_convolution2D_pad<io_T, w_T, b_T, acc_T, quant_T>(
                    in, weights, biases, out, perc_areas[i], quant_params,
                    val_min_limit, val_max_limit,
                    stride_height, stride_width,
                    dilation_height, dilation_width,
                    padding_top, padding_left,
                    padding_bot, padding_right);
        }
    }
}

#pragma MLI_CODE_SECTION_END()
} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_GROUP_CONV2D_VDSP_H_
