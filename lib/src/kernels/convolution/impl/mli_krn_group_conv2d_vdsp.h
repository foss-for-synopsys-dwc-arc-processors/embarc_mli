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

namespace mli {
namespace krn {
namespace vdsp {

#pragma MLI_CODE_SECTION_START(".mli_lib")

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

    const int row_begin = perception_area.row_beg;
    const int row_end = perception_area.row_end;
    const int clmn_begin = perception_area.clmn_beg;
    const int clmn_end = perception_area.clmn_end;
    int width = clmn_end - clmn_begin;
    constexpr int numaccuregs = sizeof(acc_T) / sizeof(vNint_t);
    // for larger kernel sizes in combination with an unroll of 4, the gather load of the input samples
    // don't fit into one vector anymore. in those cases fall back to unroll 2
    constexpr int unroll = ((numaccuregs > 2) || (fix_kernel_width * fix_kernel_height * 4 > _VDSP_NUM_8BIT_LANES)) ? 2 : 4;
    int remainder_width = width & (unroll - 1);
    int unroll_width = width - remainder_width;

    const int group_count = in.ch / weights.in_ch;
    const int filters_per_group = weights.out_ch / group_count;

    const int rows = (fix_kernel_height > 0) ? fix_kernel_height : weights.kernel_height;
    const int clmns = (fix_kernel_width > 0) ? fix_kernel_width : weights.kernel_width;

    for (int M = 0; M < group_count; M++) {
        int out_ch_idx_beg = M * filters_per_group;
        int out_ch_idx_end = out_ch_idx_beg + filters_per_group;
        for (int out_ch_idx = out_ch_idx_beg; out_ch_idx < out_ch_idx_end; out_ch_idx += get_number_lanes<acc_T>()) {
            int remaining_ch = out_ch_idx_end - out_ch_idx;
            int current_ch = MIN(remaining_ch, get_number_lanes<acc_T>()); /* nr channels computed in this loop iteration */

            const MLI_PTR(w_T) w_ptr = weights.ptr
                    + weights.out_ch_mem_stride * out_ch_idx;

            auto output_params = adjust_quant_params_v(&quant_params, out_ch_idx);

            acc_T pre_accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
            pre_accu = mli::krn::bias_additive(&biases[out_ch_idx], pre_accu, &output_params);

            pre_accu = mli::krn::weights_additive(w_ptr, pre_accu, &quant_params, clmns, rows, weights.in_ch,
                    weights.col_mem_stride,
                    weights.row_mem_stride,
                    weights.in_ch_mem_stride);

            const int h_idx_in = row_begin * stride_height - padding_top;
            const int w_idx_in = clmn_begin * stride_width - padding_left;
            MLI_CONV_OUT_PTR(io_T) out_ptr = out.ptr
                    + out.row_mem_stride * row_begin
                    + out.col_mem_stride * clmn_begin
                    + out.ch_mem_stride * out_ch_idx;
            const MLI_PTR(io_T) in_ptr = in.ptr
                    + in.row_mem_stride * h_idx_in
                    + in.col_mem_stride * w_idx_in
                    + in.ch_mem_stride * weights.in_ch * (out_ch_idx / filters_per_group);

            int out_w_inc = out.col_mem_stride;
            int out_h_inc = out.row_mem_stride - width * out_w_inc;
            int in_w_inc = in.col_mem_stride * stride_width;
            int in_h_inc = in.row_mem_stride * stride_height - width * in_w_inc;

            for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
                int W_idx = clmn_begin;
                for (int W_cnt = 0; W_cnt < remainder_width; W_cnt++, W_idx++) {

                    acc_T accu = pre_accu;

                    if ((fix_kernel_width == 1) && (fix_kernel_height == 1)) {
                        accu = mli::krn::dotprod1D_v(in_ptr, w_ptr, accu, weights.in_ch, in.ch_mem_stride, weights.in_ch_mem_stride);
                    } else if ((fix_kernel_width > 0) && (fix_kernel_height > 0)) {
                        accu = mli::krn::dotprod3D_v<io_T, w_T, acc_T, /*fixedsize*/true>(in_ptr, w_ptr, clmns, rows, weights.in_ch,
                                  in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                                  weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                                  accu);
                    } else if (weights.row_mem_stride == clmns * weights.col_mem_stride) {
                        accu = mli::krn::dotprod3D_v_nopad(in_ptr, w_ptr, clmns, rows, weights.in_ch,
                                  in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                                  weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                                  accu);
                    } else {
                        accu = mli::krn::dotprod3D_v<io_T, w_T, acc_T, /*fixedsize*/false>(in_ptr, w_ptr, clmns, rows, weights.in_ch,
                                  in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                                  weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                                  accu);
                    }

                    // Cast result to output type, apply built-in ReLU Applying and write result
                    mli::krn::result_cast_relu_store_v(out_ptr, accu, &output_params, val_min_limit, val_max_limit, current_ch);

                    out_ptr += out_w_inc;
                    in_ptr += in_w_inc;
                } // for W_idx

                for (int W_cnt = 0; W_cnt < unroll_width; W_cnt+=unroll, W_idx+=unroll) {
                    auto accu = init_accu_grp(pre_accu);

                    if ((fix_kernel_width == 1) && (fix_kernel_height == 1)) {
                        accu = mli::krn::dotprod1D_v_unroll<unroll>(in_ptr, w_ptr, accu, weights.in_ch, in.ch_mem_stride, in.col_mem_stride, weights.in_ch_mem_stride);
                    } else if ((fix_kernel_width > 0) && (fix_kernel_height > 0) && (dilation_width == stride_width)) {
                        // unrolled version with fixed kernelsize
                        accu = mli::krn::dotprod3D_v_unroll<unroll, true>(in_ptr, w_ptr, clmns, rows, weights.in_ch,
                                  in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride, in_w_inc,
                                  weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                                  accu);
                    } else if (weights.row_mem_stride == clmns * weights.col_mem_stride) {
                        accu = mli::krn::dotprod3D_v_nopad_unroll<unroll>(in_ptr, w_ptr, clmns, rows, weights.in_ch,
                                  in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                                  weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride, in_w_inc,
                                  accu);
                    } else {
                        accu = mli::krn::dotprod3D_v_unroll<unroll, false>(in_ptr, w_ptr, clmns, rows, weights.in_ch,
                                  in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride, in_w_inc,
                                  weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                                  accu);
                    }

                    // Cast result to output type, apply built-in ReLU Applying and write result
                    mli::krn::result_cast_relu_store_v(out_ptr, accu.accu0, &output_params, val_min_limit, val_max_limit, current_ch);
                    out_ptr += out_w_inc;
                    mli::krn::result_cast_relu_store_v(out_ptr, accu.accu1, &output_params, val_min_limit, val_max_limit, current_ch);
                    if (unroll > 2) {
                        out_ptr += out_w_inc;
                        mli::krn::result_cast_relu_store_v(out_ptr, accu.accu2, &output_params, val_min_limit, val_max_limit, current_ch);
                        out_ptr += out_w_inc;
                        mli::krn::result_cast_relu_store_v(out_ptr, accu.accu3, &output_params, val_min_limit, val_max_limit, current_ch);
                    }

                    out_ptr += out_w_inc;
                    in_ptr += in_w_inc * unroll;
                } // for W_idx
                out_ptr += out_h_inc;
                in_ptr += in_h_inc;
            } // for H_idx
        } // for out_ch_idx
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

    const int row_begin = perception_area.row_beg;
    const int row_end = perception_area.row_end;
    const int clmn_begin = perception_area.clmn_beg;
    const int clmn_end = perception_area.clmn_end;

    const int group_count = in.ch / weights.in_ch;
    const int filters_per_group = weights.out_ch / group_count;

    for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
        for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
            // Define area of input and filter for convolution
            // comp - compensation values for valid area definition
            const mli_compensations comp = mli_prv_valid_area_compensations(
                    H_idx, W_idx, in.height, in.width,
                    weights.kernel_height, weights.kernel_width,
                    stride_height, stride_width, padding_left, padding_top,
                    dilation_height, dilation_width);

            const int rows = weights.kernel_height - comp.kernel_top - comp.kernel_bottom;
            const int clmns = weights.kernel_width - comp.kernel_right - comp.kernel_left;
            const int h_idx_in = (H_idx * stride_height - padding_top + comp.in_top);
            const int w_idx_in = (W_idx * stride_width - padding_left + comp.in_left);

            for (int M = 0; M < group_count; M++) {
                int out_ch_idx_beg = M * filters_per_group;
                int out_ch_idx_end = out_ch_idx_beg + filters_per_group;
                for (int out_ch_idx = out_ch_idx_beg; out_ch_idx < out_ch_idx_end; out_ch_idx += get_number_lanes<acc_T>()) {
                    int remaining_ch = out_ch_idx_end - out_ch_idx;
                    int current_ch = MIN(remaining_ch, get_number_lanes<acc_T>()); /* nr channels computed in this loop iteration */
                    MLI_CONV_OUT_PTR(io_T) out_ptr = out.ptr
                            + out.row_mem_stride * H_idx
                            + out.col_mem_stride * W_idx
                            + out.ch_mem_stride * out_ch_idx;

                    const MLI_PTR(io_T) in_ptr = in.ptr
                            + in.row_mem_stride * h_idx_in
                            + in.col_mem_stride * w_idx_in
                            + in.ch_mem_stride * weights.in_ch
                            * (out_ch_idx / filters_per_group);

                    const MLI_PTR(w_T) w_ptr = weights.ptr
                            + weights.row_mem_stride * comp.kernel_top
                            + weights.col_mem_stride * comp.kernel_left
                            + weights.out_ch_mem_stride * out_ch_idx;

                    auto output_params = adjust_quant_params_v(&quant_params, out_ch_idx);

                    acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);

                    accu = mli::krn::bias_additive(&biases[out_ch_idx], accu, &output_params);

                    accu = mli::krn::dotprod3D_v(in_ptr, w_ptr, clmns, rows, weights.in_ch,
                            in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, in.ch_mem_stride,
                            weights.col_mem_stride, weights.row_mem_stride, weights.in_ch_mem_stride,
                            accu);

                    accu = mli::krn::weights_additive(w_ptr, accu, &quant_params, clmns, rows, weights.in_ch,
                            weights.col_mem_stride,
                            weights.row_mem_stride,
                            weights.in_ch_mem_stride);

                    // Cast result to output type, apply built-in ReLU Applying and write result
                    mli::krn::result_cast_relu_store_v(out_ptr, accu, &output_params, val_min_limit, val_max_limit, current_ch);

                } // for out_ch_idx
            } // for M (group_count)
        } // for W_idx
    } // for H_idx
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
