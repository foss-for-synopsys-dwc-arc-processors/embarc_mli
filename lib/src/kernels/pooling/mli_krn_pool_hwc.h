/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_POOL_HWC_H_
#define _MLI_KRN_POOL_HWC_H_

#include "mli_config.h"
#include "mli_krn_avepool_hwc.h"
#include "mli_krn_reduce_max2d.h"
#include "mli_math.h"
#include "mli_math_macros.h"
#include "mli_prv_quant.h"
#include "mli_prv_dsp.h"
#include "mli_private_types.h"
#include "mli_prv_layout.h"

namespace mli {
namespace krn {

#define POOL_FIXED_KRN_SIZE_3   (3)
#define POOL_FIXED_KRN_SIZE_2   (2)
#define POOL_NO_FIXED_KRN_SIZE  (0)

typedef enum {
    AVEPOOL,
    MAXPOOL
} pool_type;

template <pool_type type, typename io_T, int fixed_kernel_size, bool convert = false>
static MLI_FORCE_INLINE void mli_krn_pool_hwc_nopad(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const int stride_width,
        const int stride_height,
        const int padding_top,
        const int padding_bot,
        const int padding_left,
        const int padding_right,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int kernel_height,
        const int kernel_width,
        const s8asym_quant_params *params = nullptr) {

    auto input = mli_prv_load_1vec(in.ptr);
    const int num_lanes = get_number_lanes(input);

    int shift_value = params->shift;
    int16_t mul = 0;
    int32_t zp = params->offset;

    if (type == AVEPOOL) {
        MLI_ASSERT(params != nullptr);
        mli::krn::get_mul_shift_value(kernel_width * kernel_height, &mul, &shift_value);
        if (convert) {
            int norm_shift;
#ifdef AVEPOOL_16BIT_MUL
            mul = mli_math_norm_cast_fx<int32_t,int16_t>(
                                        mli_math_mul_fx<int16_t, int32_t>(params->scale, mul), &norm_shift);
#else
            mul = mli_math_norm_cast_fx<int32_t,int8_t>(
                                        mli_math_mul_fx<int16_t, int32_t>(params->scale, mul), &norm_shift);
#endif
            shift_value -= norm_shift;
        } else {
            MLI_ASSERT(params->offset == 0);
            MLI_ASSERT(params->scale  == 1);
        }
    }

    const int h_idx_in = (row_beg * stride_height - padding_top);
    const int w_idx_in = (clmn_beg * stride_width - padding_left);

    const int in_col_inc = in.col_mem_stride * stride_width;
    const int in_row_inc = in.row_mem_stride * stride_height - (in_col_inc * (clmn_end - clmn_beg));
    const int out_col_inc = out.col_mem_stride;
    const int out_row_inc = out.row_mem_stride - out_col_inc * (clmn_end - clmn_beg);

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in.height >= kernel_height && in.width >= kernel_width) {
        for (int ch_idx = 0; ch_idx < in.ch; ch_idx += num_lanes) {
            int remaining_ch = in.ch - ch_idx;
            int current_ch = MIN(remaining_ch, num_lanes); /* nr channels computed in this loop iteration */
            // Define area of input and filter for pooling
            const MLI_PTR(io_T) __restrict in_ptr = in.ptr
                                                    + in.row_mem_stride * h_idx_in
                                                    + in.col_mem_stride * w_idx_in
                                                    + ch_idx;

            MLI_OUT_PTR(io_T) __restrict out_ptr = out.ptr
                                                    + out.row_mem_stride * row_beg
                                                    + out.col_mem_stride * clmn_beg
                                                    + ch_idx;

            for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
                if (type == AVEPOOL) {
                    mli::krn::compute_avepool<io_T, fixed_kernel_size, /*varying_kernel*/ false>(
                                                in_ptr, out_ptr, mul, kernel_width, kernel_height,
                                                in.col_mem_stride, in.row_mem_stride, zp, shift_value, 
                                                current_ch);
                } else if (type == MAXPOOL) {
                    mli::krn::reduce_max2D_hwc<io_T, fixed_kernel_size, /*varying_kernel*/ false>(
                                                in_ptr, out_ptr, kernel_width, kernel_height, 
                                                in.col_mem_stride, in.row_mem_stride, current_ch);
                }

                    in_ptr += in_col_inc;
                    out_ptr += out_col_inc;
                }
                in_ptr += in_row_inc;
                out_ptr += out_row_inc;
            }
        }
    }
}

template <pool_type type, typename io_T, int fixed_kernel_size, bool convert = false>
static MLI_FORCE_INLINE void mli_krn_pool_hwc_compute_pad(
        const int stride_width,
        const int stride_height,
        const int padding_top,
        const int padding_bot,
        const int padding_left,
        const int padding_right,
        const rect_t &perception_area,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int kernel_height,
        const int kernel_width,
        const s8asym_quant_params *params = nullptr) {

    auto input = mli_prv_load_1vec(in.ptr);
    const int num_lanes = get_number_lanes(input);

    const int row_beg = perception_area.row_beg;
    const int row_end = perception_area.row_end;
    const int clmn_beg = perception_area.clmn_beg;
    const int clmn_end = perception_area.clmn_end;

    for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
        for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
            // comp - compensation values for valid area definition
            const mli_compensations comp = mli_prv_valid_area_compensations(
                    H_idx, W_idx, in.height, in.width,
                    kernel_height, kernel_width,
                    stride_height, stride_width, padding_left, padding_top,
                    1, 1);

            const int rows = kernel_height - comp.kernel_top - comp.kernel_bottom;
            const int clmns = kernel_width - comp.kernel_right - comp.kernel_left;
            const int h_idx_in = (H_idx * stride_height - padding_top + comp.in_top);
            const int w_idx_in = (W_idx * stride_width - padding_left + comp.in_left);

            int shift_value = params->shift;
            int16_t mul = 0;
            int32_t zp = params->offset;

            if (type == AVEPOOL) { 
                MLI_ASSERT(params != nullptr);
                mli::krn::get_mul_shift_value(rows * clmns, &mul, &shift_value);
                if (convert) {
                    int norm_shift;
#ifdef AVEPOOL_16BIT_MUL
                    mul = mli_math_norm_cast_fx<int32_t,int16_t>(
                                        mli_math_mul_fx<int16_t, int32_t>(params->scale, mul), &norm_shift);
#else
                    mul = mli_math_norm_cast_fx<int32_t,int8_t>(
                                                            mli_math_mul_fx<int16_t, int32_t>(params->scale, mul), &norm_shift);
#endif
                    shift_value -= norm_shift;
                } else {
                    MLI_ASSERT(params->offset == 0);
                    MLI_ASSERT(params->scale  == 1);
                }
            }

            for (int ch_idx = 0; ch_idx < in.ch; ch_idx += num_lanes) {
                int remaining_ch = in.ch - ch_idx;
                int current_ch = MIN(remaining_ch, num_lanes); /* nr channels computed in this loop iteration */
                
                // Define area of input and filter for pooling
                const MLI_PTR(io_T) in_ptr = in.ptr
                                           + in.row_mem_stride * h_idx_in // move to row
                                           + in.col_mem_stride * w_idx_in // move to column
                                           + ch_idx;                      // move to channel

                MLI_OUT_PTR(io_T) out_ptr = out.ptr
                                          + out.row_mem_stride * H_idx
                                          + out.col_mem_stride * W_idx
                                          + ch_idx;

                if (type == AVEPOOL) {
                    mli::krn::compute_avepool<io_T, fixed_kernel_size, /*varying_kernel*/ true>(
                                                in_ptr, out_ptr, mul, clmns, rows,
                                                in.col_mem_stride, in.row_mem_stride, zp, shift_value, 
                                                current_ch);
                } else if (type == MAXPOOL) {
                    mli::krn::reduce_max2D_hwc<io_T, fixed_kernel_size, /*varying_kernel*/ true>(
                                                in_ptr, out_ptr, clmns, rows, 
                                                in.col_mem_stride, in.row_mem_stride, current_ch);
                }
            }
        }
    }
}

template <pool_type type, typename io_T, int fixed_kernel_size, bool convert = false>
static MLI_FORCE_INLINE void mli_krn_pool_hwc_pad(
        const int stride_width,
        const int stride_height,
        const int padding_top,
        const int padding_bot,
        const int padding_left,
        const int padding_right,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int kernel_height,
        const int kernel_width,
        const s8asym_quant_params *params = nullptr) {

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    const int nopad_row_beg = CEIL_DIV(padding_top, stride_height);
    const int nopad_row_end = CEIL_DIV(in.height + padding_top - kernel_height + 1, stride_height);
    const int nopad_clmn_beg = CEIL_DIV(padding_left, stride_width);
    const int nopad_clmn_end = CEIL_DIV(in.width + padding_left - kernel_width + 1, stride_width);

    if ((nopad_row_end - nopad_row_beg > 0) && (nopad_clmn_end - nopad_clmn_beg > 0)) {
        mli_krn_pool_hwc_nopad<type, io_T, fixed_kernel_size, convert>(
                nopad_row_beg, nopad_row_end, nopad_clmn_beg, nopad_clmn_end,
                stride_width, stride_height, padding_top,
                padding_bot, padding_left, padding_right,
                in, out,
                kernel_height, kernel_width, params);
    }
    // Phase 2: Process border part with more complex algorithm
    // (usually significantly smaller part of computations)
    //=======================================================================
    if (padding_top || padding_left || padding_bot || padding_right) {
        rect_t perc_areas[4];
        int areas_num = 0;
        if (padding_top) {
            perc_areas[areas_num].row_beg = 0;
            perc_areas[areas_num].row_end = nopad_row_beg;
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = out.width;
        }
        if (padding_bot) {
            perc_areas[areas_num].row_beg = nopad_row_end;
            perc_areas[areas_num].row_end = out.height;
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = out.width;
        }
        if (padding_left) {
            perc_areas[areas_num].row_beg = nopad_row_beg;
            perc_areas[areas_num].row_end = nopad_row_end;
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = nopad_clmn_beg;
        }
        if (padding_right) {
            perc_areas[areas_num].row_beg = nopad_row_beg;
            perc_areas[areas_num].row_end = nopad_row_end;
            perc_areas[areas_num].clmn_beg = nopad_clmn_end;
            perc_areas[areas_num++].clmn_end = out.width;
        }


        for (int area_idx = 0; area_idx < areas_num; ++area_idx) {
            mli_krn_pool_hwc_compute_pad<type, io_T, fixed_kernel_size, convert>(stride_width, stride_height,
                                            padding_top, padding_bot, padding_left, padding_right,
                                            perc_areas[area_idx],
                                            in, out,
                                            kernel_height, kernel_width, params);
        }
    }
}

template <pool_type type, typename io_T, int fixed_kernel_size, bool convert = false>
static MLI_FORCE_INLINE void mli_krn_pool_hwc_wrapper(
        MLI_PTR(io_T) __restrict in_ptr,
        MLI_OUT_PTR(io_T) __restrict out_ptr,
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const int stride_width,
        const int stride_height,
        const int padding_top,
        const int padding_bot,
        const int padding_left,
        const int padding_right,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int kernel_height,
        const int kernel_width,
        const s8asym_quant_params *params = nullptr) {

    tensor_private_t<MLI_PTR(io_T)> in_ = in;
    tensor_private_t<MLI_OUT_PTR(io_T)> out_ = out;
    in_.ptr = in_ptr;
    out_.ptr = out_ptr;

    if (padding_top || padding_left || padding_bot || padding_right) {
        mli_krn_pool_hwc_pad<type, io_T, fixed_kernel_size, convert>(stride_width, stride_height,
                                padding_top, padding_bot, padding_left, padding_right,
                                in_, out_,
                                kernel_height, kernel_width, params);
    } else {
        mli_krn_pool_hwc_nopad<type, io_T, fixed_kernel_size, convert>(
                                /*row_beg*/ 0, row_end, /*clmn_beg*/ 0, clmn_end,
                                stride_width, stride_height,
                                /*padding_top*/ 0, /*padding_bot*/ 0, /*padding_left*/ 0, /*padding_right*/ 0,
                                in_, out_,
                                kernel_height, kernel_width, params);
    }
}

template <pool_type type, typename io_T, int fixed_kernel_size, bool convert = false>
static void mli_krn_pool_hwc(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    // Extract general avepool parameters
    int32_t stride_width = cfg->stride_width;
    int32_t stride_height = cfg->stride_height;
    int32_t padding_top = cfg->padding_top;
    int32_t padding_bot = cfg->padding_bottom;
    int32_t padding_left = cfg->padding_left;
    int32_t padding_right = cfg->padding_right;
    int32_t kernel_height = cfg->kernel_height;
    int32_t kernel_width = cfg->kernel_width;

    // Define Data dimensions
    auto in_prv = mli_prv_get_tensor_hwc<MLI_PTR(io_T)>(in,
            0); // channels

    if (fixed_kernel_size) {
        MLI_CHECK_AND_FIX(kernel_width, fixed_kernel_size);
        MLI_CHECK_AND_FIX(kernel_height, fixed_kernel_size);
    }

    // Define data dimensions
    const int32_t out_width = CEIL_DIV(in_prv.width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int32_t out_height = CEIL_DIV(in_prv.height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // Fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_H_DIM_HWC] = out_height;
    out->shape[FMAP_W_DIM_HWC] = out_width;
    out->shape[FMAP_C_DIM_HWC] = in_prv.ch;
    
    s8asym_quant_params params;
    if (type == AVEPOOL) {
        if (convert) {
            MLI_ASSERT(in->el_type == MLI_EL_SA_8);
            define_requant_params(in, out, &params);
        } else {
            MLI_ASSERT(in->el_type == MLI_EL_FX_8 || in->el_type == MLI_EL_FX_16);
            params.shift  = in->el_params.fx.frac_bits - out->el_params.fx.frac_bits;
            params.offset = 0;
            params.scale  = 1;
        }
    } else if (type == MAXPOOL) {
        out->el_params = in->el_params;
    }

    const auto out_prv = mli_prv_get_tensor_hwc<MLI_OUT_PTR(io_T)>(out);
    const int32_t row_beg = 0;
    const int32_t row_end = out_height;
    const int32_t clmn_beg = 0;
    const int32_t clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();


    mli_krn_pool_hwc_wrapper<type, io_T, fixed_kernel_size, convert>(in_prv.ptr, out_prv.ptr,
                            row_beg, row_end, clmn_beg, clmn_end,
                            stride_width, stride_height,
                            padding_top, padding_bot, padding_left, padding_right,
                            in_prv, out_prv,
                            kernel_height, kernel_width, &params);

}

} // krn
} // mli

#endif // _MLI_KRN_POOL_HWC_H_
