/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_MAXPOOL_HWC_H_
#define _MLI_KRN_MAXPOOL_HWC_H_

#include "mli_config.h"
#include "mli_math_macros.h"
#include "mli_private_types.h"
#include "mli_krn_reduce_max2d.h"

////////////////////////////////////////////////////////////////////////////////
// Setting up namespace
////////////////////////////////////////////////////////////////////////////////
namespace mli {
namespace krn {

#define MAXPOOL_FIXED_KRN_SIZE_3 3
#define MAXPOOL_FIXED_KRN_SIZE_2 2
#define MAXPOOL_NO_FIXED_KRN_SIZE 0

template <typename io_T, int fixed_kernel_size>
static MLI_FORCE_INLINE void mli_krn_maxpool_hwc_nopad(
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
        const int kernel_width) {

    bool fixed_size = fixed_kernel_size == MAXPOOL_NO_FIXED_KRN_SIZE ? false : true;

    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(in.ptr);
    const int num_lanes = get_number_lanes(input);
    const int remaining_chans = in.ch & (num_lanes - 1);

    const int h_idx_in = (row_beg * stride_height - padding_top);
    const int w_idx_in = (clmn_beg * stride_width - padding_left);

    const int in_col_inc = in.col_mem_stride * stride_width;
    const int in_row_inc = in.row_mem_stride * stride_height - (in_col_inc * (clmn_end - clmn_beg));
    const int out_col_inc = out.col_mem_stride;
    const int out_row_inc = out.row_mem_stride - out_col_inc * (clmn_end - clmn_beg);

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in.height >= kernel_height && in.width >= kernel_width) {
            for (int ch_idx = 0; ch_idx < in.ch - remaining_chans; ch_idx += num_lanes) {
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
// TODO unroll in case of MAXPOOL_FIXED_KRN_SIZE_2
// #pragma clang loop unroll_count(2)
                    for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
                        // Core Max
                        mli::krn::reduce_max2D_hwc_v(in_ptr, out_ptr, kernel_width, kernel_height,
                                in.col_mem_stride, in.row_mem_stride, fixed_size);

                        in_ptr += in_col_inc;
                        out_ptr += out_col_inc;
                    }
                    in_ptr += in_row_inc;
                    out_ptr += out_row_inc;
                }
            }
        if (remaining_chans) {
            // Define area of input and filter for pooling
            const MLI_PTR(io_T) __restrict in_ptr = in.ptr
                                                  + in.row_mem_stride * h_idx_in
                                                  + in.col_mem_stride * w_idx_in
                                                  + in.ch - remaining_chans;

            MLI_OUT_PTR(io_T) __restrict out_ptr = out.ptr
                                                 + out.row_mem_stride * row_beg
                                                 + out.col_mem_stride * clmn_beg
                                                 + out.ch - remaining_chans;

            for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
                    // Core Max
                    mli::krn::reduce_max2D_hwc(in_ptr, out_ptr, kernel_width, kernel_height, remaining_chans,
                        in.col_mem_stride, in.row_mem_stride, fixed_size);

                    in_ptr += in_col_inc;
                    out_ptr += out_col_inc;
                }
                in_ptr += in_row_inc;
                out_ptr += out_row_inc;
            }
        }
    }
}

template <typename io_T, int fixed_kernel_size>
static MLI_FORCE_INLINE void mli_krn_maxpool_hwc_pad(
        int row_beg,
        int row_end,
        int clmn_beg,
        int clmn_end,
        int stride_width,
        int stride_height,
        int padding_top,
        int padding_bot,
        int padding_left,
        int padding_right,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        int kernel_height,
        int kernel_width) {

    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(in.ptr);
    int num_lanes = get_number_lanes(input);
    int remaining_chans = in.ch & (num_lanes - 1);

    MLI_OUT_PTR(io_T) __restrict out_ptr;

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    row_beg = CEIL_DIV(padding_top, stride_height);
    row_end = out.height - CEIL_DIV(padding_bot, stride_height);
    clmn_beg = CEIL_DIV(padding_left, stride_width);
    clmn_end = out.width - CEIL_DIV(padding_right, stride_width);

    if ((row_end - row_beg > 0) && (clmn_end - clmn_beg > 0)) {
        mli_krn_maxpool_hwc_nopad<io_T, fixed_kernel_size>(
                row_beg, row_end, clmn_beg, clmn_end,
                stride_width, stride_height, padding_top,
                padding_bot, padding_left, padding_right,
                in, out,
                kernel_height, kernel_width);
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


        for (int area_idx = 0; area_idx < areas_num; ++area_idx) {
            for (int H_idx = perc_areas[area_idx].row_beg; H_idx < (int)perc_areas[area_idx].row_end; H_idx++) {
                for (int W_idx = perc_areas[area_idx].clmn_beg; W_idx < (int)perc_areas[area_idx].clmn_end; W_idx++) {
                    for (int ch_idx = 0; ch_idx < in.ch - remaining_chans; ch_idx += num_lanes) {
                        // Define area of input and filter for pooling
                        // *_comp - compensation values for valid area defining
                        int top_comp = -MIN((H_idx * stride_height) - padding_top, 0);
                        int left_comp = -MIN((W_idx * stride_width) - padding_left, 0);

                        int right_comp = -MIN(in.width - ((W_idx * stride_width) - padding_left + kernel_width), 0);
                        int bottom_comp = -MIN(in.height - ((H_idx * stride_height) - padding_top + kernel_height), 0);

                        int rows = kernel_height - top_comp - bottom_comp;
                        int clmns = kernel_width - right_comp - left_comp;

                        // Define area of input and filter for pooling
                        const MLI_PTR(io_T) in_ptr = in.ptr
                                + in.row_mem_stride * (H_idx * stride_height - padding_top + top_comp)  // move to row
                                + in.col_mem_stride * (W_idx * stride_width - padding_left + left_comp) // move to column
                                + ch_idx;                                                               // move to channel

                        out_ptr = &out.ptr[out.row_mem_stride * H_idx +
                                           out.col_mem_stride * W_idx +
                                           ch_idx];

                        // Core Max
                        mli::krn::reduce_max2D_hwc_v(in_ptr, out_ptr, clmns, rows,
                                in.col_mem_stride, in.row_mem_stride, false);
                    }
                }
            }
            if (remaining_chans) {
                for (int H_idx = perc_areas[area_idx].row_beg; H_idx < (int)perc_areas[area_idx].row_end; H_idx++) {
                    for (int W_idx = perc_areas[area_idx].clmn_beg; W_idx < (int)perc_areas[area_idx].clmn_end; W_idx++) {
                        // Define area of input and filter for pooling
                        // *_comp - compensation values for valid area defining
                        int top_comp = -MIN((H_idx * stride_height) - padding_top, 0);
                        int left_comp = -MIN((W_idx * stride_width) - padding_left, 0);

                        int right_comp = -MIN(in.width - ((W_idx * stride_width) - padding_left + kernel_width), 0);
                        int bottom_comp = -MIN(in.height - ((H_idx * stride_height) - padding_top + kernel_height), 0);

                        int rows = kernel_height - top_comp - bottom_comp;
                        int clmns = kernel_width - right_comp - left_comp;

                        const MLI_PTR(io_T) in_ptr = in.ptr
                                + in.row_mem_stride * (H_idx * stride_height - padding_top + top_comp)  // move to row
                                + in.col_mem_stride * (W_idx * stride_width - padding_left + left_comp) // move to column
                                + in.ch -  remaining_chans;                                             // move to channel

                        out_ptr = &out.ptr[out.row_mem_stride * H_idx +
                                           out.col_mem_stride * W_idx +
                                           out.ch - remaining_chans];

                        // Core Max
                        mli::krn::reduce_max2D_hwc(in_ptr, out_ptr, clmns, rows, remaining_chans,
                                in.col_mem_stride, in.row_mem_stride, false);
                    }
                }
            }
        }
    }
}

template <typename io_T, int fixed_kernel_size>
static MLI_FORCE_INLINE void mli_krn_maxpool_hwc_wrapper(
        MLI_PTR(io_T) __restrict in_ptr,
        MLI_OUT_PTR(io_T) __restrict out_ptr,
        int row_beg,
        int row_end,
        int clmn_beg,
        int clmn_end,
        int stride_width,
        int stride_height,
        int padding_top,
        int padding_bot,
        int padding_left,
        int padding_right,
        tensor_private_t<MLI_PTR(io_T)> &in,
        tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        int kernel_height,
        int kernel_width) {

    tensor_private_t<MLI_PTR(io_T)> in_ = in;
    tensor_private_t<MLI_OUT_PTR(io_T)> out_ = out;
    in_.ptr = in_ptr;
    out_.ptr = out_ptr;

    if (padding_top || padding_left || padding_bot || padding_right) {
        mli_krn_maxpool_hwc_pad<io_T, fixed_kernel_size>(row_beg, row_end, clmn_beg, clmn_end,
                                stride_width, stride_height,
                                padding_top, padding_bot, padding_left, padding_right,
                                in_, out_,
                                kernel_height, kernel_width);
    } else {
        mli_krn_maxpool_hwc_nopad<io_T, fixed_kernel_size>(/*row_beg*/ 0, row_end, /*clmn_beg*/ 0, clmn_end,
                                stride_width, stride_height,
                                /*padding_top*/ 0, /*padding_bot*/ 0, /*padding_left*/ 0, /*padding_right*/ 0,
                                in_, out_,
                                kernel_height, kernel_width);
    }
}

template <typename io_T, int fixed_kernel_size>
static void mli_krn_maxpool_hwc(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    // Extract general maxpool parameters
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

    // Define Data dimensions
    const int32_t out_width = CEIL_DIV(in_prv.width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int32_t out_height = CEIL_DIV(in_prv.height + padding_top + padding_bot - kernel_height + 1, stride_height);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->shape[FMAP_H_DIM_HWC] = out_height;
    out->shape[FMAP_W_DIM_HWC] = out_width;
    out->shape[FMAP_C_DIM_HWC] = in_prv.ch;
    out->el_params = in->el_params;

    auto out_prv = mli_prv_get_tensor_hwc<MLI_OUT_PTR(io_T)>(out);
    const int32_t row_beg = 0;
    const int32_t row_end = out_height;
    const int32_t clmn_beg = 0;
    const int32_t clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    mli_krn_maxpool_hwc_wrapper<io_T, fixed_kernel_size>(in_prv.ptr, out_prv.ptr,
                            row_beg, row_end, clmn_beg, clmn_end,
                            stride_width, stride_height,
                            padding_top, padding_bot, padding_left, padding_right,
                            in_prv, out_prv,
                            kernel_height, kernel_width);
}

} // krn
} // mli

#endif // _MLI_KRN_MAXPOOL_HWC_H_
