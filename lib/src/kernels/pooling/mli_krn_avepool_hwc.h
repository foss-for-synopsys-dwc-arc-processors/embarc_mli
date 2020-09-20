/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_AVEPOOL_HWC_H_
#define _MLI_KRN_AVEPOOL_HWC_H_

#include "mli_config.h"
#include "mli_math_macros.h"
#include "mli_private_types.h"
#include "mli_krn_reduce_sum2d.h"
#include "mli_math.h"

////////////////////////////////////////////////////////////////////////////////
// Setting up namespace
////////////////////////////////////////////////////////////////////////////////
namespace mli {
namespace krn {

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
// TODO change after vectorized version is available
// #define CHANNEL_LANES     _VDSP_NUM_8BIT_LANES
#define CHANNEL_LANES 1
#elif !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
#define CHANNEL_LANES 2
#else
#define CHANNEL_LANES 1
#endif

#define AVEPOOL_FIXED_KRN_SIZE_3 3
#define AVEPOOL_FIXED_KRN_SIZE_2 2
#define AVEPOOL_NO_FIXED_KRN_SIZE 0

#define DIV_LUT_THRESHOLD 32
static const int16_t multiplier_lut[] = {
    0, // 0
    0x0001, // 1
    0x0001, // 2
    0x5555, // 3
    0x0001, // 4
    0x6666, // 5
    0x5555, // 6
    0x4924, // 7*
    0x0001, // 8
    0x71c7, // 9
    0x6666, // 10
    0x5d17, // 11
    0x5555, // 12
    0x4ec4, // 13*
    0x4924, // 14*
    0x4444, // 15
    0x0001, // 16
    0x7878, // 17
    0x71c7, // 18
    0x6bca, // 19
    0x6666, // 20
    0x6186, // 21
    0x5d17, // 22
    0x590b, // 23
    0x5555, // 24
    0x51eb, // 25*
    0x4ec4, // 26*
    0x4bda, // 27
    0x4924, // 28*
    0x469e, // 29*
    0x4444, // 30
    0x4210, // 31*
};

static const int8_t shift_lut[] = {
    0,  // 0
    0,  // 1
    1,  // 2
    16, // 3
    2,  // 4
    17, // 5
    17, // 6
    17, // 7
    3,  // 8
    18, // 9
    18, // 10
    18, // 11
    18, // 12
    18, // 13
    18, // 14
    18, // 15
    4,  // 16
    19, // 17
    19, // 18
    19, // 19
    19, // 20
    19, // 21
    19, // 22
    19, // 23
    19, // 24
    19, // 25
    19, // 26
    19, // 27
    19, // 28
    19, // 29
    19, // 30
    19, // 31
};

//====================================================================================
// Normalized scale multiplier (1/x) for average pooling
//====================================================================================
static inline void calc_mul(unsigned div, int16_t* mul, int* shift_val) {
    unsigned int one = (1<<31); // u1.31
    unsigned int val = one / div; // u1.31
    int shift_norm_val = 0;

    if (div > 1) { 
        shift_norm_val = mli_math_norm_fx<unsigned int, int>(val) + 1;
        val <<= shift_norm_val;
    }

    *mul = val >> 17;
    *shift_val = 14 + shift_norm_val;
}

static inline void get_mul_shift_value(
        unsigned div,
        int16_t* mul, int* shift) {
    if (div < DIV_LUT_THRESHOLD) {
        *mul = multiplier_lut[div];
        *shift = (int)shift_lut[div];
    } else {
        calc_mul(div, mul, shift);
    }
}

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE void mli_krn_avepool_hwc_nopad(
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

    int remaining_chans = in.ch & (CHANNEL_LANES - 1);

    MLI_OUT_PTR(io_T) out_ptr;

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    if (in.height >= kernel_height && in.width >= kernel_width) {
            for (int ch_idx = 0; ch_idx < in.ch - remaining_chans; ch_idx += CHANNEL_LANES) {
                for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
                    for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
                        // Define area of input and filter for pooling
                        const MLI_PTR(io_T) in_ptr = in.ptr
                                + in.row_mem_stride * (H_idx * stride_height - padding_top) // move to row
                                + in.col_mem_stride * (W_idx * stride_width - padding_left) // move to column
                                + ch_idx;                                                   // move to channel

                        out_ptr = &out.ptr[out.row_mem_stride * H_idx +
                                           out.col_mem_stride * W_idx +
                                           ch_idx];
                        
                        auto acc = mli_prv_init_accu_v(io_T{0});
                        int shift_value = 0;
                        int16_t mul = 0;
                        get_mul_shift_value(kernel_width * kernel_height, &mul, &shift_value);

                        mli::krn::reduce_sum2D_v(in_ptr, mul, &acc, kernel_width, kernel_height,
                                in.col_mem_stride, in.row_mem_stride, true);

                        mli_prv_clip_and_store_output_v(out_ptr, &acc, shift_value);
                    }
                }
            }
        if (remaining_chans) {
            for (int H_idx = row_beg; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_beg; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for pooling
                    const MLI_PTR(io_T) in_ptr = in.ptr
                            + in.row_mem_stride * (H_idx * stride_height - padding_top) // move to row
                            + in.col_mem_stride * (W_idx * stride_width - padding_left) // move to column
                            + in.ch - remaining_chans;                                  // move to channel

                    out_ptr = &out.ptr[out.row_mem_stride * H_idx +
                                       out.col_mem_stride * W_idx +
                                       out.ch - remaining_chans];

                    acc_T acc = mli_prv_init_accu(io_T{0});
 
                    int shift_value = 0;
                    int16_t mul = 0;
                    get_mul_shift_value(kernel_width * kernel_height, &mul, &shift_value);

                    mli::krn::reduce_sum2D(in_ptr, mul, &acc, kernel_width, kernel_height, remaining_chans,
                        in.col_mem_stride, in.row_mem_stride, true);

                    mli_prv_clip_and_store_output(out_ptr, &acc, shift_value);
                }
            }
        }
    }
}

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE void mli_krn_avepool_hwc_pad(
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

    int remaining_chans = in.ch & (CHANNEL_LANES - 1);

    MLI_OUT_PTR(io_T) out_ptr;

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    row_beg = CEIL_DIV(padding_top, stride_height);
    row_end = out.height - CEIL_DIV(padding_bot, stride_height);
    clmn_beg = CEIL_DIV(padding_left, stride_width);
    clmn_end = out.width - CEIL_DIV(padding_right, stride_width);

    if ((row_end - row_beg > 0) && (clmn_end - clmn_beg > 0)) {
        mli_krn_avepool_hwc_nopad<io_T, acc_T>(
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
            for (int ch_idx = 0; ch_idx < in.ch - remaining_chans; ch_idx += CHANNEL_LANES) {
                for (int H_idx = perc_areas[area_idx].row_beg; H_idx < perc_areas[area_idx].row_end; H_idx++) {
                    for (int W_idx = perc_areas[area_idx].clmn_beg; W_idx < perc_areas[area_idx].clmn_end; W_idx++) {
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

                        auto acc = mli_prv_init_accu_v(io_T{0});

                        int shift_value = 0;
                        int16_t mul = 0;
                        get_mul_shift_value(rows * clmns, &mul, &shift_value);

                        mli::krn::reduce_sum2D_v(in_ptr, mul, &acc, clmns, rows,
                                in.col_mem_stride, in.row_mem_stride, false);

                        mli_prv_clip_and_store_output_v(out_ptr, &acc, shift_value);
                    }
                }
            }
            if (remaining_chans) {
                for (int H_idx = perc_areas[area_idx].row_beg; H_idx < perc_areas[area_idx].row_end; H_idx++) {
                    for (int W_idx = perc_areas[area_idx].clmn_beg; W_idx < perc_areas[area_idx].clmn_end; W_idx++) {
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

                        acc_T acc = mli_prv_init_accu(io_T{0});

                        int shift_value = 0;
                        int16_t mul = 0;
                        get_mul_shift_value(rows * clmns, &mul, &shift_value);

                        mli::krn::reduce_sum2D(in_ptr, mul, &acc, clmns, rows, remaining_chans,
                                in.col_mem_stride, in.row_mem_stride, false);

                        mli_prv_clip_and_store_output(out_ptr, &acc, shift_value);
                    }
                }
            }
        }
    }
}

template <typename io_T, typename acc_T, int fixed_kernel_size>
static void mli_krn_avepool_hwc(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
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
    auto in_prv = mli_prv_get_tensor_hwc<MLI_PTR(io_T), MLI_PTR_IS_XY>(in,
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
    out->el_params = in->el_params;
    const auto out_prv = mli_prv_get_tensor_hwc<MLI_OUT_PTR(io_T), MLI_OUT_PTR_IS_XY>(out);

    const int32_t row_beg = 0;
    const int32_t row_end = out_height;
    const int32_t clmn_beg = 0;
    const int32_t clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    /* TODO Investigating performance and size tradeoffs:
     * The pad function also calls the nopad function, so that function will be there twice (size increases).
     * The compiler will propagate constant zero values into the nopad function which will improve the performance.
     * ACTION: should measure the performance benefit and code size impact.
     * */
    if (((padding_top == 0) && (padding_bot == 0) && (padding_left == 0) && (padding_right == 0))) {
        padding_top = 0;
        padding_bot = 0;
        padding_left = 0;
        padding_right = 0;
        mli_krn_avepool_hwc_nopad<io_T, acc_T>(
            row_beg, row_end, clmn_beg, clmn_end,
            stride_width, stride_height, padding_top,
            padding_bot, padding_left, padding_right,
            in_prv, out_prv,
            kernel_height, kernel_width);
    } else {
        mli_krn_avepool_hwc_pad<io_T, acc_T>(
            row_beg, row_end, clmn_beg, clmn_end,
            stride_width, stride_height, padding_top,
            padding_bot, padding_left, padding_right,
            in_prv, out_prv,
            kernel_height, kernel_width);
    }
}

} // krn
} // mli

#endif // _MLI_KRN_AVEPOOL_HWC_H_
