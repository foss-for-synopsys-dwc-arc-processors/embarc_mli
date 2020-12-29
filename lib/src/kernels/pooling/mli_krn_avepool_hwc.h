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
#include "mli_prv_quant.h"
#include "mli_prv_dsp.h"

////////////////////////////////////////////////////////////////////////////////
// Setting up namespace
////////////////////////////////////////////////////////////////////////////////
namespace mli {
namespace krn {

#define AVEPOOL_FIXED_KRN_SIZE_3 3
#define AVEPOOL_FIXED_KRN_SIZE_2 2
#define AVEPOOL_NO_FIXED_KRN_SIZE 0

#define DIV_LUT_THRESHOLD 32
static const int16_t multiplier_lut[] = {
    0,      // 0
    0x4000, // 1
    0x4000, // 2
    0x5555, // 3
    0x4000, // 4
    0x6666, // 5
    0x5555, // 6
    0x4924, // 7*
    0x4000, // 8
    0x71c7, // 9
    0x6666, // 10
    0x5d17, // 11
    0x5555, // 12
    0x4ec4, // 13*
    0x4924, // 14*
    0x4444, // 15
    0x4000, // 16
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
     0, // 0
    14, // 1
    15, // 2
    16, // 3
    16, // 4
    17, // 5
    17, // 6
    17, // 7
    17, // 8
    18, // 9
    18, // 10
    18, // 11
    18, // 12
    18, // 13
    18, // 14
    18, // 15
    18, // 16
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
    *shift_val += 14 + shift_norm_val;
}

static inline void get_mul_shift_value(
        unsigned div,
        int16_t* mul, int* shift) {
    if (div < DIV_LUT_THRESHOLD) {
        *mul = multiplier_lut[div];
        *shift += (int)shift_lut[div];
    } else {
        calc_mul(div, mul, shift);
    }
}

template <typename io_T, typename acc_T, bool convert = false>
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
        int kernel_width,
        s8asym_quant_params *params) {

    int number_lanes = get_number_lanes<acc_T>();
    int remaining_chans = in.ch & (number_lanes - 1);

    int accum_shift_amout = 0;
    int16_t mul = 0;
    int shift_value = params->shift;
    int32_t zp = params->offset;
    get_mul_shift_value(kernel_width * kernel_height, &mul, &shift_value);
    if (convert) {
        int norm_shift;
        mul = mli_math_norm_cast_fx<int32_t,int16_t>(
                            mli_math_mul_fx<int16_t, int32_t>(params->scale, mul), &norm_shift);
        shift_value -= norm_shift;
    } else {
        MLI_ASSERT(params->offset == 0);
        MLI_ASSERT(params->scale  == 1);
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
            for (int ch_idx = 0; ch_idx < in.ch - remaining_chans; ch_idx += number_lanes) {
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
                        acc_T acc = mli_prv_init_accu_with_bias_v<acc_T>(zp, shift_value);
                        
                        auto res = mli::krn::reduce_sum2D_v(in_ptr, mul, acc, kernel_width, kernel_height,
                                in.col_mem_stride, in.row_mem_stride, &accum_shift_amout);

                        mli_prv_clip_and_store_output_v(out_ptr, res, shift_value - accum_shift_amout);

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
                    acc_T acc = mli_prv_init_accu_with_bias_v<acc_T>(zp, shift_value);

                    auto res = mli::krn::reduce_sum2D_v(in_ptr, mul, acc, kernel_width, kernel_height,
                        in.col_mem_stride, in.row_mem_stride, &accum_shift_amout);

                    mli_prv_clip_and_store_output_v(out_ptr, res, shift_value - accum_shift_amout, remaining_chans);

                    in_ptr += in_col_inc;
                    out_ptr += out_col_inc;
                }
                in_ptr += in_row_inc;
                out_ptr += out_row_inc;
            }
        }
    }
}

template <typename io_T, typename acc_T, bool convert = false>
static MLI_FORCE_INLINE void mli_krn_avepool_hwc_pad(
        int stride_width,
        int stride_height,
        int padding_top,
        int padding_bot,
        int padding_left,
        int padding_right,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        int kernel_height,
        int kernel_width,
        s8asym_quant_params *params) {

    int number_lanes = get_number_lanes<acc_T>();
    int remaining_chans = in.ch & (number_lanes - 1);
    MLI_OUT_PTR(io_T) out_ptr;

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    const int nopad_row_beg = CEIL_DIV(padding_top, stride_height);
    const int nopad_row_end = CEIL_DIV(in.height + padding_top - kernel_height + 1, stride_height);
    const int nopad_clmn_beg = CEIL_DIV(padding_left, stride_width);
    const int nopad_clmn_end = CEIL_DIV(in.width + padding_left - kernel_width + 1, stride_width);

    if ((nopad_row_end - nopad_row_beg > 0) && (nopad_clmn_end - nopad_clmn_beg > 0)) {
        mli_krn_avepool_hwc_nopad<io_T, acc_T, convert>(
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
            int top_comp_val = (perc_areas[area_idx].row_beg * stride_height) - padding_top;
            for (int H_idx = perc_areas[area_idx].row_beg; H_idx < (int)perc_areas[area_idx].row_end; H_idx++) {
                int left_comp_val = (perc_areas[area_idx].clmn_beg * stride_width) - padding_left;
                for (int W_idx = perc_areas[area_idx].clmn_beg; W_idx < (int)perc_areas[area_idx].clmn_end; W_idx++) {
                    for (int ch_idx = 0; ch_idx < in.ch - remaining_chans; ch_idx += number_lanes) {
                        // Define area of input and filter for pooling
                        // *_comp - compensation values for valid area defining
                        int top_comp = -MIN(top_comp_val, 0);
                        int left_comp = -MIN(left_comp_val, 0);

                        int right_comp = -MIN(in.width - (left_comp_val + kernel_width), 0);
                        int bottom_comp = -MIN(in.height - (top_comp_val + kernel_height), 0);

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

                        int accum_shift_amout = 0;
                        int shift_value = params->shift;
                        int16_t mul = 0;
                        int32_t zp = params->offset;
                        get_mul_shift_value(rows * clmns, &mul, &shift_value);
                        if (convert) {
                            int norm_shift;
                            mul = mli_math_norm_cast_fx<int32_t,int16_t>(
                                                mli_math_mul_fx<int16_t, int32_t>(params->scale, mul), &norm_shift);
                            shift_value -= norm_shift;
                        } else {
                            MLI_ASSERT(params->offset == 0);
                            MLI_ASSERT(params->scale  == 1);
                        }

                        acc_T acc = mli_prv_init_accu_with_bias_v<acc_T>(zp, shift_value);
                        
                        auto res = mli::krn::reduce_sum2D_v(in_ptr, mul, acc, clmns, rows,
                                in.col_mem_stride, in.row_mem_stride, &accum_shift_amout);

                        mli_prv_clip_and_store_output_v(out_ptr, res, shift_value - accum_shift_amout);

                    }
                    left_comp_val += stride_width;
                }
                top_comp_val += stride_height;
            }

            if (remaining_chans) {
                int top_comp_val = (perc_areas[area_idx].row_beg * stride_height) - padding_top;
                for (int H_idx = perc_areas[area_idx].row_beg; H_idx < (int)perc_areas[area_idx].row_end; H_idx++) {
                    int left_comp_val = (perc_areas[area_idx].clmn_beg * stride_width) - padding_left;
                    for (int W_idx = perc_areas[area_idx].clmn_beg; W_idx < (int)perc_areas[area_idx].clmn_end; W_idx++) {
                        // Define area of input and filter for pooling
                        // *_comp - compensation values for valid area defining
                        int top_comp = -MIN(top_comp_val, 0);
                        int left_comp = -MIN(left_comp_val, 0);

                        int right_comp = -MIN(in.width - (left_comp_val + kernel_width), 0);
                        int bottom_comp = -MIN(in.height - (top_comp_val + kernel_height), 0);

                        int rows = kernel_height - top_comp - bottom_comp;
                        int clmns = kernel_width - right_comp - left_comp;

                        const MLI_PTR(io_T) in_ptr = in.ptr
                                + in.row_mem_stride * (H_idx * stride_height - padding_top + top_comp)  // move to row
                                + in.col_mem_stride * (W_idx * stride_width - padding_left + left_comp) // move to column
                                + in.ch -  remaining_chans;                                             // move to channel

                        out_ptr = &out.ptr[out.row_mem_stride * H_idx +
                                           out.col_mem_stride * W_idx +
                                           out.ch - remaining_chans];

                        int accum_shift_amout = 0;
                        int shift_value = params->shift;
                        int16_t mul = 0;
                        int32_t zp = params->offset;
                        get_mul_shift_value(rows * clmns, &mul, &shift_value);
                        if (convert) {
                            int norm_shift;
                            mul = mli_math_norm_cast_fx<int32_t,int16_t>(
                                                mli_math_mul_fx<int16_t, int32_t>(params->scale, mul), &norm_shift);
                            shift_value -= norm_shift;
                        } else {
                            MLI_ASSERT(params->offset == 0);
                            MLI_ASSERT(params->scale  == 1);
                        }
                        
                        acc_T acc = mli_prv_init_accu_with_bias_v<acc_T>(zp, shift_value);

                        auto res = mli::krn::reduce_sum2D_v(in_ptr, mul, acc, clmns, rows,
                                in.col_mem_stride, in.row_mem_stride, &accum_shift_amout);

                        mli_prv_clip_and_store_output_v(out_ptr, res, shift_value - accum_shift_amout, remaining_chans);
                        left_comp_val += stride_width;
                    }
                    top_comp_val += stride_height;
                }
            }
        }
    }
}

template <typename io_T, typename acc_T, int fixed_kernel_size, bool convert = false>
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
    if (convert) {
        MLI_ASSERT(in->el_type == MLI_EL_SA_8);
        define_requant_params(in, out, &params);
    } else {
        MLI_ASSERT(in->el_type == MLI_EL_FX_8 || in->el_type == MLI_EL_FX_16);
        params.shift  = in->el_params.fx.frac_bits - out->el_params.fx.frac_bits;
        params.offset = 0;
        params.scale  = 1;
    }
    const auto out_prv = mli_prv_get_tensor_hwc<MLI_OUT_PTR(io_T)>(out);

    const int32_t row_beg = 0;
    const int32_t row_end = out_height;
    const int32_t clmn_beg = 0;
    const int32_t clmn_end = out_width;

    mli_prv_fx_init_dsp_ctrl();

    if ((padding_top || padding_bot || padding_left || padding_right)) {
        mli_krn_avepool_hwc_pad<io_T, acc_T, convert>(
            stride_width, stride_height, padding_top,
            padding_bot, padding_left, padding_right,
            in_prv, out_prv,
            kernel_height, kernel_width, &params);
    } else {
        mli_krn_avepool_hwc_nopad<io_T, acc_T, convert>(
            row_beg, row_end, clmn_beg, clmn_end,
            stride_width, stride_height, /* padding_top */ 0,
            /* padding_bot */ 0, /* padding_left */ 0 , /* padding_right */ 0,
            in_prv, out_prv,
            kernel_height, kernel_width, &params);
    }
}

} // krn
} // mli

#endif // _MLI_KRN_AVEPOOL_HWC_H_
