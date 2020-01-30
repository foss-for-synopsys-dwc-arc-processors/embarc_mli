/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_REDUCE_SUM2D_H_
#define _MLI_KRN_REDUCE_SUM2D_H_

#include "mli_config.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_load_store.h"

/******************************************************************************
 *
 * Version & platform description
 * Targets:
 *
 ******************************************************************************/

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

static inline void calc_mul(unsigned div, int16_t* mul, int* shift_val) {
    unsigned int one = (1<<31); // u1.31
    unsigned int val = one / div; // u1.31
    int shift_norm_val = 0;

    if (div > 1) { 
        shift_norm_val = fx_norm_q31(val) + 1;
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

template <typename io_T>
static inline void __attribute__((always_inline)) reduce_sum2D_chw_even(
        accum40_t *__restrict acc40,
        const MLI_PTR(io_T) __restrict in,
        const int32_t width,
        const int32_t height,
        const int32_t in_row_step,
        const int16_t mul) {
    const v2q15_t mul_v = {mul, mul};
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn += 2) {
            *acc40 = fx_a40_dmac_v2q15(*acc40, mli_prv_load_2_samples(&in[clmn]), mul_v);
        }
        in += in_row_step;
    }
}

template <typename io_T>
static inline void __attribute__((always_inline)) reduce_sum2D_chw(
        accum40_t *__restrict acc40,
        const MLI_PTR(io_T) __restrict in,
        const int32_t width,
        const int32_t height,
        const int32_t in_row_step,
        const int16_t mul) {
    const v2q15_t mul_v = {mul, mul};
    if (width & 1) {
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
            *acc40 = fx_a40_mac_q15(*acc40, in[0], mul);
#pragma clang loop unroll(full)
            for (int clmn = 1; clmn < width; clmn += 2) {
                *acc40 = fx_a40_dmac_v2q15(*acc40, mli_prv_load_2_samples(&in[clmn]), mul_v);
            }
            in += in_row_step;
        }
    } else {
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn += 2) {
                *acc40 = fx_a40_dmac_v2q15(*acc40, mli_prv_load_2_samples(&in[clmn]), mul_v);
            }
            in += in_row_step;
        }
    }
}

template <typename io_T>
static inline accum40_t reduce_sum2D_hwc(
        MLI_PTR(io_T) __restrict in,
        uint32_t width,
        uint32_t height,
        uint32_t channels,
        uint32_t in_row_step,
        int16_t mul) {
    accum40_t acc40 = fx_create_a40(0x0, 0x0);
    if (width == 1){
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
            acc40 = fx_a40_mac_q15(acc40, *in, mul);
            in += channels * in_row_step;
        }
    } else if( height == 1) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            acc40 = fx_a40_mac_q15(acc40, *in, mul);
            in += channels;
        }
    } else {
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn++) {
                acc40 = fx_a40_mac_q15(acc40, *in, mul);
                in += channels;
            }
            in += channels * (in_row_step - width);
        }
    }
    return acc40;
}

template <typename io_T>
static inline v2accum40_t __attribute__((always_inline)) reduce_sum2D_hwc_v(
        MLI_PTR(io_T) __restrict in,
        uint32_t width,
        uint32_t height,
        uint32_t channels,
        uint32_t in_row_step,
        int16_t mul) {

    v2accum40_t v2acc40 = {0, 0};
    v2q15_t v2mul = {mul, mul};
    if (width == 1){
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
            v2acc40 = fx_v2a40_mac_v2q15(v2acc40, mli_prv_load_2_samples(in), v2mul);
            in += in_row_step * channels;
        }
    } else if (height == 1){
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            v2acc40 = fx_v2a40_mac_v2q15(v2acc40, mli_prv_load_2_samples(in), v2mul);
            in += channels;
        }
    } else {
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn++) {
                v2acc40 = fx_v2a40_mac_v2q15(v2acc40, mli_prv_load_2_samples(in), v2mul);
                in += channels;
            }
            in += channels * (in_row_step - width);
        }
    }
    return v2acc40;
}

//==========================================================================
// Sequential reducing summation
//==========================================================================
template <typename io_T, typename acc_T>
inline acc_T reduce_sum(
        const io_T* __restrict in,
        const int16_t mul,
        acc_T accu,

        const int vals,
        const int step = 1) {
    for (int idx = 0; idx < vals; idx++) {
        accu = mli_math_mac_fx(accu, mul, (*in));
        in += step;
    }
    return accu;
}
//==========================================================================
// Two dimensional reducing summation across width and height 
//==========================================================================
template <typename io_T, typename acc_T>
inline acc_T __attribute__((always_inline)) reduce_sum2D(
        const MLI_PTR(io_T) __restrict in,
        const int16_t mul,
        acc_T accu,

        const int width,
        const int height,
        int in_col_step,
        int in_row_step) {
    if (width == 1){
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
            accu = mli_math_mac_fx(accu, mul, (*in));
            in += in_row_step;
        }
    } else if( height == 1) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            accu = mli_math_mac_fx(accu, mul, (*in));
            in += in_col_step;
        }
    } else {
        in_row_step -= width * in_col_step;
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn++) {
                accu = mli_math_mac_fx(accu, mul, (*in));
                in += in_col_step;
            }
            in += in_row_step;
        }
    }
    return accu;
}

template <typename io_T, typename acc_T>
static inline acc_T __attribute__((always_inline)) reduce_sum2D_v(
        const MLI_PTR(io_T) __restrict in,
        const int16_t mul,
        acc_T *v2acc,

        const int width,
        const int height,
        int in_col_step,
        int in_row_step) {

    v2q15_t v2mul = {mul, mul};
    if (width == 1){
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
            mli_math_mac_fx_vec2(v2acc, mli_prv_load_2_samples(in), v2mul);
            in += in_row_step;
        }
    } else if (height == 1){
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            mli_math_mac_fx_vec2(v2acc, mli_prv_load_2_samples(in), v2mul);
            in += in_col_step;
        }
    } else {
        in_row_step -= width * in_col_step;
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn++) {
                mli_math_mac_fx_vec2(v2acc, mli_prv_load_2_samples(in), v2mul);
                in += in_col_step;
            }
            in += in_row_step;
        }
    }
    return *v2acc;
}

template <typename io_T, typename acc_T>
inline acc_T __attribute__((always_inline)) reduce_sum2D_d(
        const MLI_PTR(io_T) __restrict in,
        const int16_t mul,
        acc_T *accu,

        const int width,
        const int height,
        int in_col_step,
        int in_row_step) {

        v2q15_t v2mul = {mul, mul};
        const __xy int16_t *__restrict v2mul_ptr = (const __xy int16_t *__restrict)&v2mul;
    if (width == 1){
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
            mli_prv_load_mac_vec2(accu, v2mul_ptr, in);
            in += in_row_step;
        }
    } else if( height == 1) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            mli_prv_load_mac_vec2(accu, v2mul_ptr, in);
            in += in_col_step;
        }
    } else {
        in_row_step -= width * in_col_step;
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn++) {
                mli_prv_load_mac_vec2(accu,v2mul_ptr, in);
                in += in_col_step;
            }
            in += in_row_step;
        }
    }
    return *accu;
}

#endif  //_MLI_KRN_REDUCE_SUM2D_H_
