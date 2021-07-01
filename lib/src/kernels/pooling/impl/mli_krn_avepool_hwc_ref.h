/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_AVEPOOL_HWC_REF_H_
#define _MLI_KRN_AVEPOOL_HWC_REF_H_

#include "mli_krn_reduce_sum2d.h"
#include "mli_prv_load_store.h"
#include "mli_prv_dsp.h"
#include "mli_math.h"
#include "mli_mem_info.h"

namespace mli {
namespace krn {
namespace ref {

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

static MLI_FORCE_INLINE void calc_mul(unsigned div, int16_t* mul, int* shift_val) {
    unsigned int one = (1<<31); // u1.31
    unsigned int val = one / div; // u1.31
    int shift_norm_val = 0;

    if (div > 1) { 
        shift_norm_val = mli_math_norm_fx<int, int>(val) + 1;
        val <<= shift_norm_val;
    }

    *mul = val >> 17;
    *shift_val += 14 + shift_norm_val;
}

static MLI_FORCE_INLINE void get_mul_shift_value(
        unsigned div,
        int16_t* mul, int* shift) {
    if (div < DIV_LUT_THRESHOLD) {
        *mul = multiplier_lut[div];
        *shift += (int)shift_lut[div];
    } else {
        calc_mul(div, mul, shift);
    }
}

static MLI_FORCE_INLINE void compute_avepool_func(
        const MLI_PTR(int8_t) __restrict in,
        MLI_OUT_PTR(int8_t) __restrict out,
        const int16_t mul,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const int32_t zp,
        const int shift_value,
        const int channels) {
    
    MLI_ASSERT(channels == 1);
    mli_acc32_t accu = mli_prv_init_accu_with_bias_v<mli_acc32_t>(zp, shift_value);
    accu = reduce_sum2D_v(in, mul, accu, width, height, col_mem_stride, row_mem_stride);
    mli_prv_clip_and_store_output_v(out, accu, shift_value);
}

static MLI_FORCE_INLINE void compute_avepool_func(
        const MLI_PTR(int16_t) __restrict in,
        MLI_OUT_PTR(int16_t) __restrict out,
        const int16_t mul,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const int32_t zp,
        const int shift_value,
        const int channels) {
    
    MLI_ASSERT(channels == 1);
    mli_acc40_t accu = mli_prv_init_accu_with_bias_v<mli_acc40_t>(zp, shift_value);
    accu = reduce_sum2D_v(in, mul, accu, width, height, col_mem_stride, row_mem_stride);
    mli_prv_clip_and_store_output_v(out, accu, shift_value);
}

template<typename io_T, int fixed_kernel_size, bool varying_kernel>
static MLI_FORCE_INLINE void compute_avepool(
        const MLI_PTR(io_T) __restrict in,
        MLI_OUT_PTR(io_T) __restrict out,
        const int16_t mul,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const int32_t zp,
        const int shift_value,
        const int channels) {

    compute_avepool_func(in, out, mul, width, height, col_mem_stride, row_mem_stride, zp ,shift_value, channels);
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_AVEPOOL_HWC_REF_H_
