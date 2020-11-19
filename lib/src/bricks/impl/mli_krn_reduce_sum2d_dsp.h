/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_REDUCE_SUM2D_DSP_H_
#define _MLI_KRN_REDUCE_SUM2D_DSP_H_

#include "mli_prv_load_store.h"
#include "mli_prv_dsp.h"
#include "mli_math.h"

#define REDUCE_SUM2D_UNROLL_FACTOR_FOR_WIDTH 7
#define REDUCE_SUM2D_UNROLL_FACTOR_FOR_HEIGHT 7

namespace mli {
namespace krn {
namespace dsp {

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE acc_T reduce_sum2D_v(
        const MLI_PTR(io_T) in,
        const int16_t mul,
        acc_T accu,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const bool fixed_size,
        int *accum_shift_amout) {

    v2q15_t v2mul = {mul, mul};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
    if (width == 1){
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
            mli_math_mac_fx_vec2(&accu, mli_prv_load_2_samples(in), v2mul);
            in += row_mem_stride;
        }
    } else if (height == 1){
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            mli_math_mac_fx_vec2(&accu, mli_prv_load_2_samples(in), v2mul);
            in += col_mem_stride;
        }
    } else {
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn++) {
                mli_math_mac_fx_vec2(&accu, mli_prv_load_2_samples(in), v2mul);
                in += col_mem_stride;
            }
            in += row_mem_stride - col_mem_stride * width;
        }
    } 
#pragma clang diagnostic pop
    
    return accu;
}

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE acc_T reduce_sum2D_d(
        const MLI_PTR(io_T) in,
        const int16_t mul,
        acc_T accu,
        const int width,
        const int height,
        int col_mem_stride,
        int row_mem_stride,
        const bool fixed_size) {

        v2q15_t v2mul = {mul, mul};
        const MLI_PTR(int16_t) __restrict v2mul_ptr = (const MLI_PTR(int16_t) __restrict)&v2mul;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
    if (width == 1){
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
            mli_prv_load_mac_vec2(&accu, v2mul_ptr, in);
            in += row_mem_stride;
        }
    } else if( height == 1) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            mli_prv_load_mac_vec2(&accu, v2mul_ptr, in);
            in += col_mem_stride;
        }
    } else {
        row_mem_stride -= width * col_mem_stride;
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn++) {
                mli_prv_load_mac_vec2(&accu, v2mul_ptr, in);
                in += col_mem_stride;
            }
            in += row_mem_stride;
        }
    }
#pragma clang diagnostic pop

    return accu;
}

} // namespace dsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_REDUCE_SUM2D_DSP_H_
