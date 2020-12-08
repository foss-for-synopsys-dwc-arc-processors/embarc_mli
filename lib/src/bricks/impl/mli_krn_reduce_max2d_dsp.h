/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_RECUCE_MAX2D_DSP_H_
#define _MLI_KRN_RECUCE_MAX2D_DSP_H_

#include "mli_prv_load_store.h"
#include "mli_prv_dsp.h"
#include "mli_math.h"

#define REDUCE_MAX2D_UNROLL_FACTOR_FOR_WIDTH 7
#define REDUCE_MAX2D_UNROLL_FACTOR_FOR_HEIGHT 7

namespace mli {
namespace krn {
namespace dsp {

template <typename io_T, bool varying_kernel>
static MLI_FORCE_INLINE void reduce_max2D_hwc_v(
		const MLI_PTR(io_T) in,
		MLI_PTR(io_T) out,
		const int width,
        const int height,
		const int col_mem_stride,
		const int row_mem_stride) {

    v2q15_t cur_max = mli_prv_load_2_samples(in);
    if (width == 1){
        for (int row = 0; row < height; row++) {
            cur_max = mli_math_max_fx(cur_max, mli_prv_load_2_samples(&in[row*row_mem_stride]));
        }
    } else if (height == 1){
        for (int clmn = 0; clmn < width; clmn++) {
            cur_max = mli_math_max_fx(cur_max, mli_prv_load_2_samples(&in[clmn*col_mem_stride]));
        }
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
        if (!varying_kernel && height <= REDUCE_MAX2D_UNROLL_FACTOR_FOR_HEIGHT &&
        		width <= REDUCE_MAX2D_UNROLL_FACTOR_FOR_WIDTH) {
#pragma clang loop unroll(full)
            for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
                for (int clmn = 0; clmn < width; clmn++) {
                    cur_max = mli_math_max_fx(cur_max, mli_prv_load_2_samples(
                        &in[(row * row_mem_stride) + (clmn * col_mem_stride)]));
                }
            }
        } else {
            for (int row = 0; row < height; row++) {
                for (int clmn = 0; clmn < width; clmn++) {
                    cur_max = mli_math_max_fx(cur_max, mli_prv_load_2_samples(
                        &in[(row * row_mem_stride) +  (clmn * col_mem_stride)]));
                }
            }
        }
#pragma clang diagnostic pop
    }

    mli_prv_store_2_samples(out, cur_max);
}

template <typename io_T, bool remaining_channels, int fixed_kernel_size, bool varying_kernel>
static MLI_FORCE_INLINE void reduce_max2D_hwc(
        const MLI_PTR(io_T) in,
        MLI_PTR(io_T) out,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const int channels) {
    if (remaining_channels) {
        mli::krn::ref::reduce_max2D_hwc<io_T, remaining_channels, fixed_kernel_size, varying_kernel>
            (in, out, width, height, col_mem_stride, row_mem_stride, channels);
    } else {
        reduce_max2D_hwc_v<io_T, varying_kernel>(in, out, width, height, col_mem_stride, row_mem_stride);
    }
}

} // namespace dsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RECUCE_MAX2D_DSP_H_
