/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_REDUCE_SUM2D_VDSP_H_
#define _MLI_KRN_REDUCE_SUM2D_VDSP_H_

#include "mli_prv_load_store.h"
#include "mli_prv_dsp.h"
#include "mli_math.h"

namespace mli {
namespace krn {
namespace vdsp {

template <typename acc_T>
static MLI_FORCE_INLINE acc_T reduce_sum2D_v(
        const MLI_PTR(int8_t) in,
        const int16_t mul,
        acc_T accu,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const bool fixed_size) {

    auto acc_short = mli_prv_init_accu<vNx4accshort_t>();

    for (int row = 0; row < height; row++) {
		for (int clmn = 0; clmn < width; clmn++) {
			acc_short = mli_math_mac_fx(acc_short, 
					mli_prv_load_nx4_samples(&in[(row * row_mem_stride) + (clmn * col_mem_stride)]), (int8_t)1);
		}
	}

    vNx4short_t acc_casted = mli_math_acc_cast_fx<vNx4short_t, vNx4accshort_t>(acc_short); 
    accu = mli_math_mac_fx(accu, acc_casted, mul);
    return accu;
}

template <typename acc_T>
static MLI_FORCE_INLINE acc_T reduce_sum2D_v(
        const MLI_PTR(int16_t) in,
        const int16_t mul,
        acc_T accu,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const bool fixed_size) {
            
    for (int row = 0; row < height; row++) {
		for (int clmn = 0; clmn < width; clmn++) {
			accu = mli_math_mac_fx(accu, in[(row * row_mem_stride) + (clmn * col_mem_stride)], mul);
		}
	}

    return accu;
}

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE acc_T reduce_sum2D(
        const MLI_PTR(io_T) in,
        const int16_t mul,
        acc_T accu,
        const int width,
        const int height,
        const int channels,
        const int col_mem_stride,
        const int row_mem_stride,
        const bool fixed_size) {

    return reduce_sum2D_v(in, mul, accu, width, height, col_mem_stride, row_mem_stride, fixed_size);
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_REDUCE_SUM2D_VDSP_H_
