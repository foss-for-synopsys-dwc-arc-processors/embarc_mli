/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_RECUCE_MAX2D_VDSP_H_
#define _MLI_KRN_RECUCE_MAX2D_VDSP_H_

#include "mli_prv_load_store.h"
#include "mli_prv_dsp.h"
#include "mli_math.h"
#include "mli_mem_info.h"

namespace mli {
namespace krn {
namespace vdsp {

template <typename io_T, int fixed_kernel_size>
static MLI_FORCE_INLINE void reduce_max2D_hwc_v(
		const MLI_PTR(io_T) __restrict in,
		MLI_PTR(io_T) __restrict out,
		const int width,
        const int height,
		const int col_mem_stride,
		const int row_mem_stride,
        const int channels) {

    int row_inc = row_mem_stride - width * col_mem_stride;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"

	auto curr_vec = mli_prv_load_1vec(in);
	if (fixed_kernel_size)
	    in += col_mem_stride;
	auto acc = mli_prv_init_accu(curr_vec);

#pragma clang loop unroll(full)
	for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
		for (int clmn = 0; clmn < width; clmn++) {
            if (fixed_kernel_size && (row == 0 && clmn == 0)) {
                continue;
            }
		    curr_vec = mli_prv_load_1vec(in);
		    acc = mli_math_max_fx(acc, curr_vec);
		    in += col_mem_stride;
		}
		in += row_inc;
	}

	auto max = mli_math_acc_cast(acc);
	
    mli_prv_store_n_samples(out, max, channels);

#pragma clang diagnostic pop
}

template <typename io_T>
static MLI_FORCE_INLINE void reduce_max2D_hwc_k2x2_padding_kernel_unroll(
        const MLI_PTR(io_T) __restrict in_ptr,
        MLI_OUT_PTR(io_T) __restrict out_ptr,
        const int clmns,
        const int rows,
        const int32_t col_mem_stride,
        const int32_t row_mem_stride,
        const int channels)
{
    MLI_ASSERT(rows == 1);
    MLI_ASSERT((clmns == 1) || (clmns == 2));

    switch (rows) {
    case 1:
        switch (clmns) {
        case 1:
            reduce_max2D_hwc_v<io_T, /*fixed_kernel_size*/ 2>(in_ptr, out_ptr, /* width = */1, /*height = */1,
                        col_mem_stride, row_mem_stride, channels);
            break;

        case 2:
            reduce_max2D_hwc_v<io_T, /*fixed_kernel_size*/ 2>(in_ptr, out_ptr, /* width = */2, /*height = */1,
                        col_mem_stride, row_mem_stride, channels);
            break;

        default:
            MLI_ASSERT(0);
            break;
        }
        break;

    default:
        MLI_ASSERT(0);
        break;
    }

}

template <typename io_T>
static MLI_FORCE_INLINE void reduce_max2D_hwc_k3x3_padding_kernel_unroll(
        const MLI_PTR(io_T) __restrict in_ptr,
        MLI_OUT_PTR(io_T) __restrict out_ptr,
        const int clmns,
        const int rows,
        const int32_t col_mem_stride,
        const int32_t row_mem_stride,
        const int channels)
{
    MLI_ASSERT((rows == 1) || (rows == 2));
    MLI_ASSERT((clmns == 1) || (clmns == 2) || (clmns == 3));

    switch (rows) {
    case 1:
        switch (clmns) {
        case 1:
            reduce_max2D_hwc_v<io_T, /*fixed_kernel_size*/ 3>(in_ptr, out_ptr, 1, 1,
                        col_mem_stride, row_mem_stride, channels);
            break;

        case 2:
            reduce_max2D_hwc_v<io_T, /*fixed_kernel_size*/ 3>(in_ptr, out_ptr, 2, 1,
                        col_mem_stride, row_mem_stride, channels);
            break;

        case 3:
            reduce_max2D_hwc_v<io_T, /*fixed_kernel_size*/ 3>(in_ptr, out_ptr, 3, 1,
                        col_mem_stride, row_mem_stride, channels);
            break;

        default:
            MLI_ASSERT(0);
            break;
        }
        break;

    case 2:
        switch (clmns) {
        case 1:
            reduce_max2D_hwc_v<io_T, /*fixed_kernel_size*/ 3>(in_ptr, out_ptr, 1, 2,
                        col_mem_stride, row_mem_stride, channels);
            break;

        case 2:
            reduce_max2D_hwc_v<io_T, /*fixed_kernel_size*/ 3>(in_ptr, out_ptr, 2, 2,
                        col_mem_stride, row_mem_stride, channels);
            break;

        case 3:
            reduce_max2D_hwc_v<io_T, /*fixed_kernel_size*/ 3>(in_ptr, out_ptr, 3, 2,
                        col_mem_stride, row_mem_stride, channels);
            break;

        default:
            MLI_ASSERT(0);
            break;
        }
        break;

    default:
        MLI_ASSERT(0);
        break;
    }
}

template <typename io_T, int fixed_kernel_size, bool varying_kernel>
static MLI_FORCE_INLINE void reduce_max2D_hwc(
        const MLI_PTR(io_T) __restrict in,
        MLI_PTR(io_T) __restrict out,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const int channels) {

    if (varying_kernel && fixed_kernel_size == 3) {
        reduce_max2D_hwc_k3x3_padding_kernel_unroll<io_T>
            (in, out, width, height, col_mem_stride, row_mem_stride, channels);
    } else if (varying_kernel && fixed_kernel_size == 2) {
        reduce_max2D_hwc_k2x2_padding_kernel_unroll<io_T>
            (in, out, width, height, col_mem_stride, row_mem_stride, channels);
    } else {
        reduce_max2D_hwc_v<io_T, fixed_kernel_size>
            (in, out, width, height, col_mem_stride, row_mem_stride, channels);
    }
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RECUCE_MAX2D_VDSP_H_
