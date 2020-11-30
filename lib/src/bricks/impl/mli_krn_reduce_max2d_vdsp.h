/*
* Copyright 2020-2020, Synopsys, Inc.
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

namespace mli {
namespace krn {
namespace vdsp {

template <typename io_T>
static MLI_FORCE_INLINE void reduce_max2D_hwc_v(
		const MLI_PTR(io_T) __restrict in,
		MLI_PTR(io_T) __restrict out,
		const int width,
        const int height,
		const int col_mem_stride,
		const int row_mem_stride,
		const bool fixed_size) {

    int row_inc = row_mem_stride - width * col_mem_stride;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"

	auto curr_vec = mli_prv_load_1vec(in);
	if (fixed_size)
	    in += col_mem_stride;
	auto acc = mli_prv_init_accu(curr_vec);

#pragma clang loop unroll(full)
	for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
		for (int clmn = 0; clmn < width; clmn++) {
            if (fixed_size && (row == 0 && clmn == 0)) {
                continue;
            }
		    curr_vec = mli_prv_load_1vec(in);
		    acc = mli_math_max_fx(acc, curr_vec);
		    in += col_mem_stride;
		}
		in += row_inc;
	}

	auto max = mli_math_acc_cast_fx<decltype(curr_vec), decltype(acc)> (acc);
	mli_prv_store_n_samples(out, max);

#pragma clang diagnostic pop
}

template <typename io_T>
static MLI_FORCE_INLINE void reduce_max2D_hwc(
		const MLI_PTR(io_T) __restrict in,
		MLI_PTR(io_T) __restrict out,
		const int width,
        const int height,
		const int channels,
		const int col_mem_stride,
		const int row_mem_stride,
		const bool fixed_size) {

    int row_inc = row_mem_stride - width * col_mem_stride;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"

    auto curr_vec = mli_prv_load_1vec(in);
    if (fixed_size)
        in += col_mem_stride;
    auto acc = mli_prv_init_accu(curr_vec);

#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            if (fixed_size && (row == 0 && clmn == 0)) {
                continue;
            }
            curr_vec = mli_prv_load_1vec(in);
            acc = mli_math_max_fx(acc, curr_vec);
            in += col_mem_stride;
        }
        in += row_inc;
    }

    auto max = mli_math_acc_cast_fx<decltype(curr_vec), decltype(acc)> (acc);
    mli_prv_store_n_samples(out, max, channels);

#pragma clang diagnostic pop
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RECUCE_MAX2D_VDSP_H_
