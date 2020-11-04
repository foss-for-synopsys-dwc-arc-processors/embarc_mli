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

	auto curr_vec = mli_prv_load_1vec(in);
	auto acc = mli_prv_init_accu(curr_vec);

	for (int row = 0, clmn = 1; row < height; row++, clmn = 0) {
		for (; clmn < width; clmn++) {
		    curr_vec = mli_prv_load_1vec(&in[(row * row_mem_stride) + (clmn * col_mem_stride)]);
		    acc = mli_math_max_fx(acc, curr_vec);
		}
	}

	auto max = mli_math_acc_cast_fx<decltype(curr_vec), decltype(acc)> (acc);
	mli_prv_store_n_samples(out, max);
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

    auto curr_vec = mli_prv_load_1vec(in);
    auto acc = mli_prv_init_accu(curr_vec);

    for (int row = 0, clmn = 1; row < height; row++, clmn = 0) {
        for (; clmn < width; clmn++) {
            curr_vec = mli_prv_load_1vec(&in[(row * row_mem_stride) + (clmn * col_mem_stride)]);
            acc = mli_math_max_fx(acc, curr_vec);
        }
    }

    auto max = mli_math_acc_cast_fx<decltype(curr_vec), decltype(acc)> (acc);
	mli_prv_store_n_samples(out, max, channels);
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RECUCE_MAX2D_VDSP_H_
