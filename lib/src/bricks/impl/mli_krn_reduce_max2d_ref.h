/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_RECUCE_MAX2D_REF_H_
#define _MLI_KRN_RECUCE_MAX2D_REF_H_

#include "mli_prv_load_store.h"
#include "mli_prv_dsp.h"
#include "mli_math.h"

namespace mli {
namespace krn {
namespace ref {

template <typename io_T, bool remaining_channels, int fixed_kernel_size, bool varying_kernel>
static MLI_FORCE_INLINE void reduce_max2D_hwc(
		const MLI_PTR(io_T) in,
		MLI_PTR(io_T) out,
		const int width,
        const int height,
		const int col_mem_stride,
		const int row_mem_stride,
		const int channels) {

	*out = in[0];
	for (int row = 0; row < height; row++) {
		for (int clmn = 0; clmn < width; clmn++) {
			*out = mli_math_max_fx(*out, in[(row_mem_stride * row) + (col_mem_stride * clmn)]);
		}
	}
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RECUCE_MAX2D_REF_H_
