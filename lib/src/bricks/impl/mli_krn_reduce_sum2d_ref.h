/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_REDUCE_SUM2D_REF_H_
#define _MLI_KRN_REDUCE_SUM2D_REF_H_

#include "mli_prv_load_store.h"
#include "mli_prv_dsp.h"
#include "mli_math.h"

namespace mli {
namespace krn {
namespace ref {

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE void reduce_sum2D_v(
        const MLI_PTR(io_T) in,
        const int16_t mul,
        acc_T * accu,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const bool fixed_size) {

    for (int row = 0; row < height; row++) {
        for (int clmn = 0; clmn < width; clmn++) {
            *accu = mli_math_mac_fx(*accu, mul, in[(row_mem_stride * row) + (col_mem_stride * clmn)]);
        }
    }
}

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE void reduce_sum2D(
        const MLI_PTR(io_T) in,
        const int16_t mul,
        acc_T * accu,
        const int width,
        const int height,
        const int channels,
        const int col_mem_stride,
        const int row_mem_stride,
        const bool fixed_size) {
    reduce_sum2D_v(in, mul, accu, width, height, col_mem_stride, row_mem_stride, fixed_size);
}

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE void reduce_sum(
        const io_T* __restrict in,
        const int16_t mul,
        acc_T * accu,
        const int vals,
        const int step) {

    for (int idx = 0; idx < vals; idx++) {
        *accu = mli_math_mac_fx(*accu, mul, (*in));
        in += step;
    }
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_REDUCE_SUM2D_REF_H_
