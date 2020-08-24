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

    // TODO
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
    
    // TODO
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_REDUCE_SUM2D_VDSP_H_
