/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_AVEPOOL_HWC_REF_H_
#define _MLI_KRN_AVEPOOL_HWC_REF_H_

#include "mli_krn_maxpool_hwc_decl.h"

#include "mli_config.h"
#include "mli_prv_dsp.h"

namespace mli {
namespace krn {
namespace ref {

template <typename io_T>
static inline void __attribute__((always_inline)) avepool_hwc_nopad(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_left,
        const int padding_right,
        const int padding_bot) {
    // TODO
    assert(0);
}

template <typename io_T>
static inline void __attribute__((always_inline)) avepool_hwc(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_left,
        const int padding_right,
        const int padding_bot) {
    // TODO
    assert(0);
}

template <typename io_T>
static inline void __attribute__((always_inline)) avepool_hwc_krnpad(
        const int row_beg,
        const int row_end,
        const int clmn_beg,
        const int clmn_end,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_top,
        const int padding_left,
        const int padding_right,
        const int padding_bot) {
    // TODO
    assert(0);
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_AVEPOOL_HWC_REF_H_
