/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_krn_maxpool_hwc_decl.h"

#include "mli_config.h"
#include "mli_prv_dsp.h"

#ifndef _MLI_KRN_MAXPOOL_HWC_REF_H_
#define _MLI_KRN_MAXPOOL_HWC_REF_H_

namespace mli {
namespace krn {
namespace ref {

template <typename io_T>
static inline void __attribute__((always_inline)) maxpool_hwc_nopad(
        int row_beg,
        int row_end,
        int clmn_beg,
        int clmn_end,
        int stride_width,
        int stride_height,
        int padding_top,
        int padding_bot,
        int padding_left,
        int padding_right,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        int kernel_height,
        int kernel_width) {
    // TODO
    assert(0);
}

template <typename io_T>
static inline void __attribute__((always_inline)) maxpool_hwc_pad(
        int row_beg,
        int row_end,
        int clmn_beg,
        int clmn_end,
        int stride_width,
        int stride_height,
        int padding_top,
        int padding_bot,
        int padding_left,
        int padding_right,
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const tensor_private_t<MLI_OUT_PTR(io_T)> &out,
        int kernel_height,
        int kernel_width) {
    // TODO
    assert(0);
}

} // namespace dsp
} // namespace ref
} // namespace mli

#endif // _MLI_KRN_MAXPOOL_HWC_REF_H_
