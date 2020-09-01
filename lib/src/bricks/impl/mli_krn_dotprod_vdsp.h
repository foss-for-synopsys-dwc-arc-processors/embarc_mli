/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_DOTPROD_VDSP_H_
#define _MLI_KRN_DOTPROD_VDSP_H_

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_math.h"
#include "mli_types.h"
#include "mli_prv_load_store.h"

namespace mli {
namespace krn {
namespace vdsp {

template <typename io_T, typename w_T, typename acc_T>
static acc_T __attribute__ ((always_inline)) dotprod2D_vv(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step) {
    in_row_step -= width * in_col_step;
    kern_row_step -= width * kern_col_step;
    for (int row = 0; row < height; row++) {
        for (int clmn = 0; clmn < width; clmn++) {
            accu = mli_math_mac_fx(accu, mli_prv_load_n_samples(in), mli_prv_load_n_samples(krn));
            in += in_col_step;
            krn += kern_col_step;
        }
        in += in_row_step;
        krn += kern_row_step;
    }
    return accu;
}

template < typename in_T, typename w_T, typename acc_T >
static inline acc_T __attribute__ ((always_inline)) dotprod3D_v (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int channels,
        int in_col_step,
        int in_row_step,
        int in_ch_step,
        int kern_col_step,
        int kern_row_step,
        int kern_ch_step,
        acc_T accu) {
    in_ch_step -= height * in_row_step;
    kern_ch_step -= height * kern_row_step;
    in_row_step -= width * in_col_step;
    kern_row_step -= width * kern_col_step;

    __builtin_assume (height > 0);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
    for (int ch = 0; ch < channels; ch++) {
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn++) {
                accu = mli_math_mac_fx(accu, mli_prv_load_n_samples(krn), *in);
                in += in_col_step;
                krn += kern_col_step;
            }
            in += in_row_step;
            krn += kern_row_step;
        }
        in += in_ch_step;
        krn += kern_ch_step;
    }
#pragma clang diagnostic pop
    return accu;
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_DOTPROD_VDSP_H_
