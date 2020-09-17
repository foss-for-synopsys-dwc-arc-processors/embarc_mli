/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_DOTPROD_REF_H_
#define _MLI_KRN_DOTPROD_REF_H_

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_math.h"
#include "mli_types.h"
#include "mli_prv_dsp.h"

namespace mli {
namespace krn {
namespace ref {

template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod1D(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int vals,
        const int in_step,
        const int krn_step) {
    for (int idx = 0; idx < vals; idx++) {
        accu = mli_math_mac_fx(accu, (*in), (*krn));
        in += in_step;
        krn += krn_step;
    }
    return accu;
}

template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod2D(
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
            accu = mli_math_mac_fx(accu, (*in), (*krn));
            in += in_col_step;
            krn += kern_col_step;
        }
        in += in_row_step;
        krn += kern_row_step;
    }
    return accu;
}

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE void dotprod2D (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step,
        acc_T * accu) {
    in_row_step -= width * in_col_step;
    kern_row_step -= width * kern_col_step;
    __builtin_assume (height > 0);
PRAGMA_CLANG(diagnostic push)
PRAGMA_CLANG(diagnostic ignored "-Wpass-failed")
PRAGMA_CLANG(loop unroll(full))
    for (int row = 0; row < height; row++) {
PRAGMA_CLANG(loop unroll(full))
        for (int clmn = 0; clmn < width; clmn++) {
            *accu = mli_math_mac_fx (*accu, in, krn);
            in += in_col_step;
            krn += kern_col_step;
        }
        in += in_row_step;
        krn += kern_row_step;
    }
PRAGMA_CLANG(diagnostic pop)
}

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE void dotprod3D (
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
        acc_T * accu) {
    in_ch_step -= height * in_row_step;
    kern_ch_step -= height * kern_row_step;
    in_row_step -= width * in_col_step;
    kern_row_step -= width * kern_col_step;

    __builtin_assume (height > 0);
PRAGMA_CLANG(diagnostic push)
PRAGMA_CLANG(diagnostic ignored "-Wpass-failed")
    for (int ch = 0; ch < channels; ch++) {
PRAGMA_CLANG(loop unroll(full))
        for (int row = 0; row < height; row++) {
PRAGMA_CLANG(loop unroll(full))
            for (int clmn = 0; clmn < width; clmn++) {
                *accu = mli_math_mac_fx(*accu, (*in), (*krn));
                in += in_col_step;
                krn += kern_col_step;
            }
            in += in_row_step;
            krn += kern_row_step;
        }
        in += in_ch_step;
        krn += kern_ch_step;
    }
PRAGMA_CLANG(diagnostic pop)
}

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE acc_T dotprod3D (
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
PRAGMA_CLANG(diagnostic push)
PRAGMA_CLANG(diagnostic ignored "-Wpass-failed")
    for (int ch = 0; ch < channels; ch++) {
PRAGMA_CLANG(loop unroll(full))
        for (int row = 0; row < height; row++) {
PRAGMA_CLANG(loop unroll(full))
            for (int clmn = 0; clmn < width; clmn++) {
                accu = mli_math_mac_fx(accu, (*in), (*krn));
                in += in_col_step;
                krn += kern_col_step;
            }
            in += in_row_step;
            krn += kern_row_step;
        }
        in += in_ch_step;
        krn += kern_ch_step;
    }
PRAGMA_CLANG(diagnostic pop)
    return accu;
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_DOTPROD_REF_H_
