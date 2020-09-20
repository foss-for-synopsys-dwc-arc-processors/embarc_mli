/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_DOTPROD_DSP_H_
#define _MLI_KRN_DOTPROD_DSP_H_

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_math.h"
#include "mli_types.h"
#include "mli_prv_dsp.h"

namespace mli {
namespace krn {
namespace dsp {

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
static MLI_FORCE_INLINE void dotprod2D_hwc_v (
        const MLI_PTR(in_T) __restrict in, 
        const MLI_PTR(w_T) __restrict krn,
        acc_T * accu,        
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step) {
    in_row_step -= width * in_col_step;
    kern_row_step -= width * kern_col_step;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang loop unroll(full)
    for (int32_t row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int32_t clmn = 0; clmn < width; clmn++) {
            v2q15_t k_v = mli_prv_load_2_samples(krn);
            krn += kern_col_step;
            v2q15_t tx = mli_prv_load_2_samples(in);
            in += in_col_step;
            mli_math_mac_fx_vec2 (accu, tx, k_v);
        }
        in += in_row_step;
        krn += kern_row_step;
    }
#pragma clang diagnostic pop
}

//The function uses pointers to pointers for in and krn. 
//The caller of the function should compensate for the increment
//done inside this function.
template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE void dotprod2D_hwc_v (
        const MLI_PTR(in_T) __restrict *in, 
        const MLI_PTR(w_T) __restrict *krn,
        acc_T * accu,        
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step) {

    in_row_step -= width * in_col_step;
    kern_row_step -= width * kern_col_step;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang loop unroll(full)
    for (int32_t row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int32_t clmn = 0; clmn < width; clmn++) {
            v2q15_t k_v = mli_prv_load_2_samples(*krn);
            *krn += kern_col_step;
            v2q15_t tx = mli_prv_load_2_samples(*in);
            *in += in_col_step;
            mli_math_mac_fx_vec2 (accu, tx, k_v);
        }
        *in += in_row_step;
        *krn += kern_row_step;
    }
#pragma clang diagnostic pop
}

//The function uses pointers to pointers for in and krn. 
//The caller of the function should compensate for the increment
//done inside this function.
template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod2D_inp_width_v(
        const MLI_PTR(io_T) __restrict *inp,
        const MLI_PTR(w_T)  __restrict *krn,
        acc_T *accu,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step,
        int in_width_step) {
    in_row_step -= width * in_col_step;
    kern_row_step -= width * kern_col_step;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            int16_t k = **krn;
            v2q15_t k_v = { k, k };
            v2q15_t in_v = {(*inp)[0], (*inp)[in_width_step]};
            mli_math_mac_fx_vec2(accu, in_v, k_v);
            *inp += in_col_step;
            *krn += kern_col_step;
        }
        *inp += in_row_step;
        *krn += kern_row_step;
    }
#pragma clang diagnostic pop
    return *accu;
}

//The function uses pointers to pointers for in and krn. 
//The caller of the function should compensate for the increment
//done inside this function.
template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod2D(
        const MLI_PTR(io_T) __restrict *in,
        const MLI_PTR(w_T)  __restrict *krn,
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
            accu = mli_math_mac_fx(accu, (**in), (**krn));
            *in += in_col_step;
            *krn += kern_col_step;
        }
        *in += in_row_step;
        *krn += kern_row_step;
    }
    return accu;
}
template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE void dotprod2D_hwc_v_point (
        const MLI_PTR(in_T) __restrict in, 
        const MLI_PTR(w_T) __restrict krn,
        acc_T * accu) {
    v2q15_t k_v = mli_prv_load_2_samples(krn);
    v2q15_t tx = mli_prv_load_2_samples(in);
    mli_math_mac_fx_vec2 (accu, tx, k_v);
}

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE void dotprod2D (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width, 
        const int height, 
        int in_row_step, 
        int kern_row_step, 
        acc_T * accu) {
    in_row_step -= width;
    kern_row_step -= width;
    __builtin_assume (height > 0);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            mli_prv_load_mac (accu, in++, krn++);
        }
        in += in_row_step;
        krn += kern_row_step;
    }
#pragma clang diagnostic pop
}

template < typename in_T, typename w_T, typename acc_T > 
static MLI_FORCE_INLINE void dotprod3D_v_simple (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width,
        const int height,
        const int in_ch,
        int in_row_step,
        int kern_row_step,
        int in_ch_step,
        int kern_ch_step,
        acc_T * accu) {
    in_ch_step -= height * in_row_step;
    kern_ch_step -= height * kern_row_step;
    in_row_step -= width;
    kern_row_step -= width;

    __builtin_assume(in_ch > 0);
    __builtin_assume(height > 1);
    for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang loop unroll(full)
        for (int row = 0; row < height-1; row++) {
#pragma clang loop unroll(full)
            for (int32_t clmn = 0; clmn < width; clmn++) {
                int16_t k = (*krn++);
                v2q15_t k_v = { k, k };
                v2q15_t tx = mli_prv_load_2_samples (in);
                in++;
                mli_math_mac_fx_vec2 (accu, tx, k_v);
            }
            in += in_row_step;
            krn += kern_row_step;
        }
#pragma clang loop unroll(full)
        for (int32_t clmn = 0; clmn < width/*+1*/; clmn++) {
            int16_t k = (*krn++);
            v2q15_t k_v = { k, k };
            v2q15_t tx = mli_prv_load_2_samples (in);
            in++;
            mli_math_mac_fx_vec2 (accu, tx, k_v);
        }
        in += in_row_step;
        krn += kern_row_step;

        in += in_ch_step;
        krn += kern_ch_step;
    }
#pragma clang diagnostic pop
}


template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod2D_inp_width_v(
        const MLI_PTR(io_T) __restrict inp,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T *accu,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step,
        int kern_col_step,
        int kern_row_step,
        int in_width_step) {
    in_row_step -= width * in_col_step;
    kern_row_step -= width * kern_col_step;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            int16_t k = *krn;
            v2q15_t k_v = { k, k };
            v2q15_t in_v = {inp[0], inp[in_width_step]};
            mli_math_mac_fx_vec2(accu, in_v, k_v);
            inp += in_col_step;
            krn += kern_col_step;
        }
        inp += in_row_step;
        krn += kern_row_step;
    }
#pragma clang diagnostic pop
    return *accu;
}


} // namespace dsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_DOTPROD_DSP_H_
