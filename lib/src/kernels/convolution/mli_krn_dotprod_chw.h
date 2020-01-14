/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_DOTPROD_CHW_H_
#define _MLI_KRN_DOTPROD_CHW_H_

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_types.h"
#include "mli_prv_dsp.h"

/******************************************************************************
 *
 * Version & platform description
 * Targets:
 *
 ******************************************************************************/
template <typename io_T, typename w_T, typename acc_T>
static inline acc_T dotprod2D(
        const io_T* __restrict in,
        const w_T*  __restrict krn,
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
static inline void dotprod2D_v_experiment (
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
}

template < typename in_T, typename w_T, typename acc_T >
static inline void __attribute__ ((always_inline)) dotprod2D (
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
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            mli_prv_load_mac (accu, in++, krn++);
        }
        in += in_row_step;
        krn += kern_row_step;
    }
}

template < typename in_T, typename w_T, typename acc_T >
static inline void __attribute__ ((always_inline)) dotprod2D_unroll2 (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width, 
        const int height, 
        int in_row_step, 
        int kern_row_step, 
        acc_T * accu) {
    MLI_ASSERT(width % 2 == 0);

    in_row_step -= width;
    kern_row_step -= width;
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
        __builtin_assume (width % 2 == 0);
        /* unroll of 2 enables the use of dmac */
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width / 2; clmn++) {
            mli_prv_load_mac_vec2 (accu, in, krn);
            in += 2;
            krn += 2;
        }
        in += in_row_step;
        krn += kern_row_step;
    }
}

template <typename in_T, typename w_T, typename acc_T>
static inline void __attribute__((always_inline)) dotprod2D_odd(
        const MLI_PTR(in_T) __restrict in,
        const MLI_PTR(w_T) __restrict krn,
        const int width,
        const int height,
        int in_row_step,
        int kern_row_step,
        acc_T *accu) {
    in_row_step -= width;
    kern_row_step -= width;
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            mli_prv_load_mac(accu, in, krn);
            in += 1;
            krn += 1;
        }
        in += in_row_step;
        krn += kern_row_step;
    }
}
template < typename in_T, typename w_T, typename acc_T >
static inline void __attribute__ ((always_inline)) dotprod2D_unroll4 (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width, 
        const int height, 
        int in_row_step, 
        int kern_row_step, 
        acc_T * accu) {
    MLI_ASSERT(width % 4 == 0);

    in_row_step -= width;
    kern_row_step -= width;
    __builtin_assume (height > 0);
    for (int row = 0; row < height; row++) {
        __builtin_assume (width % 2 == 0);
        /* unroll of 2 enables the use of dmac
         * unroll of 4 the dmac is not directly after the corresponding load
         * which reduces the stalls.
         */
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width / 4; clmn++) {
            mli_prv_load_mac_vec4 (accu, in, krn);
            in += 4;
            krn += 4;
        }
        in += in_row_step;
        krn += kern_row_step;
    }
}

template < typename in_T, typename w_T, typename acc_T >
static inline void __attribute__ ((always_inline)) dotprod2D_mac4 (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width, 
        const int height, 
        int in_row_step, 
        int kern_row_step, 
        acc_T * accu) {
    MLI_ASSERT(width % 4 == 0);

    in_row_step -= width;
    kern_row_step -= width;
#pragma clang loop unroll(full)	
    for (int row = 0; row < height; row++) {
        __builtin_assume (width % 4 == 0);
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width / 4; clmn++) {
            mli_prv_load_mac_vec4 (accu, in, krn);
            in += 4;
            krn += 4;
        }
        in += in_row_step;
        krn += kern_row_step;
    }
}
template < typename in_T, typename w_T, typename acc_T >
static inline void __attribute__ ((always_inline)) dotprod2D_unroll4_plus1 (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width, 
        const int height, 
        int in_row_step, 
        int kern_row_step, 
        acc_T * accu) {
    MLI_ASSERT(width % 4 == 1);

    in_row_step -= width;
    kern_row_step -= width;
    __builtin_assume (height > 0);
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
        /* unroll of 2 enables the use of dmac
         * unroll of 4 the dmac is not directly after the corresponding load
         * which reduces the stalls.
         */
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width / 4; clmn++) {
            mli_prv_load_mac_vec4 (accu, in, krn);
            in += 4;
            krn += 4;
        }
        mli_prv_load_mac (accu, in, krn);
        in += 1;
        krn += 1;
        in += in_row_step;
        krn += kern_row_step;
    }
}

template < typename in_T, typename w_T, typename acc_T >
static inline void __attribute__ ((always_inline)) dotprod2D_unroll4_plus2 (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width, 
        const int height, 
        int in_row_step, 
        int kern_row_step, 
        acc_T * accu) {
    MLI_ASSERT(width % 4 == 2);

    in_row_step -= width;
    kern_row_step -= width;
    __builtin_assume (height > 0);
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
        __builtin_assume (width % 2 == 0);
        /* unroll of 2 enables the use of dmac
         * unroll of 4 the dmac is not directly after the corresponding load
         * which reduces the stalls.
         */
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width / 4; clmn++) {
            mli_prv_load_mac_vec4 (accu, in, krn);
            in += 4;
            krn += 4;
        }
        mli_prv_load_mac_vec2 (accu, in, krn);
        in += 2;
        krn += 2;
        in += in_row_step;
        krn += kern_row_step;
    }
}

template < typename in_T, typename w_T, typename acc_T >
static inline void __attribute__ ((always_inline)) dotprod2D_unroll4_plus3 (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width, 
        const int height, 
        int in_row_step, 
        int kern_row_step, 
        acc_T * accu) {
    MLI_ASSERT(width % 4 == 3);

    in_row_step -= width;
    kern_row_step -= width;
    __builtin_assume (height > 0);
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
        /* unroll of 2 enables the use of dmac
         * unroll of 4 the dmac is not directly after the corresponding load
         * which reduces the stalls.
         */
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width / 4; clmn++) {
            mli_prv_load_mac_vec4 (accu, in, krn);
            in += 4;
            krn += 4;
        }
        mli_prv_load_mac_vec2 (accu, in, krn);
        in += 2;
        krn += 2;
        mli_prv_load_mac (accu, in, krn);
        in += 1;
        krn += 1;
        in += in_row_step;
        krn += kern_row_step;
    }
}

template < typename in_T, typename w_T, typename acc_T > static inline void
dotprod1D_v (
        const MLI_PTR (in_T) __restrict in, 
        const MLI_PTR (w_T) __restrict krn, 
        const int height, 
        int in_step, 
        acc_T * accu) {
    __builtin_assume (height > 0);
//#pragma clang loop unroll(full)
    for (int32_t row = 0; row < height; row++) {
        int16_t k = (*krn++);
        v2q15_t k_v = { k, k };
        v2q15_t tx = mli_prv_load_2_samples (in);
        in += in_step;
        mli_math_mac_fx_vec2 (accu, tx, k_v);
    }
}

template < typename in_T, typename w_T, typename acc_T > static inline void
dotprod1D_v_unroll2 (
        const MLI_PTR (in_T) __restrict in, 
        const MLI_PTR (w_T) __restrict krn, 
        const int height, 
        int in_step, 
        acc_T * accu) {
    MLI_ASSERT((height & 1) == 0);
//#pragma clang loop unroll(full)
    for (int32_t row = 0; row < height / 2; row++) {
#if defined __Xxy
        int16_t k0 = (*krn++);
        v2q15_t k_v0 = {k0, k0};
        int16_t k1 = (*krn++);
        v2q15_t k_v1 = {k1, k1};
#else
        v2q15_t k01 = mli_prv_load_2_samples (krn);
        krn += 2;
        v2q15_t k_v0 = { k01[0], k01[0] };
        v2q15_t k_v1 = { k01[1], k01[1] };
#endif
        v2q15_t tx = mli_prv_load_2_samples (in);
        in += in_step;
        mli_math_mac_fx_vec2 (accu, tx, k_v0);
        tx = mli_prv_load_2_samples (in);
        in += in_step;
        mli_math_mac_fx_vec2 (accu, tx, k_v1);
    }
}

template < typename in_T, typename w_T, typename acc_T > static inline void
dotprod2D_v_simple (
        const MLI_PTR (in_T) __restrict in, 
        const MLI_PTR (w_T) __restrict krn, 
        const int width, 
        const int height, 
        int in_row_step, 
        int kern_row_step, 
        acc_T * accu) {
    in_row_step -= width;
    kern_row_step -= width;

#pragma clang loop unroll(full)
    for (int32_t row = 0; row < height; row++) {
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
}

template < typename in_T, typename w_T, typename acc_T > static inline void
dotprod3D_v_simple (
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
}

/* not defining K_ODD reduces a single load for the weights.
 * to be investigated if this has a significant performance impact.
 * it could also impact the amount of unaligned loads.
 */
//#define K_ODD
template < typename in_T, typename w_T, typename acc_T, int is_even >
static inline void __attribute__ ((always_inline)) dotprod2D_v_odd_even (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width, 
        const int height, 
        int in_row_step, 
        int kern_row_step, 
        acc_T * accu) {
    in_row_step -= (width + 1);
#ifdef K_ODD
    kern_row_step -= width;
#else
    kern_row_step -= (width + 1);
#endif
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#ifdef K_ODD
        q15_t k = (*krn++);
        v2q15_t k_v0 = { k, k };
#else
        v2q15_t k01 = mli_prv_load_2_samples (krn);
        krn += 2;
        v2q15_t k_v0 = { k01[0], k01[0] };
#endif
        v2q15_t in_v0 = mli_prv_load_2_samples (in);
        in += 2;
        mli_math_mac_fx_vec2 (accu, in_v0, k_v0);
#pragma clang loop unroll(full)
        for (int32_t clmn = 0; clmn < (width - 1) / 2; clmn++) {
#ifdef K_ODD
            v2q15_t k12 = mli_prv_load_2_samples (krn);
            krn += 2;
            v2q15_t k_v1 = { k12[0], k12[0] };
            v2q15_t k_v2 = { k12[1], k12[1] };
#else
            v2q15_t k23 = mli_prv_load_2_samples (krn);
            krn += 2;
            v2q15_t k_v1 = { k01[1], k01[1] };
            v2q15_t k_v2 = { k23[0], k23[0] };
#endif

            v2q15_t in_v2 = mli_prv_load_2_samples (in);
            in += 2;
            v2q15_t in_v1 = { in_v0[1], in_v2[0] };

            mli_math_mac_fx_vec2 (accu, in_v1, k_v1);
            mli_math_mac_fx_vec2 (accu, in_v2, k_v2);

            in_v0 = in_v2;
#ifndef K_ODD
            k01 = k23;
#endif
        }
        if (is_even) {
            if ((width & 1) == 0) {
#ifdef K_ODD
                q15_t k = (*krn++);
                v2q15_t k_v = { k, k };
#else
                v2q15_t k_v = { k01[1], k01[1] };
                krn++;
#endif
                v2q15_t in_v2 = mli_prv_load_2_samples (in);
                in += 1;
                v2q15_t in_v1 = { in_v0[1], in_v2[0] };

                mli_math_mac_fx_vec2 (accu, in_v1, k_v);
            }
        }
        in += in_row_step;
        krn += kern_row_step;
    }
}

template < typename in_T, typename w_T, typename acc_T >
static inline void __attribute__ ((always_inline)) dotprod2D_v (
        const MLI_PTR (in_T) __restrict in,
        const MLI_PTR (w_T) __restrict krn,
        const int width, 
        const int height, 
        int in_row_step, 
        int kern_row_step, 
        acc_T * accu) {
#if !defined __Xxy
    /* the odd_even version of dotprod is optimized to reduce the number of loads.
     * this is not optimal for AGU based systems.
     */
    if (width & 1) {
        dotprod2D_v_odd_even < in_T, w_T, acc_T, 0 > (in, krn, width, height, in_row_step, kern_row_step, accu);
    } else {
        dotprod2D_v_odd_even < in_T, w_T, acc_T, 1 > (in, krn, width, height, in_row_step, kern_row_step, accu);
    }
#else
    dotprod2D_v_simple(in, krn, width, height, in_row_step, kern_row_step, accu);
#endif
}

#endif // _MLI_KRN_DOTPROD_CHW_H_
