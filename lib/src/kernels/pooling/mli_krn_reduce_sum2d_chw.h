/*
 *  Copyright (c) 2019, Synopsys, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1) Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2)  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3) Neither the name of the <ORGANIZATION> nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _MLI_KRN_REDUCE_SUM2D_CHW_H_
#define _MLI_KRN_REDUCE_SUM2D_CHW_H_

#include "mli_config.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_load_store.h"

/******************************************************************************
 *
 * Version & platform description
 * Targets:
 *
 ******************************************************************************/

// For width > 4, and height > 2, and even
static inline int32_t __attribute__((always_inline)) reduce_sum2D_k4_Nx2_N_even(
        const MLI_PTR(int8_t) in,
        const int32_t width,
        const int32_t height,
        const int32_t in_row_step) {
    int32_t acc = 0;
    __builtin_assume(height >= 2);
    __builtin_assume(width >= 4);
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            acc += in[clmn];
        }
        in += in_row_step;
    }
    return acc;
}

// For width > 4, and height > 2, and even
static inline int32_t __attribute__((always_inline)) reduce_sum2D_k4_Nx2_N_even(
        const MLI_PTR(int16_t) __restrict in,
        const int32_t width,
        const int32_t height,
        const int32_t in_row_step) {
    const short one_v[4] = {0x0001, 0x0001, 0x0001, 0x0001};
    const int32_t width_half = width >> 1;
    MLI_PTR(int) p_in = (MLI_PTR(int))in;

    long long acc = {0};
    __builtin_assume(height >= 2);
    __builtin_assume(width >= 2);
    __builtin_assume((height % 2) == 0);
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
        __builtin_assume(width_half >= 2);
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width_half; clmn++) {
            int one = *(int *)&one_v[clmn];
            int in_v = *p_in;
            p_in++;

            acc += (long long)(short)(in_v) * (short)one + (in_v >> 16) * (one >> 16);
        }

        p_in += (in_row_step - width) / 2;
    }
    return (int)(acc);
}

static inline int32_t __attribute__((always_inline)) reduce_sum2D(
        const MLI_PTR(int8_t) __restrict in,
        const int32_t width,
        const int32_t height,
        const int32_t in_row_step) {
    int32_t acc = 0;
    const v2q15_t one_v = {0x0001, 0x0001};
    accum40_t acc40 = fx_create_a40(0x0, 0x0);
    if (width & 1) {
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
            acc += in[0];
#pragma clang loop unroll(full)
            for (int clmn = 1; clmn < width; clmn += 2) {
                acc40 = fx_a40_dmac_v2q15(acc40, mli_prv_load_2_samples(&in[clmn]), one_v);
            }
            in += in_row_step;
        }
    } else {
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn += 2) {
                acc40 = fx_a40_dmac_v2q15(acc40, mli_prv_load_2_samples(&in[clmn]), one_v);
            }
            in += in_row_step;
        }
    }
    acc += (int32_t)fx_q31_cast_asr_rnd_a40(acc40, 1);
    return acc;
}

static inline int32_t __attribute__((always_inline)) reduce_sum2D(
        const MLI_PTR(int16_t) __restrict in,
        const int32_t width,
        const int32_t height,
        const int32_t in_row_step) {
    int32_t acc = 0;
    const v2q15_t one_v = {0x0001, 0x0001};
    accum40_t acc40 = fx_create_a40(0x0, 0x0);
    if (width & 1) {
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
            acc += in[0];
            MLI_PTR(v2q15_t) p_in = (MLI_PTR(v2q15_t))(in + 1);
#pragma clang loop unroll(full)
            for (int clmn = 1; clmn < width; clmn += 2) {
                acc40 = fx_a40_dmac_v2q15(acc40, *p_in, one_v);
                p_in++;
            }
            in += in_row_step;
        }
    } else {
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
            MLI_PTR(v2q15_t) p_in = (MLI_PTR(v2q15_t))in;
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn += 2) {
                acc40 = fx_a40_dmac_v2q15(acc40, *p_in, one_v);
                p_in++;
            }
            in += in_row_step;
        }
    }
    acc += (int32_t)fx_q31_cast_asr_rnd_a40(acc40, 1);
    return acc;
}

template <typename io_T>
static inline int32_t __attribute__((always_inline)) reduce_sum2D_odd(
        const MLI_PTR(io_T) __restrict in,
        const int32_t width,
        const int32_t height,
        const int32_t in_row_step) {
    int32_t acc = 0;
    const v2i16_t one_v = {0x0001, 0x0001};
    accum40_t acc40 = fx_create_a40(0x0, 0x0);
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
        acc += in[0];
#pragma clang loop unroll(full)
        for (int clmn = 1; clmn < width; clmn += 2) {
            acc40 = fx_a40_dmac_v2q15(acc40, mli_prv_load_2_samples(&in[clmn]), one_v);
        }
        in += in_row_step;
    }
    acc += (int32_t)fx_q31_cast_asr_rnd_a40(acc40, 1);
    return acc;
}

template <typename io_T>
static inline void __attribute__((always_inline)) reduce_sum2D_odd(
        accum40_t *__restrict acc40,
        const MLI_PTR(io_T) __restrict in,
        const int32_t width,
        const int32_t height,
        const int32_t in_row_step) {
    const v2i16_t one_v = {0x0001, 0x0001};
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
        *acc40 = fx_a40_mac_q15(*acc40, in[0], 0x0001);
#pragma clang loop unroll(full)
        for (int clmn = 1; clmn < width; clmn += 2) {
            *acc40 = fx_a40_dmac_v2q15(*acc40, mli_prv_load_2_samples(&in[clmn]), one_v);
        }
        in += in_row_step;
    }
}

template <typename io_T>
static inline int32_t __attribute__((always_inline)) reduce_sum2D_even(
        const MLI_PTR(io_T) __restrict in,
        const int32_t width,
        const int32_t height,
        const int32_t in_row_step) {
    int32_t acc = 0;
    const v2q15_t one_v = {0x0001, 0x0001};
    accum40_t acc40 = fx_create_a40(0x0, 0x0);
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn += 2) {
            acc40 = fx_a40_dmac_v2q15(acc40, mli_prv_load_2_samples(&in[clmn]), one_v);
        }
        in += in_row_step;
    }
    acc += fx_q31_cast_asr_rnd_a40(acc40, 1);
    return acc;
}

template <typename io_T>
static inline void __attribute__((always_inline)) reduce_sum2D_even(
        accum40_t *__restrict acc40,
        const MLI_PTR(io_T) __restrict in,
        const int32_t width,
        const int32_t height,
        const int32_t in_row_step) {
    const v2q15_t one_v = {0x0001, 0x0001};
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn += 2) {
            *acc40 = fx_a40_dmac_v2q15(*acc40, mli_prv_load_2_samples(&in[clmn]), one_v);
        }
        in += in_row_step;
    }
}

template <typename io_T>
static inline void __attribute__((always_inline)) reduce_sum2D(
        accum40_t *__restrict acc40,
        const MLI_PTR(io_T) __restrict in,
        const int32_t width,
        const int32_t height,
        const int32_t in_row_step) {
    const v2q15_t one_v = {0x0001, 0x0001};
    if (width & 1) {
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
            *acc40 = fx_a40_mac_q15(*acc40, in[0], 0x0001);
#pragma clang loop unroll(full)
            for (int clmn = 1; clmn < width; clmn += 2) {
                *acc40 = fx_a40_dmac_v2q15(*acc40, mli_prv_load_2_samples(&in[clmn]), one_v);
            }
            in += in_row_step;
        }
    } else {
#pragma clang loop unroll(full)
        for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
            for (int clmn = 0; clmn < width; clmn += 2) {
                *acc40 = fx_a40_dmac_v2q15(*acc40, mli_prv_load_2_samples(&in[clmn]), one_v);
            }
            in += in_row_step;
        }
    }
}

#endif  //_MLI_KRN_REDUCE_SUM2D_CHW_H_
