/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _DSP_MLI_PRV_LOAD_STORE_H_
#define _DSP_MLI_PRV_LOAD_STORE_H_

#include <assert.h>

#include "mli_config.h"
#include "mli_mem_info.h"
#include "mli_private_types.h"

#ifdef _ARC
#include <arc/arc_intrinsics.h>
#endif

// Depending on memory alignment of input pointers, certain functions below will perform
// unaligned loads/stores. Since the core supports this, we disable the related compiler warning.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-align"

static MLI_FORCE_INLINE v2q15_t mli_prv_load_2_samples (const MLI_PTR (int8_t) __restrict in) {
#if defined __Xxy
    return __builtin_convertvector (*(MLI_PTR (v2i8_t)) in, v2q15_t);
#else
    int16_t two8bitvalues = *(MLI_PTR (int16_t)) in;
    v2q15_t packedvalues = (v2q15_t) _vsext2bhl (two8bitvalues);
    return packedvalues;
#endif
}

static MLI_FORCE_INLINE v2q15_t mli_prv_load_2_samples (const MLI_PTR (int16_t) __restrict in) {
    return *(MLI_PTR (v2q15_t)) in;
}

/* workaround
 * TODO: remove this condition after reverting the workaround in pal/mli_prv_dsp.h,
 * which is using dsp/dsp/mli_prv_dsp.h while building reference variation
 */
#ifndef _REF_MLI_PRV_LOAD_STORE_H_
template <typename in_T>
static MLI_FORCE_INLINE v2q15_t mli_prv_load_1vec (const in_T __restrict in) {
    return mli_prv_load_2_samples(in);
}
#endif

static MLI_FORCE_INLINE v4q15_t mli_prv_load_4_samples (const MLI_PTR (int16_t) __restrict in) {
    return *(MLI_PTR (v4q15_t)) in;
}
static MLI_FORCE_INLINE void mli_prv_store_2_samples (MLI_OUT_PTR (int8_t) __restrict out, v2q15_t data) {
    *(MLI_OUT_PTR (v2i8_t)) out = __builtin_convertvector (data, v2i8_t);
}

static MLI_FORCE_INLINE void mli_prv_store_2_samples (MLI_OUT_PTR (int8_t) __restrict out, v2i8_t data) {
    *(MLI_OUT_PTR (v2i8_t)) out = data;
}

static MLI_FORCE_INLINE void mli_prv_store_2_samples (MLI_OUT_PTR (int16_t) __restrict out, v2q15_t data) {
    *(MLI_OUT_PTR (v2q15_t)) out = data;
}

static MLI_FORCE_INLINE void mli_prv_store_2_samples (MLI_OUT_PTR (int32_t) __restrict out, v2q15_t data) {
    *(MLI_OUT_PTR (__v2i32_t)) out = __builtin_convertvector (data, __v2i32_t);
}

static MLI_FORCE_INLINE void mli_prv_sat_and_store_2_samples (MLI_PTR (int8_t) __restrict out, v2q15_t data) {
    const v2u16_t sat_v2= {8, 8};
    *(MLI_PTR (v2i8_t)) out = __builtin_convertvector (fx_sat_v2q15(data, sat_v2), v2i8_t);
}

static MLI_FORCE_INLINE void mli_prv_sat_and_store_2_samples (MLI_PTR (int16_t) __restrict out, v2q15_t data) {
    /*You don't need to do additional saturation, because of it already built into the 16-bit FXAPI functions.*/
    *(MLI_PTR (v2q15_t)) out = data;
}

static MLI_FORCE_INLINE v2q15_t mli_prv_load_1_sample (const MLI_PTR (int8_t) __restrict in) {
    return fx_create_v2q15((q15_t) (*(MLI_PTR (q7_t)) in), 0);
}

static MLI_FORCE_INLINE v2q15_t mli_prv_load_1_sample (const MLI_PTR (int16_t) __restrict in) {
    return fx_create_v2q15(*(MLI_PTR (q15_t)) in, 0);
}

static MLI_FORCE_INLINE void mli_prv_store_1_sample (MLI_OUT_PTR (int8_t) __restrict out, v2q15_t data) {
    *(MLI_OUT_PTR (q7_t)) out = (q7_t) data[0];
}

static MLI_FORCE_INLINE void mli_prv_store_1_sample (MLI_OUT_PTR (int16_t) __restrict out, v2q15_t data) {
    *(MLI_OUT_PTR (q15_t)) out = data[0];
}

static MLI_FORCE_INLINE void mli_prv_store_1_sample (MLI_OUT_PTR (int32_t) __restrict out, v2q15_t data) {
    *(MLI_OUT_PTR (q31_t)) out = (q31_t) data[0];
}

static MLI_FORCE_INLINE void mli_prv_sat_and_store_1_sample (MLI_OUT_PTR (int8_t) __restrict out, v2q15_t data) {
    *(MLI_OUT_PTR (q7_t)) out = (int8_t)fx_sat_q15(data[0], 8);
}

static MLI_FORCE_INLINE void mli_prv_sat_and_store_1_sample (MLI_OUT_PTR (int16_t) __restrict out, v2q15_t data) {
	*(MLI_OUT_PTR (q15_t)) out = data[0];
}


template <typename out_T>
static MLI_FORCE_INLINE void mli_prv_store_n_samples(out_T __restrict out, v2q15_t data, int predicate) {
    MLI_ASSERT(predicate <= 2);
    if (predicate == 1) {
        mli_prv_store_1_sample(out, data);
    } else if(predicate == 2) {
        mli_prv_store_2_samples(out, data);
    }
}

template <typename out_T>
static MLI_FORCE_INLINE void mli_prv_store_n_samples(out_T __restrict out, v2q15_t data) {
    mli_prv_store_2_samples(out, data);
}

#pragma clang diagnostic pop

#endif //_DSP_MLI_PRV_LOAD_STORE_H_
