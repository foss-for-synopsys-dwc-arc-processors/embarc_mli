/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_LOAD_STORE_H_
#define _MLI_PRV_LOAD_STORE_H_

#include <assert.h>

#include "mli_config.h" /* for MLI_PTR */
#include "mli_private_types.h"

#ifdef _ARC
#include <arc/arc_intrinsics.h>
#endif


static inline v2q15_t __attribute__ ((always_inline)) mli_prv_load_2_samples (const MLI_PTR (int8_t) __restrict in) {
#if defined __Xxy
    return __builtin_convertvector (*(MLI_PTR (v2i8_t)) in, v2q15_t);
#else
    int16_t two8bitvalues = *(MLI_PTR (int16_t)) in;
    v2q15_t packedvalues = (v2q15_t) _vsext2bhl (two8bitvalues);
    return packedvalues;
#endif
}

static inline v2q15_t __attribute__ ((always_inline)) mli_prv_load_2_samples (const MLI_PTR (int16_t) __restrict in) {
    return *(MLI_PTR (v2q15_t)) in;
}

static inline v4q15_t __attribute__ ((always_inline)) mli_prv_load_4_samples (const MLI_PTR (int16_t) __restrict in) {
    return *(MLI_PTR (v4q15_t)) in;
}
static inline void __attribute__ ((always_inline)) mli_prv_store_2_samples (MLI_OUT_PTR (int8_t) __restrict out, v2q15_t data) {
    *(MLI_OUT_PTR (v2i8_t)) out = __builtin_convertvector (data, v2i8_t);
}

static inline void __attribute__ ((always_inline)) mli_prv_store_2_samples (MLI_OUT_PTR (int8_t) __restrict out, v2i8_t data) {
    *(MLI_OUT_PTR (v2i8_t)) out = data;
}

static inline void __attribute__ ((always_inline)) mli_prv_store_2_samples (MLI_OUT_PTR (int16_t) __restrict out, v2q15_t data) {
    *(MLI_OUT_PTR (v2q15_t)) out = data;
}


static inline void __attribute__ ((always_inline)) mli_prv_sat_and_store_2_samples (MLI_PTR (int8_t) __restrict out, v2q15_t data) {
    const v2u16_t sat_v2= {8, 8};
    *(MLI_PTR (v2i8_t)) out = __builtin_convertvector (fx_sat_v2q15(data, sat_v2), v2i8_t);
}

static inline void __attribute__ ((always_inline)) mli_prv_sat_and_store_2_samples (MLI_PTR (int16_t) __restrict out, v2q15_t data) {
    /*You don't need to do additional saturation, because of it already built into the 16-bit FXAPI functions.*/
    *(MLI_PTR (v2q15_t)) out = data;
}

static inline v2q15_t __attribute__ ((always_inline)) mli_prv_load_1_sample (const MLI_PTR (int8_t) __restrict in) {
    return fx_create_v2q15((q15_t) (*(MLI_PTR (q7_t)) in), 0);
}

static inline v2q15_t __attribute__ ((always_inline)) mli_prv_load_1_sample (const MLI_PTR (int16_t) __restrict in) {
    return fx_create_v2q15(*(MLI_PTR (q15_t)) in, 0);
}

static inline void __attribute__ ((always_inline)) mli_prv_store_1_sample (MLI_OUT_PTR (int8_t) __restrict out, v2q15_t data) {
    *(MLI_OUT_PTR (q7_t)) out = (q7_t) data[0];
}

static inline void __attribute__ ((always_inline)) mli_prv_store_1_sample (MLI_OUT_PTR (int16_t) __restrict out, v2q15_t data) {
    *(MLI_OUT_PTR (q15_t)) out = data[0];
}



#endif //_MLI_PRV_LOAD_STORE_H_
