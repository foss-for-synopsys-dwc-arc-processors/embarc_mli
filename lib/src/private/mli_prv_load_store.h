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

#ifndef _MLI_PRV_LOAD_STORE_H_
#define _MLI_PRV_LOAD_STORE_H_

#include <assert.h>

#include "mli_config.h" /* for MLI_PTR */
#include "mli_private_types.h"

#ifdef _ARC
#include <arc/arc_intrinsics.h>
#endif


static inline v2q15_t __attribute__ ((always_inline)) mli_prv_load_2_samples (const MLI_PTR (int8_t) __restrict in) {
#ifndef _ARC
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

static inline void __attribute__ ((always_inline)) mli_prv_store_2_samples (MLI_PTR (int8_t) __restrict out, v2q15_t data) {
    *(MLI_PTR (v2i8_t)) out = __builtin_convertvector (data, v2i8_t);
}

static inline void __attribute__ ((always_inline)) mli_prv_store_2_samples (MLI_PTR (int8_t) __restrict out, v2i8_t data) {
    *(MLI_PTR (v2i8_t)) out = data;
}

static inline void __attribute__ ((always_inline)) mli_prv_store_2_samples (MLI_PTR (int16_t) __restrict out, v2q15_t data) {
    *(MLI_PTR (v2q15_t)) out = data;
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

static inline void __attribute__ ((always_inline)) mli_prv_store_1_sample (MLI_PTR (int8_t) __restrict out, v2q15_t data) {
    *(MLI_PTR (q7_t)) out = (q7_t) data[0];
}

static inline void __attribute__ ((always_inline)) mli_prv_store_1_sample (MLI_PTR (int16_t) __restrict out, v2q15_t data) {
    *(MLI_PTR (q15_t)) out = data[0];
}



#endif //_MLI_PRV_LOAD_STORE_H_
