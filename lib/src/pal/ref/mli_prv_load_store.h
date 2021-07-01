/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _REF_MLI_PRV_LOAD_STORE_H_
#define _REF_MLI_PRV_LOAD_STORE_H_

#include "mli_mem_info.h"

static MLI_FORCE_INLINE int16_t mli_prv_load_1vec(const MLI_PTR (int16_t) __restrict in) {
    return *(MLI_PTR (int16_t)) in;
}

static MLI_FORCE_INLINE int8_t mli_prv_load_1vec(const MLI_PTR (int8_t) __restrict in) {
    return *(MLI_PTR (int8_t)) in;
}

template<typename io_T>
static MLI_FORCE_INLINE void mli_prv_store_n_samples(MLI_OUT_PTR (io_T) __restrict out, io_T data) {
    *out = data;
}

template<typename io_T>
static MLI_FORCE_INLINE void mli_prv_store_n_samples(MLI_OUT_PTR (io_T) __restrict out,
        io_T data, int predicate) {
    MLI_ASSERT(predicate <= 1);
    if (predicate == 1) {
        mli_prv_store_n_samples(out, data);
    }
}

#endif // _REF_MLI_PRV_LOAD_STORE_H_
