/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _VDSP_MLI_PRV_LOAD_STORE_H_
#define _VDSP_MLI_PRV_LOAD_STORE_H_

#include <arc_vector.h>
#include "mli_config.h"

// Depending on memory alignment of input pointers, certain functions below will perform
// unaligned loads/stores. Since the core supports this, we disable the related compiler warning.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-align"

static MLI_FORCE_INLINE vNx4char_t mli_prv_load_n_samples(const MLI_PTR(int8_t) __restrict in) {
	return *(MLI_PTR (vNx4char_t)) in;
}

static MLI_FORCE_INLINE vNx4short_t mli_prv_load_n_samples(const MLI_PTR(int16_t) __restrict in) {
    return *(MLI_PTR (vNx4short_t)) in;
}

static MLI_FORCE_INLINE vNx4short_t mli_prv_gather_load_n_samples(const MLI_PTR(short) in, vNx4int_t offsets) {
    vNx4short_t out;
    out.lo = vgather(in, offsets.lo);
    out.hi = vgather(in, offsets.hi);
    return out;
}

static MLI_FORCE_INLINE void mli_prv_store_n_samples(MLI_OUT_PTR (int8_t) __restrict out, vNx4char_t data) {
    *(MLI_OUT_PTR (vNx4char_t)) out = data;
}

static MLI_FORCE_INLINE void mli_prv_store_n_samples(MLI_OUT_PTR (int16_t) __restrict out, vNx4short_t data) {
    *(MLI_OUT_PTR (vNx4short_t)) out = data;
}

static MLI_FORCE_INLINE void mli_prv_store_n_samples(MLI_OUT_PTR (int16_t) __restrict out, vNx2short_t data) {
    *(MLI_OUT_PTR (vNx2short_t)) out = data;
}

static MLI_FORCE_INLINE void mli_prv_store_n_samples(MLI_OUT_PTR (int8_t) __restrict out,
        vNx4char_t data, pvNx4 predicate) {
    vvst(data, predicate, (int8_t __vccm *)(out));
}

static MLI_FORCE_INLINE void mli_prv_store_n_samples(MLI_OUT_PTR (int16_t) __restrict out,
        vNx2short_t data, pvNx2 predicate) {
    vvst(data, predicate, (int16_t __vccm *)(out));
}

static MLI_FORCE_INLINE pvNx4 mli_prv_pvNx4_init(int limit) {
    return to_pvNx4(vvci_b() < limit);
}

static MLI_FORCE_INLINE pvNx2 mli_prv_pvNx2_init(int limit) {
    return to_pvNx2(vvci_h() < limit);
}

static MLI_FORCE_INLINE void mli_prv_store_n_samples(MLI_OUT_PTR (int8_t) __restrict out,
        vNx4char_t data, int predicate_limit) {
    pvNx4 predicate = mli_prv_pvNx4_init(predicate_limit);
    vvst(data, predicate, (int8_t __vccm *)(out));
}

static MLI_FORCE_INLINE void mli_prv_store_n_samples(MLI_OUT_PTR (int16_t) __restrict out,
        vNx4short_t data, int predicate_limit) {
    if (predicate_limit > _VDSP_NUM_16BIT_LANES) {
        mli_prv_store_n_samples(out, data.lo);
        out += _VDSP_NUM_16BIT_LANES;
        pvNx2 predicate = mli_prv_pvNx2_init(predicate_limit - _VDSP_NUM_16BIT_LANES);
        vvst(data.hi, predicate, (int16_t __vccm *)(out));
    } else {
        pvNx2 predicate = mli_prv_pvNx2_init(predicate_limit);
        vvst(data.lo, predicate, (int16_t __vccm *)(out));
    }
}

#pragma clang diagnostic pop

#endif // _VDSP_MLI_PRV_LOAD_STORE_H_
