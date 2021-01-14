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
#include "arc_vector_ext.h"
#include "mli_config.h"
#include "../mli_math.h"

// Depending on memory alignment of input pointers, certain functions below will perform
// unaligned loads/stores. Since the core supports this, we disable the related compiler warning.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-align"

static MLI_FORCE_INLINE pvNx4 mli_prv_pvNx4_init(int limit) {
    return to_pvNx4(vvci_b() < limit);
}

static MLI_FORCE_INLINE pvNx2 mli_prv_pvNx2_init(int limit) {
    return to_pvNx2(vvci_h() < limit);
}

//
// load function where number of samples is based on N, and number or vectors depends on the type.
//
// this type of load functions is used in code where data from different types is combined and same
// amount of samples need to be loaded from each buffer.
static MLI_FORCE_INLINE vNx4char_t mli_prv_load_nx4_samples(const MLI_PTR(int8_t) __restrict in) {
    return *(MLI_PTR (vNx4char_t)) in;
}

static MLI_FORCE_INLINE vNx4short_t mli_prv_load_nx4_samples(const MLI_PTR(int16_t) __restrict in) {
    return *(MLI_PTR (vNx4short_t)) in;
}

static MLI_FORCE_INLINE vNx4int_t mli_prv_load_nx4_samples(const MLI_PTR(int32_t) __restrict in) {
    return *(MLI_PTR (vNx4int_t)) in;
}

static MLI_FORCE_INLINE vNx2short_t mli_prv_load_nx2_samples(const MLI_PTR(int16_t) __restrict in) {
    return *(MLI_PTR (vNx2short_t)) in;
}

static MLI_FORCE_INLINE vNx2int_t mli_prv_load_nx2_samples(const MLI_PTR(int32_t) __restrict in) {
    return *(MLI_PTR (vNx2int_t)) in;
}

static MLI_FORCE_INLINE vNint_t mli_prv_load_nx1_samples(const MLI_PTR(int32_t) __restrict in) {
    return *(MLI_PTR (vNint_t)) in;
}

/* vector load from dcache */
static MLI_FORCE_INLINE vNx4char_t mli_prv_load_nx4_samples(const int8_t* __restrict in) {
    vNx4char_t r;
    for (int i = 0; i < (1 * _VDSP_NUM_8BIT_LANES); i++) {
        r[i] = in[i];
    }
    return r;
}

static MLI_FORCE_INLINE vNx4short_t mli_prv_load_nx4_samples(const int16_t* __restrict in) {
    vNx4short_t r;
    for (int i = 0; i < _VDSP_NUM_16BIT_LANES; i++) {
        r.lo[i] = in[i];
        r.hi[i] = in[i + _VDSP_NUM_16BIT_LANES];
    }
    return r;
}

static MLI_FORCE_INLINE vNx4int_t mli_prv_load_nx4_samples(const int32_t* __restrict in) {
    vNx4int_t r;
    for (int i = 0; i < (4 * _VDSP_NUM_32BIT_LANES); i++) {
        r[i] = in[i];
    }
    return r;
}

static MLI_FORCE_INLINE vNx2short_t mli_prv_load_nx2_samples(const int16_t* __restrict in) {
    vNx2short_t r;
    for (int i = 0; i < (1 * _VDSP_NUM_16BIT_LANES); i++) {
        r[i] = in[i];
    }
    return r;
}

static MLI_FORCE_INLINE vNx2int_t mli_prv_load_nx2_samples(const int32_t* __restrict in) {
    vNx2int_t r;
    for (int i = 0; i < (2 * _VDSP_NUM_32BIT_LANES); i++) {
        r[i] = in[i];
    }
    return r;
}

static MLI_FORCE_INLINE vNint_t mli_prv_load_nx1_samples(const int32_t* __restrict in) {
    vNint_t r;
    for (int i = 0; i < (1 * _VDSP_NUM_32BIT_LANES); i++) {
        r[i] = in[i];
    }
    return r;
}

// vector gather load
static MLI_FORCE_INLINE vNx4short_t mli_prv_gather_load_nx4_samples(const MLI_PTR(short) in, vNx4int_t offsets) {
    vNx4short_t out;
    out.lo = vgather(in, offsets.lo);
    out.hi = vgather(in, offsets.hi);
    return out;
}

static MLI_FORCE_INLINE vNx2short_t mli_prv_gather_load_nx2_samples(const MLI_PTR(int16_t) in, vNx2int_t offsets, int num) {
    return vgather(in, offsets, (vNx2short_t)0, mli_prv_pvNx2_init(num));
}

static MLI_FORCE_INLINE vNx4char_t mli_prv_gather_load_nx2_samples(const MLI_PTR(int8_t) in, vNx2int_t offsets, int num) {
    return vgather_lo(in, offsets, (vNx4char_t)0, mli_prv_pvNx4_init(num));
}


//
// load functions where number of samples is based on amount of vectors, and N depends on the type
//
// This type of load functions is used in type agnostic code with a single datatype where we want to
// do operations on vectors.

static MLI_FORCE_INLINE vNx4char_t mli_prv_load_1vec(const MLI_PTR(int8_t) __restrict in) {
    return *(MLI_PTR (vNx4char_t)) in;
}

static MLI_FORCE_INLINE vNx2short_t mli_prv_load_1vec(const MLI_PTR(int16_t) __restrict in) {
    return *(MLI_PTR (vNx2short_t)) in;
}

static MLI_FORCE_INLINE vNint_t mli_prv_load_1vec(const MLI_PTR(int32_t) __restrict in) {
    return *(MLI_PTR (vNint_t)) in;
}

//
// Store functions
//
// The store functions determine the number of samples or vectors to be stored based on the type of the data argument

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

static MLI_FORCE_INLINE void mli_prv_store_n_samples(MLI_OUT_PTR (int8_t) __restrict out,
        vNx4char_t data, int predicate_limit) {
    pvNx4 predicate = mli_prv_pvNx4_init(predicate_limit);
    vvst(data, predicate, (int8_t __vccm *)(out));
}

static MLI_FORCE_INLINE void mli_prv_store_n_samples(MLI_OUT_PTR (int16_t) __restrict out,
        vNx2short_t data, int predicate_limit) {
    pvNx2 predicate = mli_prv_pvNx2_init(predicate_limit);
    vvst(data, predicate, (int16_t __vccm *)(out));
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

//-------------------------------------
// loads combined with mac operation
// _v_s means vector x scalar
// _v_v means vector x vector
//-------------------------------------

template <typename acc_T, typename l_T, typename r_T>
MLI_FORCE_INLINE acc_T mli_prv_mac_load_v_s(
        acc_T accu,
        const MLI_PTR(l_T) __restrict in1,
        const MLI_PTR(r_T) __restrict in2) {
    return mli_math_mac_fx(accu, *in1, *in2);
}

// vector * scalar
template <>
MLI_FORCE_INLINE vNx4accshort_t mli_prv_mac_load_v_s(
        vNx4accshort_t accu,
        const MLI_PTR(int8_t) __restrict in1,
        const MLI_PTR(int8_t) __restrict in2) {
    return mli_math_mac_fx(accu, mli_prv_load_nx4_samples(in1), *in2);
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_prv_mac_load_v_s(
        vNx2accint_t accu,
        const MLI_PTR(int16_t) __restrict in1,
        const MLI_PTR(int16_t) __restrict in2) {
    return mli_math_mac_fx(accu, mli_prv_load_nx2_samples(in1), *in2);
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_prv_mac_load_v_s(
        vNx4accint_t accu,
        const MLI_PTR(int8_t) __restrict in1,
        const MLI_PTR(int16_t) __restrict in2) {
    return mli_math_mac_fx(accu, mli_prv_load_nx4_samples(in1), *in2);
}

template <typename acc_T, typename l_T, typename r_T>
MLI_FORCE_INLINE acc_T mli_prv_mac_load_v_s(
        acc_T accu,
        const MLI_PTR(l_T) __restrict in1,
        const r_T  in2) {
    return mli_math_mac_fx(accu, *in1, in2);
}

// vector * scalar
template <>
MLI_FORCE_INLINE vNx4accshort_t mli_prv_mac_load_v_s(
        vNx4accshort_t accu,
        const MLI_PTR(int8_t) __restrict in1,
        const int8_t in2) {
    return mli_math_mac_fx(accu, mli_prv_load_nx4_samples(in1), in2);
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_prv_mac_load_v_s(
        vNx2accint_t accu,
        const MLI_PTR(int16_t) __restrict in1,
        const int16_t in2) {
    return mli_math_mac_fx(accu, mli_prv_load_nx2_samples(in1), in2);
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_prv_mac_load_v_s(
        vNx4accint_t accu,
        const MLI_PTR(int8_t) __restrict in1,
        const int16_t in2) {
    return mli_math_mac_fx(accu, mli_prv_load_nx4_samples(in1), in2);
}

// _v_v versions

// for scalar datatypes fall back to scalar * scalar
template <typename acc_T, typename l_T, typename r_T>
MLI_FORCE_INLINE acc_T mli_prv_mac_load_v_v(
        acc_T accu,
        const MLI_PTR(l_T) __restrict in1,
        const MLI_PTR(r_T) __restrict in2) {
    return mli_math_mac_fx(accu, *in1, *in2);
}

// vector * vector
template <>
MLI_FORCE_INLINE vNx4accshort_t mli_prv_mac_load_v_v(
        vNx4accshort_t accu,
        const MLI_PTR(int8_t) __restrict in1,
        const MLI_PTR(int8_t) __restrict in2) {
    return mli_math_mac_fx(accu, mli_prv_load_nx4_samples(in1), mli_prv_load_nx4_samples(in2));
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_prv_mac_load_v_v(
        vNx2accint_t accu,
        const MLI_PTR(int16_t) __restrict in1,
        const MLI_PTR(int16_t) __restrict in2) {
    return mli_math_mac_fx(accu, mli_prv_load_nx2_samples(in1), mli_prv_load_nx2_samples(in2));
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_prv_mac_load_v_v(
        vNx4accint_t accu,
        const MLI_PTR(int8_t) __restrict in1,
        const MLI_PTR(int16_t) __restrict in2) {
    return mli_math_mac_fx(accu, mli_prv_load_nx4_samples(in1), mli_prv_load_nx4_samples(in2));
}

//-------------------------------------
// loads combined with msub operation
// _v_s means vector x scalar
// _v_v means vector x vector
//-------------------------------------

template <typename acc_T, typename l_T, typename r_T>
MLI_FORCE_INLINE acc_T mli_prv_msub_load_v_s(
        acc_T accu,
        const MLI_PTR(l_T) __restrict in1,
        const MLI_PTR(r_T) __restrict in2) {
    return mli_math_msub_fx(accu, *in1, *in2);
}

// vector * scalar
template <>
MLI_FORCE_INLINE vNx4accshort_t mli_prv_msub_load_v_s(
        vNx4accshort_t accu,
        const MLI_PTR(int8_t) __restrict in1,
        const MLI_PTR(int8_t) __restrict in2) {
    return mli_math_msub_fx(accu, mli_prv_load_nx4_samples(in1), *in2);
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_prv_msub_load_v_s(
        vNx2accint_t accu,
        const MLI_PTR(int16_t) __restrict in1,
        const MLI_PTR(int16_t) __restrict in2) {
    return mli_math_msub_fx(accu, mli_prv_load_nx2_samples(in1), *in2);
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_prv_msub_load_v_s(
        vNx4accint_t accu,
        const MLI_PTR(int8_t) __restrict in1,
        const MLI_PTR(int16_t) __restrict in2) {
    return mli_math_msub_fx(accu, mli_prv_load_nx4_samples(in1), *in2);
}

template <typename acc_T, typename l_T, typename r_T>
MLI_FORCE_INLINE acc_T mli_prv_msub_load_v_s(
        acc_T accu,
        const MLI_PTR(l_T) __restrict in1,
        const r_T  in2) {
    return mli_math_msub_fx(accu, *in1, in2);
}

// vector * scalar
template <>
MLI_FORCE_INLINE vNx4accshort_t mli_prv_msub_load_v_s(
        vNx4accshort_t accu,
        const MLI_PTR(int8_t) __restrict in1,
        const int8_t in2) {
    return mli_math_msub_fx(accu, mli_prv_load_nx4_samples(in1), in2);
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_prv_msub_load_v_s(
        vNx2accint_t accu,
        const MLI_PTR(int16_t) __restrict in1,
        const int16_t in2) {
    return mli_math_msub_fx(accu, mli_prv_load_nx2_samples(in1), in2);
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_prv_msub_load_v_s(
        vNx4accint_t accu,
        const MLI_PTR(int8_t) __restrict in1,
        const int16_t in2) {
    return mli_math_msub_fx(accu, mli_prv_load_nx4_samples(in1), in2);
}

// _v_v versions

// for scalar datatypes fall back to scalar * scalar
template <typename acc_T, typename l_T, typename r_T>
MLI_FORCE_INLINE acc_T mli_prv_msub_load_v_v(
        acc_T accu,
        const MLI_PTR(l_T) __restrict in1,
        const MLI_PTR(r_T) __restrict in2) {
    return mli_math_msub_fx(accu, *in1, *in2);
}

// vector * vector
template <>
MLI_FORCE_INLINE vNx4accshort_t mli_prv_msub_load_v_v(
        vNx4accshort_t accu,
        const MLI_PTR(int8_t) __restrict in1,
        const MLI_PTR(int8_t) __restrict in2) {
    return mli_math_msub_fx(accu, mli_prv_load_nx4_samples(in1), mli_prv_load_nx4_samples(in2));
}

template <>
MLI_FORCE_INLINE vNx2accint_t mli_prv_msub_load_v_v(
        vNx2accint_t accu,
        const MLI_PTR(int16_t) __restrict in1,
        const MLI_PTR(int16_t) __restrict in2) {
    return mli_math_msub_fx(accu, mli_prv_load_nx2_samples(in1), mli_prv_load_nx2_samples(in2));
}

template <>
MLI_FORCE_INLINE vNx4accint_t mli_prv_msub_load_v_v(
        vNx4accint_t accu,
        const MLI_PTR(int8_t) __restrict in1,
        const MLI_PTR(int16_t) __restrict in2) {
    return mli_math_msub_fx(accu, mli_prv_load_nx4_samples(in1), mli_prv_load_nx4_samples(in2));
}


#pragma clang diagnostic pop

#endif // _VDSP_MLI_PRV_LOAD_STORE_H_
