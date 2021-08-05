/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef ARC_VECTOR_EXT_H_
#define ARC_VECTOR_EXT_H_

#include <arc_vector.h>

//////////////////////////////////////////////////
// Defines
//////////////////////////////////////////////////
// helper defines for vvconvert
#define SIGNED (1)
#define UNSIGNED (0)
#define TARGET_SZ_8 (0 << 8)
#define TARGET_SZ_10 (1 << 8)
#define TARGET_SZ_12 (2 << 8)
#define TARGET_SZ_16 (3 << 8)
#define TARGET_SZ_20 (4 << 8)
#define TARGET_SZ_24 (5 << 8)
#define TARGET_SZ_32 (6 << 8)
#define SAT (1<<7)
#define SHIFT(a) ((a)<< 1)

//////////////////////////////////////////////////
// Types
//////////////////////////////////////////////////
typedef struct {
    vNx2accint_t lo;
    vNx2accint_t hi;
} vNx4accint_t;

//////////////////////////////////////////////////
// double vector accumulator types narrowing
//
// For below functions further downcasts are
// possible with the functions in arc_vector.h
//////////////////////////////////////////////////
static MLI_FORCE_INLINE vNx2int_t
to_vNx2int_t(vNx2accint_t acc) {
    vNx2int_t v;
    v.lo = to_vNint_t(__vacc_lo(acc));
    v.hi = to_vNint_t(__vacc_hi(acc));
    return v;
}

static MLI_FORCE_INLINE vNx4short_t
to_vNx4short_t(vNx4accshort_t acc) {
    vNx4short_t v;
    v.lo = to_vNx2short_t(__vacc_lo(acc));
    v.hi = to_vNx2short_t(__vacc_hi(acc));
    return v;
}

static MLI_FORCE_INLINE vNx4int_t
to_vNx4int_t(vNx4accint_t acc) {
    vNx4int_t v;
    v.lo.lo = to_vNint_t(__vacc_lo(acc.lo));
    v.lo.hi = to_vNint_t(__vacc_hi(acc.lo));
    v.hi.lo = to_vNint_t(__vacc_lo(acc.hi));
    v.hi.hi = to_vNint_t(__vacc_hi(acc.hi));
    return v;
}

static MLI_FORCE_INLINE vNx4int_t
to_vNx4int_t(vNx4accshort_t acc) {
    vNx4int_t v;
    unsigned ctrlword_lo = UNSIGNED | TARGET_SZ_8;
    unsigned ctrlword_hi = SIGNED|TARGET_SZ_24|SHIFT(8);

    v.lo = (to_vNx2int_t(to_vNx2short_t(vvconvert(__vacc_lo(acc), ctrlword_hi))) << 8) |
            to_vNx2int_t(to_vNx2short_t(vvconvert(__vacc_lo(acc), ctrlword_lo)));

    v.hi = (to_vNx2int_t(to_vNx2short_t(vvconvert(__vacc_hi(acc), ctrlword_hi))) << 8) |
            to_vNx2int_t(to_vNx2short_t(vvconvert(__vacc_hi(acc), ctrlword_lo)));
    return v;
}

//////////////////////////////////////////////////
// MAC
//////////////////////////////////////////////////
template <typename acc_T>
static acc_T vvcmpy(acc_T acc, vNx4char_t a, int8_t b);

static MLI_FORCE_INLINE vNx4accint_t
vvcmac(vNx4accint_t acc, vNx4char_t a, int8_t b) {
    vNx4accint_t r;
    vNx4short_t a16 = to_vNx4short_t(a);
    r.lo = vvcmac(acc.lo, a16.lo, (int16_t)b);
    r.hi = vvcmac(acc.hi, a16.hi, (int16_t)b);
    return r;
}

static MLI_FORCE_INLINE vNx4accint_t
vvcmac(vNx4accint_t acc, vNx4char_t a, vNx4char_t b) {
    vNx4accint_t r;
    vNx4short_t a16 = to_vNx4short_t(a);
    vNx4short_t b16 = to_vNx4short_t(b);
    r.lo = vvcmac(acc.lo, a16.lo, b16.lo);
    r.hi = vvcmac(acc.hi, a16.hi, b16.hi);
    return r;
}

static MLI_FORCE_INLINE vNx4accint_t
vvcmpy(vNx4short_t a, vNx4short_t b) {
    vNx4accint_t acc;
    acc.lo = vvcmpy(a.lo, b.lo);
    acc.hi = vvcmpy(a.hi, b.hi);
    return acc;
}

//////////////////////////////////////////////////
// vvadd_sat
//////////////////////////////////////////////////
template <typename T>
T vvadd_sat(T L, T R);

static MLI_FORCE_INLINE vNx4short_t vvadd_sat(vNx4short_t L, vNx4short_t R) {
    vNx4short_t out;
    out.lo = vvadd_sat(L.lo, R.lo);
    out.hi = vvadd_sat(L.hi, R.hi);
    return out;
}

static MLI_FORCE_INLINE vNx2int_t vvadd_sat(vNx2int_t L, vNx2int_t R) {
    vNx2int_t out;
    out.lo = vvadd_sat(L.lo, R.lo);
    out.hi = vvadd_sat(L.hi, R.hi);
    return out;
}

static MLI_FORCE_INLINE vNx4int_t vvadd_sat(vNx4int_t L, vNx4int_t R) {
    vNx4int_t out;
    out.lo = vvadd_sat(L.lo, R.lo);
    out.hi = vvadd_sat(L.hi, R.hi);
    return out;
}

//////////////////////////////////////////////////
// vvsub_sat
//////////////////////////////////////////////////
template <typename T>
T vvsub_sat(T L, T R);

static MLI_FORCE_INLINE vNx4short_t vvsub_sat(vNx4short_t L, vNx4short_t R) {
    vNx4short_t out;
    out.lo = vvsub_sat(L.lo, R.lo);
    out.hi = vvsub_sat(L.hi, R.hi);
    return out;
}

//////////////////////////////////////////////////
// vvslm_sat
//////////////////////////////////////////////////
template <typename T>
T vvslm_sat(T L, int nbits);

static MLI_FORCE_INLINE vNx4short_t vvslm_sat(vNx4short_t L, int nbits) {
    vNx4short_t out;
    out.lo = vvslm_sat(L.lo, (vNx2short_t)nbits);
    out.hi = vvslm_sat(L.hi, (vNx2short_t)nbits);
    return out;
}

static MLI_FORCE_INLINE vNx2int_t vvslm_sat(vNx2int_t L, int nbits) {
    vNx2int_t out;
    out.lo = vvslm_sat(L.lo, (vNint_t)nbits);
    out.hi = vvslm_sat(L.hi, (vNint_t)nbits);
    return out;
}

static MLI_FORCE_INLINE vNx4int_t vvslm_sat(vNx4int_t L, int nbits) {
    vNx4int_t out;
    out.lo = vvslm_sat(L.lo, nbits);
    out.hi = vvslm_sat(L.hi, nbits);
    return out;
}


//////////////////////////////////////////////////
// MAX
//////////////////////////////////////////////////
template <typename acc_T>
static acc_T vvcmax(acc_T acc, vNx4short_t a);

static MLI_FORCE_INLINE vNx4accshort_t
vvcmax(vNx4accshort_t acc, vNx4short_t a) {
    vNx2accshort_t hi = __vacc_hi(acc);
    vNx2accshort_t lo = __vacc_lo(acc);
    hi = vvcmax(hi, a.hi);
    lo = vvcmax(lo, a.lo);
    return __vacc_concat(hi, lo);
}

//////////////////////////////////////////////////
// MIN
//////////////////////////////////////////////////
template <typename acc_T>
static acc_T vvcmin(acc_T acc, vNx4short_t a);

static MLI_FORCE_INLINE vNx4accshort_t
vvcmin(vNx4accshort_t acc, vNx4short_t a) {
    vNx2accshort_t hi = __vacc_hi(acc);
    vNx2accshort_t lo = __vacc_lo(acc);
    hi = vvcmin(hi, a.hi);
    lo = vvcmin(lo, a.lo);
    return __vacc_concat(hi, lo);
}
#endif /* ARC_VECTOR_EXT_H_ */
