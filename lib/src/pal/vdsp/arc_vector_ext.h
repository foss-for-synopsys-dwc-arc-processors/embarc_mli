/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef ARC_VECTOR_EXT_H_
#define ARC_VECTOR_EXT_H_

#include <arc_vector.h>
#include "mli_config.h"
//////////////////////////////////////////////////
// Defines
//////////////////////////////////////////////////
// helper defines for vvconvert
#define SIGNED (1)
#define UNSIGNED (0)
#define TARGET_SZ_8 (0 << 8)
#define TARGET_SZ_16 (3 << 8)
#define TARGET_SZ_24 (5 << 8)
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
// vvmpy_hi
//////////////////////////////////////////////////
template <typename T1, typename T2>
T1 vvmpy_hi_(T1 a, T2 b);

static MLI_FORCE_INLINE vNint_t
vvmpy_hi_(vNint_t a, int32_t b) {
    vNint_t r;
    r = vvmpy_hi(a, b);
    return r;
}

static MLI_FORCE_INLINE vNx2int_t
vvmpy_hi_(vNx2int_t a, int32_t b) {
    vNx2int_t r;
    r.lo = vvmpy_hi_(a.lo, b);
    r.hi = vvmpy_hi_(a.hi, b);
    return r;
}

static MLI_FORCE_INLINE vNx4int_t
vvmpy_hi_(vNx4int_t a, int32_t b) {
    vNx4int_t r;
    r.lo = vvmpy_hi_(a.lo, b);
    r.hi = vvmpy_hi_(a.hi, b);
    return r;
}

//////////////////////////////////////////////////
// relu
//////////////////////////////////////////////////
template <typename T>
T vvrelu(T a, int16_t min, int16_t max);

static MLI_FORCE_INLINE vNint_t vvrelu(vNint_t a, int16_t min, int16_t max) {
    vNint_t r;
    r = min > a ? min : a;
    r = max < r ? max : r;
    return r;
}

static MLI_FORCE_INLINE vNx2int_t vvrelu(vNx2int_t a, int16_t min, int16_t max) {
    vNx2int_t r;
    r.lo = vvrelu(a.lo, min, max);
    r.hi = vvrelu(a.hi, min, max);
    return r;
}

static MLI_FORCE_INLINE vNx4int_t vvrelu(vNx4int_t a, int16_t min, int16_t max) {
    vNx4int_t r;
    r.lo = vvrelu(a.lo, min, max);
    r.hi = vvrelu(a.hi, min, max);
    return r;
}

#endif /* ARC_VECTOR_EXT_H_ */