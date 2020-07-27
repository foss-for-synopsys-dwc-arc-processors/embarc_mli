/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _VDSP_MLI_MATH_H_
#define _VDSP_MLI_MATH_H_

////////////////////////////////////////////////////////////////////
// TODO: TO BE UPDATED
////////////////////////////////////////////////////////////////////

typedef int32_t   mli_acc32_t;

template < typename in_T > inline void *mli_math_cast_scalar_to_ptr_fx(in_T src) {
    // REMARK: Need to check from C/Cpp standard point of view
    intptr_t out_upcast = src;
    return static_cast < void *>((intptr_t *) out_upcast);
}

template <typename out_T, typename acc_T>
inline out_T mli_math_acc_cast_fx(acc_T acc, int shift_right) {
    return acc >> shift_right;
}

template <typename io_T>
inline io_T mli_math_add_fx(io_T L, io_T R) {
    return L + R;
}

template <typename io_T>
inline io_T mli_math_sub_fx(io_T L, io_T R) {
    return L - R;
}

template <typename io_T>
inline io_T mli_math_max_fx(io_T L, io_T R) {
    return (L > R) ? L : R;
}

template <typename io_T>
inline io_T mli_math_min_fx(io_T L, io_T R) {
     return (L < R) ? L : R;
}

template <typename in_T, typename acc_T>
inline acc_T mli_math_mul_fx(in_T L, in_T R) {
    return L * R;
}

#endif // _VDSP_MLI_MATH_H_
