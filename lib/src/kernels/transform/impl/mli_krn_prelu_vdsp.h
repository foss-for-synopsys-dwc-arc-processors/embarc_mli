/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_PRELU_VDSP_H_
#define _MLI_KRN_PRELU_VDSP_H_

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace vdsp {

template <typename io_T, prelu_elem_func_type func_type>
static MLI_FORCE_INLINE void mli_krn_scale_elem_v(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int8_t scale,
        const int shift) {

    vNx4char_t input = mli_prv_load_1vec(vec_in);
    vNx4accshort_t acc_res = mli_math_mul_fx<vNx4char_t, vNx4accshort_t>(input, scale);
    vNx4short_t res_short = mli_math_acc_cast_fx<vNx4short_t, vNx4accshort_t>(acc_res);
    vNx4char_t res = mli_math_cast_fx<vNx4short_t, vNx4char_t>(res_short, shift);

    if (func_type == PRELU_ELEM_FUNC_MAX) {
        mli_prv_store_n_samples(vec_out, mli_math_max_fx(input, res));
    } else {
        mli_prv_store_n_samples(vec_out, mli_math_min_fx(input, res));
    }
}

template <typename io_T, prelu_elem_func_type func_type>
static MLI_FORCE_INLINE void mli_krn_scale_elem_v(
        const MLI_PTR(int16_t) vec_in,
        MLI_OUT_PTR(int16_t) vec_out,
        const int16_t scale,
        const int shift) {

    vNx2short_t input = mli_prv_load_1vec(vec_in);
    vNx2accint_t acc = mli_math_mul_fx<vNx2short_t, vNx2accint_t>(input, scale);
    vNx2short_t res = mli_math_acc_cast_fx<vNx2short_t, vNx2accint_t>(acc, shift);

    if (func_type == PRELU_ELEM_FUNC_MAX) {
        mli_prv_store_n_samples(vec_out, mli_math_max_fx(input, res));
    } else {
        mli_prv_store_n_samples(vec_out, mli_math_min_fx(input, res));
    }
}

template <typename io_T, prelu_elem_func_type func_type>
static MLI_FORCE_INLINE void mli_krn_scale_elem_v(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int8_t scale,
        const int shift,
        const int remaining_part) {

    vNx4char_t input = mli_prv_load_1vec(vec_in);
    vNx4accshort_t acc_res = mli_math_mul_fx<vNx4char_t, vNx4accshort_t>(input, scale);
    vNx4short_t res_short = mli_math_acc_cast_fx<vNx4short_t, vNx4accshort_t>(acc_res);
    vNx4char_t res = mli_math_cast_fx<vNx4short_t, vNx4char_t>(res_short, shift);

    if (func_type == PRELU_ELEM_FUNC_MAX) {
        mli_prv_store_n_samples(vec_out, mli_math_max_fx(input, res), remaining_part);
    } else {
        mli_prv_store_n_samples(vec_out, mli_math_min_fx(input, res), remaining_part);
    }
}

template <typename io_T, prelu_elem_func_type func_type>
static MLI_FORCE_INLINE void mli_krn_scale_elem_v(
        const MLI_PTR(int16_t) vec_in,
        MLI_OUT_PTR(int16_t) vec_out,
        const int16_t scale,
        const int shift,
        const int remaining_part) {

    vNx2short_t input = mli_prv_load_1vec(vec_in);
    vNx2accint_t acc = mli_math_mul_fx<vNx2short_t, vNx2accint_t>(input, scale);
    vNx2short_t res = mli_math_acc_cast_fx<vNx2short_t, vNx2accint_t>(acc, shift);

    if (func_type == PRELU_ELEM_FUNC_MAX) {
        mli_prv_store_n_samples(vec_out, mli_math_max_fx(input, res), remaining_part);
    } else {
        mli_prv_store_n_samples(vec_out, mli_math_min_fx(input, res), remaining_part);
    }
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_PRELU_VDSP_H_