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

static MLI_FORCE_INLINE vNx4char_t calc_prelu(
        vNx4char_t input,
        vNx4char_t scale,
        const int shift ) {
    /* out  = max(0, in) + alpha * min(0, in) */
    vNx4char_t pos = mli_math_max_fx(input, 0);
    vNx4accshort_t acc = mli_math_mul_fx<vNx4char_t, vNx4accshort_t>(mli_math_min_fx(input, 0), scale);
    vNx4char_t neg = mli_math_acc_cast_fx<vNx4char_t, vNx4accshort_t>(acc, shift);

    return mli_math_add(pos, neg);
}

static MLI_FORCE_INLINE vNx2short_t calc_prelu(
        vNx2short_t input,
        vNx2short_t scale,
        const int shift ) {
    /* out  = max(0, in) + alpha * min(0, in) */
    vNx2short_t pos = mli_math_max_fx(input, 0);
    vNx2accint_t acc = mli_math_mul_fx<vNx2short_t, vNx2accint_t>(mli_math_min_fx(input, 0), scale);
    vNx2short_t neg = mli_math_acc_cast_fx<vNx2short_t, vNx2accint_t>(acc, shift);

    return mli_math_add(pos, neg);
}

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        const int8_t scale,
        MLI_OUT_PTR(int8_t) vec_out,
        const int shift) {

    vNx4char_t input = mli_prv_load_1vec(vec_in);
    mli_prv_store_n_samples(vec_out, calc_prelu(input, scale, shift));
}

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int16_t) vec_in,
        const int16_t scale,
        MLI_OUT_PTR(int16_t) vec_out,
        const int shift) {

    vNx2short_t input = mli_prv_load_1vec(vec_in);
    mli_prv_store_n_samples(vec_out, calc_prelu(input, scale, shift));
}

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        const int8_t scale,
        MLI_OUT_PTR(int8_t) vec_out,
        const int shift,
        const int remaining_part) {

    vNx4char_t input = mli_prv_load_1vec(vec_in);
    mli_prv_store_n_samples(vec_out, calc_prelu(input, scale, shift), remaining_part);
}

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int16_t) vec_in,
        const int16_t scale,
        MLI_OUT_PTR(int16_t) vec_out,
        const int shift,
        const int remaining_part) {

    vNx2short_t input = mli_prv_load_1vec(vec_in);
    mli_prv_store_n_samples(vec_out, calc_prelu(input, scale, shift), remaining_part);
}

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        const MLI_PTR(int8_t) scale_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int shift) {

    vNx4char_t input = mli_prv_load_1vec(vec_in);
    vNx4char_t scale = mli_prv_load_1vec(scale_in);
    mli_prv_store_n_samples(vec_out, calc_prelu(input, scale, shift));
}

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int16_t) vec_in,
        const MLI_PTR(int16_t) scale_in,
        MLI_OUT_PTR(int16_t) vec_out,
        const int shift) {

    vNx2short_t input = mli_prv_load_1vec(vec_in);
    vNx2short_t scale = mli_prv_load_1vec(scale_in);
    mli_prv_store_n_samples(vec_out, calc_prelu(input, scale, shift));
}

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        const MLI_PTR(int8_t) scale_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int shift,
        const int remaining_part) {

    vNx4char_t input = mli_prv_load_1vec(vec_in);
    vNx4char_t scale = mli_prv_load_1vec(scale_in);
    mli_prv_store_n_samples(vec_out, calc_prelu(input, scale, shift), remaining_part);
}

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int16_t) vec_in,
        const MLI_PTR(int16_t) scale_in,
        MLI_OUT_PTR(int16_t) vec_out,
        const int shift,
        const int remaining_part) {

    vNx2short_t input = mli_prv_load_1vec(vec_in);
    vNx2short_t scale = mli_prv_load_1vec(scale_in);
    mli_prv_store_n_samples(vec_out, calc_prelu(input, scale, shift), remaining_part);
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_PRELU_VDSP_H_