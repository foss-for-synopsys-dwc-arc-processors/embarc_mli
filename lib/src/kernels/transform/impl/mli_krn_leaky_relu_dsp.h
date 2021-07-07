/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_LEAKY_RELU_DSP_H_
#define _MLI_KRN_LEAKY_RELU_DSP_H_

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
namespace dsp {

static MLI_FORCE_INLINE v2q15_t calc_leaky_relu(
        const v2q15_t input,
        const v2q15_t scale_v,
        const int shift) {

    /* out  = max(0, in) + alpha * min(0, in) */
    v2q15_t zero = mli_prv_init_v<int16_t, v2q15_t>(0);
    v2q15_t pos = mli_math_max_fx(zero, input);
    v2q15_t neg = mli_math_acc_cast_fx<v2q15_t, v2accum40_t>(
                  mli_math_mul_fx<v2q15_t, v2accum40_t>(scale_v, mli_math_min_fx(zero, input)), shift);
    return mli_math_add_fx(pos, neg);
}

template <typename io_T>
static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift) {

    v2q15_t input = mli_prv_load_1vec(vec_in);
    v2q15_t scale_v = mli_prv_init_v<io_T, v2q15_t>(scale);
    mli_prv_sat_and_store_2_samples(vec_out, calc_leaky_relu(input, scale_v, shift));
}

template <typename io_T>
static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift,
        const int remaining_part) {

    MLI_ASSERT(remaining_part == 1);
    v2q15_t input = mli_prv_load_1vec(vec_in);
    v2q15_t scale_v = mli_prv_init_v<io_T, v2q15_t>(scale);
    mli_prv_sat_and_store_1_sample(vec_out, calc_leaky_relu(input, scale_v, shift));
}

static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params) {

    mli::krn::ref::compute_leaky_relu(vec_in, vec_out, in_zp, identity_params, alpha_params);
    mli::krn::ref::compute_leaky_relu(vec_in + 1, vec_out + 1, in_zp, identity_params, alpha_params);
}

static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params,
        const int remaining_part) {

    MLI_ASSERT(remaining_part == 1);
    mli::krn::ref::compute_leaky_relu(vec_in, vec_out, in_zp, identity_params, alpha_params);
}

} // namespace dsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_LEAKY_RELU_DSP_H_
