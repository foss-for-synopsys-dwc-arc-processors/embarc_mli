/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_LEAKY_RELU_VDSP_H_
#define _MLI_KRN_LEAKY_RELU_VDSP_H_

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

static MLI_FORCE_INLINE vNx4char_t calc_leaky_relu(
        const vNx4char_t input,
        const int8_t scale,
        const int shift ) {
    pvNx4 sel = init_predicate(input > 0);
    vNx4accshort_t acc = mli_math_mul_fx<vNx4char_t, vNx4accshort_t>(input, scale);
    vNx4char_t neg = mli_math_acc_cast_fx<vNx4char_t, vNx4accshort_t>(acc, shift);

    return mli_math_select_fx(sel, input, neg);
}

static MLI_FORCE_INLINE vNx2short_t calc_leaky_relu(
        const vNx2short_t input,
        const int16_t scale,
        const int shift ) {

    constexpr int mul_hi_shift = 16;
    pvNx2 sel = init_predicate(input > 0);
    vNx2short_t neg;
    if ( shift > mul_hi_shift) {
        neg = mli_math_mul_fx_high(input, scale);
        neg = mli_math_asr_rnd_fx(neg, shift - mul_hi_shift);
    } else {
        vNx2accint_t acc = mli_math_mul_fx<vNx2short_t, vNx2accint_t>(input, scale);
        neg = mli_math_acc_cast_fx<vNx2short_t, vNx2accint_t>(acc, shift);
    }

    return mli_math_select_fx(sel, input, neg);
}

template<typename io_T>
MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift) {

    auto input = mli_prv_load_1vec(vec_in);
    mli_prv_store_n_samples(vec_out, calc_leaky_relu(input, scale, shift));
}

template<typename io_T>
MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift,
        const int remaining_part) {

    auto input = mli_prv_load_1vec(vec_in);
    mli_prv_store_n_samples(vec_out, calc_leaky_relu(input, scale, shift), remaining_part);
}

template<typename io_T>
static MLI_FORCE_INLINE void compute_leaky_relu_fx_inner_loop(
        const MLI_PTR(io_T) __restrict vec_in,
        MLI_OUT_PTR(io_T) __restrict vec_out,
        const io_T scale,
        const int shift,
        const int count,
        const int remaining_part) {
    /* Dummy Load to get num_lanes */
    auto input = mli_prv_load_1vec(vec_in);
    int num_lanes = get_number_lanes(input);

    if (remaining_part) {
        compute_leaky_relu<io_T>(vec_in, vec_out, scale, shift, remaining_part);
        vec_in  += remaining_part;
        vec_out += remaining_part;
    }

#pragma clang loop unroll_count(4)
    for (int pos3 = remaining_part; pos3 < count; pos3 += num_lanes) {
        compute_leaky_relu<io_T>(vec_in, vec_out, scale, shift);
        vec_in  += num_lanes;
        vec_out += num_lanes;
    }
}

static MLI_FORCE_INLINE vNx4char_t calc_leaky_relu(
        const MLI_PTR(int8_t) vec_in,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params) {
    /* Load Input */
    vNx4char_t input = mli_prv_load_1vec(vec_in);
    vNx4short_t input_cast = mli_math_cast_fx<vNx4char_t, vNx4short_t>(input);
    vNx4short_t cond;
    cond.lo = input_cast.lo >= in_zp;
    cond.hi = input_cast.hi >= in_zp;
    grp_pvNx2_t select = init_predicate_grp(cond);

    int identity_shift = identity_params->shift;
    vNx4int_t input_identity_scale = mli_math_mul_fx<vNx4short_t, vNx4int_t>(input_cast, identity_params->scale);
              input_identity_scale = mli_math_asr_rnd_fx(input_identity_scale, identity_shift);

    vNx4short_t output_identity = mli_math_cast_fx<vNx4int_t, vNx4short_t>(input_identity_scale);
                output_identity = mli_math_add_fx(output_identity, (vNx4short_t)identity_params->offset);

    int alpha_shift = alpha_params->shift;
    vNx4int_t input_alpha_scale = mli_math_mul_fx<vNx4short_t, vNx4int_t>(input_cast, alpha_params->scale);
              input_alpha_scale = mli_math_asr_rnd_fx(input_alpha_scale, alpha_shift);

    vNx4short_t output_alpha = mli_math_cast_fx<vNx4int_t, vNx4short_t>(input_alpha_scale);
                output_alpha = mli_math_add_fx(output_alpha, (vNx4short_t)alpha_params->offset);
    
    vNx4short_t output = mli_math_select_fx(select, output_identity, output_alpha);
    return mli_math_cast_fx<vNx4short_t, vNx4char_t>(output);
}

static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params) {

    vNx4char_t output = calc_leaky_relu(vec_in, in_zp, identity_params, alpha_params);
    mli_prv_store_n_samples(vec_out, output);
}

static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params,
        const int remaining_part) {

    vNx4char_t output = calc_leaky_relu(vec_in, in_zp, identity_params, alpha_params);
    mli_prv_store_n_samples(vec_out, output, remaining_part);
}

static MLI_FORCE_INLINE void compute_leaky_relu_sa8_inner_loop(
        const MLI_PTR(int8_t) __restrict vec_in,
        MLI_OUT_PTR(int8_t) __restrict vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params,
        const int count,
        const int remaining_part) {

    /* Dummy Load to get num_lanes */
    auto input = mli_prv_load_1vec(vec_in);
    int num_lanes = get_number_lanes(input);

    if (remaining_part) {
        compute_leaky_relu(vec_in, vec_out, in_zp, identity_params, alpha_params, remaining_part);
        vec_in  += remaining_part;
        vec_out += remaining_part;
    }

#pragma clang loop unroll_count(2)
    for (int pos3 = remaining_part; pos3 < count; pos3 += num_lanes) {
        compute_leaky_relu(vec_in, vec_out, in_zp, identity_params, alpha_params);
        vec_in  += num_lanes;
        vec_out += num_lanes;
    }
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_LEAKY_RELU_VDSP_H_
