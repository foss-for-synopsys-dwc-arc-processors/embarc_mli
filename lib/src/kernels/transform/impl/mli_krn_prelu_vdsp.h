/*
* Copyright 2020-2021, Synopsys, Inc.
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
        const vNx4char_t input,
        const vNx4char_t scale,
        const int shift ) {
    pvNx4 sel = init_predicate(input > 0);
    vNx4accshort_t acc = mli_math_mul_fx<vNx4char_t, vNx4accshort_t>(input, scale);
    vNx4char_t neg = mli_math_acc_cast_fx<vNx4char_t, vNx4accshort_t>(acc, shift);

    return mli_math_select_fx(sel, input, neg);
}

static MLI_FORCE_INLINE vNx2short_t calc_prelu(
        const vNx2short_t input,
        const vNx2short_t scale,
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

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        const vNx4char_t scale,
        MLI_OUT_PTR(int8_t) vec_out,
        const int shift) {

    vNx4char_t input = mli_prv_load_1vec(vec_in);
    mli_prv_store_n_samples(vec_out, calc_prelu(input, scale, shift));
}

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int16_t) vec_in,
        const vNx2short_t scale,
        MLI_OUT_PTR(int16_t) vec_out,
        const int shift) {

    vNx2short_t input = mli_prv_load_1vec(vec_in);
    mli_prv_store_n_samples(vec_out, calc_prelu(input, scale, shift));
}

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        const vNx4char_t scale,
        MLI_OUT_PTR(int8_t) vec_out,
        const int shift,
        const int remaining_part) {

    vNx4char_t input = mli_prv_load_1vec(vec_in);
    mli_prv_store_n_samples(vec_out, calc_prelu(input, scale, shift), remaining_part);
}

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int16_t) vec_in,
        const vNx2short_t scale,
        MLI_OUT_PTR(int16_t) vec_out,
        const int shift,
        const int remaining_part) {

    vNx2short_t input = mli_prv_load_1vec(vec_in);
    mli_prv_store_n_samples(vec_out, calc_prelu(input, scale, shift), remaining_part);
}

template<typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu_fx_inner_loop(
        const MLI_PTR(io_T) __restrict vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) __restrict vec_out,
        const int shift,
        const int count,
        const int remaining_part) {
    /* Dummy Load to get num_lanes */
    auto input = mli_prv_load_1vec(vec_in);
    int num_lanes = get_number_lanes(input);

    if (remaining_part) {
        compute_prelu<io_T, scale_T>(vec_in, scale, vec_out, shift, remaining_part);
        vec_in  += remaining_part;
        vec_out += remaining_part;
    }

#pragma clang loop unroll_count(4)
    for (int pos3 = remaining_part; pos3 < count; pos3 += num_lanes) {
        compute_prelu<io_T, scale_T>(vec_in, scale, vec_out, shift);
        vec_in  += num_lanes;
        vec_out += num_lanes;
    }
}

static MLI_FORCE_INLINE s8asym_quant_params_v prelu_define_requant_params(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        mli_tensor *out,
        const vNx4char_t alpha_sa8,
        const s8asym_quant_params *identity_params) {

    vNx4int_t alpha_val = mli_prv_convert_sa8_fx32(alpha_sa8, 
                            slope_coeff->el_params.sa.zero_point.mem.i16,
                            slope_coeff->el_params.sa.scale.mem.i16);
    /* Normalize alpha and cast to 16bit */
    vNx4int_t norm_shift;
    vNx4short_t alpha = mli_math_norm_cast_fx(alpha_val, &norm_shift);
    
    vNx4int_t scale_alpha_shift  = identity_params->shift;
              scale_alpha_shift += slope_coeff->el_params.sa.scale_frac_bits.mem.i8;
              scale_alpha_shift -= norm_shift;
    
    vNx4short_t scale_alpha = mli_math_norm_cast_fx(
                          mli_math_mul_fx<vNx4short_t, vNx4int_t>(identity_params->scale, alpha), &norm_shift);
    scale_alpha_shift -= norm_shift;

    int16_t in_zp  = in->el_params.sa.zero_point.mem.i16;
    int16_t out_zp = out->el_params.sa.zero_point.mem.i16;
    
    vNx4int_t shift_left = mli_math_max_fx(-scale_alpha_shift, 0);
    vNx4int_t shift_right = mli_math_max_fx(scale_alpha_shift, 0);

    vNx4int_t scale_zp = mli_math_mul_fx<vNx4short_t, vNx4int_t>(scale_alpha, in_zp);
              scale_zp = mli_math_asl_fx(scale_zp, shift_left);
              scale_zp = mli_math_asr_rnd_fx(scale_zp, shift_right);

    vNx4short_t scale_alpha_offset = mli_math_sub_fx<vNx4short_t>(out_zp,
                                 mli_math_cast_fx<vNx4int_t, vNx4short_t>(scale_zp, 0));
    
    /* Define Quantization params for (In * alpha / out) ratio */
    s8asym_quant_params_v alpha_params;
    alpha_params.scale  = scale_alpha;
    alpha_params.shift  = mli_math_cast_fx<vNx4int_t, vNx4short_t>(scale_alpha_shift);
    alpha_params.offset = scale_alpha_offset;
    return alpha_params;
}

static MLI_FORCE_INLINE vNx4char_t calc_prelu(
        const MLI_PTR(int8_t) vec_in,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params) {
    /* Load Input */
    vNx4char_t input = mli_prv_load_1vec(vec_in);
    vNx4short_t input_cast = mli_math_cast_fx<vNx4char_t, vNx4short_t>(input);
    grp_pvNx2_t select = init_predicate_grp(input_cast >= in_zp);

    int scale_shift = identity_params->shift;
    int scale_shift_left = mli_math_max_fx(-scale_shift, 0);
    int scale_shift_right = mli_math_max_fx(scale_shift, 0);
    vNx4int_t input_identity_scale = mli_math_mul_fx<vNx4short_t, vNx4int_t>(identity_params->scale, input_cast);
              input_identity_scale = mli_math_asl_fx(input_identity_scale, scale_shift_left);
              input_identity_scale = mli_math_asr_rnd_fx(input_identity_scale, scale_shift_right);

    vNx4short_t output_identity = mli_math_cast_fx<vNx4int_t, vNx4short_t>(input_identity_scale);
                output_identity = mli_math_add_fx(output_identity, (vNx4short_t)identity_params->offset);

    vNx4int_t shift = mli_math_cast_fx<vNx4short_t, vNx4int_t>(alpha_params->shift);
    vNx4int_t shift_left = mli_math_max_fx(-shift, 0);
    vNx4int_t shift_right = mli_math_max_fx(shift, 0);
    vNx4int_t input_alpha_scale = mli_math_mul_fx<vNx4short_t, vNx4int_t>(alpha_params->scale, input_cast);
              input_alpha_scale = mli_math_asl_fx(input_alpha_scale, shift_left);
              input_alpha_scale = mli_math_asr_rnd_fx(input_alpha_scale, shift_right);

    vNx4short_t output_alpha = mli_math_cast_fx<vNx4int_t, vNx4short_t>(input_alpha_scale);
                output_alpha = mli_math_add_fx(output_alpha, alpha_params->offset);
    
    vNx4short_t output = mli_math_select_fx(select, output_identity, output_alpha);
    return mli_math_cast_fx<vNx4short_t, vNx4char_t>(output);
}

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params) {

    vNx4char_t output = calc_prelu(vec_in, in_zp, identity_params, alpha_params);
    mli_prv_store_n_samples(vec_out, output);
}

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params,
        const int remaining_part) {

    vNx4char_t output = calc_prelu(vec_in, in_zp, identity_params, alpha_params);
    mli_prv_store_n_samples(vec_out, output, remaining_part);
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_PRELU_VDSP_H_
