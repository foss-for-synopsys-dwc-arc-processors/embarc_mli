/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_QUANT_REF_H_
#define _MLI_PRV_QUANT_REF_H_

#include "mli_prv_quant_decl.h"

#include "mli_config.h"
#include "mli_check.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math_macros.h"
#include "mli_private_types.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"
#include "mli_krn_reduce_sum2d.h"

#include <assert.h>

namespace mli {
namespace krn {
namespace ref {

static const int kPreDivShiftS16 = 14;
static const int kPreDivShiftS32 = 30;

//==========================================================================
// Operating with quantization params set
//==========================================================================

MLI_FORCE_INLINE void define_requant_params(const mli_tensor *in, const mli_tensor *out, 
        s8asym_quant_params *params) {

    /* ****************************************************************************************************************
     *             Mathematical Derivations out_sa8 Requantization Params to use with in_sa8
     * ----------------------------------------------------------------------------------------------------------------
     *      out_sa8 = (in_scale_val/out_scale_val) *(in_sa8 - in_zp) + out_zp
     *              = scale_val * (in_sa8 - in_zp) + out_zp
     *              = scale_val * in_sa8 + out_zp - scale_val * in_zp
     *              = scale_val * in_sa8 + offset;
     *      where:
     * 
     *      scale_val = in_scale_val / out_scale_val;
     *                = in_scale * 2^(-in_scale_frac_bits) / (out_scale * 2^(-out_scale_frac_bits));
     *                = (in_scale_val * 2^kPreDivShift / out_scale_val) 
     *                 * 2^(-(kPreDivShift + in_scale_frac_bits - out_scale_frac_bits));
     *                = (in_scale_val * 2^kPreDivShift / out_scale_val) * 2^(-norm_shift) 
     *                 * 2^(-(kPreDivShift + in_scale_frac_bits - out_scale_frac_bits - norm_shift));
     *                = (in_scale_val * 2^kPreDivShift / out_scale_val) * 2^(-norm_shift) 
     *                 * 2^(-scale_shift)
     *                = scale * 2 ^(-(scale_shift))
     * 
     *      where scale = (in_scale_val * 2^kPreDivShift / out_scale_val) * 2^(-norm_shift)
     *            scale_shift = kPreDivShift + in_scale_frac_bits - out_scale_frac_bits - norm_shift
     *            norm_shift is the shift value due to normalizing the result of 
     *                       (in_scale_val * 2^kPreDivShift / out_scale_val) and casting it from int32_t to int16_t
     *            kPreDivShift is derived from norm_32(in_scale) - norm_16(out_scale)
     * 
     *      offset = out_zp - scale_val * in_zp
     *             = out_zp - (scale * in_zp) * 2^(-(scale_shift));
     * 
     * ***************************************************************************************************************/

    /* kPreDivShift = norm_32(in_scale_val) - norm_16(out_scale_val) */
    int kPreDivShift = mli_math_norm_fx<int32_t,int16_t>(in->el_params.sa.scale.mem.i16) - 
                       mli_math_norm_fx<int16_t,int16_t>(out->el_params.sa.scale.mem.i16);
    /* Normalize In/Out Scale ratio and cast to 16bit */
    int norm_shift;
    params->scale = mli_math_norm_cast_fx<int32_t,int16_t>(
                    ((int32_t)(in->el_params.sa.scale.mem.i16) << kPreDivShift) / 
                               out->el_params.sa.scale.mem.i16, &norm_shift);
    
    params->shift  = kPreDivShift;
    params->shift += in->el_params.sa.scale_frac_bits.mem.i8 - out->el_params.sa.scale_frac_bits.mem.i8; 
    params->shift -= norm_shift;
    
    int16_t in_zp = in->el_params.sa.zero_point.mem.i16;
    int16_t out_zp = out->el_params.sa.zero_point.mem.i16;
    
    params->offset = mli_math_sub_fx<int16_t>(out_zp, 
                     mli_math_cast_fx<int32_t, int16_t>(
                     mli_math_mul_fx<int16_t, int32_t>(params->scale, in_zp), params->shift));
}

template <>
MLI_FORCE_INLINE void define_quant_params(const mli_tensor* in, const mli_tensor* weights, const mli_tensor* bias,
                                const mli_tensor* out, fx_quant_specific_params* params) {
    params->bias_shift = mli_prv_calc_shift(in, weights, bias);
    params->out_shift = mli_prv_calc_shift(in, weights, out);
}

template <>
MLI_FORCE_INLINE void define_quant_params(const mli_tensor *in, const mli_tensor  *weights, const mli_tensor  *bias,
                                const mli_tensor   *out, s8asym_quant_specific_params* params) {
    params->in_offset = in->el_params.sa.zero_point.mem.i16;
    params->out_offset = out->el_params.sa.zero_point.mem.i16;

    params->weight_dim = weights->el_params.sa.dim;
    if (weights->el_params.sa.dim >= 0) {
        // per axis quantization
        params->weights_offset = weights->el_params.sa.zero_point.mem.pi16[0];
        params->weight_scales = weights->el_params.sa.scale.mem.pi16;
        params->weight_shifts = weights->el_params.sa.scale_frac_bits.mem.pi8;
    } else {
        // per tensor quantization
        params->weights_offset = weights->el_params.sa.zero_point.mem.i16;
        params->weight_scales = &weights->el_params.sa.scale.mem.i16;
        params->weight_shifts = &weights->el_params.sa.scale_frac_bits.mem.i8;
    }
    int32_t scale_unfinished = (int32_t)(in->el_params.sa.scale.mem.i16) << kPreDivShiftS16;
    scale_unfinished = scale_unfinished / out->el_params.sa.scale.mem.i16;
    params->in_to_out_scales_ratio = scale_unfinished;

    int in_to_out_norm = mli_math_norm_fx<int32_t, int>(scale_unfinished);
    int int32_to_int16_shift = 16;
    params->in_to_out_scales_ratio = mli_math_cast_fx<int32_t, int16_t>(scale_unfinished, int32_to_int16_shift - in_to_out_norm);
    params->in_to_out_shift = in->el_params.sa.scale_frac_bits.mem.i8;
    params->in_to_out_shift += (kPreDivShiftS16 - out->el_params.sa.scale_frac_bits.mem.i8);
    params->in_to_out_shift -= int32_to_int16_shift - in_to_out_norm;

}

template <>
MLI_FORCE_INLINE void adjust_quant_params(s8asym_quant_specific_params* params, int krn_idx) {
    // out multiplyer can be different across one of axis (per axis quantization for s8asym)
    if (params->weight_dim < 0) {
        krn_idx = 0;
    }
    const int32_t out_mul_scaled = (int32_t)params->in_to_out_scales_ratio * params->weight_scales[krn_idx];
    params->out_mul = out_mul_scaled;

    params->out_shift = params->in_to_out_shift;
    params->out_shift += params->weight_shifts[krn_idx];

#if !defined(FULL_ACCU)
    // When the accumulator is pre-shifted before the output multiplier,
    // we need to normalize mul for max use of the headroom.
    int norm = mli_math_norm_fx<int32_t, int>(params->out_mul);
    params->out_mul = mli_math_asl_fx(params->out_mul, norm);
    params->out_shift += norm;
#endif
    return;
}

template <>
MLI_FORCE_INLINE void adjust_quant_params(fx_quant_specific_params* in, int krn_idx) {
    // No need to adjust something during calculations for MLI_FX specific quantization
    return;
}

MLI_FORCE_INLINE int16_t quant_params_get_weigths_zeropoint(s8asym_quant_specific_params* params) {
    return params->weights_offset;
}

MLI_FORCE_INLINE int16_t quant_params_get_weigths_zeropoint(fx_quant_specific_params* params) {
    return 0;
}

static MLI_FORCE_INLINE int32_t mli_prv_calc_out_mul(
        const mli_tensor *in0,
        const mli_tensor *in1,
        const mli_tensor *out,
        int * shift){
    if ((in0->el_type == MLI_EL_FX_8) || (in0->el_type == MLI_EL_FX_16)) {
        /* mix of FX and asym datatypes is not supported */
        MLI_ASSERT((in1->el_type == MLI_EL_FX_8) || (in1->el_type == MLI_EL_FX_16));
        MLI_ASSERT((out->el_type == MLI_EL_FX_8) || (out->el_type == MLI_EL_FX_16));
        return 1;
    } else if (in0->el_type == MLI_EL_SA_8) {
        const int shiftChangeValue = 32;
        /* mix of FX and asym datatypes is not supported */
        MLI_ASSERT(in1->el_type == MLI_EL_SA_8);
        MLI_ASSERT((out->el_type == MLI_EL_SA_8) || (out->el_type == MLI_EL_SA_32));
        MLI_ASSERT((in0->el_params.sa.dim < 0) && (in1->el_params.sa.dim < 0));

        *shift = in0->el_params.sa.scale_frac_bits.mem.i8;
        *shift += in1->el_params.sa.scale_frac_bits.mem.i8;
        *shift += (kPreDivShiftS16 - out->el_params.sa.scale_frac_bits.mem.i8);
        *shift -= shiftChangeValue;

        int32_t scale_unfinished = (int32_t)(in0->el_params.sa.scale.mem.i16) << kPreDivShiftS16;
        scale_unfinished = scale_unfinished / out->el_params.sa.scale.mem.i16;
        int32_t in_to_out_scales_ratio = scale_unfinished;
        int64_t out_mul_scaled = (int64_t)in_to_out_scales_ratio * in1->el_params.sa.scale.mem.i16;
        int32_t out_mul = mli_math_cast_fx<int64_t, int32_t>(out_mul_scaled, shiftChangeValue);
        return out_mul;
    } else {
        MLI_ASSERT(0);
        return 0;
    }
}

//==========================================================================
// Calculation of weights additive (w_add) in
// dot_prod_asym = dot_prod_gen + w_add + in_add + zp_add + bias_add
//==========================================================================
template <typename w_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T weights_additive(const MLI_PTR(w_T) __restrict, acc_T init_accum,
                              const quant_T*,
                              const int, const int, int, int) {
    // By default and for FX quantization scheme, weights additive isn't required
    return init_accum;
}

template <>
MLI_FORCE_INLINE mli_acc32_t weights_additive(
        const MLI_PTR(int8_t) __restrict weights, mli_acc32_t init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, int col_step, int row_step) {
    // returns -(in_zero_point * cumsum(weights)) For S8ASYM
    if (quant_params->in_offset != 0) {
        return reduce_sum2D(weights, -quant_params->in_offset, init_accum, width, height, /*channels = */0, col_step, row_step, true);
    } else {
        return init_accum;
    }
}

template <typename w_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T weights_additive(const MLI_PTR(w_T) __restrict, acc_T init_accum,
        const quant_T*,
        const int, const int, const int, int, int, int) {
    // By default and for FX quantization scheme, weights additive isn't required
    return init_accum;
}

template <>
MLI_FORCE_INLINE mli_acc32_t weights_additive(
        const MLI_PTR(int8_t) __restrict weights, mli_acc32_t init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, const int ch, int col_step, int row_step, int ch_step) {
    // returns -(in_zero_point * cumsum(weights)) For S8ASYM
    if (quant_params->in_offset != 0) {
        for (int c = 0; c < ch; c++) {
            init_accum = reduce_sum2D(weights, -quant_params->in_offset, init_accum, width, height, /*channels = */0, col_step, row_step, true);
            weights += ch_step;
        }
        return init_accum;
    } else {
        return init_accum;
    }
}

//==========================================================================
// Calculation of input additive (in_add) in
// dot_prod_asym = dot_prod_gen + w_add + in_add + zp_add + bias_add
//==========================================================================
template <typename in_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T in_additive(const MLI_PTR(in_T) __restrict, acc_T init_accum, const quant_T* quant_params,
                              const int, const int, int, int) {
    // By default and for FX quantization scheme, input additive isn't required
    return init_accum;
}

template <>
MLI_FORCE_INLINE mli_acc32_t in_additive(
        const MLI_PTR(int8_t) __restrict in, mli_acc32_t init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width, const int height, int col_step, int row_step) {
    // returns -(wights_zero_point * cumsum(input)) For S8ASYM
    if (quant_params->weights_offset != 0) {
        init_accum = reduce_sum2D(in, -quant_params->weights_offset, init_accum, width, height, /*channels = */0, col_step, row_step, true);
        return init_accum;
    } else {
        return init_accum;
    }
}

template <typename in_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T in_additive(const MLI_PTR(in_T) __restrict, acc_T init_accum, const quant_T* quant_params,
                              const int, const int, const int, int, int, int) {
    // By default and for FX quantization scheme, input additive isn't required
    return init_accum;
}

template <>
MLI_FORCE_INLINE mli_acc32_t in_additive(
        const MLI_PTR(int8_t) __restrict in, mli_acc32_t init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width, const int height, const int ch, int col_step, int row_step, int ch_step) {
    // returns -(wights_zero_point * cumsum(input)) For S8ASYM
    if (quant_params->weights_offset != 0) {
        for (int c = 0; c < ch; c++) {
            init_accum = reduce_sum2D(in, -quant_params->weights_offset, init_accum, width, height, /*channels = */0, col_step, row_step, true);
            in += ch_step;
        }
        return init_accum;
    } else {
        return init_accum;
    }
}

//==========================================================================
// Calculation of zero points additive (zp_add) in
// dot_prod_asym = dot_prod_gen + w_add + in_add + zp_add + bias_add
//==========================================================================
template <typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T zp_additive(const quant_T*, acc_T init_accum,
                        const int) {
    // By default (for FX quantization scheme) weights additive isn't required
    return init_accum;
}


template <>
MLI_FORCE_INLINE mli_acc32_t zp_additive(const s8asym_quant_specific_params* quant_params, mli_acc32_t init_accum,
                               const int mac_serias_len) {
    if (quant_params->weights_offset != 0 || quant_params->in_offset != 0) {
        // Calculating (w_zp * in_zp * mac_serias_len) via reduce sum because of complexity with accum casts.
        // Subject for optimization
        return reduce_sum(&quant_params->weights_offset, quant_params->in_offset, init_accum, mac_serias_len, /*step = */0);
    } else {
        return init_accum;
    }
}

//==========================================================================
// Calculation of bias additive (bias_add) in
// dot_prod_asym= dot_prod_gen + w_add + in_add + zp_add + bias_add
//==========================================================================
template <>
MLI_FORCE_INLINE mli_acc32_t bias_additive(
        const MLI_PTR(int8_t) bias, mli_acc32_t init_accum, const fx_quant_specific_params* quant_params) {
    mli_acc32_t accu = mli_math_mul_fx<int8_t, mli_acc32_t>(*bias, 1);
    accu = mli_math_acc_ashift_fx(accu, -quant_params->bias_shift);
    return mli_math_add_fx(init_accum, accu);
}

template <>
MLI_FORCE_INLINE mli_acc40_t bias_additive(
        const MLI_PTR(int16_t) bias, mli_acc40_t init_accum, const fx_quant_specific_params* quant_params) {
    return mli_math_add_fx(init_accum, mli_math_cast_fx<int16_t, mli_acc40_t>(*bias, -quant_params->bias_shift));
}

template <>
MLI_FORCE_INLINE mli_acc32_t bias_additive(
        const MLI_PTR(int32_t) bias, mli_acc32_t init_accum, const s8asym_quant_specific_params* quant_params) {
    // For I8ASYM Bias is of the similar format as result accumulator.
    // To prevent saturation during dotproduct we add bias in the end. (saturate final result - not IR)
    return mli_math_add_fx(init_accum, mli_math_cast_fx<mli_acc32_t, int32_t>(*bias, /*right_shift =*/0));
}

//==========================================================================
// Calculation Scale and cast result of dotproduct to output format (requantize)
//==========================================================================
template <>
MLI_FORCE_INLINE int16_t result_cast(const mli_acc40_t acc, const fx_quant_specific_params* math_params) {
    return mli_math_cast_fx<mli_acc40_t, int16_t>(acc, math_params->out_shift);
}

template <>
MLI_FORCE_INLINE int16_t result_cast(const mli_acc32_t acc, const fx_quant_specific_params* math_params) {
    return mli_math_cast_fx<mli_acc32_t, int16_t>(acc, math_params->out_shift);
}

template <>
MLI_FORCE_INLINE int8_t result_cast(const mli_acc32_t acc, const fx_quant_specific_params* math_params) {
    return mli_math_cast_fx<mli_acc32_t, int8_t>(acc, math_params->out_shift);
}


template <>
MLI_FORCE_INLINE int8_t result_cast(
        const mli_acc32_t acc,
        const s8asym_quant_specific_params* quant_params) {

    // adding the output offset needs to happen after the output mul and output shift
    // but before the cast to the output container size.

    int32_t out_mul = quant_params->out_mul;
    int out_shift = quant_params->out_shift;
    mli_acc32_t accu = acc;
    int preshift = 0;

#if !defined(FULL_ACCU)
    // The accumulator has 8 guard bits. If we pre-shift the accumulator to make it fit into
    // 16bit, we can do the rest of the post processing in 16bit. (output multiplier and relu)
    // shifting too much will lose accuracy, shifting too little will case saturation
    // a 3bit headroom is required (1 sign bit, 1bit for the range of mul, and 1bit for the offset)
    // From the 32 bits that come out of the multiplier (16bit reduced accu x 16bit multiplier),
    // we need 8 bits for the output, 3 bits of headroom.
    // this means that the last output shift will need to shift 5 bits. the rest is shifted here.
    // adding clipping to avoid negative shift. and shifting more than 8 is also not needed.
    int target_out_shift = 32 - 8 - 3;

    // reduce out_mul to 16bit
    int int_to_short_shift = 16;
    out_mul = mli_math_asr_fx(out_mul, int_to_short_shift);
    out_shift -= int_to_short_shift;

    // preshift and clip to 16bit, but keep 32bit container for easy connection to rest of code.
    preshift = mli_math_min_fx(mli_math_max_fx(out_shift - target_out_shift, 0),8);
    accu = mli_math_asr_rnd_fx(accu, preshift);
    accu = mli_math_sat_fx(accu, 16);
#endif
    const int32_t accu_result = mli_math_cast_fx<mli_acc32_t, int32_t>(accu, 0);
    const int64_t accu_scaled = mli_math_mul_fx<int32_t, int64_t>(accu_result, out_mul);
    const int16_t out_no_offset = mli_math_cast_fx<int64_t, int16_t>(accu_scaled, out_shift - preshift);
    int8_t out_val = mli_math_cast_fx<int16_t, int8_t>(mli_math_add_fx(out_no_offset, quant_params->out_offset), 0);
    return out_val;
}

template <typename o_T, typename acc_T, typename quant_T>
static MLI_FORCE_INLINE void result_cast_relu_store(
        MLI_CONV_OUT_PTR(o_T) __restrict o_ptr,
        acc_T acc,
        const quant_T* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit) {

    o_T out = result_cast<o_T, acc_T, quant_T>(acc, quant_params);
    out = MIN(out, val_max_limit);
    out = MAX(out, val_min_limit);

    *o_ptr = (o_T) out;
}



template <typename io_T, typename acc_T, typename b_T, mli_math_type math_type>
MLI_FORCE_INLINE io_T result_cast(const acc_T acc, const b_T bias, const int32_t out_mul,
                               const conv_math_params* math_params) {
    return mli_math_cast_fx<acc_T, io_T>(acc, math_params->fx.out_shift);
}

template <>
MLI_FORCE_INLINE int8_t result_cast<int8_t, mli_acc32_t, int32_t, S8ASYM_MATH>(
        const mli_acc32_t acc,
        const int32_t bias,
        const int32_t out_mul,
        const conv_math_params* math_params) {
    const int output_shift = math_params->i8asym.out_shift;
    const int16_t out_offset = math_params->i8asym.out_offset;

    int32_t accu_result = mli_math_cast_fx<mli_acc32_t, int32_t>(acc, 0);

    // For I8ASYM Bias is of the similar format as result accumulator.
    // To prevent saturation during dotproduct we add bias in the end. (saturate final result - not IR)
    accu_result = mli_math_add_fx(accu_result, bias);

    // adding the output offset needs to happen after the output mul and output shift
    // but before the cast to the output container size.
    // because the cast and shift are combined in one function, the output offset is
    // added before, and multiplied with 1<< out_shift to compensate.
    const int64_t accu_scaled = mli_math_mul_fx<int32_t, int64_t>(accu_result, out_mul);
    const int16_t out_no_offset = mli_math_cast_fx<int64_t, int16_t>(accu_scaled, output_shift);
    int8_t out_val = mli_math_cast_fx<int16_t, int8_t>(mli_math_add_fx(out_no_offset, out_offset), 0);
    return out_val;
}

// Convert between SA8 and FX16
//=========================================================================
template <typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_prv_convert_sa8_fx16(
    const in_T in,
    const int16_t zero_point,
    const int scale) {
    int16_t in_biased_shifted_no_zp = mli_math_cast_fx<in_T, int16_t>(in, 0) - zero_point;
    return mli_math_cast_fx<int64_t, out_T>(mli_math_mul_fx<int32_t, int64_t>((int32_t)in_biased_shifted_no_zp, scale), 0);
}

template <typename in_T, typename out_T>
MLI_FORCE_INLINE out_T mli_prv_convert_fx16_sa8(
    const in_T in,
    const int16_t zero_point,
    const int scale) {

    mli_acc32_t fx_output32 = (mli_acc32_t) in;
    // Converting to float and back to asym8
    mli_acc32_t fx_output32_shifted = mli_math_acc_ashift_fx<mli_acc32_t>(fx_output32, scale) + zero_point;
    return mli_math_acc_cast_fx<out_T, mli_acc32_t>(fx_output32_shifted, 0);
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif /* _MLI_PRV_QUANT_REF_H_ */