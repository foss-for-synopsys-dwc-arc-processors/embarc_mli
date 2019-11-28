/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_QUANT_H_
#define _MLI_PRV_QUANT_H_

#include <assert.h>

#include "mli_check.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math_macros.h"
#include "mli_types.h"
#include "mli_private_types.h"
#include <arc/arc_intrinsics.h>


typedef enum {
    FX_MATH = 0,
    S8ASYM_MATH
} mli_math_type;



/**
 * @brief Quantization specific parameter to perform correct calculations in s8asym quantization scheme.
 */
struct s8asym_quant_specific_params {
    int16_t in_offset;
    int16_t out_offset;
    int16_t weights_offset;

    const int16_t *weight_scales;
    int16_t in_to_out_scales_ratio;
    
    int32_t out_mul;
    int out_shift;
};

/**
 * @brief Quantization specific parameter to perform correct calculations in MLI_FX quantization scheme.
 */
struct fx_quant_specific_params {
    int bias_shift;
    int out_shift;
};

typedef union _conv_math_params {
    fx_quant_specific_params fx;

    s8asym_quant_specific_params i8asym;
} conv_math_params;

static const int kPreDivShiftS16 = 14;

//=========================================================================
//
// Declaration
//
//=========================================================================
template <typename quant_T>
inline void define_quant_params(const mli_tensor* in, const mli_tensor* weights, const mli_tensor* bias,
                                const mli_tensor* out, quant_T* params);

template <typename quant_T>
inline void adjust_quant_params(quant_T* params, int krn_idx = 0);

template <typename w_T, typename acc_T, typename quant_T>
inline acc_T weights_additive(const w_T* __restrict weights, acc_T init_accum, const quant_T* quant_params,
                              const int width, const int height = 1, int col_step = 1, int row_step = 1);

template <typename in_T, typename acc_T, typename quant_T>
inline acc_T in_additive(const in_T* __restrict in, acc_T init_accum, const quant_T* quant_params,
                              const int width, const int height = 1, int col_step = 1, int row_step = 1);

template <typename acc_T, typename quant_T>
inline acc_T zp_additive(const quant_T* quant_params, acc_T init_accum,
                        const int mac_serias_len);

template <typename b_T, typename acc_T, typename quant_T>
inline acc_T bias_additive(const b_T bias, acc_T init_accum, const quant_T* quant_params);

template <typename o_T, typename acc_T, typename quant_T>
inline o_T result_cast(const acc_T acc, const quant_T* quant_params);

//=========================================================================
//
// Definitions
//
//=========================================================================

//==========================================================================
// Operating with quantization params set
//==========================================================================
template <>
inline void define_quant_params(const mli_tensor* in, const mli_tensor* weights, const mli_tensor* bias,
                                const mli_tensor* out, fx_quant_specific_params* params) {
    params->bias_shift = mli_prv_calc_shift(in, weights, bias);
    params->out_shift = mli_prv_calc_shift(in, weights, out); 
}

template <>
inline void define_quant_params(const mli_tensor *in, const mli_tensor  *weights, const mli_tensor  *bias,
                                const mli_tensor   *out, s8asym_quant_specific_params* params) {
    params->in_offset = in->el_params.asym.zero_point.i16;
    params->out_offset = out->el_params.asym.zero_point.i16;

    if (weights->el_params.asym.dim >= 0 ) {
        params->weights_offset = weights->el_params.asym.zero_point.pi16[0];
        params->weight_scales = weights->el_params.asym.scale.pi16;
    } else { 
        params->weights_offset = weights->el_params.asym.zero_point.i16;
        params->weight_scales = &weights->el_params.asym.scale.i16;
    }
    int32_t scale_unfinished = in->el_params.asym.scale.i16 << kPreDivShiftS16;
    scale_unfinished  = scale_unfinished / out->el_params.asym.scale.i16;
    params->in_to_out_scales_ratio = mli_math_cast_fx<int32_t, int16_t>(scale_unfinished, 0);
    params->out_shift = in->el_params.asym.scale_frac_bits;
    params->out_shift += weights->el_params.asym.scale_frac_bits;
    params->out_shift += (kPreDivShiftS16 - out->el_params.asym.scale_frac_bits);
}

template <>
inline void adjust_quant_params(s8asym_quant_specific_params* params, int krn_idx) {
    // out multiplyer can be different across one of axis (per axis quantization for s8asym)
    params->out_mul = params->in_to_out_scales_ratio * params->weight_scales[krn_idx];
    return; 
}

template <>
inline void adjust_quant_params(fx_quant_specific_params* in, int krn_idx) {
    // No need to adjust something during calculations for MLI_FX specific quantization
    return; 
}

//==========================================================================
// Calculation of weights additive (w_add) in 
// dot_prod_asym = dot_prod_gen + w_add + in_add + zp_add + bias_add
//==========================================================================
template <typename w_T, typename acc_T, typename quant_T>
inline acc_T weights_additive(const w_T* __restrict, acc_T init_accum,
                              const quant_T*,
                              const int, const int, int, int) {
    // By default and for FX quantization scheme, weights additive isn't required
    return init_accum;
}

template <>
inline mli_acc32_t weights_additive(
        const int8_t* __restrict weights, mli_acc32_t init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, int col_step, int row_step) {
    // returns -(in_zero_point * cumsum(weights)) For S8ASYM 
    if (quant_params->in_offset != 0)
        return reduce_sum2D(weights, -quant_params->in_offset, init_accum, width, height, col_step, row_step);
    else 
        return init_accum;
}

//==========================================================================
// Calculation of input additive (in_add) in 
// dot_prod_asym = dot_prod_gen + w_add + in_add + zp_add + bias_add
//==========================================================================
template <typename in_T, typename acc_T, typename quant_T>
inline acc_T in_additive(const in_T* __restrict, acc_T init_accum, const quant_T* quant_params,
                              const int, const int, int, int) {
    // By default and for FX quantization scheme, input additive isn't required
    return init_accum;
}

template <>
inline mli_acc32_t in_additive(
        const int8_t* __restrict in, mli_acc32_t init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width, const int height, int col_step, int row_step) {
    // returns -(wights_zero_point * cumsum(input)) For S8ASYM 
    if (quant_params->weights_offset != 0) {
        return reduce_sum2D(in, -quant_params->weights_offset, init_accum, width, height, col_step, row_step);
    } else {
        return init_accum;
    }
    
}

//==========================================================================
// Calculation of zero points additive (zp_add) in 
// dot_prod_asym = dot_prod_gen + w_add + in_add + zp_add + bias_add
//==========================================================================
template <typename acc_T, typename quant_T>
inline acc_T zp_additive(const quant_T*, acc_T init_accum,
                        const int) {
    // By default (for FX quantization scheme) weights additive isn't required
    return init_accum;
}


template <>
inline mli_acc32_t zp_additive(const s8asym_quant_specific_params* quant_params, mli_acc32_t init_accum,
                               const int mac_serias_len) {
    if (quant_params->weights_offset != 0 || quant_params->in_offset != 0)
        // Calculating (w_zp * in_zp * mac_serias_len) via reduce sum because of complexity with accum casts.
        // Subject for optimization
        return reduce_sum(&quant_params->weights_offset, quant_params->in_offset, init_accum, mac_serias_len, /*step = */0);
    else
        return init_accum;
}

//==========================================================================
// Calculation of bias additive (bias_add) in 
// dot_prod_asym= dot_prod_gen + w_add + in_add + zp_add + bias_add
//==========================================================================
template <>
inline mli_acc32_t bias_additive(
        const int8_t bias, mli_acc32_t init_accum, const fx_quant_specific_params* quant_params) {
    mli_acc32_t accu = mli_math_mul_fx<int8_t, mli_acc32_t>(bias, 1);
    accu = mli_math_acc_ashift_fx(accu, -quant_params->bias_shift);
    return mli_math_add_fx(init_accum, accu);
}

template <>
inline mli_acc40_t bias_additive(
        const int16_t bias, mli_acc40_t init_accum, const fx_quant_specific_params* quant_params) {
    return mli_math_add_fx(init_accum, mli_math_cast_fx<int16_t, mli_acc40_t>(bias, -quant_params->bias_shift));
}

template <>
inline mli_acc32_t bias_additive(
        const int32_t bias, mli_acc32_t init_accum, const s8asym_quant_specific_params* quant_params) {
    // For I8ASYM Bias is of the similar format as result accumulator.
    // To prevent saturation during dotproduct we add bias in the end. (saturate final result - not IR)
    return mli_math_add_fx(init_accum, mli_math_cast_fx<mli_acc32_t, int32_t>(bias, /*right_shift =*/0));
}

//==========================================================================
// Calculation Scale and cast result of dotproduct to output format (requantize)
//==========================================================================
template <>
inline int16_t result_cast(const mli_acc40_t acc, const fx_quant_specific_params* math_params) {
    return mli_math_cast_fx<mli_acc40_t, int16_t>(acc, math_params->out_shift);
}

template <>
inline int16_t result_cast(const mli_acc32_t acc, const fx_quant_specific_params* math_params) {
    return mli_math_cast_fx<mli_acc32_t, int16_t>(acc, math_params->out_shift);
}

template <>
inline int8_t result_cast(const mli_acc32_t acc, const fx_quant_specific_params* math_params) {
    return mli_math_cast_fx<mli_acc32_t, int8_t>(acc, math_params->out_shift);
}


template <>
inline int8_t result_cast(
        const mli_acc32_t acc,
        const s8asym_quant_specific_params* quant_params) {

    // adding the output offset needs to happen after the output mul and output shift
    // but before the cast to the output container size.
    // because the cast and shift are combined in one function, the output offset is
    // added before, and multiplied with 1<< out_shift to compensate.
    const int32_t accu_result = mli_math_cast_fx<mli_acc32_t, int32_t>(acc, 0);
    const int64_t accu_scaled = mli_math_mul_fx<int32_t, int64_t>(accu_result, quant_params->out_mul);
    const int16_t out_no_offset = mli_math_cast_fx<int64_t, int16_t>(accu_scaled, quant_params->out_shift);
    int8_t out_val = mli_math_cast_fx<int16_t, int8_t>(mli_math_add_fx(out_no_offset, quant_params->out_offset), 0);
    return out_val;
}



template <typename io_T, typename acc_T, typename b_T, mli_math_type math_type>
inline io_T result_cast(const acc_T acc, const b_T bias, const int32_t out_mul, 
                               const conv_math_params* math_params) {
    return mli_math_cast_fx<acc_T, io_T>(acc, math_params->fx.out_shift);
}

template <>
inline int8_t result_cast<int8_t, mli_acc32_t, int32_t, S8ASYM_MATH>(
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


#endif //_MLI_PRV_QUANT_H_