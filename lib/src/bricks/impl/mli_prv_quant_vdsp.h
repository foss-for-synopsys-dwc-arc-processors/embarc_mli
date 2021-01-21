/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_QUANT_VDSP_H_
#define _MLI_PRV_QUANT_VDSP_H_

#include "mli_prv_quant_decl.h"
#include "mli_prv_load_store.h"
#include "mli_config.h"
#include "mli_krn_dotprod.h"

namespace mli {
namespace krn {
namespace vdsp {

//funtion is temporary here until reduce_sum brick is available
MLI_FORCE_INLINE vNx4accshort_t reduce_sum2D(
        const MLI_PTR(int8_t) __restrict in,
        const int8_t mul,
        vNx4accshort_t accu,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step) {
    in_row_step -= width * in_col_step;
    for (int row = 0; row < height; row++) {
        for (int clmn = 0; clmn < width; clmn++) {
            accu = mli_math_mac_fx(accu, mli_prv_load_nx4_samples(in), mul);
            in += in_col_step;
        }
        in += in_row_step;
    }
    return accu;
}

MLI_FORCE_INLINE vNx4accshort_t reduce_sub_sum2D(
        const MLI_PTR(int8_t) __restrict in,
        const int8_t mul,
        vNx4accshort_t accu,
        const int width,
        const int height,
        int in_col_step,
        int in_row_step) {
    in_row_step -= width * in_col_step;
    for (int row = 0; row < height; row++) {
        for (int clmn = 0; clmn < width; clmn++) {
            accu = mli_math_msub_fx(accu, mli_prv_load_nx4_samples(in), mul);
            in += in_col_step;
        }
        in += in_row_step;
    }
    return accu;
}

MLI_FORCE_INLINE s8asym_quant_specific_out_params_v adjust_quant_params_v(s8asym_quant_specific_params* params, int krn_idx) {
    // out multiplyer can be different across one of axis (per axis quantization for s8asym)
    // but will be the same in case of per tensor quantization.
    vNx4short_t wscales;
    vNx4short_t wshifts;
    if (params->weight_dim < 0) {
        krn_idx = 0;
        wscales = params->weight_scales[krn_idx];
        wshifts = params->weight_shifts[krn_idx];
    } else {
        wscales = mli_prv_load_nx4_samples(&params->weight_scales[krn_idx]);
        wshifts = to_vNx4short_t(mli_prv_load_nx4_samples(&params->weight_shifts[krn_idx]));
    }
    s8asym_quant_specific_out_params_v out_params;

    vNx4int_t outmul32 = to_vNx4int_t(mli_math_mul_fx<vNx4short_t, vNx4accint_t>(wscales, (vNx4short_t)(params->in_to_out_scales_ratio)));
    vNx4int_t mul_norm = mli_math_norm_fx<vNx4int_t, vNx4int_t>(outmul32);
    int int_to_short_shift = 16;
    out_params.out_mul = to_vNx4short_t(mli_math_asr_rnd_fx(outmul32, int_to_short_shift - mul_norm));

    int out_shift = params->in_to_out_shift;
    out_params.out_shift = out_shift - sizeof(int16_t) * 8; // compensate for the mul_hi output multiplier
    out_params.out_shift += wshifts;
    out_params.out_shift -= int_to_short_shift; // for the outmul int to short
    out_params.out_shift += to_vNx4short_t(mul_norm);

    out_params.out_offset = params->out_offset;
    return out_params;
}

MLI_FORCE_INLINE fx_quant_specific_params adjust_quant_params_v(fx_quant_specific_params* in, int krn_idx) {
    // No need to adjust something during calculations for MLI_FX specific quantization
    return *in;
}

//==========================================================================
// Calculation of a dotprod with an offset on the input values.
// offset on the weights is assumed zero
//==========================================================================
template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod_inputzp_1D_v(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int vals,
        const int in_step,
        const int krn_step,
        const s8asym_quant_specific_params* quant_params) {

    for (int idx = 0; idx < vals; idx++) {
        io_T offset = (io_T)quant_params->in_offset;
        accu = mli_prv_mac_load_v_s(accu, krn, in);
        accu = mli_prv_msub_load_v_s(accu, krn, &offset);
        in += in_step;
        krn += krn_step;
    }

    return accu;
}

template <>
MLI_FORCE_INLINE vNx4accshort_t dotprod_inputzp_1D_v(
        const MLI_PTR(int8_t) __restrict in,
        const MLI_PTR(int8_t) __restrict krn,
        vNx4accshort_t accu,
        const int vals,
        const int in_step,
        const int krn_step,
        const s8asym_quant_specific_params* quant_params) {

    int unroll_count = 4;
    int idx;
    // using a second accumulators allows for using two accumulators in parallel.
    vNx4accshort_t accu2 = mli_math_mul_fx<int8_t, vNx4accshort_t>(0, 0);

// the extra unroll factor enables the compiler to combine the scalar loads into an ldd
#pragma clang loop unroll_count(2)
    for (idx = 0; idx < vals - (unroll_count - 1); idx+=unroll_count) {
        // load 4 input samples at once to reduce load bottleneck
        int32_t in4x = *(int32_t*)in;

        accu = mli_math_mac_fx(accu, mli_prv_load_nx4_samples(krn), (int8_t)in4x);
        accu2 = mli_math_msub_fx(accu2, mli_prv_load_nx4_samples(krn), (int8_t)quant_params->in_offset);
        krn += krn_step;
        in4x = in4x >> 8;
        accu = mli_math_mac_fx(accu, mli_prv_load_nx4_samples(krn), (int8_t)in4x);
        accu2 = mli_math_msub_fx(accu2, mli_prv_load_nx4_samples(krn), (int8_t)quant_params->in_offset);
        krn += krn_step;
        in4x = in4x >> 8;
        accu = mli_math_mac_fx(accu, mli_prv_load_nx4_samples(krn), (int8_t)in4x);
        accu2 = mli_math_msub_fx(accu2, mli_prv_load_nx4_samples(krn), (int8_t)quant_params->in_offset);
        krn += krn_step;
        in4x = in4x >> 8;
        accu = mli_math_mac_fx(accu, mli_prv_load_nx4_samples(krn), (int8_t)in4x);
        accu2 = mli_math_msub_fx(accu2, mli_prv_load_nx4_samples(krn), (int8_t)quant_params->in_offset);
        krn += krn_step;
        in += in_step * 4;
    }
    for ( ; idx < vals; idx++) {
        accu = mli_math_mac_fx(accu, mli_prv_load_nx4_samples(krn), *in);
        accu2 = mli_math_msub_fx(accu2, mli_prv_load_nx4_samples(krn), (int8_t)quant_params->in_offset);
        in += in_step;
        krn += krn_step;
    }
    return mli_math_add(accu, accu2);
}

template <typename io_T, typename w_T, typename acc_T>
static MLI_FORCE_INLINE acc_T dotprod_inputzp_1D_v(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict krn,
        acc_T accu,
        const int vals,
        const int in_step,
        const int krn_step,
        const fx_quant_specific_params* quant_params) {

    return mli::krn::dotprod1D_v(in, krn, accu, vals, in_step, krn_step);
}
//==========================================================================
// Calculation of weights additive (w_add) in
// dot_prod_asym = dot_prod_gen + w_add + in_add + zp_add + bias_add
//==========================================================================
template <typename w_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T weights_additive(
        const MLI_PTR(w_T) __restrict weights, acc_T init_accum,
        const quant_T* quant_params,
        const int width, const int height, const int ch, int col_step, int row_step, int ch_step) {
    return mli::krn::ref::weights_additive(weights, init_accum,
        quant_params,
        width, height, ch, col_step, row_step, ch_step);
}

template <>
MLI_FORCE_INLINE vNx4accshort_t weights_additive(
        const MLI_PTR(int8_t) __restrict weights, vNx4accshort_t init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, const int ch, int col_step, int row_step, int ch_step) {
    // returns -(in_zero_point * cumsum(weights)) For S8ASYM
    if (quant_params->in_offset != 0) {
        for (int c = 0; c < ch; c++) {
            init_accum = reduce_sub_sum2D(weights, (int8_t)quant_params->in_offset, init_accum, width, height, col_step, row_step);
            weights += ch_step;
        }
        return init_accum;
    } else {
        return init_accum;
    }
}

//==========================================================================
// Calculation of bias additive (bias_add) in
// dot_prod_asym= dot_prod_gen + w_add + in_add + zp_add + bias_add
//==========================================================================
template <typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE acc_T bias_additive(const MLI_PTR(b_T) bias, acc_T init_accum,
        const quant_T* quant_params) {
    return mli::krn::ref::bias_additive(bias, init_accum, quant_params);
}

template <>
MLI_FORCE_INLINE vNx4accshort_t bias_additive(
        const MLI_PTR(int32_t) bias, vNx4accshort_t init_accum, const s8asym_quant_specific_params* quant_params) {
    // For I8ASYM with a 24bit accumulator the bias cannot be loaded directly into the accumulator.
    // 16 bits are loaded into the accumulator and then shifted to the correct position.
    // for this reason the bias additve has to be the first operation on the accumulator.
    MLI_ASSERT(to_vNx4int_t(init_accum)[0] == 0);
    vNx4int_t bias32 = mli_prv_load_nx4_samples(bias);
    vNx4int_t norm = mli_math_norm_fx<vNx4int_t,vNx4int_t>(bias32);
    vNx4int_t shift = mli_math_max_fx(16 - norm, 0);
    vNx4short_t bias16 = to_vNx4short_t(bias32 >> shift);
    vNx4accshort_t accu = mli_math_init_accu<vNx4short_t, vNx4accshort_t>(bias16);
    accu = mli_math_asl_fx(accu, to_vNx4short_t(shift));
    return accu;
}

template <>
MLI_FORCE_INLINE vNx4accshort_t bias_additive(
        const MLI_PTR(int8_t) bias, vNx4accshort_t init_accum, const fx_quant_specific_params* quant_params) {
    MLI_ASSERT(to_vNx4int_t(init_accum)[0] == 0);
    vNx4char_t bias_v = mli_prv_load_nx4_samples(bias);
    vNx4accshort_t accu = mli_math_mul_fx<vNx4char_t, vNx4accshort_t>(bias_v, 1);
    accu = mli_math_asl_fx(accu, quant_params->bias_shift);
    return accu;
}

template <>
MLI_FORCE_INLINE vNx2accint_t bias_additive(
        const MLI_PTR(int16_t) bias, vNx2accint_t init_accum, const fx_quant_specific_params* quant_params) {
    MLI_ASSERT(to_vNx2int_t(init_accum)[0] == 0);
    vNx2short_t bias_v = mli_prv_load_nx2_samples(bias);
    vNx2accint_t accu = mli_math_mul_fx<vNx2short_t, vNx2accint_t>(bias_v, 1);
    accu = mli_math_asl_fx(accu, quant_params->bias_shift);
    return accu;
}

template <>
MLI_FORCE_INLINE vNx4accint_t bias_additive(
        const MLI_PTR(int8_t) bias, vNx4accint_t init_accum, const fx_quant_specific_params* quant_params) {
    MLI_ASSERT(to_vNx4int_t(init_accum)[0] == 0);
    vNx4char_t bias_v = mli_prv_load_nx4_samples(bias);
    vNx4accint_t accu = mli_math_mul_fx<vNx4short_t, vNx4accint_t>(to_vNx4short_t(bias_v), 1);
    accu = mli_math_asl_fx(accu, quant_params->bias_shift);
    return accu;
}

//=========================================================================
// Convert between SA8 and FX16
//=========================================================================
template<>
MLI_FORCE_INLINE vNx4short_t mli_prv_convert_sa8_fx16(
        const vNx4short_t in_val,
        const int16_t zero_point,
        const int scale) {
    vNx4int_t in_biased_shifted_no_zp = mli_math_cast_fx<vNx4short_t, vNx4int_t>(in_val) - (vNx4int_t) zero_point;
    return mli_math_cast_fx<vNx4int_t, vNx4short_t>(mli_math_bound_range_fx(in_biased_shifted_no_zp * (vNx4int_t) scale, INT16_MIN, INT16_MAX));
}

MLI_FORCE_INLINE vNx4int_t mli_prv_convert_sa8_fx32(
        const vNx4char_t in_val,
        const int16_t zero_point,
        const int scale) {
    vNx4int_t in_biased_shifted_no_zp = mli_math_cast_fx<vNx4char_t, vNx4int_t>(in_val) - (vNx4int_t) zero_point;
    return in_biased_shifted_no_zp * (vNx4int_t) scale;
}

template<>
MLI_FORCE_INLINE vNx4char_t mli_prv_convert_fx16_sa8(
        const vNx4short_t in_val,
        const int16_t zero_point,
        const int scale) {
    vNx4short_t res = mli_math_cast_fx<vNx4short_t, vNx4short_t>(in_val, scale) + zero_point;
    return to_vNx4char_t(mli_math_bound_range_fx(res, INT8_MIN, INT8_MAX));
}

template<>
MLI_FORCE_INLINE vNx4char_t mli_prv_convert_fx16_sa8(
    const vNx4int_t in,
    const int16_t zero_point,
    const int scale) {
    // Converting to float and back to asym8
    vNx4int_t in_shifted = mli_math_asr_rnd_fx<vNx4int_t>(in, (vNx4int_t) scale) + (vNx4int_t) ((int32_t) zero_point);
    return mli_math_cast_fx<vNx4int_t, vNx4char_t>(in_shifted);
}

template<>
MLI_FORCE_INLINE vNx4short_t mli_prv_convert_fx16_sa8(
    const vNx4short_t in,
    const int16_t zero_point,
    const int scale) {
    // Converting to float and back to asym8
    return mli_math_cast_fx<vNx4short_t, vNx4short_t>(in, scale) + zero_point;
}

//==========================================================================
// Storing result
//==========================================================================
template <typename o_T, typename acc_T, typename quant_T>
static MLI_FORCE_INLINE void result_cast_relu_store_v(
        MLI_CONV_OUT_PTR(o_T) __restrict o_ptr,
        acc_T acc,
        const quant_T* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        int num) {
    MLI_ASSERT(num == 1);
    mli::krn::result_cast_relu_store(o_ptr, acc, quant_params, val_min_limit, val_max_limit);
}

template <>
MLI_FORCE_INLINE void result_cast_relu_store_v(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        vNx4accshort_t acc,
        const s8asym_quant_specific_out_params_v* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        int num) {
    // adding the output offset needs to happen after the output mul and output shift
    // but before the cast to the output container size.
#ifndef FULL_ACCU
    // The accumulator has 8 guard bits. If we pre-shift the accumulator to make it fit into
    // 16bit, we can do the rest of the post processing in 16bit.
    // shifting too much will lose accuracy, shifting too little will case saturation
    // a 3bit headroom is required (1 sign bit, 1bit for the range of mul, and 1bit for the offset)
    // From the 16 bits that come out of the mul_hi, we need 8 bits for the output, 3 bits of headroom.
    // this means that the last output shift will need to shift 5 bits. the rest is shifted here.
    // adding clipping to avoid negative shift. and shifting more than 8 is also not needed.
    int target_out_shift = 16 - 8 - 3;
    vNx4short_t preshift = mli_math_min_fx(mli_math_max_fx(quant_params->out_shift - target_out_shift, 0),8);
    acc = mli_math_asr_rnd_fx(acc, preshift);

    vNx4short_t accu_result = mli_math_acc_cast_fx<vNx4short_t, vNx4accshort_t>(acc);
    vNx4short_t shift = quant_params->out_shift - preshift;
    vNx4short_t shift_left = mli_math_max_fx(-shift, 0);
    accu_result = mli_math_asl_fx(accu_result, shift_left);
    vNx4short_t accu_scaled = mli_math_mul_fx_high(accu_result, quant_params->out_mul);
    vNx4short_t shift_right = mli_math_max_fx(shift, 0);
    accu_scaled = mli_math_asr_rnd_fx(accu_scaled, shift_right);

#else
    vNx4int_t accu_result = to_vNx4int_t(acc);
    vNx4int_t accu_scaled = mli_math_mul_fx_high(accu_result, to_vNx4int_t(quant_params->out_mul)<<16);
    vNx4int_t shift = to_vNx4int_t(quant_params->out_shift);
    accu_scaled = mli_math_asr_rnd_fx(accu_scaled, shift);
#endif

    accu_scaled = accu_scaled + quant_params->out_offset;

    accu_scaled = MIN(accu_scaled, val_max_limit);
    accu_scaled = MAX(accu_scaled, val_min_limit);

    vNx4char_t out = to_vNx4char_t(accu_scaled);
    mli_prv_store_n_samples(o_ptr, out, num);
}

template <>
MLI_FORCE_INLINE void result_cast_relu_store_v(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        vNx4accshort_t acc,
        const fx_quant_specific_params* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        int num) {

    vNx4char_t out = mli_math_acc_cast_fx<vNx4char_t, vNx4accshort_t>(acc, quant_params->out_shift);

    out = MIN(out, val_max_limit);
    out = MAX(out, val_min_limit);

    mli_prv_store_n_samples(o_ptr, out, num);
}

template <>
MLI_FORCE_INLINE void result_cast_relu_store_v(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        vNx2accint_t acc,
        const fx_quant_specific_params* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        int num) {

    vNx2short_t out = mli_math_acc_cast_fx<vNx2short_t, vNx2accint_t>(acc, quant_params->out_shift);

    out = MIN(out, val_max_limit);
    out = MAX(out, val_min_limit);

    mli_prv_store_n_samples(o_ptr, out, num);
}

template <>
MLI_FORCE_INLINE void result_cast_relu_store_v(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        vNx4accint_t acc,
        const fx_quant_specific_params* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        int num) {

    vNx4short_t out = mli_math_acc_cast_fx<vNx4short_t, vNx4accint_t>(acc, quant_params->out_shift);

    out = MIN(out, val_max_limit);
    out = MAX(out, val_min_limit);

    mli_prv_store_n_samples(o_ptr, out, num);
}

} // namespace vdsp

} // namespace krn
} // namespace mli

#endif /* _MLI_PRV_QUANT_VDSP_H_ */
