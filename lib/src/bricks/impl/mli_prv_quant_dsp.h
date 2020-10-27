/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_QUANT_DSP_H_
#define _MLI_PRV_QUANT_DSP_H_

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

#include <arc/arc_intrinsics.h>
#include <assert.h>

namespace mli {
namespace krn {
namespace dsp {

static const int kPreDivShiftS16 = 14;
static const int kPreDivShiftS32 = 30;

template <>
MLI_FORCE_INLINE void adjust_quant_params(s8asym_quant_specific_params* params, int krn_idx) {
    // out multiplyer can be different across one of axis (per axis quantization for s8asym)
    if (params->weight_dim < 0) {
        krn_idx = 0;
    }
    params->out_mul = params->in_to_out_scales_ratio * params->weight_scales[krn_idx];

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

//The function uses pointers to pointers for weights.
//The caller of the function should compensate for the increment
//done inside this function.
template <typename acc_T>
MLI_FORCE_INLINE acc_T weights_additive_v(
        const MLI_PTR(int8_t) __restrict *weights, acc_T *init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, int col_step, int row_step) {

    // returns -(in_zero_point * cumsum(weights)) For S8ASYM
    if (quant_params->in_offset != 0) {
        acc_T tmp_acc = reduce_sum2D_v(weights, -quant_params->in_offset, *init_accum, width, height, col_step, row_step, true);
        return tmp_acc;
    } else {
        return *init_accum;
    }
}

template <typename acc_T>
MLI_FORCE_INLINE acc_T weights_additive_v(
        const MLI_PTR(int8_t) __restrict weights, acc_T *init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, int col_step, int row_step) {

    // returns -(in_zero_point * cumsum(weights)) For S8ASYM
    if (quant_params->in_offset != 0) {
        acc_T tmp_acc = reduce_sum2D_v(weights, -quant_params->in_offset, *init_accum, width, height, col_step, row_step, true);
        return tmp_acc;
    } else {
        return *init_accum;
    }
}

MLI_FORCE_INLINE mli_acc32_t weights_additive_d(
        const MLI_PTR(int8_t) __restrict weights, mli_acc32_t *init_accum,
        const s8asym_quant_specific_params* quant_params,
        const int width,  const int height, int col_step, int row_step) {
    // returns -(in_zero_point * cumsum(weights)) For S8ASYM
    if (quant_params->in_offset != 0) {
        mli_acc32_t tmp_acc = reduce_sum2D_d(weights, -quant_params->in_offset, *init_accum, width, height, col_step, row_step, true);
        return tmp_acc;
    } else {
        return *init_accum;
    }
}

// Depending on memory alignment of input pointers, certain functions below will perform
// unaligned loads/stores. Since the core supports this, we disable the related compiler warning.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-align"

template<>
MLI_FORCE_INLINE void result_cast_relu_store(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        int32_t conv_out,
        const s8asym_quant_specific_params* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit) {

    accum72_t accu_scaled = fx_a72_mpy_q31(conv_out, quant_params->out_mul);
    int16_t out_no_offset = fx_q15_cast_nf_asl_rnd_a72(accu_scaled, 64 - sizeof(int16_t) * 8 - quant_params->out_shift);
    int16_t out_with_offset = fx_add_q15(out_no_offset, quant_params->out_offset);

    // no saturation needed because ReLu clipping is done in 32bit domain.
    // ReLU truncation
    out_with_offset = MIN(out_with_offset, val_max_limit);
    out_with_offset = MAX(out_with_offset, val_min_limit);

    *o_ptr = (int8_t)out_with_offset;
}

MLI_FORCE_INLINE void result_cast_relu_store_v(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        __v2i32_t *conv_out_v,
        const s8asym_quant_specific_params quant_params[],
        const int16_t val_min_limit,
        const int16_t val_max_limit) {
    accum72_t accu_scaled = fx_a72_mpy_q31((*conv_out_v)[0], quant_params[0].out_mul);
    int16_t out_no_offset_ch1 = fx_q15_cast_nf_asl_rnd_a72(accu_scaled, 64 - sizeof(int16_t) * 8 - quant_params[0].out_shift);

    accu_scaled = fx_a72_mpy_q31((*conv_out_v)[1], quant_params[1].out_mul);
    int16_t out_no_offset_ch2 = fx_q15_cast_nf_asl_rnd_a72(accu_scaled, 64 - sizeof(int16_t) * 8 - quant_params[1].out_shift);

    v2q15_t v2quant_out_offset = {quant_params[0].out_offset, quant_params[1].out_offset};

    v2q15_t v2out_no_offset = {out_no_offset_ch1, out_no_offset_ch2};

    v2q15_t v2val_max_limit = {val_max_limit, val_max_limit};
    v2q15_t v2val_min_limit = {val_min_limit, val_min_limit};
    v2q15_t v2out_offset = fx_add_v2q15(v2out_no_offset, v2quant_out_offset);

    // no saturation needed because ReLu clipping is done in 32bit domain.
    // ReLU truncation
    v2out_offset = fx_min_v2q15(v2out_offset, v2val_max_limit);
    v2out_offset = fx_max_v2q15(v2out_offset, v2val_min_limit);

    // Write result
    *((v2i8_t *) o_ptr) = __builtin_convertvector((v2out_offset), v2i8_t);
}

MLI_FORCE_INLINE void result_cast_relu_store_inp_width_v(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        __v2i32_t *conv_out_v,
        const s8asym_quant_specific_params *quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int next_out_indx) {
    accum72_t accu_scaled = fx_a72_mpy_q31((*conv_out_v)[0], quant_params->out_mul);
    int16_t out_no_offset_ch1 = fx_q15_cast_nf_asl_rnd_a72(accu_scaled, 64 - sizeof(int16_t) * 8 - quant_params->out_shift);

    accu_scaled = fx_a72_mpy_q31((*conv_out_v)[1], quant_params->out_mul);
    int16_t out_no_offset_ch2 = fx_q15_cast_nf_asl_rnd_a72(accu_scaled, 64 - sizeof(int16_t) * 8 - quant_params->out_shift);

    v2q15_t v2quant_out_offset = {quant_params->out_offset, quant_params->out_offset};

    v2q15_t v2out_no_offset = {out_no_offset_ch1, out_no_offset_ch2};

    v2q15_t v2val_max_limit = {val_max_limit, val_max_limit};
    v2q15_t v2val_min_limit = {val_min_limit, val_min_limit};
    v2q15_t v2out_offset = fx_add_v2q15(v2out_no_offset, v2quant_out_offset);

    // no saturation needed because ReLu clipping is done in 32bit domain.
    // ReLU truncation
    v2out_offset = fx_min_v2q15(v2out_offset, v2val_max_limit);
    v2out_offset = fx_max_v2q15(v2out_offset, v2val_min_limit);

    // Write result
    o_ptr[0]             = (int8_t)(v2out_offset[0]);
    o_ptr[next_out_indx] = (int8_t)(v2out_offset[1]);
}

#pragma clang diagnostic pop

// Convert between SA8 and FX16
//=========================================================================
template<>
MLI_FORCE_INLINE v2q15_t mli_prv_convert_sa8_fx16(
    const v2q15_t in,
    const int16_t zero_point,
    const int scale) {
    v2q15_t zero_point_v = fx_replic_v2q15(zero_point);
    v2q15_t in_biased_shifted_no_zp = fx_sub_v2q15(in, zero_point_v);
    int16_t res_1 = mli_math_cast_fx<int64_t, int16_t>(mli_math_mul_fx<int32_t, int64_t>((int32_t)in_biased_shifted_no_zp[0], scale), 0);
    int16_t res_2 = mli_math_cast_fx<int64_t, int16_t>(mli_math_mul_fx<int32_t, int64_t>((int32_t)in_biased_shifted_no_zp[1], scale), 0);
    return fx_create_v2q15(res_1, res_2);
}

template<>
MLI_FORCE_INLINE v2q15_t mli_prv_convert_fx16_sa8(
    const v2q15_t in,
    const int16_t zero_point,
    const int scale) {

    mli_acc32_t fx_output32 = (mli_acc32_t) in[0];
    // Converting to float and back to asym8
    mli_acc32_t fx_output32_shifted = mli_math_acc_ashift_fx<mli_acc32_t>(fx_output32, scale) + zero_point;
    int8_t res_1 = mli_math_acc_cast_fx<int8_t, mli_acc32_t>(fx_output32_shifted, 0);
    fx_output32 = (mli_acc32_t) in[1];
    // Converting to float and back to asym8
    fx_output32_shifted = mli_math_acc_ashift_fx<mli_acc32_t>(fx_output32, scale) + zero_point;
    int8_t res_2 = mli_math_acc_cast_fx<int8_t, mli_acc32_t>(fx_output32_shifted, 0);
    return fx_create_v2q15(res_1, res_2);
}

template<>
MLI_FORCE_INLINE v2q15_t mli_prv_convert_fx16_sa8(
    const v2accum40_t in,
    const int16_t zero_point,
    const int scale) {

    v2q15_t x = mli_math_acc_cast_fx<v2q15_t, v2accum40_t>(in, scale) + zero_point;
    return fx_sat_v2q15_n(x, 8);
}

} // namespace dsp
} // namespace krn
} // namespace mli

#endif /* _MLI_PRV_QUANT_DSP_H_ */