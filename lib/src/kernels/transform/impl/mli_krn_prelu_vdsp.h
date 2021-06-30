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
    /* out  = max(0, in) + alpha * min(0, in) */
    vNx4char_t pos = mli_math_max_fx(input, 0);
    vNx4accshort_t acc = mli_math_mul_fx<vNx4char_t, vNx4accshort_t>(mli_math_min_fx(input, 0), scale);
    vNx4char_t neg = mli_math_acc_cast_fx<vNx4char_t, vNx4accshort_t>(acc, shift);

    return mli_math_add(pos, neg);
}

static MLI_FORCE_INLINE vNx2short_t calc_prelu(
        const vNx2short_t input,
        const vNx2short_t scale,
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

// Compute PRELU with Strides
template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift,
        const int stride_in,
        const int stride_out);

template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift,
        const int stride_in,
        const int stride_out,
        const int remaining_part);

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        const vNx4char_t scale,
        MLI_OUT_PTR(int8_t) vec_out,
        const int shift,
        const int stride_in,
        const int stride_out) {

    vNx4char_t input = mli_prv_stride_load_1vec(vec_in, stride_in);
    mli_prv_stride_store_n_samples(vec_out, calc_prelu(input, scale, shift), stride_out);
}

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int16_t) vec_in,
        const vNx2short_t scale,
        MLI_OUT_PTR(int16_t) vec_out,
        const int shift,
        const int stride_in,
        const int stride_out) {

    vNx2short_t input = mli_prv_stride_load_1vec(vec_in, stride_in);
    mli_prv_stride_store_n_samples(vec_out, calc_prelu(input, scale, shift), stride_out);
}

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        const vNx4char_t scale,
        MLI_OUT_PTR(int8_t) vec_out,
        const int shift,
        const int stride_in,
        const int stride_out,
        const int remaining_part) {

    vNx4char_t input = mli_prv_stride_load_1vec(vec_in, stride_in);
    mli_prv_stride_store_n_samples(vec_out, calc_prelu(input, scale, shift), stride_out, remaining_part);
}

template <>
MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int16_t) vec_in,
        const vNx2short_t scale,
        MLI_OUT_PTR(int16_t) vec_out,
        const int shift,
        const int stride_in,
        const int stride_out,
        const int remaining_part) {

    vNx2short_t input = mli_prv_stride_load_1vec(vec_in, stride_in);
    mli_prv_stride_store_n_samples(vec_out, calc_prelu(input, scale, shift), stride_out, remaining_part);
}

static MLI_FORCE_INLINE s8asym_quant_params_v prelu_define_requant_alpha_params(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        mli_tensor *out,
        const vNx4char_t alpha_sa8,
        const s8asym_quant_params *identity_params) {
    int16_t out_zp = out->el_params.sa.zero_point.mem.i16;
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

    /* Define Quantization params for (In * alpha / out) ratio */
    s8asym_quant_params_v alpha_params;
    alpha_params.scale  = scale_alpha;
    alpha_params.shift  = mli_math_cast_fx<vNx4int_t, vNx4short_t>(scale_alpha_shift);
    alpha_params.offset = (vNx4short_t)out_zp;
    
    return alpha_params;
}

static MLI_FORCE_INLINE vNx4char_t calc_prelu(
        const vNx4char_t input,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params) {

    vNx4short_t input_cast = mli_math_cast_fx<vNx4char_t, vNx4short_t>(input);
    input_cast = mli_math_sub(input_cast, in_zp);
    pvNx4 select = init_predicate(input >= in_zp);
    /*
     * shifting more than 24 is not needed
     * as the scaled result = ((input - in_offset) * scale) will be limited by 24 bits.
     */
    constexpr int max_shift = 24;
    constexpr int mul_hi_shift = 16;

    int identity_shift = identity_params->shift;
    identity_shift = mli_math_min_fx(identity_params->shift, max_shift);
    identity_shift -= mul_hi_shift;
    int shift_left = mli_math_max_fx(1 - identity_shift, 0);
    int shift_right = mli_math_max_fx(identity_shift, 1);
    int16_t identity_offset = identity_params->offset << shift_right;
#ifdef ROUND_UP
    identity_offset += (int)(((uint16_t)1 << shift_right) >> 1);
#else
    #error Rounding mode not supported
#endif
    vNx4short_t input_cast1 = mli_math_asl_fx(input_cast, shift_left);
    vNx4short_t input_identity_scale = mli_math_mul_fx_high(identity_params->scale, input_cast1);
                  input_identity_scale = mli_math_add_fx(input_identity_scale, (vNx4short_t)identity_offset);

    vNx4char_t output_identity = mli_math_cast_fx<vNx4short_t, vNx4char_t, false>(input_identity_scale, shift_right);

    vNx4short_t alpha_shift = mli_math_min_fx(alpha_params->shift, max_shift);
    alpha_shift -= mul_hi_shift;
    vNx4short_t shift_left1 = mli_math_max_fx(1 - alpha_shift, 0);
    vNx4short_t shift_right1 = mli_math_max_fx(alpha_shift, 1);

    vNx4short_t input_cast2 = mli_math_asl_fx(input_cast, shift_left1);
    vNx4short_t input_alpha_scale = mli_math_mul_fx_high(input_cast2, alpha_params->scale);
                input_alpha_scale = mli_math_asr_rnd_fx(input_alpha_scale, shift_right1);
                input_alpha_scale = mli_math_add_fx(input_alpha_scale, alpha_params->offset);

    vNx4char_t output_alpha = mli_math_cast_fx<vNx4short_t, vNx4char_t>(input_alpha_scale);
    return mli_math_select_fx(select, output_identity, output_alpha);
}

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params) {

    vNx4char_t input = mli_prv_load_1vec(vec_in);
    vNx4char_t output = calc_prelu(input, in_zp, identity_params, alpha_params);
    mli_prv_store_n_samples(vec_out, output);
}

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params,
        const int remaining_part) {

    vNx4char_t input = mli_prv_load_1vec(vec_in);
    vNx4char_t output = calc_prelu(input, in_zp, identity_params, alpha_params);
    mli_prv_store_n_samples(vec_out, output, remaining_part);
}

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params,
        const int stride_in,
        const int stride_out) {

    vNx4char_t input = mli_prv_stride_load_1vec(vec_in, stride_in);
    vNx4char_t output = calc_prelu(input, in_zp, identity_params, alpha_params);
    mli_prv_stride_store_n_samples(vec_out, output, stride_out);
}

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params,
        const int stride_in,
        const int stride_out,
        const int remaining_part) {

    vNx4char_t input = mli_prv_stride_load_1vec(vec_in, stride_in);
    vNx4char_t output = calc_prelu(input, in_zp, identity_params, alpha_params);
    mli_prv_stride_store_n_samples(vec_out, output, stride_out, remaining_part);
}

template <typename io_T>
static MLI_FORCE_INLINE void compute_prelu_broadcast(
        generic_tensor_private_t<MLI_PTR(io_T)> in_prv,
        generic_tensor_private_t<MLI_OUT_PTR(io_T)> out_prv,
        const MLI_PTR(io_T) slope_ptr,
        const int axis,
        const int axis_shape,
        const int axis_in_mem_stride,
        const int axis_out_mem_stride,
        const int shift) {

    /* Dummy Load to get num_lanes */
    auto input = mli_prv_load_1vec(in_prv.ptr);
    int num_lanes = get_number_lanes(input);

    /* Get Non Axis Tensor */
    auto in_non_axis_prv  = mli_prv_get_non_axis_tensor<MLI_PTR(io_T)>(&in_prv,  axis);
    auto out_non_axis_prv = mli_prv_get_non_axis_tensor<MLI_PTR(io_T)>(&out_prv, axis);

    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&in_non_axis_prv );
    mli_prv_reorder_generic_tensor<MLI_OUT_PTR(io_T)>(&out_non_axis_prv);

    if (axis_shape > in_non_axis_prv.shape[3]) {
        /* Vectorize Over Axis */
        for (int scale_idx = 0; scale_idx < axis_shape; scale_idx += num_lanes) {
            int remaining_ch = axis_shape - scale_idx;
            int remaining_part = MIN(remaining_ch, num_lanes);
            /* Define Sub Tensor */
            const MLI_PTR(io_T) vec_in  = (MLI_PTR(io_T))in_prv.ptr  + scale_idx * axis_in_mem_stride;
            MLI_OUT_PTR(io_T) vec_out = out_prv.ptr + scale_idx * axis_out_mem_stride;
            /* Load Scale Elem */
            auto scale_v = mli_prv_load_1vec(slope_ptr + scale_idx);
            /* Loop Over Sub Tensor */
            const MLI_PTR(io_T) orig_vec_in = vec_in;
            MLI_OUT_PTR(io_T) orig_vec_out = vec_out;
            for (int pos1 = 0; pos1 < in_non_axis_prv.shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_non_axis_prv.shape[2]; pos2++) {
                    for (int pos3 = 0; pos3 < in_non_axis_prv.shape[3]; pos3++) {
                        vec_in  = (MLI_PTR(io_T))orig_vec_in  + POS(&in_non_axis_prv, 0, pos1, pos2, pos3);
                        vec_out = orig_vec_out + POS(&out_non_axis_prv, 0, pos1, pos2, pos3);
                        compute_prelu<io_T, decltype(input)>(vec_in, scale_v, vec_out, shift,
                                                             axis_in_mem_stride, axis_out_mem_stride, remaining_part);
                    }
                }
            }
        }
    } else {
        int remaining_part = in_non_axis_prv.shape[3] & (num_lanes - 1);
        for (int scale_idx = 0; scale_idx < axis_shape; scale_idx++) {
            /* Define Sub Tensor */
            const MLI_PTR(io_T) vec_in  = (MLI_PTR(io_T))in_prv.ptr  + scale_idx * axis_in_mem_stride;
            MLI_OUT_PTR(io_T) vec_out = out_prv.ptr + scale_idx * axis_out_mem_stride;
            /* Load Scale Elem */
            auto scale_v = mli_prv_init_v<io_T, decltype(input)>(slope_ptr[scale_idx]);
            /* Loop Over Sub Tensor */
            const MLI_PTR(io_T) orig_vec_in = vec_in;
            MLI_OUT_PTR(io_T) orig_vec_out = vec_out;
            for (int pos1 = 0; pos1 < in_non_axis_prv.shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_non_axis_prv.shape[2]; pos2++) {
                    vec_in  = (MLI_PTR(io_T))orig_vec_in  + POS(&in_non_axis_prv, 0, pos1, pos2, 0);
                    vec_out = orig_vec_out + POS(&out_non_axis_prv, 0, pos1, pos2, 0);
                    if (remaining_part) {
                        mli::krn::compute_prelu<io_T, decltype(input)>(vec_in, scale_v, vec_out, shift,
                                                                       remaining_part);
                        vec_in  += remaining_part;
                        vec_out += remaining_part;
                    }
                    for (int pos3 = remaining_part; pos3 < in_non_axis_prv.shape[3]; pos3 += num_lanes) {
                        mli::krn::compute_prelu<io_T, decltype(input)>(vec_in, scale_v, vec_out, shift);
                        vec_in  += num_lanes;
                        vec_out += num_lanes;
                    }
                }
            }
        }
    }
}

template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu_no_broadcast(
        const MLI_PTR(io_T) __restrict vec_in,
        MLI_OUT_PTR(io_T) __restrict vec_out,
        const scale_T scale_v,
        const int shift,
        const generic_tensor_private_t<MLI_PTR(io_T)> in_prv,
        const generic_tensor_private_t<MLI_OUT_PTR(io_T)> out_prv,
        const int remaining_part) {
    /* Loop Over Sub Tensor */
    const MLI_PTR(io_T) orig_vec_in = vec_in;
    MLI_OUT_PTR(io_T) orig_vec_out = vec_out;
    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
#pragma clang loop pipeline(enable)
#pragma clang loop pipeline_options(0x10)
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                vec_in  = (MLI_PTR(io_T))orig_vec_in  + POS(&in_prv, pos0, pos1, pos2, 0);
                vec_out = orig_vec_out + POS(&out_prv, pos0, pos1, pos2, 0);
                if(remaining_part) {
                    mli::krn::compute_prelu<io_T, scale_T>(vec_in, scale_v, vec_out, shift, remaining_part);
                } else {
                    mli::krn::compute_prelu<io_T, scale_T>(vec_in, scale_v, vec_out, shift);
                }
            }
        }
    }
}

static MLI_FORCE_INLINE void compute_prelu_broadcast(
        const mli_tensor *in,
        const mli_tensor *slope_coeff,
        mli_tensor *out,
        generic_tensor_private_t<MLI_PTR(int8_t)> in_prv,
        generic_tensor_private_t<MLI_OUT_PTR(int8_t)> out_prv,
        const MLI_PTR(int8_t) slope_ptr,
        const int axis,
        const int axis_shape,
        const int axis_in_mem_stride,
        const int axis_out_mem_stride,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params) {
    
    /* Dummy Load to get num_lanes */
    auto input = mli_prv_load_1vec(in_prv.ptr);
    int num_lanes = get_number_lanes(input);

    /* Get Non Axis Tensor */
    auto in_non_axis_prv  = mli_prv_get_non_axis_tensor<MLI_PTR(int8_t)>(&in_prv,  axis);
    auto out_non_axis_prv = mli_prv_get_non_axis_tensor<MLI_PTR(int8_t)>(&out_prv, axis);

    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(int8_t)>(&in_non_axis_prv );
    mli_prv_reorder_generic_tensor<MLI_OUT_PTR(int8_t)>(&out_non_axis_prv);

    if (axis_shape > in_non_axis_prv.shape[3]) {
        /* Vectorize Over Axis */
        for (int scale_idx = 0; scale_idx < axis_shape; scale_idx += num_lanes) {
            int remaining_ch = axis_shape - scale_idx;
            int remaining_part = MIN(remaining_ch, num_lanes);
            /* Define Sub Tensor */
            const MLI_PTR(int8_t) vec_in  = (MLI_PTR(int8_t))in_prv.ptr  + scale_idx * axis_in_mem_stride;
            MLI_OUT_PTR(int8_t) vec_out = out_prv.ptr + scale_idx * axis_out_mem_stride;
            /* Load Scale Vector */
            auto scale_v = mli_prv_load_1vec(slope_ptr + scale_idx);
            auto alpha_params = mli::krn::prelu_define_requant_alpha_params(in, slope_coeff, out, scale_v, identity_params);
            /* Loop Over Sub Tensor */
            const MLI_PTR(int8_t) orig_vec_in = vec_in;
            MLI_OUT_PTR(int8_t) orig_vec_out = vec_out;
            for (int pos1 = 0; pos1 < in_non_axis_prv.shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_non_axis_prv.shape[2]; pos2++) {
#pragma clang loop pipeline(enable)
#pragma clang loop pipeline_options(0x10)
                    for (int pos3 = 0; pos3 < in_non_axis_prv.shape[3]; pos3++) {
                        vec_in  = (MLI_PTR(int8_t))orig_vec_in  + POS(&in_non_axis_prv, 0, pos1, pos2, pos3);
                        vec_out = orig_vec_out + POS(&out_non_axis_prv, 0, pos1, pos2, pos3);
                        compute_prelu(vec_in, vec_out, in_zp, identity_params, &alpha_params,
                                      axis_in_mem_stride, axis_out_mem_stride, remaining_part);
                    }
                }
            }
        }
    } else {
        int remaining_part = in_non_axis_prv.shape[3] & (num_lanes - 1);
        for (int scale_idx = 0; scale_idx < axis_shape; scale_idx++) {
            /* Define Sub Tensor */
            const MLI_PTR(int8_t) vec_in  = (MLI_PTR(int8_t))in_prv.ptr  + scale_idx * axis_in_mem_stride;
            MLI_OUT_PTR(int8_t) vec_out = out_prv.ptr + scale_idx * axis_out_mem_stride;
            /* Load Scale Elem */
            auto alpha_params = mli::krn::ref::prelu_define_requant_alpha_params(in, slope_coeff, out,
                                                                           slope_ptr[scale_idx], identity_params);

            /* Loop Over Sub Tensor */
            const MLI_PTR(int8_t) orig_vec_in = vec_in;
            MLI_OUT_PTR(int8_t) orig_vec_out = vec_out;
            for (int pos1 = 0; pos1 < in_non_axis_prv.shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_non_axis_prv.shape[2]; pos2++) {
                    vec_in  = (MLI_PTR(int8_t))orig_vec_in  + POS(&in_non_axis_prv, 0, pos1, pos2, 0);
                    vec_out = orig_vec_out + POS(&out_non_axis_prv, 0, pos1, pos2, 0);
                    if (remaining_part) {
                        mli::krn::compute_leaky_relu(vec_in, vec_out, in_zp, identity_params, &alpha_params,
                                                remaining_part);
                        vec_in  += remaining_part;
                        vec_out += remaining_part;
                    }
#pragma clang loop pipeline(enable)
#pragma clang loop pipeline_options(0x10)
                    for (int pos3 = remaining_part; pos3 < in_non_axis_prv.shape[3]; pos3 += num_lanes) {
                        mli::krn::compute_leaky_relu(vec_in, vec_out, in_zp, identity_params, &alpha_params);
                        vec_in  += num_lanes;
                        vec_out += num_lanes;
                    }
                }
            }
        }
    }
}

static MLI_FORCE_INLINE void compute_prelu_no_broadcast(
        const MLI_PTR(int8_t) __restrict vec_in,
        MLI_OUT_PTR(int8_t) __restrict vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params,
        const generic_tensor_private_t<MLI_PTR(int8_t)> in_prv,
        const generic_tensor_private_t<MLI_OUT_PTR(int8_t)> out_prv,
        const int remaining_part) {
    /* Loop Over Sub Tensor */
    const MLI_PTR(int8_t) orig_vec_in = vec_in;
    MLI_OUT_PTR(int8_t) orig_vec_out = vec_out;
    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                vec_in  = (MLI_PTR(int8_t))orig_vec_in  + POS(&in_prv, pos0, pos1, pos2, 0);
                vec_out = orig_vec_out + POS(&out_prv, pos0, pos1, pos2, 0);
                if(remaining_part) {
                    mli::krn::compute_prelu(vec_in, vec_out, in_zp, identity_params, alpha_params, remaining_part);
                } else {
                    mli::krn::compute_prelu(vec_in, vec_out, in_zp, identity_params, alpha_params);
                }
            }
        }
    }
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_PRELU_VDSP_H_
