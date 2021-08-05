/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_LEAKY_RELU_REF_H_
#define _MLI_KRN_LEAKY_RELU_REF_H_

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_mem_info.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace ref {

template <typename io_T>
static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift) {
    io_T input = vec_in[0];
    io_T zero = 0;
    /* out  = max(0, in) + alpha * min(0, in) */
    io_T pos = mli_math_max_fx(zero, input);
    io_T neg = mli_math_acc_cast_fx<io_T, mli_acc32_t>(
               mli_math_mul_fx<io_T, mli_acc32_t>( mli_math_min_fx(zero, input), scale), shift);
    vec_out[0] = mli_math_add_fx(pos, neg);
}

template <typename io_T>
static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift,
        const int remaining_part) {

    MLI_ASSERT(remaining_part == 1);
    compute_leaky_relu<io_T>(vec_in, vec_out, scale, shift);
}

static MLI_FORCE_INLINE void compute_leaky_relu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params) {

    /* Load Input */
    int8_t input = vec_in[0];
    int32_t output;
    if (input >= in_zp) {
        /* out_sa8 = (idendity_scale * (in_sa8 - in_zp)) * 2^(-(identity_shift)) + identity_offset */
        int16_t input_sub = mli_math_sub_fx((int16_t)input, in_zp);
        output = mli_math_asr_rnd_fx(
                 mli_math_mul_fx<int16_t, int32_t>(identity_params->scale, input_sub), identity_params->shift);
        output = mli_math_add_fx(output, (int32_t)identity_params->offset);
    } else {
        /* out_sa8 = (alpha_scale * (in_sa8 - in_zp)) * 2^(-(alpha_shift)) + alpha_offset */
        int16_t input_sub = mli_math_sub_fx((int16_t)input, in_zp);
        output = mli_math_asr_rnd_fx(
                 mli_math_mul_fx<int16_t, int32_t>(alpha_params->scale, input_sub), alpha_params->shift);
        output = mli_math_add_fx(output, (int32_t)alpha_params->offset);
    }

    vec_out[0] = mli_math_cast_fx<int32_t, int8_t>(output, 0);
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
        mli::krn::compute_leaky_relu<io_T>(vec_in, vec_out, scale,
                                            shift, remaining_part);
        vec_in  += remaining_part;
        vec_out += remaining_part;
    }
    for (int pos3 = remaining_part; pos3 < count; pos3 += num_lanes) {
        mli::krn::compute_leaky_relu<io_T>(vec_in, vec_out, scale, shift);
        vec_in  += num_lanes;
        vec_out += num_lanes;
    }
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
        mli::krn::compute_leaky_relu(vec_in, vec_out, in_zp, identity_params, alpha_params, remaining_part);
        vec_in  += remaining_part;
        vec_out += remaining_part;
    }

    for (int pos3 = remaining_part; pos3 < count; pos3 += num_lanes) {
        mli::krn::compute_leaky_relu(vec_in, vec_out, in_zp, identity_params, alpha_params);
        vec_in  += num_lanes;
        vec_out += num_lanes;
    }
}

template <typename io_T>
static MLI_FORCE_INLINE mli_status leaky_relu_fx_run(const mli_tensor *in,
        const mli_tensor *slope_coeff,
        mli_tensor *out) {

    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(io_T) __restrict in_ptr = mli_prv_tensor_data_ptr<MLI_PTR(io_T)>(in);
    MLI_OUT_PTR(io_T) __restrict out_ptr  = mli_prv_tensor_data_ptr<MLI_OUT_PTR(io_T)>(out);

    /* Dummy Load to get num_lanes */
    auto input = mli_prv_load_1vec(in_ptr);
    int num_lanes = get_number_lanes(input);
    int remaining_part = 0;

    int shift = mli_prv_calc_shift(in, slope_coeff, out);
    io_T scale = mli_prv_tensor_data_val<io_T>(slope_coeff);

    int shift_val = shift;
    if (std::is_same<io_T, int16_t>::value) {
        /* Normalization is needed for int16_t as we use mul_hi */
        int norm_shift;
        scale = mli_math_norm_cast_fx<io_T,io_T>(scale, &norm_shift);
        shift_val -= norm_shift;
    }

    /* Trying to squash tensor to one dim */
    int shape = mli_prv_squash_tensor_to_one_dim(in, out);
    if (shape) {
        remaining_part = shape & (num_lanes - 1);
        mli::krn::compute_leaky_relu_fx_inner_loop(in_ptr, out_ptr, scale, 
                                shift_val, shape, remaining_part);
    } else {
        /* Get Generic Private Tensor */
        auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);
        auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(io_T)>(out);
        /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
        mli_prv_squash_generic_tensor<MLI_PTR(io_T)>(&in_prv, &out_prv);

        remaining_part = in_prv.shape[3] & (num_lanes - 1);
        /* Loop Over Sub Tensor */
        const MLI_PTR(io_T) __restrict orig_vec_in = in_ptr;
        MLI_OUT_PTR(io_T) __restrict orig_vec_out = out_ptr;
        for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                    in_ptr  = (MLI_PTR(io_T))orig_vec_in  + POS(&in_prv, pos0, pos1, pos2, 0);
                    out_ptr = orig_vec_out + POS(&out_prv, pos0, pos1, pos2, 0);
                    mli::krn::compute_leaky_relu_fx_inner_loop(in_ptr, out_ptr, scale, 
                                                          shift_val, in_prv.shape[3], remaining_part);
                }
            }
        }
    }

    return MLI_STATUS_OK;
}

static MLI_FORCE_INLINE void leaky_relu_define_identity_params(const mli_tensor *in, const mli_tensor *out,
        s8asym_quant_params *params) {

    /* ****************************************************************************************************************
     *             Mathematical Derivations out_sa8 Requantization Params to use with in_sa8
     * ----------------------------------------------------------------------------------------------------------------
     *      out_sa8 = (in_scale_val/out_scale_val) *(in_sa8 - in_zp) + out_zp
     *              = scale_val * (in_sa8 - in_zp) + out_zp
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
     *      offset = out_zp
     *
     * ***************************************************************************************************************/

    int16_t scale_in = mli_hlp_tensor_scale(in, 0);
    int16_t scale_out = mli_hlp_tensor_scale(out, 0);
    int16_t out_zp = out->el_params.sa.zero_point.mem.i16;
    int kPreDivShift =  mli_math_norm_fx<int32_t, int32_t>(scale_in) -
                            mli_math_norm_fx<int16_t, int32_t>(scale_out);
    /* Normalize In/Out Scale ratio and cast to 16bit */
    int norm_shift;
    params->scale = mli_math_norm_cast_fx<int32_t, int16_t>(
                    ((int32_t)(scale_in) << kPreDivShift) /
                    scale_out, &norm_shift);

    params->shift  = kPreDivShift;
    params->shift += mli_hlp_tensor_scale_shift(in, 0) - mli_hlp_tensor_scale_shift(out, 0);
    params->shift -= norm_shift;
    params->offset = out_zp;
    int shift_left = mli_math_max_fx(-params->shift, 0);
    params->scale = mli_math_asl_fx(params->scale, shift_left);
    params->shift = mli_math_max_fx(params->shift, 0);


}
static MLI_FORCE_INLINE s8asym_quant_params leaky_relu_define_alpha_params(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        mli_tensor *out,
        const int8_t alpha_sa8,
        const s8asym_quant_params *identity_params) {
    
   /* ****************************************************************************************************************
     *             Mathematical Derivations out_sa8 Requantization Params with alpha scale to use with in_sa8
     * ----------------------------------------------------------------------------------------------------------------
     *      First we need to define Quantization Params for In/Out 
     *          out_sa8 = (in_scale_val/out_scale_val) *(in_sa8 - in_zp) + out_zp
     *      where we define scale, scale_shift and offset modified values
     *          check -> define_quant_params(const mli_tensor *in, const mli_tensor *out, s8asym_quant_params *params)
     *          for Documentation
     *      then :
     * 
     *      out_sa8 = (in_scale_val/out_scale_val) * alpha_scale * (alpha_sa8 - alpha_zp) * (in_sa8 - in_zp) + out_zp
     *              = scale_val * alpha_val * (alpha_sa8 - alpha_zp) * (in_sa8 - in_zp) + out_zp
     *              = scale_alpha_val * (in_sa8 - in_zp) + out_zp
     * 
     *      For scale_alpha_val = scale_val * alpha_val * (alpha_sa8 - alpha_zp) 
     *                        = scale * 2 ^(-(scale_shift)) * alpha_scale 
     *                          * 2^(-alpha_scale_frac_bits) * (alpha_sa8 - alpha_zp)
     *                        = scale * alpha_scale * (alpha_sa8 - alpha_zp) 
     *                          * 2^(-(scale_shift + alpha_scale_frac_bits))
     *                        = scale * alpha_scale * (alpha_sa8 - alpha_zp) * 2^(-(alpha_norm_shift))
     *                          * 2^(-(scale_shift + alpha_scale_frac_bits - alpha_norm_shift))
     *                        = scale * alpha
     *                          * 2^(-(scale_shift + alpha_scale_frac_bits - alpha_norm_shift))
     *                        = scale * alpha * 2^(-(scale_mul_norm_shift))
     *                          * 2^(-(scale_shift + alpha_scale_frac_bits - alpha_norm_shift - scale_mul_norm_shift))
     *                        = scale_alpha * 2^(-(scale_alpha_shift))
     * 
     *      where alpha = alpha_scale * (alpha_sa8 - alpha_zp) * 2^(-(alpha_norm_shift))
     *            alpha_norm_shift is the shift value due to normalizing the result of 
     *                             alpha_scale * (alpha_sa8 - alpha_zp) and casting it from int32_t to int16_t
     *            scale_alpha = scale * alpha * 2^(-(scale_mul_norm_shift))
     *            scale_mul_norm_shift is the shift value due to normalizing the result of 
     *                             scale * alpha_scale * (alpha_sa8 - alpha_zp) * 2^(-(alpha_norm_shift))
     *            scale_alpha_shift = scale_shift + alpha_scale_frac_bits - alpha_norm_shift - scale_mul_norm_shift
     * 
     *      scale_alpha_offset = out_zp
     * 
     * ***************************************************************************************************************/
    int16_t out_zp = out->el_params.sa.zero_point.mem.i16;

    int32_t alpha_val = mli_prv_convert_sa8_fx16<int8_t, int32_t>(alpha_sa8, 
                            slope_coeff->el_params.sa.zero_point.mem.i16,
                            slope_coeff->el_params.sa.scale.mem.i16);
    /* Normalize alpha and cast to 16bit */
    int norm_shift;
    int16_t alpha = mli_math_norm_cast_fx<int32_t,int16_t>(alpha_val, &norm_shift);
    
    int scale_alpha_shift  = identity_params->shift;
        scale_alpha_shift += slope_coeff->el_params.sa.scale_frac_bits.mem.i8;
        scale_alpha_shift -= norm_shift;
    
    int16_t scale_alpha = mli_math_norm_cast_fx<int32_t,int16_t>(
                          mli_math_mul_fx<int16_t, int32_t>(identity_params->scale, alpha), &norm_shift);
    scale_alpha_shift -= norm_shift;
    
    /* Define Quantization params for (In * alpha / out) ratio */
    s8asym_quant_params alpha_params;
    alpha_params.scale  = scale_alpha;
    alpha_params.shift  = scale_alpha_shift;
    alpha_params.offset = out_zp;
    return alpha_params;
}

static MLI_FORCE_INLINE mli_status leaky_relu_sa8_run(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        mli_tensor *out) {

    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(int8_t) in_ptr = mli_prv_tensor_data_ptr<MLI_PTR(int8_t)>(in);
    MLI_OUT_PTR(int8_t) out_ptr = mli_prv_tensor_data_ptr<MLI_OUT_PTR(int8_t)>(out);

    /* Input Zero Point */
    int16_t in_zp = in->el_params.sa.zero_point.mem.i16;
    int8_t scale = mli_prv_tensor_data_val<int8_t>(slope_coeff);

    /* ****************************************************************************************************************
     *                        Mathematical Derivations for Leaky RELU SA8 
     * ----------------------------------------------------------------------------------------------------------------
     *    If (in_sa8 >= in_zp)
     *       out_sa8 = (idendity_scale * (in_sa8 - in_zp)) * 2^(-(identity_shift)) + identity_offset; 
     *    else
     *       out_sa8 = (alpha_scale * (in_sa8 - in_zp)) * 2^(-(alpha_shift)) + alpha_offset;
     * 
     *    check leaky_relu_define_alpha_params for more Documentation
     * ***************************************************************************************************************/
    s8asym_quant_params identity_params;
    /* Define Requantization Params for In/Out scale ratio */
    leaky_relu_define_identity_params(in, out, &identity_params);
    s8asym_quant_params alpha_params = leaky_relu_define_alpha_params(in, slope_coeff, out, scale, &identity_params);

    /* Dummy Load to get num_lanes */
    auto input = mli_prv_load_1vec(in_ptr);
    int num_lanes = get_number_lanes(input);
    int remaining_part = 0;

    /* Trying to squash tensor to one dim */
    int shape = mli_prv_squash_tensor_to_one_dim(in, out);
    if (shape) {
        remaining_part = shape & (num_lanes - 1);
        mli::krn::compute_leaky_relu_sa8_inner_loop(in_ptr, out_ptr, in_zp, &identity_params, &alpha_params,
                                                   shape, remaining_part);
    } else {
        /* Get Generic Private Tensor */
        auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(int8_t)>(in);
        auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(int8_t)>(out);
        /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
        mli_prv_squash_generic_tensor<MLI_PTR(int8_t)>(&in_prv, &out_prv);

        remaining_part = in_prv.shape[3] & (num_lanes - 1);
        /* Loop Over Sub Tensor */
        const MLI_PTR(int8_t) __restrict orig_vec_in = in_ptr;
        MLI_OUT_PTR(int8_t) __restrict orig_vec_out = out_ptr;
        for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                    in_ptr  = (MLI_PTR(int8_t))orig_vec_in  + POS(&in_prv, pos0, pos1, pos2, 0);
                    out_ptr = orig_vec_out + POS(&out_prv, pos0, pos1, pos2, 0);
                    mli::krn::compute_leaky_relu_sa8_inner_loop(in_ptr, out_ptr, in_zp, &identity_params, &alpha_params,
                                                               in_prv.shape[3], remaining_part);
                }
            }
        }
    }

    return MLI_STATUS_OK;
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_LEAKY_RELU_REF_H_
