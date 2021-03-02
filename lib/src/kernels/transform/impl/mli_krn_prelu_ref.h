/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_PRELU_REF_H_
#define _MLI_KRN_PRELU_REF_H_

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace ref {

template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift) {
    io_T input = vec_in[0];
    io_T zero = 0;
    /* out  = max(0, in) + alpha * min(0, in) */
    io_T pos = mli_math_max_fx(zero, input);
    io_T neg = mli_math_acc_cast_fx<io_T, mli_acc32_t>(
               mli_math_mul_fx<io_T, mli_acc32_t>( mli_math_min_fx(zero, input), scale), shift);
    vec_out[0] = mli_math_add_fx(pos, neg);
}

template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift,
        const int remaining_part) {

    MLI_ASSERT(remaining_part == 1);
    compute_prelu<io_T, scale_T>(vec_in, scale, vec_out, shift);
}

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params) {

    /* Load Input */
    int8_t input = vec_in[0];
    int16_t output;
    if (input >= in_zp) {
        /* out_sa8 = (idendity_scale * in_sa8) * 2^(-(identity_shift)) + identity_offset */
        output =  mli_math_add_fx<int16_t>(
                  mli_math_cast_fx<int32_t, int16_t>(
                  mli_math_mul_fx<int16_t, int32_t>(identity_params->scale, input), 
                  identity_params->shift), identity_params->offset);
    } else {
        /* out_sa8 = (alpha_scale * in_sa8) * 2^(-(alpha_shift)) + alpha_offset */
        output =  mli_math_add_fx<int16_t>(
                  mli_math_cast_fx<int32_t, int16_t>(
                  mli_math_mul_fx<int16_t, int32_t>(alpha_params->scale, input), 
                  alpha_params->shift), alpha_params->offset);
    }
    
    vec_out[0] = mli_math_cast_fx<int16_t, int8_t>(output, 0);
}

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params,
        const int remaining_part) {

    MLI_ASSERT(remaining_part == 1);
    compute_prelu(vec_in, vec_out, in_zp, identity_params, alpha_params);
}

template <typename io_T>
static MLI_FORCE_INLINE mli_status prelu_fx_run(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        const mli_prelu_cfg *cfg, 
        mli_tensor *out) {

    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(io_T) vec_in = nullptr;
    MLI_OUT_PTR(io_T) vec_out = nullptr;

    const MLI_PTR(io_T) in_ptr = mli_prv_tensor_data_ptr<MLI_PTR(io_T)>(in);
    const MLI_PTR(io_T) slope_ptr = nullptr;
    MLI_OUT_PTR(io_T) out_ptr = mli_prv_tensor_data_ptr<MLI_OUT_PTR(io_T)>(out);

    /* Copy tensor format */
    mli_prv_copy_tensor_format_except_mem_strides(in, out);
    /* Get Generic Private Tensor */
    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);    
    auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(io_T)>(out);

    /* Define Slope Axis Params */
    int axis_shape = 1;
    int axis_in_mem_stride = 0;
    int axis_out_mem_stride = 0;
    bool broadcasting = true;
    bool is_leaky_relu = (cfg->axis == -1);
    io_T scale;
    if (is_leaky_relu) {
        // Getscalar value
        scale = mli_prv_tensor_data_val<io_T>(slope_coeff);
    } else {
        /* Broadcasting in case axis is not inner most dim */
        broadcasting = !(cfg->axis == (in_prv.rank - 1));
        in_prv.shape[cfg->axis] = 1;
        axis_shape = slope_coeff->shape[cfg->axis];
        axis_in_mem_stride = in_prv.mem_stride[cfg->axis];
        axis_out_mem_stride = out_prv.mem_stride[cfg->axis];
        slope_ptr = mli_prv_tensor_data_ptr<MLI_PTR(io_T)>(slope_coeff);
    }

    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&in_prv );
    mli_prv_reorder_generic_tensor<MLI_OUT_PTR(io_T)>(&out_prv);

    /* Dummy Load to get num_lanes */
    auto input = mli_prv_load_1vec(in_ptr);
    int num_lanes = get_number_lanes(input);
    /* use out_prv instead of in_prv shape[3] as it's modified */
    int remaining_part = out_prv.shape[3] & (num_lanes - 1);

    int shift = mli_prv_calc_shift(in, slope_coeff, out);

    for (int scale_idx = 0; scale_idx < axis_shape; ) {
        
        /* Define Sub Tensor */
        vec_in  = (MLI_PTR(io_T))in_ptr  + scale_idx * axis_in_mem_stride;
        vec_out = out_ptr + scale_idx * axis_out_mem_stride;

        decltype(input) scale_v;
        if (broadcasting) {
            /* Load Scale Elem */
            scale_v = mli_prv_init_v<io_T, decltype(input)>((is_leaky_relu)? scale: slope_ptr[scale_idx]);
            scale_idx++;
        } else {
            /* Load Scale Vector */
            scale_v = mli_prv_load_1vec(&slope_ptr[scale_idx]);
            if (remaining_part) {
                scale_idx += remaining_part;
            } else {
                scale_idx += num_lanes;
            }
        }

        /* Loop Over Sub Tensor */
        const MLI_PTR(io_T) orig_vec_in = vec_in;
        MLI_OUT_PTR(io_T) orig_vec_out = vec_out;
        for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                    vec_in  = (MLI_PTR(io_T))orig_vec_in  + POS(&in_prv, pos0, pos1, pos2, 0);
                    vec_out = orig_vec_out + POS(&out_prv, pos0, pos1, pos2, 0);
                    if (remaining_part) {
                        mli::krn::compute_prelu<io_T, decltype(input)>(vec_in, scale_v, vec_out,
                                                            shift, remaining_part);
                        vec_in  += remaining_part;
                        vec_out += remaining_part;
                    }
                    for (int pos3 = remaining_part; pos3 < in_prv.shape[3]; pos3 += num_lanes) {
                        mli::krn::compute_prelu<io_T, decltype(input)>(vec_in, scale_v, vec_out, shift);
                        vec_in  += num_lanes;
                        vec_out += num_lanes;
                    }
                }
            }
        }

        if(!broadcasting) {
            /* In case of No Broadcasting, all remaining parts are handled in the first iteration of scale */
            remaining_part = 0;
        }
    }
    
    return MLI_STATUS_OK;
}

static MLI_FORCE_INLINE s8asym_quant_params prelu_define_requant_params(const mli_tensor *in, 
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
     *              = scale_alpha_val * in_sa8 + out_zp - scale_alpha_val * in_zp
     *              = scale_alpha_val * in_sa8 + scale_alpha_offset;
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
     *      scale_alpha_offset = out_zp - scale_alpha_val * in_zp
     *                         = out_zp - (scale_alpha * in_zp) * 2^(-(scale_alpha_shift));
     * 
     * ***************************************************************************************************************/

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

    int16_t in_zp  = in->el_params.sa.zero_point.mem.i16;
    int16_t out_zp = out->el_params.sa.zero_point.mem.i16;
    
    int16_t scale_alpha_offset = mli_math_sub_fx<int16_t>(out_zp,
                                 mli_math_cast_fx<int32_t, int16_t>(
                                 mli_math_mul_fx<int16_t, int32_t>(scale_alpha, in_zp), scale_alpha_shift));
    
    /* Define Quantization params for (In * alpha / out) ratio */
    s8asym_quant_params alpha_params;
    alpha_params.scale  = scale_alpha;
    alpha_params.shift  = scale_alpha_shift;
    alpha_params.offset = scale_alpha_offset;
    return alpha_params;
}

static MLI_FORCE_INLINE mli_status prelu_sa8_run(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        const mli_prelu_cfg *cfg, 
        mli_tensor *out) {

    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(int8_t) vec_in = nullptr;
    MLI_OUT_PTR(int8_t) vec_out = nullptr;

    const MLI_PTR(int8_t) in_ptr = mli_prv_tensor_data_ptr<MLI_PTR(int8_t)>(in);
    const MLI_PTR(int8_t) slope_ptr = nullptr;
    MLI_OUT_PTR(int8_t) out_ptr = mli_prv_tensor_data_ptr<MLI_OUT_PTR(int8_t)>(out);

    /* Copy tensor format */
    for (int idx = 0; idx < (int)in->rank; idx++) {
        out->shape[idx] = in->shape[idx];
    }
    out->rank = in->rank;
    out->el_type = in->el_type;

    /* Get Generic Private Tensor */
    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(int8_t)>(in);
    auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(int8_t)>(out);

    /* Define Slope Axis Params */
    int axis_shape = 1;
    int axis_in_mem_stride = 0;
    int axis_out_mem_stride = 0;
    bool broadcasting = true;
    bool is_leaky_relu = (cfg->axis == -1);
    int8_t scale;
    if (is_leaky_relu) {
        scale = mli_prv_tensor_data_val<int8_t>(slope_coeff);
    } else {
        /* Broadcasting in case axis is not inner most dim */
        broadcasting = !(cfg->axis == (in_prv.rank - 1));
        in_prv.shape[cfg->axis] = 1;
        axis_shape = slope_coeff->shape[cfg->axis];
        axis_in_mem_stride = in_prv.mem_stride[cfg->axis];
        axis_out_mem_stride = out_prv.mem_stride[cfg->axis];
        slope_ptr = mli_prv_tensor_data_ptr<MLI_PTR(int8_t)>(slope_coeff);
    }

    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(int8_t)>(&in_prv );
    mli_prv_reorder_generic_tensor<MLI_OUT_PTR(int8_t)>(&out_prv);

    /* ****************************************************************************************************************
     *                        Mathematical Derivations for Leaky RELU SA8 
     * ----------------------------------------------------------------------------------------------------------------
     *    If (in_sa8 >= in_zp)
     *       out_sa8 = (idendity_scale * in_sa8) * 2^(-(identity_shift)) + identity_offset; 
     *    else
     *       out_sa8 = (alpha_scale * in_sa8) * 2^(-(alpha_shift)) + alpha_offset;
     * 
     *    check prelu_define_requant_params for more Documentation
     * ***************************************************************************************************************/
    s8asym_quant_params identity_params;
    /* Define Requantization Params for In/Out scale ratio */
    define_requant_params(in, out, &identity_params);

    /* Dummy Load to get num_lanes */
    auto input = mli_prv_load_1vec(in_ptr);
    int num_lanes = get_number_lanes(input);
    /* use out_prv instead of in_prv shape[3] as it's modified */
    int remaining_part = out_prv.shape[3] & (num_lanes - 1);

    /* Input Zero Point */
    int16_t in_zp = in->el_params.sa.zero_point.mem.i16;

    for (int scale_idx = 0; scale_idx < axis_shape; ) {
        
        /* Define Sub Tensor */
        vec_in  = (MLI_PTR(int8_t))in_ptr  + scale_idx * axis_in_mem_stride;
        vec_out = out_ptr + scale_idx * axis_out_mem_stride;

        decltype(input) scale_v;
        if (broadcasting) {
            /* Load Scale Elem */
            scale_v = mli_prv_init_v<int8_t, decltype(input)>((is_leaky_relu)? scale: slope_ptr[scale_idx]);
            scale_idx++;
        } else {
            /* Load Scale Vector */
            scale_v = mli_prv_load_1vec(&slope_ptr[scale_idx]);
            if (remaining_part) {
                scale_idx += remaining_part;
            } else {
                scale_idx += num_lanes;
            }
        }

        auto alpha_params = mli::krn::prelu_define_requant_params(in, slope_coeff, out, scale_v, &identity_params);
        
        /* Loop Over Sub Tensor */
        const MLI_PTR(int8_t) orig_vec_in = vec_in;
        MLI_OUT_PTR(int8_t) orig_vec_out = vec_out;
        for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                    vec_in  = (MLI_PTR(int8_t))orig_vec_in  + POS(&in_prv, pos0, pos1, pos2, 0);
                    vec_out = orig_vec_out + POS(&out_prv, pos0, pos1, pos2, 0);
                    if (remaining_part) {
                        mli::krn::compute_prelu(vec_in, vec_out, in_zp, &identity_params, &alpha_params, remaining_part);
                        vec_in  += remaining_part;
                        vec_out += remaining_part;
                    }

                    for (int pos3 = remaining_part; pos3 < in_prv.shape[3]; pos3 += num_lanes) {
                        mli::krn::compute_prelu(vec_in, vec_out, in_zp, &identity_params, &alpha_params);
                        vec_in  += num_lanes;
                        vec_out += num_lanes;
                    }
                }
            }
        }

        if(!broadcasting) {
            /* In case of No Broadcasting, all remaining parts are handled in the first iteration of scale */
            remaining_part = 0;
        }
    }

    return MLI_STATUS_OK;
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_PRELU_REF_H_