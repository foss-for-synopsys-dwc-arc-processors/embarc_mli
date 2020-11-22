/*
* Copyright 2020, Synopsys, Inc.
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

template <typename io_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const MLI_PTR(io_T) scale_in,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift) {
    io_T input = mli_prv_load_1vec(vec_in);
    io_T scale = mli_prv_load_1vec(scale_in);
    io_T zero = 0;
    /* out  = max(0, in) + alpha * min(0, in) */
    io_T pos = mli_math_max_fx(zero, input);
    io_T neg = mli_math_acc_cast_fx<io_T, mli_acc32_t>(
               mli_math_mul_fx<io_T, mli_acc32_t>( mli_math_min_fx(zero, input), scale), shift);
    io_T output = mli_math_add_fx(pos, neg);
    
    mli_prv_store_n_samples(vec_out, output);
}

template <typename io_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const MLI_PTR(io_T) scale_in,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift,
        const int remaining_part) {

    MLI_ASSERT(remaining_part == 1);
    compute_prelu<io_T>(vec_in, scale_in, vec_out, shift);
}

template <typename io_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const io_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift) {
    io_T input = mli_prv_load_1vec(vec_in);
    io_T zero = 0;
    /* out  = max(0, in) + alpha * min(0, in) */
    io_T pos = mli_math_max_fx(zero, input);
    io_T neg = mli_math_acc_cast_fx<io_T, mli_acc32_t>(
               mli_math_mul_fx<io_T, mli_acc32_t>( mli_math_min_fx(zero, input), scale), shift);
    io_T output = mli_math_add_fx(pos, neg);
    
    mli_prv_store_n_samples(vec_out, output);
}

template <typename io_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const io_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift,
        const int remaining_part) {

    MLI_ASSERT(remaining_part == 1);
    compute_prelu<io_T>(vec_in, scale, vec_out, shift);
}

template <typename io_T>
static MLI_FORCE_INLINE mli_status leaky_relu_fx_run(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        mli_tensor *out) {
    
    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(io_T) vec_in = nullptr;
    MLI_OUT_PTR(io_T) vec_out = nullptr;

    const MLI_PTR(io_T) in_ptr = (MLI_PTR(io_T))(in->data.mem.void_p);
    MLI_OUT_PTR(io_T) out_ptr = (MLI_OUT_PTR(io_T)) (out->data.mem.void_p);

    /* Copy tensor format */
    mli_prv_copy_tensor_format_except_mem_strides(in, out);
    /* Get Generic Private Tensor */
    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);    
    auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(io_T)>(out);

    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&in_prv );
    mli_prv_reorder_generic_tensor<MLI_OUT_PTR(io_T)>(&out_prv);

    /* Dummy Load to get num_lanes */
    auto input = mli_prv_load_1vec(in_ptr);
    int num_lanes = get_number_lanes(input);
    int remaining_part = in_prv.shape[3] & (num_lanes - 1);

    int shift = mli_prv_calc_shift(in, slope_coeff, out);

    io_T scale;
    // Getscalar value (casting or getting from memory)
    if (slope_coeff->rank == 0) {
        // value is stored in tensor`s field: analog of reinterpret_cast
        scale = mli_math_cast_ptr_to_scalar_fx<io_T>(slope_coeff->data.mem.void_p);
    } else {
        // pointer access to value
        scale = static_cast<io_T *>(slope_coeff->data.mem.void_p)[0];
    }

    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                vec_in  = (MLI_PTR(io_T))in_ptr  + POS(&in_prv, pos0, pos1, pos2, 0);
                vec_out = out_ptr + POS(&out_prv, pos0, pos1, pos2, 0);
                if (remaining_part) {
                    mli::krn::compute_prelu<io_T>(vec_in, scale, vec_out, 
                                                        shift, remaining_part);
                    vec_in  += remaining_part;
                    vec_out += remaining_part;
                }
                for (int pos3 = remaining_part; pos3 < in_prv.shape[3]; pos3 += num_lanes) {
                    mli::krn::compute_prelu<io_T>(vec_in, scale, vec_out, 
                                                    shift);
                    vec_in  += num_lanes;
                    vec_out += num_lanes;
                }
            }
        }
    }

    return MLI_STATUS_OK;
}

template <typename io_T>
static MLI_FORCE_INLINE mli_status prelu_fx_run(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        const mli_prelu_cfg *cfg, 
        mli_tensor *out) {

    /* Fall back to leaky_relu in case axis = -1 */
    if (cfg->axis == -1) {
        return mli::krn::leaky_relu_fx_run<io_T>(in, slope_coeff, out);
    }

    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(io_T) vec_in = nullptr;
    const MLI_PTR(io_T) scale_in = nullptr;
    MLI_OUT_PTR(io_T) vec_out = nullptr;

    const MLI_PTR(io_T) in_ptr = (MLI_PTR(io_T))(in->data.mem.void_p);
    const MLI_PTR(io_T) slope_ptr = (MLI_PTR(io_T))(slope_coeff->data.mem.void_p);
    MLI_OUT_PTR(io_T) out_ptr = (MLI_OUT_PTR(io_T)) (out->data.mem.void_p);

    /* Copy tensor format */
    mli_prv_copy_tensor_format_except_mem_strides(in, out);
    /* Get Generic Private Tensor */
    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);    
    auto slope_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(slope_coeff);
    auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(io_T)>(out);
    /* Get Non Axis Tensor */
    auto in_non_axis_prv  = mli_prv_get_non_axis_tensor<MLI_PTR(io_T)>(&in_prv,  cfg->axis);
    auto out_non_axis_prv = mli_prv_get_non_axis_tensor<MLI_OUT_PTR(io_T)>(&out_prv, cfg->axis);
    /* Get Axis Tensor */
    in_prv  = mli_prv_get_axis_tensor<MLI_PTR(io_T)>(&in_prv,  cfg->axis);
    out_prv = mli_prv_get_axis_tensor<MLI_OUT_PTR(io_T)>(&out_prv, cfg->axis);
    
    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&in_prv );
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&slope_prv );
    mli_prv_reorder_generic_tensor<MLI_OUT_PTR(io_T)>(&out_prv);

    /* Dummy Load to get num_lanes */
    auto input = mli_prv_load_1vec(in_ptr);
    int num_lanes = get_number_lanes(input);
    int remaining_part = in_prv.shape[3] & (num_lanes - 1);

    int shift = mli_prv_calc_shift(in, slope_coeff, out);

    /* For applying the function to specific axis dimension, we should first loop across other dimensions then process
    * axis dimension elements.
    * For applying the function to the whole tensor, loop body is executed only one time. (i.e. shape[i] = 1).
    */

    for (int dim0 = 0; dim0 < in_non_axis_prv.shape[0]; dim0++) {
        for (int dim1 = 0; dim1 < in_non_axis_prv.shape[1]; dim1++) {
            for (int dim2 = 0; dim2 < in_non_axis_prv.shape[2]; dim2++) {

                vec_in   = &in_ptr[dim0 * in_non_axis_prv.mem_stride[0] + 
                                dim1 * in_non_axis_prv.mem_stride[1] + 
                                dim2 * in_non_axis_prv.mem_stride[2]];
                
                vec_out  = &out_ptr[dim0 * out_non_axis_prv.mem_stride[0] + 
                                    dim1 * out_non_axis_prv.mem_stride[1] + 
                                    dim2 * out_non_axis_prv.mem_stride[2]];
                
                const MLI_PTR(io_T) orig_vec_in = vec_in;
                MLI_OUT_PTR(io_T) orig_vec_out = vec_out;
                for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
                    for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                        for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                            vec_in  = (MLI_PTR(io_T))orig_vec_in  + POS(&in_prv, pos0, pos1, pos2, 0);
                            scale_in  = (MLI_PTR(io_T))slope_ptr + POS(&slope_prv, pos0, pos1, pos2, 0);
                            vec_out = orig_vec_out + POS(&out_prv, pos0, pos1, pos2, 0);
                            if (remaining_part) {
                                mli::krn::compute_prelu<io_T>(vec_in, scale_in, vec_out, 
                                                                    shift, remaining_part);
                                vec_in  += remaining_part;
                                scale_in += remaining_part;
                                vec_out += remaining_part;
                            }
                            for (int pos3 = remaining_part; pos3 < in_prv.shape[3]; pos3 += num_lanes) {
                                mli::krn::compute_prelu<io_T>(vec_in, scale_in, vec_out, 
                                                                shift);
                                vec_in  += num_lanes;
                                scale_in += num_lanes;
                                vec_out += num_lanes;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return MLI_STATUS_OK;
}

struct prelu_sa8_requant_params {
    s8asym_quant_params identity_params;
    s8asym_quant_params alpha_params;
};

static MLI_FORCE_INLINE void prelu_define_requant_params(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        mli_tensor *out,
        int8_t alpha_sa8,
        prelu_sa8_requant_params *params) {
    
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
    
    int scale_alpha_shift  = params->identity_params.shift;
        scale_alpha_shift += slope_coeff->el_params.sa.scale_frac_bits.mem.i8;
        scale_alpha_shift -= norm_shift;
    
    int16_t scale_alpha = mli_math_norm_cast_fx<int32_t,int16_t>(
                          mli_math_mul_fx<int16_t, int32_t>(params->identity_params.scale, alpha), &norm_shift);
    scale_alpha_shift -= norm_shift;

    int16_t in_zp  = in->el_params.sa.zero_point.mem.i16;
    int16_t out_zp = out->el_params.sa.zero_point.mem.i16;
    
    int16_t scale_alpha_offset = mli_math_sub_fx<int16_t>(out_zp,
                                 mli_math_cast_fx<int32_t, int16_t>(
                                 mli_math_mul_fx<int16_t, int32_t>(scale_alpha, in_zp), scale_alpha_shift));
    
    /* Define Quantization params for (In * alpha / out) ratio */
    params->alpha_params.scale  = scale_alpha;
    params->alpha_params.shift  = scale_alpha_shift;
    params->alpha_params.offset = scale_alpha_offset;
}

static MLI_FORCE_INLINE mli_status leaky_relu_sa8_run(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        mli_tensor *out) {

    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(int8_t) vec_in = (MLI_PTR(int8_t))(in->data.mem.void_p);
    MLI_OUT_PTR(int8_t) vec_out = (MLI_OUT_PTR(int8_t)) (out->data.mem.void_p);

    /* Copy tensor format */
    for (int idx = 0; idx < (int)in->rank; idx++) {
        out->shape[idx] = in->shape[idx];
    }
    out->rank = in->rank;
    out->el_type = in->el_type;

    /* Get Generic Private Tensor */
    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(int8_t)>(in);
    auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(int8_t)>(out);
    
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
    prelu_sa8_requant_params params;
    /* Define Requantization Params for In/Out scale ratio */
    define_requant_params(in, out, &params.identity_params);
    int8_t alpha_sa8;
    if (slope_coeff->rank == 0) {
        alpha_sa8 = slope_coeff->data.mem.i8;
    } else {
        alpha_sa8 = slope_coeff->data.mem.pi8[0];
    }
    prelu_define_requant_params(in, slope_coeff, out, alpha_sa8, &params);

    /* Input Zero Point */
    int16_t in_zp = in->el_params.sa.zero_point.mem.i16;

    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                for (int pos3 = 0; pos3 < in_prv.shape[3]; pos3++) {
                    /* Load Input */
                    int8_t input = vec_in[POS(&in_prv, pos0, pos1, pos2, pos3)];
                    int16_t output;
                    if (input >= in_zp) {
                    /* out_sa8 = (idendity_scale * in_sa8) * 2^(-(identity_shift)) + identity_offset */
                    output =  mli_math_add_fx<int16_t>(
                            mli_math_cast_fx<int32_t, int16_t>(
                            mli_math_mul_fx<int16_t, int32_t>(params.identity_params.scale, input), 
                            params.identity_params.shift), params.identity_params.offset);
                    } else {
                        /* out_sa8 = (alpha_scale * in_sa8) * 2^(-(alpha_shift)) + alpha_offset */
                        output =  mli_math_add_fx<int16_t>(
                            mli_math_cast_fx<int32_t, int16_t>(
                            mli_math_mul_fx<int16_t, int32_t>(params.alpha_params.scale, input), 
                            params.alpha_params.shift), params.alpha_params.offset);
                    }
                    /* Store Output */
                    vec_out[POS(&out_prv, pos0, pos1, pos2, pos3)] = mli_math_cast_fx<int16_t, int8_t>(output, 0);
                }
            }
        }
    }

    return MLI_STATUS_OK;
}

static MLI_FORCE_INLINE mli_status prelu_sa8_run(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        const mli_prelu_cfg *cfg, 
        mli_tensor *out) {

    /* Fall back to leaky_relu in case axis = -1 */
    if (cfg->axis == -1) {
        return mli::krn::leaky_relu_sa8_run(in, slope_coeff, out);
    }

    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(int8_t) vec_in = nullptr;
    const MLI_PTR(int8_t) scale_in = (MLI_PTR(int8_t))(slope_coeff->data.mem.void_p);
    MLI_OUT_PTR(int8_t) vec_out = nullptr;

    const MLI_PTR(int8_t) in_ptr = (MLI_PTR(int8_t))(in->data.mem.void_p);
    MLI_OUT_PTR(int8_t) out_ptr = (MLI_OUT_PTR(int8_t)) (out->data.mem.void_p);

    /* Copy tensor format */
    for (int idx = 0; idx < (int)in->rank; idx++) {
        out->shape[idx] = in->shape[idx];
    }
    out->rank = in->rank;
    out->el_type = in->el_type;

    /* Get Generic Private Tensor */
    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(int8_t)>(in);
    auto slope_prv =  mli_prv_get_generic_tensor<MLI_PTR(int8_t)>(slope_coeff);
    auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(int8_t)>(out);

    /* Get Non Axis Tensor */
    auto in_non_axis_prv  = mli_prv_get_non_axis_tensor<MLI_PTR(int8_t)>(&in_prv,  cfg->axis);
    auto out_non_axis_prv = mli_prv_get_non_axis_tensor<MLI_OUT_PTR(int8_t)>(&out_prv, cfg->axis);
    /* Get Axis Tensor */
    in_prv  = mli_prv_get_axis_tensor<MLI_PTR(int8_t)>(&in_prv,  cfg->axis);
    out_prv = mli_prv_get_axis_tensor<MLI_OUT_PTR(int8_t)>(&out_prv, cfg->axis);
    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(int8_t)>(&in_prv );
    mli_prv_reorder_generic_tensor<MLI_PTR(int8_t)>(&slope_prv );
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
    prelu_sa8_requant_params params;
    /* Define Requantization Params for In/Out scale ratio */
    define_requant_params(in, out, &params.identity_params);

    /* Input Zero Point */
    int16_t in_zp = in->el_params.sa.zero_point.mem.i16;

    for (int dim0 = 0; dim0 < in_non_axis_prv.shape[0]; dim0++) {
        for (int dim1 = 0; dim1 < in_non_axis_prv.shape[1]; dim1++) {
            for (int dim2 = 0; dim2 < in_non_axis_prv.shape[2]; dim2++) {

                vec_in = &in_ptr[dim0 * in_non_axis_prv.mem_stride[0] + 
                                dim1 * in_non_axis_prv.mem_stride[1] + 
                                dim2 * in_non_axis_prv.mem_stride[2]];
                vec_out = &out_ptr[dim0 * out_non_axis_prv.mem_stride[0] + 
                                dim1 * out_non_axis_prv.mem_stride[1] + 
                                dim2 * out_non_axis_prv.mem_stride[2]];

                for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
                    for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                        for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                            for (int pos3 = 0; pos3 < in_prv.shape[3]; pos3++) {
                                /* Load Input */
                                int8_t input = vec_in[POS(&in_prv, pos0, pos1, pos2, pos3)];
                                int8_t alpha_sa8 = scale_in[POS(&slope_prv, pos0, pos1, pos2, pos3)];
                                prelu_define_requant_params(in, slope_coeff, out, alpha_sa8, &params);
                                int16_t output;
                                if (input >= in_zp) {
                                /* out_sa8 = (idendity_scale * in_sa8) * 2^(-(identity_shift)) + identity_offset */
                                output =  mli_math_add_fx<int16_t>(
                                        mli_math_cast_fx<int32_t, int16_t>(
                                        mli_math_mul_fx<int16_t, int32_t>(params.identity_params.scale, input), 
                                        params.identity_params.shift), params.identity_params.offset);
                                } else {
                                    /* out_sa8 = (alpha_scale * in_sa8) * 2^(-(alpha_shift)) + alpha_offset */
                                    output =  mli_math_add_fx<int16_t>(
                                        mli_math_cast_fx<int32_t, int16_t>(
                                        mli_math_mul_fx<int16_t, int32_t>(params.alpha_params.scale, input), 
                                        params.alpha_params.shift), params.alpha_params.offset);
                                }
                                /* Store Output */
                                vec_out[POS(&out_prv, pos0, pos1, pos2, pos3)] = mli_math_cast_fx<int16_t, int8_t>(output, 0);
                            }
                        }
                    }
                }
            }
        }
    }

    return MLI_STATUS_OK;
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_PRELU_REF_H_