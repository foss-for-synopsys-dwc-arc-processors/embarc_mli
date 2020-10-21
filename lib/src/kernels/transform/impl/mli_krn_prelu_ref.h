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

template <typename io_T, prelu_elem_func_type func_type>
static MLI_FORCE_INLINE void mli_krn_scale_elem_v(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift) {
    io_T input = mli_prv_load_1vec(vec_in);
    io_T output;
    if (func_type == PRELU_ELEM_FUNC_MAX) {
        output = mli_math_max_fx(input, mli_math_acc_cast_fx<io_T, mli_acc32_t>(
                                      mli_math_mul_fx<io_T, mli_acc32_t>(input, scale), shift));
    } else {
        output = mli_math_min_fx(input,mli_math_acc_cast_fx<io_T, mli_acc32_t>(
                                      mli_math_mul_fx<io_T, mli_acc32_t>(input, scale), shift));
    }

    mli_prv_store_n_samples(vec_out, output);
}

template <typename io_T, prelu_elem_func_type func_type>
static MLI_FORCE_INLINE void mli_krn_scale_elem_v(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift,
        const int remaining_part) {

    MLI_ASSERT(remaining_part == 1);
    mli_krn_scale_elem_v<io_T, func_type>(vec_in, vec_out, scale, shift);
}

template <typename io_T>
static MLI_FORCE_INLINE mli_status mli_krn_prelu_fx_run(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        const mli_prelu_cfg *cfg, 
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
    /* Get Non Axis Tensor */
    auto in_non_axis_prv  = mli_prv_get_non_axis_tensor<MLI_PTR(io_T)>(&in_prv,  cfg->axis);
    auto out_non_axis_prv = mli_prv_get_non_axis_tensor<MLI_OUT_PTR(io_T)>(&out_prv, cfg->axis);
    /* Get Axis Tensor */
    in_prv  = mli_prv_get_axis_tensor<MLI_PTR(io_T)>(&in_prv,  cfg->axis);
    out_prv = mli_prv_get_axis_tensor<MLI_OUT_PTR(io_T)>(&out_prv, cfg->axis);
    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&in_prv );
    mli_prv_reorder_generic_tensor<MLI_OUT_PTR(io_T)>(&out_prv);

    /* Dummy Load to get num_lanes, remaining part */
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

    /* For applying the function to specific axis dimension, we should first loop across other dimensions then process
     * axis dimension elements.
     * For applying the function to the whole tensor, loop body is executed only one time. (i.e. shape[i] = 1).
     */
    if (mli_prv_less_than_1(scale, slope_coeff->el_params.fx.frac_bits)) {
        for (int dim0 = 0; dim0 < in_non_axis_prv.shape[0]; dim0++) {
            for (int dim1 = 0; dim1 < in_non_axis_prv.shape[1]; dim1++) {
                for (int dim2 = 0; dim2 < in_non_axis_prv.shape[2]; dim2++) {

                    vec_in = &in_ptr[dim0 * in_non_axis_prv.mem_stride[0] + 
                                    dim1 * in_non_axis_prv.mem_stride[1] + 
                                    dim2 * in_non_axis_prv.mem_stride[2]];
                    vec_out = &out_ptr[dim0 * out_non_axis_prv.mem_stride[0] + 
                                    dim1 * out_non_axis_prv.mem_stride[1] + 
                                    dim2 * out_non_axis_prv.mem_stride[2]];

                    const MLI_PTR(io_T) orig_vec_in = vec_in;
                    MLI_OUT_PTR(io_T) orig_vec_out = vec_out;
                    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
                        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                                vec_in  = (MLI_PTR(io_T))orig_vec_in  + POS(&in_prv,  pos0, pos1, pos2, 0);
                                vec_out = orig_vec_out + POS(&out_prv, pos0, pos1, pos2, 0);
                                if (remaining_part) {
                                    mli::krn::mli_krn_scale_elem_v<io_T, PRELU_ELEM_FUNC_MAX>(vec_in, vec_out, 
                                                                   scale, shift, remaining_part);
                                    vec_in  += remaining_part;
                                    vec_out += remaining_part;
                                }
                                for (int pos3 = remaining_part; pos3 < in_prv.shape[3]; pos3 += num_lanes) {
                                    mli::krn::mli_krn_scale_elem_v<io_T, PRELU_ELEM_FUNC_MAX>(vec_in, vec_out, 
                                                                   scale, shift);
                                    vec_in  += num_lanes;
                                    vec_out += num_lanes;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (int dim0 = 0; dim0 < in_non_axis_prv.shape[0]; dim0++) {
            for (int dim1 = 0; dim1 < in_non_axis_prv.shape[1]; dim1++) {
                for (int dim2 = 0; dim2 < in_non_axis_prv.shape[2]; dim2++) {

                    vec_in = &in_ptr[dim0 * in_non_axis_prv.mem_stride[0] + 
                                    dim1 * in_non_axis_prv.mem_stride[1] + 
                                    dim2 * in_non_axis_prv.mem_stride[2]];
                    vec_out = &out_ptr[dim0 * out_non_axis_prv.mem_stride[0] + 
                                    dim1 * out_non_axis_prv.mem_stride[1] + 
                                    dim2 * out_non_axis_prv.mem_stride[2]];

                    const MLI_PTR(io_T) orig_vec_in = vec_in;
                    MLI_OUT_PTR(io_T) orig_vec_out = vec_out;
                    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
                        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                                vec_in  = (MLI_PTR(io_T))orig_vec_in  + POS(&in_prv,  pos0, pos1, pos2, 0);
                                vec_out = orig_vec_out + POS(&out_prv, pos0, pos1, pos2, 0);
                                if (remaining_part) {
                                    mli::krn::mli_krn_scale_elem_v<io_T, PRELU_ELEM_FUNC_MIN>(vec_in, vec_out, 
                                                                   scale, shift, remaining_part);
                                    vec_in  += remaining_part;
                                    vec_out += remaining_part;
                                }
                                for (int pos3 = remaining_part; pos3 < in_prv.shape[3]; pos3 += num_lanes) {
                                    mli::krn::mli_krn_scale_elem_v<io_T, PRELU_ELEM_FUNC_MIN>(vec_in, vec_out, 
                                                                   scale, shift);
                                    vec_in  += num_lanes;
                                    vec_out += num_lanes;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return MLI_STATUS_OK;
}

static MLI_FORCE_INLINE mli_status mli_krn_prelu_sa8_run(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        const mli_prelu_cfg *cfg, 
        mli_tensor *out) {

    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(int8_t) vec_in = nullptr;
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
    auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(int8_t)>(out);
    /* Get Non Axis Tensor */
    auto in_non_axis_prv  = mli_prv_get_non_axis_tensor<MLI_PTR(int8_t)>(&in_prv,  cfg->axis);
    auto out_non_axis_prv = mli_prv_get_non_axis_tensor<MLI_OUT_PTR(int8_t)>(&out_prv, cfg->axis);
    /* Get Axis Tensor */
    in_prv  = mli_prv_get_axis_tensor<MLI_PTR(int8_t)>(&in_prv,  cfg->axis);
    out_prv = mli_prv_get_axis_tensor<MLI_OUT_PTR(int8_t)>(&out_prv, cfg->axis);
    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(int8_t)>(&in_prv );
    mli_prv_reorder_generic_tensor<MLI_OUT_PTR(int8_t)>(&out_prv);

    int8_t scale_sa8;
    // Getscalar value
    if (slope_coeff->rank == 0) {
        scale_sa8 = slope_coeff->data.mem.i8;
    } else {
        scale_sa8 = slope_coeff->data.mem.pi8[0];
    }

    int32_t alpha = 0;
	int16_t in_zp = 0, out_zp = 0;
	int16_t zp_identity_mod = 0, zp_mod = 0 ;
    int16_t scale_identity_mul = 0, scale_mul = 0, alpha_norm = 0;
    /* ****************************************************************************************************************
     *                        Mathematical Derivations for Leaky RELU SA8 
     * ----------------------------------------------------------------------------------------------------------------
     *      If (in_sa8 > in_offset)
     *          out_sa8 = scale_identity_mul_val * in_sa8 + zp_identity_mul;   
     *      else
     *          out_sa8 = scale_mul_val * in_sa8 + zp_mod;
     * 
     *      scale_identity_mul_val = in_scale_val / out_scale_val;
     *                             = in_scale * 2^(-in_scale_frac_bits) / (out_scale * 2^(-out_scale_frac_bits));
     *                             = (in_scale_val * 2^kPreDivShift / out_scale_val) 
     *                               * 2^(-(kPreDivShift + in_scale_frac_bits - out_scale_frac_bits));
     *                             = (in_scale_val * 2^kPreDivShift / out_scale_val) * 2^(-norm_shift) 
     *                              * 2^(-(kPreDivShift + in_scale_frac_bits - out_scale_frac_bits - norm_shift));
     *                             = (in_scale_val * 2^kPreDivShift / out_scale_val) * 2^(-norm_shift) 
     *                              * 2^(-scale_identity_shift)
     *                             = scale_identity_mul * 2 ^(-(scale_identity_shift))
     * 
     *      where scale_identity_mul = (in_scale_val * 2^kPreDivShift / out_scale_val) * 2^(-norm_shift) 
     *            scale_identity_shift = kPreDivShift + in_scale_frac_bits - out_scale_frac_bits - norm_shift
     *            norm_shift is the shift value due to normalizing the result of 
     *                       (in_scale_val * 2^kPreDivShift / out_scale_val) and casting it from int32_t to int16_t
     *            kPreDivShift is derived from norm_32(in_scale) - norm_16(out_scale)
     * 
     *      zp_identity_mul = out_zp - (scale_identity_mul * in_zp) * 2^(-(scale_identity_shift));
     * 
     *      For scale_mul_val = scale_identity_mul_val * alpha_val * (alpha_sa8 - alpha_zp) 
     *                        = scale_identity_mul * 2 ^(-(scale_identity_shift)) * alpha_scale 
     *                          * 2^(-alpha_scale_frac_bits) * (alpha_sa8 - alpha_zp)
     *                        = scale_identity_mul * alpha_scale * (alpha_sa8 - alpha_zp) 
     *                          * 2^(-(scale_identity_shift + alpha_scale_frac_bits))
     *                        = scale_identity_mul * alpha_scale * (alpha_sa8 - alpha_zp) 
     *                          * 2^(-(alpha_norm_shift)) * 2^(-(scale_mul_norm_shift))
     *                          * 2^(-(scale_identity_shift + alpha_scale_frac_bits - alpha_norm_shift))
     *                        = scale_mul * 2^(-(scale_shift))
     * 
     *      where scale_mul = scale_identity_mul * alpha_scale * (alpha_sa8 - alpha_zp) * 2^(-(alpha_norm_shift))
     *            scale_shift = scale_identity_shift + alpha_scale_frac_bits - alpha_norm_shift - scale_mul_norm_shift
     *            alpha_norm_shift is the shift value due to normalizing the result of 
     *                             alpha_scale * (alpha_sa8 - alpha_zp) and casting it from int32_t to int16_t
     *            scale_mul_norm_shift is the shift value due to normalizing the result of 
     *                             scale_identity_mul * alpha_scale * (alpha_sa8 - alpha_zp) * 2^(-(alpha_norm_shift))
     * 
     *      zp_mul = out_zp - (scale_mul * in_zp) * 2^(-(scale_shift));
     * 
     *      and finally
     *          If (in_sa8 > in_offset)
     *              out_sa8 = (scale_identity_mul * in_sa8) * 2^(-(scale_identity_shift)) + zp_identity_mul; 
     *          else
     *              out_sa8 = (scale_mul * in_sa8) * 2^(-(scale_shift)) + zp_identity_mul;
     * ***************************************************************************************************************/

    /* Input Zero Point */
    in_zp = in->el_params.sa.zero_point.mem.i16;
    /* Output Zero Point */
    out_zp = out->el_params.sa.zero_point.mem.i16;

    /* kPreDivShift = norm_32(in_scale) - norm_16(out_scale) */
    int kPreDivShift = mli_math_norm_fx<int32_t,int16_t>(in->el_params.sa.scale.mem.i16) - 
                       mli_math_norm_fx<int16_t,int16_t>(out->el_params.sa.scale.mem.i16);

    int norm_shift;
    scale_identity_mul = mli_math_norm_cast_fx<int32_t,int16_t>(
                                                ((int32_t)(in->el_params.sa.scale.mem.i16) << kPreDivShift) / 
                                                out->el_params.sa.scale.mem.i16, &norm_shift);
    int scale_identity_shift  = kPreDivShift;
    scale_identity_shift += in->el_params.sa.scale_frac_bits.mem.i8 - out->el_params.sa.scale_frac_bits.mem.i8; 
    scale_identity_shift -= norm_shift;
    /* zp_identity_mod = out_zp - scale_identity_mul * in_zp */
    zp_identity_mod = mli_math_sub_fx<int16_t>(out_zp, 
                      mli_math_cast_fx<int32_t, int16_t>(
                      mli_math_mul_fx<int16_t, int32_t>(scale_identity_mul, in_zp), scale_identity_shift));

    /* alpha = slope_scale * ( scale_sa8 - scale_zp) */                                                
    alpha = mli_prv_convert_sa8_fx16<int8_t, int32_t>(scale_sa8, slope_coeff->el_params.sa.zero_point.mem.i16, 
                                                                 slope_coeff->el_params.sa.scale.mem.i16);
    /* Normalize alpha and cast to 16bit */
    alpha_norm = mli_math_norm_cast_fx<int32_t,int16_t>(alpha, &norm_shift);
    
    int scale_shift  = scale_identity_shift;
        scale_shift += slope_coeff->el_params.sa.scale_frac_bits.mem.i8;
        scale_shift -= norm_shift;
    
    /* scale_identity_mul = in_scale_fx / out_scale_fx */
    scale_mul = mli_math_norm_cast_fx<int32_t,int16_t>(
                mli_math_mul_fx<int16_t, int32_t>(scale_identity_mul, alpha_norm), &norm_shift);
    scale_shift -= norm_shift;

    /* zp_mod = out_zp - scale_mul * in_zp */
    zp_mod = mli_math_sub_fx<int16_t>(out_zp,
             mli_math_cast_fx<int32_t, int16_t>(
             mli_math_mul_fx<int32_t, int32_t>(scale_mul, in_zp), scale_shift));

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
                                int16_t output;
                                if (input > in_zp) {
                                    /* Identity -> output = scale_identity_mul * in_sa8 + zp_identity_mod */
                                   output =  mli_math_add_fx<int16_t>(
                                        mli_math_cast_fx<int32_t, int16_t>(
                                        mli_math_mul_fx<int16_t, int32_t>(scale_identity_mul, input), scale_identity_shift), zp_identity_mod);
                                } else {
                                    /* Alpha -> output = scale_mul * in_sa8 + zp_mod */
                                    output =  mli_math_add_fx<int16_t>(
                                        mli_math_cast_fx<int32_t, int16_t>(
                                        mli_math_mul_fx<int16_t, int32_t>(scale_mul, input), scale_shift), zp_mod);
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