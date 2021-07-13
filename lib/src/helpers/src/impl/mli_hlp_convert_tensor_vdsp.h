/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_HLP_CONVERT_TENSOR_VDSP_H_
#define _MLI_HLP_CONVERT_TENSOR_VDSP_H_

#include "mli_config.h"
#include "mli_mem_info.h"
#include "mli_prv_quant.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace hlp {
namespace vdsp {

#pragma MLI_CODE_SECTION_START(".mli_lib")

static MLI_FORCE_INLINE vNx4int_t calc_convert(
        vNx4char_t input,
        const int16_t scale,
        const int16_t shift,
        const int16_t in_zp,
        const int16_t out_zp) {

    vNx4short_t input_cast =  mli_math_cast_fx<vNx4char_t, vNx4short_t>(input);
    vNx4short_t src_in_zp = mli_math_sub(input_cast, in_zp);
    vNx4int_t   dst_acc = mli_math_mul_fx<vNx4short_t, vNx4int_t>(src_in_zp, scale);
    vNx4int_t   dst_acc_shf_casted = mli_math_asr_rnd_fx(dst_acc, (int)shift);
    vNx4int_t   dst_val = mli_math_add_fx<vNx4int_t>(dst_acc_shf_casted, out_zp);

    return dst_val;
}

static MLI_FORCE_INLINE vNx4int_t calc_convert(
        vNx4short_t input,
        const int16_t scale,
        const int16_t shift,
        const int16_t in_zp,
        const int16_t out_zp) {

    vNx4short_t src_in_zp = mli_math_sub(input, in_zp);
    vNx4int_t   dst_acc = mli_math_mul_fx<vNx4short_t, vNx4int_t>(src_in_zp, scale);
    vNx4int_t   dst_acc_shf_casted = mli_math_asr_rnd_fx(dst_acc, (int)shift);
    vNx4int_t   dst_val = mli_math_add_fx<vNx4int_t>(dst_acc_shf_casted, out_zp);

    return dst_val;
}

static MLI_FORCE_INLINE vNx4int_t calc_convert(
        vNx4int_t input,
        const int16_t scale,
        const int16_t shift,
        const int16_t in_zp,
        const int16_t out_zp) {
    
    constexpr int mul_pre_shift = 16;
    constexpr int mul_hi_shift = 32;
    int total_shift = shift - (mul_hi_shift - mul_pre_shift);
    int shift_right = mli_math_max_fx(total_shift, 1);
    int shift_left = mli_math_max_fx(1 - total_shift, 0);

    vNx4int_t src_in_zp = mli_math_sub(input, (int32_t)in_zp);
              src_in_zp = mli_math_asl_fx(src_in_zp, shift_left);
    vNx4int_t dst_acc = mli_math_mul_fx_high(src_in_zp, ((int32_t)scale << mul_pre_shift));
    vNx4int_t dst_acc_shf_casted = mli_math_asr_rnd_fx(dst_acc, shift_right);
    vNx4int_t dst_val = mli_math_add_fx<vNx4int_t>(dst_acc_shf_casted, out_zp);

    return dst_val;
}

template<typename out_T>
static MLI_FORCE_INLINE void store_convert(
        MLI_OUT_PTR(out_T) out_ptr,
        vNx4int_t output,
        int remaining_part = 0) {
    
    typedef decltype(mli_prv_load_nx4_samples(out_ptr)) cast_type;

    if (remaining_part) {
        mli_prv_store_n_samples(out_ptr,
                                mli_math_cast_fx<vNx4int_t, cast_type>(output, 0),
                                remaining_part);
    } else {
        mli_prv_store_n_samples(out_ptr,
                                mli_math_cast_fx<vNx4int_t, cast_type>(output, 0));
    }
}

template<>
MLI_FORCE_INLINE void store_convert<int32_t>(
        MLI_OUT_PTR(int32_t) out_ptr,
        vNx4int_t output,
        int remaining_part) {

    if (remaining_part) {
        mli_prv_store_n_samples(out_ptr, output, remaining_part);
    } else {
        mli_prv_store_n_samples(out_ptr, output);
    }
}

template <typename in_T, typename out_T, typename acc_T>
static MLI_FORCE_INLINE void compute_convert_one_dim(
        MLI_PTR(in_T) in_ptr,
        MLI_OUT_PTR(out_T) out_ptr,
        const int16_t scale,
        const int16_t shift,
        const int16_t in_zp,
        const int16_t out_zp,
        const int shape) {

    /* Dummy Load to get num_lanes */
    auto input = mli_prv_load_nx4_samples(in_ptr);
    int num_lanes = get_number_lanes(input);
    int remaining_part = shape & (num_lanes - 1);

    if (remaining_part) {
        auto convert_input = mli_prv_load_nx4_samples(in_ptr);
        auto convert_output = calc_convert(convert_input, scale, shift, in_zp, out_zp);
        store_convert<out_T>(out_ptr, convert_output, remaining_part);
        in_ptr  += remaining_part;
        out_ptr += remaining_part;
    }

    for (int pos = remaining_part; pos < shape; pos+= num_lanes) {
        auto convert_input = mli_prv_load_nx4_samples(in_ptr);
        vNx4int_t convert_output = calc_convert(convert_input, scale, shift, in_zp, out_zp);
        store_convert<out_T>(out_ptr, convert_output);
        in_ptr  += num_lanes;
        out_ptr += num_lanes;
    }
}

// TODO: Remove this when a known issue is solved.
template <>
MLI_FORCE_INLINE void compute_convert_one_dim<int32_t, int32_t, int64_t>(
        MLI_PTR(int32_t) in_ptr,
        MLI_OUT_PTR(int32_t) out_ptr,
        const int16_t scale,
        const int16_t shift,
        const int16_t in_zp,
        const int16_t out_zp,
        const int shape) {

    for (int pos = 0; pos < shape; pos++) {
        int32_t src_in_zp = mli_math_sub_fx<int32_t>(in_ptr[pos], in_zp);
        int64_t dst_acc = mli_math_mul_fx<int32_t, int64_t>(src_in_zp, scale);
        int64_t dst_acc_shf_casted = mli_math_asr_rnd_fx<int64_t>(dst_acc, shift);
        int64_t dst_val = mli_math_add_fx<int64_t>(dst_acc_shf_casted, out_zp);
        out_ptr[pos] = mli_math_cast_fx<int64_t, int32_t>(dst_val, 0);
    }
}

template <typename in_T, typename out_T, typename acc_T>
mli_status convert_quantized_data(const mli_tensor * src, mli_tensor * dst) {

    /* Copy shape and rank from source tensor to destination */
    const int rank = dst->rank = src->rank;
    for (int i = 0; i < rank; ++i)
        dst->shape[i] = src->shape[i];

    /* Get Generic Private Tensors */
    auto src_prv = mli_prv_get_generic_tensor<MLI_PTR(in_T)>(src);
    auto dst_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(out_T)>(dst);

    MLI_PTR(in_T)      __restrict src_tensor_arr = src_prv.ptr;
    MLI_OUT_PTR(out_T) __restrict dst_tensor_arr = dst_prv.ptr;

    int scale_dim = -1;
    int scales_num = 1;
    /* scale_dim and scales_num can change if one of tensors (src or dst) has SA type.
    *  Also, if both src and dst have SA type, their scale_dim and scales_num will be the same.
    *  To cover both cases, two if statements below are used.
    */
    if ((src->el_type == MLI_EL_SA_8 || src->el_type == MLI_EL_SA_32) && src->el_params.sa.dim >= 0) {
        scale_dim = src->el_params.sa.dim;
        scales_num = src_prv.shape[scale_dim];
    }
    if ((dst->el_type == MLI_EL_SA_8 || dst->el_type == MLI_EL_SA_32) && dst->el_params.sa.dim >= 0) {
        scale_dim = dst->el_params.sa.dim;
        scales_num = dst_prv.shape[scale_dim];
    }

    if (scale_dim == -1) {
        /* Per Tensor Convert */
        /* Calculate scale and scaled zero point. */
        mli::krn::s8asym_quant_params params;
        mli::krn::define_requant_params(src, dst, &params, scale_dim);
        const int16_t scale_shift = params.shift;
        const int16_t scale = params.scale;
        int16_t in_zp = mli_hlp_tensor_zero_offset(src, scale_dim);
        int16_t out_zp = mli_hlp_tensor_zero_offset(dst, scale_dim);
        /* Trying to squash tensor to one dim */
        int shape = mli_prv_squash_tensor_to_one_dim(src, dst);
        if (shape) {
            compute_convert_one_dim<in_T, out_T, acc_T>(src_tensor_arr, dst_tensor_arr,
                                                        scale, scale_shift, in_zp, out_zp, shape);
        } else {
            /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
            mli_prv_reorder_generic_tensor<MLI_PTR(in_T)>(&src_prv);
            mli_prv_reorder_generic_tensor<MLI_OUT_PTR(out_T)>(&dst_prv);

            /* Loop Over Sub Tensor */
            const MLI_PTR(in_T) __restrict orig_vec_in = src_tensor_arr;
            MLI_OUT_PTR(out_T) __restrict orig_vec_out = dst_tensor_arr;
            for (int pos0 = 0; pos0 < src_prv.shape[0]; pos0++) {
                for (int pos1 = 0; pos1 < src_prv.shape[1]; pos1++) {
                    for (int pos2 = 0; pos2 < src_prv.shape[2]; pos2++) {
                        src_tensor_arr  = (MLI_PTR(in_T))orig_vec_in  + POS(&src_prv, pos0, pos1, pos2, 0);
                        dst_tensor_arr = orig_vec_out + POS(&dst_prv, pos0, pos1, pos2, 0);
                        compute_convert_one_dim<in_T, out_T, acc_T>(src_tensor_arr, dst_tensor_arr,
                                                                    scale, scale_shift, in_zp, out_zp, src_prv.shape[3]);
                    }
                }
            }
        }
    } else {
        /* Broadcasting in case axis is not inner most dim */
        bool broadcasting = !(scale_dim == (src_prv.rank - 1));
        if (broadcasting) {
            int axis_src_mem_stride = src_prv.mem_stride[scale_dim];
            int axis_dst_mem_stride = dst_prv.mem_stride[scale_dim];
            /* Get Non Axis Tensor */
            auto src_non_axis_prv  = mli_prv_get_non_axis_tensor<MLI_PTR(in_T)>(&src_prv,  scale_dim);
            auto dst_non_axis_prv = mli_prv_get_non_axis_tensor<MLI_PTR(out_T)>(&dst_prv, scale_dim);

            /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
            mli_prv_reorder_generic_tensor<MLI_PTR(in_T)>(&src_non_axis_prv );
            mli_prv_reorder_generic_tensor<MLI_OUT_PTR(out_T)>(&dst_non_axis_prv);

            for (int scale_idx = 0; scale_idx < scales_num; scale_idx++) {
                /* Calculate scale and scaled zero point. */
                mli::krn::s8asym_quant_params params;
                mli::krn::define_requant_params(src, dst, &params, scale_idx);
                const int16_t scale_shift = params.shift;
                const int16_t scale = params.scale;
                int16_t in_zp = mli_hlp_tensor_zero_offset(src, scale_idx);
                int16_t out_zp = mli_hlp_tensor_zero_offset(dst, scale_idx);
                
                /* Define Sub Tensor */
                MLI_PTR(in_T) vec_in  = (MLI_PTR(in_T))src_prv.ptr  + scale_idx * axis_src_mem_stride;
                MLI_OUT_PTR(out_T) vec_out = dst_prv.ptr + scale_idx * axis_dst_mem_stride;

                /* Loop Over Sub Tensor */
                MLI_PTR(in_T) orig_vec_in = vec_in;
                MLI_OUT_PTR(out_T) orig_vec_out = vec_out;
                for (int pos1 = 0; pos1 < src_non_axis_prv.shape[1]; pos1++) {
                    for (int pos2 = 0; pos2 < src_non_axis_prv.shape[2]; pos2++) {
                        vec_in  = (MLI_PTR(in_T))orig_vec_in  + POS(&src_non_axis_prv, 0, pos1, pos2, 0);
                        vec_out = orig_vec_out + POS(&dst_non_axis_prv, 0, pos1, pos2, 0);
                        compute_convert_one_dim<in_T, out_T, acc_T>(vec_in, vec_out,
                                                                    scale, scale_shift, in_zp, out_zp, src_non_axis_prv.shape[3]);
                    }
                }
            }

        } else {
            /* Fall back to ref in case Axis is inner most dim */
            return mli::hlp::ref::convert_quantized_data<in_T, out_T, acc_T>(src, dst);
        }
    }
    return MLI_STATUS_OK;
}

#pragma MLI_CODE_SECTION_END()
} // namespace vdsp
} // namespace hlp
} // namespace mli

#endif  //_MLI_HLP_CONVERT_TENSOR_VDSP_H_
