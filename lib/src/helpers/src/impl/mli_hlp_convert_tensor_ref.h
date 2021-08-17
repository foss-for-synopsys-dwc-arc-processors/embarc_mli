/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_HLP_CONVERT_TENSOR_REF_H_
#define _MLI_HLP_CONVERT_TENSOR_REF_H_

#include "mli_config.h"
#include "mli_mem_info.h"
#include "mli_prv_quant.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"
#include <math.h>

template <typename in_T, typename out_T, typename acc_T>
MLI_FORCE_INLINE void calc_convert(const MLI_PTR(in_T) src_tensor_arr,
                                   MLI_OUT_PTR(out_T) dst_tensor_arr,
                                   const int16_t in_zp, const int16_t scale,
                                   const int16_t scale_shift, const int16_t out_zp) {
    if (std::is_same<acc_T, int64_t>::value) {
        const int mul_hi_shift = 32;
        int32_t src_in_zp = mli_math_sub_fx<int32_t>(*src_tensor_arr, in_zp);
        int32_t src_norm = mli_math_norm_fx<int32_t, int32_t>(src_in_zp);
        src_in_zp = mli_math_asl_fx<int32_t>(src_in_zp, src_norm);

        int32_t scale_norm = mli_math_norm_fx<int32_t, int32_t>((int32_t) scale);
        int32_t scale32 = mli_math_asl_fx<int32_t>((int32_t) scale, scale_norm);

        int64_t dst_acc = mli_math_mul_fx<int32_t, int64_t>(src_in_zp, scale32);
        int32_t acc_hi = dst_acc >> mul_hi_shift;

        int32_t dst_acc_shf_casted = mli_math_asr_rnd_fx<int32_t>(acc_hi, scale_shift + scale_norm + src_norm - mul_hi_shift);
        int32_t dst_val = mli_math_add_fx<int32_t>(dst_acc_shf_casted, out_zp);
        *dst_tensor_arr = mli_math_cast_fx<int32_t, out_T>(dst_val, 0);
    } else {
        int16_t src_in_zp = mli_math_sub_fx<int16_t>(*src_tensor_arr, in_zp);
        acc_T dst_acc = mli_math_mul_fx<int16_t, acc_T>(src_in_zp, scale);
        acc_T dst_acc_shf_casted = mli_math_asr_rnd_fx<acc_T>(dst_acc, scale_shift);
        acc_T dst_val = mli_math_add_fx<acc_T>(dst_acc_shf_casted, out_zp);
        *dst_tensor_arr = mli_math_cast_fx<acc_T, out_T>(dst_val, 0);
    }
}


namespace mli {
namespace hlp {
namespace ref {

#pragma MLI_CODE_SECTION_START(".mli_lib")

template <typename in_T, typename out_T, typename acc_T>
mli_status compute_convert_quantized_data(const mli_tensor * src, mli_tensor * dst) {
    mli_prv_fx_init_dsp_ctrl();

    /* Get Generic Private Tensors */
    auto src_prv = mli_prv_get_generic_tensor<MLI_PTR(in_T)>(src);
    auto dst_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(out_T)>(dst);

    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(in_T)>(&src_prv);
    mli_prv_reorder_generic_tensor<MLI_OUT_PTR(out_T)>(&dst_prv);

    const MLI_PTR(in_T) src_tensor_arr = src_prv.ptr;
    MLI_OUT_PTR(out_T) dst_tensor_arr = dst_prv.ptr;

    const int32_t src_tensor_size = src->data.capacity / sizeof(src_tensor_arr[0]);
    const int32_t dst_tensor_size = dst->data.capacity / sizeof(dst_tensor_arr[0]);

    int scale_dim = -1;
    int scales_num = 1;
    /* scale_dim and scales_num can change if one of tensors (src or dst) has SA type.
    *  Also, if both src and dst have SA type, their scale_dim and scales_num will be the same.
    *  To cover both cases, two if statements below are used.
    */
    if ((src->el_type == MLI_EL_SA_8 || src->el_type == MLI_EL_SA_32) && src->el_params.sa.dim >= 0) {
        scale_dim = src->el_params.sa.dim + (MLI_MAX_RANK - src->rank);
        scales_num = src_prv.shape[scale_dim];
    }
    if ((dst->el_type == MLI_EL_SA_8 || dst->el_type == MLI_EL_SA_32) && dst->el_params.sa.dim >= 0) {
        scale_dim = dst->el_params.sa.dim + (MLI_MAX_RANK - src->rank);
        scales_num = dst_prv.shape[scale_dim];
    }

    /* Transformation will be applied on slices across scales dimension (or all tensor) */
    for (int scale_idx = 0; scale_idx < scales_num; ++scale_idx) {
        /* Calculate scale and scaled zero point. */
        mli::krn::s8asym_quant_params params;
        mli::krn::define_requant_params(src, dst, &params, scale_idx);
        const int16_t scale_shift = mli_math_min_fx(params.shift, (int16_t) ((sizeof(acc_T) * 8) - 1));
        const int16_t scale = params.scale;
        const int16_t in_zp = mli_hlp_tensor_zero_offset(src, scale_idx);
        const int16_t out_zp = mli_hlp_tensor_zero_offset(dst, scale_idx);
        /* Calculate borders across all dimensions for slice where this scale is applicable */
        int dim_start[MLI_MAX_RANK] = { 0 };
        int dim_end[MLI_MAX_RANK] = { 0 };
        for (int i = 0; i < MLI_MAX_RANK; ++i) {
            dim_start[i] = (scale_dim == i) ? scale_idx : 0;
            dim_end[i] = (scale_dim == i) ? scale_idx + 1 : src_prv.shape[i];
        }

        /* Apply transformation of defined slice */
        for (int dim0_idx = dim_start[0]; dim0_idx < dim_end[0]; ++dim0_idx) {
            for (int dim1_idx = dim_start[1]; dim1_idx < dim_end[1]; ++dim1_idx) {
                for (int dim2_idx = dim_start[2]; dim2_idx < dim_end[2]; ++dim2_idx) {
                    for (int dim3_idx = dim_start[3]; dim3_idx < dim_end[3]; ++dim3_idx) {
                        const int src_pos = POS(&src_prv, dim0_idx, dim1_idx, dim2_idx, dim3_idx);
                        const int dst_pos = POS(&dst_prv, dim0_idx, dim1_idx, dim2_idx, dim3_idx);
                        MLI_ASSERT(src_pos < src_tensor_size);
                        MLI_ASSERT(dst_pos < dst_tensor_size);
                        calc_convert<in_T, out_T, acc_T>(&src_tensor_arr[src_pos], &dst_tensor_arr[dst_pos],
                                                         in_zp, scale, scale_shift, out_zp);
                    }
                }
            }
        }
    }
    return MLI_STATUS_OK;
}

mli_status convert_quantized_data(const mli_tensor * src, mli_tensor * dst) {

    mli_status ret;

    if ((src->el_type == MLI_EL_FX_8 || src->el_type == MLI_EL_SA_8) &&
            (dst->el_type == MLI_EL_FX_8 || dst->el_type == MLI_EL_SA_8)) {
        ret = compute_convert_quantized_data<int8_t, int8_t, int32_t>(src, dst);
    } else if ((src->el_type == MLI_EL_FX_8 || src->el_type == MLI_EL_SA_8) &&
            dst->el_type == MLI_EL_FX_16) {
        ret = compute_convert_quantized_data<int8_t, int16_t, int32_t>(src, dst);
    } else if ((src->el_type == MLI_EL_FX_8 || src->el_type == MLI_EL_SA_8) &&
            dst->el_type == MLI_EL_SA_32) {
        ret = compute_convert_quantized_data<int8_t, int32_t, int64_t>(src, dst);
    } else if (src->el_type == MLI_EL_FX_16 &&
            (dst->el_type == MLI_EL_FX_8 || dst->el_type == MLI_EL_SA_8)) {
        ret = compute_convert_quantized_data<int16_t, int8_t, int32_t>(src, dst);
    } else if (src->el_type == MLI_EL_FX_16 && dst->el_type == MLI_EL_FX_16) {
        ret = compute_convert_quantized_data<int16_t, int16_t, int32_t>(src, dst);
    } else if (src->el_type == MLI_EL_FX_16 && dst->el_type == MLI_EL_SA_32) {
        ret = compute_convert_quantized_data<int16_t, int32_t, int64_t>(src, dst);
    } else if (src->el_type == MLI_EL_SA_32 &&
            (dst->el_type == MLI_EL_FX_8 || dst->el_type == MLI_EL_SA_8)) {
        ret = compute_convert_quantized_data<int32_t, int8_t, int64_t>(src, dst);
    } else if (src->el_type == MLI_EL_SA_32 && dst->el_type == MLI_EL_FX_16) {
        ret = compute_convert_quantized_data<int32_t, int16_t, int64_t>(src, dst);
    } else if (src->el_type == MLI_EL_SA_32 && dst->el_type == MLI_EL_SA_32) {
        ret = compute_convert_quantized_data<int32_t, int32_t, int64_t>(src, dst);
    } else {
        ret = MLI_STATUS_TYPE_MISMATCH;
    }

    return ret;
}

template <typename t_T>
mli_status convert_float_data(const mli_tensor * src, mli_tensor * dst, convert_mode mode) {
    mli_prv_fx_init_dsp_ctrl();

    const mli_tensor* tensor = nullptr;
    const mli_tensor* float_tensor = nullptr;

    /* Defining float_tensor and tensor depending on current conversion direction */
    if (mode == mli::hlp::QUANTIZE) {
        float_tensor = src;
        tensor = dst;
    } else if (mode == mli::hlp::DEQUANTIZE) {
        float_tensor = dst;
        tensor = src;
    }

    /* Get Generic Private Tensors */
    auto tensor_prv = mli_prv_get_generic_tensor<MLI_PTR(t_T)>(tensor);
    auto float_tensor_prv = mli_prv_get_generic_tensor<MLI_PTR(float)>(float_tensor);

    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(t_T)>(&tensor_prv);
    mli_prv_reorder_generic_tensor<MLI_OUT_PTR(float)>(&float_tensor_prv);

    MLI_PTR(t_T) tensor_arr = tensor_prv.ptr;
    MLI_PTR(float) float_tensor_arr = float_tensor_prv.ptr;

    const int32_t float_tensor_size = float_tensor->data.capacity / sizeof(float_tensor_arr[0]);
    const int32_t tensor_size = tensor->data.capacity / sizeof(tensor_arr[0]);

    int scale_dim = -1;
    int scales_num = 1;
    if ((tensor->el_type == MLI_EL_SA_8 || tensor->el_type == MLI_EL_SA_32) && tensor->el_params.sa.dim >= 0) {
        scale_dim = tensor->el_params.sa.dim + (MLI_MAX_RANK - tensor->rank);
        scales_num = tensor_prv.shape[scale_dim];
    }

    /* Transformation will be applied on slices across scales dimension (or all tensor) */
    for (int scale_idx = 0; scale_idx < scales_num; ++scale_idx) {
        /* Calculate current scale and zero offset */
        float scale_val = 1.0;
        int8_t frac_bits = mli_hlp_tensor_scale_shift(tensor, scale_idx);
        float scale = (float) mli_hlp_tensor_scale(tensor, scale_idx);
        if (mode == mli::hlp::QUANTIZE) {
            scale = 1.0f / scale;
            scale_val = (float)ldexp(scale, ((int32_t) frac_bits));
        } else {
            scale_val = (float)ldexp(scale, -((int32_t) frac_bits));
        }

        int16_t zero_offset = mli_hlp_tensor_zero_offset(tensor, scale_idx);

        /* Calculate borders across all dimensions for slice where this scale is applicable */
        int dim_start[MLI_MAX_RANK] = { 0 };
        int dim_end[MLI_MAX_RANK] = { 0 };
        for (int i = 0; i < MLI_MAX_RANK; ++i) {
            dim_start[i] = (scale_dim == i) ? scale_idx : 0;
            dim_end[i] = (scale_dim == i) ? scale_idx + 1 : tensor_prv.shape[i];
        }

        /* Apply transformation of defined slice */
        for (int dim0_idx = dim_start[0]; dim0_idx < dim_end[0]; ++dim0_idx) {
            for (int dim1_idx = dim_start[1]; dim1_idx < dim_end[1]; ++dim1_idx) {
                for (int dim2_idx = dim_start[2]; dim2_idx < dim_end[2]; ++dim2_idx) {
                    for (int dim3_idx = dim_start[3]; dim3_idx < dim_end[3]; ++dim3_idx) {
                        const int float_tensor_pos = POS(&float_tensor_prv, dim0_idx, dim1_idx, dim2_idx, dim3_idx);
                        const int tensor_pos = POS(&tensor_prv, dim0_idx, dim1_idx, dim2_idx, dim3_idx);
                        MLI_ASSERT(float_tensor_pos < float_tensor_size);
                        MLI_ASSERT(tensor_pos < tensor_size);

                        if (tensor->el_type == MLI_EL_FP_32) {
                            tensor_arr[tensor_pos] = (t_T)float_tensor_arr[float_tensor_pos];
                        } else {
                            if (mode == mli::hlp::QUANTIZE) {
                                int32_t tensor_val = mli_math_float_scale(float_tensor_arr[float_tensor_pos], scale_val);
                                tensor_arr[tensor_pos] = mli_math_cast_fx<int32_t, t_T>(mli_math_add_fx<int32_t>(tensor_val, zero_offset), 0);
                            } else if (mode == mli::hlp::DEQUANTIZE) {
                                const float float_tensor_val_unscaled = static_cast<float>(mli_math_sub_fx<int32_t>((int32_t)(tensor_arr[tensor_pos]), zero_offset));
                                float_tensor_arr[float_tensor_pos] = float_tensor_val_unscaled * scale_val;
                            }
                        }
                    }
                }
            }
        }
    }
    return MLI_STATUS_OK;
}
#pragma MLI_CODE_SECTION_END()
} // namespace ref
} // namespace hlp
} // namespace mli

#endif  //_MLI_HLP_CONVERT_TENSOR_REF_H_
