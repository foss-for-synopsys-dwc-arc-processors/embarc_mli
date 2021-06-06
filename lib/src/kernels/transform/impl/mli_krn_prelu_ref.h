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
#include "mli_krn_leaky_relu.h"

namespace mli {
namespace krn {
namespace ref {

template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift) {
    mli::krn::ref::compute_leaky_relu(vec_in, vec_out, scale, shift);
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

    mli::krn::ref::compute_leaky_relu(vec_in, vec_out, in_zp, identity_params, alpha_params);
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

template <typename io_T>
static MLI_FORCE_INLINE mli_status prelu_fx_run(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        const mli_prelu_cfg *cfg, 
        mli_tensor *out) {

    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(io_T) vec_in = nullptr;
    MLI_OUT_PTR(io_T) vec_out = nullptr;

    const MLI_PTR(io_T) in_ptr = mli_prv_tensor_data_ptr<MLI_PTR(io_T)>(in);
    const MLI_PTR(io_T) slope_ptr = mli_prv_tensor_data_ptr<MLI_PTR(io_T)>(slope_coeff);
    MLI_OUT_PTR(io_T) out_ptr = mli_prv_tensor_data_ptr<MLI_OUT_PTR(io_T)>(out);

    /* Copy tensor format */
    mli_prv_copy_tensor_format_except_mem_strides(in, out);
    /* Get Generic Private Tensor */
    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);    
    auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(io_T)>(out);

    /* Define Slope Axis Params */
    int axis_shape = slope_coeff->shape[cfg->axis];
    int axis_in_mem_stride = in_prv.mem_stride[cfg->axis];
    int axis_out_mem_stride = out_prv.mem_stride[cfg->axis];
    /* Broadcasting in case axis is not inner most dim */
    bool broadcasting = !(cfg->axis == (in_prv.rank - 1));

    /* Dummy Load to get num_lanes */
    auto input = mli_prv_load_1vec(in_ptr);
    int num_lanes = get_number_lanes(input);

    int shift = mli_prv_calc_shift(in, slope_coeff, out);

    if (broadcasting) {
        /* Get Non Axis Tensor */
        auto in_non_axis_prv  = mli_prv_get_non_axis_tensor<MLI_PTR(io_T)>(&in_prv,  cfg->axis);
        auto out_non_axis_prv = mli_prv_get_non_axis_tensor<MLI_PTR(io_T)>(&out_prv, cfg->axis);

        /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
        mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&in_non_axis_prv );
        mli_prv_reorder_generic_tensor<MLI_OUT_PTR(io_T)>(&out_non_axis_prv);

        int remaining_part = in_non_axis_prv.shape[3] & (num_lanes - 1);

        for (int scale_idx = 0; scale_idx < axis_shape; scale_idx++) {
            /* Define Sub Tensor */
            vec_in  = (MLI_PTR(io_T))in_ptr  + scale_idx * axis_in_mem_stride;
            vec_out = out_ptr + scale_idx * axis_out_mem_stride;
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
                        mli::krn::compute_prelu<io_T, decltype(input)>(vec_in, scale_v, vec_out,
                                                            shift, remaining_part);
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
    } else {
        /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
        mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&in_prv );
        mli_prv_reorder_generic_tensor<MLI_OUT_PTR(io_T)>(&out_prv);

        int remaining_part = in_prv.shape[3] & (num_lanes - 1);

        if (remaining_part) {
            /* Load Scale Vector */
            auto scale_v = mli_prv_load_1vec(slope_ptr);
            mli::krn::compute_prelu_no_broadcast<io_T, decltype(input)>(in_ptr, out_ptr, scale_v,
                                                                        shift, in_prv, out_prv, remaining_part);
        }
        for (int scale_idx = remaining_part; scale_idx < axis_shape; scale_idx += num_lanes) {
            /* Define Sub Tensor */
            vec_in  = (MLI_PTR(io_T))in_ptr  + scale_idx * axis_in_mem_stride;
            vec_out = out_ptr + scale_idx * axis_out_mem_stride;
            /* Load Scale Vector */
            auto scale_v = mli_prv_load_1vec(&slope_ptr[scale_idx]);
            mli::krn::compute_prelu_no_broadcast<io_T, decltype(input)>(vec_in, vec_out, scale_v,
                                                                        shift, in_prv, out_prv);
        }
    }
    
    return MLI_STATUS_OK;
}

static MLI_FORCE_INLINE s8asym_quant_params prelu_define_requant_params(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        mli_tensor *out,
        const int8_t alpha_sa8,
        const s8asym_quant_params *identity_params) {
    
    return mli::krn::ref::leaky_relu_define_requant_params(in, slope_coeff, out, alpha_sa8, identity_params);
}

static MLI_FORCE_INLINE void compute_prelu_no_broadcast(
        const MLI_PTR(int8_t) __restrict vec_in,
        MLI_OUT_PTR(int8_t) __restrict vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params *alpha_params,
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
                    mli::krn::ref::compute_prelu(vec_in, vec_out, in_zp, identity_params, alpha_params,
                                                 remaining_part);
                } else {
                    mli::krn::ref::compute_prelu(vec_in, vec_out, in_zp, identity_params, alpha_params);
                }
            }
        }
    }
}

static MLI_FORCE_INLINE mli_status prelu_sa8_run(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        const mli_prelu_cfg *cfg, 
        mli_tensor *out) {

    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(int8_t) vec_in = nullptr;
    MLI_OUT_PTR(int8_t) vec_out = nullptr;

    const MLI_PTR(int8_t) in_ptr = mli_prv_tensor_data_ptr<MLI_PTR(int8_t)>(in);
    const MLI_PTR(int8_t) slope_ptr = mli_prv_tensor_data_ptr<MLI_PTR(int8_t)>(slope_coeff);
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
    int axis_shape = slope_coeff->shape[cfg->axis];
    int axis_in_mem_stride = in_prv.mem_stride[cfg->axis];
    int axis_out_mem_stride = out_prv.mem_stride[cfg->axis];
    /* Broadcasting in case axis is not inner most dim */
    bool broadcasting = !(cfg->axis == (in_prv.rank - 1));

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

    /* Input Zero Point */
    int16_t in_zp = in->el_params.sa.zero_point.mem.i16;

    if (broadcasting) {
        /* Get Non Axis Tensor */
        auto in_non_axis_prv  = mli_prv_get_non_axis_tensor<MLI_PTR(int8_t)>(&in_prv,  cfg->axis);
        auto out_non_axis_prv = mli_prv_get_non_axis_tensor<MLI_PTR(int8_t)>(&out_prv, cfg->axis);

        /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
        mli_prv_reorder_generic_tensor<MLI_PTR(int8_t)>(&in_non_axis_prv );
        mli_prv_reorder_generic_tensor<MLI_OUT_PTR(int8_t)>(&out_non_axis_prv);

        int remaining_part = in_non_axis_prv.shape[3] & (num_lanes - 1);

        for (int scale_idx = 0; scale_idx < axis_shape; scale_idx++) {
            /* Define Sub Tensor */
            vec_in  = (MLI_PTR(int8_t))in_ptr  + scale_idx * axis_in_mem_stride;
            vec_out = out_ptr + scale_idx * axis_out_mem_stride;
            /* Load Scale Elem */
            auto scale_v = mli_prv_init_v<int8_t, decltype(input)>(slope_ptr[scale_idx]);
            auto alpha_params = mli::krn::prelu_define_requant_params(in, slope_coeff, out, scale_v, &identity_params);

            /* Loop Over Sub Tensor */
            const MLI_PTR(int8_t) orig_vec_in = vec_in;
            MLI_OUT_PTR(int8_t) orig_vec_out = vec_out;
            for (int pos1 = 0; pos1 < in_non_axis_prv.shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_non_axis_prv.shape[2]; pos2++) {
                    vec_in  = (MLI_PTR(int8_t))orig_vec_in  + POS(&in_non_axis_prv, 0, pos1, pos2, 0);
                    vec_out = orig_vec_out + POS(&out_non_axis_prv, 0, pos1, pos2, 0);
                    if (remaining_part) {
                        mli::krn::compute_prelu(vec_in, vec_out, in_zp, &identity_params, &alpha_params,
                                                remaining_part);
                        vec_in  += remaining_part;
                        vec_out += remaining_part;
                    }
                    for (int pos3 = remaining_part; pos3 < in_non_axis_prv.shape[3]; pos3 += num_lanes) {
                        mli::krn::compute_prelu(vec_in, vec_out, in_zp, &identity_params, &alpha_params);
                        vec_in  += num_lanes;
                        vec_out += num_lanes;
                    }
                }
            }
        }
    } else {
        /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
        mli_prv_reorder_generic_tensor<MLI_PTR(int8_t)>(&in_prv );
        mli_prv_reorder_generic_tensor<MLI_OUT_PTR(int8_t)>(&out_prv);

        int remaining_part = in_prv.shape[3] & (num_lanes - 1);

        if (remaining_part) {
            /* Load Scale Vector */
            auto scale_v = mli_prv_load_1vec(slope_ptr);
            auto alpha_params = mli::krn::prelu_define_requant_params(in, slope_coeff, out, scale_v, &identity_params);

            mli::krn::compute_prelu_no_broadcast(in_ptr, out_ptr, in_zp, &identity_params, &alpha_params,
                                                 in_prv, out_prv, remaining_part);
        }
        for (int scale_idx = remaining_part; scale_idx < axis_shape; scale_idx += num_lanes) {
            /* Define Sub Tensor */
            vec_in  = (MLI_PTR(int8_t))in_ptr  + scale_idx * axis_in_mem_stride;
            vec_out = out_ptr + scale_idx * axis_out_mem_stride;
            /* Load Scale Vector */
            auto scale_v = mli_prv_load_1vec(&slope_ptr[scale_idx]);
            auto alpha_params = mli::krn::prelu_define_requant_params(in, slope_coeff, out, scale_v, &identity_params);

            mli::krn::compute_prelu_no_broadcast(vec_in, vec_out, in_zp, &identity_params, &alpha_params,
                                                 in_prv, out_prv);
        }
    }

    return MLI_STATUS_OK;
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_PRELU_REF_H_
