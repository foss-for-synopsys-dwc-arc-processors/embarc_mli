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
#include "mli_mem_info.h"
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

    const MLI_PTR(io_T) in_ptr = mli_prv_tensor_data_ptr<MLI_PTR(io_T)>(in);
    const MLI_PTR(io_T) slope_ptr = mli_prv_tensor_data_ptr<MLI_PTR(io_T)>(slope_coeff);
    MLI_OUT_PTR(io_T) out_ptr = mli_prv_tensor_data_ptr<MLI_OUT_PTR(io_T)>(out);

    /* Get Generic Private Tensor */
    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);    
    auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(io_T)>(out);

    /* Define Slope Axis Params */
    int axis_shape = slope_coeff->shape[cfg->axis];
    int axis_in_mem_stride = in_prv.mem_stride[cfg->axis];
    int axis_out_mem_stride = out_prv.mem_stride[cfg->axis];
    /* Broadcasting in case axis is not inner most dim */
    bool broadcasting = !(cfg->axis == (in_prv.rank - 1));

    int shift = mli_prv_calc_shift(in, slope_coeff, out);

    if (broadcasting) {
        mli::krn::compute_prelu_broadcast<io_T>(in_prv, out_prv, slope_ptr, cfg->axis,
                                                axis_shape, axis_in_mem_stride, axis_out_mem_stride, shift);
    } else {
        /* Dummy Load to get num_lanes */
        auto input = mli_prv_load_1vec(in_ptr);
        int num_lanes = get_number_lanes(input);

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
            const MLI_PTR(io_T) vec_in  = (MLI_PTR(io_T))in_ptr  + scale_idx * axis_in_mem_stride;
            MLI_OUT_PTR(io_T) vec_out = out_ptr + scale_idx * axis_out_mem_stride;
            /* Load Scale Vector */
            auto scale_v = mli_prv_load_1vec(&slope_ptr[scale_idx]);
            mli::krn::compute_prelu_no_broadcast<io_T, decltype(input)>(vec_in, vec_out, scale_v,
                                                                        shift, in_prv, out_prv);
        }
    }
    
    return MLI_STATUS_OK;
}

static MLI_FORCE_INLINE s8asym_quant_params prelu_define_requant_alpha_params(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        mli_tensor *out,
        const int8_t alpha_sa8,
        const s8asym_quant_params *identity_params) {
    
    return mli::krn::ref::leaky_relu_define_alpha_params(in, slope_coeff, out, alpha_sa8, identity_params);
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
                for (int pos3 = remaining_part; pos3 < in_non_axis_prv.shape[3]; pos3 += num_lanes) {
                    mli::krn::compute_leaky_relu(vec_in, vec_out, in_zp, identity_params, &alpha_params);
                    vec_in  += num_lanes;
                    vec_out += num_lanes;
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
     *    check prelu_define_requant_alpha_params for more Documentation
     * ***************************************************************************************************************/
    s8asym_quant_params identity_params;
    /* Define Requantization Params for In/Out scale ratio */
    leaky_relu_define_identity_params(in, out, &identity_params);

    /* Input Zero Point */
    int16_t in_zp = in->el_params.sa.zero_point.mem.i16;

    if (broadcasting) {
        mli::krn::compute_prelu_broadcast(in, slope_coeff, out, in_prv, out_prv, slope_ptr,
                                          cfg->axis, axis_shape, axis_in_mem_stride, axis_out_mem_stride,
                                          in_zp, &identity_params);
    } else {
        /* Dummy Load to get num_lanes */
        auto input = mli_prv_load_1vec(in_ptr);
        int num_lanes = get_number_lanes(input);

        /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
        mli_prv_reorder_generic_tensor<MLI_PTR(int8_t)>(&in_prv );
        mli_prv_reorder_generic_tensor<MLI_OUT_PTR(int8_t)>(&out_prv);

        int remaining_part = in_prv.shape[3] & (num_lanes - 1);

        if (remaining_part) {
            /* Load Scale Vector */
            auto scale_v = mli_prv_load_1vec(slope_ptr);
            auto alpha_params = mli::krn::prelu_define_requant_alpha_params(in, slope_coeff, out, scale_v, &identity_params);

            mli::krn::compute_prelu_no_broadcast(in_ptr, out_ptr, in_zp, &identity_params, &alpha_params,
                                                 in_prv, out_prv, remaining_part);
        }
        for (int scale_idx = remaining_part; scale_idx < axis_shape; scale_idx += num_lanes) {
            /* Define Sub Tensor */
            vec_in  = (MLI_PTR(int8_t))in_ptr  + scale_idx * axis_in_mem_stride;
            vec_out = out_ptr + scale_idx * axis_out_mem_stride;
            /* Load Scale Vector */
            auto scale_v = mli_prv_load_1vec(&slope_ptr[scale_idx]);
            auto alpha_params = mli::krn::prelu_define_requant_alpha_params(in, slope_coeff, out, scale_v, &identity_params);

            mli::krn::compute_prelu_no_broadcast(vec_in, vec_out, in_zp, &identity_params, &alpha_params,
                                                 in_prv, out_prv);
        }
    }

    return MLI_STATUS_OK;
}

template <typename o_T>
static MLI_FORCE_INLINE o_T scale_value(
        const int32_t in_val,
        const int32_t in_bias,
        const o_T out_bias,
        const int16_t scale,
        const int shift_right) {
    constexpr int max_shift_right = 63;
    constexpr int max_shift_left = -63;
    int32_t shift = MAX(max_shift_left, MIN(shift_right, max_shift_right));

    int32_t value = mli_math_sub_fx(in_val, in_bias);
    int64_t scaled_value =
    mli_math_mul_fx<int32_t, int64_t> (value, static_cast<int32_t>(scale));
    scaled_value = mli_math_ashift_right_fx(scaled_value, shift);
    scaled_value = mli_math_add_fx(scaled_value, static_cast<int64_t>(out_bias));
    o_T result = mli_math_cast_fx<int64_t, o_T>(scaled_value, 0);
    return result;
}

template <typename i_T, typename o_T>
static MLI_FORCE_INLINE void compute_prelu_per_axis(
        const generic_tensor_private_t<MLI_PTR(i_T)> &in_data,
        const int32_t prelu_axis,
        const i_T *in_bias,
        const o_T *out_bias,
        const int16_t *posscale,
        const int16_t *negscale,
        const int8_t *posshift,
        const int8_t *negshift,
        generic_tensor_private_t<MLI_OUT_PTR(o_T)> &out_data) {

    MLI_ASSERT(prelu_axis < (int32_t)kPreluRank);
    MLI_ASSERT(kPreluRank <=
            sizeof(in_data.shape) / sizeof(in_data.shape[0]));
    MLI_ASSERT(kPreluRank
            <= sizeof(out_data.shape) / sizeof(out_data.shape[0]));

    int pos[kPreluRank] = {0};
    for (pos[0] = 0; pos[0] < in_data.shape[0]; pos[0]++) {
        for (pos[1] = 0; pos[1] < in_data.shape[1]; pos[1]++) {
            for (pos[2] = 0; pos[2] < in_data.shape[2]; pos[2]++) {
                for (pos[3] = 0; pos[3] < in_data.shape[3]; pos[3]++) {
                    const int param_idx = pos[prelu_axis];
                    i_T in_val = mli_prv_tensor_read(in_data, pos[0], pos[1], pos[2], pos[3]);
                    o_T out_val;
                    if(in_val - in_bias[param_idx] >= 0) {
                        out_val = scale_value(in_val, in_bias[param_idx],
                                              out_bias[param_idx], posscale[param_idx],
                                              posshift[param_idx]);
                    } else {
                        out_val = scale_value(in_val, in_bias[param_idx],
                                              out_bias[param_idx], negscale[param_idx],
                                              negshift[param_idx]);
                    }
                    mli_prv_tensor_write(out_val, out_data, pos[0], pos[1], pos[2], pos[3]);
                }
            }
        }
    }
}

template <typename i_T, typename o_T>
static MLI_FORCE_INLINE void compute_prelu_per_tensor(
        const generic_tensor_private_t<MLI_PTR(i_T)> &in_data,
        const i_T in_bias,
        const o_T out_bias,
        const int16_t posscale,
        const int16_t negscale,
        const int8_t posshift,
        const int8_t negshift,
        generic_tensor_private_t<MLI_OUT_PTR(o_T)> &out_data) {

    MLI_ASSERT(kPreluRank <=
            sizeof(in_data.shape) / sizeof(in_data.shape[0]));
    MLI_ASSERT(kPreluRank
            <= sizeof(out_data.shape) / sizeof(out_data.shape[0]));

    int pos[kPreluRank] = {0};
    for (pos[0] = 0; pos[0] < in_data.shape[0]; pos[0]++) {
        for (pos[1] = 0; pos[1] < in_data.shape[1]; pos[1]++) {
            for (pos[2] = 0; pos[2] < in_data.shape[2]; pos[2]++) {
                for (pos[3] = 0; pos[3] < in_data.shape[3]; pos[3]++) {
                    i_T in_val = mli_prv_tensor_read(in_data, pos[0], pos[1], pos[2], pos[3]);
                    o_T out_val;
                    if(in_val - in_bias >= 0) {
                        out_val = scale_value(in_val, in_bias,
                                              out_bias, posscale, posshift);
                    } else {
                        out_val = scale_value(in_val, in_bias,
                                              out_bias, negscale, negshift);
                    }
                    mli_prv_tensor_write(out_val, out_data, pos[0], pos[1], pos[2], pos[3]);
                }
            }
        }
    }
}

template <typename i_T, typename o_T>
mli_status MLI_FORCE_INLINE mli_krn_prelu(Tensor<InternalBuffer, kPreluRank> &in,
                                          InternalBuffer &bias_in,
                                          InternalBuffer &posscale,
                                          InternalBuffer &negscale,
                                          InternalBuffer &posshift,
                                          InternalBuffer &negshift,
                                          InternalBuffer &bias_out,
                                          const int32_t prelu_axis,
                                          Tensor<InternalBuffer, kPreluRank> &out) {
    const auto in_prv = mli_prv_get_generic_tensor_internal<MLI_PTR(i_T)>(in);
    auto out_prv = mli_prv_get_generic_tensor_internal<MLI_OUT_PTR(o_T)>(out);
    if (prelu_axis < 0) {
        // Asserts are in checkers
        const i_T in_bias_val = bias_in.read<i_T>(0);
        const o_T out_bias_val = bias_out.read<o_T>(0);

        const int16_t posscale_val = posscale.read<int16_t>(0);
        const int16_t negscale_val = negscale.read<int16_t>(0);
        const int8_t posshift_val = posshift.read<int8_t>(0);
        const int8_t negshift_val = negshift.read<int8_t>(0);
        
        compute_prelu_per_tensor(in_prv, in_bias_val, out_bias_val,
                                   posscale_val, negscale_val, posshift_val, negshift_val, out_prv);
    } else {
        // Asserts are in checkers
        const i_T *in_bias_ptr = bias_in.get_ptr<i_T>();
        const o_T *out_bias_ptr = bias_out.get_ptr<o_T>();

        const int8_t *posshift_ptr = posshift.get_ptr<int8_t>();
        const int8_t *negshift_ptr = negshift.get_ptr<int8_t>();
        const int16_t *posscale_ptr = posscale.get_ptr<int16_t>();
        const int16_t *negscale_ptr = negscale.get_ptr<int16_t>();
        compute_prelu_per_axis(in_prv, prelu_axis, in_bias_ptr, out_bias_ptr,
                                 posscale_ptr, negscale_ptr, posshift_ptr, negshift_ptr, out_prv);
    }

    return MLI_STATUS_OK;
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_PRELU_REF_H_
