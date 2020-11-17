/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_SOFTMAX_DSP_H_
#define _MLI_KRN_SOFTMAX_DSP_H_

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_lut.h"
#include "mli_prv_activation_lut.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace dsp {

const int kSoftmaxAsymZeroPoint = -128;
const int kSoftmaxOutputShift = 8;

template <typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_subtract_max(
        const MLI_PTR(io_T) orig_vec_in, 
        MLI_PTR(io_T) orig_vec_out,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *in_prv,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *out_prv,
        int *in_frac_p) {

    MLI_PTR(io_T) vec_in  = (MLI_PTR(io_T))orig_vec_in;
    MLI_PTR(io_T) vec_out = orig_vec_out;
    
    // look for max & min values
    v2q15_t one_val = mli_prv_load_1_sample(vec_in);
    v2q15_t max_val = mli_prv_init_v(one_val[0]);
    v2q15_t min_val = mli_prv_init_v(one_val[0]);

    for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                vec_in  = (MLI_PTR(io_T))orig_vec_in + POS(in_prv,  pos0, pos1, pos2, 0);
                if (in_prv->shape[3] & 1) {
                    v2q15_t one_val = mli_prv_init_v(mli_prv_load_1_sample(vec_in)[0]);
                    max_val = mli_math_max_fx(max_val, one_val);
                    min_val = mli_math_min_fx(min_val, one_val);
                    vec_in += 1;
                }
                for (int pos3 = 0; pos3 < in_prv->shape[3] >> 1; pos3++) {
                    v2q15_t val = mli_prv_load_2_samples(vec_in);
                    max_val = mli_math_max_fx(max_val, val);
                    min_val = mli_math_min_fx(min_val, val);
                    vec_in += 2;
                }
            }
        }
    }

    max_val = mli_prv_init_v(mli_math_max_fx(max_val[0], max_val[1]));
    min_val = mli_prv_init_v(mli_math_min_fx(min_val[0], min_val[1]));
    // reset input data pointer
    vec_in = (MLI_PTR(io_T))orig_vec_in;

    // Subtract maximum from each element
    // free one more bit if saturation is expected.
    const int biased_min = static_cast<int>(min_val[0]) - static_cast<int>(max_val[0]);
    const int min_limit = -(1 << ((sizeof(io_T) * 8) - 1));
    if (biased_min < min_limit) {
        max_val = mli_math_acc_ashift_fx(max_val, 1);
        for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                    vec_in  = (MLI_PTR(io_T))orig_vec_in + POS(in_prv,  pos0, pos1, pos2, 0);
                    vec_out = orig_vec_out + POS(out_prv, pos0, pos1, pos2, 0);
                    if (in_prv->shape[3] & 1) {
                        mli_prv_store_1_sample(vec_out, mli_math_sub_fx(
                            mli_math_acc_ashift_fx(mli_prv_load_1_sample(vec_in), 1), max_val));
                        vec_in  += 1;
                        vec_out += 1;
                    }
                    for (int pos3 = 0; pos3 < in_prv->shape[3] >> 1; pos3++) {
                        mli_prv_store_2_samples(vec_out, mli_math_sub_fx(
                            mli_math_acc_ashift_fx(mli_prv_load_2_samples(vec_in), 1), max_val));
                        vec_in  += 2;
                        vec_out += 2;
                    }
                }
            }
        }
        *in_frac_p -= 1;
    } else {
        for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                    vec_in  = (MLI_PTR(io_T))orig_vec_in + POS(in_prv,  pos0, pos1, pos2, 0);
                    vec_out = orig_vec_out + POS(out_prv, pos0, pos1, pos2, 0);
                    if (in_prv->shape[3] & 1) {
                        mli_prv_store_1_sample(vec_out, mli_math_sub_fx(
                            mli_prv_load_1_sample(vec_in), max_val));
                        vec_in  += 1;
                        vec_out += 1;
                    }
                    for (int pos3 = 0; pos3 < in_prv->shape[3] >> 1; pos3++) {
                        mli_prv_store_2_samples(vec_out, mli_math_sub_fx(
                            mli_prv_load_2_samples(vec_in), max_val));
                        vec_in  += 2;
                        vec_out += 2;
                    }
                }
            }
        }
    }
}

template <typename io_T, bool convert = false>
static MLI_FORCE_INLINE mli_acc40_t sumTensor(const MLI_PTR(io_T) orig_vec_in,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *in_prv,
        const struct s8asym_quant_params *in_params = nullptr,
        struct s8asym_quant_params *out_params = nullptr) {


    const v2q15_t one_v = {1, 1};
    MLI_PTR(io_T) vec_in = (MLI_PTR(io_T))orig_vec_in;

    // Accumulation through MAC and reciprocal calculation
    mli_acc40_t sum_acc = mli_math_mul_fx<int16_t, mli_acc40_t>(0, 0);

    if (convert) {
        for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                    vec_in  = (MLI_PTR(io_T))orig_vec_in + POS(in_prv,  pos0, pos1, pos2, 0);
                    if (in_prv->shape[3] & 1) {
                        /* activation_lut */
                        v2q15_t input = mli_prv_load_1_sample(vec_in);
                        v2q15_t res = mli::krn::activation_lut_two_elem_interpolate<int16_t, true, false>
                                (input, &expneg_lut_fx16, 0, in_params, out_params);

                        /* Accumulation through MAC and reciprocal calculation */
                        res[1] = 0; // Unused
                        sum_acc = mli_math_mac_fx(sum_acc, res, one_v);
                        vec_in += 1;
                    }
                    for (int pos3 = 0; pos3 < in_prv->shape[3] >> 1; pos3++) {
                        /* activation_lut */
                        v2q15_t input = mli_prv_load_2_samples(vec_in);
                        v2q15_t res = mli::krn::activation_lut_two_elem_interpolate<int16_t, true, false>
                                (input, &expneg_lut_fx16, 0, in_params, out_params);

                        /* Accumulation through MAC and reciprocal calculation */
                        sum_acc = mli_math_mac_fx(sum_acc, res, one_v);
                        vec_in += 2;
                    }
                }
            }
        }
    } else {
        for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                    vec_in  = (MLI_PTR(io_T))orig_vec_in + POS(in_prv,  pos0, pos1, pos2, 0);
                    if (in_prv->shape[3] & 1) {
                        sum_acc = mli_math_mac_fx(sum_acc, mli_prv_load_1_sample(vec_in), one_v);
                        vec_in  += 1;
                    }
                    for (int pos3 = 0; pos3 < in_prv->shape[3] >> 1; pos3++) {
                        sum_acc = mli_math_mac_fx(sum_acc, mli_prv_load_2_samples(vec_in), one_v);
                        vec_in  += 2;
                    }
                }
            }
        }
    }

    return sum_acc;
}

template <typename io_T>
static MLI_FORCE_INLINE void normalizeTensor(MLI_PTR(io_T) orig_vec_out,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *out_prv,
        v2q15_t sum_recip,
        int shift) {

    MLI_PTR(io_T) vec_out = orig_vec_out;
    // final result: normalizing
    for (int pos0 = 0; pos0 < out_prv->shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < out_prv->shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < out_prv->shape[2]; pos2++) {
                vec_out = orig_vec_out + POS(out_prv, pos0, pos1, pos2, 0);
                if (out_prv->shape[3] & 1) {
                    v2accum40_t tmp_acc = mli_math_mul_fx<v2q15_t, v2accum40_t>(sum_recip, mli_prv_load_1_sample(vec_out));
                    mli_prv_store_1_sample(vec_out, mli_math_acc_cast_fx<v2q15_t, v2accum40_t>(tmp_acc,  shift));
                    vec_out += 1;
                }
                for (int pos3 = 0; pos3 < out_prv->shape[3] >> 1; pos3++) {
                    v2accum40_t tmp_acc = mli_math_mul_fx<v2q15_t, v2accum40_t>(sum_recip, mli_prv_load_2_samples(vec_out));
                    mli_prv_store_2_samples(vec_out, mli_math_acc_cast_fx<v2q15_t, v2accum40_t>(tmp_acc, shift));
                    vec_out += 2;
                }
            }
        }
    }
}

template <typename io_T>
static MLI_FORCE_INLINE int8_t mli_krn_softmax_get_max(
        struct generic_tensor_private_t<MLI_PTR(io_T)> *in_prv,
        const MLI_PTR(io_T) orig_vec_in)

{
    MLI_PTR(io_T) vec_in  = (MLI_PTR(io_T))orig_vec_in;
    // look for max value
    v2q15_t one_val = mli_prv_load_1_sample(vec_in);
    v2q15_t max_val = mli_prv_init_v(one_val[0]);

    for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                vec_in  = (MLI_PTR(io_T))orig_vec_in + POS(in_prv,  pos0, pos1, pos2, 0);
                if (in_prv->shape[3] & 1) {
                    v2q15_t one_val = mli_prv_init_v(mli_prv_load_1_sample(vec_in)[0]);
                    max_val = mli_math_max_fx(max_val, one_val);
                    vec_in += 1;
                }
                for (int pos3 = 0; pos3 < in_prv->shape[3] >> 1; pos3++) {
                    v2q15_t val = mli_prv_load_2_samples(vec_in);
                    max_val = mli_math_max_fx(max_val, val);
                    vec_in += 2;
                }
            }
        }
    }

    return mli_math_max_fx(max_val[0], max_val[1]);
}

template <typename io_T>
static mli_status mli_krn_softmax_fx_run(const mli_tensor *in, const mli_softmax_cfg* cfg,
        mli_tensor *out) {

    MLI_ASSERT(MLI_MAX_RANK == 4);

    const MLI_PTR(io_T) vec_in = nullptr;
    MLI_PTR(io_T) vec_out = nullptr;

    const MLI_PTR(io_T) in_ptr = (MLI_PTR(io_T))(in->data.mem.void_p);
    MLI_PTR(io_T) out_ptr = (MLI_PTR(io_T))(out->data.mem.void_p);

    /* Copy tensor format */
    mli_prv_copy_tensor_format_except_mem_strides(in, out);
    out->el_params.fx.frac_bits = (sizeof(io_T) * 8) - kTransfFuncIntBits - 1;

    /* Get Generic Private Tensor */
    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);
    auto out_prv = mli_prv_get_generic_tensor<MLI_PTR(io_T)>(out);
    /* Get Non Axis Tensor */
    auto in_non_axis_prv  = mli_prv_get_non_axis_tensor<MLI_PTR(io_T)>(&in_prv,  cfg->axis);
    auto out_non_axis_prv = mli_prv_get_non_axis_tensor<MLI_PTR(io_T)>(&out_prv, cfg->axis);
    /* Get Axis Tensor */
    in_prv  = mli_prv_get_axis_tensor<MLI_PTR(io_T)>(&in_prv,  cfg->axis);
    out_prv = mli_prv_get_axis_tensor<MLI_PTR(io_T)>(&out_prv, cfg->axis);
    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&in_prv );
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&out_prv);

    int in_frac = static_cast<int>(in->el_params.fx.frac_bits);

    /* For applying the function to specific axis dimension, we should first loop across other dimensions then process
     * axis dimension elements.
     * For applying the function to the whole tensor, loop body is executed only one time. (i.e. shape[i] = 1).
     */
    for (int dim0 = 0; dim0 < in_non_axis_prv.shape[0]; dim0++) {
        for (int dim1 = 0; dim1 < in_non_axis_prv.shape[1]; dim1++) {
            for (int dim2 = 0; dim2 < in_non_axis_prv.shape[2]; dim2++) {

                vec_in = &in_ptr[dim0 * in_non_axis_prv.mem_stride[0] + 
                                 dim1 * in_non_axis_prv.mem_stride[1] + 
                                 dim2 * in_non_axis_prv.mem_stride[2]];
                vec_out = &out_ptr[dim0 * out_non_axis_prv.mem_stride[0] + 
                                   dim1 * out_non_axis_prv.mem_stride[1] + 
                                   dim2 * out_non_axis_prv.mem_stride[2]];

                /* Subtract maximum from each element */
                mli_krn_softmax_subtract_max(vec_in, vec_out, &in_prv, &out_prv, &in_frac);

                /* Activation lookup table */
                struct generic_tensor_private_t<MLI_PTR(io_T)> out_vec_tensor = out_prv;
                out_vec_tensor.ptr = vec_out;
                mli::krn::activation_lut<io_T, false>(&out_vec_tensor, &out_vec_tensor, &expneg_lut_fx16, in_frac);

                /* Accumulation through MAC and reciprocal calculation */
                mli_acc40_t sum_acc = sumTensor<io_T>(vec_out, &out_prv);

                int sum_exp = mli_math_norm_fx<mli_acc40_t, int>(sum_acc);
    
                io_T sum_mnt = mli_math_acc_cast_fx<io_T, mli_acc40_t>(sum_acc, 16 - sum_exp);
                /* sum_mnt is normalized (that is inside [0.5, 1) range)
                 * so we use Q30(0.5) as a dividend to get Q15 result inside (0.5, 1)
                 * saturation prevents it from reaching 1
                 */
                v2q15_t sum_recip = mli_prv_init_v((int16_t)mli_math_sat_fx<int32_t>((1L << 29) / sum_mnt, 16));

                /* sum_recip * vec_out[idx] = Q15 * Q15 (default LUT output) */
                int lut_frac_bits = expneg_lut_fx16.out_frac_bits * 2;
                /* 15 - sum_exp: sum_of_exps overhead */
                int sum_exp_overhead = kMaxFracBitsFx16 - sum_exp;
                /* Normalize Output */
                normalizeTensor<io_T>(vec_out, &out_prv, sum_recip, 
                                      lut_frac_bits + sum_exp_overhead - out->el_params.fx.frac_bits);
            }
        }
    }

    return MLI_STATUS_OK;
}

static mli_status mli_krn_softmax_sa8_run(const mli_tensor *in, const mli_softmax_cfg* cfg,
        mli_tensor *out) {
    
    MLI_ASSERT(MLI_MAX_RANK == 4);

    struct s8asym_quant_params in_params;
    struct s8asym_quant_params out_params;

    in_params.scale  = in->el_params.sa.scale.mem.i16;
    in_params.shift = in->el_params.sa.scale_frac_bits.mem.i8;
    out_params.offset = kSoftmaxAsymZeroPoint;
    out_params.scale  = 1;
    out_params.shift = kSoftmaxOutputShift;

    const MLI_PTR(int8_t) vec_in  = nullptr;
    MLI_PTR(int8_t) vec_out = nullptr;

    const MLI_PTR(int8_t) in_ptr = (MLI_PTR(int8_t))(in->data.mem.void_p);
    MLI_PTR(int8_t) out_ptr = (MLI_PTR(int8_t)) (out->data.mem.void_p);

    /* Copy tensor format */
    mli_prv_copy_tensor_format_except_mem_strides(in, out);
    
    /* Get Generic Private Tensor */
    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(int8_t)>(in);
    auto out_prv = mli_prv_get_generic_tensor<MLI_PTR(int8_t)>(out);
    /* Get Non Axis Tensor */
    auto in_non_axis_prv  = mli_prv_get_non_axis_tensor<MLI_PTR(int8_t)>(&in_prv,  cfg->axis);
    auto out_non_axis_prv = mli_prv_get_non_axis_tensor<MLI_PTR(int8_t)>(&out_prv, cfg->axis);
    /* Get Axis Tensor */
    in_prv  = mli_prv_get_axis_tensor<MLI_PTR(int8_t)>(&in_prv,  cfg->axis);
    out_prv = mli_prv_get_axis_tensor<MLI_PTR(int8_t)>(&out_prv, cfg->axis);
    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(int8_t)>(&in_prv );
    mli_prv_reorder_generic_tensor<MLI_PTR(int8_t)>(&out_prv);

    /* For applying the function to specific axis dimension, we should first loop across other dimensions then process
     * axis dimension elements.
     * For applying the function to the whole tensor, loop body is executed only one time. (i.e. shape[i] = 1).
     */
    for (int dim0 = 0; dim0 < in_non_axis_prv.shape[0]; dim0++) {
        for (int dim1 = 0; dim1 < in_non_axis_prv.shape[1]; dim1++) {
            for (int dim2 = 0; dim2 < in_non_axis_prv.shape[2]; dim2++) {

                vec_in = &in_ptr[dim0 * in_non_axis_prv.mem_stride[0] + 
                                 dim1 * in_non_axis_prv.mem_stride[1] + 
                                 dim2 * in_non_axis_prv.mem_stride[2]];
                vec_out = &out_ptr[dim0 * out_non_axis_prv.mem_stride[0] + 
                                   dim1 * out_non_axis_prv.mem_stride[1] + 
                                   dim2 * out_non_axis_prv.mem_stride[2]];

                /* Subtract maximum from each input tensor element.
                 * This subtraction is done by overwriting offset with max_value.
                 * 1. Offset value is not needed here due to subtraction operation:
                 *    (in_value + offset) - (max_value + offset) = in_value - max_value
                 * 2. Replace in_params.offset with max_value 
                 */
                in_params.offset = mli_krn_softmax_get_max(&in_prv, vec_in);
                
				/* Sum the input tensor after convert it to FX16 */               
                mli_acc40_t sum_acc = sumTensor<int8_t, true>(vec_in, &in_prv, &in_params, &out_params);

                int sum_exp = mli_math_norm_fx<mli_acc40_t, int>(sum_acc);
                int16_t sum_mnt = mli_math_acc_cast_fx<int16_t, mli_acc40_t>(sum_acc, 16 - sum_exp);
                /* sum_mnt is normalized (that is inside [0.5, 1) range)
                 * so we use Q30(0.5) as a dividend to get Q15 result inside (0.5, 1)
                 * saturation prevents it from reaching 1
                 */
                v2q15_t sum_recip = mli_prv_init_v((int16_t)mli_math_sat_fx<int32_t>((1L << 29) / sum_mnt, 16)); 
                /* sum_recip * vec_out[idx] = Q15 * Q15 (default LUT output) */
                int lut_frac_bits = expneg_lut_fx16.out_frac_bits * 2;
                /* 15 - sum_exp: sum_of_exps overhead */
                int sum_exp_overhead = kMaxFracBitsFx16 - sum_exp;
				/* Output Scale Shift Value */
                int shift = lut_frac_bits + sum_exp_overhead - out_params.shift;

                const MLI_PTR(int8_t) orig_vec_in = vec_in;
                MLI_PTR(int8_t) orig_vec_out = vec_out;
                for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
                    for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                        for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                            vec_in  = (MLI_PTR(int8_t))orig_vec_in  + POS(&in_prv,  pos0, pos1, pos2, 0);
                            vec_out = orig_vec_out + POS(&out_prv, pos0, pos1, pos2, 0);
                            if(in_prv.shape[3] & 1) {
                                /* activation_lut */
                                v2q15_t input = mli_prv_load_1_sample(vec_in);
                                input = mli::krn::activation_lut_two_elem_interpolate<int16_t, true, false>
                                        (input, &expneg_lut_fx16, 0, &in_params, &out_params);

                                /* Multiply with Reciprocal of Sum */
                                v2accum40_t tmp_acc = mli_math_mul_fx<v2q15_t, v2accum40_t>(sum_recip, input);
                                
                                input = mli_prv_convert_fx16_sa8<v2accum40_t, v2q15_t>(tmp_acc, out_params.offset, shift);
                                mli_prv_store_1_sample(vec_out, input);
                                vec_in  += 1;
                                vec_out += 1;
                            }
                            for (int pos3 = 0; pos3 < in_prv.shape[3] >> 1; pos3++) {
                                /* activation_lut */
                                v2q15_t input = mli_prv_load_2_samples(vec_in);
                                input = mli::krn::activation_lut_two_elem_interpolate<int16_t, true, false>
                                        (input, &expneg_lut_fx16, 0, &in_params, &out_params);

                                /* Multiply with Reciprocal of Sum */
                                v2accum40_t tmp_acc = mli_math_mul_fx<v2q15_t, v2accum40_t>(sum_recip, input);
                                input = mli_prv_convert_fx16_sa8<v2accum40_t, v2q15_t>(tmp_acc, out_params.offset, shift);
                                mli_prv_store_2_samples(vec_out, input);
                                vec_in  += 2;
                                vec_out += 2;
                            }
                        }
                    }
                }
            }
        }
    }

    out->el_params.sa.zero_point.mem.i16 = out_params.offset;
    out->el_params.sa.scale.mem.i16 = out_params.scale;
    out->el_params.sa.scale_frac_bits.mem.i8 = (int8_t)out_params.shift;

    return MLI_STATUS_OK;
}

} // namespace dsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_SOFTMAX_DSP_H_
