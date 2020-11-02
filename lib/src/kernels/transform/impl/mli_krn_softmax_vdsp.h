/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_SOFTMAX_VDSP_H_
#define _MLI_KRN_SOFTMAX_VDSP_H_

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
#include "mli_prv_load_store.h"

namespace mli {
namespace krn {
namespace vdsp {

const int kSoftmaxAsymZeroPoint = -128;
const int kSoftmaxOutputShift = 8;

template <typename T>
static MLI_FORCE_INLINE void calculate_steps(int* steps, generic_tensor_private_t<T> prv_tsr, int vec_size){
    steps[3] = vec_size * prv_tsr.mem_stride[3];
    steps[2] = prv_tsr.mem_stride[2] - steps[3] * (prv_tsr.shape[3] / vec_size);
    steps[1] = prv_tsr.mem_stride[1] - steps[2] * prv_tsr.shape[2];
    steps[0] = prv_tsr.mem_stride[0] - steps[1] * prv_tsr.shape[1];
}

template <typename io_T, typename pred_T>
static MLI_FORCE_INLINE void mli_krn_softmax_subtract_max(const MLI_PTR(io_T) vec_in, MLI_PTR(io_T) vec_out,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *in_prv,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *out_prv,
        int *in_frac_p, int* in_step, int* out_step, 
        pred_T predicate) {
    const MLI_PTR(io_T) vec_in_begin = vec_in;
    
    auto curr_vec = mli_prv_load_1vec(vec_in);
    typedef decltype(curr_vec) vec_T;
    int num_lanes = get_number_lanes<vec_T>();
    vec_T max_vec;
    vec_T min_vec;
    if (sizeof(io_T) == sizeof(int8_t)){
        max_vec = (vec_T) INT8_MIN;
        min_vec = (vec_T) INT8_MAX;
    }
    else {
        max_vec = (vec_T) INT16_MIN;
        min_vec = (vec_T) INT16_MAX;
    }
    // Looking for maximum value
    for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                curr_vec = mli_prv_load_1vec(vec_in);
                for (int pos3 = 1; pos3 <= (in_prv->shape[3] / num_lanes); pos3++) {
                    max_vec = mli_math_max_fx(max_vec, curr_vec);
                    min_vec = mli_math_min_fx(min_vec, curr_vec);
                    vec_in += in_step[3];
                    curr_vec = mli_prv_load_1vec(vec_in);
                }
                if ((in_prv->shape[3] & (num_lanes - 1)) != 0) {
                    curr_vec = mli_math_select_fx<vec_T, pred_T>(predicate, curr_vec, 
                            (vec_T) ((sizeof(io_T) == sizeof(int8_t)) ? INT8_MIN : INT16_MIN));
                    max_vec = mli_math_max_fx(max_vec, curr_vec);
                    curr_vec = mli_math_select_fx<vec_T, pred_T>(predicate, curr_vec, 
                            (vec_T) ((sizeof(io_T) == sizeof(int8_t)) ? INT8_MAX : INT16_MAX));
                    min_vec = mli_math_min_fx(min_vec, curr_vec);
                }
                vec_in += in_step[2];
            }
            vec_in += in_step[1];
        }
        vec_in += in_step[0];
    }
    vec_in = vec_in_begin;

    io_T max_val = mli_math_intra_max(max_vec);
    io_T min_val = mli_math_intra_min(min_vec);

    // Subtract maximum from each element
    // free one more bit if saturation is expected.
    const int biased_min = static_cast<int>(min_val) - max_val;
    const int min_limit = -(1 << ((sizeof(io_T) * 8) - 1));

    if (biased_min < min_limit) {
        max_val = max_val >> 1;
        for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                    curr_vec = mli_prv_load_1vec(vec_in);
                    for (int pos3 = 1; pos3 <= (in_prv->shape[3] / num_lanes); pos3++) {
                        vec_in += in_step[3];
                        curr_vec >>= 1;
                        curr_vec = mli_math_sub_fx(curr_vec, (vec_T) max_val);
                        mli_prv_store_n_samples(vec_out, curr_vec);
                        curr_vec = mli_prv_load_1vec(vec_in);
                        vec_out +=out_step[3];
                    }
                    if ((in_prv->shape[3] & (num_lanes - 1)) != 0) {
                        int remaining_part = in_prv->shape[3] & (num_lanes - 1);
                        curr_vec = mli_prv_load_1vec(vec_in);
                        curr_vec >>= 1;
                        curr_vec = mli_math_sub_fx(curr_vec, (vec_T) max_val);
                        mli_prv_store_n_samples(vec_out, curr_vec, remaining_part);
                    }
                    vec_in += in_step[2];
                    vec_out += out_step[2];
                }
                vec_in += in_step[1];
                vec_out += out_step[1];
            }
            vec_in += in_step[0];
            vec_out += out_step[0];
        }
        *in_frac_p -= 1;
    } else {
        for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                    curr_vec = mli_prv_load_1vec(vec_in);

                    for (int pos3 = 1; pos3 <= (in_prv->shape[3] / num_lanes); pos3++) {
                        vec_in += in_step[3];
                        curr_vec = mli_math_sub_fx(curr_vec, (vec_T) max_val);
                        mli_prv_store_n_samples(vec_out, curr_vec);
                        curr_vec = mli_prv_load_1vec(vec_in);
                        vec_out += out_step[3];
                    }
                    if ((in_prv->shape[3] & (num_lanes - 1)) != 0) {
                        int remaining_part = in_prv->shape[3] & (num_lanes - 1);
                        curr_vec = mli_prv_load_1vec(vec_in);
                        curr_vec = mli_math_sub_fx(curr_vec, (vec_T) max_val);
                        mli_prv_store_n_samples(vec_out, curr_vec, remaining_part);
                    }
                    vec_in += in_step[2];
                    vec_out += out_step[2];
                }
                vec_in += in_step[1];
                vec_out += out_step[1];
            }
            vec_in += in_step[0];
            vec_out += out_step[0];
        }
    }
}

template <typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_fx_run(const MLI_PTR(io_T) vec_in, MLI_PTR(io_T) vec_out, 
        generic_tensor_private_t<MLI_PTR(io_T)> in_prv, generic_tensor_private_t<MLI_PTR(io_T)> out_prv,
        int in_frac, int* in_step, int* out_step, int frac_bits) {
    MLI_PTR(io_T) vec_out_begin = vec_out;
    
    auto curr_vec = mli_prv_load_1vec(vec_in);
    typedef decltype(curr_vec) vec_T;
    int num_lanes = get_number_lanes<vec_T>();
    
    int remaining_part = in_prv.shape[3] & (num_lanes - 1);
    auto predicate = init_predicate(remaining_part, curr_vec);
    typedef decltype(predicate) pred_T;

    mli_krn_softmax_subtract_max(vec_in, vec_out, &in_prv, &out_prv, &in_frac, in_step, out_step, predicate);

    struct generic_tensor_private_t<MLI_PTR(io_T)> out_vec_tensor = out_prv;
    out_vec_tensor.ptr = vec_out;

    mli::krn::activation_lut<io_T, false>(&out_vec_tensor, &out_vec_tensor, &expneg_lut_fx16, in_frac);

    // Accumulation through MAC and reciprocal calculation
    auto sum_vec = mli_prv_init_accu(curr_vec, (io_T) 0);
    typedef decltype(sum_vec) acc_T;
    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                curr_vec = mli_prv_load_1vec(vec_out);
                for (int pos3 = 1; pos3 <= (in_prv.shape[3] / num_lanes); pos3++) {
                    sum_vec = mli_math_mac_fx(sum_vec, curr_vec, (vec_T) 1);
                    vec_out += out_step[3];
                    curr_vec = mli_prv_load_1vec(vec_out);
                }
                if ((in_prv.shape[3] & (num_lanes - 1)) != 0) {
                    curr_vec = mli_math_select_fx<vec_T, pred_T>(predicate, curr_vec, (vec_T) 0);
                    sum_vec = mli_math_mac_fx(sum_vec, curr_vec, (vec_T) 1);
                }
                vec_out += out_step[2];
            }
            vec_out += out_step[1];
        }
        vec_out += out_step[0];
    }
    vec_out = vec_out_begin;

    mli_acc32_t  sum_acc = mli_math_intra_sum(sum_vec);

    int sum_exp = mli_math_norm_fx<mli_acc32_t, mli_acc32_t>(sum_acc);
    io_T sum_mnt = mli_math_acc_cast_fx<io_T, mli_acc32_t>(sum_acc, 16 - sum_exp);
    // sum_mnt is normalized (that is inside [0.5, 1) range)
    // so we use Q30(0.5) as a dividend to get Q15 result inside (0.5, 1)
    // saturation prevents it from reaching 1
    io_T sum_recip = (io_T) MIN((1L << 29) / sum_mnt, 32767L);

    // sum_recip * vec_out[idx] = Q15 * Q15 (default LUT output)
    int lut_frac_bits = expneg_lut_fx16.out_frac_bits * 2;
    // 15 - sum_exp: sum_of_exps overhead
    int sum_exp_overhead = kMaxFracBitsFx16 - sum_exp;
    // final result: normalizing
    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                curr_vec = mli_prv_load_1vec(vec_out);
                for (int pos3 = 1; pos3 <= (in_prv.shape[3] / num_lanes); pos3++) {
                    acc_T tmp_acc = mli_math_mul_fx<vec_T, acc_T>(sum_recip, curr_vec);
                    curr_vec = mli_math_acc_cast_fx<vec_T, acc_T>(tmp_acc, lut_frac_bits + sum_exp_overhead - frac_bits);
                    mli_prv_store_n_samples(vec_out, curr_vec);
                    vec_out += out_step[3];
                    curr_vec = mli_prv_load_1vec(vec_out);
                }
                if ((in_prv.shape[3] & (num_lanes - 1)) != 0) {
                    acc_T tmp_acc = mli_math_mul_fx<vec_T, acc_T>(sum_recip, curr_vec);
                    curr_vec = mli_math_acc_cast_fx<vec_T, decltype(tmp_acc)>(tmp_acc, lut_frac_bits + sum_exp_overhead - frac_bits);
                    mli_prv_store_n_samples(vec_out, curr_vec, remaining_part);
                }
                vec_out += out_step[2];
            }
            vec_out += out_step[1];
        }
        vec_out += out_step[0];
    }
    return ;
}

template<typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_sa8_run(const MLI_PTR(io_T) vec_in, MLI_PTR(io_T) vec_out, 
        generic_tensor_private_t<MLI_PTR(io_T)> in_prv, generic_tensor_private_t<MLI_PTR(io_T)> out_prv,
        int* in_step, int* out_step, s8asym_quant_params in_params, s8asym_quant_params out_params) {
    const MLI_PTR(io_T) vec_in_begin = vec_in;

    auto curr_vec = mli_prv_load_nx4_samples(vec_in);
    typedef decltype(curr_vec) vNx4char_t;
    int num_lanes = get_number_lanes<vNx4char_t>();
    /* Look for the maximum */
    vNx4char_t max_vec = (vNx4char_t) INT8_MIN;
    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                curr_vec = mli_prv_load_nx4_samples(vec_in);
                for (int pos3 = 1; pos3 <= (in_prv.shape[3] / num_lanes); pos3++) {
                    vec_in += in_step[3];
                    max_vec = mli_math_max_fx(max_vec, curr_vec);
                    curr_vec = mli_prv_load_nx4_samples(vec_in);
                }
                if ((in_prv.shape[3] & (num_lanes - 1)) != 0) {
                    int remaining_part = in_prv.shape[3] & (num_lanes - 1);
                    pvNx4 predicate = mli_prv_pvNx4_init(remaining_part);
                    curr_vec = mli_math_select_fx<vNx4char_t, pvNx4>(predicate, curr_vec, (vNx4char_t) INT8_MIN);
                    max_vec = mli_math_max_fx(max_vec, curr_vec);
                }
                vec_in += in_step[2];
            }
            vec_in += in_step[1];
        }
        vec_in += in_step[0];
    }
    vec_in = vec_in_begin;

    int8_t max_val = mli_math_intra_max(max_vec);

    /* Subtract maximum from each input tensor element.
    * This subtraction is done by overwriting offset with max_value.
    * 1. Offset value is not needed here due to subtraction operation:
    *    (in_value + offset) - (max_value + offset) = in_value - max_value
    * 2. Subtraction operation is done in activation_lut_vec_elem_interpolate() in
    *    mli_prv_convert_sa8_fx16() function.
    */
    in_params.offset = max_val;

    vNx4accint_t sum_vec = mli_prv_init_accu<vNx4accint_t>();
    mli_acc32_t sum_acc = mli_math_mul_fx<int16_t, mli_acc32_t>(0, 0);
    /* TODO: There is another approach that can be implemented but will leads to lower accuracy:
    * sum of exps (sum_acc) can be calculated, and each fx16 exp converted to sa8 exp and stored in out[i]
    * array in the same loop,
    * but the sa8 exp will need to be converted again to multiply it with 1/(sum of exp).
    * In this approach there is no need to call activation_lut_vec_elem_interpolate() again in the second
    * for loop (but instead out[i] is converted to int16 and multiplied by 1 / sum_of_exp).
    */
    int remaining_part_tmp = in_prv.shape[3] & (num_lanes - 1);

    grp_pvNx2_t predicate_grp = init_predicate_grp(remaining_part_tmp);

    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                curr_vec = mli_prv_load_nx4_samples(vec_in);
                for (int pos3 = 1; pos3 <= (in_prv.shape[3] / num_lanes); pos3++) {
                    /* activation_lut */
                    vNx4short_t exp_res = mli::krn::vdsp::activation_lut_vec_elem_interpolate</* convert */ true>(
                                    mli_math_cast_fx<vNx4char_t, vNx4short_t>(curr_vec), &expneg_lut_fx16, /*in_frac_bits*/ 0, &in_params);

                    /* Accumulation through MAC and reciprocal calculation */
                    sum_vec = mli_math_mac_fx(sum_vec, exp_res, (vNx4short_t) 1);
                    vec_in += in_step[3];
                    curr_vec = mli_prv_load_nx4_samples(vec_in);
                }
                if ((in_prv.shape[3] & (num_lanes - 1)) != 0) {
                    vNx4short_t exp_res = mli::krn::vdsp::activation_lut_vec_elem_interpolate</* convert */ true>(
                                    mli_math_cast_fx<vNx4char_t, vNx4short_t>(curr_vec), &expneg_lut_fx16, /*in_frac_bits*/ 0, &in_params);
                    /* Accumulation through MAC and reciprocal calculation */
                    
                    exp_res = mli_math_select_fx<vNx4short_t, grp_pvNx2_t>(predicate_grp, exp_res, (vNx4short_t) 0);
                    sum_vec = mli_math_mac_fx(sum_vec, exp_res, (vNx4short_t) 1);
                }
                vec_in += in_step[2];
            }
            vec_in += in_step[1];
        }
        vec_in += in_step[0];
    }
    vec_in =vec_in_begin;
    
    sum_acc = mli_math_intra_sum(sum_vec);
    int sum_exp = mli_math_norm_fx<mli_acc32_t, int>(sum_acc);
    int16_t sum_mnt = mli_math_acc_cast_fx<int16_t, mli_acc32_t>(sum_acc, 16 - sum_exp);
    // sum_mnt is normalized (that is inside [0.5, 1) range)
    // so we use Q30(0.5) as a dividend to get Q15 result inside (0.5, 1)
    // saturation prevents it from reaching 1
    int16_t sum_recip = (int16_t) MIN((1L << 29) / sum_mnt, 32767L);

    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                curr_vec = mli_prv_load_nx4_samples(vec_in);
                for (int pos3 = 1; pos3 <= (in_prv.shape[3] / num_lanes); pos3++) {
                    /* activation_lut */
                    vNx4short_t exp_res = mli::krn::vdsp::activation_lut_vec_elem_interpolate</* convert */ true>(
                                    mli_math_cast_fx<vNx4char_t, vNx4short_t>(curr_vec), &expneg_lut_fx16, /*in_frac_bits*/ 0, &in_params);
                    /* multiply input by sum_recip */
                    vNx4accint_t fx_output32 = mli_math_mul_fx<vNx4short_t, vNx4accint_t>((vNx4short_t) sum_recip, exp_res);
                    vNx4int_t fx_output32_non_accum = mli_math_acc_cast_fx<vNx4int_t, vNx4accint_t>(fx_output32, 0);

                    // sum_recip * vec_out[idx] = Q15 * Q15 (default LUT output)
                    int lut_frac_bits = expneg_lut_fx16.out_frac_bits * 2;
                    // 15 - sum_exp: sum_of_exps overhead
                    int sum_exp_overhead = kMaxFracBitsFx16 - sum_exp;
                    // Converting to float and back to asym8
                    
                    mli_prv_store_n_samples(vec_out, mli_prv_convert_fx16_sa8<vNx4int_t, vNx4char_t>(fx_output32_non_accum, 
                            out_params.offset, lut_frac_bits + sum_exp_overhead - out_params.shift));
                    vec_out += out_step[3];
                    vec_in += in_step[3];
                    curr_vec = mli_prv_load_nx4_samples(vec_in);
                }
                if ((in_prv.shape[3] & (num_lanes - 1)) != 0) {
                    int remaining_part = in_prv.shape[3] & (num_lanes - 1);
                    vNx4short_t exp_res = mli::krn::vdsp::activation_lut_vec_elem_interpolate</* convert */ true>(
                                    mli_math_cast_fx<vNx4char_t, vNx4short_t>(curr_vec), &expneg_lut_fx16, /*in_frac_bits*/ 0, &in_params);

                    /* multiply input by sum_recip */
                    vNx4accint_t fx_output32 = mli_math_mul_fx<vNx4short_t, vNx4accint_t>((vNx4short_t) sum_recip, exp_res);
                    vNx4int_t fx_output32_non_accum = mli_math_acc_cast_fx<vNx4int_t, vNx4accint_t>(fx_output32, 0);

                    // sum_recip * vec_out[idx] = Q15 * Q15 (default LUT output)
                    int lut_frac_bits = expneg_lut_fx16.out_frac_bits * 2;
                    // 15 - sum_exp: sum_of_exps overhead
                    int sum_exp_overhead = kMaxFracBitsFx16 - sum_exp;
                    // Converting to float and back to asym8
                    
                    mli_prv_store_n_samples(vec_out, mli_prv_convert_fx16_sa8<vNx4int_t, vNx4char_t>(fx_output32_non_accum, 
                            out_params.offset, lut_frac_bits + sum_exp_overhead - out_params.shift), remaining_part);
                }
                vec_in += in_step[2];
                vec_out += out_step[2];
            }
            vec_in += in_step[1];
            vec_out += out_step[1];
        }
        vec_in += in_step[0];
        vec_out += out_step[0];
    }
    return ;
}

template<>
MLI_FORCE_INLINE void mli_krn_softmax_sa8_run(const MLI_PTR(int16_t) vec_in, MLI_PTR(int16_t) vec_out, 
        generic_tensor_private_t<MLI_PTR(int16_t)> in_prv, generic_tensor_private_t<MLI_PTR(int16_t)> out_prv,
        int* in_step, int* out_step, s8asym_quant_params in_params, s8asym_quant_params out_params) {
    return;
}

template <typename io_T, bool is_asym>
static MLI_FORCE_INLINE mli_status mli_krn_softmax_run(const mli_tensor *in, const mli_softmax_cfg* cfg,
        mli_tensor *out) {

    MLI_ASSERT(MLI_MAX_RANK == 4);

    const MLI_PTR(io_T) vec_in = nullptr;
    MLI_PTR(io_T) vec_out = nullptr;

    const MLI_PTR(io_T) in_ptr = (MLI_PTR(io_T))(in->data.mem.void_p);
    MLI_PTR(io_T) out_ptr = (MLI_PTR(io_T)) (out->data.mem.void_p);

    /* Copy tensor format */
    mli_prv_copy_tensor_format_except_mem_strides(in, out);
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
    mli_prv_squash_generic_tensor<MLI_PTR(io_T)>(&in_prv, &out_prv);

    struct s8asym_quant_params in_params;
    struct s8asym_quant_params out_params;
    int in_frac;

    if(is_asym) {
        in_params.scale  = in->el_params.sa.scale.mem.i16;
        in_params.shift = in->el_params.sa.scale_frac_bits.mem.i8;
        out_params.offset = kSoftmaxAsymZeroPoint;
        out_params.scale  = 1;
        out_params.shift = kSoftmaxOutputShift;
    }
    else {
        in_frac = static_cast<int>(in->el_params.fx.frac_bits);
        out->el_params.fx.frac_bits = (sizeof(io_T) * 8) - kTransfFuncIntBits - 1;
    }

    int in_step[MLI_MAX_RANK];    
    int out_step[MLI_MAX_RANK];

    auto curr_vec = mli_prv_load_1vec(in_ptr);
    typedef decltype(curr_vec) vec_T;
    int num_lanes = get_number_lanes<vec_T>();

    calculate_steps(in_step, in_prv, num_lanes);
    calculate_steps(out_step, out_prv, num_lanes);

    for (int dim0 = 0; dim0 < in_non_axis_prv.shape[0]; dim0++) {
        for (int dim1 = 0; dim1 < in_non_axis_prv.shape[1]; dim1++) {
            for (int dim2 = 0; dim2 < in_non_axis_prv.shape[2]; dim2++) {

                vec_in = &in_ptr[dim0 * in_non_axis_prv.mem_stride[0] + 
                                 dim1 * in_non_axis_prv.mem_stride[1] + 
                                 dim2 * in_non_axis_prv.mem_stride[2]];
                vec_out = &out_ptr[dim0 * out_non_axis_prv.mem_stride[0] + 
                                   dim1 * out_non_axis_prv.mem_stride[1] + 
                                   dim2 * out_non_axis_prv.mem_stride[2]];
                if (is_asym) {
                    mli::krn::mli_krn_softmax_sa8_run<io_T>(vec_in, vec_out, in_prv, out_prv, in_step, out_step, in_params, out_params);
                }
                else {
                    mli::krn::mli_krn_softmax_fx_run<io_T>(vec_in, vec_out, in_prv, out_prv, in_frac, in_step, out_step, out->el_params.fx.frac_bits);
                }
            }
        }
    }
    if (is_asym) {
        out->el_params.sa.zero_point.mem.i16 = out_params.offset;
        out->el_params.sa.scale.mem.i16 = out_params.scale;
        out->el_params.sa.scale_frac_bits.mem.i8 = (int8_t)out_params.shift;
    }

    return MLI_STATUS_OK;
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_SOFTMAX_VDSP_H_
