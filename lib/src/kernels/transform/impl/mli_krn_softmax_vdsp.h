/*
* Copyright 2020-2021, Synopsys, Inc.
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
#include "mli_mem_info.h"
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

template <typename io_T, typename pred_T>
static MLI_FORCE_INLINE io_T mli_krn_softmax_get_max(
        generic_tensor_private_t<MLI_PTR(io_T)> *in_prv,
        const MLI_PTR(io_T) vec_in,  
        pred_T predicate,
        int remaining_part) {
    const MLI_PTR(io_T) vec_in_begin = vec_in;
    auto curr_vec = mli_prv_load_1vec(vec_in);
    typedef decltype(curr_vec) vec_T;
    int num_lanes = get_number_lanes<vec_T>();
    vec_T max_vec;
    if (sizeof(io_T) == sizeof(int8_t)){
        max_vec = (vec_T) INT8_MIN;
    }
    else {
        max_vec = (vec_T) INT16_MIN;
    }
    // Looking for maximum value
    for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                vec_in  = (MLI_PTR(io_T))vec_in_begin + POS(in_prv,  pos0, pos1, pos2, 0);
                curr_vec = mli_prv_load_1vec(vec_in);
                for (int pos3 = 1; pos3 <= (in_prv->shape[3] / num_lanes); pos3++) {
                    max_vec = mli_math_max_fx(max_vec, curr_vec);
                    vec_in += num_lanes;
                    curr_vec = mli_prv_load_1vec(vec_in);
                }
                if (remaining_part) {
                    curr_vec = mli_math_select_fx<vec_T, pred_T>(predicate, curr_vec, 
                            (vec_T) ((sizeof(io_T) == sizeof(int8_t)) ? INT8_MIN : INT16_MIN));
                    max_vec = mli_math_max_fx(max_vec, curr_vec);
                }
            }
        }
    }

    io_T max_val = mli_math_intra_max(max_vec);
    return max_val;
}

template <typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_fx_run(
        generic_tensor_private_t<MLI_PTR(io_T)> in_prv, 
        generic_tensor_private_t<MLI_PTR(io_T)> out_prv,
        int in_frac_bits, 
        int out_frac_bits, 
        const mli_lut *lut) {

    const MLI_PTR(io_T) vec_in = in_prv.ptr;
    MLI_PTR(io_T) vec_out = out_prv.ptr;
    MLI_PTR(io_T) vec_out_begin = vec_out;
    
    auto curr_vec = mli_prv_load_1vec(vec_in);
    typedef decltype(curr_vec) vec_T;
    int num_lanes = get_number_lanes<vec_T>();
    
    int remaining_part = in_prv.shape[3] & (num_lanes - 1);
    auto predicate = init_predicate(remaining_part, curr_vec);
    typedef decltype(predicate) pred_T;
    
    /* Look for the maximum */
    io_T max_val = mli_krn_softmax_get_max(&in_prv, vec_in, predicate, remaining_part);

    /* Use it to pass max_val as an offset */
    s8asym_quant_params in_params;
    in_params.offset = max_val;

    /* Activation lookup table */
    out_prv.ptr = vec_out;
    in_prv.ptr = (MLI_PTR(io_T))vec_in;
    mli::krn::activation_lut<io_T, /* convert */ false, /* fx_with_in_offset */ true>(
        &in_prv, &out_prv, lut, in_frac_bits, &in_params);

    // Accumulation through MAC and reciprocal calculation
    auto sum_vec = mli_prv_init_accu(curr_vec, (io_T) 0);
    typedef decltype(sum_vec) acc_T;
    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                vec_out = vec_out_begin + POS(&out_prv, pos0, pos1, pos2, 0);
                curr_vec = mli_prv_load_1vec(vec_out);
                for (int pos3 = 1; pos3 <= (in_prv.shape[3] / num_lanes); pos3++) {
                    sum_vec = mli_math_mac_fx(sum_vec, curr_vec, (vec_T) 1);
                    vec_out += num_lanes;
                    curr_vec = mli_prv_load_1vec(vec_out);
                }
                if (remaining_part) {
                    curr_vec = mli_math_select_fx<vec_T, pred_T>(predicate, curr_vec, (vec_T) 0);
                    sum_vec = mli_math_mac_fx(sum_vec, curr_vec, (vec_T) 1);
                }
            }
        }
    }

    mli_acc32_t  sum_acc = mli_math_intra_sum(sum_vec);

    int sum_exp = mli_math_norm_fx<mli_acc32_t, mli_acc32_t>(sum_acc);
    io_T sum_mnt = mli_math_acc_cast_fx<io_T, mli_acc32_t>(sum_acc, 16 - sum_exp);
    // sum_mnt is normalized (that is inside [0.5, 1) range)
    // so we use Q30(0.5) as a dividend to get Q15 result inside (0.5, 1)
    // saturation prevents it from reaching 1
    io_T sum_recip = (io_T) MIN((1L << 29) / sum_mnt, 32767L);

    // sum_recip * vec_out[idx] = Q15 * Q15 (default LUT output)
    int lut_frac_bits = lut->out_frac_bits * 2;
    // 15 - sum_exp: sum_of_exps overhead
    int sum_exp_overhead = kMaxFracBitsFx16 - sum_exp;
    
    constexpr int byte_size = 8;
    constexpr int max_shift = 2 * sizeof(io_T) * byte_size - 1;
    int shift = mli_math_min_fx(lut_frac_bits + sum_exp_overhead - out_frac_bits, max_shift);
    // final result: normalizing
    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                vec_out = vec_out_begin + POS(&out_prv, pos0, pos1, pos2, 0);
                curr_vec = mli_prv_load_1vec(vec_out);
                for (int pos3 = 1; pos3 <= (in_prv.shape[3] / num_lanes); pos3++) {
                    acc_T tmp_acc = mli_math_mul_fx<vec_T, acc_T>(sum_recip, curr_vec);
                    curr_vec = mli_math_acc_cast_fx<vec_T, acc_T>(tmp_acc, shift);
                    mli_prv_store_n_samples(vec_out, curr_vec);
                    vec_out += num_lanes;
                    curr_vec = mli_prv_load_1vec(vec_out);
                }
                if (remaining_part) {
                    acc_T tmp_acc = mli_math_mul_fx<vec_T, acc_T>(sum_recip, curr_vec);
                    curr_vec = mli_math_acc_cast_fx<vec_T, decltype(tmp_acc)>(tmp_acc, shift);
                    mli_prv_store_n_samples(vec_out, curr_vec, remaining_part);
                }
            }
        }
    }
}

template<typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_sa8_run(
        generic_tensor_private_t<MLI_PTR(io_T)> in_prv, 
        generic_tensor_private_t<MLI_PTR(io_T)> out_prv,
        s8asym_quant_params in_params, 
        s8asym_quant_params out_params, 
        const mli_lut *lut) {

    const MLI_PTR(io_T) vec_in = in_prv.ptr;
    MLI_PTR(io_T) vec_out = out_prv.ptr;
    const MLI_PTR(io_T) vec_in_begin = vec_in;
    MLI_PTR(io_T) vec_out_begin = vec_out;

    auto curr_vec = mli_prv_load_nx4_samples(vec_in);
    typedef decltype(curr_vec) vNx4char_t;
    int num_lanes = get_number_lanes<vNx4char_t>();
    int remaining_part = in_prv.shape[3] & (num_lanes - 1);
    pvNx4 predicate = mli_prv_pvNx4_init(remaining_part);
    
    /* Look for the maximum */
    int8_t max_val = mli_krn_softmax_get_max(&in_prv, vec_in, predicate, remaining_part);

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

    grp_pvNx2_t predicate_grp = init_predicate_grp(remaining_part);

    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                vec_in  = (MLI_PTR(io_T))vec_in_begin + POS(&in_prv,  pos0, pos1, pos2, 0);
                curr_vec = mli_prv_load_nx4_samples(vec_in);
                for (int pos3 = 1; pos3 <= (in_prv.shape[3] / num_lanes); pos3++) {
                    /* activation_lut */
                    vNx4short_t exp_res = mli::krn::vdsp::activation_lut_vec_elem_interpolate</* convert */ true>(
                                    mli_math_cast_fx<vNx4char_t, vNx4short_t>(curr_vec), lut, /*in_frac_bits*/ 0, &in_params);

                    /* Accumulation through MAC and reciprocal calculation */
                    sum_vec = mli_math_mac_fx(sum_vec, exp_res, (vNx4short_t) 1);
                    vec_in += num_lanes;
                    curr_vec = mli_prv_load_nx4_samples(vec_in);
                }
                if (remaining_part) {
                    vNx4short_t exp_res = mli::krn::vdsp::activation_lut_vec_elem_interpolate</* convert */ true>(
                                    mli_math_cast_fx<vNx4char_t, vNx4short_t>(curr_vec), lut, /*in_frac_bits*/ 0, &in_params);
                    /* Accumulation through MAC and reciprocal calculation */
                    
                    exp_res = mli_math_select_fx<vNx4short_t, grp_pvNx2_t>(predicate_grp, exp_res, (vNx4short_t) 0);
                    sum_vec = mli_math_mac_fx(sum_vec, exp_res, (vNx4short_t) 1);
                }
            }
        }
    }
    
    sum_acc = mli_math_intra_sum(sum_vec);
    int sum_exp = mli_math_norm_fx<mli_acc32_t, int>(sum_acc);
    int16_t sum_mnt = mli_math_acc_cast_fx<int16_t, mli_acc32_t>(sum_acc, 16 - sum_exp);
    // sum_mnt is normalized (that is inside [0.5, 1) range)
    // so we use Q30(0.5) as a dividend to get Q15 result inside (0.5, 1)
    // saturation prevents it from reaching 1
    int16_t sum_recip = (int16_t) MIN((1L << 29) / sum_mnt, 32767L);
    // sum_recip * vec_out[idx] = Q15 * Q15 (default LUT output)
    int lut_frac_bits = lut->out_frac_bits * 2;
    // 15 - sum_exp: sum_of_exps overhead
    int sum_exp_overhead = kMaxFracBitsFx16 - sum_exp;
    constexpr int max_shift = 31;
    int shift = mli_math_min_fx(lut_frac_bits + sum_exp_overhead - out_params.shift, max_shift);

    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                vec_in  = (MLI_PTR(io_T))vec_in_begin + POS(&in_prv,  pos0, pos1, pos2, 0);
                vec_out = vec_out_begin + POS(&out_prv, pos0, pos1, pos2, 0);
                curr_vec = mli_prv_load_nx4_samples(vec_in);
                for (int pos3 = 1; pos3 <= (in_prv.shape[3] / num_lanes); pos3++) {
                    /* activation_lut */
                    vNx4short_t exp_res = mli::krn::vdsp::activation_lut_vec_elem_interpolate</* convert */ true>(
                                    mli_math_cast_fx<vNx4char_t, vNx4short_t>(curr_vec), lut, /*in_frac_bits*/ 0, &in_params);
                    /* multiply input by sum_recip */
                    vNx4accint_t fx_output32 = mli_math_mul_fx<vNx4short_t, vNx4accint_t>((vNx4short_t) sum_recip, exp_res);
                    vNx4int_t fx_output32_non_accum = mli_math_acc_cast_fx<vNx4int_t, vNx4accint_t>(fx_output32, 0);

                    // Converting to float and back to asym8
                    mli_prv_store_n_samples(vec_out, mli_prv_convert_fx16_sa8<vNx4int_t, vNx4char_t>(fx_output32_non_accum, 
                            out_params.offset, shift));
                    vec_out += num_lanes;
                    vec_in += num_lanes;
                    curr_vec = mli_prv_load_nx4_samples(vec_in);
                }
                if (remaining_part) {
                    int remaining_part = in_prv.shape[3] & (num_lanes - 1);
                    vNx4short_t exp_res = mli::krn::vdsp::activation_lut_vec_elem_interpolate</* convert */ true>(
                                    mli_math_cast_fx<vNx4char_t, vNx4short_t>(curr_vec), lut, /*in_frac_bits*/ 0, &in_params);

                    /* multiply input by sum_recip */
                    vNx4accint_t fx_output32 = mli_math_mul_fx<vNx4short_t, vNx4accint_t>((vNx4short_t) sum_recip, exp_res);
                    vNx4int_t fx_output32_non_accum = mli_math_acc_cast_fx<vNx4int_t, vNx4accint_t>(fx_output32, 0);

                    // Converting to float and back to asym8
                    mli_prv_store_n_samples(vec_out, mli_prv_convert_fx16_sa8<vNx4int_t, vNx4char_t>(fx_output32_non_accum, 
                            out_params.offset, shift), remaining_part);
                }
            }
        }
    }
}

template<>
MLI_FORCE_INLINE void mli_krn_softmax_sa8_run(
        generic_tensor_private_t<MLI_PTR(int16_t)> in_prv, 
        generic_tensor_private_t<MLI_PTR(int16_t)> out_prv,
        s8asym_quant_params in_params, 
        s8asym_quant_params out_params, 
        const mli_lut *lut) {
    return;
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_SOFTMAX_VDSP_H_
