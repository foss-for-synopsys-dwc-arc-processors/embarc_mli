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

    const MLI_PTR(int8_t) vec_in = nullptr;
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
    mli_prv_squash_generic_tensor<MLI_PTR(int8_t)>(&in_prv, &out_prv);

    int in_step[4];
    in_step[3] = _VDSP_NUM_8BIT_LANES * in_prv.mem_stride[3];
    in_step[2] = in_prv.mem_stride[2] - in_step[3] * (in_prv.shape[3] / _VDSP_NUM_8BIT_LANES);
    in_step[1] = in_prv.mem_stride[1] - in_step[2] * in_prv.shape[2];
    in_step[0] = in_prv.mem_stride[0] - in_step[1] * in_prv.shape[1];
    
    int out_step[4];
    out_step[3] = _VDSP_NUM_8BIT_LANES * out_prv.mem_stride[3];
    out_step[2] = out_prv.mem_stride[2] - out_step[3] * (out_prv.shape[3] / _VDSP_NUM_8BIT_LANES);
    out_step[1] = out_prv.mem_stride[1] - out_step[2] * out_prv.shape[2];
    out_step[0] = out_prv.mem_stride[0] - out_step[1] * out_prv.shape[1];

    for (int dim0 = 0; dim0 < in_non_axis_prv.shape[0]; dim0++) {
        for (int dim1 = 0; dim1 < in_non_axis_prv.shape[1]; dim1++) {
            for (int dim2 = 0; dim2 < in_non_axis_prv.shape[2]; dim2++) {

                vec_in = &in_ptr[dim0 * in_non_axis_prv.mem_stride[0] + 
                                 dim1 * in_non_axis_prv.mem_stride[1] + 
                                 dim2 * in_non_axis_prv.mem_stride[2]];
                vec_out = &out_ptr[dim0 * out_non_axis_prv.mem_stride[0] + 
                                   dim1 * out_non_axis_prv.mem_stride[1] + 
                                   dim2 * out_non_axis_prv.mem_stride[2]];

                /* Look for the maximum */
                vNx4char_t max_vec = (vNx4char_t) INT8_MIN;
                vNx4char_t curr_vec;
                for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
                    for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                        for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                            curr_vec = mli_prv_load_nx4_samples(vec_in);
                            for (int pos3 = 1; pos3 <= (in_prv.shape[3] / _VDSP_NUM_8BIT_LANES); pos3++) {
                                vec_in += in_step[3];
                                max_vec = mli_math_max_fx(max_vec, curr_vec);
                                curr_vec = mli_prv_load_nx4_samples(vec_in);
                            }
                            if ((in_prv.shape[3] & (_VDSP_NUM_8BIT_LANES - 1)) != 0) {
                                int remaining_part = in_prv.shape[3] & (_VDSP_NUM_8BIT_LANES - 1);
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
                vec_in = &in_ptr[dim0 * in_non_axis_prv.mem_stride[0] + 
                                dim1 * in_non_axis_prv.mem_stride[1] + 
                                dim2 * in_non_axis_prv.mem_stride[2]];
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
                int remaining_part_tmp = in_prv.shape[3] & (_VDSP_NUM_8BIT_LANES - 1);

                grp_pvNx2_t predicate_grp = init_predicate_grp(remaining_part_tmp);

                for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
                    for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                        for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                            curr_vec = mli_prv_load_nx4_samples(vec_in);
                            for (int pos3 = 1; pos3 <= (in_prv.shape[3] / _VDSP_NUM_8BIT_LANES); pos3++) {
                                /* activation_lut */
                                vNx4short_t exp_res = mli::krn::vdsp::activation_lut_vec_elem_interpolate</* convert */ true>(
                                                mli_math_cast_fx<vNx4char_t, vNx4short_t>(curr_vec), &expneg_lut_fx16, /*in_frac_bits*/ 0, &in_params);

                                /* Accumulation through MAC and reciprocal calculation */
                                sum_vec = mli_math_mac_fx(sum_vec, exp_res, (vNx4short_t) 1);
                                vec_in += in_step[3];
                                curr_vec = mli_prv_load_nx4_samples(vec_in);
                            }
                            if ((in_prv.shape[3] & (_VDSP_NUM_8BIT_LANES - 1)) != 0) {
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
                vec_in = &in_ptr[dim0 * in_non_axis_prv.mem_stride[0] + 
                                dim1 * in_non_axis_prv.mem_stride[1] + 
                                dim2 * in_non_axis_prv.mem_stride[2]];
                
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
                            for (int pos3 = 1; pos3 <= (in_prv.shape[3] / _VDSP_NUM_8BIT_LANES); pos3++) {
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
                            if ((in_prv.shape[3] & (_VDSP_NUM_8BIT_LANES - 1)) != 0) {
                                int remaining_part = in_prv.shape[3] & (_VDSP_NUM_8BIT_LANES - 1);
                                // curr_vec = mli_prv_load_nx4_samples(vec_in);
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
            }
        }
    }

    out->el_params.sa.zero_point.mem.i16 = out_params.offset;
    out->el_params.sa.scale.mem.i16 = out_params.scale;
    out->el_params.sa.scale_frac_bits.mem.i8 = (int8_t)out_params.shift;

    return MLI_STATUS_OK;
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_SOFTMAX_VDSP_H_
