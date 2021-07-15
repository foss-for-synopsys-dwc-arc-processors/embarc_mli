/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_SOFTMAX_REF_H_
#define _MLI_KRN_SOFTMAX_REF_H_

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

namespace mli {
namespace krn {
namespace ref {

const int kSoftmaxAsymZeroPoint = -128;
const int kSoftmaxOutputShift = 8;

template <typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_subtract_max(const MLI_PTR(io_T) vec_in, MLI_PTR(io_T) vec_out,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *in_prv,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *out_prv,
        int *in_frac_p) {
    // Looking for maximum value
    io_T max_val = vec_in[0];
    io_T min_val = vec_in[0];
    for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                for (int pos3 = 0; pos3 < in_prv->shape[3]; pos3++) {
                    max_val = mli_math_max_fx(max_val, vec_in[POS(in_prv, pos0, pos1, pos2, pos3)]);
                    min_val = mli_math_min_fx(min_val, vec_in[POS(in_prv, pos0, pos1, pos2, pos3)]);
                }
            }
        }
    }

    // Subtract maximum from each element
    // free one more bit if saturation is expected.
    const int biased_min = static_cast<int>(min_val) - max_val;
    const int min_limit = -(1 << ((sizeof(io_T) * 8) - 1));
    if (biased_min < min_limit) {
        max_val = max_val >> 1;
        for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                    for (int pos3 = 0; pos3 < in_prv->shape[3]; pos3++) {
                        vec_out[POS(out_prv, pos0, pos1, pos2, pos3)] =
                                (vec_in[POS(in_prv, pos0, pos1, pos2, pos3)] >> 1) - max_val;
                    }
                }
            }
        }
        *in_frac_p -= 1;
    } else {
        for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                    for (int pos3 = 0; pos3 < in_prv->shape[3]; pos3++) {
                        vec_out[POS(out_prv, pos0, pos1, pos2, pos3)] =
                                vec_in[POS(in_prv, pos0, pos1, pos2, pos3)] - max_val;
                    }
                }
            }
        }
    }
}

template <typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_fx_run(const MLI_PTR(io_T) vec_in, MLI_PTR(io_T) vec_out, 
        generic_tensor_private_t<MLI_PTR(io_T)> in_prv, generic_tensor_private_t<MLI_PTR(io_T)> out_prv,
        int in_frac, int frac_bits, const mli_lut *lut) {
    /* Subtract maximum from each element */
    mli_krn_softmax_subtract_max(vec_in, vec_out, &in_prv, &out_prv, &in_frac);

    /* Activation lookup table */
    struct generic_tensor_private_t<MLI_PTR(io_T)> out_vec_tensor = out_prv;
    out_vec_tensor.ptr = vec_out;
    mli::krn::activation_lut<io_T, false>(&out_vec_tensor, &out_vec_tensor, lut, in_frac);

    // Accumulation through MAC and reciprocal calculation
    mli_acc32_t sum_acc = mli_math_mul_fx<io_T, mli_acc32_t>(0, 0);
    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                for (int pos3 = 0; pos3 < in_prv.shape[3]; pos3++) {
                    sum_acc = mli_math_mac_fx(sum_acc,
                            vec_out[POS(&out_prv, pos0, pos1, pos2, pos3)], static_cast<io_T>(1));
                }
            }
        }
    }

    int sum_exp = mli_math_norm_fx<mli_acc32_t, int>(sum_acc);
    io_T sum_mnt = mli_math_acc_cast_fx<io_T, mli_acc32_t>(sum_acc, 16 - sum_exp);
    // sum_mnt is normalized (that is inside [0.5, 1) range)
    // so we use Q30(0.5) as a dividend to get Q15 result inside (0.5, 1)
    // saturation prevents it from reaching 1
    io_T sum_recip = (io_T)MIN((1L << 29) / sum_mnt, 32767L);

    // sum_recip * vec_out[idx] = Q15 * Q15 (default LUT output)
    int lut_frac_bits = expneg_lut_fx16.out_frac_bits * 2;
    // 15 - sum_exp: sum_of_exps overhead
    int sum_exp_overhead = kMaxFracBitsFx16 - sum_exp;

    // final result: normalizing
    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                for (int pos3 = 0; pos3 < in_prv.shape[3]; pos3++) {
                    mli_acc32_t tmp_acc = mli_math_mul_fx<io_T, mli_acc32_t>(sum_recip,
                            vec_out[POS(&out_prv, pos0, pos1, pos2, pos3)]);
                    constexpr int byte_size = 8;
                    constexpr int max_shift = 2 * sizeof(io_T) * byte_size - 1;
                    int shift = mli_math_min_fx(lut_frac_bits + sum_exp_overhead - frac_bits, max_shift);
                    vec_out[POS(&out_prv, pos0, pos1, pos2, pos3)] =
                            mli_math_acc_cast_fx<io_T, mli_acc32_t>(tmp_acc, shift);
                }
            }
        }
    }
    return ;
}

template<typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_sa8_run(const MLI_PTR(io_T) vec_in, MLI_PTR(io_T) vec_out, 
        generic_tensor_private_t<MLI_PTR(io_T)> in_prv, generic_tensor_private_t<MLI_PTR(io_T)> out_prv,
        s8asym_quant_params in_params, s8asym_quant_params out_params, const mli_lut *lut) {
    /* Look for the maximum */
    int8_t max_val = vec_in[0];
    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                for (int pos3 = 0; pos3 < in_prv.shape[3]; pos3++) {
                    max_val = mli_math_max_fx(max_val, vec_in[POS(&in_prv, pos0, pos1, pos2, pos3)]);
                }
            }
        }
    }

    /* Subtract maximum from each input tensor element.
        * This subtraction is done by overwriting offset with max_value.
        * 1. Offset value is not needed here due to subtraction operation:
        *    (in_value + offset) - (max_value + offset) = in_value - max_value
        * 2. Subtraction operation is done in activation_lut_one_elem_interpolate() in
        *    mli_prv_convert_sa8_fx16() function.
        */
    in_params.offset = max_val;

    mli_acc32_t sum_acc = mli_math_mul_fx<int16_t, mli_acc32_t>(0, 0);

    /* TODO: There is another approach that can be implemented but will leads to lower accuracy:
        * sum of exps (sum_acc) can be calculated, and each fx16 exp converted to sa8 exp and stored in out[i]
        * array in the same loop,
        * but the sa8 exp will need to be converted again to multiply it with 1/(sum of exp).
        * In this approach there is no need to call activation_lut_one_elem_interpolate() again in the second
        * for loop (but instead out[i] is converted to int16 and multiplied by 1 / sum_of_exp).
        */
    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                for (int pos3 = 0; pos3 < in_prv.shape[3]; pos3++) {

                    /* activation_lut */
                    int16_t exp_res = mli::krn::activation_lut_one_elem_interpolate<int8_t, int16_t,
                            /* convert_input */ true,  /* convert_output */ false>(
                                    vec_in[POS(&in_prv, pos0, pos1, pos2, pos3)],
                                    &expneg_lut_fx16, /*in_frac_bits*/ 0, &in_params, &out_params);

                    /* Accumulation through MAC and reciprocal calculation */
                    sum_acc = mli_math_mac_fx(sum_acc, exp_res, static_cast<int16_t>(1));
                }
            }
        }
    }

    int sum_exp = mli_math_norm_fx<mli_acc32_t, int>(sum_acc);
    int16_t sum_mnt = mli_math_acc_cast_fx<int16_t, mli_acc32_t>(sum_acc, 16 - sum_exp);
    // sum_mnt is normalized (that is inside [0.5, 1) range)
    // so we use Q30(0.5) as a dividend to get Q15 result inside (0.5, 1)
    // saturation prevents it from reaching 1
    int16_t sum_recip = (int16_t)MIN((1L << 29) / sum_mnt, 32767L);


    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                for (int pos3 = 0; pos3 < in_prv.shape[3]; pos3++) {

                    /* activation_lut */
                    int16_t exp_res = mli::krn::activation_lut_one_elem_interpolate<int8_t, int16_t,
                            /* convert_input */ true,  /* convert_output */ false>(
                                    vec_in[POS(&in_prv, pos0, pos1, pos2, pos3)],
                                    lut, /*in_frac_bits*/ 0, &in_params, &out_params);

                    /* multiply input by sum_recip */
                    mli_acc32_t fx_output32 = mli_math_mul_fx<int16_t, mli_acc32_t>(sum_recip, exp_res);

                    // sum_recip * vec_out[idx] = Q15 * Q15 (default LUT output)
                    int lut_frac_bits = expneg_lut_fx16.out_frac_bits * 2;
                    // 15 - sum_exp: sum_of_exps overhead
                    int sum_exp_overhead = kMaxFracBitsFx16 - sum_exp;
                    // Converting to float and back to asym8
                    constexpr int max_shift = 31;
                    int shift = mli_math_min_fx(lut_frac_bits + sum_exp_overhead - out_params.shift, max_shift);
                    vec_out[POS(&out_prv, pos0, pos1, pos2, pos3)] =
                            mli_prv_convert_fx16_sa8<mli_acc32_t, int8_t>(fx_output32, out_params.offset, shift);
                }
            }
        }
    }
    return ;
}

template<>
MLI_FORCE_INLINE void mli_krn_softmax_sa8_run(const MLI_PTR(int16_t) vec_in, MLI_PTR(int16_t) vec_out, 
        generic_tensor_private_t<MLI_PTR(int16_t)> in_prv, generic_tensor_private_t<MLI_PTR(int16_t)> out_prv,
        s8asym_quant_params in_params, s8asym_quant_params out_params, const mli_lut *lut) {
    return;
}

template <typename io_T, bool is_asym>
static MLI_FORCE_INLINE mli_status mli_krn_softmax_run(const mli_tensor *in, const mli_softmax_cfg* cfg,
        mli_tensor *out, const mli_lut *lut) {

    MLI_ASSERT(MLI_MAX_RANK == 4);

    const MLI_PTR(io_T) vec_in = nullptr;
    MLI_PTR(io_T) vec_out = nullptr;

    const MLI_PTR(io_T) in_ptr = mli_prv_tensor_data_ptr<MLI_PTR(io_T)>(in);
    MLI_PTR(io_T) out_ptr = mli_prv_tensor_data_ptr<MLI_PTR(io_T)>(out);

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
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&in_prv );
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&out_prv);

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
                    mli::krn::mli_krn_softmax_sa8_run<io_T>(vec_in, vec_out, in_prv, out_prv, in_params, out_params, lut);
                }
                else {
                    mli::krn::mli_krn_softmax_fx_run<io_T>(vec_in, vec_out, in_prv, out_prv, in_frac, out->el_params.fx.frac_bits, lut);
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

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_SOFTMAX_REF_H_
