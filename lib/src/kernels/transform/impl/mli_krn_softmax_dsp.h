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
#include "mli_mem_info.h"
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
    v2q15_t max_val = mli_prv_init_v<int16_t, v2q15_t>(one_val[0]);
    v2q15_t min_val = mli_prv_init_v<int16_t, v2q15_t>(one_val[0]);

    for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                vec_in  = (MLI_PTR(io_T))orig_vec_in + POS(in_prv,  pos0, pos1, pos2, 0);
                if (in_prv->shape[3] & 1) {
                    v2q15_t one_val = mli_prv_init_v<int16_t, v2q15_t>(mli_prv_load_1_sample(vec_in)[0]);
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

    max_val = mli_prv_init_v<int16_t, v2q15_t>(mli_math_max_fx(max_val[0], max_val[1]));
    min_val = mli_prv_init_v<int16_t, v2q15_t>(mli_math_min_fx(min_val[0], min_val[1]));
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
        struct generic_tensor_private_t<MLI_PTR(io_T)> *in_prv, const mli_lut *lut,
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
                                (input, lut, 0, in_params, out_params);

                        /* Accumulation through MAC and reciprocal calculation */
                        res[1] = 0; // Unused
                        sum_acc = mli_math_mac_fx(sum_acc, res, one_v);
                        vec_in += 1;
                    }
                    for (int pos3 = 0; pos3 < in_prv->shape[3] >> 1; pos3++) {
                        /* activation_lut */
                        v2q15_t input = mli_prv_load_2_samples(vec_in);
                        v2q15_t res = mli::krn::activation_lut_two_elem_interpolate<int16_t, true, false>
                                (input, lut, 0, in_params, out_params);

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
    v2q15_t max_val = mli_prv_init_v<int16_t, v2q15_t>(one_val[0]);

    for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                vec_in  = (MLI_PTR(io_T))orig_vec_in + POS(in_prv,  pos0, pos1, pos2, 0);
                if (in_prv->shape[3] & 1) {
                    v2q15_t one_val = mli_prv_init_v<int16_t, v2q15_t>(mli_prv_load_1_sample(vec_in)[0]);
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
static MLI_FORCE_INLINE void mli_krn_softmax_fx_run(const MLI_PTR(io_T) vec_in, MLI_PTR(io_T) vec_out, 
        generic_tensor_private_t<MLI_PTR(io_T)> in_prv, generic_tensor_private_t<MLI_PTR(io_T)> out_prv,
        int in_frac, int frac_bits, const mli_lut *lut) {

    /* Subtract maximum from each element */
    mli_krn_softmax_subtract_max(vec_in, vec_out, &in_prv, &out_prv, &in_frac);

    /* Activation lookup table */
    struct generic_tensor_private_t<MLI_PTR(io_T)> out_vec_tensor = out_prv;
    out_vec_tensor.ptr = vec_out;
    mli::krn::activation_lut<io_T, false>(&out_vec_tensor, &out_vec_tensor, lut, in_frac);

    /* Accumulation through MAC and reciprocal calculation */
    mli_acc40_t sum_acc = sumTensor<io_T>(vec_out, &out_prv, lut);

    int sum_exp = mli_math_norm_fx<mli_acc40_t, int>(sum_acc);

    io_T sum_mnt = mli_math_acc_cast_fx<io_T, mli_acc40_t>(sum_acc, 16 - sum_exp);
    /* sum_mnt is normalized (that is inside [0.5, 1) range)
        * so we use Q30(0.5) as a dividend to get Q15 result inside (0.5, 1)
        * saturation prevents it from reaching 1
        */
    v2q15_t sum_recip = mli_prv_init_v<int16_t, v2q15_t>((int16_t)mli_math_sat_fx<int32_t>((1L << 29) / sum_mnt, 16));

    /* sum_recip * vec_out[idx] = Q15 * Q15 (default LUT output) */
    int lut_frac_bits = lut->out_frac_bits * 2;
    /* 15 - sum_exp: sum_of_exps overhead */
    int sum_exp_overhead = kMaxFracBitsFx16 - sum_exp;
    /* Normalize Output */
    normalizeTensor<io_T>(vec_out, &out_prv, sum_recip, 
                            lut_frac_bits + sum_exp_overhead - frac_bits);
    return ;
}

template<typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_sa8_run(const MLI_PTR(io_T) vec_in, MLI_PTR(io_T) vec_out, 
        generic_tensor_private_t<MLI_PTR(io_T)> in_prv, generic_tensor_private_t<MLI_PTR(io_T)> out_prv,
        s8asym_quant_params in_params, s8asym_quant_params out_params, const mli_lut *lut) {
    /* Subtract maximum from each input tensor element.
        * This subtraction is done by overwriting offset with max_value.
        * 1. Offset value is not needed here due to subtraction operation:
        *    (in_value + offset) - (max_value + offset) = in_value - max_value
        * 2. Replace in_params.offset with max_value 
        */
    in_params.offset = mli_krn_softmax_get_max(&in_prv, vec_in);
    
    /* Sum the input tensor after convert it to FX16 */               
    mli_acc40_t sum_acc = sumTensor<int8_t, true>(vec_in, &in_prv, lut, &in_params, &out_params);

    int sum_exp = mli_math_norm_fx<mli_acc40_t, int>(sum_acc);
    int16_t sum_mnt = mli_math_acc_cast_fx<int16_t, mli_acc40_t>(sum_acc, 16 - sum_exp);
    /* sum_mnt is normalized (that is inside [0.5, 1) range)
        * so we use Q30(0.5) as a dividend to get Q15 result inside (0.5, 1)
        * saturation prevents it from reaching 1
        */
    v2q15_t sum_recip = mli_prv_init_v<int16_t, v2q15_t>((int16_t)mli_math_sat_fx<int32_t>((1L << 29) / sum_mnt, 16));
    /* sum_recip * vec_out[idx] = Q15 * Q15 (default LUT output) */
    int lut_frac_bits = lut->out_frac_bits * 2;
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
                            (input, lut, 0, &in_params, &out_params);

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
                            (input, lut, 0, &in_params, &out_params);

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
    return ;
}

template<>
MLI_FORCE_INLINE void mli_krn_softmax_sa8_run(const MLI_PTR(int16_t) vec_in, MLI_PTR(int16_t) vec_out, 
        generic_tensor_private_t<MLI_PTR(int16_t)> in_prv, generic_tensor_private_t<MLI_PTR(int16_t)> out_prv,
        s8asym_quant_params in_params, s8asym_quant_params out_params, const mli_lut *lut){
    return ;
}

} // namespace dsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_SOFTMAX_DSP_H_
