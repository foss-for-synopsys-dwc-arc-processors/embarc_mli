/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_PRELU_DSP_H_
#define _MLI_KRN_PRELU_DSP_H_

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace dsp {

static MLI_FORCE_INLINE v2q15_t calc_prelu(
        const v2q15_t input,
        const v2q15_t scale_v,
        const int shift) {

    /* out  = max(0, in) + alpha * min(0, in) */
    v2q15_t zero = mli_prv_init_v<int16_t, v2q15_t>(0);
    v2q15_t pos = mli_math_max_fx(zero, input);
    v2q15_t neg = mli_math_acc_cast_fx<v2q15_t, v2accum40_t>(
                  mli_math_mul_fx<v2q15_t, v2accum40_t>(scale_v, mli_math_min_fx(zero, input)), shift);
    return mli_math_add_fx(pos, neg);
}

template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift) {

    v2q15_t input = mli_prv_load_1vec(vec_in);
    mli_prv_store_n_samples(vec_out, calc_prelu(input, scale, shift));
}

template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift,
        const int remaining_part) {

    MLI_ASSERT(remaining_part == 1);
    v2q15_t input = mli_prv_load_1vec(vec_in);
    mli_prv_store_1_sample(vec_out, calc_prelu(input, scale, shift));
}

static MLI_FORCE_INLINE s8asym_quant_params_v prelu_define_requant_params(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        mli_tensor *out,
        const v2q15_t alpha_sa8,
        const s8asym_quant_params *identity_params) {

    s8asym_quant_params scale0 = mli::krn::ref::prelu_define_requant_params(in, slope_coeff, out, 
                                                                            alpha_sa8[0], identity_params);
    s8asym_quant_params scale1 = mli::krn::ref::prelu_define_requant_params(in, slope_coeff, out, 
                                                                            alpha_sa8[1], identity_params);
    s8asym_quant_params_v alpha_params;
    alpha_params.scale  = mli_prv_init_v(scale0.scale,  scale1.scale );
    alpha_params.shift  = mli_prv_init_v(scale0.shift,  scale1.shift );
    alpha_params.offset = mli_prv_init_v(scale0.offset, scale1.offset);
    return alpha_params;
}

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params) {

    s8asym_quant_params alpha_param0 = {alpha_params->offset[0], alpha_params->shift[0], alpha_params->scale[0]};
    mli::krn::ref::compute_prelu(vec_in, vec_out, in_zp, identity_params, &alpha_param0);

    s8asym_quant_params alpha_param1 = {alpha_params->offset[1], alpha_params->shift[1], alpha_params->scale[1]};
    mli::krn::ref::compute_prelu(vec_in + 1, vec_out + 1, in_zp, identity_params, &alpha_param1);
}

static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(int8_t) vec_in,
        MLI_OUT_PTR(int8_t) vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params,
        const int remaining_part) {

    MLI_ASSERT(remaining_part == 1);
    s8asym_quant_params alpha_param = {alpha_params->offset[0], alpha_params->shift[0], alpha_params->scale[0]};
    mli::krn::ref::compute_prelu(vec_in, vec_out, in_zp, identity_params, &alpha_param);
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
    mli::krn::ref::compute_prelu_no_broadcast<io_T, scale_T>(vec_in, vec_out, scale_v, shift, in_prv, out_prv,
                                                             remaining_part);
}

static MLI_FORCE_INLINE void compute_prelu_no_broadcast(
        const MLI_PTR(int8_t) __restrict vec_in,
        MLI_OUT_PTR(int8_t) __restrict vec_out,
        const int16_t in_zp,
        const s8asym_quant_params *identity_params,
        const s8asym_quant_params_v *alpha_params,
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
                    mli::krn::compute_prelu(vec_in, vec_out, in_zp, identity_params, alpha_params, remaining_part);
                } else {
                    mli::krn::compute_prelu(vec_in, vec_out, in_zp, identity_params, alpha_params);
                }
            }
        }
    }
}

} // namespace dsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_PRELU_DSP_H_