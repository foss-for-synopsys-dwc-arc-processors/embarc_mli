/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_LUT_DSP_H_
#define _MLI_PRV_LUT_DSP_H_

#include "mli_config.h"
#include "mli_prv_lut_decl.h"
#include "mli_math.h"
#include "mli_mem_info.h"
#include "mli_prv_quant.h"

#include "mli_prv_dsp.h"
#include "mli_prv_load_store.h"

namespace mli {
namespace krn {
namespace dsp {


template <typename out_T, bool convert_input, bool convert_output>
static MLI_FORCE_INLINE v2q15_t activation_lut_two_elem_interpolate(
        const v2q15_t in,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params) {

    MLI_ASSERT(lut->length >= 0);

    if (convert_input) {
        MLI_ASSERT(in_params != nullptr);
        MLI_ASSERT(out_params != nullptr);
        /* Calculating Scaling Factor to transform SA8 to Qmn (FX16) */
        int lut_int_bits_fx8 = kMaxFracBitsFx8 - lut->in_frac_bits;
        int frac_bits_fx16 = kMaxFracBitsFx16 - lut_int_bits_fx8;
        in_frac_bits = frac_bits_fx16;
    }

    int16_t *lut_data = lut->data.mem.pi16;
    int shift_in = in_frac_bits - lut->in_frac_bits;
    // if shift amount is too high, preshift argument itself and
    // limit shift amount to prevent overflows
    int preshift_in = mli_math_max_fx(shift_in - (int)kMaxFracBitsFx16, 0);
    shift_in = mli_math_min_fx(shift_in, (int)kMaxFracBitsFx16);

    v2q15_t offset = mli_prv_init_v<int16_t, v2q15_t>(lut->input_offset);
    v2q15_t lower = mli_prv_init_v<int16_t, v2q15_t>(0);
    // input data is more precise than LUT
    v2q15_t mask = mli_prv_init_v<int16_t, v2q15_t>((1 << shift_in) - 1);
    v2q15_t upper = mli_prv_init_v<int16_t, v2q15_t>(lut->length - 2);

    /* Convert Input SA8 to FX */
    v2q15_t x = in;
    if (convert_input) {
        int shift = ((int32_t) in_params->shift - in_frac_bits) + preshift_in;
        x = mli_prv_convert_sa8_fx16<v2q15_t, v2q15_t>(x, in_params->offset, in_params->scale, shift);
    } else {
        x = mli_math_acc_ashift_fx(x, preshift_in);
    }

    v2q15_t lut_idx = mli_math_add_fx(mli_math_acc_ashift_fx(x, shift_in), offset);
    lut_idx = mli_math_bound_range_fx(lut_idx, lower, upper);
    // perform linear interpolation
    v2q15_t frac = x & mask;
    v2q15_t res = mli_prv_init_v(lut_data[lut_idx[0]], lut_data[lut_idx[1]]);
    v2q15_t next = mli_prv_init_v(lut_data[lut_idx[0] + 1], lut_data[lut_idx[1] + 1]);
    v2q15_t diff = mli_math_sub_fx(res, next);
    res = mli_math_sub_fx(res, mli_math_acc_cast_fx<v2q15_t, v2accum40_t>(
                            mli_math_mul_fx<v2q15_t, v2accum40_t>(diff, frac), shift_in));
    if (convert_output) {
        MLI_ASSERT(out_params->scale == 1);
        res = mli_prv_convert_fx16_sa8<v2q15_t, v2q15_t>(
                        res, out_params->offset, lut->out_frac_bits - out_params->shift);
    } else {
        if(sizeof(out_T) == 1) {
            res = mli_prv_v2q7_cast_rnd_v2q15(res);
        }
    }

    return res;
}

template <typename out_T, bool convert_input, bool convert_output>
static MLI_FORCE_INLINE v2q15_t activation_lut_two_elem_no_interpolate(
        const v2q15_t in,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params) {
        
    MLI_ASSERT(lut->length >= 0);

    int32_t scale_fx = 0;

    if (convert_input) {
        MLI_ASSERT(in_params != nullptr);
        MLI_ASSERT(out_params != nullptr);
        /* Calculating Scaling Factor to transform SA8 to Qmn (FX16) */
        int lut_int_bits_fx8 = kMaxFracBitsFx8 - lut->in_frac_bits;
        int frac_bits_fx16 = kMaxFracBitsFx16 - lut_int_bits_fx8;
        scale_fx = mli_math_acc_ashift_fx<int32_t>(in_params->scale, ((int32_t) in_params->shift - frac_bits_fx16));
        in_frac_bits = frac_bits_fx16;
    }

    int16_t *lut_data = lut->data.mem.pi16;
    int shift_in = in_frac_bits - lut->in_frac_bits;
    shift_in = mli_math_min_fx(shift_in, (int)kMaxFracBitsFx16);

    v2q15_t offset = mli_prv_init_v<int16_t, v2q15_t>(lut->input_offset);
    v2q15_t lower = mli_prv_init_v<int16_t, v2q15_t>(0);
    
    v2q15_t upper = mli_prv_init_v<int16_t, v2q15_t>(lut->length - 1);

    // input data isn't more precise than LUT
    v2q15_t x = in;
    /* Convert Input SA8 to FX */
    if (convert_input) {
        x = mli_prv_convert_sa8_fx16<v2q15_t, v2q15_t>(x, in_params->offset, scale_fx);
    }

    v2q15_t lut_idx = mli_math_add_fx(mli_math_acc_ashift_fx(x, shift_in), offset);
    lut_idx = mli_math_bound_range_fx(lut_idx, lower, upper);
    // no interpolation
    v2q15_t res = mli_prv_init_v(lut_data[lut_idx[0]], lut_data[lut_idx[1]]);
    if (convert_output) {
        MLI_ASSERT(out_params->scale == 1);
        res = mli_prv_convert_fx16_sa8<v2q15_t, v2q15_t>(
                        res, out_params->offset, lut->out_frac_bits - out_params->shift);
    } else {
        if(sizeof(out_T) == 1) {
            res = mli_prv_v2q7_cast_rnd_v2q15(res);
        }
    }

    return res;
}

template <typename io_T, bool convert>
static MLI_FORCE_INLINE void compute_activation_lut(
        const struct generic_tensor_private_t<MLI_PTR(io_T)> *in,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *out,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params) {

    MLI_ASSERT(in_frac_bits >= -1);  // -1 may be required by softmax
    MLI_ASSERT(lut->in_frac_bits >= 0);
    MLI_ASSERT(lut->length >= 0);
    MLI_ASSERT(MLI_MAX_RANK == 4);

    MLI_PTR(io_T) vec_in  = in->ptr;
    MLI_PTR(io_T) vec_out = out->ptr;

    if (convert) {
        MLI_ASSERT(in_params != nullptr);
        MLI_ASSERT(out_params != nullptr);
        /* Calculating Scaling Factor to transform SA8 to Qmn (FX16) */
        int lut_int_bits_fx8 = kMaxFracBitsFx8 - lut->in_frac_bits;
        int frac_bits_fx16 = kMaxFracBitsFx16 - lut_int_bits_fx8;
        in_frac_bits = frac_bits_fx16;
    }

    int shift_in = in_frac_bits - lut->in_frac_bits;
    // if shift amount is too high, preshift argument itself and
    // limit shift amount to prevent overflows
    shift_in = mli_math_min_fx(shift_in, (int)kMaxFracBitsFx16);

    if (shift_in > 0) {
        // input data is more precise than LUT

        for (int pos0 = 0; pos0 < in->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in->shape[2]; pos2++) {
                    vec_in  = in->ptr  + POS(in,  pos0, pos1, pos2, 0);
                    vec_out = out->ptr + POS(out, pos0, pos1, pos2, 0);
                    if (in->shape[3] & 1) {
                        v2q15_t input = mli_prv_load_1_sample(vec_in);
                        v2q15_t res = activation_lut_two_elem_interpolate<io_T, convert, convert>
                                (input, lut, in_frac_bits, in_params, out_params);
                        mli_prv_store_1_sample(vec_out, res);
                        vec_in  += 1;
                        vec_out += 1;
                    }
                    for (int pos3 = 0; pos3 < in->shape[3] >> 1; pos3++) {
                        v2q15_t input = mli_prv_load_2_samples(vec_in);
                        v2q15_t res = activation_lut_two_elem_interpolate<io_T, convert, convert>
                                (input, lut, in_frac_bits, in_params, out_params);
                        mli_prv_store_2_samples(vec_out, res);
                        vec_in  += 2;
                        vec_out += 2;
                    }
                }
            }
        }
    } else {
        // input data isn't more precise than LUT
        for (int pos0 = 0; pos0 < in->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in->shape[2]; pos2++) {
                    vec_in  = in->ptr  + POS(in,  pos0, pos1, pos2, 0);
                    vec_out = out->ptr + POS(out, pos0, pos1, pos2, 0);
                    if (in->shape[3] & 1) {
                        v2q15_t input = mli_prv_load_1_sample(vec_in);
                        v2q15_t res = activation_lut_two_elem_no_interpolate<io_T, convert, convert>
                                (input, lut, in_frac_bits, in_params, out_params);
                        mli_prv_store_1_sample(vec_out, res);
                        vec_in  += 1;
                        vec_out += 1;
                    }
                    for (int pos3 = 0; pos3 < in->shape[3] >> 1; pos3++) {
                        v2q15_t input = mli_prv_load_2_samples(vec_in);
                        v2q15_t res = activation_lut_two_elem_no_interpolate<io_T, convert, convert>
                                (input, lut, in_frac_bits, in_params, out_params);
                        mli_prv_store_2_samples(vec_out, res);
                        vec_in  += 2;
                        vec_out += 2;
                    }
                }
            }
        }
    }
}

} // namespace dsp
} // namespace krn
} // namespace mli

#endif // _MLI_PRV_LUT_DSP_H_
