/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_LUT_REF_H_
#define _MLI_PRV_LUT_REF_H_

#include "mli_config.h"
#include "mli_prv_lut_decl.h"
#include "mli_math.h"
#include "mli_mem_info.h"
#include "mli_prv_quant.h"

namespace mli {
namespace krn {
namespace ref {

template <typename io_T, bool convert, bool fx_with_in_offset>
static MLI_FORCE_INLINE void compute_activation_lut(
        const struct generic_tensor_private_t<MLI_PTR(io_T)> *in,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *out,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params) {

    MLI_ASSERT(lut->in_frac_bits >= 0);
    MLI_ASSERT(lut->length >= 0);
    MLI_ASSERT(MLI_MAX_RANK == 4);

    if (convert) {
        /* Calculating Scaling Factor to transform SA8 to Qmn (FX16) */
        int lut_int_bits_fx8 = kMaxFracBitsFx8 - lut->in_frac_bits;
        int frac_bits_fx16 = kMaxFracBitsFx16 - lut_int_bits_fx8;
        in_frac_bits = frac_bits_fx16;
    }

    int shift_in = in_frac_bits - lut->in_frac_bits;
        shift_in = mli_math_min_fx(shift_in, (int)kMaxFracBitsFx16);

    if (shift_in > 0) {
        // input data is more precise than LUT
        for (int pos0 = 0; pos0 < in->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in->shape[2]; pos2++) {
                    for (int pos3 = 0; pos3 < in->shape[3]; pos3++) {
                        out->ptr[POS(out, pos0, pos1, pos2, pos3)] = activation_lut_one_elem_interpolate
                                <io_T, io_T, convert, convert, fx_with_in_offset>(
                                in->ptr[POS(in, pos0, pos1, pos2, pos3)], lut, in_frac_bits, in_params, out_params);
                    }
                }
            }
        }
    } else {
        // input data isn't more precise than LUT
        for (int pos0 = 0; pos0 < in->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in->shape[2]; pos2++) {
                    for (int pos3 = 0; pos3 < in->shape[3]; pos3++) {
                        out->ptr[POS(out, pos0, pos1, pos2, pos3)] = activation_lut_one_elem_no_interpolate
                                <io_T, io_T, convert, convert, fx_with_in_offset>(
                                in->ptr[POS(in, pos0, pos1, pos2, pos3)], lut, in_frac_bits, in_params, out_params);
                    }
                }
            }
        }
    }
}

template <typename in_T, typename out_T, bool convert_input, bool convert_output, bool fx_with_in_offset>
static MLI_FORCE_INLINE out_T activation_lut_one_elem_interpolate(
        const in_T in,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params) {

    MLI_ASSERT(lut->length >= 0);

    out_T out;

    if (convert_input) {
        /* Calculating Scaling Factor to transform SA8 to Qmn (FX16) */
        int lut_int_bits_fx8 = kMaxFracBitsFx8 - lut->in_frac_bits;
        int frac_bits_fx16 = kMaxFracBitsFx16 - lut_int_bits_fx8;
        in_frac_bits = frac_bits_fx16;
    }

    int16_t *lut_data = lut->data.mem.pi16;
    int shift_in = in_frac_bits - lut->in_frac_bits;
    // if shift amount is too high, preshift argument itself and
    // limit shift amount to prevent overflows
    constexpr int max_shift = 15;
    int preshift_in = mli_math_max_fx(shift_in - (int)kMaxFracBitsFx16, 0);
        preshift_in = mli_math_min_fx(preshift_in, max_shift);
    shift_in = mli_math_min_fx(shift_in, (int)kMaxFracBitsFx16);

    int16_t mask = (1 << shift_in) - 1;

    /* Convert Input SA8 to FX */
    int32_t input;
    if (convert_input) {
        MLI_ASSERT(in_params != nullptr);
        int shift = ((int32_t) in_params->shift - in_frac_bits);
        input = mli_prv_convert_sa8_fx16<in_T, int32_t>(in, in_params->offset, in_params->scale, shift);
    } else {
        if (fx_with_in_offset) {
            MLI_ASSERT(in_params != nullptr);
            input = mli_math_sub_fx<int32_t>((int32_t)in, in_params->offset);
        } else {
            input = in;
        }
    }
    int32_t x = mli_math_asr_fx(input, preshift_in);
    int lut_idx = mli_math_add_fx(mli_math_asr_fx(x, shift_in), lut->input_offset);
        lut_idx = mli_math_bound_range_fx(lut_idx, 0, lut->length - 2);
    // perform linear interpolation
    int16_t frac = x & mask;
    int16_t res = lut_data[lut_idx];
    int16_t diff = res - lut_data[lut_idx + 1];
    res -= mli_math_acc_cast_fx<int16_t, mli_acc32_t>(
           mli_math_mul_fx<int16_t, mli_acc32_t>(diff, frac), shift_in);

    if (convert_output) {
        MLI_ASSERT(out_params != nullptr);
        MLI_ASSERT(out_params->scale == 1);
        out = mli_prv_convert_fx16_sa8<int16_t, out_T>(res, out_params->offset,
                lut->out_frac_bits - out_params->shift);
    } else {
        out = mli_math_cast_fx<int16_t, out_T>(res, 16 - sizeof(out_T) * 8);
    }

    return out;
}

template <typename in_T, typename out_T, bool convert_input, bool convert_output, bool fx_with_in_offset>
static MLI_FORCE_INLINE out_T activation_lut_one_elem_no_interpolate(
        const in_T in,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params) {

    MLI_ASSERT(lut->length >= 0);

    out_T out;
    int32_t scale_fx = 0;

    if (convert_input) {
        MLI_ASSERT(in_params != nullptr);
        /* Calculating Scaling Factor to transform SA8 to Qmn (FX16) */
        int lut_int_bits_fx8 = kMaxFracBitsFx8 - lut->in_frac_bits;
        int frac_bits_fx16 = kMaxFracBitsFx16 - lut_int_bits_fx8;
        scale_fx = mli_math_acc_ashift_fx<int32_t>(in_params->scale, ((int32_t) in_params->shift - frac_bits_fx16));
        in_frac_bits = frac_bits_fx16;
    }

    int16_t *lut_data = lut->data.mem.pi16;
    int shift_in = in_frac_bits - lut->in_frac_bits;
        shift_in = mli_math_min_fx(shift_in, (int)kMaxFracBitsFx16);

    /* Convert Input SA8 to FX */
    int32_t input;
    if (convert_input) {
        input = mli_prv_convert_sa8_fx16<in_T, int16_t>(in, in_params->offset, scale_fx);
    } else {
        if (fx_with_in_offset) {
            MLI_ASSERT(in_params != nullptr);
            input = mli_math_sub_fx<int32_t>((int32_t)in, in_params->offset);
        } else {
            input = in;
        }
    }
    int x = input;
    int lut_idx = mli_math_add_fx(mli_math_asl_fx(x, -shift_in), lut->input_offset);
        lut_idx = mli_math_bound_range_fx(lut_idx, 0, lut->length - 1);
    // no interpolation
    int16_t res = lut_data[lut_idx];

    if (convert_output) {
        MLI_ASSERT(out_params != nullptr);
        MLI_ASSERT(out_params->scale == 1);
        out = mli_prv_convert_fx16_sa8<int16_t, out_T>(res, out_params->offset,
                                                       lut->out_frac_bits - out_params->shift );
    } else {
        out = mli_math_cast_fx<int16_t, out_T>(res, 16 - sizeof(out_T) * 8);
    }

    return out;
}

template <typename io_T, bool convert, bool fx_with_in_offset>
static MLI_FORCE_INLINE void activation_lut(
        const struct generic_tensor_private_t<MLI_PTR(io_T)> *in,
        struct generic_tensor_private_t<MLI_OUT_PTR(io_T)> *out,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params) {
    mli::krn::compute_activation_lut<io_T, convert, fx_with_in_offset>(in, out, lut, in_frac_bits, in_params, out_params);
}

template <typename io_T, bool convert, bool fx_with_in_offset>
static MLI_FORCE_INLINE void activation_lut(
        const mli_tensor *in,
        const mli_tensor *out,
        const mli_lut *lut,
        int in_frac_bits,
        struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params) {

    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);
    auto out_prv =  mli_prv_get_generic_tensor<MLI_OUT_PTR(io_T)>(out);

    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_squash_generic_tensor<MLI_PTR(io_T)>(&in_prv, &out_prv);

    mli::krn::compute_activation_lut<io_T, convert, fx_with_in_offset>(&in_prv, &out_prv, lut, in_frac_bits, in_params, out_params);
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_PRV_LUT_REF_H_
