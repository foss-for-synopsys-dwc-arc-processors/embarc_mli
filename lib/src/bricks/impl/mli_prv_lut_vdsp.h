/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_LUT_VDSP_H_
#define _MLI_PRV_LUT_VDSP_H_

#include "mli_config.h"
#include "mli_prv_lut_decl.h"
#include "mli_math.h"
#include "mli_prv_quant.h"
#include "mli_prv_load_store.h"

namespace mli {
namespace krn {
namespace vdsp {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

template<typename in_T, typename out_T>
static MLI_FORCE_INLINE out_T activation_lut_load_input(const MLI_PTR(in_T) in);

template<>
vNx4short_t activation_lut_load_input(const MLI_PTR(int8_t) in) {
    return mli_math_cast_fx<vNx4char_t,vNx4short_t>(mli_prv_load_n_samples(in));
}

template<>
vNx4short_t activation_lut_load_input(const MLI_PTR(int16_t) in) {
    return mli_prv_load_n_samples(in);
}

template<typename out_T, bool convert = false>
static MLI_FORCE_INLINE void activation_lut_store_output(
    MLI_OUT_PTR(out_T) out,
    vNx4short_t data,
    struct s8asym_quant_params *out_params);

template<typename out_T, bool convert = false>
MLI_FORCE_INLINE void activation_lut_store_output(
        MLI_OUT_PTR(int8_t) out,
        vNx4short_t data,
        struct s8asym_quant_params *out_params) {
    vNx4char_t result;
    if(convert) {
        MLI_ASSERT(out_params != nullptr);
        MLI_ASSERT(out_params->scale == 1);
        result = mli_prv_convert_fx16_sa8(data, out_params->offset, kLutOutFracBits - out_params->shift);
    } else {
        result = mli_math_cast_fx<vNx4short_t, vNx4char_t>(data, 8);
    }
    mli_prv_store_n_samples(out, result);
}

template<typename out_T, bool convert = false>
MLI_FORCE_INLINE void activation_lut_store_output(
        MLI_OUT_PTR(int16_t) out,
        vNx4short_t data,
        struct s8asym_quant_params *out_params) {
    mli_prv_store_n_samples(out, data);
}

template<typename out_T, bool convert = false>
static MLI_FORCE_INLINE void activation_lut_store_output(
        MLI_OUT_PTR(out_T) out,
        vNx4short_t data,
        int elem_num,
        struct s8asym_quant_params *out_params);

template<typename out_T, bool convert = false>
MLI_FORCE_INLINE void activation_lut_store_output(
        MLI_OUT_PTR(int8_t) out,
        vNx4short_t data,
        int elem_num,
        struct s8asym_quant_params *out_params) {
    vNx4char_t result;
    if(convert) {
        MLI_ASSERT(out_params != nullptr);
        MLI_ASSERT(out_params->scale == 1);
        result = mli_prv_convert_fx16_sa8(data, out_params->offset, kLutOutFracBits - out_params->shift);
    } else {
        result = mli_math_cast_fx<vNx4short_t, vNx4char_t>(data, 8);
    }
    mli_prv_store_n_samples(out, result, elem_num);
}

template<typename out_T, bool convert = false>
MLI_FORCE_INLINE void activation_lut_store_output(
        MLI_OUT_PTR(int16_t) out,
        vNx4short_t data,
        int elem_num,
        struct s8asym_quant_params *out_params) {
    mli_prv_store_n_samples(out, data, elem_num);
}

#pragma clang diagnostic pop

template <typename io_T, bool convert>
static void activation_lut(
        const struct generic_tensor_private_t<MLI_PTR(io_T)> *in,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *out,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params) {

    /* TODO use axis_mem_stride */
    /* TODO fix this and support mem strides*/
    int axis_len = in->shape[0];

    MLI_ASSERT(in_frac_bits >= -1);  // -1 may be required by softmax
    MLI_ASSERT(axis_len >= 0);
    MLI_ASSERT(lut->frac_bits >= 0);
    MLI_ASSERT(lut->length >= 0);

    int32_t scale_fx;

    if (convert) {
        MLI_ASSERT(in_params != nullptr);
        MLI_ASSERT(out_params != nullptr);
        /* Calculating Scaling Factor to transform SA8 to Qmn (FX16) */
        int lut_int_bits_fx8 = kMaxFracBitsFx8 - lut->frac_bits;
        int frac_bits_fx16 = kMaxFracBitsFx16 - lut_int_bits_fx8;
        scale_fx = mli_math_acc_ashift_fx<int32_t>(in_params->scale, ((int32_t) in_params->shift - frac_bits_fx16));
        in_frac_bits = frac_bits_fx16;
    }

    int shift_in = in_frac_bits - lut->frac_bits;
    const MLI_PTR(short) lut_data = (const MLI_PTR(short))lut->data;
    // if shift amount is too high, preshift argument itself and
    // limit shift amount to prevent overflows
    int preshift_in = mli_math_max_fx(shift_in - (int)kMaxFracBitsFx16, 0);
    shift_in = mli_math_min_fx(shift_in, (int)kMaxFracBitsFx16);

    if (shift_in > 0) {
        // input data is more precise than LUT
        int16_t mask = (1 << shift_in) - 1;
        vNx4short_t x = activation_lut_load_input<io_T, vNx4short_t>(in) >> preshift_in;
        if (convert) {
            x = mli_prv_convert_sa8_fx16(x, in_params->offset, scale_fx);
        }
        vNx4short_t lut_idx = mli_math_add_fx<vNx4short_t>(mli_math_asr_fx(x, shift_in), lut->offset);
        /* Calculate lut_idx */
        lut_idx = mli_math_bound_range_fx(lut_idx , 0, lut->length - 2);
        vNx4int_t lut_idx_int = mli_math_cast_fx<vNx4short_t, vNx4int_t>(lut_idx);

        vNx4short_t frac = x & mask;
        for (int idx = 1; idx <= (axis_len / _VDSP_NUM_8BIT_LANES); idx++) {
            /* Update input ptr */
            in += _VDSP_NUM_8BIT_LANES;
            /* Load from LUT */
            vNx4short_t lut_values = mli_prv_gather_load_n_samples(lut_data, lut_idx_int);
            vNx4short_t lut_values_next = mli_prv_gather_load_n_samples(lut_data, lut_idx_int + 1);
            /* Load Next Input */
            x = activation_lut_load_input<io_T, vNx4short_t>(in) >> preshift_in;
            if (convert) {
                x = mli_prv_convert_sa8_fx16(x, in_params->offset, scale_fx);
            }
            lut_idx = mli_math_add_fx<vNx4short_t>(mli_math_asr_fx(x, shift_in), lut->offset);
            /* Calculate lut_idx */
            lut_idx = mli_math_bound_range_fx(lut_idx , 0, lut->length - 2);
            lut_idx_int = mli_math_cast_fx<vNx4short_t, vNx4int_t>(lut_idx);

            /* perform linear interpolation */
            vNx4short_t diffs = mli_math_sub_fx<vNx4short_t>(lut_values, lut_values_next);
            vNx4short_t diffs_mul_frac_cast =  mli_math_acc_cast_fx<vNx4short_t, vNx4accint_t>(
                                               mli_math_mul_fx<vNx4short_t, vNx4accint_t>(diffs, frac), shift_in);
            frac = x & mask;
            /* Calculate O/P */
            vNx4short_t result = mli_math_sub_fx<vNx4short_t>(lut_values, diffs_mul_frac_cast);
            /* Store O/P */
            activation_lut_store_output<io_T, convert>(out, result, out_params);
            /* Update output ptr */
            out += _VDSP_NUM_8BIT_LANES;
        }

        /* Handle the remaining part with predicates */
        if ((axis_len & (_VDSP_NUM_8BIT_LANES - 1)) != 0) {
            int remaining_part = axis_len & (_VDSP_NUM_8BIT_LANES - 1);
            /* Load from LUT */
            vNx4short_t lut_values = mli_prv_gather_load_n_samples(lut_data, lut_idx_int);
            vNx4short_t lut_values_next = mli_prv_gather_load_n_samples(lut_data, lut_idx_int + 1);

            vNx4short_t diffs = mli_math_sub_fx<vNx4short_t>(lut_values, lut_values_next);
            vNx4short_t diffs_mul_frac_cast =  mli_math_acc_cast_fx<vNx4short_t, vNx4accint_t>(
                                               mli_math_mul_fx<vNx4short_t, vNx4accint_t>(diffs, frac), shift_in);

            /* Calculate O/P */
            vNx4short_t result = mli_math_sub_fx<vNx4short_t>(lut_values, diffs_mul_frac_cast);
            /* Store O/P */
            activation_lut_store_output<io_T, convert>(out, result, remaining_part, out_params);
        }
    } else {
        // input data isn't more precise than LUT
        vNx4short_t x = activation_lut_load_input<io_T, vNx4short_t>(in);
        if (convert) {
            x = mli_prv_convert_sa8_fx16(x, in_params->offset, scale_fx);
        }
        vNx4short_t lut_idx = mli_math_add_fx<vNx4short_t>(mli_math_asl_fx(x, -shift_in), lut->offset);
        /* Calculate lut_idx_acc */
        lut_idx = mli_math_bound_range_fx(lut_idx , 0, lut->length - 1);

        for (int idx = 1; idx <= (axis_len / _VDSP_NUM_8BIT_LANES); idx++) {
            // no interpolation
            in += _VDSP_NUM_8BIT_LANES;
            /* Load from LUT */
            vNx4short_t lut_values = mli_prv_gather_load_n_samples(lut_data,
                                     mli_math_cast_fx<vNx4short_t, vNx4int_t>(lut_idx));
            /* Load Next Input */
            x = activation_lut_load_input<io_T, vNx4short_t>(in);
            if (convert) {
                x = mli_prv_convert_sa8_fx16(x, in_params->offset, scale_fx);
            }
            lut_idx = mli_math_add_fx<vNx4short_t>(mli_math_asl_fx(x, -shift_in), lut->offset);
            /* Calculate lut_idx */
            lut_idx = mli_math_bound_range_fx(lut_idx , 0, lut->length - 1);
            /* Store O/P */
            activation_lut_store_output<io_T, convert>(out, lut_values, out_params);
            out += _VDSP_NUM_8BIT_LANES;
        }

        /* Handle the remaining part with predicates */
        if ((axis_len & (_VDSP_NUM_8BIT_LANES - 1)) != 0) {
            int remaining_part = axis_len & (_VDSP_NUM_8BIT_LANES - 1);
            /* Load from LUT */
            vNx4short_t lut_values = mli_prv_gather_load_n_samples(lut_data,
                                     mli_math_cast_fx<vNx4short_t, vNx4int_t>(lut_idx));
            /* Store O/P */
            activation_lut_store_output<io_T, convert>(out, lut_values, remaining_part, out_params);
        }
    }
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_PRV_LUT_VDSP_H_
