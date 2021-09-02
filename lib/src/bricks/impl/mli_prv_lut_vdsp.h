/*
* Copyright 2020-2021, Synopsys, Inc.
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
#include "mli_mem_info.h"
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
MLI_FORCE_INLINE vNx4short_t activation_lut_load_input(const MLI_PTR(int8_t) in) {
    return mli_math_cast_fx<vNx4char_t,vNx4short_t>(mli_prv_load_nx4_samples(in));
}

template<>
MLI_FORCE_INLINE vNx4short_t activation_lut_load_input(const MLI_PTR(int16_t) in) {
    return mli_prv_load_nx4_samples(in);
}

template<typename out_T, bool convert = false>
static MLI_FORCE_INLINE void activation_lut_store_output(
    MLI_OUT_PTR(out_T) out,
    vNx4short_t data,
    const mli_lut *lut,
    struct s8asym_quant_params *out_params);

template<typename out_T, bool convert = false>
MLI_FORCE_INLINE void activation_lut_store_output(
        MLI_OUT_PTR(int8_t) out,
        vNx4short_t data,
        const mli_lut *lut,
        struct s8asym_quant_params *out_params) {
    vNx4char_t result;
    if(convert) {
        MLI_ASSERT(out_params != nullptr);
        MLI_ASSERT(out_params->scale == 1);
        result = mli_prv_convert_fx16_sa8<vNx4short_t, vNx4char_t>(data, out_params->offset, 
                                                                   lut->out_frac_bits - out_params->shift);
    } else {
        result = mli_math_cast_fx<vNx4short_t, vNx4char_t>(data, 8);
    }
    mli_prv_store_n_samples(out, result);
}

template<typename out_T, bool convert = false>
MLI_FORCE_INLINE void activation_lut_store_output(
        MLI_OUT_PTR(int16_t) out,
        vNx4short_t data,
        const mli_lut *lut,
        struct s8asym_quant_params *out_params) {
    mli_prv_store_n_samples(out, data);
}

template<typename out_T, bool convert = false>
static MLI_FORCE_INLINE void activation_lut_store_output(
        MLI_OUT_PTR(out_T) out,
        vNx4short_t data,
        int elem_num,
        const mli_lut *lut,
        struct s8asym_quant_params *out_params);

template<typename out_T, bool convert = false>
MLI_FORCE_INLINE void activation_lut_store_output(
        MLI_OUT_PTR(int8_t) out,
        vNx4short_t data,
        int elem_num,
        const mli_lut *lut,
        struct s8asym_quant_params *out_params) {
    vNx4char_t result;
    if(convert) {
        MLI_ASSERT(out_params != nullptr);
        MLI_ASSERT(out_params->scale == 1);
        result = mli_prv_convert_fx16_sa8<vNx4short_t, vNx4char_t>(data, out_params->offset, 
                                                                   lut->out_frac_bits - out_params->shift);
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
        const mli_lut *lut,
        struct s8asym_quant_params *out_params) {
    mli_prv_store_n_samples(out, data, elem_num);
}

#pragma clang diagnostic pop

template <bool convert, bool fx_with_in_offset>
static MLI_FORCE_INLINE vNx4short_t activation_lut_vec_elem_interpolate(
        vNx4short_t in,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params) {

    MLI_ASSERT(lut->in_frac_bits >= 0);
    MLI_ASSERT(lut->length >= 0);

    if (convert) {
        /* Calculating Scaling Factor to transform SA8 to Qmn (FX16) */
        int lut_int_bits_fx8 = kMaxFracBitsFx8 - lut->in_frac_bits;
        int frac_bits_fx16 = kMaxFracBitsFx16 - lut_int_bits_fx8;
        in_frac_bits = frac_bits_fx16;
    }

    int shift_in = in_frac_bits - lut->in_frac_bits;
    const MLI_PTR(short) lut_data = (const MLI_PTR(short))lut->data.mem.pi16;
    // if shift amount is too high, preshift argument itself and
    // limit shift amount to prevent overflows
    constexpr int max_shift = 15;
    int preshift_in = mli_math_max_fx(shift_in - (int)kMaxFracBitsFx16, 0);
        preshift_in = mli_math_min_fx(preshift_in, max_shift);
    shift_in = mli_math_min_fx(shift_in, (int)kMaxFracBitsFx16);

    // input data is more precise than LUT
    int16_t mask = (1 << shift_in) - 1;
    vNx4short_t x = in;
    vNx4int_t lut_idx_int;
    vNx4short_t frac;
    if (convert) {
        MLI_ASSERT(in_params != nullptr);
        int shift = (int32_t) in_params->shift - in_frac_bits;
        vNx4int_t x_int = mli_prv_convert_sa8_fx16<vNx4short_t, vNx4int_t>(x, in_params->offset, in_params->scale, shift);
        x_int = mli_math_asr_fx(x_int, preshift_in);
        frac = mli_math_cast_fx<vNx4int_t, vNx4short_t>(x_int & mask);
        lut_idx_int = mli_math_add_fx<vNx4int_t>(mli_math_asr_fx(x_int, shift_in), lut->input_offset);
        lut_idx_int = mli_math_bound_range_fx(lut_idx_int , 0, lut->length - 2);

    } else {
        if (fx_with_in_offset) {
            MLI_ASSERT(in_params != nullptr);
            vNx4int_t x_int = mli_math_mul_fx<vNx4short_t, vNx4int_t>(x, 1);
                      x_int = mli_math_sub(x_int, in_params->offset);
            x_int = mli_math_asr_fx(x_int, preshift_in);
            frac = mli_math_cast_fx<vNx4int_t, vNx4short_t>(x_int & mask);
            lut_idx_int = mli_math_add_fx<vNx4int_t>(mli_math_asr_fx(x_int, shift_in), lut->input_offset);
            lut_idx_int = mli_math_bound_range_fx(lut_idx_int , 0, lut->length - 2);
        } else {
            x = mli_math_asr_fx(x, preshift_in);
            frac = x & mask;
            vNx4short_t lut_idx = mli_math_add_fx<vNx4short_t>(mli_math_asr_fx(x, shift_in), lut->input_offset);
                        lut_idx = mli_math_bound_range_fx(lut_idx , 0, lut->length - 2);
            lut_idx_int = mli_math_mul_fx<vNx4short_t, vNx4int_t>(lut_idx, 1);
        }
    }

    /* Load from LUT */
    vNx4short_t lut_values = mli_prv_gather_load_nx4_samples(lut_data, lut_idx_int);
    vNx4short_t lut_values_next = mli_prv_gather_load_nx4_samples(lut_data, lut_idx_int + 1);
    /* perform linear interpolation */
    vNx4short_t diffs = mli_math_sub_fx<vNx4short_t>(lut_values, lut_values_next);
    vNx4short_t diffs_mul_frac_cast =  mli_math_acc_cast_fx<vNx4short_t, vNx4accint_t>(
                                        mli_math_mul_fx<vNx4short_t, vNx4accint_t>(diffs, frac), shift_in);

    /* Calculate O/P */
    return mli_math_sub_fx<vNx4short_t>(lut_values, diffs_mul_frac_cast);
}

template <bool convert, bool fx_with_in_offset>
static MLI_FORCE_INLINE vNx4short_t activation_lut_vec_elem_no_interpolate(
        vNx4short_t in,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params) {

    MLI_ASSERT(lut->in_frac_bits >= 0);
    MLI_ASSERT(lut->length >= 0);

    int32_t scale_fx;

    if (convert) {
        MLI_ASSERT(in_params != nullptr);
        /* Calculating Scaling Factor to transform SA8 to Qmn (FX16) */
        int lut_int_bits_fx8 = kMaxFracBitsFx8 - lut->in_frac_bits;
        int frac_bits_fx16 = kMaxFracBitsFx16 - lut_int_bits_fx8;
        scale_fx = mli_math_acc_ashift_fx<int32_t>(in_params->scale, ((int32_t) in_params->shift - frac_bits_fx16));
        in_frac_bits = frac_bits_fx16;
    }

    int shift_in = in_frac_bits - lut->in_frac_bits;
    const MLI_PTR(short) lut_data = (const MLI_PTR(short))lut->data.mem.pi16;
    
    shift_in = mli_math_min_fx(shift_in, (int)kMaxFracBitsFx16);

    // input data isn't more precise than LUT
    vNx4int_t lut_idx_int;
    vNx4short_t x = in;
    if (fx_with_in_offset) {
            MLI_ASSERT(in_params != nullptr);
            vNx4int_t x_int = mli_math_mul_fx<vNx4short_t, vNx4int_t>(x, 1);
                      x_int = mli_math_sub(x_int, in_params->offset);
            lut_idx_int = mli_math_add_fx<vNx4int_t>(mli_math_asl_fx(x_int, -shift_in), lut->input_offset);
            lut_idx_int = mli_math_bound_range_fx(lut_idx_int , 0, lut->length - 1);
    } else {
        if (convert) {
            x = mli_prv_convert_sa8_fx16<vNx4short_t, vNx4short_t>(x, in_params->offset, scale_fx);
        }
        vNx4short_t lut_idx = mli_math_add_fx<vNx4short_t>(mli_math_asl_fx(x, -shift_in), lut->input_offset);
                    lut_idx = mli_math_bound_range_fx(lut_idx , 0, lut->length - 1);
        lut_idx_int = mli_math_mul_fx<vNx4short_t, vNx4int_t>(lut_idx, 1);
    }

    /* Load from LUT */
    vNx4short_t lut_values = mli_prv_gather_load_nx4_samples(lut_data, lut_idx_int);
    
    return lut_values;
}

template <typename io_T, bool convert, bool fx_with_in_offset>
static MLI_FORCE_INLINE void load_input_and_get_lut_idx(
        MLI_PTR(io_T) __restrict in_ptr,
        vNx4short_t &x,
        vNx4int_t &x_int,
        vNx4int_t &lut_idx_int,
        int16_t in_frac_bits,
        int preshift_in,
        int shift_in,
        const mli_lut *lut,
        const struct s8asym_quant_params *in_params) {

    x = activation_lut_load_input<io_T, vNx4short_t>(in_ptr);
    if (convert) {
        MLI_ASSERT(in_params != nullptr);
        int shift = (int32_t) in_params->shift - in_frac_bits;
        x_int = mli_prv_convert_sa8_fx16<vNx4short_t, vNx4int_t>(x, in_params->offset, in_params->scale, shift);
        x_int = mli_math_asr_fx(x_int, preshift_in);
        lut_idx_int = mli_math_add_fx<vNx4int_t>(mli_math_asr_fx(x_int, shift_in), lut->input_offset);
        lut_idx_int = mli_math_bound_range_fx(lut_idx_int , 0, lut->length - 2);

    } else {
        if (fx_with_in_offset) {
            MLI_ASSERT(in_params != nullptr);
            x_int = mli_math_mul_fx<vNx4short_t, vNx4int_t>(x, 1);
            x_int = mli_math_sub(x_int, in_params->offset);
            x_int = mli_math_asr_fx(x_int, preshift_in);
            lut_idx_int = mli_math_add_fx<vNx4int_t>(mli_math_asr_fx(x_int, shift_in), lut->input_offset);
            lut_idx_int = mli_math_bound_range_fx(lut_idx_int , 0, lut->length - 2);
        } else {
            x = mli_math_asr_fx(x, preshift_in);
            vNx4short_t lut_idx = mli_math_add_fx<vNx4short_t>(mli_math_asr_fx(x, shift_in), lut->input_offset);
                        lut_idx = mli_math_bound_range_fx(lut_idx , 0, lut->length - 2);
            lut_idx_int = mli_math_mul_fx<vNx4short_t, vNx4int_t>(lut_idx, 1);
        }
    }
}

template <typename io_T, bool convert, bool fx_with_in_offset>
static MLI_FORCE_INLINE void compute_activation_lut_func(
        MLI_PTR(io_T) __restrict in_ptr,
        MLI_OUT_PTR(io_T) __restrict out_ptr,
        const struct generic_tensor_private_t<MLI_PTR(io_T)> *in,
        struct generic_tensor_private_t<MLI_OUT_PTR(io_T)> *out,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params) {

    MLI_ASSERT(lut->in_frac_bits >= 0);
    MLI_ASSERT(lut->length >= 0);
    MLI_ASSERT(MLI_MAX_RANK == 4);

    MLI_PTR(io_T) __restrict input_ptr  = in_ptr;
    MLI_OUT_PTR(io_T) __restrict output_ptr = out_ptr;

    if (convert) {
        /* Calculating Scaling Factor to transform SA8 to Qmn (FX16) */
        int lut_int_bits_fx8 = kMaxFracBitsFx8 - lut->in_frac_bits;
        int frac_bits_fx16 = kMaxFracBitsFx16 - lut_int_bits_fx8;
        in_frac_bits = frac_bits_fx16;
    }

    int shift_in = in_frac_bits - lut->in_frac_bits;
    const MLI_PTR(short) lut_data = (const MLI_PTR(short))lut->data.mem.pi16;
    // if shift amount is too high, preshift argument itself and
    // limit shift amount to prevent overflows
    constexpr int max_shift = 15;
    int preshift_in = mli_math_max_fx(shift_in - (int)kMaxFracBitsFx16, 0);
        preshift_in = mli_math_min_fx(preshift_in, max_shift);
    shift_in = mli_math_min_fx(shift_in, (int)kMaxFracBitsFx16);

    int remaining_part = in->shape[3] & (_VDSP_NUM_8BIT_LANES - 1);
    if (shift_in > 0) {
        // input data is more precise than LUT
        int16_t mask = (1 << shift_in) - 1;
        for (int pos0 = 0; pos0 < in->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in->shape[2]; pos2++) {
                    input_ptr  = in_ptr  + POS(in,  pos0, pos1, pos2, 0);
                    output_ptr = out_ptr + POS(out, pos0, pos1, pos2, 0);
                    if (remaining_part) {
                        vNx4short_t x = activation_lut_load_input<io_T, vNx4short_t>(input_ptr);
                        vNx4short_t res = mli::krn::activation_lut_vec_elem_interpolate<convert, fx_with_in_offset>
                                            (x, lut, in_frac_bits, in_params);
                        /* Store O/P */
                        activation_lut_store_output<io_T, convert>(output_ptr, res, remaining_part, lut, out_params);
                        input_ptr  += remaining_part;
                        output_ptr += remaining_part;
                    }

                    /* Manual software pipelining */
                    vNx4short_t x, frac;
                    vNx4int_t x_int, lut_idx_int;
                    vNx4short_t _lut_values, _lut_values_next, _frac;

                    load_input_and_get_lut_idx<io_T, convert, fx_with_in_offset>(input_ptr, x, x_int, lut_idx_int,
                            in_frac_bits, preshift_in, shift_in, lut, in_params);
                    if (convert || fx_with_in_offset) {
                        frac = mli_math_cast_fx<vNx4int_t, vNx4short_t>(x_int & mask);
                    } else {
                        frac = x & mask;
                    }
                    input_ptr  += _VDSP_NUM_8BIT_LANES;

                    if (in->shape[3] >= _VDSP_NUM_8BIT_LANES && !convert) {
                        /* Load from LUT */
                        _lut_values = mli_prv_gather_load_nx4_samples(lut_data, lut_idx_int);
                        _lut_values_next = mli_prv_gather_load_nx4_samples(lut_data, lut_idx_int + 1);

                        load_input_and_get_lut_idx<io_T, convert, fx_with_in_offset>(input_ptr, x, x_int, lut_idx_int,
                                in_frac_bits, preshift_in, shift_in, lut, in_params);

                        _frac = frac;
                        if (convert || fx_with_in_offset) {
                            frac = mli_math_cast_fx<vNx4int_t, vNx4short_t>(x_int & mask);
                        } else {
                            frac = x & mask;
                        }
                        input_ptr  += _VDSP_NUM_8BIT_LANES;

                        for (int pos3 = remaining_part; pos3 < in->shape[3] - _VDSP_NUM_8BIT_LANES; pos3 += _VDSP_NUM_8BIT_LANES) {
                            /* Load from LUT */
                            vNx4short_t lut_values = mli_prv_gather_load_nx4_samples(lut_data, lut_idx_int);
                            vNx4short_t lut_values_next = mli_prv_gather_load_nx4_samples(lut_data, lut_idx_int + 1);

                            load_input_and_get_lut_idx<io_T, convert, fx_with_in_offset>(input_ptr, x, x_int, lut_idx_int,
                                    in_frac_bits, preshift_in, shift_in, lut, in_params);

                            /* perform linear interpolation */
                            vNx4short_t diffs = mli_math_sub_fx<vNx4short_t>(_lut_values, _lut_values_next);
                            vNx4short_t diffs_mul_frac_cast =  mli_math_acc_cast_fx<vNx4short_t, vNx4accint_t>(
                                                                mli_math_mul_fx<vNx4short_t, vNx4accint_t>(diffs, _frac), shift_in);

                            /* Calculate O/P */
                            vNx4short_t res = mli_math_sub_fx<vNx4short_t>(_lut_values, diffs_mul_frac_cast);

                            /* Store O/P */
                            activation_lut_store_output<io_T, convert>(output_ptr, res, lut, out_params);
                            output_ptr += _VDSP_NUM_8BIT_LANES;

                            _frac = frac;
                            if (convert || fx_with_in_offset) {
                                frac = mli_math_cast_fx<vNx4int_t, vNx4short_t>(x_int & mask);
                            } else {
                                frac = x & mask;
                            }

                            input_ptr  += _VDSP_NUM_8BIT_LANES;

                            _lut_values = lut_values;
                            _lut_values_next = lut_values_next;
                        }

                        /* perform linear interpolation */
                        vNx4short_t diffs = mli_math_sub_fx<vNx4short_t>(_lut_values, _lut_values_next);
                        vNx4short_t diffs_mul_frac_cast =  mli_math_acc_cast_fx<vNx4short_t, vNx4accint_t>(
                                mli_math_mul_fx<vNx4short_t, vNx4accint_t>(diffs, _frac), shift_in);

                        /* Calculate O/P */
                        vNx4short_t res = mli_math_sub_fx<vNx4short_t>(_lut_values, diffs_mul_frac_cast);

                        /* Store O/P */
                        activation_lut_store_output<io_T, convert>(output_ptr, res, lut, out_params);
                    } else {
                        for (int pos3 = remaining_part; pos3 < in->shape[3]; pos3 += _VDSP_NUM_8BIT_LANES) {
                            /* Load from LUT */
                            vNx4short_t lut_values = mli_prv_gather_load_nx4_samples(lut_data, lut_idx_int);
                            vNx4short_t lut_values_next = mli_prv_gather_load_nx4_samples(lut_data, lut_idx_int + 1);

                            load_input_and_get_lut_idx<io_T, convert, fx_with_in_offset>(input_ptr, x, x_int, lut_idx_int,
                                    in_frac_bits, preshift_in, shift_in, lut, in_params);

                            /* perform linear interpolation */
                            vNx4short_t diffs = mli_math_sub_fx<vNx4short_t>(lut_values, lut_values_next);
                            vNx4short_t diffs_mul_frac_cast =  mli_math_acc_cast_fx<vNx4short_t, vNx4accint_t>(
                                                                mli_math_mul_fx<vNx4short_t, vNx4accint_t>(diffs, frac), shift_in);

                            if (convert || fx_with_in_offset) {
                                frac = mli_math_cast_fx<vNx4int_t, vNx4short_t>(x_int & mask);
                            } else {
                                frac = x & mask;
                            }

                            /* Calculate O/P */
                            vNx4short_t res = mli_math_sub_fx<vNx4short_t>(lut_values, diffs_mul_frac_cast);

                            /* Store O/P */
                            activation_lut_store_output<io_T, convert>(output_ptr, res, lut, out_params);
                            input_ptr  += _VDSP_NUM_8BIT_LANES;
                            output_ptr += _VDSP_NUM_8BIT_LANES;
                        }
                    }

                }
            }
        }
    } else {
        // input data isn't more precise than LUT
        for (int pos0 = 0; pos0 < in->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in->shape[2]; pos2++) {
                    input_ptr  = in_ptr  + POS(in,  pos0, pos1, pos2, 0);
                    output_ptr = out_ptr + POS(out, pos0, pos1, pos2, 0);
                    if (remaining_part) {
                        vNx4short_t x = activation_lut_load_input<io_T, vNx4short_t>(input_ptr);
                        vNx4short_t res = mli::krn::activation_lut_vec_elem_no_interpolate<convert, fx_with_in_offset>
                                            (x, lut, in_frac_bits, in_params);
                        /* Store O/P */
                        activation_lut_store_output<io_T, convert>(output_ptr, res, remaining_part, lut, out_params);
                        input_ptr  += remaining_part;
                        output_ptr += remaining_part;
                    }
                    for (int pos3 = remaining_part; pos3 < in->shape[3]; pos3 += _VDSP_NUM_8BIT_LANES) {
                        vNx4short_t x = activation_lut_load_input<io_T, vNx4short_t>(input_ptr);
                        vNx4short_t res = mli::krn::activation_lut_vec_elem_no_interpolate<convert, fx_with_in_offset>
                                            (x, lut, in_frac_bits, in_params);
                        /* Store O/P */
                        activation_lut_store_output<io_T, convert>(output_ptr, res, lut, out_params);
                        input_ptr  += _VDSP_NUM_8BIT_LANES;
                        output_ptr += _VDSP_NUM_8BIT_LANES;
                    }
                }
            }
        }
    }
}

template <typename io_T, bool convert, bool fx_with_in_offset>
static MLI_FORCE_INLINE void compute_activation_lut(
        const struct generic_tensor_private_t<MLI_PTR(io_T)> *in,
        struct generic_tensor_private_t<MLI_OUT_PTR(io_T)> *out,
        const mli_lut *lut,
        int8_t in_frac_bits,
        const struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params) {
    compute_activation_lut_func<io_T, convert, fx_with_in_offset>(in->ptr, out->ptr, in, out, lut, in_frac_bits, in_params, out_params);
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_PRV_LUT_VDSP_H_
