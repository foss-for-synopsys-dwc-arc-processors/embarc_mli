/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _VDSP_MLI_PRV_DSP_H_
#define _VDSP_MLI_PRV_DSP_H_

#include "../mli_math.h"
#include "mli_math_macros.h"
#include "mli_private_types.h"
#include "arc_vector_ext.h"



//=========================================================================
// This file contains functions that combine the math functions from
// mli_math.h with the loadstore functions from mli_prv_load_store.h
// It also implements the template versions for the different types
// and type combinations.
//=========================================================================


//=========================================================================
//
// Declaration
//
//=========================================================================


// Multiply and accumulate for vectors of 1, 2, and 4 elements
//=========================================================================
template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE void  mli_prv_load_mac(
        acc_T * accu,
        MLI_PTR(in_T) __restrict in,
        MLI_PTR(w_T) __restrict k);

template < typename in_T, typename w_T, typename acc_T >
static MLI_FORCE_INLINE void  mli_prv_load_mac(
        acc_T * accu,
        MLI_PTR(in_T) __restrict in,
        w_T k);


//=========================================================================
//
// Implementation
//
//=========================================================================

// Depending on memory alignment of input pointers, certain functions below will perform
// unaligned loads/stores. Since the core supports this, we disable the related compiler warning.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-align"

static MLI_FORCE_INLINE unsigned  mli_prv_fx_init_dsp_ctrl() {
    return 0;
}

static MLI_FORCE_INLINE void  mli_prv_clip_div_and_store_result(
        MLI_PTR(int16_t) __restrict o_ptr,
        const int kernel_size,
        const int32_t accum_32) {
    int32_t temp = mli_math_asl_fx<int32_t>(accum_32, 1) / kernel_size;
    temp = mli_math_asl_fx<int32_t>(temp, 16 - 1);
    *o_ptr = (int16_t) mli_math_cast_fx<int32_t, int16_t>(temp);
}

static MLI_FORCE_INLINE void  mli_prv_clip_div_and_store_result(
        MLI_PTR(int8_t) __restrict o_ptr,
        const int kernel_size,
        const int32_t accum_32) {
    int32_t temp = mli_math_asl_fx<int32_t>(accum_32, 1) / kernel_size;
    temp = mli_math_asl_fx<int32_t>(temp, 32 - sizeof(int8_t) * 8 - 1);
    *o_ptr = (int8_t) mli_math_cast_fx<int32_t, int8_t>(temp);
}

//=========================================================================

static MLI_FORCE_INLINE void  mli_prv_clip_and_store_output(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        int32_t * ip_in,
        const int out_shift) {
    int32_t temp = mli_math_asr_rnd_fx<int32_t>(*ip_in, out_shift);
    *o_ptr = (int8_t) mli_math_sat_fx<int32_t>(temp, 8);
}

static MLI_FORCE_INLINE void  mli_prv_clip_and_store_output(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        int32_t * ip_in,
        const int out_shift) {
    int32_t temp = mli_math_asr_rnd_fx<int32_t>(*ip_in, out_shift);
    *o_ptr = (int16_t) mli_math_sat_fx<int32_t>(temp, 16);
}

//=========================================================================

static MLI_FORCE_INLINE void  mli_prv_shift_clip_and_store_output(
        MLI_PTR(int8_t) __restrict o_ptr,
        int32_t * ip_in,
        const int out_shift) {
    int32_t temp = mli_math_asr_rnd_fx<int32_t>(*ip_in, out_shift);
    *o_ptr = (int8_t) mli_math_sat_fx<int32_t>(temp, 8);
}

static MLI_FORCE_INLINE void  mli_prv_shift_clip_and_store_output(
        MLI_PTR(int16_t) __restrict o_ptr,
        int32_t * ip_in,
        const int out_shift) {
    int32_t temp = mli_math_asr_rnd_fx<int32_t>(*ip_in, out_shift);
    *o_ptr = (int16_t) mli_math_sat_fx<int32_t>(temp, 16);
}

//=========================================================================

static MLI_FORCE_INLINE void  mli_prv_clip_relu_store_output(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        int32_t conv_out,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit) {
    conv_out = mli_math_asr_rnd_fx<int32_t>(conv_out, out_shift);
    // no saturation needed because ReLu clipping is done in 32bit domain.
    // ReLU truncation
    conv_out = MIN(conv_out, val_max_limit);
    conv_out = MAX(conv_out, val_min_limit);

    int16_t out_val = (int16_t) conv_out;

    // Write result
    *o_ptr = out_val;
}

static MLI_FORCE_INLINE void  mli_prv_clip_relu_store_output(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        int32_t conv_out,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit) {
    conv_out = mli_math_asr_rnd_fx<int32_t>(conv_out, out_shift);

    // no saturation needed because ReLu clipping is done in 32bit domain.
    // ReLU truncation
    conv_out = MIN(conv_out, val_max_limit);
    conv_out = MAX(conv_out, val_min_limit);

    int8_t out_val = (int8_t) conv_out;

    // Write result
    *o_ptr = out_val;
}

static MLI_FORCE_INLINE void  mli_prv_clip_relu_store_output(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        int32_t conv_out,
        const s8asym_quant_specific_params* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit) {

    int64_t accu_scaled = mli_math_mul_fx<int32_t, int64_t>(conv_out, quant_params->out_mul);
    int64_t shifted_acc = mli_math_asr_rnd_fx<int64_t>(accu_scaled, (-quant_params->out_shift));
    int16_t out_no_offset = (int16_t)mli_math_sat_fx<int64_t>(shifted_acc, 16);
    int16_t out_with_offset = mli_math_add_fx<int16_t>(out_no_offset, quant_params->out_offset);

    // no saturation needed because ReLu clipping is done in 32bit domain.
    // ReLU truncation
    out_with_offset = MIN(out_with_offset, val_max_limit);
    out_with_offset = MAX(out_with_offset, val_min_limit);

    // Write result
    *o_ptr = (int8_t)out_with_offset;
}

// Initialize Accumulator
template<typename T>
static MLI_FORCE_INLINE T mli_prv_init_accu();

template<>
MLI_FORCE_INLINE vNx4accshort_t mli_prv_init_accu<vNx4accshort_t>() {
    return vvcmpy((vNx4char_t)0, (int8_t)0);
}

template<>
MLI_FORCE_INLINE vNx2accint_t mli_prv_init_accu<vNx2accint_t>() {
    return vvcmpy((vNx2short_t)0, (short)0);
}

template<>
MLI_FORCE_INLINE vNx4accint_t mli_prv_init_accu<vNx4accint_t>() {
    vNx4accint_t r;
    r.lo = vvcmpy((vNx2short_t)0, (int16_t)0);
    r.hi = vvcmpy((vNx2short_t)0, (int16_t)0);
    return r;
}

static MLI_FORCE_INLINE int32_t  mli_prv_init_accu(int8_t inp_val) {
    int32_t acc = inp_val;
    // _setacc(acc, 1);

    return acc;
}

static MLI_FORCE_INLINE int32_t  mli_prv_init_accu(int32_t inp_val) {
    int32_t acc = inp_val;
    // _setacc(acc, 1);

    return acc;
}

static MLI_FORCE_INLINE int32_t  mli_prv_init_accu_with_bias(
        const MLI_PTR(int8_t) __restrict in,
        const int8_t bias,
        const int bias_shift) {
    int32_t accu = mli_math_asr_rnd_fx<int32_t>((int32_t) bias, -bias_shift);
    // _setacc(accu, 1);

    return accu;
}

static MLI_FORCE_INLINE int32_t  mli_prv_init_accu_with_bias(
        const MLI_PTR(int16_t) __restrict in,
        const int8_t bias,
        const int bias_shift) {
    int32_t accu = mli_math_asr_rnd_fx<int32_t>((int32_t) bias, -bias_shift);
    // _setacc(accu, 1);

    return accu;
}

// Multiply and accumulate for vectors of 1, 2, and 4 elements
//=========================================================================
// Note:
// Some implementations make use of intrinsics that make use of the HW accumulator
// without passing it as an input argument to the intrinsic.
// For this to work correct it is important to use these functions in combination
// with the accumulator init functions that are defined elsewhere in this file.
//
static MLI_FORCE_INLINE void mli_prv_load_mac(
        int32_t * accu,
        const MLI_PTR(int16_t) __restrict in,
        const MLI_PTR(int8_t) __restrict k) {
    *accu += mli_math_mul_fx<int16_t, int32_t>(*in, (int16_t)(*k << 8));
}

static MLI_FORCE_INLINE void mli_prv_load_mac(
        int32_t * accu,
        const MLI_PTR(int16_t) __restrict in,
        const MLI_PTR(int16_t) __restrict k) {
    *accu += mli_math_mul_fx<int16_t, int32_t>(*in, *k);
}

// Accumulator Math
//////////////////////////////////////////////////
// double vector accumulator types vvconvert (targetsize is e.g. TARGET_SZ_8)
//////////////////////////////////////////////////
static MLI_FORCE_INLINE vNx2int_t
mli_math_acc_ashift_fx(vNx2accint_t acc, int out_shift, int target_sz) {
    vNx2int_t v;
    unsigned ctrlword = target_sz | SIGNED | SAT | SHIFT(out_shift);
    v.lo = to_vNint_t(vvconvert(__vacc_lo(acc), ctrlword));
    v.hi = to_vNint_t(vvconvert(__vacc_hi(acc), ctrlword));
    return v;
}

static MLI_FORCE_INLINE vNx4short_t
mli_math_acc_ashift_fx(vNx4accshort_t acc, int out_shift, int target_sz) {
    vNx4short_t v;
    unsigned ctrlword = target_sz | SIGNED | SAT | SHIFT(out_shift);
    v.lo = to_vNx2short_t(vvconvert(__vacc_lo(acc), ctrlword));
    v.hi = to_vNx2short_t(vvconvert(__vacc_hi(acc), ctrlword));
    return v;
}

static MLI_FORCE_INLINE vNx4int_t
mli_math_acc_ashift_fx(vNx4accint_t acc, int out_shift, int target_sz) {
    vNx4int_t v;
    unsigned ctrlword = target_sz | SIGNED | SAT | SHIFT(out_shift);
    v.lo.lo = to_vNint_t(vvconvert(__vacc_lo(acc.lo), ctrlword));
    v.lo.hi = to_vNint_t(vvconvert(__vacc_hi(acc.lo), ctrlword));
    v.hi.lo = to_vNint_t(vvconvert(__vacc_lo(acc.hi), ctrlword));
    v.hi.hi = to_vNint_t(vvconvert(__vacc_hi(acc.hi), ctrlword));
    return v;
}
#pragma clang diagnostic pop

#endif // _VDSP_MLI_PRV_DSP_H_
