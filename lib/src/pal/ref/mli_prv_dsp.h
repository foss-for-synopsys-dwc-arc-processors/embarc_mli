/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _REF_MLI_PRV_DSP_H_
#define _REF_MLI_PRV_DSP_H_

#include "../mli_math.h"
#include "mli_math_macros.h"
#include "mli_mem_info.h"
#include "mli_private_types.h"

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
PRAGMA_CLANG(diagnostic push)
PRAGMA_CLANG(diagnostic ignored "-Wcast-align")

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
    *o_ptr = (int8_t) mli_math_sat_fx<int32_t>(temp, 24);
}

static MLI_FORCE_INLINE void  mli_prv_clip_and_store_output(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        int32_t * ip_in,
        const int out_shift) {
    int32_t temp = mli_math_asr_rnd_fx<int32_t>(*ip_in, out_shift);
    *o_ptr = (int16_t) mli_math_sat_fx<int32_t>(temp, 16);
}

static MLI_FORCE_INLINE void  mli_prv_clip_and_store_output(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        int64_t * ip_in,
        const int out_shift) {
    int64_t temp = mli_math_asr_rnd_fx<int64_t>(*ip_in, out_shift);
    *o_ptr = (int16_t) mli_math_sat_fx<int64_t>(temp, 48);
}

static MLI_FORCE_INLINE void mli_prv_clip_and_store_output_v(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        int64_t ip_in,
        const int out_shift) {
    mli_prv_clip_and_store_output(o_ptr, &ip_in, out_shift);
}

static MLI_FORCE_INLINE void mli_prv_clip_and_store_output_v(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        int32_t ip_in,
        const int out_shift) {
    mli_prv_clip_and_store_output(o_ptr, &ip_in, out_shift);
}

static MLI_FORCE_INLINE void mli_prv_clip_and_store_output_v(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        int64_t ip_in,
        const int out_shift,
        int num) {
    mli_prv_clip_and_store_output(o_ptr, &ip_in, out_shift);
}

static MLI_FORCE_INLINE void mli_prv_clip_and_store_output_v(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        int32_t ip_in,
        const int out_shift,
        int num) {
    mli_prv_clip_and_store_output(o_ptr, &ip_in, out_shift);
}

//=========================================================================

static MLI_FORCE_INLINE void  mli_prv_shift_clip_and_store_output(
        MLI_PTR(int8_t) __restrict o_ptr,
        int32_t * ip_in,
        const int out_shift) {
    int32_t temp = mli_math_asr_rnd_fx<int32_t>(*ip_in, out_shift);
    *o_ptr = (int8_t) mli_math_sat_fx<int32_t>(temp, 24);
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

template<typename T>
static MLI_FORCE_INLINE T mli_prv_init_accu();

template<>
MLI_FORCE_INLINE int32_t mli_prv_init_accu<int32_t>() {
    int32_t acc = 0;
    return acc;
}

template<>
MLI_FORCE_INLINE int64_t mli_prv_init_accu<int64_t>() {
    int64_t acc = 0;
    return acc;
}

static MLI_FORCE_INLINE int32_t  mli_prv_init_accu(int8_t inp_val) {
    int32_t acc = inp_val;
    return acc;
}

static MLI_FORCE_INLINE int32_t  mli_prv_init_accu(int32_t inp_val) {
    int32_t acc = inp_val;
    return acc;
}


static MLI_FORCE_INLINE int32_t mli_prv_init_accu_v(int8_t inp_val) {
    int32_t acc = inp_val;
    return acc;
}

static MLI_FORCE_INLINE int64_t mli_prv_init_accu_v(int16_t inp_val) {
    int64_t acc = inp_val;
    return acc;
}

static MLI_FORCE_INLINE int32_t  mli_prv_init_accu_with_bias(
        const MLI_PTR(int8_t) __restrict in,
        const int8_t bias,
        const int bias_shift) {
    int32_t accu = mli_math_asr_rnd_fx<int32_t>((int32_t) bias, -bias_shift);
    return accu;
}

static MLI_FORCE_INLINE int32_t  mli_prv_init_accu_with_bias(
        const MLI_PTR(int16_t) __restrict in,
        const int8_t bias,
        const int bias_shift) {
    int32_t accu = mli_math_asr_rnd_fx<int32_t>((int32_t) bias, -bias_shift);
    return accu;
}

template<typename T>
static MLI_FORCE_INLINE T mli_prv_init_accu_with_bias_v(
        const int16_t bias,
        const int bias_shift);

template<>
MLI_FORCE_INLINE mli_acc40_t mli_prv_init_accu_with_bias_v(
        const int16_t bias,
        const int bias_shift) {
    mli_acc40_t accu = mli_math_asl_fx((mli_acc40_t) bias, bias_shift);
    return accu;
}

template<>
MLI_FORCE_INLINE mli_acc32_t mli_prv_init_accu_with_bias_v(
        const int16_t bias,
        const int bias_shift) {
    mli_acc32_t accu = mli_math_asl_fx((mli_acc32_t) bias, bias_shift);
    return accu;
}

// Multiply and accumulate
//=========================================================================
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

template<typename in_T, typename out_T>
static MLI_FORCE_INLINE out_T mli_prv_init_v(in_T in) {
    return static_cast<out_T>(in);
}

PRAGMA_CLANG(diagnostic pop)

#endif // _REF_MLI_PRV_DSP_H_