/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_DSP_H_
#define _MLI_PRV_DSP_H_

#include <arc/arc_reg.h>        // Defines DSP_CTRL

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_math.h"
#include "mli_math_macros.h"
#include "mli_prv_load_store.h"
#include "mli_private_types.h"

#include <arc/arc_intrinsics.h>

#if ((_ARCVER >= 0x50) && (_ARCVER < 0x60))
#define _ARCVER_ARCv2HS
#endif

#if defined(_ARCVER_ARCv2HS)
#define LOOP_PIPELINE_ENABLE _Pragma("clang loop pipeline(enable)")
#else
#define LOOP_PIPELINE_ENABLE 
#endif

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
static inline void __attribute__ ((always_inline)) mli_prv_load_mac(
        acc_T * accu,
        MLI_PTR(in_T) __restrict in,
        MLI_PTR(w_T) __restrict k);

template < typename in_T, typename w_T, typename acc_T >
static inline void __attribute__ ((always_inline)) mli_prv_load_mac(
        acc_T * accu,
        MLI_PTR(in_T) __restrict in,
        w_T k);

template < typename in_T, typename w_T, typename acc_T >
static inline void __attribute__ ((always_inline)) mli_prv_load_mac_vec2(
        acc_T * accu,
        MLI_PTR(in_T) __restrict in,
        MLI_PTR(w_T) __restrict k);

template < typename in_T, typename w_T, typename acc_T >
static inline void __attribute__ ((always_inline)) mli_prv_load_mac_vec4(
        acc_T * accu,
        MLI_PTR(in_T) __restrict in,
        MLI_PTR(w_T) __restrict k);

//=========================================================================
//
// Implementation
//
//=========================================================================


static inline void __attribute__ ((always_inline)) mli_prv_clip_div_and_store_result(
        MLI_PTR(int16_t) __restrict o_ptr, 
        const int kernel_size, 
        const int32_t accum_32) {
    int32_t temp = fx_asl_q31(accum_32, 1) / kernel_size;
    temp = fx_asl_q31(temp, 16 - 1);
    *o_ptr = (int16_t) fx_q15_cast_rnd_q31(temp);
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_div_and_store_result(
        MLI_PTR(int8_t) __restrict o_ptr, 
        const int kernel_size, 
        const int32_t accum_32) {
    int32_t temp = fx_asl_q31(accum_32, 1) / kernel_size;
    temp = fx_asl_q31(temp, 32 - sizeof(int8_t) * 8 - 1);
    *o_ptr = (int8_t) fx_q7_cast_rnd_q31(temp);
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_div_and_store_result(
            MLI_PTR(int16_t) __restrict o_ptr, 
            const int kernel_size, 
            const accum40_t accum_40) {
    int32_t temp = (int32_t) fx_q31_cast_a40(accum_40) / kernel_size;
    temp = fx_asl_q31(temp, 32 - sizeof(int16_t) * 8 - 1);
    *o_ptr = (int16_t) fx_q15_cast_rnd_q31(temp);
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_div_and_store_result(
        MLI_PTR(int8_t) __restrict o_ptr,
        const int kernel_size,
        const accum40_t accum_40) {
    int32_t temp = (int32_t) fx_q31_cast_a40(accum_40) / kernel_size;
    temp = fx_asl_q31(temp, 32 - sizeof(int8_t) * 8 - 1);
    *o_ptr = (int16_t) fx_q7_cast_rnd_q31(temp);
}

//=========================================================================

static inline void __attribute__ ((always_inline)) mli_prv_clip_and_store_output(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        accum40_t * ip_out_v,
        const int out_shift) {
    *o_ptr = fx_q15_cast_asl_rnd_a40(*ip_out_v, 16 - out_shift - 1);
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_and_store_output(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        accum40_t * ip_out_v,
        const int out_shift) {
    *o_ptr = fx_q7_cast_asl_rnd_a40(*ip_out_v, 32 - sizeof(int8_t) * 8 - out_shift - 1);
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_and_store_output(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        int32_t * ip_in, 
        const int out_shift) {
    q31_t temp = fx_asr_rnd_q31(*ip_in, out_shift);
    temp = fx_asl_q31(temp, 32 - sizeof(int8_t) * 8);
    *o_ptr = (int8_t) fx_q7_cast_q31(temp);
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_and_store_output(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        int32_t * ip_in, 
        const int out_shift) {
    q31_t temp = fx_asr_rnd_q31(*ip_in, out_shift);
    temp = fx_asl_q31(temp, 32 - sizeof(int16_t) * 8);
    *o_ptr = (int16_t) fx_q15_cast_q31(temp);
}

//=========================================================================

static inline void __attribute__ ((always_inline)) mli_prv_shift_clip_and_store_output(
        MLI_PTR(int16_t) __restrict o_ptr,
        accum40_t * ip_out_v,
        const int out_shift) {
    *o_ptr = fx_q15_cast_nf_asl_rnd_a40(*ip_out_v, 32 - sizeof(int16_t) * 8 - out_shift);
}

static inline void __attribute__ ((always_inline)) mli_prv_shift_clip_and_store_output(
        MLI_PTR(int8_t) __restrict o_ptr,
        accum40_t * ip_out_v,
        const int out_shift) {
    *o_ptr = fx_q7_cast_nf_asl_rnd_a40(*ip_out_v, 32 - sizeof(int8_t) * 8 - out_shift);
}

static inline void __attribute__ ((always_inline)) mli_prv_shift_clip_and_store_output(
        MLI_PTR(int8_t) __restrict o_ptr,
        int32_t * ip_in,
        const int out_shift) {
    q31_t temp = fx_asr_rnd_q31(*ip_in, out_shift);
    temp = fx_asl_q31(temp, 32 - sizeof(int8_t) * 8);
    *o_ptr = (int8_t) fx_q7_cast_q31(temp);
}

static inline void __attribute__ ((always_inline)) mli_prv_shift_clip_and_store_output(
        MLI_PTR(int16_t) __restrict o_ptr,
        int32_t * ip_in,
        const int out_shift) {
    q31_t temp = fx_asr_rnd_q31(*ip_in, out_shift);
    temp = fx_asl_q31(temp, 32 - sizeof(int16_t) * 8);
    *o_ptr = (int16_t) fx_q15_cast_q31(temp);
}

//=========================================================================

static inline void __attribute__ ((always_inline)) mli_prv_clip_and_store_output_v(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        __v2i32_t * acc_v, 
        const int out_shift) {
    v2i16_t out_v;

    (*acc_v)[0] = fx_asr_rnd_q31((*acc_v)[0], out_shift);
    (*acc_v)[1] = fx_asr_rnd_q31((*acc_v)[1], out_shift);
    (*acc_v)[0] = fx_asl_q31((*acc_v)[0], (32 - sizeof(int16_t)* 8));
    (*acc_v)[1] = fx_asl_q31((*acc_v)[1], (32 - sizeof(int16_t)* 8));
    (*acc_v)[0] = fx_asr_q31((*acc_v)[0], (32 - sizeof(int16_t)* 8));
    (*acc_v)[1] = fx_asr_q31((*acc_v)[1], (32 - sizeof(int16_t)* 8));

    out_v = __builtin_convertvector((*acc_v), v2i16_t);
    *((v2i16_t *) o_ptr) = out_v;
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_and_store_output_v(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        __v2i32_t * acc_v, 
        const int out_shift) {
    v2i8_t out_v;

    (*acc_v)[0] = fx_asr_rnd_q31((*acc_v)[0], out_shift);
    (*acc_v)[1] = fx_asr_rnd_q31((*acc_v)[1], out_shift);
    (*acc_v)[0] = fx_asl_q31((*acc_v)[0], (32 - sizeof(int8_t)* 8));
    (*acc_v)[1] = fx_asl_q31((*acc_v)[1], (32 - sizeof(int8_t)* 8));
    (*acc_v)[0] = fx_asr_q31((*acc_v)[0], (32 - sizeof(int8_t)* 8));
    (*acc_v)[1] = fx_asr_q31((*acc_v)[1], (32 - sizeof(int8_t)* 8));

    out_v = __builtin_convertvector((*acc_v), v2i8_t);
    *((v2i8_t *) o_ptr) = out_v;
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_and_store_output_v(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        v2accum40_t *__restrict acc_v, 
        const int out_shift) {
    v2q15_t out_v;

    out_v = fx_v2q15_cast_nf_asl_rnd_v2a40(*acc_v, (32 - sizeof(int16_t)* 8 - out_shift));
    v2q15_t * v2o_ptr = (v2q15_t *)o_ptr;
    *v2o_ptr = out_v;
}

v2i8_t FXAPI fx_v2q7_cast_nf_asl_rnd_v2a40(v2accum40_t VQ, int I) {
  v2q15_t r;
  int sel = I & 255;
  sel |= 0x0300;    // Vector accumulator select
  sel |= 0x0400;    // Saturation enable
  sel |= 0x0800;    // Signed
  sel |= 0x1000;    // Round using DSP_CTRL.RM
  sel |= (3<<14);   // Round at byte: 8b
  sel |= (0<<16);   // Clear PA bit sensitive, NO extra shift
  r = (v2q15_t) __v2acc40_getacc( VQ.q, sel );
  return __builtin_convertvector((r >> 8), v2i8_t);
}

static void __attribute__ ((always_inline)) mli_prv_clip_and_store_output_v(
        MLI_OUT_PTR(int8_t) __restrict o_ptr,
        v2accum40_t * __restrict acc_v,
        const int out_shift) {
    *((v2i8_t *) o_ptr) = fx_v2q7_cast_nf_asl_rnd_v2a40(*acc_v, (32 - sizeof(int8_t)* 8 - out_shift));
}

//=========================================================================

static inline void __attribute__ ((always_inline)) mli_prv_clip_relu_store_output_v(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        __v2i32_t * accu_v,
        const int out_shift, 
        const int16_t val_min_limit, 
        const int16_t val_max_limit) {
    v2i8_t out_v;
    __v2i32_t acc_v = *accu_v;

    acc_v[0] = fx_asr_rnd_q31(acc_v[0], out_shift);
    acc_v[1] = fx_asr_rnd_q31(acc_v[1], out_shift);

    // acc_v[0]
    // no saturation needed because ReLu clipping is done in 32bit domain.
    // ReLU truncation
    acc_v[0] = MIN(acc_v[0], val_max_limit);
    acc_v[1] = MIN(acc_v[1], val_max_limit);

    acc_v[0] = MAX(acc_v[0], val_min_limit);
    acc_v[1] = MAX(acc_v[1], val_min_limit);

    out_v = __builtin_convertvector(acc_v, v2i8_t);
    *((v2i8_t *) o_ptr) = out_v;
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_relu_store_output_v(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        __v2i32_t * accu_v,
        const int out_shift, 
        const int16_t val_min_limit, 
        const int16_t val_max_limit) {
    v2q15_t out_v;
    __v2i32_t acc_v = *accu_v;

    acc_v[0] = fx_asr_rnd_q31(acc_v[0], out_shift);
    acc_v[1] = fx_asr_rnd_q31(acc_v[1], out_shift);

    // acc_v[0]
    // no saturation needed because ReLu clipping is done in 32bit domain.
    // ReLU truncation
    acc_v[0] = MIN(acc_v[0], val_max_limit);
    acc_v[1] = MIN(acc_v[1], val_max_limit);

    acc_v[0] = MAX(acc_v[0], val_min_limit);
    acc_v[1] = MAX(acc_v[1], val_min_limit);

    out_v = __builtin_convertvector(acc_v, v2q15_t);
    *((v2q15_t *) o_ptr) = out_v;
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_relu_store_output_v(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        v2accum40_t *acc_v,
        const int out_shift, 
        const int16_t val_min_limit, 
        const int16_t val_max_limit) {
    v2q15_t out_v;
    v2q15_t v2_val_max_limit = { val_max_limit, val_max_limit };
    v2q15_t v2_val_min_limit = { val_min_limit, val_min_limit };
    //acc_v = acc_v >> out_shift;
    out_v = fx_v2q15_cast_nf_asl_rnd_v2a40(*acc_v, 16 - out_shift);
    // no saturation needed because ReLu clipping is done in 32bit domain.
    // ReLU truncation
    out_v = fx_min_v2q15(out_v, v2_val_max_limit);
    out_v = fx_max_v2q15(out_v, v2_val_min_limit);

    //out_v = __builtin_convertvector(acc_v, __v2i16_t);
    *((v2q15_t *) o_ptr) = out_v;
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_relu_store_output_v(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        v2accum40_t *acc_v,
        const int out_shift, 
        const int16_t val_min_limit, 
        const int16_t val_max_limit) {
    v2q15_t out_v;
    v2q15_t v2_val_max_limit = { val_max_limit, val_max_limit };
    v2q15_t v2_val_min_limit = { val_min_limit, val_min_limit };
    //acc_v = acc_v >> out_shift;
    out_v = fx_v2q15_cast_nf_asl_rnd_v2a40(*acc_v, 16 - out_shift);
    // no saturation needed because ReLu clipping is done in 32bit domain.
    // ReLU truncation
    out_v = fx_min_v2q15(out_v, v2_val_max_limit);
    out_v = fx_max_v2q15(out_v, v2_val_min_limit);

    v2i8_t out_v8 = __builtin_convertvector(out_v, v2i8_t);
    *((v2i8_t *) o_ptr) = out_v8;
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_relu_store_output_v(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        __v2i32_t *conv_out_v,
        const s8asym_quant_specific_params quant_params[],
        const int16_t val_min_limit,
        const int16_t val_max_limit) {
    accum72_t accu_scaled = fx_a72_mpy_q31((*conv_out_v)[0], quant_params[0].out_mul);
    int16_t out_no_offset_ch1 = fx_q15_cast_nf_asl_rnd_a72(accu_scaled, 64 - sizeof(int16_t) * 8 - quant_params[0].out_shift);

    accu_scaled = fx_a72_mpy_q31((*conv_out_v)[1], quant_params[1].out_mul);
    int16_t out_no_offset_ch2 = fx_q15_cast_nf_asl_rnd_a72(accu_scaled, 64 - sizeof(int16_t) * 8 - quant_params[1].out_shift);

    v2q15_t v2quant_out_offset = {quant_params[0].out_offset, quant_params[1].out_offset};

    v2q15_t v2out_no_offset = {out_no_offset_ch1, out_no_offset_ch2};

    v2q15_t v2val_max_limit = {val_max_limit, val_max_limit};
    v2q15_t v2val_min_limit = {val_min_limit, val_min_limit};
    v2q15_t v2out_offset = fx_add_v2q15(v2out_no_offset, v2quant_out_offset);

    // no saturation needed because ReLu clipping is done in 32bit domain.
    // ReLU truncation
    v2out_offset = fx_min_v2q15(v2out_offset, v2val_max_limit);
    v2out_offset = fx_max_v2q15(v2out_offset, v2val_min_limit);

    // Write result
    *((v2i8_t *) o_ptr) = __builtin_convertvector((v2out_offset), v2i8_t);
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_relu_store_output_inp_width_v(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        __v2i32_t *conv_out_v,
        const s8asym_quant_specific_params *quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit,
        const int next_out_indx) {
    accum72_t accu_scaled = fx_a72_mpy_q31((*conv_out_v)[0], quant_params->out_mul);
    int16_t out_no_offset_ch1 = fx_q15_cast_nf_asl_rnd_a72(accu_scaled, 64 - sizeof(int16_t) * 8 - quant_params->out_shift);

    accu_scaled = fx_a72_mpy_q31((*conv_out_v)[1], quant_params->out_mul);
    int16_t out_no_offset_ch2 = fx_q15_cast_nf_asl_rnd_a72(accu_scaled, 64 - sizeof(int16_t) * 8 - quant_params->out_shift);

    v2q15_t v2quant_out_offset = {quant_params->out_offset, quant_params->out_offset};

    v2q15_t v2out_no_offset = {out_no_offset_ch1, out_no_offset_ch2};

    v2q15_t v2val_max_limit = {val_max_limit, val_max_limit};
    v2q15_t v2val_min_limit = {val_min_limit, val_min_limit};
    v2q15_t v2out_offset = fx_add_v2q15(v2out_no_offset, v2quant_out_offset);

    // no saturation needed because ReLu clipping is done in 32bit domain.
    // ReLU truncation
    v2out_offset = fx_min_v2q15(v2out_offset, v2val_max_limit);
    v2out_offset = fx_max_v2q15(v2out_offset, v2val_min_limit);

    // Write result
    o_ptr[0]             = (int8_t)(v2out_offset[0]);
    o_ptr[next_out_indx] = (int8_t)(v2out_offset[1]);
}

//=========================================================================

static inline void __attribute__ ((always_inline)) mli_prv_clip_relu_store_output(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        accum40_t conv_out,
        const int out_shift, 
        const int16_t val_min_limit, 
        const int16_t val_max_limit) {
    int16_t out_val = fx_q15_cast_asl_rnd_a40(conv_out, 16 - out_shift - 1);

    out_val = MIN(out_val, val_max_limit);
    out_val = MAX(out_val, val_min_limit);

    // Write result
    *o_ptr = out_val;
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_relu_store_output(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        accum40_t conv_out,
        const int out_shift, 
        const int16_t val_min_limit, 
        const int16_t val_max_limit) {
    int8_t out_val = fx_q7_cast_asl_rnd_a40(conv_out, 32 - sizeof(int8_t) * 8 - out_shift - 1);

    out_val = MIN(out_val, val_max_limit);
    out_val = MAX(out_val, val_min_limit);

    // Write result
    *o_ptr = out_val;
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_relu_store_output(
        MLI_CONV_OUT_PTR(int16_t) __restrict o_ptr,
        int32_t conv_out,
        const int out_shift, 
        const int16_t val_min_limit, 
        const int16_t val_max_limit) {
    conv_out = fx_asr_rnd_q31(conv_out, out_shift);
    // no saturation needed because ReLu clipping is done in 32bit domain.
    // ReLU truncation
    conv_out = MIN(conv_out, val_max_limit);
    conv_out = MAX(conv_out, val_min_limit);

    int16_t out_val = (int16_t) conv_out;

    // Write result
    *o_ptr = out_val;
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_relu_store_output(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        int32_t conv_out,
        const int out_shift,
        const int16_t val_min_limit,
        const int16_t val_max_limit) {
    conv_out = fx_asr_rnd_q31(conv_out, out_shift);

    // no saturation needed because ReLu clipping is done in 32bit domain.
    // ReLU truncation
    conv_out = MIN(conv_out, val_max_limit);
    conv_out = MAX(conv_out, val_min_limit);

    int8_t out_val = (int8_t) conv_out;

    // Write result
    *o_ptr = out_val;
}

static inline void __attribute__ ((always_inline)) mli_prv_clip_relu_store_output(
        MLI_CONV_OUT_PTR(int8_t) __restrict o_ptr,
        int32_t conv_out,
        const s8asym_quant_specific_params* quant_params,
        const int16_t val_min_limit,
        const int16_t val_max_limit) {

    accum72_t accu_scaled = fx_a72_mpy_q31(conv_out, quant_params->out_mul);
    int16_t out_no_offset = fx_q15_cast_nf_asl_rnd_a72(accu_scaled, 64 - sizeof(int16_t) * 8 - quant_params->out_shift);
    int16_t out_with_offset = fx_add_q15(out_no_offset, quant_params->out_offset);

    // no saturation needed because ReLu clipping is done in 32bit domain.
    // ReLU truncation
    out_with_offset = MIN(out_with_offset, val_max_limit);
    out_with_offset = MAX(out_with_offset, val_min_limit);

    // Write result
    *o_ptr = (int8_t)out_with_offset;
}

template < typename io_T, typename w_T >
static inline int32_t __attribute__ ((always_inline)) mli_prv_qmpy_v4i16x8(
        const MLI_PTR(int8_t) __restrict pIn, 
        const MLI_PTR(int8_t) __restrict pWt) {
    unsigned tmp = *(MLI_PTR(unsigned)) pIn;
    unsigned in1_v2i16 = (unsigned) (_vsext2bhl(tmp));
    unsigned in2_v2i16 = (unsigned) (_vsext2bhm(tmp));
    unsigned wt1_v4i8 = (unsigned) ((*(MLI_PTR(unsigned)) pWt));
    _dmpyhbl(in1_v2i16, wt1_v4i8);
    return _dmachbm(in2_v2i16, wt1_v4i8);
}

template < typename io_T, typename w_T >
static inline int32_t __attribute__ ((always_inline)) mli_prv_qmac_v4i16x8(
        const MLI_PTR(int8_t) __restrict pIn, 
        const MLI_PTR(int8_t) __restrict pWt) {
    unsigned tmp = *(MLI_PTR(unsigned)) pIn;
    unsigned in1_v2i16 = (unsigned) (_vsext2bhl(tmp));
    unsigned in2_v2i16 = (unsigned) (_vsext2bhm(tmp));
    unsigned wt1_v4i8 = (unsigned) ((*(MLI_PTR(unsigned)) pWt));
    _dmachbl(in1_v2i16, wt1_v4i8);
    return _dmachbm(in2_v2i16, wt1_v4i8);
}

template < typename io_T, typename w_T >
static inline int32_t __attribute__ ((always_inline)) mli_prv_qmac_v4i16x8(
        const MLI_PTR(int16_t) __restrict pIn, 
        const MLI_PTR(int8_t) __restrict pWt) {
    unsigned tmp1 = *(MLI_PTR(unsigned)) pIn;
    unsigned tmp2 = *(MLI_PTR(unsigned)) (pIn + 2);
    unsigned wt1_v4i8 = (unsigned) ((*(MLI_PTR(unsigned)) pWt));
    _dmachbl(tmp1, wt1_v4i8);
    return _dmachbm(tmp2, wt1_v4i8);
}

template < typename io_T, typename w_T >
static inline int32_t __attribute__ ((always_inline)) mli_prv_qmpy_v4i16x8(
        const MLI_PTR(int16_t) __restrict pIn, 
        const MLI_PTR(int8_t) __restrict pWt) {
    unsigned tmp1 = *(MLI_PTR(unsigned)) pIn;
    unsigned tmp2 = *(MLI_PTR(unsigned)) (pIn + 2);
    unsigned wt1_v4i8 = (unsigned) ((*(MLI_PTR(unsigned)) pWt));
    _dmpyhbl(tmp1, wt1_v4i8);
    return _dmachbm(tmp2, wt1_v4i8);
}

static inline v2accum40_t __attribute__ ((always_inline)) mli_prv_init_accu_v(int16_t inp_val) {
    v2accum40_t acc_v = {inp_val, inp_val};

    return acc_v;
}

static inline __v2i32_t __attribute__ ((always_inline)) mli_prv_init_accu_v(int8_t inp_val) {
    __v2i32_t acc_v = {inp_val, inp_val};

    return acc_v;
}

static inline int32_t __attribute__ ((always_inline)) mli_prv_init_accu(int8_t inp_val) {
    int32_t acc = inp_val;
    _setacc(acc, 1);

    return acc;
}

static inline int32_t __attribute__ ((always_inline)) mli_prv_init_accu(int32_t inp_val) {
    int32_t acc = inp_val;
    _setacc(acc, 1);

    return acc;
}

static inline accum40_t __attribute__ ((always_inline)) mli_prv_init_accu(int16_t inp_val) {
    accum40_t acc = {inp_val};

    return acc;
}

static inline v2accum40_t __attribute__ ((always_inline)) mli_prv_init_accu_with_bias_v(
        const MLI_PTR(int16_t) __restrict in, 
        const int16_t bias, 
        const int bias_shift) {
    v2q15_t v2bias = {bias, bias};
    v2q15_t v2one = {0x0001, 0x0001};
    v2accum40_t acc_v = fx_v2a40_mpy_nf_v2q15(v2bias, v2one);
    acc_v = fx_asl_v2a40_n(acc_v, bias_shift);

    return acc_v;
}

static inline __v2i32_t __attribute__ ((always_inline)) mli_prv_init_accu_with_bias_v(
        const MLI_PTR(int8_t) __restrict in, 
        const int8_t bias, 
        const int bias_shift) {
    int32_t accu = fx_asr_rnd_q31((int32_t) bias, -bias_shift);
    __v2i32_t accu_v = { accu, accu };
    return accu_v;
}

#ifdef USE_40BIT_ACCU_FOR_16x8
static inline v2accum40_t __attribute__ ((always_inline)) mli_prv_init_accu_with_bias_v(
        const MLI_PTR(int16_t) __restrict in, 
        const int8_t bias, 
        const int bias_shift) {
    v2q15_t v2bias = {bias, bias};
    v2q15_t v2one = {0x0001, 0x0001};
    v2accum40_t acc_v = fx_v2a40_mpy_nf_v2q15(v2bias, v2one);
    acc_v = fx_asl_v2a40_n(acc_v, bias_shift);

    return acc_v;
}
#else
static inline __v2i32_t __attribute__ ((always_inline)) mli_prv_init_accu_with_bias_v(
        const MLI_PTR(int16_t) __restrict in, 
        const int8_t bias, 
        const int bias_shift) {
    int32_t accu = fx_asr_rnd_q31((int32_t) bias, -bias_shift);
    __v2i32_t accu_v = { accu, accu };
    return accu_v;
}

#endif

static inline accum40_t __attribute__ ((always_inline)) mli_prv_init_accu_with_bias(
        const MLI_PTR(int16_t) __restrict in,
        const int16_t bias,
        const int bias_shift) {
    accum40_t accu = fx_a40_mpy_q15(bias, 1);
    accu = fx_asl_a40(accu, bias_shift);

    return accu;
}

static inline int32_t __attribute__ ((always_inline)) mli_prv_init_accu_with_bias(
        const MLI_PTR(int8_t) __restrict in,
        const int8_t bias,
        const int bias_shift) {
    int32_t accu = fx_asr_rnd_q31((int32_t) bias, -bias_shift);
    _setacc(accu, 1);

    return accu;
}
#ifdef USE_40BIT_ACCU_FOR_16x8

static inline accum40_t __attribute__ ((always_inline)) mli_prv_init_accu_with_bias(
        const MLI_PTR(int16_t) __restrict in,
        const int8_t bias,
        const int bias_shift) {
    accum40_t accu = fx_a40_mpy_q15(bias, 1);
    accu = fx_asl_a40(accu, bias_shift);

    return accu;
}
#else
static inline int32_t __attribute__ ((always_inline)) mli_prv_init_accu_with_bias(
        const MLI_PTR(int16_t) __restrict in,
        const int8_t bias,
        const int bias_shift) {
    int32_t accu = fx_asr_rnd_q31((int32_t) bias, -bias_shift);
    _setacc(accu, 1);

    return accu;
}
    
#endif
static inline v2q15_t __attribute__ ((always_inline)) mli_prv_load_add_vec2(
        const MLI_PTR(int16_t) __restrict in, 
        const MLI_PTR(int16_t) __restrict k) {
   return fx_add_v2q15(mli_prv_load_2_samples(in), mli_prv_load_2_samples(k));
}

static inline v2i8_t __attribute__ ((always_inline)) mli_prv_load_add_vec2(
        const MLI_PTR(int8_t) __restrict in, 
        const MLI_PTR(int8_t) __restrict k) {
    //in case with AGU repacking it should give improve of performance
    v2q15_t res = fx_add_v2q15(mli_prv_load_2_samples(in), mli_prv_load_2_samples(k));
    const v2u16_t sat_v2= {8, 8};
    return __builtin_convertvector(fx_sat_v2q15(res, sat_v2), v2i8_t);
}

static inline v2q15_t __attribute__ ((always_inline)) mli_prv_load_sub_vec2(
        const MLI_PTR(int16_t) __restrict in, 
        const MLI_PTR(int16_t) __restrict k) {
   return fx_sub_v2q15(mli_prv_load_2_samples(in), mli_prv_load_2_samples(k));
}

static inline v2i8_t __attribute__ ((always_inline)) mli_prv_load_sub_vec2(
        const MLI_PTR(int8_t) __restrict in, 
        const MLI_PTR(int8_t) __restrict k) {
    v2q15_t res = fx_sub_v2q15(mli_prv_load_2_samples(in), mli_prv_load_2_samples(k));
    const v2u16_t sat_v2= {8, 8};
    return __builtin_convertvector(fx_sat_v2q15(res, sat_v2), v2i8_t);
}

static inline v2q15_t __attribute__ ((always_inline)) mli_prv_load_max_vec2(
        const MLI_PTR(int16_t) __restrict in, 
        const MLI_PTR(int16_t) __restrict k) {
   return fx_max_v2q15(mli_prv_load_2_samples(in), mli_prv_load_2_samples(k));
}

static inline v2q15_t __attribute__ ((always_inline)) mli_prv_load_max_vec2(
        const MLI_PTR(int8_t) __restrict in, 
        const MLI_PTR(int8_t) __restrict k) {
   return fx_max_v2q15(mli_prv_load_2_samples(in), mli_prv_load_2_samples(k));
}

static inline v2q15_t __attribute__ ((always_inline)) mli_prv_load_min_vec2(
        const MLI_PTR(int16_t) __restrict in,
        const MLI_PTR(int16_t) __restrict k) {
   return fx_min_v2q15(mli_prv_load_2_samples(in), mli_prv_load_2_samples(k));
}

static inline v2q15_t __attribute__ ((always_inline)) mli_prv_load_min_vec2(
        const MLI_PTR(int8_t) __restrict in, 
        const MLI_PTR(int8_t) __restrict k) {
   return fx_min_v2q15(mli_prv_load_2_samples(in), mli_prv_load_2_samples(k));
}

// Multiply and accumulate for vectors of 1, 2, and 4 elements
//=========================================================================
// Note:
// Some implementations make use of intrinsics that make use of the HW accumulator
// without passing it as an input argument to the intrinsic.
// For this to work correct it is important to use these functions in combination
// with the accumulator init functions that are defined elsewhere in this file.
//

static inline void __attribute__ ((always_inline)) mli_prv_load_mac(
        accum40_t * accu,
        const MLI_PTR(int16_t) __restrict in,
        const MLI_PTR(int16_t) __restrict k) {
    *accu = fx_a40_mac_q15(*accu, *in, *k);
}

static inline void __attribute__ ((always_inline)) mli_prv_load_mac(
        int32_t * accu,
        const MLI_PTR(int8_t) __restrict in,
        const MLI_PTR(int8_t) __restrict k) {
    /* casting the in pointer to unsigned to make sure no sign extension happens on the load
     * this way the 'second' byte contains zeros. and it is safe to use dmac.
     * the sign extension happens inside the dmachbl operation.
     * for the load of 'k' we need sign extension because we need a 16bit value.
     * the value of the second half is don't care because it will be multiplied by 0
     */
    *accu = _dmachbl(*k, *(MLI_PTR(uint8_t)) in);
}

static inline void __attribute__ ((always_inline)) mli_prv_load_mac(
        int32_t * accu, const MLI_PTR(int16_t) __restrict in,
        const MLI_PTR(int8_t) __restrict k) {
    /* casting the in pointer to unsigned to make sure no sign extension happens on the load
     * this way the 'second' byte contains zeros. and it is safe to use dmac.
     * the sign extension happens inside the dmachbl operation.
     * for the load of 'in' we need sign extension because we need a 16bit value.
     * the value of the second half is don't care because it will be multiplied by 0
     */
    *accu = _dmachbl(*in, *(MLI_PTR(uint8_t)) k);
}

static inline void __attribute__ ((always_inline)) mli_prv_load_mac(
        accum40_t * accu,
        const MLI_PTR(int16_t) __restrict in,
        const MLI_PTR(int8_t) __restrict k) {
    *accu = fx_a40_mac_q15(*accu, *in, *k);
}

static inline void __attribute__ ((always_inline)) mli_prv_load_mac(
        accum40_t * accu,
        const MLI_PTR(int16_t) __restrict in,
        const int16_t k) {
    *accu = fx_a40_mac_q15(*accu, *in, k);
}

static inline void __attribute__ ((always_inline)) mli_prv_load_mac(
        int32_t * accu,
        const MLI_PTR(int8_t) __restrict in,
        const int8_t k) {
    /* casting the in pointer to unsigned to make sure no sign extension happens on the load
     * this way the 'second' byte contains zeros. and it is safe to use dmac.
     * the sign extension happens inside the dmachbl operation.
     * for the load of 'k' we need sign extension because we need a 16bit value.
     * the value of the second half is don't care because it will be multiplied by 0
     */
    *accu = _dmachbl(k, *(MLI_PTR(uint8_t)) in);
}

static inline void __attribute__ ((always_inline)) mli_prv_load_mac(
        int32_t * accu, const MLI_PTR(int16_t) __restrict in,
        const int8_t k) {
    /* casting the in pointer to unsigned to make sure no sign extension happens on the load
     * this way the 'second' byte contains zeros. and it is safe to use dmac.
     * the sign extension happens inside the dmachbl operation.
     * for the load of 'in' we need sign extension because we need a 16bit value.
     * the value of the second half is don't care because it will be multiplied by 0
     */
    *accu = _dmachbl(*in, (uint8_t)k);
}

static inline void __attribute__ ((always_inline)) mli_prv_load_mac(
        int32_t * accu,
        const MLI_PTR(int8_t) __restrict in,
        const int16_t k) {
    /* casting the in pointer to unsigned to make sure no sign extension happens on the load
     * this way the 'second' byte contains zeros. and it is safe to use dmac.
     * the sign extension happens inside the dmachbl operation.
     * for the load of 'k' we need sign extension because we need a 16bit value.
     * the value of the second half is don't care because it will be multiplied by 0
     */
    *accu = _dmachbl(k, *(MLI_PTR(uint8_t)) in);
}

static inline void __attribute__ ((always_inline)) mli_prv_load_mac_vec2(
        accum40_t * accu, 
        const MLI_PTR(int16_t) __restrict in, 
        const MLI_PTR(int16_t) __restrict k) {
    *accu = fx_a40_dmac_v2q15(*accu, mli_prv_load_2_samples(in), mli_prv_load_2_samples(k));
}

static inline void __attribute__ ((always_inline)) mli_prv_load_mac_vec2(
        accum40_t * accu, 
        const MLI_PTR(int16_t) __restrict in, 
        const MLI_PTR(int8_t) __restrict k) {
    *accu = fx_a40_dmac_v2q15(*accu, mli_prv_load_2_samples(in), mli_prv_load_2_samples(k));
}

static inline void __attribute__ ((always_inline)) mli_prv_load_mac_vec2(
        int32_t * accu, 
        const MLI_PTR(int8_t) __restrict in, 
        const MLI_PTR(int8_t) __restrict k) {
    int16_t two8bitvalues = *(MLI_PTR(int16_t)) in;
    *accu = _dmachbl((int32_t) mli_prv_load_2_samples(k), two8bitvalues);
}

static inline void __attribute__ ((always_inline)) mli_prv_load_mac_vec2(
        int32_t * accu, 
        const MLI_PTR(int16_t) __restrict in, 
        const MLI_PTR(int8_t) __restrict k) {
    int16_t two8bitvalues = *(MLI_PTR(int16_t)) k;
    *accu = _dmachbl((int32_t) mli_prv_load_2_samples(in), two8bitvalues);
}

static inline void __attribute__ ((always_inline)) mli_prv_load_mac_vec2(
        accum40_t * accu, 
        const MLI_PTR(int8_t) in, 
        const MLI_PTR(int8_t) k) {

    *accu = fx_a40_dmac_v2q15(*accu, mli_prv_load_2_samples(in), mli_prv_load_2_samples(k));
}

template < typename in_T, typename w_T, typename acc_T > 
static inline void __attribute__ ((always_inline)) mli_prv_load_mac_vec4(acc_T * accu, MLI_PTR(in_T) in, MLI_PTR(w_T) k) {
    mli_prv_load_mac_vec2(accu, in, k);
    in += 2;
    k += 2;
    mli_prv_load_mac_vec2(accu, in, k);
}

//=========================================================================
//  Multiply and accumulate for 'in' vector (4 elements) with constant k 
//  
//  constant k  repilcate to v2k
//  count the sum: 
//        accu = accu + v2k.h1*sext(four8bitvalues.b1) + v2k.h0*sext(four8bitvalues.b0) 
//        accu = accu + v2k.h1*sext(four8bitvalues.b3) + v2k.h0*sext(four8bitvalues.b2) 
//=========================================================================
static inline void __attribute__ ((always_inline)) mli_prv_load_mac_vec4(
        int32_t * accu,
        const MLI_PTR(int8_t) __restrict in,
        const int16_t k) {
    v2q15_t v2k=fx_replic_v2q15((q15_t) k);
    int32_t four8bitvalues = *(MLI_PTR(int32_t)) in;
    *accu = _dmachbl((int32_t)v2k,four8bitvalues);
    *accu = _dmachbm((int32_t)v2k,four8bitvalues);
}


#ifdef __Xdsp_wide
static inline void __attribute__ ((always_inline)) mli_prv_load_mac_vec4(
        accum40_t * accu, 
        const MLI_PTR(int16_t) in, 
        const MLI_PTR(int16_t) k) {

    *accu = fx_a40_qmac_v4q15(*accu, mli_prv_load_4_samples(in), mli_prv_load_4_samples(k));
}
#endif

static inline void __attribute__ ((always_inline)) mli_prv_load_mac_vec4(
        int32_t * accu, 
        const MLI_PTR(int8_t) in, 
        const MLI_PTR(int8_t) k) {
    int32_t four8bitvalues = *(MLI_PTR(int32_t)) in;
#if defined __Xxy
    *accu = _dmachbl((int32_t) mli_prv_load_2_samples(k), four8bitvalues);
    k += 2;
    *accu = _dmachbm((int32_t) mli_prv_load_2_samples(k), four8bitvalues);
#else
    int32_t four8bit_weights = *(MLI_PTR(int32_t)) k;
    *accu = _dmachbl((int32_t) (v2q15_t) _vsext2bhl(four8bit_weights), four8bitvalues);
    *accu = _dmachbm((int32_t) (v2q15_t) _vsext2bhm(four8bit_weights), four8bitvalues);
#endif
}

static inline void __attribute__ ((always_inline)) mli_prv_load_mac_vec4(
        int32_t * accu, 
        const MLI_PTR(int16_t) in, 
        const MLI_PTR(int8_t) k) {
    int32_t four8bitvalues = *(MLI_PTR(int32_t)) k;
    *accu = _dmachbl((int32_t) mli_prv_load_2_samples(in), four8bitvalues);
    in += 2;
    *accu = _dmachbm((int32_t) mli_prv_load_2_samples(in), four8bitvalues);
}

static inline unsigned __attribute__ ((always_inline)) mli_prv_init_dsp_ctrl(unsigned ctrl_info) {
    unsigned t, old = _lr(DSP_CTRL);
    _sr(ctrl_info, DSP_CTRL);
    t = _lr(DSP_CTRL);

    /* Check if the new mode is set correctly
     * If the check fails, there is a disconnect between the
     * compiler and the debugger.  NSIM will only allow bits
     * to be set in DSP_CTRL that are enabled in the HW.
     * The code needs to be recompiled with a -Xdsp_ctrl option
     * that matches your hardware OR you need to invoke
     * the debugger with different opitons.
     */
    MLI_ASSERT((t & 31) == (ctrl_info & 31));

    return old;
}

static inline unsigned __attribute__ ((always_inline)) mli_prv_fx_init_dsp_ctrl() {
    unsigned mode = 0;

#if (defined(__Xdsp_version) && __Xdsp_version > 1) || defined(__Xdsp2)
    mode |= 0x10;
#endif

    // select the rounding mode
#if defined(__ROUNDING_MODE_TRUNC__)
#   error "Rounding 'trunc' is not supported"
#elif defined(__ROUNDING_MODE_UP__)
    mode |= 2;
#else
    mode |= 3;
#endif
    //select guarded accumulator mode
#if defined(__GUARD_ACCUM__)
    mode |= 4;
#endif
    // select post-accum shift mode
#if !defined(__POST_ACCUM_SHIFT__) && !defined(__Xdsp_postshift_mode)
    mode |= 8;
#endif

    return mli_prv_init_dsp_ctrl(mode);
}


#endif //_MLI_PRV_DSP_H_
