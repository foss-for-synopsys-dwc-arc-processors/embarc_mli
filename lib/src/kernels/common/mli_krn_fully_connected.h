/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_FULLY_CONNECTED_H_
#define _MLI_KRN_FULLY_CONNECTED_H_

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"
#include "math.h"
#include "mli_prv_quant.h"

/******************************************************************************
 *
 * Version & platform description
 * Targets:
 *
 ******************************************************************************/

template <typename io_T, typename w_T>
static void __attribute__((always_inline)) full_connection(
        const MLI_PTR(io_T) __restrict in_ptr,
        const MLI_PTR(w_T) __restrict w_ptr,
        const MLI_PTR(w_T) bias_p,
        MLI_CONV_OUT_PTR(io_T) __restrict o_ptr,
        const int ch_out,
        const int inp_size,
        const int w_ch_out_mem_stride,
        const int bias_shift,
        const int out_shift) {
    if (_Rarely(inp_size < 8)) {
        for (int i = 0; i < ch_out; i++) {
            auto ip_out = mli_prv_init_accu_with_bias(in_ptr, *bias_p++, bias_shift);
            for (int j = 0; j < inp_size; j++) {
                mli_prv_load_mac(&ip_out, in_ptr++, w_ptr++);
            }
            in_ptr -= inp_size;
            w_ptr += w_ch_out_mem_stride - inp_size;

            mli_prv_clip_and_store_output(o_ptr++, &ip_out, out_shift);
        }
    } else {
        if ((inp_size & 0x3) == 0) {
            const MLI_PTR(io_T) start_in_ptr = in_ptr;
            for (int i = 0; i < ch_out; i++) {
                auto ip_out = mli_prv_init_accu_with_bias(in_ptr, *bias_p++, bias_shift);

LOOP_PIPELINE_ENABLE
LOOP_PIPELINE_ENABLE_BACKTRACKING
                for (int j = 0; j < (inp_size / 4); j++) {
                    mli_prv_load_mac_vec4(&ip_out, in_ptr, w_ptr);
                    in_ptr += 4;
                    w_ptr += 4;
                }
                in_ptr -= inp_size;
                w_ptr += w_ch_out_mem_stride - inp_size;
                MLI_EXTRA_ASSERT(start_in_ptr == in_ptr);

                mli_prv_clip_and_store_output(o_ptr++, &ip_out, out_shift);
            }
        } else {
            const MLI_PTR(io_T) start_in_ptr = in_ptr;
            for (int i = 0; i < ch_out; i++) {
                auto ip_out = mli_prv_init_accu_with_bias(in_ptr, *bias_p++, bias_shift);

                int odd_rest_of_inp_size = (inp_size & 0x3);
                for (int k = 0; k < odd_rest_of_inp_size; k++) {
                    mli_prv_load_mac(&ip_out, in_ptr++, w_ptr++);
                }

                int even_inp_size = inp_size - odd_rest_of_inp_size;
LOOP_PIPELINE_ENABLE
LOOP_PIPELINE_ENABLE_BACKTRACKING
                for (int j = 0; j < (even_inp_size / 4); j++) {
                    mli_prv_load_mac_vec4(&ip_out, in_ptr, w_ptr);
                    in_ptr += 4;
                    w_ptr += 4;
                }
                in_ptr -= inp_size;
                w_ptr += w_ch_out_mem_stride - inp_size;
                MLI_EXTRA_ASSERT(start_in_ptr == in_ptr);

                mli_prv_clip_and_store_output(o_ptr++, &ip_out, out_shift);
            }
        }
    }
}

template <typename io_T, typename w_T>
static void __attribute__((always_inline)) fully_connected_prepare_and_run_fx(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        mli_tensor* out) {
    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(io_T) in_ptr = (MLI_PTR(io_T))(in->data.mem.void_p);
    const MLI_PTR(w_T) w_ptr = (MLI_PTR(w_T))(weights->data.mem.void_p);
    const MLI_PTR(w_T) b_ptr = (MLI_PTR(w_T))(bias->data.mem.void_p);
    MLI_CONV_OUT_PTR(io_T) out_ptr = (MLI_CONV_OUT_PTR(io_T))(out->data.mem.void_p);

    int ch_out = weights->shape[0];
    int in_sz = mli_prv_count_elem_num(in);
    int w_ch_out_mem_stride_from_tensor = weights->mem_stride[0];
    int w_ch_out_mem_stride = (w_ch_out_mem_stride_from_tensor != 0) ?
        w_ch_out_mem_stride_from_tensor : in_sz;

    // Define shift values
    const int bias_shift = mli_prv_calc_shift(in, weights, bias);
    const int out_shift = mli_prv_calc_shift(in, weights, out);

    // Run basic calculation
    full_connection<io_T, w_T>(in_ptr, w_ptr, b_ptr, out_ptr, ch_out, in_sz, w_ch_out_mem_stride,
            bias_shift, out_shift);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->shape[0] = ch_out;
    out->rank = 1;
}

template <typename io_T, typename w_T, typename b_T, typename acc_T>
static void __attribute__((always_inline)) ip_op(
        const MLI_PTR(io_T) __restrict in,
        const MLI_PTR(w_T)  __restrict weights,
        const MLI_PTR(b_T)  __restrict biases,
              MLI_CONV_OUT_PTR(io_T) __restrict out,

        const int in_elements,
        const int out_elements,
        const int w_ch_out_mem_stride,
        const int32_t bias_mul,
        const int bias_shift,
        const int32_t out_mul,
        const int out_shift,
        const int16_t input_offset,
        const int16_t output_offset) {
    const int left_shift = out_shift > 0 ? 0 : -out_shift;
    const int right_shift = out_shift > 0 ? out_shift : 0;
    // Matrix-Vector multiplication
    //==============================
    if (_Rarely(in_elements < 8)) {
        for (int o_idx = 0; o_idx < out_elements; o_idx++) {

            acc_T accu = mli_math_init_accu<b_T, acc_T, true>(biases[o_idx], bias_mul, bias_shift);

            for (int i_idx = 0; i_idx < in_elements; i_idx++){
                mli_prv_load_mac(&accu, in++, weights);
                mli_prv_load_mac(&accu, weights, (const int16_t)-input_offset);
                weights++;
            }
            in -= in_elements;
            const mli_acc32_t accu_result = mli_math_cast_fx<mli_acc32_t, acc_T >(accu, -left_shift);
            const accum72_t accu_scaled = mli_math_mul_fx<mli_acc32_t, accum72_t>(accu_result, out_mul);
            // adding the output offset needs to happen after the output mul and output shift
            // but before the cast to the output container size.
            // because the cast and shift are combined in one function, the output offset is
            // added before, and multiplied with 1<< right_shift to compensate.
            const int16_t out_no_offset = mli_math_cast_fx<accum72_t, int16_t>(accu_scaled, right_shift);
            out[o_idx] = mli_math_cast_fx<int16_t, io_T>(mli_math_add_fx(out_no_offset, output_offset), 0);
        }
    } else {
        if ((in_elements & 0x3) == 0) {
            for (int o_idx = 0; o_idx < out_elements; o_idx++) {

                acc_T accu = mli_math_init_accu<b_T, acc_T, true>(biases[o_idx], bias_mul, bias_shift);
LOOP_PIPELINE_ENABLE
LOOP_PIPELINE_ENABLE_BACKTRACKING
                for (int i_idx = 0; i_idx < in_elements/4; i_idx++){
                    mli_prv_load_mac_vec4(&accu, in, weights);
                    in += 4;
                    mli_prv_load_mac_vec4(&accu, weights, (const int16_t) -input_offset);
                    weights += 4;
                }
                in -= in_elements;
                weights += w_ch_out_mem_stride - in_elements;

                const mli_acc32_t accu_result = mli_math_cast_fx<mli_acc32_t, acc_T >(accu, -left_shift);
                const accum72_t accu_scaled = mli_math_mul_fx<mli_acc32_t, accum72_t>(accu_result, out_mul);
                // adding the output offset needs to happen after the output mul and output shift
                // but before the cast to the output container size.
                // because the cast and shift are combined in one function, the output offset is
                // added before, and multiplied with 1<< right_shift to compensate.
                const int16_t out_no_offset = mli_math_cast_fx<accum72_t, int16_t>(accu_scaled, right_shift);
                out[o_idx] = mli_math_cast_fx<int16_t, io_T>(mli_math_add_fx(out_no_offset, output_offset), 0);
            }
        } else {
            for (int o_idx = 0; o_idx < out_elements; o_idx++) {

            acc_T accu = mli_math_init_accu<b_T, acc_T, true>(biases[o_idx], bias_mul, bias_shift);

            int odd_rest_of_inp_size = (in_elements & 0x3);
            
            for (int k = 0; k < odd_rest_of_inp_size; k++) {
                mli_prv_load_mac(&accu, in++, weights);
                mli_prv_load_mac(&accu, weights,  (const int16_t) -input_offset);
                weights++;
            }
            
            int even_inp_size = in_elements - odd_rest_of_inp_size;
LOOP_PIPELINE_ENABLE
LOOP_PIPELINE_ENABLE_BACKTRACKING
            for (int i_idx = 0; i_idx < even_inp_size/4; i_idx++){
                mli_prv_load_mac_vec4(&accu, in, weights);
                in += 4;
                mli_prv_load_mac_vec4(&accu, weights, (const int16_t)-input_offset );
                weights += 4;
            }
            in -= in_elements;
            weights += w_ch_out_mem_stride - in_elements;
            const mli_acc32_t accu_result = mli_math_cast_fx<mli_acc32_t, acc_T >(accu, -left_shift);
            const accum72_t accu_scaled = mli_math_mul_fx<mli_acc32_t, accum72_t>(accu_result, out_mul);
            // adding the output offset needs to happen after the output mul and output shift
            // but before the cast to the output container size.
            // because the cast and shift are combined in one function, the output offset is
            // added before, and multiplied with 1<< right_shift to compensate.
            const int16_t out_no_offset = mli_math_cast_fx<accum72_t, int16_t>(accu_scaled, right_shift);
            out[o_idx] = mli_math_cast_fx<int16_t, io_T>(mli_math_add_fx(out_no_offset, output_offset), 0);
            }
        }
    }
}

template <typename io_T, typename w_T, typename b_T>
static void __attribute__((always_inline)) fully_connected_prepare_and_run(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        mli_tensor* out) {
    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(io_T) in_ptr = (MLI_PTR(io_T))(in->data.mem.void_p);
    const MLI_PTR(w_T)  w_ptr  = (MLI_PTR(w_T)) (weights->data.mem.void_p);
    const MLI_PTR(b_T)  b_ptr  = (MLI_PTR(b_T)) (bias->data.mem.void_p);
    MLI_CONV_OUT_PTR(io_T) out_ptr = (MLI_CONV_OUT_PTR(io_T)) (out->data.mem.void_p);

    int ch_out = bias->shape[0];
    int in_sz = weights->shape[1];
    int w_ch_out_mem_stride_from_tensor = weights->mem_stride[0];
    int w_ch_out_mem_stride = (w_ch_out_mem_stride_from_tensor != 0) ?
        w_ch_out_mem_stride_from_tensor : in_sz;

    // Define shift values
    int bias_shift = 0;
    int out_shift = 0;

    int32_t out_mul = mli::krn::mli_prv_calc_out_mul(in, weights, out, &out_shift);
    int32_t bias_mul = mli::krn::mli_prv_calc_out_mul(in, weights, bias, &bias_shift);
    int16_t input_offset = mli_hlp_tensor_zero_offset(in, 0);
    int16_t output_offset = mli_hlp_tensor_zero_offset(out, 0);
    MLI_ASSERT(mli_hlp_tensor_zero_offset(weights, 0) == 0);

    // Run basic calculation
    ip_op<io_T, w_T, b_T, mli_acc32_t>(in_ptr, w_ptr, b_ptr, out_ptr, in_sz, ch_out, w_ch_out_mem_stride,
            bias_mul, bias_shift, out_mul, out_shift, input_offset, output_offset);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->shape[0] = ch_out;
    out->rank = 1;
}

#endif  //_MLI_KRN_FULLY_CONNECTED_H_
