/*
* Copyright 2019, Synopsys, Inc.
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
        const int bias_shift,
        const int out_shift) {
    if (_Rarely(inp_size < 8)) {
        for (int i = 0; i < ch_out; i++) {
            auto ip_out = mli_prv_init_accu_with_bias(in_ptr, *bias_p++, bias_shift);
            for (int j = 0; j < inp_size; j++) {
                mli_prv_load_mac(&ip_out, in_ptr++, w_ptr++);
            }
            in_ptr -= inp_size;

            mli_prv_clip_and_store_output(o_ptr++, &ip_out, out_shift);
        }
    } else {
        if ((inp_size & 0x3) == 0) {
            const MLI_PTR(io_T) start_in_ptr = in_ptr;
            for (int i = 0; i < ch_out; i++) {
                auto ip_out = mli_prv_init_accu_with_bias(in_ptr, *bias_p++, bias_shift);

LOOP_PIPELINE_ENABLE
                for (int j = 0; j < (inp_size / 4); j++) {
                    mli_prv_load_mac_vec4(&ip_out, in_ptr, w_ptr);
                    in_ptr += 4;
                    w_ptr += 4;
                }
                in_ptr -= inp_size;
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
                for (int j = 0; j < (even_inp_size / 4); j++) {
                    mli_prv_load_mac_vec4(&ip_out, in_ptr, w_ptr);
                    in_ptr += 4;
                    w_ptr += 4;
                }
                in_ptr -= inp_size;
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

    const MLI_PTR(io_T) in_ptr = (MLI_PTR(io_T))(in->data);
    const MLI_PTR(w_T) w_ptr = (MLI_PTR(w_T))(weights->data);
    const MLI_PTR(w_T) b_ptr = (MLI_PTR(w_T))(bias->data);
    MLI_CONV_OUT_PTR(io_T) out_ptr = (MLI_CONV_OUT_PTR(io_T))(out->data);

    int ch_out = weights->shape[0];
    int in_sz = mli_prv_count_elem_num(in);

    // Define shift values
    const int bias_shift = mli_prv_calc_shift(in, weights, bias);
    const int out_shift = mli_prv_calc_shift(in, weights, out);

    // Run basic calculation
    full_connection<io_T, w_T>(in_ptr, w_ptr, b_ptr, out_ptr, ch_out, in_sz, bias_shift, out_shift);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->shape[0] = ch_out;
    out->rank = 1;
}

template <typename io_T, typename w_T, typename b_T, typename acc_T>
static inline void ip_op(
        const io_T* __restrict in,
        const w_T*  __restrict weights,
        const b_T*  __restrict biases,
              io_T* __restrict out,

        const int in_elements,
        const int out_elements,
        const int32_t bias_mul,
        const int bias_shift,
        const int32_t out_mul,
        const int out_shift,
        const io_T input_offset,
        const io_T output_offset) {
    // Matrix-Vector multiplication
    //==============================
    for (int o_idx = 0; o_idx < out_elements; o_idx++) {
        int w_idx = o_idx * in_elements;

        acc_T accu = mli_math_init_accu<b_T, acc_T, true>(biases[o_idx], bias_mul, bias_shift);

        for (int i_idx = 0; i_idx < in_elements; i_idx++, w_idx++){
            accu = mli_math_mac_fx(accu, in[i_idx], weights[w_idx]);
            accu = mli_math_mac_fx(accu, (io_T)-input_offset, weights[w_idx]);
        }

        accu = mli_math_scale_mul<acc_T, true>(accu, out_mul);

        // adding the output offset needs to happen after the output mul and output shift
        // but before the cast to the output container size.
        // because the cast and shift are combined in one function, the output offset is
        // added before, and multiplied with 1<< out_shift to compensate.
        accu = mli_math_mac_fx(accu, (int16_t)(1<<out_shift), (io_T)output_offset);
        out[o_idx] = mli_math_acc_cast_fx<io_T, acc_T> (accu, out_shift);
    }
}

template <typename io_T, typename w_T, typename b_T>
static void fully_connected_prepare_and_run(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        mli_tensor* out) {
    fx_init_dsp_ctrl();

    const io_T * in_ptr = static_cast<io_T *>(in->data);
    const w_T * w_ptr = static_cast<w_T *>(weights->data);
    const b_T * b_ptr = static_cast<b_T *>(bias->data);
    io_T * out_ptr = static_cast<io_T *>(out->data);

    int ch_out = bias->shape[0];
    int in_sz = mli_prv_count_elem_num(in);

    // Define shift values
    int bias_shift = mli_prv_calc_shift(in, weights, bias);
    int out_shift = mli_prv_calc_shift(in, weights, out);

    int32_t out_mul = mli_prv_calc_out_mul(in, weights, out, &out_shift);;
    int32_t bias_mul = mli_prv_calc_bias_mul(in, weights, bias);
    io_T input_offset = mli_hlp_tensor_zero_offset(in, 0);
    io_T output_offset = mli_hlp_tensor_zero_offset(out, 0);
    MLI_ASSERT(mli_hlp_tensor_zero_offset(weights, 0) == 0);

    // Run basic calculation
    ip_op<io_T, w_T, b_T, mli_acc32_t>(in_ptr, w_ptr, b_ptr, out_ptr, in_sz, ch_out, bias_mul, bias_shift, out_mul, out_shift, input_offset, output_offset);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->shape[0] = ch_out;
    out->rank = 1;
}

#endif  //_MLI_KRN_FULLY_CONNECTED_H_
