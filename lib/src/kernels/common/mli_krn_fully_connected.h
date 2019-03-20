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
        MLI_PTR(io_T) __restrict o_ptr,
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
    MLI_PTR(io_T) out_ptr = (MLI_PTR(io_T))(out->data);

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

#endif  //_MLI_KRN_FULLY_CONNECTED_H_
