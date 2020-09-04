/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_COMMON_H_
#define _MLI_KRN_COMMON_H_

#include "mli_api.h"
#include "mli_check.h"
#include "mli_helpers_api.h"
#include "mli_krn_eltwise.h"
#include "mli_prv_dsp.h"
#include "mli_math.h"
#include "mli_types.h"

static inline accum40_t __attribute__ ((always_inline)) mli_prv_ashift_accu(accum40_t accu, const int shift_right)
{
    return fx_asr_a40(accu, shift_right);
}

static inline int32_t __attribute__ ((always_inline)) mli_prv_ashift_accu(int32_t accu, const int shift_right)
{
    accu = fx_asr_rnd_q31(accu, shift_right);
    _setacc(accu,1);
    return accu; 
}

namespace mli {

template <typename io_T, typename w_T>
static void __attribute__ ((always_inline)) rnn_dense_op_fx(
        const MLI_PTR (io_T) __restrict in,
        const MLI_PTR (io_T) __restrict state,
        const MLI_PTR (w_T) __restrict weights,
        const MLI_PTR (w_T) __restrict biases,
        MLI_CONV_OUT_PTR (io_T) __restrict out,
        const int inp_size,
        const int s_size,
        const int ch_out,
        const int bias_shift,
        const int in_to_state_fraq_dif,
        const int out_shift)
{

    if (_Rarely (inp_size < 8 && s_size < 8)) {
        for (int i = 0; i < ch_out; i++) {
            const MLI_PTR (io_T) __restrict in_ptr = in;
            const MLI_PTR (io_T) __restrict s_ptr = state;

            auto ip_out = mli_prv_init_accu_with_bias(in_ptr, *biases++, bias_shift);

            for (int j = 0; j < inp_size; j++) {
                mli_prv_load_mac(&ip_out, in_ptr++, weights++);
            }

            ip_out = mli_prv_ashift_accu(ip_out, in_to_state_fraq_dif);
        
            for (int k = 0; k < s_size; k++) {
                mli_prv_load_mac(&ip_out, s_ptr++, weights++);
            }
        
            mli_prv_clip_and_store_output(out++, &ip_out, out_shift);
        }
    } else if ((inp_size%4 == 0) && (s_size%4 == 0)) {
        for (int i = 0; i < ch_out; i++) {
            const MLI_PTR (io_T) __restrict in_ptr = in;
            const MLI_PTR (io_T) __restrict s_ptr = state;

            auto ip_out = mli_prv_init_accu_with_bias(in_ptr, *biases++, bias_shift);

LOOP_PIPELINE_ENABLE
LOOP_PIPELINE_ENABLE_BACKTRACKING
            for (int jj = 0; jj < (inp_size/4); jj++) {
                mli_prv_load_mac_vec4(&ip_out, in_ptr, weights);
                in_ptr += 4;
                weights += 4;
            }

            ip_out = mli_prv_ashift_accu(ip_out, in_to_state_fraq_dif);

LOOP_PIPELINE_ENABLE
LOOP_PIPELINE_ENABLE_BACKTRACKING
            for (int kk = 0; kk < (s_size/4); kk++) {
                mli_prv_load_mac_vec4(&ip_out, s_ptr, weights);
                s_ptr += 4;
                weights += 4;
            }
            s_ptr -= s_size;

            mli_prv_clip_and_store_output(out++, &ip_out, out_shift);
        }
    } else {
        for (int i = 0; i < ch_out; i++) {
            const MLI_PTR (io_T) __restrict in_ptr = in;
            const MLI_PTR (io_T) __restrict s_ptr = state;

            auto ip_out = mli_prv_init_accu_with_bias(in_ptr, *biases++, bias_shift);

            for (int j = 0; j < (inp_size&3); j++) {
                mli_prv_load_mac(&ip_out, in_ptr++, weights++);
            }

LOOP_PIPELINE_ENABLE
LOOP_PIPELINE_ENABLE_BACKTRACKING
            for (int jj = 0; jj < (inp_size/4); jj++) {
                mli_prv_load_mac_vec4(&ip_out, in_ptr, weights);
                in_ptr += 4;
                weights += 4;
            }

            ip_out = mli_prv_ashift_accu(ip_out, in_to_state_fraq_dif);

            for (int k = 0; k < (s_size&3); k++) {
                mli_prv_load_mac(&ip_out, s_ptr++, weights++);
            }

LOOP_PIPELINE_ENABLE
LOOP_PIPELINE_ENABLE_BACKTRACKING
            for (int kk = 0; kk < (s_size/4); kk++) {
                mli_prv_load_mac_vec4(&ip_out, s_ptr, weights);
                s_ptr += 4;
                weights += 4;
            }

            mli_prv_clip_and_store_output(out++, &ip_out, out_shift);
        }
    }
}
//==================================================================////==================================================================
template <typename io_T, typename w_T>
static void __attribute__ ((always_inline))  basic_rnn_cell_prepare_and_run_fx(
        const mli_tensor *in,
        const mli_tensor *prev_out,
        const mli_tensor *weights,
        const mli_tensor *bias,
        const mli_rnn_cell_cfg *cfg,
        mli_tensor *out) {
    // WARNING: In the row with other usual input restrictions, this procedure
    // MUST NOT be used in the BATCH mode with multiple weights matrix.
    // It is a shape mismatch between in + output vector and weights matrix in this case
    const int prev_elements = static_cast<int>(mli_prv_count_elem_num(prev_out));
    const int out_elements = static_cast<int>(mli_prv_count_elem_num(bias));
    const int batch_sz = static_cast<int>((cfg->mode == RNN_ONE_TO_ONE) ? 1 : in->shape[0]);
    const int in_elements = static_cast<int>((cfg->mode == RNN_ONE_TO_ONE) ? 
            mli_prv_count_elem_num(in) : mli_prv_count_elem_num_part(in, 1));

    const MLI_PTR (w_T) w_ptr = (const MLI_PTR (w_T)) weights->data.mem.void_p;
    const MLI_PTR (w_T) b_ptr = (const MLI_PTR (w_T)) bias->data.mem.void_p;
    const MLI_PTR (io_T) in_ptr = (const MLI_PTR (io_T)) in->data.mem.void_p;
    MLI_PTR (io_T) state_ptr = (MLI_PTR (io_T)) prev_out->data.mem.void_p;

    mli_tensor dense_out = {{ 0 }};
    dense_out.data.mem.void_p = (cfg->mode != RNN_BATCH_TO_LAST) ? out->data.mem.void_p : cfg->ir_tsr->data.mem.void_p;
    dense_out.data.capacity = (cfg->mode != RNN_BATCH_TO_LAST) ? out->data.capacity : cfg->ir_tsr->data.capacity;
    dense_out.shape[0] = out_elements;
    dense_out.mem_stride[0] = 0;
    dense_out.rank = 1;
    dense_out.el_type = in->el_type;
    // 1sign and 3 integer bits for typical rnn nonlinearity (TANH/SIGM) is enough
    dense_out.el_params.fx.frac_bits = (cfg->act == RNN_ACT_NONE) ? out->el_params.fx.frac_bits : 
            (sizeof(io_T) * 8) - 1 - 3;

    MLI_CONV_OUT_PTR (io_T) dense_out_ptr = (MLI_CONV_OUT_PTR (io_T)) dense_out.data.mem.void_p;

    mli_tensor rnn_out = {
        .data.mem.void_p = out->data.mem.void_p,
        .data.capacity = out_elements * sizeof(io_T),
        .shape = {static_cast<unsigned>(out_elements)},
        .rank = 1,
        .el_type = in->el_type,
        .el_params = (cfg->act == RNN_ACT_NONE) ? out->el_params : prev_out->el_params};

    int b_half = batch_sz&1;
    if(b_half == 0 && cfg->act == RNN_ACT_NONE && cfg->mode == RNN_BATCH_TO_LAST) {
        rnn_out.data.mem.void_p = cfg->ir_tsr->data.mem.void_p;
    }

    MLI_CONV_OUT_PTR (io_T) rnn_out_ptr = (MLI_CONV_OUT_PTR (io_T)) rnn_out.data.mem.void_p;

    // Define shift values
    const int bias_shift = mli_prv_calc_shift (in, weights, bias);
    const int in_to_state_dif = in->el_params.fx.frac_bits - prev_out->el_params.fx.frac_bits;
    const int out_shift = mli_prv_calc_shift (prev_out, weights, &dense_out);

    // Perform sequential run (or only one run for RNN_ONE_VEC)
    for (int batch = 0; batch < batch_sz; batch++) {
        if(cfg->mode == RNN_BATCH_TO_LAST && cfg->act == RNN_ACT_NONE) {
            // Applying Dense
            //=======================================
            rnn_dense_op_fx(in_ptr, state_ptr, w_ptr, b_ptr, rnn_out_ptr, in_elements,
                    prev_elements, out_elements, bias_shift, in_to_state_dif, out_shift);
            // Update pointers for next batch
            //=======================================
            state_ptr = (MLI_PTR (io_T)) rnn_out.data.mem.void_p;
            b_half ^= 1;
            rnn_out.data.mem.void_p = b_half ? out->data.mem.void_p : cfg->ir_tsr->data.mem.void_p;
            rnn_out_ptr = (MLI_CONV_OUT_PTR (io_T)) rnn_out.data.mem.void_p;
        }
        else {
            // Applying Dense
            //=======================================
            rnn_dense_op_fx(in_ptr, state_ptr, w_ptr, b_ptr, dense_out_ptr, in_elements,
                    prev_elements, out_elements, bias_shift, in_to_state_dif, out_shift);
            // Applying Non-Linearity
            //=======================================
            if (cfg->act == RNN_ACT_TANH) {
                if (sizeof(io_T)==sizeof(int8_t)) 
                    mli_krn_tanh_fx8 (&dense_out, &rnn_out);
                else 
                    mli_krn_tanh_fx16 (&dense_out, &rnn_out);
            }
            else if (cfg->act == RNN_ACT_SIGM) {
                if (sizeof(io_T)==sizeof(int8_t)) 
                    mli_krn_sigm_fx8 (&dense_out, &rnn_out);
                else 
                    mli_krn_sigm_fx16 (&dense_out, &rnn_out);
            }
            // Update pointers for next batch
            //=======================================
            state_ptr = (MLI_PTR (io_T)) rnn_out.data.mem.void_p;
            if (cfg->mode == RNN_BATCH_TO_BATCH) {
                rnn_out.data.mem.void_p = static_cast < io_T * >(rnn_out.data.mem.void_p) + out_elements;
                dense_out.data.mem.void_p = static_cast < io_T * >(dense_out.data.mem.void_p) + out_elements;
                dense_out_ptr += out_elements;
            }
        }
        in_ptr += in_elements;
    }

    // Fill output tensor params
    out->el_type = rnn_out.el_type;
    out->el_params.fx.frac_bits = rnn_out.el_params.fx.frac_bits;
    if (cfg->mode == RNN_ONE_TO_ONE || cfg->mode == RNN_BATCH_TO_LAST) {
        out->rank = bias->rank;
        for (int k = 0; k < bias->rank; k++)
            out->shape[k] = bias->shape[k];
    } else {
        out->rank = 2;
        out->shape[0] = batch_sz;
        out->shape[1] = out_elements;
    }
}

//==================================================================
//
//==================================================================
template <typename io_T, typename w_T>
static void __attribute__ ((always_inline)) lstm_cell_prepare_and_run_fx(
        const mli_tensor *in,
        const mli_tensor *prev_out,
        const mli_tensor *weights,
        const mli_tensor *bias,
        const mli_rnn_cell_cfg *cfg,
        mli_tensor *cell,
        mli_tensor *out) {
    const int lstm_out_elements = static_cast<int>(mli_prv_count_elem_num(prev_out));
    const int dense_out_elements = static_cast<int>(mli_prv_count_elem_num(bias));
    const int batch_sz = static_cast<int>((cfg->mode == RNN_ONE_TO_ONE) ? 1 : in->shape[0]);
    const int in_elements = static_cast<int>((cfg->mode == RNN_ONE_TO_ONE) ? mli_prv_count_elem_num (in) : mli_prv_count_elem_num_part(in, 1));

    const MLI_PTR (w_T) w_ptr = (const MLI_PTR (w_T)) weights->data.mem.void_p;
    const MLI_PTR (w_T) b_ptr = (const MLI_PTR (w_T)) bias->data.mem.void_p;
    const MLI_PTR (io_T) in_ptr = (const MLI_PTR (io_T)) in->data.mem.void_p;
    const MLI_PTR (io_T) prev_ptr = (const MLI_PTR (io_T)) prev_out->data.mem.void_p;
    MLI_CONV_OUT_PTR (io_T) dense_out_ptr = (MLI_CONV_OUT_PTR (io_T)) cfg->ir_tsr->data.mem.void_p;

    // Fill intermediate tensor of dense output
    mli_tensor *ir_tensor = cfg->ir_tsr;
    ir_tensor->rank = bias->rank;
    ir_tensor->shape[0] = bias->shape[0];
    ir_tensor->shape[1] = bias->shape[1];
    ir_tensor->mem_stride[0] = 0;
    ir_tensor->mem_stride[1] = 0;
    ir_tensor->el_type = in->el_type;
    // 1sign and 3 integer bits for TANH/SIGM input is enough
    ir_tensor->el_params.fx.frac_bits = (sizeof(io_T) * 8) - 1 - 3;

    // Define shift values
    const int dense_bias_shift = mli_prv_calc_shift(in, weights, bias);
    const int in_to_state_dif = in->el_params.fx.frac_bits - prev_out->el_params.fx.frac_bits;
    const int dense_out_shift = mli_prv_calc_shift(prev_out, weights, ir_tensor);

    // Paricular subtensors of intermediate tensor (mli_tensor.mem_stride[] should be zero and cannot be left uninitialized)
    mli_tensor in_gate = {{ 0 }}, forget_gate = {{ 0 }}, out_gate = {{ 0 }}; // Various gates to controll info flow
    mli_tensor g_tsr = {{ 0 }}; // Information tensors

    // Init subtensors
    mli_point_to_subtsr_cfg iterator = {.start_coord = {0}, .coord_num=1, .first_out_dim_size=1};
    mli_hlp_point_to_subtensor(ir_tensor, &iterator, &in_gate); iterator.start_coord[0]++;
    mli_hlp_point_to_subtensor(ir_tensor, &iterator, &g_tsr); iterator.start_coord[0]++;
    mli_hlp_point_to_subtensor(ir_tensor, &iterator, &forget_gate); iterator.start_coord[0]++;
    mli_hlp_point_to_subtensor(ir_tensor, &iterator, &out_gate);

    // lstm output for one step
    mli_tensor rnn_out = {
        .data.mem.void_p = out->data.mem.void_p,
        .data.capacity = out->data.capacity,
        .shape = {static_cast<unsigned>(lstm_out_elements)},
        .rank = 1,
        .el_type = in->el_type};
    if (cfg->act == RNN_ACT_NONE)   // fx Parameters in case of No activation
        rnn_out.el_params.fx.frac_bits = prev_out->el_params.fx.frac_bits;

    io_T *f_g = (io_T *) forget_gate.data.mem.void_p;
    io_T *i_g = (io_T *) in_gate.data.mem.void_p;
    io_T *g = (io_T *) g_tsr.data.mem.void_p;
    io_T *c = (io_T *) cell->data.mem.void_p;

    // For elementwise (assuming gates have 7 (fx8) or 15 (fx16) fractional bits)
    int eltwise_ir_shift = ((sizeof(io_T) * 8) - 1) - (int) cell->el_params.fx.frac_bits;
    int eltwise_o_shift = (sizeof(io_T) * 8) - 1;

    for (int batch = 0; batch < batch_sz; batch++) {

        // Step 1: Applying Dense
        //=======================================
        rnn_dense_op_fx(in_ptr, prev_ptr, w_ptr, b_ptr, dense_out_ptr,
                in_elements, lstm_out_elements, dense_out_elements, dense_bias_shift, in_to_state_dif, dense_out_shift);

        // Step2: Applying non-linearity
        //=======================================
        in_gate.el_params.fx.frac_bits = out_gate.el_params.fx.frac_bits = ir_tensor->el_params.fx.frac_bits;
        g_tsr.el_params.fx.frac_bits = forget_gate.el_params.fx.frac_bits = ir_tensor->el_params.fx.frac_bits;

        if (sizeof(io_T)==sizeof(int8_t)) {
            mli_krn_sigm_fx8(&in_gate, &in_gate);
            mli_krn_tanh_fx8(&g_tsr, &g_tsr);
            mli_krn_sigm_fx8(&forget_gate, &forget_gate);
            mli_krn_sigm_fx8(&out_gate, &out_gate);
        }
        else {
            mli_krn_sigm_fx16(&in_gate, &in_gate);
            mli_krn_tanh_fx16(&g_tsr, &g_tsr);
            mli_krn_sigm_fx16(&forget_gate, &forget_gate);
            mli_krn_sigm_fx16(&out_gate, &out_gate);
        }

        // Step3: Pointwise operations
        //=======================================
//      LOGICALY (NOT BITWISE) EQUAL TO THE NEXT:
//      //Forget some old info
//      eltwise_prepare_and_run_fx<io_T, ELTWISE_MUL>(&forget_gate, cell, cell);
//      // Decide what new info we want to add to the mem cell
//      eltwise_prepare_and_run_fx<io_T, ELTWISE_MUL>(&in_gate, &g_tsr, &new_cell_info);
//      //Adding new info into cell
//      eltwise_prepare_and_run_fx<io_T, ELTWISE_ADD>(cell, &new_cell_info, cell);
        for (int idx = 0; idx < lstm_out_elements; idx++) {
            mli_acc32_t new_val = mli_math_mul_fx<io_T, mli_acc32_t>(i_g[idx], g[idx]);
            new_val = mli_math_acc_ashift_fx(new_val, eltwise_ir_shift);
            new_val = mli_math_mac_fx(new_val, f_g[idx], c[idx]);
            c[idx] = mli_math_acc_cast_fx<io_T, mli_acc32_t>(new_val, eltwise_o_shift);
        }

        // Step4: Calculate output: Activation + pointwise operation
        //===========================================================
        if (cfg->act == RNN_ACT_NONE) {
            eltwise_prepare_and_run_fx<io_T, ELTWISE_MUL>(cell, &out_gate, &rnn_out);
        } else {             // Non - Linear activation
            if (sizeof(io_T)==sizeof(int8_t)) {
                if (cfg->act == RNN_ACT_TANH)
                    mli_krn_tanh_fx8(cell, &rnn_out);
                else            // RNN_ACT_SIGM:
                    mli_krn_sigm_fx8(cell, &rnn_out);
            } else {
                if (cfg->act == RNN_ACT_TANH)
                    mli_krn_tanh_fx16(cell, &rnn_out);
                else            // RNN_ACT_SIGM:
                    mli_krn_sigm_fx16(cell, &rnn_out);
            }

            eltwise_prepare_and_run_fx<io_T, ELTWISE_MUL>(&rnn_out, &out_gate, &rnn_out);
        }

        // Step 5: Update pointers and tensors for next batch
        //=======================================
        in_ptr += in_elements;
        prev_ptr = (MLI_PTR (io_T)) rnn_out.data.mem.void_p;
        if (cfg->mode == RNN_BATCH_TO_BATCH) {
            rnn_out.data.mem.void_p = static_cast<io_T*>(rnn_out.data.mem.void_p) + lstm_out_elements;
            rnn_out.data.capacity -= lstm_out_elements;
        }
    }

    // Fill output tensor params
    out->el_type = rnn_out.el_type;
    out->el_params.fx.frac_bits = rnn_out.el_params.fx.frac_bits;
    if (cfg->mode == RNN_ONE_TO_ONE || cfg->mode == RNN_BATCH_TO_LAST) {
        out->rank = 1;
        out->shape[0] = lstm_out_elements;
    } else {
        out->rank = 2;
        out->shape[0] = batch_sz;
        out->shape[1] = lstm_out_elements;
    }
}

}

#endif // _MLI_KRN_COMMON_H_
