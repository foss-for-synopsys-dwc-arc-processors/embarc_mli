/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_GRU_CELL_H_
#define _MLI_KRN_GRU_CELL_H_

#include <type_traits>

#include "mli_api.h"
#include "mli_math.h"
#include "mli_private_types.h"
#include "mli_prv_quant.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"
#include "mli_krn_eltwise.h"

#include "mli_krn_rnn_dense_op.h"

namespace mli {
namespace krn {

#pragma MLI_CODE_SECTION_START(".mli_lib")

//========================================================================================
// Common routine for pre-calculation of various basic rnn cell parameters and running it.
//========================================================================================

static inline void inc_scales_for_new_gate(mli_element_params* params, int prev_gates) {
	params->sa.scale.mem.pi16 += prev_gates;
    params->sa.scale_frac_bits.mem.pi8 += prev_gates;
}

static inline void dec_scales_for_new_gate(mli_element_params* params, int prev_gates) {
    params->sa.scale.mem.pi16 -= prev_gates;
    params->sa.scale_frac_bits.mem.pi8 -= prev_gates;
}

template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void gru_cell_prepare_and_run(
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out, 
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut, 
        const mli_rnn_cell_cfg * cfg, 
        mli_tensor *out) {

    constexpr bool asym = std::is_same<quant_T, s8asym_quant_specific_params>::value;

    const int8_t num_inputs = 2;
    const int8_t num_gates = 2;
    const int batch_sz = in->shape[0];
    const int inputs_elements[] = {(int)mli_prv_count_elem_num_part(in, 1), (int)mli_prv_count_elem_num(prev_out)};
    const int gru_out_elements = static_cast<int>(mli_prv_count_elem_num(prev_out));

    const mli_tensor * weights[] = {weights_in, weights_out};
    const MLI_PTR (io_T) inputs_ptr[] = {mli_prv_tensor_data_ptr<MLI_PTR (io_T)>(in), 
                                         mli_prv_tensor_data_ptr<MLI_PTR (io_T)>(prev_out)};

    if (cfg->direction == RNN_DIR_BACKWARD) 
        inputs_ptr[0] += (batch_sz - 1) * inputs_elements[0];

    mli_element_params one_el_params;
    mli_element_params ir_asym_params;
    if (asym) {
        one_el_params.sa.dim = ir_asym_params.sa.dim = -1;
        one_el_params.sa.scale.mem.i16 = ir_asym_params.sa.scale.mem.i16 = 1;
        one_el_params.sa.zero_point.mem.i16 = ir_asym_params.sa.zero_point.mem.i16 = 0;
        one_el_params.sa.scale_frac_bits.mem.i8 = 0;
        ir_asym_params.sa.scale_frac_bits.mem.i8 = 6;
        one_el_params.sa.scale.capacity = ir_asym_params.sa.scale.capacity = 0;
        one_el_params.sa.zero_point.capacity = ir_asym_params.sa.zero_point.capacity = 0;
        one_el_params.sa.scale_frac_bits.capacity = ir_asym_params.sa.scale_frac_bits.capacity = 0;
    } else {
        one_el_params.fx.frac_bits = 0;
        // 1sign and 3 integer bits for TANH/SIGM input is enough
        ir_asym_params.fx.frac_bits = (sizeof(io_T) * 8) - 1 - 3;
    }

    mli_tensor one = {{ 0 }};
    int16_t one_data[] = {1};
    one.data.capacity = 1;
    one.data.mem.pi16 = one_data;
    one.mem_stride[0] = 1;
    one.shape[0] = 1;
    one.rank = 1;
    one.el_type = in->el_type;
    one.el_params = one_el_params;

    mli_tensor ir_tensor = {{ 0 }};
    ir_tensor.data = cfg->scratch_data;
    ir_tensor.shape[0] = bias->shape[0];
    ir_tensor.shape[1] = bias->shape[1];
    ir_tensor.mem_stride[0] = ir_tensor.shape[1];
    ir_tensor.mem_stride[1] = 1;
    ir_tensor.rank = bias->rank; 
    ir_tensor.el_type = in->el_type;
    ir_tensor.el_params = ir_asym_params;

    quant_T in_to_out_params[2];
    define_quant_params(in, weights_in, bias, &ir_tensor, &in_to_out_params[0]);
    define_quant_params(prev_out, weights_out, bias, &ir_tensor, &in_to_out_params[1]);

    const int w_ch_out_mem_strides[] = {(int)weights_in->mem_stride[KRNL_RNN_W_IN_ELEMS_DIM],
                                                    (int)weights_out->mem_stride[KRNL_RNN_W_IN_ELEMS_DIM]};

    const int w_gate_mem_strides[] = {(int)weights_in->mem_stride[0],
                                                  (int)weights_out->mem_stride[0]};

    // Paricular subtensors of intermediate tensor
    mli_tensor reset_gate, update_gate, new_gate; // Various gates to control info flow
    mli_tensor w_in_new_g, w_out_new_g, b_new_g;

    // Init subtensors
    mli_sub_tensor_cfg iterator = {/*.offset =*/ {0}, /*.size = */{1, ir_tensor.shape[1]}, /*.sub_tensor_rank =*/2};
    mli_sub_tensor_cfg weight_iterator = {/*.offset =*/ {2,0}, /*.size = */{1, bias->shape[1]}, /*.sub_tensor_rank =*/2};
    mli_hlp_create_subtensor(&ir_tensor, &iterator, &update_gate); iterator.offset[0]++;
    mli_hlp_create_subtensor(&ir_tensor, &iterator, &reset_gate); iterator.offset[0]++;
    mli_hlp_create_subtensor(&ir_tensor, &iterator, &new_gate); iterator.offset[0]++;
    mli_hlp_create_subtensor(bias, &weight_iterator, &b_new_g);

    w_in_new_g.data = weights_in->data; 
    w_in_new_g.rank = 2;
    w_in_new_g.shape[0] = weights_in->shape[1];
    w_in_new_g.shape[1] = weights_in->shape[2];
    w_in_new_g.mem_stride[0] = weights_in->mem_stride[1];
    w_in_new_g.mem_stride[1] = weights_in->mem_stride[2];
    w_in_new_g.el_params = weights_in->el_params;
    w_in_new_g.el_type = weights_in->el_type;
    mli_prv_tensor_inc_data_ptr<w_T*>(&w_in_new_g, num_gates * w_gate_mem_strides[0]);

    w_out_new_g.data = weights_out->data; 
    w_out_new_g.rank = 2;
    w_out_new_g.shape[0] = weights_out->shape[1];
    w_out_new_g.shape[1] = weights_out->shape[2];
    w_out_new_g.mem_stride[0] = weights_out->mem_stride[1];
    w_out_new_g.mem_stride[1] = weights_out->mem_stride[2];
    w_out_new_g.el_params = weights_out->el_params;
    w_out_new_g.el_type = weights_out->el_type;
    mli_prv_tensor_inc_data_ptr<w_T*>(&w_out_new_g, num_gates * w_gate_mem_strides[1]);

    const MLI_PTR (w_T) w_new_g_ptr[] = {
        mli_prv_tensor_data_ptr<MLI_PTR (w_T)>(weights_in) + num_gates * w_gate_mem_strides[0], 
        mli_prv_tensor_data_ptr<MLI_PTR (w_T)>(weights_out) + num_gates * w_gate_mem_strides[1]
    };

    const MLI_PTR (b_T) b_new_g_ptr = mli_prv_tensor_data_ptr<MLI_PTR (b_T)>(&b_new_g);

    mli_tensor rnn_out = {{ 0 }};
    rnn_out.data = out->data;
    rnn_out.rank = 2;
    rnn_out.shape[0] = 1;
    rnn_out.shape[1] = static_cast<unsigned>(gru_out_elements);
    rnn_out.mem_stride[0] = rnn_out.shape[1];
    rnn_out.mem_stride[1] = 1;
    rnn_out.el_type = in->el_type;

    mli_tensor current_hidden = {{ 0 }};
    current_hidden.data = prev_out->data;
    current_hidden.rank = 2;
    current_hidden.shape[0] = 1;
    current_hidden.shape[1] = prev_out->shape[0];
    current_hidden.mem_stride[0] = current_hidden.shape[1];
    current_hidden.mem_stride[1] = 1;

    current_hidden.el_type = in->el_type;
    current_hidden.el_params = prev_out->el_params;
    mli_tensor current_out = current_hidden;
    current_out.data = out->data;
    current_out.el_params = ir_tensor.el_params;

    mli_tensor prev_out_reset = {{ 0 }};
    prev_out_reset.data = reset_gate.data;
    prev_out_reset.rank = reset_gate.rank;
    prev_out_reset.shape[0] = reset_gate.shape[0];
    prev_out_reset.shape[1] = reset_gate.shape[1];
    prev_out_reset.mem_stride[0] = reset_gate.mem_stride[0];
    prev_out_reset.mem_stride[1] = reset_gate.mem_stride[1];
    prev_out_reset.el_type = in->el_type;
    prev_out_reset.el_params = ir_tensor.el_params;

    const MLI_PTR (io_T) inputs_new_ptr[] = {inputs_ptr[0], 
                                             mli_prv_tensor_data_ptr<MLI_PTR (io_T)>(&prev_out_reset)};

    for (int batch = 0; batch < batch_sz; batch++) {

        // Step 1: Applying Dense
        //=======================================
        mli::krn::rnn_dense_op_stacked<io_T, w_T, b_T, acc_T, quant_T>(
            inputs_ptr, weights, bias, num_gates, num_inputs, inputs_elements,
            in_to_out_params, w_ch_out_mem_strides, w_gate_mem_strides, &ir_tensor);

        // Step 2: Applying non-linearity
        //=======================================
        update_gate.el_params = reset_gate.el_params = new_gate.el_params = ir_tensor.el_params;
        
        mli_tensor update_gate_input = update_gate;
        mli_tensor reset_gate_input = reset_gate;
        mli_tensor new_gate_input = new_gate;

        if (asym) {
            mli_krn_sigm_sa8(&update_gate_input, sigm_lut, &update_gate);
            mli_krn_sigm_sa8(&reset_gate_input, sigm_lut, &reset_gate);
        } else {
            if (sizeof(io_T)==sizeof(int8_t)) {
                mli_krn_sigm_fx8(&update_gate_input, sigm_lut, &update_gate);
                mli_krn_sigm_fx8(&reset_gate_input, sigm_lut, &reset_gate);
            } else if (sizeof(io_T)==sizeof(int16_t)) {
                mli_krn_sigm_fx16(&update_gate_input, sigm_lut, &update_gate);
                mli_krn_sigm_fx16(&reset_gate_input, sigm_lut, &reset_gate);
            } else {
                MLI_ASSERT(0);
            }
        }

        // Step 3: Pointwise operations
        //=======================================
        mli::krn::eltwise_prepare_and_run<io_T, ELTWISE_MUL, /*convert*/ asym>(&reset_gate, &current_hidden, &prev_out_reset);
        mli::krn::eltwise_prepare_and_run<io_T, ELTWISE_MUL, /*convert*/ asym>(&update_gate, &current_hidden, &current_out);
        mli::krn::eltwise_prepare_and_run<io_T, ELTWISE_SUB, /*convert*/ asym>(&one, &update_gate, &update_gate);

        // Step 4: New gate
        //=======================================
        mli_relu_cfg relu_none = {MLI_RELU_NONE};
        mli_minmax_t val_limit = mli_prv_get_relu_limits<io_T, asym>(&relu_none, &new_gate_input);

        if (asym) {
            inc_scales_for_new_gate(&w_in_new_g.el_params, num_gates);
            inc_scales_for_new_gate(&w_out_new_g.el_params, num_gates);
            inc_scales_for_new_gate(&b_new_g.el_params, num_gates);
        }

        define_quant_params(in, &w_in_new_g, &b_new_g, &new_gate_input, &in_to_out_params[0]);
        define_quant_params(&prev_out_reset, &w_out_new_g, &b_new_g, &new_gate_input, &in_to_out_params[1]);

        MLI_PTR (io_T) new_gate_ptr = mli_prv_tensor_data_ptr<MLI_PTR (io_T)>(&new_gate_input);
        mli::krn::rnn_dense_op<io_T, w_T, b_T, acc_T, quant_T>(
            inputs_new_ptr, w_new_g_ptr, b_new_g_ptr, new_gate_ptr, num_inputs, inputs_elements, gru_out_elements,
            w_ch_out_mem_strides, in_to_out_params, (io_T)val_limit.min, (io_T)val_limit.max);

        if (asym) {
            mli_krn_tanh_sa8(&new_gate_input, tanh_lut, &new_gate);
        } else {
            if (sizeof(io_T)==sizeof(int8_t)) {
                mli_krn_tanh_fx8(&new_gate_input, tanh_lut, &new_gate);
            } else if (sizeof(io_T)==sizeof(int16_t)) {
                mli_krn_tanh_fx16(&new_gate_input, tanh_lut, &new_gate);
            } else {
                MLI_ASSERT(0);
            }
        }

        if (asym) {
            dec_scales_for_new_gate(&w_in_new_g.el_params, num_gates);
            dec_scales_for_new_gate(&w_out_new_g.el_params, num_gates);
            dec_scales_for_new_gate(&b_new_g.el_params, num_gates);
        }

        // Step 5: Calculate output: Activation + pointwise operation
        //===========================================================
        mli_tensor temp;
        temp.data = new_gate.data;
        temp.rank = new_gate.rank;
        temp.shape[0] = new_gate.shape[0];
        temp.shape[1] = new_gate.shape[1];
        temp.mem_stride[0] = new_gate.mem_stride[0];
        temp.mem_stride[1] = new_gate.mem_stride[1];
        temp.el_type = new_gate.el_type;
        temp.el_params = current_hidden.el_params;
   
        rnn_out.el_params = out->el_params;
        mli::krn::eltwise_prepare_and_run<io_T, ELTWISE_MUL, /*convert*/ asym>(&new_gate, &update_gate, &temp);
        mli::krn::eltwise_prepare_and_run<io_T, ELTWISE_ADD, /*convert*/ asym>(&temp, &current_out, &rnn_out);

        current_hidden.data = rnn_out.data;
        current_hidden.el_params = rnn_out.el_params;

        // Step 6: Update pointers and tensors for next batch
        //=======================================
        inputs_ptr[0] += cfg->direction == RNN_DIR_FORWARD ? inputs_elements[0] : -inputs_elements[0];
        inputs_new_ptr[0] += cfg->direction == RNN_DIR_FORWARD ? inputs_elements[0] : -inputs_elements[0];
        inputs_ptr[1] = mli_prv_tensor_data_ptr<MLI_PTR (io_T)>(&current_hidden);

        if (asym) {
            define_quant_params(in, weights_in, bias, &ir_tensor, &in_to_out_params[0]);
            define_quant_params(&current_hidden, weights_out, bias, &ir_tensor, &in_to_out_params[1]);
        } else {
            define_quant_params(&current_hidden, weights_out, bias, &ir_tensor, &in_to_out_params[1]);
        }

        if (cfg->results == RNN_OUT_ALL) {
            mli_prv_tensor_inc_data_ptr<io_T*>(&rnn_out, gru_out_elements);
        }

        current_out = rnn_out;
    }

    // Fill output tensor params
    out->el_type = rnn_out.el_type;
    if (cfg->results == RNN_OUT_LAST) {
        out->rank = 2;
        out->shape[0] = 1;
        out->shape[1] = gru_out_elements;
    } else {
        out->rank = 2;
        out->shape[0] = batch_sz;
        out->shape[1] = gru_out_elements;
    }
}

#pragma MLI_CODE_SECTION_END()
} // namespace mli
} // namespace krn

#endif  //_MLI_KRN_GRU_CELL_H_
