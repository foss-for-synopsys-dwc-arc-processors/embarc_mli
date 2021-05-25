/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_LSTM_CELL_H_
#define _MLI_KRN_LSTM_CELL_H_

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

template <typename io_T, mli_eltwise_type func_type, bool convert>
MLI_FORCE_INLINE void lstm_eltwise(
        const mli_tensor * in1,
        const mli_tensor * in2,
        mli_tensor * out) {

    mli_prv_fx_init_dsp_ctrl();
    int pre_op_shift1 = 0, pre_op_shift2 = 0, post_op_shift = 0;

    constexpr int byte_size = 8;
    if(!convert)    
    {
        /*
        * max_shift will be determined according to the size of the out register to avoid
        * overflow in the rounding value.
        */
        int max_shift = sizeof(io_T) * byte_size;
        if (func_type == ELTWISE_MUL) {
            max_shift = 2 * max_shift - 1;
            post_op_shift = mli_prv_calc_shift(in1, in2, out);
        } else if (func_type == ELTWISE_MIN || func_type == ELTWISE_MAX) {
            max_shift = max_shift - 1;
            post_op_shift = in1->el_params.fx.frac_bits - out->el_params.fx.frac_bits;
        } else {
            max_shift = 2 * max_shift - 1;
            pre_op_shift1 = MIN(in1->el_params.fx.frac_bits -  in2->el_params.fx.frac_bits, 0);
            pre_op_shift2 = MIN(in2->el_params.fx.frac_bits -  in1->el_params.fx.frac_bits, 0);
            post_op_shift = MAX(in1->el_params.fx.frac_bits, in2->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;
        }
        post_op_shift = MIN(post_op_shift, max_shift);
    }

    /* Extract general parameters for function */
    //assumption that always 0 
    bool scalar_op1 = 0;
    bool scalar_op2 = 0;

    /* Extract in/out as scalar values */
    io_T in1_scalar = mli_prv_tensor_data_val<io_T>(in1);
    io_T in2_scalar = mli_prv_tensor_data_val<io_T>(in2);
            
     /* Fill output tensor parameters
    //======================================
     */
    //in1_sz = mli_prv_count_elem_num(in1);
    //in2_sz = mli_prv_count_elem_num(in2);
    //assumption that in1_sz and in2_sz are always the same 
    // assuming that always in1 is out no need to update out
    MLI_ASSERT( in1 == out);
    
    MLI_ASSERT( in1->rank == 2 && in1->shape[0] == 1 || in1->rank == 1);
    //assumption that always in2->mem_stride[]==0 and out->mem_stride[]==0 
    int flatten_count = in1->shape[in1->rank - 1];
 
    auto in1_ptr = mli_prv_tensor_data_ptr<MLI_PTR(io_T)>(in1);
    auto in2_ptr = mli_prv_tensor_data_ptr<MLI_PTR(io_T)>(in2);
    auto out_ptr = mli_prv_tensor_data_ptr<MLI_PTR(io_T)>(out);

    mli::krn::eltwise_innerloop<io_T, func_type, /*convert*/ false>(
                in1_ptr, in2_ptr , out_ptr, 0, 0, 0, flatten_count,
                in1_scalar, in2_scalar, scalar_op1, scalar_op2,
                /*in_offset1*/ 0, /*in_offset2*/ 0, /*out_offset*/ 0, 
                /*scale16_1*/ 1, /*scale16_2*/ 1, pre_op_shift1, pre_op_shift2, post_op_shift);
}

//========================================================================================
// Common routine for pre-calculation of various basic rnn cell parameters and running it.
//========================================================================================

template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void lstm_cell_prepare_and_run(
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out, 
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut, 
        const mli_rnn_cell_cfg * cfg, 
        mli_tensor * cell,
        mli_tensor *out) {

    constexpr bool asym = std::is_same<quant_T, s8asym_quant_specific_params>::value;

    const int inputs_elements[] = {(int)mli_prv_count_elem_num_part(in, 1), (int)mli_prv_count_elem_num(prev_out)};
    const int lstm_out_elements = static_cast<int>(mli_prv_count_elem_num(prev_out));
    const int seq_len = in->shape[0];
    const int8_t num_gates = 4;
    const int8_t num_inputs = 2;

    const mli_tensor * weights[2] = {weights_in, weights_out};
    const MLI_PTR (io_T) inputs_ptr[] = {mli_prv_tensor_data_ptr<MLI_PTR (io_T)>(in), 
                                         mli_prv_tensor_data_ptr<MLI_PTR (io_T)>(prev_out)};

    if (cfg->direction == RNN_DIR_BACKWARD) 
        inputs_ptr[0] += (seq_len - 1) * inputs_elements[0];

    // Fill intermediate tensor of dense output
    mli_tensor ir_tensor;
    ir_tensor.data = cfg->scratch_data;
    ir_tensor.rank = bias->rank;
    ir_tensor.shape[0] = bias->shape[0];
    ir_tensor.shape[1] = bias->shape[1];
    ir_tensor.mem_stride[0] = 0;
    ir_tensor.mem_stride[1] = 0;
    ir_tensor.el_type = in->el_type;

    if (asym) {
        mli_element_params ir_asym_params;
        ir_asym_params.sa.dim = -1;
        ir_asym_params.sa.scale.mem.i16 = 1;
        ir_asym_params.sa.zero_point.mem.i16 = 0;
        ir_asym_params.sa.scale_frac_bits.mem.i16 = 4;
        ir_asym_params.sa.scale.capacity = ir_asym_params.sa.zero_point.capacity = 0;
        ir_asym_params.sa.scale_frac_bits.capacity = 0;
        ir_tensor.el_params = ir_asym_params;
    } else { 
        // 1sign and 3 integer bits for TANH/SIGM input is enough
        ir_tensor.el_params.fx.frac_bits = (sizeof(io_T) * 8) - 1 - 3;    
    }

    quant_T in_to_out_params[2];
    define_quant_params(in, weights_in, bias, &ir_tensor, &in_to_out_params[0]);
    define_quant_params(prev_out, weights_out, bias, &ir_tensor, &in_to_out_params[1]);


    const int w_ch_out_mem_stride_from_tensors[] = {(int)weights_in->mem_stride[KRNL_RNN_W_IN_ELEMS_DIM], 
                                                    (int)weights_out->mem_stride[KRNL_RNN_W_IN_ELEMS_DIM]};

    const int w_gate_mem_stride_from_tensors[] = {(int)weights_in->mem_stride[0], 
                                                  (int)weights_out->mem_stride[0]};

    const int w_ch_out_mem_strides[] = {(w_ch_out_mem_stride_from_tensors[0] != 0) 
                                            ? w_ch_out_mem_stride_from_tensors[0] : lstm_out_elements, 
                                        (w_ch_out_mem_stride_from_tensors[1] != 0) 
                                            ? w_ch_out_mem_stride_from_tensors[1]: lstm_out_elements};

    const int w_gate_mem_strides[] = {(w_gate_mem_stride_from_tensors[0] != 0) 
                                        ? w_gate_mem_stride_from_tensors[0] : lstm_out_elements * inputs_elements[0], 
                                      (w_gate_mem_stride_from_tensors[1] != 0) 
                                        ? w_gate_mem_stride_from_tensors[1]: lstm_out_elements * inputs_elements[1]};

    // Paricular subtensors of intermediate tensor (mli_tensor.mem_stride[] should be zero and cannot be left uninitialized)
    mli_tensor in_gate, forget_gate , out_gate ; // Various gates to controll info flow
    mli_tensor g_tsr; // Information tensors
    for (int r=0; r < MLI_MAX_RANK; r++)
    {
        in_gate.mem_stride[r]=0;
        forget_gate.mem_stride[r]=0;
        out_gate.mem_stride[r]=0;
        g_tsr.mem_stride[r]=0;
    }

    // Init subtensors
    mli_point_to_subtsr_cfg iterator = {/*.start_coord =*/ {0}, /*.coord_num=*/ 1, /*.first_out_dim_size=*/ 1};
    mli_hlp_point_to_subtensor(&ir_tensor, &iterator, &in_gate); iterator.start_coord[0]++;
    mli_hlp_point_to_subtensor(&ir_tensor, &iterator, &g_tsr); iterator.start_coord[0]++;
    mli_hlp_point_to_subtensor(&ir_tensor, &iterator, &forget_gate); iterator.start_coord[0]++;
    mli_hlp_point_to_subtensor(&ir_tensor, &iterator, &out_gate);

    mli_tensor rnn_out;
    rnn_out.data = out->data;
    rnn_out.rank = 2;
    rnn_out.shape[0] = 1;
    rnn_out.shape[1] = static_cast<unsigned>(lstm_out_elements);
    rnn_out.mem_stride[0] = 0;
    rnn_out.mem_stride[1] = 0;
    rnn_out.el_type = in->el_type;

    for (int timestep = 0; timestep < seq_len; timestep++) {
        
        // Step 1: Applying Dense
        //=======================================
        rnn_dense_op_stacked<io_T, w_T, b_T, acc_T, quant_T>(
            inputs_ptr, weights, bias, num_gates, num_inputs, inputs_elements,
            in_to_out_params, w_ch_out_mem_strides, w_gate_mem_strides, &ir_tensor);


        // Step 2: Applying non-linearity
        //=======================================
        in_gate.el_params = out_gate.el_params = ir_tensor.el_params;
        g_tsr.el_params = forget_gate.el_params = ir_tensor.el_params;
        
        mli_tensor in_gate_input = in_gate;
        mli_tensor g_tsr_input = g_tsr;
        mli_tensor forget_gate_input = forget_gate;
        mli_tensor out_gate_input = out_gate;

        if (asym) {
            mli_krn_sigm_sa8(&in_gate_input, sigm_lut, &in_gate);
            mli_krn_tanh_sa8(&g_tsr_input, tanh_lut, &g_tsr);
            mli_krn_sigm_sa8(&forget_gate_input, sigm_lut, &forget_gate);
            mli_krn_sigm_sa8(&out_gate_input, sigm_lut, &out_gate);
        } else {
            if (sizeof(io_T)==sizeof(int8_t)) {
                mli_krn_sigm_fx8(&in_gate_input, sigm_lut, &in_gate);
                mli_krn_tanh_fx8(&g_tsr_input, tanh_lut, &g_tsr);
                mli_krn_sigm_fx8(&forget_gate_input, sigm_lut, &forget_gate);
                mli_krn_sigm_fx8(&out_gate_input, sigm_lut, &out_gate);
            } else if (sizeof(io_T)==sizeof(int16_t)) {
                mli_krn_sigm_fx16(&in_gate_input, sigm_lut, &in_gate);
                mli_krn_tanh_fx16(&g_tsr_input, tanh_lut, &g_tsr);
                mli_krn_sigm_fx16(&forget_gate_input, sigm_lut, &forget_gate);
                mli_krn_sigm_fx16(&out_gate_input, sigm_lut, &out_gate);
            } else {
                MLI_ASSERT(0);
            }
        }

        // Step 3: Pointwise operations
        //=======================================
        if (asym) {
            mli::krn::eltwise_prepare_and_run<io_T, ELTWISE_MUL, /*convert*/ asym>(cell,&forget_gate, cell);
            mli::krn::eltwise_prepare_and_run<io_T, ELTWISE_MUL, /*convert*/ asym>(&g_tsr, &in_gate, &g_tsr);
            mli::krn::eltwise_prepare_and_run<io_T, ELTWISE_ADD, /*convert*/ asym>(cell, &g_tsr, cell);
        } else {
            lstm_eltwise<io_T, ELTWISE_MUL, /*convert*/ false>(cell, &forget_gate, cell);
            lstm_eltwise<io_T, ELTWISE_MUL, /*convert*/ false>(&g_tsr, &in_gate, &g_tsr);
            lstm_eltwise<io_T, ELTWISE_ADD, /*convert*/ false>(cell, &g_tsr, cell);
        }

        // Step 4: Calculate output: Activation + pointwise operation
        //===========================================================
        mli_tensor temp;
        temp.data = rnn_out.data;
        temp.rank = rnn_out.rank;
        temp.shape[0] = rnn_out.shape[0];
        temp.shape[1] = rnn_out.shape[1];
        temp.mem_stride[0] = rnn_out.mem_stride[0];
        temp.mem_stride[1] = rnn_out.mem_stride[1];
        temp.el_type = rnn_out.el_type;
        temp.el_params = out->el_params;

        if (cfg->act == RNN_ACT_NONE) {
            mli::krn::eltwise_prepare_and_run<io_T, ELTWISE_MUL, /*convert*/ asym>(cell, &out_gate, &temp);
        } else {
            if (asym) {
                if (cfg->act == RNN_ACT_TANH)
                    mli_krn_tanh_sa8(cell, tanh_lut, &rnn_out);
                else if (cfg->act == RNN_ACT_SIGM)
                    mli_krn_sigm_sa8(cell, sigm_lut, &rnn_out);
                else
                    MLI_ASSERT(0);
            } else {
                if (sizeof(io_T)==sizeof(int8_t)) {
                    if (cfg->act == RNN_ACT_TANH)
                        mli_krn_tanh_fx8(cell, tanh_lut, &rnn_out);
                    else if (cfg->act == RNN_ACT_SIGM)
                        mli_krn_sigm_fx8(cell, sigm_lut, &rnn_out);
                    else
                        MLI_ASSERT(0);
                } else if (sizeof(io_T)==sizeof(int16_t)) {
                    if (cfg->act == RNN_ACT_TANH)
                        mli_krn_tanh_fx16(cell, tanh_lut, &rnn_out);
                    else if (cfg->act == RNN_ACT_SIGM)
                        mli_krn_sigm_fx16(cell, sigm_lut, &rnn_out);
                    else
                        MLI_ASSERT(0);
                } else {
                    MLI_ASSERT(0);
                }
            }

            mli::krn::eltwise_prepare_and_run<io_T, ELTWISE_MUL, /*convert*/ asym>(&rnn_out, &out_gate, &temp);
        }
        rnn_out.el_params = out->el_params;

        // Step 5: Update pointers and tensors for next timestep
        //=======================================
        inputs_ptr[0] += cfg->direction == RNN_DIR_FORWARD ? inputs_elements[0] : -inputs_elements[0];
        inputs_ptr[1] = mli_prv_tensor_data_ptr<MLI_PTR (io_T)>(&rnn_out);

        if (asym) {
            rnn_out.el_params = out->el_params;
            define_quant_params(in, weights_in, bias, &ir_tensor, &in_to_out_params[0]);
            define_quant_params(out, weights_out, bias, &ir_tensor, &in_to_out_params[1]);
        } else {
            define_quant_params(&rnn_out, weights_out, bias, &ir_tensor, &in_to_out_params[1]);
        }

        if (cfg->results == RNN_OUT_ALL) {
            mli_prv_tensor_inc_data_ptr<io_T*>(&rnn_out, lstm_out_elements);
        }
    }

    // Fill output tensor params
    out->el_type = rnn_out.el_type;
    if (cfg->results == RNN_OUT_LAST) {
        out->rank = 2;
        out->shape[0] = 1;
        out->shape[1] = lstm_out_elements;
    } else {
        out->rank = 2;
        out->shape[0] = seq_len;
        out->shape[1] = lstm_out_elements;
    }
}

#pragma MLI_CODE_SECTION_END()
} // namespace mli
} // namespace krn

#endif  //_MLI_KRN_LSTM_CELL_H_
