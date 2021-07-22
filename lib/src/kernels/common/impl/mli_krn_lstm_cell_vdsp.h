/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_LSTM_CELL_VDSP_H_
#define _MLI_KRN_LSTM_CELL_VDSP_H_

#include <type_traits>

#include "mli_api.h"
#include "mli_math.h"
#include "mli_mem_info.h"
#include "mli_private_types.h"
#include "mli_prv_quant.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"
#include "mli_krn_eltwise.h"
#include "mli_prv_activation_lut.h"

#include "mli_krn_rnn_dense_op.h"

namespace mli {
namespace krn {
namespace vdsp {    

#pragma MLI_CODE_SECTION_START(".mli_lib")

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

    MLI_ASSERT(in->rank==2);
    MLI_ASSERT(prev_out->rank==1);
    __builtin_assume(prev_out->rank==1);
    __builtin_assume(in->rank==2);
    const uint32_t lstm_out_elements =mli_prv_count_elem_num(prev_out);
    const int inputs_elements[] = {(int)mli_prv_count_elem_num_part(in, 1), (int)lstm_out_elements};

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
    ir_tensor.mem_stride[0] = ir_tensor.shape[1];
    ir_tensor.mem_stride[1] = 1;
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
        ir_tensor.el_params.fx.frac_bits = MIN(ir_tensor.el_params.fx.frac_bits, in->el_params.fx.frac_bits + weights_in->el_params.fx.frac_bits);
    }

    quant_T in_to_out_params[2];
    define_quant_params(in, weights_in, bias, &ir_tensor, &in_to_out_params[0]);
    define_quant_params(prev_out, weights_out, bias, &ir_tensor, &in_to_out_params[1]);


    const int w_ch_out_mem_strides[] = {(int)weights_in->mem_stride[KRNL_RNN_W_IN_ELEMS_DIM],
                                        (int)weights_out->mem_stride[KRNL_RNN_W_IN_ELEMS_DIM]};

    const int w_gate_mem_strides[] = {(int)weights_in->mem_stride[0],
                                      (int)weights_out->mem_stride[0]};

    // Paricular subtensors of intermediate tensor
    mli_tensor tmp_gate; // tmp tensor store parameters of various gates (in_gate, forget_gate, out_gate) that control information flow  
    mli_tensor g_tsr; // Information tensor
    
    g_tsr = tmp_gate = ir_tensor; 
    // update shape
    tmp_gate.shape[0] = 1;
    g_tsr.shape[0] = 1;
    // update data ptr
    //mli_prv_tensor_inc_data_ptr<io_T*>(&tmp_gate,                           0 );
    mli_data_container dtcntr_in_gate = tmp_gate.data; //store data of in_gate 
    mli_prv_tensor_inc_data_ptr<io_T*>(&g_tsr,         ir_tensor.mem_stride[0]);
    mli_prv_tensor_inc_data_ptr<io_T*>(&tmp_gate, (2 * ir_tensor.mem_stride[0]));
    mli_data_container dtcntr_forget_gate = tmp_gate.data; //store data of forget_gate 
    mli_prv_tensor_inc_data_ptr<io_T*>(&tmp_gate,      ir_tensor.mem_stride[0]);
    mli_data_container dtcntr_out_gate = tmp_gate.data; //store data of out_gate 

        

    mli_tensor rnn_out;
    rnn_out.data = out->data;
    rnn_out.rank = 2;
    rnn_out.shape[0] = 1;
    rnn_out.shape[1] = lstm_out_elements;
    rnn_out.mem_stride[0] = rnn_out.shape[1];
    rnn_out.mem_stride[1] = 1;
    rnn_out.el_type = in->el_type;

    cell->rank = 2;
    cell->shape[0] = tmp_gate.shape[0];
    cell->shape[1] = tmp_gate.shape[1];
    cell->mem_stride[0] = tmp_gate.mem_stride[0];
    cell->mem_stride[1] = tmp_gate.mem_stride[1];

    struct s8asym_quant_params out_params_sigm;
    struct s8asym_quant_params out_params_tanh;
    
    if (asym) {
        out_params_tanh.offset = K_TANH_ASYM_ZERO_POINT;
        out_params_tanh.scale  = 1;
        out_params_tanh.shift = K_TANH_OUTPUT_SHIFT;
    
        out_params_sigm.offset = K_SIGM_ASYM_ZERO_POINT;
        out_params_sigm.scale  = 1;
        out_params_sigm.shift = K_SIGM_OUTPUT_SHIFT; 

        g_tsr.el_params.sa.zero_point.mem.i16 = out_params_tanh.offset;
        g_tsr.el_params.sa.scale.mem.i16 = out_params_tanh.scale;
        g_tsr.el_params.sa.scale_frac_bits.mem.i8 = (int8_t)out_params_tanh.shift;

        tmp_gate.el_params.sa.zero_point.mem.i16 = out_params_sigm.offset;
        tmp_gate.el_params.sa.scale.mem.i16 = out_params_sigm.scale;
        tmp_gate.el_params.sa.scale_frac_bits.mem.i8 = (int8_t)out_params_sigm.shift;
    } else {
            if (sizeof(io_T)==sizeof(int8_t)) {
                tmp_gate.el_params.fx.frac_bits = 7;
                g_tsr.el_params.fx.frac_bits = 7;
            } else if (sizeof(io_T)==sizeof(int16_t)) {
                tmp_gate.el_params.fx.frac_bits = 15;
                g_tsr.el_params.fx.frac_bits = 15;
            } else {
                MLI_ASSERT(0);
            }
    }

    for (int timestep = 0; timestep < seq_len; timestep++) {

        MLI_ASSERT(bias->rank==2);
        __builtin_assume(bias->rank==2);
        // Step 1: Applying Dense
        //=======================================
        rnn_dense_op_stacked<io_T, w_T, b_T, acc_T, quant_T>(
            inputs_ptr, weights, bias, num_gates, num_inputs, inputs_elements,
            in_to_out_params, w_ch_out_mem_strides, w_gate_mem_strides, &ir_tensor);


        // Step 2: Applying non-linearity
        //=======================================
        __builtin_assume(tmp_gate.rank==2);
        __builtin_assume(g_tsr.rank==2);
        if (asym) {
            struct s8asym_quant_params in_params;
           
            in_params.offset = ir_tensor.el_params.sa.zero_point.mem.i16;
            in_params.scale  = ir_tensor.el_params.sa.scale.mem.i16;
            in_params.shift = ir_tensor.el_params.sa.scale_frac_bits.mem.i8;
            tmp_gate.data = dtcntr_in_gate; // switch data to in_gate            
            mli_prv_activation_lut_sa8(&tmp_gate, &tmp_gate, sigm_lut, &in_params, &out_params_sigm);
            mli_prv_activation_lut_sa8(&g_tsr, &g_tsr, tanh_lut, &in_params, &out_params_tanh);
            tmp_gate.data = dtcntr_forget_gate;  // switch data ptr to forget_gate
            tmp_gate.shape[1]  *=2; // increase len to combine calculation of forget_gate and out_gate
            mli_prv_activation_lut_sa8(&tmp_gate, &tmp_gate, sigm_lut, &in_params, &out_params_sigm);
            tmp_gate.shape[1]  = g_tsr.shape[1]; //restore len
            
        } else {
            if (sizeof(io_T)==sizeof(int8_t)) {
                auto frac_bits = ir_tensor.el_params.fx.frac_bits;
                tmp_gate.data = dtcntr_in_gate; // switch data to in_gate
                mli_prv_activation_lut_fx8(&tmp_gate, &tmp_gate, sigm_lut, frac_bits);
                mli_prv_activation_lut_fx8(&g_tsr, &g_tsr, tanh_lut, frac_bits);
                tmp_gate.data = dtcntr_forget_gate; // switch data to forget_gate
                tmp_gate.shape[1]  *=2; // increase len to combine calculation of forget_gate and out_gate
                mli_prv_activation_lut_fx8(&tmp_gate, &tmp_gate, sigm_lut, frac_bits);
                tmp_gate.shape[1]  = g_tsr.shape[1]; //restore len
                
            } else if (sizeof(io_T)==sizeof(int16_t)) {
                auto frac_bits = ir_tensor.el_params.fx.frac_bits;
                tmp_gate.data = dtcntr_in_gate; // switch data to in_gate
                mli_prv_activation_lut_fx16(&tmp_gate, &tmp_gate, sigm_lut, frac_bits);
                mli_prv_activation_lut_fx16(&g_tsr, &g_tsr, tanh_lut, frac_bits);
                tmp_gate.data = dtcntr_forget_gate; // switch data to forget_gate
                tmp_gate.shape[1]  *=2; // increase len to combine calculation of forget_gate and out_gate
                mli_prv_activation_lut_fx16(&tmp_gate, &tmp_gate, sigm_lut, frac_bits);
                tmp_gate.shape[1]  = g_tsr.shape[1]; //restore len
                
            } else {
                MLI_ASSERT(0);
            }
        }

        // Step 3: Pointwise operations
        //=======================================
        //tmp_gate.data = dtcntr_forget_gate; // switch data to forget_gate
        mli::krn::eltwise_prepare_and_run<io_T, ELTWISE_MUL, /*convert*/ asym, /*no_scalar*/ true, /*no_out_update*/ true, /*shape_1d*/ true>(cell, &tmp_gate, cell);
        tmp_gate.data = dtcntr_in_gate; // switch data to in_gate
        mli::krn::eltwise_prepare_and_run<io_T, ELTWISE_MUL, /*convert*/ asym, /*no_scalar*/ true, /*no_out_update*/ true, /*shape_1d*/ true>(&g_tsr, &tmp_gate, &g_tsr);
        mli::krn::eltwise_prepare_and_run<io_T, ELTWISE_ADD, /*convert*/ asym, /*no_scalar*/ true, /*no_out_update*/ true, /*shape_1d*/ true>(cell, &g_tsr, cell);

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
            tmp_gate.data = dtcntr_out_gate; // switch data to out_gate
            mli::krn::eltwise_prepare_and_run<io_T, ELTWISE_MUL, /*convert*/ asym, /*no_scalar*/ true, /*no_out_update*/ true, /*shape_1d*/ true>(cell, &tmp_gate, &temp);
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

            tmp_gate.data = dtcntr_out_gate; // switch data to out_gate
            mli::krn::eltwise_prepare_and_run<io_T, ELTWISE_MUL, /*convert*/ asym, /*no_scalar*/ true, /*no_out_update*/ true, /*shape_1d*/ true>(&rnn_out, &tmp_gate, &temp);
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
            mli_prv_tensor_inc_data_ptr<io_T*>(&rnn_out, (int)lstm_out_elements);
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
} // namespace vdsp
} // namespace mli
} // namespace krn

#endif  //_MLI_KRN_LSTM_CELL_VDSP_H_