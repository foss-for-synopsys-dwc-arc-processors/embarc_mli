/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "har_smartphone_model.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mli_api.h"
#include "mli_config.h"

#include "har_smartphone_constants.h"
#include "tests_aux.h"


//==============================================================
//
//
// Data related to the Module
//
//
//==============================================================


// Intermediate data buffers (enough size for max intermediate results)
//==============================
#define LSTM_CELL_SZ (32)
#define INOUT_BUF_SZ_MOST (128*LSTM_CELL_SZ)
#define LSTM_IR_BUF_SZ (4*LSTM_CELL_SZ)
#define LUT_BUF_SZ (512)

// Despite the name of buf we keep all in/out data
// in the same bank (typically first in operand)
// Weights and lstm memory in the another (typically second input operand)
// 11d has got only 2 separate banks of memory
static d_type  _Y    x_mem_buf[INOUT_BUF_SZ_MOST];
static d_type  _Y    y_mem_buf[INOUT_BUF_SZ_MOST];
static d_type  _Y    lstm_ir_mem_buf[LSTM_IR_BUF_SZ];
static d_type  _X    lstm_cell_mem_buf[LSTM_CELL_SZ];
static int16_t  _Y   tanh_lut_mem_buf[LUT_BUF_SZ];
static int16_t  _X   sigm_lut_mem_buf[LUT_BUF_SZ];

// Module intermediate tensors 
//=============================
static mli_tensor input_float = {
    .data = {
        .capacity = 0,
        .mem = { .pf32 = NULL }
    },
    .mem_stride = {9, 1},
    .shape = {128, 9},
    .rank = 2,
    .el_type = MLI_EL_FP_32,
};

static mli_tensor L0_move_out = {
    .data = {
        .capacity = sizeof(y_mem_buf),
        .mem = {.pf32 = (float*)y_mem_buf }
    },
    .mem_stride = {9, 1},
    .shape = {128, 9},
    .rank = 2,
    .el_type = MLI_EL_FP_32,
};

static mli_tensor L0_convert_out = {
    .data = {
        .capacity = sizeof(x_mem_buf),
        .mem = {.D_FIELD = (d_type*)x_mem_buf }
    },
    .mem_stride = {9, 1},
    .shape = {128, 9},
    .rank = 2,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = sizeof(d_type) * 8 - 1 - 2,
};

static mli_tensor L1_fc_out = {
    .data = {
        .capacity = sizeof(y_mem_buf),
        .mem = {.D_FIELD = (d_type*)y_mem_buf }
    },
    .mem_stride = {32, 1},
    .shape = {128, 32},
    .rank = 2,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = FC1_OUT_FRAQ,
};

static mli_tensor L2_lstm_cell = {
    .data = {
        .capacity = sizeof(lstm_cell_mem_buf),
        .mem = {.D_FIELD = (d_type*)lstm_cell_mem_buf }
    },
    .mem_stride = {1},
    .shape = {LSTM_CELL_SZ},
    .rank = 1,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = LSTM2_CELL_FRAQ
};

static mli_tensor L2_lstm_prev = {
    .data = {
        .capacity = sizeof(x_mem_buf),
        .mem = {.D_FIELD = (d_type*)x_mem_buf }
    },
    .mem_stride = {1},
    .shape = {LSTM_CELL_SZ},
    .rank = 1,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = LSTM2_OUT_FRAQ
};

static mli_tensor L2_lstm_out = {
    .data = {
        .capacity = sizeof(x_mem_buf),
        .mem = {.D_FIELD = (d_type*)x_mem_buf }
    },
    .mem_stride = {32, 1},
    .shape = {128, 32},
    .rank = 2,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = LSTM2_OUT_FRAQ
};

static mli_tensor L3_lstm_cell = {
    .data = {
        .capacity = sizeof(lstm_cell_mem_buf),
        .mem = {.D_FIELD = (d_type*)lstm_cell_mem_buf }
    },
    .mem_stride = {1},
    .shape = {LSTM_CELL_SZ},
    .rank = 1,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = LSTM3_CELL_FRAQ
};

static mli_tensor L3_lstm_prev = {
    .data = {
        .capacity = sizeof(y_mem_buf),
        .mem = {.D_FIELD = (d_type*)y_mem_buf }
    },
    .mem_stride = {1},
    .shape = {LSTM_CELL_SZ},
    .rank = 1,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = LSTM3_OUT_FRAQ
};

static mli_tensor L3_lstm_out = {
    .data = {
        .capacity = sizeof(y_mem_buf),
        .mem = {.D_FIELD = (d_type*)y_mem_buf }
    },
    .mem_stride = {32, 1},
    .shape = {1, 32},
    .rank = 2,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = LSTM3_OUT_FRAQ
};


static mli_tensor output = {
    .data = {
        .capacity = sizeof(x_mem_buf),
        .mem = { .D_FIELD = (d_type *)x_mem_buf }
    },
    .mem_stride = {1},
    .shape = {6},
    .rank = 1,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = FC4_OUT_FRAQ,
};

static mli_lut tanh_lut = {
    .data = {
        .capacity = sizeof(int16_t) * LUT_BUF_SZ,
        .mem = { .pi16 = (int16_t *)tanh_lut_mem_buf}
    },

};

static mli_lut sigm_lut = {
    .data = {
        .capacity = sizeof(int16_t) * LUT_BUF_SZ,
        .mem = { .pi16 = (int16_t *)sigm_lut_mem_buf}
    },

};


// Interface variables: Available to user via main model header
//===========================================================
mli_tensor * const har_smartphone_net_input = &input_float;
mli_tensor * const har_smartphone_net_output = &output;

//==============================================================
//  Model description and configuration
//==============================================================
static const mli_tensor zero_tsr_fx16 = {
    .data = {
        .capacity = 0,
        .mem = {.i16 = 0 }
    },
    .el_type = MLI_EL_FX_16,
    .rank = 0,
    .shape = {0},
    .mem_stride = {1},
    .el_params.fx.frac_bits = 0,
};

// Layer 1: Fully Connected related data
//===================================
static const mli_tensor L1_fc_wt = {
    .data = {
        .capacity = FC1_W_ELEMENTS * sizeof(w_type),
        .mem = { .W_FIELD = (w_type *)L1_fc_wt_buf }
    },
    .mem_stride = FC1_W_MEM_STRIDE,
    .shape = FC1_W_SHAPE,
    .rank = FC1_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = FC1_W_FRAQ,
};

static const mli_tensor L1_fc_bias = {
    .data = {
        .capacity = FC1_B_ELEMENTS * sizeof(b_type),
        .mem = { .B_FIELD = (b_type *)L1_fc_bias_buf }
    },
    .mem_stride = FC1_B_MEM_STRIDE,
    .shape = FC1_B_SHAPE,
    .rank = FC1_B_RANK,
    .el_type = B_EL_TYPE,
    .el_params.fx.frac_bits = FC1_B_FRAQ,
};

static const mli_fully_connected_cfg fc1_config = {
    .relu = {
        .type = MLI_RELU_GEN
    }
};

// LSTM Layer 2 related data
//===================================
static const mli_tensor L2_lstm_wt_in = {
    .data = {
        .capacity = LSTM2_W_IN_ELEMENTS * sizeof(w_type),
        .mem = { .W_FIELD = (w_type *)L2_lstm_wt_in_buf }
    },
    .mem_stride = LSTM2_W_IN_MEM_STRIDE,
    .shape = LSTM2_W_IN_SHAPE,
    .rank = LSTM2_W_IN_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = LSTM2_W_FRAQ,
};

static const mli_tensor L2_lstm_wt_out = {
    .data = {
        .capacity = LSTM2_W_OUT_ELEMENTS * sizeof(w_type),
        .mem = { .W_FIELD = (w_type *)L2_lstm_wt_out_buf }
    },
    .mem_stride = LSTM2_W_OUT_MEM_STRIDE,
    .shape = LSTM2_W_OUT_SHAPE,
    .rank = LSTM2_W_OUT_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = LSTM2_W_FRAQ,
};

static const mli_tensor L2_lstm_bias = {
    .data = {
        .capacity = LSTM2_B_ELEMENTS * sizeof(b_type),
        .mem = { .B_FIELD = (b_type *)L2_lstm_bias_buf }
    },
    .mem_stride = LSTM2_B_MEM_STRIDE,
    .shape = LSTM2_B_SHAPE,
    .rank = LSTM2_B_RANK,
    .el_type = B_EL_TYPE,
    .el_params.fx.frac_bits = LSTM2_B_FRAQ,
};

static const mli_rnn_cell_cfg L2_lstm_cfg = {
    .direction = RNN_DIR_FORWARD,
    .results = RNN_OUT_ALL,
    .act = RNN_ACT_TANH,
    .scratch_data = {
        .capacity = sizeof(lstm_ir_mem_buf),
        .mem = { .D_FIELD = (d_type *)lstm_ir_mem_buf }
    }
};


// LSTM Layer 3 related data
//===================================
static const mli_tensor L3_lstm_wt_in = {
    .data = {
        .capacity = LSTM3_W_IN_ELEMENTS * sizeof(w_type),
        .mem = { .W_FIELD = (w_type *)L3_lstm_wt_in_buf }
    },
    .mem_stride = LSTM3_W_IN_MEM_STRIDE,
    .shape = LSTM3_W_IN_SHAPE,
    .rank = LSTM3_W_IN_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = LSTM3_W_FRAQ,
};

static const mli_tensor L3_lstm_wt_out = {
    .data = {
        .capacity = LSTM3_W_OUT_ELEMENTS * sizeof(w_type),
        .mem = { .W_FIELD = (w_type *)L3_lstm_wt_out_buf }
    },
    .mem_stride = LSTM3_W_OUT_MEM_STRIDE,
    .shape = LSTM3_W_OUT_SHAPE,
    .rank = LSTM3_W_OUT_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = LSTM3_W_FRAQ,
};

static const mli_tensor L3_lstm_bias = {
    .data = {
        .capacity = LSTM3_B_ELEMENTS * sizeof(b_type),
        .mem = { .B_FIELD = (b_type *)L3_lstm_bias_buf }
    },
    .mem_stride = LSTM3_B_MEM_STRIDE,
    .shape = LSTM3_B_SHAPE,
    .rank = LSTM3_B_RANK,
    .el_type = B_EL_TYPE,
    .el_params.fx.frac_bits = LSTM3_B_FRAQ,
};

static const mli_rnn_cell_cfg L3_lstm_cfg = {
    .direction = RNN_DIR_FORWARD,
    .results = RNN_OUT_LAST,
    .act = RNN_ACT_TANH,
    .scratch_data = {
        .capacity = sizeof(lstm_ir_mem_buf),
        .mem = { .D_FIELD = (d_type *)lstm_ir_mem_buf }
    }
};

// FC4 Layer related data
//===================================
static const mli_tensor L4_fc_wt = {
    .data = {
        .capacity = FC4_W_ELEMENTS * sizeof(w_type),
        .mem = { .W_FIELD = (w_type *)L4_fc_wt_buf }
    },
    .mem_stride = FC4_W_MEM_STRIDE,
    .shape = FC4_W_SHAPE,
    .rank = FC4_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = FC4_W_FRAQ,
};

static const mli_tensor L4_fc_bias = {
    .data = {
        .capacity = FC4_B_ELEMENTS * sizeof(b_type),
        .mem = { .B_FIELD = (b_type *)L4_fc_bias_buf }
    },
    .mem_stride = FC4_B_MEM_STRIDE,
    .shape = FC4_B_SHAPE,
    .rank = FC4_B_RANK,
    .el_type = B_EL_TYPE,
    .el_params.fx.frac_bits = FC4_B_FRAQ,
};

static const mli_fully_connected_cfg fc4_config = {
    .relu = {
        .type = MLI_RELU_NONE
    }
};

//==============================================================
//  Wrappers on MLI Lib calls declaration
//  Next functions calls mli_lib kernels for appropriate data types
//  (MODEL_BIT_DEPTH define)
//
//==============================================================
static inline mli_status nn_fully_connected(
        const mli_tensor *in,
        const mli_tensor *weights,
        const mli_tensor  *bias,
        const mli_fully_connected_cfg *cfg,
        mli_tensor   *out);


static inline mli_status nn_lstm_cell(
        const mli_tensor *in,
        const mli_tensor *prev_out,
        const mli_tensor *weights_in,
        const mli_tensor *weights_out,
        const mli_tensor *bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg *cfg,
        mli_tensor *cell,
        mli_tensor *out);

#if defined(CUSTOM_USER_LSTM_LAYER3)
static inline mli_status rnn_dense(
        const mli_tensor** in,
        const mli_tensor** weights,
        const mli_tensor* bias,
        const mli_rnn_dense_cfg* cfg,
        mli_tensor* out);

static inline mli_status nn_sigm(const mli_tensor *in, const mli_lut *lut, mli_tensor *out);

static inline mli_status nn_tanh(const mli_tensor *in, const mli_lut *lut, mli_tensor *out);

static inline mli_status nn_eltwise_mul(const mli_tensor *in1, const mli_tensor *in2, mli_tensor *out);

static inline mli_status nn_eltwise_add(const mli_tensor *in1, const mli_tensor *in2, mli_tensor *out);
#endif

//==============================================================
//  Declaration of helper functions and user specific kernels
//==============================================================
mli_tensor mli_tsr_make_fx16(int16_t* data, uint32_t len, uint32_t rank,
                             const uint32_t* shape, int8_t frac_bits);

static mli_status user_fc_on_multiple_samples(
        const mli_tensor* input,
        mli_tensor* output,
        const mli_fully_connected_cfg* cfg);

static mli_status user_lstm(
        const mli_tensor* in,
        const mli_tensor* prev_out,
        const mli_tensor* weights_in,
        const mli_tensor* weights_out,
        const mli_tensor* bias,
        const mli_lut* tanh_lut,
        const mli_lut* sigm_lut,
        const mli_rnn_cell_cfg* lstm_cfg,
        mli_tensor* cell,
        mli_tensor* out);

static void check_result(
        const char * ir_root,
        const char * ref_file,
        mli_tensor *pred_tsr,
        unsigned cycles,
        mli_status ret_code);

// Initialize the lut for tanh and sigm
//==============================================================
mli_status har_smartphone_init() {
    uint32_t tanh_lut_size = mli_krn_tanh_get_lut_size();
    uint32_t sigm_lut_size = mli_krn_sigm_get_lut_size();
    if (tanh_lut_size > tanh_lut.data.capacity || sigm_lut_size > sigm_lut.data.capacity) {
        return MLI_STATUS_NOT_ENGH_MEM;
    }

    if (mli_krn_tanh_create_lut(&tanh_lut) != MLI_STATUS_OK || mli_krn_sigm_create_lut(&sigm_lut) != MLI_STATUS_OK) {
        return MLI_STATUS_BAD_FUNC_CFG;
    }

    return MLI_STATUS_OK;
}

//==============================================================
//
//  HAR Smartphone graph based. Layer-by-Layer execution
//
//==============================================================
void har_smartphone_net(const char * debug_ir_root) {
    if (debug_ir_root == NULL) {
        // Version A: without return status checking and profiling wrappers
        //========================================================================================
        
        // Move Input Data to CCM
        //=======================================
        mli_mov_cfg_t mov_cfg;
        mli_mov_cfg_for_copy(&mov_cfg);
        mli_mov_tensor_sync(&input_float, &mov_cfg, &L0_move_out);

        // Convert Input Data
        //=======================================
        mli_hlp_convert_tensor(&L0_move_out, &L0_convert_out);

        // LAYER 1
        //=======================================
        user_fc_on_multiple_samples(&L0_convert_out, &L1_fc_out, &fc1_config);

        // LAYER 2
        //=======================================
        // Clearing the state (eltwise_mul by zero) and run LSTM
        mli_krn_eltwise_mul_fx16(&L2_lstm_cell, &zero_tsr_fx16, &L2_lstm_cell);
        mli_krn_eltwise_mul_fx16(&L2_lstm_prev, &zero_tsr_fx16, &L2_lstm_prev);
        nn_lstm_cell(&L1_fc_out, &L2_lstm_prev, &L2_lstm_wt_in, &L2_lstm_wt_out, &L2_lstm_bias,
                     &tanh_lut, &sigm_lut, &L2_lstm_cfg, &L2_lstm_cell, &L2_lstm_out);

        // LAYER 3
        //=======================================
        // Clearing the state (eltwise_mul by zero) and run LSTM
        mli_krn_eltwise_mul_fx16(&L3_lstm_cell, &zero_tsr_fx16, &L3_lstm_cell);
        mli_krn_eltwise_mul_fx16(&L3_lstm_prev, &zero_tsr_fx16, &L3_lstm_prev);
        user_lstm(&L2_lstm_out, &L3_lstm_prev, &L3_lstm_wt_in, &L3_lstm_wt_out, &L3_lstm_bias,
                  &tanh_lut, &sigm_lut, &L3_lstm_cfg, &L3_lstm_cell, &L3_lstm_out);

        // LAYER 4
        //=======================================
        nn_fully_connected(&L3_lstm_out, &L4_fc_wt, &L4_fc_bias, &fc4_config, &output);
    } else {
        // Version B: Wrapped by service code for profiling and IR results checking purpose
        //========================================================================================

        mli_status ret = MLI_STATUS_OK;
        unsigned mov_cycles = 0;
        unsigned convert_cycles = 0;
        unsigned layer1_cycles = 0;
        unsigned layer2_cycles = 0;
        unsigned layer3_cycles = 0;
        unsigned layer4_cycles = 0;

        // Move Input Data to CCM
        //=======================================
        mli_mov_cfg_t mov_cfg;
        mli_mov_cfg_for_copy(&mov_cfg);
        PROFILE(ret = mli_mov_tensor_sync(&input_float, &mov_cfg, &L0_move_out));
        check_result(debug_ir_root, "ir_mov.idx", &L0_move_out, cycle_cnt, ret);
        mov_cycles += cycle_cnt;

        PROFILE(ret = mli_hlp_convert_tensor(&L0_move_out, &L0_convert_out));
        check_result(debug_ir_root, "ir_in.idx", &L0_convert_out, cycle_cnt, ret);
        convert_cycles += cycle_cnt;

        // LAYER 1
        //=======================================
        PROFILE(ret = user_fc_on_multiple_samples(&L0_convert_out, &L1_fc_out, &fc1_config));
        check_result(debug_ir_root, "ir_relu1.idx", &L1_fc_out, cycle_cnt, ret);
        layer1_cycles += cycle_cnt;

        // LAYER 2
        //=======================================
        // Clearing the state (eltwise_mul by zero) and run LSTM
        PROFILE(mli_krn_eltwise_mul_fx16(&L2_lstm_cell, &zero_tsr_fx16, &L2_lstm_cell)); 
        layer2_cycles += cycle_cnt;
        PROFILE(mli_krn_eltwise_mul_fx16(&L2_lstm_prev, &zero_tsr_fx16, &L2_lstm_prev));
        layer2_cycles += cycle_cnt;
        PROFILE(ret = nn_lstm_cell(&L1_fc_out, &L2_lstm_prev, &L2_lstm_wt_in, &L2_lstm_wt_out, &L2_lstm_bias,
                                   &tanh_lut, &sigm_lut, &L2_lstm_cfg, &L2_lstm_cell, &L2_lstm_out));
        layer2_cycles += cycle_cnt;
        check_result(debug_ir_root, "ir_lstm2.idx", &L2_lstm_out, layer2_cycles, ret);


        // LAYER 3
        //=======================================
        // Clearing the state (eltwise_mul by zero) and run LSTM
        PROFILE(mli_krn_eltwise_mul_fx16(&L3_lstm_cell, &zero_tsr_fx16, &L3_lstm_cell)); 
        layer3_cycles += cycle_cnt;
        PROFILE(mli_krn_eltwise_mul_fx16(&L3_lstm_prev, &zero_tsr_fx16, &L3_lstm_prev));
        layer3_cycles += cycle_cnt;
        PROFILE(user_lstm(&L2_lstm_out, &L3_lstm_prev, &L3_lstm_wt_in, &L3_lstm_wt_out, &L3_lstm_bias,
                          &tanh_lut, &sigm_lut, &L3_lstm_cfg, &L3_lstm_cell, &L3_lstm_out));
        layer3_cycles += cycle_cnt;
        check_result(debug_ir_root, "ir_lstm3.idx", &L3_lstm_out, cycle_cnt, ret);

        // LAYER 4
        //=======================================
        PROFILE(ret = nn_fully_connected(&L3_lstm_out, &L4_fc_wt, &L4_fc_bias, &fc4_config, &output));
        check_result(debug_ir_root, "ir_fc4.idx", &output, cycle_cnt, ret);
        layer4_cycles += cycle_cnt;

        const unsigned total = convert_cycles + layer1_cycles + layer2_cycles + layer3_cycles + layer4_cycles;
        printf("\n\nSummary:\n"
                "\tMovement: %u cycles\n"
                "\tConversion: %u cycles\n"
                "\tLayer1: %u cycles\n"
                "\tLayer2: %u cycles\n"
                "\tLayer3: %u cycles\n"
                "\tLayer4: %u cycles\n"
                "\n\tTotal: %u cycles\n\n",
                mov_cycles,
                convert_cycles,
                layer1_cycles, layer2_cycles,
                layer3_cycles, layer4_cycles,
                total);
    }
}


mli_tensor mli_tsr_make_fx16(int16_t* data, uint32_t len, uint32_t rank,
                             const uint32_t* shape, int8_t frac_bits) {
    mli_tensor ret_val = { 0 };
    if (data == NULL || rank > MLI_MAX_RANK)
        return ret_val;

    uint32_t len_by_shape = 1;
    for (int i = 0; i < rank; ++i)
        len_by_shape *= shape[i];
    if (len < len_by_shape)
        return ret_val;

    ret_val.data.mem.pi16 = data;
    ret_val.data.capacity = len * sizeof(data[0]);
    ret_val.rank = rank;
    uint32_t cur_mem_stride = 1;
    for (int i = 0; i < rank; ++i) {
        ret_val.mem_stride[rank - i - 1] = cur_mem_stride;
        ret_val.shape[i] = shape[i];
        cur_mem_stride *= shape[i];
    }
    ret_val.el_type = MLI_EL_FX_16;
    ret_val.el_params.fx.frac_bits = frac_bits;

    return ret_val;
}


//==============================================================
//  Fully connected on batch: User Implementatioon
//==============================================================
static mli_status user_fc_on_multiple_samples(const mli_tensor* layer_input, mli_tensor* layer_output,
                                              const mli_fully_connected_cfg* cfg) {
    mli_status ret_val = MLI_STATUS_OK;
    mli_tensor fc_in = *layer_input;
    mli_tensor fc_out = *layer_output;
    const mli_sub_tensor_cfg in_iterator = {/*.offset =*/ {0, 0}, /*.size = */{1, layer_input->shape[1]},
                                            /*.sub_tensor_rank =*/1 };
    const mli_sub_tensor_cfg out_iterator = {/*.offset =*/ {0, 0}, /*.size = */{1, layer_output->shape[1]},
                                             /*.sub_tensor_rank =*/1 };

    // Create initial in/out tensors pointing to the first sample from batch
    ret_val = mli_hlp_create_subtensor(layer_input, &in_iterator, &fc_in);
    if (ret_val == MLI_STATUS_OK)
        ret_val = mli_hlp_create_subtensor(layer_output, &out_iterator, &fc_out);

    for (uint32_t batch_idx = 0; batch_idx < layer_input->shape[0]; batch_idx++) {
        if (ret_val != MLI_STATUS_OK)
            break;

        ret_val = nn_fully_connected(&fc_in, &L1_fc_wt, &L1_fc_bias, cfg, &fc_out);

        // Manually update data containers of in/out tensors 
        // to get the next sample from batch
        fc_in.data.mem.D_FIELD += layer_input->mem_stride[0];
        fc_in.data.capacity -= layer_input->mem_stride[0] * sizeof(d_type);
        fc_out.data.mem.D_FIELD += layer_output->mem_stride[0];
        fc_out.data.capacity -= layer_output->mem_stride[0] * sizeof(d_type);
    }
    return ret_val;
}

//==============================================================
//  User Implementatioon of LSTM cell through other MLI Kernels.
//==============================================================

static mli_status user_lstm(
        const mli_tensor* in,
        const mli_tensor* prev_out,
        const mli_tensor* weights_in,
        const mli_tensor* weights_out,
        const mli_tensor* bias,
        const mli_lut* tanh_lut,
        const mli_lut* sigm_lut,
        const mli_rnn_cell_cfg* lstm_cfg,
        mli_tensor* cell,
        mli_tensor* out) {
#if defined(CUSTOM_USER_LSTM_LAYER3)
    mli_status ret_val = MLI_STATUS_OK;

    const int kGates = 4;
    const int kInGateIdx = 0;
    const int kGGateIdx = 1;
    const int kFgtGateIdx = 2;
    const int kOutGateIdx = 3;

    // Parse weights and biases per-gate
    mli_tensor w_in_tensors[4];
    mli_tensor w_out_tensors[4];
    mli_tensor bias_tensors[4];
    {
        mli_sub_tensor_cfg w_in_iterator = {
            /*.offset =*/ {0, 0, 0},
            /*.size = */{1, weights_in->shape[1], weights_in->shape[2]},
            /*.sub_tensor_rank =*/ 2 };
        mli_sub_tensor_cfg w_out_iterator = {
            /*.offset =*/ {0, 0, 0},
            /*.size = */{1, weights_out->shape[1], weights_out->shape[2]},
            /*.sub_tensor_rank =*/ 2 };
        mli_sub_tensor_cfg b_iterator = {
            /*.offset =*/ {0, 0},
            /*.size = */{1, bias->shape[1]},
            /*.sub_tensor_rank =*/ 1 };
        for (int i = 0; i < kGates; ++i) {
            ret_val = mli_hlp_create_subtensor(weights_in, &w_in_iterator, &w_in_tensors[i]);
            if (ret_val == MLI_STATUS_OK)
                ret_val = mli_hlp_create_subtensor(weights_out, &w_out_iterator, &w_out_tensors[i]);
            if (ret_val == MLI_STATUS_OK)
                ret_val = mli_hlp_create_subtensor(bias, &b_iterator, &bias_tensors[i]);

            if (ret_val != MLI_STATUS_OK)
                return ret_val;
            ++w_in_iterator.offset[0];
            ++w_out_iterator.offset[0];
            ++b_iterator.offset[0];
        }
    }
    const mli_tensor* rnn_w_in_gate[2] = { &w_in_tensors[kInGateIdx], &w_out_tensors[kInGateIdx] };
    const mli_tensor* rnn_w_g_gate[2] = { &w_in_tensors[kGGateIdx], &w_out_tensors[kGGateIdx] };
    const mli_tensor* rnn_w_f_gate[2] = { &w_in_tensors[kFgtGateIdx], &w_out_tensors[kFgtGateIdx] };
    const mli_tensor* rnn_w_out_gate[2] = { &w_in_tensors[kOutGateIdx], &w_out_tensors[kOutGateIdx] };

    const uint32_t gate_rank = 1;
    const int16_t gate_len = bias->shape[1];
    const uint32_t* gate_shape = &bias->shape[1];
    const int8_t gate_frac_bits = (sizeof(d_type) * 8) - 1 - 3;
    const uint32_t seq_len = in->shape[0];

    mli_tensor step_prev_out = *prev_out;
    mli_tensor step_in = { 0 };  // To be fiiled in the loop
    mli_tensor step_out = { 0 }; // To be fiiled later depending on mode
    const mli_tensor* rnn_in[2] = { &step_in, &step_prev_out };
    const mli_rnn_dense_cfg rnn_cfg = { sizeof(rnn_in) / sizeof(rnn_in[0]) }; // Assume 2 inputs
    mli_sub_tensor_cfg in_iterator = {/*.offset =*/ {0, 0}, /*.size = */{1, in->shape[1]},
                                      /*.sub_tensor_rank =*/1 };
    mli_sub_tensor_cfg out_iterator = {/*.offset =*/ {0, 0}, /*.size = */{1, out->shape[1]},
                                       /*.sub_tensor_rank =*/1 };

    if (lstm_cfg->results == RNN_OUT_LAST) 
        ret_val = mli_hlp_create_subtensor(out, &out_iterator, &step_out);

    for (uint32_t  sample_idx = 0; sample_idx < seq_len; sample_idx++) {
        if (ret_val != MLI_STATUS_OK)
            break;

        // Prepare step: Constructing current in\out and all gates tensors 
        if (lstm_cfg->direction == RNN_DIR_FORWARD)
            in_iterator.offset[0] =  sample_idx;
        else
            in_iterator.offset[0] = seq_len - sample_idx - 1;
        ret_val = mli_hlp_create_subtensor(in, &in_iterator, &step_in);
        if (ret_val == MLI_STATUS_OK && lstm_cfg->results == RNN_OUT_ALL) {
            out_iterator.offset[0] = sample_idx;
            ret_val = mli_hlp_create_subtensor(out, &out_iterator, &step_out);
        }

        int16_t* gate_data = lstm_cfg->scratch_data.mem.pi16;
        mli_tensor i_gate = mli_tsr_make_fx16(gate_data, gate_len, gate_rank, gate_shape, gate_frac_bits);
        gate_data += gate_len;
        mli_tensor g_gate = mli_tsr_make_fx16(gate_data, gate_len, gate_rank, gate_shape, gate_frac_bits);
        gate_data += gate_len;
        mli_tensor f_gate = mli_tsr_make_fx16(gate_data, gate_len, gate_rank, gate_shape, gate_frac_bits);
        gate_data += gate_len;
        mli_tensor o_gate = mli_tsr_make_fx16(gate_data, gate_len, gate_rank, gate_shape, gate_frac_bits);

        //Step 1: Fully connected
        if (MLI_STATUS_OK == ret_val)
            ret_val = rnn_dense(rnn_in, rnn_w_in_gate, &bias_tensors[kInGateIdx], &rnn_cfg, &i_gate);
        if (MLI_STATUS_OK == ret_val)
            ret_val = rnn_dense(rnn_in, rnn_w_g_gate, &bias_tensors[kGGateIdx], &rnn_cfg, &g_gate);
        if (MLI_STATUS_OK == ret_val)
            ret_val = rnn_dense(rnn_in, rnn_w_f_gate, &bias_tensors[kFgtGateIdx], &rnn_cfg, &f_gate);
        if (MLI_STATUS_OK == ret_val)
            ret_val = rnn_dense(rnn_in, rnn_w_out_gate, &bias_tensors[kOutGateIdx], &rnn_cfg, &o_gate);

        //Step 1: Applying non-linearity
        if (MLI_STATUS_OK == ret_val)
            ret_val = nn_sigm(&i_gate, sigm_lut, &i_gate);
        if (MLI_STATUS_OK == ret_val)
            ret_val = nn_tanh(&g_gate, tanh_lut, &g_gate);
        if (MLI_STATUS_OK == ret_val)
            ret_val = nn_sigm(&f_gate, sigm_lut, &f_gate);
        if (MLI_STATUS_OK == ret_val)
            ret_val = nn_sigm(&o_gate, sigm_lut, &o_gate);

        // Step 3: Pointwise operations
        if (MLI_STATUS_OK == ret_val)
            ret_val = nn_eltwise_mul(&f_gate, cell, cell);
        if (MLI_STATUS_OK == ret_val)
            ret_val = nn_eltwise_mul(&i_gate, &g_gate, &g_gate);
        if (MLI_STATUS_OK == ret_val)
            ret_val = nn_eltwise_add(cell, &g_gate, cell);

        // Step 4: Calculate output for current step 
        if (MLI_STATUS_OK == ret_val)
            ret_val = nn_tanh(cell, tanh_lut, &g_gate);
        if (MLI_STATUS_OK == ret_val)
            ret_val = nn_eltwise_mul(&g_gate, &o_gate, &step_out);
        step_prev_out = step_out;
    }

    return ret_val;
#else
    // The whole function might be replaced with MLI function
    return nn_lstm_cell(in, prev_out, weights_in, weights_out, bias, tanh_lut, sigm_lut, lstm_cfg, cell, out);
#endif
}


//==============================================================
//  Checking kernel result. Debug function
//==============================================================
static void check_result(const char * ir_root, const char * ref_file, mli_tensor *pred_tsr,
                         unsigned cycles, mli_status ret_code) {
    if (ret_code != MLI_STATUS_OK) {
        printf("ERROR: MLI Code for %s (%d) is not OK\n", ref_file, ret_code);
        assert(0);
    }

    if (ir_root != NULL) {
        ref_to_pred_output err;
        test_status test_result = measure_ref_to_pred(ir_root, ref_file, *pred_tsr, &err);
        if (test_result == TEST_PASSED) {
            printf("%s: \n\tS/N=%-10.1f (%-4.1f db)\n\t%u cycles\n",
                    ref_file,
                    err.ref_vec_length / err.noise_vec_length,
                    err.ref_to_noise_snr,
                    cycles);
        }
        else if (test_result == TEST_FAILED) {
            printf("ERROR: Test suit returns FAILD code for %s\n", ref_file);
            assert(0);
        }
        else
            printf("%s(w/o IR check):\t%u cycles\n", ref_file, cycles);
    }
}

//========================================================================================
//  MLI Functions wrappers: Kernels w/o weights
//========================================================================================
static inline mli_status nn_fully_connected(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_fully_connected_cfg* cfg,
        mli_tensor* out) {
#if (MODEL_BIT_DEPTH == MODEL_FX_16)
    return mli_krn_fully_connected_fx16(in, weights, bias, cfg, out);
#else //MODEL_BIT_DEPTH == MODEL_FX_8W16D
    return mli_krn_fully_connected_fx16_fx8_fx8(in, weights, bias, cfg, out);
#endif
}

static inline mli_status nn_lstm_cell(
        const mli_tensor* in,
        const mli_tensor* prev_out,
        const mli_tensor* weights_in,
        const mli_tensor* weights_out,
        const mli_tensor* bias,
        const mli_lut* tanh_lut,
        const mli_lut* sigm_lut,
        const mli_rnn_cell_cfg* cfg,
        mli_tensor* cell,
        mli_tensor* out) {
#if (MODEL_BIT_DEPTH == MODEL_FX_16)
    return mli_krn_lstm_cell_fx16(in, prev_out, weights_in, weights_out, bias, 
                                  tanh_lut, sigm_lut, cfg, cell, out);
#else //MODEL_BIT_DEPTH == MODEL_FX_8W16D
    return mli_krn_lstm_cell_fx16_fx8_fx8(in, prev_out, weights_in, weights_out, bias, 
                                          tanh_lut, sigm_lut, cfg, cell, out);
#endif
}


// The following layers are used only in custom user LSTM.
//========================================================================================
#if defined(CUSTOM_USER_LSTM_LAYER3)
static inline mli_status rnn_dense(
        const mli_tensor** in,
        const mli_tensor** weights,
        const mli_tensor* bias,
        const mli_rnn_dense_cfg* cfg,
        mli_tensor* out) {
#if (MODEL_BIT_DEPTH == MODEL_FX_16)
    return mli_krn_rnn_dense_fx16(in, weights, bias, cfg, out);
#else //MODEL_BIT_DEPTH == MODEL_FX_8W16D
    return mli_krn_rnn_dense_fx16_fx8_fx8(in, weights, bias, cfg, out);
#endif
}

static inline mli_status nn_sigm(const mli_tensor* in, const mli_lut* lut, mli_tensor* out) {
    return mli_krn_sigm_fx16(in, lut, out);
}

static inline mli_status nn_tanh(const mli_tensor* in, const mli_lut* lut, mli_tensor* out) {
    return mli_krn_tanh_fx16(in, lut, out);
}

static inline mli_status nn_eltwise_mul(const mli_tensor* in1, const mli_tensor* in2, mli_tensor* out) {
    return mli_krn_eltwise_mul_fx16(in1, in2, out);
}

static inline mli_status nn_eltwise_add(const mli_tensor* in1, const mli_tensor* in2, mli_tensor* out) {
    return mli_krn_eltwise_add_fx16(in1, in2, out);
}
#endif //CUSTOM_USER_LSTM_LAYER3

