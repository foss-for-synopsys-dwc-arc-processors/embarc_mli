/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

/**
 * 
 * SA8 version of HAR LSTM. See *.h for general description
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
#define INOUT_BUF_SZ_MOST (128*9*sizeof(float))
#define INOUT_BUF_SZ_SEC_MOST (128*LSTM_CELL_SZ)
#define LSTM_IR_BUF_SZ (4*LSTM_CELL_SZ)
#define LUT_BUF_SZ (512)

// Despite the name of buf we keep all in/out data
// in the same bank (typically first in operand)
// Weights and lstm memory in the another (typically second input operand)
// 11d has got only 2 separate banks of memory
static d_type  _Y    x_mem_buf[INOUT_BUF_SZ_SEC_MOST];
static d_type  _Y    y_mem_buf[INOUT_BUF_SZ_MOST];
static d_type  _Y    lstm_ir_mem_buf[LSTM_IR_BUF_SZ * LSTM_CELL_SZ];
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
        .mem = { .pf32 = (float *)y_mem_buf }
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
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
        .zero_point = {.capacity = 0, .mem = {.i16 = -1}},
        .scale = {.capacity = 0, .mem = {.i16 = 28262}},
        .dim = -1,
        .scale_frac_bits = {.capacity = 0, .mem = {.i8 = 20 }},
    }
};

static mli_tensor L1_fc_out = {
    .data = {
        .capacity = sizeof(y_mem_buf),
        .mem = {.D_FIELD = (d_type*)y_mem_buf }
    },
    .mem_stride = {32, 1},
    .shape = {128, 32},
    .rank = 2,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
        .zero_point = {.capacity = 0, .mem = {.i16 = -128}},
        .scale = {.capacity = 0, .mem = {.i16 = 16803}},
        .dim = -1,
        .scale_frac_bits = {.capacity = 0, .mem = {.i8 = 20 }},
    }
};

static mli_tensor L2_lstm_cell = {
    .data = {
        .capacity = sizeof(lstm_cell_mem_buf),
        .mem = {.D_FIELD = (d_type*)lstm_cell_mem_buf }
    },
    .mem_stride = {1},
    .shape = {LSTM_CELL_SZ},
    .rank = 1,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
        .zero_point = {.capacity = 0, .mem = {.i16 = -5}},
        .scale = {.capacity = 0, .mem = {.i16 = 18552}},
        .dim = -1,
        .scale_frac_bits = {.capacity = 0, .mem = {.i8 = 18 }},
    }
};

static mli_tensor L2_lstm_prev = {
    .data = {
        .capacity = sizeof(x_mem_buf),
        .mem = {.D_FIELD = (d_type*)x_mem_buf }
    },
    .mem_stride = {1},
    .shape = {LSTM_CELL_SZ},
    .rank = 1,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
        .zero_point = {.capacity = 0, .mem = {.i16 = 0}},
        .scale = {.capacity = 0, .mem = {.i16 = 16384}},
        .dim = -1,
        .scale_frac_bits = {.capacity = 0, .mem = {.i8 = 21 }},
    }
};

static mli_tensor L2_lstm_out = {
    .data = {
        .capacity = sizeof(x_mem_buf),
        .mem = {.D_FIELD = (d_type*)x_mem_buf }
    },
    .mem_stride = {32, 1},
    .shape = {128, 32},
    .rank = 2,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
        .zero_point = {.capacity = 0, .mem = {.i16 = 0}},
        .scale = {.capacity = 0, .mem = {.i16 = 16384}},
        .dim = -1,
        .scale_frac_bits = {.capacity = 0, .mem = {.i8 = 21 }},
    }
};

static mli_tensor L3_lstm_cell = {
    .data = {
        .capacity = sizeof(lstm_cell_mem_buf),
        .mem = {.D_FIELD = (d_type*)lstm_cell_mem_buf }
    },
    .mem_stride = {1},
    .shape = {LSTM_CELL_SZ},
    .rank = 1,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
        .zero_point = {.capacity = 0, .mem = {.i16 = 4}},
        .scale = {.capacity = 0, .mem = {.i16 = 30630}},
        .dim = -1,
        .scale_frac_bits = {.capacity = 0, .mem = {.i8 = 19 }},
    }
};

static mli_tensor L3_lstm_prev = {
    .data = {
        .capacity = sizeof(y_mem_buf),
        .mem = {.D_FIELD = (d_type*)y_mem_buf }
    },
    .mem_stride = {1},
    .shape = {LSTM_CELL_SZ},
    .rank = 1,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
        .zero_point = {.capacity = 0, .mem = {.i16 = 0}},
        .scale = {.capacity = 0, .mem = {.i16 = 16384}},
        .dim = -1,
        .scale_frac_bits = {.capacity = 0, .mem = {.i8 = 21 }},
    }
};

static mli_tensor L3_lstm_out = {
    .data = {
        .capacity = sizeof(y_mem_buf),
        .mem = {.D_FIELD = (d_type*)y_mem_buf }
    },
    .mem_stride = {32, 1},
    .shape = {1, 32},
    .rank = 2,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
        .zero_point = {.capacity = 0, .mem = {.i16 = 0}},
        .scale = {.capacity = 0, .mem = {.i16 = 16384}},
        .dim = -1,
        .scale_frac_bits = {.capacity = 0, .mem = {.i8 = 21 }},
    }
};


static mli_tensor output = {
    .data = {
        .capacity = sizeof(x_mem_buf),
        .mem = { .D_FIELD = (d_type *)x_mem_buf }
    },
    .mem_stride = {1},
    .shape = {6},
    .rank = 1,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
        .zero_point = {.capacity = 0, .mem = {.i16 = 29}},
        .scale = {.capacity = 0, .mem = {.i16 = 21838}},
        .dim = -1,
        .scale_frac_bits = {.capacity = 0, .mem = {.i8 = 19 }},
    }
};

static mli_lut tanh_lut = {
    .data = {
        .capacity = sizeof(tanh_lut_mem_buf),
        .mem = { .pi16 = (int16_t *)tanh_lut_mem_buf}
    },

};

static mli_lut sigm_lut = {
    .data = {
        .capacity = sizeof(sigm_lut_mem_buf),
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

static const mli_tensor zero_tsr_sa8 = {
    .data = {
        .capacity = 0,
        .mem = {.i8 = 0 }
    },
    .el_type = MLI_EL_SA_8,
    .rank = 0,
    .shape = {0},
    .mem_stride = {1},
    .el_params.sa = {
        .dim = -1,
        .zero_point = {.capacity = 0, .mem = {.i16 = 0}},
        .scale = {.capacity = 0, .mem = {.i16 = 1}},
        .scale_frac_bits = {.capacity = 0, .mem = {.i8 = 0}},
    }
};

// Layer 0: Convert related data
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
    .el_params.sa = {
        .zero_point = { .capacity = FC1_B_ELEMENTS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)FC1_W_ZP }},
        .scale = { .capacity = FC1_B_ELEMENTS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)FC1_W_SCALE }},
        .dim = FC1_W_DIM,
        .scale_frac_bits = { .capacity = FC1_B_ELEMENTS * sizeof(int8_t), .mem = { .pi8 = (int8_t *)FC1_W_FRAQ }},
    }
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
    .el_params.sa = {
        .zero_point = { .capacity = FC1_B_ELEMENTS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)FC1_B_ZP }},
        .scale = { .capacity = FC1_B_ELEMENTS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)FC1_B_SCALE }},
        .dim = FC1_B_DIM,
        .scale_frac_bits = { .capacity = FC1_B_ELEMENTS * sizeof(int8_t), .mem = { .pi8 = (int8_t *)FC1_B_FRAQ }},
    }
};

static const mli_fully_connected_cfg fc1_config = {
    .relu = {
        .type = MLI_RELU_GEN
    }
};

static const mli_tensor L2_lstm_wt_in = {
    .data = {
        .capacity = LSTM2_W_IN_ELEMENTS * sizeof(w_type),
        .mem = { .W_FIELD = (w_type *)L2_lstm_wt_in_buf }
    },
    .mem_stride = LSTM2_W_IN_MEM_STRIDE,
    .shape = LSTM2_W_IN_SHAPE,
    .rank = LSTM2_W_IN_RANK,
    .el_type = W_EL_TYPE,
    .el_params.sa = {
        .zero_point = { .capacity = LSTM2_SA_PARAMS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)LSTM2_W_IN_ZP }},
        .scale = { .capacity = LSTM2_SA_PARAMS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)LSTM2_W_IN_SCALE }},
        .dim = LSTM2_W_IN_DIM,
        .scale_frac_bits = { .capacity = LSTM2_SA_PARAMS * sizeof(int8_t), .mem = { .pi8 = (int8_t *)LSTM2_W_IN_FRAQ }},
    }
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
    .el_params.sa = {
        .zero_point = { .capacity = LSTM2_SA_PARAMS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)LSTM2_W_OUT_ZP }},
        .scale = { .capacity = LSTM2_SA_PARAMS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)LSTM2_W_OUT_SCALE }},
        .dim = LSTM2_W_OUT_DIM,
        .scale_frac_bits = { .capacity = LSTM2_SA_PARAMS * sizeof(int8_t), .mem = { .pi8 = (int8_t *)LSTM2_W_OUT_FRAQ }},
    }
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
    .el_params.sa = {
        .zero_point = { .capacity = LSTM2_SA_PARAMS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)LSTM2_B_ZP }},
        .scale = { .capacity = LSTM2_SA_PARAMS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)LSTM2_B_SCALE }},
        .dim = LSTM2_B_DIM,
        .scale_frac_bits = { .capacity = LSTM2_SA_PARAMS * sizeof(int8_t), .mem = { .pi8 = (int8_t *)LSTM2_B_SA_FRAQ }},
    }
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

static const mli_tensor L3_lstm_wt_in = {
    .data = {
        .capacity = LSTM3_W_IN_ELEMENTS * sizeof(w_type),
        .mem = { .W_FIELD = (w_type *)L3_lstm_wt_in_buf }
    },
    .mem_stride = LSTM3_W_IN_MEM_STRIDE,
    .shape = LSTM3_W_IN_SHAPE,
    .rank = LSTM3_W_IN_RANK,
    .el_type = W_EL_TYPE,
    .el_params.sa = {
        .zero_point = { .capacity = LSTM3_SA_PARAMS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)LSTM3_W_IN_ZP }},
        .scale = { .capacity = LSTM3_SA_PARAMS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)LSTM3_W_IN_SCALE }},
        .dim = LSTM3_W_IN_DIM,
        .scale_frac_bits = { .capacity = LSTM3_SA_PARAMS * sizeof(int8_t), .mem = { .pi8 = (int8_t *)LSTM3_W_IN_FRAQ }},
    }
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
    .el_params.sa = {
        .zero_point = { .capacity = LSTM3_SA_PARAMS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)LSTM3_W_OUT_ZP }},
        .scale = { .capacity = LSTM3_SA_PARAMS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)LSTM3_W_OUT_SCALE }},
        .dim = LSTM3_W_OUT_DIM,
        .scale_frac_bits = { .capacity = LSTM3_SA_PARAMS * sizeof(int8_t), .mem = { .pi8 = (int8_t *)LSTM3_W_OUT_FRAQ }},
    }
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
    .el_params.sa = {
        .zero_point = { .capacity = LSTM3_SA_PARAMS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)LSTM3_B_ZP }},
        .scale = { .capacity = LSTM3_SA_PARAMS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)LSTM3_B_SCALE }},
        .dim = LSTM3_B_DIM,
        .scale_frac_bits = { .capacity = LSTM3_SA_PARAMS * sizeof(int8_t), .mem = { .pi8 = (int8_t *)LSTM3_B_FRAQ }},
    }
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


// Layer 4: Fully Connected related data
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
    .el_params.sa = {
        .zero_point = { .capacity = FC4_B_ELEMENTS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)FC4_W_ZP }},
        .scale = { .capacity = FC4_B_ELEMENTS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)FC4_W_SCALE }},
        .dim = FC4_W_DIM,
        .scale_frac_bits = { .capacity = FC4_B_ELEMENTS * sizeof(int8_t), .mem = { .pi8 = (int8_t *)FC4_W_FRAQ }},
    }
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
    .el_params.sa = {
        .zero_point = { .capacity = FC4_B_ELEMENTS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)FC4_B_ZP }},
        .scale = { .capacity = FC4_B_ELEMENTS * sizeof(int16_t), .mem = { .pi16 = (int16_t *)FC4_B_SCALE }},
        .dim = FC4_B_DIM,
        .scale_frac_bits = { .capacity = FC4_B_ELEMENTS * sizeof(int8_t), .mem = { .pi8 = (int8_t *)FC4_B_FRAQ }},
    }
};

static const mli_fully_connected_cfg fc4_config = {
    .relu = {
        .type = MLI_RELU_NONE
    }
};

//==============================================================
//  Declaration of helper functions and user specific kernels
//==============================================================
static mli_status user_fc_on_multiple_samples(
        const mli_tensor *input, 
        mli_tensor *output,
        const mli_fully_connected_cfg *cfg);

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
        mli_krn_eltwise_mul_sa8(&L2_lstm_cell, &zero_tsr_sa8, &L2_lstm_cell);
        mli_krn_eltwise_mul_sa8(&L2_lstm_prev, &zero_tsr_sa8, &L2_lstm_prev);
        mli_krn_lstm_cell_sa8_sa8_sa32(&L1_fc_out, &L2_lstm_prev, &L2_lstm_wt_in, &L2_lstm_wt_out, &L2_lstm_bias,
                                       &tanh_lut, &sigm_lut, &L2_lstm_cfg, &L2_lstm_cell, &L2_lstm_out);

        // LAYER 3
        //=======================================
        mli_krn_eltwise_mul_sa8(&L3_lstm_cell, &zero_tsr_sa8, &L3_lstm_cell);
        mli_krn_eltwise_mul_sa8(&L3_lstm_prev, &zero_tsr_sa8, &L3_lstm_prev);
        mli_krn_lstm_cell_sa8_sa8_sa32(&L2_lstm_out, &L3_lstm_prev, &L3_lstm_wt_in, &L3_lstm_wt_out, &L3_lstm_bias,
                                       &tanh_lut, &sigm_lut, &L3_lstm_cfg, &L3_lstm_cell, &L3_lstm_out);

        // LAYER 4
        //=======================================
        mli_krn_fully_connected_sa8_sa8_sa32(&L3_lstm_out, &L4_fc_wt, &L4_fc_bias, &fc4_config, &output);
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
        PROFILE(mli_krn_eltwise_mul_sa8(&L2_lstm_cell, &zero_tsr_sa8, &L2_lstm_cell));
        layer2_cycles += cycle_cnt;
        PROFILE(mli_krn_eltwise_mul_sa8(&L2_lstm_prev, &zero_tsr_sa8, &L2_lstm_prev));
        layer2_cycles += cycle_cnt;
        PROFILE(ret = mli_krn_lstm_cell_sa8_sa8_sa32(&L1_fc_out, &L2_lstm_prev, &L2_lstm_wt_in, &L2_lstm_wt_out,
                                                     &L2_lstm_bias, &tanh_lut, &sigm_lut, &L2_lstm_cfg, &L2_lstm_cell,
                                                     &L2_lstm_out));
        layer2_cycles += cycle_cnt;
        check_result(debug_ir_root, "ir_lstm2.idx", &L2_lstm_out, cycle_cnt, ret);


        // LAYER 3
        //=======================================
        PROFILE(mli_krn_eltwise_mul_sa8(&L3_lstm_cell, &zero_tsr_sa8, &L3_lstm_cell));
        layer3_cycles += cycle_cnt;
        PROFILE(mli_krn_eltwise_mul_sa8(&L3_lstm_prev, &zero_tsr_sa8, &L3_lstm_prev));
        layer3_cycles += cycle_cnt;
        PROFILE(ret = mli_krn_lstm_cell_sa8_sa8_sa32(&L2_lstm_out, &L3_lstm_prev, &L3_lstm_wt_in, &L3_lstm_wt_out,
                                                     &L3_lstm_bias, &tanh_lut, &sigm_lut, &L3_lstm_cfg, &L3_lstm_cell,
                                                     &L3_lstm_out));
        layer3_cycles += cycle_cnt;
        check_result(debug_ir_root, "ir_lstm3.idx", &L3_lstm_out, cycle_cnt, ret);


        // LAYER 4
        //=======================================
        PROFILE(ret = mli_krn_fully_connected_sa8_sa8_sa32(&L3_lstm_out, &L4_fc_wt, &L4_fc_bias, &fc4_config, &output));
        check_result(debug_ir_root, "ir_fc4.idx", &output, cycle_cnt, ret);
        layer4_cycles += cycle_cnt;

        const unsigned total = mov_cycles + convert_cycles + layer1_cycles + 
                               layer2_cycles + layer3_cycles + layer4_cycles;

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

//==============================================================
//  Fully connected on batch: User Implementation
//==============================================================
static mli_status user_fc_on_multiple_samples(const mli_tensor *layer_input, mli_tensor *layer_output, 
                                              const mli_fully_connected_cfg *cfg) {
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
        
        ret_val = mli_krn_fully_connected_sa8_sa8_sa32(&fc_in, &L1_fc_wt, &L1_fc_bias, cfg, &fc_out);
        
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

