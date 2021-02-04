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

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "mli_api.h"

#include "mli_types.h"
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

inline void set_mli_tensor_shape1(mli_tensor* tensor, uint32_t shape0) {
    tensor->rank = 1;
    tensor->shape[0] = shape0;
    tensor->mem_stride[0] = 1;
}

inline void set_mli_tensor_shape2(mli_tensor* tensor, uint32_t shape0, uint32_t shape1) {
    tensor->rank = 2;
    tensor->shape[0] = shape0;
    tensor->shape[1] = shape1;
    tensor->mem_stride[0] = 1 * shape1;
    tensor->mem_stride[1] = 1;
}

inline void set_mli_tensor_shape3(mli_tensor* tensor, uint32_t shape0, uint32_t shape1, uint32_t shape2) {
    tensor->rank = 3;
    tensor->shape[0] = shape0;
    tensor->shape[1] = shape1;
    tensor->shape[2] = shape2;
    tensor->mem_stride[0] = 1 * shape2 * shape1;
    tensor->mem_stride[1] = 1 * shape2;
    tensor->mem_stride[2] = 1;
}

// Intermediate data buffers (enough size for max intermediate results)
//==============================
#define LSTM_CELL_SZ (32)
#define INOUT_BUF_SZ_MOST (138*LSTM_CELL_SZ)
#define LSTM_IR_BUF_SZ (20*LSTM_CELL_SZ)
#define LUT_BUF_SZ (512)

// Despite the name of buf we keep all in/out data
// in the same bank (typically first in operand)
// Weights and lstm memory in the another (typically second input operand)
// 11d has got only 2 separate banks of memory
static d_type  _Y    x_mem_buf[INOUT_BUF_SZ_MOST];
static d_type  _Y    y_mem_buf[INOUT_BUF_SZ_MOST];
static d_type  _Y    lstm_ir_mem_buf[LSTM_IR_BUF_SZ * LSTM_CELL_SZ];
static d_type  _X    lstm_cell_mem_buf[LSTM_CELL_SZ];
static int16_t  _Y   tanh_lut_mem_buf[LUT_BUF_SZ];
static int16_t  _X   sigm_lut_mem_buf[LUT_BUF_SZ];

// Module Input/Output tensors and their's external interface
//============================================================
static mli_tensor input_float = {
    .data = {
    .capacity = sizeof(float) * IN_POINTS,
    .mem = { .pf32 = NULL }
},
    .mem_stride = {9, 1},
    .shape = {128, 9},
    .rank = 2,
    .el_type = MLI_EL_FP_32,
};

static mli_tensor input = {
    .data = {
    .capacity = sizeof(float) * IN_POINTS,
    .mem = { .pf32 = (float *)y_mem_buf }
},
    .mem_stride = {9, 1},
    .shape = {128, 9},
    .rank = 2,
    .el_type = MLI_EL_FP_32,
};

static mli_tensor output = {
    .data = {
        .capacity = sizeof(d_type) * OUT_POINTS,
        .mem = { .D_FIELD = (d_type *)y_mem_buf }
    },
    .mem_stride = {1},
    .shape = {6},
    .rank = 1,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
        .zero_point.mem = { .i16 = 29 },
        .scale.mem = { .i16 = 21838 },
        .dim = -1,
        .scale_frac_bits.mem = { .i8 = 19 }
    }
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

static const mli_fully_connected_cfg default_fc_config = {
    .relu = {
        .type = MLI_RELU_NONE
    }    
};

// Interface variables: Available to user via main model header
//===========================================================
mli_tensor * const har_smartphone_net_input = &input_float;
mli_tensor * const har_smartphone_net_output = &output;

//==============================================================
//  Model description and configuration
//==============================================================
#pragma Data(".mli_data")

// Output and helper tensors
//===============================================
static mli_tensor lstm_ir_tensor = {
    .data = {
        .capacity = sizeof(lstm_ir_mem_buf),
        .mem = { .D_FIELD = (d_type *)lstm_ir_mem_buf }
    },
    .shape = {0, 0, 0, 0},
    .rank = 4,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
    .zero_point.mem = { .i16 = 0 },
    .scale.mem = { .i16 = 1 },
    .dim = -1,
    .scale_frac_bits.mem = { .i8 = 7 }
    }
};

// Layer 0: Convert related data
//===================================
static mli_tensor L0_convert_out = {
    .data = {
        .capacity = sizeof(x_mem_buf),
        .mem = { .D_FIELD = (d_type *)x_mem_buf }
    },
    .mem_stride = {9, 1},
    .shape = {128, 9},
    .rank = 2,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
    .zero_point.mem = { .i16 = -1 },
    .zero_point.capacity = 0,
    .scale.mem = { .i16 = 28262 },
    .dim = -1,
    .scale_frac_bits.mem = { .i8 = 20 },
    }
};

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

static const mli_relu_cfg L1_relu_cfg = {.type = MLI_RELU_GEN};

static mli_tensor L1_fc_out = {
    .data = {
        .capacity = sizeof(y_mem_buf),
        .mem = { .D_FIELD = (d_type *)y_mem_buf }
    },
    .mem_stride = {32, 1},
    .shape = {128, 32},
    .rank = 2,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
    .zero_point.mem = { .i16 = -128 },
    .scale.mem = { .i16 = 16803 },
    .dim = -1,
    .scale_frac_bits.mem = { .i8 = 20 }
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

static mli_tensor L2_lstm_cell = {
    .data = {
        .capacity = sizeof(lstm_cell_mem_buf),
        .mem = { .D_FIELD = (d_type *)lstm_cell_mem_buf }
    },
    .mem_stride = {1},
    .shape = {LSTM_CELL_SZ},
    .rank = 1,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
        .zero_point.mem = { .i16 = -5 },
        .scale.mem = { .i16 = 18552 },
        .dim = -1,
        .scale_frac_bits.mem = { .i8 = 18 }
    }
};

static mli_tensor L2_lstm_prev = {
    .data = {
        .capacity = sizeof(lstm_cell_mem_buf),
        .mem = { .D_FIELD = NULL }
    },
    .mem_stride = {1},
    .shape = {LSTM_CELL_SZ},
    .rank = 1,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
        .zero_point.mem = { .i16 = 0 },
        .scale.mem = { .i16 = 16384 },
        .dim = -1,
        .scale_frac_bits.mem = { .i8 = 21 }
    }
};

static mli_tensor L2_lstm_out = {
    .data = {
        .capacity = sizeof(y_mem_buf),
        .mem = { .D_FIELD = (d_type *)y_mem_buf }
    },
    .mem_stride = {32, 1},
    .shape = {4, 32},
    .rank = 2,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
    .zero_point.mem = { .i16 = 0 },
    .scale.mem = { .i16 = 16384 },
    .dim = -1,
    .scale_frac_bits.mem = { .i8 = 21 }
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

static mli_tensor L3_lstm_cell = {
    .data = {
        .capacity = sizeof(lstm_cell_mem_buf),
        .mem = { .D_FIELD = (d_type *)lstm_cell_mem_buf }
    },
    .mem_stride = {1},
    .shape = {LSTM_CELL_SZ},
    .rank = 1,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
        .zero_point.mem = { .i16 = 4 },
        .scale.mem = { .i16 = 30630 },
        .dim = -1,
        .scale_frac_bits.mem = { .i8 = 19 }
    }
};

static mli_tensor L3_lstm_prev = {
    .data = {
        .capacity = sizeof(lstm_cell_mem_buf),
        .mem = { .D_FIELD = NULL }
    },
    .mem_stride = {1},
    .shape = {LSTM_CELL_SZ},
    .rank = 1,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
        .zero_point.mem = { .i16 = 0 },
        .scale.mem = { .i16 = 16384 },
        .dim = -1,
        .scale_frac_bits.mem = { .i8 = 21 }
    }
};

static mli_tensor L3_lstm_out = {
    .data = {
        .capacity = sizeof(x_mem_buf),
        .mem = { .D_FIELD = (d_type *)x_mem_buf }
    },
    .mem_stride = {32, 1},
    .shape = {1, 32},
    .rank = 2,
    .el_type = MLI_EL_SA_8,
    .el_params.sa = {
    .zero_point.mem = { .i16 = 0 },
    .scale.mem = { .i16 = 16384 },
    .dim = -1,
    .scale_frac_bits.mem = { .i8 = 21 }
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
#pragma Data()

#if defined(CUSTOM_USER_LSTM_LAYER3)
static inline mli_status nn_rnn_cell(
        const mli_tensor ** in,
        const mli_tensor ** weights,
        const mli_tensor * bias,
        const mli_rnn_dense_cfg *cfg,
        mli_tensor *out,
        int gate);
#endif

//==============================================================
//  Declaration of helper functions and user specific kernels
//==============================================================
static mli_status user_fc_on_multiple_samples(
        const mli_tensor *input, 
        mli_tensor *output,
        const mli_relu_cfg *relu_cfg);

static mli_status user_lstm(
        const mli_tensor *in,
        const mli_tensor *prev_out,
        const mli_tensor  *weights_in,
        const mli_tensor  *weights_out,
        const mli_tensor  *bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg *lstm_cfg,
        mli_tensor *lstm_ir_tensor,
        mli_tensor *cell,
        mli_tensor *out);

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
        mli_mov_tensor_sync(&input_float, &mov_cfg, &input);

        // Convert Input Data
        //=======================================
        mli_hlp_convert_tensor(&input, &L0_convert_out);

        // LAYER 1
        //=======================================
        user_fc_on_multiple_samples(&L0_convert_out, &L1_fc_out, &L1_relu_cfg);

        // LAYER 2
        //=======================================
        d_type *cell_ptr = (d_type *)L2_lstm_cell.data.mem.D_FIELD;
        d_type *prev_out_ptr = L2_lstm_prev.data.mem.D_FIELD = (d_type *)L0_convert_out.data.mem.D_FIELD;
        for (int idx = 0; idx < LSTM_CELL_SZ; idx++) {
            prev_out_ptr[idx] = L2_lstm_prev.el_params.sa.zero_point.mem.i16;
            cell_ptr[idx] = L2_lstm_cell.el_params.sa.zero_point.mem.i16;
        }

        mli_krn_lstm_cell_sa8_sa8_sa32(&L1_fc_out, &L2_lstm_prev, &L2_lstm_wt_in, &L2_lstm_wt_out, &L2_lstm_bias,
                        &tanh_lut, &sigm_lut, &L2_lstm_cfg, &L2_lstm_cell, &L2_lstm_out);

        // LAYER 3
        //=======================================
        cell_ptr = (d_type *)L3_lstm_cell.data.mem.D_FIELD;
        prev_out_ptr = L3_lstm_prev.data.mem.D_FIELD = (d_type *)L1_fc_out.data.mem.D_FIELD;
        for (int idx = 0; idx < LSTM_CELL_SZ; idx++) {
            prev_out_ptr[idx] = L3_lstm_prev.el_params.sa.zero_point.mem.i16;
            cell_ptr[idx] = L3_lstm_cell.el_params.sa.zero_point.mem.i16;
        }

        user_lstm(&L2_lstm_out, &L3_lstm_prev, &L3_lstm_wt_in, &L3_lstm_wt_out, &L3_lstm_bias,
                        &tanh_lut, &sigm_lut, &L3_lstm_cfg, &lstm_ir_tensor, &L3_lstm_cell, &L3_lstm_out);

        // LAYER 4
        //=======================================
        mli_krn_fully_connected_sa8_sa8_sa32(&L3_lstm_out, &L4_fc_wt, &L4_fc_bias, &default_fc_config, &output);
    } else {
        // Version A: Wrapped by service code for profiling and IR results checking purpose
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
        PROFILE(ret = mli_mov_tensor_sync(&input_float, &mov_cfg, &input));
        check_result(debug_ir_root, "ir_mov.idx", &input, cycle_cnt, ret);
        mov_cycles += cycle_cnt;

        PROFILE(ret = mli_hlp_convert_tensor(&input, &L0_convert_out));
        check_result(debug_ir_root, "ir_in.idx", &L0_convert_out, cycle_cnt, ret);
        convert_cycles += cycle_cnt;

        // LAYER 1
        //=======================================
        PROFILE(ret = user_fc_on_multiple_samples(&L0_convert_out, &L1_fc_out, &L1_relu_cfg));
        check_result(debug_ir_root, "ir_relu1.idx", &L1_fc_out, cycle_cnt, ret);
        layer1_cycles += cycle_cnt;

        // LAYER 2
        //=======================================
        d_type *cell_ptr = (d_type *)L2_lstm_cell.data.mem.D_FIELD;
        d_type *prev_out_ptr = L2_lstm_prev.data.mem.D_FIELD = (d_type *)L0_convert_out.data.mem.D_FIELD;
        for (int idx = 0; idx < LSTM_CELL_SZ; idx++) {
            prev_out_ptr[idx] = L2_lstm_prev.el_params.sa.zero_point.mem.i16;
            cell_ptr[idx] = L2_lstm_cell.el_params.sa.zero_point.mem.i16;
        }

        PROFILE(ret = mli_krn_lstm_cell_sa8_sa8_sa32(&L1_fc_out, &L2_lstm_prev, &L2_lstm_wt_in, &L2_lstm_wt_out, &L2_lstm_bias,
            &tanh_lut, &sigm_lut, &L2_lstm_cfg, &L2_lstm_cell, &L2_lstm_out));
        
        check_result(debug_ir_root, "ir_lstm2.idx", &L2_lstm_out, cycle_cnt, ret);
        layer2_cycles += cycle_cnt;

        // LAYER 3
        //=======================================
        cell_ptr = (d_type *)L3_lstm_cell.data.mem.D_FIELD;
        prev_out_ptr = L3_lstm_prev.data.mem.D_FIELD = (d_type *)L1_fc_out.data.mem.D_FIELD;
        for (int idx = 0; idx < LSTM_CELL_SZ; idx++) {
            prev_out_ptr[idx] = L3_lstm_prev.el_params.sa.zero_point.mem.i16;
            cell_ptr[idx] = L3_lstm_cell.el_params.sa.zero_point.mem.i16;
        }

        PROFILE(ret = user_lstm(&L2_lstm_out, &L3_lstm_prev,
                &L3_lstm_wt_in, &L3_lstm_wt_out, &L3_lstm_bias, &tanh_lut, &sigm_lut, 
                &L3_lstm_cfg, &lstm_ir_tensor, &L3_lstm_cell, &L3_lstm_out));
        check_result(debug_ir_root, "ir_lstm3.idx", &L3_lstm_out, cycle_cnt, ret);
        layer3_cycles += cycle_cnt;

        // LAYER 4
        //=======================================
        PROFILE(ret = mli_krn_fully_connected_sa8_sa8_sa32(&L3_lstm_out, &L4_fc_wt, &L4_fc_bias, &default_fc_config, &output));
        check_result(debug_ir_root, "ir_fc4.idx", &output, cycle_cnt, ret);
        layer4_cycles += cycle_cnt;

        const unsigned total = mov_cycles + convert_cycles + layer1_cycles + layer2_cycles + layer3_cycles + layer4_cycles;
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
//  Fully connected on batch: User Implementatioon
//==============================================================
static mli_status user_fc_on_multiple_samples(const mli_tensor *layer_input, mli_tensor *layer_output, 
    const mli_relu_cfg *relu_cfg) {
    mli_status ret_val = MLI_STATUS_OK;
    mli_tensor fc1_in = {.rank=1, .shape={0}};
    mli_tensor fc1_out = {
        .data = {
            .capacity = layer_output->data.capacity,
            .mem = { .D_FIELD = layer_output->data.mem.D_FIELD}
        },
        .rank = 2,
        .el_type = layer_input->el_type,
        .el_params = layer_output->el_params
    };

    const mli_fully_connected_cfg cfg = {.relu = *relu_cfg};
    mli_point_to_subtsr_cfg iterator = {.start_coord = {0}, .coord_num = 1, .first_out_dim_size = 1};
    ret_val = mli_hlp_point_to_subtensor(layer_input, &iterator, &fc1_in);
    if (ret_val != MLI_STATUS_OK)
                return ret_val;

    unsigned next_out_add = mli_hlp_count_elem_num(&L1_fc_bias, 0);
    unsigned next_in_add = fc1_in.shape[1];

    for (int batch_idx = 0; batch_idx < layer_input->shape[0]; batch_idx++) {
        ret_val = mli_krn_fully_connected_sa8_sa8_sa32(&fc1_in, &L1_fc_wt, &L1_fc_bias, &cfg, &fc1_out);
        if (ret_val != MLI_STATUS_OK)
            return ret_val;

        fc1_in.data.mem.D_FIELD = (d_type *) fc1_in.data.mem.D_FIELD + next_in_add;
        fc1_out.data.mem.D_FIELD = (d_type *) fc1_out.data.mem.D_FIELD + next_out_add;
        fc1_out.data.capacity -= next_out_add;
    }

    layer_output->rank = 2;
    layer_output->shape[0] = layer_input->shape[0];
    layer_output->shape[1] = fc1_out.shape[0];
    layer_output->el_type = fc1_out.el_type;
    layer_output->el_params = fc1_out.el_params;

    return ret_val;
}

//==============================================================
//  User Implementatioon of LSTM cell through other MLI Kernels.
//==============================================================

static mli_status user_lstm(
        const mli_tensor *in,
        const mli_tensor *prev_out,
        const mli_tensor *weights_in,
        const mli_tensor *weights_out,
        const mli_tensor *bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg *lstm_cfg,
        mli_tensor *lstm_ir_tensor,
        mli_tensor *cell,
        mli_tensor *out) {
#if !defined(CUSTOM_USER_LSTM_LAYER3)
    // Might be replaced with MLI function
    return mli_krn_lstm_cell_sa8_sa8_sa32(in, prev_out, weights_in, weights_out, bias, tanh_lut, sigm_lut, lstm_cfg, cell, out);
#else
    mli_status ret_val = MLI_STATUS_OK;
    int gates = 4;
    mli_rnn_dense_cfg rnn_cfg = {.inputs_num = 2};
    mli_tensor *rnn_prev = prev_out;
    mli_tensor *ir_tensor = lstm_ir_tensor;
    ir_tensor->rank = bias->rank;
    ir_tensor->shape[0] = bias->shape[0];
    ir_tensor->shape[1] = bias->shape[1];
    ir_tensor->mem_stride[0] = bias->shape[1];
    ir_tensor->mem_stride[1] = 1;
    ir_tensor->el_type = in->el_type;

    mli_element_params ir_asym_params;

    ir_asym_params.sa.dim = -1;
    ir_asym_params.sa.scale.mem.i16 = 1;
    ir_asym_params.sa.zero_point.mem.i16 = 0;
    ir_asym_params.sa.scale_frac_bits.mem.i8 = 4;
    ir_asym_params.sa.scale.capacity = ir_asym_params.sa.zero_point.capacity = ir_asym_params.sa.scale_frac_bits.capacity = 0;

    ir_tensor->el_params = ir_asym_params;

    // Various gates to controll info flow.
    mli_tensor in_gate = {{ 0 }}, g_tsr = {{ 0 }}, forget_gate = {{ 0 }}, out_gate = {{ 0 }};

    mli_tensor rnn_out = {{ 0 }};
    rnn_out.data = out->data;
    rnn_out.rank = 2;
    rnn_out.shape[0] = 1;
    rnn_out.shape[1] = LSTM_CELL_SZ;
    rnn_out.mem_stride[0] = LSTM_CELL_SZ;
    rnn_out.mem_stride[1] = 1;
    rnn_out.el_type = in->el_type;

    //Iteration 0: Started outside of main cycle for initialization purpose
    //===============================================================
    //Step 1: Fully connected
    mli_tensor rnn_in = {{ 0 }};
    mli_point_to_subtsr_cfg iterator = {.start_coord = {0}, .coord_num = 1, .first_out_dim_size = 1};
    
    ret_val = mli_hlp_point_to_subtensor(in, &iterator, &rnn_in);
    if (ret_val != MLI_STATUS_OK)
        return ret_val;

    mli_tensor* inputs[2] = {&rnn_in, rnn_prev};
    const mli_tensor* weights[2] = {weights_in, weights_out};

    const int lstm_out_elements = mli_hlp_count_elem_num(prev_out, 0);

    // Init subtensors (current iterators state is suitable for it)
    mli_hlp_point_to_subtensor(ir_tensor, &iterator, &in_gate);
    iterator.start_coord[0]++;
    mli_hlp_point_to_subtensor(ir_tensor, &iterator, &g_tsr);
    iterator.start_coord[0]++;
    mli_hlp_point_to_subtensor(ir_tensor, &iterator, &forget_gate);
    iterator.start_coord[0]++;
    mli_hlp_point_to_subtensor(ir_tensor, &iterator, &out_gate);

    for (int gate = 0; gate < gates; ++gate) {
        ret_val = nn_rnn_cell(inputs, weights, bias, &rnn_cfg, ir_tensor, gate);
        if (ret_val != MLI_STATUS_OK)
            return ret_val;
    }

    ir_tensor->rank = 2;
    ir_tensor->shape[0] = 4;

    rnn_prev = out;
    d_type *out_start = rnn_prev->data.mem.D_FIELD;
    unsigned next_in_add =  mli_hlp_count_elem_num(bias, 1) * mli_hlp_tensor_element_size(&rnn_in);

    // Manual reshape
    in_gate.shape[0] = g_tsr.shape[0] = forget_gate.shape[0] = out_gate.shape[0] =
            mli_hlp_count_elem_num(&in_gate, 0);
    in_gate.rank = g_tsr.rank = forget_gate.rank = out_gate.rank = 1;
    in_gate.mem_stride[0] = g_tsr.mem_stride[0] = forget_gate.mem_stride[0] = out_gate.mem_stride[0] = 1;

    mli_tensor in_gate_input = in_gate;
    mli_tensor g_tsr_input = g_tsr;
    mli_tensor forget_gate_input = forget_gate;
    mli_tensor out_gate_input = out_gate;
    //Iteration 0.3-127: outside of main cycle for initialization purpose
    //===============================================================
    for (int batch_idx = 0; batch_idx < in->shape[0]; batch_idx++) {
        //Step 2: Applying non-linearity
        ret_val = mli_krn_sigm_sa8(&in_gate_input, sigm_lut, &in_gate);
        if (ret_val == MLI_STATUS_OK)
            ret_val = mli_krn_tanh_sa8(&g_tsr_input, tanh_lut, &g_tsr);
        if (ret_val == MLI_STATUS_OK)
            ret_val = mli_krn_sigm_sa8(&forget_gate_input, sigm_lut, &forget_gate);
        if (ret_val == MLI_STATUS_OK)
            ret_val = mli_krn_sigm_sa8(&out_gate_input, sigm_lut, &out_gate);
        if (ret_val != MLI_STATUS_OK)
            return ret_val;

        // Step 3: Pointwise operations
        ret_val = mli_krn_eltwise_mul_sa8(&forget_gate, cell, cell);
        if (ret_val == MLI_STATUS_OK)
            ret_val = mli_krn_eltwise_mul_sa8(&in_gate, &g_tsr, &g_tsr);
        if (ret_val == MLI_STATUS_OK)
            ret_val = mli_krn_eltwise_add_sa8(cell, &g_tsr, cell);
        if (ret_val != MLI_STATUS_OK)
            return ret_val;

        // Step 4: Calculate next output
        mli_tensor temp;
        temp.data = rnn_out.data;
        temp.rank = rnn_out.rank;
        temp.shape[0] = rnn_out.shape[0];
        temp.mem_stride[0] = 1;
        temp.el_type = rnn_out.el_type;
        temp.el_params = out->el_params;
        
        if(lstm_cfg->results == RNN_OUT_LAST) {
            rnn_out.rank = 1;
            rnn_out.shape[0] = bias->shape[1];
            rnn_out.mem_stride[0] = 1;
        }

        ret_val = mli_krn_tanh_sa8(cell, tanh_lut, &rnn_out);

        if (ret_val == MLI_STATUS_OK)
            ret_val = mli_krn_eltwise_mul_sa8(&rnn_out, &out_gate, &temp);
        if (ret_val != MLI_STATUS_OK)
            return ret_val;

        rnn_out.el_params = out->el_params;

        //Next sample: Step 1: Fully connected
        if (batch_idx < in->shape[0]-1) {
            inputs[0]->data.mem.D_FIELD = inputs[0]->data.mem.D_FIELD + next_in_add;
            inputs[1]->data.mem.D_FIELD = out->data.mem.D_FIELD;
            
            if(lstm_cfg->results == RNN_OUT_ALL) {
                out->data.mem.D_FIELD = out->data.mem.D_FIELD + next_in_add;
                rnn_out.data = out->data;
            }

            for (int gate = 0; gate < gates; ++gate) {
                ret_val = nn_rnn_cell(inputs, weights, bias, &rnn_cfg, ir_tensor, gate);
                if (ret_val != MLI_STATUS_OK)
                    return ret_val;
            }

            ir_tensor->rank = 2;
            ir_tensor->shape[0] = 4;

            // Gate tensors point to RNN cell result, but structures have changed due to non-linearity.
            // Restore element params.
            in_gate.el_params = forget_gate.el_params =
                    g_tsr.el_params = out_gate.el_params =
                    ir_tensor->el_params;
        }
    }

    if(lstm_cfg->results == RNN_OUT_ALL) {
        out->rank = 2;
        out->shape[0] = in->shape[0];
        out->shape[1] = out->shape[1];
        out->mem_stride[0] = out->shape[1];
        out->mem_stride[1] = 1;
        out->data.mem.D_FIELD = out_start;
    } else {
        out->rank = 2;
        out->shape[0] = 1;
        out->shape[1] = out->shape[1];
        out->mem_stride[0] = out->shape[1];
        out->mem_stride[1] = 1;
        out->data.mem.D_FIELD = out_start;
    }

    return ret_val;
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

#if defined(CUSTOM_USER_LSTM_LAYER3)
static inline mli_status nn_rnn_cell(
        const mli_tensor ** in,
        const mli_tensor ** weights,
        const mli_tensor * bias,
        const mli_rnn_dense_cfg *cfg,
        mli_tensor *out, 
        int gate) {

    out->rank = 1;
    out->shape[0] = bias->shape[1];
    out->mem_stride[0] = 1; 

    const mli_tensor w_in_part = {
        .data = {
            .capacity = weights[0]->data.capacity,
            .mem = { .W_FIELD = (w_type *)weights[0]->data.mem.W_FIELD + gate * mli_hlp_count_elem_num(weights[0], 1) * sizeof(w_type)}
        },
        .shape = {weights[0]->shape[1], weights[0]->shape[2]},
        .mem_stride = {weights[0]->mem_stride[1], weights[0]->mem_stride[2]},
        .rank = 2,
        .el_type = weights[0]->el_type,
        .el_params.sa = {
            .zero_point = { .capacity = 0, .mem = { .i16 = weights[0]->el_params.sa.zero_point.mem.pi16[0] } },
            .scale = { .capacity = 0, .mem = { .i16 = weights[0]->el_params.sa.scale.mem.pi16[gate] } },
            .dim = -1,
            .scale_frac_bits = { .capacity = 0, .mem = { .i8 = weights[0]->el_params.sa.scale_frac_bits.mem.pi8[gate] } }
        }
    };

    const mli_tensor w_out_part = {
        .data = {
            .capacity = weights[1]->data.capacity,
            .mem = { .W_FIELD = (w_type *)weights[1]->data.mem.W_FIELD + gate * mli_hlp_count_elem_num(weights[1], 1) * sizeof(w_type)}
        },
        .shape = {weights[1]->shape[1], weights[1]->shape[2]},
        .mem_stride = {weights[1]->mem_stride[1], weights[1]->mem_stride[2]},
        .rank = 2,
        .el_type = weights[1]->el_type,
        .el_params.sa = {
            .zero_point = { .capacity = 0, .mem = { .i16 = weights[1]->el_params.sa.zero_point.mem.pi16[0] } },
            .scale = { .capacity = 0, .mem = { .i16 = weights[1]->el_params.sa.scale.mem.pi16[gate] } },
            .dim = -1,
            .scale_frac_bits = { .capacity = 0, .mem = { .i8 = weights[1]->el_params.sa.scale_frac_bits.mem.pi8[gate] } }
        }
    };

    const mli_tensor bias_part = {
        .data = {
            .capacity = bias->data.capacity,
            .mem = { .B_FIELD = (b_type *)bias->data.mem.B_FIELD + gate * mli_hlp_count_elem_num(bias, 1) * sizeof(b_type)}
        },
        .shape = {bias->shape[1]},
        .mem_stride = {bias->mem_stride[1]},
        .rank = 1,
        .el_type = bias->el_type,
        .el_params.sa = {
            .zero_point = { .capacity = 0, .mem = { .i16 = bias->el_params.sa.zero_point.mem.pi16[0] } },
            .scale = { .capacity = 0, .mem = { .i16 = bias->el_params.sa.scale.mem.pi16[gate] } },
            .dim = -1,
            .scale_frac_bits = { .capacity = 0, .mem = { .i8 = bias->el_params.sa.scale_frac_bits.mem.pi8[gate] } }
        }
    };

    const mli_tensor* weights_part[2] = {&w_in_part, &w_out_part};
    out->data.mem.D_FIELD = (d_type *)out->data.mem.D_FIELD + gate * mli_hlp_count_elem_num(bias, 1);
    mli_krn_rnn_dense_sa8_sa8_sa32(in, weights_part, &bias_part, cfg, out);

    return MLI_STATUS_OK;
}
#endif
