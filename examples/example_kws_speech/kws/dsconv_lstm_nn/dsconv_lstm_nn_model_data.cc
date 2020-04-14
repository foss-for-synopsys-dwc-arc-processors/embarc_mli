/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "dsconv_lstm_nn_model_data.h"

namespace mli_kws {

//======================================================
//
// Quantized model parameters (statically allocated)
//
//======================================================
#include "dsconv_lstm_nn_coefficients.inc"

#pragma Data(".mli_model")
static const int8_t  L1_conv_wt_buf[] = DSCONV_LSTM_CONV1_W;
static const int8_t  L1_conv_bias_buf[] = DSCONV_LSTM_CONV1_B;

static const int8_t  L2a_conv_dw_wt_buf[] = DSCONV_LSTM_DW2_W;
static const int8_t  L2a_conv_dw_bias_buf[] = DSCONV_LSTM_DW2_B;

static const int8_t  L2b_conv_pw_wt_buf[] = DSCONV_LSTM_PW2_W;
static const int8_t  L2b_conv_pw_bias_buf[] = DSCONV_LSTM_PW2_B;

static const int8_t  L3a_conv_dw_wt_buf[] = DSCONV_LSTM_DW3_W;
static const int8_t  L3a_conv_dw_bias_buf[] = DSCONV_LSTM_DW3_B;

static const int8_t  L3b_conv_pw_wt_buf[] = DSCONV_LSTM_PW3_W;
static const int8_t  L3b_conv_pw_bias_buf[] = DSCONV_LSTM_PW3_B;

static const int8_t  L4a_conv_dw_wt_buf[] = DSCONV_LSTM_DW4_W;
static const int8_t  L4a_conv_dw_bias_buf[] = DSCONV_LSTM_DW4_B;

static const int8_t  L4b_conv_pw_wt_buf[] = DSCONV_LSTM_PW4_W;
static const int8_t  L4b_conv_pw_bias_buf[] = DSCONV_LSTM_PW4_B;

static const int8_t  L5_lstm_wt_buf[] = DSCONV_LSTM_LSTM5_W;
static const int8_t  L5_lstm_bias_buf[] = DSCONV_LSTM_LSTM5_B;

static const int8_t  L6_fc_wt_buf[] = DSCONV_LSTM_FC6_W;
static const int8_t  L6_fc_bias_buf[] = DSCONV_LSTM_FC6_B;
#pragma Data()

const uint32_t kDsconvLstmModelCoeffTotalSize = 
    sizeof(L1_conv_wt_buf) + sizeof(L1_conv_bias_buf) + 
    sizeof(L2a_conv_dw_wt_buf) + sizeof(L2a_conv_dw_bias_buf) + 
    sizeof(L2b_conv_pw_wt_buf) + sizeof(L2b_conv_pw_bias_buf) + 
    sizeof(L3a_conv_dw_wt_buf) + sizeof(L3a_conv_dw_bias_buf) + 
    sizeof(L3b_conv_pw_wt_buf) + sizeof(L3b_conv_pw_bias_buf) + 
    sizeof(L4a_conv_dw_wt_buf) + sizeof(L4a_conv_dw_bias_buf) + 
    sizeof(L4b_conv_pw_wt_buf) + sizeof(L4b_conv_pw_bias_buf) + 
    sizeof(L5_lstm_wt_buf) + sizeof(L5_lstm_bias_buf) + 
    sizeof(L6_fc_wt_buf) + sizeof(L6_fc_bias_buf);

//======================================================
//
//  Model description and configuration
//
//======================================================

const dsconv_lstm_model_data kDsconvLstmModelStruct = {
    // Layer 1: Conv2D related tensors
    //===================================
    .L1_conv_wt = {
        .data = (void *)L1_conv_wt_buf,
        .capacity = sizeof(L1_conv_wt_buf),
        .shape = {64,1,3,3},
        .rank = 4,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 7,
    },
    .L1_conv_bias = {
        .data = (void *)L1_conv_bias_buf,
        .capacity = sizeof(L1_conv_bias_buf),
        .shape = {64},
        .rank = 1,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 6,
    },
    .L1_conv_out_fraq = 3,

    // Layer 2: Depthwise Separable Conv 2D related tensors
    //======================================================
    .L2a_conv_dw_wt = {
        .data = (void *)L2a_conv_dw_wt_buf,
        .capacity = sizeof(L2a_conv_dw_wt_buf),
        .shape = {64,1,3,3},
        .rank = 4,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 5,
    },
    .L2a_conv_dw_bias = {
        .data = (void *)L2a_conv_dw_bias_buf,
        .capacity = sizeof(L2a_conv_dw_bias_buf),
        .shape = {64},
        .rank = 1,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 7,
    },
    .L2a_conv_dw_out_fraq = 2,

    .L2b_conv_pw_wt = {
        .data = (void *)L2b_conv_pw_wt_buf,
        .capacity = sizeof(L2b_conv_pw_wt_buf),
        .shape = {64,64,1,1},
        .rank = 4,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 7,
    },
    .L2b_conv_pw_bias = {
        .data = (void *)L2b_conv_pw_bias_buf,
        .capacity = sizeof(L2b_conv_pw_bias_buf),
        .shape = {64},
        .rank = 1,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 7,
    },
    .L2b_conv_pw_out_fraq = 2,

    // Layer 3: Depthwise Separable Conv 2D related tensors
    //======================================================
    .L3a_conv_dw_wt = {
        .data = (void *)L3a_conv_dw_wt_buf,
        .capacity = sizeof(L3a_conv_dw_wt_buf),
        .shape = {64,1,3,3},
        .rank = 4,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 5,
    },
    .L3a_conv_dw_bias = {
        .data = (void *)L3a_conv_dw_bias_buf,
        .capacity = sizeof(L3a_conv_dw_bias_buf),
        .shape = {64},
        .rank = 1,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 7,
    },
    .L3a_conv_dw_out_fraq = 2,

    .L3b_conv_pw_wt = {
        .data = (void *)L3b_conv_pw_wt_buf,
        .capacity = sizeof(L3b_conv_pw_wt_buf),
        .shape = {64,64,1,1},
        .rank = 4,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 7,
    },
    .L3b_conv_pw_bias = {
        .data = (void *)L3b_conv_pw_bias_buf,
        .capacity = sizeof(L3b_conv_pw_bias_buf),
        .shape = {64},
        .rank = 1,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 6,
    },
    .L3b_conv_pw_out_fraq = 3,


    // Layer 4: Depthwise Separable Conv 2D related tensors
    //======================================================
    .L4a_conv_dw_wt = {
        .data = (void *)L4a_conv_dw_wt_buf,
        .capacity = sizeof(L4a_conv_dw_wt_buf),
        .shape = {64,1,3,3},
        .rank = 4,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 6,
    },
    .L4a_conv_dw_bias = {
        .data = (void *)L4a_conv_dw_bias_buf,
        .capacity = sizeof(L4a_conv_dw_bias_buf),
        .shape = {64},
        .rank = 1,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 7,
    },
    .L4a_conv_dw_out_fraq = 3,
    
    .L4b_conv_pw_wt = {
        .data = (void *)L4b_conv_pw_wt_buf,
        .capacity = sizeof(L4b_conv_pw_wt_buf),
        .shape = {64,64,1,1},
        .rank = 4,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 7,
    },
    .L4b_conv_pw_bias = {
        .data = (void *)L4b_conv_pw_bias_buf,
        .capacity = sizeof(L4b_conv_pw_bias_buf),
        .shape = {64},
        .rank = 1,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 7,
    },
    .L4b_conv_pw_out_fraq = 3,
    
    // Layer 5: LSTM Layer
    //======================================================
    .L5_lstm_wt = {
        .data = (void *)L5_lstm_wt_buf,
        .capacity = sizeof(L5_lstm_wt_buf),
        .shape = {4,32,96},
        .rank = 3,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 6,
    },
    .L5_lstm_bias = {
        .data = (void *)L5_lstm_bias_buf,
        .capacity = sizeof(L5_lstm_bias_buf),
        .shape = {4, 32},
        .rank = 2,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 6,
    },
    .L5_lstm_cell_fraq = 10,
    
    // Layer 6: Fully connected related tensors
    //======================================================
    .L6_fc_wt = {
        .data = (void *)L6_fc_wt_buf,
        .capacity = sizeof(L6_fc_wt_buf),
        .shape = {12,32},
        .rank = 2,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 5,
    },
    .L6_fc_bias = {
        .data = (void *)L6_fc_bias_buf,
        .capacity = sizeof(L6_fc_bias_buf),
        .shape = {12},
        .rank = 1,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = 7,
    },
    .L6_fc_out_fraq = 10,

    // Configuration objects for layers
    //===============================================
    .leaky_relu_slope_coeff = {
        .data = (void *)(LRELU_SLOPE_COEFF_VAL), // 0.2
        .capacity = 0,
        .shape = {0}, 
        .rank = 0,
        .el_type = MLI_EL_FX_8,
        .el_params.fx.frac_bits = LRELU_SLOPE_COEFF_FRAQ_BITS,
    },
    .L1_conv_cfg = {
        .stride_height = 2, .stride_width = 1,
        .padding_bottom = 0, .padding_top = 0,
        .padding_left = 0, .padding_right = 0,
        .relu.type = MLI_RELU_NONE
    },
    .depthw_conv_cfg = {
        .stride_height = 1, .stride_width = 1,
        .padding_bottom = 0, .padding_top = 0,
        .padding_left = 0, .padding_right = 0,
        .relu.type = MLI_RELU_NONE
    },
    .pointw_conv_cfg = {
        .stride_height = 1, .stride_width = 1,
        .padding_bottom = 0, .padding_top = 0,
        .padding_left = 0, .padding_right = 0,
        .relu.type = MLI_RELU_NONE
    },
    .avg_pool_cfg = {
        .kernel_height = 1, .kernel_width = 5,
        .stride_height = 1, .stride_width = 1,
        .padding_bottom = 0, .padding_top = 0,
        .padding_left = 0, .padding_right = 0
    },
    .permute_chw2hwc_cfg = {
        .perm_dim = {1, 2, 0}
    },
    .lstm_mode = RNN_BATCH_TO_LAST,
    .lstm_act = RNN_ACT_TANH,
};

} //namespace mli_kws 
