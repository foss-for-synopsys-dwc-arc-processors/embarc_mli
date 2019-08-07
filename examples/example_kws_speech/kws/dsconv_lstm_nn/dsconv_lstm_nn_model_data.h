/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _DSCONV_LSTM_NN_MODEL_DATA_H_
#define _DSCONV_LSTM_NN_MODEL_DATA_H_

#include <mli_api.h>

namespace mli_kws {

// Constant params model was trained for: input audio
constexpr int kAudioSampleRate = 16000;
constexpr int kAudioFrameLength = 480;
constexpr int kAudioFrameStride = 240;
constexpr int kAudioSubframesInFrame = kAudioFrameLength / kAudioFrameStride;

// Constant params model was trained for: feature extractor params
constexpr int kFbankFraqBits = 3;
constexpr int kFbankNumBins = 13;
constexpr int kFbankLowFreq = 20;
constexpr int kFbankHighFreq = 7600;

// Constant params model was trained for: model IO size
constexpr int kFeatureVectorsForInference = 65;
constexpr int kOutputClassesNum = 12;
constexpr int kSilenceID = 0;

// ClassID-to-STR converter
inline const char* dsconv_lstm_model_id_to_str(int id) {
    switch (id) {
        case  0: return "_silence_";
        case  1: return "_unknown_";
        case  2: return "yes";
        case  3: return "no";
        case  4: return "up";
        case  5: return "down";
        case  6: return "left";
        case  7: return "right";
        case  8: return "on";
        case  9: return "off";
        case 10: return "stop";
        case 11: return "go";
        default: return "<UNKN_ID>";
    }
    return "<UNKN_ID>";
}
// DSCONV_LSTM: structure for all model MLI tensors and configs
struct dsconv_lstm_model_data {

    mli_tensor  L1_conv_wt;
    mli_tensor  L1_conv_bias;
    mli_tensor  L2a_conv_dw_wt;
    mli_tensor  L2a_conv_dw_bias;
    mli_tensor  L2b_conv_pw_wt;
    mli_tensor  L2b_conv_pw_bias;
    mli_tensor  L3a_conv_dw_wt;
    mli_tensor  L3a_conv_dw_bias;
    mli_tensor  L3b_conv_pw_wt;
    mli_tensor  L3b_conv_pw_bias;
    mli_tensor  L4a_conv_dw_wt;
    mli_tensor  L4a_conv_dw_bias;
    mli_tensor  L4b_conv_pw_wt;
    mli_tensor  L4b_conv_pw_bias;
    mli_tensor  L5_lstm_wt;
    mli_tensor  L5_lstm_bias;
    mli_tensor  L6_fc_wt;
    mli_tensor  L6_fc_bias;

    mli_tensor      leaky_relu_slope_coeff;
    mli_conv2d_cfg  L1_conv_cfg;
    mli_conv2d_cfg  depthw_conv_cfg;
    mli_conv2d_cfg  pointw_conv_cfg;
    mli_pool_cfg    avg_pool_cfg;
    mli_permute_cfg permute_chw2hwc_cfg;
    mli_rnn_mode    lstm_mode;
    mli_rnn_out_activation lstm_act;

    uint8_t L1_conv_out_fraq;
    uint8_t L2a_conv_dw_out_fraq;
    uint8_t L2b_conv_pw_out_fraq;
    uint8_t L3a_conv_dw_out_fraq;
    uint8_t L3b_conv_pw_out_fraq;
    uint8_t L4a_conv_dw_out_fraq;
    uint8_t L4b_conv_pw_out_fraq;
    uint8_t L5_lstm_cell_fraq;
    uint8_t L6_fc_out_fraq;
};

// Total size of all DSCONV_LSTM trained coefficients
extern const uint32_t kDsconvLstmModelCoeffTotalSize;

// Global structure instance with all required DSCONV_LSTM tensors and parameters
extern const dsconv_lstm_model_data kDsconvLstmModelStruct;

}

#endif // _DSCONV_LSTM_NN_MODEL_DATA_H_
