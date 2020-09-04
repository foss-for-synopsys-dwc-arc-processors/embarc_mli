/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "dsconv_lstm_nn_model_impl.h"
#include "dsconv_lstm_nn_model_data.h"
#include "../kws_factory.h"

#include "tensor_transform.h"

using namespace mli_kws;
using au_fex::fbank_features;

// Init static member with a model description global structure 
const dsconv_lstm_model_data* const kws_dsconv_lstm_nn::model = &kDsconvLstmModelStruct;

// Constant sizes of buffers
constexpr int kModelIRsizeBytesMax = 22528;
constexpr int kModelIRsizeBytesMaxHalf = kModelIRsizeBytesMax / 2;

constexpr int kModelIRsizeBytesSecondMax = 17280;
constexpr int kModelIRsizeBytesSecondMaxHalf = kModelIRsizeBytesSecondMax / 2;

constexpr int kFexIRsizeFloatsMax = 1024;

constexpr int kSubframesInSamplesBuf = 4;
constexpr int kFeatureVectorsStridePerInf = 16;

// Fast temp mem Bank X: Memory map used by module during the single processing.
// Pointer to exact memory region is to be provided by application to process(..) function
struct kws_dsconv_lstm_nn::fastmem_x_map {
    union nn_t {
        struct fex_phase_t {
            float ir_buf_x[kFexIRsizeFloatsMax];
        } fex_phase;

        struct cnn_phase_t {
            int8_t ir_data_x[kModelIRsizeBytesSecondMax];
        } cnn_phase ;

        struct rnn_phase_t {
            int8_t ir_data_x[kModelIRsizeBytesSecondMaxHalf];
            int8_t lstm_ir_data[kModelIRsizeBytesSecondMaxHalf];
        } rnn_phase;
    } nn;
};

// Fast temp mem Bank Y: Memory map used by module during the single processing.
// Pointer to exact memory region is to be provided by application to process(..) function.
struct kws_dsconv_lstm_nn::fastmem_y_map {
    union nn_t {
        struct fex_phase_t {
            float ir_buf_y[kFexIRsizeFloatsMax];
            float out_features[kFbankNumBins];
        } fex_phase;

        struct cnn_phase_t{
            int8_t ir_data_y[kModelIRsizeBytesMax];
        } cnn_phase;

        struct rnn_phase_t{
            int8_t ir_data_y[kModelIRsizeBytesMaxHalf];
            int8_t lstm_cell_data[kModelIRsizeBytesMaxHalf];
        } rnn_phase;
    } nn;
};

// Fast state mem Bank X: Memory map used by module during it's lifetime.
// Pointer to exact memory region is to be provided by application in construction time.
struct kws_dsconv_lstm_nn::state_fastmem_x_map {
    sample_t samples_buf[kSubframesInSamplesBuf][kAudioFrameStride];
    int8_t features_buf[kFeatureVectorsForInference][kFbankNumBins];
    uint32_t fex_state_len;
    float fex_state_start;
};

//==============================================================
//
//
// Factory related methods
//
//
//==============================================================

// Provide information on the module to the user
//==============================================================
kws_info dsconv_lstm_factory::info() {
    const fbank_features::params fex_config {kFbankNumBins,kAudioFrameLength, kAudioSampleRate,
                                             kFbankLowFreq, kFbankHighFreq};
    fbank_features::info aufex_info = fbank_features::get_info(fex_config);
    return kws_info {
        /* .input_samples_num = */    kAudioFrameStride,
        /* .output_values_num = */    kOutputClassesNum,   
        /* .timestamp_duration_ms =*/ ((kAudioFrameStride) * 1000 / kAudioSampleRate),
        /* .state_fastmem_a_sz = */   sizeof(kws_dsconv_lstm_nn::state_fastmem_x_map)
                                        + aufex_info.state_fastbuf_a_len * sizeof(float),
        /* .temp_fastmem_a_sz = */    sizeof(kws_dsconv_lstm_nn::fastmem_x_map),
        /* .temp_fastmem_b_sz = */    sizeof(kws_dsconv_lstm_nn::fastmem_y_map),
        /* .coeff_fastmem_sz = */     kDsconvLstmModelCoeffTotalSize,
        /* .dynamic_mem_sz = */       sizeof(kws_dsconv_lstm_nn) + (sizeof(float) * kOutputClassesNum)
                                        + sizeof(fbank_features) + aufex_info.dynamic_mem_sz, 
        /* .silence_id; */            kSilenceID,
        /* .name = */                 "Depthwise CNN + LSTM Model on FBANKs features (monolit nn inference)" 
    };
}

// Allocate a model and return pointer to user.
//==============================================================
kws_module* dsconv_lstm_factory::create_module(void* state_fastmem_a) {
    // Initial params check
    kws_dsconv_lstm_nn* instance = nullptr;
    bool is_valid = (state_fastmem_a != nullptr);
    
    // Instance allocation
    if (is_valid) {
        instance = new kws_dsconv_lstm_nn(state_fastmem_a);
        is_valid = is_valid && (instance != nullptr && instance->probs != nullptr);
    }

    // Feature Extractor Module construction
    if (is_valid) {
        const fbank_features::params fex_config {kFbankNumBins,kAudioFrameLength, 
                                                 kAudioSampleRate, kFbankLowFreq, kFbankHighFreq};
        fbank_features::info fex_info = fbank_features::get_info(fex_config);
        instance->state_mem->fex_state_len = fex_info.state_fastbuf_a_len;

        instance->fex_module.reset(fbank_features::create_fbank_fex_module(
                                        fex_config, 
                                        &(instance->state_mem->fex_state_start), 
                                        instance->state_mem->fex_state_len));
        const unsigned int fex_temp_buf_a_size = sizeof(kws_dsconv_lstm_nn::fastmem_x_map::nn_t::fex_phase_t::ir_buf_x);
        const unsigned int fex_temp_buf_b_size = sizeof(kws_dsconv_lstm_nn::fastmem_y_map::nn_t::fex_phase_t::ir_buf_y);
        is_valid = is_valid && fex_info.temp_fastbuf_a_len * sizeof(float) <= fex_temp_buf_a_size;
        is_valid = is_valid && fex_info.temp_fastbuf_b_len * sizeof(float) <= fex_temp_buf_b_size;
        is_valid = is_valid && (instance->fex_module);
    }

    // Check instance and return it
    if (!is_valid && instance != nullptr) {
        delete instance;
        instance = nullptr;
    }
    return instance;
}


//==============================================================
//
//
// KWS module related methods
//
//
//==============================================================

// Constructor
//====================================================
kws_dsconv_lstm_nn::kws_dsconv_lstm_nn(void* state_fastmem_a)
        : fex_module()
        , state_mem(static_cast<state_fastmem_x_map *>(state_fastmem_a))
        , probs(new float[kOutputClassesNum])
        , subframes_tail(0)
        , subframes_num(0)
        , features_tail(0)
        , features_frames(0)
        , last_result_frame_idx(0)
        , current_frame_idx(0)
{}

// Reset module state: further process from scratch
//====================================================
kws_status kws_dsconv_lstm_nn::reset() {
    subframes_tail = subframes_num = 0;
    features_tail = features_frames = 0;
    current_frame_idx = last_result_frame_idx = 0;
    return KWS_STATUS_NONE;
}

// General process function: fill buffers, extract features, envoke NN 
//========================================================================
kws_status kws_dsconv_lstm_nn::process(const sample_t* in_samples, void* temp_fast_mem_a, void* temp_fast_mem_b) {
    kws_status ret_val = KWS_STATUS_NONE;
    fastmem_x_map* fatmem_x = static_cast<fastmem_x_map*>(temp_fast_mem_a);
    fastmem_y_map* fatmem_y = static_cast<fastmem_y_map*>(temp_fast_mem_b);

    // Step 1: Extract features from audio
    memcpy(&state_mem->samples_buf[subframes_tail][0], in_samples, kAudioFrameStride * sizeof(in_samples[0]));
    subframes_num++;
    subframes_tail++;

    // Extract features from the single input in case it's length is enough
    if (subframes_num >= kAudioSubframesInFrame) {
        ret_val = audio_features_extract(&state_mem->samples_buf[subframes_tail - subframes_num][0], 
                                         &state_mem->features_buf[features_tail][0], fatmem_x, fatmem_y);
        subframes_num--;
        features_tail++;
        features_frames++;
    }

    // Move data in the beginning of buffer
    if (ret_val == KWS_STATUS_NONE && subframes_tail == kSubframesInSamplesBuf) {
        memmove(&state_mem->samples_buf[0][0], &state_mem->samples_buf[subframes_tail - subframes_num][0],
            (subframes_num) * kAudioFrameStride * sizeof(state_mem->samples_buf[0][0]));
        subframes_tail = subframes_num;
    }

    // Step 2: Perform NN processing
    if (ret_val == KWS_STATUS_NONE && features_frames >= kFeatureVectorsForInference) {
        ret_val = nn_inference(&state_mem->features_buf[features_tail - features_frames][0], probs.get(),
                               fatmem_x, fatmem_y);
        if (ret_val == KWS_STATUS_NONE) {
            last_result_frame_idx = current_frame_idx;
            ret_val = KWS_STATUS_RESULT_READY;
            features_frames -= kFeatureVectorsStridePerInf;
        }
    }

    // Move data in the beginning of buffer
    if (features_tail == kFeatureVectorsForInference) {
        memmove(&state_mem->features_buf[0][0], &state_mem->features_buf[features_tail - features_frames][0],
            (features_frames) * kFbankNumBins * sizeof(state_mem->features_buf[0][0]));
        features_tail = features_frames;
    }

    current_frame_idx++;
    return ret_val;
}

// Wrap and output KWS result
//====================================================
kws_result kws_dsconv_lstm_nn::get_result() {
    const timestamp_t end = last_result_frame_idx;
    const timestamp_t start = (last_result_frame_idx > kFeatureVectorsForInference)? 
        last_result_frame_idx - kFeatureVectorsForInference : 0;
    return kws_result{start, end, probs.get()};
}

// Return CString regarding to KeyWord ID 
//====================================================
const char* kws_dsconv_lstm_nn::label_id_to_str(int id) const {
    return dsconv_lstm_model_id_to_str(id);
}

// Feature extraction using external FEX module
//====================================================
kws_status kws_dsconv_lstm_nn::audio_features_extract(const sample_t *in_frame, int8_t *out_features,
                                                      fastmem_x_map *x_mem, fastmem_y_map *y_mem) {
    kws_status ret = KWS_STATUS_NONE;
    const int fbanks_num = fex_module->compute(in_frame, y_mem->nn.fex_phase.out_features, 
                                               y_mem->nn.fex_phase.ir_buf_y, x_mem->nn.fex_phase.ir_buf_x);
    if (fbanks_num == kFbankNumBins) {
        mli_tensor out_fx_features = {
            .data.mem.void_p = (void *)out_features, 
            .data.capacity = kFbankNumBins * sizeof(out_features[0]), 
            .el_type = MLI_EL_FX_8,
            .el_params.fx.frac_bits = kFbankFraqBits,
        };
        if (MLI_STATUS_OK != 
                mli_hlp_float_to_fx_tensor(y_mem->nn.fex_phase.out_features, kFbankNumBins,&out_fx_features))
            ret = KWS_STATUS_ERROR;
    } else {
        ret = KWS_STATUS_ERROR;
    }

    return ret;
}

// MLI Based Full Neural Network Inference
//====================================================
kws_status kws_dsconv_lstm_nn::nn_inference(const int8_t *in_features, float *out_probs, 
                                            fastmem_x_map *x_mem, fastmem_y_map *y_mem) {
    kws_status ret = KWS_STATUS_NONE;
    const dsconv_lstm_model_data *m = model;
    mli_tensor ir_X = { (void *)x_mem->nn.cnn_phase.ir_data_x, sizeof(x_mem->nn.cnn_phase.ir_data_x) };
    mli_tensor ir_Y = { (void *)y_mem->nn.cnn_phase.ir_data_y, sizeof(y_mem->nn.cnn_phase.ir_data_y) };
    
    // Convolution Phase
    {
        mli_tensor input = {
            .data.mem.void_p = (void *)in_features, 
            .data.capacity = kFeatureVectorsForInference * kFbankNumBins * sizeof(in_features[0]),
            .shape = {1, kFeatureVectorsForInference, kFbankNumBins}, 
            .rank = 3,
            .el_type = MLI_EL_FX_8, 
            .el_params.fx.frac_bits = kFbankFraqBits,
        };

        // LAYER 1
        ir_Y.el_params.fx.frac_bits = m->L1_conv_out_fraq;
        mli_krn_conv2d_chw_fx8_generic(&input, &m->L1_conv_wt, &m->L1_conv_bias, &m->L1_conv_cfg, &ir_Y);
        mli_krn_leaky_relu_fx8(&ir_Y, &m->leaky_relu_slope_coeff, &ir_Y);

        // LAYER 2
        ir_X.el_params.fx.frac_bits = m->L2a_conv_dw_out_fraq;
        mli_krn_depthwise_conv2d_chw_fx8_k3x3_str1_nopad(&ir_Y, &m->L2a_conv_dw_wt, &m->L2a_conv_dw_bias,
                                                         &m->depthw_conv_cfg, &ir_X);
        mli_krn_leaky_relu_fx8(&ir_X, &m->leaky_relu_slope_coeff, &ir_X);
        ir_Y.el_params.fx.frac_bits = m->L2b_conv_pw_out_fraq;
        mli_krn_conv2d_chw_fx8_k1x1_str1_nopad(&ir_X, &m->L2b_conv_pw_wt, &m->L2b_conv_pw_bias,
                                               &m->pointw_conv_cfg, &ir_Y);
        mli_krn_leaky_relu_fx8(&ir_Y, &m->leaky_relu_slope_coeff, &ir_Y);

        // LAYER 3
        ir_X.el_params.fx.frac_bits = m->L3a_conv_dw_out_fraq;
        mli_krn_depthwise_conv2d_chw_fx8_k3x3_str1_nopad(&ir_Y, &m->L3a_conv_dw_wt, &m->L3a_conv_dw_bias,
                                                         &m->depthw_conv_cfg, &ir_X);
        mli_krn_leaky_relu_fx8(&ir_X, &m->leaky_relu_slope_coeff, &ir_X);
        ir_Y.el_params.fx.frac_bits = m->L3b_conv_pw_out_fraq;
        mli_krn_conv2d_chw_fx8_k1x1_str1_nopad(&ir_X, &m->L3b_conv_pw_wt, &m->L3b_conv_pw_bias,
                                               &m->pointw_conv_cfg, &ir_Y);
        mli_krn_leaky_relu_fx8(&ir_Y, &m->leaky_relu_slope_coeff, &ir_Y);

        // LAYER 4
        ir_X.el_params.fx.frac_bits = m->L4a_conv_dw_out_fraq;
        mli_krn_depthwise_conv2d_chw_fx8_k3x3_str1_nopad(&ir_Y, &m->L4a_conv_dw_wt, &m->L4a_conv_dw_bias,
                                                         &m->depthw_conv_cfg, &ir_X);
        mli_krn_leaky_relu_fx8(&ir_X, &m->leaky_relu_slope_coeff, &ir_X);
        ir_Y.el_params.fx.frac_bits = m->L4b_conv_pw_out_fraq;
        mli_krn_conv2d_chw_fx8_k1x1_str1_nopad(&ir_X, &m->L4b_conv_pw_wt, &m->L4b_conv_pw_bias,
                                               &m->pointw_conv_cfg, &ir_Y);
        mli_krn_leaky_relu_fx8(&ir_Y, &m->leaky_relu_slope_coeff, &ir_X);

        const uint8_t fx8_to_fx16_extra_bits = 8;
        ir_Y.el_type = MLI_EL_FX_16; 
        ir_Y.el_params.fx.frac_bits = ir_X.el_params.fx.frac_bits + fx8_to_fx16_extra_bits;
        mli_hlp_convert_tensor(&ir_X, &ir_Y);
        mli_krn_avepool_chw_fx16_generic(&ir_Y, &m->avg_pool_cfg, &ir_X);

    }

    // LSTM (RNN) Phase
    {
        // Update intermediate tensor for the RNN phase
        ir_Y.data.mem.void_p = (void *)y_mem->nn.rnn_phase.ir_data_y;
        ir_Y.data.capacity = sizeof(y_mem->nn.rnn_phase.ir_data_y);

        // Move timestep dimension at first (least frequently changing)
        mli_krn_permute_fx16(&ir_X, &m->permute_chw2hwc_cfg, &ir_Y);

        // Update intermediate tensor for the RNN phase
        ir_X.data.mem.void_p = (void *)x_mem->nn.rnn_phase.ir_data_x;
        ir_X.data.capacity = sizeof(x_mem->nn.rnn_phase.ir_data_x);

        // LAYER 5
        // init structures for LSTM layer 
        const uint32_t lstm_cell_size = m->L5_lstm_bias.shape[1];
        mli_tensor lstm_prev_out = { 
            .data.mem.void_p = ir_X.data.mem.void_p, 
            .data.capacity = ir_X.data.capacity, 
            .shape = {lstm_cell_size}, 
            .rank = 1, 
            .el_type = MLI_EL_FX_16, 
            .el_params.fx.frac_bits = 7,
        };
        mli_tensor lstm_ir = { (void *)x_mem->nn.rnn_phase.lstm_ir_data, sizeof(x_mem->nn.rnn_phase.lstm_ir_data) };
        const mli_rnn_cell_cfg lstm_cfg = {m->lstm_mode, m->lstm_act, &lstm_ir};
        mli_tensor lstm_cell = {
            .data.mem.void_p = (void *)y_mem->nn.rnn_phase.lstm_cell_data,
            .data.capacity = sizeof(y_mem->nn.rnn_phase.lstm_cell_data),
            .shape = {lstm_cell_size},
            .rank = 1,
            .el_type = MLI_EL_FX_16,
            .el_params.fx.frac_bits = m->L5_lstm_cell_fraq,
        };

        // Clear state buffers and state tensors description completion
        int16_t *cell_ptr = (int16_t *)lstm_cell.data.mem.void_p;
        int16_t *prev_out_ptr = (int16_t *)lstm_prev_out.data.mem.void_p;
        for (uint32_t idx = 0; idx < lstm_cell_size; idx++)
            cell_ptr[idx] = prev_out_ptr[idx] = 0;
        lstm_prev_out.el_params.fx.frac_bits = sizeof(int16_t) * 8 - 1;

        mli_krn_lstm_cell_fx8w16d(&ir_Y, &lstm_prev_out, &m->L5_lstm_wt, &m->L5_lstm_bias,
                                  &lstm_cfg, &lstm_cell, &ir_X);

        // LAYER 6
        ir_Y.el_params.fx.frac_bits = m->L6_fc_out_fraq;
        mli_krn_fully_connected_fx8w16d(&ir_X, &m->L6_fc_wt, &m->L6_fc_bias, &ir_Y);
        mli_krn_softmax_fx16(&ir_Y, &ir_X);
    }
    
    // Transform probabilities into floats
    mli_hlp_fx_tensor_to_float(&ir_X, out_probs, kOutputClassesNum);

    return ret;
}
