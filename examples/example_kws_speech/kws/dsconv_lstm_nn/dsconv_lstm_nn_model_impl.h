/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _DSCONV_LSTM_NN_MODEL_IMPL_H_
#define _DSCONV_LSTM_NN_MODEL_IMPL_H_


#include <memory>

#include <mli_api.h>

#include "../kws_module.h"
#include "../kws_factory.h"
#include "audio_features.h"


namespace mli_kws {


// Forward Declaration of model description structure
struct dsconv_lstm_model_data;

//====================================================================
// Depthwise CNN + LSTM Model on FBANKs features (monolit nn inference)
//
// Next processing steps are used:
// - Extraction of FBANK Features from input audio frames (floats -> fx8)
// - Convolution layer 1 on input fbank features and Leaky ReLU activation (slope = 0.2) (fx8)
// - Depthwise + pointwise convolution layers 2-4 with Leaky ReLU activation (slope = 0.2) (fx8)
// - Average pooling across frequency dimension (not-time) (fx8 -> fx16)
// - LSTM sequential processing across time dimension (fx8w16d)
// - Fully connected and softmax (fx8w16d)
//
// Usage is similar to base class.
//==============================================================
class kws_dsconv_lstm_nn: public kws_module {
public:

    // Interface methods. See *kws_module* base class
    ~kws_dsconv_lstm_nn() = default;
    kws_status reset() override;
    kws_status process(const sample_t* in_samples, void* temp_fastmem_a, void* temp_fastmem_b) override;
    kws_result get_result() override;
    const char* label_id_to_str(int id) const override;

private:
    // Memory maps of fast banks for internal usage (incomplete types)
    struct fastmem_x_map;
    struct fastmem_y_map;
    struct state_fastmem_x_map;

    // Factory is a friend of this class and the only ebtry point for construction
    friend class dsconv_lstm_factory;

    // Constructor is not accessable for user (constructin by friend factory class)
    kws_dsconv_lstm_nn(void* state_fastmem_a);

    // Privat methods for high loaded processing
    kws_status audio_features_extract(const sample_t *in_frame, int8_t *out_features,
                                      fastmem_x_map *x_mem, fastmem_y_map *y_mem);

    kws_status nn_inference(const int8_t *in_features, float *out_probs,
                            fastmem_x_map *x_mem, fastmem_y_map *y_mem);

    // Pointer to staticaly allocated model description
    static const dsconv_lstm_model_data* const model;

    // Other class members
    std::unique_ptr<au_fex::fbank_features> fex_module;
    state_fastmem_x_map *state_mem;
    std::unique_ptr<float []> probs;
    size_t subframes_tail;
    size_t subframes_num;
    size_t features_tail;
    size_t features_frames;
    timestamp_t last_result_frame_idx;
    timestamp_t current_frame_idx;
};

} // namespace mli_kws {


#endif // _DSCONV_LSTM_NN_MODEL_IMPL_H_