/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "face_trigger_model.h"

#include "mli_api.h"
#include "mli_config.h"

#include "face_trigger_constants.h"
#include "tests_aux.h"

//==============================================================
//
//
// Data related to the Module
//
//
//==============================================================

// Defining data sections attributes
//===================================
#if (ARC_PLATFORM == V2DSP_XY)
#if defined (__GNUC__) && !defined (__CCAC__)
//ARC GNU tools
// Model Weights attribute
#define _Wdata_attr __attribute__((section(".mli_model")))
#define _W  _Wdata_attr

// Bank X (XCCM) attribute
#define __Xdata_attr __attribute__((section(".Xdata")))
#define _X  __Xdata_attr

// Bank Y (YCCM) attribute
#define __Ydata_attr __attribute__((section(".Ydata")))
#define _Y  __Ydata_attr

// Bank Z (DCCM) attribute
#define __Zdata_attr __attribute__((section(".Zdata")))
#define _Z  __Zdata_attr

#else
//Metaware tools
// Model Weights attribute
#define _Wdata_attr __attribute__((section(".mli_model")))
#define _W __xy _Wdata_attr

// Bank X (XCCM) attribute
#define __Xdata_attr __attribute__((section(".Xdata")))
#define _X __xy __Xdata_attr

// Bank Y (YCCM) attribute
#define __Ydata_attr __attribute__((section(".Ydata")))
#define _Y __xy __Ydata_attr

// Bank Z (DCCM) attribute
#define __Zdata_attr __attribute__((section(".Zdata")))
#define _Z __xy __Zdata_attr
#endif // if defined (__GNUC__) && !defined (__CCAC__)

#else
#define _X __attribute__((section(".mli_ir_buf")))
#define _Y __attribute__((section(".mli_ir_buf")))
#define _Z __attribute__((section(".mli_ir_buf")))
#define _W __attribute__((section(".mli_model")))
#endif




// Intermediate data buffers (enough size for max intermediate results)
//==============================
#define kMaxBufPoints (FT_MODEL_IN_POINTS)
#define kSecondMaxBufPoints (4*16*16)
#define kIrBufPointsL2 (2*16*16)

static int16_t  _X    x_mem_buf[kMaxBufPoints];
static int16_t  _Y    y1_mem_buf[kSecondMaxBufPoints];
static int16_t  _Y    y2_mem_buf[kIrBufPointsL2];


// Model coefficients buffer
// Note: Actual data type is int16_t. int8_t only for compatibility)
//=============================================================================
static const int8_t _W kL1convWeightsBuf[] = CONV1_WEIGHTS;
static const int8_t _W kL1convBiasBuf[] = CONV1_BIAS;

static const int8_t _W kL2AconvWeightsBuf[] = CONV2_A_WEIGHTS;
static const int8_t _W kL2AconvBiasBuf[] = CONV2_A_BIAS;

static const int8_t _W kL2BconvWeightsBuf[] = CONV2_B_WEIGHTS;
static const int8_t _W kL2BconvBiasBuf[] = CONV2_B_BIAS;

static const int8_t _W kL3convWeightsBuf[] = CONV3_WEIGHTS;
static const int8_t _W kL3convBiasBuf[] = CONV3_BIAS;


static const int8_t _W kL4fcWeightsBuf[] = FC4_WEIGHTS;
static const int8_t _W kL4convBiasBuf[] = FC4_BIAS;

// Decision threshold and LookUp activation table
//=============================================================================
static const int16_t kOutputDecisionThreshold = 0;

static const uint8_t _W kUserActivationLookUp[] = ACT_LUT;


//==============================================================
//
//
// Model description and configuration
//
//
//==============================================================
#pragma Data(".mli_data")
// Configuration objects for layers
//===============================================

static const mli_conv2d_cfg shared_conv_cfg = {
    .stride_height = 2, .stride_width = 2,
    .padding_bottom = 0, .padding_top = 0,
    .padding_left = 0, .padding_right = 0,
    .relu.type = MLI_RELU_NONE
};


// Intermediate and helper tensors
//===============================================
static mli_tensor ir_tensor_X = {
    .data = (void *)x_mem_buf,
    .capacity = sizeof(x_mem_buf),
};

static mli_tensor ir_tensor_Y = {
    .data = (void *)y1_mem_buf,
    .capacity = sizeof(y1_mem_buf),
};

static mli_tensor custom_l2_ir_tensor = {
    .data = (void *)y2_mem_buf,
    .capacity = sizeof(y2_mem_buf),
};



// Conv 1 Layer related tensors
//===================================
static const mli_tensor L1_conv_wt = {
    .data = (void *)kL1convWeightsBuf,
    .capacity = sizeof(kL1convWeightsBuf),
    .shape = {4,1,6,6},
    .rank = 4,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 12,
};


static const mli_tensor L1_conv_bias = {
    .data = (void *)kL1convBiasBuf,
    .capacity = sizeof(kL1convBiasBuf),
    .shape = {4},
    .rank = 1,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 12,
};

static const int8_t kL1ConvOutFracBits = 16;

// Conv 2 Layer related data
//===================================
//2.a Convolution: considering each input fmap separately
static const mli_tensor L2a_conv_wt = {
    .data = (void *)kL2AconvWeightsBuf,
    .capacity = sizeof(kL2AconvWeightsBuf),
    .shape = {8,1,4,4},
    .rank = 4,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 12,
};

static const mli_tensor L2a_conv_bias = {
    .data = (void *)kL2AconvBiasBuf,
    .capacity = sizeof(kL2AconvBiasBuf),
    .shape = {8},
    .rank = 1,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 12,
};

//2.b Convolution: considering input fmaps in pairs
static const mli_tensor L2b_conv_wt = {
    .data = (void *)(kL2BconvWeightsBuf),
    .capacity = sizeof(kL2BconvWeightsBuf),
    .shape = {6,2,4,4},
    .rank = 4,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 12,
};

static const mli_tensor L2b_conv_bias = {
    .data = (void *)(kL2BconvBiasBuf),
    .capacity = sizeof(kL2BconvBiasBuf),
    .shape = {6},
    .rank = 1,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 12,
};

static const int8_t kL2ConvOutFracBits = 16;

// Conv 3 Layer related data
//===================================
static const mli_tensor L3_conv_wt = {
    .data = (void *)kL3convWeightsBuf,
    .capacity = sizeof(kL3convWeightsBuf),
    .shape = {14,1,6,6},
    .rank = 4,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 12,
};

static const mli_tensor L3_conv_bias = {
    .data = (void *)kL3convBiasBuf,
    .capacity = sizeof(kL3convBiasBuf),
    .shape = {14},
    .rank = 1,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 12,
};

static const int8_t kL3ConvOutFracBits = 16;

// FC4 Layer related data
//===================================
static const mli_tensor L4_fc_wt = {
    .data = (void *)kL4fcWeightsBuf,
    .capacity = sizeof(kL4fcWeightsBuf),
    .shape = {1, 14},
    .rank = 2,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 12,
};

static const mli_tensor L4_fc_bias = {
    .data = (void *)kL4convBiasBuf,
    .capacity = sizeof(kL4convBiasBuf),
    .shape = {1},
    .rank = 1,
    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 12,
};

static const int8_t kL4ConvOutFracBits = 10;
#pragma Data()


// Subtensor increment parameters
typedef struct {
    int first_out_dim_inc;
} mli_increment_subtsr_cfg;

//==============================================================
//  Declaration of user specific functions and kernels
//==============================================================
static inline void increment_sub_tensor(mli_tensor *t, mli_increment_subtsr_cfg *cfg);

static void user_custom_preprocessing(const uint8_t *in_image36x36, mli_tensor *output);

static void user_custom_activation(const mli_tensor *input, mli_tensor *output);

static void user_custom_convolution_layer2(const mli_tensor *input, mli_tensor *temp_data, mli_tensor *output);

//Profiling vars
int total_cycles = 0;
int run_num = 0;

//==============================================================
//
//  Face Trigger graph Layer-by-Layer execution
//
//==============================================================
int mli_face_trigger_process(const uint8_t *image_buffer){

#ifndef PROFILE_ON
    // Preprocessing: Expend data to 16bit
    //=============================================
    user_custom_preprocessing(image_buffer, &ir_tensor_X);

    // Layer 1: Convolution on grayscale image
    //=============================================
    ir_tensor_Y.el_params.fx.frac_bits = kL1ConvOutFracBits;
    mli_krn_conv2d_chw_fx16_generic(&ir_tensor_X, &L1_conv_wt, &L1_conv_bias,
                            &shared_conv_cfg, &ir_tensor_Y);
    user_custom_activation(&ir_tensor_Y, &ir_tensor_Y);

    // Layer 2: Custom convolution layer
    //=============================================
    ir_tensor_X.el_params.fx.frac_bits = kL2ConvOutFracBits;
    user_custom_convolution_layer2(&ir_tensor_Y, &custom_l2_ir_tensor, &ir_tensor_X);
    user_custom_activation(&ir_tensor_X, &ir_tensor_X);

    // Layer 3: Depthwise convolution layer
    //=============================================
    ir_tensor_Y.el_params.fx.frac_bits = kL3ConvOutFracBits;
    mli_krn_depthwise_conv2d_chw_fx16_generic(&ir_tensor_X, &L3_conv_wt, &L3_conv_bias,
                                              &shared_conv_cfg, &ir_tensor_Y);
    user_custom_activation(&ir_tensor_Y, &ir_tensor_Y);

    // Layer 4: fully connected layer
    //=============================================
    ir_tensor_X.el_params.fx.frac_bits = kL4ConvOutFracBits;
    mli_krn_fully_connected_fx16(&ir_tensor_Y, &L4_fc_wt, &L4_fc_bias,
                                 &ir_tensor_X);
#else
    mli_status ret = MLI_STATUS_OK;
    unsigned preproc_cycles = 0;
    unsigned layer1_cycles = 0;
    unsigned activ1_cycles = 0;
    unsigned layer2_cycles = 0;
    unsigned activ2_cycles = 0;
    unsigned layer3_cycles = 0;
    unsigned activ3_cycles = 0;
    unsigned layer4_cycles = 0;

    // Preprocessing: Expend data to 16bit
    //=============================================
    PROFILE(user_custom_preprocessing(image_buffer, &ir_tensor_X));
    preproc_cycles += cycle_cnt;

    // Layer 1: Convolution on grayscale image
//=============================================
    ir_tensor_Y.el_params.fx.frac_bits = kL1ConvOutFracBits;
    PROFILE(mli_krn_conv2d_chw_fx16_generic(&ir_tensor_X, &L1_conv_wt, &L1_conv_bias,
      &shared_conv_cfg, &ir_tensor_Y));
    layer1_cycles += cycle_cnt;
    
    PROFILE(user_custom_activation(&ir_tensor_Y, &ir_tensor_Y));
    activ1_cycles += cycle_cnt;

    // Layer 2: Custom convolution layer
    //=============================================
    ir_tensor_X.el_params.fx.frac_bits = kL2ConvOutFracBits;
    PROFILE(user_custom_convolution_layer2(&ir_tensor_Y, &custom_l2_ir_tensor, &ir_tensor_X));
    layer2_cycles += cycle_cnt;
    PROFILE(user_custom_activation(&ir_tensor_X, &ir_tensor_X));
    activ2_cycles += cycle_cnt;

    // Layer 3: Depthwise convolution layer
    //=============================================
    ir_tensor_Y.el_params.fx.frac_bits = kL3ConvOutFracBits;
    PROFILE(mli_krn_depthwise_conv2d_chw_fx16_generic(&ir_tensor_X, &L3_conv_wt, &L3_conv_bias,
      &shared_conv_cfg, &ir_tensor_Y));
    layer3_cycles += cycle_cnt;
    PROFILE(user_custom_activation(&ir_tensor_Y, &ir_tensor_Y));
    activ3_cycles += cycle_cnt;

    // Layer 4: fully connected layer
    //=============================================
    ir_tensor_X.el_params.fx.frac_bits = kL4ConvOutFracBits;
    PROFILE(ret = mli_krn_fully_connected_fx16(&ir_tensor_Y, &L4_fc_wt, &L4_fc_bias,
      &ir_tensor_X));
    layer4_cycles += cycle_cnt;

    const unsigned total = preproc_cycles + layer1_cycles + activ1_cycles + layer2_cycles + activ2_cycles + layer3_cycles + activ3_cycles + layer4_cycles;
    total_cycles += total;
    run_num++;
    if (print_summary) {
      printf("\n\nSummary:\n"
        "\tImage Preprocessing: %u cycles\n"
        "\tLayer1: %u cycles\n"
        "\tActivation1: %u cycles\n"
        "\tLayer2: %u cycles\n"
        "\tActivation2: %u cycles\n"
        "\tLayer3: %u cycles\n"
        "\tActivation3: %u cycles\n"
        "\tLayer4: %u cycles\n"
        "\n\tTotal: %u cycles\n\n",
        preproc_cycles, layer1_cycles, activ1_cycles, layer2_cycles, activ2_cycles, layer3_cycles, activ3_cycles, layer4_cycles, total);
      print_summary = false;
    }
#endif
    // Decision by threshold
    //=============================================
    const int16_t out_score = ((int16_t *)ir_tensor_X.data)[0];
    return (out_score > kOutputDecisionThreshold)? 1: 0;
}


//=============================================================================
// Input preprocessing procedure
//=============================================================================
static void user_custom_preprocessing(const uint8_t *in_image36x36, mli_tensor *output) {
    // Crop input image on the right and bottom
    // Reason: Layer 2 doesn't analyze 2 pixels on the right and bottom
    const int columns_in = 36;
    const int rows_out = 32;
    const int columns_out = 32;
    int16_t * vec_out = (int16_t *)(output->data);

    for (int r_idx = 0; r_idx < rows_out; r_idx++)
        for (int c_idx = 0; c_idx < columns_out; c_idx++)
            vec_out[r_idx*columns_out + c_idx] = (int16_t)in_image36x36[r_idx*columns_in + c_idx];

    // Fill output tensor
    output->shape[FMAP_C_DIM_CHW] = 1;
    output->shape[FMAP_H_DIM_CHW] = rows_out;
    output->shape[FMAP_W_DIM_CHW] = columns_out;
    output->rank = 3;
    output->el_type = MLI_EL_FX_16;
    output->el_params.fx.frac_bits = 8;
}


//=============================================================================
// Custom activation function
//=============================================================================
static void user_custom_activation(const mli_tensor *input, mli_tensor *output) {
    const int16_t * vec_in = (int16_t *)input->data;
    const uint32_t el_num = mli_hlp_count_elem_num(input, 0);
    int16_t * vec_out = (int16_t *)(output->data);

    for (uint32_t i = 0; i < el_num; i++) {
        int16_t index = (vec_in[i] >> (sizeof(int16_t) * 8 - ACT_LUT_IDX_BITS));
        index &= ((1<<ACT_LUT_IDX_BITS) - 1);                     // 0x3FF for 10 bits
        vec_out[i] = (int16_t)kUserActivationLookUp[index];
    }

    // Update output shape
    for(unsigned idx = 0; idx < input->rank; idx++)
        output->shape[idx] = input->shape[idx];
    output->rank = input->rank;
    output->el_type = input->el_type;
    output->el_params.fx.frac_bits = 8;
}


// function to increment a subtensor. first_out_dim_inc is the increment amount in the first output dimension
// so for CHW layout it is the number of channels to increment
static inline void increment_sub_tensor(mli_tensor *t, mli_increment_subtsr_cfg *cfg) {
    
    int inc = cfg->first_out_dim_inc;
    // multiply the size of the other dimensions except the first.
    for (int i = 1; i < t->rank; i++) {
        inc *= t->shape[i];
    }
    t->data += inc * mli_hlp_tensor_element_size(t);
    t->capacity -=  inc * mli_hlp_tensor_element_size(t);
}

//=============================================================================
// Layer 2: Complex convolution function
//=============================================================================
#define FMAPS_NUM (4)
static void user_custom_convolution_layer2(
        const mli_tensor *input,
        mli_tensor *temp_data,
        mli_tensor *output) {
    mli_tensor filter_weights = {0};
    mli_tensor filter_bias = {0};
    mli_tensor input_sub = {0};
    mli_tensor out_feature_map = {
        .data=output->data, .capacity=output->capacity,
        .el_params.fx.frac_bits = output->el_params.fx.frac_bits};

    mli_point_to_subtsr_cfg fmap_sub = {.start_coord = {0}, .coord_num = 1, .first_out_dim_size = 1};
    mli_increment_subtsr_cfg fmap_inc = {.first_out_dim_inc = 1};
    mli_point_to_subtsr_cfg coef_sub = {.start_coord = {0}, .coord_num = 1, .first_out_dim_size = 2};
    mli_increment_subtsr_cfg coef_inc = {.first_out_dim_inc = 2};

    //2.a Considering each input fmap separately
    //=======================================================
    // run a convolution with two output planes for each input plane
    // first point to the first plane in the input and weights
    mli_hlp_point_to_subtensor(&L2a_conv_wt, &coef_sub, &filter_weights);
    mli_hlp_point_to_subtensor(&L2a_conv_bias, &coef_sub, &filter_bias);
    mli_hlp_point_to_subtensor(input, &fmap_sub, &input_sub);
    for (int i = 0; i < FMAPS_NUM; i++) {
        
        mli_krn_conv2d_chw_fx16_generic(&input_sub, &filter_weights, &filter_bias,
                        &shared_conv_cfg, &out_feature_map);

        increment_sub_tensor(&input_sub, &fmap_inc);
        increment_sub_tensor(&filter_bias, &coef_inc);
        increment_sub_tensor(&filter_weights, &coef_inc);
        increment_sub_tensor(&out_feature_map, &coef_inc);

    }

    //2.b Convolution: use input fmaps in adjacent pairs
    //=======================================================
    // run a convolution with 1 output plane for each pair of input planes.
    coef_sub.first_out_dim_size = 1;
    coef_inc.first_out_dim_inc = 1;
    fmap_sub.first_out_dim_size = 2;
    fmap_sub.start_coord[0] = 0;
    fmap_inc.first_out_dim_inc = 2;
    mli_hlp_point_to_subtensor(&L2b_conv_wt, &coef_sub, &filter_weights);
    mli_hlp_point_to_subtensor(&L2b_conv_bias, &coef_sub, &filter_bias);
    mli_hlp_point_to_subtensor(input, &fmap_sub, &input_sub);
    for (int i = 0; i < FMAPS_NUM-1; i++) {
        mli_krn_conv2d_chw_fx16_generic(&input_sub, &filter_weights, &filter_bias,
                                        &shared_conv_cfg, &out_feature_map);
                                        
        increment_sub_tensor(&input_sub, &fmap_inc);
        increment_sub_tensor(&filter_bias, &coef_inc);
        increment_sub_tensor(&filter_weights, &coef_inc);
        increment_sub_tensor(&out_feature_map, &coef_inc);

    }

    //2.c Convolution: use input fmaps in pairs (concatenate not-sequential ones)
    //=======================================================
    // run a convolution with 1 output plain for each non acjacent pair of input planes.
    // the pairs are in the following look up table
    const char fmap_idxs[3][2] = {{0, 2}, {1, 3}, {0, 3}};
    mli_tensor input0 = {0};
    mli_tensor input1 = {0};
    mli_tensor *concat_inputs[2] = {&input0, &input1};
    mli_concat_cfg concat_cfg = {.tensors_num = 2, .axis = 0};
    fmap_sub.first_out_dim_size = 1;
    for (int i = 0; i < FMAPS_NUM-1; i++) {
        fmap_sub.start_coord[0] = fmap_idxs[i][0];
        mli_hlp_point_to_subtensor(input, &fmap_sub, &input0);
        fmap_sub.start_coord[0] = fmap_idxs[i][1];
        mli_hlp_point_to_subtensor(input, &fmap_sub, &input1);

        // Concatenate two separate feature maps into one tensor and pass it to convolution
        mli_krn_concat_fx16((const mli_tensor **)concat_inputs, &concat_cfg, temp_data);
        mli_krn_conv2d_chw_fx16_generic(temp_data, &filter_weights, &filter_bias,
                                        &shared_conv_cfg, &out_feature_map);
        increment_sub_tensor(&filter_bias, &coef_inc);
        increment_sub_tensor(&filter_weights, &coef_inc);
        increment_sub_tensor(&out_feature_map, &coef_inc);
    }

    // Fill output tensor
    output->shape[0] = L2a_conv_bias.shape[0] +  L2b_conv_bias.shape[0];
    for(unsigned i = 1; i < input->rank; i++)
        output->shape[i] = out_feature_map.shape[i];
    output->rank = out_feature_map.rank;
    output->el_type = out_feature_map.el_type;
}

