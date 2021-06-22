#include "emnist_model.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "mli_api.h"
#include "mli_types.h"
#include "mli_config.h"

#include "emnist_constants.h"
#include "tests_aux.h"
#include "tensor_transform.h"

#if (MODEL_BIT_DEPTH == MODEL_SA_8)
#define D_EL_TYPE (MLI_EL_SA_8)
#else
#define D_EL_TYPE (MLI_EL_FX_16)
#endif
//==============================================================
//
//
// Data related to the Module
//
//
//==============================================================

const char debug_ir_root[] = "model/idx";

inline void set_mli_tensor_params(mli_tensor* tensor, int16_t zero_point, int8_t scale_frac_bits, int16_t scale) {
    tensor->el_params.sa.zero_point.mem.i16 = zero_point;
    tensor->el_params.sa.scale_frac_bits.mem.i8 = scale_frac_bits;
    tensor->el_params.sa.scale.mem.i16 = scale;
}

inline void set_mli_tensor_shape1(mli_tensor* tensor, uint32_t shape0) {
    tensor->rank = 1;
    tensor->shape[0] = shape0;
    tensor->mem_stride[0] = 1;
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
#define IR_BUF_SZ_MOST (16*28*28)
#define IR_BUF_SZ_NEXT (8*28*28)
#define LUT_BUF_SIZE (512)
static d_type _Z  x_mem_buf[IR_BUF_SZ_MOST];
static d_type _Y  y_mem_buf[IR_BUF_SZ_NEXT];
static int16_t  _X    lut_mem_buf[LUT_BUF_SIZE];
// Module Input/Output tensors and their's external interface
//============================================================
static mli_tensor input = {
    .data = {
        .capacity = sizeof(d_type) * IN_POINTS,
        .mem = { .D_FIELD = (d_type *)x_mem_buf }
    },
    .mem_stride = { 28, 1, 1 },
    .shape = {28, 28, 1},
    .rank = 3,
    .el_type = D_EL_TYPE,
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
    .el_params.sa = {
        .zero_point.mem = { .i16 = 0 },
        .scale.mem = { .i16 = 0 },
        .dim = -1,
        .scale_frac_bits.mem = { .i8 = 22 }
    }
#else
        .el_params.fx.frac_bits = 7
#endif
};
static mli_tensor output = {
    .data = {
        .capacity = sizeof(d_type) * OUT_POINTS,
        .mem = { .D_FIELD = (d_type *)y_mem_buf }
    },
    .mem_stride = { 1 },
    .shape = {26},
    .rank = 1,
    .el_type = D_EL_TYPE,
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
    .el_params.sa = {
        .zero_point.mem = { .i16 = 0 },
        .scale.mem = { .i16 = 1 },
        .dim = -1,
        .scale_frac_bits.mem = { .i8 = 0 }
    }
#else
    .el_params.fx.frac_bits = 0,
#endif 
};

static mli_lut output_lut = {
	.data = {
		.capacity = sizeof(int16_t) * LUT_BUF_SIZE,
		.mem = { .pi16 = (int16_t *)lut_mem_buf}
	},

};

// Interface variables: Available to user via main model header
//===========================================================
mli_tensor * const emnist_cf_net_input = &input;
mli_tensor * const emnist_cf_net_output = &output;

char const letters[26] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};


//==============================================================
//  Model description and configuration
//==============================================================
#pragma Data(".mli_data")

// Configuration objects for layers
//===============================================

static const mli_permute_cfg permute_hwc2chw_cfg = {
        .perm_dim = {0, 1, 2} // 2 0 1
};

static const mli_conv2d_cfg shared_conv_cfg = {
    .stride_height = 1, .stride_width = 1,
    .padding_bottom = 2, .padding_top = 2,
    .padding_left = 2, .padding_right = 2,
    .dilation_width = 1, .dilation_height = 1,
    .relu.type = MLI_RELU_GEN
};

static const mli_pool_cfg shared_pool_cfg = {
    .kernel_height = 2, .kernel_width = 2,
    .stride_height = 2, .stride_width = 2,
    .padding_bottom = 0, .padding_top = 0,
    .padding_left = 0, .padding_right = 0
};

static const mli_tensor L1_conv_wt = {
    .data = {
        .capacity = CONV1_W_ELEMENTS * sizeof(w_type),
        .mem = { .W_FIELD = (w_type *)L1_conv_wt_buf }
    },
    .shape = CONV1_W_SHAPE,
    .mem_stride = CONV1_W_MEM_STRIDE,
    .rank = CONV1_W_RANK,
    .el_type = W_EL_TYPE,
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
    .el_params.sa = {
        .zero_point.mem = { .pi16 = (int16_t*)CONV1_W_ZP },
        .scale.mem = { .pi16 = (int16_t*)CONV1_W_SCALE },
        .dim = CONV1_W_DIM,
        .scale_frac_bits.mem = { .pi8 = (int8_t*)CONV1_W_FRAQ }
    }
#else
    .el_params.fx.frac_bits = CONV1_W_FRAQ
#endif
};

static const mli_tensor L1_conv_bias = {
    .data = {
        .capacity = CONV1_B_ELEMENTS * sizeof(b_type),
        .mem = { .B_FIELD = (b_type *)L1_conv_bias_buf }
    },
    .shape = CONV1_B_SHAPE,
    .mem_stride = CONV1_B_MEM_STRIDE,
    .rank = CONV1_B_RANK,
    .el_type = B_EL_TYPE,
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
    .el_params.sa = {
        .zero_point.mem = { .pi16 = (int16_t*)CONV1_B_ZP },
        .scale.mem = { .pi16 = (int16_t*)CONV1_B_SCALE },
        .dim = CONV1_B_DIM,
        .scale_frac_bits.mem = { .pi8 = (int8_t*)CONV1_B_FRAQ }
    }
#else
    .el_params.fx.frac_bits = CONV1_B_FRAQ
#endif
};



static const mli_tensor L2_conv_wt = {
    .data = {
        .capacity = CONV2_W_ELEMENTS * sizeof(w_type),
        .mem = { .W_FIELD = (w_type *)L2_conv_wt_buf }
    },
    .shape = CONV2_W_SHAPE,
    .mem_stride = CONV2_W_MEM_STRIDE,
    .rank = CONV2_W_RANK,
    .el_type = W_EL_TYPE,
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
    .el_params.sa = {
        .zero_point.mem = { .pi16 = (int16_t*)CONV2_W_ZP },
        .scale.mem = { .pi16 = (int16_t*)CONV2_W_SCALE },
        .dim = CONV2_W_DIM,
        .scale_frac_bits.mem = { .pi8 = (int8_t*)CONV2_W_FRAQ }
    }
#else
    .el_params.fx.frac_bits = CONV2_W_FRAQ
#endif
};

static const mli_tensor L2_conv_bias = {
    .data = {
        .capacity = CONV2_B_ELEMENTS * sizeof(b_type),
        .mem = { .B_FIELD = (b_type *)L2_conv_bias_buf }
    },
    .shape = CONV2_B_SHAPE,
    .mem_stride = CONV2_B_MEM_STRIDE,
    .rank = CONV2_B_RANK,
    .el_type = B_EL_TYPE,
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
    .el_params.sa = {
        .zero_point.mem = { .pi16 = (int16_t*)CONV2_B_ZP },
        .scale.mem = { .pi16 = (int16_t*)CONV2_B_SCALE },
        .dim = CONV2_B_DIM,
        .scale_frac_bits.mem = { .pi8 = (int8_t*)CONV2_B_FRAQ }
    }
#else
    .el_params.fx.frac_bits = CONV2_B_FRAQ
#endif
};



static const mli_tensor L3_conv_wt = {
    .data = {
        .capacity = CONV3_W_ELEMENTS * sizeof(w_type),
        .mem = { .W_FIELD = (w_type *)L3_conv_wt_buf }
    },
    .shape = CONV3_W_SHAPE,
    .mem_stride = CONV3_W_MEM_STRIDE,
    .rank = CONV3_W_RANK,
    .el_type = W_EL_TYPE,
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
    .el_params.sa = {
        .zero_point.mem = { .pi16 = (int16_t*)CONV3_W_ZP },
        .scale.mem = { .pi16 = (int16_t*)CONV3_W_SCALE },
        .dim = CONV3_W_DIM,
        .scale_frac_bits.mem = { .pi8 = (int8_t*)CONV3_W_FRAQ }
    }
#else
    .el_params.fx.frac_bits = CONV3_W_FRAQ
#endif
};

static const mli_tensor L3_conv_bias = {
    .data = {
        .capacity = CONV3_B_ELEMENTS * sizeof(b_type),
        .mem = { .B_FIELD = (b_type *)L3_conv_bias_buf }
    },
    .shape = CONV3_B_SHAPE,
    .mem_stride = CONV3_B_MEM_STRIDE,
    .rank = CONV3_B_RANK,
    .el_type = B_EL_TYPE,
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
    .el_params.sa = {
        .zero_point.mem = { .pi16 = (int16_t*)CONV3_B_ZP },
        .scale.mem = { .pi16 = (int16_t*)CONV3_B_SCALE },
        .dim = CONV3_B_DIM,
        .scale_frac_bits.mem = { .pi8 = (int8_t*)CONV3_B_FRAQ }
    }
#else
    .el_params.fx.frac_bits = CONV3_B_FRAQ
#endif
};




static mli_tensor L4_fc_wt = {
    .data = {
        .capacity = FC4_W_ELEMENTS * sizeof(w_type),
        .mem = { .W_FIELD = (w_type *)L4_fc_wt_buf }
    },
    .shape = FC4_W_SHAPE,
    .mem_stride = FC4_W_MEM_STRIDE,
    .rank = FC4_W_RANK,
    .el_type = W_EL_TYPE,
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
    .el_params.sa = {
        .zero_point.mem = { .pi16 = (int16_t*)FC4_W_ZP },
        .scale.mem = { .pi16 = (int16_t*)FC4_W_SCALE },
        .dim = FC4_W_DIM,
        .scale_frac_bits.mem = { .pi8 = (int8_t*)FC4_W_FRAQ }
    }
#else
    .el_params.fx.frac_bits = FC4_W_FRAQ
#endif
};

static mli_tensor L4_fc_bias = {
    .data = {
        .capacity = FC4_B_ELEMENTS * sizeof(b_type),
        .mem = { .B_FIELD = (b_type *)L4_fc_bias_buf }
    },
    .shape = FC4_B_SHAPE,
    .mem_stride = FC4_B_MEM_STRIDE,
    .rank = FC4_B_RANK,
    .el_type = W_EL_TYPE,
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
    .el_params.sa = {
        .zero_point.mem = { .pi16 = (int16_t*)FC4_B_ZP },
        .scale.mem = { .pi16 = (int16_t*)FC4_B_SCALE },
        .dim = FC4_B_DIM,
        .scale_frac_bits.mem = { .pi8 = (int8_t*)FC4_B_FRAQ }
    }
#else
    .el_params.fx.frac_bits = FC4_B_FRAQ,
#endif
};


static mli_tensor L5_fc_wt = {
    .data = {
        .capacity = FC5_W_ELEMENTS * sizeof(w_type),
        .mem = { .W_FIELD = (w_type *)L5_fc_wt_buf }
    },
    .shape = FC5_W_SHAPE,
    .mem_stride = FC5_W_MEM_STRIDE,
    .rank = FC5_W_RANK,
    .el_type = W_EL_TYPE,
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
    .el_params.sa = {
        .zero_point.mem = { .pi16 = (int16_t*)FC5_W_ZP },
        .scale.mem = { .pi16 = (int16_t*)FC5_W_SCALE },
        .dim = FC5_W_DIM,
        .scale_frac_bits.mem = { .pi8 = (int8_t*)FC5_W_FRAQ }
    }
#else
    .el_params.fx.frac_bits = FC5_W_FRAQ
#endif
};

static mli_tensor L5_fc_bias = {
    .data = {
        .capacity = FC5_B_ELEMENTS * sizeof(b_type),
        .mem = { .B_FIELD = (b_type *)L5_fc_bias_buf }
    },
    .shape = FC5_B_SHAPE,
    .mem_stride = FC5_B_MEM_STRIDE,
    .rank = FC5_B_RANK,
    .el_type = W_EL_TYPE,
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
    .el_params.sa = {
        .zero_point.mem = { .pi16 = (int16_t*)FC5_B_ZP },
        .scale.mem = { .pi16 = (int16_t*)FC5_B_SCALE },
        .dim = FC5_B_DIM,
        .scale_frac_bits.mem = { .pi8 = (int8_t*)FC5_B_FRAQ }
    }
#else
    .el_params.fx.frac_bits = FC5_B_FRAQ,
#endif
};

// Intermediate result tensors
//===============================================
static mli_tensor ir_tensor_X = {
    .data = {
        .capacity = sizeof(x_mem_buf),
        .mem = { .D_FIELD = (d_type *)x_mem_buf }
    },
    .shape = {0, 0, 0, 0},
    .rank = 4,
    .el_type = D_EL_TYPE,
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
    .el_params.sa = {
    .zero_point.mem = { .i16 = 0 },
    .scale.mem = { .i16 = 1 },
    .dim = -1,
    .scale_frac_bits.mem = { .i8 = 0 }
    }
#else
    .el_params.fx.frac_bits = FRQ_BITS(0, d_type),
#endif
};

static mli_tensor ir_tensor_Y = {
    .data = {
        .capacity = sizeof(y_mem_buf),
        .mem = { .D_FIELD = (d_type *)y_mem_buf }
    },
    .shape = {0, 0, 0, 0},
    .rank = 4,
    .el_type = D_EL_TYPE,
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
    .el_params.sa = {
    .zero_point.mem = { .i16 = 0 },
    .scale.mem = { .i16 = 1 },
    .dim = -1,
    .scale_frac_bits.mem = { .i8 = 0 }
    }
#else
    .el_params.fx.frac_bits = FRQ_BITS(0, d_type),
#endif
};

#pragma Data()

static void preprocessing(mli_tensor* net_input_) {
    d_type* const dst = (d_type * const)net_input_->data.mem.D_FIELD;
    if (net_input_->el_params.fx.frac_bits == 7) {
        for (int idx = 0; idx < IN_POINTS; idx++) {
            dst[idx] = dst[idx] - 128;
        }
    }
    else if (net_input_->el_params.fx.frac_bits > 7) {
        int shift_left = net_input_->el_params.fx.frac_bits - 7;
        for (int idx = 0; idx < IN_POINTS; idx++) {
            dst[idx] = (dst[idx] - 128) << shift_left;
        }
    }
    else {
        int shift_right = 7 - net_input_->el_params.fx.frac_bits;
        for (int idx = 0; idx < IN_POINTS; idx++) {
            dst[idx] = (dst[idx] - 128) >> shift_right;
        }
    }
}
static void check_result(
    const char * ir_root,
    const char * ref_file,
    mli_tensor *pred_tsr,
    unsigned cycles,
    mli_status ret_code) {
    if (ret_code != MLI_STATUS_OK) {
        printf("ERROR: MLI Code for %s (%d) is not OK\n", ref_file, ret_code);
        assert(0);
    }

    if (ir_root != NULL) {
        ref_to_pred_output err;
        test_status test_result = measure_ref_to_pred(ir_root, ref_file, *pred_tsr, &err);
        if (test_result == TEST_PASSED) {
            printf("%s: \n\tS/N=%-10.1f (%-4.1f db)\n\tMAX_ABS_ERROR=%-10.1f\n",
                    ref_file,
                    err.ref_vec_length / err.noise_vec_length,
                    err.ref_to_noise_snr,
                    err.max_abs_err);
        }
        else if (test_result == TEST_FAILED) {
            printf("ERROR: Test suit returns FAILD code for %s\n", ref_file);
            assert(0);
        } else {
            printf("%s(w/o IR check):\t%u cycles\n", ref_file, cycles);
        }
    }
}
static void tensor_to_float (const mli_tensor * src, float *dst, uint32_t dst_size) {
    const float scale_val = 1.0f / (float) (1u << (src->el_params.fx.frac_bits));
    if (src->el_type == MLI_EL_FX_16) {
        int16_t *src_arr = src->data.mem.pi16;
        for (int idx = 0; idx < dst_size; idx++)
            dst[idx] = (float) (scale_val * src_arr[idx]);
    } else {
        int8_t *src_arr = src->data.mem.pi8;
        for (int idx = 0; idx < dst_size; idx++)
            dst[idx] = (float) (scale_val * src_arr[idx]);
    }
}


void top_n_pred(int8_t n, char *top_letters, float *top_letters_probs) {
    uint8_t flags[OUT_POINTS] = {0};
    float pred_data[OUT_POINTS] = {0};
    //d_type* const out = (d_type * const)emnist_cf_net_output->data;
    tensor_to_float(emnist_cf_net_output, pred_data, OUT_POINTS);
    for (int top = 0; top < n; top++) {
        float max = -1;
        uint8_t max_idx = -1;
        
        for (int idx = 0; idx < OUT_POINTS; idx++) {
            if(pred_data[idx] > max && flags[idx] != 1) {
                max = pred_data[idx];
                max_idx = idx;
            }
        }

        top_letters[top] = letters[max_idx];
        top_letters_probs[top] = pred_data[max_idx];
        flags[max_idx] = 1;
    }
}

// Initialize the lut for softmax
//==============================================================
mli_status emnist_init() {
	uint32_t lut_size = mli_krn_softmax_get_lut_size();
	if (lut_size > output_lut.data.capacity) {
		return MLI_STATUS_NOT_ENGH_MEM;
	}
	return mli_krn_softmax_create_lut(&output_lut);
}

void all_pred(float *pred_data) {
    tensor_to_float(emnist_cf_net_output, pred_data, OUT_POINTS);
}
static inline mli_status softmax(const mli_tensor *in,  mli_tensor *out) {
    mli_softmax_cfg cfg = {0};
    cfg.axis = -1;
    return mli_krn_softmax_fx16(in, &output_lut, &cfg, out);
}

static const mli_relu_cfg relu_cfg = {.type = MLI_RELU_GEN};
static inline mli_status relu(const mli_tensor *in, const mli_relu_cfg *cfg, mli_tensor *out) {
    return mli_krn_relu_fx16(in, cfg, out);
}

static inline mli_status mli_krn_permute_fx(const mli_tensor *in, const mli_permute_cfg *cfg, mli_tensor *out) {
    return mli_krn_permute_fx16(in, cfg, out);
}

static inline mli_status maxpool_hwcn(const mli_tensor *in, const mli_pool_cfg *cfg, mli_tensor *out) {
    return mli_krn_maxpool_hwc_fx16(in, cfg, out);
}


static inline mli_status conv2d_hwcn(
        const mli_tensor *in,
        const mli_tensor *weights,
        const mli_tensor *bias,
        const mli_conv2d_cfg *cfg,
        mli_tensor *out) {
    return mli_krn_conv2d_hwcn_fx16(in, weights, bias, cfg, out);
}

static inline mli_status fully_connected(
        const mli_tensor *in,
        const mli_tensor *weights,
        const mli_tensor *bias,
        mli_tensor *out) {
    mli_fully_connected_cfg cfg = { 0 };
    cfg.relu.type = MLI_RELU_NONE;
    return mli_krn_fully_connected_fx16(in, weights, bias, &cfg, out);
}

void emnist_cf_net() {

        unsigned cycles = 0;
    
        preprocessing(&input);

        mli_status ret = MLI_STATUS_OK;
        ret = mli_krn_permute_fx(&input, &permute_hwc2chw_cfg, &ir_tensor_Y);
        cycles += cycle_cnt;

#if (MODEL_BIT_DEPTH == MODEL_SA_8)
        set_mli_tensor_params(&ir_tensor_X, -128, 20, QMN(int16_t, 20, 0.027559));
#else
        ir_tensor_X.el_params.fx.frac_bits = CONV1_OUT_FRAQ;
#endif
        set_mli_tensor_shape3(&ir_tensor_X, CONV1_OUT_H, CONV1_OUT_W, CONV1_OUT_C);
        ret = conv2d_hwcn(&ir_tensor_Y, &L1_conv_wt, &L1_conv_bias, &shared_conv_cfg, &ir_tensor_X);
        check_result(debug_ir_root, "ir_acti1.idx", &ir_tensor_X, 0, ret);
    
        set_mli_tensor_shape3(&ir_tensor_Y, CONV1_OUT_H/2, CONV1_OUT_W/2, CONV1_OUT_C);
        ret = maxpool_hwcn(&ir_tensor_X, &shared_pool_cfg, &ir_tensor_Y);
        check_result(debug_ir_root, "ir_pool1.idx", &ir_tensor_Y, 0, ret);
    

#if (MODEL_BIT_DEPTH == MODEL_SA_8)
        set_mli_tensor_params(&ir_tensor_X, -128, 18, QMN(int16_t, 18, 0.08580));
#else
        ir_tensor_X.el_params.fx.frac_bits = CONV2_OUT_FRAQ;
#endif
        set_mli_tensor_shape3(&ir_tensor_X, CONV2_OUT_H, CONV2_OUT_W, CONV2_OUT_C);
        ret = conv2d_hwcn(&ir_tensor_Y, &L2_conv_wt, &L2_conv_bias, &shared_conv_cfg, &ir_tensor_X);
        check_result(debug_ir_root, "ir_acti2.idx", &ir_tensor_X, 0, ret);
    

        set_mli_tensor_shape3(&ir_tensor_Y, CONV2_OUT_H/2, CONV2_OUT_W/2, CONV2_OUT_C);
        ret = maxpool_hwcn(&ir_tensor_X, &shared_pool_cfg, &ir_tensor_Y);
        check_result(debug_ir_root, "ir_pool2.idx", &ir_tensor_Y, 0, ret);
    

#if (MODEL_BIT_DEPTH == MODEL_SA_8)
        set_mli_tensor_params(&ir_tensor_X, -128, 18, QMN(int16_t, 18, 0.10678));
#else
        ir_tensor_X.el_params.fx.frac_bits = CONV3_OUT_FRAQ;
#endif
        set_mli_tensor_shape3(&ir_tensor_X, CONV3_OUT_H, CONV3_OUT_W, CONV3_OUT_C);
        ret = conv2d_hwcn(&ir_tensor_Y, &L3_conv_wt, &L3_conv_bias, &shared_conv_cfg, &ir_tensor_X);
        check_result(debug_ir_root, "ir_acti3.idx", &ir_tensor_X, 0, ret);
    
        set_mli_tensor_shape3(&ir_tensor_Y, CONV3_OUT_H/2, CONV3_OUT_W/2, CONV3_OUT_C);
        ret = maxpool_hwcn(&ir_tensor_X, &shared_pool_cfg, &ir_tensor_Y);
        check_result(debug_ir_root, "ir_pool3.idx", &ir_tensor_Y, 0, ret);
    
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
        set_mli_tensor_params(&ir_tensor_X, -11, 17, QMN(int16_t, 17, 0.149543));
#else
        ir_tensor_X.el_params.fx.frac_bits = FC4_OUT_FRAQ;
#endif
        set_mli_tensor_shape1(&ir_tensor_X, FC4_OUT_SIZE);
        ret = fully_connected(&ir_tensor_Y, &L4_fc_wt, &L4_fc_bias, &ir_tensor_X);
        check_result(debug_ir_root, "ir_dense4.idx", &ir_tensor_X, 0, ret);
    
        ret = relu(&ir_tensor_X, &relu_cfg, &ir_tensor_Y);
        check_result(debug_ir_root, "ir_acti4.idx", &ir_tensor_Y, 0, ret);
    
#if (MODEL_BIT_DEPTH == MODEL_SA_8)
        set_mli_tensor_params(&ir_tensor_X, -11, 17, QMN(int16_t, 17, 0.149543));
#else
        ir_tensor_X.el_params.fx.frac_bits = FC5_OUT_FRAQ;
#endif
        set_mli_tensor_shape1(&ir_tensor_X, FC5_OUT_SIZE);
        ret = fully_connected(&ir_tensor_Y, &L5_fc_wt, &L5_fc_bias, &ir_tensor_X);
        check_result(debug_ir_root, "ir_dense5.idx", &ir_tensor_X, 0, ret);
    
        ret = softmax(&ir_tensor_X, &output);
        check_result(debug_ir_root, "ir_acti5.idx", &output, 0, ret);  
}


