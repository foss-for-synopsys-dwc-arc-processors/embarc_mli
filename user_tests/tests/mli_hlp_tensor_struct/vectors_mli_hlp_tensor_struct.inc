/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <stdint.h>

#include "mli_types.h"
#include "test_tensor_quantizer.h"

using mli::tst::tensor_quantizer;

// Generated input vectors Declaration
//========================================
extern mli::tst::tensor_quantizer input_1_fx16;
extern mli::tst::tensor_quantizer input_1_fx8;
extern mli::tst::tensor_quantizer input_1_sa8;
extern mli::tst::tensor_quantizer input_1_sa8_per_axis;
extern mli::tst::tensor_quantizer input_1_sa32;
extern mli::tst::tensor_quantizer input_1_sa32_per_axis;
extern mli::tst::tensor_quantizer input_1_fp32;

extern mli::tst::tensor_quantizer input_2_fx16;
extern mli::tst::tensor_quantizer input_2_fx8;
extern mli::tst::tensor_quantizer input_2_sa8;
extern mli::tst::tensor_quantizer input_2_sa32;
extern mli::tst::tensor_quantizer input_2_fp32;

// Extracted Output vectors Declaration
//===================================================
extern mli::tst::tensor_quantizer test_1_out_fx16;
extern mli::tst::tensor_quantizer test_1_out_fx8;
extern mli::tst::tensor_quantizer test_1_out_sa8;
extern mli::tst::tensor_quantizer test_1_out_sa8_per_axis;
extern mli::tst::tensor_quantizer test_1_out_sa32;
extern mli::tst::tensor_quantizer test_1_out_sa32_per_axis;
extern mli::tst::tensor_quantizer test_1_out_fp32;

extern mli::tst::tensor_quantizer test_2_out_fx16;
extern mli::tst::tensor_quantizer test_2_out_fx8;
extern mli::tst::tensor_quantizer test_2_out_sa8;
extern mli::tst::tensor_quantizer test_2_out_sa8_per_axis;
extern mli::tst::tensor_quantizer test_2_out_sa32;
extern mli::tst::tensor_quantizer test_2_out_sa32_per_axis;
extern mli::tst::tensor_quantizer test_2_out_fp32;

// Tests configuration structures Declaration
//========================================
extern const mli_sub_tensor_cfg test_1_cfg;
extern const mli_sub_tensor_cfg test_2_cfg;

// Generated input vectors 
//========================================
static const float input_1_data[] = {
    0.061785f, -0.247152f, -0.050968f, 0.193333f, -0.010555f, -0.457387f, 0.091005f, 0.029531f, -0.131128f, -0.124796f, 
    0.038729f, 0.318163f, 0.018775f, 0.000024f, 0.065362f, 0.106736f, -0.298480f, 0.330821f, 0.227728f, 0.175145f, 
    -0.454469f, -0.038041f, 0.131463f, 0.102645f, 0.082620f, -0.190452f, 0.063674f, -0.016948f, -0.240415f, 0.255931f, 
    -0.091566f, -0.081387f, -0.367663f, 0.105258f, 0.239669f, 0.527339f
};

static const float input_1_scale = 0.004152277f;
static const float input_1_zero_point = 0.0f;
static const int8_t input_1_scale_frac[] = {22};
static const int input_1_sa_dim = -1;

static const float input_1_scales[] = {0.0017931332f, 0.0023502372f, 0.004152277f};
static const float input_1_zero_points[] = {0.0f, 0.0f, 0.0f};
static const int8_t input_1_scales_frac[] = {22, 23, 21};
static const int input_1_sa_dim_per_axis = 3;

static const int8_t input_1_fx8_frac = 7;
static const int8_t input_1_fx16_frac = 15;
static const int8_t input_1_fp_exp_frac = 0;
static const float input_1_fx8_exp_scale = 1.f / 128.f;
static const float input_1_fx16_exp_scale = 1.f / 32768.f;
static const float input_1_fp_exp_scale = 1.f;
static const float input_1_fx_fp_exp_zero = 0.f;

#define INPUT_1_ELEM_COUNT {4*3*1*3, 3*1*3, 3*1, 3}

#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(3 * (3 * 2))  * 2, 3 * 2, 3, 1}, \
    /* .shape = */ {4, 3, 1, 3}, \
    /* .rank =  */ 4

static const mli_tensor input_1_tsr_fx16 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_1_tsr_fx8 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_1_tsr_sa8 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_1_tsr_sa32 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_1_tsr_fp32 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FP_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx16_frac, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]));

tensor_quantizer input_1_fx8(input_1_tsr_fx8, input_1_fx8_frac, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]));

tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, 
    &input_1_zero_point, 1, input_1_scale_frac, 1);

tensor_quantizer input_1_sa8_per_axis(input_1_tsr_sa8, input_1_sa_dim_per_axis, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]), input_1_scales, 
    sizeof(input_1_scales) / sizeof(input_1_scales[0]), input_1_zero_points, 
    sizeof(input_1_zero_points) / sizeof(input_1_zero_points[0]), input_1_scales_frac, 
    sizeof(input_1_scales_frac) / sizeof(input_1_scales_frac[0]));

tensor_quantizer input_1_sa32(input_1_tsr_sa32, input_1_sa_dim, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, 
    &input_1_zero_point, 1, input_1_scale_frac, 1);

tensor_quantizer input_1_sa32_per_axis(input_1_tsr_sa32, input_1_sa_dim_per_axis, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]), input_1_scales, 
    sizeof(input_1_scales) / sizeof(input_1_scales[0]), input_1_zero_points, 
    sizeof(input_1_zero_points) / sizeof(input_1_zero_points[0]), input_1_scales_frac, 
    sizeof(input_1_scales_frac) / sizeof(input_1_scales_frac[0]));

tensor_quantizer input_1_fp32(input_1_tsr_fp32, (float*)input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));


static const float input_2_data[] = { -0.42424242f };

static const float input_2_scale = 0.003340491f;
static const float input_2_zero_point = 0.0f;
static const int8_t input_2_scale_frac[] = { 23 };
static const int input_2_sa_dim = -1;
static const int input_2_fx8_frac = 7;
static const int input_2_fx16_frac = 15;

#define INPUT_2_ELEM_COUNT {1}

#define TEST_2_INPUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ { 0 }, \
    /* .rank = */  0

static const mli_tensor input_2_tsr_fx16 = {
    TEST_2_INPUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_2_tsr_fx8 = {
    TEST_2_INPUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_2_tsr_sa8 = {
    TEST_2_INPUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_2_tsr_sa32 = {
    TEST_2_INPUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_2_tsr_fp32 = {
    TEST_2_INPUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FP_32,
    /* .el_params = */ { 0 }
};


tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx16_frac, input_2_data, 
    sizeof(input_2_data) / sizeof(input_2_data[0]));

tensor_quantizer input_2_fx8(input_2_tsr_fx8, input_2_fx8_frac, input_2_data, 
    sizeof(input_2_data) / sizeof(input_2_data[0]));

tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data, 
    sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1, 
    &input_2_zero_point, 1, input_2_scale_frac, 1);

tensor_quantizer input_2_sa32(input_2_tsr_sa32, input_2_sa_dim, input_2_data, 
    sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1, 
    &input_2_zero_point, 1, input_2_scale_frac, 1);

tensor_quantizer input_2_fp32(input_2_tsr_fp32, (float*)input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]));

// Extracted Output vectors 
//===================================================

static const float test_1_out_data[] = {
    0.061785f, -0.247152f, 0.193333f, -0.010555f, -0.124796f, 0.038729f, 0.018775f, 0.000024f
};

static const float test_1_out_scales[] = {0.0017931332f, 0.0023502372f};
static const float test_1_out_zero_points[] = {0.0f, 0.0f};
static const int8_t test_1_out_scales_frac[] = {22, 23};
static const int test_1_out_sa_dim_per_axis = 3;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(3 * (3 * 2))  * 2, 3 * 2, 3, 1}, \
    /* .shape = */ {2, 2, 1, 2}, \
    /* .rank =  */ 4

static const mli_tensor test_1_out_tsr_fx16 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor test_1_out_tsr_fx8 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor test_1_out_tsr_sa8 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor test_1_out_tsr_sa32 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const mli_tensor test_1_out_tsr_fp32 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FP_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, input_1_fx16_frac, test_1_out_data, 
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));

tensor_quantizer test_1_out_fx8(test_1_out_tsr_fx8, input_1_fx8_frac, test_1_out_data, 
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));

tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, input_1_sa_dim, test_1_out_data, 
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &input_1_scale, 1, 
    &input_1_zero_point, 1, input_1_scale_frac, 1);

tensor_quantizer test_1_out_sa8_per_axis(test_1_out_tsr_sa8, test_1_out_sa_dim_per_axis, test_1_out_data, 
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), test_1_out_scales, 
    sizeof(test_1_out_scales) / sizeof(test_1_out_scales[0]), test_1_out_zero_points, 
    sizeof(test_1_out_zero_points) / sizeof(test_1_out_zero_points[0]), test_1_out_scales_frac, 
    sizeof(test_1_out_scales_frac) / sizeof(test_1_out_scales_frac[0]));

tensor_quantizer test_1_out_sa32(test_1_out_tsr_sa32, input_1_sa_dim, test_1_out_data, 
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &input_1_scale, 1, 
    &input_1_zero_point, 1, input_1_scale_frac, 1);

tensor_quantizer test_1_out_sa32_per_axis(test_1_out_tsr_sa32, test_1_out_sa_dim_per_axis, test_1_out_data, 
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), test_1_out_scales, 
    sizeof(test_1_out_scales) / sizeof(test_1_out_scales[0]), test_1_out_zero_points, 
    sizeof(test_1_out_zero_points) / sizeof(test_1_out_zero_points[0]), test_1_out_scales_frac, 
    sizeof(test_1_out_scales_frac) / sizeof(test_1_out_scales_frac[0]));



static const float test_2_out_data[] = {
    -0.298480f, 0.330821f, -0.190452f, 0.063674f, 0.239669f, 0.527339f
};

static const float test_2_out_scales[] = {0.0023502372f, 0.004152277f};
static const float test_2_out_zero_points[] = {0.0f, 0.0f};
static const int8_t test_2_out_scales_frac[] = {23, 21};
static const int test_2_out_sa_dim_per_axis = 2;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 3 * 2, 3, 1 }, \
    /* .shape = */ {3, 1, 2}, \
    /* .rank =  */ 3
  
static const mli_tensor test_2_out_tsr_fx16 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor test_2_out_tsr_fx8 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor test_2_out_tsr_sa8 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor test_2_out_tsr_sa32 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const mli_tensor test_2_out_tsr_fp32 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FP_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, input_1_fx16_frac, test_2_out_data, 
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));

tensor_quantizer test_2_out_fx8(test_2_out_tsr_fx8, input_1_fx8_frac, test_2_out_data, 
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));

tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, input_1_sa_dim, test_2_out_data, 
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &input_1_scale, 1, 
    &input_1_zero_point, 1, input_1_scale_frac, 1);

tensor_quantizer test_2_out_sa8_per_axis(test_2_out_tsr_sa8, test_2_out_sa_dim_per_axis, test_2_out_data, 
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), test_2_out_scales, 
    sizeof(test_2_out_scales) / sizeof(test_2_out_scales[0]), test_2_out_zero_points, 
    sizeof(test_2_out_zero_points) / sizeof(test_2_out_zero_points[0]), test_2_out_scales_frac, 
    sizeof(test_2_out_scales_frac) / sizeof(test_2_out_scales_frac[0]));

tensor_quantizer test_2_out_sa32(test_2_out_tsr_sa32, input_1_sa_dim, test_2_out_data, 
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &input_1_scale, 1, 
    &input_1_zero_point, 1, input_1_scale_frac, 1);

tensor_quantizer test_2_out_sa32_per_axis(test_2_out_tsr_sa32, test_2_out_sa_dim_per_axis, test_2_out_data, 
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), test_2_out_scales, 
    sizeof(test_2_out_scales) / sizeof(test_2_out_scales[0]), test_2_out_zero_points, 
    sizeof(test_2_out_zero_points) / sizeof(test_2_out_zero_points[0]), test_2_out_scales_frac, 
    sizeof(test_2_out_scales_frac) / sizeof(test_2_out_scales_frac[0]));



// Tests configuration structures
//========================================

const mli_sub_tensor_cfg test_1_cfg = {
    /* offset = */ { 0 },                
    /* size = */ {2, 2, 1, 2}, 
    /* sub_tensor_rank = */ 4    
};

const mli_sub_tensor_cfg test_2_cfg = {
    /* offset = */ { 1, 2, 0, 1 },                
    /* size = */ { 3, 1, 1, 2 }, 
    /* sub_tensor_rank = */ 3
};