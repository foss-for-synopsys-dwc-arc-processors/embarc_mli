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
extern mli::tst::tensor_quantizer input_1_sa8;

extern mli::tst::tensor_quantizer input_2_fx16;
extern mli::tst::tensor_quantizer input_2_sa8;

extern mli::tst::tensor_quantizer input_3_fx16;
extern mli::tst::tensor_quantizer input_3_sa8;

extern mli::tst::tensor_quantizer input_4_fx16;
extern mli::tst::tensor_quantizer input_4_sa8;

extern mli::tst::tensor_quantizer input_5_fx16;
extern mli::tst::tensor_quantizer input_5_sa8;

// Generated weights vectors Declaration
//========================================
extern mli::tst::tensor_quantizer weights_1_fx16;
extern mli::tst::tensor_quantizer weights_1_memstr_fx16;
extern mli::tst::tensor_quantizer weights_1_fx8;
extern mli::tst::tensor_quantizer weights_1_memstr_fx8;
extern mli::tst::tensor_quantizer weights_1_sa8;
extern mli::tst::tensor_quantizer weights_1_memstr_sa8;

extern mli::tst::tensor_quantizer weights_2_fx16;
extern mli::tst::tensor_quantizer weights_2_memstr_fx16;
extern mli::tst::tensor_quantizer weights_2_fx8;
extern mli::tst::tensor_quantizer weights_2_memstr_fx8;
extern mli::tst::tensor_quantizer weights_2_sa8;
extern mli::tst::tensor_quantizer weights_2_memstr_sa8;

extern mli::tst::tensor_quantizer weights_3_fx16;
extern mli::tst::tensor_quantizer weights_3_memstr_fx16;
extern mli::tst::tensor_quantizer weights_3_fx8;
extern mli::tst::tensor_quantizer weights_3_memstr_fx8;
extern mli::tst::tensor_quantizer weights_3_sa8;
extern mli::tst::tensor_quantizer weights_3_memstr_sa8;

extern mli::tst::tensor_quantizer weights_4_fx16;
extern mli::tst::tensor_quantizer weights_4_memstr_fx16;
extern mli::tst::tensor_quantizer weights_4_fx8;
extern mli::tst::tensor_quantizer weights_4_memstr_fx8;
extern mli::tst::tensor_quantizer weights_4_sa8;
extern mli::tst::tensor_quantizer weights_4_memstr_sa8;

extern mli::tst::tensor_quantizer weights_5_fx16;
extern mli::tst::tensor_quantizer weights_5_memstr_fx16;
extern mli::tst::tensor_quantizer weights_5_fx8;
extern mli::tst::tensor_quantizer weights_5_memstr_fx8;
extern mli::tst::tensor_quantizer weights_5_sa8;
extern mli::tst::tensor_quantizer weights_5_memstr_sa8;

extern mli::tst::tensor_quantizer weights_6_fx16;
extern mli::tst::tensor_quantizer weights_6_memstr_fx16;
extern mli::tst::tensor_quantizer weights_6_fx8;
extern mli::tst::tensor_quantizer weights_6_memstr_fx8;
extern mli::tst::tensor_quantizer weights_6_sa8;
extern mli::tst::tensor_quantizer weights_6_memstr_sa8;

extern mli::tst::tensor_quantizer weights_7_fx16;
extern mli::tst::tensor_quantizer weights_7_memstr_fx16;
extern mli::tst::tensor_quantizer weights_7_fx8;
extern mli::tst::tensor_quantizer weights_7_memstr_fx8;
extern mli::tst::tensor_quantizer weights_7_sa8;
extern mli::tst::tensor_quantizer weights_7_memstr_sa8;

extern mli::tst::tensor_quantizer weights_8_fx16;
extern mli::tst::tensor_quantizer weights_8_memstr_fx16;
extern mli::tst::tensor_quantizer weights_8_fx8;
extern mli::tst::tensor_quantizer weights_8_memstr_fx8;
extern mli::tst::tensor_quantizer weights_8_sa8;
extern mli::tst::tensor_quantizer weights_8_memstr_sa8;

extern mli::tst::tensor_quantizer weights_9_fx16;
extern mli::tst::tensor_quantizer weights_9_memstr_fx16;
extern mli::tst::tensor_quantizer weights_9_fx8;
extern mli::tst::tensor_quantizer weights_9_memstr_fx8;
extern mli::tst::tensor_quantizer weights_9_sa8;
extern mli::tst::tensor_quantizer weights_9_memstr_sa8;

extern mli::tst::tensor_quantizer bias_1_fx16;
extern mli::tst::tensor_quantizer bias_1_fx8;
extern mli::tst::tensor_quantizer bias_1_i1_w1_sa32;

extern mli::tst::tensor_quantizer bias_2_fx16;
extern mli::tst::tensor_quantizer bias_2_fx8;
extern mli::tst::tensor_quantizer bias_2_i3_w3_sa32;

extern mli::tst::tensor_quantizer bias_3_fx16;
extern mli::tst::tensor_quantizer bias_3_fx8;
extern mli::tst::tensor_quantizer bias_3_i3_w6_sa32;

// Extracted Output vectors Declaration
//===================================================
extern mli::tst::tensor_quantizer test_1_out_fx16;
extern mli::tst::tensor_quantizer test_1_out_sa8;
extern mli::tst::tensor_quantizer test_2_out_fx16;
extern mli::tst::tensor_quantizer test_2_out_sa8;
extern mli::tst::tensor_quantizer test_3_out_fx16;
extern mli::tst::tensor_quantizer test_3_out_sa8;

// Tests configuration structures Declaration
//========================================
extern const mli_rnn_dense_cfg test_1_cfg;
extern const mli_rnn_dense_cfg test_2_cfg;
extern const mli_rnn_dense_cfg test_3_cfg;


// Generated vectors 
//========================================
static const float input_1_data[] = {
    -0.073981f, -0.393418f, -0.046866f, -0.256056f, -0.168622f, 0.024018f, 0.312583f, 0.40492f, -0.128252f, -0.195426f, 
    0.331349f, -0.02585f, -0.210009f, -0.228304f, 0.007194f, -0.234319f
};

static const float input_1_scale = 0.0040117498f;
static const float input_1_zero_point = 0.0057512522f;
static const int8_t input_1_scales_frac[] = {22};
static const int input_1_sa_dim = -1;

static const int input_1_fx8_frac = 8;

#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 8}, \
    /* .rank =  */ 2

static const mli_tensor input_1_tsr_fx16 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_1_tsr_sa8 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);


static const float input_2_data[] = {
    -0.952562f, -0.99843f, -0.801853f, 0.470273f, -0.983646f, 0.139618f, 0.02511f, -0.998669f
};

static const float input_2_scale = 0.0148377959f;
static const float input_2_zero_point = -0.2641976774f;
static const int8_t input_2_scales_frac[] = {21};
static const int input_2_sa_dim = -1;

static const int input_2_fx8_frac = 7;

#define INPUT_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 1, 2, 4}, \
    /* .rank =  */ 4

static const mli_tensor input_2_tsr_fx16 = {
    INPUT_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_2_tsr_sa8 = {
    INPUT_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx8_frac + 8, input_2_data, 
    sizeof(input_2_data) / sizeof(input_2_data[0]));
tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data, 
    sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);


static const float input_3_data[] = {
    -0.076125f, -0.318358f, 0.019112f, 0.283709f, -0.304293f, -0.265468f, -0.058526f, 0.395846f, 0.22952f, -0.22914f, 
    -0.367296f, 0.086669f
};

static const float input_3_scale = 0.0038348811f;
static const float input_3_zero_point = 0.0142749846f;
static const int8_t input_3_scales_frac[] = {23};
static const int input_3_sa_dim = -1;

static const int input_3_fx8_frac = 8;

#define INPUT_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {12}, \
    /* .rank =  */ 1

static const mli_tensor input_3_tsr_fx16 = {
    INPUT_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_3_tsr_sa8 = {
    INPUT_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_3_fx16(input_3_tsr_fx16, input_3_fx8_frac + 8, input_3_data, 
    sizeof(input_3_data) / sizeof(input_3_data[0]));
tensor_quantizer input_3_sa8(input_3_tsr_sa8, input_3_sa_dim, input_3_data, 
    sizeof(input_3_data) / sizeof(input_3_data[0]), &input_3_scale, 1, &input_3_zero_point, 1, input_3_scales_frac, 1);


static const float input_4_data[] = {
    0.314014f, -0.448195f, 0.038448f, -0.131283f, -0.152695f, -0.154024f, 0.22274f, 0.311522f, 0.134357f, 0.304217f
};

static const float input_4_scale = 0.0038301926f;
static const float input_4_zero_point = -0.0670904517f;
static const int8_t input_4_scales_frac[] = {23};
static const int input_4_sa_dim = -1;

static const int input_4_fx8_frac = 8;

#define INPUT_4_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {10}, \
    /* .rank =  */ 1

static const mli_tensor input_4_tsr_fx16 = {
    INPUT_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_4_tsr_sa8 = {
    INPUT_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_4_fx16(input_4_tsr_fx16, input_4_fx8_frac + 8, input_4_data, 
    sizeof(input_4_data) / sizeof(input_4_data[0]));
tensor_quantizer input_4_sa8(input_4_tsr_sa8, input_4_sa_dim, input_4_data, 
    sizeof(input_4_data) / sizeof(input_4_data[0]), &input_4_scale, 1, &input_4_zero_point, 1, input_4_scales_frac, 1);


static const float input_5_data[] = {
    0.996371f, 0.075272f, 0.043633f, 0.715202f, 0.958081f, 0.289251f, -0.778187f, -0.354781f, -0.715274f, 
    0.906402f, 0.999899f
};

static const float input_5_scale = 0.0089351051f;
static const float input_5_zero_point = 0.1108561158f;
static const int8_t input_5_scales_frac[] = {21};
static const int input_5_sa_dim = -1;

static const int input_5_fx8_frac = 7;

#define INPUT_5_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {11}, \
    /* .rank =  */ 1

static const mli_tensor input_5_tsr_fx16 = {
    INPUT_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_5_tsr_sa8 = {
    INPUT_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_5_fx16(input_5_tsr_fx16, input_5_fx8_frac + 8, input_5_data, 
    sizeof(input_5_data) / sizeof(input_5_data[0]));
tensor_quantizer input_5_sa8(input_5_tsr_sa8, input_5_sa_dim, input_5_data, 
    sizeof(input_5_data) / sizeof(input_5_data[0]), &input_5_scale, 1, &input_5_zero_point, 1, input_5_scales_frac, 1);


static const float weights_1_data[] = {
    -0.872902f, 0.283538f, -0.886071f, -0.094742f, -0.183509f, 0.770262f, -0.454453f, -0.532863f, 0.81164f, 0.831813f, 
    -0.709488f, 0.136591f, -0.553718f, 0.584037f, -0.991321f, 0.493198f, -0.655145f, -0.677211f, 0.966649f, 0.480896f, 
    0.819295f, -0.926538f, -0.014077f, 0.774531f, -0.310206f, 0.084446f, -0.615242f, 0.927212f, 0.545517f, -0.148083f, 
    0.320255f, -0.828719f, 0.896024f, -0.134106f, 0.692385f, 0.088915f, 0.259131f, -0.71838f, 0.624308f, 0.629463f, 
    0.597316f, -0.055897f, -0.78733f, -0.101873f, 0.528092f, 0.526355f, 0.086753f, 0.77831f, -0.458645f, 0.493737f, 
    -0.797352f, -0.512717f, 0.205136f, 0.625629f, 0.430907f, -0.648192f, -0.174327f, -0.554505f, 0.58488f, 0.470591f, 
    -0.332807f, 0.65744f, 0.566606f, -0.033807f, -0.285517f, 0.451981f, -0.285224f, -0.667111f, -0.109348f, -0.746968f, 
    -0.11036f, 0.381027f, -0.679044f, 0.883652f, -0.667485f, 0.140246f, -0.989824f, 0.672941f, -0.339687f, -0.973294f, 
    0.024519f, -0.880346f, -0.612857f, 0.857061f, 0.207116f, -0.033765f, -0.453525f, 0.801123f, 0.998073f, -0.355006f, 
    0.085517f, 0.143042f, 0.590336f, -0.71351f, -0.67329f, -0.158132f, -0.971189f, -0.879448f, 0.228875f, 0.098749f, 
    0.132038f, 0.965143f, 0.008677f, 0.566014f, -0.077979f, -0.847934f, 0.183069f, 0.010655f, -0.405113f, 0.871698f, 
    -0.41775f, 0.525155f, -0.058603f, 0.701131f, 0.915945f, -0.023276f, -0.824718f, -0.508782f, 0.692749f, -0.919684f, 
    -0.638924f, -0.006848f, -0.601909f, 0.314812f, 0.197433f, -0.866103f, 0.662088f, -0.070292f
};

static const float weights_1_scale = 0.0100308834f;
static const float weights_1_zero_point = 0.f;
static const int8_t weights_1_scales_frac[] = {21};
static const int weights_1_sa_dim = -1;

static const int weights_1_fx8_frac = 7;

#define WEIGHTS_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {16, 8}, \
    /* .rank =  */ 2

#define WEIGHTS_1_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {1 * 10, 1}, \
    /* .shape = */ {16, 8}, \
    /* .rank =  */ 2

static const mli_tensor weights_1_tsr_fx16 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_1_memstr_tsr_fx16 = {
    WEIGHTS_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_1_tsr_fx8 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_1_memstr_tsr_fx8 = {
    WEIGHTS_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_1_tsr_sa8 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_1_memstr_tsr_sa8 = {
    WEIGHTS_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_1_fx16(weights_1_tsr_fx16, weights_1_fx8_frac + 8, weights_1_data, 
    sizeof(weights_1_data) / sizeof(weights_1_data[0]));
tensor_quantizer weights_1_memstr_fx16(weights_1_memstr_tsr_fx16, weights_1_fx8_frac + 8, weights_1_data, 
    sizeof(weights_1_data) / sizeof(weights_1_data[0]));
tensor_quantizer weights_1_fx8(weights_1_tsr_fx8, weights_1_fx8_frac, weights_1_data, 
    sizeof(weights_1_data) / sizeof(weights_1_data[0]));
tensor_quantizer weights_1_memstr_fx8(weights_1_memstr_tsr_fx8, weights_1_fx8_frac, weights_1_data, 
    sizeof(weights_1_data) / sizeof(weights_1_data[0]));
tensor_quantizer weights_1_sa8(weights_1_tsr_sa8, weights_1_sa_dim, weights_1_data, 
    sizeof(weights_1_data) / sizeof(weights_1_data[0]), &weights_1_scale, 1, &weights_1_zero_point, 1, 
    weights_1_scales_frac, 1);
tensor_quantizer weights_1_memstr_sa8(weights_1_memstr_tsr_sa8, weights_1_sa_dim, weights_1_data, 
    sizeof(weights_1_data) / sizeof(weights_1_data[0]), &weights_1_scale, 1, &weights_1_zero_point, 1, 
    weights_1_scales_frac, 1);


static const float weights_2_data[] = {
    -0.880469f, 0.481715f, -0.582038f, 0.625354f, 0.014959f, 0.281606f, 0.519045f, 0.170123f, -0.093325f, -0.971667f, 
    -0.386996f, -0.603977f, 0.286631f, 0.667821f, -0.261829f, 0.54141f, -0.43002f, 0.085242f, 0.122564f, -0.282372f, 
    -0.210558f, 0.384373f, 0.137122f, 0.393832f, -0.994149f, 0.620714f, 0.617644f, 0.213706f, -0.260536f, 0.526041f, 
    -0.934093f, 0.663371f, -0.542811f, 0.223351f, -0.030592f, 0.739394f, -0.024723f, 0.946724f, -0.256401f, 0.088833f, 
    -0.218757f, 0.744111f, -0.199538f, 0.469036f, -0.833373f, -0.820712f, -0.481447f, 0.766668f, 0.29619f, 0.170564f, 
    0.848644f, -0.093552f, -0.065663f, 0.865838f, -0.484487f, -0.547828f, 0.958263f, -0.604791f, -0.699281f, 0.2972f, 
    -0.033738f, 0.057846f, -0.545146f, -0.734215f
};

static const float weights_2_scale = 0.0099914465f;
static const float weights_2_zero_point = 0.f;
static const int8_t weights_2_scales_frac[] = {21};
static const int weights_2_sa_dim = -1;

static const int weights_2_fx8_frac = 7;

#define WEIGHTS_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {8, 8}, \
    /* .rank =  */ 2

#define WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {1 * 10, 1}, \
    /* .shape = */ {8, 8}, \
    /* .rank =  */ 2

static const mli_tensor weights_2_tsr_fx16 = {
    WEIGHTS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_2_memstr_tsr_fx16 = {
    WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_2_tsr_fx8 = {
    WEIGHTS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_2_memstr_tsr_fx8 = {
    WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_2_tsr_sa8 = {
    WEIGHTS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_2_memstr_tsr_sa8 = {
    WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_2_fx16(weights_2_tsr_fx16, weights_2_fx8_frac + 8, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_memstr_fx16(weights_2_memstr_tsr_fx16, weights_2_fx8_frac + 8, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_fx8(weights_2_tsr_fx8, weights_2_fx8_frac, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_memstr_fx8(weights_2_memstr_tsr_fx8, weights_2_fx8_frac, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_sa8(weights_2_tsr_sa8, weights_2_sa_dim, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]), &weights_2_scale, 1, &weights_2_zero_point, 1, 
    weights_2_scales_frac, 1);
tensor_quantizer weights_2_memstr_sa8(weights_2_memstr_tsr_sa8, weights_2_sa_dim, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]), &weights_2_scale, 1, &weights_2_zero_point, 1, 
    weights_2_scales_frac, 1);


static const float weights_3_data[] = {
    0.015026f, -0.604522f, 0.972991f, 0.178238f, 0.012757f, -0.490461f, 0.034445f, 0.075015f, -0.420235f, 0.024611f, 
    -0.628725f, 0.283891f, -0.15452f, 0.066271f, 0.841047f, 0.155695f, -0.022805f, 0.542771f, 0.252279f, 0.662706f, 
    -0.627041f, 0.793147f, 0.232069f, -0.741373f, 0.16466f, -0.615994f, 0.913126f, -0.378348f, -0.255843f, -0.211288f, 
    -0.421016f, -0.234961f, 0.221957f, -0.166632f, -0.870848f, -0.337083f, -0.622263f, 0.669093f, 0.887145f, 0.672694f, 
    -0.668846f, 0.061069f, -0.38573f, -0.152247f, 0.601258f, -0.190988f, 0.724209f, -0.800925f, 0.62501f, -0.866201f, 
    0.495877f, -0.891299f, -0.060686f, 0.940005f, -0.933861f, 0.297426f, 0.314375f, -0.363516f, -0.757311f, -0.943266f, 
    -0.276722f, -0.183053f, 0.865775f, -0.356736f, -0.410069f, -0.297989f, -0.365746f, -0.215597f, 0.684491f, 0.609358f, 
    -0.159429f, -0.383429f, -0.147451f, 0.633353f, 0.868655f, 0.14916f, 0.574434f, 0.263585f, -0.588772f, 0.559462f, 
    -0.607289f, -0.204487f, 0.319349f, -0.48442f, 0.638781f, -0.439422f, -0.497268f, 0.697713f, -0.01569f, 0.751421f, 
    0.358696f, 0.403405f, -0.777839f, -0.939702f, 0.826571f, -0.167634f, -0.819267f, -0.44593f, 0.831926f, -0.97175f, 
    0.664126f, -0.816574f, -0.538096f, 0.040512f, -0.291409f, 0.226655f, 0.494502f, -0.962663f, 0.980261f, 0.641164f, 
    -0.678284f, -0.005294f, 0.973382f, 0.24777f, 0.186091f, 0.387701f, -0.421456f, 0.293454f, 0.284007f, 0.931364f
};

static const float weights_3_scale = 0.0098518683f;
static const float weights_3_zero_point = 0.f;
static const int8_t weights_3_scales_frac[] = {21};
static const int weights_3_sa_dim = -1;

static const int weights_3_fx8_frac = 7;

#define WEIGHTS_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {12, 10}, \
    /* .rank =  */ 2

#define WEIGHTS_3_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {1 * 12, 1}, \
    /* .shape = */ {12, 10}, \
    /* .rank =  */ 2

static const mli_tensor weights_3_tsr_fx16 = {
    WEIGHTS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_3_memstr_tsr_fx16 = {
    WEIGHTS_3_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_3_tsr_fx8 = {
    WEIGHTS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_3_memstr_tsr_fx8 = {
    WEIGHTS_3_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_3_tsr_sa8 = {
    WEIGHTS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_3_memstr_tsr_sa8 = {
    WEIGHTS_3_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_3_fx16(weights_3_tsr_fx16, weights_3_fx8_frac + 8, weights_3_data, 
    sizeof(weights_3_data) / sizeof(weights_3_data[0]));
tensor_quantizer weights_3_memstr_fx16(weights_3_memstr_tsr_fx16, weights_3_fx8_frac + 8, weights_3_data, 
    sizeof(weights_3_data) / sizeof(weights_3_data[0]));
tensor_quantizer weights_3_fx8(weights_3_tsr_fx8, weights_3_fx8_frac, weights_3_data, 
    sizeof(weights_3_data) / sizeof(weights_3_data[0]));
tensor_quantizer weights_3_memstr_fx8(weights_3_memstr_tsr_fx8, weights_3_fx8_frac, weights_3_data, 
    sizeof(weights_3_data) / sizeof(weights_3_data[0]));
tensor_quantizer weights_3_sa8(weights_3_tsr_sa8, weights_3_sa_dim, weights_3_data, 
    sizeof(weights_3_data) / sizeof(weights_3_data[0]), &weights_3_scale, 1, &weights_3_zero_point, 1, 
    weights_3_scales_frac, 1);
tensor_quantizer weights_3_memstr_sa8(weights_3_memstr_tsr_sa8, weights_3_sa_dim, weights_3_data, 
    sizeof(weights_3_data) / sizeof(weights_3_data[0]), &weights_3_scale, 1, &weights_3_zero_point, 1, 
    weights_3_scales_frac, 1);


static const float weights_4_data[] = {
    0.612769f, -0.160872f, -0.447303f, 0.754497f, 0.312195f, -0.511806f, 0.296291f, -0.67721f, 0.765434f, 0.471681f, 
    -0.133281f, -0.651817f, -0.597316f, 0.417499f, -0.825452f, 0.309209f, -0.004455f, 0.118624f, 0.698215f, 0.129028f, 
    -0.155313f, 0.780785f, 0.539934f, 0.018578f, 0.80025f, -0.022086f, -0.821528f, -0.25381f, 0.907979f, -0.643602f, 
    -0.827321f, -0.839362f, 0.942719f, 0.305953f, 0.674378f, 0.768928f, 0.208686f, 0.678803f, -0.950114f, -0.104785f, 
    -0.662878f, 0.585382f, 0.953455f, 0.825913f, 0.614613f, 0.642426f, 0.218471f, 0.434546f, 0.093401f, 0.953946f, 
    -0.748132f, -0.759824f, -0.341094f, 0.151738f, 0.287715f, -0.03519f, -0.166817f, -0.632708f, 0.345924f, 0.142081f, 
    0.392096f, -0.825497f, -0.971186f, 0.569363f, -0.292958f, -0.152954f, -0.217273f, 0.311386f, -0.625829f, 0.964166f, 
    0.53799f, 0.272196f, -0.373703f, -0.1318f, 0.068349f, 0.85092f, -0.215687f, -0.938263f, 0.641931f, 0.470001f, 
    -0.625628f, -0.176997f, 0.243189f, -0.072861f, -0.268354f, 0.979128f, 0.759063f, 0.096839f, 0.598179f, -0.904727f, 
    0.181323f, -0.259034f, 0.907931f, 0.251817f, -0.379035f, 0.128546f, -0.383206f, -0.248289f, 0.33665f, 0.162395f
};

static const float weights_4_scale = 0.0098404791f;
static const float weights_4_zero_point = 0.f;
static const int8_t weights_4_scales_frac[] = {21};
static const int weights_4_sa_dim = -1;

static const int weights_4_fx8_frac = 7;

#define WEIGHTS_4_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {10, 10}, \
    /* .rank =  */ 2

#define WEIGHTS_4_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {1 * 12, 1}, \
    /* .shape = */ {10, 10}, \
    /* .rank =  */ 2

static const mli_tensor weights_4_tsr_fx16 = {
    WEIGHTS_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_4_memstr_tsr_fx16 = {
    WEIGHTS_4_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_4_tsr_fx8 = {
    WEIGHTS_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_4_memstr_tsr_fx8 = {
    WEIGHTS_4_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_4_tsr_sa8 = {
    WEIGHTS_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_4_memstr_tsr_sa8 = {
    WEIGHTS_4_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_4_fx16(weights_4_tsr_fx16, weights_4_fx8_frac + 8, weights_4_data, 
    sizeof(weights_4_data) / sizeof(weights_4_data[0]));
tensor_quantizer weights_4_memstr_fx16(weights_4_memstr_tsr_fx16, weights_4_fx8_frac + 8, weights_4_data, 
    sizeof(weights_4_data) / sizeof(weights_4_data[0]));
tensor_quantizer weights_4_fx8(weights_4_tsr_fx8, weights_4_fx8_frac, weights_4_data, 
    sizeof(weights_4_data) / sizeof(weights_4_data[0]));
tensor_quantizer weights_4_memstr_fx8(weights_4_memstr_tsr_fx8, weights_4_fx8_frac, weights_4_data, 
    sizeof(weights_4_data) / sizeof(weights_4_data[0]));
tensor_quantizer weights_4_sa8(weights_4_tsr_sa8, weights_4_sa_dim, weights_4_data, 
    sizeof(weights_4_data) / sizeof(weights_4_data[0]), &weights_4_scale, 1, &weights_4_zero_point, 1, 
    weights_4_scales_frac, 1);
tensor_quantizer weights_4_memstr_sa8(weights_4_memstr_tsr_sa8, weights_4_sa_dim, weights_4_data, 
    sizeof(weights_4_data) / sizeof(weights_4_data[0]), &weights_4_scale, 1, &weights_4_zero_point, 1, 
    weights_4_scales_frac, 1);


static const float weights_5_data[] = {
    -0.61941f, 0.186396f, 0.049477f, 0.169398f, -0.789263f, 0.10592f, 0.329574f, -0.243221f, 0.502142f, 0.91506f, 
    -0.306106f, -0.402894f, -0.145196f, 0.027889f, -0.585279f, 0.296929f, 0.697228f, 0.262795f, 0.344464f, 0.200412f, 
    -0.914251f, 0.002258f, -0.87838f, 0.736657f, 0.205193f, 0.838853f, -0.871333f, 0.611819f, -0.127905f, 0.265697f, 
    0.669204f, -0.03834f, -0.454756f, -0.98959f, -0.044805f, 0.040481f, -0.80163f, -0.159296f, 0.43255f, 0.085688f, 
    -0.939324f, -0.208825f, 0.473686f, -0.42485f, -0.077341f, 0.100405f, 0.370831f, -0.416232f, 0.078806f, 0.671878f, 
    -0.676209f, 0.071415f, 0.816714f, -0.349117f, -0.311784f, 0.65024f, 0.32043f, -0.860963f, -0.739784f, 0.845725f, 
    0.929724f, -0.308182f, 0.703878f, 0.303432f, 0.467589f, 0.548392f, -0.805244f, -0.403818f, 0.093942f, 0.922039f, 
    0.705473f, -0.452479f, -0.075569f, -0.879728f, 0.063239f, 0.889364f, -0.967662f, -0.604151f, -0.9101f, 0.302685f, 
    0.594242f, -0.786558f, -0.259673f, 0.835226f, 0.016364f, 0.870262f, 0.576205f, -0.370272f, -0.621583f, -0.82978f, 
    0.641897f, 0.972204f, -0.551842f, 0.656226f, -0.661035f, 0.393614f, 0.953059f, 0.235269f, 0.288858f, -0.813262f, 
    0.324253f, -0.466492f, 0.16283f, 0.743783f, 0.220078f, -0.614126f, -0.034551f, -0.297453f, -0.626471f, 0.292769f
};

static const float weights_5_scale = 0.009945632f;
static const float weights_5_zero_point = 0.f;
static const int8_t weights_5_scales_frac[] = {21};
static const int weights_5_sa_dim = -1;

static const int weights_5_fx8_frac = 7;

#define WEIGHTS_5_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {11, 10}, \
    /* .rank =  */ 2

#define WEIGHTS_5_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {1 * 12, 1}, \
    /* .shape = */ {11, 10}, \
    /* .rank =  */ 2

static const mli_tensor weights_5_tsr_fx16 = {
    WEIGHTS_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_5_memstr_tsr_fx16 = {
    WEIGHTS_5_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_5_tsr_fx8 = {
    WEIGHTS_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_5_memstr_tsr_fx8 = {
    WEIGHTS_5_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_5_tsr_sa8 = {
    WEIGHTS_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_5_memstr_tsr_sa8 = {
    WEIGHTS_5_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_5_fx16(weights_5_tsr_fx16, weights_5_fx8_frac + 8, weights_5_data, 
    sizeof(weights_5_data) / sizeof(weights_5_data[0]));
tensor_quantizer weights_5_memstr_fx16(weights_5_memstr_tsr_fx16, weights_5_fx8_frac + 8, weights_5_data, 
    sizeof(weights_5_data) / sizeof(weights_5_data[0]));
tensor_quantizer weights_5_fx8(weights_5_tsr_fx8, weights_5_fx8_frac, weights_5_data, 
    sizeof(weights_5_data) / sizeof(weights_5_data[0]));
tensor_quantizer weights_5_memstr_fx8(weights_5_memstr_tsr_fx8, weights_5_fx8_frac, weights_5_data, 
    sizeof(weights_5_data) / sizeof(weights_5_data[0]));
tensor_quantizer weights_5_sa8(weights_5_tsr_sa8, weights_5_sa_dim, weights_5_data, 
    sizeof(weights_5_data) / sizeof(weights_5_data[0]), &weights_5_scale, 1, &weights_5_zero_point, 1, 
    weights_5_scales_frac, 1);
tensor_quantizer weights_5_memstr_sa8(weights_5_memstr_tsr_sa8, weights_5_sa_dim, weights_5_data, 
    sizeof(weights_5_data) / sizeof(weights_5_data[0]), &weights_5_scale, 1, &weights_5_zero_point, 1, 
    weights_5_scales_frac, 1);


static const float weights_6_data[] = {
    -0.620125f, -0.335248f, 0.703854f, -0.62983f, -0.955257f, -0.91857f, 0.723968f, 0.263853f, -0.10535f, -0.014266f, 
    -0.474739f, 0.125218f, 0.05659f, -0.132623f, 0.463404f, 0.196584f, 0.081711f, 0.355448f, 0.24952f, -0.692822f, 
    -0.48014f, -0.826529f, -0.353655f, -0.748169f, -0.087522f, 0.869225f, 0.555907f, -0.16133f, -0.787513f, 0.191783f, 
    -0.949263f, -0.859875f, -0.724693f, 0.350522f, 0.300906f, 0.147623f, 0.03224f, 0.481536f, -0.44883f, -0.027051f, 
    0.438929f, 0.60086f, 0.088868f, -0.722345f, -0.139806f, 0.690815f, -0.662184f, 0.869286f, -0.882663f, -0.497618f, 
    -0.333346f, -0.531086f, -0.827961f, 0.273114f, 0.717359f, 0.758486f, 0.632248f, 0.592136f, 0.172292f, -0.366309f, 
    -0.142095f, 0.339565f, 0.658203f, 0.320576f, 0.940589f, 0.119947f, 0.703381f, 0.313803f, -0.78686f, 0.622464f, 
    0.12432f, 0.347216f, -0.948423f, -0.026209f, 0.317702f, -0.439601f, 0.918853f, 0.683045f, -0.571682f, 0.90908f, 
    -0.011911f, 0.362072f, 0.433114f, -0.961987f, -0.747947f, -0.097197f, -0.837664f, -0.023768f, -0.357495f, 0.985421f, 
    -0.185669f, -0.975865f, -0.807715f, -0.668111f, 0.056585f, 0.136282f, -0.463853f, 0.122949f, 0.292836f, 0.566401f, 
    -0.023331f, -0.70503f, -0.637818f, 0.127703f, -0.934736f, -0.496645f, 0.834192f, 0.021296f
};

static const float weights_6_scale = 0.0099037252f;
static const float weights_6_zero_point = 0.f;
static const int8_t weights_6_scales_frac[] = {21};
static const int weights_6_sa_dim = -1;

static const int weights_6_fx8_frac = 7;

#define WEIGHTS_6_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {12, 9}, \
    /* .rank =  */ 2

#define WEIGHTS_6_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {1 * 12, 1}, \
    /* .shape = */ {12, 9}, \
    /* .rank =  */ 2

static const mli_tensor weights_6_tsr_fx16 = {
    WEIGHTS_6_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_6_memstr_tsr_fx16 = {
    WEIGHTS_6_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_6_tsr_fx8 = {
    WEIGHTS_6_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_6_memstr_tsr_fx8 = {
    WEIGHTS_6_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_6_tsr_sa8 = {
    WEIGHTS_6_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_6_memstr_tsr_sa8 = {
    WEIGHTS_6_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_6_fx16(weights_6_tsr_fx16, weights_6_fx8_frac + 8, weights_6_data, 
    sizeof(weights_6_data) / sizeof(weights_6_data[0]));
tensor_quantizer weights_6_memstr_fx16(weights_6_memstr_tsr_fx16, weights_6_fx8_frac + 8, weights_6_data, 
    sizeof(weights_6_data) / sizeof(weights_6_data[0]));
tensor_quantizer weights_6_fx8(weights_6_tsr_fx8, weights_6_fx8_frac, weights_6_data, 
    sizeof(weights_6_data) / sizeof(weights_6_data[0]));
tensor_quantizer weights_6_memstr_fx8(weights_6_memstr_tsr_fx8, weights_6_fx8_frac, weights_6_data, 
    sizeof(weights_6_data) / sizeof(weights_6_data[0]));
tensor_quantizer weights_6_sa8(weights_6_tsr_sa8, weights_6_sa_dim, weights_6_data, 
    sizeof(weights_6_data) / sizeof(weights_6_data[0]), &weights_6_scale, 1, &weights_6_zero_point, 1, 
    weights_6_scales_frac, 1);
tensor_quantizer weights_6_memstr_sa8(weights_6_memstr_tsr_sa8, weights_6_sa_dim, weights_6_data, 
    sizeof(weights_6_data) / sizeof(weights_6_data[0]), &weights_6_scale, 1, &weights_6_zero_point, 1, 
    weights_6_scales_frac, 1);


static const float weights_7_data[] = {
    0.454037f, 0.95495f, -0.350683f, -0.522488f, 0.846219f, -0.364125f, -0.023798f, 0.063983f, -0.908256f, -0.0597f, 
    -0.50762f, 0.156999f, 0.807397f, -0.839224f, -0.875243f, -0.599003f, -0.60934f, -0.836705f, -0.507154f, 0.367933f, 
    0.981761f, 0.096121f, 0.247487f, 0.540161f, 0.940034f, -0.528012f, -0.986945f, -0.556695f, 0.554088f, 0.700075f, 
    -0.512869f, -0.220017f, 0.001336f, -0.809439f, 0.066613f, 0.941961f, 0.084519f, -0.664931f, -0.890666f, -0.950368f, 
    0.779705f, 0.68319f, -0.490535f, 0.021956f, -0.293213f, 0.688015f, -0.360702f, -0.277593f, 0.251079f, -0.196938f, 
    0.068969f, -0.244814f, -0.367923f, 0.80701f, 0.742864f, 0.647913f, 0.820969f, -0.428372f, 0.201981f, -0.687662f, 
    -0.809401f, 0.55219f, 0.729294f, 0.731935f, -0.279524f, 0.644115f, -0.123078f, -0.455653f, 0.588157f, 0.630501f, 
    0.328082f, -0.497352f, -0.176509f, -0.650837f, 0.670634f, 0.493144f, -0.400565f, 0.302529f, 0.822232f, 0.231506f, 
    -0.532435f, -0.622323f, 0.037151f, 0.89705f, 0.610212f, 0.500725f, 0.438244f, -0.067325f, 0.013352f, -0.016628f
};

static const float weights_7_scale = 0.0099190464f;
static const float weights_7_zero_point = 0.f;
static const int8_t weights_7_scales_frac[] = {21};
static const int weights_7_sa_dim = -1;

static const int weights_7_fx8_frac = 7;

#define WEIGHTS_7_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {10, 9}, \
    /* .rank =  */ 2

#define WEIGHTS_7_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {1 * 12, 1}, \
    /* .shape = */ {10, 9}, \
    /* .rank =  */ 2

static const mli_tensor weights_7_tsr_fx16 = {
    WEIGHTS_7_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_7_memstr_tsr_fx16 = {
    WEIGHTS_7_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_7_tsr_fx8 = {
    WEIGHTS_7_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_7_memstr_tsr_fx8 = {
    WEIGHTS_7_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_7_tsr_sa8 = {
    WEIGHTS_7_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_7_memstr_tsr_sa8 = {
    WEIGHTS_7_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_7_fx16(weights_7_tsr_fx16, weights_7_fx8_frac + 8, weights_7_data, 
    sizeof(weights_7_data) / sizeof(weights_7_data[0]));
tensor_quantizer weights_7_memstr_fx16(weights_7_memstr_tsr_fx16, weights_7_fx8_frac + 8, weights_7_data, 
    sizeof(weights_7_data) / sizeof(weights_7_data[0]));
tensor_quantizer weights_7_fx8(weights_7_tsr_fx8, weights_7_fx8_frac, weights_7_data, 
    sizeof(weights_7_data) / sizeof(weights_7_data[0]));
tensor_quantizer weights_7_memstr_fx8(weights_7_memstr_tsr_fx8, weights_7_fx8_frac, weights_7_data, 
    sizeof(weights_7_data) / sizeof(weights_7_data[0]));
tensor_quantizer weights_7_sa8(weights_7_tsr_sa8, weights_7_sa_dim, weights_7_data, 
    sizeof(weights_7_data) / sizeof(weights_7_data[0]), &weights_7_scale, 1, &weights_7_zero_point, 1, 
    weights_7_scales_frac, 1);
tensor_quantizer weights_7_memstr_sa8(weights_7_memstr_tsr_sa8, weights_7_sa_dim, weights_7_data, 
    sizeof(weights_7_data) / sizeof(weights_7_data[0]), &weights_7_scale, 1, &weights_7_zero_point, 1, 
    weights_7_scales_frac, 1);


static const float weights_8_data[] = {
    -0.212369f, -0.375899f, 0.505953f, 0.783606f, 0.085324f, 0.691483f, -0.95568f, -0.389791f, 0.491432f, -0.045871f, 
    0.862165f, 0.463398f, 0.254853f, -0.779802f, 0.665015f, 0.610872f, -0.09195f, 0.341467f, -0.101666f, -0.438089f, 
    -0.596836f, -0.983518f, -0.073694f, 0.930277f, -0.0641f, -0.696587f, -0.305712f, 0.205578f, 0.554677f, -0.716381f, 
    -0.548447f, 0.65165f, -0.415792f, 0.673124f, -0.498276f, -0.229748f, 0.662939f, 0.099508f, -0.386263f, -0.40106f, 
    0.560754f, 0.806698f, 0.803967f, 0.425846f, -0.233268f, -0.738372f, 0.999556f, -0.303527f, 0.208585f, -0.345272f, 
    0.865302f, -0.522877f, -0.429093f, 0.377838f, -0.28236f, 0.870183f, -0.854466f, 0.750466f, -0.350703f, 0.04112f, 
    0.466716f, 0.100708f, -0.145216f, 0.558994f, -0.763463f, 0.56247f, -0.170661f, 0.52397f, -0.688517f, 0.788807f, 
    0.910634f, -0.098054f, -0.353718f, -0.791868f, 0.140207f, 0.501478f, 0.262305f, 0.610177f, 0.280953f, 0.198271f, 
    -0.716103f, 0.24793f, -0.73563f, -0.450427f, -0.939971f, -0.827016f, -0.87219f, -0.411049f, 0.689599f, -0.947581f, 
    -0.417005f, 0.77111f, -0.523912f, 0.741901f, -0.726866f, -0.591473f, 0.583859f, 0.228531f, 0.372288f
};

static const float weights_8_scale = 0.010045792f;
static const float weights_8_zero_point = 0.f;
static const int8_t weights_8_scales_frac[] = {21};
static const int weights_8_sa_dim = -1;

static const int weights_8_fx8_frac = 7;

#define WEIGHTS_8_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {11, 9}, \
    /* .rank =  */ 2

#define WEIGHTS_8_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {1 * 12, 1}, \
    /* .shape = */ {11, 9}, \
    /* .rank =  */ 2

static const mli_tensor weights_8_tsr_fx16 = {
    WEIGHTS_8_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_8_memstr_tsr_fx16 = {
    WEIGHTS_8_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_8_tsr_fx8 = {
    WEIGHTS_8_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_8_memstr_tsr_fx8 = {
    WEIGHTS_8_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_8_tsr_sa8 = {
    WEIGHTS_8_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_8_memstr_tsr_sa8 = {
    WEIGHTS_8_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_8_fx16(weights_8_tsr_fx16, weights_8_fx8_frac + 8, weights_8_data, 
    sizeof(weights_8_data) / sizeof(weights_8_data[0]));
tensor_quantizer weights_8_memstr_fx16(weights_8_memstr_tsr_fx16, weights_8_fx8_frac + 8, weights_8_data, 
    sizeof(weights_8_data) / sizeof(weights_8_data[0]));
tensor_quantizer weights_8_fx8(weights_8_tsr_fx8, weights_8_fx8_frac, weights_8_data, 
    sizeof(weights_8_data) / sizeof(weights_8_data[0]));
tensor_quantizer weights_8_memstr_fx8(weights_8_memstr_tsr_fx8, weights_8_fx8_frac, weights_8_data, 
    sizeof(weights_8_data) / sizeof(weights_8_data[0]));
tensor_quantizer weights_8_sa8(weights_8_tsr_sa8, weights_8_sa_dim, weights_8_data, 
    sizeof(weights_8_data) / sizeof(weights_8_data[0]), &weights_8_scale, 1, &weights_8_zero_point, 1, 
    weights_8_scales_frac, 1);
tensor_quantizer weights_8_memstr_sa8(weights_8_memstr_tsr_sa8, weights_8_sa_dim, weights_8_data, 
    sizeof(weights_8_data) / sizeof(weights_8_data[0]), &weights_8_scale, 1, &weights_8_zero_point, 1, 
    weights_8_scales_frac, 1);


static const float weights_9_data[] = {
    0.748788f, 0.75015f, 0.540454f, -0.419529f, 0.966013f, 0.573554f, -0.356106f, 0.163919f, -0.456666f, -0.296149f, 
    0.571023f, -0.63092f, 0.787088f, -0.468868f, 0.27717f, -0.786735f, 0.452829f, -0.406493f, -0.918751f, 0.6926f, 
    0.30458f, 0.038585f, 0.522657f, 0.902679f, -0.518962f, 0.036681f, -0.445712f, -0.568662f, 0.960567f, 0.175662f, 
    .165107f, 0.13479f, 0.188154f, 0.755263f, 0.132477f, 0.600766f, -0.335736f, -0.564843f, 0.088733f, -0.779877f, 
    -0.894105f, 0.592711f, 0.580784f, 0.402803f, 0.964219f, -0.503205f, 0.1731f, 0.874018f, 0.628634f, -0.197715f, 
    0.381513f, 0.640914f, -0.978743f, -0.095866f, -0.393065f, -0.009353f, 0.346044f, -0.649642f, -0.981214f, 0.076264f, 
    0.524804f, -0.83772f, 0.453198f, 0.535425f, -0.693031f, 0.973179f, -0.083707f, 0.6303f, 0.648089f, 0.636187f, 
    0.623542f, 0.110564f
};

static const float weights_9_scale = 0.0098614432f;
static const float weights_9_zero_point = 0.f;
static const int8_t weights_9_scales_frac[] = {21};
static const int weights_9_sa_dim = -1;

static const int weights_9_fx8_frac = 7;

#define WEIGHTS_9_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {8, 9}, \
    /* .rank =  */ 2

#define WEIGHTS_9_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {1 * 12, 1}, \
    /* .shape = */ {8, 9}, \
    /* .rank =  */ 2

static const mli_tensor weights_9_tsr_fx16 = {
    WEIGHTS_9_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_9_memstr_tsr_fx16 = {
    WEIGHTS_9_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_9_tsr_fx8 = {
    WEIGHTS_9_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_9_memstr_tsr_fx8 = {
    WEIGHTS_9_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_9_tsr_sa8 = {
    WEIGHTS_9_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_9_memstr_tsr_sa8 = {
    WEIGHTS_9_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_9_fx16(weights_9_tsr_fx16, weights_9_fx8_frac + 8, weights_9_data, 
    sizeof(weights_9_data) / sizeof(weights_9_data[0]));
tensor_quantizer weights_9_memstr_fx16(weights_9_memstr_tsr_fx16, weights_9_fx8_frac + 8, weights_9_data, 
    sizeof(weights_9_data) / sizeof(weights_9_data[0]));
tensor_quantizer weights_9_fx8(weights_9_tsr_fx8, weights_9_fx8_frac, weights_9_data, 
    sizeof(weights_9_data) / sizeof(weights_9_data[0]));
tensor_quantizer weights_9_memstr_fx8(weights_9_memstr_tsr_fx8, weights_9_fx8_frac, weights_9_data, 
    sizeof(weights_9_data) / sizeof(weights_9_data[0]));
tensor_quantizer weights_9_sa8(weights_9_tsr_sa8, weights_9_sa_dim, weights_9_data, 
    sizeof(weights_9_data) / sizeof(weights_9_data[0]), &weights_9_scale, 1, &weights_9_zero_point, 1, 
    weights_9_scales_frac, 1);
tensor_quantizer weights_9_memstr_sa8(weights_9_memstr_tsr_sa8, weights_9_sa_dim, weights_9_data, 
    sizeof(weights_9_data) / sizeof(weights_9_data[0]), &weights_9_scale, 1, &weights_9_zero_point, 1, 
    weights_9_scales_frac, 1);


static const float bias_1_data[] = {
    0.176345f, 0.009683f, -0.056492f, -0.024836f, 0.141631f, 0.152541f, 0.091757f, 0.114015f
};

static const float bias_1_scale = 0.0000402414f;
static const float bias_1_zero_point = 0.f;
static const int8_t bias_1_scales_frac[] = {29};
static const int bias_1_sa_dim = -1;

static const float bias_1_i1_w1_scale = 0.0000402414f;
static const float bias_1_i1_w1_zero_point = 0.f;
static const int8_t bias_1_i1_w1_scales_frac[] = {29};
static const int bias_1_i1_w1_sa_dim = -1;

static const int bias_1_fx8_frac = 9;

#define BIAS_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {8}, \
    /* .rank =  */ 1

static const mli_tensor bias_1_tsr_fx16 = {
    BIAS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_1_tsr_fx8 = {
    BIAS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_1_i1_w1_tsr_sa32 = {
    BIAS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer bias_1_fx16(bias_1_tsr_fx16, bias_1_fx8_frac + 8, bias_1_data, 
    sizeof(bias_1_data) / sizeof(bias_1_data[0]));
tensor_quantizer bias_1_fx8(bias_1_tsr_fx8, bias_1_fx8_frac, bias_1_data, 
    sizeof(bias_1_data) / sizeof(bias_1_data[0]));
tensor_quantizer bias_1_i1_w1_sa32(bias_1_i1_w1_tsr_sa32, bias_1_i1_w1_sa_dim, bias_1_data, 
    sizeof(bias_1_data) / sizeof(bias_1_data[0]), &bias_1_i1_w1_scale, 1, &bias_1_i1_w1_zero_point, 1, 
    bias_1_i1_w1_scales_frac, 1);


static const float bias_2_data[] = {
    0.208238f, -0.127407f, 0.085261f, 0.121543f, -0.062138f, 0.11798f, 0.01566f, -0.095964f, 0.161241f, 0.220358f
};

static const float bias_2_scale = 0.0002979996f;
static const float bias_2_zero_point = 0.f;
static const int8_t bias_2_scales_frac[] = {26};
static const int bias_2_sa_dim = -1;

static const float bias_2_i3_w3_scale = 0.0000377807f;
static const float bias_2_i3_w3_zero_point = 0.f;
static const int8_t bias_2_i3_w3_scales_frac[] = {29};
static const int bias_2_i3_w3_sa_dim = -1;

static const int bias_2_fx8_frac = 9;

#define BIAS_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {10}, \
    /* .rank =  */ 1

static const mli_tensor bias_2_tsr_fx16 = {
    BIAS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_2_tsr_fx8 = {
    BIAS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_2_i3_w3_tsr_sa32 = {
    BIAS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer bias_2_fx16(bias_2_tsr_fx16, bias_2_fx8_frac + 8, bias_2_data, 
    sizeof(bias_2_data) / sizeof(bias_2_data[0]));
tensor_quantizer bias_2_fx8(bias_2_tsr_fx8, bias_2_fx8_frac, bias_2_data, 
    sizeof(bias_2_data) / sizeof(bias_2_data[0]));
tensor_quantizer bias_2_i3_w3_sa32(bias_2_i3_w3_tsr_sa32, bias_2_i3_w3_sa_dim, bias_2_data, 
    sizeof(bias_2_data) / sizeof(bias_2_data[0]), &bias_2_i3_w3_scale, 1, &bias_2_i3_w3_zero_point, 1, 
    bias_2_i3_w3_scales_frac, 1);


static const float bias_3_data[] = {
    0.443208f, 0.255715f, -0.26626f, -0.876383f, 0.521495f, -0.810034f, -0.763066f, -0.773739f, -0.178584f
};

static const float bias_3_scale = 0.0001526533f;
static const float bias_3_zero_point = 0.f;
static const int8_t bias_3_scales_frac[] = {27};
static const int bias_3_sa_dim = -1;

static const float bias_3_i3_w6_scale = 0.0000379796f;
static const float bias_3_i3_w6_zero_point = 0.f;
static const int8_t bias_3_i3_w6_scales_frac[] = {29};
static const int bias_3_i3_w6_sa_dim = -1;

static const int bias_3_fx8_frac = 7;

#define BIAS_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {9}, \
    /* .rank =  */ 1

static const mli_tensor bias_3_tsr_fx16 = {
    BIAS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_3_tsr_fx8 = {
    BIAS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_3_i3_w6_tsr_sa32 = {
    BIAS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer bias_3_fx16(bias_3_tsr_fx16, bias_3_fx8_frac + 8, bias_3_data, 
    sizeof(bias_3_data) / sizeof(bias_3_data[0]));
tensor_quantizer bias_3_fx8(bias_3_tsr_fx8, bias_3_fx8_frac, bias_3_data, 
    sizeof(bias_3_data) / sizeof(bias_3_data[0]));
tensor_quantizer bias_3_i3_w6_sa32(bias_3_i3_w6_tsr_sa32, bias_3_i3_w6_sa_dim, bias_3_data, 
    sizeof(bias_3_data) / sizeof(bias_3_data[0]), &bias_3_i3_w6_scale, 1, &bias_3_i3_w6_zero_point, 1, 
    bias_3_i3_w6_scales_frac, 1);


static const float test_1_out_data[] = {
    0.566755f, 0.721686f, 2.126752f, -0.704047f, 0.052606f, -1.765898f, 0.521044f, 0.046221f
};

static const float test_1_out_scale = 0.019561056f;
static const float test_1_out_zero_point = 0.1804270744f;
static const int8_t test_1_out_scales_frac[] = {20};
static const int test_1_out_sa_dim = -1;

static const int test_1_out_fx8_frac = 5;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {8}, \
    /* .rank =  */ 1

static const mli_tensor test_1_out_tsr_fx16 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor test_1_out_tsr_sa8 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data, 
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim, test_1_out_data, 
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_1_out_scale, 1, &test_1_out_zero_point, 1, 
    test_1_out_scales_frac, 1);


static const float test_2_out_data[] = {
    -1.216035f, 1.547995f, 0.285401f, 0.02231f, -1.779811f, -1.294535f, 1.164403f, 0.139236f, 0.852021f, 2.547481f
};

static const float test_2_out_scale = 0.02560528f;
static const float test_2_out_zero_point = 0.3838348389f;
static const int8_t test_2_out_scales_frac[] = {20};
static const int test_2_out_sa_dim = -1;

static const int test_2_out_fx8_frac = 5;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {10}, \
    /* .rank =  */ 1


static const mli_tensor test_2_out_tsr_fx16 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor test_2_out_tsr_sa8 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_fx8_frac + 8, test_2_out_data, 
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_sa_dim, test_2_out_data, 
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &test_2_out_scale, 1, &test_2_out_zero_point, 1, 
    test_2_out_scales_frac, 1);


static const float test_3_out_data[] = {
    0.645183f, 2.011113f, -1.177894f, -0.983762f, 0.05449f, -3.620857f, 0.560062f, -1.293246f, 0.099845f
};

static const float test_3_out_scale = 0.0354211926f;
static const float test_3_out_zero_point = -0.804872036f;
static const int8_t test_3_out_scales_frac[] = {19};
static const int test_3_out_sa_dim = -1;

static const int test_3_out_fx8_frac = 5;

#define TEST_3_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {9}, \
    /* .rank =  */ 1

static const mli_tensor test_3_out_tsr_fx16 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor test_3_out_tsr_sa8 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer test_3_out_fx16(test_3_out_tsr_fx16, test_3_out_fx8_frac + 8, test_3_out_data, 
    sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));
tensor_quantizer test_3_out_sa8(test_3_out_tsr_sa8, test_3_out_sa_dim, test_3_out_data, 
    sizeof(test_3_out_data) / sizeof(test_3_out_data[0]), &test_3_out_scale, 1, &test_3_out_zero_point, 1, 
    test_3_out_scales_frac, 1);

const mli_rnn_dense_cfg test_1_cfg = {/* .inputs_num = */ 2};
const mli_rnn_dense_cfg test_2_cfg = {/* .inputs_num = */ 3};
const mli_rnn_dense_cfg test_3_cfg = {/* .inputs_num = */ 4};
