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

// Extracted Output vectors Declaration
//===================================================
extern mli::tst::tensor_quantizer test_1_out_fx8;
extern mli::tst::tensor_quantizer test_1_out_fx16;
extern mli::tst::tensor_quantizer test_1_out_sa8;
extern mli::tst::tensor_quantizer test_1_out_sa32;
extern mli::tst::tensor_quantizer test_2_out_fx16;
extern mli::tst::tensor_quantizer test_2_out_sa32;
extern mli::tst::tensor_quantizer test_3_out_fx8;
extern mli::tst::tensor_quantizer test_3_out_fx16;
extern mli::tst::tensor_quantizer test_3_out_sa8;
extern mli::tst::tensor_quantizer test_3_out_sa32;

// Tests configuration structures Declaration
//========================================
extern const mli_argmax_cfg test_1_cfg;
extern const mli_argmax_cfg test_2_cfg;
extern const mli_argmax_cfg test_3_cfg;

// Generated input vectors 
//========================================
static const float input_1_data[] = { -0.42424242f };

static const float input_1_scale = 0.003340491f;
static const float input_1_zero_point = 0.0f;
static const int8_t input_1_scales_frac[] = { 23 };
static const int input_1_sa_dim = -1;
static const int input_1_fx8_frac = 7;

#define TEST_1_INPUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ { 1 }, \
    /* .rank = */  1

static const mli_tensor input_1_tsr_sa8 = {
    TEST_1_INPUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_1_tsr_fx16 = {
    TEST_1_INPUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const float input_2_data[] = {
    -0.795256615f, -0.020611584f, -0.857786238f, -0.116273165f, 0.808907866f, -0.496227741f, -0.107275546f, -0.577136159f,
    1.860774279f, 0.930648685f, 1.950116396f, 1.566918135f, 0.589830399f, -0.950313091f, 0.618810654f, -0.962112069f,
    1.893128872f, -0.582279801f, 1.939082861f, -0.719187677f, 1.063522339f, 1.109600544f, 1.949721098f, -0.861565351f,
    0.142124653f, -0.689962804f, 1.484006405f, 1.528692007f, -0.571019948f, -0.376762152f, -0.228962064f, 1.455527067f,
    -0.751569092f, 0.435355544f, 0.228472352f, 1.998298407f, 0.528104305f, 0.669281363f, 1.963288307f, -0.903628111f,
    -0.531457663f, 0.510696411f, -0.32744509f, -0.207189679f, -0.556229711f, -0.869290471f, -0.953169167f, 1.123666763f,
    -0.810569465f, 0.351910353f, 0.803937912f, 0.019652009f, -0.049965858f, 0.62204957f, 0.900914192f, 0.991265893f,
    1.660414934f, 0.52547884f, -0.973419189f, 1.005276918f, 0.573811293f, 0.358962655f, 0.842565656f, 1.532562733f,
    0.070606351f, -0.333183408f, 0.164556742f, 1.287754774f, 0.305704474f, 0.906447887f, -0.721507311f, 1.87564373f,
    0.568320155f, -0.962049961f, 1.042712688f, 0.401276469f, 1.858359337f, -0.10336417f, -0.097108901f, 1.075629711f,
    0.781458378f, 0.160974979f, 0.460997581f, 0.180617571f, -0.582523704f, -0.813086033f, 1.32623148f, -0.808524489f,
    -0.4848876f, 1.06777668f, 0.391311884f, -0.760229707f, 0.372676373f, 1.040057898f, -0.851530135f, 1.943858385f,
    0.56839323f, 0.60644412f, 1.348023891f, -0.123570859f, -0.975745201f, 0.277356744f, 0.937498331f, 1.385216713f,
    1.256720543f, -0.355837345f, 1.507807732f, -0.943606257f, -0.654643536f, 0.141403437f, -0.733771205f, 1.77818656f,
    0.254359484f, 0.034042001f, 1.820943594f, 0.470926762f, 0.997800827f, -0.947544813f, -0.973272562f, 1.56578064f,
    -0.972122073f, -0.172074139f, -0.780366302f, -0.018243432f, -0.442839742f, -0.381048918f, -0.363562047f, -0.161547184f,
    1.9712286f, -0.079811931f, 0.247272134f, 0.334599495f, 1.427205563f, 1.751028538f, 1.87986064f, -0.979990005f,
    0.431378245f, -0.714987516f, 1.234764576f, -0.018693268f, -0.588499546f, -0.086845994f, 1.424705029f, 1.664449692f,
    1.395023346f, 0.789302588f, -0.676198661f
};

static const float input_2_scale = 0.011725545f;
static const float input_2_zero_point = 0.509154201f;
static const int8_t input_2_scales_frac[] = { 21 };
static const int input_2_sa_dim = -1;
static const int input_2_fx8_frac = 6;

#define TEST_2_INPUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 13*6*1, 6*1, 1}, \
    /* .shape = */ { 7, 7, 3 }, \
    /* .rank = */  3

static const mli_tensor input_2_tsr_sa8 = {
    TEST_2_INPUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_2_tsr_fx16 = {
    TEST_2_INPUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const float input_3_data[] = {
    -0.795256615f, -0.020611584f, -0.857786238f, -0.116273165f, 0.808907866f, -0.496227741f, -0.107275546f, -0.577136159f,
    1.860774279f, 0.930648685f, 1.949721098f, 1.566918135f, 0.589830399f, -0.950313091f, 0.618810654f, -0.962112069f,
    1.893128872f, -0.582279801f, 1.939082861f, -0.719187677f, 1.063522339f, 1.109600544f, 1.950116396f, -0.861565351f,
    0.142124653f, -0.689962804f, 1.484006405f, 1.528692007f, -0.571019948f, -0.376762152f, -0.228962064f, 1.455527067f,
    -0.751569092f, 0.435355544f, 0.228472352f, 1.998298407f, 0.528104305f, 0.669281363f, 1.963288307f, -0.903628111f,
    -0.531457663f, 0.510696411f, -0.32744509f, -0.207189679f, -0.556229711f, -0.869290471f, -0.953169167f, 1.123666763f,
    -0.810569465f, 0.351910353f, 0.803937912f, 0.019652009f, -0.049965858f, 0.62204957f, 0.900914192f, 0.991265893f,
    1.660414934f, 0.52547884f, -0.973419189f, 1.005276918f, 0.573811293f, 0.358962655f, 0.842565656f, 1.532562733f,
    0.070606351f, -0.333183408f, 0.164556742f, 1.287754774f, 0.305704474f, 0.906447887f, -0.721507311f, 1.87564373f,
    0.568320155f, -0.962049961f, 1.042712688f, 0.401276469f, 1.858359337f, -0.10336417f, -0.097108901f, 1.075629711f,
    0.781458378f, 0.160974979f, 0.460997581f, 0.180617571f, -0.582523704f, -0.813086033f, 1.32623148f, -0.808524489f,
    -0.4848876f, 1.06777668f, 0.391311884f, -0.760229707f, 0.372676373f, 1.040057898f, -0.851530135f, 1.943858385f,
    0.56839323f, 0.60644412f, 1.348023891f, -0.123570859f, -0.975745201f, 0.277356744f, 0.937498331f, 1.385216713f,
    1.256720543f, -0.355837345f, 1.507807732f, -0.943606257f, -0.654643536f, 0.141403437f, -0.733771205f, 1.77818656f,
    0.254359484f, 0.034042001f, 1.820943594f, 0.470926762f, 0.997800827f, -0.947544813f, -0.973272562f, 1.56578064f,
    -0.972122073f, -0.172074139f, -0.780366302f, -0.018243432f, -0.442839742f, -0.381048918f
};

static const float input_3_scale = 0.011725545f;
static const float input_3_zero_point = 0.509154201f;
static const int8_t input_3_scales_frac[] = { 21 };
static const int input_3_sa_dim = -1;
static const int input_3_fx8_frac = 6;

#define TEST_3_INPUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ { 7, 6, 3 }, \
    /* .rank = */  3

static const mli_tensor input_3_tsr_sa8 = {
    TEST_3_INPUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_3_tsr_fx16 = {
    TEST_3_INPUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

// Extracted Output vectors 
//===================================================
static const float test_1_out_data[] = { 0.0f };

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ { 1, 1 }, \
    /* .rank = */  2

static const mli_tensor test_1_out_tsr_fx8 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

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

static const mli_tensor test_1_out_tsr_sa32 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_2_out_data[] = { 104.0f, 470.0f, 110.0f, 19.0f, 79.0f };

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ { 1, 5 }, \
    /* .rank = */  2

static const mli_tensor test_2_out_tsr_fx16 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor test_2_out_tsr_sa32 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_3_out_data[] = {
    18.0f, 114.0f, 111.0f, 63.0f,
    22.0f, 10.0f, 16.0f, 76.0f,
    35.0f, 38.0f, 95.0f, 71.0f
};

#define TEST_3_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ { 3, 4 }, \
    /* .rank = */  2

static const mli_tensor test_3_out_tsr_fx8 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

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

static const mli_tensor test_3_out_tsr_sa32 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_output_scale = 1.0f;
static const float test_output_zero_point = 0.0f;
static const int8_t test_output_scales_frac[] = { 0 };
static const int test_output_sa_dim = -1;
static const int test_output_fx8_frac = 0;

// Quantizers
//===================================================

tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));

tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1,
    &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx8_frac + 8, input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]));

tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1,
    &input_2_zero_point, 1, input_2_scales_frac, 1);

tensor_quantizer input_3_fx16(input_3_tsr_fx16, input_3_fx8_frac + 8, input_3_data, sizeof(input_3_data) / sizeof(input_3_data[0]));

tensor_quantizer input_3_sa8(input_3_tsr_sa8, input_3_sa_dim, input_3_data, sizeof(input_3_data) / sizeof(input_3_data[0]), &input_3_scale, 1,
    &input_3_zero_point, 1, input_3_scales_frac, 1);

tensor_quantizer test_1_out_fx8(test_1_out_tsr_fx8, test_output_fx8_frac, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_output_fx8_frac, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));

tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_output_sa_dim, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_output_scale, 1,
    &test_output_zero_point, 1, test_output_scales_frac, 1);

tensor_quantizer test_1_out_sa32(test_1_out_tsr_sa32, test_output_sa_dim, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_output_scale, 1,
&test_output_zero_point, 1, test_output_scales_frac, 1);

tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_output_fx8_frac, test_2_out_data, sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));

tensor_quantizer test_2_out_sa32(test_2_out_tsr_sa32, test_output_sa_dim, test_2_out_data, sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &test_output_scale, 1,
    &test_output_zero_point, 1, test_output_scales_frac, 1);

tensor_quantizer test_3_out_fx8(test_3_out_tsr_fx8, test_output_fx8_frac, test_3_out_data, sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));

tensor_quantizer test_3_out_fx16(test_3_out_tsr_fx16, test_output_fx8_frac, test_3_out_data, sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));

tensor_quantizer test_3_out_sa8(test_3_out_tsr_sa8, test_output_sa_dim, test_3_out_data, sizeof(test_3_out_data) / sizeof(test_3_out_data[0]), &test_output_scale, 1,
    &test_output_zero_point, 1, test_output_scales_frac, 1);

tensor_quantizer test_3_out_sa32(test_3_out_tsr_sa32, test_output_sa_dim, test_3_out_data, sizeof(test_3_out_data) / sizeof(test_3_out_data[0]), &test_output_scale, 1,
    &test_output_zero_point, 1, test_output_scales_frac, 1);

// Tests configuration structures
//========================================

const mli_argmax_cfg test_1_cfg = {
    /* axis = */ -1,
    /* topk = */ 1
};

const mli_argmax_cfg test_2_cfg = {
    /* axis = */ -1,
    /* topk = */ 5
};

const mli_argmax_cfg test_3_cfg = {
    /* axis = */ 2,
    /* topk = */ 4
};