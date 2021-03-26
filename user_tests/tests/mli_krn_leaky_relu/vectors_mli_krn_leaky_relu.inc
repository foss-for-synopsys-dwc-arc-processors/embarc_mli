
/*
* Copyright 2020, Synopsys, Inc.
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

extern mli::tst::tensor_quantizer input_1_fx16;
extern mli::tst::tensor_quantizer input_1_sa8;
extern mli::tst::tensor_quantizer input_2_fx16;
extern mli::tst::tensor_quantizer input_2_sa8;
extern mli::tst::tensor_quantizer test_1_out_fx16;
extern mli::tst::tensor_quantizer test_1_out_sa8;
extern mli::tst::tensor_quantizer test_2_out_fx16;
extern mli::tst::tensor_quantizer test_2_out_sa8;

static const float input_1_data[] = {
    -2.999756f, -2.952881f, -2.906006f, -2.859131f, -2.812256f, -2.765381f, -2.718506f, -2.671631f, -2.624756f, -2.577881f, 
    -2.531006f, -2.484131f, -2.437256f, -2.390381f, -2.343506f, -2.296631f, -2.249756f, -2.202881f, -2.156006f, -2.109131f, 
    -2.062256f, -2.015381f, -1.968506f, -1.921631f, -1.874756f, -1.827881f, -1.781006f, -1.734131f, -1.687256f, -1.640381f, 
    -1.593506f, -1.546631f, -1.499756f, -1.452881f, -1.406006f, -1.359131f, -1.312256f, -1.265381f, -1.218506f, -1.171631f, 
    -1.124756f, -1.077881f, -1.031006f, -0.984131f, -0.937256f, -0.890381f, -0.843506f, -0.796631f, -0.749756f, -0.702881f, 
    -0.656006f, -0.609131f, -0.562256f, -0.515381f, -0.468506f, -0.421631f, -0.374756f, -0.327881f, -0.281006f, -0.234131f, 
    -0.187256f, -0.140381f, -0.093506f, -0.046631f,  0.000244f,  0.047119f,  0.093994f,  0.140869f,  0.187744f,  0.234619f, 
     0.281494f,  0.328369f,  0.375244f,  0.422119f,  0.468994f,  0.515869f,  0.562744f,  0.609619f,  0.656494f,  0.703369f, 
     0.750244f,  0.797119f,  0.843994f,  0.890869f,  0.937744f,  0.984619f,  1.031494f,  1.078369f,  1.125244f,  1.172119f, 
     1.218994f,  1.265869f,  1.312744f,  1.359619f,  1.406494f,  1.453369f,  1.500244f,  1.547119f,  1.593994f,  1.640869f, 
     1.687744f,  1.734619f,  1.781494f,  1.828369f,  1.875244f,  1.922119f,  1.968994f,  2.015869f,  2.062744f,  2.109619f, 
     2.156494f,  2.203369f,  2.250244f,  2.297119f,  2.343994f,  2.390869f,  2.437744f,  2.484619f,  2.531494f,  2.578369f, 
     2.625244f,  2.672119f,  2.718994f,  2.765869f,  2.812744f,  2.859619f,  2.906494f,  2.953369f};

static const float input_1_scale = 0.0233455878f;
static const float input_1_zero_point = -0.0231933594f;
static const int8_t input_1_scales_frac[] = {20};
static const int input_1_sa_dim = -1;

static const int input_1_fx8_frac = 5;

#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {16 + 2, 1}, \
    /* .shape = */ {8, 16}, \
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

static const float input_2_data[] = {0.200000f};

static const float input_2_scale = 0.0233455878f;
static const float input_2_zero_point = -0.0231933594f;
static const int8_t input_2_scales_frac[] = {20};
static const int input_2_sa_dim = -1;

static const int input_2_fx8_frac = 9;

#define INPUT_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {0}, \
    /* .rank =  */ 0


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

static const float input_3_data[] = {1.200000f};

static const float input_3_scale = 0.0233455878f;
static const float input_3_zero_point = -0.0231933594f;
static const int8_t input_3_scales_frac[] = {20};
static const int input_3_sa_dim = -1;

static const int input_3_fx8_frac = 6;

#define INPUT_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1}, \
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

static const float test_1_out_data[] = {
    -0.599951f, -0.590576f, -0.581201f, -0.571826f, -0.562451f, -0.553076f, -0.543701f, -0.534326f, -0.524951f, -0.515576f, 
    -0.506201f, -0.496826f, -0.487451f, -0.478076f, -0.468701f, -0.459326f, -0.449951f, -0.440576f, -0.431201f, -0.421826f, 
    -0.412451f, -0.403076f, -0.393701f, -0.384326f, -0.374951f, -0.365576f, -0.356201f, -0.346826f, -0.337451f, -0.328076f, 
    -0.318701f, -0.309326f, -0.299951f, -0.290576f, -0.281201f, -0.271826f, -0.262451f, -0.253076f, -0.243701f, -0.234326f, 
    -0.224951f, -0.215576f, -0.206201f, -0.196826f, -0.187451f, -0.178076f, -0.168701f, -0.159326f, -0.149951f, -0.140576f, 
    -0.131201f, -0.121826f, -0.112451f, -0.103076f, -0.093701f, -0.084326f, -0.074951f, -0.065576f, -0.056201f, -0.046826f, 
    -0.037451f, -0.028076f, -0.018701f, -0.009326f,  0.000244f,  0.047119f,  0.093994f,  0.140869f,  0.187744f,  0.234619f, 
     0.281494f,  0.328369f,  0.375244f,  0.422119f,  0.468994f,  0.515869f,  0.562744f,  0.609619f,  0.656494f,  0.703369f, 
     0.750244f,  0.797119f,  0.843994f,  0.890869f,  0.937744f,  0.984619f,  1.031494f,  1.078369f,  1.125244f,  1.172119f, 
     1.218994f,  1.265869f,  1.312744f,  1.359619f,  1.406494f,  1.453369f,  1.500244f,  1.547119f,  1.593994f,  1.640869f, 
     1.687744f,  1.734619f,  1.781494f,  1.828369f,  1.875244f,  1.922119f,  1.968994f,  2.015869f,  2.062744f,  2.109619f, 
     2.156494f,  2.203369f,  2.250244f,  2.297119f,  2.343994f,  2.390869f,  2.437744f,  2.484619f,  2.531494f,  2.578369f, 
     2.625244f,  2.672119f,  2.718994f,  2.765869f,  2.812744f,  2.859619f,  2.906494f,  2.953369f};

static const float test_1_out_scale = 0.0139345899f;
static const float test_1_out_zero_point = 1.1767089367f;
static const int8_t test_1_out_scales_frac[] = {21};
static const int test_1_out_sa_dim = -1;

static const int test_1_out_fx8_frac = 5;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {16 + 4, 1}, \
    /* .shape = */ {8, 16}, \
    /* .rank =  */ 2


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

static const float test_2_out_data[] = {
    -3.599707f, -3.543457f, -3.487207f, -3.430957f, -3.374707f, -3.318457f, -3.262207f, -3.205957f, -3.149707f, -3.093457f, 
    -3.037207f, -2.980957f, -2.924707f, -2.868457f, -2.812207f, -2.755957f, -2.699707f, -2.643457f, -2.587207f, -2.530957f, 
    -2.474707f, -2.418457f, -2.362207f, -2.305957f, -2.249707f, -2.193457f, -2.137207f, -2.080957f, -2.024707f, -1.968457f, 
    -1.912207f, -1.855957f, -1.799707f, -1.743457f, -1.687207f, -1.630957f, -1.574707f, -1.518457f, -1.462207f, -1.405957f, 
    -1.349707f, -1.293457f, -1.237207f, -1.180957f, -1.124707f, -1.068457f, -1.012207f, -0.955957f, -0.899707f, -0.843457f, 
    -0.787207f, -0.730957f, -0.674707f, -0.618457f, -0.562207f, -0.505957f, -0.449707f, -0.393457f, -0.337207f, -0.280957f, 
    -0.224707f, -0.168457f, -0.112207f, -0.055957f,  0.000244f,  0.047119f,  0.093994f,  0.140869f,  0.187744f,  0.234619f, 
     0.281494f,  0.328369f,  0.375244f,  0.422119f,  0.468994f,  0.515869f,  0.562744f,  0.609619f,  0.656494f,  0.703369f, 
     0.750244f,  0.797119f,  0.843994f,  0.890869f,  0.937744f,  0.984619f,  1.031494f,  1.078369f,  1.125244f,  1.172119f, 
     1.218994f,  1.265869f,  1.312744f,  1.359619f,  1.406494f,  1.453369f,  1.500244f,  1.547119f,  1.593994f,  1.640869f, 
     1.687744f,  1.734619f,  1.781494f,  1.828369f,  1.875244f,  1.922119f,  1.968994f,  2.015869f,  2.062744f,  2.109619f, 
     2.156494f,  2.203369f,  2.250244f,  2.297119f,  2.343994f,  2.390869f,  2.437744f,  2.484619f,  2.531494f,  2.578369f, 
     2.625244f,  2.672119f,  2.718994f,  2.765869f,  2.812744f,  2.859619f,  2.906494f,  2.953369f};

static const float test_2_out_scale = 0.0256983377f;
static const float test_2_out_zero_point = -0.3231689930f;
static const int8_t test_2_out_scales_frac[] = {20};
static const int test_2_out_sa_dim = -1;

static const int test_2_out_fx8_frac = 5;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {16 + 8, 1}, \
    /* .shape = */ {8, 16}, \
    /* .rank =  */ 2


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

tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx8_frac + 8, input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]));
tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data, 
sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);

tensor_quantizer input_3_fx16(input_3_tsr_fx16, input_3_fx8_frac + 8, input_3_data, sizeof(input_3_data) / sizeof(input_3_data[0]));
tensor_quantizer input_3_sa8(input_3_tsr_sa8, input_3_sa_dim, input_3_data, 
sizeof(input_3_data) / sizeof(input_3_data[0]), &input_3_scale, 1, &input_3_zero_point, 1, input_3_scales_frac, 1);

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim, test_1_out_data, 
sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_1_out_scale, 1, &test_1_out_zero_point, 1, test_1_out_scales_frac, 1);

tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_fx8_frac + 8, test_2_out_data, sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_sa_dim, test_2_out_data, 
sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &test_2_out_scale, 1, &test_2_out_zero_point, 1, test_2_out_scales_frac, 1);

