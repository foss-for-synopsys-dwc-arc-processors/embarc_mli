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
extern mli::tst::tensor_quantizer test_3_out_fx16;
extern mli::tst::tensor_quantizer test_3_out_sa8;
extern mli::tst::tensor_quantizer test_4_out_fx16;
extern mli::tst::tensor_quantizer test_4_out_sa8;

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

static const float input_2_data[] = {
    3.000244f, 3.039307f, 3.078369f, 3.117432f, 3.156494f, 3.195557f, 3.234619f, 3.273682f, 3.312744f, 3.351807f, 
    3.390869f, 3.429932f, 3.468994f, 3.508057f, 3.547119f, 3.586182f, 3.625244f, 3.664307f, 3.703369f, 3.742432f, 
    3.781494f, 3.820557f, 3.859619f, 3.898682f, 3.937744f, 3.976807f, 4.015869f, 4.054932f, 4.093994f, 4.133057f, 
    4.172119f, 4.211182f, 4.250244f, 4.289307f, 4.328369f, 4.367432f, 4.406494f, 4.445557f, 4.484619f, 4.523682f, 
    4.562744f, 4.601807f, 4.640869f, 4.679932f, 4.718994f, 4.758057f, 4.797119f, 4.836182f, 4.875244f, 4.914307f, 
    4.953369f, 4.992432f, 5.031494f, 5.070557f, 5.109619f, 5.148682f, 5.187744f, 5.226807f, 5.265869f, 5.304932f, 
    5.343994f, 5.383057f, 5.422119f, 5.461182f, 5.500244f, 5.539307f, 5.578369f, 5.617432f, 5.656494f, 5.695557f, 
    5.734619f, 5.773682f, 5.812744f, 5.851807f, 5.890869f, 5.929932f, 5.968994f, 6.008057f, 6.047119f, 6.086182f, 
    6.125244f, 6.164307f, 6.203369f, 6.242432f, 6.281494f, 6.320557f, 6.359619f, 6.398682f, 6.437744f, 6.476807f, 
    6.515869f, 6.554932f, 6.593994f, 6.633057f, 6.672119f, 6.711182f, 6.750244f, 6.789307f, 6.828369f, 6.867432f, 
    6.906494f, 6.945557f, 6.984619f, 7.023682f, 7.062744f, 7.101807f, 7.140869f, 7.179932f, 7.218994f, 7.258057f, 
    7.297119f, 7.336182f, 7.375244f, 7.414307f, 7.453369f, 7.492432f, 7.531494f, 7.570557f, 7.609619f, 7.648682f, 
    7.687744f, 7.726807f, 7.765869f, 7.804932f, 7.843994f, 7.883057f, 7.922119f, 7.961182f};

static const float input_2_scale = 0.0194546562f;
static const float input_2_zero_point = 5.4807128906f;
static const int8_t input_2_scales_frac[] = {20};
static const int input_2_sa_dim = -1;

static const int input_2_fx8_frac = 4;

#define INPUT_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {((2 + 1) * 8 + 4) * 4 + 2 ,(2 + 1) * 8 + 4, (2 + 1), 1}, \
    /* .shape = */ {2, 4, 8, 2}, \
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

static const float test_1_out_data[] = {
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000244f, 0.047119f, 0.093994f, 0.140869f, 0.187744f, 0.234619f, 
    0.281494f, 0.328369f, 0.375244f, 0.422119f, 0.468994f, 0.515869f, 0.562744f, 0.609619f, 0.656494f, 0.703369f, 
    0.750244f, 0.797119f, 0.843994f, 0.890869f, 0.937744f, 0.984619f, 1.031494f, 1.078369f, 1.125244f, 1.172119f, 
    1.218994f, 1.265869f, 1.312744f, 1.359619f, 1.406494f, 1.453369f, 1.500244f, 1.547119f, 1.593994f, 1.640869f, 
    1.687744f, 1.734619f, 1.781494f, 1.828369f, 1.875244f, 1.922119f, 1.968994f, 2.015869f, 2.062744f, 2.109619f, 
    2.156494f, 2.203369f, 2.250244f, 2.297119f, 2.343994f, 2.390869f, 2.437744f, 2.484619f, 2.531494f, 2.578369f, 
    2.625244f, 2.672119f, 2.718994f, 2.765869f, 2.812744f, 2.859619f, 2.906494f, 2.953369f};

static const float test_1_out_scale = 0.0115818400f;
static const float test_1_out_zero_point = 1.4766845703f;
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
    -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, 
    -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, 
    -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, 
    -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, -1.000000f, 
    -1.000000f, -1.000000f, -1.000000f, -0.984131f, -0.937256f, -0.890381f, -0.843506f, -0.796631f, -0.749756f, -0.702881f, 
    -0.656006f, -0.609131f, -0.562256f, -0.515381f, -0.468506f, -0.421631f, -0.374756f, -0.327881f, -0.281006f, -0.234131f, 
    -0.187256f, -0.140381f, -0.093506f, -0.046631f,  0.000244f,  0.047119f,  0.093994f,  0.140869f,  0.187744f,  0.234619f, 
     0.281494f,  0.328369f,  0.375244f,  0.422119f,  0.468994f,  0.515869f,  0.562744f,  0.609619f,  0.656494f,  0.703369f, 
     0.750244f,  0.797119f,  0.843994f,  0.890869f,  0.937744f,  0.984619f,  1.000000f,  1.000000f,  1.000000f,  1.000000f, 
     1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f, 
     1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f, 
     1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f, 
     1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f};

static const float test_2_out_scale = 0.0078431377f;
static const float test_2_out_zero_point = 0.0000000000f;
static const int8_t test_2_out_scales_frac[] = {21};
static const int test_2_out_sa_dim = -1;

static const int test_2_out_fx8_frac = 7;

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

static const float test_3_out_data[] = {
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000244f, 0.047119f, 0.093994f, 0.140869f, 0.187744f, 0.234619f, 
    0.281494f, 0.328369f, 0.375244f, 0.422119f, 0.468994f, 0.515869f, 0.562744f, 0.609619f, 0.656494f, 0.703369f, 
    0.750244f, 0.797119f, 0.843994f, 0.890869f, 0.937744f, 0.984619f, 1.031494f, 1.078369f, 1.125244f, 1.172119f, 
    1.218994f, 1.265869f, 1.312744f, 1.359619f, 1.406494f, 1.453369f, 1.500244f, 1.547119f, 1.593994f, 1.640869f, 
    1.687744f, 1.734619f, 1.781494f, 1.828369f, 1.875244f, 1.922119f, 1.968994f, 2.015869f, 2.062744f, 2.109619f, 
    2.156494f, 2.203369f, 2.250244f, 2.297119f, 2.343994f, 2.390869f, 2.437744f, 2.484619f, 2.531494f, 2.578369f, 
    2.625244f, 2.672119f, 2.718994f, 2.765869f, 2.812744f, 2.859619f, 2.906494f, 2.953369f};

static const float test_3_out_scale = 0.0115818400f;
static const float test_3_out_zero_point = 1.4766845703f;
static const int8_t test_3_out_scales_frac[] = {21};
static const int test_3_out_sa_dim = -1;

static const int test_3_out_fx8_frac = 5;

#define TEST_3_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {16 + 12, 1}, \
    /* .shape = */ {8, 16}, \
    /* .rank =  */ 2


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

static const float test_4_out_data[] = {
    3.000244f, 3.039307f, 3.078369f, 3.117432f, 3.156494f, 3.195557f, 3.234619f, 3.273682f, 3.312744f, 3.351807f, 
    3.390869f, 3.429932f, 3.468994f, 3.508057f, 3.547119f, 3.586182f, 3.625244f, 3.664307f, 3.703369f, 3.742432f, 
    3.781494f, 3.820557f, 3.859619f, 3.898682f, 3.937744f, 3.976807f, 4.015869f, 4.054932f, 4.093994f, 4.133057f, 
    4.172119f, 4.211182f, 4.250244f, 4.289307f, 4.328369f, 4.367432f, 4.406494f, 4.445557f, 4.484619f, 4.523682f, 
    4.562744f, 4.601807f, 4.640869f, 4.679932f, 4.718994f, 4.758057f, 4.797119f, 4.836182f, 4.875244f, 4.914307f, 
    4.953369f, 4.992432f, 5.031494f, 5.070557f, 5.109619f, 5.148682f, 5.187744f, 5.226807f, 5.265869f, 5.304932f, 
    5.343994f, 5.383057f, 5.422119f, 5.461182f, 5.500244f, 5.539307f, 5.578369f, 5.617432f, 5.656494f, 5.695557f, 
    5.734619f, 5.773682f, 5.812744f, 5.851807f, 5.890869f, 5.929932f, 5.968994f, 6.000000f, 6.000000f, 6.000000f, 
    6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 
    6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 
    6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 
    6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 
    6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f, 6.000000f};

static const float test_4_out_scale = 0.0117637487f;
static const float test_4_out_zero_point = 4.5001220703f;
static const int8_t test_4_out_scales_frac[] = {21};
static const int test_4_out_sa_dim = -1;

static const int test_4_out_fx8_frac = 4;

#define TEST_4_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {((2 + 0) * 8 + 2) * 4 + 4 ,(2 + 0) * 8 + 2, (2 + 0), 1}, \
    /* .shape = */ {2, 4, 8, 2}, \
    /* .rank =  */ 4


static const mli_tensor test_4_out_tsr_fx16 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_4_out_tsr_sa8 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx8_frac + 8, input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]));
tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data, 
sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim, test_1_out_data, 
sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_1_out_scale, 1, &test_1_out_zero_point, 1, test_1_out_scales_frac, 1);

tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_fx8_frac + 8, test_2_out_data, sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_sa_dim, test_2_out_data, 
sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &test_2_out_scale, 1, &test_2_out_zero_point, 1, test_2_out_scales_frac, 1);

tensor_quantizer test_3_out_fx16(test_3_out_tsr_fx16, test_3_out_fx8_frac + 8, test_3_out_data, sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));
tensor_quantizer test_3_out_sa8(test_3_out_tsr_sa8, test_3_out_sa_dim, test_3_out_data, 
sizeof(test_3_out_data) / sizeof(test_3_out_data[0]), &test_3_out_scale, 1, &test_3_out_zero_point, 1, test_3_out_scales_frac, 1);

tensor_quantizer test_4_out_fx16(test_4_out_tsr_fx16, test_4_out_fx8_frac + 8, test_4_out_data, sizeof(test_4_out_data) / sizeof(test_4_out_data[0]));
tensor_quantizer test_4_out_sa8(test_4_out_tsr_sa8, test_4_out_sa_dim, test_4_out_data, 
sizeof(test_4_out_data) / sizeof(test_4_out_data[0]), &test_4_out_scale, 1, &test_4_out_zero_point, 1, test_4_out_scales_frac, 1);
