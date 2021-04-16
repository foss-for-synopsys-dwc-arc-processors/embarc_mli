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
    -2.999756f, -2.976318f, -2.952881f, -2.929443f, -2.906006f, -2.882568f, -2.859131f, -2.835693f,
    -2.812256f, -2.788818f, -2.765381f, -2.741943f, -2.718506f, -2.695068f, -2.671631f, -2.648193f,
    -2.624756f, -2.601318f, -2.577881f, -2.554443f, -2.531006f, -2.507568f, -2.484131f, -2.460693f,
    -2.437256f, -2.413818f, -2.390381f, -2.366943f, -2.343506f, -2.320068f, -2.296631f, -2.273193f,
    -2.249756f, -2.226318f, -2.202881f, -2.179443f, -2.156006f, -2.132568f, -2.109131f, -2.085693f,
    -2.062256f, -2.038818f, -2.015381f, -1.991943f, -1.968506f, -1.945068f, -1.921631f, -1.898193f,
    -1.874756f, -1.851318f, -1.827881f, -1.804443f, -1.781006f, -1.757568f, -1.734131f, -1.710693f,
    -1.687256f, -1.663818f, -1.640381f, -1.616943f, -1.593506f, -1.570068f, -1.546631f, -1.523193f,
    -1.499756f, -1.476318f, -1.452881f, -1.429443f, -1.406006f, -1.382568f, -1.359131f, -1.335693f,
    -1.312256f, -1.288818f, -1.265381f, -1.241943f, -1.218506f, -1.195068f, -1.171631f, -1.148193f,
    -1.124756f, -1.101318f, -1.077881f, -1.054443f, -1.031006f, -1.007568f, -0.984131f, -0.960693f,
    -0.937256f, -0.913818f, -0.890381f, -0.866943f, -0.843506f, -0.820068f, -0.796631f, -0.773193f,
    -0.749756f, -0.726318f, -0.702881f, -0.679443f, -0.656006f, -0.632568f, -0.609131f, -0.585693f,
    -0.562256f, -0.538818f, -0.515381f, -0.491943f, -0.468506f, -0.445068f, -0.421631f, -0.398193f,
    -0.374756f, -0.351318f, -0.327881f, -0.304443f, -0.281006f, -0.257568f, -0.234131f, -0.210693f,
    -0.187256f, -0.163818f, -0.140381f, -0.116943f, -0.093506f, -0.070068f, -0.046631f, -0.023193f,
    0.000244f, 0.023682f, 0.047119f, 0.070557f, 0.093994f, 0.117432f, 0.140869f, 0.164307f,
    0.187744f, 0.211182f, 0.234619f, 0.258057f, 0.281494f, 0.304932f, 0.328369f, 0.351807f,
    0.375244f, 0.398682f, 0.422119f, 0.445557f, 0.468994f, 0.492432f, 0.515869f, 0.539307f,
    0.562744f, 0.586182f, 0.609619f, 0.633057f, 0.656494f, 0.679932f, 0.703369f, 0.726807f,
    0.750244f, 0.773682f, 0.797119f, 0.820557f, 0.843994f, 0.867432f, 0.890869f, 0.914307f,
    0.937744f, 0.961182f, 0.984619f, 1.008057f, 1.031494f, 1.054932f, 1.078369f, 1.101807f,
    1.125244f, 1.148682f, 1.172119f, 1.195557f, 1.218994f, 1.242432f, 1.265869f, 1.289307f,
    1.312744f, 1.336182f, 1.359619f, 1.383057f, 1.406494f, 1.429932f, 1.453369f, 1.476807f,
    1.500244f, 1.523682f, 1.547119f, 1.570557f, 1.593994f, 1.617432f, 1.640869f, 1.664307f,
    1.687744f, 1.711182f, 1.734619f, 1.758057f, 1.781494f, 1.804932f, 1.828369f, 1.851807f,
    1.875244f, 1.898682f, 1.922119f, 1.945557f, 1.968994f, 1.992432f, 2.015869f, 2.039307f,
    2.062744f, 2.086182f, 2.109619f, 2.133057f, 2.156494f, 2.179932f, 2.203369f, 2.226807f,
    2.250244f, 2.273682f, 2.297119f, 2.320557f, 2.343994f, 2.367432f, 2.390869f, 2.414307f,
    2.437744f, 2.461182f, 2.484619f, 2.508057f, 2.531494f, 2.554932f, 2.578369f, 2.601807f,
    2.625244f, 2.648682f, 2.672119f, 2.695557f, 2.718994f, 2.742432f, 2.765869f, 2.789307f,
    2.812744f, 2.836182f, 2.859619f, 2.883057f, 2.906494f, 2.929932f, 2.953369f, 2.976807f};

static const float input_1_scale = 0.0234375000f;
static const float input_1_zero_point = -0.0114746094f;
static const int8_t input_1_scales_frac[] = {20};
static const int input_1_sa_dim = -1;

static const int input_1_fx8_frac = 5;

#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 2, 4, 16}, \
    /* .rank =  */ 4


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

#define INPUT_1_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 2*(4*(16*4+2)+4), 4*(16*4+2)+4, 16*4+2, 1 }, \
    /* .shape = */ {2, 2, 4, 16}, \
    /* .rank =  */ 4


static const mli_tensor input_1_memstr_tsr_fx16 = {
    INPUT_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_1_memstr_tsr_sa8 = {
    INPUT_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const float input_2_data[] = {
    3.000244f, 3.039307f, 3.078369f, 3.117432f, 3.156494f, 3.195557f, 3.234619f, 3.273682f,
    3.312744f, 3.351807f, 3.390869f, 3.429932f, 3.468994f, 3.508057f, 3.547119f, 3.586182f,
    3.625244f, 3.664307f, 3.703369f, 3.742432f, 3.781494f, 3.820557f, 3.859619f, 3.898682f,
    3.937744f, 3.976807f, 4.015869f, 4.054932f, 4.093994f, 4.133057f, 4.172119f, 4.211182f,
    4.250244f, 4.289307f, 4.328369f, 4.367432f, 4.406494f, 4.445557f, 4.484619f, 4.523682f,
    4.562744f, 4.601807f, 4.640869f, 4.679932f, 4.718994f, 4.758057f, 4.797119f, 4.836182f,
    4.875244f, 4.914307f, 4.953369f, 4.992432f, 5.031494f, 5.070557f, 5.109619f, 5.148682f,
    5.187744f, 5.226807f, 5.265869f, 5.304932f, 5.343994f, 5.383057f, 5.422119f, 5.461182f,
    5.500244f, 5.539307f, 5.578369f, 5.617432f, 5.656494f, 5.695557f, 5.734619f, 5.773682f,
    5.812744f, 5.851807f, 5.890869f, 5.929932f, 5.968994f, 6.008057f, 6.047119f, 6.086182f,
    6.125244f, 6.164307f, 6.203369f, 6.242432f, 6.281494f, 6.320557f, 6.359619f, 6.398682f,
    6.437744f, 6.476807f, 6.515869f, 6.554932f, 6.593994f, 6.633057f, 6.672119f, 6.711182f,
    6.750244f, 6.789307f, 6.828369f, 6.867432f, 6.906494f, 6.945557f, 6.984619f, 7.023682f,
    7.062744f, 7.101807f, 7.140869f, 7.179932f, 7.218994f, 7.258057f, 7.297119f, 7.336182f,
    7.375244f, 7.414307f, 7.453369f, 7.492432f, 7.531494f, 7.570557f, 7.609619f, 7.648682f,
    7.687744f, 7.726807f, 7.765869f, 7.804932f, 7.843994f, 7.883057f, 7.922119f, 7.961182f};

static const float input_2_scale = 0.039062503937f;
static const float input_2_zero_point = 3.000244f;
static const int8_t input_2_scales_frac[] = {19};
static const int input_2_sa_dim = -1;

static const int input_2_fx8_frac = 4;

#define INPUT_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
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
    -0.995052f, -0.994815f, -0.994568f, -0.994307f, -0.994035f, -0.993750f, -0.993451f, -0.993137f,
    -0.992809f, -0.992465f, -0.992105f, -0.991728f, -0.991333f, -0.990919f, -0.990485f, -0.990031f,
    -0.989555f, -0.989056f, -0.988534f, -0.987987f, -0.987414f, -0.986814f, -0.986186f, -0.985528f,
    -0.984838f, -0.984116f, -0.983360f, -0.982569f, -0.981740f, -0.980872f, -0.979963f, -0.979012f,
    -0.978016f, -0.976973f, -0.975881f, -0.974738f, -0.973542f, -0.972289f, -0.970979f, -0.969607f,
    -0.968172f, -0.966670f, -0.965098f, -0.963454f, -0.961734f, -0.959934f, -0.958052f, -0.956083f,
    -0.954023f, -0.951870f, -0.949618f, -0.947264f, -0.944803f, -0.942231f, -0.939542f, -0.936733f,
    -0.933797f, -0.930730f, -0.927526f, -0.924179f, -0.920685f, -0.917037f, -0.913228f, -0.909253f,
    -0.905104f, -0.900776f, -0.896261f, -0.891552f, -0.886643f, -0.881525f, -0.876191f, -0.870634f,
    -0.864845f, -0.858817f, -0.852541f, -0.846009f, -0.839213f, -0.832144f, -0.824794f, -0.817155f,
    -0.809217f, -0.800972f, -0.792412f, -0.783528f, -0.774311f, -0.764754f, -0.754849f, -0.744586f,
    -0.733959f, -0.722960f, -0.711582f, -0.699818f, -0.687662f, -0.675107f, -0.662149f, -0.648783f,
    -0.635003f, -0.620808f, -0.606193f, -0.591157f, -0.575699f, -0.559818f, -0.543515f, -0.526791f,
    -0.509649f, -0.492093f, -0.474127f, -0.455757f, -0.436991f, -0.417837f, -0.398303f, -0.378402f,
    -0.358145f, -0.337544f, -0.316615f, -0.295374f, -0.273836f, -0.252020f, -0.229944f, -0.207630f,
    -0.185097f, -0.162368f, -0.139466f, -0.116413f, -0.093234f, -0.069954f, -0.046597f, -0.023189f,
    0.000244f, 0.023677f, 0.047084f, 0.070440f, 0.093718f, 0.116895f, 0.139945f, 0.162844f,
    0.185569f, 0.208097f, 0.230407f, 0.252477f, 0.274287f, 0.295819f, 0.317055f, 0.337977f,
    0.358570f, 0.378820f, 0.398714f, 0.418240f, 0.437386f, 0.456144f, 0.474506f, 0.492463f,
    0.510011f, 0.527144f, 0.543859f, 0.560153f, 0.576026f, 0.591475f, 0.606502f, 0.621108f,
    0.635295f, 0.649065f, 0.662423f, 0.675373f, 0.687919f, 0.700067f, 0.711823f, 0.723193f,
    0.734184f, 0.744804f, 0.755059f, 0.764957f, 0.774507f, 0.783716f, 0.792594f, 0.801147f,
    0.809385f, 0.817317f, 0.824950f, 0.832294f, 0.839357f, 0.846148f, 0.852674f, 0.858945f,
    0.864968f, 0.870752f, 0.876305f, 0.881634f, 0.886747f, 0.891653f, 0.896357f, 0.900868f,
    0.905192f, 0.909337f, 0.913309f, 0.917114f, 0.920759f, 0.924251f, 0.927594f, 0.930795f,
    0.933859f, 0.936792f, 0.939599f, 0.942286f, 0.944856f, 0.947314f, 0.949666f, 0.951916f,
    0.954067f, 0.956124f, 0.958092f, 0.959972f, 0.961770f, 0.963489f, 0.965132f, 0.966702f,
    0.968202f, 0.969636f, 0.971007f, 0.972316f, 0.973567f, 0.974762f, 0.975904f, 0.976995f,
    0.978037f, 0.979032f, 0.979983f, 0.980891f, 0.981758f, 0.982586f, 0.983376f, 0.984132f,
    0.984853f, 0.985541f, 0.986199f, 0.986827f, 0.987426f, 0.987999f, 0.988545f, 0.989067f,
    0.989565f, 0.990040f, 0.990494f, 0.990928f, 0.991341f, 0.991736f, 0.992113f, 0.992473f,
    0.992816f, 0.993144f, 0.993457f, 0.993756f, 0.994041f, 0.994313f, 0.994573f, 0.994820f};

static const float test_1_out_scale = 0.0078034233f;
static const float test_1_out_zero_point = -0.0001159906f;
static const int8_t test_1_out_scales_frac[] = {22};
static const int test_1_out_sa_dim = -1;

static const int test_1_out_fx8_frac = 7;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 2, 4, 16}, \
    /* .rank =  */ 4


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

#define TEST_1_OUT_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 2*(4*(16*4+2)+4), 4*(16*4+2)+4, 16*4+2, 1 }, \
    /* .shape = */ {2, 2, 4, 16}, \
    /* .rank =  */ 4


static const mli_tensor test_1_out_memstr_tsr_fx16 = {
    TEST_1_OUT_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_1_out_memstr_tsr_sa8 = {
    TEST_1_OUT_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float test_2_out_data[] = {
    0.995057f, 0.995428f, 0.995771f, 0.996088f, 0.996381f, 0.996653f, 0.996904f, 0.997136f,
    0.997351f, 0.997550f, 0.997734f, 0.997904f, 0.998061f, 0.998207f, 0.998342f, 0.998466f,
    0.998581f, 0.998688f, 0.998786f, 0.998878f, 0.998962f, 0.999040f, 0.999112f, 0.999179f,
    0.999240f, 0.999297f, 0.999350f, 0.999399f, 0.999444f, 0.999486f, 0.999525f, 0.999560f,
    0.999593f, 0.999624f, 0.999652f, 0.999678f, 0.999703f, 0.999725f, 0.999745f, 0.999765f,
    0.999782f, 0.999799f, 0.999814f, 0.999828f, 0.999841f, 0.999853f, 0.999864f, 0.999874f,
    0.999884f, 0.999892f, 0.999900f, 0.999908f, 0.999915f, 0.999921f, 0.999927f, 0.999933f,
    0.999938f, 0.999942f, 0.999947f, 0.999951f, 0.999954f, 0.999958f, 0.999961f, 0.999964f,
    0.999967f, 0.999969f, 0.999972f, 0.999974f, 0.999976f, 0.999977f, 0.999979f, 0.999981f,
    0.999982f, 0.999984f, 0.999985f, 0.999986f, 0.999987f, 0.999988f, 0.999989f, 0.999990f,
    0.999990f, 0.999991f, 0.999992f, 0.999992f, 0.999993f, 0.999994f, 0.999994f, 0.999995f,
    0.999995f, 0.999995f, 0.999996f, 0.999996f, 0.999996f, 0.999997f, 0.999997f, 0.999997f,
    0.999997f, 0.999998f, 0.999998f, 0.999998f, 0.999998f, 0.999998f, 0.999998f, 0.999999f,
    0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f,
    0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f};

static const float test_2_out_scale = 0.0000193832f;
static const float test_2_out_zero_point = 0.9975285530f;
static const int8_t test_2_out_scales_frac[] = {30};
static const int test_2_out_sa_dim = -1;

static const int test_2_out_fx8_frac = 7;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 4, 8, 2}, \
    /* .rank =  */ 4


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

tensor_quantizer input_1_memstr_fx16(input_1_memstr_tsr_fx16, input_1_fx8_frac + 8, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_memstr_sa8(input_1_memstr_tsr_sa8, input_1_sa_dim, input_1_data, 
sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx8_frac + 8, input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]));
tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data, 
sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim, test_1_out_data, 
sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_1_out_scale, 1, &test_1_out_zero_point, 1, test_1_out_scales_frac, 1);

tensor_quantizer test_1_out_memstr_fx16(test_1_out_memstr_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_memstr_sa8(test_1_out_memstr_tsr_sa8, test_1_out_sa_dim, test_1_out_data, 
sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_1_out_scale, 1, &test_1_out_zero_point, 1, test_1_out_scales_frac, 1);

tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_fx8_frac + 8, test_2_out_data, sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_sa_dim, test_2_out_data, 
sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &test_2_out_scale, 1, &test_2_out_zero_point, 1, test_2_out_scales_frac, 1);
