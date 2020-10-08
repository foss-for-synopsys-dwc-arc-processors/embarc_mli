/*
* Copyright 2019-2020, Synopsys, Inc.
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
extern mli::tst::tensor_quantizer input_3_memstr_fx16;
extern mli::tst::tensor_quantizer input_3_memstr_sa8;


// Extracted Output vectors Declaration
//===================================================

extern mli::tst::tensor_quantizer test_3_out_fx16;
extern mli::tst::tensor_quantizer test_3_out_sa8;

extern mli::tst::tensor_quantizer test_4_out_fx16;
extern mli::tst::tensor_quantizer test_4_out_sa8;

extern mli::tst::tensor_quantizer test_5_out_fx16;
extern mli::tst::tensor_quantizer test_5_out_sa8;

extern mli::tst::tensor_quantizer test_6_out_fx16;
extern mli::tst::tensor_quantizer test_6_out_sa8;

// Tests configuration structures Declaration
//========================================
extern const mli_softmax_cfg test_1_cfg;
extern const mli_softmax_cfg test_2_cfg;
extern const mli_softmax_cfg test_3_cfg;
extern const mli_softmax_cfg test_4_cfg;
extern const mli_softmax_cfg test_5_cfg;
extern const mli_softmax_cfg test_6_cfg;



// Generated input vectors 
//========================================
constexpr int input_sa_dim = -1;


static const float input_1_data[] = {
    -1.919013619f, -3.533751488f, 0.860428214f, 2.761076927f, 3.128014326f, 3.955837488f, 2.771568537f, 0.069868460f, 
    -3.750723839f, -4.342554092f
};

static const float input_1_scale[] = { 0.032542713f };
static const float input_1_zero_point[] = { -0.19335818f };
static const int8_t input_1_scales_frac[] = { 19 };
constexpr int input_1_fx16_frac = 12;


#define INPUT_1_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /*.mem_stride =*/ { 0 }, \
    /*.shape =*/ {10}, \
    /*.rank =*/ 1


static const mli_tensor input_1_tsr_sa8 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_1_tsr_fx16 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx16_frac, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_sa_dim,
    input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]),
    input_1_scale, sizeof(input_1_scale) / sizeof(input_1_scale[0]),
    input_1_zero_point, sizeof(input_1_zero_point) / sizeof(input_1_zero_point[0]),
    input_1_scales_frac, sizeof(input_1_scales_frac) / sizeof(input_1_scales_frac[0]));


static const float input_2_data[] = {
    -4.474763870f, -3.259614706f, -5.320953846f, 0.120907545f, -3.312439919f, 4.264426708f, -2.489409924f, 1.311650872f, 
    3.822283030f, 5.749427795f
};

static const float input_2_scale[] = { 0.04341326f };
static const float input_2_zero_point[] = { 0.21423721f };
static const int8_t input_2_scales_frac[] = { 19 };
static const int input_2_fx16_frac = 12;


#define INPUT_2_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /*.mem_stride =*/ { 0 }, \
    /*.shape =*/ {10}, \
    /*.rank =*/ 1


static const mli_tensor input_2_tsr_sa8 = {
    INPUT_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_2_tsr_fx16 = {
    INPUT_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx16_frac, input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]));
tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_sa_dim, 
    input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]),
    input_2_scale, sizeof(input_2_scale) / sizeof(input_2_scale[0]),
    input_2_zero_point, sizeof(input_2_zero_point) / sizeof(input_2_zero_point[0]),
    input_2_scales_frac, sizeof(input_2_scales_frac) / sizeof(input_2_scales_frac[0]));


static const float input_3_data[] = {
    4.733333111f, -2.733333349f, 5.800000191f, 2.066666603f, 3.666666746f, -0.066666670f, -1.933333278f, -1.666666627f,
    2.333333254f, 7.933333397f, -0.866666675f, 1.799999952f, 6.066666603f, 9.000000000f, 11.666666985f, 10.066666603f,
    11.399999619f, 11.933333397f, 12.199999809f, -2.466666698f, 8.733333588f, 2.866666555f, 4.199999809f, 7.400000095f,
    2.599999905f, 1.000000000f, 5.533333302f, 8.199999809f, 5.000000000f, 8.466666222f, 7.133333206f, 5.266666889f,
    6.866666794f, 3.933333397f, -1.133333325f, 3.133333445f, 10.333333015f, 7.666666508f, 12.733333588f, -0.600000024f,
    4.466666698f, 0.733333349f, 10.600000381f, -3.000000000f, 10.866666794f, 0.200000003f, 9.533333778f, -2.200000048f,
    -0.333333343f, 6.599999905f, 3.400000095f, 1.533333302f, 9.800000191f, 0.466666669f, 1.266666651f, 12.466666222f,
    9.266666412f, 11.133333206f, -1.399999976f, 6.333333492f
};

static const float input_3_scale[] = { 0.06169934570789337f };
static const float input_3_zero_point[] = {4.86666679f};
static const int8_t input_3_scales_frac[] = { 19 };
static const int input_3_fx16_frac = 11;


#define INPUT_3_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /*.mem_stride =*/ { 0 }, \
    /*.shape =*/ {3, 4, 5}, \
    /*.rank =*/ 3


static const mli_tensor input_3_tsr_sa8 = {
    INPUT_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_3_tsr_fx16 = {
    INPUT_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_3_fx16(input_3_tsr_fx16, input_3_fx16_frac, input_3_data, sizeof(input_3_data) / sizeof(input_3_data[0]));
tensor_quantizer input_3_sa8(input_3_tsr_sa8, input_sa_dim,
                             input_3_data, sizeof(input_3_data) / sizeof(input_3_data[0]),
                             input_3_scale, sizeof(input_3_scale) / sizeof(input_3_scale[0]),
                             input_3_zero_point, sizeof(input_3_zero_point) / sizeof(input_3_zero_point[0]),
                             input_3_scales_frac, sizeof(input_3_scales_frac) / sizeof(input_3_scales_frac[0]));

#define INPUT_3_MEMSTR_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /*.mem_stride =*/ { (4*(5+1))*2, (5+1)*1, 1}, \
    /*.shape =*/ {3, 4, 5}, \
    /*.rank =*/ 3


static const mli_tensor input_3_memstr_tsr_sa8 = {
    INPUT_3_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_3_memstr_tsr_fx16 = {
    INPUT_3_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_3_memstr_fx16(input_3_memstr_tsr_fx16, input_3_fx16_frac, 
                                     input_3_data, sizeof(input_3_data) / sizeof(input_3_data[0]));
tensor_quantizer input_3_memstr_sa8(input_3_memstr_tsr_sa8, input_sa_dim,
                                    input_3_data, sizeof(input_3_data) / sizeof(input_3_data[0]),
                                    input_3_scale, sizeof(input_3_scale) / sizeof(input_3_scale[0]),
                                    input_3_zero_point, sizeof(input_3_zero_point) / sizeof(input_3_zero_point[0]),
                                    input_3_scales_frac, sizeof(input_3_scales_frac) / sizeof(input_3_scales_frac[0]));

// Extracted Output vectors 
//===================================================

static const float test_out_scale[] = { 1.0f / 256.0f };
static const float test_out_zero_point[] = { 0.5 };
static const int8_t test_out_scales_frac[] = { 23 };
constexpr int test_out_frac_fx16 = 15;
constexpr int test_out_sa_dim = -1;

static const float test_1_out_data[] = {
    0.001327858f, 0.000264168f, 0.021391824f, 0.143116325f, 0.206560850f, 0.472679824f, 0.144625738f, 0.009703138f,
    0.000212643f, 0.000117658f
};

#define TEST_1_2_OUT_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /*.mem_stride =*/ { 0 }, \
    /*.shape =*/ { 10 }, \
    /*.rank =*/ 1


static mli_tensor test_1_out_tsr_sa8 = {
    TEST_1_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static mli_tensor test_1_out_tsr_fx16 = {
    TEST_1_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_out_frac_fx16, 
    test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_out_sa_dim, 
    test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]),
    test_out_scale, sizeof(test_out_scale) / sizeof(test_out_scale[0]),
    test_out_zero_point, sizeof(test_out_zero_point) / sizeof(test_out_zero_point[0]),
    test_out_scales_frac, sizeof(test_out_scales_frac) / sizeof(test_out_scales_frac[0]));

//=====================================================
static const float test_2_out_data[] = {
    0.000026139f, 0.000088109f, 0.000011215f, 0.002589191f, 0.000083576f, 0.163181737f, 0.000190334f, 0.008517208f,
    0.104869939f, 0.720442533f
};

static mli_tensor test_2_out_tsr_sa8 = {
    TEST_1_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static mli_tensor test_2_out_tsr_fx16 = {
    TEST_1_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_out_frac_fx16, 
    test_2_out_data, sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_out_sa_dim, 
    test_2_out_data, sizeof(test_2_out_data) / sizeof(test_2_out_data[0]),
    test_out_scale, sizeof(test_out_scale) / sizeof(test_out_scale[0]),
    test_out_zero_point, sizeof(test_out_zero_point) / sizeof(test_out_zero_point[0]),
    test_out_scales_frac, sizeof(test_out_scales_frac) / sizeof(test_out_scales_frac[0]));

//====================================================
static const float test_3_out_data[] = {
    0.017741788f, 0.003295369f, 0.008149140f, 0.004804607f, 0.000745838f, 0.191891298f, 0.000010285f, 0.000051871f,
    0.064677194f, 0.336904407f, 0.000327521f, 0.029589532f, 0.022200603f, 0.993541718f, 0.999966860f, 0.083165988f,
    0.683715940f, 0.683360100f, 0.369739503f, 0.000150564f, 0.968669236f, 0.891154647f, 0.001645282f, 0.995165110f,
    0.000256683f, 0.557574332f, 0.017986018f, 0.999917746f, 0.930828810f, 0.574288607f, 0.976325572f, 0.947746992f,
    0.049408358f, 0.006262688f, 0.000002761f, 0.000081066f, 0.235303447f, 0.009586493f, 0.630260050f, 0.000973650f,
    0.013588941f, 0.105549999f, 0.990205586f, 0.000030285f, 0.998997509f, 0.250534296f, 0.982003748f, 0.000030430f,
    0.004493994f, 0.088807024f, 0.023346871f, 0.022663461f, 0.928391039f, 0.000195527f, 0.000030431f, 0.916752994f,
    0.080980577f, 0.307053417f, 0.000000459f, 0.998875797f
}; 

#define TEST_3_4_5_OUT_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /*.mem_stride =*/ { 0 }, \
    /*.shape =*/ { 3, 4, 5 }, \
    /*.rank =*/ 3


static mli_tensor test_3_out_tsr_sa8 = {
    TEST_3_4_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static mli_tensor test_3_out_tsr_fx16 = {
    TEST_3_4_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer test_3_out_fx16(test_3_out_tsr_fx16, test_out_frac_fx16, 
    test_3_out_data, sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));
tensor_quantizer test_3_out_sa8(test_3_out_tsr_sa8, test_out_sa_dim,
    test_3_out_data, sizeof(test_3_out_data) / sizeof(test_3_out_data[0]),
    test_out_scale, sizeof(test_out_scale) / sizeof(test_out_scale[0]),
    test_out_zero_point, sizeof(test_out_zero_point) / sizeof(test_out_zero_point[0]),
    test_out_scales_frac, sizeof(test_out_scales_frac) / sizeof(test_out_scales_frac[0]));

static const float test_4_out_data[] = { 
    0.231487468f, 0.000132372f, 0.672628284f, 0.016084552f, 0.079667315f, 0.000334075f, 0.000051661f, 0.000067449f,
    0.003682569f, 0.995864272f, 0.000003359f, 0.000048335f, 0.003445523f, 0.064741932f, 0.931760788f, 0.050752819f,
    0.192539275f, 0.328203529f, 0.428504139f, 0.000000183f, 0.781650841f, 0.002213868f, 0.008398678f, 0.206040889f,
    0.001695664f, 0.000308941f, 0.028752604f, 0.413804978f, 0.016867606f, 0.540265799f, 0.509791613f, 0.078833319f,
    0.390463918f, 0.020780234f, 0.000130986f, 0.000061735f, 0.082689509f, 0.005745554f, 0.911501646f, 0.000001476f,
    0.000939991f, 0.000022478f, 0.433307976f, 0.000000538f, 0.565729022f, 0.000083947f, 0.949336171f, 0.000007615f,
    0.000049247f, 0.050523020f, 0.001657903f, 0.000256375f, 0.997801006f, 0.000088232f, 0.000196365f, 0.765386343f,
    0.031198841f, 0.201753706f, 0.000000727f, 0.001660382f 
};

static mli_tensor test_4_out_tsr_sa8 = {
    TEST_3_4_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static mli_tensor test_4_out_tsr_fx16 = {
    TEST_3_4_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer test_4_out_fx16(test_4_out_tsr_fx16, test_out_frac_fx16, 
    test_4_out_data, sizeof(test_4_out_data) / sizeof(test_4_out_data[0]));
tensor_quantizer test_4_out_sa8(test_4_out_tsr_sa8, test_out_sa_dim, 
    test_4_out_data, sizeof(test_4_out_data) / sizeof(test_4_out_data[0]),
    test_out_scale, sizeof(test_out_scale) / sizeof(test_out_scale[0]),
    test_out_zero_point, sizeof(test_out_zero_point) / sizeof(test_out_zero_point[0]),
    test_out_scales_frac, sizeof(test_out_scales_frac) / sizeof(test_out_scales_frac[0]));


static const float test_5_out_data[] = {
    0.000078522f, 0.000000045f, 0.000228161f, 0.000005456f, 0.000027024f, 0.000000646f, 0.000000100f, 0.000000130f,
    0.000007123f, 0.001926351f, 0.000000290f, 0.000004179f, 0.000297887f, 0.005597354f, 0.080556676f, 0.016264105f,
    0.061700597f, 0.105175182f, 0.137317225f, 0.000000059f, 0.004287173f, 0.000012143f, 0.000046065f, 0.001130086f,
    0.000009300f, 0.000001878f, 0.000174755f, 0.002515053f, 0.000102519f, 0.003283665f, 0.000865565f, 0.000133850f,
    0.000662961f, 0.000035282f, 0.000000222f, 0.000015853f, 0.021234494f, 0.001475446f, 0.234071717f, 0.000000379f,
    0.000060142f, 0.000001438f, 0.027723886f, 0.000000034f, 0.036196440f, 0.000000844f, 0.009541281f, 0.000000077f,
    0.000000495f, 0.000507780f, 0.000020698f, 0.000003201f, 0.012457143f, 0.000001102f, 0.000002452f, 0.179282039f,
    0.007307932f, 0.047258247f, 0.000000170f, 0.000388924f
};


static mli_tensor test_5_out_tsr_sa8 = {
    TEST_3_4_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static mli_tensor test_5_out_tsr_fx16 = {
    TEST_3_4_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer test_5_out_fx16(test_5_out_tsr_fx16, test_out_frac_fx16, 
    test_5_out_data, sizeof(test_5_out_data) / sizeof(test_5_out_data[0]));
tensor_quantizer test_5_out_sa8(test_5_out_tsr_sa8, test_out_sa_dim, 
    test_5_out_data, sizeof(test_5_out_data) / sizeof(test_5_out_data[0]),
    test_out_scale, sizeof(test_out_scale) / sizeof(test_out_scale[0]),
    test_out_zero_point, sizeof(test_out_zero_point) / sizeof(test_out_zero_point[0]),
    test_out_scales_frac, sizeof(test_out_scales_frac) / sizeof(test_out_scales_frac[0]));



static const float test_6_out_data[] = { 
    0.000078522f, 0.000000045f, 0.000228161f, 0.000005456f, 0.000027024f, 0.000000646f, 0.000000100f, 0.000000130f,
    0.000007123f, 0.001926351f, 0.000000290f, 0.000004179f, 0.000297887f, 0.005597354f, 0.080556676f, 0.016264105f,
    0.061700597f, 0.105175182f, 0.137317225f, 0.000000059f, 0.004287173f, 0.000012143f, 0.000046065f, 0.001130086f,
    0.000009300f, 0.000001878f, 0.000174755f, 0.002515053f, 0.000102519f, 0.003283665f, 0.000865565f, 0.000133850f,
    0.000662961f, 0.000035282f, 0.000000222f, 0.000015853f, 0.021234494f, 0.001475446f, 0.234071717f, 0.000000379f,
    0.000060142f, 0.000001438f, 0.027723886f, 0.000000034f, 0.036196440f, 0.000000844f, 0.009541281f, 0.000000077f,
    0.000000495f, 0.000507780f, 0.000020698f, 0.000003201f, 0.012457143f, 0.000001102f, 0.000002452f, 0.179282039f,
    0.007307932f, 0.047258247f, 0.000000170f, 0.000388924f
};

#define TEST_6_OUT_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /*.mem_stride =*/ { (5*2)*4*2, 5*2, 1 }, \
    /*.shape =*/ { 3, 4, 5 }, \
    /*.rank =*/ 3


static mli_tensor test_6_out_tsr_sa8 = {
    TEST_6_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static mli_tensor test_6_out_tsr_fx16 = {
    TEST_6_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer test_6_out_fx16(test_6_out_tsr_fx16, test_out_frac_fx16, 
    test_6_out_data, sizeof(test_6_out_data) / sizeof(test_6_out_data[0]));
tensor_quantizer test_6_out_sa8(test_6_out_tsr_sa8, test_out_sa_dim, 
    test_6_out_data, sizeof(test_6_out_data) / sizeof(test_6_out_data[0]),
    test_out_scale, sizeof(test_out_scale) / sizeof(test_out_scale[0]),
    test_out_zero_point, sizeof(test_out_zero_point) / sizeof(test_out_zero_point[0]),
    test_out_scales_frac, sizeof(test_out_scales_frac) / sizeof(test_out_scales_frac[0]));

// Tests configuration structures
//========================================
const mli_softmax_cfg test_1_cfg = {/* .axis = */-1 };

const mli_softmax_cfg test_2_cfg = {/* .axis = */-1 };

const mli_softmax_cfg test_3_cfg = {/* .axis = */0};

const mli_softmax_cfg test_4_cfg = {/* .axis = */2};

const mli_softmax_cfg test_5_cfg = {/* .axis = */-1};

const mli_softmax_cfg test_6_cfg = {/* .axis = */-1};


