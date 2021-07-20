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
extern mli::tst::tensor_quantizer input_1_memstr_fx16;
extern mli::tst::tensor_quantizer input_1_sa8;
extern mli::tst::tensor_quantizer input_1_memstr_sa8;

extern mli::tst::tensor_quantizer input_2_fx16;
extern mli::tst::tensor_quantizer input_2_sa8;

// Generated weights vectors Declaration
//========================================

extern mli::tst::tensor_quantizer weights_1_fx16;
extern mli::tst::tensor_quantizer weights_1_fx8;
extern mli::tst::tensor_quantizer weights_1_sa8_per_axis;

extern mli::tst::tensor_quantizer weights_2_fx16;
extern mli::tst::tensor_quantizer weights_2_memstr_fx16;
extern mli::tst::tensor_quantizer weights_2_fx8;
extern mli::tst::tensor_quantizer weights_2_memstr_fx8;
extern mli::tst::tensor_quantizer weights_2_sa8;
extern mli::tst::tensor_quantizer weights_2_memstr_sa8_per_axis;

extern mli::tst::tensor_quantizer weights_3_fx16;
extern mli::tst::tensor_quantizer weights_3_fx8;
extern mli::tst::tensor_quantizer weights_3_sa8_per_axis;

extern mli::tst::tensor_quantizer weights_4_fx16;
extern mli::tst::tensor_quantizer weights_4_fx8;
extern mli::tst::tensor_quantizer weights_4_sa8_per_axis;

extern mli::tst::tensor_quantizer weights_5_fx8;
extern mli::tst::tensor_quantizer weights_5_fx16;
extern mli::tst::tensor_quantizer weights_5_sa8;

extern mli::tst::tensor_quantizer bias_1_fx16;
extern mli::tst::tensor_quantizer bias_1_fx8;
extern mli::tst::tensor_quantizer bias_1_sa32_per_axis;

extern mli::tst::tensor_quantizer bias_2_fx16;
extern mli::tst::tensor_quantizer bias_2_fx8;
extern mli::tst::tensor_quantizer bias_2_i1_w2_sa32;
extern mli::tst::tensor_quantizer bias_2_i1_w2_sa32_per_axis;
extern mli::tst::tensor_quantizer bias_2_i1_w3_sa32_per_axis;
extern mli::tst::tensor_quantizer bias_2_i1_w4_sa32_per_axis;

extern mli::tst::tensor_quantizer bias_3_fx8;
extern mli::tst::tensor_quantizer bias_3_fx16;
extern mli::tst::tensor_quantizer bias_3_i2_w5_sa32;

// Extracted Output vectors Declaration
//===================================================

extern mli::tst::tensor_quantizer test_1_out_fx16;
extern mli::tst::tensor_quantizer test_1_out_sa8;

extern mli::tst::tensor_quantizer test_2_out_fx16;
extern mli::tst::tensor_quantizer test_2_out_sa8;

extern mli::tst::tensor_quantizer test_3_out_fx16;
extern mli::tst::tensor_quantizer test_3_out_sa8;

extern mli::tst::tensor_quantizer test_4_out_fx16;
extern mli::tst::tensor_quantizer test_4_out_sa8;

extern mli::tst::tensor_quantizer test_5_out_fx16;
extern mli::tst::tensor_quantizer test_5_out_sa8;

extern mli::tst::tensor_quantizer test_6_out_fx16;
extern mli::tst::tensor_quantizer test_6_out_sa8;

extern mli::tst::tensor_quantizer test_7_out_fx16;
extern mli::tst::tensor_quantizer test_7_out_sa8;

extern mli::tst::tensor_quantizer test_8_out_fx16;
extern mli::tst::tensor_quantizer test_8_out_sa8;

extern mli::tst::tensor_quantizer test_9_out_fx16;
extern mli::tst::tensor_quantizer test_9_out_sa8;

extern mli::tst::tensor_quantizer test_10_out_fx16;
extern mli::tst::tensor_quantizer test_10_out_sa8;

// Tests configuration structures Declaration
//========================================

extern const mli_conv2d_cfg test_1_cfg;
extern const mli_conv2d_cfg test_2_cfg;
extern const mli_conv2d_cfg test_3_cfg;
extern const mli_conv2d_cfg test_4_cfg;
extern const mli_conv2d_cfg test_5_cfg;
extern const mli_conv2d_cfg test_6_cfg;
extern const mli_conv2d_cfg test_7_cfg;
extern const mli_conv2d_cfg test_8_cfg;
extern const mli_conv2d_cfg test_9_cfg;
extern const mli_conv2d_cfg test_10_cfg;

// Generated input vectors 
//========================================
static const float input_1_data[] = {
    1.499679f, -0.195957f, -0.101497f, 1.08470f, -0.127001f, 1.921374f, 1.265131f, 1.050833f, 0.828557f, 1.759060f, 
    0.740656f, 0.602529f, 0.770518f, -0.055207f, 1.764953f, 0.056915f, 1.798162f, -0.974392f, -0.193173f, 1.791836f, 
    0.087221f, 0.050762f, -0.795762f, -0.861908f, -0.521138f, 0.569642f, -0.466072f, 0.950747f, 0.105315f, 1.743229f, 
    -0.345538f, 0.211855f, -0.675123f, -0.174242f, -0.210372f, 0.379746f, 1.743715f, -0.732649f, -0.078323f, 0.358176f, 
    -0.621651f, 0.547825f, -0.846128f, -0.170151f, 1.669240f, 1.788261f, 0.269071f, -0.584010f, -0.169977f, -0.151245f, 
    -0.711375f, 0.605190f, -0.521042f, 0.967539f, -0.951895f, 1.476784f, 1.189694f, 0.493913f, 1.585523f, 0.662098f, 
    -0.136224f, -0.129915f, 0.816717f, 1.840368f, 1.695302f, 1.008417f, -0.657568f, 1.430704f, 1.290535f, 0.754588f, 
    0.868083f, 0.100421f, 0.658160f, 0.164856f, 0.491118f, -0.730381f, -0.211975f, -0.864242f, -0.159679f, 1.634750f, 
    -0.874014f, 0.053465f, -0.872567f, 1.262317f, 0.582659f, 1.509164f, 0.571157f, 1.600416f, 1.445452f, 1.635805f, 
    -0.016744f, 1.974257f, 1.793061f, -0.750511f, -0.211884f, -0.806919f, -0.263645f, -0.336632f, 0.57630f, 1.501349f, 
    0.066899f, 1.175289f, -0.585119f, 1.680775f, 1.642291f, 1.548584f, -0.574892f, 1.02780f, 0.628294f, 0.380053f, 
    0.004982f, -0.463713f, -0.902412f, 0.729615f, 0.346627f, 1.194765f, 0.783888f, 1.006757f, -0.953241f, 0.968442f, 
    1.131668f, 1.994732f, 0.113902f, 0.782195f, 0.953184f, -0.217187f, 1.710644f, 0.946193f, 1.710289f, 1.380116f, 
    1.373977f, 0.230183f, 1.108058f, -0.623367f, 1.587298f, 1.012207f, -0.438984f, 0.206704f, 0.643520f, 0.858104f, 
    0.951818f, -0.029513f, -0.948736f, 1.503883f, 1.769106f, 0.514857f, -0.052131f
};

static const float input_1_scale = 0.0116436249f;
static const float input_1_zero_point = 0.5101699829f;
static const int8_t input_1_scales_frac[] = {21};
static const int input_1_sa_dim = -1;

static const int input_1_fx8_frac = 6;

#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {7, 7, 3}, \
    /* .rank =  */ 3

#define INPUT_1_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(13 * 3) * 2, (3 * 2) * 1, 1}, \
    /* .shape = */ {7, 7, 3}, \
    /* .rank =  */ 3

#define INPUT_1b_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(13 * 3) * 2, (3 * 2) * 1, 1}, \
    /* .shape = */ {6, 7, 3}, \
    /* .rank =  */ 3

static const mli_tensor input_1_tsr_fx16 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_1_memstr_tsr_fx16 = {
    INPUT_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_1_tsr_sa8 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_1_memstr_tsr_sa8 = {
    INPUT_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_1b_memstr_tsr_fx16 = {
    INPUT_1b_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_1b_memstr_tsr_sa8 = {
    INPUT_1b_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float input_2_data[] = {
    41587.f, 46283.f, -6322.f, 22712.f, -3568.f, -3045.f, 41330.f, 31563.f, -14780.f, 25440.f, 2596.f, 37906.f,
    24698.f, -8836.f, 49181.f, 18121.f, 36335.f, -13315.f, 39941.f, 58422.f, 20540.f, -13695.f, 60515.f, 31796.f,
    -5480.f, -15268.f, 16507.f, 25887.f, 50366.f, 28873.f, 57826.f, -6425.f, 21999.f, 56271.f, 30938.f, 1992.f,
    50332.f, 45526.f, 60748.f, 5147.f, 8723.f, -1447.f, 38624.f, -15050.f, 56015.f, 50225.f, -2359.f, -1834.f,
    5402.f, 62746.f, 14403.f, 34990.f, -12898.f, 59689.f, -15420.f, 25038.f, 52217.f, 46022.f, 16980.f, -3400.f,
    37631.f, 35055.f, 49731.f, 44233.f, 18970.f, -11146.f, 35900.f, 27804.f, 43530.f, 36423.f, 44457.f, 10531.f,
    -5659.f, 21296.f, 14068.f, 63861.f, 55001.f, 53493.f, 58948.f, 54903.f, 41573.f, 18334.f, 18972.f, -11686.f,
    31398.f, -10333.f, 26472.f, 61970.f, 51453.f, 21636.f, -14469.f, -6701.f, -2362.f, 23282.f, 26633.f, 42634.f,
    33993.f, 23151.f, -7815.f, 61490.f, 18023.f, 52519.f, -15784.f, 8720.f, 57115.f, -9813.f, 48361.f, -11385.f
};

static const float input_2_scale = 312.3333435059f;
static const float input_2_zero_point = 24038.5f;
static const int8_t input_2_scales_frac[] = {6};
static const int input_2_sa_dim = -1;

static const int input_2_fx8_frac = -9;

#define INPUT_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {6, 6, 3}, \
    /* .rank =  */ 3


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


static const float weights_1_data[] = {
    0.070644f, -0.021585f, 0.190618f, -0.195205f, 0.063108f, 0.129999f, 0.109678f, 0.021456f, -0.277439f, 0.043154f, 
    -0.057055f, -0.415196f, -0.036425f, 0.329322f, 0.282428f, 0.043816f, 0.078477f, -0.064667f, 0.179238f, 0.181760f, 
    -0.014273f, -0.072606f, 0.070096f, -0.108414f, -0.104719f, -0.062017f, -0.070193f, -0.220118f, 0.200723f, 0.474011f, 
    0.051944f, 0.151496f, -0.133954f, -0.070044f, 0.210833f, -0.248594f
};

static const float weights_1_scales[] = {0.0017332132f, 0.0025930898f, 0.0037323702f};
static const float weights_1_zero_points[] = {0.0f, 0.0f, 0.0f};
static const int8_t weights_1_scales_frac[] = {23, 23, 23};
static const int weights_1_sa_dim = 3;

static const int weights_1_fx8_frac = 8;

#define WEIGHTS_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 4, 1, 3}, \
    /* .rank =  */ 4

static const mli_tensor weights_1_tsr_fx16 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_1_tsr_fx8 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_1_tsr_sa8 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const float weights_2_data[] = {
    0.061785f, -0.247152f, -0.050968f, 0.193333f, -0.010555f, -0.457387f, 0.091005f, 0.029531f, -0.131128f, -0.124796f, 
    0.038729f, 0.318163f, 0.018775f, 0.000024f, 0.065362f, 0.106736f, -0.298480f, 0.330821f, 0.227728f, 0.175145f, 
    -0.454469f, -0.038041f, 0.131463f, 0.102645f, 0.082620f, -0.190452f, 0.063674f, -0.016948f, -0.240415f, 0.255931f, 
    -0.091566f, -0.081387f, -0.367663f, 0.105258f, 0.239669f, 0.527339f
};

static const float weights_2_scale = 0.004152277f;
static const float weights_2_zero_point = 0.0f;
static const int8_t weights_2_scale_frac[] = {22};
static const int weights_2_sa_dim = -1;

static const float weights_2_scales[] = {0.0017931332f, 0.0023502372f, 0.004152277f};
static const float weights_2_zero_points[] = {0.0f, 0.0f, 0.0f};
static const int8_t weights_2_scales_frac[] = {22, 22, 22};
static const int weights_2_sa_dim_per_axis = 3;

static const int weights_2_fx8_frac = 7;

#define WEIGHTS_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 3, 1, 3}, \
    /* .rank =  */ 4

#define WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(3 * (3 * 2))  * 2, 3 * 2, 3, 1}, \
    /* .shape = */ {4, 3, 1, 3}, \
    /* .rank =  */ 4

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


static const float weights_3_data[] = {
    -0.312838f, 2.917907f, 0.151440f, 0.449363f, -2.032399f, -2.220524f, 0.637657f, 0.990262f,
    0.889516f, -0.655975f, -0.266191f, -1.906472f, 0.477778f, 0.759943f, 1.282664f, 0.663656f,
    0.319935f, -0.407204f, 0.437790f, -0.171670f, -0.836023f, 0.114243f, -1.676214f, -2.453241f,
    0.621572f, -1.540178f, -0.571634f
};

static const float weights_3_scales[] = {0.0052256342f, 0.0229756478f, 0.0193168558f};
static const float weights_3_zero_points[] = {0.0f, 0.0f, 0.0f};
static const int8_t weights_3_scales_frac[] = {22, 20, 20};
static const int weights_3_sa_dim = 3;

static const int weights_3_fx8_frac = 5;

#define WEIGHTS_3_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {((3*1 + 1)*1 + 4)*3, (3*1 + 1)*1, (3*1), 1}, \
    /* .shape = */ {3, 3, 1, 3}, \
    /* .rank =  */ 4


static const mli_tensor weights_3_tsr_fx8 = {
    WEIGHTS_3_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_3_tsr_fx16 = {
    WEIGHTS_3_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_3_tsr_sa8 = {
    WEIGHTS_3_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const float weights_4_data[] = {
    -0.332096f, 0.149141f, 0.173252f, -0.171514f, -0.185494f, 2.397222f, -1.013184f, 0.565661f, -1.537474f, 
    -0.274718f, 1.756464f, 0.753059f, -1.120236f, -1.778918f, -1.533642f, 0.027517f, 1.638698f, -3.600690f, 
    -0.645360f, -2.566396f, 3.851239f, -0.645629f, 1.211123f, -1.826542f, -0.137540f, -1.068214f, -0.938217f, 
    -0.575475f, -0.007711f, 0.135225f, -0.592028f, 0.758051f, -0.825912f, 0.047567f, -0.440687f, 3.035970f, 
    -0.547786f, -0.143698f, -0.670205f, 0.268794f, -0.442165f, 3.059151f, -0.587010f, 1.499530f, 1.713755f, 
    -0.009335f, 0.122824f, 2.271225f, 1.057685f, 1.328502f, -3.107481f, 0.099054f, 0.085906f, -2.297469f, 
    0.663333f, 0.225171f, 1.451750f, -0.606511f, 0.187434f, 0.483175f, -0.612586f, -0.187096f, 1.310985f, 
    0.341193f, -0.101912f, 0.102854f, 0.080621f, -0.939357f, 0.141440f, 0.178186f, -0.857409f, -0.746881f, 
    0.418920f, 1.515816f, 2.419256f
};

static const float weights_4_scales[] = {0.0088207582f, 0.0202078428f, 0.0303247143f};
static const float weights_4_zero_points[] = {0.0f, 0.0f, 0.0f};
static const int8_t weights_4_scales_frac[] = {21, 20, 20};
static const int weights_4_sa_dim = 3;

static const int weights_4_fx8_frac = 5;

#define WEIGHTS_4_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {((3*1 + 1)*1 + 4)*5, (3*1 + 1)*1, (3*1), 1}, \
    /* .shape = */ {5, 5, 1, 3}, \
    /* .rank =  */ 4


static const mli_tensor weights_4_tsr_fx8 = {
    WEIGHTS_4_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_4_tsr_fx16 = {
    WEIGHTS_4_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_4_tsr_sa8 = {
    WEIGHTS_4_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const float weights_5_data[] = {
    32032.f, -21740.f, -46566.f, -75524.f, -8320.f, 100694.f, 91536.f, -40203.f, 119184.f,
    -25412.f, 17887.f, 106504.f, -76390.f, -55172.f, -105900.f, 104672.f, -5102.f, -52242.f,
    47684.f, -2342.f, -123480.f, 106014.f, -56850.f, -33076.f, 79262.f, -14923.f, -55410.f
};

static const float weights_5_scales[] = {834.7559204102f, 447.6377868652f, 972.2834472656f};
static const float weights_5_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t weights_5_scales_frac[] = {5, 6, 5};
static const int weights_5_sa_dim = 3;

static const int weights_5_fx8_frac = -10;

#define WEIGHTS_5_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 3, 1, 3}, \
    /* .rank =  */ 4


static const mli_tensor weights_5_tsr_fx8 = {
    WEIGHTS_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_5_tsr_fx16 = {
    WEIGHTS_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_5_tsr_sa8 = {
    WEIGHTS_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const float bias_1_data[] = {-0.161161f, 0.359392f, -0.036637f};

static const float bias_1_scales[] = {0.00002018f, 0.00003019f, 0.00004346f};
static const float bias_1_zero_points[] = {0.0f, 0.0f, 0.0f};
static const int8_t bias_1_scales_frac[] = {27, 27, 26};
static const int bias_1_sa_dim = 0;

static const int bias_1_fx8_frac = 8;

#define BIAS_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3}, \
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

static const mli_tensor bias_1_tsr_sa32 = {
    BIAS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};


static const float bias_2_data[] = {-0.161161f, 0.359392f, -0.036637f};

static const float bias_2_i1_w2_scale = 0.00004835f;
static const float bias_2_i1_w2_zero_point = 0.0f;
static const int8_t bias_2_i1_w2_scale_frac[] = {26};
static const int bias_2_i1_w2_sa_dim = -1;

static const float bias_2_i1_w2_scales[] = {0.00002088f, 0.00002737f, 0.00004835f};
static const float bias_2_i1_w2_zero_points[] = {0.0f, 0.0f, 0.0f};
static const int8_t bias_2_i1_w2_scales_frac[] = {28, 27, 26};
static const int bias_2_i1_w2_sa_dim_per_axis = 0;

static const float bias_2_i1_w3_scales[] = {0.0000608453f, 0.0002675198f, 0.0002249182f};
static const float bias_2_i1_w3_zero_points[] = {0.0f, 0.0f, 0.0f};
static const int8_t bias_2_i1_w3_scales_frac[] = {29, 26, 27};
static const int bias_2_i1_w3_sa_dim = 0;

static const float bias_2_i1_w4_scales[] = {0.0001027056f, 0.0002352925f, 0.0003530896f};
static const float bias_2_i1_w4_zero_points[] = {0.0f, 0.0f, 0.0f};
static const int8_t bias_2_i1_w4_scales_frac[] = {28, 27, 26};
static const int bias_2_i1_w4_sa_dim = 0;

static const int bias_2_fx8_frac = 8;

#define BIAS_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3}, \
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

static const mli_tensor bias_2_tsr_sa32 = {
    BIAS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float bias_3_data[] = {-539190912.f, -790461248.f, -699932160.f};

static const float bias_3_i2_w5_scales[] = {260722.109375f, 139812.203125f, 303676.53125f};
static const float bias_3_i2_w5_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t bias_3_i2_w5_scales_frac[] = {-3, -3, -4};
static const int bias_3_i2_w5_sa_dim = 0;

static const int bias_3_fx8_frac = -23;

#define BIAS_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3}, \
    /* .rank =  */ 1


static const mli_tensor bias_3_tsr_fx8 = {
    BIAS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor bias_3_tsr_fx16 = {
    BIAS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor bias_3_i2_w5_tsr_sa32 = {
    BIAS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_1_out_data[] = {
    -0.097726f, 0.343363f, -0.926802f, 0.113766f, 0.752121f, -0.492807f, -0.046339f, 0.504339f, 1.107545f, -0.150885f, 
    0.729244f, -0.234922f, -0.027222f, 0.753985f, 0.125550f, -0.568329f, 0.579819f, 0.386437f, -0.433139f, 1.012842f, 
    -0.052301f, -0.149513f, 0.307539f, -0.007198f, -0.237716f, 0.084325f, -0.990453f, -0.145694f, 0.852486f, -1.460101f, 
    -0.735259f, 0.704190f, 0.838192f, 0.283441f, 0.694336f, 0.472566f, -0.008191f, 0.479724f, 0.374354f, -0.188989f, 
    0.019189f, 0.072077f, -0.383516f, 1.080392f, -0.485482f, 0.013403f, 0.652063f, 0.355017f, -0.335874f, 0.521644f, 
    -0.024881f, -0.394823f, 0.988437f, 0.34160f, 0.169621f, 0.595592f, -0.864213f, -0.328260f, 1.100599f, -0.367612f, 
    -0.126902f, 0.571401f, 0.813532f, 0.056844f, 1.755135f, 0.149099f, -0.932518f, 1.697406f, 0.843652f, 0.067234f, 
    1.006558f, 0.141784f, -0.444260f, 0.444014f, -1.425449f, 0.282817f, 1.088960f, -0.386008f, -0.643830f, 0.613607f, 
    0.140695f, -0.119943f, 1.120076f, 0.563234f, -0.522121f, 0.719019f, -0.254816f, -0.040066f, 1.603856f, -0.153388f, 
    -0.504888f, 0.743092f, 0.739818f, -0.566575f, 1.402207f, 1.234275f, 0.086161f, 0.698818f, -0.354220f, -0.519111f, 
    1.252208f, -0.586787f, -0.553884f, 0.490592f, 0.075649f, -0.268424f, 0.600361f, -0.921062f, -1.100061f, 0.262831f, 
    0.042061f, -0.573621f, 0.576206f, 0.802844f, -0.152952f, -0.063976f, -0.833202f, -0.218865f, 1.160578f, -0.694619f, 
    -0.352917f, 0.257159f, 0.692647f, -0.334151f, 1.357913f, 0.316832f, -0.172682f, 0.663047f, -0.493310f, -0.087035f, 
    0.583610f, 0.055120f, 0.188188f, 0.758183f, -0.704189f, 0.019148f, 0.169735f, 0.146673f, -0.317734f, 0.048283f, 
    0.278271f, 0.067201f, 0.828022f, 0.395341f, -0.155313f, 0.104455f, 0.384951f
};

static const float test_1_out_scale = 0.0126087721f;
static const float test_1_out_zero_point = 0.1475169659f;
static const int8_t test_1_out_scales_frac[] = {21};
static const int test_1_out_sa_dim = -1;

static const int test_1_out_fx8_frac = 6;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {7, 7, 3}, \
    /* .rank =  */ 3

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
    0.203490f, 0.262528f, 0.0f, 0.0f, 0.029437f, 1.798222f, 0.0f, 0.385419f, 0.448258f, 0.215679f, 0.0f, 0.0f, 0.0f, 
    0.478291f, 1.172920f, 0.0f, 0.164021f, 0.0f, 0.261174f, 0.0f, 0.27280f, 0.0f, 0.627163f, 0.204590f, 0.298193f, 
    0.056678f, 0.0f, 0.0f, 0.0f, 0.0f, 0.027674f, 0.0f, 0.864396f, 0.0f, 0.628065f, 0.241818f, 0.374835f, 0.0f, 
    0.0f, 0.0f, 0.494496f, 0.0f, 0.040495f, 0.399270f, 0.091016f, 0.096882f, 0.0f, 0.531969f
};

static const float test_2_out_scale = 0.0070518525f;
static const float test_2_out_zero_point = 0.8991112113f;
static const int8_t test_2_out_scales_frac[] = {22};
static const int test_2_out_sa_dim = -1;

static const int test_2_out_fx8_frac = 6;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 4, 3}, \
    /* .rank =  */ 3

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
    -0.393397f, 1.123979f, 0.262144f, -0.578174f, 0.851972f, 0.044634f, 
    -0.838294f, 1.155926f, -0.103201f
};

static const float test_3_out_scale = 0.0078204684f;
static const float test_3_out_zero_point = 0.1588161588f;
static const int8_t test_3_out_scales_frac[] = {21};
static const int test_3_out_sa_dim = -1;

static const int test_3_out_fx8_frac = 6;

#define TEST_3_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 1, 3}, \
    /* .rank =  */ 3

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
    -0.237716f, 0.084325f, -0.990453f, 0.283441f, 0.694336f, 0.472566f, -0.040066f, 1.0f, -0.153388f, 0.086161f, 
    0.698818f, -0.354220f
};

static const float test_4_out_scale = 0.0078056967f;
static const float test_4_out_zero_point = 0.0047736764f;
static const int8_t test_4_out_scales_frac[] = {22};
static const int test_4_out_sa_dim = -1;

static const int test_4_out_fx8_frac = 7;

#define TEST_4_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(((3*1)+1)*2)+3, (3*1)+1, 1}, \
    /* .shape = */ {2, 2, 3}, \
    /* .rank =  */ 3

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

static const float test_5_out_data[] = {
    0.740725f, 1.575873f, 0.0f, 2.896098f, 0.0f, 4.030248f, 0.0f, 0.556138f, 0.0f, 1.434916f, 0.0f, 0.0f, 
    0.240326f, 0.0f, 0.0f, 2.691257f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.045688f, 3.382314f, 0.561977f, 
    0.257052f, 0.569224f, 1.370535f, 0.029328f, 1.075759f, 0.205622f, 0.0f, 0.0f, 1.049827f, 0.0f, 0.0f, 0.0f, 
    2.800543f, 0.0f, 0.0f, 0.0f, 0.0f, 0.402417f, 0.0f, 0.0f, 0.281607f, 5.437880f, 0.0f, 0.0f, 3.588021f, 
    0.0f, 0.0f, 1.173560f, 0.0f, 1.535824f, 1.032547f, 0.0f, 1.587831f, 0.0f, 1.671238f, 0.936969f, 2.428011f, 
    0.0f, 0.0f, 1.435224f, 0.0f, 1.477373f, 6.0f, 0.0f, 2.415473f, 0.185508f, 0.0f, 2.000233f, 3.630160f, 0.0f
};

static const float test_5_out_scale = 0.0235294122f;
static const float test_5_out_zero_point = 3.0f;
static const int8_t test_5_out_scales_frac[] = {20};
static const int test_5_out_sa_dim = -1;

static const int test_5_out_fx8_frac = 4;

#define TEST_5_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {5, 5, 3}, \
    /* .rank =  */ 3


static const mli_tensor test_5_out_tsr_fx16 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_5_out_tsr_sa8 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const float test_6_out_data[] = {
    0.957095f, 0.626361f, 1.431675f, 0.564856f, 0.828118f, 0.0f, 0.003515f, 2.140245f, 1.152858f, 0.513510f, 
    2.410225f, 0.654422f, 0.113393f, 0.0f, 0.629954f, 0.0f, 0.0f, 0.0f, 0.561977f, 0.257052f, 0.569224f, 0.0f, 
    0.146223f, 0.0f, 2.154483f, 0.318256f, 0.0f, 0.0f, 3.588021f, 0.0f, 1.535824f, 1.032547f, 0.0f, 0.0f, 
    6.0f, 0.0f, 2.668583f, 3.062790f, 0.0f, 0.0f, 3.505633f, 0.554232f, 0.528308f, 6.0f, 0.0f, 0.700901f, 
    4.886393f, 0.0f
};

static const float test_6_out_scale = 0.0235294122f;
static const float test_6_out_zero_point = 3.0f;
static const int8_t test_6_out_scales_frac[] = {20};
static const int test_6_out_sa_dim = -1;

static const int test_6_out_fx8_frac = 4;

#define TEST_6_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(((3*1)+1)*4)+3, (3*1)+1, 1}, \
    /* .shape = */ {4, 4, 3}, \
    /* .rank =  */ 3


static const mli_tensor test_6_out_tsr_fx16 = {
    TEST_6_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_6_out_tsr_sa8 = {
    TEST_6_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float test_7_out_data[] = {
    0.0f, 1.798977f, 6.0f, 0.0f, 3.213652f, 6.0f, 0.0f, 0.0f, 3.246345f, 2.113257f, 0.0f, 0.0f, 0.0f, 0.0f, 
    1.220649f, 0.0f, 1.271335f, 3.669997f, 0.0f, 2.457561f, 0.173070f, 0.0f, 2.801335f, 0.0f, 0.683534f, 
    0.482768f, 4.611751f, 1.082141f, 0.0f, 6.0f, 2.898520f, 6.0f, 3.812624f, 1.808096f, 0.0f, 3.672356f, 0.0f, 
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.518527f
};

static const float test_7_out_scale = 0.0235294122f;
static const float test_7_out_zero_point = 3.0f;
static const int8_t test_7_out_scales_frac[] = {20};
static const int test_7_out_sa_dim = -1;

static const int test_7_out_fx8_frac = 4;

#define TEST_7_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(((3*1)+1)*4)+3, (3*1)+1, 1}, \
    /* .shape = */ {4, 4, 3}, \
    /* .rank =  */ 3


static const mli_tensor test_7_out_tsr_fx16 = {
    TEST_7_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_7_out_tsr_sa8 = {
    TEST_7_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float test_8_out_data[] = {
    1.0f, 1.0f, -1.0f, 1.0f, 0.851317f, 1.0f, -1.0f, -0.799316f, 0.170510f, 1.0f, -0.083452f, -1.0f, -1.0f, 
    -1.0f, -1.0f, -0.966495f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -0.305496f, -1.0f, -1.0f, 
    0.582066f, -0.835701f, 1.0f, 0.860890f, -1.0f, -1.0f, -0.350228f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 
    -0.189379f, 1.0f, -1.0f, 0.858548f, -1.0f, 0.930343f, 1.0f, -0.707815f, -1.0f, 0.360904f, -1.0f, -1.0f, 
    1.0f, 0.307821f, 0.074646f, -1.0f, 1.0f, -1.0f, -1.0f, 0.133816f, -0.985730f, -0.111945f, -1.0f, -1.0f, 
    1.0f, 1.0f, 1.0f, -0.205824f, -1.0f, 1.0f, 0.058403f, 0.835156f, -1.0f, 1.0f, -1.0f, -1.0f, -0.718767f, 
    0.092004f, -1.0f, 0.503506f, 0.116331f, -1.0f, 1.0f, -1.0f, 1.0f, 0.501670f, 1.0f, -1.0f, 1.0f, -1.0f, 
    1.0f, 0.130482f, 1.0f, -1.0f, 0.155762f, 1.0f, -1.0f, 0.532347f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 
    0.452635f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 0.493259f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 0.742988f, 1.0f, 
    -0.759602f, 0.612960f, 1.0f, 1.0f, -0.125504f, -1.0f, 0.630144f, -0.195336f, 1.0f, -1.0f, 1.0f, 
    -0.233183f, 1.0f, 1.0f, -1.0f, -1.0f, -0.684735f, -0.033894f, -1.0f, -0.483035f, 1.0f, 1.0f, 0.107179f, 
    1.0f, -1.0f, 0.070196f, -0.998963f, -1.0f, 0.081494f, -1.0f, -1.0f
};

static const float test_8_out_scale = 0.0078431377f;
static const float test_8_out_zero_point = 0.0f;
static const int8_t test_8_out_scales_frac[] = {21};
static const int test_8_out_sa_dim = -1;

static const int test_8_out_fx8_frac = 7;

#define TEST_8_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(((3*1)+1)*7)+3, (3*1)+1, 1}, \
    /* .shape = */ {7, 7, 3}, \
    /* .rank =  */ 3


static const mli_tensor test_8_out_tsr_fx16 = {
    TEST_8_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_8_out_tsr_sa8 = {
    TEST_8_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float test_9_out_data[] = {
    0.0f, 0.0f, 1.433628f, 0.587324f, 1.874803f, 5.592649f, 0.0f, 3.903254f, 7.072687f, 1.97420f, 0.177926f, 
    2.141395f, 0.0f, 0.0f, 5.888912f, 0.0f, 0.0f, 0.0f, 0.0f, 0.941083f, 0.032147f, 1.758663f, 0.281915f, 
    5.879355f, 0.062643f, 1.193907f, 0.0f, 1.565769f, 5.139726f, 0.0f, 0.179767f, 0.0f, 0.0f, 0.0f, 1.439335f, 
    14.907073f, 0.0f, 0.0f, 1.867380f, 0.0f, 0.0f, 0.348317f, 0.041212f, 2.763275f, 1.282343f, 0.0f, 0.0f, 
    0.0f, 0.259735f, 4.376644f, 2.297802f, 1.043119f, 1.919704f, 4.187203f, 0.0f, 0.0f, 0.0f, 0.0f, 1.214675f, 
    4.179674f, 0.0f, 3.131628f, 7.949909f, 0.0f, 0.0f, 0.0f, 0.0f, 3.677073f, 1.758717f, 0.241961f, 0.0f, 
    0.0f, 1.173234f, 0.0f, 0.0f, 0.0f, 0.0f, 10.01250f, 0.0f, 2.444105f, 0.0f, 0.382074f, 0.0f, 0.0f, 0.0f, 
    0.643762f, 0.740929f, 0.0f, 0.0f, 0.0f, 0.752109f, 0.0f, 3.868686f, 0.0f, 1.882916f, 8.367089f, 1.150580f, 
    4.344887f, 0.0f, 0.0f, 5.602156f, 0.0f, 1.343273f, 0.0f, 9.790434f, 0.0f, 0.648245f, 3.168720f, 0.0f, 
    6.309370f, 0.0f, 0.0f, 0.470472f, 4.566372f, 0.0f, 0.0f, 4.556415f, 0.0f, 0.0f, 1.024385f, 0.0f, 
    3.366853f, 0.0f, 0.0f, 0.571266f, 0.0f, 0.570028f, 0.0f, 0.0f, 0.0f, 0.0f, 1.557938f, 0.0f, 2.060237f, 
    10.258909f, 0.0f, 0.0f, 9.406365f, 0.591361f, 0.0f, 2.197687f, 0.0f, 5.757831f, 0.0f, 0.0f, 5.199497f, 0.0f
};

static const float test_9_out_scale = 0.0584591106f;
static const float test_9_out_zero_point = 7.4535365105f;
static const int8_t test_9_out_scales_frac[] = {19};
static const int test_9_out_sa_dim = -1;

static const int test_9_out_fx8_frac = 3;

#define TEST_9_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(((3*1)+1)*7)+3, (3*1)+1, 1}, \
    /* .shape = */ {7, 7, 3}, \
    /* .rank =  */ 3


static const mli_tensor test_9_out_tsr_fx16 = {
    TEST_9_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_9_out_tsr_sa8 = {
    TEST_9_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float test_10_out_data[] = {
    8325484032.f, 0.f, 0.f, 11192914944.f, 1458333184.f, 1028864640.f, 12933103616.f, 0.f, 0.f,
    6624958976.f, 0.f, 0.f, 10769914880.f, 0.f, 0.f, 13671384064.f, 0.f, 0.f, 7699801600.f, 0.f,
    1284153856.f, 11849158656.f, 0.f, 0.f, 16207603712.f, 0.f, 514985472.f, 12639497216.f, 0.f, 0.f,
    4064989696.f, 0.f, 0.f, 12925231104.f, 0.f, 971207424.f, 4844002816.f, 0.f, 0.f, 7522769920.f, 0.f,
    0.f, 7888317440.f, 0.f, 1395462144.f, 5603478016.f, 0.f, 0.f
};

static const float test_10_out_scale = 63559232.f;
static const float test_10_out_zero_point = 8103801856.f;
static const int8_t test_10_out_scales_frac[] = {-11};
static const int test_10_out_sa_dim = -1;

static const int test_10_out_fx8_frac = -27;

#define TEST_10_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 4, 3}, \
    /* .rank =  */ 3


static const mli_tensor test_10_out_tsr_fx16 = {
    TEST_10_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_10_out_tsr_sa8 = {
    TEST_10_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

// Quantizers
//===================================================
tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]));

tensor_quantizer input_1_memstr_fx16(input_1_memstr_tsr_fx16, input_1_fx8_frac + 8, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]));

tensor_quantizer input_1b_memstr_fx16(input_1b_memstr_tsr_fx16, input_1_fx8_frac + 8, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]));

tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, 
    &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_1_memstr_sa8(input_1_memstr_tsr_sa8, input_1_sa_dim, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, 
    &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_1b_memstr_sa8(input_1b_memstr_tsr_sa8, input_1_sa_dim, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, 
    &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx8_frac + 8, input_2_data,
    sizeof(input_2_data) / sizeof(input_2_data[0]));

tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data, 
    sizeof(input_2_data) / sizeof(input_2_data[0]),
    &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);


tensor_quantizer weights_1_fx16(weights_1_tsr_fx16, weights_1_fx8_frac + 8, weights_1_data, 
    sizeof(weights_1_data) / sizeof(weights_1_data[0]));

tensor_quantizer weights_1_fx8(weights_1_tsr_fx8, weights_1_fx8_frac, weights_1_data, 
    sizeof(weights_1_data) / sizeof(weights_1_data[0]));

tensor_quantizer weights_1_sa8_per_axis(weights_1_tsr_sa8, weights_1_sa_dim, weights_1_data, 
    sizeof(weights_1_data) / sizeof(weights_1_data[0]), weights_1_scales, 
    sizeof(weights_1_scales) / sizeof(weights_1_scales[0]), weights_1_zero_points, 
    sizeof(weights_1_zero_points) / sizeof(weights_1_zero_points[0]), weights_1_scales_frac, 
    sizeof(weights_1_scales_frac) / sizeof(weights_1_scales_frac[0]));

tensor_quantizer weights_2_fx16(weights_2_tsr_fx16, weights_2_fx8_frac + 8, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));

tensor_quantizer weights_2_memstr_fx16(weights_2_memstr_tsr_fx16, weights_2_fx8_frac + 8, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));

tensor_quantizer weights_2_fx8(weights_2_tsr_fx8, weights_2_fx8_frac, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));

tensor_quantizer weights_2_memstr_fx8(weights_2_memstr_tsr_fx8, weights_2_fx8_frac, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));

tensor_quantizer weights_2_sa8(weights_2_tsr_sa8, weights_2_sa_dim, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]), &weights_2_scale, 1, 
    &weights_2_zero_point, 1, weights_2_scale_frac, 1);

tensor_quantizer weights_2_memstr_sa8_per_axis(weights_2_memstr_tsr_sa8, weights_2_sa_dim_per_axis, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]), weights_2_scales, 
    sizeof(weights_2_scales) / sizeof(weights_2_scales[0]), weights_2_zero_points, 
    sizeof(weights_2_zero_points) / sizeof(weights_2_zero_points[0]), weights_2_scales_frac, 
    sizeof(weights_2_scales_frac) / sizeof(weights_2_scales_frac[0]));

tensor_quantizer weights_3_fx16(weights_3_tsr_fx16, weights_3_fx8_frac + 8, weights_3_data, 
    sizeof(weights_3_data) / sizeof(weights_3_data[0]));

tensor_quantizer weights_3_fx8(weights_3_tsr_fx8, weights_3_fx8_frac, weights_3_data, 
    sizeof(weights_3_data) / sizeof(weights_3_data[0]));

tensor_quantizer weights_3_sa8_per_axis(weights_3_tsr_sa8, weights_3_sa_dim, weights_3_data, 
    sizeof(weights_3_data) / sizeof(weights_3_data[0]), weights_3_scales, 
    sizeof(weights_3_scales) / sizeof(weights_3_scales[0]), weights_3_zero_points, 
    sizeof(weights_3_zero_points) / sizeof(weights_3_zero_points[0]), weights_3_scales_frac, 
    sizeof(weights_3_scales_frac) / sizeof(weights_3_scales_frac[0]));

tensor_quantizer weights_4_fx16(weights_4_tsr_fx16, weights_4_fx8_frac + 8, weights_4_data, 
    sizeof(weights_4_data) / sizeof(weights_4_data[0]));

tensor_quantizer weights_4_fx8(weights_4_tsr_fx8, weights_4_fx8_frac, weights_4_data, 
    sizeof(weights_4_data) / sizeof(weights_4_data[0]));

tensor_quantizer weights_4_sa8_per_axis(weights_4_tsr_sa8, weights_4_sa_dim, weights_4_data, 
    sizeof(weights_4_data) / sizeof(weights_4_data[0]), weights_4_scales, 
    sizeof(weights_4_scales) / sizeof(weights_4_scales[0]), weights_4_zero_points, 
    sizeof(weights_4_zero_points) / sizeof(weights_4_zero_points[0]), weights_4_scales_frac, 
    sizeof(weights_4_scales_frac) / sizeof(weights_4_scales_frac[0]));

tensor_quantizer weights_5_fx8(weights_5_tsr_fx8, weights_5_fx8_frac, weights_5_data,
    sizeof(weights_5_data) / sizeof(weights_5_data[0]));

tensor_quantizer weights_5_fx16(weights_5_tsr_fx16, weights_5_fx8_frac + 8, weights_5_data,
    sizeof(weights_5_data) / sizeof(weights_5_data[0]));

tensor_quantizer weights_5_sa8(weights_5_tsr_sa8, weights_5_sa_dim, weights_5_data, 
    sizeof(weights_5_data) / sizeof(weights_5_data[0]), 
    weights_5_scales, sizeof(weights_5_scales) / sizeof(weights_5_scales[0]), 
    weights_5_zero_points, sizeof(weights_5_zero_points) / sizeof(weights_5_zero_points[0]), 
    weights_5_scales_frac, sizeof(weights_5_scales_frac) / sizeof(weights_5_scales_frac[0]));

tensor_quantizer bias_1_fx16(bias_1_tsr_fx16, bias_1_fx8_frac + 8, bias_1_data, 
    sizeof(bias_1_data) / sizeof(bias_1_data[0]));

tensor_quantizer bias_1_fx8(bias_1_tsr_fx8, bias_1_fx8_frac, bias_1_data, 
    sizeof(bias_1_data) / sizeof(bias_1_data[0]));

tensor_quantizer bias_1_sa32_per_axis(bias_1_tsr_sa32, bias_1_sa_dim, bias_1_data, 
    sizeof(bias_1_data) / sizeof(bias_1_data[0]), bias_1_scales, 
    sizeof(bias_1_scales) / sizeof(bias_1_scales[0]), bias_1_zero_points, 
    sizeof(bias_1_zero_points) / sizeof(bias_1_zero_points[0]), bias_1_scales_frac, 
    sizeof(bias_1_scales_frac) / sizeof(bias_1_scales_frac[0]));


tensor_quantizer bias_2_fx16(bias_2_tsr_fx16, bias_2_fx8_frac + 8, bias_2_data, 
    sizeof(bias_2_data) / sizeof(bias_2_data[0]));

tensor_quantizer bias_2_fx8(bias_2_tsr_fx8, bias_2_fx8_frac, bias_2_data, 
    sizeof(bias_2_data) / sizeof(bias_2_data[0]));

tensor_quantizer bias_2_i1_w2_sa32(bias_2_tsr_sa32, bias_2_i1_w2_sa_dim, bias_2_data, 
    sizeof(bias_2_data) / sizeof(bias_2_data[0]), &bias_2_i1_w2_scale, 1, 
    &bias_2_i1_w2_zero_point, 1, bias_2_i1_w2_scale_frac, 1);

tensor_quantizer bias_2_i1_w2_sa32_per_axis(bias_2_tsr_sa32, bias_2_i1_w2_sa_dim_per_axis, bias_2_data, 
    sizeof(bias_2_data) / sizeof(bias_2_data[0]), bias_2_i1_w2_scales, 
    sizeof(bias_2_i1_w2_scales) / sizeof(bias_2_i1_w2_scales[0]), bias_2_i1_w2_zero_points, 
    sizeof(bias_2_i1_w2_zero_points) / sizeof(bias_2_i1_w2_zero_points[0]), bias_2_i1_w2_scales_frac, 
    sizeof(bias_2_i1_w2_scales_frac) / sizeof(bias_2_i1_w2_scales_frac[0]));

tensor_quantizer bias_2_i1_w3_sa32_per_axis(bias_2_tsr_sa32, bias_2_i1_w3_sa_dim, bias_2_data, 
    sizeof(bias_2_data) / sizeof(bias_2_data[0]), bias_2_i1_w3_scales, 
    sizeof(bias_2_i1_w3_scales) / sizeof(bias_2_i1_w3_scales[0]), bias_2_i1_w3_zero_points, 
    sizeof(bias_2_i1_w3_zero_points) / sizeof(bias_2_i1_w3_zero_points[0]), bias_2_i1_w3_scales_frac, 
    sizeof(bias_2_i1_w3_scales_frac) / sizeof(bias_2_i1_w3_scales_frac[0]));

tensor_quantizer bias_2_i1_w4_sa32_per_axis(bias_2_tsr_sa32, bias_2_i1_w4_sa_dim, bias_2_data, 
    sizeof(bias_2_data) / sizeof(bias_2_data[0]), bias_2_i1_w4_scales, 
    sizeof(bias_2_i1_w4_scales) / sizeof(bias_2_i1_w4_scales[0]), bias_2_i1_w4_zero_points, 
    sizeof(bias_2_i1_w4_zero_points) / sizeof(bias_2_i1_w4_zero_points[0]), bias_2_i1_w4_scales_frac, 
    sizeof(bias_2_i1_w4_scales_frac) / sizeof(bias_2_i1_w4_scales_frac[0]));

tensor_quantizer bias_3_fx8(bias_3_tsr_fx8, bias_3_fx8_frac, bias_3_data,
    sizeof(bias_3_data) / sizeof(bias_3_data[0]));

tensor_quantizer bias_3_fx16(bias_3_tsr_fx16, bias_3_fx8_frac + 8, bias_3_data,
    sizeof(bias_3_data) / sizeof(bias_3_data[0]));

tensor_quantizer bias_3_i2_w5_sa32(bias_3_i2_w5_tsr_sa32, bias_3_i2_w5_sa_dim, bias_3_data, 
    sizeof(bias_3_data) / sizeof(bias_3_data[0]),
    bias_3_i2_w5_scales, sizeof(bias_3_i2_w5_scales) / sizeof(bias_3_i2_w5_scales[0]), 
    bias_3_i2_w5_zero_points, sizeof(bias_3_i2_w5_zero_points) / sizeof(bias_3_i2_w5_zero_points[0]), 
    bias_3_i2_w5_scales_frac, sizeof(bias_3_i2_w5_scales_frac) / sizeof(bias_3_i2_w5_scales_frac[0]));

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data, 
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
    
tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim, test_1_out_data, 
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_1_out_scale, 1, 
    &test_1_out_zero_point, 1, test_1_out_scales_frac, 1);


tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_fx8_frac + 8, test_2_out_data, 
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));

tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_sa_dim, test_2_out_data, 
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &test_2_out_scale, 1, 
    &test_2_out_zero_point, 1, test_2_out_scales_frac, 1);


tensor_quantizer test_3_out_fx16(test_3_out_tsr_fx16, test_3_out_fx8_frac + 8, test_3_out_data, 
    sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));

tensor_quantizer test_3_out_sa8(test_3_out_tsr_sa8, test_3_out_sa_dim, test_3_out_data, 
    sizeof(test_3_out_data) / sizeof(test_3_out_data[0]), &test_3_out_scale, 1, 
    &test_3_out_zero_point, 1, test_3_out_scales_frac, 1);


tensor_quantizer test_4_out_fx16(test_4_out_tsr_fx16, test_4_out_fx8_frac + 8, test_4_out_data, 
    sizeof(test_4_out_data) / sizeof(test_4_out_data[0]));

tensor_quantizer test_4_out_sa8(test_4_out_tsr_sa8, test_4_out_sa_dim, test_4_out_data, 
    sizeof(test_4_out_data) / sizeof(test_4_out_data[0]), &test_4_out_scale, 1, 
    &test_4_out_zero_point, 1, test_4_out_scales_frac, 1);


tensor_quantizer test_5_out_fx16(test_5_out_tsr_fx16, test_5_out_fx8_frac + 8, test_5_out_data, 
    sizeof(test_5_out_data) / sizeof(test_5_out_data[0]));

tensor_quantizer test_5_out_sa8(test_5_out_tsr_sa8, test_5_out_sa_dim, test_5_out_data, 
    sizeof(test_5_out_data) / sizeof(test_5_out_data[0]), &test_5_out_scale, 1, 
    &test_5_out_zero_point, 1, test_5_out_scales_frac, 1);

tensor_quantizer test_6_out_fx16(test_6_out_tsr_fx16, test_6_out_fx8_frac + 8, test_6_out_data, 
    sizeof(test_6_out_data) / sizeof(test_6_out_data[0]));

tensor_quantizer test_6_out_sa8(test_6_out_tsr_sa8, test_6_out_sa_dim, test_6_out_data, 
    sizeof(test_6_out_data) / sizeof(test_6_out_data[0]), &test_6_out_scale, 1,
    &test_6_out_zero_point, 1, test_6_out_scales_frac, 1);

tensor_quantizer test_7_out_fx16(test_7_out_tsr_fx16, test_7_out_fx8_frac + 8, test_7_out_data,
    sizeof(test_7_out_data) / sizeof(test_7_out_data[0]));

tensor_quantizer test_7_out_sa8(test_7_out_tsr_sa8, test_7_out_sa_dim, test_7_out_data, 
    sizeof(test_7_out_data) / sizeof(test_7_out_data[0]), &test_7_out_scale, 1, 
    &test_7_out_zero_point, 1, test_7_out_scales_frac, 1);

tensor_quantizer test_8_out_fx16(test_8_out_tsr_fx16, test_8_out_fx8_frac + 8, test_8_out_data,
    sizeof(test_8_out_data) / sizeof(test_8_out_data[0]));

tensor_quantizer test_8_out_sa8(test_8_out_tsr_sa8, test_8_out_sa_dim, test_8_out_data, 
    sizeof(test_8_out_data) / sizeof(test_8_out_data[0]), &test_8_out_scale, 1, 
    &test_8_out_zero_point, 1, test_8_out_scales_frac, 1);

tensor_quantizer test_9_out_fx16(test_9_out_tsr_fx16, test_9_out_fx8_frac + 8, test_9_out_data,
    sizeof(test_9_out_data) / sizeof(test_9_out_data[0]));

tensor_quantizer test_9_out_sa8(test_9_out_tsr_sa8, test_9_out_sa_dim, test_9_out_data, 
    sizeof(test_9_out_data) / sizeof(test_9_out_data[0]), &test_9_out_scale, 1, 
    &test_9_out_zero_point, 1, test_9_out_scales_frac, 1);

tensor_quantizer test_10_out_fx16(test_10_out_tsr_fx16, test_10_out_fx8_frac + 8, test_10_out_data,
    sizeof(test_10_out_data) / sizeof(test_10_out_data[0]));
    tensor_quantizer test_10_out_sa8(test_10_out_tsr_sa8, test_10_out_sa_dim, test_10_out_data, 
    sizeof(test_10_out_data) / sizeof(test_10_out_data[0]),
    &test_10_out_scale, 1, &test_10_out_zero_point, 1, test_10_out_scales_frac, 1);


// Tests configuration structures
//========================================
const mli_conv2d_cfg test_1_cfg = {
    /* .relu.type = */ MLI_RELU_NONE,
    /* .stride_width = */ 1,
    /* .stride_height = */ 1,
    /* .padding_left = */ 1,
    /* .padding_right = */ 2,
    /* .padding_top = */ 1,
    /* .padding_bottom = */ 1,
    /* .dilation_width = */ 1,
    /* .dilation_height = */ 1
};

const mli_conv2d_cfg test_2_cfg = {
    /* .relu.type = */ MLI_RELU_GEN,
    /* .stride_width = */ 2,
    /* .stride_height = */ 2,
    /* .padding_left = */ 1,
    /* .padding_right = */ 1,
    /* .padding_top = */ 1,
    /* .padding_bottom = */ 2,
    /* .dilation_width = */ 1,
    /* .dilation_height = */ 1
};

const mli_conv2d_cfg test_3_cfg = {
    /* .relu.type = */ MLI_RELU_NONE,
    /* .stride_width = */ 1,
    /* .stride_height = */ 1,
    /* .padding_left = */ 0,
    /* .padding_right = */ 0,
    /* .padding_top = */ 0,
    /* .padding_bottom = */ 0,
    /* .dilation_width = */ 2,
    /* .dilation_height = */ 2
};

const mli_conv2d_cfg test_4_cfg = {
    /* .relu.type = */ MLI_RELU_1,
    /* .stride_width = */ 3,
    /* .stride_height = */ 3,
    /* .padding_left = */ 0,
    /* .padding_right = */ 0,
    /* .padding_top = */ 0,
    /* .padding_bottom = */ 0,
    /* .dilation_width = */ 1,
    /* .dilation_height = */ 1
};

const mli_conv2d_cfg test_5_cfg = {
    /* .relu.type = */ MLI_RELU_6,
    /* .stride_width = */ 1,
    /* .stride_height = */ 1,
    /* .padding_left = */ 0,
    /* .padding_right = */ 0,
    /* .padding_top = */ 0,
    /* .padding_bottom = */ 0,
    /* .dilation_width = */ 1,
    /* .dilation_height = */ 1
};

const mli_conv2d_cfg test_6_cfg = {
    /* .relu.type = */MLI_RELU_6,
    /* .stride_height = */2,
    /* .stride_width = */2,
    /* .padding_left = */1,
    /* .padding_right = */1,
    /* .padding_top = */1,
    /* .padding_bottom = */1,
    /* .dilation_height = */1,
    /* .dilation_width = */1
};

const mli_conv2d_cfg test_7_cfg = {
    /* .relu.type = */MLI_RELU_6,
    /* .stride_height = */2,
    /* .stride_width = */2,
    /* .padding_left = */2,
    /* .padding_right = */2,
    /* .padding_top = */2,
    /* .padding_bottom = */2,
    /* .dilation_height = */1,
    /* .dilation_width = */1
};

const mli_conv2d_cfg test_8_cfg = {
    /* .relu.type = */MLI_RELU_1,
    /* .stride_height = */1,
    /* .stride_width = */1,
    /* .padding_left = */2,
    /* .padding_right = */2,
    /* .padding_top = */2,
    /* .padding_bottom = */2,
    /* .dilation_height = */2,
    /* .dilation_width = */2
};

const mli_conv2d_cfg test_9_cfg = {
    /* .relu.type = */MLI_RELU_GEN,
    /* .stride_height = */1,
    /* .stride_width = */1,
    /* .padding_left = */4,
    /* .padding_right = */4,
    /* .padding_top = */4,
    /* .padding_bottom = */4,
    /* .dilation_height = */2,
    /* .dilation_width = */2
};

const mli_conv2d_cfg test_10_cfg = {
    /* .relu.type = */MLI_RELU_GEN,
    /* .stride_width = */1,
    /* .stride_height = */1,
    /* .padding_left = */0,
    /* .padding_right = */0,
    /* .padding_top = */0,
    /* .padding_bottom = */0,
    /* .dilation_height = */1,
    /* .dilation_width = */1
};
