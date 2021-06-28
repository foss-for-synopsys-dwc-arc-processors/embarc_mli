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
extern mli::tst::tensor_quantizer input_1_memstr_fx16;
extern mli::tst::tensor_quantizer input_1_sa8;
extern mli::tst::tensor_quantizer input_1_memstr_sa8;
extern mli::tst::tensor_quantizer input_2_memstr_fx16;
extern mli::tst::tensor_quantizer input_2_memstr_sa8;

// Generated weights vectors Declaration
//========================================

extern mli::tst::tensor_quantizer weights_1_fx8;
extern mli::tst::tensor_quantizer weights_1_fx16;
extern mli::tst::tensor_quantizer weights_1_sa8;

extern mli::tst::tensor_quantizer weights_2_memstr_fx8;
extern mli::tst::tensor_quantizer weights_2_memstr_fx16;
extern mli::tst::tensor_quantizer weights_2_memstr_sa8;

extern mli::tst::tensor_quantizer weights_3_memstr_fx8;
extern mli::tst::tensor_quantizer weights_3_memstr_fx16;
extern mli::tst::tensor_quantizer weights_3_memstr_sa8;

extern mli::tst::tensor_quantizer weights_4_memstr_fx8;
extern mli::tst::tensor_quantizer weights_4_memstr_fx16;
extern mli::tst::tensor_quantizer weights_4_memstr_sa8;

extern mli::tst::tensor_quantizer weights_5_memstr_fx8;
extern mli::tst::tensor_quantizer weights_5_memstr_fx16;
extern mli::tst::tensor_quantizer weights_5_memstr_sa8;

extern mli::tst::tensor_quantizer bias_1_fx8;
extern mli::tst::tensor_quantizer bias_1_fx16;
extern mli::tst::tensor_quantizer bias_1_i1_w1_sa32;
extern mli::tst::tensor_quantizer bias_1_i1_w2_sa32;
extern mli::tst::tensor_quantizer bias_1_i1_w5_sa32;
extern mli::tst::tensor_quantizer bias_2_fx8;
extern mli::tst::tensor_quantizer bias_2_fx16;
extern mli::tst::tensor_quantizer bias_2_i2_w3_sa32;
extern mli::tst::tensor_quantizer bias_2_i2_w4_sa32;

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


// Generated input vectors 
//========================================
static const float input_1_data[] = {
    0.760813f, 1.402546f, 1.659607f, 1.925337f, 1.900578f, 0.473811f, 1.476568f, 0.071224f, 0.236099f, 0.229136,
     -0.130896f, 1.164252f, -0.505863f, 0.308291f, 1.090899f, 1.09429f, 0.840339f, 1.858756f, -0.723445f, 
    0.75131f, -0.10565f, 1.115949f, 1.564747f, 0.811586f, -0.20983f, 1.276726f, -0.99979f, -0.826718f, -0.90529f,
     1.821834f, 0.680334f, -0.111956f, 0.025245f, 0.370005f, 0.354926f, 0.833301f, 1.079554f, -0.093351f, 
    0.73275f, -0.772586f, 0.300372f, -0.325865f, 1.756499f, 1.706079f, 1.309113f, 0.088898f, 1.375918f, 0.495281f
    , 0.778721f, -0.82619f, 0.324342f, 1.729644f, -0.486106f, 1.936264f, -0.675234f, -0.116795f, 0.510489f, 
    1.258417f, -0.754757f, 0.288059f, 1.217644f, 0.396291f, 1.99292f, -0.774432f, -0.845276f, 0.622998f, -
    0.590172f, 1.756666f, -0.900931f, 1.049657f, 1.553634f, 0.357499f, 0.357556f, 0.30842f, 0.64465f, 0.283418f, 
    0.083297f, 1.189218f, 0.719238f, 1.221781f, -0.796109f, -0.145757f, -0.977839f, -0.929623f, 0.577857f, 
    1.278931f, 1.030622f, 0.690176f, 0.099787f, 0.052687f, 1.471032f, 1.837287f, 0.328458f, -0.908108f, -
    0.961513f, 1.671578f, 1.852382f, 1.360534f, 1.218412f, 1.512003f, 1.671298f, 1.922946f, 0.401419f, 1.664331f,
     -0.097103f, 0.958267f, 1.990651f, -0.008188f, -0.921605f, 1.003292f, 0.636269f, 0.034368f
};

static const float input_1_scale = 0.011736121f;
static const float input_1_zero_point = 0.4965649843f;
static const int8_t input_1_scales_frac[] = {21};
static const int input_1_sa_dim = -1;

static const int input_1_fx8_frac = 6;

#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 4, 7}, \
    /* .rank =  */ 3


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
    /* .mem_stride = */ {7*12*1, 12*1, 1}, \
    /* .shape = */ {4, 4, 7}, \
    /* .rank =  */ 3


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
    -0.647099f, 2.476065f, 2.78113f, 0.098039f, -0.252552f, -0.310425f, 2.586953f, 0.472886f, -1.119638f, 0.106922f, 
    -0.341404f, 2.035476f, 2.752784f, 1.600119f, -1.552247f, 1.962506f, 0.528303f, 0.197099f, 0.31535f, -0.460923f, 
    2.06248f, -0.293721f, -0.720762f, 0.637147f, -1.732849f, 1.167793f, 1.678034f, -0.892122f, 0.216196f, 1.509061f, 
    -1.303623f, -0.704221f, 1.420887f, 0.82633f, 0.431202f, -1.28906f, -0.791522f, 0.945917f, -0.000907f, 2.866447f, 
    -0.279242f, 2.624524f, 0.934735f, -1.650267f, -0.810861f, 2.664924f, -1.361016f, 2.269444f, -0.25824f, 1.487748f, 
    -1.87167f, -1.026415f, 1.788182f, 2.134732f, 0.727358f, 1.804821f, 2.446338f, 1.28467f, 0.878259f, 0.310652f, 
    -0.663583f, 0.426819f, -1.621781f, 1.180519f, -1.285246f, -1.709839f, 2.186806f, 0.309499f, -1.83694f, 1.641386f, 
    -0.030603f, -1.9107f, 2.502872f, 0.750357f, -1.649416f, 1.189143f, 1.696514f, -1.3433f, 1.123679f, -1.629001f, 
    -1.163403f, 1.670298f, 1.301242f, 0.210077f, 2.425141f, -1.384105f, 0.39949f, 1.071621f, 0.268424f, -0.279311f, 
    -1.929832f, 1.174778f, 2.712368f, 1.259047f, 0.037541f, -1.488749f, 0.424381f, -0.327814f, 1.452811f, 0.273991f, 
    0.258519f, 0.042854f, -0.346574f, -0.615438f, -0.79936f, -1.042715f, 1.770572f, -1.297297f, 1.649647f, -0.298651f, 
    -1.048315f, 2.919409f, -0.240338f, 1.922086f, -0.913307f, 0.441175f, -0.214455f, 2.137821f, -1.785499f, 0.820243f, 
    2.749585f, -0.501614f, 0.777862f, 0.920702f, 1.217655f, 0.06251f, -1.516837f, 1.644894f, -0.847783f, -0.345033f, 
    -1.713706f, -1.233106f, -0.582036f, 1.028212f, 1.301552f, 2.143174f, 1.806924f, 2.207125f, -0.579078f, 1.906502f, 
    0.982897f, -0.865886f, -0.989911f, 1.919585f, -1.447719f, 0.344478f, 1.894133f, -0.52895f, 2.858046f, 2.373685f, 
    -1.349178f, 0.362629f, 2.065136f, -1.441838f, 0.724043f, 2.071793f, 2.334071f, -0.52413f, 2.822581f, 2.40418f, 
    0.036858f, 2.324529f, 0.740669f, 2.940693f, 0.317033f, 1.637373f, -1.629464f, -0.746877f, 0.00927f, 1.425529f, 
    1.31462f, -0.908295f, -0.468471f, 0.516123f, 1.388376f, 1.530926f, 0.201057f, -1.503039f, 0.645908f, -1.16199f, 
    2.467333f, 0.957414f, 0.49212f, 2.376532f, 1.713727f, 1.630455f, -0.228587f, -1.105875f, -0.187591f, -1.187659f, 
    -0.359362f, 2.593583f, -1.517742f, 0.332665f, 1.637244f, 1.99322f, 0.203073f, 1.909803f, 1.451433f, 0.550977f, 
    2.933184f, -1.04719f, -1.221934f, 2.949385f, 1.593447f, 0.817332f, 0.671072f, -0.552762f, 0.499745f, 0.830748f, 
    -1.974666f, 2.115046f, -1.418005f, -0.173569f, -0.284441f, -0.885543f, -1.857606f, 1.274526f, 1.336182f, 0.679476f, 
    -1.579514f, 0.95177f, -1.523659f, 1.676437f, -1.655538f, 0.942799f, 0.264698f, 0.264226f, 2.191175f, -1.748495f, 
    -0.807964f, -1.651592f, -1.570845f, -1.87921f, -1.39217f, -0.303364f, -1.318291f, -0.870099f, 0.85997f, -0.907592f, 
    1.153272f, -1.236432f, -1.130912f, -1.73134f, 2.574367f, 0.109466f, 2.412357f, 0.433988f, 0.383013f, 2.387408f, 
    2.274064f, -0.161339f, -1.298869f, -0.005196f, 0.525504f, -0.39202f, 1.484107f, -1.451721f, 2.152957f, -0.8672f, 
    -0.943481f, 1.142861f, 0.184091f, 1.221996f, 2.729434f, 1.887683f, 1.727574f, 0.81034f, -1.176404f, 2.767824f, 
    -0.844923f, 1.67775f, -1.405007f, 1.50139f, 1.086983f, -0.698317f, 0.42956f, -1.312916f, -0.549957f, 1.191849f, 
    2.787362f, -1.667168f, 0.586058f, 2.610515f, 1.065882f, -1.418429f, 0.493697f, -0.266616f, 0.365524f, 1.723124f, 
    0.808873f, -0.438026f, 0.893164f, 2.160716f
};

static const float input_2_scale = 0.0193100013f;
static const float input_2_zero_point = 0.4873595238f;
static const int8_t input_2_scales_frac[] = {20};
static const int input_2_sa_dim = -1;

static const int input_2_fx8_frac = 5;

#define INPUT_2_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {8*10*1, 10*1, 1}, \
    /* .shape = */ {6, 7, 7}, \
    /* .rank =  */ 3


static const mli_tensor input_2_memstr_tsr_fx16 = {
    INPUT_2_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_2_memstr_tsr_sa8 = {
    INPUT_2_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data,
                              sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
                            sizeof(input_1_data) / sizeof(input_1_data[0]),
                            &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_1_memstr_fx16(input_1_memstr_tsr_fx16, input_1_fx8_frac + 8, input_1_data,
                                    sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_memstr_sa8(input_1_memstr_tsr_sa8, input_1_sa_dim, input_1_data, 
                                    sizeof(input_1_data) / sizeof(input_1_data[0]),
                                    &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_2_memstr_fx16(input_2_memstr_tsr_fx16, input_2_fx8_frac + 8, input_2_data,
                                     sizeof(input_2_data) / sizeof(input_2_data[0]));
tensor_quantizer input_2_memstr_sa8(input_2_memstr_tsr_sa8, input_2_sa_dim, input_2_data,
                                    sizeof(input_2_data) / sizeof(input_2_data[0]),
                                    &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);

// Generated weights vectors
//========================================
static const float weights_1_data[] = {
    -0.514681f, -0.813339f, 0.28109f, 0.119411f, 0.629025f, 0.521227f, 0.411447f, -0.684829f, -0.070376f, 0.059774f, 
    0.25139f, 0.439824f, 0.488577f, 0.19706f, -0.702872f, -0.243088f, 0.131168f, 0.325799f, -0.524818f, -0.28095f, 
    0.474624f, 0.046544f, -0.408008f, 1.335377f, 0.171412f, -0.665568f, -0.013372f, 0.429435f, -1.296239f, -1.38123f, 
    -0.26686f, 0.156087f, -0.184014f, 0.118936f, -0.918997f, -0.013775f, -0.407927f, -0.68799f, 1.168229f, -0.119136f,
     0.291377f, -0.267657f, 0.105409f, 1.384503f, -0.744617f, 0.382196f, -0.279554f, -0.252394f, 0.691139f, 0.09661f, 
    0.791492f, -0.084297f, -0.471391f, -0.401745f, -0.204422f, -1.481886f, 0.304261f, -0.195589f, -0.049959f, 
    1.412329f, -0.218979f, -0.884448f, -0.513236f, -0.509409f, -0.077728f, 0.98941f, -0.261367f, 0.314638f, -1.015214f
    , 0.285719f, 0.095967f, 1.321387f, 0.057658f, -0.36223f, -0.292453f, 0.155974f, -0.304052f, -1.706459f, 0.179586f,
     0.294703f, 1.218515f, -0.281901f, -0.011824f, -0.5502f, -0.16435f, 1.126442f, 0.874174f, 0.083975f, -0.419333f, 
    -0.09473f, -0.074018f, 1.004564f, -0.836062f, 0.158232f, -0.741527f, -1.145911f, -0.563787f, -0.041813f, -0.785309f
    , -0.13277f, 1.332846f, -1.225953f, -0.398022f, -0.057896f, -0.819458f, -0.503983f, -0.988575f, 0.496907f, 
    -0.410957f, -0.263573f, -0.956622f, -0.166665f, -0.157697f, -0.218268f, 0.042462f, 0.174852f, 0.422301f, -0.090437f
    , 0.155348f, 1.094301f, 0.076985f, -0.146415f, -0.806523f, -0.100553f, 0.628148f, -0.600378f, -0.000983f, 
    -0.522097f, -0.53389f, -0.110325f, 0.264378f, 0.712703f, -0.141585f, 0.206686f, 0.912617f, -0.055176f, 0.471613f, 
    0.06365f, -0.366392f, -0.18045f, 0.040575f, 0.302553f, 0.469871f, 0.590352f, 0.673516f, 0.357208f, -0.29236f, 
    0.196226f, -0.686653f, -0.753475f, -0.275288f, -0.239034f, 1.225803f, -0.286841f, 0.295394f, 1.119199f, -0.107007f
    , -0.06875f, 1.391504f, -0.295184f, 0.852699f, -0.511842f, 0.151853f, 0.442586f, 1.088541f, -0.262988f, -0.550834f
    , -0.425935f, 0.304138f, -0.67877f, 0.253705f, 0.05506f, -0.501335f, 0.471534f, -0.021045f, 0.112798f, 0.57899f, 
    -0.46456f, -0.713893f, 0.545605f, -0.038559f, -0.679545f, 0.164071f, 0.263258f, 0.318081f, 0.276203f, -0.198273f, 
    -0.303624f, -0.483597f, 0.513492f, -0.889671f, -0.010094f, 0.383561f, -0.221573f, -1.982108f, 0.435262f, -0.690922f
    , -0.064783f, 0.246325f, -0.338102f, -0.264832f, -0.005156f, -0.400871f, 0.344227f, -0.025399f, -0.457112f, 
    0.456221f, -0.016413f, 0.623502f, 0.11311f, 0.360878f, 1.530652f, -1.053973f, 0.098192f, -0.319295f, 1.321422f, 
    -0.042735f, 0.610309f, 0.986145f, 0.582277f, 0.746768f, 0.673027f, -0.17056f, 0.24916f, 2.256253f, -0.272432f, 
    0.135918f, 0.166599f, 0.218894f, -0.471416f, -0.288898f, -0.004881f, -0.285753f, -1.840852f, -0.269143f, 
    -0.861721f, 1.060404f, 0.148629f, -0.430828f, 0.60332f, -0.185052f, -0.304387f, -1.674397f, -0.668502f, -1.064901f,
     0.35301f, 0.128075f, 0.277548f, 0.893357f, 0.661585f, -0.121742f, 0.9772f
};

static const float weights_1_scales[] = {0.0054420424f, 0.0120523814f, 0.0177657735f};
static const float weights_1_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t weights_1_scales_frac[] = {22, 21, 20};
static const int weights_1_sa_dim = 3;

static const int weights_1_fx8_frac = 5;

#define WEIGHTS_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 4, 7, 3}, \
    /* .rank =  */ 4


static const mli_tensor weights_1_tsr_fx8 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_1_tsr_fx16 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_1_tsr_sa8 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

#define WEIGHTS_1_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {5*8*7*1, 8*7*1, 7*1, 1}, \
    /* .shape = */ {3, 4, 7, 3}, \
    /* .rank =  */ 4


static const mli_tensor weights_1_memstr_tsr_fx8 = {
    WEIGHTS_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_1_memstr_tsr_fx16 = {
    WEIGHTS_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_1_memstr_tsr_sa8 = {
    WEIGHTS_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float weights_2_data[] = {
    -0.429326f, 0.179115f, 0.385005f, 0.324496f, -0.711918f, -0.815551f, 0.253848f, -0.117591f, -0.957254f, -0.450211f,
     -0.434503f, -1.174495f, 0.055922f, -0.203394f, 0.344811f, 0.279997f, -0.017828f, -0.589972f, -0.058008f, 0.187947f
    , -1.889045f, -0.04663f, -0.943767f, -0.527539f, -0.199537f, 0.022817f, -0.107378f, -0.424022f, -0.40301f, 
    -0.101448f, 0.157511f, -0.870898f, 0.384631f, 0.675249f, -0.903358f, -0.356891f, 0.186255f, 1.32339f, -0.872754f, 
    0.343803f, 0.664403f, -1.193943f, -0.452426f, 0.571556f, 0.160856f, 0.131433f, -0.235488f, -0.913688f, -0.03364f, 
    1.071222f, 0.622198f, -0.301478f, 0.597027f, -0.664929f, 0.267884f, -0.24353f, -0.013121f, -0.456754f, 0.187704f, 
    0.807097f, -0.194894f, -1.16848f, -0.368003f, -0.081259f, -0.164851f, 0.768034f, 0.012424f, -0.405287f, 0.003002f, 
    0.141326f, 0.669743f, -0.963211f, 0.000831f, -0.435706f, 1.04353f, 0.33007f, 0.030285f, -0.095872f, 0.450822f, 
    -0.276708f, 2.138779f, 0.099941f, 1.042223f, -0.816672f, -0.035444f, 1.173744f, -0.425477f, -0.348314f, -0.616082f, 
    -1.029526f, -0.088265f, -0.56447f, 1.117046f, -0.069244f, -0.490999f, 0.167999f, 0.116474f, 0.075978f, 0.180008f, 
    0.065848f, 0.598555f, 0.845499f, 0.189212f, 0.11691f, 0.164211f, -0.279765f, 0.73441f, 0.778085f, -0.178877f, 
    0.486882f, 0.027965f, 0.038553f, 0.286721f, 0.32019f, 0.180967f, -0.687635f, -0.178808f, -0.239555f, 1.125391f, 
    0.828247f, -0.339891f, 1.113509f, 1.49536f, 0.532904f, -0.194239f, -0.170096f, 0.060172f, -0.859277f, -0.770439f, 
    -0.164874f, 0.927485f, -0.402342f, 0.249215f, -0.600724f, -0.646714f, -0.333532f, -0.786812f, 0.025745f, 0.034507f, 
    -0.094337f, -0.19666f, 0.323121f, 0.06882f, 1.553635f, 0.188626f, 0.598296f, 0.62439f, 0.075021f, 0.513109f, 
    1.291583f, 0.117459f, 0.527565f, 0.162041f, 0.18463f, -0.57616f, -0.389615f, 0.283135f, -0.657172f, -0.849878f, 
    0.008829f, 0.199749f, -0.530664f, 0.278764f, -0.337648f, 0.373837f, -0.280634f, 0.394343f, 1.268408f, 0.623402f, 
    0.125548f, 0.447495f, -0.116811f, 0.486062f, 0.887078f, 0.101421f, 0.886914f, 0.552681f, 0.265728f, 1.167428f, 
    0.117578f, -0.27736f, -0.918578f, 2.065294f, -0.36874f, 0.638232f, -0.414111f, 0.214881f, 1.061407f, 0.400319f, 
    -0.182019f, -1.420189f, 0.880795f, 0.026581f, 0.694936f, -0.039205f, -0.050977f, 0.384079f, 0.030458f, 0.379287f, 
    0.371574f, -0.177958f, 0.059795f, 0.812588f, -0.025217f, 0.142205f, 1.185495f, -0.451813f, -0.338401f, 0.331788f, 
    1.334148f, 0.281718f, 0.737145f, -0.056727f, 0.453142f, 0.45675f, -0.386746f, -0.263556f, -0.178901f, 0.081319f, 
    -0.027898f, -0.308265f, -0.529952f, -0.163682f, -0.546096f, 0.556747f, -0.226577f, 0.929046f, 0.821552f, 0.135522f, 
    0.948511f, 1.18204f, -0.246047f, -0.485889f, -1.305959f, 0.310535f, 0.169456f, -0.439719f, -0.142477f, 0.154759f, 
    -1.078846f, 0.127324f, 0.544206f, 0.233832f, 0.164121f, -0.154471f, 1.365787f, 0.481901f, -0.253203f, 1.036454f, 
    -0.028946f, -0.940035f, -1.848657f
};

static const float weights_2_scales[] = {0.0053169183f, 0.0111825867f, 0.0168407764f};
static const float weights_2_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t weights_2_scales_frac[] = {22, 21, 20};
static const int weights_2_sa_dim = 3;

static const int weights_2_fx8_frac = 5;
#define WEIGHTS_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 3, 7, 3}, \
    /* .rank =  */ 4


#define WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {3*7*5*1, 7*5*1, 5*1, 1}, \
    /* .shape = */ {4, 3, 7, 3}, \
    /* .rank =  */ 4


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
    -0.568325f, 0.556923f, -0.175918f, -1.959237f, 0.93365f, 0.03384f, -0.184951f, -0.237696f, -2.490783f, 0.941078f, 
    -0.159079f, -0.469947f, -0.503309f, 2.349342f, -0.607883f, 0.157542f, 0.420767f, 0.117734f, -1.614202f, -0.497699f, 
    0.293307f, 0.403358f, 0.085876f, -2.161647f, -1.593411f, 0.30968f, 0.877831f, -0.328494f, -0.3652f, 2.348464f, 
    0.174293f, 0.281362f, 0.913772f, -0.488018f, -3.460793f, -0.017046f, 0.645641f, 0.9744f, -0.369711f, -0.596923f, 
    0.369745f, -0.458423f, -0.055117f, -0.070666f, 3.610572f, -0.222589f, -0.641287f, 0.983673f, 0.593106f, -1.545344f, 
    0.200386f, 0.458f, -0.905701f, 1.423956f, 0.787442f, 0.363842f, -0.681295f, -0.172728f, -0.864614f, -0.078431f, 
    -0.180629f, -0.112668f, -0.675433f, -0.199799f, -3.813763f, -0.138011f, 0.349825f, -1.14939f, 0.016379f, 2.524868f, 
    0.425882f, -0.960518f, -0.9648f, 1.099253f, -1.599318f, -0.395179f, 0.085577f, -1.153976f, -0.54708f, 1.700083f, 
    0.052158f, -0.334696f, 0.58405f, 1.995081f, -0.361351f, 0.045629f, 0.663931f, -1.441008f, 0.466744f, 1.221385f, 
    0.403004f, 0.420114f, -0.112318f, 0.274544f, 1.151413f, -0.307708f, 0.594736f, -0.446652f, 0.466613f, 1.521286f, 
    -0.292203f, 0.290332f, -0.408482f, -0.323337f, 2.951766f, 0.274716f, -0.702756f, 0.535653f, 0.253097f, -0.439136f, 
    0.349553f, 0.341441f, -0.707367f, -1.37812f, -1.032227f, 0.524842f, 0.719258f, 0.666874f, -0.829071f, 1.764994f, 
    0.430323f, 0.114432f, -1.733507f, -1.931313f, 1.54837f, 0.325035f, -0.765372f, 0.492583f, -2.074497f, 2.171005f, 
    0.132712f, 1.377192f, -0.006062f, 0.313715f, -2.427777f, -0.499898f, 0.31632f, 0.57408f, -0.74169f, -2.1614f
};

static const float weights_3_scales[] = {0.0044750031f, 0.0108440351f, 0.0136496602f, 0.0196124688f, 0.0300296284f};
static const float weights_3_zero_points[] = {0.f, 0.f, 0.f, 0.f, 0.f};
static const int8_t weights_3_scales_frac[] = {22, 21, 21, 20, 20};
static const int weights_3_sa_dim = 3;

static const int weights_3_fx8_frac = 5;

#define WEIGHTS_3_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {2*8*8*1, 8*8*1, 8*1, 1}, \
    /* .shape = */ {2, 2, 7, 5}, \
    /* .rank =  */ 4


static const mli_tensor weights_3_memstr_tsr_fx8 = {
    WEIGHTS_3_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_3_memstr_tsr_fx16 = {
    WEIGHTS_3_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_3_memstr_tsr_sa8 = {
    WEIGHTS_3_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float weights_4_data[] = {
    -0.344528f, -0.400265f, -1.08055f, 0.012469f, 0.998779f, -0.101981f, 0.165913f, 0.043027f, -1.375209f, -0.310435f, 
    -0.179534f, 0.606649f, -0.347562f, 0.316881f, 0.896875f, -0.128827f, -0.031994f, 2.370403f, -1.093224f, 0.462575f, 
    -0.332328f, -0.066358f, 1.36381f, -1.555277f, -0.465427f, 0.425626f, -0.770176f, -0.594575f, -0.764326f, 0.875896f, 
    -0.195446f, 0.188799f, -0.4947f, 0.043171f, 4.298376f, -0.019003f, -0.062621f, -0.258114f, -0.462567f, 2.256041f, 
    -0.631018f, 0.465196f, -0.825707f, -0.582521f, 0.649454f, -0.157315f, -0.591495f, -0.753531f, 0.316908f, -1.718512f, 
    0.138504f, 0.235726f, -0.125792f, 0.730334f, 0.475339f, 0.056903f, -0.544169f, 0.494625f, 0.282371f, 1.38829f, 
    0.589399f, -0.096448f, 0.550256f, -1.173738f, 0.039171f, 0.023616f, -0.256351f, -0.447438f, -0.770704f, -1.008304f, 
    0.501402f, -0.25799f, 1.637521f, -1.818654f, -0.955313f, -0.082082f, -0.074182f, -1.100386f, -0.923971f, 0.904258f, 
    0.45864f, 0.525086f, -0.329574f, -0.321235f, -1.978449f, 0.160233f, -0.307255f, 0.021051f, 2.813702f, -2.577382f, 
    -0.595801f, 0.133921f, -0.620319f, -1.194616f, 0.176514f, -0.293285f, 0.704665f, -0.447282f, -1.338537f, 1.867033f, 
    0.277741f, -0.133158f, 1.977369f, 0.153608f, 1.947577f, -0.319231f, 0.059418f, 0.953515f, -0.997706f, 0.375757f, 
    0.169746f, -0.415115f, 1.480352f, -0.538222f, 1.911838f, -0.068688f, -0.119414f, -0.793959f, -0.859162f, -2.618059f, 
    0.079576f, -0.834957f, -0.155428f, -2.593916f, 0.29742f, 0.059524f, 0.325789f, -0.62481f, 2.665011f, -0.936745f, 
    0.535077f, 1.387748f, -0.626463f, -1.216554f, -0.625897f, -0.089359f, 0.022044f, 0.020826f, -0.396088f, -1.219117f, 
    0.525558f, -0.172327f, -0.050953f, -0.129161f, -0.272397f, -0.132263f, -0.4808f, 0.401722f, 1.444905f, 0.05363f, 
    -0.246401f, 0.732324f, -2.409892f, 2.529404f, 1.144917f, 0.063518f, -0.879115f, 0.226441f, -0.792624f, 0.261524f, 
    -0.015104f, -0.423009f, 1.078093f, 0.305413f, -1.773685f, -0.087819f, -0.414863f, -0.148039f, 0.059361f, 0.469314f, 
    -0.461809f, 0.083518f, -0.296167f, -1.259929f, -0.115881f, -0.205164f, -0.265681f, -0.511037f, -0.226399f, 
    -1.312477f, 0.225248f, -0.412041f, 1.357132f, 0.450887f, 2.281838f, 0.044501f, -0.647276f, 0.172765f, 1.051127f, 
    -0.364579f, 0.068451f, 0.041836f, -0.965833f, -1.088727f, -2.244696f, 0.114904f, -0.680508f, 0.687137f, -0.29416f, 
    -1.655159f, 0.089186f, 0.131592f, -1.62994f, -0.734126f, 0.22862f, 0.403415f, -0.243093f, -0.23304f, -1.380456f, 
    0.039886f, 0.07141f, 1.074497f, 0.434464f, -1.934811f, 1.54225f, 0.098378f, -0.206137f, 1.069564f, 0.521499f, 
    0.092448f, 0.336436f, 0.204255f, -0.943613f, 0.791198f, -0.574255f, 0.413339f, 0.636192f, -0.152099f, -1.02228f, 
    -0.584094f, 0.69757f, -0.577982f, 0.098476f, 1.312457f, 0.307292f, 0.171578f, -0.109153f, 0.638828f, -0.489385f, 
    -0.10175f, 0.026741f, -0.623864f, -0.469475f, 1.56992f, 0.171912f, -0.229814f, 0.476844f, -1.322632f, -0.763036f, 
    0.623563f, -0.291115f, 0.638934f, -1.457311f, 0.501265f, 0.947283f, 0.289122f, -0.599772f, 1.203439f, -1.051804f, 
    0.322984f, 0.160236f, -0.445703f, 0.096532f, 1.363351f, -0.090388f, 0.208069f, -1.313838f, -0.783447f, -3.629148f, 
    1.879543f, 0.374365f, 0.211391f, 0.830345f, 0.119511f, 1.013509f, -0.424619f, 0.400326f, 0.945775f, -0.50562f, 
    0.973882f, -0.040006f, -0.370981f, -0.393675f, 1.076107f, 2.921102f, -0.100603f, 0.625333f, -0.621957f, 1.971185f, 
    2.289851f, -0.055599f, -0.09115f, -0.935977f, -0.294526f, 3.296776f, 0.493616f, -0.138253f, -1.339782f, -0.469213f, 
    -0.049293f, -0.152734f, -0.847742f, -0.925832f, 0.195313f, -0.671833f, 0.43588f, -0.18745f, 0.554728f, -1.705622f, 
    1.960333f, -0.231184f, -0.169869f, -0.215575f, -0.947481f, -0.926851f, 0.186117f, -0.524379f, -1.911913f, 0.095015f, 
    2.45384f, 0.111198f, -0.317669f, 0.909094f, -1.791777f, 2.919982f, -0.357588f, -0.404184f, -0.761821f, 2.239159f, 
    -1.253623f, 0.191457f, -0.023778f, 2.120935f, 0.330579f, 0.490848f, 0.303447f, -0.144574f, 0.424506f, 0.860524f, 
    1.498205f, 0.419289f, 0.199116f, 0.365511f, 0.389156f, 0.327333f, -0.020039f, -0.339562f, 0.665056f, 1.516554f, 
    2.632515f, -0.380281f, 1.051786f, -2.458638f, 2.236818f, 2.512681f, -0.075958f, 0.607398f, 0.280214f, 0.442161f, 
    0.410975f, -0.279364f, 0.780394f, -0.117713f, -2.876006f, 2.501356f, 0.367281f, -0.303711f, 0.598453f, -1.202611f, 
    -1.365246f, 0.334674f, 0.339662f, 0.32637f, -1.348392f, 1.78494f, 0.315385f, -0.271596f, -0.212651f, 0.567281f, 
    -1.479356f, 0.39542f, 0.29205f, -1.112317f, -1.298018f, 1.106737f, 0.672203f, -0.418357f, 1.274226f, 1.698393f, 
    -1.640805f, 0.37544f, -0.426982f, 0.860954f, -1.922582f, 2.685942f, -0.437885f, -0.868749f, -0.186408f, 0.76756f, 
    -0.329038f, 0.363715f, 0.47353f, 0.288331f, -0.026119f, 0.135635f, -0.21518f, 0.252147f, -0.075959f, -1.333704f, 
    0.581556f, 0.115971f, -1.156862f, 3.378402f, 3.23628f, -0.739628f, 0.33939f, 0.012322f, -0.102582f, -0.621741f, 
    -0.575499f, -0.135253f, 0.296085f, -2.052871f, 2.683316f, -1.319392f, 0.047499f, -0.422186f, -0.837261f, -2.117929f, 
    1.005204f, 0.154911f, 0.022836f, -0.081964f, -1.343868f, -1.121452f, 0.608304f, 0.384536f, -0.431087f, 3.143294f, 
    -0.548279f, -0.412794f, -1.502661f, -0.726238f, 1.453353f, 0.999264f, 0.031758f, 0.171434f, 1.285993f, -0.026693f, 
    2.246813f, -0.074892f, -0.217566f, 1.101416f, -0.58148f, -0.243949f, 0.107617f, -0.510983f, -1.563234f, 1.066056f, 
    0.164167f, -0.349182f, 0.287206f, 0.52598f, -0.454179f, 2.664182f, -0.322559f, -0.633737f, -0.779715f, 0.589299f, 
    -0.52149f, -0.147741f, 0.278779f, -1.175372f, -0.373419f, 0.254431f, 0.060055f, -0.279541f, 0.41905f, -1.127611f, 
    0.125829f, -0.171088f, -1.019682f, -0.562197f, -0.459236f, 2.64527f, -0.067399f, 0.590729f, -0.406416f, -0.256065f, 
    1.512456f, 0.215486f, -0.179604f, 0.254883f, 1.040926f, -0.221491f, 0.001435f, -0.093505f, -0.51147f, -0.360849f, 
    -0.09995f, -0.164271f, 0.414829f, -0.192947f, -0.756399f, -0.351075f, -0.235017f, -1.223457f, 2.04016f, 0.700707f, 
    1.236093f, 0.289558f, -0.794435f, -1.265162f, -0.792799f, -0.77841f, 0.032208f, 0.009678f, 0.315327f, 1.439786f, 
    -1.086806f, 0.033071f, -0.879137f, 1.107125f, 1.194098f, 0.021064f, 0.670591f, -1.059507f, 1.262136f, 0.057799f, 
    1.377281f, 0.252204f, 0.019616f, 0.07012f, 0.251184f, 3.745033f, -0.558119f, 0.325928f, -0.260324f, 0.552917f, 
    -0.067908f, 0.287515f, -0.1643f, 0.364718f, -1.353579f, 0.091137f, -0.178431f, 0.473426f, -1.89655f, -0.339556f, 
    -2.865804f, -0.113939f, -0.13395f, 1.731968f, 1.046747f, 0.191813f, 0.285615f, 0.258428f, -0.173027f, 0.301153f, 
    -1.587649f
};

static const float weights_4_scales[] = {0.0054926742f, 0.0118319755f, 0.0266015865f, 0.028575968f, 0.0338454805f};
static const float weights_4_zero_points[] = {0.f, 0.f, 0.f, 0.f, 0.f};
static const int8_t weights_4_scales_frac[] = {22, 21, 20, 20, 19};
static const int weights_4_sa_dim = 3;

static const int weights_4_fx8_frac = 4;

#define WEIGHTS_4_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {4*8*8*1, 8*8*1, 8*1, 1}, \
    /* .shape = */ {4, 4, 7, 5}, \
    /* .rank =  */ 4


static const mli_tensor weights_4_memstr_tsr_fx8 = {
    WEIGHTS_4_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_4_memstr_tsr_fx16 = {
    WEIGHTS_4_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_4_memstr_tsr_sa8 = {
    WEIGHTS_4_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float weights_5_data[] = {
    0.162124f, -0.346638f, 0.845082f, 0.080916f, -0.158546f, -0.791927f, 0.112239f, -1.218158f, 0.174856f, -0.180816f,
    0.456323f, -1.192600f, -0.167070f, -0.112744f, 1.795874f, 0.201216f, -0.626168f, 1.975422f, -0.010648f, 0.514551f, 
    1.412555f, 0.316056f, -0.843186f, 1.071855f, 0.041489f, 0.083707f, -0.201573f, -0.228914f, -0.571671f, -0.514952f, 
    -0.086343f, -0.450086f, 0.785266f, 0.160865f, -0.859547f, 0.973818f, 0.095020f, -0.113663f, -0.678512f, 0.150543f, 
    -0.053690f, 0.594102f, 0.180646f, -0.609229f, 1.355713f, 0.051250f, -0.209586f, -0.523649f, -0.229502f, -0.025293f, 
    -1.294115f, -0.119114f, -0.011287f, 0.480028f, -0.042190f, 0.952942f, -0.112790f, -0.091807f, 0.520569f, -0.025375f,
    -0.246422f, 0.445169f, 0.087788f, 0.135852f, -1.110783f, -1.154933f, -0.317469f, -0.574920f, -1.332593f, -0.339507f, 
    0.507098f, 1.066522f, -0.292976f, 0.187145f, -0.103800f, -0.024647f, 1.115174f, -0.563868f, 0.281211f, -0.422818f, 
    -1.013970f, 0.555492f, 0.145126f, -1.099089f, 0.604236f, 1.122268f, -0.408094f, 0.018153f, -0.393905f, -0.033450f, 
    -0.127945f, -0.934519f, -0.229592f, -0.161013f, 0.133046f, 0.509755f, -0.152131f, -0.646317f, -0.031912f, 0.185736f, 
    -0.784153f, -1.109624f, 0.160113f, 0.058988f, -0.389235f, -0.453502f, -0.631380f, -1.169693f, -0.152213f, 0.608713f, 
    0.649977f, -0.545584f, 0.283266f, -0.374898f, -0.041339f, 0.767390f, 1.445703f, -0.111479f, -0.230255f, 0.780456f, 
    0.027445f, -0.158883f, 0.259358f, -0.074331f, -0.193207f, -0.371220f, 0.057664f, 0.628225f, 1.759768f, -0.073189f, 
    0.049544f, -0.011524f, -0.077964f, -0.539795f, 0.389784f, 0.314573f, 0.244022f, -1.029533f, 0.282667f, -0.232915f, 
    -0.173739f, -0.174215f, -0.403591f, -0.504128f, 0.416343f, 0.670114f, -0.217452f, 0.002100f, -0.212137f, 0.374115f, 
    -0.061205f, 0.780235f, -0.994838f, -0.023874f, -0.450788f, -0.022117f, 0.070457f, -0.261881f, -0.135130f, -0.283063f,
    0.437815f, -0.146060f, 0.020644f, -1.297438f, -0.733107f, -0.212464f, 1.094582f, 0.081259f, -0.107395f, -0.427986f,
    0.854578f, -0.089741f, 0.165120f, -1.317230f, -0.209578f, 0.102788f, -1.051647f, 0.224085f, -0.826160f, -0.008636f,
    -0.182198f, 0.283343f, -0.559180f, 0.599075f, -0.313968f, -0.175495f, 0.269183f, 0.542840f, -0.410377f
};

static const float weights_5_scales[] = {0.0047577615f, 0.0102160443f, 0.0155545054f};
static const float weights_5_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t weights_5_scales_frac[] = {22, 21, 21};
static const int weights_5_sa_dim = 3;

static const int weights_5_fx8_frac = 6;

#define WEIGHTS_5_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {3*8*4*1, 8*4*1, 4*1, 1}, \
    /* .shape = */ {3, 3, 7, 3}, \
    /* .rank =  */ 4


static const mli_tensor weights_5_memstr_tsr_fx8 = {
    WEIGHTS_5_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_5_memstr_tsr_fx16 = {
    WEIGHTS_5_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_5_memstr_tsr_sa8 = {
    WEIGHTS_5_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float bias_1_data[] = {-0.149134f, -0.723882f, 0.276216f};

static const float bias_1_i1_w1_scales[] = {0.0000638685f, 0.0001414482f, 0.0002085013f};
static const float bias_1_i1_w1_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t bias_1_i1_w1_scales_frac[] = {28, 27, 27};
static const int bias_1_i1_w1_sa_dim = 0;

static const float bias_1_i1_w2_scales[] = {0.0000624f, 0.0001312402f, 0.0001976454f};
static const float bias_1_i1_w2_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t bias_1_i1_w2_scales_frac[] = {28, 27, 27};
static const int bias_1_i1_w2_sa_dim = 0;

static const float bias_1_i1_w5_scales[] = {0.0000558376646f, 0.0001198967f, 0.0001825496f};
static const float bias_1_i1_w5_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t bias_1_i1_w5_scales_frac[] = {28, 28, 27};
static const int bias_1_i1_w5_sa_dim = 0;


static const int bias_1_fx8_frac = 7;

#define BIAS_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3}, \
    /* .rank =  */ 1


static const mli_tensor bias_1_tsr_fx8 = {
    BIAS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor bias_1_tsr_fx16 = {
    BIAS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor bias_1_i1_w1_tsr_sa32 = {
    BIAS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};


static const mli_tensor bias_1_i1_w2_tsr_sa32 = {
    BIAS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_1_i1_w5_tsr_sa32 = {
    BIAS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};


static const float bias_2_data[] = {-0.195573f, 0.092147f, -0.123712f, -0.580636f, -0.688343f};

static const float bias_2_i2_w3_scales[] = {0.0000864123f, 0.0002093983f, 0.000263575f, 0.0003787168f, 0.0005798722f};
static const float bias_2_i2_w3_zero_points[] = {0.f, 0.f, 0.f, 0.f, 0.f};
static const int8_t bias_2_i2_w3_scales_frac[] = {28, 27, 26, 26, 25};
static const int bias_2_i2_w3_sa_dim = 0;

static const float bias_2_i2_w4_scales[] = {0.0001060635f, 0.0002284755f, 0.0005136767f, 0.000551802f, 0.0006535563f};
static const float bias_2_i2_w4_zero_points[] = {0.f, 0.f, 0.f, 0.f, 0.f};
static const int8_t bias_2_i2_w4_scales_frac[] = {28, 27, 25, 25, 25};
static const int bias_2_i2_w4_sa_dim = 0;

static const int bias_2_fx8_frac = 7;

#define BIAS_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {5}, \
    /* .rank =  */ 1


static const mli_tensor bias_2_tsr_fx8 = {
    BIAS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor bias_2_tsr_fx16 = {
    BIAS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor bias_2_i2_w3_tsr_sa32 = {
    BIAS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};


static const mli_tensor bias_2_i2_w4_tsr_sa32 = {
    BIAS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_1_fx8(weights_1_tsr_fx8, weights_1_fx8_frac, weights_1_data,
                        sizeof(weights_1_data) / sizeof(weights_1_data[0]));
//tensor_quantizer weights_1_fx16(weights_1_tsr_fx16, weights_1_fx8_frac + 8, weights_1_data, 189);
tensor_quantizer weights_1_fx16(weights_1_tsr_fx16, weights_1_fx8_frac + 8, weights_1_data,
                        sizeof(weights_1_data) / sizeof(weights_1_data[0]));
tensor_quantizer weights_1_sa8(weights_1_tsr_sa8, weights_1_sa_dim, weights_1_data,
                        sizeof(weights_1_data) / sizeof(weights_1_data[0]), 
                        weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]), 
                        weights_1_zero_points, sizeof(weights_1_zero_points) / sizeof(weights_1_zero_points[0]), 
                        weights_1_scales_frac, sizeof(weights_1_scales_frac) / sizeof(weights_1_scales_frac[0]));

tensor_quantizer weights_1_memstr_fx8(weights_1_memstr_tsr_fx8, weights_1_fx8_frac, weights_1_data,
                        sizeof(weights_1_data) / sizeof(weights_1_data[0]));
tensor_quantizer weights_1_memstr_fx16(weights_1_memstr_tsr_fx16, weights_1_fx8_frac + 8, weights_1_data,
                        sizeof(weights_1_data) / sizeof(weights_1_data[0]));
tensor_quantizer weights_1_memstr_sa8(weights_1_memstr_tsr_sa8, weights_1_sa_dim, weights_1_data, 
                        sizeof(weights_1_data) / sizeof(weights_1_data[0]),
                        weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]), 
                        weights_1_zero_points, sizeof(weights_1_zero_points) / sizeof(weights_1_zero_points[0]), 
                        weights_1_scales_frac, sizeof(weights_1_scales_frac) / sizeof(weights_1_scales_frac[0]));

tensor_quantizer weights_2_fx8(weights_2_tsr_fx8, weights_2_fx8_frac, weights_2_data,
                        sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_memstr_fx8(weights_2_tsr_fx8, weights_2_fx8_frac, weights_2_data,
                        sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_fx16(weights_2_tsr_fx16, weights_2_fx8_frac + 8, weights_2_data,
                        sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_memstr_fx16(weights_2_tsr_fx16, weights_2_fx8_frac + 8, weights_2_data,
                        sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_sa8(weights_2_tsr_sa8, weights_2_sa_dim, weights_2_data, 
                        sizeof(weights_2_data) / sizeof(weights_2_data[0]),
                        weights_2_scales, sizeof(weights_2_scales) / sizeof(weights_2_scales[0]), 
                        weights_2_zero_points, sizeof(weights_2_zero_points) / sizeof(weights_2_zero_points[0]), 
                        weights_2_scales_frac, sizeof(weights_2_scales_frac) / sizeof(weights_2_scales_frac[0]));
tensor_quantizer weights_2_memstr_sa8(weights_2_tsr_sa8, weights_2_sa_dim, weights_2_data, 
                        sizeof(weights_2_data) / sizeof(weights_2_data[0]),
                        weights_2_scales, sizeof(weights_2_scales) / sizeof(weights_2_scales[0]), 
                        weights_2_zero_points, sizeof(weights_2_zero_points) / sizeof(weights_2_zero_points[0]), 
                        weights_2_scales_frac, sizeof(weights_2_scales_frac) / sizeof(weights_2_scales_frac[0]));

tensor_quantizer weights_3_memstr_fx8(weights_3_memstr_tsr_fx8, weights_3_fx8_frac, weights_3_data,
                                      sizeof(weights_3_data) / sizeof(weights_3_data[0]));
tensor_quantizer weights_3_memstr_fx16(weights_3_memstr_tsr_fx16, weights_3_fx8_frac + 8, weights_3_data,
                                       sizeof(weights_3_data) / sizeof(weights_3_data[0]));
tensor_quantizer weights_3_memstr_sa8(weights_3_memstr_tsr_sa8, weights_3_sa_dim, weights_3_data, 
                                      sizeof(weights_3_data) / sizeof(weights_3_data[0]),
                                      weights_3_scales, sizeof(weights_3_scales) / sizeof(weights_3_scales[0]), 
                                      weights_3_zero_points, sizeof(weights_3_zero_points) / sizeof(weights_3_zero_points[0]), 
                                      weights_3_scales_frac, sizeof(weights_3_scales_frac) / sizeof(weights_3_scales_frac[0]));

tensor_quantizer weights_4_memstr_fx8(weights_4_memstr_tsr_fx8, weights_4_fx8_frac, weights_4_data,
                                      sizeof(weights_4_data) / sizeof(weights_4_data[0]));
tensor_quantizer weights_4_memstr_fx16(weights_4_memstr_tsr_fx16, weights_4_fx8_frac + 8, weights_4_data,
                                       sizeof(weights_4_data) / sizeof(weights_4_data[0]));
tensor_quantizer weights_4_memstr_sa8(weights_4_memstr_tsr_sa8, weights_4_sa_dim, weights_4_data, 
                                      sizeof(weights_4_data) / sizeof(weights_4_data[0]),
                                      weights_4_scales, sizeof(weights_4_scales) / sizeof(weights_4_scales[0]), 
                                      weights_4_zero_points, sizeof(weights_4_zero_points) / sizeof(weights_4_zero_points[0]), 
                                      weights_4_scales_frac, sizeof(weights_4_scales_frac) / sizeof(weights_4_scales_frac[0]));

tensor_quantizer weights_5_memstr_fx8(weights_5_memstr_tsr_fx8, weights_5_fx8_frac, weights_5_data,
                                      sizeof(weights_5_data) / sizeof(weights_5_data[0]));
tensor_quantizer weights_5_memstr_fx16(weights_5_memstr_tsr_fx16, weights_5_fx8_frac + 8, weights_5_data,
                                       sizeof(weights_5_data) / sizeof(weights_5_data[0]));
tensor_quantizer weights_5_memstr_sa8(weights_5_memstr_tsr_sa8, weights_5_sa_dim, weights_5_data, 
                                      sizeof(weights_5_data) / sizeof(weights_5_data[0]),
                                      weights_5_scales, sizeof(weights_5_scales) / sizeof(weights_5_scales[0]), 
                                      weights_5_zero_points, sizeof(weights_5_zero_points) / sizeof(weights_5_zero_points[0]), 
                                      weights_5_scales_frac, sizeof(weights_5_scales_frac) / sizeof(weights_5_scales_frac[0]));


tensor_quantizer bias_1_fx8(bias_1_tsr_fx8, bias_1_fx8_frac, bias_1_data,
                            sizeof(bias_1_data) / sizeof(bias_1_data[0]));
tensor_quantizer bias_1_fx16(bias_1_tsr_fx16, bias_1_fx8_frac + 8, bias_1_data,
                             sizeof(bias_1_data) / sizeof(bias_1_data[0]));
tensor_quantizer bias_1_i1_w1_sa32(bias_1_i1_w1_tsr_sa32, bias_1_i1_w1_sa_dim, bias_1_data, 
                                   sizeof(bias_1_data) / sizeof(bias_1_data[0]),
                                   bias_1_i1_w1_scales, sizeof(bias_1_i1_w1_scales) / sizeof(bias_1_i1_w1_scales[0]), 
                                   bias_1_i1_w1_zero_points, sizeof(bias_1_i1_w1_zero_points) / sizeof(bias_1_i1_w1_zero_points[0]), 
                                   bias_1_i1_w1_scales_frac, sizeof(bias_1_i1_w1_scales_frac) / sizeof(bias_1_i1_w1_scales_frac[0]));
tensor_quantizer bias_1_i1_w2_sa32(bias_1_i1_w2_tsr_sa32, bias_1_i1_w2_sa_dim, bias_1_data, 
                                   sizeof(bias_1_data) / sizeof(bias_1_data[0]),
                                   bias_1_i1_w2_scales, sizeof(bias_1_i1_w2_scales) / sizeof(bias_1_i1_w2_scales[0]), 
                                   bias_1_i1_w2_zero_points, sizeof(bias_1_i1_w2_zero_points) / sizeof(bias_1_i1_w2_zero_points[0]), 
                                   bias_1_i1_w2_scales_frac, sizeof(bias_1_i1_w2_scales_frac) / sizeof(bias_1_i1_w2_scales_frac[0]));
tensor_quantizer bias_1_i1_w5_sa32(bias_1_i1_w5_tsr_sa32, bias_1_i1_w5_sa_dim, bias_1_data, 
                                   sizeof(bias_1_data) / sizeof(bias_1_data[0]),
                                   bias_1_i1_w5_scales, sizeof(bias_1_i1_w5_scales) / sizeof(bias_1_i1_w5_scales[0]), 
                                   bias_1_i1_w5_zero_points, sizeof(bias_1_i1_w5_zero_points) / sizeof(bias_1_i1_w5_zero_points[0]), 
                                   bias_1_i1_w5_scales_frac, sizeof(bias_1_i1_w5_scales_frac) / sizeof(bias_1_i1_w5_scales_frac[0]));


tensor_quantizer bias_2_fx8(bias_2_tsr_fx8, bias_2_fx8_frac, bias_2_data,
                            sizeof(bias_2_data) / sizeof(bias_2_data[0]));
tensor_quantizer bias_2_fx16(bias_2_tsr_fx16, bias_2_fx8_frac + 8, bias_2_data,
                             sizeof(bias_2_data) / sizeof(bias_2_data[0]));
tensor_quantizer bias_2_i2_w3_sa32(bias_2_i2_w3_tsr_sa32, bias_2_i2_w3_sa_dim, bias_2_data, 
                                   sizeof(bias_2_data) / sizeof(bias_2_data[0]),
                                   bias_2_i2_w3_scales, sizeof(bias_2_i2_w3_scales) / sizeof(bias_2_i2_w3_scales[0]), 
                                   bias_2_i2_w3_zero_points, sizeof(bias_2_i2_w3_zero_points) / sizeof(bias_2_i2_w3_zero_points[0]), 
                                   bias_2_i2_w3_scales_frac, sizeof(bias_2_i2_w3_scales_frac) / sizeof(bias_2_i2_w3_scales_frac[0]));
tensor_quantizer bias_2_i2_w4_sa32(bias_2_i2_w4_tsr_sa32, bias_2_i2_w4_sa_dim, bias_2_data, 
                                   sizeof(bias_2_data) / sizeof(bias_2_data[0]),
                                   bias_2_i2_w4_scales, sizeof(bias_2_i2_w4_scales) / sizeof(bias_2_i2_w4_scales[0]), 
                                   bias_2_i2_w4_zero_points, sizeof(bias_2_i2_w4_zero_points) / sizeof(bias_2_i2_w4_zero_points[0]), 
                                   bias_2_i2_w4_scales_frac, sizeof(bias_2_i2_w4_scales_frac) / sizeof(bias_2_i2_w4_scales_frac[0]));

// Extracted Output vectors 
//===================================================
static const float test_1_out_data[] = {
    0.182443f, -5.460881f, -1.241028f, 1.255288f, -4.978712f, -0.382022f, 0.132413f, -2.471303f, -3.133288f, -0.543137f
    , -2.901602f, 2.217625f, -0.462109f, -2.947853f, -1.469382f, 2.066833f, 0.634305f, -0.634317f, 0.791142f, 
    -3.922265f, 2.442199f, 1.260735f, -0.549783f, -0.822284f, -1.587682f, -0.617362f, 0.574562f, -0.945824f, 0.47555f, 
    1.883455f, -2.297307f, 0.050218f, 6.073983f, -0.077768f, 0.222888f, -1.983227f, -1.680174f, -1.616613f, -1.929004f,
     -0.474543f, 0.725694f, 3.093736f, -1.779776f, -3.695127f, 7.047446f, -1.753271f, -1.819515f, 1.226207f, 2.332019f,
     -5.483848f, -3.906916f, 1.385945f, -0.593567f, 11.051339f, -0.720665f, -5.364254f, 0.373693f, 0.109426f, -
    1.527467f, 5.526871f, 1.111737f, -9.551408f, 0.472277f, 5.155255f, -0.583936f, 0.830366f, 1.88085f, -7.64768f, 
    -10.738862f, 1.159131f, -2.175375f, 6.572781f, -0.569206f, -0.263289f, -2.596092f, 0.504312f, 0.914679f, 3.157492f, 
    -1.747266f, -1.673466f, 2.854959f, -1.521472f, 2.686754f, -1.344923f, -1.940461f, -3.725044f, 2.379254f, -1.356561f
    , -0.557641f, -0.363773f, -1.436045f, -0.293904f, 3.56606f, -1.394564f, -0.786934f, 2.210115f, -0.077802f, 
    -1.545802f, -1.555384f, 0.000402f, -7.37569f, 1.934162f, 0.342423f, -4.263429f, -5.021095f, 0.422951f, -7.327517f, 
    2.325003f, 2.593388f, -7.091592f, -6.205179f, 1.075053f, 1.223205f, 4.940702f, 0.333508f, -7.952691f, 0.811212f, 
    0.792129f, 2.072053f, 6.515614f, -1.06555f, -0.007848f, -0.571216f, -0.553233f, -0.788939f, -1.645887f, -0.758075f,
     0.34902f, 2.006872f, -0.95722f, 1.018669f, -0.399273f, -2.136623f, -0.766078f, -1.71823f, -0.121529f, 0.437723f, 
    3.153871f, -1.348384f, -3.322972f, 0.803526f, -1.227272f, -2.042775f, 0.906853f, 0.608711f, -2.473988f, -2.554814f,
     -0.88894f, -3.482285f, 1.601813f, -2.094563f, -4.319683f, 9.751204f, -1.362408f, -4.998181f, 7.500357f, -0.382723f
    , -8.928977f, 2.673067f, 2.436469f, 2.202399f, 4.422984f, 1.111997f, -7.192806f, 1.01661f, 0.519441f, 0.941847f, 
    0.753118f, -1.261297f, -0.742263f, -1.410058f, -1.593839f, 4.436145f, 0.206275f, -2.26141f, -2.955983f, 2.811399f, 
    0.557433f, 2.015875f, -5.551959f, -1.41633f, -2.543018f, -3.716955f, -1.168153f, 2.690913f, 3.494143f, -2.235671f, 
    -1.975619f, 6.15405f, -0.46963f, -1.004044f, 1.5235f
};

static const float test_1_out_scale = 0.0854517668f;
static const float test_1_out_zero_point = 0.1562385559f;
static const int8_t test_1_out_scales_frac[] = {18};
static const int test_1_out_sa_dim = -1;

static const int test_1_out_fx8_frac = 3;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {8, 8, 3}, \
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
    0.f, 0.f, 0.f, 1.014343f, 0.f, 0.f, 0.f, 0.208717f, 0.f, 0.f, 0.f, 0.162998f, 0.480589f, 0.f, 0.f, 0.364283f, 0.f, 
    0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 2.092508f, 0.f, 0.019416f, 0.f, 1.491065f, 0.f, 0.f, 0.760421f, 0.547392f, 
    0.756998f, 0.f, 1.031112f, 0.320258f, 0.900643f, 0.f, 0.f, 1.671057f, 0.f, 2.049336f, 3.126048f, 0.067327f, 0.f, 
    0.f, 0.f, 0.f, 0.0693f, 0.f, 0.286158f, 0.590411f, 0.f, 0.f, 4.010118f, 0.f, 0.f, 0.424088f, 0.f, 0.f, 1.634596f, 
    0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.718636f, 1.188677f, 0.f, 0.f, 0.f, 0.658715f, 0.f, 0.244679f, 0.456688f, 
    3.896796f, 7.324594f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.012536f, 0.f, 0.f, 3.366194f, 0.f, 0.f, 0.f, 0.975371f, 0.f, 
    0.486165f, 0.883176f, 3.892157f, 0.57055f, 0.f, 0.f, 0.f, 0.173385f, 0.61111f, 0.f, 0.227229f, 0.f, 5.307415f, 
    0.07667f, 3.11912f, 2.307226f, 0.f, 0.389976f, 1.998013f, 0.605408f, 0.f, 0.f, 0.f, 0.f, 0.970468f, 0.f, 0.f, 
    0.86588f, 0.f, 0.f, 0.297465f, 0.442886f, 0.f, 0.426823f, 0.314596f, 0.849621f, 0.f, 0.285587f, 0.093924f, 0.f, 0.f
    , 0.f, 0.57641f, 0.571771f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.138237f, 0.f, 0.f, 0.f, 0.698061f, 0.f, 0.279345f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.414647f, 0.f, 0.460943f, 0.f, 0.113113f, 0.f, 0.f, 0.176507f, 0.f, 0.f, 0.f, 0.181688f,
     0.f, 0.894726f, 0.f, 0.f, 0.156238f, 0.f, 0.35252f, 1.205521f, 0.f, 0.718973f, 0.253731f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.141807f, 0.374454f, 0.f, 0.f, 2.756646f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.658572f, 0.669675f, 0.f, 0.753628f, 0.f
    , 0.f, 0.043999f, 0.f, 3.659676f, 3.841667f, 0.48909f, 0.774836f, 0.f, 0.f, 0.f, 1.902702f, 0.f, 1.107323f, 
    0.897029f, 0.f, 1.658566f, 0.589323f, 0.094615f, 0.f, 0.f, 0.f, 1.110364f, 1.891054f, 0.f, 0.f, 0.f, 0.398777f, 0.f
    , 1.049992f, 0.462992f, 1.233619f, 0.f, 0.326271f, 0.f, 0.f, 0.381586f, 0.679473f, 2.678473f, 0.506054f, 1.469515f,
     6.25795f, 0.167463f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 5.324824f, 0.f, 2.624474f, 0.f, 0.20533f, 0.020327f, 
    0.475209f, 0.700342f, 0.f, 0.f, 0.117357f, 0.f, 0.283634f, 0.593213f, 0.566166f, 0.f, 0.115056f, 0.f, 0.f, 0.f, 
    0.466587f, 2.560954f, 0.539397f, 1.535939f, 1.669516f, 0.215235f, 0.f, 0.f, 0.f, 1.892701f, 0.f, 0.f, 0.f, 
    0.981125f, 0.f, 0.f, 1.393466f, 0.f, 0.f, 0.f, 1.522563f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.117998f, 0.f, 0.f, 1.077776f, 
    0.f, 0.f, 0.924622f, 0.f, 0.567549f, 0.f, 0.f, 0.f, 0.120779f, 0.f, 0.f, 0.f, 0.445954f, 0.406511f, 0.01424f, 0.f, 
    2.195706f, 0.f, 0.f, 1.156821f, 0.195036f, 1.558148f, 0.6547f, 0.49833f, 0.05523f, 0.802931f, 0.f, 0.f, 0.f, 
    0.141625f, 1.273738f, 1.677557f, 0.269416f, 1.817225f, 0.f, 0.502503f, 0.f, 1.799565f, 0.580707f, 0.f, 0.f, 
    0.128724f, 0.f, 1.924314f, 0.f, 0.754534f, 0.067696f, 0.f, 1.506513f, 2.497983f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.325121f, 0.330374f, 0.f, 1.33614f, 1.492716f, 0.f, 1.342819f, 3.218542f, 0.f, 0.868488f, 4.386188f, 0.263232f, 
    0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.876079f, 0.001184f, 0.f, 0.f, 0.224076f, 0.f, 1.808878f, 0.630456f, 0.f, 
    2.769839f, 0.f, 0.f, 0.f, 0.011746f, 0.019461f, 0.661962f, 0.621815f, 0.f, 1.759813f, 0.f, 1.363267f, 3.051361f, 
    0.736795f, 1.86696f, 2.90574f, 0.593383f, 0.f, 0.f, 0.f, 0.516191f, 1.514394f, 0.f, 0.f, 2.489074f, 0.f, 0.f, 
    0.374156f, 0.f, 0.f, 1.55346f, 0.189265f, 0.415893f, 0.605659f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.679582f, 0.331438f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.874765f, 0.598331f, 0.f, 0.f, 
    0.f, 1.318401f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 2.88294f, 0.225704f, 0.730519f, 0.f, 0.017577f, 0.075078f, 0.f, 
    0.f, 0.f, 0.f, 0.f, 0.169732f, 0.57365f, 0.f, 0.f, 0.f, 0.284928f, 0.36717f, 0.029338f, 0.879405f, 1.099316f, 0.f, 
    4.250101f, 0.392174f, 3.609787f, 0.070906f, 0.f, 4.86571f, 5.519163f, 0.881549f, 0.f, 5.213828f, 0.f, 0.f, 
    1.854637f, 0.f, 2.387873f, 4.295587f, 0.417242f, 0.f, 1.30092f, 0.f, 0.092692f, 0.f, 0.f, 3.411382f, 3.016245f, 0.f
    , 0.f, 0.f, 0.032883f, 0.f, 2.074305f, 0.618627f, 3.19344f, 3.093224f, 0.914504f, 0.087862f, 2.689775f, 0.f, 
    1.792185f, 5.404583f, 0.f, 0.f, 4.004668f, 0.f, 0.f, 0.144006f, 1.467646f, 0.f, 0.095475f, 0.368385f, 4.481642f, 
    3.413462f, 0.13236f, 0.99875f, 0.f, 0.070708f, 1.427507f, 2.371948f, 0.f, 0.f, 4.180378f, 0.f, 0.628137f, 2.57987f,
     0.473402f, 1.278843f, 1.304523f, 0.046061f, 0.f, 0.f, 0.f, 0.194725f, 2.961133f, 0.274531f, 3.337937f, 4.541351f, 
    0.635391f, 0.f, 0.175009f, 0.606972f, 2.219942f, 0.107168f, 0.f, 1.208021f, 0.215219f, 0.89806f, 0.f, 0.f, 0.f, 
    0.533943f, 0.939043f, 0.747017f, 1.25312f, 1.061651f, 0.587364f, 0.f, 0.f
};

static const float test_2_out_scale = 0.0287238974f;
static const float test_2_out_zero_point = 3.6622970104f;
static const int8_t test_2_out_scales_frac[] = {20};
static const int test_2_out_sa_dim = -1;

static const int test_2_out_fx8_frac = 4;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {16, 12, 3}, \
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
    0.463167f, -1.f, 1.f, 0.79043f, -1.f, -1.f, 0.677935f, -1.f, 1.f, 0.665595f, -1.f, -1.f, 1.f, -1.f, -1.f, 0.818153f
    , -0.440296f, 1.f, -0.654155f, -0.837459f, -1.f, -1.f, 1.f, -1.f, -1.f, -1.f, -0.177336f, 0.663107f, -1.f, 1.f, 
    0.088687f, -1.f, 1.f, -0.205148f, -1.f, -0.628634f, -1.f, -1.f, 1.f, 0.133401f, -0.786462f, -1.f, -1.f, -1.f, 1.f, 
    1.f, -1.f, -1.f, -1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, -1.f, -0.780275f, -1.f, 1.f, 1.f, -1.f, -1.f, 1.f, -1.f, 
    -1.f, 0.431204f, -1.f, -1.f, -1.f, 1.f, -1.f, 1.f, -1.f, -1.f, 1.f, -0.913491f, -1.f, 1.f, -1.f, -1.f, 1.f, -1.f, 
    -1.f, 1.f, -0.983698f, -1.f, -1.f, -1.f, 1.f, -1.f, -0.498285f, 1.f, 1.f, -0.963813f, -1.f, 1.f, 0.002756f, -1.f, 
    1.f, -1.f, -0.315328f, 1.f, -1.f, -1.f, -0.322612f, -0.526929f, -1.f, 1.f, 1.f, -1.f, -1.f, 1.f, -1.f, 1.f, 1.f, 
    -1.f, 0.454239f, 1.f, 0.282633f, 1.f, -1.f, -1.f, 1.f, -1.f, -1.f, 1.f
};

static const float test_3_out_scale = 0.0078431377f;
static const float test_3_out_zero_point = 0.f;
static const int8_t test_3_out_scales_frac[] = {21};
static const int test_3_out_sa_dim = -1;

static const int test_3_out_fx8_frac = 7;

#define TEST_3_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {6, 7, 3}, \
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
    0.f, 0.f, 0.f, 1.014343f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.480589f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
    0.776924f, 0.608372f, 0.f, 0.f, 0.760421f, 0.547392f, 0.756998f, 0.f, 1.031112f, 0.320258f, 0.900643f, 0.f, 0.f, 
    1.671057f, 0.186532f, 2.648945f, 1.378837f, 0.f, 0.f, 0.0693f, 0.f, 0.f, 4.324313f, 0.f, 0.f, 0.424088f, 0.f, 0.f, 
    0.f, 0.f, 0.f, 0.f, 0.f, 1.718636f, 1.188677f, 0.f, 0.f, 0.f, 0.658715f, 0.f, 0.244679f, 0.457681f, 4.05972f, 6.f, 
    0.f, 0.f, 0.012536f, 0.f, 0.f, 2.833407f, 0.975371f, 0.f, 0.486165f, 0.529645f, 3.378214f, 0.f, 0.173385f, 0.61111f
    , 0.f, 0.227229f, 0.f, 5.307415f, 1.364041f, 1.683594f, 0.f, 0.f, 1.811919f, 1.412189f, 0.557993f, 0.f, 0.f, 0.f, 
    0.f, 1.004311f, 0.f, 0.f, 0.f, 0.640238f, 0.f, 0.f, 0.610194f, 0.f, 0.f, 0.877143f, 0.f, 0.f, 1.026173f, 0.395742f,
    0.f, 0.253731f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 2.854884f, 0.f, 0.f, 0.f, 0.281773f, 0.f, 1.135984f, 0.f, 0.f, 
    0.043999f, 0.f, 5.158394f, 0.f, 0.f, 0.f, 1.902702f, 0.f, 1.107323f, 0.897029f, 0.f, 1.658566f, 0.589323f, 
    0.094615f, 0.f, 0.f, 0.f, 0.773731f, 1.309071f, 0.398777f, 0.f, 1.049992f, 0.938397f, 0.820491f, 0.f, 0.381586f, 
    0.679473f, 2.678473f, 0.822651f, 1.524691f, 4.130105f, 0.f, 0.f, 0.f, 0.f, 0.f, 5.324824f, 0.f, 1.903447f, 0.f, 
    1.877028f, 0.369143f, 0.f, 0.863724f, 0.f, 0.f, 1.820124f, 0.f, 0.f, 0.282092f, 0.f, 0.91974f, 0.80931f, 0.f, 
    0.317004f, 0.f, 1.454752f, 0.f, 0.f, 0.f, 1.861731f, 0.186475f, 1.999347f, 1.771951f, 0.49833f, 0.05523f, 0.802931f
    , 0.f, 0.f, 0.f, 0.560175f, 3.814845f, 0.f, 0.502503f, 0.f, 1.799565f, 0.858565f, 0.f, 1.532303f, 0.f, 0.754534f, 
    0.067696f, 0.f, 1.481408f, 0.087694f, 0.f, 0.f, 0.f, 0.f, 0.325121f, 0.330374f, 0.f, 1.33614f, 1.492716f, 0.f, 
    1.342819f, 3.218542f, 0.378859f, 1.240347f, 3.815045f, 0.f, 0.f, 0.f, 0.f, 0.f, 2.677526f, 0.224076f, 0.f, 
    1.808878f, 0.657102f, 0.f, 0.f, 0.011746f, 0.019461f, 0.661962f, 0.621815f, 0.f, 1.759813f, 0.f, 0.390441f, 0.f, 
    0.7093f, 1.291295f, 0.786553f, 0.f, 0.f, 0.f, 1.184681f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
    0.55565f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.169732f, 0.57365f, 0.f, 0.f, 0.f, 0.284928f, 1.61562f, 0.360112f, 4.853291f, 
    0.392174f, 3.609787f, 0.070906f, 0.f, 3.78398f, 6.f, 0.f, 0.f, 1.854637f, 0.f, 1.709325f, 5.320291f, 0.f, 0.092692f
    , 0.f, 0.f, 3.411382f, 3.016245f, 0.f, 0.f, 0.f, 0.032883f, 0.f, 2.074305f, 1.682265f, 4.005184f, 5.506783f, 0.f, 
    1.792185f, 5.404583f, 0.f, 0.f, 3.872458f, 1.467646f, 0.f, 0.095475f, 0.649879f, 6.f, 2.668502f, 0.070708f, 
    1.427507f, 2.371948f, 0.f, 0.f, 4.180378f, 0.f, 0.628137f, 2.57987f, 0.473402f, 1.278843f, 1.304523f, 0.f, 0.f, 0.f
    , 0.274531f, 3.337937f, 4.541351f, 1.391497f, 0.f, 0.005962f, 0.f, 1.208021f, 0.215219f, 0.56626f, 1.111132f, 0.f, 
    0.747017f, 1.25312f, 1.061651f, 0.587364f, 0.f, 0.f
};

static const float test_4_out_scale = 0.0235294122f;
static const float test_4_out_zero_point = 3.f;
static const int8_t test_4_out_scales_frac[] = {20};
static const int test_4_out_sa_dim = -1;

static const int test_4_out_fx8_frac = 4;

#define TEST_4_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {11*5*1, 5*1, 1}, \
    /* .shape = */ {13, 9, 3}, \
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
    0.182443f, -5.460881f, -1.241028f, 1.255288f, -4.978712f, -0.382022f, 0.132413f, -2.471303f, -3.133288f, -0.543137f
    , -2.901602f, 2.217625f, -0.462109f, -2.947853f, -1.469382f, 2.066833f, 0.634305f, -0.634317f, 0.791142f, 
    -3.922265f, 2.442199f, 1.260735f, -0.549783f, -0.822284f, -1.587682f, -0.617362f, 0.574562f, -0.945824f, 0.47555f, 
    1.883455f, -2.297307f, 0.050218f, 6.073983f, -0.077768f, 0.222888f, -1.983227f, -1.680174f, -1.616613f, -1.929004f,
    -0.474543f, 0.725694f, 3.093736f, -1.779776f, -3.695127f, 7.047446f, -1.753271f, -1.819515f, 1.226207f, 1.930053f,
    -3.566968f, -2.09146f, 1.034727f, 1.192267f, 8.293514f, -0.676529f, -5.796121f, 0.433521f, -0.422057f, -2.330728f,
    5.53787f, 0.996022f, -4.965669f, -1.631708f, 1.65142f, 0.233911f, 2.885056f, 1.349058f, -4.314198f, -6.08424f, 
    0.124025f, 1.395203f, 4.779737f, 0.252832f, -2.640762f, -1.53924f, 0.202084f, -2.509716f, 3.03404f, -0.19327f, 
    -0.292015f, 0.216387f, 0.382349f, 0.079379f, 0.265216f, -0.03342f, -5.309621f, 2.380201f, 3.354701f, -1.541728f, 
    -1.778474f, 0.382658f, -4.057364f, -4.378407f, 0.885972f, -4.294459f, 2.069259f, -0.569206f, -0.263289f, -2.596092f,
    0.504312f, 0.914679f, 3.157492f, -1.747266f, -1.673466f, 2.854959f, -1.521472f, 2.686754f, -1.344923f, -1.940461f,
    -3.725044f, 2.379254f, -1.356561f, -0.557641f, -0.363773f, -1.436045f, -0.293904f, 3.56606f, -1.394564f, 
    -0.786934f, 2.210115f, 0.337984f, -0.712302f, -3.122503f, -0.270711f, -3.39138f, 5.429394f, 0.735647f, -3.456115f, 
    2.656832f, 1.149093f, -2.278186f, 1.669154f, 1.861715f, -4.297037f, -4.439114f, 0.175686f, 1.773576f, 5.85935f, 
    -0.437591f, -5.617468f, 0.909389f, -0.944263f, -0.953722f, 7.244059f, -0.56492f, -1.557382f, 1.843335f, 0.121979f, 
    -4.708191f, -3.219017f, -0.542358f, -1.531196f, -7.401711f, -0.875276f, -5.773212f, 0.932065f, 0.582538f, -3.518437f
    , -1.489849f, 0.750233f, -1.274252f, -0.642432f, 0.621964f, -3.059104f, 0.178039f, 1.587258f, 2.301893f, -0.452229f
    , -1.06555f, -0.007848f, -0.571216f, -0.553233f, -0.788939f, -1.645887f, -0.758075f, 0.34902f, 2.006872f, -0.95722f
    , 1.018669f, -0.399273f, -2.136623f, -0.766078f, -1.71823f, -0.121529f, 0.437723f, 3.153871f, -1.348384f, 
    -3.322972f, 0.803526f, -1.227272f, -2.042775f, 0.906853f, 0.289063f, -0.459471f, -1.425286f, -0.054811f, -1.530056f,
    2.063652f, -0.70005f, -2.138505f, 4.576154f, -0.331649f, -3.666306f, 4.784592f, 0.843284f, -4.260267f, 2.542039f, 
    1.119949f, 1.757097f, 2.218418f, 0.666655f, -4.20837f, -3.8506f, -0.077349f, 1.28692f, 0.419073f, 0.170514f, 
    -2.738399f, -0.853313f, -0.983263f, -2.676111f, -0.185623f, -1.543647f, -2.90506f, 5.451266f, -1.179893f, -2.055756f
    , 2.991981f, -1.375142f, -5.392591f, 0.407243f, 1.167386f, -0.27858f, 2.480781f, 0.296208f, -3.708318f, 5.143426f, 
    0.447657f, -1.068954f, 0.61026f, -1.261297f, -0.742263f, -1.410058f, -1.593839f, 4.436145f, 0.206275f, -2.26141f, 
    -2.955983f, 2.811399f, 0.557433f, 2.015875f, -5.551959f, -1.41633f, -2.543018f, -3.716955f, -1.168153f, 2.690913f, 
    3.494143f, -2.235671f, -1.975619f, 6.15405f, -0.46963f, -1.004044f, 1.5235f, 1.230741f, -1.613683f, -2.28937f, 
    2.029547f, -1.552357f, 2.345528f, 0.801853f, -4.339289f, 2.772879f, -0.499599f, -1.745083f, 4.529621f, 2.180319f, 
    -6.485736f, -0.861432f, 1.895524f, 2.005064f, 6.141337f, 0.214224f, -5.282022f, -4.112389f, -0.481068f, -0.265676f, 
    3.628133f
};

static const float test_5_out_scale = 0.0615499057f;
static const float test_5_out_zero_point = 0.4459013939f;
static const int8_t test_5_out_scales_frac[] = {19};
static const int test_5_out_sa_dim = -1;

static const int test_5_out_fx8_frac = 3;

#define TEST_5_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {9*8*1, 8*1, 1}, \
    /* .shape = */ {12, 8, 3}, \
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
    0.109683f, 0.f, 0.457518f, 0.292227f, 0.f, 0.f, 0.f, 0.f, 1.595493f, 6.f, 1.155381f, 4.056403f, 0.697891f, 0.f, 
    0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.009259f, 0.f, 1.50331f, 0.f, 0.f, 0.391764f, 0.f, 0.f, 0.301781f, 6.f, 0.279297f, 
    0.726381f, 0.f, 3.382336f, 4.001736f, 0.f, 0.f, 1.516286f, 0.f, 0.f, 0.576868f, 1.669597f, 0.297368f, 0.f, 
    0.260843f, 0.815974f, 0.f, 0.f, 0.f, 3.917051f, 1.571339f, 0.725238f, 2.551107f, 0.f, 0.f, 0.054169f, 0.f, 0.f, 
    0.f, 3.227523f, 0.024772f, 3.790867f, 0.f, 0.f, 6.f, 0.f, 3.773921f, 0.f, 3.2233f, 0.f, 0.f, 0.5202f, 0.f, 
    1.897129f, 6.f, 0.577519f, 3.987985f, 0.825489f, 0.f, 0.f, 0.f, 2.236537f, 0.f, 1.931621f, 6.f, 0.f, 2.154737f, 
    3.489267f, 0.f, 0.f, 0.f, 2.162251f, 0.f, 0.f, 6.f, 0.f, 2.031881f, 0.f, 0.f, 0.f, 0.133293f, 0.178428f, 3.344342f, 
    1.345227f, 0.f, 0.108747f, 1.447427f, 3.710516f, 1.922274f, 0.f, 0.f, 1.195964f, 0.f, 0.f, 5.308278f, 0.f, 0.f, 
    0.218493f, 0.f, 0.f, 0.09175f, 2.745521f, 1.31784f, 0.130206f, 6.f, 0.f, 0.f, 3.298417f, 0.f, 3.591253f, 0.f, 
    1.796774f, 0.f, 1.744599f, 0.f, 0.193461f, 2.678981f, 0.f, 1.182585f, 0.f, 0.506508f, 4.50901f, 2.380171f, 0.f, 
    0.f, 0.070061f, 2.482965f, 0.f, 0.f, 0.f, 0.f, 0.585651f, 0.f, 0.f, 6.f, 0.009773f, 0.498412f, 4.462146f, 0.f, 0.f, 
    0.f, 1.685122f, 3.222003f, 0.f, 0.f, 0.345772f, 4.111879f, 0.f, 1.299842f, 6.f, 0.f, 0.423892f, 0.182416f, 6.f, 
    0.f, 0.f, 1.270145f, 0.f, 3.334949f, 0.f, 1.611233f, 1.287335f, 0.436375f, 0.f, 0.351639f, 0.715957f, 0.f, 0.f, 
    0.f, 1.208663f, 0.f, 1.282022f, 0.f, 0.f, 5.092241f, 0.f, 1.700527f, 4.098455f, 0.175568f, 0.f, 0.f, 0.f, 0.f, 0.f, 
    6.f, 0.f, 0.f, 2.423279f, 0.790603f, 4.158094f, 1.314851f, 1.887146f, 0.f, 1.207476f, 5.197121f, 0.f, 0.f, 
    0.296084f, 0.f, 0.515182f, 0.473713f, 0.f, 0.f, 3.844239f, 0.f, 2.171844f, 0.061549f, 0.f, 0.f, 0.f, 1.036042f, 
    0.386492f, 0.f, 0.f, 0.93423f, 0.f, 0.f, 0.f, 0.f, 2.071953f, 0.f, 0.12535f, 1.452377f, 5.328173f, 1.041893f, 0.f, 
    4.784076f, 1.734066f, 1.057033f, 0.f, 0.f, 2.787976f, 0.f, 0.f, 6.f, 0.f, 0.695138f, 0.227793f, 0.f, 0.f, 
    2.211977f, 0.f, 0.f, 4.70696f, 0.f, 1.672544f, 0.f, 0.f, 0.326651f, 6.f, 0.f, 0.f, 0.f, 1.65157f, 0.f, 1.488527f, 
    2.722763f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 2.620761f, 0.f, 0.784203f, 2.886619f, 0.f, 0.f, 1.019799f, 0.268447f, 
    3.472416f, 0.f, 0.f, 0.773975f, 1.537597f, 0.f, 1.774826f, 6.f, 0.508791f, 1.505414f, 0.f, 0.f, 6.f, 0.593139f, 
    0.f, 0.f, 0.f, 0.f, 0.291676f, 2.061306f, 0.f, 0.f, 3.759059f, 1.181035f, 0.f, 0.f, 0.f, 5.57907f, 0.f, 0.f, 0.f, 
    1.298159f, 1.979551f, 0.237584f, 0.f, 0.f, 0.588316f, 6.f, 0.00817f, 1.095521f, 0.641961f, 0.f, 0.f, 0.797703f, 
    0.f, 0.f, 1.029973f, 6.f, 0.f, 2.323977f, 2.12675f, 0.f, 0.f, 0.f, 4.247802f, 0.f, 2.477363f, 2.222142f, 0.613596f, 
    0.f, 0.f, 1.278385f, 0.f, 0.658708f, 0.f, 0.f, 0.231149f, 1.382964f, 0.f, 2.873325f, 0.f, 0.f, 6.f, 0.f, 0.393463f, 
    0.f, 0.f, 0.f, 0.f, 1.749029f, 0.f, 0.f, 1.567543f, 1.227356f, 2.854445f, 0.f, 0.f, 0.f, 0.f, 1.353055f, 0.f, 0.f, 
    6.f, 1.49196f, 1.174797f, 0.f, 0.f, 0.f, 0.f, 0.122226f, 0.424614f, 0.f, 0.f, 0.f, 0.499778f, 0.f, 1.737761f, 0.f, 
    0.432792f, 0.834636f, 0.f, 4.676056f, 6.f, 2.131729f, 0.f, 0.f, 0.f, 6.f, 0.f, 0.826343f, 0.f, 0.f, 3.418039f, 0.f, 
    0.795031f, 0.f, 0.f, 0.f, 0.106193f, 0.f, 0.f, 0.256952f, 6.f, 0.029383f, 0.f, 2.959972f, 0.f, 0.f, 0.588003f, 
    2.054121f, 0.f, 0.f, 1.800437f, 0.521175f, 0.f, 1.084507f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.440775f, 0.f, 6.f, 
    0.f, 0.503795f, 0.f, 0.f, 0.f, 0.55181f, 0.f, 0.f, 0.f, 0.059078f, 0.f, 3.280165f, 0.f, 3.852277f, 0.356285f, 0.f, 
    5.854979f, 0.f, 3.096708f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.762485f, 0.f, 1.43591f, 2.823121f, 6.f, 
    0.767548f, 0.f, 0.f, 6.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.116515f, 0.044869f, 2.909683f, 1.81416f, 2.436089f, 
    5.103088f, 0.8314f, 0.f, 0.f, 0.f, 1.071154f, 0.f, 5.378685f, 6.f, 2.945085f, 3.071891f, 2.660661f, 0.f, 0.374578f, 
    2.049059f, 0.f, 0.f, 6.f, 0.f, 3.071057f, 0.f, 2.396651f, 0.f, 6.f, 0.f, 0.f, 0.f, 0.511811f, 4.479856f, 0.f, 
    2.922506f, 1.887252f, 0.f, 0.f, 0.f, 0.827982f, 0.f, 0.f, 3.77871f, 0.f, 2.667964f, 0.f, 2.002911f, 0.f, 0.f, 0.f, 
    0.f, 2.322087f, 1.437781f, 2.244118f, 0.569552f, 0.f, 0.f, 4.785712f, 0.f, 1.018603f, 3.555861f, 2.987246f, 6.f, 
    0.f, 5.126541f, 5.153133f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.719448f, 0.f, 2.502758f, 0.f, 4.955278f, 0.f, 
    1.787752f, 0.f, 0.f, 1.913436f, 0.f, 1.66003f, 1.66162f, 0.048368f, 2.804704f, 0.515052f, 0.f, 0.f, 0.f, 5.209158f, 
    1.213798f, 0.f, 0.f, 0.f, 6.f, 0.f, 0.f, 0.92065f, 0.f, 0.f, 0.199136f, 0.f, 2.081343f, 0.f, 6.f, 0.737126f, 0.f, 
    0.f, 0.f, 0.f, 1.399849f, 0.f, 0.f, 0.f, 6.f, 0.120651f, 0.f, 0.934061f, 5.962024f, 0.f, 0.f, 0.364249f, 0.f, 0.f, 
    0.f, 0.f, 0.f, 2.038712f, 6.f, 0.f, 0.f, 1.623705f, 1.42434f, 0.f, 5.957133f, 1.098408f, 0.19525f, 0.f, 2.462419f, 
    0.559546f, 2.951918f, 0.f, 0.f, 0.f, 6.f, 0.f, 0.f, 0.f, 3.54642f, 0.403911f, 1.360778f, 0.602278f, 0.f, 0.f, 0.f, 
    0.f, 1.343728f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.771486f, 0.145112f, 0.f, 0.f, 0.f, 1.096589f, 0.f, 0.f, 
    4.245515f, 0.f, 0.f, 0.409818f, 1.495909f, 1.024328f, 0.f, 1.267362f, 0.815871f, 0.f, 0.f, 0.f, 6.f, 0.097905f, 
    0.084664f, 5.169346f, 0.f, 0.f, 0.f, 0.f, 0.621962f, 6.f, 0.608554f, 0.f, 0.f, 2.397951f, 0.63425f, 0.f, 0.f, 0.f, 
    4.822704f, 2.213126f, 0.f, 1.191506f, 2.577979f, 0.f, 0.f, 3.324433f, 1.154811f, 0.f, 0.f, 0.f, 0.f, 0.758051f, 
    0.f, 2.367014f, 0.115854f, 0.f, 0.134193f, 0.f, 0.f, 0.f, 6.f, 1.645228f, 2.74406f, 0.632816f, 1.296696f, 0.f, 0.f, 
    0.f, 0.f, 0.f, 0.f, 0.f, 3.358515f, 0.f, 0.f, 6.f, 0.474522f, 2.753243f, 0.f, 2.339234f, 0.f, 0.f, 0.55918f, 
    0.912825f, 0.f, 0.f, 0.f, 2.581152f, 0.f, 0.249537f, 6.f, 0.f, 2.09954f, 0.803202f, 0.f, 0.f, 0.109034f, 3.082171f, 
    1.102823f, 1.941421f, 0.f, 0.346354f, 0.621936f, 0.707661f, 1.340015f, 0.f, 0.f, 0.f, 0.f, 1.901561f, 0.848242f, 
    0.f, 2.611092f, 0.f, 1.033275f, 6.f, 2.228781f, 2.448407f, 0.f, 0.f, 0.f, 0.f, 1.288306f, 1.609535f, 0.f, 6.f, 0.f, 
    0.f, 2.19005f, 0.f, 2.300975f, 0.f, 3.268804f, 0.076487f, 2.346999f, 6.f, 0.f, 4.643297f, 2.191251f, 0.f, 0.f, 0.f, 
    0.968446f, 0.f, 0.825108f, 0.f, 1.786297f, 1.13819f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.705121f, 0.f, 0.f, 0.f, 
    0.f, 0.f, 2.521783f, 0.f, 0.f, 5.254272f, 0.f, 1.329757f, 0.f, 0.f, 0.f, 6.f, 0.f, 1.314329f, 0.f, 2.339489f, 6.f, 
    0.00742f, 3.985039f, 0.24108f, 0.f, 0.f
};

static const float test_6_out_scale = 0.0235294122f;
static const float test_6_out_zero_point = 3.f;
static const int8_t test_6_out_scales_frac[] = {20};
static const int test_6_out_sa_dim = -1;

static const int test_6_out_fx8_frac = 4;

#define TEST_6_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {16*8*1, 8*1, 1}, \
    /* .shape = */ {12, 14, 5}, \
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
    1.612303f, 0.f, 3.682693f, 0.229883f, 5.027158f, 0.154888f, 0.f, 0.f, 5.575845f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
    0.f, 0.f, 3.877062f, 0.86577f, 2.397208f, 0.f, 6.f, 0.f, 6.f, 0.262792f, 0.f, 0.f, 6.f, 0.f, 0.f, 0.048185f, 
    0.275913f, 1.012161f, 3.303212f, 0.186678f, 0.f, 5.105071f, 0.551304f, 0.f, 1.838547f, 0.f, 3.470774f, 0.f, 
    4.90499f, 0.f, 0.f, 3.209132f, 0.582248f, 0.f, 0.739024f, 0.f, 0.f, 0.f, 0.f, 2.993297f, 0.f, 0.f, 3.644969f, 
    1.002667f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.666512f, 0.314131f, 0.f, 0.f, 0.879693f, 0.f, 5.311986f, 4.507118f, 6.f, 
    0.f, 0.f, 3.441069f, 0.f, 6.f, 0.611324f, 0.f, 0.f, 0.f, 6.f, 0.880783f, 1.682268f, 0.f, 0.f, 0.935548f, 0.f, 0.f, 
    6.f, 4.367838f, 6.f, 2.354864f, 0.f, 6.f, 1.208347f, 6.f, 0.447068f, 0.f, 0.f, 0.f, 0.f, 2.594644f, 0.020454f, 6.f, 
    0.989673f, 0.f, 0.f, 0.f, 3.185384f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 2.934988f, 1.318326f, 6.f, 6.f, 6.f, 
    5.642881f, 0.f, 0.f, 0.f, 0.f, 0.f, 2.999222f, 0.f, 0.f, 0.f, 2.559106f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.633486f, 0.f, 
    0.f, 0.f, 1.628865f, 0.f, 2.338941f, 6.f, 6.f, 0.f, 0.f, 0.f, 0.f, 6.f, 1.945178f, 0.f, 6.f, 0.f, 0.f, 0.f, 
    5.543451f, 0.f, 0.f, 0.f, 0.f, 0.406046f, 0.f, 0.f, 6.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
    0.856572f, 6.f, 1.313051f, 5.714135f, 2.297852f, 0.f, 3.328464f, 5.076805f, 0.f, 0.f, 0.822006f, 0.f, 0.f, 0.f, 
    3.011547f, 5.216634f, 0.f, 6.f, 4.519185f, 0.f, 0.f, 0.f, 1.309921f, 0.f, 0.f, 0.f, 6.f, 4.929738f, 2.19647f, 
    0.862602f, 0.f, 1.323422f, 6.f, 6.f, 2.799471f, 1.364977f, 0.f, 0.f, 6.f, 0.f, 0.886486f, 0.f, 0.f, 0.f, 0.f, 0.f, 
    0.f, 6.f, 0.f, 1.915524f, 0.f, 6.f, 0.471624f, 6.f, 0.f, 0.f, 0.f, 0.f, 6.f, 1.930647f, 6.f, 0.f, 0.f, 2.613482f, 
    0.f, 1.437104f, 0.f, 0.f, 0.f, 0.f, 1.100758f, 6.f, 3.722325f, 0.f, 0.f, 0.f, 2.4696f, 0.99466f, 6.f, 0.f, 0.f, 
    0.f, 0.f, 6.f, 0.f, 2.601814f, 0.f, 6.f, 6.f, 0.849792f, 0.f, 0.f, 0.f, 0.f, 0.024752f, 1.727437f, 4.829096f, 6.f, 
    2.193951f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.381432f, 0.f, 0.f, 2.373422f, 0.f, 0.f, 0.f, 0.f, 0.967325f, 0.f, 
    0.f, 6.f, 0.f, 0.f, 3.20644f, 0.406951f, 5.442923f, 6.f, 1.564964f, 0.f, 6.f, 0.97099f, 0.f, 3.344471f, 0.f, 0.f, 
    0.f, 0.f, 0.f, 0.f, 6.f, 0.f, 6.f, 0.f, 2.905131f, 6.f, 0.f, 6.f, 0.f, 0.f, 0.f, 6.f, 0.f, 0.297123f, 0.f, 0.f, 
    6.f, 0.f, 4.127302f, 0.232429f, 0.f, 0.f, 1.49025f, 0.f, 0.f, 0.f, 0.f, 6.f, 0.f, 3.450019f, 0.f, 0.f, 0.f, 0.f, 
    0.f, 0.f, 0.f, 0.f, 0.f, 5.865136f, 0.f, 0.f, 1.02392f, 2.168701f, 0.f, 6.f, 0.f, 6.f, 0.f, 6.f, 0.f, 0.f, 
    5.276888f, 2.871476f, 5.381645f, 0.f, 0.f, 4.970076f, 2.17714f, 0.236733f, 0.f, 4.765553f, 0.f, 0.f, 0.f, 6.f, 0.f, 
    6.f, 1.803472f, 3.319622f, 0.f, 0.979825f, 6.f, 2.748632f, 0.819249f, 6.f, 0.f, 4.124219f, 0.f, 0.f, 3.476707f, 
    0.f, 6.f, 0.f, 2.079974f, 0.f, 0.f, 6.f, 3.552917f, 4.753067f, 0.f, 0.f, 6.f, 1.615141f, 0.f, 5.257527f, 0.f, 0.f, 
    1.106647f, 2.306467f, 0.f, 0.f, 1.918906f, 0.323499f, 0.f, 1.10438f, 6.f, 6.f, 1.098096f, 0.f, 2.193317f, 6.f, 0.f, 
    2.188545f, 0.f, 3.010049f, 0.f, 6.f, 4.013948f, 0.f, 2.90343f, 6.f, 6.f, 1.907748f, 0.f, 1.458995f, 0.f, 6.f, 
    0.775694f, 2.998086f, 0.f, 5.264673f, 6.f, 0.133925f, 0.f, 1.462426f, 0.f, 6.f, 1.401253f, 0.f, 2.551511f, 0.f, 
    0.f, 0.505411f, 5.560861f, 1.462368f, 0.f, 6.f, 0.f, 0.f, 0.f, 2.028553f, 3.725128f, 0.f, 2.617194f, 0.f, 
    4.787561f, 6.f, 0.025402f, 2.977197f, 0.f, 6.f, 0.f, 1.265011f, 0.84491f, 0.f, 1.356137f, 2.945519f, 0.07234f, 0.f, 
    5.740277f, 6.f, 0.503783f, 0.f, 0.f, 0.f, 0.701316f, 4.874044f, 0.f, 0.295278f, 0.f, 0.f, 6.f, 0.f, 0.f, 5.072755f, 
    6.f, 6.f, 2.348612f, 2.890643f, 0.f, 0.f, 6.f, 0.f, 0.f, 6.f, 0.532404f, 6.f, 0.f, 6.f, 0.f, 0.f, 6.f, 0.f, 0.f, 
    0.f, 1.206985f, 6.f, 0.910795f, 3.605764f, 6.f, 0.f, 4.756282f, 0.f, 0.f, 6.f, 6.f, 6.f, 0.f, 0.785079f, 0.f, 6.f, 
    0.f, 2.458167f, 0.f, 6.f, 6.f, 0.f, 0.f, 3.593995f, 0.f, 0.f, 6.f, 0.f, 0.f, 0.f, 6.f, 1.258649f, 3.150318f, 
    0.476788f, 6.f, 0.f, 5.961527f, 0.f, 0.f, 6.f, 1.805623f, 5.768168f, 2.210758f, 0.f, 0.f, 0.f, 2.739697f, 0.f, 0.f, 
    0.f, 0.f, 6.f, 0.899466f, 0.f, 0.f, 0.613579f, 0.f, 0.f, 3.140562f, 0.f, 6.f, 6.f, 0.f, 0.f, 0.640394f, 0.f, 0.f, 
    0.f, 0.f, 0.f, 5.821669f, 6.f, 0.f, 1.636458f, 3.601084f, 6.f, 0.695965f, 0.f, 0.f, 0.f, 0.f, 6.f, 2.997504f, 0.f, 
    3.043015f, 6.f, 0.f, 0.395517f, 0.f, 5.693463f, 0.f, 6.f, 0.f, 0.f, 2.685422f, 0.f, 5.10919f, 0.f, 0.f, 0.766391f, 
    0.f, 6.f, 0.f, 1.067741f, 0.f, 3.204967f, 0.f, 0.049364f, 0.f, 1.445834f, 0.f, 6.f, 0.f, 6.f, 0.f, 0.f, 6.f, 
    0.56214f, 0.f, 0.f, 0.f, 6.f, 0.f, 6.f, 0.f, 6.f, 6.f, 4.297416f, 0.f, 6.f, 0.f, 0.f, 0.f, 0.f, 6.f, 1.515633f, 
    0.f, 0.962949f, 6.f, 0.f, 0.f, 6.f, 0.f, 3.391435f, 0.f, 5.703098f, 6.f, 0.123188f, 0.74395f, 1.311747f, 0.f, 6.f, 
    0.f, 0.f, 6.f, 0.f, 0.f, 0.f, 2.024087f, 0.f, 0.f, 6.f, 0.f, 0.f, 6.f, 6.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
    0.218827f, 3.539775f, 0.f, 0.f, 6.f, 0.026009f, 0.f, 1.993029f, 0.f, 0.472078f, 1.526834f, 0.f, 0.f, 0.f, 0.f, 
    0.288532f, 0.f, 0.f, 0.f, 6.f, 0.f, 0.f, 0.f, 5.20411f, 2.876878f, 2.40597f, 0.f, 6.f, 0.f, 6.f, 1.409823f, 0.f, 
    0.f, 0.f, 0.f, 0.025228f, 1.600513f, 0.f, 0.f, 6.f, 1.168078f, 0.f, 0.f, 0.f, 0.824512f, 1.234117f, 3.922958f, 0.f, 
    3.501705f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.7285f, 0.f, 6.f, 0.f, 0.f, 0.f, 0.f, 6.f, 0.f, 0.f, 0.f, 0.263715f, 0.f, 0.f, 
    0.f, 0.f, 0.627931f, 0.f, 0.f, 4.004261f, 0.f, 1.692368f, 0.f, 4.079147f, 0.f, 6.f, 0.f, 1.174236f, 0.f, 0.f, 0.f, 
    0.f, 0.f, 6.f, 4.476387f, 6.f, 1.621733f, 0.f, 0.757781f, 0.f, 6.f, 0.f, 2.089746f, 0.f, 0.f, 3.368918f, 4.436606f, 
    0.f, 0.f, 0.f, 6.f, 0.965486f, 0.f, 6.f, 1.450709f, 0.478212f, 0.f, 0.630363f, 0.f, 6.f, 0.f, 2.770687f, 0.f, 6.f, 
    4.487434f, 6.f, 0.f, 0.f, 0.f, 5.13682f, 3.160626f, 2.649455f, 0.f, 0.f, 5.103724f, 3.963688f, 0.004791f, 
    2.462662f, 0.f, 0.f, 6.f, 1.246739f, 1.096148f, 0.534888f, 6.f, 0.f, 0.68324f, 1.37247f, 0.f, 0.f, 2.285978f
};

static const float test_7_out_scale = 0.0235294122f;
static const float test_7_out_zero_point = 3.f;
static const int8_t test_7_out_scales_frac[] = {20};
static const int test_7_out_sa_dim = -1;

static const int test_7_out_fx8_frac = 4;

#define TEST_7_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {16*7*1, 7*1, 1}, \
    /* .shape = */ {12, 14, 5}, \
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
    -0.312073f, -2.104259f, 4.237396f, 0.176413f, -4.830071f, 3.872828f, -1.257123f, 1.113210f, 0.849369f, 0.027653f, 
    -1.796032f, 1.744342f, 0.056987f, -1.008066f, -2.180445f, -0.172698f, -2.327000f, 0.974763f, -0.291775f, -3.667994f,
    0.286258f, 0.086746f, -2.842753f, 2.004754f, -0.711948f, 0.959996f, -4.076325f, -0.151023f, -3.230109f, -0.641899f,
    -2.105740f, 2.232733f, 2.895964f, -0.331767f, -1.306114f, 0.524086f, -1.362217f, -3.114600f, -1.518389f, 0.355776f,
    -0.596459f, -0.220465f, -2.567690f, 0.945127f, 0.468914f, -0.028781f, -0.964155f, 0.865604f, 1.501502f, -1.153668f,
    -1.304271f, -1.243225f, 1.284765f, -3.427740f, 0.472312f, -0.758425f, -9.494400f, -0.462167f, 0.638443f, 0.455736f,
    0.075623f, -3.356467f, 5.081804f, 0.844033f, -5.781651f, 1.273198f, -0.044709f, -4.611673f, 3.766028f, -0.855004f,
    -0.726928f, -0.322268f, -0.748086f, -0.518450f, -1.148292f, -0.610379f, -3.393690f, -0.178071f, -0.891492f, -1.562042f,
    -0.001133f, 0.389083f, 0.215367f, 0.113189f, -0.787688f, -0.677359f, -4.355668f, 0.768722f, -1.867935f, -1.649504f,
    -3.083479f, 3.043895f, 2.019896f, -1.224171f, -4.188751f, 0.690549f, -0.429612f, -0.199854f, 3.518472f, 0.533419f,
    -0.690197f, 0.888332f, -1.056530f, 2.832343f, 1.741581f, 0.118035f, -2.264989f, 0.784811f, 0.318075f, -0.288014f,
    3.783076f, -0.278995f, -1.780685f, 0.830560f, 0.243512f, -4.750223f, -3.653229f, -0.845027f, -0.348321f, -0.280776f,
    0.881041f, -0.863688f, -6.104002f, 0.439807f, -0.883192f, -0.908384f, -0.034352f, 3.042615f, 3.025584f, -0.883616f,
    -2.287746f, 0.711725f, -0.287448f, -2.571106f, -2.370283f, 0.668616f, -0.048195f, -0.662126f, -2.222355f, -2.506699f,
    0.698049f, 0.246063f, -0.114123f, 0.532040f, 1.208862f, 0.397897f, 0.918417f, -0.763991f, 1.164098f, 0.194816f, 1.286364f,
    -0.644931f, 7.787173f, 0.640600f, -0.548635f, 4.177498f, -0.005623f, 1.087955f, 5.720635f, -0.241582f, -4.261491f, 
    1.011570f, -0.885413f, -3.144013f, 7.373176f, 0.636597f, -1.728365f, 1.220859f, -0.199257f, -1.146761f, -2.794411f,
    0.210389f, -1.469495f, -0.521409f, 0.655103f, -1.395990f, -7.156060f, 1.545581f, -0.523282f, -3.404558f, -1.774872f, 
    -4.906569f, -5.075408f, 0.323576f, -2.828363f, -1.496868f, -1.970722f, -0.492529f, -2.152939f, 0.586508f, -1.692890f,
    -1.400761f
};

static const float test_8_out_scale = 0.0677708760f;
static const float test_8_out_zero_point = -0.8536138535f;
static const int8_t test_8_out_scales_frac[] = {18};
static const int test_8_out_sa_dim = -1;

static const int test_8_out_fx8_frac = 3;

#define TEST_8_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {8*5*1, 5*1, 1}, \
    /* .shape = */ {8, 8, 3}, \
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

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data,
                                sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim, test_1_out_data, 
                                sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), 
                                &test_1_out_scale, 1, &test_1_out_zero_point, 1, test_1_out_scales_frac, 1);

tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_fx8_frac + 8, test_2_out_data,
                                sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_sa_dim, test_2_out_data, 
                                sizeof(test_2_out_data) / sizeof(test_2_out_data[0]),
                                &test_2_out_scale, 1, &test_2_out_zero_point, 1, test_2_out_scales_frac, 1);

tensor_quantizer test_3_out_fx16(test_3_out_tsr_fx16, test_3_out_fx8_frac + 8, test_3_out_data,
                                sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));
tensor_quantizer test_3_out_sa8(test_3_out_tsr_sa8, test_3_out_sa_dim, test_3_out_data, 
                                sizeof(test_3_out_data) / sizeof(test_3_out_data[0]),
                                &test_3_out_scale, 1, &test_3_out_zero_point, 1, test_3_out_scales_frac, 1);

tensor_quantizer test_4_out_fx16(test_4_out_tsr_fx16, test_4_out_fx8_frac + 8, test_4_out_data,
                                sizeof(test_4_out_data) / sizeof(test_4_out_data[0]));
tensor_quantizer test_4_out_sa8(test_4_out_tsr_sa8, test_4_out_sa_dim, test_4_out_data, 
                                sizeof(test_4_out_data) / sizeof(test_4_out_data[0]),
                                &test_4_out_scale, 1, &test_4_out_zero_point, 1, test_4_out_scales_frac, 1);

tensor_quantizer test_5_out_fx16(test_5_out_tsr_fx16, test_5_out_fx8_frac + 8, test_5_out_data,
                                sizeof(test_5_out_data) / sizeof(test_5_out_data[0]));
tensor_quantizer test_5_out_sa8(test_5_out_tsr_sa8, test_5_out_sa_dim, test_5_out_data, 
                                sizeof(test_5_out_data) / sizeof(test_5_out_data[0]),
                                &test_5_out_scale, 1, &test_5_out_zero_point, 1, test_5_out_scales_frac, 1);

tensor_quantizer test_6_out_fx16(test_6_out_tsr_fx16, test_6_out_fx8_frac + 8, test_6_out_data,
                                 sizeof(test_6_out_data) / sizeof(test_6_out_data[0]));
tensor_quantizer test_6_out_sa8(test_6_out_tsr_sa8, test_6_out_sa_dim, test_6_out_data, 
                                sizeof(test_6_out_data) / sizeof(test_6_out_data[0]),
                                &test_6_out_scale, 1, &test_6_out_zero_point, 1, test_6_out_scales_frac, 1);

tensor_quantizer test_7_out_fx16(test_7_out_tsr_fx16, test_7_out_fx8_frac + 8, test_7_out_data,
                                 sizeof(test_7_out_data) / sizeof(test_7_out_data[0]));
tensor_quantizer test_7_out_sa8(test_7_out_tsr_sa8, test_7_out_sa_dim, test_7_out_data, 
                                sizeof(test_7_out_data) / sizeof(test_7_out_data[0]),
                                &test_7_out_scale, 1, &test_7_out_zero_point, 1, test_7_out_scales_frac, 1);

tensor_quantizer test_8_out_fx16(test_8_out_tsr_fx16, test_8_out_fx8_frac + 8, test_8_out_data,
                                 sizeof(test_8_out_data) / sizeof(test_8_out_data[0]));
tensor_quantizer test_8_out_sa8(test_8_out_tsr_sa8, test_8_out_sa_dim, test_8_out_data, 
                                sizeof(test_8_out_data) / sizeof(test_8_out_data[0]),
                                &test_8_out_scale, 1, &test_8_out_zero_point, 1, test_8_out_scales_frac, 1);



// Tests configuration structures
//========================================
const mli_conv2d_cfg test_1_cfg = {
    /* .relu.type = */MLI_RELU_NONE,
    /* .stride_width = */2,
    /* .stride_height = */2,
    /* .padding_left = */1,
    /* .padding_right = */1,
    /* .padding_top = */0,
    /* .padding_bottom = */1,
    /* .dilation_height = */1,
    /* .dilation_width = */1
};

const mli_conv2d_cfg test_2_cfg = {
    /* .relu.type = */MLI_RELU_GEN,
    /* .stride_width = */3,
    /* .stride_height = */4,
    /* .padding_left = */0,
    /* .padding_right = */0,
    /* .padding_top = */0,
    /* .padding_bottom = */0,
    /* .dilation_height = */1,
    /* .dilation_width = */1
};

const mli_conv2d_cfg test_3_cfg = {
    /* .relu.type = */MLI_RELU_1,
    /* .stride_width = */1,
    /* .stride_height = */1,
    /* .padding_left = */0,
    /* .padding_right = */0,
    /* .padding_top = */0,
    /* .padding_bottom = */0,
    /* .dilation_height = */1,
    /* .dilation_width = */1
};

const mli_conv2d_cfg test_4_cfg = {
    /* .relu.type = */MLI_RELU_6,
    /* .stride_width = */2,
    /* .stride_height = */3,
    /* .padding_left = */0,
    /* .padding_right = */0,
    /* .padding_top = */0,
    /* .padding_bottom = */0,
    /* .dilation_height = */1,
    /* .dilation_width = */1
};

const mli_conv2d_cfg test_5_cfg = {
    /* .relu.type = */MLI_RELU_NONE,
    /* .stride_width = */2,
    /* .stride_height = */3,
    /* .padding_left = */1,
    /* .padding_right = */1,
    /* .padding_top = */0,
    /* .padding_bottom = */0,
    /* .dilation_height = */1,
    /* .dilation_width = */1
};

const mli_conv2d_cfg test_6_cfg = {
    /* .relu.type = */MLI_RELU_6,
    /* .stride_width = */2,
    /* .stride_height = */2,
    /* .padding_left = */0,
    /* .padding_right = */0,
    /* .padding_top = */0,
    /* .padding_bottom = */0,
    /* .dilation_height = */1,
    /* .dilation_width = */1
};

const mli_conv2d_cfg test_7_cfg = {
    /* .relu.type = */MLI_RELU_6,
    /* .stride_width = */2,
    /* .stride_height = */2,
    /* .padding_left = */1,
    /* .padding_right = */1,
    /* .padding_top = */1,
    /* .padding_bottom = */1,
    /* .dilation_height = */1,
    /* .dilation_width = */1
};

const mli_conv2d_cfg test_8_cfg = {
    /* .relu.type = */MLI_RELU_NONE,
    /* .stride_width = */2,
    /* .stride_height = */2,
    /* .padding_left = */0,
    /* .padding_right = */1,
    /* .padding_top = */0,
    /* .padding_bottom = */1,
    /* .dilation_height = */1,
    /* .dilation_width = */1
};

