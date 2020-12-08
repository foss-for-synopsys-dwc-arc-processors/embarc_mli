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

// Generated weights vectors Declaration
//========================================

extern mli::tst::tensor_quantizer weights_1_fx8;
extern mli::tst::tensor_quantizer weights_1_fx16;
extern mli::tst::tensor_quantizer weights_1_sa8;

extern mli::tst::tensor_quantizer weights_2_memstr_fx8;
extern mli::tst::tensor_quantizer weights_2_memstr_fx16;
extern mli::tst::tensor_quantizer weights_2_memstr_sa8;

extern mli::tst::tensor_quantizer bias_1_fx8;
extern mli::tst::tensor_quantizer bias_1_fx16;
extern mli::tst::tensor_quantizer bias_1_i1_w2_sa32;

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

// Tests configuration structures Declaration
//========================================
extern const mli_conv2d_cfg test_1_cfg;
extern const mli_conv2d_cfg test_2_cfg;
extern const mli_conv2d_cfg test_3_cfg;
extern const mli_conv2d_cfg test_4_cfg;
extern const mli_conv2d_cfg test_5_cfg;


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


tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data,
                                sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
                            sizeof(input_1_data) / sizeof(input_1_data[0]),
                            &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

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

tensor_quantizer input_1_memstr_fx16(input_1_memstr_tsr_fx16, input_1_fx8_frac + 8, input_1_data,
                                    sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_memstr_sa8(input_1_memstr_tsr_sa8, input_1_sa_dim, input_1_data, 
                                    sizeof(input_1_data) / sizeof(input_1_data[0]),
                                    &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

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

static const float bias_1_data[] = {-0.149134f, -0.723882f, 0.276216f};

static const float bias_1_i1_w1_scales[] = {0.0000638685f, 0.0001414482f, 0.0002085013f};
static const float bias_1_i1_w1_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t bias_1_i1_w1_scales_frac[] = {28, 27, 27};
static const int bias_1_i1_w1_sa_dim = 0;

static const float bias_1_i1_w2_scales[] = {0.0000624f, 0.0001312402f, 0.0001976454f};
static const float bias_1_i1_w2_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t bias_1_i1_w2_scales_frac[] = {28, 27, 27};
static const int bias_1_i1_w2_sa_dim = 0;

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

tensor_quantizer weights_1_fx8(weights_1_tsr_fx8, weights_1_fx8_frac, weights_1_data,
                        sizeof(weights_1_data) / sizeof(weights_1_data[0]));
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

tensor_quantizer bias_1_fx8(bias_1_tsr_fx8, bias_1_fx8_frac, bias_1_data, sizeof(bias_1_data) / sizeof(bias_1_data[0]));
tensor_quantizer bias_1_fx16(bias_1_tsr_fx16, bias_1_fx8_frac + 8, bias_1_data, sizeof(bias_1_data) / sizeof(bias_1_data[0]));
tensor_quantizer bias_1_i1_w1_sa32(bias_1_i1_w1_tsr_sa32, bias_1_i1_w1_sa_dim, bias_1_data, 
    sizeof(bias_1_data) / sizeof(bias_1_data[0]), bias_1_i1_w1_scales, sizeof(bias_1_i1_w1_scales) / sizeof(bias_1_i1_w1_scales[0]), 
    bias_1_i1_w1_zero_points, sizeof(bias_1_i1_w1_zero_points) / sizeof(bias_1_i1_w1_zero_points[0]), 
    bias_1_i1_w1_scales_frac, sizeof(bias_1_i1_w1_scales_frac) / sizeof(bias_1_i1_w1_scales_frac[0]));
tensor_quantizer bias_1_i1_w2_sa32(bias_1_i1_w2_tsr_sa32, bias_1_i1_w2_sa_dim, bias_1_data, 
    sizeof(bias_1_data) / sizeof(bias_1_data[0]), bias_1_i1_w2_scales, sizeof(bias_1_i1_w2_scales) / sizeof(bias_1_i1_w2_scales[0]), 
    bias_1_i1_w2_zero_points, sizeof(bias_1_i1_w2_zero_points) / sizeof(bias_1_i1_w2_zero_points[0]), 
    bias_1_i1_w2_scales_frac, sizeof(bias_1_i1_w2_scales_frac) / sizeof(bias_1_i1_w2_scales_frac[0]));

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


