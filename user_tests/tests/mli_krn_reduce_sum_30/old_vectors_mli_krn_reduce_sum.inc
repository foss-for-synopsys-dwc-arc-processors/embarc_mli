/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <stdint.h>

#include "mli_types.h"
#include "mli_types.hpp"
#include "test_tensor_quantizer.h"

using mli::tst::tensor_quantizer;

namespace lib_mli = ::snps_arc::metaware::mli;

extern mli::tst::tensor_quantizer input_1_sa8;
extern mli::tst::tensor_quantizer input_1_fx8;
extern mli::tst::tensor_quantizer input_1_fx16;
extern mli::tst::tensor_quantizer input_1_sa32;
extern mli::tst::tensor_quantizer input_2_sa8;
extern mli::tst::tensor_quantizer input_2_fx8;
extern mli::tst::tensor_quantizer input_2_fx16;
extern mli::tst::tensor_quantizer input_2_sa32;
extern mli::tst::tensor_quantizer input_3_sa8;
extern mli::tst::tensor_quantizer input_3_fx8;
extern mli::tst::tensor_quantizer input_3_fx16;
extern mli::tst::tensor_quantizer input_3_sa32;
extern mli::tst::tensor_quantizer input_4_sa8;
extern mli::tst::tensor_quantizer input_4_fx8;
extern mli::tst::tensor_quantizer input_4_fx16;
extern mli::tst::tensor_quantizer input_4_sa32;
extern mli::tst::tensor_quantizer test_1_out_sa8;
extern mli::tst::tensor_quantizer test_1_out_fx8;
extern mli::tst::tensor_quantizer test_1_out_fx16;
extern mli::tst::tensor_quantizer test_1_out_sa32;
extern mli::tst::tensor_quantizer test_2_out_sa8;
extern mli::tst::tensor_quantizer test_2_out_fx8;
extern mli::tst::tensor_quantizer test_2_out_fx16;
extern mli::tst::tensor_quantizer test_2_out_sa32;
extern mli::tst::tensor_quantizer test_3_out_sa8;
extern mli::tst::tensor_quantizer test_3_out_fx8;
extern mli::tst::tensor_quantizer test_3_out_fx16;
extern mli::tst::tensor_quantizer test_3_out_sa32;
extern mli::tst::tensor_quantizer test_4_out_sa8;
extern mli::tst::tensor_quantizer test_4_out_fx8;
extern mli::tst::tensor_quantizer test_4_out_fx16;
extern mli::tst::tensor_quantizer test_4_out_sa32;
extern mli::tst::tensor_quantizer test_5_out_sa8;
extern mli::tst::tensor_quantizer test_5_out_fx8;
extern mli::tst::tensor_quantizer test_5_out_fx16;
extern mli::tst::tensor_quantizer test_5_out_sa32;
extern mli::tst::tensor_quantizer test_6_out_sa8;
extern mli::tst::tensor_quantizer test_6_out_fx8;
extern mli::tst::tensor_quantizer test_6_out_fx16;
extern mli::tst::tensor_quantizer test_6_out_sa32;
extern mli::tst::tensor_quantizer test_7_out_sa8;
extern mli::tst::tensor_quantizer test_7_out_fx8;
extern mli::tst::tensor_quantizer test_7_out_fx16;
extern mli::tst::tensor_quantizer test_7_out_sa32;
extern mli::tst::tensor_quantizer test_8_out_sa8;
extern mli::tst::tensor_quantizer test_8_out_fx8;
extern mli::tst::tensor_quantizer test_8_out_fx16;
extern mli::tst::tensor_quantizer test_8_out_sa32;
extern const lib_mli::ReduceOpConfig test_1_cfg;
extern const lib_mli::ReduceOpConfig test_2_cfg;
extern const lib_mli::ReduceOpConfig test_3_cfg;
extern const lib_mli::ReduceOpConfig test_4_cfg;
static const float input_1_data[] = {33.810513f, -9.318748f, 0.656403f, 8.150326f, -15.778461f, 0.041311f, -0.017808f, -35.094486f, 20.353161f, 12.009970f, -12.508579f, -3.430965f, 10.105988f, -5.227128f, -4.854981f, -29.064829f};

static const float input_1_scale = 4.5936665535f;
static const float input_1_zero_point = -0.6419868469f;
static const int8_t input_1_scales_frac[] = {12};
static const int input_1_sa_dim = -1;

static const int input_1_fx8_frac = 1;

#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 2, 2, 2}, \
    /* .rank =  */ 4


static const mli_tensor input_1_tsr_sa8 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_1_tsr_fx8 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_1_tsr_fx16 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_1_tsr_sa32 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float input_2_data[] = {0.554580f, 0.123881f, 0.274460f, -1.526525f, 1.650700f, 0.154336f, -0.387140f, 2.029072f, -0.045386f, -1.450679f, -0.405228f, -2.288315f, 1.049397f, -0.416474f, -0.742554f, 1.072470f, -1.651076f, 0.535429f, -2.064415f, -0.662159f, -1.204220f, 1.461976f, 1.766161f, -0.329414f, 0.840733f, -0.179986f, 0.568062f, -0.752837f, -1.708339f, -1.803099f, 0.383122f, 2.247595f, 0.269412f, -0.524605f, 1.912019f, 0.237302f, 0.101434f, 0.252578f, -0.132377f, -0.309476f, -1.434963f, 0.501624f, -0.094775f, 1.193086f, -0.368818f, -1.906370f, -0.099611f, 1.699537f, -0.383423f, -0.889857f, -1.193592f, -1.050017f, -0.300194f, -1.179982f, 1.497639f, -0.282635f, 0.108648f, 1.438240f, 1.503319f, -0.212733f};

static const float input_2_scale = 0.0768798292f;
static const float input_2_zero_point = -0.0203599930f;
static const int8_t input_2_scales_frac[] = {18};
static const int input_2_sa_dim = -1;

static const int input_2_fx8_frac = 5;

#define INPUT_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 6, 5, 2}, \
    /* .rank =  */ 4


static const mli_tensor input_2_tsr_sa8 = {
    INPUT_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_2_tsr_fx8 = {
    INPUT_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_2_tsr_fx16 = {
    INPUT_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_2_tsr_sa32 = {
    INPUT_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float input_3_data[] = {0.331974f, 0.735027f, -0.192855f, -1.778013f, 0.654706f, 0.894352f, 0.415503f, -0.923545f, -0.196027f, -0.590770f, -0.299711f, 1.296885f, 1.529580f, 0.669418f, 0.548745f, 0.676629f, -0.012242f, -0.075663f, -0.673645f, -0.055867f, 2.259947f, 0.869039f, -0.342117f, -0.471927f, -0.864490f, 0.374370f, 0.391546f, -1.443122f, 0.486335f, -0.569472f, 1.426721f, 0.156844f, 1.717730f, -0.458127f, -0.287984f, 0.299808f, 1.055948f, 0.565883f, -1.233524f, 0.182901f, 0.022245f, -0.429069f, -0.648105f, 1.747577f, -0.390386f, -0.845923f, 0.637113f, 0.130623f, -0.075814f, 0.781302f, 0.488625f, 0.362190f, 0.964200f, 0.283636f, -0.616922f, -0.362282f, -0.520972f, 0.209722f, -1.077985f, -1.832426f, 0.088155f, -1.336521f, -1.951978f, 0.373331f, -0.711053f, 0.496125f, -0.530854f, -1.121962f, -1.243518f, -0.514927f, -0.273383f, -0.823460f, -0.281108f, -1.536102f, 0.014489f, -1.793649f, -0.193856f, 0.697294f, 1.091939f, 1.714380f, -1.556288f, 0.758725f, 1.206282f, 0.980557f, -0.929324f, 0.411797f, 1.860056f, -1.497274f, 0.476338f, 1.112291f, -0.696591f, 0.582970f, -1.070892f, -0.812209f, -0.817080f, 0.192308f, -0.090981f, 0.954115f, -0.940279f, -0.131276f, 0.961076f, 0.560777f, -1.324641f, 0.665204f, -0.478418f, 0.385387f, 0.300680f, 1.219319f, 0.116560f, -1.853806f, -1.369860f, 0.731154f, -0.258316f, 1.137979f, 0.273439f, -0.749066f, 1.082407f, -0.807398f, -2.052635f, 2.063796f, -1.910749f, 0.636413f, 0.938852f, 0.147066f, -1.683397f, 1.015015f, -1.441544f, -1.343629f, -0.361668f, 0.788505f, 0.300114f, 0.810768f, -0.443644f, -0.172183f, -1.137137f, -0.316865f, 0.593128f, -0.299828f, 0.475460f, -0.573274f, -0.592392f, 0.039169f, -0.194881f, 0.592385f, 0.178290f, 1.740346f, -0.219495f, -0.227253f, -1.040131f, 0.022778f, 0.682829f, -0.740296f, -0.049672f, 0.718557f, -0.126909f, -0.519137f, 0.973093f, 0.097020f, 0.396565f, 1.171393f, 1.145316f, -0.347204f, -0.525430f, 0.284309f, -0.132788f, -1.899861f, 1.275362f, 0.019361f, 0.736378f, -1.172917f, 1.288204f, -0.588263f, -0.107667f, -1.401425f, -0.166950f, -0.229804f, -0.185361f, 0.438505f, 0.060346f, 0.975783f, -0.468158f, -0.996016f, -0.019106f, 0.628368f, 1.244642f, 0.813921f, -1.765441f, 0.377058f, 1.460372f, 0.062576f, -0.516042f, 0.309161f, -0.503242f, 0.639113f, -0.012994f, 1.393000f, 0.499503f, -0.723885f, 0.604631f, 0.936012f, -0.568456f, -0.425794f, -2.305183f, 0.903406f, -0.587754f, -2.122350f, -1.768772f, -0.222950f, 0.760660f, -0.089065f, 0.471778f, -1.101500f, -1.341366f, 0.885721f, 0.492144f, -0.030509f, 1.282093f, -0.951190f, -0.420266f, 1.026371f, 1.837794f, 0.008426f, 1.679519f, -0.602511f, 0.329750f, -0.173824f, -1.588248f, 0.257973f, 0.932750f, -0.148267f, -0.070455f, 1.392927f, -1.249714f, -1.497588f, -0.654039f, -1.196623f, 1.346289f, 0.309689f, -0.864093f, -0.614479f, 2.861067f, -0.610090f, -0.762025f, 0.304307f, 0.638252f, 0.221682f, -0.908626f, -1.159187f, 0.596141f, -0.054190f, -1.641433f, 1.171001f, 0.775329f, -0.240954f, -0.260264f, 0.231460f, -0.687009f, -1.348312f, 0.322518f, -1.439087f, 0.390727f, -1.441208f, -0.099209f, -0.135645f, 1.782629f, -1.064383f, -1.414946f, -0.646177f, -1.123732f, -0.589762f, 0.420801f, 0.771977f, 1.659561f, -0.736674f, 2.638539f, 1.982185f, -0.203527f, -2.291420f, -0.083112f, 0.072508f, -2.299249f, 0.073832f, -0.200154f, -0.785711f, 0.626629f, 1.005348f, -0.486177f, 0.229480f, -0.654792f, -0.786606f, 0.079497f, 0.523422f, -0.563293f, -2.213670f, 0.585785f, -1.596558f, 0.121999f, 1.779026f, -0.016830f, -1.189750f, -1.461179f, 1.170462f, -0.016948f, 0.539738f, 0.926302f, -0.049002f, -0.308604f, -0.947348f, -3.082505f, -1.518000f, -0.069594f, -0.175262f, 0.403560f, -0.126687f, 0.159358f, 0.787799f, -0.885045f, -0.287158f, -1.607213f, 0.972893f, 0.038418f, 0.489822f, 1.542483f, 1.079546f, -0.060001f, 0.787685f, -1.431704f, 0.433909f, 0.138910f, -1.618269f, 1.337320f, -0.704077f, 0.683229f, -0.419935f, 0.993543f, 1.312262f, -0.830049f, -1.530399f, -0.131410f, 0.848917f, -0.371312f, -0.613524f, 0.488282f, -2.761518f, -0.664716f, -1.383796f, 0.294814f, -0.736249f, -1.115168f, -0.353831f, -1.468408f, -0.013124f, -0.682372f, -0.377419f, -0.717138f, 0.912263f, -0.108854f, 0.524512f, 0.479785f, -0.806151f, 0.569973f, 0.349108f, 0.313649f, -0.398693f, -0.426844f, 0.785552f, 2.181960f, -0.830127f, 0.534448f, -0.366253f, 0.784857f, 0.438313f, 1.060126f, -2.794785f, -0.601083f, -0.988185f, -0.107988f, -0.723866f, -0.640976f, 0.681860f, 1.901928f, -2.823831f, -0.310450f, 0.312054f, 0.004733f, -0.345135f, -1.353468f, -0.942672f, 0.724982f, 1.075763f, 0.096675f, -0.836172f, 0.095711f, -0.240040f, 0.545605f, 0.118299f, -0.621790f, 0.190673f, -1.179405f, 0.820012f, 1.182335f, -1.178504f, 0.421047f, -0.991262f, 1.071684f, 0.904119f, -0.230178f, 0.994934f, -0.431091f, -0.643010f, -0.568720f, -0.140851f, -0.323412f, 0.154345f, -0.615059f, 0.764696f, 1.020969f, -0.495862f, -1.427175f, 0.723873f, 1.138557f, -0.163248f, -1.286517f, 0.266761f, 1.524668f, -1.876327f, 0.477926f, -0.313635f, -0.855674f, 0.498200f, -1.298830f, 0.296945f, 1.313259f, 0.176689f, -0.876322f, -1.779309f, -1.281958f, -1.841222f, 1.599632f, -0.249484f, -1.045164f, -1.298209f, -1.067245f, -0.335843f, -1.282611f, 1.676308f, -0.747389f, -0.782583f, 0.307758f, 1.450584f, 1.323976f, -0.487819f, 0.255428f, 0.149295f, 0.529894f, 0.255705f, -0.943828f, -1.260190f, 0.175656f, -1.183538f, 1.363223f, 0.039930f, -1.026078f, -0.321635f, -0.507307f, -0.131240f, 0.945849f, -1.444428f, -1.867577f, 0.338051f, -0.177283f, 0.163225f, 0.444547f, -0.181048f, 0.799690f, -0.893001f, 0.064378f, -0.830762f, -1.129906f, 0.615449f, 0.569053f, -1.941980f, -1.121999f, 0.291984f, 1.070232f, -0.665960f, -0.183810f, -1.413347f, 2.510811f, 0.475121f, 0.894613f, 0.658021f, 0.933244f, 0.608083f, -0.335188f, 0.744013f, 0.079140f, 1.296576f, -0.735548f, -0.541261f, 0.661154f, 1.279645f, 0.145161f, -0.712528f, -0.118956f, -0.498224f, 1.896838f, 0.898709f, 1.264443f, 0.301545f, -0.555123f, 1.173264f, 1.653949f, 0.581959f, -0.024252f, 1.080974f, -0.371115f, 0.764541f, -0.711584f, -0.970366f, 0.566445f, 0.337930f, 1.904306f, -0.280107f, 0.023448f, -1.139939f, -1.869280f, -1.780815f, 0.463108f, -1.176639f, 1.126787f, 1.422267f, -0.285403f, 1.658425f, 0.227200f, 1.124955f, -0.225706f, 0.959171f, -0.943317f, 0.622683f, -0.459217f, -0.130560f, 0.631719f, 0.172712f, -0.074375f, 0.727779f, 0.803206f, -1.385007f, 0.558944f, 0.671001f, -0.751719f, 0.385801f, -1.324233f, 1.439324f, -0.237654f, -0.078560f, 1.197392f, -0.142741f, 0.598888f, -1.047400f, -0.569755f, -1.392457f, -0.208185f, 1.031035f, 0.223608f, 0.884278f, -1.026208f, -0.883227f, -0.465543f, 0.572573f, 0.356429f, -0.015289f, 0.622629f, -0.151398f, -0.021524f, 1.304835f, -0.586269f, 0.182681f, 1.611945f, 0.101905f, 0.706553f, -0.606682f, 1.956520f, -2.249631f, -1.134322f, -1.053844f, 1.619804f, 0.352481f, -0.262327f, -1.281758f, 0.364129f, 0.114034f, -0.692969f, -1.491527f, -1.102758f, 0.213413f, 0.311011f, 0.073144f, -0.151585f, -1.768737f, 0.048770f, -1.032535f, 0.041790f, 0.465051f, -0.675776f, -1.023071f, 0.993611f, 0.351797f, -0.636626f, -3.000316f, -0.049125f, 0.629138f, 1.121557f, -0.677859f, -0.643031f, -0.442907f, 0.837922f, -1.174718f, -0.112877f, -0.925745f, 0.268548f, -1.554685f, 0.376160f, -0.496647f, -0.224038f, 0.525754f, 1.143813f, -1.696987f, 1.003555f, 0.558156f, -0.294692f, 1.404000f, 0.188650f, 1.293437f, -0.592790f, -0.866720f, 1.224657f, 2.164272f, 0.633785f, 0.891185f, -0.473898f, -1.325168f, -0.225969f, 1.500319f, -1.860459f, -0.495327f, -1.914540f, 0.108269f, 0.874244f, -1.242732f, -0.409392f, 1.325747f, -0.088467f, -1.239247f, -1.316261f, -2.014647f, 0.773295f, -0.746122f, 1.050681f, 0.867029f, -0.660030f, -0.527954f, 0.778634f, 0.149428f, 0.267829f, -1.647090f, 0.897712f, -0.875326f, -2.269504f, -0.616925f, 0.421996f, 2.097126f, -0.372826f, -1.329782f, 2.123474f, 0.160238f, -0.261173f, -1.075987f, -1.003978f, 1.478673f, -0.270691f, -0.537809f, -0.509913f, 1.567866f, -0.589414f, 0.628160f, 0.266142f, 0.239266f, -0.877251f, 0.658554f, 1.608632f, -0.052184f, -0.919746f, -1.291374f, 0.190718f, -0.271984f, -0.104374f, 0.011829f, -1.475613f, 0.781485f, 0.234318f, -0.074616f, 0.231857f, 0.926143f, 0.053425f, -0.850234f, 1.166233f, -0.433324f, -0.725470f, -1.561996f, 0.007915f, -1.047644f, -0.519133f, 0.449287f, 0.033270f, 1.016287f, 0.305137f, -0.403895f, -0.329703f, -0.507374f, 0.646371f, 0.021466f, -0.410920f, 0.844923f, 0.096286f, -0.465535f, -1.512397f, 0.437472f, 0.238709f, 0.051331f, -0.434066f, -0.833486f, -0.226698f, 0.231426f, -0.127851f, -0.144712f, -0.468176f, 0.338905f, 0.233626f, -0.752411f, 0.760264f, 0.791357f, 0.813596f, -0.130646f, -0.194161f, -1.159171f, 0.754660f, -2.135793f, -0.390959f, -0.339840f, 0.856219f, -0.000613f, -0.964469f, -1.426765f, -0.729134f, 0.106609f, 0.055047f, -0.197626f, -0.684329f, -0.210131f, 0.997763f, -1.100428f, 0.033643f, 1.618569f, 0.554328f, -0.455936f, 0.660972f, -0.752024f, 0.007048f, 1.554035f, 1.894136f, -1.708630f, 0.700067f, 1.948613f, -0.545339f, 0.258991f, 1.507241f, -0.492521f, 0.452024f, -0.059207f, -1.925367f, 0.346121f, -0.184120f, 0.928131f, 2.093387f, 0.683891f, -0.367397f, 0.450151f, 0.201782f, 1.022980f, -1.438788f, -1.166327f, -2.045182f, 0.607085f, -0.019007f, -0.640138f, -0.725247f, -0.572191f, -0.927816f, -0.321402f, -0.494216f, 1.389974f, 0.323307f, -0.818523f, -0.583886f, 1.198940f, 0.452427f, 0.295183f, 0.720647f, 1.586113f, -1.217319f, -0.745674f, -1.740620f, 0.494881f, -0.378818f, -0.671851f, -0.762229f, 0.691356f, -0.142080f, -0.530522f, 1.255361f, -1.046073f, -1.264379f, 2.594646f, -0.367732f, -0.543471f, -0.727420f, -0.136895f, 0.782744f, 0.753463f, 0.544058f, -0.278602f, -1.490074f, 0.282956f, 1.518797f, 1.069279f, -0.168258f, 0.802538f, -0.385492f, -0.055148f, -1.234365f, -0.315428f, 0.140565f, -0.169042f, 0.769963f, 1.883674f, 1.321638f, -0.304953f, -0.731183f, -0.652660f, -0.977799f, 0.034013f, 0.421119f, 0.183040f, 0.193650f, 0.158200f, -0.132054f, -0.742356f, 1.724646f, 1.047532f, -0.908256f, -0.905242f, -0.586270f, 0.033029f, 1.259046f, 1.014847f, 1.067750f, 1.087834f, 1.106402f, 0.736690f, 0.454504f, -0.119054f, -1.743768f, -0.129222f, 1.801978f, -0.539966f, -0.625693f, 1.017416f, 0.080625f, -0.480053f, 1.557663f, 0.624156f, 0.757578f, -0.564575f, 0.497052f, 0.376493f, 0.777078f, 1.149170f, 1.126073f, 0.170491f, 0.689499f, -1.145078f, 0.643612f, 0.684315f, -0.388854f, -0.773748f, -0.058757f, -1.213453f, 0.340881f, 0.294701f, -0.948568f, 0.047568f, 1.067713f, 1.062610f, 1.010541f, -0.150360f, -0.338133f, -0.390863f, 0.592680f, 2.351867f, 1.405004f, 1.585773f, 1.349905f, -1.612765f, -0.200719f, -0.432644f, 1.249578f, 1.688386f, -0.663918f, 0.038660f, 0.557177f, -0.488457f, -0.628522f, -0.677439f, 0.884888f, -0.909118f, 1.135317f, 0.086851f, 0.751667f, 1.545038f, -0.668927f, -0.816678f, 0.426734f, -1.031639f, -0.730026f, -0.113475f, 0.156858f, -1.243373f, -0.382122f, 0.582409f, 1.342994f, -1.959364f, 1.303778f, 0.879939f, -1.082737f, -1.466776f, 0.416408f, -0.418382f, 0.135707f, 0.858200f, -0.370886f, 0.161811f, 0.473566f, 0.885880f, 1.755204f, 1.694499f, 0.240223f, -0.235218f, 2.363380f, 0.637837f, 0.425601f, -1.095883f, 0.775224f, -0.773075f, -0.455524f, -1.328443f, -0.515960f, 0.062203f, -1.492490f, -0.890168f, -0.077298f, -1.256239f, -1.245609f, -0.827385f, -1.399928f, 0.611581f, -0.988388f, 0.765927f, 0.920914f, -1.192153f, 0.040771f, -0.476436f, -1.132051f, -0.333159f, -0.240153f, 0.708190f, 1.021623f, -1.095978f, -0.404438f, 0.528620f, 1.173811f, -2.434089f, 1.274662f, -0.167363f, 1.528185f, -0.422497f, -0.828602f, 0.044862f, 1.166579f, 0.030610f, -2.351926f, 0.682637f, -1.605997f, -0.976377f, -1.159282f, -1.834470f, -1.250412f, -0.985846f, -0.611231f, -0.053254f, -0.333982f, -1.480404f, -0.181690f, 0.972736f, 0.948211f, 0.134652f, 0.619387f, 0.892259f, 1.361653f, 1.696450f, -0.320278f, -0.271657f, 1.472799f, 1.561071f, 1.540657f, -2.343559f, -0.219689f, -3.255515f, 0.729936f, 0.055616f, 0.052179f, -0.137695f, 1.303371f, -0.293107f, 1.033114f, 0.553598f, 0.356816f, 0.462595f, 0.110706f, -1.093726f, 0.716220f, 0.713328f, -0.143884f, 0.730710f, 0.898929f, -1.281530f, -0.142136f, -0.443249f, -0.905160f, 0.322314f, -0.843178f, -0.434382f, 0.245399f, 0.349850f, 0.321546f, 2.233908f, -0.503687f, 0.250186f, -1.205844f, 0.481356f, 1.153454f, 1.213860f, -0.635812f, -0.279464f, 1.262388f, -0.093371f, 0.958674f, -0.972422f, 1.383369f, 1.027248f, 0.464046f, -0.798536f, -0.159710f, 0.151261f, 0.263412f, -0.104779f, -0.218438f, 0.192456f, 1.395560f, -2.007881f};

static const float input_3_scale = 0.0056687514f;
static const float input_3_zero_point = -0.1972241402f;
static const int8_t input_3_scales_frac[] = {22};
static const int input_3_sa_dim = -1;

static const int input_3_fx8_frac = 5;

#define INPUT_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 9, 4, 15}, \
    /* .rank =  */ 4


static const mli_tensor input_3_tsr_sa8 = {
    INPUT_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_3_tsr_fx8 = {
    INPUT_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_3_tsr_fx16 = {
    INPUT_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_3_tsr_sa32 = {
    INPUT_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float input_4_data[] = {-1.371873f, -1.544873f, -0.686978f, -1.740260f, 0.553120f, 0.095720f, -0.253777f, -0.681543f, 0.156858f, 0.178250f, 0.330263f, -0.729015f, 0.649313f, -1.468553f, 1.249436f, -3.076721f, -0.593652f, -1.202292f, 0.789744f, -1.485153f, 1.821844f, -1.544586f, 2.028460f, 0.448007f, 0.030719f, 0.435786f, 2.192579f, 0.893080f, -2.233381f, 1.194225f, 0.231777f, -1.617023f, 0.760483f, -1.211108f, 0.181193f, 0.233243f, -0.648398f, 0.647565f, -1.021602f, 0.191182f, -0.547315f, -0.182666f, 1.922891f, 0.112044f, -1.087372f, -0.117230f, -0.769263f, 0.026488f, 0.161925f, 0.084981f, 0.083811f, -0.544460f, 0.449203f, 1.057358f, -0.620454f, -0.058400f, 0.060698f, 1.253749f, 1.079337f, -0.730048f, 0.072355f, -1.493982f, -0.965491f, 0.015023f, 0.629938f, 0.954395f, -0.310799f, -1.352944f, 0.396887f, -0.173041f, 0.423960f, 0.160124f, 0.715728f, 0.654629f, 0.843599f, -0.459159f, 0.995498f, -0.503029f, 0.084938f, 1.860592f, 0.356802f, -1.166131f, 1.490428f, 1.000092f, 1.140474f, 0.197208f, 0.345824f, -0.233004f, -0.699932f, 1.529995f, -0.368361f, -0.480698f, -0.443495f, 1.862343f, 0.878136f, -0.367912f, -0.045622f, 0.510451f, 0.878468f, -1.610490f, 0.903784f, -0.218026f, -0.686035f, -0.390316f, 0.791927f, 1.758432f, 0.147920f, 0.556904f, -0.814024f, 0.491345f, 0.822647f, -1.488031f, -0.512910f, 0.790799f, -1.758901f, -0.373900f, 0.600697f, -1.690561f, -0.396653f, 0.054838f, 0.890394f, -1.278736f, 1.109510f, 0.305675f, 1.196846f, 1.163220f};

static const float input_4_scale = 0.0421543978f;
static const float input_4_zero_point = -0.4420711994f;
static const int8_t input_4_scales_frac[] = {19};
static const int input_4_sa_dim = -1;

static const int input_4_fx8_frac = 5;

#define INPUT_4_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {7, 6, 3}, \
    /* .rank =  */ 3


static const mli_tensor input_4_tsr_sa8 = {
    INPUT_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_4_tsr_fx8 = {
    INPUT_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_4_tsr_fx16 = {
    INPUT_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_4_tsr_sa32 = {
    INPUT_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_1_out_data[] = {54.163673f, 2.691222f, -11.852176f, 4.719360f, -5.672473f, -5.185817f, -4.872789f, -64.159317f};

static const float test_1_out_scale = input_1_scale; //16.9032840729f;
static const float test_1_out_zero_point = 2 * input_1_zero_point; //-4.9978218079f;
static const int8_t* test_1_out_scales_frac = input_1_scales_frac; //[] = {10};
static const int test_1_out_sa_dim = -1;

static const int test_1_out_fx8_frac = input_1_fx8_frac; //0;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 2, 2, 2}, \
    /* .rank =  */ 4


static const mli_tensor test_1_out_tsr_sa8 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


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


static const mli_tensor test_1_out_tsr_sa32 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_2_out_data[] = {18.032051f, -9.277436f, 0.638596f, -26.944160f, 30.459148f, 6.782842f, -17.363560f, -32.495792f};

static const float test_2_out_scale = input_1_scale; //8.9935626984f;
static const float test_2_out_zero_point = input_1_zero_point; //-1.0183219910f;
static const int8_t* test_2_out_scales_frac = input_1_scales_frac; //[] = {11};
static const int test_2_out_sa_dim = -1;

static const int test_2_out_fx8_frac = input_1_fx8_frac; //1;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 1, 2, 2}, \
    /* .rank =  */ 4


static const mli_tensor test_2_out_tsr_sa8 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_2_out_tsr_fx8 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


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

static const float test_3_out_data[] = {2.047214f, -0.669915f, -3.813875f, -1.759049f, 0.262397f, -1.603360f, 2.533609f, 1.903394f, -2.381591f, 0.598020f, 1.615820f, -1.287127f};

static const float test_3_out_scale = input_2_scale; //0.5770440698f;
static const float test_3_out_zero_point = 5 * input_2_zero_point;//-0.6401331425f;
static const int8_t* test_3_out_scales_frac = input_2_scales_frac; //[] = {15};
static const int test_3_out_sa_dim = -1;

static const int test_3_out_fx8_frac = input_2_fx8_frac; //5;

#define TEST_3_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 6, 1, 2}, \
    /* .rank =  */ 4


static const mli_tensor test_3_out_tsr_sa8 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


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


static const mli_tensor test_3_out_tsr_sa32 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_4_out_data[] = {0.678461f, -1.252065f, 1.805035f, 1.641932f, -1.496065f, -2.693543f, 0.632922f, 0.329917f, -1.115646f, -2.726574f, 0.257756f, 1.436747f, 0.660747f, -0.184775f, -3.511438f, 2.630717f, -0.255193f, 2.149321f, 0.354012f, -0.441854f, -0.933339f, 1.098310f, -2.275188f, 1.599927f, -1.273280f, -2.243609f, -1.480176f, 1.215004f, 1.546888f, 1.290586f};

static const float test_4_out_scale = input_2_scale; //0.2117984444f;
static const float test_4_out_zero_point = 2 * input_2_zero_point;//-0.4403605461f;
static const int8_t* test_4_out_scales_frac = input_2_scales_frac;//[] = {17};
static const int test_4_out_sa_dim = -1;

static const int test_4_out_fx8_frac = input_2_fx8_frac; //5;

#define TEST_4_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 6, 5, 1}, \
    /* .rank =  */ 4


static const mli_tensor test_4_out_tsr_sa8 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_4_out_tsr_fx8 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_4_out_tsr_fx16 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_4_out_tsr_sa32 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_5_out_data[] = {-0.127243f, 0.604467f, 0.438863f, -1.605301f, 0.580331f, 1.622131f, 1.218709f, -2.308552f, 0.362917f, 0.080231f, -1.051431f, 1.682686f, 0.205346f, 2.108742f, 0.311091f, 0.598069f, 1.185150f, -0.218405f, -0.074757f, -1.103268f, 1.690192f, -0.523418f, -0.550302f, 0.559109f, -0.640882f, 1.258648f, -0.634662f, -2.326349f, 0.020792f, 0.003101f, 1.783150f, 0.141555f, 2.340359f, -0.609525f, -0.309508f, 1.604643f, 0.469680f, 0.748564f, 0.378421f, 0.284805f, 0.728798f, -1.035750f, 1.308415f, -0.502054f, -1.524709f, -1.899767f, 2.256917f, 0.483104f, -0.338141f, -0.500456f, 0.852754f, 0.476224f, 0.271231f, -1.207891f, -1.719680f, -0.148869f, -0.209962f, 0.282866f, -1.229570f, -3.601163f, 0.136925f, -2.369056f, -1.910188f, 0.838381f, -1.386829f, -0.526945f, 0.462757f, -0.770165f, -1.880143f, -3.515243f, -0.322508f, -0.194322f, 0.840449f, -2.213962f, -0.628542f, -2.236556f, 0.644066f, -0.477424f, 0.979062f, 0.788636f, -1.287739f, -0.795959f, 1.582442f, 0.483910f, -1.153362f, 0.937551f, 3.003869f, -3.194261f, 1.479893f, 1.670447f, -0.991283f, 1.986970f, -0.882242f, 0.481228f, -1.409870f, -0.674412f, 1.133676f, 3.118387f, -0.306494f, 0.759909f, 0.487178f, -0.764390f, -1.550610f, 2.165523f, -2.338877f, -0.109941f, -1.613860f, 1.327588f, 0.990804f, -3.096538f, -1.779251f, 2.056901f, -0.346783f, -0.101268f, -1.042822f, -2.763713f, 1.855701f, -1.553520f, -1.001955f, 2.930825f, -2.570779f, 0.108459f, 1.717486f, 0.296494f, -1.415567f, -0.632075f, -0.543832f, -2.218954f, -2.631173f, 0.171579f, 0.722110f, 2.907894f, -0.816470f, -1.501965f, 0.986337f, -0.156627f, 0.331956f, -1.375815f, -0.528518f, 0.905398f, -0.863082f, -0.498640f, -0.704794f, 2.160251f, -0.411123f, 2.368506f, 0.046647f, 0.012013f, -1.917382f, 0.681332f, 2.291461f, -0.792479f, -0.969418f, -0.572817f, 0.063808f, -0.791121f, 0.868719f, 0.108848f, -1.079048f, 1.952879f, 1.379634f, -0.421820f, -0.293573f, 1.210452f, -0.079363f, -2.750095f, 2.441595f, -0.413963f, 0.010908f, -2.734912f, 1.296119f, -1.635906f, -0.626800f, -0.952137f, -0.133680f, 0.786483f, 0.119777f, 0.034609f, -0.269357f, 0.468409f, 0.178213f, -0.974550f, -0.430027f, 1.473291f, 1.340928f, 0.348386f, -3.277838f, 0.814530f, 1.699080f, 0.113908f, -0.950109f, -0.524325f, -0.729940f, 0.870539f, -0.140845f, 1.248289f, 0.031327f, -0.384980f, 0.838257f, 0.183600f, 0.191809f, 0.365562f, -1.491588f, 0.772760f, -0.781916f, -3.281520f, -1.014111f, -2.358743f, 0.369701f, -0.428905f, 1.327996f, -1.102113f, -2.305835f, -0.541045f, -0.236989f, 0.076100f, 1.337139f, -1.148816f, -1.104596f, 0.816241f, 2.835557f, -1.092003f, 1.713163f, 1.016059f, 0.884078f, -0.629759f, -0.927277f, -0.494051f, 0.939797f, 1.405768f, 1.823681f, -0.315702f, -0.549646f, 0.451024f, -1.199378f, -0.937632f, 2.853530f, -0.182832f, -0.412069f, -0.673686f, 0.935701f, -0.263969f, -0.946144f, 1.232438f, 2.731640f, 0.905573f, -1.276023f, -0.709036f, 0.797922f, 0.968790f, -3.080221f, 0.004673f, -1.269853f, 0.366130f, -0.279271f, -0.408678f, -1.412256f, -1.920504f, -0.605298f, -1.760488f, -0.103489f, -0.051233f, 0.224098f, -0.954167f, 1.198743f, 0.134557f, -0.962518f, -0.350995f, -0.403086f, 0.996351f, -0.796518f, 0.026303f, -0.081058f, -0.241793f, 2.259720f, 1.310334f, -0.965756f, -1.600064f, -0.225193f, -0.458015f, -1.043888f, -0.972241f, -1.464533f, 1.808934f, 0.258897f, 0.461877f, -1.213598f, 0.092585f, 0.127952f, -0.033143f, 0.623555f, 0.244819f, -2.053367f, -1.930714f, 2.104582f, -0.527279f, -0.046260f, 2.581564f, -0.402322f, -1.244898f, -2.695544f, 0.855034f, 0.123617f, 0.370696f, 1.696265f, 1.834672f, 1.013033f, -1.252301f, -3.813688f, -2.170660f, -1.047392f, -0.141249f, 0.824679f, 0.056353f, 0.353008f, 0.945999f, -1.017099f, -1.029514f, 0.117433f, 2.020424f, -0.869838f, -0.415419f, 0.956213f, 1.112575f, 1.199045f, 1.802532f, -0.363954f, 1.521744f, 1.245312f, -0.881579f, 1.791824f, -0.823131f, -1.060539f, -0.549157f, 2.795521f, 0.772295f, -1.455743f, -0.512983f, -0.050785f, 0.368864f, 1.186351f, 0.010632f, 1.245860f, -3.326093f, -0.167664f, -1.007303f, 1.071892f, 0.412920f, 0.010906f, -0.183340f, -0.778909f, -1.158202f, -0.038760f, 0.306896f, -1.105992f, 0.138515f, -0.167611f, -0.688941f, 0.820666f, -0.511450f, -0.378594f, 0.396676f, 1.381361f, 0.663917f, 0.583697f, 0.635192f, 1.843826f, -1.220990f, 1.127128f, 1.985615f, 2.189861f, 2.024086f, 2.410030f, -4.407550f, -0.801802f, -1.420830f, 1.141590f, 0.964520f, -1.304893f, 0.720520f, 2.459105f, -3.312287f, -0.938972f, -0.365385f, 0.889621f, -1.254253f, -0.218150f, -0.855821f, 1.476649f, 2.620800f, -0.572252f, -1.652851f, 0.522445f, -1.271679f, -0.184421f, 0.004824f, -0.464932f, -1.052700f, -1.561527f, 1.402421f, 2.525330f, -3.137868f, 1.724825f, -0.111324f, -0.011053f, -0.562658f, 0.186229f, 0.576552f, -0.295384f, 0.215190f, -0.939606f, 0.020961f, 0.150154f, 1.040225f, 1.140146f, 2.459195f, 1.261193f, -0.731080f, 0.936205f, 1.361710f, 1.564157f, -1.259131f, -0.511293f, -0.506314f, 1.069145f, -3.204769f, -0.038035f, -0.251431f, -2.348164f, -0.391968f, -1.376128f, -0.959294f, 0.067650f, -0.650696f, -2.276250f, -1.167728f, -2.270346f, -1.075295f, 2.520546f, -1.441637f, -1.004393f, -1.774644f, -2.199297f, -0.669002f, -1.522764f, 2.384498f, 0.274234f, -1.878561f, -0.096679f, 1.979204f, 2.497787f, -2.921908f, 1.530090f, -0.018067f, 2.058079f, -0.166792f, -1.772430f, -1.215328f, 1.342235f, -1.152928f, -0.988702f, 0.722566f, -2.632075f, -1.298012f, -1.666589f, -1.965710f, -0.304563f, -2.430274f, -2.478807f, 0.284797f, -0.511265f, -1.317179f, 0.262856f, 0.791688f, 1.747900f, -0.758349f, 0.683765f, 0.061497f, 0.231747f, 2.311899f, 0.248775f, -2.213637f, 0.350800f, 1.853055f, 2.610889f, -3.009519f, -0.403499f, -4.668862f, 3.240747f, 0.530737f, 0.946793f, 0.520326f, 2.236615f, 0.314976f, 0.697927f, 1.297611f, 0.435956f, 1.759171f, -0.624841f, -1.634986f, 1.377374f, 1.992974f, 0.001277f, 0.018182f, 0.779973f, -1.779755f, 1.754702f, 0.455461f, 0.359283f, 0.623859f, -1.398301f, 0.738881f, 1.899348f, 0.931809f, 0.297294f, 3.314882f, -0.874802f, 1.014727f, -1.917428f, -0.489010f, 1.719898f, 1.551790f, 1.268495f, -0.559571f, 1.285836f, -1.233311f, -0.910606f, -2.753237f, 1.846477f, -0.149391f, 1.590832f, 0.623731f, -0.445114f, 1.809686f, 0.490612f, 1.020176f, -0.444144f, 1.151626f, 0.452243f, -1.385199f};

static const float test_5_out_scale = input_3_scale; //0.0148121407f;
static const float test_5_out_zero_point = 2 * input_3_zero_point; //-0.6769900322f;
static const int8_t* test_5_out_scales_frac = input_3_scales_frac; //[] = {21};
static const int test_5_out_sa_dim = -1;

static const int test_5_out_fx8_frac = input_3_fx8_frac; //4;

#define TEST_5_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 9, 4, 15}, \
    /* .rank =  */ 4


static const mli_tensor test_5_out_tsr_sa8 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_5_out_tsr_fx8 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_5_out_tsr_fx16 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_5_out_tsr_sa32 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_6_out_data[] = {1.718693f, -2.156846f, -4.098929f, 0.374849f, 3.237508f, 1.635359f, -2.063490f, -7.675720f, -0.637620f, -1.219739f, -2.119409f, 3.982372f, 4.787103f, -2.777408f, -2.500396f, -1.045107f, -1.996328f, -3.306064f, 0.436799f, 1.446171f, 1.489875f, -2.552640f, -1.425160f, 2.212006f, -1.877483f, 0.960415f, -2.787334f, -4.429412f, 1.995563f, 1.096392f, 4.508386f, -1.184809f, 3.603678f, -0.331452f, 3.943500f, 4.691895f, 0.249844f, -3.460968f, -4.127361f, 1.836922f, 4.407874f, -1.547406f, 0.768747f, -3.045654f, -0.122104f, -3.279730f, -0.720067f, -1.109803f, -1.895646f, -5.015722f, -0.671613f, 3.852948f, -2.061430f, -2.436765f, -0.883994f, -2.225420f, 2.595149f, 2.086603f, -6.971809f, 0.558777f, -4.427082f, -2.958861f, 4.313403f, 4.241808f, 4.572990f, -3.662713f, 1.062590f, -4.910040f, -2.807753f, -1.007273f, -1.849616f, 1.338105f, -1.018598f, -2.280023f, 0.722048f, -0.255343f, 1.066671f, -0.452180f, 0.880192f, -2.767972f, 0.100063f, -2.460425f, -1.695762f, 0.906227f, 1.434818f, 1.610349f, 4.673390f, -4.546101f, 0.538920f, 6.168218f, 1.425478f, -3.213965f, -2.610528f, -2.059594f, 1.390134f, 0.758860f, -1.974292f, 2.921604f, -0.397054f, 1.648243f, 4.422281f, -3.591679f, 0.551220f, 3.449567f, -3.237339f, -6.416937f, 1.706796f, -2.088973f, 2.710159f, 0.223697f, 2.849523f, -4.857301f, -1.314798f, 2.644274f, 1.165682f, 1.493946f, 3.310841f, -1.130813f, 6.097398f, -0.902329f};

static const float test_6_out_scale = input_3_scale;// 0.1163356155f;
static const float test_6_out_zero_point = 9 * input_3_zero_point; //-0.7537512779f;
static const int8_t* test_6_out_scales_frac = input_3_scales_frac; //[] = {18};
static const int test_6_out_sa_dim = -1;

static const int test_6_out_fx8_frac = input_3_fx8_frac; //4;

#define TEST_6_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 1, 4, 15}, \
    /* .rank =  */ 4


static const mli_tensor test_6_out_tsr_sa8 = {
    TEST_6_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_6_out_tsr_fx8 = {
    TEST_6_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_6_out_tsr_fx16 = {
    TEST_6_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_6_out_tsr_sa32 = {
    TEST_6_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_7_out_data[] = {-2.317638f, -1.775586f, 1.396713f, -1.924762f, 4.474327f, -0.449129f, 0.052602f, 0.369447f, 2.132541f, -3.498059f, -0.044885f, 2.282997f, 2.077049f, -5.825384f, 4.737879f, -3.174224f, 1.105538f, 3.498552f};

static const float test_7_out_scale = input_4_scale; //0.6213684082f;
static const float test_7_out_zero_point = 7 * input_4_zero_point; //-0.5437521935f;
static const int8_t* test_7_out_scales_frac = input_4_scales_frac; //[] = {15};
static const int test_7_out_sa_dim = -1;

static const int test_7_out_fx8_frac = input_4_fx8_frac; //4;

#define TEST_7_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 6, 3}, \
    /* .rank =  */ 3


static const mli_tensor test_7_out_tsr_sa8 = {
    TEST_7_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_7_out_tsr_fx8 = {
    TEST_7_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_7_out_tsr_fx16 = {
    TEST_7_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_7_out_tsr_sa32 = {
    TEST_7_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_8_out_data[] = {-3.603724f, -1.091420f, -0.778463f, -0.220502f, 0.430196f, -4.872665f, 1.126435f, 0.931880f, 2.659084f, -0.146076f, -0.624763f, -0.796672f, -1.022434f, -0.538799f, 0.947563f, -0.860006f, 0.330717f, 0.962101f, -0.618157f, 1.603038f, -2.387118f, 1.599357f, -1.266856f, 0.411044f, 2.213956f, 0.033310f, 2.302332f, 1.324389f, 1.683506f, 0.597059f, -1.292554f, 2.372567f, 1.343296f, -0.924733f, -0.284424f, 2.463256f, 0.499967f, -1.210142f, -1.532103f, -2.032376f, 0.721168f, 2.665741f};

static const float test_8_out_scale = input_4_scale; //0.1838635653f;
static const float test_8_out_zero_point = 3 * input_4_zero_point; //-1.1034617424f;
static const int8_t* test_8_out_scales_frac = input_4_scales_frac; //[] = {17};
static const int test_8_out_sa_dim = -1;

static const int test_8_out_fx8_frac = input_4_fx8_frac; //4;

#define TEST_8_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {7, 6, 1}, \
    /* .rank =  */ 3


static const mli_tensor test_8_out_tsr_sa8 = {
    TEST_8_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_8_out_tsr_fx8 = {
    TEST_8_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_8_out_tsr_fx16 = {
    TEST_8_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_8_out_tsr_sa32 = {
    TEST_8_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);
tensor_quantizer input_1_fx8(input_1_tsr_fx8, input_1_fx8_frac, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_sa32(input_1_tsr_sa32, input_1_sa_dim, input_1_data, 
sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data, 
sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);
tensor_quantizer input_2_fx8(input_2_tsr_fx8, input_2_fx8_frac, input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]));
tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx8_frac + 8, input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]));
tensor_quantizer input_2_sa32(input_2_tsr_sa32, input_2_sa_dim, input_2_data, 
sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);

tensor_quantizer input_3_sa8(input_3_tsr_sa8, input_3_sa_dim, input_3_data, 
sizeof(input_3_data) / sizeof(input_3_data[0]), &input_3_scale, 1, &input_3_zero_point, 1, input_3_scales_frac, 1);
tensor_quantizer input_3_fx8(input_3_tsr_fx8, input_3_fx8_frac, input_3_data, sizeof(input_3_data) / sizeof(input_3_data[0]));
tensor_quantizer input_3_fx16(input_3_tsr_fx16, input_3_fx8_frac + 8, input_3_data, sizeof(input_3_data) / sizeof(input_3_data[0]));
tensor_quantizer input_3_sa32(input_3_tsr_sa32, input_3_sa_dim, input_3_data, 
sizeof(input_3_data) / sizeof(input_3_data[0]), &input_3_scale, 1, &input_3_zero_point, 1, input_3_scales_frac, 1);

tensor_quantizer input_4_sa8(input_4_tsr_sa8, input_4_sa_dim, input_4_data, 
sizeof(input_4_data) / sizeof(input_4_data[0]), &input_4_scale, 1, &input_4_zero_point, 1, input_4_scales_frac, 1);
tensor_quantizer input_4_fx8(input_4_tsr_fx8, input_4_fx8_frac, input_4_data, sizeof(input_4_data) / sizeof(input_4_data[0]));
tensor_quantizer input_4_fx16(input_4_tsr_fx16, input_4_fx8_frac + 8, input_4_data, sizeof(input_4_data) / sizeof(input_4_data[0]));
tensor_quantizer input_4_sa32(input_4_tsr_sa32, input_4_sa_dim, input_4_data, 
sizeof(input_4_data) / sizeof(input_4_data[0]), &input_4_scale, 1, &input_4_zero_point, 1, input_4_scales_frac, 1);

tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim, test_1_out_data, 
sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_1_out_scale, 1, &test_1_out_zero_point, 1, test_1_out_scales_frac, 1);
tensor_quantizer test_1_out_fx8(test_1_out_tsr_fx8, test_1_out_fx8_frac, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_sa32(test_1_out_tsr_sa32, test_1_out_sa_dim, test_1_out_data, 
sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_1_out_scale, 1, &test_1_out_zero_point, 1, test_1_out_scales_frac, 1);

tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_sa_dim, test_2_out_data, 
sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &test_2_out_scale, 1, &test_2_out_zero_point, 1, test_2_out_scales_frac, 1);
tensor_quantizer test_2_out_fx8(test_2_out_tsr_fx8, test_2_out_fx8_frac, test_2_out_data, sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_fx8_frac + 8, test_2_out_data, sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_sa32(test_2_out_tsr_sa32, test_2_out_sa_dim, test_2_out_data, 
sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &test_2_out_scale, 1, &test_2_out_zero_point, 1, test_2_out_scales_frac, 1);

tensor_quantizer test_3_out_sa8(test_3_out_tsr_sa8, test_3_out_sa_dim, test_3_out_data, 
sizeof(test_3_out_data) / sizeof(test_3_out_data[0]), &test_3_out_scale, 1, &test_3_out_zero_point, 1, test_3_out_scales_frac, 1);
tensor_quantizer test_3_out_fx8(test_3_out_tsr_fx8, test_3_out_fx8_frac, test_3_out_data, sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));
tensor_quantizer test_3_out_fx16(test_3_out_tsr_fx16, test_3_out_fx8_frac + 8, test_3_out_data, sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));
tensor_quantizer test_3_out_sa32(test_3_out_tsr_sa32, test_3_out_sa_dim, test_3_out_data, 
sizeof(test_3_out_data) / sizeof(test_3_out_data[0]), &test_3_out_scale, 1, &test_3_out_zero_point, 1, test_3_out_scales_frac, 1);

tensor_quantizer test_4_out_sa8(test_4_out_tsr_sa8, test_4_out_sa_dim, test_4_out_data, 
sizeof(test_4_out_data) / sizeof(test_4_out_data[0]), &test_4_out_scale, 1, &test_4_out_zero_point, 1, test_4_out_scales_frac, 1);
tensor_quantizer test_4_out_fx8(test_4_out_tsr_fx8, test_4_out_fx8_frac, test_4_out_data, sizeof(test_4_out_data) / sizeof(test_4_out_data[0]));
tensor_quantizer test_4_out_fx16(test_4_out_tsr_fx16, test_4_out_fx8_frac + 8, test_4_out_data, sizeof(test_4_out_data) / sizeof(test_4_out_data[0]));
tensor_quantizer test_4_out_sa32(test_4_out_tsr_sa32, test_4_out_sa_dim, test_4_out_data, 
sizeof(test_4_out_data) / sizeof(test_4_out_data[0]), &test_4_out_scale, 1, &test_4_out_zero_point, 1, test_4_out_scales_frac, 1);

tensor_quantizer test_5_out_sa8(test_5_out_tsr_sa8, test_5_out_sa_dim, test_5_out_data, 
sizeof(test_5_out_data) / sizeof(test_5_out_data[0]), &test_5_out_scale, 1, &test_5_out_zero_point, 1, test_5_out_scales_frac, 1);
tensor_quantizer test_5_out_fx8(test_5_out_tsr_fx8, test_5_out_fx8_frac, test_5_out_data, sizeof(test_5_out_data) / sizeof(test_5_out_data[0]));
tensor_quantizer test_5_out_fx16(test_5_out_tsr_fx16, test_5_out_fx8_frac + 8, test_5_out_data, sizeof(test_5_out_data) / sizeof(test_5_out_data[0]));
tensor_quantizer test_5_out_sa32(test_5_out_tsr_sa32, test_5_out_sa_dim, test_5_out_data, 
sizeof(test_5_out_data) / sizeof(test_5_out_data[0]), &test_5_out_scale, 1, &test_5_out_zero_point, 1, test_5_out_scales_frac, 1);

tensor_quantizer test_6_out_sa8(test_6_out_tsr_sa8, test_6_out_sa_dim, test_6_out_data, 
sizeof(test_6_out_data) / sizeof(test_6_out_data[0]), &test_6_out_scale, 1, &test_6_out_zero_point, 1, test_6_out_scales_frac, 1);
tensor_quantizer test_6_out_fx8(test_6_out_tsr_fx8, test_6_out_fx8_frac, test_6_out_data, sizeof(test_6_out_data) / sizeof(test_6_out_data[0]));
tensor_quantizer test_6_out_fx16(test_6_out_tsr_fx16, test_6_out_fx8_frac + 8, test_6_out_data, sizeof(test_6_out_data) / sizeof(test_6_out_data[0]));
tensor_quantizer test_6_out_sa32(test_6_out_tsr_sa32, test_6_out_sa_dim, test_6_out_data, 
sizeof(test_6_out_data) / sizeof(test_6_out_data[0]), &test_6_out_scale, 1, &test_6_out_zero_point, 1, test_6_out_scales_frac, 1);

tensor_quantizer test_7_out_sa8(test_7_out_tsr_sa8, test_7_out_sa_dim, test_7_out_data, 
sizeof(test_7_out_data) / sizeof(test_7_out_data[0]), &test_7_out_scale, 1, &test_7_out_zero_point, 1, test_7_out_scales_frac, 1);
tensor_quantizer test_7_out_fx8(test_7_out_tsr_fx8, test_7_out_fx8_frac, test_7_out_data, sizeof(test_7_out_data) / sizeof(test_7_out_data[0]));
tensor_quantizer test_7_out_fx16(test_7_out_tsr_fx16, test_7_out_fx8_frac + 8, test_7_out_data, sizeof(test_7_out_data) / sizeof(test_7_out_data[0]));
tensor_quantizer test_7_out_sa32(test_7_out_tsr_sa32, test_7_out_sa_dim, test_7_out_data, 
sizeof(test_7_out_data) / sizeof(test_7_out_data[0]), &test_7_out_scale, 1, &test_7_out_zero_point, 1, test_7_out_scales_frac, 1);

tensor_quantizer test_8_out_sa8(test_8_out_tsr_sa8, test_8_out_sa_dim, test_8_out_data, 
sizeof(test_8_out_data) / sizeof(test_8_out_data[0]), &test_8_out_scale, 1, &test_8_out_zero_point, 1, test_8_out_scales_frac, 1);
tensor_quantizer test_8_out_fx8(test_8_out_tsr_fx8, test_8_out_fx8_frac, test_8_out_data, sizeof(test_8_out_data) / sizeof(test_8_out_data[0]));
tensor_quantizer test_8_out_fx16(test_8_out_tsr_fx16, test_8_out_fx8_frac + 8, test_8_out_data, sizeof(test_8_out_data) / sizeof(test_8_out_data[0]));
tensor_quantizer test_8_out_sa32(test_8_out_tsr_sa32, test_8_out_sa_dim, test_8_out_data, 
sizeof(test_8_out_data) / sizeof(test_8_out_data[0]), &test_8_out_scale, 1, &test_8_out_zero_point, 1, test_8_out_scales_frac, 1);

const lib_mli::ReduceOpConfig test_1_cfg = {/* .reduce_axis = */0};
const lib_mli::ReduceOpConfig test_2_cfg = {/* .reduce_axis = */1};
const lib_mli::ReduceOpConfig test_3_cfg = {/* .reduce_axis = */2};
const lib_mli::ReduceOpConfig test_4_cfg = {/* .reduce_axis = */3};
