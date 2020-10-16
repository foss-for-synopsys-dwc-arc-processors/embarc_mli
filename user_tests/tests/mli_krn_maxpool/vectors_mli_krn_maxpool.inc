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
extern mli::tst::tensor_quantizer input_1_memstr_fx16;
extern mli::tst::tensor_quantizer input_1_memstr_sa8;
extern mli::tst::tensor_quantizer input_2_fx16;
extern mli::tst::tensor_quantizer input_2_sa8;
extern mli::tst::tensor_quantizer input_2_memstr_fx16;
extern mli::tst::tensor_quantizer input_2_memstr_sa8;

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

// Tests configuration structures Declaration
//========================================
extern const mli_pool_cfg test_1_cfg;
extern const mli_pool_cfg test_2_cfg;
extern const mli_pool_cfg test_3_cfg;
extern const mli_pool_cfg test_4_cfg;
extern const mli_pool_cfg test_5_cfg;
extern const mli_pool_cfg test_6_cfg;
extern const mli_pool_cfg test_7_cfg;

// Generated input vectors 
//========================================
static const float input_1_data[] = {
    -1.0208817f, -0.8298214f, -1.932184f, 0.6934353f, -1.7505208f, -1.246126f, -0.53347284f, -1.4600027f, 
    -1.3607147f, -1.281774f, -0.50772935f, 4.1487026f, -0.77798414f, -1.3925005f, -0.5142991f, -1.1159647f, 
    -0.8101392f, -0.17724936f, -1.1963716f, -0.7952166f, -1.2653972f, -0.88291013f, -0.6640177f, -1.3240529f, 
    -0.97957563f, -0.9251775f, -1.1379572f, -1.4294345f, -1.5532452f, -0.8813524f, -0.9426002f, 3.8262439f, 
    -2.090427f, -0.74025494f, -0.6036644f, -0.07939405f, -0.53401566f, -1.2474333f, -0.7999948f, -0.51255316f, 
    -0.96615934f, -0.39136377f, -1.790088f, 1.1752788f, -0.022130655f, -0.97790897f, -0.4759894f, -0.7745756f, 
    -1.2394418f, -0.7535136f, -0.2698061f, -1.001979f, -0.9991048f, -0.8394695f, -0.57024693f, -0.12245901f, 
    -1.1363801f, -0.53055f, -0.68545854f, -1.7850646f, -1.0222378f, -0.8701343f, -0.8209722f, -0.6272182f, 
    -1.8813713f, -0.77548456f, -1.159168f, -1.5629027f, -0.5221229f, -0.41030183f, -1.1462226f, -0.35349816f, 
    -1.0022037f, -0.76476085f, -1.3823316f, 0.3977814f, -2.2095082f, -0.8484699f, -0.43158662f, -1.8777373f, 
    -0.9004187f, -0.8409259f, -0.3629074f, -0.13844715f, -1.0140629f, -1.5337949f, -0.07536522f, -0.15452062f, 
    -1.0448629f, -0.6140291f, -0.5976109f, -0.49987403f, -0.6917698f, -0.054450992f, -1.7211186f, -0.2483633f, 
    -0.59834987f, -0.9150817f, -0.1780283f, -2.0421097f, -0.18439361f, -0.59537864f, -1.4005606f, -0.570461f, 
    -0.9611658f, 5.7020626f, 1.7272635f, -0.5664344f, -1.1812046f, -1.1092098f, -0.97227633f, -1.0969243f, 
    -1.2781904f, -1.1132153f, -0.94637513f, -1.0739485f, -1.1702527f, -1.2923127f, 5.5173225f, -1.5807818f, 
    -1.3413498f, -0.5708303f, -0.9714974f, -0.57537293f, -0.3764346f, -0.13081476f, -0.40145862f, -1.3266933f, 
    -0.21215406f, -1.7354064f, -0.86064327f, -1.0640323f, -1.3622235f, -0.48641676f, -0.89712596f, -0.259477f, 
    -1.5857067f, -0.3514547f, -0.53609174f, -0.7329528f, -1.5519696f, -0.93344027f, -1.6197375f, -1.9038055f, 
    9.38265f, -0.2854435f, 1.5007421f, -1.027377f, -0.43101507f, -0.43036938f, -0.4811966f, -0.7980605f, 
    -0.9209712f, -1.8968945f, -0.47662362f, -0.5755466f, -0.20752004f, -0.9453483f, 1.2316046f, -1.0371754f, 
    -1.4653764f, -0.30277327f, -1.0482235f, -1.4650018f, -0.73384345f, -0.39905387f, -0.672037f, -0.5363644f, 
    13.776071f, -0.36480594f, 0.118329674f, -1.8524983f, -0.60652167f, -0.38973653f, -0.9931739f, -1.2642691f, 
    -0.5898443f, -0.9667472f, -1.1146904f, -0.1907647f, -0.09214993f, -0.6396065f, -1.0407407f, -0.5273835f, 
    -0.36038703f, -0.46577036f, -1.570283f, -0.9791494f, -0.6096593f, -0.4702232f, -0.29078558f, -0.39806372f, 
    -1.4011889f, -1.1749331f, -1.4473574f, 1.5637262f, -0.7043828f, -0.50682294f, -0.2524946f, -0.9743713f, 
    -1.4100316f, -1.0260569f, -0.878518f, -0.004238435f, -0.33911297f, -1.376158f, -0.44796264f, -0.41051114f, 
    -1.3625238f, -0.8575139f, -1.7807142f, -0.62484133f, -0.9673174f, -1.5155011f, -1.1700306f, -0.6728259f, 
    -1.0930632f, -0.6412508f, -0.50708824f, -1.2454348f, -0.96550345f, 3.3667939f, -0.73581403f, -1.2488192f, 
    -0.7679549f, -1.5094521f, -0.19964458f, -0.6037384f, -0.2245226f, -0.3218449f, -0.59314287f, -0.28298002f, 
    -0.69846594f, -1.1509708f, -1.3424629f, -1.0375327f, -2.1278927f, 1.6777706f, -1.7084557f, -0.84061897f, 
    -1.4983107f, -1.8735646f, -0.84359f, -0.9614733f, -0.8846538f, -0.5090583f, -1.127739f, -0.59923893f, 
    -0.52797675f, -0.97746736f, -0.95026535f, -0.51463515f, -0.769657f, -0.7156831f, -2.0584335f, -0.76898444f, 
    -0.60043067f, -0.8718043f, -0.62196857f, 2.0931923f, -0.75475687f, -0.44737282f, -1.2504976f, -1.2517412f, 
    -1.0543921f, -1.340414f, -0.80402863f, -1.2949302f, -0.89274836f, -1.561686f
};

static const int input_1_fx8_frac = 3;

static const int input_1_sa_dim = -1;
static const float input_1_scale = 0.06268854439258575f;
static const float input_1_zero_point = 5.783281326293945f;
static const int8_t input_1_scales_frac[] = { 17 };


#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ {0},\
    /* .mem_stride =  */ {0},\
    /* .shape = */ {10, 9, 3}, \
    /* .rank = */ 3

#define INPUT_1_MEMSTR_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /* .mem_stride = */ { 10*5*1, 5*1, 1}, \
    /* .shape = */ {10, 9, 3}, \
    /* .rank = */  3

static const mli_tensor input_1_tsr_fx16 = {
    INPUT_1_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_FX_16,
    /* .el_params = */ {0},
};

static const mli_tensor input_1_tsr_sa8 = {
    INPUT_1_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_SA_8,
    /* .el_params = */ {0},
};

static const mli_tensor input_1_memstr_tsr_sa8 = {
    INPUT_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_1_memstr_tsr_fx16 = {
    INPUT_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};
static const float input_2_data[] = {
    0.4465427f, -0.056221493f, -0.018236352f, -0.15031485f, -0.06928342f, -0.26181605f, 2.066586f, 1.8027046f, 
    -0.032395303f, 3.1937637f, 0.27961677f, 1.6054269f, 1.086699f, -0.016432973f, 3.4857635f, 3.0043883f, 
    2.9626255f, 3.630303f, -0.39707074f, 0.9035027f, 3.5336874f, -0.24137643f, -0.110906914f, 0.9798653f, 
    2.002687f, -0.00734325f, -0.03848969f, 2.5772076f, -0.36123234f, 2.6362073f, 0.52838767f, 1.7111558f, 
    2.162468f, -0.283807f, 3.3326924f, 2.8516436f, 0.21475048f, -0.1701427f, -0.3258896f, -0.03584147f, 
    2.2078326f, 2.7977722f, -0.0146258045f, -0.14005794f, -0.013674864f, 3.6897643f, -0.286613f, -0.20862353f,
     2.0960107f, -0.23507906f, -0.08538145f, 2.7387376f, -0.045278095f, 1.5223976f, 1.3795925f, -0.0867767f, 
    -0.10649354f, -0.05818292f, 3.1680465f, -0.37786537f, -0.29186904f, 0.013138759f, -0.14004405f, 
    1.3208586f, 1.1828204f, 0.9226358f, -0.045111377f, 1.0175475f, 2.6387482f, 2.793783f, 1.1437613f, 
    2.7573779f, -0.35955495f, -0.21820198f, 2.4142916f, 0.2707324f, -0.024879782f, 1.374852f, 0.19728103f, 
    3.3662264f, 0.28313023f, 2.9338703f, 1.550406f, 3.3809109f, 0.8559161f, 3.655233f, 1.402587f, 
    -0.36726174f, 2.359338f, 3.0822647f, -0.283711f, 3.684314f, 1.3242136f, 2.3701413f, -0.13567844f, 
    3.5529766f, 1.795638f, 1.20558f, 3.672202f, 1.2861155f, 1.1368436f, -0.34717497f, 1.2236949f, 2.3917294f,
    0.17357905f, 2.0080235f, 1.3282361f, -0.15458947f, 1.2262021f, 0.8083336f, -0.0014908476f, 1.9095477f 
};

static const int input_2_fx8_frac = 5;

static const int input_2_sa_dim = -1;
static const float input_2_scale = 0.016026804223656654f;
static const float input_2_zero_point = 1.6463468074798584f;
static const int8_t input_2_scales_frac[] = { 19 };


#define INPUT_2_TSR_SHARED_DESCR \
    /* .data = */ {0},\
    /* .mem_stride =  */ {0},\
    /* .shape = */ {4, 4, 7}, \
    /* .rank = */ 3

#define INPUT_2_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ {0},\
    /* .mem_stride =  */ {8*8*1, 8*1, 1},\
    /* .shape = */ {4, 4, 7}, \
    /* .rank = */ 3

static const mli_tensor input_2_tsr_fx16 = {
    INPUT_2_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_FX_16,
    /* .el_params = */ {0},
};


static const mli_tensor input_2_tsr_sa8 = {
    INPUT_2_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_SA_8,
    /* .el_params = */ {0},
};

static const mli_tensor input_2_memstr_tsr_sa8 = {
    INPUT_2_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_2_memstr_tsr_fx16 = {
    INPUT_2_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

// Extracted Output vectors 
//===================================================
static const float test_1_out_data[] = {
    0.6934353f, 3.8262439f, 4.1487026f, 0.6934353f, 3.8262439f, 4.1487026f, -0.41030183f, 1.1752788f, 
    4.1487026f, -0.41030183f, 1.1752788f, 4.1487026f, 0.3977814f, 1.1752788f, -0.022130655f, 0.3977814f, 
    1.1752788f, -0.022130655f, -0.15452062f, 3.8262439f, -0.07536522f, -0.054450992f, 3.8262439f, 
    -0.07536522f, -0.054450992f, 1.1752788f, -0.022130655f, -0.054450992f, 1.1752788f, -0.022130655f, 
    0.3977814f, 1.1752788f, -0.022130655f, 5.7020626f, 1.7272635f, -0.022130655f, -0.15452062f, 5.5173225f, 
    -0.07536522f, -0.054450992f, 5.5173225f, -0.07536522f, -0.054450992f, 5.5173225f, -0.13081476f, 
    -0.054450992f, 5.5173225f, -0.13081476f, 0.3977814f, -0.18439361f, -0.13081476f, 5.7020626f, 1.7272635f, 
    -0.13081476f, 9.38265f, 5.5173225f, 1.5007421f, 9.38265f, 5.5173225f, 1.5007421f, 9.38265f, 5.5173225f, 
    1.5007421f, 9.38265f, 5.5173225f, 1.5007421f, -0.054450992f, -0.18439361f, 1.2316046f, 5.7020626f, 
    1.7272635f, 1.2316046f, 13.776071f, 5.5173225f, 1.5007421f, 13.776071f, 5.5173225f, 1.5007421f, 
    13.776071f, 5.5173225f, 1.5007421f, 9.38265f, 5.5173225f, 1.5007421f, -0.09214993f, -0.36038703f, 
    1.2316046f, -0.09214993f, -0.36038703f, 1.2316046f, 13.776071f, -0.2854435f, 1.5007421f, 13.776071f, 
    -0.2854435f, 1.5007421f, 13.776071f, -0.2854435f, 1.5007421f, 9.38265f, -0.2854435f, 1.5007421f, 
    -0.09214993f, -0.36038703f, 1.2316046f, -0.09214993f, -0.36038703f, 1.2316046f, 13.776071f, -0.19964458f,
    3.3667939f, 13.776071f, -0.19964458f, 3.3667939f, 13.776071f, -0.19964458f, 0.118329674f, -0.09214993f, 
    -0.19964458f, -0.004238435f, 1.6777706f, -0.3218449f, -0.004238435f, 1.6777706f, -0.36038703f, 
    -0.1907647f, 1.5637262f, -0.19964458f, 3.3667939f, 1.5637262f, -0.19964458f, 3.3667939f, 1.5637262f, 
    2.0931923f, -0.004238435f, -0.2245226f, 2.0931923f, -0.004238435f, 1.6777706f, 2.0931923f, -0.004238435f, 
    1.6777706f, 2.0931923f, -0.44796264f
};

static const int test_1_out_fx8_frac = 3;

static const int test_1_out_sa_dim = -1;
static const float test_1_out_scale = input_1_scale;
static const float test_1_out_zero_point = input_1_zero_point;
static const int8_t* test_1_out_scales_frac = input_1_scales_frac;


#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ {0},\
    /* .mem_stride =  */ {0},\
    /* .shape = */ {8, 6, 3}, \
    /* .rank = */ 3



static const mli_tensor test_1_out_tsr_fx16 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_FX_16,
    /* .el_params = */ {0},
};


static const mli_tensor test_1_out_tsr_sa8 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_SA_8,
    /* .el_params = */ {0},
};

static const float test_2_out_data[] = {
    0.6934353f, 3.8262439f, -0.8813524f, 0.6934353f, 3.8262439f, 4.1487026f, -0.41030183f, 1.1752788f, 
    4.1487026f, 0.3977814f, 1.1752788f, -0.022130655f, 0.3977814f, -0.6640177f, -0.2698061f, -0.53055f, 
    3.8262439f, -0.07536522f, -0.15452062f, 5.5173225f, -0.07536522f, -0.054450992f, 5.5173225f, 
    -0.022130655f, 0.3977814f, 1.1752788f, -0.022130655f, 5.7020626f, 1.7272635f, -0.2698061f, -0.259477f, 
    -0.3629074f, -0.07536522f, 13.776071f, 5.5173225f, 1.5007421f, 9.38265f, 5.5173225f, 1.5007421f, 
    -0.09214993f, -0.18439361f, 1.2316046f, 5.7020626f, 1.7272635f, 1.2316046f, -0.259477f, -0.29078558f, 
    3.3667939f, 13.776071f, -0.19964458f, 3.3667939f, 9.38265f, -0.19964458f, 1.5007421f, 1.6777706f, 
    -0.36038703f, 1.2316046f, 1.6777706f, -0.36038703f, 1.2316046f, -0.4702232f, -0.29078558f, 3.3667939f, 
    1.5637262f, -0.19964458f, 3.3667939f, -0.2245226f, 2.0931923f, -0.004238435f, 1.6777706f, 2.0931923f, 
    -0.44796264f, 1.6777706f, -0.62484133f, -0.6728259f 
};

static const int test_2_out_fx8_frac = 3;

static const int test_2_out_sa_dim = -1;
static const float test_2_out_scale = input_1_scale;
static const float test_2_out_zero_point = input_1_zero_point;
static const int8_t* test_2_out_scales_frac = input_1_scales_frac;


#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ {0},\
    /* .mem_stride =  */ {0},\
    /* .shape = */ {5, 5, 3}, \
    /* .rank = */ 3



static const mli_tensor test_2_out_tsr_fx16 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_FX_16,
    /* .el_params = */ {0},
};


static const mli_tensor test_2_out_tsr_sa8 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_SA_8,
    /* .el_params = */ {0},
};

static const float test_3_out_data[] = {
    0.6934353f, 3.8262439f, -0.07939405f, -0.41030183f, 1.1752788f, 4.1487026f, 0.3977814f, -0.4759894f, -0.2698061f,
     -0.15452062f, -0.12245901f, -0.07536522f, 9.38265f, 5.5173225f, 1.5007421f, 5.7020626f, 1.7272635f, 
    1.2316046f, 13.776071f, -0.29078558f, 3.3667939f, 9.38265f, -0.19964458f, 1.5007421f, 1.6777706f, -0.36038703f,
     1.2316046f, -0.73581403f, -0.59923893f, 3.3667939f, -0.2245226f, 2.0931923f, -0.59314287f, 1.6777706f, 
    -0.89274836f, -0.80402863f
};

static const int test_3_out_fx8_frac = 3;

static const int test_3_out_sa_dim = -1;
static const float test_3_out_scale = input_1_scale;
static const float test_3_out_zero_point = input_1_zero_point;
static const int8_t* test_3_out_scales_frac = input_1_scales_frac;


#define TEST_3_OUT_TSR_SHARED_DESCR \
    /* .data = */ {0},\
    /* .mem_stride =  */ {4*6*1, 6*1, 1},\
    /* .shape = */ {4, 3, 3}, \
    /* .rank = */ 3



static const mli_tensor test_3_out_tsr_fx16 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_FX_16,
    /* .el_params = */ {0},
};


static const mli_tensor test_3_out_tsr_sa8 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_SA_8,
    /* .el_params = */ {0},
};

static const float test_4_out_data[] = {
    3.684314f, 3.655233f, 3.3662264f, 3.6897643f, 3.5529766f, 3.0822647f, 3.5336874f
};

static const int test_4_out_fx8_frac = 5;

static const int test_4_out_sa_dim = -1;
static const float test_4_out_scale = input_2_scale;
static const float test_4_out_zero_point = input_2_zero_point;
static const int8_t* test_4_out_scales_frac = input_2_scales_frac;


#define TEST_4_OUT_TSR_SHARED_DESCR \
    /* .data = */ {0},\
    /* .mem_stride =  */ {0},\
    /* .shape = */ {1, 1, 7}, \
    /* .rank = */ 3



static const mli_tensor test_4_out_tsr_fx16 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_FX_16,
    /* .el_params = */ {0},
};


static const mli_tensor test_4_out_tsr_sa8 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_SA_8,
    /* .el_params = */ {0},
};

static const float test_5_out_data[] = {
    3.4857635f, 3.0043883f, 3.1937637f, 3.6897643f, 2.4142916f, 2.6387482f, 3.5336874f, 3.4857635f, 3.0043883f, 
    3.3662264f, 3.6897643f, 2.9338703f, 2.6387482f, 3.5336874f, 3.684314f, 3.655233f, 3.1680465f, 3.6897643f, 
    3.5529766f, 3.0822647f, 3.3326924f, 3.684314f, 2.7573779f, 3.3662264f, 3.6897643f, 3.5529766f, 2.6387482f, 
    3.3809109f
};

static const int test_5_out_fx8_frac = 5;

static const int test_5_out_sa_dim = -1;
static const float test_5_out_scale = input_2_scale;
static const float test_5_out_zero_point = input_2_zero_point;
static const int8_t* test_5_out_scales_frac = input_2_scales_frac;



#define TEST_5_OUT_TSR_SHARED_DESCR \
    /* .data = */ {0},\
    /* .mem_stride =  */ {3*8*1, 8*1, 1},\
    /* .shape = */ {2, 2, 7}, \
    /* .rank = */ 3



static const mli_tensor test_5_out_tsr_fx16 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_FX_16,
    /* .el_params = */ {0},
};


static const mli_tensor test_5_out_tsr_sa8 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_SA_8,
    /* .el_params = */ {0},
};

static const float test_6_out_data[] = {
    0.6934353f, 3.8262439f, -0.8813524f, -0.53347284f, -0.50772935f, 4.1487026f, -0.51255316f, 1.1752788f, 
    -0.022130655f, -0.88291013f, -0.4759894f, -0.2698061f, -0.97957563f, -0.9251775f, -0.8394695f, -0.53055f, 
    -0.12245901f, -0.07536522f, -0.15452062f, -0.49987403f, -0.6140291f, -0.054450992f, -0.9150817f, -0.1780283f, 
    0.3977814f, -0.18439361f, -0.59537864f, 5.7020626f, 1.7272635f, -0.5664344f, -0.259477f, -0.7329528f, 
    -0.3514547f, 9.38265f, 5.5173225f, 1.5007421f, -0.4811966f, -0.3764346f, -0.13081476f, -0.20752004f, 
    -0.47662362f, 1.2316046f, -1.0371754f, -0.48641676f, -0.30277327f, -0.39905387f, -0.29078558f, -0.39806372f, 
    13.776071f, -0.36480594f, 0.118329674f, -0.33911297f, -0.878518f, -0.004238435f, -0.09214993f, -0.36038703f, 
    -0.46577036f, -1.5155011f, -0.9791494f, -0.6096593f, -0.9614733f, -0.59923893f, 3.3667939f, -0.73581403f, 
    -0.19964458f, -0.51463515f, -0.2245226f, 2.0931923f, -0.59314287f, 1.6777706f, -1.0375327f, -0.80402863f, 
    -1.2949302f, -0.89274836f, -0.84359f
};

static const int test_6_out_fx8_frac = 3;

static const int test_6_out_sa_dim = -1;
static const float test_6_out_scale = input_1_scale;
static const float test_6_out_zero_point = input_1_zero_point;
static const int8_t* test_6_out_scales_frac = input_1_scales_frac;


#define TEST_6_OUT_TSR_SHARED_DESCR \
    /* .data = */ {0},\
    /* .mem_stride =  */ {7*7*1, 7*1, 1},\
    /* .shape = */ {5, 5, 3}, \
    /* .rank = */ 3



static const mli_tensor test_6_out_tsr_fx16 = {
    TEST_6_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_FX_16,
    /* .el_params = */ {0},
};


static const mli_tensor test_6_out_tsr_sa8 = {
    TEST_6_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_SA_8,
    /* .el_params = */ {0},
};

static const float test_7_out_data[] = {
    0.6934353f, 3.8262439f, -0.8813524f, 0.6934353f, 3.8262439f, 4.1487026f, -0.41030183f, 1.1752788f, 
    4.1487026f, 0.3977814f, 1.1752788f, -0.022130655f, 0.3977814f, -0.6640177f, -0.2698061f, -0.53055f, 
    -0.12245901f, -0.07536522f, -0.15452062f, 5.5173225f, -0.07536522f, -0.054450992f, 5.5173225f, 
    -0.13081476f, 0.3977814f, -0.18439361f, -0.13081476f, 5.7020626f, 1.7272635f, -0.5664344f, -0.259477f, 
    -0.672037f, -0.3514547f, 13.776071f, 5.5173225f, 1.5007421f, 9.38265f, 5.5173225f, 1.5007421f, 
    -0.09214993f, -0.36038703f, 1.2316046f, -0.20752004f, -0.36038703f, 1.2316046f, -0.39905387f, -0.29078558f,
     3.3667939f, 13.776071f, -0.19964458f, 3.3667939f, -0.2245226f, -0.19964458f, -0.004238435f, 1.6777706f, 
    -0.36038703f, -0.1907647f, 1.6777706f, -0.36038703f, -0.46577036f, -0.9614733f, -0.59923893f, 3.3667939f, 
    -0.73581403f, -0.19964458f, 3.3667939f, -0.2245226f, 2.0931923f, -0.59314287f, 1.6777706f, 2.0931923f, 
    -0.75475687f, 1.6777706f, -0.89274836f, -0.80402863f
};

static const int test_7_out_fx8_frac = 3;

static const int test_7_out_sa_dim = -1;
static const float test_7_out_scale = input_1_scale;
static const float test_7_out_zero_point = input_1_zero_point;
static const int8_t* test_7_out_scales_frac = input_1_scales_frac;

#define TEST_7_OUT_TSR_SHARED_DESCR \
    /* .data = */ {0},\
    /* .mem_stride =  */ {7*7*1, 7*1, 1},\
    /* .shape = */ {5, 5, 3}, \
    /* .rank = */ 3



static const mli_tensor test_7_out_tsr_fx16 = {
    TEST_7_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_FX_16,
    /* .el_params = */ {0},
};


static const mli_tensor test_7_out_tsr_sa8 = {
    TEST_7_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_SA_8,
    /* .el_params = */ {0},
};

// Tensor quantizers 
//========================================
tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8,
    input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));


tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data,
    sizeof(input_1_data) / sizeof(input_1_data[0]),
    &input_1_scale, 1,
    &input_1_zero_point, 1,
    input_1_scales_frac, 1);

tensor_quantizer input_1_memstr_fx16(input_1_memstr_tsr_fx16, input_1_fx8_frac + 8,
    input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));

tensor_quantizer input_1_memstr_sa8(input_1_memstr_tsr_sa8, input_1_sa_dim, input_1_data,
    sizeof(input_1_data) / sizeof(input_1_data[0]),
    &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);


tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx8_frac + 8,
    input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]));


tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data,
    sizeof(input_2_data) / sizeof(input_2_data[0]),
    &input_2_scale, 1,
    &input_2_zero_point, 1,
    input_2_scales_frac, 1);

tensor_quantizer input_2_memstr_fx16(input_2_memstr_tsr_fx16, input_2_fx8_frac + 8,
    input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]));

tensor_quantizer input_2_memstr_sa8(input_2_memstr_tsr_sa8, input_2_sa_dim, input_2_data,
    sizeof(input_2_data) / sizeof(input_2_data[0]),
    &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);


tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8,
    test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));


tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim, test_1_out_data,
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]),
    &test_1_out_scale, 1,
    &test_1_out_zero_point, 1,
    test_1_out_scales_frac, 1);


tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_fx8_frac + 8,
    test_2_out_data, sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));


tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_sa_dim, test_2_out_data,
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]),
    &test_2_out_scale, 1,
    &test_2_out_zero_point, 1,
    test_2_out_scales_frac, 1);


tensor_quantizer test_3_out_fx16(test_3_out_tsr_fx16, test_3_out_fx8_frac + 8,
    test_3_out_data, sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));


tensor_quantizer test_3_out_sa8(test_3_out_tsr_sa8, test_3_out_sa_dim, test_3_out_data,
    sizeof(test_3_out_data) / sizeof(test_3_out_data[0]),
    &test_3_out_scale, 1,
    &test_3_out_zero_point, 1,
    test_3_out_scales_frac, 1);


tensor_quantizer test_4_out_fx16(test_4_out_tsr_fx16, test_4_out_fx8_frac + 8,
    test_4_out_data, sizeof(test_4_out_data) / sizeof(test_4_out_data[0]));


tensor_quantizer test_4_out_sa8(test_4_out_tsr_sa8, test_4_out_sa_dim, test_4_out_data,
    sizeof(test_4_out_data) / sizeof(test_4_out_data[0]),
    &test_4_out_scale, 1,
    &test_4_out_zero_point, 1,
    test_4_out_scales_frac, 1);


tensor_quantizer test_5_out_fx16(test_5_out_tsr_fx16, test_5_out_fx8_frac + 8,
    test_5_out_data, sizeof(test_5_out_data) / sizeof(test_5_out_data[0]));


tensor_quantizer test_5_out_sa8(test_5_out_tsr_sa8, test_5_out_sa_dim, test_5_out_data,
    sizeof(test_5_out_data) / sizeof(test_5_out_data[0]),
    &test_5_out_scale, 1,
    &test_5_out_zero_point, 1,
    test_5_out_scales_frac, 1);


tensor_quantizer test_6_out_fx16(test_6_out_tsr_fx16, test_6_out_fx8_frac + 8,
    test_6_out_data, sizeof(test_6_out_data) / sizeof(test_6_out_data[0]));


tensor_quantizer test_6_out_sa8(test_6_out_tsr_sa8, test_6_out_sa_dim, test_6_out_data,
    sizeof(test_6_out_data) / sizeof(test_6_out_data[0]),
    &test_6_out_scale, 1,
    &test_6_out_zero_point, 1,
    test_6_out_scales_frac, 1);


tensor_quantizer test_7_out_fx16(test_7_out_tsr_fx16, test_7_out_fx8_frac + 8,
    test_7_out_data, sizeof(test_7_out_data) / sizeof(test_7_out_data[0]));


tensor_quantizer test_7_out_sa8(test_7_out_tsr_sa8, test_7_out_sa_dim, test_7_out_data,
    sizeof(test_7_out_data) / sizeof(test_7_out_data[0]),
    &test_7_out_scale, 1,
    &test_7_out_zero_point, 1,
    test_7_out_scales_frac, 1);

// Tests configuration structures
//========================================

const mli_pool_cfg test_1_cfg = {
    /* kernel_width = */4, /* kernel_height = */3, /* stride_width = */1, /* stride_height = */1,
    /* padding_left = */0, /* padding_right = */0, /* padding_top = */0, /* padding_bottom = */0 
};
const mli_pool_cfg test_2_cfg = {
    /* kernel_width = */3, /* kernel_height = */4, /* stride_width = */2, /* stride_height = */2,
    /* padding_left = */1, /* padding_right = */1, /* padding_top = */1, /* padding_bottom = */1
};
const mli_pool_cfg test_3_cfg = {
    /* kernel_width = */3, /* kernel_height = */4, /* stride_width = */3, /* stride_height = */3,
    /* padding_left = */0, /* padding_right = */0, /* padding_top = */1, /* padding_bottom = */2
};
const mli_pool_cfg test_4_cfg = {
    /* kernel_width = */4, /* kernel_height = */4, /* stride_width = */1, /* stride_height = */1,
    /* padding_left = */0, /* padding_right = */0, /* padding_top = */0, /* padding_bottom = */0
};
const mli_pool_cfg test_5_cfg = {
    /* kernel_width = */4, /* kernel_height = */4, /* stride_width = */2, /* stride_height = */2,
    /* padding_left = */1, /* padding_right = */1, /* padding_top = */1, /* padding_bottom = */2 
};
const mli_pool_cfg test_6_cfg = {
    /* kernel_width = */2, /* kernel_height = */2, /* stride_width = */2, /* stride_height = */2,
    /* padding_left = */0, /* padding_right = */1, /* padding_top = */0, /* padding_bottom = */0 
};
const mli_pool_cfg test_7_cfg = {
    /* kernel_width = */3, /* kernel_height = */3, /* stride_width = */2, /* stride_height = */2,
    /* padding_left = */1, /* padding_right = */1, /* padding_top = */0, /* padding_bottom = */1 
};