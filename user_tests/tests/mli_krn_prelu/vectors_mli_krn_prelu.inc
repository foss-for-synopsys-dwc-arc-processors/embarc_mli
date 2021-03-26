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

extern mli::tst::tensor_quantizer input_1_fx16;
extern mli::tst::tensor_quantizer input_1_sa8;
extern mli::tst::tensor_quantizer input_2_fx16;
extern mli::tst::tensor_quantizer input_2_sa8;
extern mli::tst::tensor_quantizer alpha_1_fx16;
extern mli::tst::tensor_quantizer alpha_1_sa8;
extern mli::tst::tensor_quantizer alpha_2_fx16;
extern mli::tst::tensor_quantizer alpha_2_sa8;
extern mli::tst::tensor_quantizer alpha_3_fx16;
extern mli::tst::tensor_quantizer alpha_3_sa8;
extern mli::tst::tensor_quantizer alpha_4_fx16;
extern mli::tst::tensor_quantizer alpha_4_sa8;
extern mli::tst::tensor_quantizer alpha_5_fx16;
extern mli::tst::tensor_quantizer alpha_5_sa8;
extern mli::tst::tensor_quantizer alpha_6_fx16;
extern mli::tst::tensor_quantizer alpha_6_sa8;
extern mli::tst::tensor_quantizer alpha_7_fx16;
extern mli::tst::tensor_quantizer alpha_7_sa8;
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
extern const mli_prelu_cfg test_1_cfg;
extern const mli_prelu_cfg test_2_cfg;
extern const mli_prelu_cfg test_3_cfg;
extern const mli_prelu_cfg test_4_cfg;
extern const mli_prelu_cfg test_5_cfg;
extern const mli_prelu_cfg test_6_cfg;
extern const mli_prelu_cfg test_7_cfg;
static const float input_1_data[] = {
       4.733333f,  -2.733333f,   5.800000f,   2.066667f,   3.666667f,  -0.066667f,  -1.933333f,  -1.666667f,   2.333333f,   7.933333f, 
      -0.866667f,   1.800000f,   6.066667f,   9.000000f,  11.666667f,  10.066667f,  11.400000f,  11.933333f,  12.200000f,  -2.466667f, 
       8.733334f,   2.866667f,   4.200000f,   7.400000f,   2.600000f,   1.000000f,   5.533333f,   8.200000f,   5.000000f,   8.466666f, 
       7.133333f,   5.266667f,   6.866667f,   3.933333f,  -1.133333f,   3.133333f,  10.333333f,   7.666667f,  12.733334f,  -0.600000f, 
       4.466667f,   0.733333f,  10.600000f,  -3.000000f,  10.866667f,   0.200000f,   9.533334f,  -2.200000f,  -0.333333f,   6.600000f, 
       3.400000f,   1.533333f,   9.800000f,   0.466667f,   1.266667f,  12.466666f,   9.266666f,  11.133333f,  -1.400000f,   6.333333f};

static const float input_1_scale = 0.0616993457f;
static const float input_1_zero_point = 4.8666667938f;
static const int8_t input_1_scales_frac[] = {19};
static const int input_1_sa_dim = -1;

static const int input_1_fx8_frac = 3;

#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(5 + 1) * 4 + 2, (5 + 1), 1}, \
    /* .shape = */ {3, 4, 5}, \
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

static const float input_2_data[] = {
     -0.014464f, -0.127124f, -0.346039f,  0.183577f, -0.146219f, -0.229402f, -0.023406f,  0.667822f,  1.901911f, -0.275810f, 
     -0.131176f,  0.977532f,  0.351972f, -0.052234f, -0.300959f,  0.352285f,  1.343946f, -0.052462f,  0.123676f,  2.070206f, 
      1.131116f, -0.117366f, -0.026335f,  0.098247f,  1.728101f, -0.228697f, -0.269607f, -0.029644f,  0.599574f, -0.073787f, 
      0.983436f, -0.086342f,  0.245085f, -0.370017f,  0.068616f,  0.842401f,  0.113727f,  1.902350f, -0.053947f, -0.037896f, 
      0.738433f, -0.160815f,  0.136564f, -0.149495f, -0.103959f, -0.101112f, -0.183924f, -0.008274f,  0.707035f, -0.253770f, 
      0.759652f,  0.645774f, -0.201239f, -0.084330f, -0.047119f,  1.220286f, -0.164200f,  0.027409f, -0.076359f,  2.251054f, 
     -0.017310f, -0.350310f, -0.360060f,  0.602139f,  0.126229f, -0.125738f,  0.396302f, -0.120944f, -0.285312f,  0.344301f, 
      0.985496f, -0.202224f,  0.202073f,  0.605014f, -0.044279f,  0.961571f,  1.598490f,  0.902155f,  1.857274f,  1.300972f, 
      0.263176f,  0.234102f,  0.238046f,  0.242233f,  1.962631f,  0.369178f,  1.334183f, -0.130084f, -0.008512f,  1.495837f, 
      0.099535f,  0.668074f,  0.266050f,  1.410501f, -0.069969f,  1.889247f,  0.468879f,  0.469579f,  1.712759f,  0.044378f, 
     -0.154750f,  1.257971f,  0.378508f, -0.342435f, -0.215832f, -0.027610f,  1.770307f,  0.632215f, -0.193755f, -0.192682f, 
      0.082068f,  1.312118f,  0.808710f,  0.027000f, -0.405419f, -0.020981f, -0.061781f, -0.326719f, -0.055818f, -0.340564f, 
     -0.193347f,  1.564731f, -0.067227f, -0.042573f, -0.111758f,  0.113742f,  0.825400f,  1.943771f};

static const float input_2_scale = 0.0104175406f;
static const float input_2_zero_point = 0.9228175879f;
static const int8_t input_2_scales_frac[] = {21};
static const int input_2_sa_dim = -1;

static const int input_2_fx8_frac = 5;

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

static const float alpha_1_data[] = {0.269882f, 1.702468f, 1.740227f};

static const float alpha_1_scale = 0.0057660593f;
static const float alpha_1_zero_point = 1.0050547123f;
static const int8_t alpha_1_scales_frac[] = {22};
static const int alpha_1_sa_dim = -1;

static const int alpha_1_fx8_frac = 6;

#define WEIGHTS_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 1, 1}, \
    /* .rank =  */ 3


static const mli_tensor alpha_1_tsr_fx16 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor alpha_1_tsr_sa8 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float alpha_2_data[] = {-0.505353f, -0.450047f, 0.016671f, 0.772134f};

static const float alpha_2_scale = 0.0050097532f;
static const float alpha_2_zero_point = 0.1333907843f;
static const int8_t alpha_2_scales_frac[] = {22};
static const int alpha_2_sa_dim = -1;

static const int alpha_2_fx8_frac = 7;

#define WEIGHTS_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 4, 1}, \
    /* .rank =  */ 3


static const mli_tensor alpha_2_tsr_fx16 = {
    WEIGHTS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor alpha_2_tsr_sa8 = {
    WEIGHTS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float alpha_3_data[] = {1.419029f, -1.642624f, 2.368335f, -0.693205f, -1.906772f};

static const float alpha_3_scale = 0.0167651270f;
static const float alpha_3_zero_point = 0.2307810783f;
static const int8_t alpha_3_scales_frac[] = {20};
static const int alpha_3_sa_dim = -1;

static const int alpha_3_fx8_frac = 5;

#define WEIGHTS_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 1, 5}, \
    /* .rank =  */ 3


static const mli_tensor alpha_3_tsr_fx16 = {
    WEIGHTS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor alpha_3_tsr_sa8 = {
    WEIGHTS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float alpha_4_data[] = {-0.097156f, -1.190724f};

static const float alpha_4_scale = 0.0042885053f;
static const float alpha_4_zero_point = -0.6439400315f;
static const int8_t alpha_4_scales_frac[] = {22};
static const int alpha_4_sa_dim = -1;

static const int alpha_4_fx8_frac = 6;

#define WEIGHTS_4_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 1, 1, 1}, \
    /* .rank =  */ 4


static const mli_tensor alpha_4_tsr_fx16 = {
    WEIGHTS_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor alpha_4_tsr_sa8 = {
    WEIGHTS_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float alpha_5_data[] = {0.063095f, 0.738169f, 0.286040f, -1.257899f};

static const float alpha_5_scale = 0.0078277169f;
static const float alpha_5_zero_point = -0.2598646283f;
static const int8_t alpha_5_scales_frac[] = {21};
static const int alpha_5_sa_dim = -1;

static const int alpha_5_fx8_frac = 6;

#define WEIGHTS_5_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 4, 1, 1}, \
    /* .rank =  */ 4


static const mli_tensor alpha_5_tsr_fx16 = {
    WEIGHTS_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor alpha_5_tsr_sa8 = {
    WEIGHTS_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float alpha_6_data[] = {-1.454904f, 0.389350f, -0.020371f, 1.286899f, 0.113903f, -0.130907f, -0.644746f, -0.751419f};

static const float alpha_6_scale = 0.0107521666f;
static const float alpha_6_zero_point = -0.0840026140f;
static const int8_t alpha_6_scales_frac[] = {21};
static const int alpha_6_sa_dim = -1;

static const int alpha_6_fx8_frac = 6;

#define WEIGHTS_6_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 1, 8, 1}, \
    /* .rank =  */ 4


static const mli_tensor alpha_6_tsr_fx16 = {
    WEIGHTS_6_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor alpha_6_tsr_sa8 = {
    WEIGHTS_6_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float alpha_7_data[] = {-1.992258f, -1.145236f};

static const float alpha_7_scale = 0.0033216532f;
static const float alpha_7_zero_point = -1.5687470436f;
static const int8_t alpha_7_scales_frac[] = {23};
static const int alpha_7_sa_dim = -1;

static const int alpha_7_fx8_frac = 6;

#define WEIGHTS_7_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 1, 1, 2}, \
    /* .rank =  */ 4


static const mli_tensor alpha_7_tsr_fx16 = {
    WEIGHTS_7_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor alpha_7_tsr_sa8 = {
    WEIGHTS_7_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float test_1_out_data[] = {
       4.733333f,  -0.737678f,   5.800000f,   2.066667f,   3.666667f,  -0.017992f,  -0.521772f,  -0.449803f,   2.333333f,   7.933333f, 
      -0.233898f,   1.800000f,   6.066667f,   9.000000f,  11.666667f,  10.066667f,  11.400000f,  11.933333f,  12.200000f,  -0.665709f, 
       8.733334f,   2.866667f,   4.200000f,   7.400000f,   2.600000f,   1.000000f,   5.533333f,   8.200000f,   5.000000f,   8.466666f, 
       7.133333f,   5.266667f,   6.866667f,   3.933333f,  -1.929464f,   3.133333f,  10.333333f,   7.666667f,  12.733334f,  -1.021481f, 
       4.466667f,   0.733333f,  10.600000f,  -5.220682f,  10.866667f,   0.200000f,   9.533334f,  -3.828500f,  -0.580076f,   6.600000f, 
       3.400000f,   1.533333f,   9.800000f,   0.466667f,   1.266667f,  12.466666f,   9.266666f,  11.133333f,  -2.436318f,   6.333333f};

static const float test_1_out_scale = 0.0704079047f;
static const float test_1_out_zero_point = 3.7563257217f;
static const int8_t test_1_out_scales_frac[] = {18};
static const int test_1_out_sa_dim = -1;

static const int test_1_out_fx8_frac = 3;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(5 + 2) * 4 + 3, (5 + 2), 1}, \
    /* .shape = */ {3, 4, 5}, \
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
     4.733333f, 1.381297f,  5.800000f, 2.066667f,  3.666667f,  0.030003f,  0.870091f,  0.750078f,  2.333333f,  7.933333f, 
    -0.014448f, 1.800000f,  6.066667f, 9.000000f, 11.666667f, 10.066667f, 11.400000f, 11.933333f, 12.200000f, -1.904598f, 
     8.733334f, 2.866667f,  4.200000f, 7.400000f,  2.600000f,  1.000000f,  5.533333f,  8.200000f,  5.000000f,  8.466666f, 
     7.133333f, 5.266667f,  6.866667f, 3.933333f, -0.018894f,  3.133333f, 10.333333f,  7.666667f, 12.733334f, -0.463281f, 
     4.466667f, 0.733333f, 10.600000f, 1.516058f, 10.866667f,  0.200000f,  9.533334f,  0.990103f,  0.150016f,  6.600000f, 
     3.400000f, 1.533333f,  9.800000f, 0.466667f,  1.266667f, 12.466666f,  9.266666f, 11.133333f, -1.080988f,  6.333333f};

static const float test_2_out_scale = 0.0574036539f;
static const float test_2_out_zero_point = 5.4143676758f;
static const int8_t test_2_out_scales_frac[] = {19};
static const int test_2_out_sa_dim = -1;

static const int test_2_out_fx8_frac = 3;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(5 + 3) * 4 + 4, (5 + 3), 1}, \
    /* .shape = */ {3, 4, 5}, \
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
     4.733333f, 4.489841f,  5.800000f, 2.066667f,  3.666667f, -0.094602f,  3.175741f, -3.947225f,  2.333333f, 7.933333f, 
    -1.229825f, 1.800000f,  6.066667f, 9.000000f, 11.666667f, 10.066667f, 11.400000f, 11.933333f, 12.200000f, 4.703372f, 
     8.733334f, 2.866667f,  4.200000f, 7.400000f,  2.600000f,  1.000000f,  5.533333f,  8.200000f,  5.000000f, 8.466666f, 
     7.133333f, 5.266667f,  6.866667f, 3.933333f,  2.161009f,  3.133333f, 10.333333f,  7.666667f, 12.733334f, 1.144064f, 
     4.466667f, 0.733333f, 10.600000f, 2.079616f, 10.866667f,  0.200000f,  9.533334f, -5.210337f,  0.231068f, 6.600000f, 
     3.400000f, 1.533333f,  9.800000f, 0.466667f,  1.266667f, 12.466666f,  9.266666f, 11.133333f,  0.970487f, 6.333333f};

static const float test_3_out_scale = 0.0703673363f;
static const float test_3_out_zero_point = 3.7614984512f;
static const int8_t test_3_out_scales_frac[] = {18};
static const int test_3_out_sa_dim = -1;

static const int test_3_out_fx8_frac = 3;

#define TEST_3_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(5 + 4) * 4 + 5, (5 + 4), 1}, \
    /* .shape = */ {3, 4, 5}, \
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
    0.001405f, 0.012351f, 0.033620f, 0.183577f, 0.014206f, 0.022288f, 0.002274f, 0.667822f, 1.901911f, 0.026797f, 
    0.012744f, 0.977532f, 0.351972f, 0.005075f, 0.029240f, 0.352285f, 1.343946f, 0.005097f, 0.123676f, 2.070206f, 
    1.131116f, 0.011403f, 0.002559f, 0.098247f, 1.728101f, 0.022219f, 0.026194f, 0.002880f, 0.599574f, 0.007169f, 
    0.983436f, 0.008389f, 0.245085f, 0.035949f, 0.068616f, 0.842401f, 0.113727f, 1.902350f, 0.005241f, 0.003682f, 
    0.738433f, 0.015624f, 0.136564f, 0.014524f, 0.010100f, 0.009824f, 0.017869f, 0.000804f, 0.707035f, 0.024655f, 
    0.759652f, 0.645774f, 0.019551f, 0.008193f, 0.004578f, 1.220286f, 0.015953f, 0.027409f, 0.007419f, 2.251054f, 
    0.001682f, 0.034035f, 0.034982f, 0.602139f, 0.126229f, 0.149719f, 0.396302f, 0.144011f, 0.339728f, 0.344301f, 
    0.985496f, 0.240794f, 0.202073f, 0.605014f, 0.052724f, 0.961571f, 1.598490f, 0.902155f, 1.857274f, 1.300972f, 
    0.263176f, 0.234102f, 0.238046f, 0.242233f, 1.962631f, 0.369178f, 1.334183f, 0.154895f, 0.010135f, 1.495837f, 
    0.099535f, 0.668074f, 0.266050f, 1.410501f, 0.083313f, 1.889247f, 0.468879f, 0.469579f, 1.712759f, 0.044378f, 
    0.184264f, 1.257971f, 0.378508f, 0.407745f, 0.256996f, 0.032875f, 1.770307f, 0.632215f, 0.230709f, 0.229431f, 
    0.082068f, 1.312118f, 0.808710f, 0.027000f, 0.482742f, 0.024983f, 0.073565f, 0.389032f, 0.066464f, 0.405518f, 
    0.230223f, 1.564731f, 0.080048f, 0.050693f, 0.133073f, 0.113742f, 0.825400f, 1.943771f};

static const float test_4_out_scale = 0.0088245105f;
static const float test_4_out_zero_point = 1.1259289980f;
static const int8_t test_4_out_scales_frac[] = {21};
static const int test_4_out_sa_dim = -1;

static const int test_4_out_fx8_frac = 5;

#define TEST_4_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
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

static const float test_5_out_data[] = {
    -0.000913f, -0.008021f, -0.021833f,  0.183577f, -0.009226f, -0.014474f, -0.001477f,  0.667822f,  1.901911f, -0.017402f, 
    -0.008277f,  0.977532f,  0.351972f, -0.003296f, -0.018989f,  0.352285f,  1.343946f, -0.038726f,  0.123676f,  2.070206f, 
     1.131116f, -0.086636f, -0.019440f,  0.098247f,  1.728101f, -0.168817f, -0.199016f, -0.021883f,  0.599574f, -0.054467f, 
     0.983436f, -0.063735f,  0.245085f, -0.105840f,  0.068616f,  0.842401f,  0.113727f,  1.902350f, -0.015431f, -0.010840f, 
     0.738433f, -0.046000f,  0.136564f, -0.042761f, -0.029736f, -0.028922f, -0.052610f, -0.002367f,  0.707035f,  0.319217f, 
     0.759652f,  0.645774f,  0.253138f,  0.106079f,  0.059270f,  1.220286f,  0.206547f,  0.027409f,  0.096052f,  2.251054f, 
     0.021774f,  0.440654f,  0.452918f,  0.602139f,  0.126229f, -0.007933f,  0.396302f, -0.007631f, -0.018002f,  0.344301f, 
     0.985496f, -0.012759f,  0.202073f,  0.605014f, -0.002794f,  0.961571f,  1.598490f,  0.902155f,  1.857274f,  1.300972f, 
     0.263176f,  0.234102f,  0.238046f,  0.242233f,  1.962631f,  0.369178f,  1.334183f, -0.096024f, -0.006283f,  1.495837f, 
     0.099535f,  0.668074f,  0.266050f,  1.410501f, -0.051649f,  1.889247f,  0.468879f,  0.469579f,  1.712759f,  0.044378f, 
    -0.044265f,  1.257971f,  0.378508f, -0.097950f, -0.061736f, -0.007897f,  1.770307f,  0.632215f, -0.055422f, -0.055115f, 
     0.082068f,  1.312118f,  0.808710f,  0.027000f,  0.509976f,  0.026392f,  0.077715f,  0.410979f,  0.070214f,  0.428395f, 
     0.243211f,  1.564731f,  0.084564f,  0.053553f,  0.140581f,  0.113742f,  0.825400f,  1.943771f};

static const float test_5_out_scale = 0.0096081169f;
static const float test_5_out_zero_point = 1.0260190964f;
static const int8_t test_5_out_scales_frac[] = {21};
static const int test_5_out_sa_dim = -1;

static const int test_5_out_fx8_frac = 5;

#define TEST_5_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 4, 8, 2}, \
    /* .rank =  */ 4


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
     0.021044f,  0.184953f, -0.134730f,  0.183577f,  0.002979f,  0.004673f, -0.030122f,  0.667822f,  1.901911f, -0.031416f, 
     0.017172f,  0.977532f,  0.351972f,  0.033678f,  0.226147f,  0.352285f,  1.343946f,  0.076328f,  0.123676f,  2.070206f, 
     1.131116f,  0.002391f, -0.033891f,  0.098247f,  1.728101f, -0.026049f,  0.035293f,  0.003881f,  0.599574f,  0.047574f, 
     0.983436f,  0.064879f,  0.245085f,  0.538339f,  0.068616f,  0.842401f,  0.113727f,  1.902350f, -0.069424f, -0.048768f, 
     0.738433f, -0.018317f,  0.136564f,  0.019570f,  0.067027f,  0.065191f,  0.138204f,  0.006217f,  0.707035f,  0.369211f, 
     0.759652f,  0.645774f,  0.004099f,  0.001718f, -0.060637f,  1.220286f, -0.018703f,  0.027409f,  0.009996f,  2.251054f, 
     0.011161f,  0.225861f,  0.270556f,  0.602139f,  0.126229f,  0.182936f,  0.396302f, -0.047089f,  0.005812f,  0.344301f, 
     0.985496f, -0.260242f,  0.202073f,  0.605014f,  0.005796f,  0.961571f,  1.598490f,  0.902155f,  1.857274f,  1.300972f, 
     0.263176f,  0.234102f,  0.238046f,  0.242233f,  1.962631f,  0.369178f,  1.334183f, -0.167405f, -0.000970f,  1.495837f, 
     0.099535f,  0.668074f,  0.266050f,  1.410501f,  0.052576f,  1.889247f,  0.468879f,  0.469579f,  1.712759f,  0.044378f, 
     0.003152f,  1.257971f,  0.378508f, -0.440679f, -0.024584f, -0.003145f,  1.770307f,  0.632215f,  0.124923f,  0.124231f, 
     0.082068f,  1.312118f,  0.808710f,  0.027000f, -0.157850f, -0.008169f,  0.001259f,  0.006655f, -0.071832f, -0.438271f, 
    -0.022023f,  1.564731f,  0.008800f,  0.005573f,  0.072056f,  0.113742f,  0.825400f,  1.943771f};

static const float test_6_out_scale = 0.0105558140f;
static const float test_6_out_zero_point = 0.9051877260f;
static const int8_t test_6_out_scales_frac[] = {21};
static const int test_6_out_sa_dim = -1;

static const int test_6_out_fx8_frac = 5;

#define TEST_6_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 4, 8, 2}, \
    /* .rank =  */ 4


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
    0.028816f, 0.145587f, 0.689399f, 0.183577f, 0.291306f, 0.262720f, 0.046632f, 0.667822f, 1.901911f, 0.315868f, 
    0.261336f, 0.977532f, 0.351972f, 0.059820f, 0.599588f, 0.352285f, 1.343946f, 0.060082f, 0.123676f, 2.070206f, 
    1.131116f, 0.134412f, 0.052466f, 0.098247f, 1.728101f, 0.261912f, 0.537127f, 0.033950f, 0.599574f, 0.084503f, 
    0.983436f, 0.098882f, 0.245085f, 0.423757f, 0.068616f, 0.842401f, 0.113727f, 1.902350f, 0.107476f, 0.043400f, 
    0.738433f, 0.184171f, 0.136564f, 0.171207f, 0.207113f, 0.115797f, 0.366424f, 0.009475f, 0.707035f, 0.290627f, 
    0.759652f, 0.645774f, 0.400919f, 0.096578f, 0.093873f, 1.220286f, 0.327128f, 0.027409f, 0.152127f, 2.251054f, 
    0.034486f, 0.401187f, 0.717332f, 0.602139f, 0.126229f, 0.144000f, 0.396302f, 0.138509f, 0.568415f, 0.344301f, 
    0.985496f, 0.231595f, 0.202073f, 0.605014f, 0.088215f, 0.961571f, 1.598490f, 0.902155f, 1.857274f, 1.300972f, 
    0.263176f, 0.234102f, 0.238046f, 0.242233f, 1.962631f, 0.369178f, 1.334183f, 0.148977f, 0.016958f, 1.495837f, 
    0.099535f, 0.668074f, 0.266050f, 1.410501f, 0.139396f, 1.889247f, 0.468879f, 0.469579f, 1.712759f, 0.044378f, 
    0.308301f, 1.257971f, 0.378508f, 0.392169f, 0.429992f, 0.031620f, 1.770307f, 0.632215f, 0.386010f, 0.220666f, 
    0.082068f, 1.312118f, 0.808710f, 0.027000f, 0.807699f, 0.024028f, 0.123084f, 0.374170f, 0.111204f, 0.390026f, 
    0.385198f, 1.564731f, 0.133933f, 0.048756f, 0.222651f, 0.113742f, 0.825400f, 1.943771f};

static const float test_7_out_scale = 0.0087905042f;
static const float test_7_out_zero_point = 1.1302647591f;
static const int8_t test_7_out_scales_frac[] = {21};
static const int test_7_out_sa_dim = -1;

static const int test_7_out_fx8_frac = 5;

#define TEST_7_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 4, 8, 2}, \
    /* .rank =  */ 4


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

tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx8_frac + 8, input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]));
tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data, 
sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);

tensor_quantizer alpha_1_fx16(alpha_1_tsr_fx16, alpha_1_fx8_frac + 8, alpha_1_data, sizeof(alpha_1_data) / sizeof(alpha_1_data[0]));
tensor_quantizer alpha_1_sa8(alpha_1_tsr_sa8, alpha_1_sa_dim, alpha_1_data, 
sizeof(alpha_1_data) / sizeof(alpha_1_data[0]), &alpha_1_scale, 1, &alpha_1_zero_point, 1, alpha_1_scales_frac, 1);

tensor_quantizer alpha_2_fx16(alpha_2_tsr_fx16, alpha_2_fx8_frac + 8, alpha_2_data, sizeof(alpha_2_data) / sizeof(alpha_2_data[0]));
tensor_quantizer alpha_2_sa8(alpha_2_tsr_sa8, alpha_2_sa_dim, alpha_2_data, 
sizeof(alpha_2_data) / sizeof(alpha_2_data[0]), &alpha_2_scale, 1, &alpha_2_zero_point, 1, alpha_2_scales_frac, 1);

tensor_quantizer alpha_3_fx16(alpha_3_tsr_fx16, alpha_3_fx8_frac + 8, alpha_3_data, sizeof(alpha_3_data) / sizeof(alpha_3_data[0]));
tensor_quantizer alpha_3_sa8(alpha_3_tsr_sa8, alpha_3_sa_dim, alpha_3_data, 
sizeof(alpha_3_data) / sizeof(alpha_3_data[0]), &alpha_3_scale, 1, &alpha_3_zero_point, 1, alpha_3_scales_frac, 1);

tensor_quantizer alpha_4_fx16(alpha_4_tsr_fx16, alpha_4_fx8_frac + 8, alpha_4_data, sizeof(alpha_4_data) / sizeof(alpha_4_data[0]));
tensor_quantizer alpha_4_sa8(alpha_4_tsr_sa8, alpha_4_sa_dim, alpha_4_data, 
sizeof(alpha_4_data) / sizeof(alpha_4_data[0]), &alpha_4_scale, 1, &alpha_4_zero_point, 1, alpha_4_scales_frac, 1);

tensor_quantizer alpha_5_fx16(alpha_5_tsr_fx16, alpha_5_fx8_frac + 8, alpha_5_data, sizeof(alpha_5_data) / sizeof(alpha_5_data[0]));
tensor_quantizer alpha_5_sa8(alpha_5_tsr_sa8, alpha_5_sa_dim, alpha_5_data, 
sizeof(alpha_5_data) / sizeof(alpha_5_data[0]), &alpha_5_scale, 1, &alpha_5_zero_point, 1, alpha_5_scales_frac, 1);

tensor_quantizer alpha_6_fx16(alpha_6_tsr_fx16, alpha_6_fx8_frac + 8, alpha_6_data, sizeof(alpha_6_data) / sizeof(alpha_6_data[0]));
tensor_quantizer alpha_6_sa8(alpha_6_tsr_sa8, alpha_6_sa_dim, alpha_6_data, 
sizeof(alpha_6_data) / sizeof(alpha_6_data[0]), &alpha_6_scale, 1, &alpha_6_zero_point, 1, alpha_6_scales_frac, 1);

tensor_quantizer alpha_7_fx16(alpha_7_tsr_fx16, alpha_7_fx8_frac + 8, alpha_7_data, sizeof(alpha_7_data) / sizeof(alpha_7_data[0]));
tensor_quantizer alpha_7_sa8(alpha_7_tsr_sa8, alpha_7_sa_dim, alpha_7_data, 
sizeof(alpha_7_data) / sizeof(alpha_7_data[0]), &alpha_7_scale, 1, &alpha_7_zero_point, 1, alpha_7_scales_frac, 1);

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

tensor_quantizer test_5_out_fx16(test_5_out_tsr_fx16, test_5_out_fx8_frac + 8, test_5_out_data, sizeof(test_5_out_data) / sizeof(test_5_out_data[0]));
tensor_quantizer test_5_out_sa8(test_5_out_tsr_sa8, test_5_out_sa_dim, test_5_out_data, 
sizeof(test_5_out_data) / sizeof(test_5_out_data[0]), &test_5_out_scale, 1, &test_5_out_zero_point, 1, test_5_out_scales_frac, 1);

tensor_quantizer test_6_out_fx16(test_6_out_tsr_fx16, test_6_out_fx8_frac + 8, test_6_out_data, sizeof(test_6_out_data) / sizeof(test_6_out_data[0]));
tensor_quantizer test_6_out_sa8(test_6_out_tsr_sa8, test_6_out_sa_dim, test_6_out_data, 
sizeof(test_6_out_data) / sizeof(test_6_out_data[0]), &test_6_out_scale, 1, &test_6_out_zero_point, 1, test_6_out_scales_frac, 1);

tensor_quantizer test_7_out_fx16(test_7_out_tsr_fx16, test_7_out_fx8_frac + 8, test_7_out_data, sizeof(test_7_out_data) / sizeof(test_7_out_data[0]));
tensor_quantizer test_7_out_sa8(test_7_out_tsr_sa8, test_7_out_sa_dim, test_7_out_data, 
sizeof(test_7_out_data) / sizeof(test_7_out_data[0]), &test_7_out_scale, 1, &test_7_out_zero_point, 1, test_7_out_scales_frac, 1);

const mli_prelu_cfg test_1_cfg = {/*axis*/0};
const mli_prelu_cfg test_2_cfg = {/*axis*/1};
const mli_prelu_cfg test_3_cfg = {/*axis*/2};
const mli_prelu_cfg test_4_cfg = {/*axis*/0};
const mli_prelu_cfg test_5_cfg = {/*axis*/1};
const mli_prelu_cfg test_6_cfg = {/*axis*/2};
const mli_prelu_cfg test_7_cfg = {/*axis*/3};
