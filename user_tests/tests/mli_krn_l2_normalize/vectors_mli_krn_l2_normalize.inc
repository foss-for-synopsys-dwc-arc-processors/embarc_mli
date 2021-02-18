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

extern mli::tst::tensor_quantizer epsilon_fx16;
extern mli::tst::tensor_quantizer epsilon_sa8;
extern mli::tst::tensor_quantizer input_1_fx16;
extern mli::tst::tensor_quantizer input_1_sa8;
extern mli::tst::tensor_quantizer input_2_fx16;
extern mli::tst::tensor_quantizer input_2_sa8;
extern mli::tst::tensor_quantizer input_3_fx16;
extern mli::tst::tensor_quantizer input_3_sa8;
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
extern const mli_l2_normalize_cfg test_1_cfg;
extern const mli_l2_normalize_cfg test_2_cfg;
extern const mli_l2_normalize_cfg test_3_cfg;
extern const mli_l2_normalize_cfg test_4_cfg;
extern const mli_l2_normalize_cfg test_5_cfg;
extern const mli_l2_normalize_cfg test_6_cfg;
extern const mli_l2_normalize_cfg test_7_cfg;
extern const mli_l2_normalize_cfg test_8_cfg;

static const float epsilon_data[] = {0.0000000001f};

static const float epsilon_scale = 1.000000f;
static const float epsilon_zero_point = 0.000000f;
static const int8_t epsilon_scales_frac[] = {48};
static const int epsilon_sa_dim = -1;

static const int epsilon_fx8_frac = 40;

#define EPSILON_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {0}, \
    /* .rank =  */ 0


static const mli_tensor epsilon_tsr_fx16 = {
    EPSILON_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor epsilon_tsr_sa8 = {
    EPSILON_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


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
    /* .mem_stride = */ {0}, \
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
    -0.196436f,  1.251724f, -0.315056f,  0.088186f,  0.201181f, -0.113343f,  1.386356f, -0.121149f, -0.221007f,  0.263518f, 
    -0.396502f,  0.401217f,  0.365816f,  0.503809f, -0.121512f,  0.896004f, -0.038972f,  0.456809f, -0.245499f, -0.125369f, 
     0.764411f, -0.150907f, -0.049561f,  0.212580f,  0.996462f,  0.298890f,  1.193738f, -0.021064f, -0.054614f,  0.754287f, 
     0.005484f, -0.186344f, -0.048063f, -0.353363f,  0.536566f,  0.069397f,  0.929472f, -0.085906f,  0.901522f, -0.047931f, 
     1.326120f,  0.331985f, -0.078394f,  1.304024f,  0.903325f,  1.575325f, -0.237606f, -0.127100f,  0.028617f,  0.706279f, 
    -0.199204f,  0.094334f, -0.100702f,  0.081564f, -0.233470f,  0.791553f,  0.295598f, -0.037702f,  0.638925f, -0.020275f, 
     0.737763f, -0.104850f, -0.049147f,  0.409932f, -0.005188f, -0.017840f, -0.315708f,  0.123700f,  0.613385f, -0.118198f, 
    -0.201318f, -0.419595f,  0.762317f, -0.325845f,  1.236359f, -0.104461f, -0.043952f, -0.150563f, -0.124493f,  0.135427f, 
    -0.308457f, -0.203477f,  0.779945f,  0.693177f,  1.023374f, -0.285378f,  2.360960f,  0.218219f, -0.168041f,  0.467199f, 
     0.113511f, -0.117993f,  0.308877f, -0.096351f, -0.139591f,  1.410012f, -0.239563f,  0.145219f,  0.369902f, -0.303721f, 
    -0.092921f, -0.279719f,  0.096952f, -0.027382f,  0.676855f, -0.001419f, -0.017476f,  0.292802f, -0.013528f, -0.300737f, 
    -0.045962f, -0.139116f, -0.319746f, -0.000157f,  1.130625f, -0.189847f, -0.089942f, -0.198470f,  0.965037f, -0.240541f, 
    -0.239220f,  0.552013f, -0.183854f, -0.405221f, -0.238537f, -0.073910f,  0.039353f, -0.237175f};

static const float input_2_scale = 0.0109041352f;
static const float input_2_zero_point = 0.9706827402f;
static const int8_t input_2_scales_frac[] = {21};
static const int input_2_sa_dim = -1;

static const int input_2_fx8_frac = 5;

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

static const float input_3_data[] = {
       0.000000f,    0.000000f,    0.000000f,    0.000000f, 
      10.000000f,   10.000000f,   10.000000f,   10.000000f, 
    1000.000000f, 1000.000000f, 1000.000000f, 1000.000000f};

static const float input_3_scale = 3.9215686321f;
static const float input_3_zero_point = 500.0000000000f;
static const int8_t input_3_scales_frac[] = {13};
static const int input_3_sa_dim = -1;

static const int input_3_fx8_frac = -3;

#define INPUT_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 4}, \
    /* .rank =  */ 2


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
     0.091081f, -0.052596f, 0.111606f,  0.039768f,  0.070556f, -0.001283f, -0.037202f, -0.032071f,  0.044899f,  0.152657f, 
    -0.016677f,  0.034636f, 0.116738f,  0.173182f,  0.224496f,  0.193708f,  0.219364f,  0.229627f,  0.234758f, -0.047465f, 
     0.168051f,  0.055162f, 0.080818f,  0.142394f,  0.050030f,  0.019242f,  0.106475f,  0.157788f,  0.096212f,  0.162920f, 
     0.137263f,  0.101344f, 0.132132f,  0.075687f, -0.021808f,  0.060293f,  0.198839f,  0.147526f,  0.245021f, -0.011545f, 
     0.085950f,  0.014111f, 0.203970f, -0.057727f,  0.209102f,  0.003848f,  0.183445f, -0.042333f, -0.006414f,  0.127000f, 
     0.065424f,  0.029505f, 0.188576f,  0.008980f,  0.024374f,  0.239890f,  0.178314f,  0.214233f, -0.026939f,  0.121869f};

static const float test_1_out_scale = 0.0011872480f;
static const float test_1_out_zero_point = 0.0936467201f;
static const int8_t test_1_out_scales_frac[] = {24};
static const int test_1_out_sa_dim = -1;

static const int test_1_out_fx8_frac = 9;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
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
     0.525177f, -0.303271f, 0.643527f,  0.229303f,  0.406827f, -0.007703f, -0.223388f, -0.192576f,  0.269606f,  0.916660f, 
    -0.053966f,  0.112083f, 0.377762f,  0.560417f,  0.726466f,  0.437836f,  0.495827f,  0.519024f,  0.530622f, -0.107284f, 
     0.682690f,  0.224089f, 0.328317f,  0.578462f,  0.203244f,  0.071512f,  0.395700f,  0.586399f,  0.357561f,  0.605469f, 
     0.597504f,  0.441148f, 0.575168f,  0.329465f, -0.094931f,  0.170463f,  0.562166f,  0.417091f,  0.692734f, -0.032642f, 
     0.277046f,  0.045485f, 0.657468f, -0.186076f,  0.674008f,  0.016937f,  0.807343f, -0.186310f, -0.028229f,  0.558930f, 
     0.321595f,  0.145033f, 0.926951f,  0.044141f,  0.119810f,  0.617714f,  0.459156f,  0.551648f, -0.069369f,  0.313812f};

static const float test_2_out_scale = 0.0048244027f;
static const float test_2_out_zero_point = 0.3118399978f;
static const int8_t test_2_out_scales_frac[] = {22};
static const int test_2_out_sa_dim = -1;

static const int test_2_out_fx8_frac = 7;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
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
     0.434586f, -0.678544f, 0.453402f,  0.250563f,  0.311801f, -0.065233f, -0.172757f, -0.192633f,  0.422116f,  0.594324f, 
    -0.109021f,  0.311790f, 0.452188f,  0.915280f,  0.989554f,  0.616563f,  0.634697f,  0.661806f,  0.689653f, -0.361514f, 
     0.801842f,  0.711644f, 0.328325f,  0.897177f,  0.221095f,  0.978492f,  0.494442f,  0.947754f,  0.904534f,  0.634279f, 
     0.897324f,  0.912274f, 0.511817f,  0.400011f, -0.096128f,  0.191910f,  0.575310f,  0.425183f,  0.719802f, -0.087936f, 
     0.410102f,  0.182048f, 0.828631f, -0.363720f,  0.924065f,  0.195698f,  0.851870f, -0.254275f, -0.060302f,  0.494438f, 
     0.427696f,  0.265599f, 0.730458f,  0.047459f,  0.107437f,  0.763558f,  0.515924f,  0.617440f, -0.079141f,  0.928211f};

static const float test_3_out_scale = 0.0065415595f;
static const float test_3_out_zero_point = 0.1555048823f;
static const int8_t test_3_out_scales_frac[] = {22};
static const int test_3_out_sa_dim = -1;

static const int test_3_out_fx8_frac = 7;

#define TEST_3_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
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
    -0.031665f,  0.201776f, -0.050787f,  0.014216f,  0.032430f, -0.018271f,  0.223479f, -0.019529f, -0.035626f,  0.042479f, 
    -0.063916f,  0.064676f,  0.058969f,  0.081213f, -0.019588f,  0.144435f, -0.006282f,  0.073637f, -0.039574f, -0.020209f, 
     0.123222f, -0.024326f, -0.007989f,  0.034268f,  0.160628f,  0.048181f,  0.192429f, -0.003395f, -0.008804f,  0.121590f, 
     0.000884f, -0.030038f, -0.007748f, -0.056962f,  0.086494f,  0.011187f,  0.149830f, -0.013848f,  0.145324f, -0.007726f, 
     0.213769f,  0.053516f, -0.012637f,  0.210207f,  0.145615f,  0.253940f, -0.038302f, -0.020488f,  0.004613f,  0.113851f, 
    -0.032111f,  0.015206f, -0.016233f,  0.013148f, -0.037635f,  0.127597f,  0.047650f, -0.006078f,  0.102994f, -0.003268f, 
     0.118926f, -0.016902f, -0.007922f,  0.066081f, -0.000836f, -0.002876f, -0.050892f,  0.019940f,  0.098877f, -0.019053f, 
    -0.032452f, -0.067638f,  0.122885f, -0.052526f,  0.199300f, -0.016839f, -0.007085f, -0.024271f, -0.020068f,  0.021831f, 
    -0.049723f, -0.032800f,  0.125726f,  0.111739f,  0.164967f, -0.046003f,  0.380584f,  0.035177f, -0.027088f,  0.075312f, 
     0.018298f, -0.019020f,  0.049791f, -0.015532f, -0.022502f,  0.227292f, -0.038617f,  0.023409f,  0.059628f, -0.048960f, 
    -0.014979f, -0.045090f,  0.015629f, -0.004414f,  0.109108f, -0.000229f, -0.002817f,  0.047199f, -0.002181f, -0.048479f, 
    -0.007409f, -0.022425f, -0.051543f, -0.000025f,  0.182255f, -0.030603f, -0.014499f, -0.031993f,  0.155563f, -0.038775f, 
    -0.038562f,  0.088984f, -0.029637f, -0.065321f, -0.038452f, -0.011914f,  0.006344f, -0.038232f};

static const float test_4_out_scale = 0.0017577332f;
static const float test_4_out_zero_point = 0.1564728469f;
static const int8_t test_4_out_scales_frac[] = {24};
static const int test_4_out_sa_dim = -1;

static const int test_4_out_fx8_frac = 8;

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

static const float test_5_out_data[] = {
    -0.155035f,  0.987909f, -0.962987f,  0.269547f,  0.871245f, -0.490849f,  0.996203f, -0.087055f, -0.642597f,  0.766204f, 
    -0.702915f,  0.711274f,  0.587551f,  0.809187f, -0.134385f,  0.990929f, -0.085005f,  0.996381f, -0.890594f, -0.454799f, 
     0.981065f, -0.193678f, -0.227051f,  0.973883f,  0.957839f,  0.287305f,  0.999844f, -0.017643f, -0.072216f,  0.997389f, 
     0.029416f, -0.999567f, -0.134775f, -0.990876f,  0.991740f,  0.128267f,  0.995756f, -0.092032f,  0.998590f, -0.053091f, 
     0.970064f,  0.242849f, -0.060009f,  0.998198f,  0.497442f,  0.867497f, -0.881771f, -0.471678f,  0.040484f,  0.999180f, 
    -0.903784f,  0.427990f, -0.777079f,  0.629403f, -0.282902f,  0.959149f,  0.991964f, -0.126520f,  0.999497f, -0.031717f, 
     0.990051f, -0.140705f, -0.119038f,  0.992890f, -0.279223f, -0.960226f, -0.931081f,  0.364814f,  0.981935f, -0.189216f, 
    -0.432579f, -0.901596f,  0.919521f, -0.393041f,  0.996450f, -0.084191f, -0.280223f, -0.959935f, -0.676762f,  0.736201f, 
    -0.834740f, -0.550645f,  0.747460f,  0.664306f,  0.963248f, -0.268612f,  0.995756f,  0.092036f, -0.338450f,  0.940984f, 
     0.693287f, -0.720661f,  0.954632f, -0.297788f, -0.098518f,  0.995135f, -0.855152f,  0.518377f,  0.772856f, -0.634581f, 
    -0.315256f, -0.949007f,  0.962355f, -0.271794f,  0.999998f, -0.002096f, -0.059578f,  0.998224f, -0.044937f, -0.998990f, 
    -0.313709f, -0.949519f, -1.000000f, -0.000491f,  0.986194f, -0.165595f, -0.412769f, -0.910836f,  0.970312f, -0.241856f, 
    -0.397627f,  0.917547f, -0.413175f, -0.910652f, -0.955198f, -0.295967f,  0.163684f, -0.986513f};

static const float test_5_out_scale = 0.0078431275f;
static const float test_5_out_zero_point = -0.0000010133f;
static const int8_t test_5_out_scales_frac[] = {21};
static const int test_5_out_sa_dim = -1;

static const int test_5_out_fx8_frac = 7;

#define TEST_5_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {((2 + 1) * 8 + 2) * 4 + 4 ,(2 + 1) * 8 + 2, (2 + 1), 1}, \
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
    -0.999651f,  0.999898f, -0.706375f,  0.580493f,  0.311650f, -0.692129f,  0.989620f, -0.277398f, -0.278448f,  0.628822f, 
    -0.305382f,  0.967738f,  0.992859f,  0.958129f, -0.698487f,  0.988770f, -0.125349f,  0.913477f, -0.300242f, -0.177974f, 
     0.598436f, -0.467463f, -0.020987f,  0.697792f,  0.986077f,  0.538903f,  0.995509f, -0.175741f, -0.174114f,  0.991940f, 
     0.039255f, -0.131018f, -0.196708f, -0.924939f,  0.823316f,  0.222749f,  0.995040f, -0.293581f,  0.994267f, -0.868300f, 
     0.890690f,  0.999991f, -0.976043f,  0.975706f,  0.999888f,  0.982261f, -0.981800f, -0.674505f,  0.089142f,  1.000000f, 
    -0.173516f,  0.444987f, -0.745828f,  0.380117f, -0.235145f,  0.956797f,  0.777340f, -0.068140f,  0.961004f, -0.049971f, 
     0.951502f, -0.817341f, -0.780596f,  0.865567f, -0.026399f, -0.014251f, -0.707838f,  0.814265f,  0.950197f, -0.721774f, 
    -0.143707f, -0.960755f,  0.960451f, -0.777549f,  0.952230f, -0.251960f, -0.119291f, -0.286337f, -0.715623f,  0.149448f, 
    -0.992113f, -0.406891f,  0.953863f,  0.984035f,  0.801171f, -0.884013f,  0.999780f,  0.716300f, -0.166289f,  0.842368f, 
     0.094662f, -0.984436f,  0.984725f, -0.126708f, -0.999229f,  0.991380f, -0.980462f,  0.380115f,  0.567584f, -0.974876f, 
    -0.099476f, -0.955934f,  0.106926f, -0.496040f,  0.454610f, -0.004274f, -0.217577f,  0.219082f, -0.014974f, -0.187519f, 
    -0.189918f, -0.738270f, -0.996019f, -0.000222f,  0.984831f, -0.895537f, -0.666139f, -0.924938f,  0.971960f, -0.290756f, 
    -0.629081f,  0.997676f, -0.276534f, -0.998751f, -0.307644f, -0.576155f,  0.625036f, -0.500793f};

static const float test_6_out_scale = 0.0078417705f;
static const float test_6_out_zero_point = 0.0001742840f;
static const int8_t test_6_out_scales_frac[] = {21};
static const int test_6_out_sa_dim = -1;

static const int test_6_out_fx8_frac = 7;

#define TEST_6_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {((2 + 0) * 8 + 4) * 4 + 6 ,(2 + 0) * 8 + 4, (2 + 0), 1}, \
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
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 
    0.500000f, 0.500000f, 0.500000f, 0.500000f};

static const float test_7_out_scale = 0.0019607844f;
static const float test_7_out_zero_point = 0.2500000000f;
static const int8_t test_7_out_scales_frac[] = {23};
static const int test_7_out_sa_dim = -1;

static const int test_7_out_fx8_frac = 8;

#define TEST_7_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 4}, \
    /* .rank =  */ 2


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
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 
    0.009999f, 0.009999f, 0.009999f, 0.009999f, 
    0.999950f, 0.999950f, 0.999950f, 0.999950f};

static const float test_8_out_scale = 0.0039213724f;
static const float test_8_out_zero_point = 0.4999749959f;
static const int8_t test_8_out_scales_frac[] = {22};
static const int test_8_out_sa_dim = -1;

static const int test_8_out_fx8_frac = 7;

#define TEST_8_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 4}, \
    /* .rank =  */ 2


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

tensor_quantizer epsilon_fx16(epsilon_tsr_fx16, epsilon_fx8_frac + 8, epsilon_data, sizeof(epsilon_data) / sizeof(epsilon_data[0]));
tensor_quantizer epsilon_sa8(epsilon_tsr_sa8, epsilon_sa_dim, epsilon_data, 
sizeof(epsilon_data) / sizeof(epsilon_data[0]), &epsilon_scale, 1, &epsilon_zero_point, 1, epsilon_scales_frac, 1);

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

tensor_quantizer test_8_out_fx16(test_8_out_tsr_fx16, test_8_out_fx8_frac + 8, test_8_out_data, sizeof(test_8_out_data) / sizeof(test_8_out_data[0]));
tensor_quantizer test_8_out_sa8(test_8_out_tsr_sa8, test_8_out_sa_dim, test_8_out_data, 
sizeof(test_8_out_data) / sizeof(test_8_out_data[0]), &test_8_out_scale, 1, &test_8_out_zero_point, 1, test_8_out_scales_frac, 1);

const mli_l2_normalize_cfg test_1_cfg = {/*axis=*/-1};
const mli_l2_normalize_cfg test_2_cfg = {/*axis=*/2};
const mli_l2_normalize_cfg test_3_cfg = {/*axis=*/0};
const mli_l2_normalize_cfg test_4_cfg = {/*axis=*/-1};
const mli_l2_normalize_cfg test_5_cfg = {/*axis=*/3};
const mli_l2_normalize_cfg test_6_cfg = {/*axis=*/0};
const mli_l2_normalize_cfg test_7_cfg = {/*axis=*/1};
const mli_l2_normalize_cfg test_8_cfg = {/*axis=*/0};
