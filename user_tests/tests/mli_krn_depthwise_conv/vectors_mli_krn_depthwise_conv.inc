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

// Generated input vectors Declaration
//========================================

extern mli::tst::tensor_quantizer input_1_fx16;
extern mli::tst::tensor_quantizer input_1_memstr_fx16;
extern mli::tst::tensor_quantizer input_1_sa8;
extern mli::tst::tensor_quantizer input_1_memstr_sa8;

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

extern mli::tst::tensor_quantizer bias_1_fx16;
extern mli::tst::tensor_quantizer bias_1_fx8;
extern mli::tst::tensor_quantizer bias_1_sa32_per_axis;

extern mli::tst::tensor_quantizer bias_2_fx16;
extern mli::tst::tensor_quantizer bias_2_fx8;
extern mli::tst::tensor_quantizer bias_2_i1_w2_sa32;
extern mli::tst::tensor_quantizer bias_2_i1_w2_sa32_per_axis;

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
    1.499679f, -0.195957f, -0.101497f, 1.084700f, -0.127001f, 1.921374f, 1.265131f, 1.050833f, 0.828557f, 1.759060f, 
    0.740656f, 0.602529f, 0.770518f, -0.055207f, 1.764953f, 0.056915f, 1.798162f, -0.974392f, -0.193173f, 1.791836f, 
    0.087221f, 0.050762f, -0.795762f, -0.861908f, -0.521138f, 0.569642f, -0.466072f, 0.950747f, 0.105315f, 1.743229f, 
    -0.345538f, 0.211855f, -0.675123f, -0.174242f, -0.210372f, 0.379746f, 1.743715f, -0.732649f, -0.078323f, 0.358176f, 
    -0.621651f, 0.547825f, -0.846128f, -0.170151f, 1.669240f, 1.788261f, 0.269071f, -0.584010f, -0.169977f, -0.151245f, 
    -0.711375f, 0.605190f, -0.521042f, 0.967539f, -0.951895f, 1.476784f, 1.189694f, 0.493913f, 1.585523f, 0.662098f, 
    -0.136224f, -0.129915f, 0.816717f, 1.840368f, 1.695302f, 1.008417f, -0.657568f, 1.430704f, 1.290535f, 0.754588f, 
    0.868083f, 0.100421f, 0.658160f, 0.164856f, 0.491118f, -0.730381f, -0.211975f, -0.864242f, -0.159679f, 1.634750f, 
    -0.874014f, 0.053465f, -0.872567f, 1.262317f, 0.582659f, 1.509164f, 0.571157f, 1.600416f, 1.445452f, 1.635805f, 
    -0.016744f, 1.974257f, 1.793061f, -0.750511f, -0.211884f, -0.806919f, -0.263645f, -0.336632f, 0.576300f, 1.501349f, 
    0.066899f, 1.175289f, -0.585119f, 1.680775f, 1.642291f, 1.548584f, -0.574892f, 1.027800f, 0.628294f, 0.380053f, 
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


static const float bias_1_data[] = {-0.161161f, 0.359392f, -0.036637f};

static const float bias_1_scales[] = {0.00002018, 0.00003019, 0.00004346};
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

static const float bias_2_i1_w2_scale = 0.00004835;
static const float bias_2_i1_w2_zero_point = 0.0f;
static const int8_t bias_2_i1_w2_scale_frac[] = {26};
static const int bias_2_i1_w2_sa_dim = -1;

static const float bias_2_i1_w2_scales[] = {0.00002088, 0.00002737, 0.00004835};
static const float bias_2_i1_w2_zero_points[] = {0.0f, 0.0f, 0.0f};
static const int8_t bias_2_i1_w2_scales_frac[] = {28, 27, 26};
static const int bias_2_i1_w2_sa_dim_per_axis = 0;

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

static const mli_tensor bias_2_i1_w2_tsr_sa32 = {
    BIAS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};


static const float test_1_out_data[] = {
    -0.097726f, 0.343363f, -0.926802f, 0.113766f, 0.752121f, -0.492807f, -0.046339f, 0.504339f, 1.107545f, -0.150885f, 
    0.729244f, -0.234922f, -0.027222f, 0.753985f, 0.125550f, -0.568329f, 0.579819f, 0.386437f, -0.433139f, 1.012842f, 
    -0.052301f, -0.149513f, 0.307539f, -0.007198f, -0.237716f, 0.084325f, -0.990453f, -0.145694f, 0.852486f, -1.460101f, 
    -0.735259f, 0.704190f, 0.838192f, 0.283441f, 0.694336f, 0.472566f, -0.008191f, 0.479724f, 0.374354f, -0.188989f, 
    0.019189f, 0.072077f, -0.383516f, 1.080392f, -0.485482f, 0.013403f, 0.652063f, 0.355017f, -0.335874f, 0.521644f, 
    -0.024881f, -0.394823f, 0.988437f, 0.341600f, 0.169621f, 0.595592f, -0.864213f, -0.328260f, 1.100599f, -0.367612f, 
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
    0.478291f, 1.172920f, 0.0f, 0.164021f, 0.0f, 0.261174f, 0.0f, 0.272800f, 0.0f, 0.627163f, 0.204590f, 0.298193f, 
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
    -0.838294f, 1.155926f, -0.103201f};

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
    -0.237716f, 0.084325f, -0.990453f, 0.283441f, 0.694336f, 0.472566f, -0.040066f, 1.000000f, -0.153388f, 0.086161f, 
    0.698818f, -0.354220f};

static const float test_4_out_scale = 0.0078056967f;
static const float test_4_out_zero_point = 0.0047736764f;
static const int8_t test_4_out_scales_frac[] = {22};
static const int test_4_out_sa_dim = -1;

static const int test_4_out_fx8_frac = 7;

#define TEST_4_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
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
    0.175386f, 0.096368f, 0.0f, 0.831568f, 0.111765f, 0.0f, 0.0f, 0.0f, 0.0f, 0.581390f, 0.627241f, 0.0f, 0.0f, 
    0.784977f, 0.949691f, 0.297051f, 0.904105f, 0.420333f, 0.0f, 0.164021f, 0.0f, 0.0f, 0.0f, 1.373218f, 0.261174f, 
    0.0f, 0.272800f, 0.0f, 1.319053f, 1.564838f, 0.0f, 0.170072f, 1.345576f, 0.509405f, 1.106807f, 0.306912f, 0.0f, 
    0.816069f, 0.0f, 0.0f, 0.212155f, 0.0f, 0.0f, 0.151108f, 0.0f, 0.101346f, 0.0f, 0.960344f, 0.0f, 0.0f, 
    0.0f, 0.0f, 0.885595f, 1.198048f, 0.027674f, 0.0f, 0.864396f, 0.195193f, 0.0f, 0.266053f
};

static const float test_5_out_scale = 0.0061366213f;
static const float test_5_out_zero_point = 0.7824192047f;
static const int8_t test_5_out_scales_frac[] = {22};
static const int test_5_out_sa_dim = -1;

static const int test_5_out_fx8_frac = 6;

#define TEST_5_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 5, 3}, \
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

// Quantizers
//===================================================
tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]));

tensor_quantizer input_1_memstr_fx16(input_1_memstr_tsr_fx16, input_1_fx8_frac + 8, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]));

tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_1_memstr_sa8(input_1_memstr_tsr_sa8, input_1_sa_dim, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);


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
    sizeof(weights_2_data) / sizeof(weights_2_data[0]), &weights_2_scale, 1, &weights_2_zero_point, 1, weights_2_scale_frac, 1);

tensor_quantizer weights_2_memstr_sa8_per_axis(weights_2_memstr_tsr_sa8, weights_2_sa_dim_per_axis, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]), weights_2_scales, 
    sizeof(weights_2_scales) / sizeof(weights_2_scales[0]), weights_2_zero_points, 
    sizeof(weights_2_zero_points) / sizeof(weights_2_zero_points[0]), weights_2_scales_frac, 
    sizeof(weights_2_scales_frac) / sizeof(weights_2_scales_frac[0]));


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

tensor_quantizer bias_2_i1_w2_sa32(bias_2_i1_w2_tsr_sa32, bias_2_i1_w2_sa_dim, bias_2_data, 
    sizeof(bias_2_data) / sizeof(bias_2_data[0]), &bias_2_i1_w2_scale, 1, &bias_2_i1_w2_zero_point, 1, bias_2_i1_w2_scale_frac, 1);

tensor_quantizer bias_2_i1_w2_sa32_per_axis(bias_2_i1_w2_tsr_sa32, bias_2_i1_w2_sa_dim_per_axis, bias_2_data, 
    sizeof(bias_2_data) / sizeof(bias_2_data[0]), bias_2_i1_w2_scales, 
    sizeof(bias_2_i1_w2_scales) / sizeof(bias_2_i1_w2_scales[0]), bias_2_i1_w2_zero_points, 
    sizeof(bias_2_i1_w2_zero_points) / sizeof(bias_2_i1_w2_zero_points[0]), bias_2_i1_w2_scales_frac, 
    sizeof(bias_2_i1_w2_scales_frac) / sizeof(bias_2_i1_w2_scales_frac[0]));


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
