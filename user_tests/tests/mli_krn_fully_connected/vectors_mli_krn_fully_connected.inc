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
extern mli::tst::tensor_quantizer input_1_sa8;

extern mli::tst::tensor_quantizer input_2_fx16;
extern mli::tst::tensor_quantizer input_2_sa8;

extern mli::tst::tensor_quantizer input_3_fx16;
extern mli::tst::tensor_quantizer input_3_sa8;

// Generated weights vectors Declaration
//========================================
extern mli::tst::tensor_quantizer weights_1_fx16;
extern mli::tst::tensor_quantizer weights_1_sa8;
extern mli::tst::tensor_quantizer weights_1_sa8_per_axis;

extern mli::tst::tensor_quantizer weights_2_memstr_fx16;
extern mli::tst::tensor_quantizer weights_2_memstr_sa8;
extern mli::tst::tensor_quantizer weights_2_memstr_sa8_per_axis;

extern mli::tst::tensor_quantizer weights_3_fx16;
extern mli::tst::tensor_quantizer weights_3_sa8;
extern mli::tst::tensor_quantizer weights_3_sa8_per_axis;

extern mli::tst::tensor_quantizer weights_4_fx16;
extern mli::tst::tensor_quantizer weights_4_sa8;
extern mli::tst::tensor_quantizer weights_4_sa8_per_axis;

extern mli::tst::tensor_quantizer bias_1_fx16;
extern mli::tst::tensor_quantizer bias_1_fx8;
extern mli::tst::tensor_quantizer bias_1_sa32;
extern mli::tst::tensor_quantizer bias_1_sa32_per_axis;

extern mli::tst::tensor_quantizer bias_2_fx16;
extern mli::tst::tensor_quantizer bias_2_fx8;
extern mli::tst::tensor_quantizer bias_2_i1_w2_sa32;
extern mli::tst::tensor_quantizer bias_2_i1_w2_sa32_per_axis;

extern mli::tst::tensor_quantizer bias_3_fx16;
extern mli::tst::tensor_quantizer bias_3_fx8;
extern mli::tst::tensor_quantizer bias_3_i2_w3_sa32;
extern mli::tst::tensor_quantizer bias_3_i2_w3_sa32_per_axis;

extern mli::tst::tensor_quantizer bias_4_fx16;
extern mli::tst::tensor_quantizer bias_4_fx8;
extern mli::tst::tensor_quantizer bias_4_i3_w4_sa32;
extern mli::tst::tensor_quantizer bias_4_i3_w4_sa32_per_axis;

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
extern const mli_fully_connected_cfg test_1_cfg;
extern const mli_fully_connected_cfg test_2_cfg;
extern const mli_fully_connected_cfg test_3_cfg;
extern const mli_fully_connected_cfg test_4_cfg;
extern const mli_fully_connected_cfg test_5_cfg;


// Generated input vectors 
//========================================
static const float input_1_data[] = {
    -0.908895f, 0.395841f, 0.649044f, -0.571899f, 0.763308f, -0.734995f, -0.830512f, -0.823701f, -0.848582f, 0.461255f, 
    0.580638f, 0.393583f, 0.693779f, 0.633978f, -0.391279f, -0.434601f
};

static const float input_1_scale = 0.0065576606f;
static const float input_1_zero_point = -0.0727936625f;
static const int8_t input_1_scales_frac[] = {22};
static const int input_1_sa_dim = -1;
static const int input_1_fx8_frac = 7;

#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ {16}, \
    /* .rank =  */ 1

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
    1.519544f, -1.316821f, 4.255250f, 2.226602f, 3.547555f, 6.295907f, -0.826965f, -1.337350f, -0.472715f, -1.574683f, 
    1.473053f, -0.264385f, -0.409806f, 3.729059f, -0.487928f, 0.821070f, -0.037560f, 0.881305f, -1.571814f, -0.891369f, 
    -0.295402f, 5.737370f, 7.732080f, -0.486213f
};

static const float input_2_scale = 0.0364971124f;
static const float input_2_zero_point = 3.0786986351f;
static const int8_t input_2_scales_frac[] = {19};
static const int input_2_sa_dim = -1;
static const int input_2_fx8_frac = 4;

#define INPUT_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ {2, 3, 4}, \
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


static const float input_3_data[] = {
    -674.f, 52680.f, 28859.f, -486.f, 12770.f, 63099.f, 36506.f, 59798.f, 55153.f, 48468.f, 42168.f, 41656.f, 
    16596.f, -8266.f, 31462.f, 13677.f, 8671.f, 27913.f, 37478.f, 46331.f, 17071.f, -4269.f, 61068.f, 12080.f,
    4356.f, -13792.f, -12752.f, -11244.f, 38794.f, 63602.f, 34196.f, 33288.f, 59888.f, 61638.f, -9204.f, 43626.f
};

static const float input_3_scale = 304.4235229492f;
static const float input_3_zero_point = 25022.f;
static const int8_t input_3_scales_frac[] = {6};
static const int input_3_sa_dim = -1;

static const int input_3_fx8_frac = -9;

#define INPUT_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 4, 3}, \
    /* .rank =  */ 3


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

// Generated weights vectors
//========================================
static const float weights_1_data[] = {
    -0.032052f, -0.093988f, -0.421309f, -0.218596f, 0.708457f, -1.049523f, 0.946744f, -1.278018f, -0.827760f, -0.284491f, 
    0.159829f, -0.163374f, 0.128148f, 0.138870f, 0.049217f, -0.575348f, -0.246227f, 0.351754f, 1.128148f, -1.108376f, 
    0.047544f, 0.134401f, 0.211448f, -0.168552f, -0.102179f, -0.445093f, 0.082635f, 0.500884f, -1.302601f, 0.481398f, 
    -0.058289f, 0.133062f, -0.381894f, -0.159978f, -0.246925f, -0.752307f, 1.238527f, 0.560511f, -0.361804f, -0.736474f, 
    -0.150557f, 0.126989f, 0.390134f, -0.056405f, 0.121626f, 0.289270f, -0.584562f, 0.036469f, -0.585951f, 1.986397f, 
    -0.157422f, 0.321986f, -0.078606f, 0.355532f, -0.516700f, -0.586291f, -1.052906f, -1.583220f, 0.339286f, -0.269013f, 
    0.119790f, -0.068143f, -0.577316f, -0.349255f, 0.985706f, -0.275252f, -0.487659f, 0.488674f, 0.660994f, -2.010060f, 
    -0.030322f, -0.123350f, -0.188193f, 0.206856f, -0.127155f, 0.159915f, -0.058445f, -0.075921f, -1.440821f, -0.024006f, 
    0.001644f, 0.108816f, 0.933067f, -0.239177f, -0.286313f, -0.477576f, 0.048699f, 1.161955f, -1.278339f, 0.016663f, 
    0.183060f, 0.155239f, -0.137598f, 0.551259f, 0.430256f, -0.174185f, 0.358912f, -0.114811f, 1.344614f, -1.041834f, 
    -0.115143f, 0.008658f, -0.072136f, -0.366301f, 0.009962f, -1.098268f, -2.557152f, -0.147772f, 0.016725f, 0.689081f, 
    -0.001592f, 0.443742f, 0.005161f, -0.834467f, -1.193272f, -0.225043f, 0.076506f, -0.693056f, -0.397510f, -0.279218f, 
    0.039682f, 0.088732f, -0.374209f, 0.276235f, 0.573385f, -0.266817f, 1.103967f, 0.214483f, 0.421550f, -1.533991f, 
    0.017447f, 0.220219f, -0.045236f, 0.086321f, 0.953155f, 0.483784f, -0.444382f, 0.439544f, -1.235744f, -0.191433f, 
    -0.075254f, -0.373287f, -0.083081f, 0.046965f, -0.494169f, 0.697905f, 0.611067f, -0.756727f, 0.599701f, -0.655624f, 
    -0.199125f, 0.230636f, 0.505653f, 0.129240f, -0.375650f, -0.416485f, 0.302787f, -1.492259f, 0.216486f, 0.132823f
};

static const float weights_1_scale = 0.020135058f;
static const float weights_1_zero_point = 0.0f;
static const int8_t weights_1_scales_frac[] = {20};
static const int weights_1_sa_dim = -1;

static const float weights_1_scales_1[] = {
    0.0015679142f, 0.0034940278f, 0.0073469807f, 0.0065706093f, 0.009395845f, 0.008647776f, 0.020135058f, 
    0.0124662975f, 0.011345044f, 0.01582724f
};
static const float weights_1_zero_points_1[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static const int8_t weights_1_scales_fracs_1[] = {20, 20, 20, 20, 20, 20, 20, 20, 20, 20};
static const int weights_1_sa_dim_1 = 1;

static const int weights_1_fx8_frac = 5;

#define WEIGHTS_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ {16, 10}, \
    /* .rank =  */ 2

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
    0.165477f, 0.068997f, 0.018857f, -0.070015f, -0.171419f, -0.072857f, 1.074393f, 0.478832f, 0.113766f, 0.180469f, 
    0.007470f, -0.362102f, 0.077590f, -0.266461f, 0.337411f, 0.366312f, -0.116023f, -0.296277f, -0.033833f, -0.583667f, 
    0.951421f, 0.316189f, -0.061582f, -0.183663f, -0.009630f, -0.310497f, 0.097745f, 0.448799f, 0.258771f, -0.477061f, 
    -0.036857f, -0.428004f, -0.078752f, -0.159545f, -0.112365f, 0.214311f, 0.606028f, 0.034075f, -0.434215f, -0.193215f, 
    0.005434f, 0.004871f, -0.212723f, 0.094367f, -0.269956f, 0.184606f, 0.215898f, 0.070009f, -0.233572f, -0.074285f, 
    0.260854f, -0.136420f, 0.569657f, -0.013810f, 0.376198f, 0.055545f, 0.067138f, -0.091076f, -0.382014f, 0.188029f, 
    0.319592f, -0.517103f, -0.712752f, 0.420927f, -0.146667f, -0.034667f, 0.109799f, 0.287206f, 0.007287f, -0.062518f, 
    0.596078f, 0.147157f, -0.004379f, 0.152732f, -0.542774f, 0.561854f, 0.553240f, 0.171836f, 0.328378f, -0.039341f, 
    -0.017184f, -0.301906f, -0.091577f, -0.068146f, -0.522880f, 0.850562f, 0.647824f, 1.001104f, 0.120496f, 0.032015f, 
    -0.635666f, 0.336033f, -0.098269f, 0.616363f, 0.361688f, 0.282971f, -0.081876f, 0.155467f, 0.416778f, 0.581155f, 
    0.451897f, -0.232321f, -1.007298f, -0.761464f, 0.068209f, 0.010016f, 0.160997f, 0.025816f, -0.095067f, -0.025573f, 
    -0.238789f, -0.858919f, 0.048224f, 0.049603f, 0.162394f, -0.339993f, 0.076866f, -0.252099f, 0.711314f, 0.597713f, 
    0.115158f, 0.111328f, 0.370476f, -0.402291f, 0.854234f, -0.321867f, 0.165566f, 1.036673f
};

static const float weights_2_scales_2[] = {
    0.0018391474f, 0.0024448556f, 0.0050052465f, 0.0045958045f, 0.0074915006f, 0.006697338f, 
    0.008459788f, 0.008162776f
};
static const float weights_2_zero_points_2[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static const int8_t weights_2_scales_fracs_2[] = {21, 21, 21, 21, 21, 21, 21, 21};
static const int weights_2_sa_dim_2 = 1;

static const int weights_2_fx8_frac = 6;

#define WEIGHTS_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ {16, 8}, \
    /* .rank =  */ 2

#define WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 1 * 10, 1 }, \
    /* .shape = */ {16, 8}, \
    /* .rank =  */ 2

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
    0.097457f, 0.015074f, -0.277234f, 0.627382f, -0.319384f, 0.275178f, -0.108849f, 0.155456f, -0.107056f, 0.049122f, 
    -0.001397f, 0.135584f, -0.071387f, -0.116455f, 0.957716f, -0.109634f, -0.067469f, 0.074831f, 0.125173f, -0.761531f, 
    -0.083283f, -0.802975f, -0.865671f, 0.509259f, 0.136741f, 0.102805f, 0.790759f, -0.195794f, -0.613777f, 0.767797f, 
    0.265791f, -0.582687f, 0.121584f, -0.075150f, -0.078265f, -0.387553f, 0.028563f, 1.621765f, 0.830583f, 0.599023f, 
    -0.124724f, -0.104824f, 0.009574f, 0.643093f, -0.425436f, -0.050905f, 0.982248f, 0.386483f, -0.167748f, -0.151750f, 
    -0.298218f, 0.317917f, -0.749690f, -0.525343f, 0.591371f, -0.164374f, -0.143942f, -0.044116f, -0.022382f, -0.046972f, 
    -0.409024f, -0.067486f, -0.159582f, 0.894585f, 0.139214f, -0.008569f, 0.514483f, -0.080893f, 0.057814f, -0.232124f, 
    -0.277465f, -1.995032f, 0.104527f, 0.105825f, 0.095852f, -0.174129f, -0.167814f, 0.401860f, 0.819575f, -0.081066f, 
    0.214935f, -0.024127f, 0.128330f, 0.908297f, 0.504948f, 0.896274f, 0.485511f, 1.001398f, -0.121791f, 0.074259f, 
    -0.206917f, 0.496773f, -0.353467f, -0.589501f, 0.957454f, 0.265891f, 0.067361f, -0.339442f, -0.089399f, 0.483913f, 
    0.119770f, -0.756203f, 0.402404f, 0.857700f, -0.055121f, 0.111482f, -0.133582f, 0.126972f, -0.989068f, 1.245212f, 
    -1.400330f, -0.972428f, 0.074963f, 0.227611f, 0.101504f, -0.650491f, -1.596210f, 0.354532f, -0.081922f, -0.532497f, 
    0.068938f, 0.271948f, 0.404880f, -0.622163f, 0.201512f, -0.058914f, -0.235090f, 1.076714f, 0.071157f, 0.073390f, 
    0.337247f, 0.835986f, 0.325217f, -0.614370f, -0.903711f, 1.750579f, -0.325262f, 0.094982f, -0.061446f, 0.165219f, 
    0.070897f, -1.064192f, -0.226336f, -0.193685f, -0.077496f, 0.263476f, 0.146926f, 0.020473f, 0.075859f, 0.712951f, 
    0.800052f, -0.145057f, 0.005754f, 0.062949f, 0.449850f, 0.197527f, -0.324643f, -0.105659f, 0.775672f, 0.622120f, 
    -0.023928f, -0.004995f, 0.409109f, -0.115917f, -0.222046f, 0.219908f, 0.185287f, 0.200464f, 0.075711f, -0.300791f, 
    0.056775f, 0.062213f, -0.108139f, -1.737894f, 0.758916f, -1.695432f, -0.050334f, 0.055489f, 0.209034f, 0.206017f, 
    -0.170781f, 0.586594f, 1.065130f, -0.643725f, -0.106767f, 0.110212f, 0.071015f, -0.463168f, 0.186020f, 0.132796f, 
    -0.374275f, 0.479161f
};

static const float weights_3_scales[] = {
    0.0025611166f, 0.002672773f, 0.0062264497f, 0.007151945f, 0.012568581f, 0.013684208f, 
    0.0110262185f, 0.01570891f
};
static const float weights_3_zero_points[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static const int8_t weights_3_scales_fracs[] = {20, 20, 20, 20, 20, 20, 20, 20};
static const int weights_3_sa_dim = 1;

static const int weights_3_fx8_frac = 6;

#define WEIGHTS_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ {24, 8}, \
    /* .rank =  */ 2

static const mli_tensor weights_3_tsr_fx16 = {
    WEIGHTS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_3_tsr_fx8 = {
    WEIGHTS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_3_tsr_sa8 = {
    WEIGHTS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const float weights_4_data[] = {
    -37798.f, 17539.f, 127380.f, -57113.f, -65318.f, 57991.f, 53494.f, 56842.f, -59365.f, -58440.f, 63503.f,
    -92040.f, 8892.f, -74480.f, 11816.f, -35286.f, -126782.f, 58495.f, 7156.f, -26529.f, -20754.f, -32442.f,
    -51703.f, -60096.f, 17366.f, 103456.f, 50471.f, 121316.f, -35268.f, -42236.f, -125680.f, 44329.f, 99714.f,
    -47937.f, 29080.f, -86232.f, 19292.f, 114326.f, -35481.f, 46324.f, 26066.f, 2146.f, 43226.f, -3223.f, 39210.f,
    -25123.f, 49258.f, 22975.f, -65428.f, -89086.f, -17107.f, -85988.f, -17513.f, -17042.f, 38736.f, -68692.f,
    -102390.f, 43565.f, 63790.f, -20337.f, 109634.f, -59175.f, -103964.f, -94434.f, -39244.f, -82820.f, 31054.f,
    49288.f, -60886.f, -11742.f, 12706.f, -54067.f, 90604.f, 63284.f, 43586.f, 21871.f, -30246.f, 19878.f, 39058.f,
    -127358.f, 57170.f, -21140.f, 2194.f, -45860.f, 116342.f, -57831.f, -103494.f, -35483.f, 58410.f, 25385.f,
    -97552.f, 60488.f, -22125.f, -22626.f, 16749.f, -20150.f, 59677.f, 121158.f, 81948.f, -7245.f, 29826.f, 996.f,
    7010.f, 41029.f, -36652.f, -123620.f, 23541.f, 22452.f, -9744.f, -55732.f, 57478.f, 117018.f, -53628.f, 63810.f,
    -87464.f, 31823.f, -123716.f, 63503.f, 12672.f, -79954.f, -58853.f, 86448.f, -16931.f, 119338.f, -45440.f,
    -83848.f, 90362.f, -56106.f, 54554.f, -19455.f, -127304.f, 9487.f, -26436.f, -119728.f, -49397.f, 96052.f,
    -18788.f, -70488.f, 21002.f, 3482.f, -91688.f, 15414.f, 7634.f, 50866.f, 1120.f, -41228.f, 30538.f, -18840.f,
    58404.f, -83276.f, 33897.f, 68956.f, 57567.f, -86608.f, 27224.f, -9065.f, -100688.f, -9560.f, 122090.f, 47452.f,
    68626.f, 113094.f, -18728.f, -104252.f, 15664.f, -52552.f, -10822.f, 51092.f, -78886.f, 39572.f, -105338.f,
    -55573.f, 29064.f, 44153.f, -23358.f, 78382.f, 15229.f, -21368.f, 44620.f, 124750.f, 63649.f, 29526.f, 58982.f,
    54555.f, -5944.f, -13758.f, 42280.f, 6460.f, 49170.f, 109650.f, -33655.f, 105114.f, 50134.f, -15942.f, 60272.f,
    -63198.f, 62784.f, 5220.f, 85012.f, 37534.f, -17198.f, 17480.f, 28386.f, -7600.f, 61116.f, -116220.f, 57591.f,
    126990.f, 63126.f, -21098.f, 91472.f, 55350.f, -44322.f, 813.f, -22858.f, 6584.f, 29170.f, 83158.f, -15723.f,
    23352.f, 18652.f, -63060.f, -35969.f, 112086.f, 50922.f, -38531.f, 109658.f, 51734.f, 38984.f, 41142.f, -91198.f,
    34420.f, 22218.f, 93124.f, 54377.f, -107966.f, -41665.f, -96030.f, 34024.f, -9781.f, -99816.f, 27758.f, 91026.f,
    -43429.f, 65668.f, 7310.f, -2545.f, 50792.f, -40576.f, -115546.f, 57640.f, -82608.f
};

static const float weights_4_scales[] = {973.3858032227f, 502.4409484863f, 1002.9921264648f, 500.0236206055f, 1002.3936767578f, 501.1732177734f, 955.2440795898f};
static const float weights_4_zero_points[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
static const int8_t weights_4_scales_frac[] = {5, 6, 5, 6, 5, 6, 5};
static const int weights_4_sa_dim = 1;

static const int weights_4_fx8_frac = -10;

#define WEIGHTS_4_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3*4*3, 7}, \
    /* .rank =  */ 2


static const mli_tensor weights_4_tsr_fx8 = {
    WEIGHTS_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_4_tsr_fx16 = {
    WEIGHTS_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_4_tsr_sa8 = {
    WEIGHTS_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float bias_1_data[] = {
    -0.018097f, 0.948008f, -0.075758f, -0.929695f, -0.589073f, -0.254283f, 0.302161f, -0.222462f, 
    0.050303f, 0.257336f
};

static const float bias_1_w1_per_axis_sa_spec_data[] = {
    -0.00011308f, 0.86583763f, -0.06195948f, -0.88614585f, -0.62361642f, 0.08568881f,
    0.3485856f, -0.04798725f, 0.2467104f, 0.60665255f
};

static const float bias_1_w1_per_tsr_sa_spec_data[] = {
    0.002244661f, 0.866307026f, -0.059813608f, -0.885848779f, -0.620846767f, 0.087145654f,
    0.348582617f, -0.050967004f, 0.243479676f, 0.605794336f
};

static const float bias_1_scale = 0.00013203887f;
static const float bias_1_zero_point = 0.0f;
static const int8_t bias_1_scales_frac[] = {21};
static const int bias_1_sa_dim = -1;

static const float bias_1_scales_1[] = {
    0.00001028f, 0.00002291f, 0.00004818f, 0.00004309f, 0.00006161f, 0.00005671f, 0.00013204f, 
    0.00008175f, 0.0000744f, 0.00010379f
};
static const float bias_1_zero_points_1[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static const int8_t bias_1_scales_fracs_1[] = {27, 27, 27, 27, 27, 27, 27, 27, 27, 27};
static const int bias_1_sa_dim_1 = 0;

static const int bias_1_fx8_frac = 7;

#define BIAS_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ {10}, \
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


static const float bias_2_data[] = {
    -0.021062f, 0.395218f, -0.262359f, 0.276163f, -0.796129f, -0.345023f, 
    -1.157919f, 0.772353f
};

static const float bias2_w2_per_axis_sa_spec_data[] = {
    -0.022384288f, 0.431194894f, -0.233467936f, 0.220457495f, -1.053326004f, -0.340195567f,
    -1.32694049f, 0.626928252f
};


static const float bias_2_i1_w2_scale = 0.0000535334f;
static const float bias_2_i1_w2_zero_point = 0.0f;
static const int8_t bias_2_i1_w2_scales_frac[] = {32};
static const int bias_2_i1_w2_sa_dim = -1;

static const float bias_2_i1_w2_scales[] = {
    0.00001206050f, 0.00001603253f, 0.00003282271f, 0.00003013773f, 0.00004912672f,
    0.00004391887f, 0.00005547642f, 0.00005352871f
};
static const float bias_2_i1_w2_zero_points[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static const int8_t bias_2_i1_w2_scales_fracs[] = {31, 30, 29, 30, 29, 29, 29, 29};
static const int bias_2_i1_w2_sa_dim_2 = 0;

static const int bias_2_fx8_frac = 6;

#define BIAS_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ {8}, \
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


static const float bias_3_data[] = {
    -0.413987f, -0.036830f, -0.774361f, -0.733625f, 0.407069f, 0.270810f, 
    0.239078f, -0.396362f
};

static const float bias3_w3_per_axis_sa_spec_data[] = {
    -1.01143887f, 1.90641965f, 7.3551735f, 4.55262423f, -14.89876688f, 1.72153521f, 16.13824786f,
    4.83833187f
};

static const float bias_3_i2_w3_scale = 0.0005360940f;
static const float bias_3_i2_w3_zero_point = 0.0f;
static const int8_t bias_3_i2_w3_scales_frac[] = {32};
static const int bias_3_i2_w3_sa_dim = -1;

static const float bias_3_i2_w3_scales[] = {
    0.00009347f, 0.00009755f, 0.00022725f, 0.00026103f, 0.00045872f, 0.00049943f, 
    0.00040243f, 0.00057333f
};
static const float bias_3_i2_w3_zero_points[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static const int8_t bias_3_i2_w3_scales_fracs[] = {24, 24, 24, 24, 24, 24, 24, 24};
static const int bias_3_i2_w3_sa_dim_2 = 0;

static const int bias_3_fx8_frac = 7;

#define BIAS_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ {8}, \
    /* .rank =  */ 1

static const mli_tensor bias_3_tsr_fx16 = {
    BIAS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_3_tsr_fx8 = {
    BIAS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_3_i2_w3_tsr_sa32 = {
    BIAS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float bias_4_data[] = {-830091072.f, 1377955072.f, -361819296.f, -1783368064.f, -236796528.f, -1604637056.f, 1327439232.f};
static const float bias_4_spec_data[] = {3946410153.f, -2239717777.f, -9290104645.f, 9387342839.f, 8364832620.f, 12199258424.f, -3736764339.f};

static const float bias_4_i3_w4_scales[] = {296321.53125f, 152954.84375f, 305334.40625f, 152218.953125f, 305152.21875f, 152568.921875f, 290798.78125f};
static const float bias_4_i3_w4_zero_points[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
static const int8_t bias_4_i3_w4_scales_frac[] = {-4, -3, -4, -3, -4, -3, -4};
static const int bias_4_i3_w4_sa_dim = 0;

static const int bias_4_fx8_frac = -24;

#define BIAS_4_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {7}, \
    /* .rank =  */ 1


static const mli_tensor bias_4_tsr_fx8 = {
    BIAS_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor bias_4_tsr_fx16 = {
    BIAS_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor bias_4_i3_w4_tsr_sa32 = {
    BIAS_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

// Extracted Output vectors 
//===================================================
static const float test_1_out_data[] = {
    0.234904f, 1.404309f, 0.332936f, -0.790707f, -0.043650f, 0.990013f, -1.746598f, 1.905630f, 
    1.290431f, 3.022623f
};

static const float test_1_out_scale = 0.0187028293f;
static const float test_1_out_zero_point = 0.6380126476f;
static const int8_t test_1_out_scales_frac[] = {20};
static const int test_1_out_sa_dim = -1;
static const int test_1_out_fx8_frac = 5;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ {10}, \
    /* .rank =  */ 1

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
    0.234904f, 1.404309f, 0.332936f, 0.0f, 0.0f, 0.990013f, 0.0f, 1.905630f, 
    1.290431f, 3.022623f
};

static const float test_2_out_scale = 0.0118534248f;
static const float test_2_out_zero_point = 1.5113116503f;
static const int8_t test_2_out_scales_frac[] = {21};
static const int test_2_out_sa_dim = -1;
static const int test_2_out_fx8_frac = 5;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ {10}, \
    /* .rank =  */ 1

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


static const float test_3_out_data[] = {-0.044243f, 0.384159f, -0.666019f, 0.449173f, -0.456802f, 1.0f, -1.0f, -1.0f};

static const float test_3_out_scale = 0.0078431377f;
static const float test_3_out_zero_point = 0.0f;
static const int8_t test_3_out_scales_frac[] = {21};
static const int test_3_out_sa_dim = -1;
static const int test_3_out_fx8_frac = 7;

#define TEST_3_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ {8}, \
    /* .rank =  */ 1

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


static const float test_4_out_data[] = {0.0f, 0.0f, 1.917626f, 2.554689f, 0.0f, 3.271442f, 6.0f, 0.0f};

static const float test_4_out_scale = 0.0235294122f;
static const float test_4_out_zero_point = 3.0000000000f;
static const int8_t test_4_out_scales_frac[] = {20};
static const int test_4_out_sa_dim = -1;
static const int test_4_out_fx8_frac = 4;

#define TEST_4_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ {8}, \
    /* .rank =  */ 1

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
    -7333843968.f, -6154032128.f, 3043232256.f, 11726278656.f, 3220792832.f, 6539892224.f, -29828861952.f
};

static const float test_5_out_scale = 275452512.f;
static const float test_5_out_zero_point = -3701690368.f;
static const int8_t test_5_out_scales_frac[] = {-14};
static const int test_5_out_sa_dim = -1;

static const int test_5_out_fx8_frac = -29;

#define TEST_5_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {7}, \
    /* .rank =  */ 1


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

tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, 
    &input_1_zero_point, 1, input_1_scales_frac, 1);


tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx8_frac + 8, input_2_data,
    sizeof(input_2_data) / sizeof(input_2_data[0]));

tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data,
    sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1,
    &input_2_zero_point, 1, input_2_scales_frac, 1);

    
tensor_quantizer input_3_fx16(input_3_tsr_fx16, input_3_fx8_frac + 8, input_3_data,
    sizeof(input_3_data) / sizeof(input_3_data[0]));
tensor_quantizer input_3_sa8(input_3_tsr_sa8, input_3_sa_dim, input_3_data, 
    sizeof(input_3_data) / sizeof(input_3_data[0]),
    &input_3_scale, 1, &input_3_zero_point, 1, input_3_scales_frac, 1);

tensor_quantizer weights_1_fx16(weights_1_tsr_fx16, weights_1_fx8_frac + 8, weights_1_data,
    sizeof(weights_1_data) / sizeof(weights_1_data[0]));

tensor_quantizer weights_1_fx8(weights_1_tsr_fx8, weights_1_fx8_frac, weights_1_data,
    sizeof(weights_1_data) / sizeof(weights_1_data[0]));

tensor_quantizer weights_1_sa8(weights_1_tsr_sa8, weights_1_sa_dim, weights_1_data,
    sizeof(weights_1_data) / sizeof(weights_1_data[0]), &weights_1_scale, 1,
    &weights_1_zero_point, 1, weights_1_scales_frac, 1);

tensor_quantizer weights_1_sa8_per_axis(weights_1_tsr_sa8, weights_1_sa_dim_1, weights_1_data,
    sizeof(weights_1_data) / sizeof(weights_1_data[0]), weights_1_scales_1, 10,
    weights_1_zero_points_1, 10, weights_1_scales_fracs_1, 10);


tensor_quantizer weights_2_memstr_fx16(weights_2_memstr_tsr_fx16, weights_2_fx8_frac + 8, weights_2_data,
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));

tensor_quantizer weights_2_memstr_fx8(weights_2_memstr_tsr_fx8, weights_2_fx8_frac, weights_2_data,
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));

tensor_quantizer weights_2_memstr_sa8_per_axis(weights_2_memstr_tsr_sa8, weights_2_sa_dim_2, weights_2_data,
    sizeof(weights_2_data) / sizeof(weights_2_data[0]), weights_2_scales_2, 8, 
    weights_2_zero_points_2, 8, weights_2_scales_fracs_2, 8);


tensor_quantizer weights_3_fx16(weights_3_tsr_fx16, weights_3_fx8_frac + 8, weights_3_data,
    sizeof(weights_3_data) / sizeof(weights_3_data[0]));

tensor_quantizer weights_3_fx8(weights_3_tsr_fx8, weights_3_fx8_frac, weights_3_data,
    sizeof(weights_3_data) / sizeof(weights_3_data[0]));

tensor_quantizer weights_3_sa8_per_axis(weights_3_tsr_sa8, weights_3_sa_dim, weights_3_data,
    sizeof(weights_3_data) / sizeof(weights_3_data[0]), weights_3_scales, 8,
    weights_3_zero_points, 8, weights_3_scales_fracs, 8);

tensor_quantizer weights_4_fx8(weights_4_tsr_fx8, weights_4_fx8_frac, weights_4_data,
    sizeof(weights_4_data) / sizeof(weights_4_data[0]));

tensor_quantizer weights_4_fx16(weights_4_tsr_fx16, weights_4_fx8_frac + 8, weights_4_data,
    sizeof(weights_4_data) / sizeof(weights_4_data[0]));

tensor_quantizer weights_4_sa8(weights_4_tsr_sa8, weights_4_sa_dim, weights_4_data, 
    sizeof(weights_4_data) / sizeof(weights_4_data[0]),
    weights_4_scales, sizeof(weights_4_scales) / sizeof(weights_4_scales[0]), 
    weights_4_zero_points, sizeof(weights_4_zero_points) / sizeof(weights_4_zero_points[0]), 
    weights_4_scales_frac, sizeof(weights_4_scales_frac) / sizeof(weights_4_scales_frac[0]));

tensor_quantizer bias_1_fx16(bias_1_tsr_fx16, bias_1_fx8_frac + 8, bias_1_data,
    sizeof(bias_1_data) / sizeof(bias_1_data[0]));

tensor_quantizer bias_1_fx8(bias_1_tsr_fx8, bias_1_fx8_frac, bias_1_data,
    sizeof(bias_1_data) / sizeof(bias_1_data[0]));

tensor_quantizer bias_1_sa32(bias_1_tsr_sa32, bias_1_sa_dim, bias_1_data,
    sizeof(bias_1_data) / sizeof(bias_1_data[0]), &bias_1_scale, 1,
    &bias_1_zero_point, 1, bias_1_scales_frac, 1);

tensor_quantizer bias_1_sa32_spec(bias_1_tsr_sa32, bias_1_sa_dim, bias_1_w1_per_tsr_sa_spec_data,
    sizeof(bias_1_data) / sizeof(bias_1_data[0]), &bias_1_scale, 1,
    &bias_1_zero_point, 1, bias_1_scales_frac, 1);

tensor_quantizer bias_1_sa32_per_axis(bias_1_tsr_sa32, bias_1_sa_dim_1, bias_1_data,
    sizeof(bias_1_data) / sizeof(bias_1_data[0]), bias_1_scales_1, 10,
    bias_1_zero_points_1, 10, bias_1_scales_fracs_1, 10);

tensor_quantizer bias_1_sa32_per_axis_spec(bias_1_tsr_sa32, bias_1_sa_dim_1, bias_1_w1_per_axis_sa_spec_data,
    sizeof(bias_1_data) / sizeof(bias_1_data[0]), bias_1_scales_1, 10,
    bias_1_zero_points_1, 10, bias_1_scales_fracs_1, 10);


tensor_quantizer bias_2_fx16(bias_2_tsr_fx16, bias_2_fx8_frac + 8, bias_2_data,
    sizeof(bias_2_data) / sizeof(bias_2_data[0]));

tensor_quantizer bias_2_fx8(bias_2_tsr_fx8, bias_2_fx8_frac, bias_2_data,
    sizeof(bias_2_data) / sizeof(bias_2_data[0]));

tensor_quantizer bias_2_i1_w2_sa32(bias_2_i1_w2_tsr_sa32, bias_2_i1_w2_sa_dim, bias_2_data,
    sizeof(bias_2_data) / sizeof(bias_2_data[0]), &bias_2_i1_w2_scale, 1,
    &bias_2_i1_w2_zero_point, 1, bias_2_i1_w2_scales_frac, 1);

tensor_quantizer bias_2_i1_w2_sa32_per_axis(bias_2_i1_w2_tsr_sa32, bias_2_i1_w2_sa_dim_2, bias_2_data,
    sizeof(bias_2_data) / sizeof(bias_2_data[0]), bias_2_i1_w2_scales, 8,
    bias_2_i1_w2_zero_points, 8, bias_2_i1_w2_scales_fracs, 8);

tensor_quantizer bias_2_i1_w2_sa32_per_axis_spec(bias_2_i1_w2_tsr_sa32,
    bias_2_i1_w2_sa_dim_2, bias2_w2_per_axis_sa_spec_data,
    sizeof(bias_2_data) / sizeof(bias_2_data[0]), bias_2_i1_w2_scales, 8,
    bias_2_i1_w2_zero_points, 8, bias_2_i1_w2_scales_fracs, 8);


tensor_quantizer bias_3_fx16(bias_3_tsr_fx16, bias_3_fx8_frac + 8, bias_3_data,
    sizeof(bias_3_data) / sizeof(bias_3_data[0]));

tensor_quantizer bias_3_fx8(bias_3_tsr_fx8, bias_3_fx8_frac, bias_3_data,
    sizeof(bias_3_data) / sizeof(bias_3_data[0]));

tensor_quantizer bias_3_i2_w3_sa32(bias_3_i2_w3_tsr_sa32, bias_3_i2_w3_sa_dim, bias_3_data,
    sizeof(bias_3_data) / sizeof(bias_3_data[0]), &bias_3_i2_w3_scale, 1,
    &bias_3_i2_w3_zero_point, 1, bias_3_i2_w3_scales_frac, 1);

tensor_quantizer bias_3_i2_w3_sa32_per_axis(bias_3_i2_w3_tsr_sa32, bias_3_i2_w3_sa_dim_2, bias_3_data,
    sizeof(bias_3_data) / sizeof(bias_3_data[0]), bias_3_i2_w3_scales, 8,
    bias_3_i2_w3_zero_points, 8, bias_3_i2_w3_scales_fracs, 8);

tensor_quantizer bias_3_i2_w3_sa32_per_axis_spec(bias_3_i2_w3_tsr_sa32, bias_3_i2_w3_sa_dim_2, bias3_w3_per_axis_sa_spec_data,
    sizeof(bias_3_data) / sizeof(bias_3_data[0]), bias_3_i2_w3_scales, 8,
    bias_3_i2_w3_zero_points, 8, bias_3_i2_w3_scales_fracs, 8);

tensor_quantizer bias_4_fx8(bias_4_tsr_fx8, bias_4_fx8_frac, bias_4_data,
    sizeof(bias_4_data) / sizeof(bias_4_data[0]));

tensor_quantizer bias_4_fx16(bias_4_tsr_fx16, bias_4_fx8_frac + 8, bias_4_data,
    sizeof(bias_4_data) / sizeof(bias_4_data[0]));

tensor_quantizer bias_4_i3_w4_sa32(bias_4_i3_w4_tsr_sa32, bias_4_i3_w4_sa_dim, bias_4_data, 
    sizeof(bias_4_data) / sizeof(bias_4_data[0]),
    bias_4_i3_w4_scales, sizeof(bias_4_i3_w4_scales) / sizeof(bias_4_i3_w4_scales[0]), 
    bias_4_i3_w4_zero_points, sizeof(bias_4_i3_w4_zero_points) / sizeof(bias_4_i3_w4_zero_points[0]), 
    bias_4_i3_w4_scales_frac, sizeof(bias_4_i3_w4_scales_frac) / sizeof(bias_4_i3_w4_scales_frac[0]));

tensor_quantizer bias_4_i3_w4_sa32_spec(bias_4_i3_w4_tsr_sa32, bias_4_i3_w4_sa_dim, bias_4_spec_data, 
    sizeof(bias_4_spec_data) / sizeof(bias_4_spec_data[0]),
    bias_4_i3_w4_scales, sizeof(bias_4_i3_w4_scales) / sizeof(bias_4_i3_w4_scales[0]), 
    bias_4_i3_w4_zero_points, sizeof(bias_4_i3_w4_zero_points) / sizeof(bias_4_i3_w4_zero_points[0]), 
    bias_4_i3_w4_scales_frac, sizeof(bias_4_i3_w4_scales_frac) / sizeof(bias_4_i3_w4_scales_frac[0]));

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
    sizeof(test_5_out_data) / sizeof(test_5_out_data[0]),
    &test_5_out_scale, 1, &test_5_out_zero_point, 1, test_5_out_scales_frac, 1);
    

const mli_fully_connected_cfg test_1_cfg = {/* .relu.type = */ MLI_RELU_NONE};
const mli_fully_connected_cfg test_2_cfg = {/* .relu.type = */ MLI_RELU_GEN};
const mli_fully_connected_cfg test_3_cfg = {/* .relu.type = */ MLI_RELU_1};
const mli_fully_connected_cfg test_4_cfg = {/* .relu.type = */ MLI_RELU_6};
const mli_fully_connected_cfg test_5_cfg = {/* .relu.type = */ MLI_RELU_NONE};
