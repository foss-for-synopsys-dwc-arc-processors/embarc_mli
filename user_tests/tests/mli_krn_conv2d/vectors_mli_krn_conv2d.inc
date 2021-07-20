/*
* Copyright 2019-2021, Synopsys, Inc.
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

// Generated weights vectors Declaration
//========================================

extern mli::tst::tensor_quantizer weights_1_fx16;
extern mli::tst::tensor_quantizer weights_1_fx8;
extern mli::tst::tensor_quantizer weights_1_sa8;

extern mli::tst::tensor_quantizer weights_2_fx16;
extern mli::tst::tensor_quantizer weights_2_fx8;
extern mli::tst::tensor_quantizer weights_2_sa8;

extern mli::tst::tensor_quantizer weights_2_memstr_fx16;
extern mli::tst::tensor_quantizer weights_2_memstr_fx8;
extern mli::tst::tensor_quantizer weights_2_memstr_sa8;

extern mli::tst::tensor_quantizer weights_3_memstr_fx16;
extern mli::tst::tensor_quantizer weights_3_memstr_fx8;
extern mli::tst::tensor_quantizer weights_3_memstr_sa8;

extern mli::tst::tensor_quantizer weights_4_memstr_fx16;
extern mli::tst::tensor_quantizer weights_4_memstr_fx8;
extern mli::tst::tensor_quantizer weights_4_memstr_sa8;

extern mli::tst::tensor_quantizer weights_5_memstr_fx16;
extern mli::tst::tensor_quantizer weights_5_memstr_fx8;
extern mli::tst::tensor_quantizer weights_5_memstr_sa8;

extern mli::tst::tensor_quantizer weights_6_fx16;
extern mli::tst::tensor_quantizer weights_6_fx8;
extern mli::tst::tensor_quantizer weights_6_sa8;

extern mli::tst::tensor_quantizer bias_1_fx16;
extern mli::tst::tensor_quantizer bias_1_fx8;
extern mli::tst::tensor_quantizer bias_1_sa32;
extern mli::tst::tensor_quantizer bias_1_w2_sa32;
extern mli::tst::tensor_quantizer bias_1_w2_per_tensor_sa32;
extern mli::tst::tensor_quantizer bias_1_w3_sa32;
extern mli::tst::tensor_quantizer bias_1_w4_sa32;
extern mli::tst::tensor_quantizer bias_1_w5_sa32;

extern mli::tst::tensor_quantizer bias_2_fx16;
extern mli::tst::tensor_quantizer bias_2_fx8;
extern mli::tst::tensor_quantizer bias_2_i2_w6_sa32;


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

extern mli::tst::tensor_quantizer test_11_out_fx16;
extern mli::tst::tensor_quantizer test_11_out_sa8;

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
    -0.795256615f, -0.020611584f, -0.857786238f, -0.116273165f, 0.808907866f, -0.496227741f, -0.107275546f, -0.577136159f,
    1.860774279f, 0.930648685f, 1.950116396f, 1.566918135f, 0.589830399f, -0.950313091f, 0.618810654f, -0.962112069f,
    1.893128872f, -0.582279801f, 1.939082861f, -0.719187677f, 1.063522339f, 1.109600544f, 1.949721098f, -0.861565351f,
    0.142124653f, -0.689962804f, 1.484006405f, 1.528692007f, -0.571019948f, -0.376762152f, -0.228962064f, 1.455527067f,
    -0.751569092f, 0.435355544f, 0.228472352f, 1.998298407f, 0.528104305f, 0.669281363f, 1.963288307f, -0.903628111f,
    -0.531457663f, 0.510696411f, -0.32744509f, -0.207189679f, -0.556229711f, -0.869290471f, -0.953169167f, 1.123666763f,
    -0.810569465f, 0.351910353f, 0.803937912f, 0.019652009f, -0.049965858f, 0.62204957f, 0.900914192f, 0.991265893f,
    1.660414934f, 0.52547884f, -0.973419189f, 1.005276918f, 0.573811293f, 0.358962655f, 0.842565656f, 1.532562733f,
    0.070606351f, -0.333183408f, 0.164556742f, 1.287754774f, 0.305704474f, 0.906447887f, -0.721507311f, 1.87564373f,
    0.568320155f, -0.962049961f, 1.042712688f, 0.401276469f, 1.858359337f, -0.10336417f, -0.097108901f, 1.075629711f,
    0.781458378f, 0.160974979f, 0.460997581f, 0.180617571f, -0.582523704f, -0.813086033f, 1.32623148f, -0.808524489f,
    -0.4848876f, 1.06777668f, 0.391311884f, -0.760229707f, 0.372676373f, 1.040057898f, -0.851530135f, 1.943858385f,
    0.56839323f, 0.60644412f, 1.348023891f, -0.123570859f, -0.975745201f, 0.277356744f, 0.937498331f, 1.385216713f,
    1.256720543f, -0.355837345f, 1.507807732f, -0.943606257f, -0.654643536f, 0.141403437f, -0.733771205f, 1.77818656f,
    0.254359484f, 0.034042001f, 1.820943594f, 0.470926762f, 0.997800827f, -0.947544813f, -0.973272562f, 1.56578064f,
    -0.972122073f, -0.172074139f, -0.780366302f, -0.018243432f, -0.442839742f, -0.381048918f, -0.363562047f, -0.161547184f,
    1.9712286f, -0.079811931f, 0.247272134f, 0.334599495f, 1.427205563f, 1.751028538f, 1.87986064f, -0.979990005f,
    0.431378245f, -0.714987516f, 1.234764576f, -0.018693268f, -0.588499546f, -0.086845994f, 1.424705029f, 1.664449692f,
    1.395023346f, 0.789302588f, -0.676198661f
};

static const float input_1_scale = 0.011679563f;
static const float input_1_zero_point = 0.509154201f;
static const int8_t input_1_scales_frac[] = { 21 };
static const int input_1_sa_dim = -1;
static const int input_1_fx8_frac = 6;


#define INPUT_1_TSR_SHARED_DESCR \
    /*.data = */ { 0 }, \
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ { 7, 7, 3 }, \
    /* .rank = */ 3


static const mli_tensor input_1_tsr_sa8 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */MLI_EL_SA_8,
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
    /*.el_params = */ { 0 }
};

tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data,
                              sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data,
                             sizeof(input_1_data) / sizeof(input_1_data[0]),
                             &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);


#define INPUT_1_MEMSTR_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /* .mem_stride = */ { 13*6*1, 6*1, 1}, \
    /* .shape = */ { 6, 7, 3 }, \
    /* .rank = */  3


static const mli_tensor input_1_memstr_tsr_sa8 = {
    INPUT_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_1_memstr_tsr_fx8 = {
    INPUT_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_1_memstr_tsr_fx16 = {
    INPUT_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_1_memstr_fx16(input_1_memstr_tsr_fx16, input_1_fx8_frac + 8, 
                                     input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_memstr_sa8(input_1_memstr_tsr_sa8, input_1_sa_dim,input_1_data,
                                    sizeof(input_1_data) / sizeof(input_1_data[0]),
                                    &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

static const float input_2_data[] = {
    -674.f, 52680.f, 28859.f, -486.f, 12770.f, 63099.f, 36506.f, 59798.f, 55153.f, 48468.f, 42168.f, 41656.f, 7762.f,
    13766.f, 55561.f, 53583.f, 51199.f, 35859.f, 16596.f, -8266.f, 31462.f, 13677.f, 8671.f, 27913.f, 37478.f, 46331.f,
    17071.f, -4269.f, 61068.f, 12080.f, -8963.f, -11318.f, 2334.f, 34704.f, 50401.f, 38689.f, 4356.f, -13792.f, -12752.f,
    -11244.f, 38794.f, 63602.f, 34196.f, 33288.f, 59888.f, 61638.f, -9204.f, 43626.f, 1818.f, 19120.f, -1348.f, 30510.f,
    36178.f, -10044.f, 10256.f, 21676.f, 25908.f, 58067.f, 49974.f, 23021.f, 46667.f, 30566.f, 59008.f, 1322.f, 9146.f,
    60414.f, 52671.f, 47712.f, -7561.f, 18550.f, 20058.f, 55690.f, 51432.f, -1271.f, -11979.f, -1017.f, 6651.f, 11956.f,
    15458.f, -9137.f, 52790.f, 29327.f, 41480.f, 8407.f, 32171.f, -7421.f, 2472.f, 43455.f, 10774.f, 22176.f, -5968.f,
    11464.f, 22987.f, 63836.f, -5504.f, 5478.f, 32273.f, 45561.f, 54836.f, 7976.f, 48722.f, -13464.f, 46133.f, 34716.f,
    39587.f, 40679.f, 3592.f, 40094.f
};

static const float input_2_scale = 304.4235229492f;
static const float input_2_zero_point = 25022.f;
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

tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx8_frac + 8, input_2_data,
                              sizeof(input_2_data) / sizeof(input_2_data[0]));
tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data, 
                             sizeof(input_2_data) / sizeof(input_2_data[0]),
                             &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);

// Generated weights vectors
//========================================
static const float weights_1_data[] = {
    0.121539377f, 0.038981315f, 0.238740608f, -0.227366433f, -1.423697591f, -1.002833486f, -0.483930886f, 0.100503266f,
    -0.139419615f, 0.436625689f, -0.532677531f, 0.427570045f, -0.307641238f, 0.172509328f, -0.026271259f, 0.425837994f,
    -0.288608491f, 0.288695902f, 0.13327387f, 1.159123302f, -0.318038374f, 0.075211421f, -0.132531196f, -0.11317683f,
    0.227300689f, 0.278074265f, 0.686661422f, -0.682077289f, 0.161586493f, -0.164802447f, 0.799581826f, -0.564442813f,
    -0.396176994f, 0.236066803f, -0.308561832f, -0.13669531f, 0.100037456f, -0.632492363f, -0.20194006f, -0.16947329f,
    0.454397738f, 1.601993442f, 0.25155741f, 0.518023372f, -0.107394643f, -0.029803216f, 0.441627651f, -0.720269144f,
    0.308284104f, 0.007023496f, 0.221299991f, 0.276993394f, 0.671937406f, -0.216236874f, -0.195448056f, -0.383312225f,
    -0.080290787f, -0.104013339f, 0.05112939f, 0.828456044f, 0.519129336f, -0.083231583f, 2.043134451f, 0.073937632f,
    -0.242123961f, -0.151708424f, -0.477973312f, 0.270733327f, 0.314504147f, -1.155037999f, 0.117129698f, 0.158947885f,
    -0.653497517f, 0.022939235f, -0.248011231f, 0.169625223f, 0.447813988f, -0.003180373f, -0.02626197f, 0.45241487f,
    0.059201516f, -0.067172617f, -0.345578939f, -0.327685088f, 0.039380759f, 0.021876009f, -0.546851277f, -0.177791998f,
    0.220277116f, -0.3018246f, -0.381017804f, 0.045308102f, 0.088454835f, 0.085999437f, 0.003846063f, -0.091018461f,
    -0.90490979f, -0.382344782f, 0.111589707f, -0.155282184f, 0.004263277f, -0.185914531f, 0.176349491f, 1.426594615f,
    -0.694690406f, 0.048129093f, 0.401795059f, -0.437018067f, -0.051059466f, -0.027009116f, -0.500988185f, -0.742791295f,
    0.154998183f, -0.228159264f, 0.042742822f, -0.466717333f, 0.563317478f, 0.247517765f, 1.114162087f, -0.04944453f,
    -0.45109126f, 0.174054205f, 0.289583415f, 0.326953709f, 0.316643655f, -1.05029583f, 0.071101822f, -0.114677481f,
    -0.15621753f, -0.548054278f, 0.439401656f, -0.105688043f, 1.153027654f, 0.078443088f, 0.274467528f, -0.085858665f,
    0.174084067f, -0.442948669f, 0.486325681f, -0.919982672f, -0.119199298f, -0.039415505f, -0.064425007f, -0.059484687f,
    -0.174552381f, -0.087301873f, 0.108685292f, -0.063917093f, 0.198944986f, 0.054434042f, 0.144530609f, 0.372599989f,
    0.437938243f, -0.193609491f, -0.043178584f, -0.089182042f, 0.491961032f, -0.406788021f, -0.061561186f, 0.465097427f,
    -0.788807869f, 0.059773225f, 0.264351815f, -0.156099781f, 0.214262009f, -1.20488143f, -0.064080834f, -0.20592095f,
    0.090724051f, 0.371410072f, 0.183330402f, -0.18974714f, 0.08137878f, -0.699047506f, 0.548414052f, -0.199038938f,
    -0.032763209f, 0.292395085f, 0.731802106f, 0.663361192f, -0.041131146f, -1.132700443f, -0.004237484f, -0.079310201f,
    -0.007845246f, 0.379243433f, -0.613977134f, 0.311061054f, 0.012565632f, -0.027481252f, -0.082511358f, -0.262109607f,
    -0.0922819f, 0.770100713f, 0.823133767f, -0.929636836f, -0.002228576f, -0.276713401f, -0.73335439f, 0.097396046f,
    0.771846235f, -0.24578993f, -1.138297677f, 0.041287374f, 0.363226146f, -0.208227709f, -0.247083694f, 0.43671298f,
    0.435876101f, -0.459643662f, -0.014785624f, -0.329277813f, 0.320278764f, 0.17764692f, -0.373586893f, 0.096097499f,
    -0.30733937f, 0.111224152f, 0.173332736f, 0.082185537f, -0.08492846f, -0.085071102f, 0.104451664f, 0.481835127f,
    -0.014706351f, -0.059814841f, -0.294265062f, 0.699438274f, 0.426571071f, -0.23051928f, -0.545369089f, -0.035941798f,
    0.170828208f, 0.146618083f, -0.666299522f, 0.733037353f, -0.224516228f, -0.740856171f, 0.036224466f, -0.28800568f,
    0.165565833f, -0.201040745f, -0.399984539f, 0.824885786f, -0.741401076f, -0.11888241f, -0.185713932f, -0.254682899f,
    -0.426853001f, -0.0115926f, 0.517828643f, -0.662262499f
};

static const float weights_1_scales[] = { 0.001980767f, 0.004078924f, 0.00629592f, 0.006523276f, 0.011210217f, 0.011233028f, 0.016087674f };
static const float weights_1_zero_points[] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
static const int8_t weights_1_scales_frac[] = { 23, 22, 22, 22, 21, 21, 20};
static const int weights_1_sa_dim = 3;
static const int weights_1_fx8_frac = 5;

#define WEIGHTS_1_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 4, 3, 7}, \
    /* .rank = */  4


static const mli_tensor weights_1_tsr_sa8 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_1_tsr_fx8 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_1_tsr_fx16 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_1_fx16(weights_1_tsr_fx16, weights_1_fx8_frac + 8, weights_1_data, 
                                       sizeof(weights_1_data) / sizeof(weights_1_data[0]));
tensor_quantizer weights_1_fx8(weights_1_tsr_fx8, weights_1_fx8_frac, weights_1_data, 
                                      sizeof(weights_1_data) / sizeof(weights_1_data[0]));
tensor_quantizer weights_1_sa8(weights_1_tsr_sa8, weights_1_sa_dim, weights_1_data, 
                                      sizeof(weights_1_data) / sizeof(weights_1_data[0]),
                                      weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                                      weights_1_zero_points, sizeof(weights_1_zero_points) / sizeof(weights_1_zero_points[0]),
                                      weights_1_scales_frac, sizeof(weights_1_scales_frac) / sizeof(weights_1_scales_frac[0]));

static const float weights_2_data[] = {
    -0.001188097f, 0.158430964f, 0.033262957f, -0.633298337f, -0.1396368f, 0.305942148f, -0.591342807f, -0.036758021f,
    0.232412338f, -0.178800613f, 0.080816858f, 0.481414646f, -0.508018792f, 0.708433688f, 0.008211083f, -0.177805245f,
    -0.376847386f, -0.13070485f, -0.336240917f, 1.120456338f, 0.502928853f, -0.036441471f, -0.020082472f, -0.340529919f,
    -0.0340732f, -1.01299572f, 0.221572533f, -0.337901235f, -0.089239083f, 0.300923795f, -0.217772543f, -0.115643874f,
    -0.272480905f, 0.303629249f, 0.32568109f, -0.03776769f, 0.204064757f, -0.107153647f, -0.035749901f, 0.70189321f,
    -0.097298183f, -0.378773808f, -0.112897478f, 0.141247198f, -0.024257997f, 0.139342397f, -0.194187388f, 0.006307581f,
    -0.032069135f, 0.077514552f, 0.093124047f, 0.455886424f, 0.254220098f, 0.122940816f, -0.381508678f, -0.00430326f,
    -0.06681338f, -0.266334146f, 0.234112456f, 0.596589029f, -0.956062317f, -0.022876985f, -0.759260952f, -0.138965741f,
    -0.032402743f, -0.162632108f, 0.144866064f, -0.888238132f, -0.057009783f, 0.228629276f, 0.04191963f, 0.001033493f,
    0.143752724f, -0.404819876f, -0.184150264f, 0.132288873f, -0.06303715f, 0.036310095f, -0.014957452f, 0.028623329f,
    0.933860242f, 0.046874043f, -0.613898635f, 0.127052143f, -0.15063639f, 0.122040689f, -0.240940541f, 0.126907215f,
    -0.432094723f, 0.053468354f, 0.568058372f, 0.006249467f, 0.070788667f, -0.612377226f, -0.1148412f, -0.807911396f,
    0.072652124f, 0.692732394f, -0.015197292f, -0.070149951f, -0.304998279f, -0.03614476f, -1.098420262f, -0.74943161f,
    -0.739642859f, 0.121449545f, 0.057953008f, -0.186060697f, 0.074108623f, 0.317548603f, 0.438116908f, -1.105246902f,
    -0.037531693f, -0.022304447f, -0.113697089f, -0.101493947f, -0.076509289f, -0.509375155f, -0.925535202f, 0.167301416f,
    -0.136420399f, -0.189717218f, 0.152643248f, 0.639494777f, -0.752813995f, 0.1146871f, -0.163981646f, 0.012904882f,
    0.674863279f, -0.293668032f, 0.475825787f, -0.080334291f, 0.464278638f, -0.114536725f, -0.177354872f, -0.130889043f,
    -0.31429866f, 0.042126495f, 0.971474588f, -0.113744892f, -0.122762255f, -0.296576411f, 0.083119228f, 0.036537632f,
    -0.436970115f, -0.850781262f, -0.648254752f, 0.044025764f, -0.056047019f, -0.083228536f, 0.349574864f, 0.113673046f,
    -0.702896714f, 1.108441234f, -0.061534978f, -0.333887547f, -0.020187877f, -0.768115163f, -0.109793335f, 0.013849158f,
    -0.116258159f, -0.083387949f, 0.050520938f, -0.556613564f, -0.626676619f, -0.440787971f, -0.434929341f, 0.238853112f,
    0.051639326f, 0.016697606f, -0.083599262f, -0.031606335f, -0.217136189f, 0.05245354f, -0.295803815f, 0.002231798f,
    0.070277952f, -0.06251011f, 0.433514416f, 0.259230793f, 0.740238488f, 0.560710132f, -0.215274885f, -0.228348419f,
    0.286687076f, 0.299754798f, 0.244238213f, 0.297419786f, -0.37243551f, 0.011011983f, -0.214312419f, -0.176681072f,
    0.643409312f, -0.074228197f, 0.98294723f, -0.698858798f, -0.09920653f, -0.016969206f, -0.061422843f, 0.305310547f,
    0.337030739f, 0.823659658f, 0.506570637f, 0.12859568f, -0.279233128f, -0.238742113f, -0.305386782f, -0.654793024f,
    -0.130563483f, -0.492927164f, 0.003850749f, -0.121581905f, 0.154081061f, 0.051160473f, -0.250011474f, -0.567618132f,
    0.863942802f, -0.14374198f, -0.083179489f, -0.016421042f, 0.42755881f, 0.977829158f, -0.52741611f, -0.621286809f,
    -0.027401676f, 0.055229425f, -0.21033898f, -0.015223696f, -0.315685958f, -0.112352826f, 0.216336951f, 0.095024176f,
    -0.170186132f, 0.126614869f, -0.612870872f, 0.21274367f, -0.77916801f, -0.044506028f, 0.18726787f, 0.039244618f,
    0.166786805f, -0.182345957f, -0.290119529f, -0.236063749f, 0.226787373f, 0.018864114f, -0.104759611f, -0.389850825f,
    -0.110729717f, 0.965348959f, -0.071828105f, 0.657099783f
};

static const float weights_2_scales[] = { 0.001695078f, 0.002629036f, 0.005313884f, 0.00735323f, 0.008648979f, 0.00882249f, 0.008727884f };
static const float weights_2_per_tensor_scales[] = { 0.00882249f };
static const float weights_2_zero_points[] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
static const float weights_2_per_tensor_zero_points[] = { 0.f };
static const int8_t weights_2_scales_frac[] = { 24, 23, 22, 22, 21, 21, 21 };
static const int8_t weights_2_per_tensor_scales_frac[] = { 21 };
static const int weights_2_sa_dim = 3;
static const int weights_2_per_tensor_sa_dim = -1;
static const int weights_2_fx8_frac = 6;

#define WEIGHTS_2_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ { 4, 3, 3, 7 }, \
    /* .rank = */  4


static const mli_tensor weights_2_tsr_sa8 = {
    WEIGHTS_2_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_2_tsr_fx8 = {
    WEIGHTS_2_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_2_tsr_fx16 = {
    WEIGHTS_2_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_2_fx16(weights_2_tsr_fx16, weights_2_fx8_frac + 8, weights_2_data,
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_fx8(weights_2_tsr_fx8, weights_2_fx8_frac, weights_2_data,
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_sa8(weights_2_tsr_sa8, weights_2_per_tensor_sa_dim, weights_2_data,
    sizeof(weights_2_data) / sizeof(weights_2_data[0]),
    weights_2_per_tensor_scales, sizeof(weights_2_per_tensor_scales) / sizeof(weights_2_per_tensor_scales[0]),
    weights_2_per_tensor_zero_points, sizeof(weights_2_per_tensor_zero_points) / sizeof(weights_2_per_tensor_zero_points[0]),
    weights_2_per_tensor_scales_frac, sizeof(weights_2_per_tensor_scales_frac) / sizeof(weights_2_per_tensor_scales_frac[0]));

#define WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /* .mem_stride = */ {3*3*8*1, 3*8*1, 8*1, 1}, \
    /* .shape = */ { 4, 3, 3, 7 }, \
    /* .rank = */  4


static const mli_tensor weights_2_memstr_tsr_sa8 = {
    WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_2_memstr_tsr_fx8 = {
    WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_2_memstr_tsr_fx16 = {
    WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_2_memstr_fx16(weights_2_memstr_tsr_fx16, weights_2_fx8_frac + 8, weights_2_data,
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_memstr_fx8(weights_2_memstr_tsr_fx8, weights_2_fx8_frac, weights_2_data,
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_memstr_sa8(weights_2_memstr_tsr_sa8, weights_2_sa_dim, weights_2_data,
    sizeof(weights_2_data) / sizeof(weights_2_data[0]),
    weights_2_scales, sizeof(weights_2_scales) / sizeof(weights_2_scales[0]),
    weights_2_zero_points, sizeof(weights_2_zero_points) / sizeof(weights_2_zero_points[0]),
    weights_2_scales_frac, sizeof(weights_2_scales_frac) / sizeof(weights_2_scales_frac[0]));

static const float weights_3_data[] = {
-0.065f, -0.114f, 0.06f, 0.566f, -0.657f, -1.442f, 0.533f, -0.23f, -0.2f, -0.521f, -0.286f, -0.413f, -0.071f, 
-1.362f, -0.141f, -0.205f, 0.468f, -0.193f, 0.543f, 0.589f, -0.999f};

static const float weights_3_scales[] = {
0.0018110237f, 0.0016141732f, 0.0041023623f, 0.004456693f, 0.005173228f, 0.011354331f, 0.0107244095f};
static const float weights_3_zero_points[] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
static const int8_t weights_3_scales_frac[] = {24, 24, 22, 22, 22, 21, 21};
static const int weights_3_sa_dim = 3;

static const int weights_3_fx8_frac = 6;

#define WEIGHTS_3_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {1*(3+2)*(7+1)*1, (3+2)*(7+1)*1, (7+1)*1, 1}, \
    /* .shape = */ {1, 1, 3, 7}, \
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
    0.062f, 0.124f, -0.668f, -0.249f, -1.317f, 0.308f, 0.878f, -0.068f, -0.177f, 0.468f, 0.407f, -0.135f, -0.341f, 
    -0.8f, -0.098f, 0.045f, -0.308f, 0.205f, 1.412f, 0.572f, -0.238f, -0.061f, -0.176f, 0.362f, 0.616f, -0.773f, 1.104f, 
    0.213f, -0.158f, 0.133f, -0.011f, -0.278f, -0.342f, -0.039f, -0.567f, 0.064f, 0.381f, -0.512f, 0.809f, 0.776f, 
    -0.172f, 1.445f, -0.174f, 0.174f, 0.299f, 0.662f, 0.326f, 0.134f, 1.248f, -0.065f, -0.127f, 0.33f, -0.756f, -0.345f, 
    -0.018f, -0.154f, -0.113f, -0.145f, 0.33f, 0.148f, 0.149f, 0.138f, 1.356f, 0.043f, 0.031f, -0.006f, -0.012f, 0.052f, 
    -0.627f, 0.565f, -0.151f, 0.293f, -0.095f, -0.092f, -0.968f, -0.196f, 0.339f, -0.158f, 0.055f, -0.391f, -0.259f, 
    -1.133f, 0.726f, -0.067f, -0.038f, -0.45f, -0.315f, 0.248f, 0.112f, -0.411f, -0.007f, 0.153f, -0.01f, 0.231f, 
    0.589f, 0.191f, 1.011f, -0.047f, 0.1f, -0.119f, 0.276f, -0.257f, -0.105f, 0.481f, -0.938f, -0.026f, 0.055f, 0.459f, 
    -0.538f, -0.714f, -0.975f, -0.431f, -0.099f, 0.139f, -0.193f, 0.071f, -0.046f, 0.976f, -0.864f, 0.042f, 0.302f, 
    0.094f, -0.361f, 0.569f, 0.342f, -0.386f, 0.042f, 0.193f, -0.162f, 0.946f, 0.77f, 0.205f, -0.566f, 0.122f, 0.1f, 
    0.129f, -0.511f, 0.065f, -0.85f, 0.158f, 0.032f, -0.326f, 0.068f, 0.188f, -0.508f, -0.555f, -0.199f, 0.074f, 
    -0.021f, 0.322f, 0.189f, -0.208f, -0.235f, -0.753f, 0.073f, -0.26f, 0.047f, -0.159f, 0.244f, 0.474f, -0.682f, 
    -0.015f, 0.053f, -0.284f, -0.189f, 1.117f, 0.107f, -0.114f, 0.017f, 0.032f, 0.161f, 0.205f, 0.123f, -1.433f, 
    -0.511f, -0.024f, -0.088f, 0.032f, -0.029f, 0.056f, -0.377f, 0.864f, -0.054f, 0.177f, -0.268f, 0.356f, -0.236f, 
    0.277f, 0.467f
};

static const float weights_4_scales[] = {0.0013700787f, 0.003543307f, 0.0052598426f, 0.0074488185f, 0.011118109f, 0.011283465f, 0.011377953f};
static const float weights_4_zero_points[] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
static const int8_t weights_4_scales_frac[] = {24, 23, 22, 22, 21, 21, 21};
static const int weights_4_sa_dim = 3;

static const int weights_4_fx8_frac = 6;

#define WEIGHTS_4_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {3*(3+2)*(7+2)*1, (3+2)*(7+2)*1, (7+2)*1, 1}, \
    /* .shape = */ {3, 3, 3, 7}, \
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
    -0.029f, -0.22f, -0.136f, 0.281f, 0.072f, -0.162f, -0.973f, 0.066f, 0.027f, 0.612f, -0.402f, -0.162f, 0.182f, 
    0.715f, -0.03f, 0.166f, 0.434f, 0.6f, 0.048f, -0.985f, 0.498f, -0.081f, -0.354f, 0.022f, 0.099f, -0.42f, 
    -0.093f, -0.77f, -0.037f, 0.239f, 0.603f, 0.065f, 0.895f, 0.098f, -0.456f, -0.071f, -0.185f, -0.278f, 
    -0.432f, 0.568f, -0.35f, 0.131f, 0.194f, -0.072f, -0.134f, -0.483f, 0.759f, -1.059f, 0.133f, 0.119f, -0.37f, 
    0.295f, -0.135f, -0.556f, 0.354f, 0.12f, -0.199f, 0.011f, -0.411f, -0.32f, -0.021f, -0.495f, -0.712f, 0.04f, 
    0.046f, -0.748f, -0.234f, -0.187f, -0.263f, -0.322f, -0.119f, 0.092f, -0.414f, 0.264f, -0.036f, -0.492f, 
    -1.139f, -0.243f, 0.224f, 0.12f, -0.56f, -0.04f, 0.737f, 0.374f, 0.058f, 0.099f, 0.293f, 0.022f, -0.079f, 
    -0.796f, 0.202f, -0.038f, -0.281f, 0.358f, -0.886f, -0.877f, -0.247f, 0.036f, -0.043f, -0.116f, 0.087f, 
    0.166f, -0.09f, 0.283f, -1.028f, 0.148f, -0.438f, -0.209f, 0.122f, 0.638f, 0.383f, 0.466f, 0.006f, 0.226f, 
    0.075f, 0.161f, -0.342f, -0.038f, -1.303f, 0.034f, -0.241f, -0.024f, 0.202f, 0.762f, 1.615f, 0.12f, 0.158f, 
    -0.452f, 0.626f, -0.134f, -0.366f, -0.228f, -0.895f, -0.075f, -0.006f, -0.23f, -0.366f, 0.057f, 0.107f, 
    0.689f, -0.1f, 0.163f, 0.038f, 0.511f, -0.154f, 0.176f, -0.174f, 0.081f, -0.269f, 0.663f, -0.688f, -0.265f, 
    -0.607f, 0.338f, 0.092f, -0.009f, 0.507f, 0.119f, 0.362f, 0.42f, 0.302f, -0.111f, 0.211f, -0.046f, -0.317f, 
    0.148f, -0.725f, 0.639f, 0.057f, -0.041f, 0.406f, -0.037f, -0.129f, 0.439f, -0.376f, -0.079f, 0.123f, 0.07f, 
    -0.021f, -0.316f, 0.634f, 0.279f, 0.128f, -0.087f, 0.116f, -0.223f, -0.106f, 0.359f, 0.617f, 0.029f, -0.004f, 
    -0.747f, 0.242f, 0.382f, -0.19f, 0.826f, 0.095f, 0.235f, -0.049f, -0.081f, -0.16f, 1.114f, 0.439f, 0.134f, 
    -0.217f, 0.147f, 1.043f, -0.429f, -0.207f, -1.353f, 0.232f, -0.099f, 0.372f, -0.017f, -0.16f, -0.1f, -0.264f, 
    0.221f, 0.043f, 0.754f, 0.336f, -0.492f, 0.659f, -0.224f, 0.073f, 0.271f, 0.261f, 0.391f, 0.177f, -0.101f, 
    -1.18f, -0.111f, -0.347f, -0.053f, -0.192f, 0.633f, 1.521f, -0.213f, 0.167f, -0.235f, 0.158f, -0.528f, 
    0.096f, -0.499f, -0.166f, 0.2f, 0.057f, 0.247f, 0.168f, -0.477f, 0.985f, -0.334f, 0.168f, 0.147f, -0.67f, 
    -0.121f, 0.166f, 0.056f, -0.374f, -0.081f, 0.25f, -0.111f, -0.26f, 0.183f, -1.715f, -0.711f, -0.117f, 
    -0.032f, 0.091f, 0.517f, 0.035f, -0.345f, 0.31f, 0.04f, 0.017f, 0.036f, 0.076f, 0.465f, 0.064f, 0.877f, 
    -0.094f, 0.213f, -0.004f, 0.864f, -0.006f, -0.025f, 0.536f, 0.069f, -0.1f, 0.382f, -0.607f, 0.908f, 0.582f, 
    0.411f, -0.066f, -0.192f, 0.254f, 0.641f, 0.764f, 0.173f, -0.076f, -0.172f, 0.064f, 0.445f, -0.115f, 0.219f, 
    0.402f, 0.081f, 0.055f, -0.103f, 0.963f, 0.394f, -0.135f, 0.714f, -1.106f, -0.018f, 0.15f, 0.191f, 0.151f, 
    0.263f, 0.002f, -0.726f, -0.057f, 0.036f, -0.47f, -0.343f, 0.717f, -0.249f, -1.019f, 0.069f, 0.286f, -0.249f, 
    -0.557f, 1.251f, -1.265f, -0.111f, -0.041f, -0.281f, 0.354f, 0.46f, 0.576f, 0.099f, 0.057f, -0.14f, 0.352f, 
    -0.335f, -0.247f, -0.018f, -0.414f, -0.431f, 0.087f, 0.11f, 0.378f, 0.027f, 0.196f, 0.172f, -0.919f, 0.109f, 
    0.216f, 0.008f, -0.613f, -0.124f, -0.34f, 0.35f, -0.087f, -0.063f, 0.063f, 0.227f, 0.187f, 1.193f, -0.724f, 
    0.062f, -0.256f, -0.375f, -0.342f, 0.097f, -1.073f, -1.411f, -0.103f, -0.278f, 0.442f, -0.386f, 0.058f, 
    -0.365f, -0.273f, 0.062f, -0.429f, 0.041f, 0.118f, 0.494f, 0.855f, 0.96f, 0.067f, 0.159f, -0.357f, -0.406f, 
    0.754f, 0.461f, 0.785f, -0.201f, -0.179f, -0.339f, -0.144f, -0.662f, -0.096f, 0.434f, 0.045f, 0.151f, 
    -0.428f, 0.538f, 0.081f, 0.237f, -0.09f, -0.089f, -0.029f, -0.171f, 0.204f, -0.45f, 0.076f, 0.502f, -0.182f, 
    0.066f, -0.113f, 0.241f, 0.226f, 0.547f, -0.31f, 0.133f, 0.162f, -0.227f, 0.322f, -0.022f, 0.543f, -1.479f, 
    -0.014f, 0.3f, -0.145f, 0.158f, 0.241f, 0.827f, -0.875f, -0.117f, 0.108f, 0.122f, 0.165f, 0.921f, 0.721f, 
    0.231f, 0.041f, -0.192f, -0.311f, 0.451f, -0.148f, -0.53f, 0.568f, 0.112f, -0.171f, -0.315f, -0.218f,0.173f, 
    0.085f, 0.776f, -0.043f, 0.128f, -0.296f, 0.598f, 0.749f, 0.285f, 0.178f, -0.095f, 0.178f, -0.038f, 0.603f, 
    -0.298f, -0.579f, 0.611f, -0.032f, 0.298f, 0.38f, -0.342f, 0.182f, -0.479f, 0.698f, -0.01f, -0.119f, 0.247f, 
    -0.177f, -0.392f, 0.505f, 0.334f, -0.123f, 0.432f, 0.056f, 0.491f, 1.194f, -0.261f, -0.378f, 0.104f, 0.277f, 
    -0.034f, 0.137f, 0.225f, 0.134f, -0.048f, -0.002f, 0.371f, 0.041f, -0.185f, 0.146f, 0.016f, 1.011f, 0.029f, 
    -0.351f, -0.162f, -1.203f, -0.577f, -0.578f, -0.75f, -0.131f, 0.191f, -0.119f, -0.017f, -0.68f, 0.707f, 
    0.503f
};

static const float weights_5_scales[] = {0.0019133858f, 0.003559055f, 0.007582677f, 0.009472441f, 0.009850394f, 0.013503937f, 0.011645669f};
static const float weights_5_zero_points[] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
static const int8_t weights_5_scales_frac[] = {24, 23, 22, 21, 21, 21, 21};
static const int weights_5_sa_dim = 3;

static const int weights_5_fx8_frac = 6;

#define WEIGHTS_5_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {5*(3+2)*(7+2)*1, (3+2)*(7+2)*1, (7+2)*1, 1}, \
    /* .shape = */ {5, 5, 3, 7}, \
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

static const float weights_6_data[] = {
    -37798.f, 17539.f, 127380.f, -57113.f, -65318.f, 57991.f, 53494.f, 56842.f, -59365.f, -58440.f,
    63503.f, -92040.f, 8892.f, -74480.f, 11816.f, -35286.f, -126782.f, 58495.f, 7156.f, -26529.f,
    -20754.f, -32442.f, -51703.f, -60096.f, 17366.f, 103456.f, 50471.f, 121316.f, -35268.f, -42236.f,
    -125680.f, 44329.f, 99714.f, -47937.f, 29080.f, -86232.f, 19292.f, 114326.f, -35481.f, 46324.f,
    26066.f, 2146.f, 43226.f, -3223.f, 39210.f, -25123.f, 49258.f, 22975.f, -65428.f, -89086.f, -17107.f,
    -85988.f, -17513.f, -17042.f, 38736.f, -68692.f, -102390.f, 43565.f, 63790.f, -20337.f, 109634.f,
    -59175.f, -103964.f, -94434.f, -39244.f, -82820.f, 31054.f, 49288.f, -60886.f, -11742.f, 12706.f,
    -54067.f, 90604.f, 63284.f, 43586.f, 21871.f, -30246.f, 19878.f, 39058.f, -127358.f, 57170.f, -21140.f,
    2194.f, -45860.f, 116342.f, -57831.f, -103494.f, -35483.f, 58410.f, 25385.f, -97552.f, 60488.f, -22125.f,
    -22626.f, 16749.f, -20150.f, 59677.f, 121158.f, 81948.f, -7245.f, 29826.f, 996.f, 7010.f, 41029.f,
    -36652.f, -123620.f, 23541.f, 22452.f, -9744.f, -55732.f, 57478.f, 117018.f, -53628.f, 63810.f, -87464.f,
    31823.f, -123716.f, 63503.f, 12672.f, -79954.f, -58853.f, 86448.f, -16931.f, 119338.f, -45440.f,
    -83848.f, 90362.f, -56106.f, 54554.f, -19455.f, -127304.f, 9487.f, -26436.f, -119728.f, -49397.f,
    96052.f, -18788.f, -70488.f, 21002.f, 3482.f, -91688.f, 15414.f, 7634.f, 50866.f, 1120.f, -41228.f,
    30538.f, -18840.f, 58404.f, -83276.f, 33897.f, 68956.f, 57567.f, -86608.f, 27224.f, -9065.f, -100688.f,
    -9560.f, 122090.f, 47452.f, 68626.f, 113094.f, -18728.f, -104252.f, 15664.f, -52552.f, -10822.f, 51092.f,
    -78886.f, 39572.f, -105338.f, -55573.f, 29064.f, 44153.f, -23358.f, 78382.f, 15229.f, -21368.f, 44620.f,
    124750.f, 63649.f, 29526.f, 58982.f, 54555.f, -5944.f, -13758.f, 42280.f, 6460.f, 49170.f, 109650.f,
    -33655.f, 105114.f, 50134.f, -15942.f, 60272.f, -63198.f, 62784.f, 5220.f, 85012.f, 37534.f, -17198.f,
    17480.f, 28386.f, -7600.f, 61116.f, -116220.f, 57591.f, 126990.f, 63126.f, -21098.f, 91472.f, 55350.f,
    -44322.f, 813.f, -22858.f, 6584.f, 29170.f, 83158.f, -15723.f, 23352.f, 18652.f, -63060.f, -35969.f,
    112086.f, 50922.f, -38531.f, 109658.f, 51734.f, 38984.f, 41142.f, -91198.f, 34420.f, 22218.f, 93124.f,
    54377.f, -107966.f, -41665.f, -96030.f, 34024.f, -9781.f, -99816.f, 27758.f, 91026.f, -43429.f, 65668.f,
    7310.f, -2545.f, 50792.f, -40576.f, -115546.f, 57640.f, -82608.f
};

static const float weights_6_scales[] = {973.3858032227f, 502.4409484863f, 1002.9921264648f, 500.0236206055f, 1002.3936767578f, 501.1732177734f, 955.2440795898f};
static const float weights_6_zero_points[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
static const int8_t weights_6_scales_frac[] = {5, 6, 5, 6, 5, 6, 5};
static const int weights_6_sa_dim = 3;

static const int weights_6_fx8_frac = -10;

#define WEIGHTS_6_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 4, 3, 7}, \
    /* .rank =  */ 4


static const mli_tensor weights_6_tsr_fx8 = {
    WEIGHTS_6_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_6_tsr_fx16 = {
    WEIGHTS_6_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor weights_6_tsr_sa8 = {
    WEIGHTS_6_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


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

tensor_quantizer weights_6_fx8(weights_6_tsr_fx8, weights_6_fx8_frac, weights_6_data,
                               sizeof(weights_6_data) / sizeof(weights_6_data[0]));
tensor_quantizer weights_6_fx16(weights_6_tsr_fx16, weights_6_fx8_frac + 8, weights_6_data,
                                sizeof(weights_6_data) / sizeof(weights_6_data[0]));
tensor_quantizer weights_6_sa8(weights_6_tsr_sa8, weights_6_sa_dim, weights_6_data, 
                               sizeof(weights_6_data) / sizeof(weights_6_data[0]),
                               weights_6_scales, sizeof(weights_6_scales) / sizeof(weights_6_scales[0]), 
                               weights_6_zero_points, sizeof(weights_6_zero_points) / sizeof(weights_6_zero_points[0]), 
                               weights_6_scales_frac, sizeof(weights_6_scales_frac) / sizeof(weights_6_scales_frac[0]));


static const float bias_1_data[] = { 0.478962898f, -0.367202461f, 0.641515911f, 0.216426253f, 0.080585465f, -0.270597309f, 0.376852006f };
static const float bias_1_scales[] = { 0.0000231345f, 0.00004764f, 0.0000735336f, 0.000076189f, 0.0001309304f, 0.0001311969f, 0.000187897f};
static const float bias_1_zero_points[] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
static const int8_t bias_1_scales_frac[] = { 30, 29, 28, 28, 27, 27, 27 };
static const int8_t bias_1_per_tensor_scales_frac[] = { 27 };
static const int bias_1_sa_dim = 0;
static const int bias_1_fx8_frac = 7;

static const float bias_1_w2_scales[] = {
    0.0000197978f, 0.000030706f, 0.0000620638f, 0.0000858825f, 0.0001010163f, 0.0001030428f, 0.0001019379f 
 };
static const float bias_1_w2_per_tensor_scales[] = { 0.0001030428f, };
static const int8_t bias_1_w2_scales_frac[] = { 30, 29, 28, 28, 28, 28, 28 };
static const int8_t bias_1_w2_per_tensor_scales_frac[] = { 28 };
static const float bias_1_w2_per_tensor_zero_points[] = { 0.f };
static const int bias_1_per_tensor_sa_dim = -1;

static const float bias_1_w3_scales[] = {
    0.0000211519654f, 0.0000188528376f, 0.0000479137989f, 0.0000520522267f, 0.0000604210423f, 0.0001326136242f, 0.0001252564164f 
};
static const int8_t bias_1_w3_scales_frac[] = {30, 30, 29, 29, 29, 27, 27};

static const float bias_1_w4_scales[] = {
    0.0000160019205f, 0.0000413842773f, 0.0000614326630f, 0.0000869989449f, 0.0001298546545f, 0.0001317859403f, 0.0001328895189f
};
static const int8_t bias_1_w4_scales_frac[] = {30, 29, 28, 28, 27, 27, 27};

static const float bias_1_w5_scales[] = {
    0.00002234751f, 0.0000415682071f, 0.0000885623537f, 0.0001106339714f, 0.0001150482973f, 0.0001577200829f, 0.0001360163248f
};
static const int8_t bias_1_w5_scales_frac[] = {30, 29, 28, 28, 28, 27, 27};


#define BIAS_1_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {7}, \
    /* .rank = */  1


static const mli_tensor bias_1_tsr_sa32 = {
    BIAS_1_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_1_tsr_fx8 = {
    BIAS_1_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_1_tsr_fx16 = {
    BIAS_1_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const float bias_2_data[] = {-830091072.f, 1377955072.f, -361819296.f, -1783368064.f, -236796528.f, -1604637056.f, 1327439232.f};
static const float bias_2_i2_w6_scales[] = {296321.53125f, 152954.84375f, 305334.40625f, 152218.953125f, 305152.21875f, 152568.921875f, 290798.78125f};
static const float bias_2_i2_w6_zero_points[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
static const int8_t bias_2_i2_w6_scales_frac[] = {-4, -3, -4, -3, -4, -3, -4};
static const int bias_2_i2_w6_sa_dim = 0;

static const int bias_2_fx8_frac = -24;

#define BIAS_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {7}, \
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


static const mli_tensor bias_2_i2_w6_tsr_sa32 = {
    BIAS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer bias_1_fx16(bias_1_tsr_fx16, bias_1_fx8_frac + 8, bias_1_data,
                             sizeof(bias_1_data) / sizeof(bias_1_data[0]));
tensor_quantizer bias_1_fx8(bias_1_tsr_fx8, bias_1_fx8_frac, bias_1_data, 
                            sizeof(bias_1_data) / sizeof(bias_1_data[0]));
tensor_quantizer bias_1_sa32(bias_1_tsr_sa32, bias_1_sa_dim, bias_1_data,
                             sizeof(bias_1_data) / sizeof(bias_1_data[0]),
                             bias_1_scales, sizeof(bias_1_scales) / sizeof(bias_1_scales[0]),
                             bias_1_zero_points, sizeof(bias_1_zero_points) / sizeof(bias_1_zero_points[0]),
                             bias_1_scales_frac, sizeof(bias_1_scales_frac) / sizeof(bias_1_scales_frac[0]));
                                   

tensor_quantizer bias_1_w2_sa32(bias_1_tsr_sa32, bias_1_sa_dim, bias_1_data,
                                sizeof(bias_1_data) / sizeof(bias_1_data[0]),
                                bias_1_w2_scales, sizeof(bias_1_w2_scales) / sizeof(bias_1_w2_scales[0]),
                                bias_1_zero_points, sizeof(bias_1_zero_points) / sizeof(bias_1_zero_points[0]),
                                bias_1_w2_scales_frac, sizeof(bias_1_w2_scales_frac) / sizeof(bias_1_w2_scales_frac[0]));

tensor_quantizer bias_1_w2_per_tensor_sa32(bias_1_tsr_sa32, bias_1_per_tensor_sa_dim, bias_1_data,
    sizeof(bias_1_data) / sizeof(bias_1_data[0]),
    bias_1_w2_per_tensor_scales, sizeof(bias_1_w2_per_tensor_scales) / sizeof(bias_1_w2_per_tensor_scales[0]),
    bias_1_w2_per_tensor_zero_points, sizeof(bias_1_w2_per_tensor_zero_points) / sizeof(bias_1_w2_per_tensor_zero_points[0]),
    bias_1_w2_per_tensor_scales_frac, sizeof(bias_1_w2_per_tensor_scales_frac) / sizeof(bias_1_w2_per_tensor_scales_frac[0]));

tensor_quantizer bias_1_w3_sa32(bias_1_tsr_sa32, bias_1_sa_dim, bias_1_data,
                                sizeof(bias_1_data) / sizeof(bias_1_data[0]),
                                bias_1_w3_scales, sizeof(bias_1_w3_scales) / sizeof(bias_1_w3_scales[0]),
                                bias_1_zero_points, sizeof(bias_1_zero_points) / sizeof(bias_1_zero_points[0]),
                                bias_1_w3_scales_frac, sizeof(bias_1_w3_scales_frac) / sizeof(bias_1_w3_scales_frac[0]));

tensor_quantizer bias_1_w4_sa32(bias_1_tsr_sa32, bias_1_sa_dim, bias_1_data,
                                sizeof(bias_1_data) / sizeof(bias_1_data[0]),
                                bias_1_w4_scales, sizeof(bias_1_w4_scales) / sizeof(bias_1_w4_scales[0]),
                                bias_1_zero_points, sizeof(bias_1_zero_points) / sizeof(bias_1_zero_points[0]),
                                bias_1_w4_scales_frac, sizeof(bias_1_w4_scales_frac) / sizeof(bias_1_w4_scales_frac[0]));

tensor_quantizer bias_1_w5_sa32(bias_1_tsr_sa32, bias_1_sa_dim, bias_1_data,
                                sizeof(bias_1_data) / sizeof(bias_1_data[0]),
                                bias_1_w5_scales, sizeof(bias_1_w5_scales) / sizeof(bias_1_w5_scales[0]),
                                bias_1_zero_points, sizeof(bias_1_zero_points) / sizeof(bias_1_zero_points[0]),
                                bias_1_w5_scales_frac, sizeof(bias_1_w5_scales_frac) / sizeof(bias_1_w5_scales_frac[0]));

tensor_quantizer bias_2_fx8(bias_2_tsr_fx8, bias_2_fx8_frac, bias_2_data,
                            sizeof(bias_2_data) / sizeof(bias_2_data[0]));
tensor_quantizer bias_2_fx16(bias_2_tsr_fx16, bias_2_fx8_frac + 8, bias_2_data,
                             sizeof(bias_2_data) / sizeof(bias_2_data[0]));
tensor_quantizer bias_2_i2_w6_sa32(bias_2_i2_w6_tsr_sa32, bias_2_i2_w6_sa_dim, bias_2_data, 
                                   sizeof(bias_2_data) / sizeof(bias_2_data[0]),
                                   bias_2_i2_w6_scales, sizeof(bias_2_i2_w6_scales) / sizeof(bias_2_i2_w6_scales[0]), 
                                   bias_2_i2_w6_zero_points, sizeof(bias_2_i2_w6_zero_points) / sizeof(bias_2_i2_w6_zero_points[0]), 
                                   bias_2_i2_w6_scales_frac, sizeof(bias_2_i2_w6_scales_frac) / sizeof(bias_2_i2_w6_scales_frac[0]));

// Extracted Output vectors 
//===================================================
static const float test_1_out_data[] = {
    0.535328388f, -0.221749097f, -1.524278402f, 1.549519181f, 1.215887189f, -1.485789418f, -3.493813038f, 0.360762626f,
    -0.376689941f, -0.057757556f, -0.464838862f, -1.955297709f, 2.368459463f, -2.457421303f, 0.502509952f, 1.423059464f,
    -1.285085678f, 4.264955044f, 1.281325817f, 1.840798497f, -10.452395439f, 0.344621837f, 0.75870806f, 1.432294846f,
    2.087225914f, -2.11764431f, -3.513283014f,
    -2.168069601f, 0.372738004f, -0.360535502f, 1.355887175f, 0.360819638f, 0.958503306f, -2.431846857f, 0.690559626f,
    0.620592475f, 1.104650736f, -1.257079363f, 1.243721843f, -2.325690746f, 2.846367598f, -1.435010672f, 0.136842549f,
    -1.265274763f, -0.545728743f, 0.306806445f, 1.21085012f, 0.26382795f, -10.976535797f, 0.826310098f, -1.348274708f,
    -0.607777297f, -0.780666173f, 2.986452103f, 4.337287903f, -0.246931463f, 0.546687961f, -1.727869272f, 1.866532087f,
    -1.532512903f, 1.56780386f, 1.558582783f, -2.939911604f, 0.21952635f, -0.239676908f, 2.272932529f, 2.806646347f,
    -0.103639379f, 8.606427193f, -11.131134033f, 0.603303134f, -1.624239683f, -0.558938086f, 0.720144689f, 3.11555934f,
    -0.174777776f, 6.196098328f, 0.241342127f, -0.016108692f, -0.735345423f, 1.148536205f, 1.377587557f, 3.344902039f,
    -2.246030569f, 0.351803362f, 2.010327101f, -0.7744928f, 1.889515519f, -1.44070363f, -1.481731772f, 1.649418354f,
    -0.339770615f, -1.550270557f, -0.048358917f, 0.905905008f, 1.102876663f, 3.742504358f, -5.023919106f, 1.01051414f,
    -0.58178854f, -1.655605793f, -3.067297935f, 1.408858776f, 7.106957436f, 0.154514045f, 0.811818719f, -0.024253339f,
    -1.962688208f, -0.479133546f, 0.770052791f, 2.981967211f, -2.215245962f, 0.186186433f, -1.367815733f, 1.907425165f,
    1.422053456f, 1.793094635f, -1.298845768f,
    -6.626911163f, 0.048112571f, -1.272145748f, -3.239808321f, 0.479271412f, 1.548650861f, 4.541175842f, 2.681740284f,
    1.06196928f, 1.287585258f, 2.691403389f, 0.075811028f, -1.801050663f, 3.8727777f, -0.641383529f, 0.731324196f,
    -1.213650227f, 1.138151407f, -0.917452216f, 1.654405832f, 2.661809683f, 0.500878811f, -0.134554505f, -0.234148443f,
    -0.038627148f, -2.071105957f, 2.453381062f, 4.181341171f, 1.748562813f, 0.708138049f, 2.034394741f, -4.372180462f,
    0.530447364f, 5.304075241f, -0.324032634f, 2.621093273f, 0.495161742f, -0.996799827f, -1.573622942f, 1.327330232f,
    1.759609461f, 4.728504181f, -6.743413925f, -0.069207847f, -0.784472466f, 0.332671881f, 2.241839409f, -0.030126348f,
    2.310925007f, -1.49981904f, 0.660584927f, -1.595312119f, 3.246035814f, -1.231983781f, -2.221083879f, 2.692591906f,
    -9.371782303f, 1.023940563f, -0.376934856f, 2.37006712f, 2.265807152f, 4.717998505f, -0.165601283f, -0.808206558f,
    -0.225964427f, 1.246556044f, -0.24988997f, -0.883403182f, -3.278885603f, -0.750352144f, -0.764114141f, 0.812959611f,
    -0.125019491f, -0.067988932f, -0.297365963f, 1.852548718f, -0.043699116f, -0.510064483f
};

static const float test_1_out_scale = 0.077402197f;
static const float test_1_out_zero_point = -1.262352943f;
static const int8_t test_1_out_scales_frac[] = {18};
static const int test_1_out_sa_dim = -1;
static const int test_1_out_fx8_frac = 3;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ { 7, 4, 7 }, \
    /* .rank = */  3


static mli_tensor test_1_out_tsr_sa8 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_SA_8,
    /* .el_type = */ { 0 }
};

static mli_tensor test_1_out_tsr_fx8 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static mli_tensor test_1_out_tsr_fx16 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data,
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim, test_1_out_data,
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]),
    &test_1_out_scale, 1, &test_1_out_zero_point, 1,
    test_1_out_scales_frac, sizeof(test_1_out_scales_frac) / sizeof(test_1_out_scales_frac[0]));


static const float test_2_out_data[] = {
    0.f, 0.f, 1.298560977f, 0.039126217f, 2.642681599f, 0.97795701f, 0.f, 0.744094372f,
    0.338498652f, 0.f, 0.f, 1.495568275f, 1.8384794f, 3.753365278f, 1.115882158f, 0.f,
    0.f, 0.187201381f, 0.f, 0.f, 0.f, 0.365388334f, 0.f, 0.053870738f,
    0.312182069f, 0.f, 0.f, 3.12265873f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.35820505f, 0.680491805f, 0.f, 0.f, 0.f, 0.f,
    0.f, 1.066405892f, 0.063340724f, 0.f, 0.568200171f, 0.f, 0.f, 0.f,
    0.f, 0.270341933f, 0.f, 0.f, 2.322947025f, 0.f, 1.923987508f, 4.97557354f,
    0.f, 0.f, 1.194999933f, 0.f, 2.326817513f, 0.f, 0.f, 0.828509331f,
    0.f, 0.f, 0.f, 0.f, 0.f, 1.745433688f, 0.872721314f, 0.f,
    0.161613882f, 1.157350183f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
    2.676425934f, 0.f, 0.f, 0.f, 0.626036942f, 0.f, 0.f, 0.f,
    0.f, 2.294606924f, 0.f, 0.018213838f, 0.f, 0.f, 0.630528629f, 0.f,
    0.f, 0.896081448f, 0.758919716f, 0.f, 0.82399267f, 0.f, 0.f, 0.f,
    0.f, 1.221440792f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
    0.546371043f, 0.f, 0.f, 4.566282749f, 2.228814602f, 0.f, 0.249573261f, 0.053962529f,
    0.f, 0.505144f, 0.3803505f, 0.f, 0.f, 0.f, 0.f, 0.021784961f,
    0.065716028f, 0.444164813f, 0.f, 0.f, 4.844591141f, 0.712916374f, 0.f, 0.496141791f,
    2.768072605f, 1.603745699f, 0.f, 0.f, 0.542137027f, 0.f, 0.f, 1.92160058f,
    0.f, 0.f, 2.792612314f, 0.587710798f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.044365346f, 0.817780077f, 0.020003438f, 0.104035318f, 3.069861412f, 2.370106697f, 0.f,
    0.f, 0.f, 0.f, 0.f, 1.503202081f, 0.f, 0.f, 0.390048593f,
    0.405348003f, 0.f, 0.f, 0.617841363f, 0.f, 1.465770006f, 0.f, 0.844059944f,
    0.f, 0.0039047f, 0.f, 4.768614769f, 0.f, 0.f, 0.431787938f, 0.f,
    0.f, 0.f, 0.f, 1.532243371f, 0.f, 0.471905977f, 0.f, 0.730849862f,
    2.061485291f, 0.f, 0.f, 2.384791613f
};

static const float test_2_out_scale = 0.019512054f;
static const float test_2_out_zero_point = 2.48778677f;
static const int8_t test_2_out_scales_frac[] = {20};
static const int test_2_out_sa_dim = -1;
static const int test_2_out_fx8_frac = 4;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ { 4, 7, 7 }, \
    /* .rank = */  3


static mli_tensor test_2_out_tsr_sa8 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static mli_tensor test_2_out_tsr_fx8 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static mli_tensor test_2_out_tsr_fx16 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_fx8_frac + 8, test_2_out_data,
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_sa_dim, test_2_out_data,
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]),
    &test_2_out_scale, 1, &test_2_out_zero_point, 1,
    test_2_out_scales_frac, sizeof(test_2_out_scales_frac) / sizeof(test_2_out_scales_frac[0]));


static const float test_3_out_data[] = {
    0.044364989f, -1.961867809f, -0.258086085f, -1.600380182f, 0.47328198f, 3.765779257f, 1.42001152f,
    0.295486212f, -0.705313623f, 0.407233655f, 4.214907169f, 5.011917591f, -3.013315678f, -6.079471111f,
    0.426969975f, -0.177058637f, 0.140124142f, 0.628331244f, 1.224997997f, 4.644593239f, -3.699924946f
};

static const float test_3_out_scale = 0.04349564f;
static const float test_3_out_zero_point = -0.53377676f;
static const int8_t test_3_out_scales_frac[] = {19};
static const int test_3_out_sa_dim = -1;
static const int test_3_out_fx8_frac = 4;

#define TEST_3_OUT_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ { 3, 1, 7 }, \
    /* .rank = */  3


static mli_tensor test_3_out_tsr_sa8 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static mli_tensor test_3_out_tsr_fx8 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static mli_tensor test_3_out_tsr_fx16 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer test_3_out_fx16(test_3_out_tsr_fx16, test_3_out_fx8_frac + 8, test_3_out_data,
    sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));
tensor_quantizer test_3_out_sa8(test_3_out_tsr_sa8, test_3_out_sa_dim, test_3_out_data,
    sizeof(test_3_out_data) / sizeof(test_3_out_data[0]),
    &test_3_out_scale, 1, &test_3_out_zero_point, 1,
    test_3_out_scales_frac, sizeof(test_3_out_scales_frac) / sizeof(test_3_out_scales_frac[0]));

static const float test_4_out_data[] = {
    0.631586492f, -1.f, 1.f, -0.829536438f, 1.f, -1.f, 1.f, 0.136842549f,
    -1.f, -0.545728743f, 0.306806445f, 1.f, 0.26382795f, -1.f, 0.283196002f, -1.f,
    1.f, 0.420753598f, -0.661842644f, -0.068618447f, -1.f, 1.f, 1.f, 1.f,
    0.075811028f, -1.f, 1.f, -0.641383529f
};

static const float test_4_out_scale = 0.007843138f;
static const float test_4_out_zero_point = 0.f;
static const int8_t test_4_out_scales_frac[] = {21};
static const int test_4_out_sa_dim = -1;
static const int test_4_out_fx8_frac = 7;

#define TEST_4_OUT_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /* .mem_stride = */ { 30, 10, 1 }, \
    /* .shape = */ { 2, 2, 7 }, \
    /* .rank = */  3


static mli_tensor test_4_out_tsr_sa8 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static mli_tensor test_4_out_tsr_fx8 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static mli_tensor test_4_out_tsr_fx16 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

tensor_quantizer test_4_out_fx16(test_4_out_tsr_fx16, test_4_out_fx8_frac + 8, test_4_out_data,
    sizeof(test_4_out_data) / sizeof(test_4_out_data[0]));
tensor_quantizer test_4_out_sa8(test_4_out_tsr_sa8, test_4_out_sa_dim, test_4_out_data,
    sizeof(test_4_out_data) / sizeof(test_4_out_data[0]),
    &test_4_out_scale, 1, &test_4_out_zero_point, 1,
    test_4_out_scales_frac, sizeof(test_4_out_scales_frac) / sizeof(test_4_out_scales_frac[0]));


static const float test_5_out_data[] = {
    0.f, 0.f, 0.f, 1.747632504f, 0.f, 0.788578153f, 0.f, 0.f,
    0.f, 0.417308658f, 3.193252563f, 0.f, 0.f, 0.f, 0.706569195f, 0.f,
    0.f, 0.f, 0.f, 2.456426382f, 0.154037565f, 0.846633494f, 0.f, 0.f,
    0.f, 0.137218222f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.335994244f,
    0.f, 0.f, 1.388865113f, 0.f, 0.f, 1.194999933f, 0.f, 2.326817513f,
    0.f, 0.f, 0.828509331f, 0.f, 0.f, 0.f, 0.f, 0.f,
    1.745433688f, 0.872721314f, 0.f, 0.161613882f, 1.157350183f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 2.676425934f, 0.f, 0.f, 0.f, 0.626036942f,
    0.f, 0.f, 0.f, 0.f, 2.294606924f, 0.f, 0.638227344f, 0.f,
    0.f, 0.289771914f, 0.712056279f, 0.f, 0.f, 0.453725517f, 0.f, 0.589880943f,
    0.909006f, 0.f, 0.f, 0.444470555f, 0.f, 0.f, 0.f, 6.f,
    0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.45031774f, 0.f,
    0.f, 0.f, 0.174193859f, 0.f, 0.f, 0.f, 0.f, 0.142332017f,
    0.f, 1.221440792f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
    0.546371043f, 0.f, 0.f, 4.566282749f, 2.228814602f, 0.f, 0.249573261f, 0.053962529f,
    0.f, 0.505144f, 0.3803505f, 0.f, 0.f, 0.f, 0.f, 0.021784961f,
    0.06571573f, 0.444164813f, 0.f, 0.f, 4.844591141f, 0.712916434f, 0.f, 0.49614197f,
    2.768073082f, 1.603746176f, 0.f, 0.f
};

static const float test_5_out_scale = 0.023529412f;
static const float test_5_out_zero_point = 3.f;
static const int8_t test_5_out_scales_frac[] = {20};
static const int test_5_out_sa_dim = -1;
static const int test_5_out_fx8_frac = 4;

#define TEST_5_OUT_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /* .mem_stride = */ { 0 }, \
    /* .shape = */ { 4, 5, 7 }, \
    /* .rank = */  3


static mli_tensor test_5_out_tsr_sa8 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static mli_tensor test_5_out_tsr_fx8 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_8,
    /* .el_params = */ { 0 } 
};

static mli_tensor test_5_out_tsr_fx16 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    /* .el_type =  */ MLI_EL_FX_16,
    /* .el_params = */ { 0 } 
};

tensor_quantizer test_5_out_fx16(test_5_out_tsr_fx16, test_5_out_fx8_frac + 8, test_5_out_data,
    sizeof(test_5_out_data) / sizeof(test_5_out_data[0]));
tensor_quantizer test_5_out_sa8(test_5_out_tsr_sa8, test_5_out_sa_dim, test_5_out_data,
    sizeof(test_5_out_data) / sizeof(test_5_out_data[0]),
    &test_5_out_scale, 1, &test_5_out_zero_point, 1,
    test_5_out_scales_frac, sizeof(test_5_out_scales_frac) / sizeof(test_5_out_scales_frac[0]));


    
static const float test_6_out_data[] = {
    0.656343f, 0.f, 0.203095f, 0.f, 0.145804f, 0.37239f, 0.837982f, 0.37044f, 0.f, 0.f, 0.01504f, 0.f, 0.f, 0.f, 
    0.356308f, 0.f, 1.80661f, 0.f, 1.399823f, 1.021067f, 0.f, 0.f, 0.f, 0.414662f, 0.f, 0.f, 0.f, 0.f, 0.571944f, 
    0.f, 1.461622f, 0.702629f, 0.42156f, 0.f, 1.367366f, 0.188182f, 0.f, 0.f, 0.f, 0.f, 0.639393f, 0.f, 0.368379f, 
    0.f, 1.630286f, 1.314375f, 0.f, 0.f, 1.327458f, 0.079884f, 0.f, 0.f, 0.453122f, 0.f, 0.f, 0.f, 0.419171f, 0.f, 
    1.704029f, 0.207785f, 1.07798f, 0.447526f, 0.f, 0.564056f, 0.f, 0.854414f, 1.317693f, 0.f, 0.f, 2.345759f, 
    0.265045f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.116356f, 0.f, 1.483807f, 0.011823f, 0.785274f, 0.262396f, 0.f, 
    0.013878f, 0.f, 1.243325f, 0.f, 0.523273f, 0.076734f, 0.f, 0.587926f, 0.f, 1.103194f, 0.f, 1.171069f, 
    1.370968f, 0.108878f, 0.626329f, 0.f, 0.4695f, 0.197701f, 0.079254f, 0.f, 1.04019f, 0.596259f, 0.f, 1.611836f, 
    0.f, 1.655519f, 1.712434f, 0.089193f, 0.337355f, 0.f, 0.785779f, 0.f, 0.904329f, 1.346778f, 0.f, 0.401469f, 
    0.f, 0.959846f, 0.121784f, 0.426083f, 0.070999f, 0.f, 0.f, 0.f, 0.956195f, 0.122382f, 0.f, 0.f, 0.f, 0.526949f, 
    0.f, 1.650666f, 0.598227f, 0.683233f, 0.f, 0.978458f, 0.240302f, 0.f, 0.883246f, 0.275925f, 0.012853f, 0.f, 
    0.f, 0.410086f, 0.f, 0.540754f, 1.127968f, 0.f, 0.f, 1.430392f, 0.128979f, 0.f, 0.123539f, 0.f, 0.f, 0.f, 0.f, 
    0.321525f, 0.f, 1.949609f, 0.573828f, 0.801506f, 0.f, 0.f, 0.516271f, 0.f, 1.664833f, 0.611998f, 0.670719f, 
    0.f, 0.948409f, 0.040032f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.127694f, 0.f, 0.441009f, 0.f, 0.124483f, 0.253343f, 
    0.f, 0.337003f, 0.f, 0.495524f, 0.140834f, 0.f, 0.f, 0.f, 0.516838f, 0.f, 1.650859f, 0.f, 1.519252f, 1.408281f, 
    0.f, 0.492485f, 0.f, 1.34535f, 0.f, 1.391847f, 1.558643f, 0.f, 0.575833f, 0.f, 1.235487f, 0.583408f, 0.339832f, 
    0.f, 1.24855f, 0.333127f, 0.f, 2.057292f, 0.673472f, 0.804464f, 0.f, 0.149072f, 0.112464f, 0.f, 0.990537f, 
    0.104525f, 0.188667f, 0.f, 0.f, 0.672309f, 0.f, 1.272268f, 0.372018f, 0.715359f, 0.140233f, 1.362874f, 0.f, 
    0.f, 0.564213f, 0.108331f, 0.f, 0.f, 0.f, 0.288345f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.592454f, 0.f, 0.185161f, 
    0.f, 0.053849f, 0.231168f, 0.568373f, 0.300078f, 0.f, 0.631617f, 1.143563f, 0.f, 0.f, 0.94418f, 0.111598f, 0.f, 
    0.97239f, 0.91982f, 0.f, 0.f, 0.f, 0.543631f, 0.f, 1.824523f, 0.f, 1.955303f, 2.08711f, 0.f, 0.69176f, 0.f, 
    0.307628f, 0.f, 0.366597f, 0.683784f, 0.872662f, 0.63573f, 0.f, 0.69281f, 0.406295f, 0.068555f, 0.f, 1.350944f, 
    0.261807f, 0.f, 1.626403f, 0.f, 1.456542f, 1.426183f, 0.f, 0.3801f, 0.f, 0.664491f, 0.035955f, 0.212586f, 
    0.024014f, 0.f, 0.f, 0.f, 0.694637f, 0.160617f, 0.f, 0.f, 0.f, 0.544258f, 0.f, 0.023354f, 0.f, 0.158041f, 
    0.690793f, 0.f, 0.485981f, 0.f, 0.449923f, 1.03423f, 0.f, 0.f, 1.648353f, 0.f, 0.f, 0.672996f, 0.f, 0.453036f, 
    0.733841f, 0.f, 0.302091f, 0.f, 0.f, 0.910775f, 0.f, 0.f, 0.720892f
};

static const float test_6_out_scale = 0.0091990568f;
static const float test_6_out_zero_point = 1.1728796959f;
static const int8_t test_6_out_scales_frac[] = {21};
static const int test_6_out_sa_dim = -1;

static const int test_6_out_fx8_frac = 5;

#define TEST_6_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(7*1+3)*7+6, 7*1+3, 1}, \
    /* .shape = */ {7, 7, 7}, \
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
    0.498553f, 0.f, 0.69108f, 1.166635f, 0.f, 1.394715f, 0.f, 0.428758f, 0.f, 2.143944f, 0.f, 0.f, 0.f, 0.f, 0.f, 
    0.713277f, 0.f, 0.f, 0.f, 0.469849f, 0.945567f, 0.219694f, 0.f, 0.f, 0.332456f, 0.f, 0.f, 0.f, 0.068731f, 0.f, 
    2.136943f, 0.876932f, 0.f, 0.222984f, 1.864027f, 0.653401f, 0.054455f, 0.641873f, 0.f, 1.508578f, 3.491822f, 
    0.f, 0.503047f, 0.f, 2.464174f, 3.428593f, 0.160063f, 1.03251f, 2.932822f, 0.7687f, 0.f, 0.f, 0.152532f, 
    3.097333f, 0.f, 0.f, 0.473338f, 0.f, 2.445764f, 0.f, 0.f, 3.631407f, 0.f, 0.392623f, 0.391439f, 1.564334f, 
    2.480505f, 0.898404f, 0.f, 2.49491f, 0.22597f, 0.f, 0.f, 0.f, 1.316278f, 3.099524f, 0.741207f, 0.53755f, 0.f, 
    1.450086f, 0.861116f, 0.99304f, 2.13177f, 0.f, 0.567547f, 0.f, 1.156383f, 0.f, 0.f, 0.616374f, 0.f, 0.15475f, 
    0.f, 2.743253f, 3.945569f, 0.f, 4.844812f, 2.18671f, 0.870535f, 0.687069f, 0.f, 0.634597f, 2.415865f, 
    0.433902f, 0.077811f, 0.071847f, 0.f, 0.41472f, 0.455603f, 0.f, 0.f, 0.453105f
};

static const float test_7_out_scale = 0.0189992636f;
static const float test_7_out_zero_point = 2.4224061966f;
static const int8_t test_7_out_scales_frac[] = {20};
static const int test_7_out_sa_dim = -1;

static const int test_7_out_fx8_frac = 4;

#define TEST_7_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(7*1+3)*4+6, 7*1+3, 1}, \
    /* .shape = */ {4, 4, 7}, \
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
    0.382504f, 0.f, 1.560793f, 0.f, 0.f, 3.987979f, 0.f, 0.322384f, 0.f, 1.672614f, 0.f, 0.f, 0.873936f, 6.f, 
    1.181278f, 0.265367f, 1.022901f, 1.715667f, 1.423097f, 2.541872f, 0.f, 1.039007f, 1.462328f, 0.f, 0.f, 6.f, 
    0.f, 0.f, 1.210744f, 0.f, 3.183197f, 0.f, 0.321438f, 4.042826f, 4.07325f, 0.f, 2.192997f, 0.337671f, 0.57454f, 
    0.f, 0.f, 0.f, 0.357112f, 0.111237f, 3.829905f, 0.f, 6.f, 0.f, 2.260155f, 0.340313f, 3.042266f, 0.13709f, 
    3.760564f, 6.f, 2.648774f, 0.f, 0.f, 0.f, 4.09014f, 0.f, 0.f, 5.250613f, 3.164089f, 0.277474f, 0.682404f, 
    1.387101f, 0.f, 3.581748f, 2.642064f, 5.776623f, 0.798394f, 0.f, 3.544884f, 3.016122f, 6.f, 6.f, 0.f, 0.58618f, 
    0.429943f, 0.f, 2.681408f, 0.435036f, 0.f, 2.346607f, 0.f, 0.f, 3.155254f, 4.307629f, 3.497239f, 3.685683f, 
    0.25366f, 0.054509f, 1.189736f, 1.822809f, 0.489343f, 0.f, 0.f, 0.f, 0.847542f, 0.f, 1.304549f, 0.f, 1.74483f, 
    0.f, 0.f, 1.499071f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.725827f
};

static const float test_8_out_scale = 0.0235294122f;
static const float test_8_out_zero_point = 3.f;
static const int8_t test_8_out_scales_frac[] = {20};
static const int test_8_out_sa_dim = -1;

static const int test_8_out_fx8_frac = 4;

#define TEST_8_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(7*1+3)*7+6, 7*1+3, 1}, \
    /* .shape = */ {4, 4, 7}, \
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
    0.461678f, 0.685434f, 0.59567f, 0.f, 0.366094f, 0.971305f, 2.559584f, 0.223252f, 0.972306f, 0.11715f, 0.f, 
    1.467746f, 1.923826f, 0.f, 0.588679f, 0.f, 1.157487f, 0.f, 1.631999f, 0.f, 0.301974f, 0.512703f, 0.f, 
    0.193098f, 2.374771f, 0.f, 4.151912f, 0.f, 0.331396f, 0.f, 0.838276f, 0.f, 0.f, 0.f, 0.f, 0.212303f, 0.893865f, 
    0.2883f, 0.871271f, 0.f, 1.154489f, 1.93349f, 0.740616f, 0.f, 0.060074f, 0.470487f, 1.122215f, 0.f, 0.f, 
    0.704913f, 0.f, 1.492651f, 2.350322f, 0.f, 0.f, 0.f, 0.42473f, 0.f, 0.198252f, 0.245917f, 0.f, 1.515597f, 0.f, 
    0.251648f, 0.723338f, 0.027884f, 0.721309f, 3.406739f, 0.f, 1.093632f, 0.573814f, 0.80446f, 0.435915f, 0.f, 
    0.845756f, 1.108398f, 0.700992f, 1.165142f, 0.f, 1.162684f, 1.80387f, 1.938243f, 0.338713f, 0.f, 0.646221f, 
    0.f, 0.98068f, 1.188727f, 0.493009f, 1.417801f, 0.f, 0.434669f, 0.357629f, 0.328591f, 0.f, 0.f, 0.f, 0.f, 
    0.149283f, 0.130071f, 0.058296f, 0.f, 2.802057f, 0.041349f, 1.885831f, 0.f, 0.359761f, 1.757257f, 0.f, 
    0.181688f, 0.f, 1.412108f, 0.648425f, 0.923331f, 1.49873f, 2.800381f, 1.977833f, 0.f, 2.727125f, 0.351715f, 
    0.f, 1.00102f, 0.f, 0.f, 0.077601f, 0.f, 0.042084f, 0.f, 0.355407f, 4.249274f, 4.684985f, 3.79153f, 2.841051f, 
    0.f, 0.f, 0.f, 0.515304f, 0.f, 0.f, 0.f, 0.619649f, 0.f, 0.f, 1.870229f, 0.f, 2.977245f, 1.499815f, 0.027843f, 
    0.f, 2.289968f, 0.580741f, 0.f, 0.f, 0.f, 0.988704f, 0.186368f, 1.024223f, 1.010739f, 0.49948f, 0.f, 1.671561f, 
    0.31907f, 0.f, 2.850888f, 0.492554f, 0.f, 2.022866f, 0.246435f, 0.f, 0.f, 0.990746f, 0.f, 0.845832f, 2.876812f, 
    0.f, 0.889922f, 1.002685f, 0.f, 3.009646f, 0.851012f, 3.365976f, 4.048658f, 0.77157f, 0.f, 0.794887f, 
    4.320246f, 0.117149f, 0.526992f, 0.685967f, 0.165421f, 0.f, 0.f, 0.765257f, 0.f, 0.f, 2.049095f, 0.446953f, 
    0.f, 0.769721f, 0.f, 1.38291f, 0.f, 0.608502f, 0.90498f, 0.533457f, 1.157039f, 0.f, 2.697205f, 0.f, 2.188462f, 
    0.226364f, 0.f, 1.542841f, 0.f, 1.758619f, 0.f, 0.f, 0.171964f, 0.f, 0.f, 1.244521f, 2.311857f, 0.85399f, 
    4.110386f, 0.646141f, 0.f, 2.789012f, 3.021591f, 2.745941f, 0.f, 0.f, 0.419217f, 0.f, 0.f, 0.f, 1.954778f, 
    1.380297f, 2.026916f, 0.490505f, 0.f, 0.f, 3.317868f, 0.f, 3.785811f, 0.f, 0.110096f, 0.f, 2.988263f, 
    2.385898f, 0.f, 1.612129f, 3.784957f, 0.051919f, 0.808896f, 1.60202f, 0.268048f, 0.f, 0.f, 1.504182f, 
    0.669256f, 0.f, 0.f, 1.484462f, 0.f, 0.f, 6.094067f, 0.592991f, 0.f, 0.907984f, 2.911446f, 1.641763f, 0.64755f, 
    1.98973f, 0.101974f, 0.241425f, 0.05719f, 0.f, 0.f, 0.227798f, 0.770615f, 0.145535f, 1.159709f, 0.f, 0.f, 0.f, 
    0.f, 3.764453f, 0.070166f, 0.f, 0.519944f, 0.496741f, 0.f, 0.586974f, 0.f, 0.721774f, 1.063324f, 0.864775f, 
    0.031425f, 1.974998f, 0.902096f, 0.f, 0.355025f, 0.f, 0.053238f, 3.192802f, 2.898672f, 0.670719f, 6.088238f, 
    0.241613f, 0.f, 1.251353f, 1.131299f, 0.239175f, 3.589928f, 1.160302f, 0.475477f, 1.47899f, 0.f, 3.219539f, 
    4.42058f, 3.926777f, 2.52368f, 0.f, 0.289335f, 0.f, 0.248436f, 0.f, 0.f, 6.421874f, 0.983724f, 0.f, 0.f, 
    0.975994f, 2.64414f, 3.593419f, 0.449156f, 0.29427f, 0.f, 0.f, 2.931132f, 2.057409f, 0.f, 2.618293f
};

static const float test_9_out_scale = 0.0251838192f;
static const float test_9_out_zero_point = 3.2109370232f;
static const int8_t test_9_out_scales_frac[] = {20};
static const int test_9_out_sa_dim = -1;

static const int test_9_out_fx8_frac = 4;

#define TEST_9_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(7*1+3)*7+6, 7*1+3, 1}, \
    /* .shape = */ {7, 7, 7}, \
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
    0.635133f, 0.f, 1.192481f, 0.f, 0.f, 3.698598f, 4.647196f, 0.254133f, 0.428313f, 1.53546f, 0.f, 0.501393f, 0.f, 
    4.750391f, 0.404154f, 0.f, 0.654892f, 0.f, 2.651023f, 1.332102f, 3.036563f, 0.669307f, 0.83995f, 0.f, 0.f, 0.f, 
    0.f, 0.f, 0.77478f, 0.881377f, 1.025105f, 0.f, 3.509822f, 4.796329f, 2.490216f, 1.513501f, 0.f, 1.937922f, 0.f, 
    2.040922f, 0.f, 0.f, 0.443511f, 1.003047f, 0.f, 1.659597f, 4.254647f, 0.f, 0.f, 0.504223f, 1.096079f, 
    0.656901f, 1.089815f, 0.f, 0.43155f, 0.f, 0.185194f, 0.f, 2.458277f, 4.350532f, 0.f, 3.154658f, 0.f, 1.609827f, 
    0.f, 0.f, 0.f, 4.742415f, 0.f, 1.513012f, 0.643888f, 0.653661f, 0.305154f, 0.f, 3.374336f, 0.f, 1.127988f, 
    0.88888f, 0.f, 4.721356f, 0.408374f, 2.238085f, 6.745949f, 0.f, 0.631239f, 0.f, 0.317524f, 0.114174f, 
    4.112437f, 0.f, 0.f, 0.769246f, 0.f, 0.694724f, 0.f, 1.489552f, 1.340646f, 0.f, 0.501362f, 0.480516f, 
    2.855238f, 1.703427f, 2.911497f, 0.f, 0.f, 0.43054f, 0.f, 3.152053f, 0.f, 0.f, 4.562704f, 1.371198f, 0.349209f, 
    0.000925f, 0.f, 0.680604f, 4.023891f, 0.f, 10.167797f, 1.082417f, 1.410061f, 1.324281f, 0.f, 0.962943f, 0.f, 
    0.159776f, 0.794715f, 0.f, 0.043767f, 0.480981f, 4.652651f, 0.f, 2.926259f, 0.648087f, 0.679294f, 2.033376f, 
    2.726506f, 0.608113f, 0.f, 0.f, 0.666106f, 0.089992f, 1.685356f, 2.256398f, 4.139259f, 6.348581f, 0.f, 
    1.016131f, 0.f, 5.116295f, 0.247569f, 3.105422f, 4.917112f, 0.f, 0.353523f, 0.f, 2.787939f, 0.f, 2.568628f, 
    0.f, 0.f, 0.537192f, 0.f, 0.948445f, 0.f, 0.13819f, 0.979903f, 3.083597f, 1.290102f, 0.f, 1.392604f, 0.779007f, 
    0.f, 2.663505f, 0.805992f, 1.020511f, 0.f, 1.513466f, 0.078003f, 0.f, 0.f, 0.f, 0.257509f, 0.f, 2.950367f, 
    1.723192f, 1.845632f, 3.883855f, 3.287359f, 1.043573f, 0.f, 0.988005f, 0.914659f, 2.12749f, 0.f, 0.f, 0.38219f, 
    0.f, 1.980602f, 1.792575f, 2.242915f, 5.833778f, 1.04755f, 0.061713f, 0.f, 0.f, 0.101442f, 0.f, 0.843206f, 
    0.462077f, 0.553659f, 0.488379f, 2.921236f, 1.350652f, 1.010145f, 4.207087f, 0.f, 0.819846f, 0.f, 0.514559f, 
    1.95906f, 1.907756f, 2.268581f, 0.506986f, 0.f, 1.903872f, 0.f, 0.f, 6.581852f, 0.f, 3.838022f, 0.f, 0.f, 
    1.17135f, 0.f, 0.888273f, 6.284151f, 0.f, 0.310482f, 0.687716f, 0.f, 0.f, 6.000235f, 0.f, 0.f, 2.233798f, 0.f, 
    2.944568f, 0.f, 0.f, 0.f, 0.399425f, 0.706643f, 0.f, 0.745612f, 0.f, 0.76595f, 0.f, 0.f, 0.731454f, 0.f, 
    1.322511f, 0.f, 2.870625f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.720082f, 0.363607f, 0.f, 4.531674f, 
    0.261315f, 0.f, 5.924967f, 0.362845f, 0.142553f, 0.f, 2.857198f, 0.992112f, 1.645181f, 1.914987f, 0.f, 
    0.850257f, 0.f, 0.302332f, 0.610107f, 1.459751f, 4.468338f, 0.f, 0.f, 0.f, 2.148523f, 3.273672f, 2.08194f, 
    0.085512f, 2.943926f, 0.f, 0.369194f, 1.887733f, 2.411426f, 0.f, 0.060005f, 0.f, 0.047355f, 1.341573f, 0.f, 
    2.921771f, 0.f, 0.096868f, 0.f, 0.f, 0.340791f, 0.617452f, 0.f, 1.426828f, 0.f, 5.056797f, 1.064062f, 0.f, 
    0.842631f, 0.f, 2.24167f, 4.367232f, 0.f, 0.f, 0.721143f, 0.368186f, 1.957134f, 0.918331f, 0.f, 0.f, 1.146812f, 
    0.f, 3.584839f, 0.f, 3.181296f, 0.f, 0.f
};

static const float test_10_out_scale = 0.0398737155f;
static const float test_10_out_zero_point = 5.0838985443f;
static const int8_t test_10_out_scales_frac[] = {19};
static const int test_10_out_sa_dim = -1;

static const int test_10_out_fx8_frac = 3;

#define TEST_10_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(7*1+3)*7+6, 7*1+3, 1}, \
    /* .shape = */ {7, 7, 7}, \
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

static const float test_11_out_data[] = {
    -7333843968.f, -6154032128.f, 3043232256.f, 11726278656.f, 3220792832.f, 6539892224.f, -29828861952.f,
    -4668388352.f, 3718503168.f, -13789184000.f, 16171816960.f, 22837057536.f, 13803910144.f, -5043655680.f,
    8374614016.f, -686402816.f, -38821883904.f, 20203468800.f, 29980436480.f, 13674545152.f, -2236483584.f,
    4800590336.f, -1999792640.f, 13921811456.f, 12712160256.f, 4674600448.f, 14096794624.f, -22270711808.f,
    2117026048.f, -1699832576.f, -8062188032.f, 14887353344.f, 18557353984.f, 16541390848.f, -2263584256.f,
    6187534336.f, 1785957376.f, -20238200832.f, 8610623488.f, 16639879168.f, 14661462016.f, 3595582464.f,
    -24798640128.f, 3541465856.f, -9104544768.f, 2977540096.f, 3550413824.f, 2342983680.f, -7996243456.f,
    -3898777856.f, -1747437824.f, -14680812544.f, 12643473408.f, 28782987264.f, 10040888320.f, -6682178048.f,
    15206917120.f, -14617193472.f, 2641565184.f, 3774700032.f, -480409408.f, 13166043136.f, -6499435520.f,
    12754702336.f, -5144712192.f, -20408668160.f, 15423592448.f, 31418503168.f, 2867204608.f, -8045949440.f,
    -19055181824.f, -5561644032.f, 11221115904.f, 6097220096.f, 11398113280.f, 5006050304.f, -9395949568.f,
    16510922752.f, 5825272832.f, -4934623744.f, 8208686592.f, -9126788096.f, 19024678912.f, -13331911680.f
};

static const float test_11_out_scale = 275452512.f;
static const float test_11_out_zero_point = -3701690368.f;
static const int8_t test_11_out_scales_frac[] = {-14};
static const int test_11_out_sa_dim = -1;

static const int test_11_out_fx8_frac = -29;

#define TEST_11_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 3, 7}, \
    /* .rank =  */ 3


static const mli_tensor test_11_out_tsr_fx16 = {
    TEST_11_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_11_out_tsr_sa8 = {
    TEST_11_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};
tensor_quantizer test_6_out_fx16(test_6_out_tsr_fx16, test_6_out_fx8_frac + 8, test_6_out_data, sizeof(test_6_out_data) / sizeof(test_6_out_data[0]));
tensor_quantizer test_6_out_sa8(test_6_out_tsr_sa8, test_6_out_sa_dim, test_6_out_data, 
sizeof(test_6_out_data) / sizeof(test_6_out_data[0]), &test_6_out_scale, 1, &test_6_out_zero_point, 1, test_6_out_scales_frac, 1);

tensor_quantizer test_7_out_fx16(test_7_out_tsr_fx16, test_7_out_fx8_frac + 8, test_7_out_data, sizeof(test_7_out_data) / sizeof(test_7_out_data[0]));
tensor_quantizer test_7_out_sa8(test_7_out_tsr_sa8, test_7_out_sa_dim, test_7_out_data, 
sizeof(test_7_out_data) / sizeof(test_7_out_data[0]), &test_7_out_scale, 1, &test_7_out_zero_point, 1, test_7_out_scales_frac, 1);

tensor_quantizer test_8_out_fx16(test_8_out_tsr_fx16, test_8_out_fx8_frac + 8, test_8_out_data, sizeof(test_8_out_data) / sizeof(test_8_out_data[0]));
tensor_quantizer test_8_out_sa8(test_8_out_tsr_sa8, test_8_out_sa_dim, test_8_out_data, 
sizeof(test_8_out_data) / sizeof(test_8_out_data[0]), &test_8_out_scale, 1, &test_8_out_zero_point, 1, test_8_out_scales_frac, 1);

tensor_quantizer test_9_out_fx16(test_9_out_tsr_fx16, test_9_out_fx8_frac + 8, test_9_out_data, sizeof(test_9_out_data) / sizeof(test_9_out_data[0]));
tensor_quantizer test_9_out_sa8(test_9_out_tsr_sa8, test_9_out_sa_dim, test_9_out_data, 
sizeof(test_9_out_data) / sizeof(test_9_out_data[0]), &test_9_out_scale, 1, &test_9_out_zero_point, 1, test_9_out_scales_frac, 1);

tensor_quantizer test_10_out_fx16(test_10_out_tsr_fx16, test_10_out_fx8_frac + 8, test_10_out_data, sizeof(test_10_out_data) / sizeof(test_10_out_data[0]));
tensor_quantizer test_10_out_sa8(test_10_out_tsr_sa8, test_10_out_sa_dim, test_10_out_data, 
sizeof(test_10_out_data) / sizeof(test_10_out_data[0]), &test_10_out_scale, 1, &test_10_out_zero_point, 1, test_10_out_scales_frac, 1);

tensor_quantizer test_11_out_fx16(test_11_out_tsr_fx16, test_11_out_fx8_frac + 8, test_11_out_data, sizeof(test_11_out_data) / sizeof(test_11_out_data[0]));
tensor_quantizer test_11_out_sa8(test_11_out_tsr_sa8, test_11_out_sa_dim, test_11_out_data, 
sizeof(test_11_out_data) / sizeof(test_11_out_data[0]), &test_11_out_scale, 1, &test_11_out_zero_point, 1, test_11_out_scales_frac, 1);


// Tests configuration structures
//========================================
const mli_conv2d_cfg test_1_cfg = {
    /* .relu.type = */  MLI_RELU_NONE,
    /* .stride_width = */  2,
    /* .stride_height = */  1,
    /* .padding_left = */  1,
    /* .padding_right = */  2,
    /* .padding_top = */  1,
    /* .padding_bottom = */  1,
    /* .dilation_width = */  1,
    /* .dilation_height = */  1
};

const mli_conv2d_cfg test_2_cfg = {
    /* .relu.type = */  MLI_RELU_GEN,
    /* .stride_width = */  1,
    /* .stride_height = */  2,
    /* .padding_left = */  1,
    /* .padding_right = */  1,
    /* .padding_top = */  1,
    /* .padding_bottom = */  2,
    /* .dilation_width = */  1,
    /* .dilation_height = */  1
};

const mli_conv2d_cfg test_3_cfg = {
    /* .relu.type = */  MLI_RELU_NONE,
    /* .stride_width = */  1,
    /* .stride_height = */  1,
    /* .padding_left = */  0,
    /* .padding_right = */  0,
    /* .padding_top = */  0,
    /* .padding_bottom = */  0,
    /* .dilation_width = */  2,
    /* .dilation_height = */  2
};

const mli_conv2d_cfg test_4_cfg = {
    /* .relu.type = */  MLI_RELU_1,
    /* .stride_width = */  3,
    /* .stride_height = */  3,
    /* .padding_left = */  0,
    /* .padding_right = */  0,
    /* .padding_top = */  0,
    /* .padding_bottom = */  0,
    /* .dilation_width = */  1,
    /* .dilation_height = */  1
};

const mli_conv2d_cfg test_5_cfg = {
    /* .relu.type = */  MLI_RELU_6,
    /* .stride_width = */  1,
    /* .stride_height = */  1,
    /* .padding_left = */  0,
    /* .padding_right = */  0,
    /* .padding_top = */  0,
    /* .padding_bottom = */  0,
    /* .dilation_width = */  1,
    /* .dilation_height = */  1
};

const mli_conv2d_cfg test_6_cfg = {
    /* .relu.type = */MLI_RELU_6,
    /* .stride_width = */1,
    /* .stride_height = */1,
    /* .padding_left = */0,
    /* .padding_right = */0,
    /* .padding_top = */0,
    /* .padding_bottom = */0,
    /* .dilation_width = */2,
    /* .dilation_height = */2
};

const mli_conv2d_cfg test_7_cfg = {
    /* .relu.type = */MLI_RELU_6,
    /* .stride_width = */2,
    /* .stride_height = */2,
    /* .padding_left = */1,
    /* .padding_right = */1,
    /* .padding_top = */1,
    /* .padding_bottom = */1,
    /* .dilation_width = */1,
    /* .dilation_height = */1
};

const mli_conv2d_cfg test_8_cfg = {
    /* .relu.type = */MLI_RELU_6,
    /* .stride_width = */2,
    /* .stride_height = */2,
    /* .padding_left = */2,
    /* .padding_right = */2,
    /* .padding_top = */2,
    /* .padding_bottom = */2,
    /* .dilation_width = */1,
    /* .dilation_height = */1
};

const mli_conv2d_cfg test_9_cfg = {
    /* .relu.type = */MLI_RELU_GEN,
    /* .stride_width = */1,
    /* .stride_height = */1,
    /* .padding_left = */2,
    /* .padding_right = */2,
    /* .padding_top = */2,
    /* .padding_bottom = */2,
    /* .dilation_width = */2,
    /* .dilation_height = */2
};

const mli_conv2d_cfg test_10_cfg = {
    /* .relu.type = */MLI_RELU_GEN,
    /* .stride_width = */1,
    /* .stride_height = */1,
    /* .padding_left = */4,
    /* .padding_right = */4,
    /* .padding_top = */4,
    /* .padding_bottom = */4,
    /* .dilation_width = */2,
    /* .dilation_height = */2
};

const mli_conv2d_cfg test_11_cfg = {
    /* .relu.type = */MLI_RELU_NONE,
    /* .stride_width = */1,
    /* .stride_height = */1,
    /* .padding_left = */0,
    /* .padding_right = */0,
    /* .padding_top = */0,
    /* .padding_bottom = */0,
    /* .dilation_width = */1,
    /* .dilation_height = */1
};

