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
#include "test_quantizer.h"

using mli::tst::tensor_quantizer;

// Generated input vectors Declaration
//========================================

extern mli::tst::tensor_quantizer input_1_fx16;
extern mli::tst::tensor_quantizer input_1_sa8;
extern mli::tst::tensor_quantizer input_1_memstr_fx16;
extern mli::tst::tensor_quantizer input_1_memstr_sa8;

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

extern mli::tst::tensor_quantizer bias_1_fx16;
extern mli::tst::tensor_quantizer bias_1_fx8;
extern mli::tst::tensor_quantizer bias_1_sa32;
extern mli::tst::tensor_quantizer bias_1_w2_sa32;


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
    -0.795256615f, -0.020611584f, -0.857786238f, -0.116273165f, 0.808907866f, -0.496227741f, -0.107275546f, -0.577136159f,
    1.860774279f, 0.930648685f, 1.950116396f, 1.566918135f, 0.589830399f, -0.950313091f, 0.618810654f, -0.962112069f,
    1.893128872f, -0.582279801f, 1.939082861f, -0.719187677f, 1.063522339f, 1.109600544f, 1.949721098f, -0.861565351f,
    0.142124653f, -0.689962804f, 1.484006405f, 1.528692007f, -0.571019948f, -0.376762152f, -0.228962064f, 1.455527067f,
    -0.751569092f, 0.435355544f, 0.228472352f, 1.998298407f, 0.528104305f, 0.669281363f, 1.963288307f, -0.903628111f,
    -0.531457663f, 0.510696411f, -0.327445090f, -0.207189679f, -0.556229711f, -0.869290471f, -0.953169167f, 1.123666763f,
    -0.810569465f, 0.351910353f, 0.803937912f, 0.019652009f, -0.049965858f, 0.622049570f, 0.900914192f, 0.991265893f,
    1.660414934f, 0.525478840f, -0.973419189f, 1.005276918f, 0.573811293f, 0.358962655f, 0.842565656f, 1.532562733f,
    0.070606351f, -0.333183408f, 0.164556742f, 1.287754774f, 0.305704474f, 0.906447887f, -0.721507311f, 1.875643730f,
    0.568320155f, -0.962049961f, 1.042712688f, 0.401276469f, 1.858359337f, -0.103364170f, -0.097108901f, 1.075629711f,
    0.781458378f, 0.160974979f, 0.460997581f, 0.180617571f, -0.582523704f, -0.813086033f, 1.326231480f, -0.808524489f,
    -0.48488760f, 1.067776680f, 0.391311884f, -0.760229707f, 0.372676373f, 1.040057898f, -0.851530135f, 1.943858385f,
    0.568393230f, 0.606444120f, 1.348023891f, -0.123570859f, -0.975745201f, 0.277356744f, 0.937498331f, 1.385216713f,
    1.256720543f, -0.355837345f, 1.507807732f, -0.943606257f, -0.654643536f, 0.141403437f, -0.733771205f, 1.778186560f,
    0.254359484f, 0.034042001f, 1.820943594f, 0.470926762f, 0.997800827f, -0.947544813f, -0.973272562f, 1.565780640f,
    -0.972122073f, -0.172074139f, -0.780366302f, -0.018243432f, -0.442839742f, -0.381048918f, -0.363562047f, -0.161547184f,
    1.97122860f, -0.079811931f, 0.247272134f, 0.334599495f, 1.427205563f, 1.751028538f, 1.879860640f, -0.979990005f,
    0.431378245f, -0.714987516f, 1.234764576f, -0.018693268f, -0.588499546f, -0.086845994f, 1.424705029f, 1.664449692f,
    1.395023346f, 0.789302588f, -0.676198661f
};

static const float input_1_scale = 0.011679563f;
static const float input_1_zero_point = 0.509154201f;
static const int8_t input_1_scales_frac[] = { 21 };


#define INPUT_1_TSR_SHARED_DESCR \
    .data = {.capacity=0, .mem.pi8=nullptr},\
    .shape = { 7, 7, 3 }, \
    .rank = 3, \
    .mem_stride = { 0 }

static const mli_tensor input_1_tsr_sa8 = {
    INPUT_1_TSR_SHARED_DESCR,

    .el_type = MLI_EL_SA_8,
    .el_params.sa.dim = -1,
    .el_params.sa.zero_point = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale_frac_bits = {.capacity = 0, .mem.pi8 = nullptr} // 21
};

static const mli_tensor input_1_tsr_fx8 = {
    INPUT_1_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_8,
    .el_params.fx.frac_bits = 6
};

static const mli_tensor input_1_tsr_fx16 = {
    INPUT_1_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 14
};

tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]),
    &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

#define INPUT_1_MEMSTR_TSR_SHARED_DESCR \
    .data = {.capacity=0, .mem.pi8=nullptr},\
    .shape = { 7, 7, 3 }, \
    .rank = 3, \
    .mem_stride = { 13*6*1, 6*1, 1}

static const mli_tensor input_1_memstr_tsr_sa8 = {
    INPUT_1_MEMSTR_TSR_SHARED_DESCR,

    .el_type = MLI_EL_SA_8,
    .el_params.sa.dim = -1,
    .el_params.sa.zero_point = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale_frac_bits = {.capacity = 0, .mem.pi8 = nullptr} // 37
};

static const mli_tensor input_1_memstr_tsr_fx8 = {
    INPUT_1_MEMSTR_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_8,
    .el_params.fx.frac_bits = 6
};

static const mli_tensor input_1_memstr_tsr_fx16 = {
    INPUT_1_MEMSTR_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 14,
};

tensor_quantizer input_1_memstr_fx16(input_1_memstr_tsr_fx16, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_memstr_sa8(input_1_memstr_tsr_sa8, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]),
                                    &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

// Generated weights vectors
//========================================
static const float weights_1_data[] = {
    0.121539377f, 0.038981315f, 0.238740608f, -0.227366433f, -1.423697591f, -1.002833486f, -0.483930886f, 0.100503266f,
    -0.139419615f, 0.436625689f, -0.532677531f, 0.427570045f, -0.307641238f, 0.172509328f, -0.026271259f, 0.425837994f,
    -0.288608491f, 0.288695902f, 0.133273870f, 1.159123302f, -0.318038374f, 0.075211421f, -0.132531196f, -0.113176830f,
    0.227300689f, 0.278074265f, 0.686661422f, -0.682077289f, 0.161586493f, -0.164802447f, 0.799581826f, -0.564442813f,
    -0.396176994f, 0.236066803f, -0.308561832f, -0.136695310f, 0.100037456f, -0.632492363f, -0.201940060f, -0.169473290f,
    0.454397738f, 1.601993442f, 0.251557410f, 0.518023372f, -0.107394643f, -0.029803216f, 0.441627651f, -0.720269144f,
    0.308284104f, 0.007023496f, 0.221299991f, 0.276993394f, 0.671937406f, -0.216236874f, -0.195448056f, -0.383312225f,
    -0.080290787f, -0.104013339f, 0.051129390f, 0.828456044f, 0.519129336f, -0.083231583f, 2.043134451f, 0.073937632f,
    -0.242123961f, -0.151708424f, -0.477973312f, 0.270733327f, 0.314504147f, -1.155037999f, 0.117129698f, 0.158947885f,
    -0.653497517f, 0.022939235f, -0.248011231f, 0.169625223f, 0.447813988f, -0.003180373f, -0.026261970f, 0.452414870f,
    0.059201516f, -0.067172617f, -0.345578939f, -0.327685088f, 0.039380759f, 0.021876009f, -0.546851277f, -0.177791998f,
    0.220277116f, -0.30182460f, -0.381017804f, 0.045308102f, 0.088454835f, 0.085999437f, 0.003846063f, -0.091018461f,
    -0.904909790f, -0.382344782f, 0.111589707f, -0.155282184f, 0.004263277f, -0.185914531f, 0.176349491f, 1.426594615f,
    -0.694690406f, 0.048129093f, 0.401795059f, -0.437018067f, -0.051059466f, -0.027009116f, -0.500988185f, -0.742791295f,
    0.154998183f, -0.228159264f, 0.042742822f, -0.466717333f, 0.563317478f, 0.247517765f, 1.114162087f, -0.049444530f,
    -0.451091260f, 0.174054205f, 0.289583415f, 0.326953709f, 0.316643655f, -1.050295830f, 0.071101822f, -0.114677481f,
    -0.156217530f, -0.548054278f, 0.439401656f, -0.105688043f, 1.153027654f, 0.078443088f, 0.274467528f, -0.085858665f,
    0.174084067f, -0.442948669f, 0.486325681f, -0.919982672f, -0.119199298f, -0.039415505f, -0.064425007f, -0.059484687f,
    -0.174552381f, -0.087301873f, 0.108685292f, -0.063917093f, 0.198944986f, 0.054434042f, 0.144530609f, 0.372599989f,
    0.437938243f, -0.193609491f, -0.043178584f, -0.089182042f, 0.491961032f, -0.406788021f, -0.061561186f, 0.465097427f,
    -0.788807869f, 0.059773225f, 0.264351815f, -0.156099781f, 0.214262009f, -1.204881430f, -0.064080834f, -0.205920950f,
    0.090724051f, 0.371410072f, 0.183330402f, -0.189747140f, 0.081378780f, -0.699047506f, 0.548414052f, -0.199038938f,
    -0.032763209f, 0.292395085f, 0.731802106f, 0.663361192f, -0.041131146f, -1.132700443f, -0.004237484f, -0.079310201f,
    -0.007845246f, 0.379243433f, -0.613977134f, 0.311061054f, 0.012565632f, -0.027481252f, -0.082511358f, -0.262109607f,
    -0.09228190f, 0.770100713f, 0.823133767f, -0.929636836f, -0.002228576f, -0.276713401f, -0.733354390f, 0.097396046f,
    0.771846235f, -0.245789930f, -1.138297677f, 0.041287374f, 0.363226146f, -0.208227709f, -0.247083694f, 0.436712980f,
    0.435876101f, -0.459643662f, -0.014785624f, -0.329277813f, 0.320278764f, 0.177646920f, -0.373586893f, 0.096097499f,
    -0.307339370f, 0.111224152f, 0.173332736f, 0.082185537f, -0.084928460f, -0.085071102f, 0.104451664f, 0.481835127f,
    -0.014706351f, -0.059814841f, -0.294265062f, 0.699438274f, 0.426571071f, -0.230519280f, -0.545369089f, -0.035941798f,
    0.170828208f, 0.146618083f, -0.666299522f, 0.733037353f, -0.224516228f, -0.740856171f, 0.036224466f, -0.288005680f,
    0.165565833f, -0.201040745f, -0.399984539f, 0.824885786f, -0.741401076f, -0.118882410f, -0.185713932f, -0.254682899f,
    -0.426853001f, -0.01159260f, 0.517828643f, -0.662262499f
};

static const float weights_1_scales[] = { 0.001980767f, 0.004078924f, 0.006295920f, 0.006523276f, 0.011210217f, 0.011233028f, 0.016087674f };
static const float weights_1_zero_points[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static const int8_t weights_1_scales_frac[] = { 23, 22, 22, 22, 21, 21, 20};

#define WEIGHTS_1_TSR_SHARED_DESCR \
    .data = {.capacity=0, .mem.pi8=nullptr},\
    .shape = {3, 4, 3, 7}, \
    .rank = 4, \
    .mem_stride = {0}

static const mli_tensor weights_1_tsr_sa8 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    .el_type = MLI_EL_SA_8,
    .el_params.sa.dim = 3,
    .el_params.sa.zero_point = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale = {.capacity = 0, .mem.pi8 = nullptr},
     .el_params.sa.scale_frac_bits = {.capacity = 0, .mem.pi8 = nullptr} //36
};

static const mli_tensor weights_1_tsr_fx8 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_8,
    .el_params.fx.frac_bits = 5
};

static const mli_tensor weights_1_tsr_fx16 = {
    WEIGHTS_1_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 13
};

tensor_quantizer weights_1_fx16(weights_1_tsr_fx16, weights_1_data,
                                       sizeof(weights_1_data) / sizeof(weights_1_data[0]));
tensor_quantizer weights_1_fx8(weights_1_tsr_fx8, weights_1_data,
                                      sizeof(weights_1_data) / sizeof(weights_1_data[0]));
tensor_quantizer weights_1_sa8(weights_1_tsr_sa8, weights_1_data,
                                      sizeof(weights_1_data) / sizeof(weights_1_data[0]),
                                      weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                                      weights_1_zero_points, sizeof(weights_1_zero_points) / sizeof(weights_1_zero_points[0]),
                                      weights_1_scales_frac, sizeof(weights_1_scales_frac) / sizeof(weights_1_scales_frac[0]));

static const float weights_2_data[] = {
    -0.001188097f, 0.158430964f, 0.033262957f, -0.633298337f, -0.13963680f, 0.305942148f, -0.591342807f, -0.036758021f,
    0.232412338f, -0.178800613f, 0.080816858f, 0.481414646f, -0.508018792f, 0.708433688f, 0.008211083f, -0.177805245f,
    -0.376847386f, -0.130704850f, -0.336240917f, 1.120456338f, 0.502928853f, -0.036441471f, -0.020082472f, -0.340529919f,
    -0.03407320f, -1.012995720f, 0.221572533f, -0.337901235f, -0.089239083f, 0.300923795f, -0.217772543f, -0.115643874f,
    -0.272480905f, 0.303629249f, 0.325681090f, -0.037767690f, 0.204064757f, -0.107153647f, -0.035749901f, 0.701893210f,
    -0.097298183f, -0.378773808f, -0.112897478f, 0.141247198f, -0.024257997f, 0.139342397f, -0.194187388f, 0.006307581f,
    -0.032069135f, 0.077514552f, 0.093124047f, 0.455886424f, 0.254220098f, 0.122940816f, -0.381508678f, -0.004303260f,
    -0.066813380f, -0.266334146f, 0.234112456f, 0.596589029f, -0.956062317f, -0.022876985f, -0.759260952f, -0.138965741f,
    -0.032402743f, -0.162632108f, 0.144866064f, -0.888238132f, -0.057009783f, 0.228629276f, 0.041919630f, 0.001033493f,
    0.143752724f, -0.404819876f, -0.184150264f, 0.132288873f, -0.063037150f, 0.036310095f, -0.014957452f, 0.028623329f,
    0.933860242f, 0.046874043f, -0.613898635f, 0.127052143f, -0.150636390f, 0.122040689f, -0.240940541f, 0.126907215f,
    -0.432094723f, 0.053468354f, 0.568058372f, 0.006249467f, 0.070788667f, -0.612377226f, -0.11484120f, -0.807911396f,
    0.072652124f, 0.692732394f, -0.015197292f, -0.070149951f, -0.304998279f, -0.036144760f, -1.098420262f, -0.749431610f,
    -0.739642859f, 0.121449545f, 0.057953008f, -0.186060697f, 0.074108623f, 0.317548603f, 0.438116908f, -1.105246902f,
    -0.037531693f, -0.022304447f, -0.113697089f, -0.101493947f, -0.076509289f, -0.509375155f, -0.925535202f, 0.167301416f,
    -0.136420399f, -0.189717218f, 0.152643248f, 0.639494777f, -0.752813995f, 0.11468710f, -0.163981646f, 0.012904882f,
    0.674863279f, -0.293668032f, 0.475825787f, -0.080334291f, 0.464278638f, -0.114536725f, -0.177354872f, -0.130889043f,
    -0.314298660f, 0.042126495f, 0.971474588f, -0.113744892f, -0.122762255f, -0.296576411f, 0.083119228f, 0.036537632f,
    -0.436970115f, -0.850781262f, -0.648254752f, 0.044025764f, -0.056047019f, -0.083228536f, 0.349574864f, 0.113673046f,
    -0.702896714f, 1.108441234f, -0.061534978f, -0.333887547f, -0.020187877f, -0.768115163f, -0.109793335f, 0.013849158f,
    -0.116258159f, -0.083387949f, 0.050520938f, -0.556613564f, -0.626676619f, -0.440787971f, -0.434929341f, 0.238853112f,
    0.051639326f, 0.016697606f, -0.083599262f, -0.031606335f, -0.217136189f, 0.052453540f, -0.295803815f, 0.002231798f,
    0.070277952f, -0.062510110f, 0.433514416f, 0.259230793f, 0.740238488f, 0.560710132f, -0.215274885f, -0.228348419f,
    0.286687076f, 0.299754798f, 0.244238213f, 0.297419786f, -0.372435510f, 0.011011983f, -0.214312419f, -0.176681072f,
    0.643409312f, -0.074228197f, 0.982947230f, -0.698858798f, -0.099206530f, -0.016969206f, -0.061422843f, 0.305310547f,
    0.337030739f, 0.823659658f, 0.506570637f, 0.128595680f, -0.279233128f, -0.238742113f, -0.305386782f, -0.654793024f,
    -0.130563483f, -0.492927164f, 0.003850749f, -0.121581905f, 0.154081061f, 0.051160473f, -0.250011474f, -0.567618132f,
    0.863942802f, -0.143741980f, -0.083179489f, -0.016421042f, 0.427558810f, 0.977829158f, -0.527416110f, -0.621286809f,
    -0.027401676f, 0.055229425f, -0.210338980f, -0.015223696f, -0.315685958f, -0.112352826f, 0.216336951f, 0.095024176f,
    -0.170186132f, 0.126614869f, -0.612870872f, 0.212743670f, -0.779168010f, -0.044506028f, 0.187267870f, 0.039244618f,
    0.166786805f, -0.182345957f, -0.290119529f, -0.236063749f, 0.226787373f, 0.018864114f, -0.104759611f, -0.389850825f,
    -0.110729717f, 0.965348959f, -0.071828105f, 0.657099783f
};

static const float weights_2_scales[] = { 0.001695078f, 0.002629036f, 0.005313884f, 0.007353230f, 0.008648979f, 0.008822490f, 0.008727884f };
static const float weights_2_zero_points[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
static const int8_t weights_2_scales_frac[] = { 24, 23, 22, 22, 21, 21, 21 };

#define WEIGHTS_2_TSR_SHARED_DESCR \
    .data = {.capacity=0, .mem.pi8=nullptr},\
    .shape = { 4, 3, 3, 7 }, \
    .rank = 4, \
    .mem_stride = {0}

static const mli_tensor weights_2_tsr_sa8 = {
    WEIGHTS_2_TSR_SHARED_DESCR,

    .el_type = MLI_EL_SA_8,
    .el_params.sa.dim = 3,
    .el_params.sa.zero_point = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale_frac_bits = {.capacity = 0, .mem.pi8 = nullptr} //37
};

static const mli_tensor weights_2_tsr_fx8 = {
    WEIGHTS_2_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_8,
    .el_params.fx.frac_bits = 6
};

static const mli_tensor weights_2_tsr_fx16 = {
    WEIGHTS_2_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 14
};

tensor_quantizer weights_2_fx16(weights_2_tsr_fx16, weights_2_data,
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_fx8(weights_2_tsr_fx8, weights_2_data,
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_sa8(weights_2_tsr_sa8, weights_2_data,
    sizeof(weights_2_data) / sizeof(weights_2_data[0]),
    weights_2_scales, sizeof(weights_2_scales) / sizeof(weights_2_scales[0]),
    weights_2_zero_points, sizeof(weights_2_zero_points) / sizeof(weights_2_zero_points[0]),
    weights_2_scales_frac, sizeof(weights_2_scales_frac) / sizeof(weights_2_scales_frac[0]));

#define WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR \
    .data = {.capacity=0, .mem.pi8=nullptr},\
    .shape = { 4, 3, 3, 7 }, \
    .rank = 4, \
    .mem_stride = {3*3*8*1, 3*8*1, 8*1, 1}

static const mli_tensor weights_2_memstr_tsr_sa8 = {
    WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR,

    .el_type = MLI_EL_SA_8,
    .el_params.sa.dim = 3,
    .el_params.sa.zero_point = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale_frac_bits = {.capacity = 0, .mem.pi8 = nullptr} //37
};

static const mli_tensor weights_2_memstr_tsr_fx8 = {
    WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_8,
    .el_params.fx.frac_bits = 6
};

static const mli_tensor weights_2_memstr_tsr_fx16 = {
    WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 14
};

tensor_quantizer weights_2_memstr_fx16(weights_2_memstr_tsr_fx16, weights_2_data,
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_memstr_fx8(weights_2_memstr_tsr_fx8, weights_2_data,
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));
tensor_quantizer weights_2_memstr_sa8(weights_2_memstr_tsr_sa8, weights_2_data,
    sizeof(weights_2_data) / sizeof(weights_2_data[0]),
    weights_2_scales, sizeof(weights_2_scales) / sizeof(weights_2_scales[0]),
    weights_2_zero_points, sizeof(weights_2_zero_points) / sizeof(weights_2_zero_points[0]),
    weights_2_scales_frac, sizeof(weights_2_scales_frac) / sizeof(weights_2_scales_frac[0]));

static const float bias_1_data[] = { 0.478962898f, -0.367202461f, 0.641515911f, 0.216426253f, 0.080585465f, -0.270597309f, 0.376852006f };
static const float bias_1_scales[] = { 0.0000231345f, 0.000047640f, 0.0000735336f, 0.0000761890f, 0.0001309304f, 0.0001311969f, 0.0001878970f};
static const float bias_1_zero_points[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
static const int8_t bias_1_scales_frac[] = { 30, 29, 28, 28, 27, 27, 27 };

#define BIAS_1_TSR_SHARED_DESCR \
    .data = {.capacity=0, .mem.pi8=nullptr},\
    .shape = {7}, \
    .rank = 1, \
    .mem_stride = {0}

static const mli_tensor bias_1_tsr_sa32 = {
    BIAS_1_TSR_SHARED_DESCR,

    .el_type = MLI_EL_SA_32,
    .el_params.sa.dim = 0,
    .el_params.sa.zero_point = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale_frac_bits = {.capacity = 0, .mem.pi8 = nullptr} //43
};

static const mli_tensor bias_1_tsr_fx8 = {
    BIAS_1_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_8,
    .el_params.fx.frac_bits = 7
};

static const mli_tensor bias_1_tsr_fx16 = {
    BIAS_1_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 15
};

static const float bias_1_w2_scales[] = { 0.0000197978f, 0.0000307060f, 0.0000620638f, 0.0000858825f, 0.0001010163f, 0.0001030428f, 0.0001019379f };
static const int8_t bias_1_w2_scales_frac[] = { 30, 29, 28, 28, 28, 28, 28 };

static const mli_tensor bias_1_w2_tsr_sa32 = {
    BIAS_1_TSR_SHARED_DESCR,

    .el_type = MLI_EL_SA_32,
    .el_params.sa.dim = 0,
    .el_params.sa.zero_point = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale_frac_bits = {.capacity = 0, .mem.pi8 = nullptr} // 44
};

tensor_quantizer bias_1_fx16(bias_1_tsr_fx16, bias_1_data,
                                   sizeof(bias_1_data) / sizeof(bias_1_data[0]));
tensor_quantizer bias_1_fx8(bias_1_tsr_fx8, bias_1_data,
                                   sizeof(bias_1_data) / sizeof(bias_1_data[0]));
tensor_quantizer bias_1_sa32(bias_1_tsr_sa32, bias_1_data,
                                   sizeof(bias_1_data) / sizeof(bias_1_data[0]),
                                   bias_1_scales, sizeof(bias_1_scales) / sizeof(bias_1_scales[0]),
                                   bias_1_zero_points, sizeof(bias_1_zero_points) / sizeof(bias_1_zero_points[0]),
                                   bias_1_scales_frac, sizeof(bias_1_scales_frac) / sizeof(bias_1_scales_frac[0]));
                                   

tensor_quantizer bias_1_w2_sa32(bias_1_w2_tsr_sa32, bias_1_data,
    sizeof(bias_1_data) / sizeof(bias_1_data[0]),
    bias_1_w2_scales, sizeof(bias_1_w2_scales) / sizeof(bias_1_w2_scales[0]),
    bias_1_zero_points, sizeof(bias_1_zero_points) / sizeof(bias_1_zero_points[0]),
    bias_1_w2_scales_frac, sizeof(bias_1_w2_scales_frac) / sizeof(bias_1_w2_scales_frac[0]));


// Extracted Output vectors 
//===================================================
static const float test_1_out_data[] = {
    0.535328388f, -0.221749097f, -1.524278402f, 1.549519181f, 1.215887189f, -1.485789418f, -3.493813038f, 0.360762626f,
    -0.376689941f, -0.057757556f, -0.464838862f, -1.955297709f, 2.368459463f, -2.457421303f, 0.502509952f, 1.423059464f,
    -1.285085678f, 4.264955044f, 1.281325817f, 1.840798497f, -10.452395439f, 0.344621837f, 0.758708060f, 1.432294846f,
    2.087225914f, -2.117644310f, -3.513283014f,
    -2.168069601f, 0.372738004f, -0.360535502f, 1.355887175f, 0.360819638f, 0.958503306f, -2.431846857f, 0.690559626f,
    0.620592475f, 1.104650736f, -1.257079363f, 1.243721843f, -2.325690746f, 2.846367598f, -1.435010672f, 0.136842549f,
    -1.265274763f, -0.545728743f, 0.306806445f, 1.210850120f, 0.263827950f, -10.976535797f, 0.826310098f, -1.348274708f,
    -0.607777297f, -0.780666173f, 2.986452103f, 4.337287903f, -0.246931463f, 0.546687961f, -1.727869272f, 1.866532087f,
    -1.532512903f, 1.567803860f, 1.558582783f, -2.939911604f, 0.219526350f, -0.239676908f, 2.272932529f, 2.806646347f,
    -0.103639379f, 8.606427193f, -11.131134033f, 0.603303134f, -1.624239683f, -0.558938086f, 0.720144689f, 3.115559340f,
    -0.174777776f, 6.196098328f, 0.241342127f, -0.016108692f, -0.735345423f, 1.148536205f, 1.377587557f, 3.344902039f,
    -2.246030569f, 0.351803362f, 2.010327101f, -0.77449280f, 1.889515519f, -1.440703630f, -1.481731772f, 1.649418354f,
    -0.339770615f, -1.550270557f, -0.048358917f, 0.905905008f, 1.102876663f, 3.742504358f, -5.023919106f, 1.010514140f,
    -0.581788540f, -1.655605793f, -3.067297935f, 1.408858776f, 7.106957436f, 0.154514045f, 0.811818719f, -0.024253339f,
    -1.962688208f, -0.479133546f, 0.770052791f, 2.981967211f, -2.215245962f, 0.186186433f, -1.367815733f, 1.907425165f,
    1.422053456f, 1.793094635f, -1.298845768f,
    -6.626911163f, 0.048112571f, -1.272145748f, -3.239808321f, 0.479271412f, 1.548650861f, 4.541175842f, 2.681740284f,
    1.061969280f, 1.287585258f, 2.691403389f, 0.075811028f, -1.801050663f, 3.87277770f, -0.641383529f, 0.731324196f,
    -1.213650227f, 1.138151407f, -0.917452216f, 1.654405832f, 2.661809683f, 0.500878811f, -0.134554505f, -0.234148443f,
    -0.038627148f, -2.071105957f, 2.453381062f, 4.181341171f, 1.748562813f, 0.708138049f, 2.034394741f, -4.372180462f,
    0.530447364f, 5.304075241f, -0.324032634f, 2.621093273f, 0.495161742f, -0.996799827f, -1.573622942f, 1.327330232f,
    1.759609461f, 4.728504181f, -6.743413925f, -0.069207847f, -0.784472466f, 0.332671881f, 2.241839409f, -0.030126348f,
    2.310925007f, -1.499819040f, 0.660584927f, -1.595312119f, 3.246035814f, -1.231983781f, -2.221083879f, 2.692591906f,
    -9.371782303f, 1.023940563f, -0.376934856f, 2.370067120f, 2.265807152f, 4.717998505f, -0.165601283f, -0.808206558f,
    -0.225964427f, 1.246556044f, -0.249889970f, -0.883403182f, -3.278885603f, -0.750352144f, -0.764114141f, 0.812959611f,
    -0.125019491f, -0.067988932f, -0.297365963f, 1.852548718f, -0.043699116f, -0.510064483f
};

static const float test_1_out_scale = 0.077402197f;
static const float test_1_out_zero_point = -1.262352943f;
static const int8_t test_1_out_scales_frac[] = {18};

#define TEST_1_OUT_TSR_SHARED_DESCR \
    .data = {.capacity=0, .mem.pi8=nullptr},\
    .shape = { 7, 4, 7 }, \
    .rank = 3, \
    .mem_stride = { 0 }

static mli_tensor test_1_out_tsr_sa8 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    .el_type = MLI_EL_SA_8,
    .el_params.sa.dim = -1,
    .el_params.sa.zero_point = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale_frac_bits = {.capacity = 0, .mem.pi8 = nullptr} //34
};

static mli_tensor test_1_out_tsr_fx8 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_8,
    .el_params.fx.frac_bits = 3
};

static mli_tensor test_1_out_tsr_fx16 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 11
};

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_data,
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_data,
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]),
    &test_1_out_scale, 1, &test_1_out_zero_point, 1,
    test_1_out_scales_frac, sizeof(test_1_out_scales_frac) / sizeof(test_1_out_scales_frac[0]));


static const float test_2_out_data[] = {
    0.0f, 0.0f, 1.298560977f, 0.039126217f, 2.642681599f, 0.977957010f, 0.0f, 0.744094372f,
    0.338498652f, 0.0f, 0.0f, 1.495568275f, 1.83847940f, 3.753365278f, 1.115882158f, 0.0f,
    0.0f, 0.187201381f, 0.0f, 0.0f, 0.0f, 0.365388334f, 0.0f, 0.053870738f,
    0.312182069f, 0.0f, 0.0f, 3.122658730f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.358205050f, 0.680491805f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.066405892f, 0.063340724f, 0.0f, 0.568200171f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.270341933f, 0.0f, 0.0f, 2.322947025f, 0.0f, 1.923987508f, 4.975573540f,
    0.0f, 0.0f, 1.194999933f, 0.0f, 2.326817513f, 0.0f, 0.0f, 0.828509331f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.745433688f, 0.872721314f, 0.0f,
    0.161613882f, 1.157350183f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    2.676425934f, 0.0f, 0.0f, 0.0f, 0.626036942f, 0.0f, 0.0f, 0.0f,
    0.0f, 2.294606924f, 0.0f, 0.018213838f, 0.0f, 0.0f, 0.630528629f, 0.0f,
    0.0f, 0.896081448f, 0.758919716f, 0.0f, 0.823992670f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.221440792f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.546371043f, 0.0f, 0.0f, 4.566282749f, 2.228814602f, 0.0f, 0.249573261f, 0.053962529f,
    0.0f, 0.5051440f, 0.38035050f, 0.0f, 0.0f, 0.0f, 0.0f, 0.021784961f,
    0.065716028f, 0.444164813f, 0.0f, 0.0f, 4.844591141f, 0.712916374f, 0.0f, 0.496141791f,
    2.768072605f, 1.603745699f, 0.0f, 0.0f, 0.542137027f, 0.0f, 0.0f, 1.921600580f,
    0.0f, 0.0f, 2.792612314f, 0.587710798f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.044365346f, 0.817780077f, 0.020003438f, 0.104035318f, 3.069861412f, 2.370106697f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 1.503202081f, 0.0f, 0.0f, 0.390048593f,
    0.405348003f, 0.0f, 0.0f, 0.617841363f, 0.0f, 1.465770006f, 0.0f, 0.844059944f,
    0.0f, 0.00390470f, 0.0f, 4.768614769f, 0.0f, 0.0f, 0.431787938f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.532243371f, 0.0f, 0.471905977f, 0.0f, 0.730849862f,
    2.061485291f, 0.0f, 0.0f, 2.384791613f
};

static const float test_2_out_scale = 0.019512054f;
static const float test_2_out_zero_point = 2.487786770f;
static const int8_t test_2_out_scales_frac[] = {20};

#define TEST_2_OUT_TSR_SHARED_DESCR \
    .data = {.capacity=0, .mem.pi8=nullptr},\
    .shape = { 4, 7, 7 }, \
    .rank = 3, \
    .mem_stride = { 0 }

static mli_tensor test_2_out_tsr_sa8 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    .el_type = MLI_EL_SA_8,
    .el_params.sa.dim = -1,
    .el_params.sa.zero_point = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale_frac_bits = {.capacity = 0, .mem.pi8 = nullptr} //36
};

static mli_tensor test_2_out_tsr_fx8 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_8,
    .el_params.fx.frac_bits = 4
};

static mli_tensor test_2_out_tsr_fx16 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 12
};

tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_data,
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_data,
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]),
    &test_2_out_scale, 1, &test_2_out_zero_point, 1,
    test_2_out_scales_frac, sizeof(test_2_out_scales_frac) / sizeof(test_2_out_scales_frac[0]));


static const float test_3_out_data[] = {
    0.044364989f, -1.961867809f, -0.258086085f, -1.600380182f, 0.473281980f, 3.765779257f, 1.420011520f,
    0.295486212f, -0.705313623f, 0.407233655f, 4.214907169f, 5.011917591f, -3.013315678f, -6.079471111f,
    0.426969975f, -0.177058637f, 0.140124142f, 0.628331244f, 1.224997997f, 4.644593239f, -3.699924946f
};

static const float test_3_out_scale = 0.043495640f;
static const float test_3_out_zero_point = -0.533776760f;
static const int8_t test_3_out_scales_frac[] = {19};

#define TEST_3_OUT_TSR_SHARED_DESCR \
    .data = {.capacity=0, .mem.pi8=nullptr},\
    .shape = { 3, 1, 7 }, \
    .rank = 3, \
    .mem_stride = { 0 }

static mli_tensor test_3_out_tsr_sa8 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    .el_type = MLI_EL_SA_8,
    .el_params.sa.dim = -1,
    .el_params.sa.zero_point = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale_frac_bits = {.capacity = 0, .mem.pi8 = nullptr} //35
};

static mli_tensor test_3_out_tsr_fx8 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_8,
    .el_params.fx.frac_bits = 4
};

static mli_tensor test_3_out_tsr_fx16 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 12
};

tensor_quantizer test_3_out_fx16(test_3_out_tsr_fx16, test_3_out_data,
    sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));
tensor_quantizer test_3_out_sa8(test_3_out_tsr_sa8, test_3_out_data,
    sizeof(test_3_out_data) / sizeof(test_3_out_data[0]),
    &test_3_out_scale, 1, &test_3_out_zero_point, 1,
    test_3_out_scales_frac, sizeof(test_3_out_scales_frac) / sizeof(test_3_out_scales_frac[0]));

static const float test_4_out_data[] = {
    0.631586492f, -1.0f, 1.0f, -0.829536438f, 1.0f, -1.0f, 1.0f, 0.136842549f,
    -1.0f, -0.545728743f, 0.306806445f, 1.0f, 0.263827950f, -1.0f, 0.283196002f, -1.0f,
    1.0f, 0.420753598f, -0.661842644f, -0.068618447f, -1.0f, 1.0f, 1.0f, 1.0f,
    0.075811028f, -1.0f, 1.0f, -0.641383529f
};

static const float test_4_out_scale = 0.007843138f;
static const float test_4_out_zero_point = 0.0f;
static const int8_t test_4_out_scales_frac[] = {21};

#define TEST_4_OUT_TSR_SHARED_DESCR \
    .data = {.capacity=0, .mem.pi8=nullptr},\
    .shape = { 2, 2, 7 }, \
    .rank = 3, \
    .mem_stride = { 0 }

static mli_tensor test_4_out_tsr_sa8 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    .el_type = MLI_EL_SA_8,
    .el_params.sa.dim = -1,
    .el_params.sa.zero_point = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale_frac_bits = {.capacity = 0, .mem.pi8 = nullptr} //37
};

static mli_tensor test_4_out_tsr_fx8 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_8,
    .el_params.fx.frac_bits = 7
};

static mli_tensor test_4_out_tsr_fx16 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 15
};

tensor_quantizer test_4_out_fx16(test_4_out_tsr_fx16, test_4_out_data,
    sizeof(test_4_out_data) / sizeof(test_4_out_data[0]));
tensor_quantizer test_4_out_sa8(test_4_out_tsr_sa8, test_4_out_data,
    sizeof(test_4_out_data) / sizeof(test_4_out_data[0]),
    &test_4_out_scale, 1, &test_4_out_zero_point, 1,
    test_4_out_scales_frac, sizeof(test_4_out_scales_frac) / sizeof(test_4_out_scales_frac[0]));


static const float test_5_out_data[] = {
    0.0f, 0.0f, 0.0f, 1.747632504f, 0.0f, 0.788578153f, 0.0f, 0.0f,
    0.0f, 0.417308658f, 3.193252563f, 0.0f, 0.0f, 0.0f, 0.706569195f, 0.0f,
    0.0f, 0.0f, 0.0f, 2.456426382f, 0.154037565f, 0.846633494f, 0.0f, 0.0f,
    0.0f, 0.137218222f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.335994244f,
    0.0f, 0.0f, 1.388865113f, 0.0f, 0.0f, 1.194999933f, 0.0f, 2.326817513f,
    0.0f, 0.0f, 0.828509331f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    1.745433688f, 0.872721314f, 0.0f, 0.161613882f, 1.157350183f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 2.676425934f, 0.0f, 0.0f, 0.0f, 0.626036942f,
    0.0f, 0.0f, 0.0f, 0.0f, 2.294606924f, 0.0f, 0.638227344f, 0.0f,
    0.0f, 0.289771914f, 0.712056279f, 0.0f, 0.0f, 0.453725517f, 0.0f, 0.589880943f,
    0.9090060f, 0.0f, 0.0f, 0.444470555f, 0.0f, 0.0f, 0.0f, 6.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.450317740f, 0.0f,
    0.0f, 0.0f, 0.174193859f, 0.0f, 0.0f, 0.0f, 0.0f, 0.142332017f,
    0.0f, 1.221440792f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.546371043f, 0.0f, 0.0f, 4.566282749f, 2.228814602f, 0.0f, 0.249573261f, 0.053962529f,
    0.0f, 0.5051440f, 0.38035050f, 0.0f, 0.0f, 0.0f, 0.0f, 0.021784961f,
    0.065715730f, 0.444164813f, 0.0f, 0.0f, 4.844591141f, 0.712916434f, 0.0f, 0.496141970f,
    2.768073082f, 1.603746176f, 0.0f, 0.0f
};

static const float test_5_out_scale = 0.023529412f;
static const float test_5_out_zero_point = 3.0f;
static const int8_t test_5_out_scales_frac[] = {20};

#define TEST_5_OUT_TSR_SHARED_DESCR \
    .data = {.capacity=0, .mem.pi8=nullptr},\
    .shape = { 4, 5, 7 }, \
    .rank = 3, \
    .mem_stride = { 0 }

static mli_tensor test_5_out_tsr_sa8 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    .el_type = MLI_EL_SA_8,
    .el_params.sa.dim = -1,
    .el_params.sa.zero_point = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale = {.capacity = 0, .mem.pi8 = nullptr},
    .el_params.sa.scale_frac_bits = {.capacity = 0, .mem.pi8 = nullptr} // 36
};

static mli_tensor test_5_out_tsr_fx8 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_8,
    .el_params.fx.frac_bits = 4
};

static mli_tensor test_5_out_tsr_fx16 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    .el_type = MLI_EL_FX_16,
    .el_params.fx.frac_bits = 12
};

tensor_quantizer test_5_out_fx16(test_5_out_tsr_fx16, test_5_out_data,
    sizeof(test_5_out_data) / sizeof(test_5_out_data[0]));
tensor_quantizer test_5_out_sa8(test_5_out_tsr_sa8, test_5_out_data,
    sizeof(test_5_out_data) / sizeof(test_5_out_data[0]),
    &test_5_out_scale, 1, &test_5_out_zero_point, 1,
    test_5_out_scales_frac, sizeof(test_5_out_scales_frac) / sizeof(test_5_out_scales_frac[0]));

// Tests configuration structures
//========================================
const mli_conv2d_cfg test_1_cfg = {
    .stride_height = 1,
    .stride_width = 2,
    .padding_bottom = 1,
    .padding_top = 1,
    .padding_left = 1,
    .padding_right = 2,
    .dilation_height = 1,
    .dilation_width = 1,
    .relu.type = MLI_RELU_NONE
};

const mli_conv2d_cfg test_2_cfg = {
    .stride_height = 2,
    .stride_width = 1,
    .padding_bottom = 2,
    .padding_top = 1,
    .padding_left = 1,
    .padding_right = 1,
    .dilation_height = 1,
    .dilation_width = 1,
    .relu.type = MLI_RELU_GEN
};

const mli_conv2d_cfg test_3_cfg = {
    .stride_height = 1,
    .stride_width = 1,
    .padding_bottom = 0,
    .padding_top = 0,
    .padding_left = 0,
    .padding_right = 0,
    .dilation_height = 2,
    .dilation_width = 2,
    .relu.type = MLI_RELU_NONE
};

const mli_conv2d_cfg test_4_cfg = {
    .stride_height = 3,
    .stride_width = 3,
    .padding_bottom = 0,
    .padding_top = 0,
    .padding_left = 0,
    .padding_right = 0,
    .dilation_height = 1,
    .dilation_width = 1,
    .relu.type = MLI_RELU_1
};

const mli_conv2d_cfg test_5_cfg = {
    .stride_height = 1,
    .stride_width = 1,
    .padding_bottom = 0,
    .padding_top = 0,
    .padding_left = 0,
    .padding_right = 0,
    .dilation_height = 1,
    .dilation_width = 1,
    .relu.type = MLI_RELU_6
};


