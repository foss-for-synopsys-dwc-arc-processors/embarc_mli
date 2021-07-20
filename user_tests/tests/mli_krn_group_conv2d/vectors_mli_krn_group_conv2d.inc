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

extern mli::tst::tensor_quantizer weights_3_fx16;
extern mli::tst::tensor_quantizer weights_3_fx8;
extern mli::tst::tensor_quantizer weights_3_sa8;

extern mli::tst::tensor_quantizer weights_4_fx16;
extern mli::tst::tensor_quantizer weights_4_fx8;
extern mli::tst::tensor_quantizer weights_4_sa8;

extern mli::tst::tensor_quantizer weights_5_fx16;
extern mli::tst::tensor_quantizer weights_5_fx8;
extern mli::tst::tensor_quantizer weights_5_sa8;

extern mli::tst::tensor_quantizer weights_6_fx16;
extern mli::tst::tensor_quantizer weights_6_fx8;
extern mli::tst::tensor_quantizer weights_6_sa8;

extern mli::tst::tensor_quantizer weights_7_memstr_fx16;
extern mli::tst::tensor_quantizer weights_7_memstr_fx8;
extern mli::tst::tensor_quantizer weights_7_memstr_sa8;

extern mli::tst::tensor_quantizer weights_8_memstr_fx16;
extern mli::tst::tensor_quantizer weights_8_memstr_fx8;
extern mli::tst::tensor_quantizer weights_8_memstr_sa8;

extern mli::tst::tensor_quantizer bias_1_fx16;
extern mli::tst::tensor_quantizer bias_1_fx8;
extern mli::tst::tensor_quantizer bias_1_sa32;
extern mli::tst::tensor_quantizer bias_1_i1_w2_sa32;

extern mli::tst::tensor_quantizer bias_2_fx16;
extern mli::tst::tensor_quantizer bias_2_fx8;
extern mli::tst::tensor_quantizer bias_2_i1_w2_sa32;

extern mli::tst::tensor_quantizer bias_3_fx16;
extern mli::tst::tensor_quantizer bias_3_fx8;
extern mli::tst::tensor_quantizer bias_3_i2_w3_sa32;

extern mli::tst::tensor_quantizer bias_4_fx16;
extern mli::tst::tensor_quantizer bias_4_fx8;
extern mli::tst::tensor_quantizer bias_4_i1_w4_sa32;

extern mli::tst::tensor_quantizer bias_5_fx16;
extern mli::tst::tensor_quantizer bias_5_fx8;
extern mli::tst::tensor_quantizer bias_5_i1_w5_sa32;

extern mli::tst::tensor_quantizer bias_6_fx16;
extern mli::tst::tensor_quantizer bias_6_fx8;
extern mli::tst::tensor_quantizer bias_6_i1_w1_sa32;

extern mli::tst::tensor_quantizer bias_7_fx16;
extern mli::tst::tensor_quantizer bias_7_fx8;
extern mli::tst::tensor_quantizer bias_7_i1_w2_sa32;

extern mli::tst::tensor_quantizer bias_8_fx16;
extern mli::tst::tensor_quantizer bias_8_fx8;
extern mli::tst::tensor_quantizer bias_8_i2_w6_sa32;

extern mli::tst::tensor_quantizer bias_9_fx16;
extern mli::tst::tensor_quantizer bias_9_fx8;
extern mli::tst::tensor_quantizer bias_9_i1_w7_sa32;

extern mli::tst::tensor_quantizer bias_10_fx16;
extern mli::tst::tensor_quantizer bias_10_fx8;
extern mli::tst::tensor_quantizer bias_10_i1_w8_sa32;


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
extern const mli_conv2d_cfg test_9_cfg;
extern const mli_conv2d_cfg test_10_cfg;


// Generated input vectors 
//========================================

static const float input_1_data[] = {
    0.34744f, 1.795847f, 1.651111f, 1.44922f, -0.346746f, -1.45628f, -0.592795f, -0.165616f, 1.626643f, 1.598544f, 
    1.064811f, -1.090194f, -1.225704f, 1.007413f, -0.768389f, 0.692621f, 0.77789f, 1.210412f, 0.574106f, 0.762663f, 
    0.314541f, 1.603806f, -1.419915f, 0.05261f, 0.6023f, 1.378407f, -1.072125f, -0.466605f, -0.395504f, -0.42098f, 
    -0.377963f, 0.515759f, 0.286495f, -1.571231f, -0.165395f, -1.689665f, 0.819661f, 1.825678f, 1.238641f, 0.404465f, 
    -1.859867f, -0.569145f, 1.422361f, 1.612728f, -0.466029f, 0.426143f, -0.259091f, 1.206133f, -0.049095f, -1.788858f, 
    0.30646f, 1.120754f, -1.61946f, 1.507252f, -0.050776f, 1.322809f, -1.2715f, -0.686755f, 0.03957f, 0.946473f, 
    -1.414973f, 1.363127f, -1.700314f, -1.830491f, -1.81938f, -1.550009f, 1.324454f, -1.268708f, -1.721221f, 1.37611f, 
    0.372898f, 0.191534f, -0.859499f, -0.607218f, -0.394164f, 1.325558f, -1.526041f, -0.043891f, -1.093957f, -0.897904f, 
    0.812086f, 1.298046f, 0.635478f, 1.57971f, -1.347965f, 1.326561f, 0.613892f, 1.304487f, -0.461795f, 1.556651f, 
    1.073473f, -1.896251f, -0.99792f, 0.445872f, -0.983644f, 1.709636f, 1.289485f, 1.907383f, 0.792595f, 1.899567f, 
    1.671263f, 1.064476f, -1.355642f, 1.944411f, -0.425785f, -0.288949f, -0.344154f, -1.596044f, 0.368159f, 1.00835f, 
    -0.179722f, -1.843663f, -1.07034f, -0.919365f, -0.682547f, 1.630967f, 0.145336f, -1.016343f, 0.630849f, -1.203746f, 
    -0.240684f, 1.028098f, -0.437805f, -0.80184f, -0.118378f, 0.453418f, -1.474567f, 0.388828f, 1.101009f, 1.142011f, 
    -0.057817f, 1.933877f, 1.743309f, 1.073f, -1.028248f, -1.927287f, -0.020817f, 0.624922f, 1.399929f, -0.131262f, 
    1.969751f, -1.162651f, 0.939535f, 1.710138f, -1.055977f, -1.243387f, -0.552681f, 1.365294f, 1.712248f, -1.925052f, 
    0.376912f, 1.527325f, 1.803207f, 1.272869f, -1.175655f, 1.607178f, 0.40778f, -0.589185f, 1.392584f, -1.400213f, 
    -1.89645f, 1.235455f, 1.303012f, -0.881752f, -0.182945f, 1.403686f, 0.426116f, 0.240478f, -1.14255f, -1.778158f, 
    1.459836f, -1.046473f, 0.663171f, -1.520842f, -0.822953f, -1.173249f, -0.262207f, 0.26402f, 1.63585f, 0.843291f, 
    0.043888f, 0.695283f, -0.435317f, -0.780613f, 0.249963f, -1.247092f, 0.352851f, -0.150589f, 0.320703f, 1.507977f, 
    -1.243862f, 1.683575f, 1.213054f, -0.150545f, 1.082279f, 1.367875f, -1.605141f, 0.635952f, -0.406748f, 1.219466f, 
    -1.785135f, 0.383121f, -0.250743f, -1.606144f, 1.47841f, 1.744553f, 1.003202f, 1.696796f, 1.343074f, 1.042688f, 
    -0.133826f, -0.560333f, -1.668181f, 0.622716f, 0.604172f, -1.990211f, 1.687279f, -1.072742f, -1.138825f, 0.716816f, 
    1.252456f, 1.991191f, 1.046682f, 1.666403f, 1.484994f, -0.891333f, 1.505836f, -0.597143f, 0.49413f, 0.580634f, 
    0.089311f, -1.427835f, 1.086891f, -0.664677f, 1.938149f, -1.984167f, -1.704359f, -1.982156f, 1.019358f, 0.045067f, 
    -0.985831f, 0.327749f, 1.614845f, 0.442398f, -0.822166f, 1.611151f, 0.334586f, -1.277282f, 1.58429f, -1.896746f, 
    -1.12432f, -0.703809f, -0.272531f, -1.24627f, -0.64669f, 0.544943f, -1.372965f, -1.866776f, 1.771305f, -0.308543f, 
    -0.450579f, 1.142159f, -0.746637f, 1.999281f, 0.529871f, -0.335318f, 1.886214f, -1.737756f, -0.54133f, -0.601543f, 
    0.490079f, 0.203233f, 1.476444f, -1.478539f, 1.363326f, -0.915504f, -1.108039f, -1.296286f, 1.695655f, -1.700782f, 
    -1.397174f, -0.468571f, -0.848796f, 0.39771f, -1.740154f, 1.839934f, 1.20117f, -0.076158f, -1.433971f, -1.901771f, 
    1.286168f, 0.436081f, 1.109198f, -0.009222f, -0.216738f, -1.715814f, 0.26169f, -0.502698f, 0.128274f, -1.952157f, 
    -0.491841f, 1.098805f, 1.567678f, -0.828167f, 1.222766f, -1.79252f, 1.294323f, -0.032705f, -0.422944f, -1.870089f, 
    -0.88375f, 1.588325f, 0.785798f, -0.321389f, -1.347978f, 1.528877f, 1.186481f, -0.753827f, -0.061427f, 0.036234f, 
    -1.073123f, 1.88172f, 0.270056f, 1.840576f, 0.061087f, 1.057956f, -0.342434f, -1.891923f, 1.12311f, -0.96602f, 
    1.445812f, -1.600796f, 1.967282f, 1.96066f, -0.613705f, 0.486882f, -1.111951f, -0.476156f, 1.627519f, 0.524183f, 
    1.747849f, -1.651426f, -1.414045f, 1.763412f, 0.03855f, 0.7919f, 1.862356f, 1.727543f, -0.857955f, -1.919962f, 
    0.474524f, 1.465667f, 0.788604f, 0.413622f, 1.878906f, 0.069336f, 1.061801f, 1.878505f, 0.289565f, -0.126157f, 
    -1.385275f, -1.818636f, 0.533229f, -0.579588f, 0.406406f, 1.752979f, 1.043567f, 0.487761f, -0.533945f, 0.188833f, 
    -0.35372f, -1.67811f, -0.72235f, 1.944072f, -0.395024f, -0.053314f, -1.562372f, 1.644131f, 1.607226f, 0.563918f, 
    1.510931f, -1.862164f, 0.046538f, 1.495018f, 0.211205f, -1.854244f, -0.024829f, 0.105314f, -1.942805f, -1.183336f, 
    -1.92882f, -1.526889f, 0.43275f, 1.624353f, -1.444882f, -0.773358f, -0.022195f, 1.259018f, 0.690966f, 1.9509f, 
    1.569988f, -0.992417f, 1.530996f, 0.856251f, 0.097789f, -1.691667f, -1.61545f, -1.876118f, -1.027863f, 1.085757f, 
    1.174307f, -1.517541f, 1.83185f, 1.499249f, -1.256844f, 0.761835f, 0.379233f, -1.139825f, 1.466577f, 0.473047f, 
    0.678838f, 1.182235f, -1.282221f, -1.482653f, -0.569446f, -0.076292f, -1.541736f, 1.813431f, 0.80331f, -0.609877f, 
    -1.111129f, -0.340465f, -0.924899f, -1.843309f, 0.67319f, -0.220493f, -0.284193f, 1.902251f, 1.834526f, 0.513818f, 
    -0.136004f 
};

static const float input_1_scale = 0.0156450681f;
static const float input_1_zero_point = 0.0045347214f;
static const int8_t input_1_scales_frac[] = {20};
static const int input_1_sa_dim = -1;

static const int input_1_fx8_frac = 6;

#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {7, 7, 9}, \
    /* .rank =  */ 3

#define INPUT_1b_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {6, 7, 9}, \
    /* .rank =  */ 3

#define INPUT_1_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {13*6*1, 6*1, 1}, \
    /* .shape = */ {7, 7, 9}, \
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

static const mli_tensor input_1b_tsr_fx16 = {
    INPUT_1b_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor input_1b_tsr_sa8 = {
    INPUT_1b_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]));

tensor_quantizer input_1b_fx16(input_1b_tsr_fx16, input_1_fx8_frac + 8, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]));

tensor_quantizer input_1_memstr_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]));

tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_1b_sa8(input_1b_tsr_sa8, input_1_sa_dim, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_1_memstr_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
    sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

static const float input_2_data[] = {
    -0.126698f, 0.928649f, 0.786023f, 0.70801f, 1.112568f, -0.937178f, 3.009114f, 1.432832f, 2.555978f, -0.778635f, 
    2.58849f, 1.091639f, -0.601002f, 1.975144f, 0.641855f, -0.219165f, -0.378103f, 0.610834f, -0.711721f, 0.087156f, 
    0.665624f, 0.06839f, 5.556412f, -0.425597f, 0.600027f, -0.98535f, -0.029234f, 0.269464f, -0.153647f, 1.114534f, 
    1.491544f, 0.206799f, 1.501578f, 0.968592f, -0.098752f, 0.313456f, -0.441835f, -0.163739f, 0.423341f, 1.565129f, 
    0.809351f, -0.561295f, 1.576849f, 1.836003f, 0.844229f, 2.37553f, -0.502458f, 3.900731f, 2.445101f, -0.767367f, 
    -0.009792f, 3.573427f, 3.325302f, -0.555837f, 0.519953f, 1.267449f, 0.330502f, -0.035386f, 0.073609f, -0.497959f, 
    -0.44624f, 0.380939f, -0.104122f, 1.793071f, 0.687861f, 1.844548f, 0.966939f, 1.289068f, -0.077277f, 0.583427f, 
    0.496239f, 1.17726f, 0.406624f, 2.061589f, 3.970343f, 0.214806f, 0.716618f, 1.18626f, 0.117666f, 4.837132f, 
    -0.327397f, -0.021366f, 0.590314f, -0.249105f, 2.749338f, 3.138922f, -0.596374f, -0.805298f, -0.338488f, 7.507203f, 
    0.211f, 0.730495f, 2.652461f, 1.521816f, -0.374881f, 1.497885f
};

static const float input_2_scale = 0.0333041288f;
static const float input_2_zero_point = 3.2609262466f;
static const int8_t input_2_scales_frac[] = {19};
static const int input_2_sa_dim = -1;

static const int input_2_fx8_frac = 4;

#define INPUT_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 4, 6}, \
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
    sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);


// Generated weights vectors 
//========================================

static const float weights_1_data[] = {
    0.599116f, -0.957041f, 1.025789f, 0.645013f, 1.05513f, 0.840126f, -1.283282f, -0.217461f, -0.011488f, 0.429128f, 
    0.339057f, 0.594426f, -0.496567f, -0.439712f, 0.204459f, 1.15608f, -0.845947f, 0.193653f, -1.421526f, -1.187298f, 
    0.119052f, -1.359803f, -1.314799f, -1.174896f, -0.059042f, 0.943217f, 1.407256f, 1.315936f, 0.135136f, 1.448148f, 
    1.221068f, 0.73742f, 0.884614f, 1.158972f, 1.306176f, -1.440367f, -0.08709f, 1.185679f, -0.095599f, -0.279709f, 
    -0.06994f, -1.312558f, 1.308933f, -0.987147f, 0.204602f, -0.061492f, 1.020418f, 0.123223f, -1.404979f, -1.289267f, 
    -0.796229f, -0.611431f, -0.042228f, -0.244851f, 1.47993f, -0.674358f, 1.213408f, 1.17362f, -0.43831f, 1.325195f, 
    0.491526f, -0.46933f, 0.249245f, 0.036526f, 1.267311f, 0.792006f, 0.080032f, -0.526502f, -1.41251f, 0.33182f, 
    1.486164f, -0.490578f, -0.577771f, -0.574189f, -0.169707f, -0.551301f, 1.155509f, -1.068225f, -0.630293f, 0.709249f, 
    0.611133f, -0.094436f, -0.389813f, 0.012281f, 0.680534f, -0.330619f, -0.343507f, 0.380732f, 0.659929f, 1.110383f, 
    -0.038605f, 0.576644f, -0.068291f, 1.203978f, -1.136088f, -0.58124f, 0.799399f, -0.431878f, -1.320261f, -0.070444f, 
    -1.371032f, -0.248137f, -1.238347f, -0.848517f, -0.615651f, -0.149915f, -1.364621f, 0.37917f
};

static const float weights_1_scales[] = {0.0276622456f, 0.0277787633f, 0.0270681866f};
static const float weights_1_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t weights_1_scales_frac[] = {20, 20, 20};
static const int weights_1_sa_dim = 3;

static const int weights_1_fx8_frac = 6;

#define WEIGHTS_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 4, 3, 3}, \
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

tensor_quantizer weights_1_fx8(weights_1_tsr_fx8, weights_1_fx8_frac, weights_1_data, 
    sizeof(weights_1_data) / sizeof(weights_1_data[0]));

tensor_quantizer weights_1_fx16(weights_1_tsr_fx16, weights_1_fx8_frac + 8, weights_1_data, 
    sizeof(weights_1_data) / sizeof(weights_1_data[0]));

tensor_quantizer weights_1_sa8(weights_1_tsr_sa8, weights_1_sa_dim, weights_1_data, 
    sizeof(weights_1_data) / sizeof(weights_1_data[0]), weights_1_scales, 
    sizeof(weights_1_scales) / sizeof(weights_1_scales[0]), weights_1_zero_points, 
    sizeof(weights_1_zero_points) / sizeof(weights_1_zero_points[0]), weights_1_scales_frac, 
    sizeof(weights_1_scales_frac) / sizeof(weights_1_scales_frac[0]));


static const float weights_2_data[] = {
    -0.291052f, -0.427529f, 0.14381f, 0.400898f, 1.043984f, -1.07869f, 0.830195f, 0.141929f, -0.436863f, 0.853777f, 
    -0.979459f, -0.777343f, 0.227816f, -0.325332f, 0.195321f, -0.109276f, 0.22211f, 1.183573f, -1.058738f, 1.371129f, 
    -0.176213f, 1.229239f, -0.948821f, -1.188107f, -1.337789f, -1.160645f, -0.860161f, -0.250731f, 1.385696f, -1.418006f, 
    -0.179544f, 0.599501f, 1.009745f, 0.706817f, -0.260571f, 1.275583f, -1.054866f, -0.319359f, -0.708735f, -0.449993f, 
    -1.25018f, 1.166596f, 0.451804f, 0.95704f, 0.691216f, 0.3665f, 0.397718f, -0.249592f, -0.455882f, 1.387704f, 
    1.280796f, -0.351868f, 1.409526f, 1.382051f, 0.546113f, -0.539793f, -0.324905f, 0.097324f, -1.467627f, -1.099779f, 
    0.575692f, -1.130292f, -1.167656f, -0.412624f, -0.84857f, -0.981388f, 0.309201f, -0.787837f, -1.482409f, -0.025015f, 
    -0.426958f, 0.109809f, -1.408774f, -0.941872f, 0.02369f, -0.537461f, -0.823215f, -0.22367f, -0.150818f, 0.007654f, 
    1.456254f, -0.310223f, 0.953662f, -0.403724f, -0.575552f, 0.908165f, 0.145221f, 0.608107f, -0.866952f, 0.830559f, 
    1.228824f, 0.815209f, -1.134585f, -0.291341f, -0.339855f, -0.104027f, -1.258391f, 1.392009f, -0.646005f, -0.006177f, 
    -0.019174f, -0.767491f, 1.438075f, -1.240953f, 1.446894f, -1.308846f, 0.278932f, 0.309422f 
};

static const float weights_2_scales[] = {0.0268799085f, 0.0274322815f, 0.0277085733f};
static const float weights_2_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t weights_2_scales_frac[] = {20, 20, 20};
static const int weights_2_sa_dim = 3;

static const int weights_2_fx8_frac = 6;

#define WEIGHTS_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 3, 3, 3}, \
    /* .rank =  */ 4

#define WEIGHTS_2_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {3*(3+1)*(3+1)*1, (3+1)*(3+1)*1, (3+1)*1, 1}, \
    /* .shape = */ {4, 3, 3, 3}, \
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

tensor_quantizer weights_2_fx8(weights_2_tsr_fx8, weights_2_fx8_frac, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));

tensor_quantizer weights_2_memstr_fx8(weights_2_memstr_tsr_fx8, weights_2_fx8_frac, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));

tensor_quantizer weights_2_fx16(weights_2_tsr_fx16, weights_2_fx8_frac + 8, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));

tensor_quantizer weights_2_memstr_fx16(weights_2_memstr_tsr_fx16, weights_2_fx8_frac + 8, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]));

tensor_quantizer weights_2_sa8(weights_2_tsr_sa8, weights_2_sa_dim, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]), weights_2_scales, 
    sizeof(weights_2_scales) / sizeof(weights_2_scales[0]), 
    weights_2_zero_points, sizeof(weights_2_zero_points) / sizeof(weights_2_zero_points[0]), 
    weights_2_scales_frac, sizeof(weights_2_scales_frac) / sizeof(weights_2_scales_frac[0]));

tensor_quantizer weights_2_memstr_sa8(weights_2_memstr_tsr_sa8, weights_2_sa_dim, weights_2_data, 
    sizeof(weights_2_data) / sizeof(weights_2_data[0]), weights_2_scales, 
    sizeof(weights_2_scales) / sizeof(weights_2_scales[0]), 
    weights_2_zero_points, sizeof(weights_2_zero_points) / sizeof(weights_2_zero_points[0]), 
    weights_2_scales_frac, sizeof(weights_2_scales_frac) / sizeof(weights_2_scales_frac[0]));


static const float weights_3_data[] = {
    -1.099211f, 0.14328f, 0.713995f, -1.388396f, -0.633342f, 1.031406f, -1.156235f, -1.037323f, 1.012432f, 0.614725f, 
    1.494054f, -0.999417f, 0.725834f, -0.934642f, -0.78084f, -0.878779f, -0.753668f, 0.074169f, 0.854229f, -1.4087f, 
    1.319065f, 1.153341f, -0.740705f, -0.185719f, 1.010994f, 0.388403f, 1.133544f, -1.178498f, -1.057148f, -0.168803f, 
    -1.473734f, -0.312192f, 1.065553f, 0.372157f, -0.640058f, -0.169515f, 0.344737f, -0.9202f, 0.09609f, -0.672267f, 
    -1.330042f, -0.50917f, 1.381277f, -1.343965f, 0.763633f, -1.355451f, 0.040416f, -0.06497f, -0.814266f, -0.630522f, 
    1.164706f, 0.544234f, -0.913119f, 1.456264f, 0.630999f, 1.423339f, 0.765145f, 0.237088f, -0.387604f, -0.187486f, 
    0.844082f, 0.749716f, -1.127191f, -1.454228f, 1.184528f, 0.644599f, 0.53982f, -0.69408f, -1.356243f, 0.686504f, 
    1.020267f, -0.595069f, -1.401258f, -1.266855f, -0.860634f, -1.46705f, 0.94661f, 1.156553f, -0.051253f, -0.027294f, 
    0.602751f, 0.085623f, 0.487756f, -0.337362f, 0.429829f, 0.990202f, -1.255305f, -0.937096f, -0.274874f, -1.304501f, 
    0.240647f, 0.358226f, 0.515374f, -0.70224f, 0.74863f, -0.887448f, -0.766403f, 0.202741f, -0.065361f, -0.192919f, 
    1.456514f, 0.322423f, -0.881341f, -0.444883f, 1.044491f, -0.114037f, 0.447241f, 1.18756f, 0.588087f, -0.39129f, 
    -0.822657f, -1.190439f, -0.572196f, 0.932961f, -0.604967f, -0.995419f, 1.004461f, -1.041163f, -1.411765f, 1.408691f, 
    -0.572496f, 0.254428f, -0.142774f, -1.234648f, -1.067017f, -1.171099f, -1.164588f, -1.401344f, -0.18192f, -1.096604f, 
    -1.303108f, -0.844576f, 0.628173f, 0.959907f, -0.314636f, 0.670827f, -0.994268f, 0.504003f, 0.954777f, -0.972154f, 
    -0.003079f, 0.531385f, -1.112697f, 1.143665f, -0.909078f, 0.536288f, 0.342767f, 1.13515f, -1.191937f, -0.202415f, 
    0.335692f, 0.856849f, 0.553463f, 0.323195f, -1.370499f, -0.80307f, 0.714555f, 0.655908f, 1.019708f, 0.243696f, 
    0.768885f, -1.051909f, -1.247447f, 0.152513f, 1.30555f, -0.476248f, 1.236347f, 1.335135f, -0.805034f, -0.487353f, 
    -0.074671f, -1.361339f, 0.744396f, -0.092615f, -1.395746f, 1.071974f, -0.623828f, 0.99479f, -1.320602f, -0.0381f, 
    1.391514f, -0.543023f, -0.000013f, -0.739013f, -0.002672f, 1.093107f, 1.051374f, -1.300983f, -1.044859f, -0.057661f, 
    0.389916f, 0.212912f
};

static const float weights_3_scales[] = {0.0152514521f, 0.0152488425f, 0.0156445447f, 0.0153617784f};
static const float weights_3_zero_points[] = {0.f, 0.f, 0.f, 0.f};
static const int8_t weights_3_scales_frac[] = {21, 21, 20, 21};
static const int weights_3_sa_dim = 3;

static const int weights_3_fx8_frac = 6;

#define WEIGHTS_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 4, 3, 4}, \
    /* .rank =  */ 4


static const mli_tensor weights_3_tsr_fx8 = {
    WEIGHTS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_3_tsr_fx16 = {
    WEIGHTS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_3_tsr_sa8 = {
    WEIGHTS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_3_fx8(weights_3_tsr_fx8, weights_3_fx8_frac, weights_3_data, 
    sizeof(weights_3_data) / sizeof(weights_3_data[0]));

tensor_quantizer weights_3_fx16(weights_3_tsr_fx16, weights_3_fx8_frac + 8, weights_3_data, 
    sizeof(weights_3_data) / sizeof(weights_3_data[0]));

tensor_quantizer weights_3_sa8(weights_3_tsr_sa8, weights_3_sa_dim, weights_3_data, 
    sizeof(weights_3_data) / sizeof(weights_3_data[0]), weights_3_scales, 
    sizeof(weights_3_scales) / sizeof(weights_3_scales[0]), 
    weights_3_zero_points, sizeof(weights_3_zero_points) / sizeof(weights_3_zero_points[0]), 
    weights_3_scales_frac, sizeof(weights_3_scales_frac) / sizeof(weights_3_scales_frac[0]));


static const float weights_4_data[] = {
    -0.719342f, 0.373054f, -0.907701f, 0.466273f, -1.087744f, -0.968199f, 0.479328f, 0.022109f, -0.301937f, -0.921682f, 
    -0.161145f, -0.508726f, -0.202059f, 0.488647f, 0.74206f, -0.277094f, 0.240183f, 0.416178f, 1.082067f, 0.712138f, 
    -1.204442f, 0.619261f, 1.336648f, -0.253904f, -1.122485f, 1.343918f, -0.577164f, -1.377996f, 1.132262f, -0.644173f, 
    -0.547496f, 0.517939f, -0.015077f, 0.993394f, -0.314189f, -1.455925f, 1.276243f, 1.404688f, -0.347006f, -0.328845f, 
    -0.534283f, 0.349469f, 0.406634f, -0.123586f, -0.399496f, -0.09538f, 1.434243f, 1.477504f, -0.505389f, -0.970828f, 
    -1.055286f, 1.240519f, 1.295479f, 0.634935f, 1.45037f, -0.978124f, 1.205364f, -0.201763f, -0.725985f, -1.312768f, 
    0.045377f, -1.237732f, -0.303917f, 0.354355f, 1.110189f, -1.461154f, -0.583547f, 0.962269f, -0.968384f, -1.388294f, 
    -1.322699f, -0.352843f, 0.55242f, -0.235703f, -1.386272f, 0.359412f, 1.469686f, 1.174704f, 0.84968f, 1.231472f, 
    -0.069636f
};

static const float weights_4_scales[] = {
    0.0362592451f, 0.0358560793f, 0.0369376019f, 0.015481526f, 0.0367421396f, 0.032819204f, 0.0347073451f, 
    0.0335979573f, 0.0363981314f
};

static const float weights_4_zero_points[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
static const int8_t weights_4_scales_frac[] = {19, 19, 19, 21, 19, 19, 19, 19, 19};
static const int weights_4_sa_dim = 3;

static const int weights_4_fx8_frac = 6;

#define WEIGHTS_4_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 3, 1, 9}, \
    /* .rank =  */ 4


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

tensor_quantizer weights_4_fx8(weights_4_tsr_fx8, weights_4_fx8_frac, weights_4_data, 
    sizeof(weights_4_data) / sizeof(weights_4_data[0]));

tensor_quantizer weights_4_fx16(weights_4_tsr_fx16, weights_4_fx8_frac + 8, weights_4_data, 
    sizeof(weights_4_data) / sizeof(weights_4_data[0]));

tensor_quantizer weights_4_sa8(weights_4_tsr_sa8, weights_4_sa_dim, weights_4_data, 
    sizeof(weights_4_data) / sizeof(weights_4_data[0]), weights_4_scales, 
    sizeof(weights_4_scales) / sizeof(weights_4_scales[0]), 
    weights_4_zero_points, sizeof(weights_4_zero_points) / sizeof(weights_4_zero_points[0]), 
    weights_4_scales_frac, sizeof(weights_4_scales_frac) / sizeof(weights_4_scales_frac[0]));


static const float weights_5_data[] = {
    -0.829969f, -0.607751f, 0.923353f, 1.493069f, 0.441025f, -0.701544f, -0.837726f, -0.100275f, 1.147081f, -0.590371f, 
    -1.386117f, -1.011521f, 0.674457f, -0.746186f, -0.993227f, -0.700931f, -0.7101f, -0.697989f, 0.086856f, 1.25833f, 
    -0.459897f, 0.138725f, 0.406101f, 0.411456f, -0.451467f, -0.602175f, -0.239672f, 0.965451f, -0.773031f, -0.855753f, 
    -0.872041f, -0.583814f, -0.968178f, -0.76778f, 0.840435f, -1.218603f
};

static const float weights_5_scales[] = {0.0853182077f};
static const float weights_5_zero_points[] = {0.f};
static const int8_t weights_5_scales_frac[] = {18};
static const int weights_5_sa_dim = 3;

static const int weights_5_fx8_frac = 6;

#define WEIGHTS_5_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 2, 9, 1}, \
    /* .rank =  */ 4


static const mli_tensor weights_5_tsr_fx8 = {
    WEIGHTS_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_5_tsr_fx16 = {
    WEIGHTS_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_5_tsr_sa8 = {
    WEIGHTS_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_5_fx8(weights_5_tsr_fx8, weights_5_fx8_frac, weights_5_data, 
    sizeof(weights_5_data) / sizeof(weights_5_data[0]));

tensor_quantizer weights_5_fx16(weights_5_tsr_fx16, weights_5_fx8_frac + 8, weights_5_data, 
    sizeof(weights_5_data) / sizeof(weights_5_data[0]));

tensor_quantizer weights_5_sa8(weights_5_tsr_sa8, weights_5_sa_dim, weights_5_data, 
    sizeof(weights_5_data) / sizeof(weights_5_data[0]), weights_5_scales, 
    sizeof(weights_5_scales) / sizeof(weights_5_scales[0]), 
    weights_5_zero_points, sizeof(weights_5_zero_points) / sizeof(weights_5_zero_points[0]), 
    weights_5_scales_frac, sizeof(weights_5_scales_frac) / sizeof(weights_5_scales_frac[0]));


static const float weights_6_data[] = {
    -0.148327f, 0.26408f, -0.139476f, -0.314901f, 0.045452f, -0.108698f, 0.342114f, -0.305877f, 0.255444f, 0.230926f, 
    -0.156448f, 0.226737f, 0.194082f, -0.003199f, -0.051591f, -0.03538f, 0.31787f, -0.468218f, 0.237157f, -0.46416f, 
    -0.265753f, 0.234505f, -0.072885f, -0.121499f, 0.169543f, -0.253977f, -0.092765f, -0.235683f, -0.264479f, 0.444963f, 
    0.409693f, 0.428938f, -0.037149f, -0.481214f, -0.101243f, -0.384948f, -0.2277f, -0.32487f, 0.347877f, -0.317341f, 
    0.047197f, -0.439019f, 0.125852f, 0.433259f, -0.170459f, 0.057731f, 0.014833f, 0.053226f
};

static const float weights_6_scales[] = {0.013526381f, 0.0204772055f, 0.0174337402f, 0.0197515022f};
static const float weights_6_zero_points[] = {0.f, 0.f, 0.f, 0.f};
static const int8_t weights_6_scales_frac[] = {21, 20, 20, 20};
static const int weights_6_sa_dim = 3;

static const int weights_6_fx8_frac = 8;

#define WEIGHTS_6_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 2, 3, 4}, \
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

tensor_quantizer weights_6_fx8(weights_6_tsr_fx8, weights_6_fx8_frac, weights_6_data, 
    sizeof(weights_6_data) / sizeof(weights_6_data[0]));

tensor_quantizer weights_6_fx16(weights_6_tsr_fx16, weights_6_fx8_frac + 8, weights_6_data, 
    sizeof(weights_6_data) / sizeof(weights_6_data[0]));

tensor_quantizer weights_6_sa8(weights_6_tsr_sa8, weights_6_sa_dim, weights_6_data, 
    sizeof(weights_6_data) / sizeof(weights_6_data[0]), weights_6_scales, 
    sizeof(weights_6_scales) / sizeof(weights_6_scales[0]), 
    weights_6_zero_points, sizeof(weights_6_zero_points) / sizeof(weights_6_zero_points[0]), 
    weights_6_scales_frac, sizeof(weights_6_scales_frac) / sizeof(weights_6_scales_frac[0]));


static const float weights_7_data[] = {
    -0.590371f, -0.100275f, 0.320175f, -0.459897f, -0.383404f, 1.299925f, -0.583814f, -0.2179f, -1.436085f, 0.086856f, 
    -0.7101f, -1.393318f, -0.855753f, 1.313684f, -0.316963f, -0.701544f, -1.201008f, -1.217581f, 0.965451f, -0.602175f, 
    -0.695801f, 1.493069f, 1.191503f, 0.697392f, -0.993227f, 0.091547f, 0.159442f, -0.607751f, 0.840435f, 0.50565f, 
    0.674457f, -0.887911f, 0.564818f, 0.411456f, 1.244663f, -0.888256f, -1.386117f, 1.147081f, 0.551898f, 0.138725f, 
    1.387913f, -1.417591f, -0.968178f, 1.296999f, 0.563838f, 1.25833f, -0.697989f, 0.695386f, -0.872041f, -0.5852f, 
    1.362935f, -0.837726f, -1.096997f, -1.03849f, -0.773031f, -0.239672f, 0.476433f, 0.441025f, -0.549055f, -0.054397f, 
    -0.700931f, -0.456982f, 0.576189f, 0.923353f, -1.218603f, -0.545817f, -0.746186f, -0.616798f, 0.476321f, -0.451467f, 
    -0.331918f, -0.2596f, -1.011521f, -0.256656f, 0.256399f, 0.406101f, 1.201238f, 1.477726f, -0.76778f, -0.943317f, 
    -0.851421f  
};

static const float weights_7_scales[] = {0.0373267159f, 0.0346978381f, 0.0369431563f};
static const float weights_7_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t weights_7_scales_frac[] = {19, 19, 19};
static const int weights_7_sa_dim = 3;

static const int weights_7_fx8_frac = 6;

#define WEIGHTS_7_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {3*(3+2)*(3+2)*1, (3+2)*(3+2)*1, (3+2)*1, 1}, \
    /* .shape = */ {3, 3, 3, 3}, \
    /* .rank =  */ 4


static const mli_tensor weights_7_memstr_tsr_fx8 = {
    WEIGHTS_7_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_7_memstr_tsr_fx16 = {
    WEIGHTS_7_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_7_memstr_tsr_sa8 = {
    WEIGHTS_7_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_7_memstr_fx8(weights_7_memstr_tsr_fx8, weights_7_fx8_frac, weights_7_data, 
    sizeof(weights_7_data) / sizeof(weights_7_data[0]));

tensor_quantizer weights_7_memstr_fx16(weights_7_memstr_tsr_fx16, weights_7_fx8_frac + 8, weights_7_data, 
    sizeof(weights_7_data) / sizeof(weights_7_data[0]));

tensor_quantizer weights_7_memstr_sa8(weights_7_memstr_tsr_sa8, weights_7_sa_dim, weights_7_data, 
    sizeof(weights_7_data) / sizeof(weights_7_data[0]), weights_7_scales, 
    sizeof(weights_7_scales) / sizeof(weights_7_scales[0]), 
    weights_7_zero_points, sizeof(weights_7_zero_points) / sizeof(weights_7_zero_points[0]), 
    weights_7_scales_frac, sizeof(weights_7_scales_frac) / sizeof(weights_7_scales_frac[0]));


static const float weights_8_data[] = {
    -0.704877f, 1.035034f, 0.849127f, 0.250362f, -1.024443f, -0.484717f, 0.195743f, 0.52415f, 1.325264f, 0.940646f, 
    -0.832692f, 0.992065f, 1.124879f, 1.078469f, 0.023182f, 0.788705f, -0.398872f, 0.676f, 0.652909f, 0.954792f, 
    0.775759f, 0.421754f, 1.15018f, -1.044632f, 0.876632f, -0.094915f, 0.792491f, 0.996059f, -0.176096f, 0.489679f, 
    0.274044f, 0.485402f, 0.13195f, -1.320384f, 0.082236f, -0.584685f, 0.75683f, -1.173981f, 0.97196f, -0.644314f, 
    -1.413625f, -1.214739f, -0.32853f, 1.065212f, -0.307409f, -1.394656f, -0.871613f, 0.881731f, -0.633288f, -1.337325f, 
    -0.866988f, -0.193804f, -1.228034f, 1.181859f, -0.70653f, -0.425108f, 0.100525f, -1.431769f, -1.27503f, 1.086997f, 
    -1.4592f, 0.553393f, -1.338327f, -0.751172f, -0.927433f, 1.387365f, 0.264185f, 0.598631f, 1.29213f, -1.04175f, 
    0.183705f, -1.163349f, 0.698175f, -0.393903f, 1.487649f, -0.765646f, -1.484652f, 0.743235f, -0.64907f, -0.630801f, 
    0.446667f, 0.804163f, 0.481621f, -0.209915f, 0.873962f, 0.783699f, -1.316577f, 1.284951f, -0.862636f, 1.25897f, 
    -1.147934f, 0.767565f, -0.109869f, -1.349882f, 1.251665f, 0.383593f, -0.802056f, -1.112355f, -1.016856f, 1.285377f, 
    -0.373947f, -0.320957f, 0.454418f, 0.192227f, -0.39489f, -0.826976f, 1.416549f, -0.184875f, -0.128744f, 0.509501f, 
    -0.691616f, 0.130002f, 1.491502f, -0.833604f, 1.460116f, 0.477953f, -0.388169f, 1.435615f, -0.932816f, 0.990809f, 
    1.375668f, -1.412393f, 0.351713f, 0.500304f, 0.627325f, -0.185581f, 0.938807f, -0.891686f, -0.103235f, -0.843384f, 
    -0.055476f, 0.258946f, -0.432886f, 0.397042f, 0.789858f, 0.077423f, 0.254962f, 1.187185f, 0.719093f, 0.343169f, 
    -0.684755f, -0.048555f, -0.712781f, -1.153779f, -0.248013f, -0.486174f, 0.05112f, -0.061364f, 1.422476f, 0.208933f, 
    -0.529444f, 0.711907f, -0.547436f, 1.480752f, 0.816922f, -0.3018f, 1.208148f, -1.482086f, -0.606723f, 1.007616f, 
    -1.385796f, 1.384693f, 0.051792f, 0.987806f, 0.925697f, 0.146416f, -0.489453f, 1.016317f, 0.056561f, 0.251009f, 
    -1.115105f, -0.702977f, -1.415132f, 1.305998f, -1.347228f, 1.350351f, 1.066485f, -0.129595f, -1.07282f, 0.11232f, 
    -1.01023f, 1.306143f, -0.687448f, -0.960959f, 1.270385f, -0.681959f, -0.498674f, 0.484894f, 0.925955f, 0.370443f, 
    0.062946f, -1.007421f, 0.180016f, 0.544324f, -0.830588f, 0.415717f, 1.246952f, -0.829548f, -0.322224f, 0.773096f, 
    0.770475f, -0.330651f, -0.849134f, 0.043755f, 0.637424f, -0.610444f, -0.563792f, -0.21351f, 1.128617f, 0.881724f, 
    -0.34218f, -0.17413f, 0.782114f, -1.258166f, -1.286584f, -0.292003f, 0.694564f, 1.411646f, 1.26232f, -0.195146f, 
    -1.132873f, -0.997212f, 1.403898f, -1.257369f, -0.334995f  
};

static const float weights_8_scales[] = {0.0132210003f, 0.0133169806f, 0.0132825822f};
static const float weights_8_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t weights_8_scales_frac[] = {21, 21, 21};
static const int weights_8_sa_dim = 3;

static const int weights_8_fx8_frac = 6;

#define WEIGHTS_8_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {5*(3+2)*(3+2)*1, (3+2)*(3+2)*1, (3+2)*1, 1}, \
    /* .shape = */ {5, 5, 3, 3}, \
    /* .rank =  */ 4


static const mli_tensor weights_8_memstr_tsr_fx8 = {
    WEIGHTS_8_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_8_memstr_tsr_fx16 = {
    WEIGHTS_8_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor weights_8_memstr_tsr_sa8 = {
    WEIGHTS_8_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

tensor_quantizer weights_8_memstr_fx8(weights_8_memstr_tsr_fx8, weights_8_fx8_frac, weights_8_data, 
    sizeof(weights_8_data) / sizeof(weights_8_data[0]));

tensor_quantizer weights_8_memstr_fx16(weights_8_memstr_tsr_fx16, weights_8_fx8_frac + 8, weights_8_data, 
    sizeof(weights_8_data) / sizeof(weights_8_data[0]));

tensor_quantizer weights_8_memstr_sa8(weights_8_memstr_tsr_sa8, weights_8_sa_dim, weights_8_data, 
    sizeof(weights_8_data) / sizeof(weights_8_data[0]), weights_8_scales, 
    sizeof(weights_8_scales) / sizeof(weights_8_scales[0]), 
    weights_8_zero_points, sizeof(weights_8_zero_points) / sizeof(weights_8_zero_points[0]), 
    weights_8_scales_frac, sizeof(weights_8_scales_frac) / sizeof(weights_8_scales_frac[0]));


// Generated bias vectors 
//========================================

static const float bias_1_data[] = {
    -0.131594f, -0.164709f, -0.060911f
};

static const float bias_1_scales[] = {0.0004327777f, 0.0004346006f, 0.0004234836f};
static const float bias_1_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t bias_1_scales_frac[] = {26, 26, 26};
static const int bias_1_sa_dim = 0;

static const int bias_1_fx8_frac = 9;

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

static const mli_tensor bias_1_tsr_sa32 = {
    BIAS_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer bias_1_fx8(bias_1_tsr_fx8, bias_1_fx8_frac, bias_1_data, 
    sizeof(bias_1_data) / sizeof(bias_1_data[0]));

tensor_quantizer bias_1_fx16(bias_1_tsr_fx16, bias_1_fx8_frac + 8, bias_1_data, 
    sizeof(bias_1_data) / sizeof(bias_1_data[0]));

tensor_quantizer bias_1_sa32(bias_1_tsr_sa32, bias_1_sa_dim, bias_1_data, 
    sizeof(bias_1_data) / sizeof(bias_1_data[0]), bias_1_scales, sizeof(bias_1_scales) / sizeof(bias_1_scales[0]), 
    bias_1_zero_points, sizeof(bias_1_zero_points) / sizeof(bias_1_zero_points[0]), 
    bias_1_scales_frac, sizeof(bias_1_scales_frac) / sizeof(bias_1_scales_frac[0]));


static const float bias_2_data[] = {
    0.12295f, -0.01622f, -0.143007f
};

static const float bias_2_i1_w2_scales[] = {0.000420538f, 0.0004291799f, 0.0004335025f};
static const float bias_2_i1_w2_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t bias_2_i1_w2_scales_frac[] = {26, 26, 26};
static const int bias_2_i1_w2_sa_dim = 0;

static const int bias_2_fx8_frac = 9;

#define BIAS_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3}, \
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

static const mli_tensor bias_2_i1_w2_tsr_sa32 = {
    BIAS_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer bias_2_fx8(bias_2_tsr_fx8, bias_2_fx8_frac, bias_2_data, 
    sizeof(bias_2_data) / sizeof(bias_2_data[0]));

tensor_quantizer bias_2_fx16(bias_2_tsr_fx16, bias_2_fx8_frac + 8, bias_2_data, 
    sizeof(bias_2_data) / sizeof(bias_2_data[0]));

tensor_quantizer bias_2_i1_w2_sa32(bias_2_i1_w2_tsr_sa32, bias_2_i1_w2_sa_dim, bias_2_data, 
    sizeof(bias_2_data) / sizeof(bias_2_data[0]), bias_2_i1_w2_scales, 
    sizeof(bias_2_i1_w2_scales) / sizeof(bias_2_i1_w2_scales[0]), 
    bias_2_i1_w2_zero_points, sizeof(bias_2_i1_w2_zero_points) / sizeof(bias_2_i1_w2_zero_points[0]), 
    bias_2_i1_w2_scales_frac, sizeof(bias_2_i1_w2_scales_frac) / sizeof(bias_2_i1_w2_scales_frac[0]));


static const float bias_3_data[] = {
    -0.050741f, 0.056134f, 0.007795f, -0.023348f
};

static const float bias_3_i2_w3_scales[] = {0.0005079363f, 0.0005078494f, 0.0005210279f, 0.0005116106f};
static const float bias_3_i2_w3_zero_points[] = {0.f, 0.f, 0.f, 0.f};
static const int8_t bias_3_i2_w3_scales_frac[] = {25, 25, 25, 25};
static const int bias_3_i2_w3_sa_dim = 0;

static const int bias_3_fx8_frac = 11;

#define BIAS_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4}, \
    /* .rank =  */ 1


static const mli_tensor bias_3_tsr_fx8 = {
    BIAS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_3_tsr_fx16 = {
    BIAS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_3_i2_w3_tsr_sa32 = {
    BIAS_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer bias_3_fx8(bias_3_tsr_fx8, bias_3_fx8_frac, bias_3_data, 
    sizeof(bias_3_data) / sizeof(bias_3_data[0]));

tensor_quantizer bias_3_fx16(bias_3_tsr_fx16, bias_3_fx8_frac + 8, bias_3_data, 
    sizeof(bias_3_data) / sizeof(bias_3_data[0]));

tensor_quantizer bias_3_i2_w3_sa32(bias_3_i2_w3_tsr_sa32, bias_3_i2_w3_sa_dim, bias_3_data, 
    sizeof(bias_3_data) / sizeof(bias_3_data[0]), bias_3_i2_w3_scales, 
    sizeof(bias_3_i2_w3_scales) / sizeof(bias_3_i2_w3_scales[0]), 
    bias_3_i2_w3_zero_points, sizeof(bias_3_i2_w3_zero_points) / sizeof(bias_3_i2_w3_zero_points[0]), 
    bias_3_i2_w3_scales_frac, sizeof(bias_3_i2_w3_scales_frac) / sizeof(bias_3_i2_w3_scales_frac[0]));


static const float bias_4_data[] = {
    -0.329861f, -0.24223f, -0.278861f, -0.174201f, -0.130191f, 0.042459f, 0.306955f, -0.095607f, -0.103371f
};

static const float bias_4_i1_w4_scales[] = {
    0.0005672784f, 0.0005609708f, 0.0005778913f, 0.0002422095f, 0.0005748333f, 0.0005134587f, 0.0005429988f, 
    0.0005256424f, 0.0005694513f
};

static const float bias_4_i1_w4_zero_points[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
static const int8_t bias_4_i1_w4_scales_frac[] = {25, 25, 25, 27, 25, 25, 25, 25, 25};
static const int bias_4_i1_w4_sa_dim = 0;

static const int bias_4_fx8_frac = 8;

#define BIAS_4_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {9}, \
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

static const mli_tensor bias_4_i1_w4_tsr_sa32 = {
    BIAS_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer bias_4_fx8(bias_4_tsr_fx8, bias_4_fx8_frac, bias_4_data, 
    sizeof(bias_4_data) / sizeof(bias_4_data[0]));

tensor_quantizer bias_4_fx16(bias_4_tsr_fx16, bias_4_fx8_frac + 8, bias_4_data, 
    sizeof(bias_4_data) / sizeof(bias_4_data[0]));

tensor_quantizer bias_4_i1_w4_sa32(bias_4_i1_w4_tsr_sa32, bias_4_i1_w4_sa_dim, bias_4_data, 
    sizeof(bias_4_data) / sizeof(bias_4_data[0]), bias_4_i1_w4_scales, 
    sizeof(bias_4_i1_w4_scales) / sizeof(bias_4_i1_w4_scales[0]), 
    bias_4_i1_w4_zero_points, sizeof(bias_4_i1_w4_zero_points) / sizeof(bias_4_i1_w4_zero_points[0]), 
    bias_4_i1_w4_scales_frac, sizeof(bias_4_i1_w4_scales_frac) / sizeof(bias_4_i1_w4_scales_frac[0]));


static const float bias_5_data[] = {-0.126657f};

static const float bias_5_i1_w5_scale = 0.0013348091f;
static const float bias_5_i1_w5_zero_point = 0.f;
static const int8_t bias_5_i1_w5_scales_frac[] = {24};
static const int bias_5_i1_w5_sa_dim = 0;

static const int bias_5_fx8_frac = 9;

#define BIAS_5_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1}, \
    /* .rank =  */ 1


static const mli_tensor bias_5_tsr_fx8 = {
    BIAS_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_5_tsr_fx16 = {
    BIAS_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_5_i1_w5_tsr_sa32 = {
    BIAS_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer bias_5_fx8(bias_5_tsr_fx8, bias_5_fx8_frac, bias_5_data, 
    sizeof(bias_5_data) / sizeof(bias_5_data[0]));

tensor_quantizer bias_5_fx16(bias_5_tsr_fx16, bias_5_fx8_frac + 8, bias_5_data, 
    sizeof(bias_5_data) / sizeof(bias_5_data[0]));

tensor_quantizer bias_5_i1_w5_sa32(bias_5_i1_w5_tsr_sa32, bias_5_i1_w5_sa_dim, bias_5_data, 
    sizeof(bias_5_data) / sizeof(bias_5_data[0]), &bias_5_i1_w5_scale, 1, 
    &bias_5_i1_w5_zero_point, 1, bias_5_i1_w5_scales_frac, 1);


static const float bias_6_data[] = {
    0.122248f, 0.123632f, 0.053742f
};

static const float bias_6_i1_w1_scales[] = {0.0004327777f, 0.0004346006f, 0.0004234836f};
static const float bias_6_i1_w1_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t bias_6_i1_w1_scales_frac[] = {26, 26, 26};
static const int bias_6_i1_w1_sa_dim = 0;

static const int bias_6_fx8_frac = 10;

#define BIAS_6_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3}, \
    /* .rank =  */ 1


static const mli_tensor bias_6_tsr_fx8 = {
    BIAS_6_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_6_tsr_fx16 = {
    BIAS_6_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_6_i1_w1_tsr_sa32 = {
    BIAS_6_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer bias_6_fx8(bias_6_tsr_fx8, bias_6_fx8_frac, bias_6_data, 
    sizeof(bias_6_data) / sizeof(bias_6_data[0]));

tensor_quantizer bias_6_fx16(bias_6_tsr_fx16, bias_6_fx8_frac + 8, bias_6_data, 
    sizeof(bias_6_data) / sizeof(bias_6_data[0]));

tensor_quantizer bias_6_i1_w1_sa32(bias_6_i1_w1_tsr_sa32, bias_6_i1_w1_sa_dim, bias_6_data, 
    sizeof(bias_6_data) / sizeof(bias_6_data[0]), bias_6_i1_w1_scales, 
    sizeof(bias_6_i1_w1_scales) / sizeof(bias_6_i1_w1_scales[0]), 
    bias_6_i1_w1_zero_points, sizeof(bias_6_i1_w1_zero_points) / sizeof(bias_6_i1_w1_zero_points[0]), 
    bias_6_i1_w1_scales_frac, sizeof(bias_6_i1_w1_scales_frac) / sizeof(bias_6_i1_w1_scales_frac[0]));


static const float bias_7_data[] = {
    0.02682f, 0.165051f, -0.061406f
};

static const float bias_7_i1_w2_scales[] = {0.000420538f, 0.0004291799f, 0.0004335025f};
static const float bias_7_i1_w2_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t bias_7_i1_w2_scales_frac[] = {26, 26, 26};
static const int bias_7_i1_w2_sa_dim = 0;

static const int bias_7_fx8_frac = 9;

#define BIAS_7_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3}, \
    /* .rank =  */ 1


static const mli_tensor bias_7_tsr_fx8 = {
    BIAS_7_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_7_tsr_fx16 = {
    BIAS_7_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_7_i1_w2_tsr_sa32 = {
    BIAS_7_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer bias_7_fx8(bias_7_tsr_fx8, bias_7_fx8_frac, bias_7_data, 
    sizeof(bias_7_data) / sizeof(bias_7_data[0]));

tensor_quantizer bias_7_fx16(bias_7_tsr_fx16, bias_7_fx8_frac + 8, bias_7_data, 
    sizeof(bias_7_data) / sizeof(bias_7_data[0]));

tensor_quantizer bias_7_i1_w2_sa32(bias_7_i1_w2_tsr_sa32, bias_7_i1_w2_sa_dim, bias_7_data, 
    sizeof(bias_7_data) / sizeof(bias_7_data[0]), bias_7_i1_w2_scales, 
    sizeof(bias_7_i1_w2_scales) / sizeof(bias_7_i1_w2_scales[0]), 
    bias_7_i1_w2_zero_points, sizeof(bias_7_i1_w2_zero_points) / sizeof(bias_7_i1_w2_zero_points[0]), 
    bias_7_i1_w2_scales_frac, sizeof(bias_7_i1_w2_scales_frac) / sizeof(bias_7_i1_w2_scales_frac[0]));


static const float bias_8_data[] = {
    0.259429f, 0.190487f, 0.180013f, -0.0286f
};

static const float bias_8_i2_w6_scales[] = {0.0004504843f, 0.0006819755f, 0.0005806155f, 0.0006578066f};
static const float bias_8_i2_w6_zero_points[] = {0.f, 0.f, 0.f, 0.f};
static const int8_t bias_8_i2_w6_scales_frac[] = {26, 25, 25, 25};
static const int bias_8_i2_w6_sa_dim = 0;

static const int bias_8_fx8_frac = 8;

#define BIAS_8_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4}, \
    /* .rank =  */ 1


static const mli_tensor bias_8_tsr_fx8 = {
    BIAS_8_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_8_tsr_fx16 = {
    BIAS_8_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_8_i2_w6_tsr_sa32 = {
    BIAS_8_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer bias_8_fx8(bias_8_tsr_fx8, bias_8_fx8_frac, bias_8_data, 
    sizeof(bias_8_data) / sizeof(bias_8_data[0]));

tensor_quantizer bias_8_fx16(bias_8_tsr_fx16, bias_8_fx8_frac + 8, bias_8_data, 
    sizeof(bias_8_data) / sizeof(bias_8_data[0]));

tensor_quantizer bias_8_i2_w6_sa32(bias_8_i2_w6_tsr_sa32, bias_8_i2_w6_sa_dim, bias_8_data, 
    sizeof(bias_8_data) / sizeof(bias_8_data[0]), bias_8_i2_w6_scales, 
    sizeof(bias_8_i2_w6_scales) / sizeof(bias_8_i2_w6_scales[0]), 
    bias_8_i2_w6_zero_points, sizeof(bias_8_i2_w6_zero_points) / sizeof(bias_8_i2_w6_zero_points[0]), 
    bias_8_i2_w6_scales_frac, sizeof(bias_8_i2_w6_scales_frac) / sizeof(bias_8_i2_w6_scales_frac[0]));


static const float bias_9_data[] = {
    0.1407f, -0.146251f, -0.106485f
};

static const float bias_9_i1_w7_scales[] = {0.000583979f, 0.0005428501f, 0.0005779782f};
static const float bias_9_i1_w7_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t bias_9_i1_w7_scales_frac[] = {25, 25, 25};
static const int bias_9_i1_w7_sa_dim = 0;

static const int bias_9_fx8_frac = 9;

#define BIAS_9_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3}, \
    /* .rank =  */ 1


static const mli_tensor bias_9_tsr_fx8 = {
    BIAS_9_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_9_tsr_fx16 = {
    BIAS_9_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_9_i1_w7_tsr_sa32 = {
    BIAS_9_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer bias_9_fx8(bias_9_tsr_fx8, bias_9_fx8_frac, bias_9_data, 
    sizeof(bias_9_data) / sizeof(bias_9_data[0]));

tensor_quantizer bias_9_fx16(bias_9_tsr_fx16, bias_9_fx8_frac + 8, bias_9_data, 
    sizeof(bias_9_data) / sizeof(bias_9_data[0]));

tensor_quantizer bias_9_i1_w7_sa32(bias_9_i1_w7_tsr_sa32, bias_9_i1_w7_sa_dim, bias_9_data, 
    sizeof(bias_9_data) / sizeof(bias_9_data[0]), bias_9_i1_w7_scales, 
    sizeof(bias_9_i1_w7_scales) / sizeof(bias_9_i1_w7_scales[0]), 
    bias_9_i1_w7_zero_points, sizeof(bias_9_i1_w7_zero_points) / sizeof(bias_9_i1_w7_zero_points[0]), 
    bias_9_i1_w7_scales_frac, sizeof(bias_9_i1_w7_scales_frac) / sizeof(bias_9_i1_w7_scales_frac[0]));


static const float bias_10_data[] = {
    -0.068465f, 0.103489f, -0.067884f
};

static const float bias_10_i1_w8_scales[] = {0.0002068435f, 0.0002083451f, 0.0002078069f};
static const float bias_10_i1_w8_zero_points[] = {0.f, 0.f, 0.f};
static const int8_t bias_10_i1_w8_scales_frac[] = {27, 27, 27};
static const int bias_10_i1_w8_sa_dim = 0;

static const int bias_10_fx8_frac = 10;

#define BIAS_10_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3}, \
    /* .rank =  */ 1


static const mli_tensor bias_10_tsr_fx8 = {
    BIAS_10_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_10_tsr_fx16 = {
    BIAS_10_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};

static const mli_tensor bias_10_i1_w8_tsr_sa32 = {
    BIAS_10_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer bias_10_fx8(bias_10_tsr_fx8, bias_10_fx8_frac, bias_10_data, 
    sizeof(bias_10_data) / sizeof(bias_10_data[0]));

tensor_quantizer bias_10_fx16(bias_10_tsr_fx16, bias_10_fx8_frac + 8, bias_10_data, 
    sizeof(bias_10_data) / sizeof(bias_10_data[0]));

tensor_quantizer bias_10_i1_w8_sa32(bias_10_i1_w8_tsr_sa32, bias_10_i1_w8_sa_dim, bias_10_data, 
    sizeof(bias_10_data) / sizeof(bias_10_data[0]), bias_10_i1_w8_scales, 
    sizeof(bias_10_i1_w8_scales) / sizeof(bias_10_i1_w8_scales[0]), 
    bias_10_i1_w8_zero_points, sizeof(bias_10_i1_w8_zero_points) / sizeof(bias_10_i1_w8_zero_points[0]), 
    bias_10_i1_w8_scales_frac, sizeof(bias_10_i1_w8_scales_frac) / sizeof(bias_10_i1_w8_scales_frac[0]));


// Generated output vectors 
//========================================

static const float test_1_out_data[] = {
    -4.203309f, 6.595673f, 1.005346f, -7.56347f, 2.437803f, -4.151514f, -0.358215f, 3.121201f, -7.735502f, -0.183801f, 
    -2.557331f, 0.41927f, 4.559401f, 1.974583f, -1.75534f, 6.953897f, -8.332093f, 13.654872f, -5.298742f, 4.746768f, 
    7.180227f, -2.876125f, -7.549429f, -4.206837f, -1.669751f, 6.0284f, -3.140554f, 10.631813f, -6.963354f, 
    -13.236053f, -1.938898f, -3.567002f, 1.880314f, -2.909009f, 2.491353f, -1.335037f, -0.330126f, -1.448121f, 
    10.293348f, -7.741118f, 1.340472f, 1.543093f, -3.167654f, 0.303567f, -7.003404f, 0.233948f, 4.310973f, -0.993946f, 
    13.592759f, -10.434213f, 0.484252f, 0.659447f, -5.236037f, 10.383711f, 0.910686f, 2.566551f, 6.315306f, 2.644721f, 
    0.513249f, -2.181569f, -2.59475f, 5.316718f, -4.831538f, -0.520448f, -7.194818f, -10.972982f, 1.312318f, -1.747258f, 
    4.722891f, 5.384421f, -1.151866f, 1.040806f, -4.238006f, 2.103667f, 12.301698f, -2.668591f, -3.017245f, 7.905983f, 
    -2.832315f, 0.039682f, -10.435492f, -2.078593f, -3.014738f, 2.690046f
};

static const float test_1_out_scale = 0.1054546088f;
static const float test_1_out_zero_point = 0.2094097137f;
static const int8_t test_1_out_scales_frac[] = {18};
static const int test_1_out_sa_dim = -1;

static const int test_1_out_fx8_frac = 3;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {7, 4, 3}, \
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

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data, 
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));

tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim, test_1_out_data, 
    sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_1_out_scale, 1, 
    &test_1_out_zero_point, 1, test_1_out_scales_frac, 1);


static const float test_2_out_data[] = {
    0.f, 3.334203f, 4.535436f, 0.f, 3.272548f, 8.999142f, 0.f, 1.656603f, 0.f, 0.f, 1.803915f, 0.f, 3.534855f, 
    2.336643f, 1.329538f, 5.750436f, 4.01232f, 0.f, 0.056521f, 0.f, 0.f, 0.f, 4.447572f, 3.700812f, 0.f, 0.f, 
    0.f, 0.f, 0.f, 2.285427f, 0.f, 1.310576f, 0.f, 1.580776f, 0.f, 0.f, 6.964923f, 10.383367f, 6.376374f, 0.f, 
    0.f, 1.782822f, 3.68795f, 7.803767f, 0.f, 0.112908f, 1.616996f, 0.f, 0.017248f, 0.f, 0.f, 9.861785f, 0.f, 
    5.968733f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.307624f, 0.f, 4.162376f, 0.f, 3.548999f, 0.f, 0.f, 0.f, 0.f, 0.f, 
    5.370814f, 0.f, 0.965384f, 0.f, 6.719813f, 1.695364f, 1.462528f, 0.f, 3.924559f, 2.807482f, 0.f, 0.f, 
    3.033085f, 1.879653f, 0.f
};

static const float test_2_out_scale = 0.0407190844f;
static const float test_2_out_zero_point = 5.1916832924f;
static const int8_t test_2_out_scales_frac[] = {19};
static const int test_2_out_sa_dim = -1;

static const int test_2_out_fx8_frac = 3;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 7, 3}, \
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

tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_fx8_frac + 8, test_2_out_data, 
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));

tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_sa_dim, test_2_out_data, 
    sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &test_2_out_scale, 1, 
    &test_2_out_zero_point, 1, test_2_out_scales_frac, 1);


static const float test_3_out_data[] = {0.957427f, -2.945011f, -14.240964f, -28.660917f};

static const float test_3_out_scale = 0.1161503717f;
static const float test_3_out_zero_point = -13.8517456055f;
static const int8_t test_3_out_scales_frac[] = {18};
static const int test_3_out_sa_dim = -1;

static const int test_3_out_fx8_frac = 2;

#define TEST_3_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 1, 4}, \
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

tensor_quantizer test_3_out_fx16(test_3_out_tsr_fx16, test_3_out_fx8_frac + 8, test_3_out_data, 
    sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));

tensor_quantizer test_3_out_sa8(test_3_out_tsr_sa8, test_3_out_sa_dim, test_3_out_data, 
    sizeof(test_3_out_data) / sizeof(test_3_out_data[0]), &test_3_out_scale, 1, 
    &test_3_out_zero_point, 1, test_3_out_scales_frac, 1);


static const float test_4_out_data[] = {
    0.f, 0.f, 2.570564f, 0.f, 3.23953f, 2.11739f, 0.298622f, 2.230946f, 0.f, 5.963995f, 3.34019f, 3.077807f, 
    0.240005f, 0.f, 1.031842f, 0.f, 1.924531f, 0.f, 3.268057f, 0.f, 1.930758f, 0.f, 3.529444f, 1.843976f, 0.f, 
    0.f, 0.f, 2.792207f, 4.281549f, 0.f, 0.990146f, 0.f, 4.310933f, 0.889513f, 0.f, 0.f, 1.794729f, 0.f, 1.543967f,
    1.200018f, 0.003838f, 0.f, 2.726492f, 0.f, 0.f, 0.f, 0.f, 0.668866f, 0.f, 1.955437f, 0.f, 3.644825f, 0.f, 
    1.694384f, 0.f, 0.926878f, 0.f, 0.f, 0.f, 4.006671f, 0.161543f, 0.f, 0.452262f, 4.567684f, 0.f, 2.065584f, 
    0.f, 0.261816f, 2.705401f, 0.f, 0.350874f, 0.f, 2.478328f, 0.f, 0.599952f, 0.f, 0.f, 3.834558f, 3.055648f, 
    5.177128f, 0.f, 0.f, 0.761392f, 1.269717f, 0.165119f, 0.f, 0.646161f, 0.f, 0.f, 0.f, 0.184351f, 0.f, 
    0.50734f, 0.734455f, 0.545188f, 0.f, 0.f, 3.948277f, 0.f, 5.740386f, 0.777035f, 0.f, 0.342937f, 3.772167f, 
    2.659051f, 1.724036f, 0.f, 0.887073f, 1.048669f, 0.f, 0.f, 0.f, 0.f, 3.91402f, 3.006148f, 7.326628f, 0.f, 
    0.f, 4.190682f, 3.420397f, 0.f, 4.660341f, 0.f, 0.f, 0.f, 1.648864f, 0.f, 0.f, 1.310742f, 0.844033f, 0.f, 
    0.f, 0.265358f, 1.50546f, 0.f, 3.546515f, 0.f, 0.41124f, 0.80631f, 0.f, 4.606476f, 0.232175f, 3.371388f, 
    0.255721f, 5.880758f, 0.f, 1.313719f, 0.f, 2.454969f, 0.f, 1.821706f, 0.f, 0.322076f, 0.f, 0.f, 0.f, 
    0.558972f, 1.522013f, 0.f, 0.f, 1.270885f, 0.f, 0.f, 0.f, 0.f, 1.498855f, 0.f, 3.81513f, 0.879893f, 0.f, 
    0.f, 0.f, 0.f, 0.f, 1.915731f, 0.513069f, 0.f, 1.831481f, 0.f, 0.366028f, 2.083612f, 2.812579f, 2.176717f, 
    0.414214f, 0.f, 0.f, 0.f, 0.273479f, 0.f, 0.f, 1.76207f, 0.f, 0.17416f, 0.741718f, 0.f, 6.80555f, 
    1.548975f, 0.f, 0.f, 0.039124f, 4.968717f, 0.f, 1.309213f, 1.83206f, 4.214101f, 2.155789f, 0.f, 0.f, 0.f, 
    0.f, 0.226741f, 0.f, 2.084999f, 4.348601f, 0.f, 4.125805f, 0.f, 0.f, 1.412037f, 0.317576f, 0.f, 0.f, 
    0.65389f, 0.562906f, 0.f 
};

static const float test_4_out_scale = 0.0287318733f;
static const float test_4_out_zero_point = 3.6633138657f;
static const int8_t test_4_out_scales_frac[] = {20};
static const int test_4_out_sa_dim = -1;

static const int test_4_out_fx8_frac = 4;

#define TEST_4_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {5, 5, 9}, \
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

tensor_quantizer test_4_out_fx16(test_4_out_tsr_fx16, test_4_out_fx8_frac + 8, test_4_out_data, 
    sizeof(test_4_out_data) / sizeof(test_4_out_data[0]));

tensor_quantizer test_4_out_sa8(test_4_out_tsr_sa8, test_4_out_sa_dim, test_4_out_data, 
    sizeof(test_4_out_data) / sizeof(test_4_out_data[0]), &test_4_out_scale, 1, 
    &test_4_out_zero_point, 1, test_4_out_scales_frac, 1);


static const float test_5_out_data[] = {
    -1.2766f, -7.863432f, 1.135503f, 1.947458f, 1.054694f, 2.356742f, -0.52071f, 5.203207f, -1.942583f, 1.984076f, 
    -5.439644f, 0.071168f, -0.82437f, -4.742188f, -4.094831f, 5.037953f, -11.207453f, -8.680915f, 3.005073f, 1.031898f, 
    -4.484271f, -3.687214f, -4.894147f, -4.168667f, -10.102245f, 1.917058f, -0.56688f, -0.83082f
};

static const float test_5_out_scale = 0.0643555298f;
static const float test_5_out_zero_point = -3.002122879f;
static const int8_t test_5_out_scales_frac[] = {18};
static const int test_5_out_sa_dim = -1;

static const int test_5_out_fx8_frac = 3;

#define TEST_5_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 7, 1}, \
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

tensor_quantizer test_5_out_fx16(test_5_out_tsr_fx16, test_5_out_fx8_frac + 8, test_5_out_data, 
    sizeof(test_5_out_data) / sizeof(test_5_out_data[0]));

tensor_quantizer test_5_out_sa8(test_5_out_tsr_sa8, test_5_out_sa_dim, test_5_out_data, 
    sizeof(test_5_out_data) / sizeof(test_5_out_data[0]), &test_5_out_scale, 1, 
    &test_5_out_zero_point, 1, test_5_out_scales_frac, 1);


static const float test_6_out_data[] = {
    0.f, 6.f, 1.119999f, 3.146295f, 6.f, 6.f, 0.070041f, 0.f, 0.533923f, 0.f, 0.f, 6.f, 0.f, 0.f, 1.603333f, 
    0.48779f, 4.599314f, 0.f, 0.f, 2.392008f, 6.f, 0.f, 0.f, 3.747266f, 0.f, 0.f, 2.804699f
};

static const float test_6_out_scale = 0.0235294122f;
static const float test_6_out_zero_point = 3.f;
static const int8_t test_6_out_scales_frac[] = {20};
static const int test_6_out_sa_dim = -1;

static const int test_6_out_fx8_frac = 4;

#define TEST_6_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {30, 10, 1}, \
    /* .shape = */ {3, 3, 3}, \
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

tensor_quantizer test_6_out_fx16(test_6_out_tsr_fx16, test_6_out_fx8_frac + 8, test_6_out_data, 
    sizeof(test_6_out_data) / sizeof(test_6_out_data[0]));

tensor_quantizer test_6_out_sa8(test_6_out_tsr_sa8, test_6_out_sa_dim, test_6_out_data, 
    sizeof(test_6_out_data) / sizeof(test_6_out_data[0]), &test_6_out_scale, 1, 
    &test_6_out_zero_point, 1, test_6_out_scales_frac, 1);


static const float test_7_out_data[] = {
    0.f, 3.515474f, 4.617037f, 0.f, 1.985187f, 0.f, 0.f, 0.f, 0.f, 2.252705f, 0.f, 0.f, 0.f, 2.936583f, 0.f, 
    1.202998f, 0.f, 0.195904f
};

static const float test_7_out_scale = 0.0181060266f;
static const float test_7_out_zero_point = 2.3085184097f;
static const int8_t test_7_out_scales_frac[] = {20};
static const int test_7_out_sa_dim = -1;

static const int test_7_out_fx8_frac = 4;

#define TEST_7_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 3, 3}, \
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

tensor_quantizer test_7_out_fx16(test_7_out_tsr_fx16, test_7_out_fx8_frac + 8, test_7_out_data, 
    sizeof(test_7_out_data) / sizeof(test_7_out_data[0]));

tensor_quantizer test_7_out_sa8(test_7_out_tsr_sa8, test_7_out_sa_dim, test_7_out_data,
    sizeof(test_7_out_data) / sizeof(test_7_out_data[0]), &test_7_out_scale, 1, 
    &test_7_out_zero_point, 1, test_7_out_scales_frac, 1);


static const float test_8_out_data[] = {
    0.198123f, -0.13067f, 0.55243f, 0.178868f, -0.793809f, -1.f, 0.251102f, 1.f, 0.0838f, -0.377189f, 0.550262f, 
    0.580767f, 0.348409f, -0.934795f, 1.f, 1.f, -0.388695f, 0.913123f, -0.016406f, -0.374269f, 0.090984f, 0.08972f, 1.f, 
    1.f, 0.08128f, 0.426455f, 0.270948f, -0.121359f, -0.326392f, -0.298172f, 0.509185f, -0.362075f, 0.643565f, -0.14642f, 
    0.617233f, -0.762947f, -0.014466f, -1.f, 0.838984f, -1.f, 0.212098f, 0.026127f, 0.423236f, 0.165756f, 0.009827f, 
    0.200701f, -0.641129f, -1.f, -0.270156f, -0.517608f, 1.f, -0.179054f, 0.504732f, -0.596899f, 0.744064f, -1.f, 1.f, 
    -1.f, 1.f, -0.070012f, -0.161682f, 0.925545f, 1.f, -1.f, 0.429767f, 0.298243f, -0.085608f, -0.543035f, 0.364865f, 
    0.511966f, 1.f, -1.f, -0.601628f, -0.164895f, 0.230949f, 0.203222f, 0.472913f, -1.f, 0.146941f, 0.194164f, -1.f, 
    -1.f, 0.009781f, 0.131992f, -0.616455f, 1.f, 0.209566f, -0.768585f, 0.08377f, 1.f, -0.497608f, -1.f, 0.068681f, 
    -0.402662f, -1.f, -0.834601f, 0.49266f, 0.539663f, 0.824787f, -1.f, 0.675395f, -0.327114f, 0.235589f, -0.001013f, 
    -0.078472f, 0.715988f, 0.842383f, -1.f, 0.942298f, -0.358432f, 0.289818f, -0.54689f, 0.316335f, 0.007191f, 
    -0.123326f, -0.385413f, 0.154857f, 0.287871f, 0.084229f, -0.097181f, -0.061465f, 0.154977f, 0.252421f, -0.512955f, 
    1.f, -1.f, 0.339268f, -0.271578f, 1.f, -0.627714f, -0.376263f, -0.772987f, -0.053542f, -0.100525f, 0.147329f, 
    -0.320745f, -0.15804f, 0.437618f, -0.997956f, 1.f, 0.93889f, 0.779327f, -0.394837f, -0.053527f
};

static const float test_8_out_scale = 0.0078431377f;
static const float test_8_out_zero_point = 0.f;
static const int8_t test_8_out_scales_frac[] = {21};
static const int test_8_out_sa_dim = -1;

static const int test_8_out_fx8_frac = 7;


#define TEST_8_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {6, 6, 4}, \
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

tensor_quantizer test_8_out_fx16(test_8_out_tsr_fx16, test_8_out_fx8_frac + 8, test_8_out_data, 
    sizeof(test_8_out_data) / sizeof(test_8_out_data[0]));

tensor_quantizer test_8_out_sa8(test_8_out_tsr_sa8, test_8_out_sa_dim, test_8_out_data, 
    sizeof(test_8_out_data) / sizeof(test_8_out_data[0]), &test_8_out_scale, 1, 
    &test_8_out_zero_point, 1, test_8_out_scales_frac, 1);


static const float test_9_out_data[] = {
    0.f, 0.324315f, 4.774007f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 5.055168f, 0.f, 0.f, 1.372424f, 0.f, 2.932685f, 
    0.f, 0.f, 4.834651f, 1.318798f, 2.162142f, 0.f, 3.2711f, 0.f, 0.f, 3.722245f, 0.f, 4.923825f, 0.f, 2.163748f, 
    7.245541f, 2.559371f, 6.71715f, 0.f, 0.f, 0.f, 0.f, 0.662758f, 0.f, 0.f, 5.279433f, 0.f, 0.511729f, 0.f, 
    5.043289f, 0.f, 5.07621f, 7.621825f, 0.f, 5.048098f, 1.92084f, 0.f, 0.f, 5.564611f, 6.154945f, 0.f, 0.f, 
    5.655184f, 3.714609f, 0.f, 0.f, 0.f, 0.575388f, 4.343388f, 0.f, 5.186674f, 6.102503f, 8.070898f, 0.f, 0.f, 
    0.f, 4.455057f, 0.f, 8.433863f, 0.f, 0.f, 0.f, 3.27062f, 0.f, 0.f, 0.f, 1.728364f, 0.f, 0.899699f, 
    2.545474f, 0.f, 0.f, 0.f, 0.f, 3.967861f, 1.114252f, 0.f, 0.f, 3.195547f, 7.244598f, 0.f, 2.207935f, 
    4.962389f, 0.f, 3.268717f, 2.903937f, 0.f, 0.f, 0.f, 0.f, 0.f, 4.132554f, 0.f, 3.54799f, 0.f, 0.f, 0.f, 
    0.f, 0.f, 0.f, 5.42614f, 1.606486f, 1.111508f, 0.f, 4.114276f, 6.014443f, 6.580969f, 4.895285f, 0.f, 
    1.059912f, 0.f, 9.997729f, 0.f, 1.011876f, 3.070468f, 0.f, 0.f, 5.157227f, 0.f, 8.533406f, 1.775449f, 
    2.435084f, 5.740183f, 0.f, 3.624646f, 0.f, 0.f, 4.493617f, 1.499491f, 5.084453f, 3.35638f, 0.f, 0.2235f
};

static const float test_9_out_scale = 0.0392067805f;
static const float test_9_out_zero_point = 4.9988646507f;
static const int8_t test_9_out_scales_frac[] = {19};
static const int test_9_out_sa_dim = -1;

static const int test_9_out_fx8_frac = 3;

#define TEST_9_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {7, 7, 3}, \
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

tensor_quantizer test_9_out_fx16(test_9_out_tsr_fx16, test_9_out_fx8_frac + 8, test_9_out_data, 
    sizeof(test_9_out_data) / sizeof(test_9_out_data[0]));

tensor_quantizer test_9_out_sa8(test_9_out_tsr_sa8, test_9_out_sa_dim, test_9_out_data, 
    sizeof(test_9_out_data) / sizeof(test_9_out_data[0]), &test_9_out_scale, 1, 
    &test_9_out_zero_point, 1, test_9_out_scales_frac, 1);


static const float test_10_out_data[] = {
    -7.848892f, -6.152408f, 4.26403f, -9.462627f, 1.776017f, 0.54884f, -13.467508f, -9.181556f, 16.437769f, 6.921103f, 
    2.89089f, -6.264424f, -0.125646f, 1.815216f, -1.046473f, 1.279558f, 3.912849f, -7.92703f, 8.664543f, -3.704524f, 
    2.274545f, -11.504289f, -1.276218f, -7.811656f, 14.789411f, -8.726789f, -0.727665f 
};

static const float test_10_out_scale = 0.1172755957f;
static const float test_10_out_zero_point = 1.4851303101f;
static const int8_t test_10_out_scales_frac[] = {18};
static const int test_10_out_sa_dim = -1;

static const int test_10_out_fx8_frac = 2;

#define TEST_10_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 3, 3}, \
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

tensor_quantizer test_10_out_fx16(test_10_out_tsr_fx16, test_10_out_fx8_frac + 8, test_10_out_data, 
    sizeof(test_10_out_data) / sizeof(test_10_out_data[0]));
    
tensor_quantizer test_10_out_sa8(test_10_out_tsr_sa8, test_10_out_sa_dim, test_10_out_data, 
    sizeof(test_10_out_data) / sizeof(test_10_out_data[0]), &test_10_out_scale, 1, 
    &test_10_out_zero_point, 1, test_10_out_scales_frac, 1);


// Tests configuration structures
//========================================

const mli_conv2d_cfg test_1_cfg = {
    /* .relu.type = */ MLI_RELU_NONE,
    /* .stride_width = */ 2,
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
    /* .stride_width = */ 1,
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
    /* .dilation_width = */ 1,
    /* .dilation_height = */ 1
};

const mli_conv2d_cfg test_4_cfg = {
    /* .relu.type = */ MLI_RELU_GEN,
    /* .stride_width = */ 1,
    /* .stride_height = */ 1,
    /* .padding_left = */ 1,
    /* .padding_right = */ 1,
    /* .padding_top = */ 1,
    /* .padding_bottom = */ 1,
    /* .dilation_width = */ 2,
    /* .dilation_height = */ 2
};

const mli_conv2d_cfg test_5_cfg = {
    /* .relu.type = */ MLI_RELU_NONE,
    /* .stride_width = */ 1,
    /* .stride_height = */ 2,
    /* .padding_left = */ 1,
    /* .padding_right = */ 1,
    /* .padding_top = */ 1,
    /* .padding_bottom = */ 1,
    /* .dilation_width = */ 2,
    /* .dilation_height = */ 2
};

const mli_conv2d_cfg test_6_cfg = {
    /* .relu.type = */ MLI_RELU_6,
    /* .stride_width = */ 3,
    /* .stride_height = */ 3,
    /* .padding_left = */ 1,
    /* .padding_right = */ 2,
    /* .padding_top = */ 1,
    /* .padding_bottom = */ 1,
    /* .dilation_width = */ 1,
    /* .dilation_height = */ 1
};

const mli_conv2d_cfg test_7_cfg = {
    /* .relu.type = */ MLI_RELU_GEN,
    /* .stride_width = */ 3,
    /* .stride_height = */ 3,
    /* .padding_left = */ 1,
    /* .padding_right = */ 1,
    /* .padding_top = */ 1,
    /* .padding_bottom = */ 0,
    /* .dilation_width = */ 1,
    /* .dilation_height = */ 1
};

const mli_conv2d_cfg test_8_cfg = {
    /* .relu.type = */ MLI_RELU_1,
    /* .stride_width = */ 1,
    /* .stride_height = */ 1,
    /* .padding_left = */ 2,
    /* .padding_right = */ 2,
    /* .padding_top = */ 2,
    /* .padding_bottom = */ 2,
    /* .dilation_width = */ 2,
    /* .dilation_height = */ 2
};

const mli_conv2d_cfg test_9_cfg = {
    /* .relu.type = */ MLI_RELU_GEN,
    /* .stride_width = */ 1,
    /* .stride_height = */ 1,
    /* .padding_left = */ 2,
    /* .padding_right = */ 2,
    /* .padding_top = */ 2,
    /* .padding_bottom = */ 2,
    /* .dilation_width = */ 2,
    /* .dilation_height = */ 2
};

const mli_conv2d_cfg test_10_cfg = {
    /* .relu.type = */ MLI_RELU_NONE,
    /* .stride_width = */ 1,
    /* .stride_height = */ 1,
    /* .padding_left = */ 2,
    /* .padding_right = */ 2,
    /* .padding_top = */ 2,
    /* .padding_bottom = */ 2,
    /* .dilation_width = */ 2,
    /* .dilation_height = */ 2
};
