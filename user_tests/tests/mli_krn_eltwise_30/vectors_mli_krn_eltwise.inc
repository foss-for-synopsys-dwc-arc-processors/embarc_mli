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
extern mli::tst::tensor_quantizer test_9_out_fx16;
extern mli::tst::tensor_quantizer test_9_out_sa8;
extern mli::tst::tensor_quantizer test_10_out_fx16;
extern mli::tst::tensor_quantizer test_10_out_sa8;
extern mli::tst::tensor_quantizer test_11_out_fx16;
extern mli::tst::tensor_quantizer test_11_out_sa8;


static const float input_1_data[] = {
    -0.154767f, -0.118563f,  0.226941f,  0.283774f,  0.245932f, -0.068952f,
    -0.315049f, -0.299830f,  0.174177f, -0.019029f, -0.145222f, -0.314864f,
     0.833865f,  0.273061f, -0.087248f,  0.279299f,  0.345489f, -0.496154f,
    -0.355891f,  0.094294f,  0.336745f, -0.185674f, -0.390024f, -0.247575f,
    -0.149485f,  0.289462f, -0.039426f,  0.001710f, -0.414436f, -0.223744f,
    -0.038971f,  0.558261f,  0.146462f,  0.616282f,  0.389316f, -0.066083f,
     0.159876f, -0.161529f, -0.027262f,  0.108276f,  0.206893f,  0.040067f,
    -0.341104f,  0.107336f, -0.268049f, -0.111010f, -0.211149f,  0.129771f,
    -0.124119f,  0.009234f,  0.207962f,  0.223424f,  0.354668f, -0.278795f,
     0.077915f, -0.263391f,  0.224826f,  0.142887f, -0.162642f,  0.427009f,
     0.477671f,  0.577053f,  0.201815f,  0.126930f, -0.637734f,  0.112142f,
     0.263059f,  0.216071f, -0.257113f,  0.173853f, -0.242127f,  0.072917f,
     0.270399f, -0.039309f, -0.029376f,  0.338695f, -0.285461f,  0.244499f,
     0.024809f, -0.161522f, -0.085716f,  0.488852f,  0.363817f, -0.131824f,
    -0.440843f, -0.062320f, -0.165981f, -0.065595f, -0.100688f, -0.366461f,
     0.773151f,  0.036817f, -0.343643f,  0.148965f,  0.102216f,  0.054143f,
    -0.547653f,  0.383607f,  0.184353f,  0.342847f,  0.235109f, -0.164047f,
    -0.415784f, -0.687819f, -0.430796f,  0.205749f, -0.150957f, -0.495271f,
     0.319163f,  0.208912f,  0.136296f, -0.255625f, -0.060121f,  0.119739f,
    -0.488017f, -0.496901f, -0.066240f,  0.654600f,  0.363814f, -0.128497f,
    -0.229125f,  0.527085f, -0.109751f,  0.237886f,  0.187139f,  0.091283f,
     0.363140f, -0.575530f};

static const float input_1_scale = 0.0059673889f;
static const float input_1_zero_point = 0.0730233788f;
static const int8_t input_1_scales_frac[] = {22};
static const int input_1_sa_dim = -1;

static const int input_1_fx8_frac = 7;

#define INPUT_1_TSR_SHARED_DESCR \
  /* .data = */ { 0 }            \
  , /* .mem_stride = */ {0}, /* .shape = */ {8, 16}, /* .rank =  */ 2

static const mli_tensor input_1_tsr_fx16 = {INPUT_1_TSR_SHARED_DESCR,

                                            /* .el_type = */ MLI_EL_FX_16,
                                            /* .el_params = */ {0}};

static const mli_tensor input_1_tsr_sa8 = {INPUT_1_TSR_SHARED_DESCR,

                                           /* .el_type = */ MLI_EL_SA_8,
                                           /* .el_params = */ {0}};

static const float input_2_data[] = {
    -0.091750f, -0.051321f,  0.282265f, -0.109082f, -0.022779f,  0.045961f,
     0.149066f,  0.200496f,  0.022000f,  0.012408f,  0.008006f, -0.120819f,
     0.263768f,  0.237886f, -0.145260f,  0.157669f,  0.156119f,  0.169200f,
     0.019774f,  0.092650f, -0.078201f, -0.001856f,  0.109068f, -0.020643f,
     0.049847f, -0.265108f,  0.118578f, -0.024900f, -0.163252f, -0.055075f,
    -0.014320f, -0.054766f, -0.158402f,  0.176820f, -0.067452f,  0.017345f,
    -0.041596f,  0.134407f, -0.303202f,  0.256677f,  0.107938f,  0.062214f,
    -0.258394f, -0.018200f,  0.140401f,  0.079273f, -0.112908f, -0.315468f,
    -0.157668f, -0.127504f, -0.133055f, -0.305548f,  0.086594f, -0.038669f,
     0.233342f, -0.146538f,  0.020152f, -0.012567f,  0.111566f, -0.306916f,
    -0.245183f,  0.183883f, -0.029849f, -0.298754f, -0.191261f,  0.087108f,
    -0.025554f, -0.241929f, -0.111014f,  0.085338f,  0.005598f, -0.000472f,
     0.014796f, -0.208077f, -0.042013f, -0.011296f,  0.126863f,  0.096162f,
    -0.179204f, -0.354448f, -0.135553f,  0.046803f, -0.033813f,  0.215600f,
    -0.038830f,  0.105875f, -0.242753f, -0.222373f,  0.064287f,  0.034292f,
    -0.187089f,  0.156937f,  0.012316f,  0.113855f,  0.225760f,  0.108602f,
     0.132049f,  0.060413f, -0.096945f,  0.160177f, -0.249445f,  0.066459f,
     0.044761f,  0.010666f, -0.302838f, -0.025816f, -0.206852f, -0.049250f,
     0.005334f,  0.010188f,  0.027012f, -0.109258f, -0.098318f,  0.015853f,
     0.035378f,  0.228086f,  0.186415f,  0.074828f,  0.112086f, -0.218000f,
    -0.071108f, -0.178859f, -0.292142f,  0.049861f, -0.048190f, -0.128079f,
     0.161887f,  0.059452f};

static const float input_2_scale = 0.0024969154f;
static const float input_2_zero_point = -0.0360917449f;
static const int8_t input_2_scales_frac[] = {23};
static const int input_2_sa_dim = -1;

static const int input_2_fx8_frac = 8;

#define INPUT_2_TSR_SHARED_DESCR \
  /* .data = */ { 0 }            \
  , /* .mem_stride = */ {0}, /* .shape = */ {8, 16}, /* .rank =  */ 2

static const mli_tensor input_2_tsr_fx16 = {INPUT_2_TSR_SHARED_DESCR,

                                            /* .el_type = */ MLI_EL_FX_16,
                                            /* .el_params = */ {0}};

static const mli_tensor input_2_tsr_sa8 = {INPUT_2_TSR_SHARED_DESCR,

                                           /* .el_type = */ MLI_EL_SA_8,
                                           /* .el_params = */ {0}};

static const float input_3_data[] = {0.500000f};

static const float input_3_scale = 1.0f;
static const float input_3_zero_point = 0.000000000f;
static const int8_t input_3_scales_frac[] = {23};
static const int input_3_sa_dim = -1;

static const int input_3_fx8_frac = 8;

#define INPUT_3_TSR_SHARED_DESCR \
  /* .data = */ { 0 }            \
  , /* .mem_stride = */ {0}, /* .shape = */ {0}, /* .rank =  */ 0

static const mli_tensor input_3_tsr_fx16 = {INPUT_3_TSR_SHARED_DESCR,

                                            /* .el_type = */ MLI_EL_FX_16,
                                            /* .el_params = */ {0}};

static const mli_tensor input_3_tsr_sa8 = {INPUT_3_TSR_SHARED_DESCR,

                                           /* .el_type = */ MLI_EL_SA_8,
                                           /* .el_params = */ {0}};


static const float test_1_out_data[] = {
    -0.246516f, -0.169883f,  0.509206f,  0.174692f,  0.223154f, -0.022991f,
    -0.165983f, -0.099335f,  0.196178f, -0.006622f, -0.137216f, -0.435684f,
     1.097633f,  0.510947f, -0.232507f,  0.436968f,  0.501608f, -0.326954f,
    -0.336117f,  0.186945f,  0.258543f, -0.187530f, -0.280956f, -0.268218f,
    -0.099638f,  0.024354f,  0.079152f, -0.023190f, -0.577688f, -0.278819f,
    -0.053291f,  0.503495f, -0.011941f,  0.793102f,  0.321864f, -0.048738f,
     0.118280f, -0.027123f, -0.330464f,  0.364953f,  0.314831f,  0.102281f,
    -0.599498f,  0.089137f, -0.127648f, -0.031737f, -0.324057f, -0.185697f,
    -0.281787f, -0.118270f,  0.074907f, -0.082124f,  0.441261f, -0.317464f,
     0.311257f, -0.409929f,  0.244978f,  0.130320f, -0.051076f,  0.120092f,
     0.232488f,  0.760936f,  0.171966f, -0.171824f, -0.828994f,  0.199250f,
     0.237505f, -0.025858f, -0.368128f,  0.259190f, -0.236529f,  0.072445f,
     0.285196f, -0.247386f, -0.071389f,  0.327399f, -0.158598f,  0.340661f,
    -0.154395f, -0.515971f, -0.221269f,  0.535655f,  0.330004f,  0.083776f,
    -0.479673f,  0.043555f, -0.408734f, -0.287969f, -0.036400f, -0.332169f,
     0.586062f,  0.193754f, -0.331327f,  0.262820f,  0.327976f,  0.162745f,
    -0.415604f,  0.444020f,  0.087408f,  0.503025f, -0.014336f, -0.097588f,
    -0.371023f, -0.677152f, -0.733634f,  0.179933f, -0.357809f, -0.544521f,
     0.324496f,  0.219100f,  0.163309f, -0.364883f, -0.158440f,  0.135592f,
    -0.452639f, -0.268815f,  0.120175f,  0.729428f,  0.475900f, -0.346497f,
    -0.300233f,  0.348225f, -0.401892f,  0.287747f,  0.138949f, -0.036797f,
     0.525026f, -0.516078f};

static const float test_1_out_scale = 0.0075554014f;
static const float test_1_out_zero_point = 0.1343196630f;
static const int8_t test_1_out_scales_frac[] = {22};
static const int test_1_out_sa_dim = -1;

static const int test_1_out_fx8_frac = 6;

#define TEST_1_OUT_TSR_SHARED_DESCR \
  /* .data = */ { 0 }               \
  , /* .mem_stride = */ {0}, /* .shape = */ {8, 16}, /* .rank =  */ 2

static const mli_tensor test_1_out_tsr_fx16 = {TEST_1_OUT_TSR_SHARED_DESCR,

                                               /* .el_type = */ MLI_EL_FX_16,
                                               /* .el_params = */ {0}};

static const mli_tensor test_1_out_tsr_sa8 = {TEST_1_OUT_TSR_SHARED_DESCR,

                                              /* .el_type = */ MLI_EL_SA_8,
                                              /* .el_params = */ {0}};

static const float test_2_out_data[] = {
    0.408250f, 0.448679f, 0.782265f, 0.390918f, 0.477221f, 0.545961f, 0.649066f,
    0.700496f, 0.522000f, 0.512408f, 0.508006f, 0.379181f, 0.763768f, 0.737886f,
    0.354740f, 0.657669f, 0.656119f, 0.669200f, 0.519774f, 0.592650f, 0.421799f,
    0.498144f, 0.609068f, 0.479357f, 0.549847f, 0.234892f, 0.618578f, 0.475100f,
    0.336748f, 0.444925f, 0.485680f, 0.445234f, 0.341598f, 0.676820f, 0.432548f,
    0.517345f, 0.458404f, 0.634407f, 0.196798f, 0.756677f, 0.607938f, 0.562214f,
    0.241606f, 0.481800f, 0.640401f, 0.579273f, 0.387092f, 0.184532f, 0.342332f,
    0.372496f, 0.366945f, 0.194452f, 0.586594f, 0.461331f, 0.733342f, 0.353462f,
    0.520152f, 0.487433f, 0.611566f, 0.193084f, 0.254817f, 0.683883f, 0.470151f,
    0.201246f, 0.308739f, 0.587108f, 0.474446f, 0.258071f, 0.388986f, 0.585338f,
    0.505598f, 0.499528f, 0.514796f, 0.291923f, 0.457987f, 0.488704f, 0.626863f,
    0.596162f, 0.320796f, 0.145552f, 0.364447f, 0.546803f, 0.466187f, 0.715600f,
    0.461170f, 0.605875f, 0.257247f, 0.277627f, 0.564287f, 0.534292f, 0.312911f,
    0.656937f, 0.512316f, 0.613855f, 0.725760f, 0.608602f, 0.632050f, 0.560413f,
    0.403055f, 0.660177f, 0.250555f, 0.566459f, 0.544761f, 0.510666f, 0.197162f,
    0.474184f, 0.293148f, 0.450750f, 0.505334f, 0.510188f, 0.527012f, 0.390742f,
    0.401682f, 0.515853f, 0.535378f, 0.728086f, 0.686415f, 0.574828f, 0.612086f,
    0.282000f, 0.428892f, 0.321141f, 0.207858f, 0.549861f, 0.451810f, 0.371921f,
    0.661887f, 0.559452f};

static const float test_2_out_scale = 0.0036528209f;
static const float test_2_out_zero_point = 0.4639082551f;
static const int8_t test_2_out_scales_frac[] = {23};
static const int test_2_out_sa_dim = -1;

static const int test_2_out_fx8_frac = 7;

#define TEST_2_OUT_TSR_SHARED_DESCR \
  /* .data = */ { 0 }               \
  , /* .mem_stride = */ {0}, /* .shape = */ {8, 16}, /* .rank =  */ 2

static const mli_tensor test_2_out_tsr_fx16 = {TEST_2_OUT_TSR_SHARED_DESCR,

                                               /* .el_type = */ MLI_EL_FX_16,
                                               /* .el_params = */ {0}};

static const mli_tensor test_2_out_tsr_sa8 = {TEST_2_OUT_TSR_SHARED_DESCR,

                                              /* .el_type = */ MLI_EL_SA_8,
                                              /* .el_params = */ {0}};

static const float test_3_out_data[] = {
    -0.063017f, -0.067242f, -0.055324f,  0.392856f,  0.268711f, -0.114913f,
    -0.464115f, -0.500326f,  0.152177f, -0.031437f, -0.153227f, -0.194045f,
     0.570098f,  0.035174f,  0.058012f,  0.121630f,  0.189370f, -0.665354f,
    -0.375664f,  0.001644f,  0.414946f, -0.183817f, -0.499092f, -0.226931f,
    -0.199332f,  0.554571f, -0.158004f,  0.026610f, -0.251184f, -0.168670f,
    -0.024651f,  0.613028f,  0.304864f,  0.439461f,  0.456767f, -0.083428f,
     0.201472f, -0.295936f,  0.275940f, -0.148401f,  0.098955f, -0.022147f,
    -0.082711f,  0.125536f, -0.408450f, -0.190283f, -0.098241f,  0.445239f,
     0.033548f,  0.136739f,  0.341017f,  0.528973f,  0.268074f, -0.240127f,
    -0.155427f, -0.116852f,  0.204675f,  0.155454f, -0.274208f,  0.733925f,
     0.722854f,  0.393171f,  0.231664f,  0.425684f, -0.446473f,  0.025035f,
     0.288613f,  0.458000f, -0.146099f,  0.088515f, -0.247725f,  0.073389f,
     0.255603f,  0.168768f,  0.012636f,  0.349992f, -0.412325f,  0.148337f,
     0.204013f,  0.192926f,  0.049838f,  0.442049f,  0.397630f, -0.347423f,
    -0.402013f, -0.168195f,  0.076772f,  0.156778f, -0.164975f, -0.400754f,
     0.960240f, -0.120120f, -0.355959f,  0.035110f, -0.123544f, -0.054459f,
    -0.679703f,  0.323194f,  0.281298f,  0.182670f,  0.484554f, -0.230505f,
    -0.460545f, -0.698485f, -0.127958f,  0.231564f,  0.055895f, -0.446021f,
     0.313829f,  0.198724f,  0.109284f, -0.146367f,  0.038197f,  0.103886f,
    -0.523395f, -0.724987f, -0.252655f,  0.579772f,  0.251727f,  0.089502f,
    -0.158017f,  0.705944f,  0.182391f,  0.188025f,  0.235330f,  0.219362f,
     0.201253f, -0.634982f};

static const float test_3_out_scale = 0.0066087358f;
static const float test_3_out_zero_point = 0.1176264882f;
static const int8_t test_3_out_scales_frac[] = {22};
static const int test_3_out_sa_dim = -1;

static const int test_3_out_fx8_frac = 7;

#define TEST_3_OUT_TSR_SHARED_DESCR \
  /* .data = */ { 0 }               \
  , /* .mem_stride = */ {0}, /* .shape = */ {8, 16}, /* .rank =  */ 2

static const mli_tensor test_3_out_tsr_fx16 = {TEST_3_OUT_TSR_SHARED_DESCR,

                                               /* .el_type = */ MLI_EL_FX_16,
                                               /* .el_params = */ {0}};

static const mli_tensor test_3_out_tsr_sa8 = {TEST_3_OUT_TSR_SHARED_DESCR,

                                              /* .el_type = */ MLI_EL_SA_8,
                                              /* .el_params = */ {0}};

static const float test_4_out_data[] = {
    -0.654766f, -0.618563f, -0.273059f, -0.216226f, -0.254068f, -0.568952f,
    -0.815049f, -0.799830f, -0.325823f, -0.519029f, -0.645222f, -0.814864f,
     0.333865f, -0.226939f, -0.587248f, -0.220701f, -0.154511f, -0.996154f,
    -0.855891f, -0.405706f, -0.163255f, -0.685674f, -0.890024f, -0.747575f,
    -0.649485f, -0.210538f, -0.539426f, -0.498290f, -0.914436f, -0.723744f,
    -0.538971f,  0.058261f, -0.353538f,  0.116282f, -0.110684f, -0.566083f,
    -0.340124f, -0.661529f, -0.527262f, -0.391724f, -0.293107f, -0.459933f,
    -0.841104f, -0.392664f, -0.768049f, -0.611010f, -0.711149f, -0.370229f,
    -0.624119f, -0.490766f, -0.292038f, -0.276576f, -0.145332f, -0.778795f,
    -0.422085f, -0.763391f, -0.275174f, -0.357113f, -0.662642f, -0.072991f,
    -0.022329f,  0.077053f, -0.298185f, -0.373070f, -1.137733f, -0.387858f,
    -0.236941f, -0.283929f, -0.757113f, -0.326147f, -0.742127f, -0.427083f,
    -0.229601f, -0.539309f, -0.529376f, -0.161305f, -0.785461f, -0.255501f,
    -0.475191f, -0.661522f, -0.585716f, -0.011148f, -0.136183f, -0.631824f,
    -0.940843f, -0.562320f, -0.665981f, -0.565595f, -0.600688f, -0.866461f,
     0.273151f, -0.463183f, -0.843643f, -0.351035f, -0.397784f, -0.445857f,
    -1.047653f, -0.116393f, -0.315647f, -0.157153f, -0.264891f, -0.664047f,
    -0.915784f, -1.187819f, -0.930796f, -0.294251f, -0.650957f, -0.995271f,
    -0.180837f, -0.291088f, -0.363704f, -0.755625f, -0.560121f, -0.380261f,
    -0.988017f, -0.996901f, -0.566240f,  0.154600f, -0.136186f, -0.628497f,
    -0.729125f,  0.027085f, -0.609751f, -0.262114f, -0.312861f, -0.408717f,
    -0.136860f, -1.075530f};

static const float test_4_out_scale = 0.0059673889f;
static const float test_4_out_zero_point = -0.4269766212f;
static const int8_t test_4_out_scales_frac[] = {22};
static const int test_4_out_sa_dim = -1;

static const int test_4_out_fx8_frac = 6;

#define TEST_4_OUT_TSR_SHARED_DESCR \
  /* .data = */ { 0 }               \
  , /* .mem_stride = */ {0}, /* .shape = */ {8, 16}, /* .rank =  */ 2

static const mli_tensor test_4_out_tsr_fx16 = {TEST_4_OUT_TSR_SHARED_DESCR,

                                               /* .el_type = */ MLI_EL_FX_16,
                                               /* .el_params = */ {0}};

static const mli_tensor test_4_out_tsr_sa8 = {TEST_4_OUT_TSR_SHARED_DESCR,

                                              /* .el_type = */ MLI_EL_SA_8,
                                              /* .el_params = */ {0}};

static const float test_5_out_data[] = {
    0.591750f, 0.551321f, 0.217735f, 0.609082f, 0.522779f, 0.454039f, 0.350934f,
    0.299504f, 0.478000f, 0.487592f, 0.491994f, 0.620819f, 0.236232f, 0.262114f,
    0.645260f, 0.342331f, 0.343881f, 0.330800f, 0.480226f, 0.407350f, 0.578201f,
    0.501856f, 0.390932f, 0.520643f, 0.450153f, 0.765108f, 0.381422f, 0.524900f,
    0.663252f, 0.555075f, 0.514320f, 0.554766f, 0.658402f, 0.323180f, 0.567452f,
    0.482655f, 0.541596f, 0.365593f, 0.803202f, 0.243323f, 0.392062f, 0.437786f,
    0.758394f, 0.518200f, 0.359599f, 0.420727f, 0.612908f, 0.815468f, 0.657668f,
    0.627504f, 0.633055f, 0.805548f, 0.413406f, 0.538669f, 0.266658f, 0.646538f,
    0.479848f, 0.512567f, 0.388434f, 0.806916f, 0.745183f, 0.316117f, 0.529849f,
    0.798754f, 0.691261f, 0.412892f, 0.525554f, 0.741929f, 0.611014f, 0.414662f,
    0.494402f, 0.500472f, 0.485204f, 0.708077f, 0.542013f, 0.511296f, 0.373137f,
    0.403838f, 0.679204f, 0.854448f, 0.635553f, 0.453197f, 0.533813f, 0.284400f,
    0.538830f, 0.394125f, 0.742753f, 0.722373f, 0.435713f, 0.465708f, 0.687089f,
    0.343063f, 0.487684f, 0.386145f, 0.274240f, 0.391398f, 0.367950f, 0.439587f,
    0.596945f, 0.339823f, 0.749445f, 0.433541f, 0.455239f, 0.489334f, 0.802838f,
    0.525816f, 0.706852f, 0.549250f, 0.494666f, 0.489812f, 0.472988f, 0.609258f,
    0.598318f, 0.484147f, 0.464622f, 0.271914f, 0.313585f, 0.425172f, 0.387914f,
    0.718000f, 0.571108f, 0.678859f, 0.792142f, 0.450139f, 0.548190f, 0.628079f,
    0.338113f, 0.440548f};

static const float test_5_out_scale = 0.00422119484f;
static const float test_5_out_zero_point = 0.5360917449f;
static const int8_t test_5_out_scales_frac[] = {22};
static const int test_5_out_sa_dim = -1;

static const int test_5_out_fx8_frac = 7;

#define TEST_5_OUT_TSR_SHARED_DESCR \
  /* .data = */ { 0 }               \
  , /* .mem_stride = */ {0}, /* .shape = */ {8, 16}, /* .rank =  */ 2

static const mli_tensor test_5_out_tsr_fx16 = {TEST_5_OUT_TSR_SHARED_DESCR,

                                               /* .el_type = */ MLI_EL_FX_16,
                                               /* .el_params = */ {0}};

static const mli_tensor test_5_out_tsr_sa8 = {TEST_5_OUT_TSR_SHARED_DESCR,

                                              /* .el_type = */ MLI_EL_SA_8,
                                              /* .el_params = */ {0}};


static const float test_6_out_data[] = {
     0.014200f,  0.006085f,  0.064058f, -0.030955f, -0.005602f, -0.003169f,
    -0.046963f, -0.060115f,  0.003832f, -0.000236f, -0.001163f,  0.038042f,
     0.219947f,  0.064957f,  0.012674f,  0.044037f,  0.053937f, -0.083949f,
    -0.007037f,  0.008736f, -0.026334f,  0.000345f, -0.042539f,  0.005111f,
    -0.007451f, -0.076739f, -0.004675f, -0.000043f,  0.067658f,  0.012323f,
     0.000558f, -0.030574f, -0.023200f,  0.108971f, -0.026260f, -0.001146f,
    -0.006650f, -0.021711f,  0.008266f,  0.027792f,  0.022332f,  0.002493f,
     0.088139f, -0.001953f, -0.037634f, -0.008800f,  0.023840f, -0.040939f,
     0.019570f, -0.001177f, -0.027670f, -0.068267f,  0.030712f,  0.010781f,
     0.018181f,  0.038597f,  0.004531f, -0.001796f, -0.018145f, -0.131056f,
    -0.117117f,  0.106110f, -0.006024f, -0.037921f,  0.121973f,  0.009768f,
    -0.006722f, -0.052274f,  0.028543f,  0.014836f, -0.001355f, -0.000034f,
     0.004001f,  0.008179f,  0.001234f, -0.003826f, -0.036215f,  0.023512f,
    -0.004446f,  0.057251f,  0.011619f,  0.022880f, -0.012302f, -0.028421f,
     0.017118f, -0.006598f,  0.040292f,  0.014587f, -0.006473f, -0.012567f,
    -0.144648f,  0.005778f, -0.004232f,  0.016960f,  0.023076f,  0.005880f,
    -0.072317f,  0.023175f, -0.017872f,  0.054916f, -0.058647f, -0.010902f,
    -0.018611f, -0.007337f,  0.130461f, -0.005312f,  0.031226f,  0.024392f,
     0.001702f,  0.002128f,  0.003682f,  0.027929f,  0.005911f,  0.001898f,
    -0.017265f, -0.113336f, -0.012348f,  0.048983f,  0.040779f,  0.028012f,
     0.016293f, -0.094274f,  0.032063f,  0.011861f, -0.009018f, -0.011691f,
     0.058788f, -0.034216f};

static const float test_6_out_scale = 0.0014297848f;
static const float test_6_out_zero_point = 0.0376494378f;
static const int8_t test_6_out_scales_frac[] = {24};
static const int test_6_out_sa_dim = -1;

static const int test_6_out_fx8_frac = 9;

#define TEST_6_OUT_TSR_SHARED_DESCR \
  /* .data = */ { 0 }               \
  , /* .mem_stride = */ {0}, /* .shape = */ {8, 16}, /* .rank =  */ 2

static const mli_tensor test_6_out_tsr_fx16 = {TEST_6_OUT_TSR_SHARED_DESCR,

                                               /* .el_type = */ MLI_EL_FX_16,
                                               /* .el_params = */ {0}};

static const mli_tensor test_6_out_tsr_sa8 = {TEST_6_OUT_TSR_SHARED_DESCR,

                                              /* .el_type = */ MLI_EL_SA_8,
                                              /* .el_params = */ {0}};

static const float test_7_out_data[] = {
    -0.077383f, -0.059281f,  0.113471f,  0.141887f,  0.122966f, -0.034476f,
    -0.157525f, -0.149915f,  0.087089f, -0.009515f, -0.072611f, -0.157432f,
     0.416933f,  0.136530f, -0.043624f,  0.139649f,  0.172744f, -0.248077f,
    -0.177945f,  0.047147f,  0.168372f, -0.092837f, -0.195012f, -0.123787f,
    -0.074742f,  0.144731f, -0.019713f,  0.000855f, -0.207218f, -0.111872f,
    -0.019485f,  0.279131f,  0.073231f,  0.308141f,  0.194658f, -0.033042f,
     0.079938f, -0.080765f, -0.013631f,  0.054138f,  0.103447f,  0.020033f,
    -0.170552f,  0.053668f, -0.134025f, -0.055505f, -0.105575f,  0.064885f,
    -0.062060f,  0.004617f,  0.103981f,  0.111712f,  0.177334f, -0.139398f,
     0.038957f, -0.131695f,  0.112413f,  0.071444f, -0.081321f,  0.213504f,
     0.238836f,  0.288527f,  0.100907f,  0.063465f, -0.318867f,  0.056071f,
     0.131529f,  0.108036f, -0.128557f,  0.086926f, -0.121064f,  0.036459f,
     0.135200f, -0.019654f, -0.014688f,  0.169348f, -0.142731f,  0.122249f,
     0.012404f, -0.080761f, -0.042858f,  0.244426f,  0.181909f, -0.065912f,
    -0.220421f, -0.031160f, -0.082991f, -0.032798f, -0.050344f, -0.183231f,
     0.386576f,  0.018409f, -0.171822f,  0.074483f,  0.051108f,  0.027072f,
    -0.273827f,  0.191803f,  0.092176f,  0.171424f,  0.117555f, -0.082023f,
    -0.207892f, -0.343909f, -0.215398f,  0.102874f, -0.075478f, -0.247636f,
     0.159581f,  0.104456f,  0.068148f, -0.127813f, -0.030061f,  0.059870f,
    -0.244009f, -0.248451f, -0.033120f,  0.327300f,  0.181907f, -0.064249f,
    -0.114562f,  0.263542f, -0.054875f,  0.118943f,  0.093570f,  0.045641f,
     0.181570f, -0.287765f};

static const float test_7_out_scale = 0.0029836944f;
static const float test_7_out_zero_point = 0.0365116894f;
static const int8_t test_7_out_scales_frac[] = {23};
static const int test_7_out_sa_dim = -1;

static const int test_7_out_fx8_frac = 8;

#define TEST_7_OUT_TSR_SHARED_DESCR \
  /* .data = */ { 0 }               \
  , /* .mem_stride = */ {0}, /* .shape = */ {8, 16}, /* .rank =  */ 2

static const mli_tensor test_7_out_tsr_fx16 = {TEST_7_OUT_TSR_SHARED_DESCR,

                                               /* .el_type = */ MLI_EL_FX_16,
                                               /* .el_params = */ {0}};

static const mli_tensor test_7_out_tsr_sa8 = {TEST_7_OUT_TSR_SHARED_DESCR,

                                              /* .el_type = */ MLI_EL_SA_8,
                                              /* .el_params = */ {0}};


static const float test_8_out_data[] = {
    -0.091750f, -0.051321f,  0.282265f,  0.283774f,  0.245932f,  0.045961f,
     0.149066f,  0.200496f,  0.174177f,  0.012408f,  0.008006f, -0.120819f,
     0.833865f,  0.273061f, -0.087248f,  0.279299f,  0.345489f,  0.169200f,
     0.019774f,  0.094294f,  0.336745f, -0.001856f,  0.109068f, -0.020643f,
     0.049847f,  0.289462f,  0.118578f,  0.001710f, -0.163252f, -0.055075f,
    -0.014320f,  0.558261f,  0.146462f,  0.616282f,  0.389316f,  0.017345f,
     0.159876f,  0.134407f, -0.027262f,  0.256677f,  0.206893f,  0.062214f,
    -0.258394f,  0.107336f,  0.140401f,  0.079273f, -0.112908f,  0.129771f,
    -0.124119f,  0.009234f,  0.207962f,  0.223424f,  0.354668f, -0.038669f,
     0.233342f, -0.146538f,  0.224826f,  0.142887f,  0.111566f,  0.427009f,
     0.477671f,  0.577053f,  0.201815f,  0.126930f, -0.191261f,  0.112142f,
     0.263059f,  0.216071f, -0.111014f,  0.173853f,  0.005598f,  0.072917f,
     0.270399f, -0.039309f, -0.029376f,  0.338695f,  0.126863f,  0.244499f,
     0.024809f, -0.161522f, -0.085716f,  0.488852f,  0.363817f,  0.215600f,
    -0.038830f,  0.105875f, -0.165981f, -0.065595f,  0.064287f,  0.034292f,
     0.773151f,  0.156937f,  0.012316f,  0.148965f,  0.225760f,  0.108602f,
     0.132049f,  0.383607f,  0.184353f,  0.342847f,  0.235109f,  0.066459f,
     0.044761f,  0.010666f, -0.302838f,  0.205749f, -0.150957f, -0.049250f,
     0.319163f,  0.208912f,  0.136296f, -0.109258f, -0.060121f,  0.119739f,
     0.035378f,  0.228086f,  0.186415f,  0.654600f,  0.363814f, -0.128497f,
    -0.071108f,  0.527085f, -0.109751f,  0.237886f,  0.187139f,  0.091283f,
     0.363140f,  0.059452f};

static const float test_8_out_scale = 0.0044576600f;
static const float test_8_out_zero_point = 0.2655138373f;
static const int8_t test_8_out_scales_frac[] = {22};
static const int test_8_out_sa_dim = -1;

static const int test_8_out_fx8_frac = 7;

#define TEST_8_OUT_TSR_SHARED_DESCR \
  /* .data = */ { 0 }               \
  , /* .mem_stride = */ {0}, /* .shape = */ {8, 16}, /* .rank =  */ 2

static const mli_tensor test_8_out_tsr_fx16 = {TEST_8_OUT_TSR_SHARED_DESCR,

                                               /* .el_type = */ MLI_EL_FX_16,
                                               /* .el_params = */ {0}};

static const mli_tensor test_8_out_tsr_sa8 = {TEST_8_OUT_TSR_SHARED_DESCR,

                                              /* .el_type = */ MLI_EL_SA_8,
                                              /* .el_params = */ {0}};

static const float test_9_out_data[] = {
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f,
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.833865f, 0.500000f,
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f,
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f,
    0.500000f, 0.500000f, 0.500000f, 0.558261f, 0.500000f, 0.616282f, 0.500000f,
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f,
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f,
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f,
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.577053f, 0.500000f,
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f,
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f,
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f,
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.773151f,
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f,
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f,
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f,
    0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.500000f, 0.654600f, 0.500000f,
    0.500000f, 0.500000f, 0.527085f, 0.500000f, 0.500000f, 0.500000f, 0.500000f,
    0.500000f, 0.500000f};

static const float test_9_out_scale = 0.0052631578947f;
static const float test_9_out_zero_point = 0.6669327021f;
static const int8_t test_9_out_scales_frac[] = {22};
static const int test_9_out_sa_dim = -1;

static const int test_9_out_fx8_frac = 7;

#define TEST_9_OUT_TSR_SHARED_DESCR \
  /* .data = */ { 0 }               \
  , /* .mem_stride = */ {0}, /* .shape = */ {8, 16}, /* .rank =  */ 2

static const mli_tensor test_9_out_tsr_fx16 = {TEST_9_OUT_TSR_SHARED_DESCR,

                                               /* .el_type = */ MLI_EL_FX_16,
                                               /* .el_params = */ {0}};

static const mli_tensor test_9_out_tsr_sa8 = {TEST_9_OUT_TSR_SHARED_DESCR,

                                              /* .el_type = */ MLI_EL_SA_8,
                                              /* .el_params = */ {0}};

static const float test_10_out_data[] = {
    -0.154767f, -0.118563f,  0.226941f, -0.109082f, -0.022779f, -0.068952f,
    -0.315049f, -0.299830f,  0.022000f, -0.019029f, -0.145222f, -0.314864f,
     0.263768f,  0.237886f, -0.145260f,  0.157669f,  0.156119f, -0.496154f,
    -0.355891f,  0.092650f, -0.078201f, -0.185674f, -0.390024f, -0.247575f,
    -0.149485f, -0.265108f, -0.039426f, -0.024900f, -0.414436f, -0.223744f,
    -0.038971f, -0.054766f, -0.158402f,  0.176820f, -0.067452f, -0.066083f,
    -0.041596f, -0.161529f, -0.303202f,  0.108276f,  0.107938f,  0.040067f,
    -0.341104f, -0.018200f, -0.268049f, -0.111010f, -0.211149f, -0.315468f,
    -0.157668f, -0.127504f, -0.133055f, -0.305548f,  0.086594f, -0.278795f,
     0.077915f, -0.263391f,  0.020152f, -0.012567f, -0.162642f, -0.306916f,
    -0.245183f,  0.183883f, -0.029849f, -0.298754f, -0.637734f,  0.087108f,
    -0.025554f, -0.241929f, -0.257113f,  0.085338f, -0.242127f, -0.000472f,
     0.014796f, -0.208077f, -0.042013f, -0.011296f, -0.285461f,  0.096162f,
    -0.179204f, -0.354448f, -0.135553f,  0.046803f, -0.033813f, -0.131824f,
    -0.440843f, -0.062320f, -0.242753f, -0.222373f, -0.100688f, -0.366461f,
    -0.187089f,  0.036817f, -0.343643f,  0.113855f,  0.102216f,  0.054143f,
    -0.547653f,  0.060413f, -0.096945f,  0.160177f, -0.249445f, -0.164047f,
    -0.415784f, -0.687819f, -0.430796f, -0.025816f, -0.206852f, -0.495271f,
     0.005334f,  0.010188f,  0.027012f, -0.255625f, -0.098318f,  0.015853f,
    -0.488017f, -0.496901f, -0.066240f,  0.074828f,  0.112086f, -0.218000f,
    -0.229125f, -0.178859f, -0.292142f,  0.049861f, -0.048190f, -0.128079f,
     0.161887f, -0.575530f};

static const float test_10_out_scale = 0.0037317127f;
static const float test_10_out_zero_point = -0.2120254040f;
static const int8_t test_10_out_scales_frac[] = {23};
static const int test_10_out_sa_dim = -1;

static const int test_10_out_fx8_frac = 7;

#define TEST_10_OUT_TSR_SHARED_DESCR \
  /* .data = */ { 0 }                \
  , /* .mem_stride = */ {0}, /* .shape = */ {8, 16}, /* .rank =  */ 2

static const mli_tensor test_10_out_tsr_fx16 = {TEST_10_OUT_TSR_SHARED_DESCR,

                                                /* .el_type = */ MLI_EL_FX_16,
                                                /* .el_params = */ {0}};

static const mli_tensor test_10_out_tsr_sa8 = {TEST_10_OUT_TSR_SHARED_DESCR,

                                               /* .el_type = */ MLI_EL_SA_8,
                                               /* .el_params = */ {0}};

static const float test_11_out_data[] = {
    -0.154767f, -0.118563f,  0.226941f,  0.283774f,  0.245932f, -0.068952f,
    -0.315049f, -0.299830f,  0.174177f, -0.019029f, -0.145222f, -0.314864f,
     0.500000f,  0.273061f, -0.087248f,  0.279299f,  0.345489f, -0.496154f,
    -0.355891f,  0.094294f,  0.336745f, -0.185674f, -0.390024f, -0.247575f,
    -0.149485f,  0.289462f, -0.039426f,  0.001710f, -0.414436f, -0.223744f,
    -0.038971f,  0.500000f,  0.146462f,  0.500000f,  0.389316f, -0.066083f,
     0.159876f, -0.161529f, -0.027262f,  0.108276f,  0.206893f,  0.040067f,
    -0.341104f,  0.107336f, -0.268049f, -0.111010f, -0.211149f,  0.129771f,
    -0.124119f,  0.009234f,  0.207962f,  0.223424f,  0.354668f, -0.278795f,
     0.077915f, -0.263391f,  0.224826f,  0.142887f, -0.162642f,  0.427009f,
     0.477671f,  0.500000f,  0.201815f,  0.126930f, -0.637734f,  0.112142f,
     0.263059f,  0.216071f, -0.257113f,  0.173853f, -0.242127f,  0.072917f,
     0.270399f, -0.039309f, -0.029376f,  0.338695f, -0.285461f,  0.244499f,
     0.024809f, -0.161522f, -0.085716f,  0.488852f,  0.363817f, -0.131824f,
    -0.440843f, -0.062320f, -0.165981f, -0.065595f, -0.100688f, -0.366461f,
     0.500000f,  0.036817f, -0.343643f,  0.148965f,  0.102216f,  0.054143f,
    -0.547653f,  0.383607f,  0.184353f,  0.342847f,  0.235109f, -0.164047f,
    -0.415784f, -0.687819f, -0.430796f,  0.205749f, -0.150957f, -0.495271f,
     0.319163f,  0.208912f,  0.136296f, -0.255625f, -0.060121f,  0.119739f,
    -0.488017f, -0.496901f, -0.066240f,  0.500000f,  0.363814f, -0.128497f,
    -0.229125f,  0.500000f, -0.109751f,  0.237886f,  0.187139f,  0.091283f,
     0.363140f, -0.575530f};

static const float test_11_out_scale = 0.0046581128f;
static const float test_11_out_zero_point = -0.0939093828f;
static const int8_t test_11_out_scales_frac[] = {22};
static const int test_11_out_sa_dim = -1;

static const int test_11_out_fx8_frac = 7;

#define TEST_11_OUT_TSR_SHARED_DESCR \
  /* .data = */ { 0 }                \
  , /* .mem_stride = */ {0}, /* .shape = */ {8, 16}, /* .rank =  */ 2

static const mli_tensor test_11_out_tsr_fx16 = {TEST_11_OUT_TSR_SHARED_DESCR,

                                                /* .el_type = */ MLI_EL_FX_16,
                                                /* .el_params = */ {0}};

static const mli_tensor test_11_out_tsr_sa8 = {TEST_11_OUT_TSR_SHARED_DESCR,

                                               /* .el_type = */ MLI_EL_SA_8,
                                               /* .el_params = */ {0}};


tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8,
                              input_1_data,
                              sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data,
                             sizeof(input_1_data) / sizeof(input_1_data[0]),
                             &input_1_scale, 1, &input_1_zero_point, 1,
                             input_1_scales_frac, 1);

tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx8_frac + 8,
                              input_2_data,
                              sizeof(input_2_data) / sizeof(input_2_data[0]));
tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data,
                             sizeof(input_2_data) / sizeof(input_2_data[0]),
                             &input_2_scale, 1, &input_2_zero_point, 1,
                             input_2_scales_frac, 1);

tensor_quantizer input_3_fx16(input_3_tsr_fx16, input_3_fx8_frac + 8,
                              input_3_data,
                              sizeof(input_3_data) / sizeof(input_3_data[0]));
tensor_quantizer input_3_sa8(input_3_tsr_sa8, input_3_sa_dim, input_3_data,
                             sizeof(input_3_data) / sizeof(input_3_data[0]),
                             &input_3_scale, 1, &input_3_zero_point, 1,
                             input_3_scales_frac, 1);

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8,
                                 test_1_out_data,
                                 sizeof(test_1_out_data) /
                                     sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim,
                                test_1_out_data, sizeof(test_1_out_data) /
                                                     sizeof(test_1_out_data[0]),
                                &test_1_out_scale, 1, &test_1_out_zero_point, 1,
                                test_1_out_scales_frac, 1);


tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_fx8_frac + 8,
                                 test_2_out_data,
                                 sizeof(test_2_out_data) /
                                     sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_sa_dim,
                                test_2_out_data, sizeof(test_2_out_data) /
                                                     sizeof(test_2_out_data[0]),
                                &test_2_out_scale, 1, &test_2_out_zero_point, 1,
                                test_2_out_scales_frac, 1);

tensor_quantizer test_3_out_fx16(test_3_out_tsr_fx16, test_3_out_fx8_frac + 8,
                                 test_3_out_data,
                                 sizeof(test_3_out_data) /
                                     sizeof(test_3_out_data[0]));
tensor_quantizer test_3_out_sa8(test_3_out_tsr_sa8, test_3_out_sa_dim,
                                test_3_out_data, sizeof(test_3_out_data) /
                                                     sizeof(test_3_out_data[0]),
                                &test_3_out_scale, 1, &test_3_out_zero_point, 1,
                                test_3_out_scales_frac, 1);

tensor_quantizer test_4_out_fx16(test_4_out_tsr_fx16, test_4_out_fx8_frac + 8,
                                 test_4_out_data,
                                 sizeof(test_4_out_data) /
                                     sizeof(test_4_out_data[0]));
tensor_quantizer test_4_out_sa8(test_4_out_tsr_sa8, test_4_out_sa_dim,
                                test_4_out_data, sizeof(test_4_out_data) /
                                                     sizeof(test_4_out_data[0]),
                                &test_4_out_scale, 1, &test_4_out_zero_point, 1,
                                test_4_out_scales_frac, 1);

tensor_quantizer test_5_out_fx16(test_5_out_tsr_fx16, test_5_out_fx8_frac + 8,
                                 test_5_out_data,
                                 sizeof(test_5_out_data) /
                                     sizeof(test_5_out_data[0]));
tensor_quantizer test_5_out_sa8(test_5_out_tsr_sa8, test_5_out_sa_dim,
                                test_5_out_data, sizeof(test_5_out_data) /
                                                     sizeof(test_5_out_data[0]),
                                &test_5_out_scale, 1, &test_5_out_zero_point, 1,
                                test_5_out_scales_frac, 1);

tensor_quantizer test_6_out_fx16(test_6_out_tsr_fx16, test_6_out_fx8_frac + 8,
                                 test_6_out_data,
                                 sizeof(test_6_out_data) /
                                     sizeof(test_6_out_data[0]));
tensor_quantizer test_6_out_sa8(test_6_out_tsr_sa8, test_6_out_sa_dim,
                                test_6_out_data, sizeof(test_6_out_data) /
                                                     sizeof(test_6_out_data[0]),
                                &test_6_out_scale, 1, &test_6_out_zero_point, 1,
                                test_6_out_scales_frac, 1);


tensor_quantizer test_7_out_fx16(test_7_out_tsr_fx16, test_7_out_fx8_frac + 8,
                                 test_7_out_data,
                                 sizeof(test_7_out_data) /
                                     sizeof(test_7_out_data[0]));
tensor_quantizer test_7_out_sa8(test_7_out_tsr_sa8, test_7_out_sa_dim,
                                test_7_out_data, sizeof(test_7_out_data) /
                                                     sizeof(test_7_out_data[0]),
                                &test_7_out_scale, 1, &test_7_out_zero_point, 1,
                                test_7_out_scales_frac, 1);

tensor_quantizer test_8_out_fx16(test_8_out_tsr_fx16, test_8_out_fx8_frac + 8,
                                 test_8_out_data,
                                 sizeof(test_8_out_data) /
                                     sizeof(test_8_out_data[0]));
tensor_quantizer test_8_out_sa8(test_8_out_tsr_sa8, test_8_out_sa_dim,
                                test_8_out_data, sizeof(test_8_out_data) /
                                                     sizeof(test_8_out_data[0]),
                                &test_8_out_scale, 1, &test_8_out_zero_point, 1,
                                test_8_out_scales_frac, 1);

tensor_quantizer test_9_out_fx16(test_9_out_tsr_fx16, test_9_out_fx8_frac + 8,
                                 test_9_out_data,
                                 sizeof(test_9_out_data) /
                                     sizeof(test_9_out_data[0]));
tensor_quantizer test_9_out_sa8(test_9_out_tsr_sa8, test_9_out_sa_dim,
                                test_9_out_data, sizeof(test_9_out_data) /
                                                     sizeof(test_9_out_data[0]),
                                &test_9_out_scale, 1, &test_9_out_zero_point, 1,
                                test_9_out_scales_frac, 1);

tensor_quantizer test_10_out_fx16(test_10_out_tsr_fx16,
                                  test_10_out_fx8_frac + 8, test_10_out_data,
                                  sizeof(test_10_out_data) /
                                      sizeof(test_10_out_data[0]));
tensor_quantizer test_10_out_sa8(test_10_out_tsr_sa8, test_10_out_sa_dim,
                                 test_10_out_data,
                                 sizeof(test_10_out_data) /
                                     sizeof(test_10_out_data[0]),
                                 &test_10_out_scale, 1, &test_10_out_zero_point,
                                 1, test_10_out_scales_frac, 1);

tensor_quantizer test_11_out_fx16(test_11_out_tsr_fx16,
                                  test_11_out_fx8_frac + 8, test_11_out_data,
                                  sizeof(test_11_out_data) /
                                      sizeof(test_11_out_data[0]));
tensor_quantizer test_11_out_sa8(test_11_out_tsr_sa8, test_11_out_sa_dim,
                                 test_11_out_data,
                                 sizeof(test_11_out_data) /
                                     sizeof(test_11_out_data[0]),
                                 &test_11_out_scale, 1, &test_11_out_zero_point,
                                 1, test_11_out_scales_frac, 1);

// MAX/MIN parameters
//===================================================

/* FX element paramaeters */
static const int input_12_fx8_frac = 7;
static const int input_13_fx8_frac = 7;
static const int input_23_fx8_frac = 8;
/* SA8 element paramaeters */
static const int input_sa_dim = -1;
static const float input_12_zero_point = 0.0f;
static const float input_13_zero_point = 0.0f;
static const float input_23_zero_point = 0.0f;
static const float input_12_scale = 1.0f;
static const float input_13_scale = 1.0f;
static const float input_23_scale = 1.0f;
static const int8_t input_12_scales_frac[] = {0};
static const int8_t input_13_scales_frac[] = {0};
static const int8_t input_23_scales_frac[] = {0};

extern mli::tst::tensor_quantizer input_1_fx16_12;
extern mli::tst::tensor_quantizer input_2_fx16_12;
extern mli::tst::tensor_quantizer input_1_sa8_12;
extern mli::tst::tensor_quantizer input_2_sa8_12;
extern mli::tst::tensor_quantizer input_1_fx16_13;
extern mli::tst::tensor_quantizer input_3_fx16_13;
extern mli::tst::tensor_quantizer input_1_sa8_13;
extern mli::tst::tensor_quantizer input_3_sa8_13;
extern mli::tst::tensor_quantizer input_2_fx16_23;
extern mli::tst::tensor_quantizer input_3_fx16_23;
extern mli::tst::tensor_quantizer input_2_sa8_23;
extern mli::tst::tensor_quantizer input_3_sa8_23;

// inputs 1,2 with same sa8/fx parameters from in1 and in2
tensor_quantizer input_1_fx16_12(input_1_tsr_fx16, input_12_fx8_frac + 8,
                                 input_1_data, sizeof(input_1_data) /
                                                   sizeof(input_1_data[0]));
tensor_quantizer input_2_fx16_12(input_2_tsr_fx16, input_12_fx8_frac + 8,
                                 input_2_data, sizeof(input_2_data) /
                                                   sizeof(input_2_data[0]));
tensor_quantizer input_1_sa8_12(input_1_tsr_sa8, input_sa_dim, input_1_data,
                                sizeof(input_1_data) / sizeof(input_1_data[0]),
                                &input_12_scale, 1, &input_12_zero_point, 1,
                                input_12_scales_frac, 1);
tensor_quantizer input_2_sa8_12(input_2_tsr_sa8, input_sa_dim, input_2_data,
                                sizeof(input_2_data) / sizeof(input_2_data[0]),
                                &input_12_scale, 1, &input_12_zero_point, 1,
                                input_12_scales_frac, 1);

// inputs 1,3 with same sa8/fx parameters from in1 and in3
tensor_quantizer input_1_fx16_13(input_1_tsr_fx16, input_13_fx8_frac + 8,
                                 input_1_data, sizeof(input_1_data) /
                                                   sizeof(input_1_data[0]));
tensor_quantizer input_3_fx16_13(input_3_tsr_fx16, input_13_fx8_frac + 8,
                                 input_3_data, sizeof(input_3_data) /
                                                   sizeof(input_3_data[0]));
tensor_quantizer input_1_sa8_13(input_1_tsr_sa8, input_sa_dim, input_1_data,
                                sizeof(input_1_data) / sizeof(input_1_data[0]),
                                &input_13_scale, 1, &input_13_zero_point, 1,
                                input_13_scales_frac, 1);
tensor_quantizer input_3_sa8_13(input_3_tsr_sa8, input_sa_dim, input_3_data,
                                sizeof(input_3_data) / sizeof(input_3_data[0]),
                                &input_13_scale, 1, &input_13_zero_point, 1,
                                input_13_scales_frac, 1);

// inputs 2,3 with same sa8/fx parameters from in2 and in3
tensor_quantizer input_2_fx16_23(input_2_tsr_fx16, input_23_fx8_frac + 8,
                                 input_2_data, sizeof(input_2_data) /
                                                   sizeof(input_2_data[0]));
tensor_quantizer input_3_fx16_23(input_3_tsr_fx16, input_23_fx8_frac + 8,
                                 input_3_data, sizeof(input_3_data) /
                                                   sizeof(input_3_data[0]));
tensor_quantizer input_2_sa8_23(input_2_tsr_sa8, input_sa_dim, input_2_data,
                                sizeof(input_2_data) / sizeof(input_2_data[0]),
                                &input_23_scale, 1, &input_23_zero_point, 1,
                                input_23_scales_frac, 1);
tensor_quantizer input_3_sa8_23(input_3_tsr_sa8, input_sa_dim, input_3_data,
                                sizeof(input_3_data) / sizeof(input_3_data[0]),
                                &input_23_scale, 1, &input_23_zero_point, 1,
                                input_23_scales_frac, 1);
