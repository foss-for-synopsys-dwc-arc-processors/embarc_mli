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
extern mli::tst::tensor_quantizer test_1_out_fx16;
extern mli::tst::tensor_quantizer test_1_out_sa8;
extern mli::tst::tensor_quantizer test_2_out_fx16;
extern mli::tst::tensor_quantizer test_2_out_sa8;

static const float input_1_data[] = {
    -2.999756f, -2.976318f, -2.952881f, -2.929443f, -2.906006f, -2.882568f, -2.859131f, -2.835693f,
    -2.812256f, -2.788818f, -2.765381f, -2.741943f, -2.718506f, -2.695068f, -2.671631f, -2.648193f,
    -2.624756f, -2.601318f, -2.577881f, -2.554443f, -2.531006f, -2.507568f, -2.484131f, -2.460693f,
    -2.437256f, -2.413818f, -2.390381f, -2.366943f, -2.343506f, -2.320068f, -2.296631f, -2.273193f,
    -2.249756f, -2.226318f, -2.202881f, -2.179443f, -2.156006f, -2.132568f, -2.109131f, -2.085693f,
    -2.062256f, -2.038818f, -2.015381f, -1.991943f, -1.968506f, -1.945068f, -1.921631f, -1.898193f,
    -1.874756f, -1.851318f, -1.827881f, -1.804443f, -1.781006f, -1.757568f, -1.734131f, -1.710693f,
    -1.687256f, -1.663818f, -1.640381f, -1.616943f, -1.593506f, -1.570068f, -1.546631f, -1.523193f,
    -1.499756f, -1.476318f, -1.452881f, -1.429443f, -1.406006f, -1.382568f, -1.359131f, -1.335693f,
    -1.312256f, -1.288818f, -1.265381f, -1.241943f, -1.218506f, -1.195068f, -1.171631f, -1.148193f,
    -1.124756f, -1.101318f, -1.077881f, -1.054443f, -1.031006f, -1.007568f, -0.984131f, -0.960693f,
    -0.937256f, -0.913818f, -0.890381f, -0.866943f, -0.843506f, -0.820068f, -0.796631f, -0.773193f,
    -0.749756f, -0.726318f, -0.702881f, -0.679443f, -0.656006f, -0.632568f, -0.609131f, -0.585693f,
    -0.562256f, -0.538818f, -0.515381f, -0.491943f, -0.468506f, -0.445068f, -0.421631f, -0.398193f,
    -0.374756f, -0.351318f, -0.327881f, -0.304443f, -0.281006f, -0.257568f, -0.234131f, -0.210693f,
    -0.187256f, -0.163818f, -0.140381f, -0.116943f, -0.093506f, -0.070068f, -0.046631f, -0.023193f,
    0.000244f, 0.023682f, 0.047119f, 0.070557f, 0.093994f, 0.117432f, 0.140869f, 0.164307f,
    0.187744f, 0.211182f, 0.234619f, 0.258057f, 0.281494f, 0.304932f, 0.328369f, 0.351807f,
    0.375244f, 0.398682f, 0.422119f, 0.445557f, 0.468994f, 0.492432f, 0.515869f, 0.539307f,
    0.562744f, 0.586182f, 0.609619f, 0.633057f, 0.656494f, 0.679932f, 0.703369f, 0.726807f,
    0.750244f, 0.773682f, 0.797119f, 0.820557f, 0.843994f, 0.867432f, 0.890869f, 0.914307f,
    0.937744f, 0.961182f, 0.984619f, 1.008057f, 1.031494f, 1.054932f, 1.078369f, 1.101807f,
    1.125244f, 1.148682f, 1.172119f, 1.195557f, 1.218994f, 1.242432f, 1.265869f, 1.289307f,
    1.312744f, 1.336182f, 1.359619f, 1.383057f, 1.406494f, 1.429932f, 1.453369f, 1.476807f,
    1.500244f, 1.523682f, 1.547119f, 1.570557f, 1.593994f, 1.617432f, 1.640869f, 1.664307f,
    1.687744f, 1.711182f, 1.734619f, 1.758057f, 1.781494f, 1.804932f, 1.828369f, 1.851807f,
    1.875244f, 1.898682f, 1.922119f, 1.945557f, 1.968994f, 1.992432f, 2.015869f, 2.039307f,
    2.062744f, 2.086182f, 2.109619f, 2.133057f, 2.156494f, 2.179932f, 2.203369f, 2.226807f,
    2.250244f, 2.273682f, 2.297119f, 2.320557f, 2.343994f, 2.367432f, 2.390869f, 2.414307f,
    2.437744f, 2.461182f, 2.484619f, 2.508057f, 2.531494f, 2.554932f, 2.578369f, 2.601807f,
    2.625244f, 2.648682f, 2.672119f, 2.695557f, 2.718994f, 2.742432f, 2.765869f, 2.789307f,
    2.812744f, 2.836182f, 2.859619f, 2.883057f, 2.906494f, 2.929932f, 2.953369f, 2.976807f};

static const float input_1_scale = 0.0234375000f;
static const float input_1_zero_point = -0.0114746094f;
static const int8_t input_1_scales_frac[] = {20};
static const int input_1_sa_dim = -1;

static const int input_1_fx8_frac = 5;

#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 2, 4, 16}, \
    /* .rank =  */ 4


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

#define INPUT_1_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 2*(4*(16*4+2)+4), 4*(16*4+2)+4, 16*4+2, 1 }, \
    /* .shape = */ {2, 2, 4, 16}, \
    /* .rank =  */ 4


static const mli_tensor input_1_memstr_tsr_fx16 = {
    INPUT_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_1_memstr_tsr_sa8 = {
    INPUT_1_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float input_2_data[] = {
    3.000244f, 3.039307f, 3.078369f, 3.117432f, 3.156494f, 3.195557f, 3.234619f, 3.273682f,
    3.312744f, 3.351807f, 3.390869f, 3.429932f, 3.468994f, 3.508057f, 3.547119f, 3.586182f,
    3.625244f, 3.664307f, 3.703369f, 3.742432f, 3.781494f, 3.820557f, 3.859619f, 3.898682f,
    3.937744f, 3.976807f, 4.015869f, 4.054932f, 4.093994f, 4.133057f, 4.172119f, 4.211182f,
    4.250244f, 4.289307f, 4.328369f, 4.367432f, 4.406494f, 4.445557f, 4.484619f, 4.523682f,
    4.562744f, 4.601807f, 4.640869f, 4.679932f, 4.718994f, 4.758057f, 4.797119f, 4.836182f,
    4.875244f, 4.914307f, 4.953369f, 4.992432f, 5.031494f, 5.070557f, 5.109619f, 5.148682f,
    5.187744f, 5.226807f, 5.265869f, 5.304932f, 5.343994f, 5.383057f, 5.422119f, 5.461182f,
    5.500244f, 5.539307f, 5.578369f, 5.617432f, 5.656494f, 5.695557f, 5.734619f, 5.773682f,
    5.812744f, 5.851807f, 5.890869f, 5.929932f, 5.968994f, 6.008057f, 6.047119f, 6.086182f,
    6.125244f, 6.164307f, 6.203369f, 6.242432f, 6.281494f, 6.320557f, 6.359619f, 6.398682f,
    6.437744f, 6.476807f, 6.515869f, 6.554932f, 6.593994f, 6.633057f, 6.672119f, 6.711182f,
    6.750244f, 6.789307f, 6.828369f, 6.867432f, 6.906494f, 6.945557f, 6.984619f, 7.023682f,
    7.062744f, 7.101807f, 7.140869f, 7.179932f, 7.218994f, 7.258057f, 7.297119f, 7.336182f,
    7.375244f, 7.414307f, 7.453369f, 7.492432f, 7.531494f, 7.570557f, 7.609619f, 7.648682f,
    7.687744f, 7.726807f, 7.765869f, 7.804932f, 7.843994f, 7.883057f, 7.922119f, 7.961182f};

static const float input_2_scale = 0.039062503937f;
static const float input_2_zero_point = 3.000244f;
static const int8_t input_2_scales_frac[] = {19};
static const int input_2_sa_dim = -1;

static const int input_2_fx8_frac = 4;

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

static const float test_1_out_data[] = {
    0.047437f, 0.048507f, 0.049601f, 0.050717f, 0.051857f, 0.053022f, 0.054211f, 0.055426f,
    0.056665f, 0.057931f, 0.059224f, 0.060543f, 0.061890f, 0.063265f, 0.064668f, 0.066100f,
    0.067562f, 0.069054f, 0.070576f, 0.072129f, 0.073713f, 0.075329f, 0.076978f, 0.078660f,
    0.080376f, 0.082125f, 0.083909f, 0.085728f, 0.087583f, 0.089474f, 0.091402f, 0.093368f,
    0.095371f, 0.097412f, 0.099492f, 0.101612f, 0.103771f, 0.105971f, 0.108213f, 0.110495f,
    0.112820f, 0.115187f, 0.117597f, 0.120051f, 0.122549f, 0.125092f, 0.127680f, 0.130313f,
    0.132992f, 0.135718f, 0.138491f, 0.141311f, 0.144179f, 0.147095f, 0.150060f, 0.153074f,
    0.156137f, 0.159250f, 0.162413f, 0.165627f, 0.168891f, 0.172207f, 0.175573f, 0.178992f,
    0.182462f, 0.185984f, 0.189559f, 0.193185f, 0.196865f, 0.200597f, 0.204382f, 0.208219f,
    0.212110f, 0.216053f, 0.220049f, 0.224098f, 0.228199f, 0.232354f, 0.236560f, 0.240819f,
    0.245130f, 0.249493f, 0.253907f, 0.258373f, 0.262889f, 0.267456f, 0.272073f, 0.276739f,
    0.281455f, 0.286219f, 0.291031f, 0.295891f, 0.300797f, 0.305749f, 0.310747f, 0.315789f,
    0.320875f, 0.326003f, 0.331174f, 0.336386f, 0.341637f, 0.346928f, 0.352257f, 0.357624f,
    0.363026f, 0.368463f, 0.373933f, 0.379436f, 0.384970f, 0.390534f, 0.396127f, 0.401746f,
    0.407392f, 0.413063f, 0.418756f, 0.424472f, 0.430207f, 0.435962f, 0.441733f, 0.447521f,
    0.453322f, 0.459137f, 0.464962f, 0.470797f, 0.476641f, 0.482490f, 0.488344f, 0.494202f,
    0.500061f, 0.505920f, 0.511778f, 0.517632f, 0.523481f, 0.529324f, 0.535159f, 0.540985f,
    0.546799f, 0.552600f, 0.558387f, 0.564158f, 0.569912f, 0.575648f, 0.581362f, 0.587056f,
    0.592726f, 0.598371f, 0.603990f, 0.609582f, 0.615146f, 0.620679f, 0.626181f, 0.631651f,
    0.637087f, 0.642489f, 0.647854f, 0.653182f, 0.658472f, 0.663723f, 0.668934f, 0.674104f,
    0.679232f, 0.684317f, 0.689358f, 0.694355f, 0.699306f, 0.704211f, 0.709069f, 0.713881f,
    0.718644f, 0.723358f, 0.728024f, 0.732640f, 0.737205f, 0.741721f, 0.746185f, 0.750598f,
    0.754960f, 0.759270f, 0.763528f, 0.767733f, 0.771887f, 0.775987f, 0.780035f, 0.784030f,
    0.787972f, 0.791861f, 0.795698f, 0.799481f, 0.803212f, 0.806891f, 0.810516f, 0.814090f,
    0.817611f, 0.821080f, 0.824497f, 0.827863f, 0.831177f, 0.834441f, 0.837653f, 0.840815f,
    0.843927f, 0.846990f, 0.850002f, 0.852966f, 0.855881f, 0.858748f, 0.861567f, 0.864339f,
    0.867064f, 0.869742f, 0.872375f, 0.874961f, 0.877503f, 0.880000f, 0.882453f, 0.884863f,
    0.887229f, 0.889553f, 0.891835f, 0.894075f, 0.896274f, 0.898433f, 0.900552f, 0.902631f,
    0.904672f, 0.906674f, 0.908638f, 0.910565f, 0.912456f, 0.914310f, 0.916128f, 0.917912f,
    0.919661f, 0.921375f, 0.923056f, 0.924705f, 0.926320f, 0.927904f, 0.929456f, 0.930978f,
    0.932469f, 0.933930f, 0.935361f, 0.936764f, 0.938138f, 0.939484f, 0.940803f, 0.942095f,
    0.943361f, 0.944600f, 0.945814f, 0.947003f, 0.948167f, 0.949306f, 0.950423f, 0.951515f};

static const float test_1_out_scale = 0.0035454053f;
static const float test_1_out_zero_point = 0.4994760752f;
static const int8_t test_1_out_scales_frac[] = {23};
static const int test_1_out_sa_dim = -1;

static const int test_1_out_fx8_frac = 7;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 2, 4, 16}, \
    /* .rank =  */ 4


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

#define TEST_1_OUT_MEMSTR_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ { 2*(4*(16*4+2)+4), 4*(16*4+2)+4, 16*4+2, 1 }, \
    /* .shape = */ {2, 2, 4, 16}, \
    /* .rank =  */ 4


static const mli_tensor test_1_out_memstr_tsr_fx16 = {
    TEST_1_OUT_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_1_out_memstr_tsr_sa8 = {
    TEST_1_OUT_MEMSTR_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};

static const float test_2_out_data[] = {
    0.952585f, 0.954319f, 0.955992f, 0.957606f, 0.959164f, 0.960667f, 0.962116f, 0.963515f,
    0.964863f, 0.966164f, 0.967418f, 0.968627f, 0.969793f, 0.970916f, 0.971999f, 0.973043f,
    0.974049f, 0.975018f, 0.975952f, 0.976852f, 0.977719f, 0.978554f, 0.979359f, 0.980134f,
    0.980881f, 0.981600f, 0.982292f, 0.982959f, 0.983601f, 0.984219f, 0.984815f, 0.985388f,
    0.985940f, 0.986471f, 0.986983f, 0.987475f, 0.987949f, 0.988405f, 0.988845f, 0.989267f,
    0.989674f, 0.990066f, 0.990443f, 0.990806f, 0.991155f, 0.991491f, 0.991814f, 0.992125f,
    0.992425f, 0.992713f, 0.992990f, 0.993257f, 0.993513f, 0.993760f, 0.993998f, 0.994226f,
    0.994446f, 0.994658f, 0.994862f, 0.995057f, 0.995246f, 0.995427f, 0.995602f, 0.995770f,
    0.995931f, 0.996086f, 0.996236f, 0.996379f, 0.996517f, 0.996650f, 0.996778f, 0.996901f,
    0.997020f, 0.997133f, 0.997243f, 0.997348f, 0.997450f, 0.997547f, 0.997641f, 0.997731f,
    0.997818f, 0.997901f, 0.997981f, 0.998059f, 0.998133f, 0.998204f, 0.998273f, 0.998339f,
    0.998403f, 0.998464f, 0.998522f, 0.998579f, 0.998633f, 0.998686f, 0.998736f, 0.998784f,
    0.998831f, 0.998876f, 0.998919f, 0.998960f, 0.999000f, 0.999038f, 0.999075f, 0.999110f,
    0.999144f, 0.999177f, 0.999209f, 0.999239f, 0.999268f, 0.999296f, 0.999323f, 0.999349f,
    0.999374f, 0.999398f, 0.999421f, 0.999443f, 0.999464f, 0.999485f, 0.999505f, 0.999524f,
    0.999542f, 0.999559f, 0.999576f, 0.999592f, 0.999608f, 0.999623f, 0.999637f, 0.999651f};

static const float test_2_out_scale = 0.0001845736f;
static const float test_2_out_zero_point = 0.9761183262f;
static const int8_t test_2_out_scales_frac[] = {27};
static const int test_2_out_sa_dim = -1;

static const int test_2_out_fx8_frac = 7;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 4, 8, 2}, \
    /* .rank =  */ 4


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

tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_1_memstr_fx16(input_1_memstr_tsr_fx16, input_1_fx8_frac + 8, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_memstr_sa8(input_1_memstr_tsr_sa8, input_1_sa_dim, input_1_data, 
sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx8_frac + 8, input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]));
tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data, 
sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim, test_1_out_data, 
sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_1_out_scale, 1, &test_1_out_zero_point, 1, test_1_out_scales_frac, 1);

tensor_quantizer test_1_out_memstr_fx16(test_1_out_memstr_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_memstr_sa8(test_1_out_memstr_tsr_sa8, test_1_out_sa_dim, test_1_out_data, 
sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_1_out_scale, 1, &test_1_out_zero_point, 1, test_1_out_scales_frac, 1);

tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_fx8_frac + 8, test_2_out_data, sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_sa_dim, test_2_out_data, 
sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &test_2_out_scale, 1, &test_2_out_zero_point, 1, test_2_out_scales_frac, 1);
