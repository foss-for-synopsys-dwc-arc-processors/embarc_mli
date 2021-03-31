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

// Tests configuration structures Declaration
//========================================
extern const mli_pool_cfg test_1_cfg;
extern const mli_pool_cfg test_2_cfg;
extern const mli_pool_cfg test_3_cfg;
extern const mli_pool_cfg test_4_cfg;
extern const mli_pool_cfg test_5_cfg;
extern const mli_pool_cfg test_6_cfg;
extern const mli_pool_cfg test_7_cfg;
extern const mli_pool_cfg test_8_cfg;

// Generated input vectors 
//========================================
static const float input_1_data[] = {
    -1.4040972f, -0.4529037f, -1.5962168f, -0.73494565f, -0.814128f, -0.7360624f, -0.120489374f, -1.0616391f, 
    -1.0163001f, -0.38303176f, -0.17544968f, -1.510983f, 1.8074336f, 3.0970626f, -1.54559f, -0.62684023f, 
    -0.7810305f, -0.65703535f, -0.62637943f, -0.44984207f, -0.6267744f, 10.593016f, -1.646564f, -0.7886187f, 
    5.6685967f, -1.5031856f, 3.4522443f, -0.2770916f, -0.3191755f, -0.18863408f, -0.5507039f, -1.7014537f, 
    -0.10716601f, -0.3843876f, -1.58752f, -0.58320874f, -0.76907414f, -1.1279624f, -0.71781516f, -1.371721f, 
    -0.6732101f, -0.93712145f, -1.3035706f, -1.3740748f, -0.6109958f, -0.39783722f, -0.64684796f, -1.7753627f, 
    -1.0078254f, -1.021867f, -1.0740795f, -1.243933f, -1.2341982f, -1.0360813f, -1.5717467f, -0.53784484f, 
    -1.173897f, -0.3888874f, -0.73509926f, -0.6296618f, -1.366994f, -0.12383132f, 1.2512428f, -0.77808017f, 
    -1.2270421f, -0.8131619f, -1.2114267f, -0.50910205f, -1.1695606f, -0.79659796f, -1.5574596f, -1.8897877f, 
    -0.26204118f, -0.95596135f, -0.6523025f, -0.54907614f, -0.72944057f, -0.33907992f, -1.7348433f, -1.1490866f,
    -2.0406806f, -0.8932164f, -1.2296221f, -0.48654634f, -1.193977f, -0.90377367f, 0.11085541f, -1.3249632f, 
    -1.572813f, -0.97322553f, -0.3745244f, -0.06398462f, -1.1979241f, -0.5966208f, -0.8863991f, -1.3440627f, 
    -0.96825576f, -1.3900144f, -1.3871597f, -1.5732166f, -0.9593854f, -0.7035985f, -1.4057658f, -0.86527133f, 
    -0.39088795f, -0.61055124f, -0.12121102f, -0.7022372f, -0.5630828f, -0.7391643f, 0.9815041f, -1.3275728f, 
    -1.3884043f, -1.124226f, -1.242418f, -1.2940874f, -0.6737328f, -2.1215305f, -0.60663337f, -1.0424935f, 
    -1.4408526f, -0.5885472f, 2.8008537f, -0.95094913f, -1.4891198f, -1.0007299f, -1.1550088f, -0.6758578f, 
    -0.3805352f, -0.6456276f, -1.7131983f, -0.33921084f, -0.39684492f, -0.87239605f, -0.5904598f, -0.7963563f, 
    -0.86228615f, -1.0794955f, 3.3609083f, -0.7886797f, -0.7327272f, -0.8171242f, -1.738547f, -0.70010686f, 
    0.4805395f, -0.65775436f, -0.74258316f, -0.5500512f, -1.1157719f, 1.4043885f, 0.99633455f, -1.3868202f, 
    -1.0996146f, -1.0642086f, -0.47607106f, -0.8666448f, -0.8845047f, -0.5995666f, -0.31296152f, -0.43325025f, 
    -0.76519936f, -1.0361811f, -0.5409625f, -0.57662904f, -1.0636926f, -0.9681042f, -0.01695835f, -0.69471425f, 
    -0.5913803f, -0.6655751f, -0.85448754f, -1.1140425f, -1.3287685f, -0.66801554f, -0.3051004f, -1.4155285f, 
    -0.5044277f, -0.8607243f, -1.7193509f, -2.183599f, -0.84390146f, -0.8469301f, -0.7387878f, -0.91895664f, 
    -0.5191655f, -1.4403934f, -1.4027972f, 1.4825351f, -0.004922455f, -1.6287858f, -0.993925f, -0.5332026f, 
    -0.9907754f, -0.8310245f, -1.7791178f, -0.16133723f, -0.6725665f, -1.0128365f, -1.0444493f, -0.99604607f, 
    -1.0821317f, -1.830801f, -1.4580777f, -1.2813649f, -1.00773f, -1.1029631f, -0.103657424f, -0.54025275f, 
    -1.0179627f, -0.9554991f, -1.2328091f, -2.7421093f, -1.9471334f, -1.633418f, -0.7367817f, 0.37797228f, 
    -0.47541645f, -1.6205299f, -1.1839347f, -1.2660786f, -0.13790841f, -0.58898354f, -0.1948162f, -1.892627f, 
    -0.57638615f, -1.1338538f, -0.17755578f, -0.5904507f, -0.49317116f, -0.6892894f, -0.744993f, -0.73453325f, 
    -0.4382035f, -1.354786f, 1.7248597f, -0.017363198f, -1.2726744f, -0.81926113f, -0.41005805f, -0.13540052f, 
    -1.3979547f, -0.6301549f, -0.65024376f, -1.2523519f, -0.35487315f, -0.43734974f, -0.5712727f, -0.499639f, 
    -0.9425663f, -0.3704904f, -0.9666545f, -1.1614499f, -0.48116407f, -0.35847312f, -0.9520558f, -1.1955967f, 
    -0.98160136f, -0.7986757f, -0.19397384f, -1.069739f, -0.31441155f, -1.2069892f, -0.5760116f, -0.41730127f,
    -1.3380334f, -0.67576665f, -0.56093484f, -0.7343522f, -0.56852627f, -0.49403435f};

static const int input_1_fx8_frac = 3;

static const int input_1_sa_dim = -1;
static const float input_1_scale = 0.05229460820555687f;
static const float input_1_zero_point = 3.9254531860351562f;
static const int8_t input_1_scales_frac[] = {18};


#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ {0},\
    /* .mem_stride =  */ {0},\
    /* .shape = */ {10, 9, 3}, \
    /* .rank = */ 3

#define INPUT_1_MEMSTR_TSR_SHARED_DESCR \
    /*.data = */ { 0 },\
    /* .mem_stride = */ { 10*6*1, 5*1, 1}, \
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
    1.8512048f, -0.27873313f, -0.2539165f, -0.09343147f, 1.3933007f, -0.035857026f, 3.475007f, -0.3416654f, 
    2.8354964f, 0.9961257f, -0.0056447755f, 3.4955933f, 1.7421336f, 2.571423f, 0.28003702f, -0.2826468f, 
    1.649162f, -0.3406625f, 2.869344f, 2.9216738f, 3.8200698f, 3.587048f, -0.034718625f, 0.025198033f, 2.7714105f,
     -0.078165896f, 3.453137f, 3.455557f, -0.19390273f, -0.30502352f, -0.00654175f, 2.6836565f, -0.22652344f, 
    -0.27297968f, -0.15204771f, -0.13844837f, -0.2574886f, 3.5509603f, 3.4010718f, -0.17557037f, 0.6529491f, 
    -0.2763976f, 1.8601494f, 1.4764289f, -0.39108914f, -0.3156588f, 1.1903664f, 2.683753f, 3.1480494f, -0.371396f,
     -0.30660808f, -0.29761833f, 1.9714252f, -0.17730962f, 2.409669f, 2.7738173f, 1.5323026f, 3.8726993f, 
    1.4933829f, -0.15895593f, 3.6559548f, 2.3233144f, 1.9393194f, -0.08760921f, 2.7103262f, 1.8685579f, 
    3.8130794f, 1.5535879f, 1.9012792f, 2.944067f, 0.38000348f, 1.9509495f, -0.1601986f, -0.3219304f, 1.923927f, 
    -0.07862248f, 2.2163744f, 0.008998514f, -0.08998683f, 1.4108609f, -0.24628736f, 3.2213886f, -0.23165321f, 
    2.6168575f, 1.975287f, 3.8832178f, 0.27048218f, 2.0273023f, 2.432747f, -0.35175195f, 3.2401395f, 3.8122609f, 
    1.2806444f, 3.5090775f, 3.3002853f, 1.009292f, 1.3647448f, 1.4804814f, 3.1808455f, 2.0768278f, 3.2460022f, 
    1.2692984f, 1.5676126f, -0.13965546f, -0.1544277f, 1.2516915f, 3.2490647f, -0.27865565f, 3.5846298f, 
    -0.07445458f, -0.369498f, 1.4749277f};

static const int input_2_fx8_frac = 5;

static const int input_2_sa_dim = -1;
static const float input_2_scale = 0.016761986538767815f;
static const float input_2_zero_point = 1.7460644245147705f;
static const int8_t input_2_scales_frac[] = {19};


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
static const float input_3_data[] = {
    51035.0f, 37503.0f, 5141.0f, 112202.0f, 55772.0f, 98830.0f, 72590.0f, 83633.0f, 13016.0f, 99245.0f,
    84714.0f, -5802.0f, 52382.0f, 3155.0f, 94029.0f, -11833.0f, 7529.0f, 96442.0f, 126398.0f, 19754.0f,
    -22277.0f, 17227.0f, -16372.0f, 26213.0f, 26673.0f, 73592.0f, 52758.0f, 23884.0f, 18598.0f, 56971.0f,
    26515.0f, -5908.0f, 92030.0f, 43698.0f, 29386.0f, 79323.0f, 111834.0f, -29272.0f, 102251.0f, 56578.0f,
    111880.0f, -14901.0f, 45796.0f, 17693.0f, 127967.0f, 90659.0f, 32155.0f, 75653.0f, 76592.0f, 114130.0f,
    88899.0f, -15226.0f, 46407.0f, 42421.0f, 84588.0f, 88884.0f, 28078.0f, 116193.0f, 109170.0f, -8395.0f,
    18691.0f, -26728.0f, 50091.0f, 112026.0f, -11656.0f, 72961.0f, 117989.0f, -27650.0f, 119814.0f, 86034.0f,
    56607.0f, 15584.0f, 17579.0f, -7900.0f, 47999.0
};

static const int input_3_fx8_frac = -10;

static const int input_3_sa_dim = -1;
static const float input_3_scale = 616.62353515625f;
static const float input_3_zero_point = 49347.5f;
static const int8_t input_3_scales_frac[] = {5};


#define INPUT_3_TSR_SHARED_DESCR \
    /* .data = */ {0},\
    /* .mem_stride =  */ {0},\
    /* .shape = */ {5, 5, 3}, \
    /* .rank = */ 3



static const mli_tensor input_3_tsr_fx16 = {
    INPUT_3_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_FX_16,
    /* .el_params = */ {0},
};


static const mli_tensor input_3_tsr_sa8 = {
    INPUT_3_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_SA_8,
    /* .el_params = */ {0},
};


static const float test_8_out_data[] = {
    61074.0f, 20225.0f, 54552.11f, 56245.445f, 38624.555f, 50278.0f, 64753.11f, 13429.333f,
    80382.555f, 69040.555f, 40646.555f, 51202.0
};

static const int test_8_out_fx8_frac = -10;

static const int test_8_out_sa_dim = -1;
static const float test_8_out_scale = 369.3381520669291f;
static const float test_8_out_zero_point = 46905.9453125f;
static const int8_t test_8_out_scales_frac[] = {6};


#define TEST_8_OUT_TSR_SHARED_DESCR \
    /* .data = */ {0},\
    /* .mem_stride =  */ {0},\
    /* .shape = */ {2, 2, 3}, \
    /* .rank = */ 3



static const mli_tensor test_8_out_tsr_fx16 = {
    TEST_8_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_FX_16,
    /* .el_params = */ {0},
};


static const mli_tensor test_8_out_tsr_sa8 = {
    TEST_8_OUT_TSR_SHARED_DESCR,

    /*.el_type = */ MLI_EL_SA_8,
    /* .el_params = */ {0},
};


static const float test_1_out_data[] = {
    -0.7274608f, -0.82200414f, -0.6518219f, -0.52102566f, -0.55328125f, -0.70961565f, -0.6087317f, -0.5917716f, 
    -0.8500264f, -0.55993056f, -0.53174335f, -1.0755409f, 0.3539277f, -0.60402817f, -1.0055257f, 0.6427222f, 
    -1.0874633f, -0.6698795f, -0.8228039f, -0.9275102f, -0.4590952f, -0.85928005f, -0.9260159f, -0.5925675f, 
    -0.93718463f, -1.0077845f, -0.8643983f, -0.86691386f, -0.9476204f, -1.0999044f, -0.9536629f, -0.9640861f, 
    -1.0228332f, -0.9877928f, -1.0004016f, -1.0501878f, -1.0955828f, -0.86852497f, -0.48093888f, -1.1139873f, 
    -0.82497644f, -0.40042472f, -1.0977678f, -0.94241947f, -0.61997855f, -1.019092f, -0.90912557f, -0.73170525f, 
    -0.96295327f, -1.0266465f, -0.5663385f, -0.92073154f, -1.0398668f, -0.86805576f, -0.5677765f, -0.98714584f, 
    -0.6383918f, -0.59568226f, -0.96711636f, -0.35124865f, -0.7425348f, -1.0658743f, -0.4963659f, -0.77652866f, 
    -0.8580299f, -0.46334204f, -0.8532273f, -1.0121686f, -0.3016803f, -0.757654f, -0.94284254f, -0.73418516f, 
    -0.52009386f, -0.8886239f, -0.6995642f, -0.55306077f, -0.9671047f, -0.29435602f, -0.709775f, -1.1672087f, 
    -0.43871236f, -0.74412465f, -1.0255961f, -0.4185157f, -0.7186293f, -1.045494f, -0.38847187f, -0.71337f, 
    -0.7984283f, -0.8328368f, -0.40098915f, -0.84406334f, -0.91192585f, -0.37764314f, -0.97377473f, -0.7206771f, 
    -0.56715554f, -1.1881474f, -0.73570305f, -0.64036566f, -1.1268371f, -0.73516136f, -0.7535588f, -1.2000264f, 
    -0.83580786f, -0.81885153f, -0.8691988f, -0.8592851f, -0.84250027f, -0.82584286f, -0.8856628f, -0.84115916f, 
    -0.8568272f, -0.8648258f, -0.7893283f, -1.0463793f, -0.9130947f, -0.6819749f, -0.93400335f, -0.95669895f, 
    -0.6551984f, -1.0314169f, -1.0552264f, -0.80562323f, -0.7248756f, -0.8674271f, -0.79756594f, -0.79181856f, 
    -0.90337205f, -0.81115055f, -0.8051219f, -0.959251f, -0.7368264f, -0.9003164f, -0.8310999f, -0.67813796f, 
    -0.7402739f, -0.8223334f, -0.73902434f, -0.9315955f, -0.82390267f, -0.7595373f, -0.8321367f, -0.652342f};


// Extracted Output vectors 
//===================================================
static const int test_1_out_fx8_frac = 6;

static const int test_1_out_sa_dim = -1;
static const float test_1_out_scale = 0.00722646526992321f;
static const float test_1_out_zero_point = -0.2786521315574646f;
static const int8_t test_1_out_scales_frac[] = {21};


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
    -0.82124543f, -0.76010084f, -0.7386064f, -0.60851043f, -0.9504584f, -0.54034626f, -0.6036566f, -0.48091874f, 
    -1.0946723f, 0.5580942f, -1.0181209f, -0.93489295f, 1.9543223f, -1.214057f, -0.3043826f, -0.84578484f, 
    -0.9443172f, -0.32722148f, -0.9852595f, -1.0277171f, -0.5417098f, -1.0569338f, -0.9577957f, -0.77582985f, 
    -0.91798097f, -1.114875f, -0.87864417f, -0.9493084f, -0.9633336f, -0.81408966f, -0.36529544f, -0.8131897f, 
    -0.5111303f, -0.60284907f, -0.9188316f, -0.7744484f, -0.65048146f, -1.0540577f, -0.5804473f, -0.8562321f, 
    -1.0533959f, -0.9036769f, -0.8372873f, -0.4966841f, -0.60215676f, -0.41319638f, -0.7284927f, -0.9569835f, 
    -0.3700428f, -0.8253343f, -0.83521175f, -0.6331319f, -1.0405108f, -0.7459362f, -0.51547396f, -0.9397137f, 
    -1.0342627f, -1.090369f, -0.61506253f, -0.643658f, -1.0307802f, -0.73965f, -0.9108591f, -0.6904709f, 
    -0.7258327f, -0.96510863f, -0.90169704f, -0.8079943f, -0.802503f, -0.5943025f, -0.8944641f, -0.7846443f, 
    -1.192638f, -0.9605661f, -0.5682958f};

static const int test_2_out_fx8_frac = 6;

static const int test_2_out_sa_dim = -1;
static const float test_2_out_scale = 0.012425016611814499f;
static const float test_2_out_zero_point = 0.3701326847076416f;
static const int8_t test_2_out_scales_frac[] = {20};


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
    -0.75548255f, -0.8148439f, -0.5311005f, -0.6036566f, -0.48091874f, -1.0946723f, 1.1599641f, -1.0374436f, 
    -0.5423039f, -0.6771192f, -0.99284625f, -0.43583474f, -0.69266796f, -0.956554f, -0.62348634f, -0.8929116f, 
    -0.8235538f, -0.69623166f, -0.42251906f, -0.8997714f, -0.8999738f, -0.6331319f, -1.0405108f, -0.7459362f, 
    -0.78720456f, -0.60656893f, -0.7485725f, -0.6884043f, -0.9120386f, -0.8151117f, -0.7053821f, -0.61914366f, 
    -0.7925622f, -0.6286218f, -0.4796468f, -0.5884315f};

static const int test_3_out_fx8_frac = 6;

static const int test_3_out_sa_dim = -1;
static const float test_3_out_scale = 0.008841711096465588f;
static const float test_3_out_zero_point = 0.03264594078063965f;
static const int8_t test_3_out_scales_frac[] = {20};


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
1.1616756f, 1.361278f, 1.0394869f, 1.4587243f, 1.4738181f, 1.1232897f, 2.160826f};

static const int test_4_out_fx8_frac = 5;

static const int test_4_out_sa_dim = -1;
static const float test_4_out_scale = 0.01259965690102164f;
static const float test_4_out_zero_point = 1.6001564264297485f;
static const int8_t test_4_out_scales_frac[] = {21};


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
    0.5713413f, 1.3024454f, 0.9718271f, 0.96239156f, 1.74222f, 1.3152938f, 2.187318f, 0.57523525f, 0.88908356f, 
    0.96132874f, 1.1918671f, 1.5359068f, 1.7171465f, 2.585535f, 1.3689876f, 1.8542868f, 1.486737f, 1.7442387f, 
    1.4368215f, 0.8981146f, 1.5983952f, 1.0996107f, 1.3433509f, 1.3842108f, 1.8284348f, 1.1154265f, 0.9103296f, 
    1.8026388f};

static const int test_5_out_fx8_frac = 5;

static const int test_5_out_sa_dim = -1;
static const float test_5_out_scale = 0.01242864695120984f;
static const float test_5_out_zero_point = 1.57843816280365f;
static const int8_t test_5_out_scales_frac[] = {20};


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
    -0.7417096f, -0.8219152f, -0.65701985f, -0.41424572f, -0.9881428f, -0.9570768f, -0.37367457f, 0.0671868f, 
    -0.9376856f, 2.1402435f, -0.94128025f, -1.0662088f, 2.2123318f, -1.3686919f, 1.2080815f, -1.0119568f, 
    -0.85158503f, -0.54481244f, -0.96114045f, -0.7469178f, -0.4332672f, -0.8932253f, -1.0857438f, -1.4476427f, 
    -0.9475249f, -0.87751466f, -0.5214672f, -1.1726973f, -0.6351488f, -1.3714589f, 0.16847408f, -0.9446336f, 
    -0.48873615f, -0.9251333f, -1.0742556f, -0.7897291f, -0.48637962f, -1.1450648f, 0.52622443f, -0.9373374f, 
    -0.8661735f, -0.47483808f, -0.4150476f, -0.8187977f, -0.81332046f, -1.0321571f, -0.6046342f, -1.0176818f, 
    -0.7278024f, -0.915739f, -0.9043678f, -1.0010889f, -1.4239801f, -1.0182623f, -0.88398f, -1.2815418f, 
    -1.2704535f, -1.5181077f, 0.3728767f, 0.18652491f, -0.8912799f, -0.65323764f, -0.7882086f, -0.54508114f, 
    -0.8488276f, -0.82008564f, -0.65431875f, -0.79470825f, -0.80321664f, -0.40985602f, -0.41979986f, -0.59657776f,
     -1.0661534f, -0.59934056f, -0.572139f};

static const int test_6_out_fx8_frac = 5;

static const int test_6_out_sa_dim = -1;
static const float test_6_out_scale = 0.014629174023866653f;
static const float test_6_out_zero_point = 0.3471120595932007f;
static const int8_t test_6_out_scales_frac[] = {20};


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
    -0.82124543f, -0.76010084f, -0.7386064f, -0.60851043f, -0.9504584f, -0.54034626f, -0.6036566f, -0.48091874f, 
    -1.0946723f, 0.5580942f, -1.0181209f, -0.93489295f, 1.9543223f, -1.214057f, -0.3043826f, -0.9897471f, 
    -0.92231816f, -0.3869953f, -1.1243275f, -0.87951875f, -0.56581414f, -1.0265375f, -0.9242558f, -0.78266966f, 
    -0.92294884f, -1.1484121f, -0.7870324f, -0.89045143f, -0.908434f, -0.7337594f, -0.13919507f, -0.728687f, 
    -0.6188919f, -0.48230278f, -0.9428231f, -0.8036763f, -0.65181965f, -1.1453661f, -0.33735788f, -0.7030608f, 
    -1.0473423f, -0.9291642f, -0.78033024f, -0.49783185f, -0.6206882f, -0.9783539f, -0.6961625f, -0.9739409f, 
    -0.82942635f, -0.7465589f, -0.87190264f, -0.9471562f, -1.0361981f, -0.9459362f, -0.5814788f, -0.9793452f, 
    -1.1257701f, -1.2341995f, -0.5926224f, -0.63335353f, -0.8912799f, -0.65323764f, -0.7882086f, -0.6696126f, 
    -0.672143f, -0.8019821f, -0.7053821f, -0.61914366f, -0.7925622f, -0.42798853f, -0.53119034f, -0.6759181f, 
    -1.0724003f, -0.57112646f, -0.46015334f};

static const int test_7_out_fx8_frac = 6;

static const int test_7_out_sa_dim = -1;
static const float test_7_out_scale = 0.012504007667303085f;
static const float test_7_out_zero_point = 0.3600614070892334f;
static const int8_t test_7_out_scales_frac[] = {20};


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
tensor_quantizer input_1_memstr_sa8(input_1_memstr_tsr_sa8, input_1_sa_dim,input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]),
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

tensor_quantizer input_2_memstr_sa8(input_2_memstr_tsr_sa8, input_2_sa_dim,input_2_data,
                                    sizeof(input_2_data) / sizeof(input_2_data[0]),
                                    &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);

tensor_quantizer input_3_fx16(input_3_tsr_fx16, input_3_fx8_frac + 8,
                              input_3_data, sizeof(input_3_data) / sizeof(input_3_data[0]));


tensor_quantizer input_3_sa8(input_3_tsr_sa8, input_3_sa_dim, input_3_data,
                             sizeof(input_3_data) / sizeof(input_3_data[0]),
                             &input_3_scale, 1, &input_3_zero_point, 1, input_3_scales_frac, 1);

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


tensor_quantizer test_8_out_fx16(test_8_out_tsr_fx16, test_8_out_fx8_frac + 8,
                                test_8_out_data, sizeof(test_8_out_data) / sizeof(test_8_out_data[0]));


tensor_quantizer test_8_out_sa8(test_8_out_tsr_sa8, test_8_out_sa_dim, test_8_out_data,
                                sizeof(test_8_out_data) / sizeof(test_8_out_data[0]),
                                &test_8_out_scale, 1, &test_8_out_zero_point, 1, test_8_out_scales_frac, 1);
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


const mli_pool_cfg test_8_cfg = {
    /* kernel_width = */3, /* kernel_height = */3, /* stride_width = */2, /* stride_height = */2,
    /* padding_left = */0, /* padding_right = */0, /* padding_top = */0, /* padding_bottom = */0
};
