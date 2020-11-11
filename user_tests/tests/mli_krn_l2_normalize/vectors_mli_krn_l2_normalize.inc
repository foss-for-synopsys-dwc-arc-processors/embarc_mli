
#include <stdint.h>

#include "mli_types.h"
#include "test_tensor_quantizer.h"

using mli::tst::tensor_quantizer;

extern mli::tst::tensor_quantizer input_1_fx16;
extern mli::tst::tensor_quantizer input_1_sa8;
extern mli::tst::tensor_quantizer test_1_out_fx16;
extern mli::tst::tensor_quantizer test_1_out_sa8;
extern mli::tst::tensor_quantizer test_2_out_fx16;
extern mli::tst::tensor_quantizer test_2_out_sa8;
extern mli::tst::tensor_quantizer test_3_out_fx16;
extern mli::tst::tensor_quantizer test_3_out_sa8;
extern const mli_l2_normalize_cfg test_1_cfg;
extern const mli_l2_normalize_cfg test_2_cfg;
extern const mli_l2_normalize_cfg test_3_cfg;
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

tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim, test_1_out_data, 
sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_1_out_scale, 1, &test_1_out_zero_point, 1, test_1_out_scales_frac, 1);

tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_fx8_frac + 8, test_2_out_data, sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_sa_dim, test_2_out_data, 
sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &test_2_out_scale, 1, &test_2_out_zero_point, 1, test_2_out_scales_frac, 1);

tensor_quantizer test_3_out_fx16(test_3_out_tsr_fx16, test_3_out_fx8_frac + 8, test_3_out_data, sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));
tensor_quantizer test_3_out_sa8(test_3_out_tsr_sa8, test_3_out_sa_dim, test_3_out_data, 
sizeof(test_3_out_data) / sizeof(test_3_out_data[0]), &test_3_out_scale, 1, &test_3_out_zero_point, 1, test_3_out_scales_frac, 1);

const mli_l2_normalize_cfg test_1_cfg = {/*axis=*/-1};
const mli_l2_normalize_cfg test_2_cfg = {/*axis=*/2};
const mli_l2_normalize_cfg test_3_cfg = {/*axis=*/0};
