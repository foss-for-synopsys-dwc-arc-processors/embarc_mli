
#include <stdint.h>

#include "mli_types.h"
#include "test_tensor_quantizer.h"

using mli::tst::tensor_quantizer;

extern mli::tst::tensor_quantizer input_1_fx16;
extern mli::tst::tensor_quantizer input_1_sa8;
extern mli::tst::tensor_quantizer test_1_out_fx16;
extern mli::tst::tensor_quantizer test_1_out_sa8;
extern const mli_l2_normalize_cfg test_1_cfg;
static const float input_1_data[] = {
     4.733333f, -2.733333f,  5.800000f,  2.066667f,  3.666667f,  -0.066667f, -1.933333f, -1.666667f,  2.333333f,  7.933333f, 
    -0.866667f,  1.800000f,  6.066667f,  9.000000f, 11.666667f,  10.066667f, 11.400000f, 11.933333f, 12.200000f, -2.466667f, 
     8.733334f,  2.866667f,  4.200000f,  7.400000f,  2.600000f,   1.000000f,  5.533333f,  8.200000f,  5.000000f,  8.466666f, 
     7.133333f,  5.266667f,  6.866667f,  3.933333f, -1.133333f,   3.133333f, 10.333333f,  7.666667f, 12.733334f, -0.600000f, 
     4.466667f,  0.733333f, 10.600000f, -3.000000f, 10.866667f,   0.200000f,  9.533334f, -2.200000f, -0.333333f,  6.600000f, 
     3.400000f,  1.533333f,  9.800000f,  0.466667f,  1.266667f,  12.466666f,  9.266666f, 11.133333f, -1.400000f,  6.333333f};

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

tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim, test_1_out_data, 
sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_1_out_scale, 1, &test_1_out_zero_point, 1, test_1_out_scales_frac, 1);

const mli_l2_normalize_cfg test_1_cfg = {/*Axis=*/ -1};
