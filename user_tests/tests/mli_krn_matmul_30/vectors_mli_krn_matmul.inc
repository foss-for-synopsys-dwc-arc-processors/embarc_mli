
#include <stdint.h>

#include "mli_types.h"
#include "test_tensor_quantizer.h"

using mli::tst::tensor_quantizer;

extern mli::tst::tensor_quantizer input_1_sa8;
extern mli::tst::tensor_quantizer input_1_fx8;
extern mli::tst::tensor_quantizer input_1_fx16;
extern mli::tst::tensor_quantizer input_1_sa32;
extern mli::tst::tensor_quantizer input_2_sa8;
extern mli::tst::tensor_quantizer input_2_fx8;
extern mli::tst::tensor_quantizer input_2_fx16;
extern mli::tst::tensor_quantizer input_2_sa32;
extern mli::tst::tensor_quantizer input_3_sa8;
extern mli::tst::tensor_quantizer input_3_fx8;
extern mli::tst::tensor_quantizer input_3_fx16;
extern mli::tst::tensor_quantizer input_3_sa32;
extern mli::tst::tensor_quantizer input_4_sa8;
extern mli::tst::tensor_quantizer input_4_fx8;
extern mli::tst::tensor_quantizer input_4_fx16;
extern mli::tst::tensor_quantizer input_4_sa32;
extern mli::tst::tensor_quantizer input_5_sa8;
extern mli::tst::tensor_quantizer input_5_fx8;
extern mli::tst::tensor_quantizer input_5_fx16;
extern mli::tst::tensor_quantizer input_5_sa32;
extern mli::tst::tensor_quantizer input_6_sa8;
extern mli::tst::tensor_quantizer input_6_fx8;
extern mli::tst::tensor_quantizer input_6_fx16;
extern mli::tst::tensor_quantizer input_6_sa32;
extern mli::tst::tensor_quantizer input_7_sa8;
extern mli::tst::tensor_quantizer input_7_fx8;
extern mli::tst::tensor_quantizer input_7_fx16;
extern mli::tst::tensor_quantizer input_7_sa32;
extern mli::tst::tensor_quantizer input_8_sa8;
extern mli::tst::tensor_quantizer input_8_fx8;
extern mli::tst::tensor_quantizer input_8_fx16;
extern mli::tst::tensor_quantizer input_8_sa32;
extern mli::tst::tensor_quantizer test_1_out_sa8;
extern mli::tst::tensor_quantizer test_1_out_fx8;
extern mli::tst::tensor_quantizer test_1_out_fx16;
extern mli::tst::tensor_quantizer test_1_out_sa32;
extern mli::tst::tensor_quantizer test_2_out_sa8;
extern mli::tst::tensor_quantizer test_2_out_fx8;
extern mli::tst::tensor_quantizer test_2_out_fx16;
extern mli::tst::tensor_quantizer test_2_out_sa32;
extern mli::tst::tensor_quantizer test_3_out_sa8;
extern mli::tst::tensor_quantizer test_3_out_fx8;
extern mli::tst::tensor_quantizer test_3_out_fx16;
extern mli::tst::tensor_quantizer test_3_out_sa32;
extern mli::tst::tensor_quantizer test_4_out_sa8;
extern mli::tst::tensor_quantizer test_4_out_fx8;
extern mli::tst::tensor_quantizer test_4_out_fx16;
extern mli::tst::tensor_quantizer test_4_out_sa32;
extern mli::tst::tensor_quantizer test_5_out_sa8;
extern mli::tst::tensor_quantizer test_5_out_fx8;
extern mli::tst::tensor_quantizer test_5_out_fx16;
extern mli::tst::tensor_quantizer test_5_out_sa32;
extern mli::tst::tensor_quantizer test_6_out_sa8;
extern mli::tst::tensor_quantizer test_6_out_fx8;
extern mli::tst::tensor_quantizer test_6_out_fx16;
extern mli::tst::tensor_quantizer test_6_out_sa32;

static const float input_1_data[] = {48.000000f, 69.000000f, -102.000000f, 119.000000f};

static const float input_1_scale = 1.0f;
static const float input_1_zero_point = 0.0f;
static const int8_t input_1_scales_frac[] = {0};
static const int input_1_sa_dim = -1;

static const int input_1_fx8_frac = 0;

#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 1}, \
    /* .rank =  */ 2


static const mli_tensor input_1_tsr_sa8 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
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
    /* .el_params = */ { 0 }
};


static const mli_tensor input_1_tsr_sa32 = {
    INPUT_1_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float input_2_data[] = {-60.000000f, 84.000000f, 24.000000f, -24.000000f, -35.000000f, 58.000000f};

static const float input_2_scale = 1.0f;
static const float input_2_zero_point = 0.0f;
static const int8_t input_2_scales_frac[] = {0};
static const int input_2_sa_dim = -1;

static const int input_2_fx8_frac = 0;

#define INPUT_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 6}, \
    /* .rank =  */ 2


static const mli_tensor input_2_tsr_sa8 = {
    INPUT_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_2_tsr_fx8 = {
    INPUT_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_2_tsr_fx16 = {
    INPUT_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_2_tsr_sa32 = {
    INPUT_2_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float input_3_data[] = {15.000000f, -104.000000f, -55.000000f, -38.000000f, -17.000000f, -85.000000f, 91.000000f, 9.000000f, 40.000000f, 103.000000f, -59.000000f, 49.000000f, 0.000000f, 8.000000f, 45.000000f, -127.000000f, -52.000000f, -72.000000f};

static const float input_3_scale = 1.0f;
static const float input_3_zero_point = 0.0f;
static const int8_t input_3_scales_frac[] = {0};
static const int input_3_sa_dim = -1;

static const int input_3_fx8_frac = 0;

#define INPUT_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 9}, \
    /* .rank =  */ 2


static const mli_tensor input_3_tsr_sa8 = {
    INPUT_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_3_tsr_fx8 = {
    INPUT_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_3_tsr_fx16 = {
    INPUT_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_3_tsr_sa32 = {
    INPUT_3_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float input_4_data[] = {30.000000f, 13.000000f, 77.000000f, -48.000000f, -98.000000f, 9.000000f, -73.000000f, 39.000000f, -10.000000f, 71.000000f, 22.000000f, -83.000000f, 92.000000f, 8.000000f, 82.000000f, 107.000000f, 112.000000f, 70.000000f, -27.000000f, 53.000000f, 106.000000f, 42.000000f, -98.000000f, -101.000000f, -63.000000f, 125.000000f, 63.000000f, 87.000000f, 99.000000f, -115.000000f, 7.000000f, 105.000000f, 35.000000f, 60.000000f, -21.000000f, -17.000000f, -11.000000f, 114.000000f, -34.000000f, 103.000000f, 107.000000f, -110.000000f, -122.000000f, -48.000000f, -78.000000f};

static const float input_4_scale = 1.0f;
static const float input_4_zero_point = 0.0f;
static const int8_t input_4_scales_frac[] = {0};
static const int input_4_sa_dim = -1;

static const int input_4_fx8_frac = 0;

#define INPUT_4_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {9, 5}, \
    /* .rank =  */ 2


static const mli_tensor input_4_tsr_sa8 = {
    INPUT_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_4_tsr_fx8 = {
    INPUT_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_4_tsr_fx16 = {
    INPUT_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_4_tsr_sa32 = {
    INPUT_4_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float input_5_data[] = {8.000000f, 24.000000f, -35.000000f, 120.000000f, 97.000000f, 22.000000f, -27.000000f, 36.000000f, 98.000000f, -60.000000f, -4.000000f, 92.000000f};

static const float input_5_scale = 1.0f;
static const float input_5_zero_point = 0.0f;
static const int8_t input_5_scales_frac[] = {0};
static const int input_5_sa_dim = -1;

static const int input_5_fx8_frac = 0;

#define INPUT_5_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 4}, \
    /* .rank =  */ 2


static const mli_tensor input_5_tsr_sa8 = {
    INPUT_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_5_tsr_fx8 = {
    INPUT_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_5_tsr_fx16 = {
    INPUT_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_5_tsr_sa32 = {
    INPUT_5_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float input_6_data[] = {98.000000f, -108.000000f, 28.000000f, -123.000000f, -108.000000f, -94.000000f, -4.000000f, -97.000000f, 58.000000f, 102.000000f, -54.000000f, 94.000000f, 82.000000f, 0.000000f, 32.000000f, 3.000000f, -44.000000f, 51.000000f, 19.000000f, -106.000000f};

static const float input_6_scale = 1.0f;
static const float input_6_zero_point = 0.0f;
static const int8_t input_6_scales_frac[] = {0};
static const int input_6_sa_dim = -1;

static const int input_6_fx8_frac = 0;

#define INPUT_6_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 5}, \
    /* .rank =  */ 2


static const mli_tensor input_6_tsr_sa8 = {
    INPUT_6_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_6_tsr_fx8 = {
    INPUT_6_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_6_tsr_fx16 = {
    INPUT_6_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_6_tsr_sa32 = {
    INPUT_6_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float input_7_data[] = {-8.000000f, -60.000000f, 76.000000f, 59.000000f, -80.000000f, -111.000000f, -115.000000f, -34.000000f, -20.000000f, 106.000000f, 10.000000f, -51.000000f, -60.000000f, -17.000000f, 15.000000f, -3.000000f, -107.000000f, -74.000000f, 125.000000f, 16.000000f, -68.000000f, -92.000000f, -79.000000f, -123.000000f, -84.000000f, 73.000000f, 91.000000f, 123.000000f, -73.000000f, -84.000000f, 112.000000f, 88.000000f, 78.000000f, 51.000000f, -88.000000f, 4.000000f};

static const float input_7_scale = 1.0f;
static const float input_7_zero_point = 0.0f;
static const int8_t input_7_scales_frac[] = {0};
static const int input_7_sa_dim = -1;

static const int input_7_fx8_frac = 0;

#define INPUT_7_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 9}, \
    /* .rank =  */ 2


static const mli_tensor input_7_tsr_sa8 = {
    INPUT_7_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_7_tsr_fx8 = {
    INPUT_7_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_7_tsr_fx16 = {
    INPUT_7_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_7_tsr_sa32 = {
    INPUT_7_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float input_8_data[] = {30.000000f, 13.000000f, 77.000000f, -48.000000f, -98.000000f, 9.000000f, -73.000000f, 39.000000f, -10.000000f, 71.000000f, 22.000000f, -83.000000f, 92.000000f, 8.000000f, 82.000000f, 107.000000f, 112.000000f, 70.000000f, -27.000000f, 53.000000f, 106.000000f, 42.000000f, -98.000000f, -101.000000f, -63.000000f, 125.000000f, 63.000000f, 87.000000f, 99.000000f, -115.000000f, 7.000000f, 105.000000f, 35.000000f, 60.000000f, -21.000000f, -17.000000f, -11.000000f, 114.000000f, -34.000000f, 103.000000f, 107.000000f, -110.000000f, -122.000000f, -48.000000f, -78.000000f};

static const float input_8_scale = 1.0f;
static const float input_8_zero_point = 0.0f;
static const int8_t input_8_scales_frac[] = {0};
static const int input_8_sa_dim = -1;

static const int input_8_fx8_frac = 0;

#define INPUT_8_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {9, 5}, \
    /* .rank =  */ 2


static const mli_tensor input_8_tsr_sa8 = {
    INPUT_8_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_8_tsr_fx8 = {
    INPUT_8_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_8_tsr_fx16 = {
    INPUT_8_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor input_8_tsr_sa32 = {
    INPUT_8_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_1_out_data[] = {-2880.000000f, 4032.000000f, 1152.000000f, -1152.000000f, -1680.000000f, 2784.000000f, -4140.000000f, 5796.000000f, 1656.000000f, -1656.000000f, -2415.000000f, 4002.000000f, 6120.000000f, -8568.000000f, -2448.000000f, 2448.000000f, 3570.000000f, -5916.000000f, -7140.000000f, 9996.000000f, 2856.000000f, -2856.000000f, -4165.000000f, 6902.000000f};

static const float test_1_out_scale = 1.0f;
static const float test_1_out_zero_point = 0.0f;
static const int8_t test_1_out_scales_frac[] = {0};
static const int test_1_out_sa_dim = -1;

static const int test_1_out_fx8_frac = -7;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 6}, \
    /* .rank =  */ 2


static const mli_tensor test_1_out_tsr_sa8 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_1_out_tsr_fx8 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_1_out_tsr_fx16 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_1_out_tsr_sa32 = {
    TEST_1_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_2_out_data[] = {-13425.000000f, 7083.000000f, -17019.000000f, -2558.000000f, -8636.000000f, 2401.000000f, -93.000000f, 11680.000000f, -2711.000000f, -13017.000000f};

static const float test_2_out_scale = 1.0f;
static const float test_2_out_zero_point = 0.0f;
static const int8_t test_2_out_scales_frac[] = {0};
static const int test_2_out_sa_dim = -1;

static const int test_2_out_fx8_frac = -8;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 5}, \
    /* .rank =  */ 2


static const mli_tensor test_2_out_tsr_sa8 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_2_out_tsr_fx8 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_2_out_tsr_fx16 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_2_out_tsr_sa32 = {
    TEST_2_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_3_out_data[] = {778.000000f, -9530.000000f, 1146.000000f, 2688.000000f, -12256.000000f, 9004.000000f, -14686.000000f, 204.000000f, -9971.000000f, -12912.000000f, 15736.000000f, -14768.000000f, 12928.000000f, -13786.000000f, -26584.000000f};

static const float test_3_out_scale = 1.0f;
static const float test_3_out_zero_point = 0.0f;
static const int8_t test_3_out_scales_frac[] = {0};
static const int test_3_out_sa_dim = -1;

static const int test_3_out_fx8_frac = -8;

#define TEST_3_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 5}, \
    /* .rank =  */ 2


static const mli_tensor test_3_out_tsr_sa8 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_3_out_tsr_fx8 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_3_out_tsr_fx16 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_3_out_tsr_sa32 = {
    TEST_3_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_4_out_data[] = {-17517.000000f, -15278.000000f, 888.000000f, -7694.000000f, 24161.000000f, -10319.000000f, 7394.000000f, -644.000000f, 6236.000000f, -22880.000000f, -23287.000000f, -34903.000000f, -11126.000000f, -20308.000000f, -259.000000f, 34528.000000f, 40937.000000f, -3837.000000f, -4176.000000f, -43150.000000f};

static const float test_4_out_scale = 1.0f;
static const float test_4_out_zero_point = 0.0f;
static const int8_t test_4_out_scales_frac[] = {0};
static const int test_4_out_sa_dim = -1;

static const int test_4_out_fx8_frac = -9;

#define TEST_4_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 5}, \
    /* .rank =  */ 2


static const mli_tensor test_4_out_tsr_sa8 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_4_out_tsr_fx8 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_4_out_tsr_fx16 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_4_out_tsr_sa32 = {
    TEST_4_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_5_out_data[] = {-13425.000000f, 7083.000000f, -17019.000000f, -2558.000000f, -8636.000000f, 2401.000000f, -93.000000f, 11680.000000f, -2711.000000f, -13017.000000f};

static const float test_5_out_scale = 1.0f;
static const float test_5_out_zero_point = 0.0f;
static const int8_t test_5_out_scales_frac[] = {0};
static const int test_5_out_sa_dim = -1;

static const int test_5_out_fx8_frac = -8;

#define TEST_5_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 5}, \
    /* .rank =  */ 2


static const mli_tensor test_5_out_tsr_sa8 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_5_out_tsr_fx8 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_5_out_tsr_fx16 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_5_out_tsr_sa32 = {
    TEST_5_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_6_out_data[] = {12865.000000f, -9560.000000f, -8316.000000f, 15692.000000f, 12277.000000f, 13137.000000f, 8068.000000f, -15955.000000f, -4641.000000f, 2609.000000f, -8660.000000f, 5062.000000f, 10919.000000f, -2833.000000f, -4308.000000f, -7117.000000f, -10791.000000f, -5881.000000f, 3672.000000f, -13260.000000f, 3052.000000f, 20054.000000f, 1592.000000f, -4110.000000f, -6062.000000f, -5300.000000f, 2484.000000f};

static const float test_6_out_scale = 1.0f;
static const float test_6_out_zero_point = 0.0f;
static const int8_t test_6_out_scales_frac[] = {0};
static const int test_6_out_sa_dim = -1;

static const int test_6_out_fx8_frac = -8;

#define TEST_6_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {3, 9}, \
    /* .rank =  */ 2


static const mli_tensor test_6_out_tsr_sa8 = {
    TEST_6_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_6_out_tsr_fx8 = {
    TEST_6_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_6_out_tsr_fx16 = {
    TEST_6_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_6_out_tsr_sa32 = {
    TEST_6_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

tensor_quantizer input_1_sa8(input_1_tsr_sa8, input_1_sa_dim, input_1_data, 
sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);
tensor_quantizer input_1_fx8(input_1_tsr_fx8, input_1_fx8_frac, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_fx16(input_1_tsr_fx16, input_1_fx8_frac + 8, input_1_data, sizeof(input_1_data) / sizeof(input_1_data[0]));
tensor_quantizer input_1_sa32(input_1_tsr_sa32, input_1_sa_dim, input_1_data, 
sizeof(input_1_data) / sizeof(input_1_data[0]), &input_1_scale, 1, &input_1_zero_point, 1, input_1_scales_frac, 1);

tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data, 
sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);
tensor_quantizer input_2_fx8(input_2_tsr_fx8, input_2_fx8_frac, input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]));
tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx8_frac + 8, input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]));
tensor_quantizer input_2_sa32(input_2_tsr_sa32, input_2_sa_dim, input_2_data, 
sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);

tensor_quantizer input_3_sa8(input_3_tsr_sa8, input_3_sa_dim, input_3_data, 
sizeof(input_3_data) / sizeof(input_3_data[0]), &input_3_scale, 1, &input_3_zero_point, 1, input_3_scales_frac, 1);
tensor_quantizer input_3_fx8(input_3_tsr_fx8, input_3_fx8_frac, input_3_data, sizeof(input_3_data) / sizeof(input_3_data[0]));
tensor_quantizer input_3_fx16(input_3_tsr_fx16, input_3_fx8_frac + 8, input_3_data, sizeof(input_3_data) / sizeof(input_3_data[0]));
tensor_quantizer input_3_sa32(input_3_tsr_sa32, input_3_sa_dim, input_3_data, 
sizeof(input_3_data) / sizeof(input_3_data[0]), &input_3_scale, 1, &input_3_zero_point, 1, input_3_scales_frac, 1);

tensor_quantizer input_4_sa8(input_4_tsr_sa8, input_4_sa_dim, input_4_data, 
sizeof(input_4_data) / sizeof(input_4_data[0]), &input_4_scale, 1, &input_4_zero_point, 1, input_4_scales_frac, 1);
tensor_quantizer input_4_fx8(input_4_tsr_fx8, input_4_fx8_frac, input_4_data, sizeof(input_4_data) / sizeof(input_4_data[0]));
tensor_quantizer input_4_fx16(input_4_tsr_fx16, input_4_fx8_frac + 8, input_4_data, sizeof(input_4_data) / sizeof(input_4_data[0]));
tensor_quantizer input_4_sa32(input_4_tsr_sa32, input_4_sa_dim, input_4_data, 
sizeof(input_4_data) / sizeof(input_4_data[0]), &input_4_scale, 1, &input_4_zero_point, 1, input_4_scales_frac, 1);

tensor_quantizer input_5_sa8(input_5_tsr_sa8, input_5_sa_dim, input_5_data, 
sizeof(input_5_data) / sizeof(input_5_data[0]), &input_5_scale, 1, &input_5_zero_point, 1, input_5_scales_frac, 1);
tensor_quantizer input_5_fx8(input_5_tsr_fx8, input_5_fx8_frac, input_5_data, sizeof(input_5_data) / sizeof(input_5_data[0]));
tensor_quantizer input_5_fx16(input_5_tsr_fx16, input_5_fx8_frac + 8, input_5_data, sizeof(input_5_data) / sizeof(input_5_data[0]));
tensor_quantizer input_5_sa32(input_5_tsr_sa32, input_5_sa_dim, input_5_data, 
sizeof(input_5_data) / sizeof(input_5_data[0]), &input_5_scale, 1, &input_5_zero_point, 1, input_5_scales_frac, 1);

tensor_quantizer input_6_sa8(input_6_tsr_sa8, input_6_sa_dim, input_6_data, 
sizeof(input_6_data) / sizeof(input_6_data[0]), &input_6_scale, 1, &input_6_zero_point, 1, input_6_scales_frac, 1);
tensor_quantizer input_6_fx8(input_6_tsr_fx8, input_6_fx8_frac, input_6_data, sizeof(input_6_data) / sizeof(input_6_data[0]));
tensor_quantizer input_6_fx16(input_6_tsr_fx16, input_6_fx8_frac + 8, input_6_data, sizeof(input_6_data) / sizeof(input_6_data[0]));
tensor_quantizer input_6_sa32(input_6_tsr_sa32, input_6_sa_dim, input_6_data, 
sizeof(input_6_data) / sizeof(input_6_data[0]), &input_6_scale, 1, &input_6_zero_point, 1, input_6_scales_frac, 1);

tensor_quantizer input_7_sa8(input_7_tsr_sa8, input_7_sa_dim, input_7_data, 
sizeof(input_7_data) / sizeof(input_7_data[0]), &input_7_scale, 1, &input_7_zero_point, 1, input_7_scales_frac, 1);
tensor_quantizer input_7_fx8(input_7_tsr_fx8, input_7_fx8_frac, input_7_data, sizeof(input_7_data) / sizeof(input_7_data[0]));
tensor_quantizer input_7_fx16(input_7_tsr_fx16, input_7_fx8_frac + 8, input_7_data, sizeof(input_7_data) / sizeof(input_7_data[0]));
tensor_quantizer input_7_sa32(input_7_tsr_sa32, input_7_sa_dim, input_7_data, 
sizeof(input_7_data) / sizeof(input_7_data[0]), &input_7_scale, 1, &input_7_zero_point, 1, input_7_scales_frac, 1);

tensor_quantizer input_8_sa8(input_8_tsr_sa8, input_8_sa_dim, input_8_data, 
sizeof(input_8_data) / sizeof(input_8_data[0]), &input_8_scale, 1, &input_8_zero_point, 1, input_8_scales_frac, 1);
tensor_quantizer input_8_fx8(input_8_tsr_fx8, input_8_fx8_frac, input_8_data, sizeof(input_8_data) / sizeof(input_8_data[0]));
tensor_quantizer input_8_fx16(input_8_tsr_fx16, input_8_fx8_frac + 8, input_8_data, sizeof(input_8_data) / sizeof(input_8_data[0]));
tensor_quantizer input_8_sa32(input_8_tsr_sa32, input_8_sa_dim, input_8_data, 
sizeof(input_8_data) / sizeof(input_8_data[0]), &input_8_scale, 1, &input_8_zero_point, 1, input_8_scales_frac, 1);

tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim, test_1_out_data, 
sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_1_out_scale, 1, &test_1_out_zero_point, 1, test_1_out_scales_frac, 1);
tensor_quantizer test_1_out_fx8(test_1_out_tsr_fx8, test_1_out_fx8_frac, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_sa32(test_1_out_tsr_sa32, test_1_out_sa_dim, test_1_out_data, 
sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), &test_1_out_scale, 1, &test_1_out_zero_point, 1, test_1_out_scales_frac, 1);

tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_sa_dim, test_2_out_data, 
sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &test_2_out_scale, 1, &test_2_out_zero_point, 1, test_2_out_scales_frac, 1);
tensor_quantizer test_2_out_fx8(test_2_out_tsr_fx8, test_2_out_fx8_frac, test_2_out_data, sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_fx8_frac + 8, test_2_out_data, sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_sa32(test_2_out_tsr_sa32, test_2_out_sa_dim, test_2_out_data, 
sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &test_2_out_scale, 1, &test_2_out_zero_point, 1, test_2_out_scales_frac, 1);

tensor_quantizer test_3_out_sa8(test_3_out_tsr_sa8, test_3_out_sa_dim, test_3_out_data, 
sizeof(test_3_out_data) / sizeof(test_3_out_data[0]), &test_3_out_scale, 1, &test_3_out_zero_point, 1, test_3_out_scales_frac, 1);
tensor_quantizer test_3_out_fx8(test_3_out_tsr_fx8, test_3_out_fx8_frac, test_3_out_data, sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));
tensor_quantizer test_3_out_fx16(test_3_out_tsr_fx16, test_3_out_fx8_frac + 8, test_3_out_data, sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));
tensor_quantizer test_3_out_sa32(test_3_out_tsr_sa32, test_3_out_sa_dim, test_3_out_data, 
sizeof(test_3_out_data) / sizeof(test_3_out_data[0]), &test_3_out_scale, 1, &test_3_out_zero_point, 1, test_3_out_scales_frac, 1);

tensor_quantizer test_4_out_sa8(test_4_out_tsr_sa8, test_4_out_sa_dim, test_4_out_data, 
sizeof(test_4_out_data) / sizeof(test_4_out_data[0]), &test_4_out_scale, 1, &test_4_out_zero_point, 1, test_4_out_scales_frac, 1);
tensor_quantizer test_4_out_fx8(test_4_out_tsr_fx8, test_4_out_fx8_frac, test_4_out_data, sizeof(test_4_out_data) / sizeof(test_4_out_data[0]));
tensor_quantizer test_4_out_fx16(test_4_out_tsr_fx16, test_4_out_fx8_frac + 8, test_4_out_data, sizeof(test_4_out_data) / sizeof(test_4_out_data[0]));
tensor_quantizer test_4_out_sa32(test_4_out_tsr_sa32, test_4_out_sa_dim, test_4_out_data, 
sizeof(test_4_out_data) / sizeof(test_4_out_data[0]), &test_4_out_scale, 1, &test_4_out_zero_point, 1, test_4_out_scales_frac, 1);

tensor_quantizer test_5_out_sa8(test_5_out_tsr_sa8, test_5_out_sa_dim, test_5_out_data, 
sizeof(test_5_out_data) / sizeof(test_5_out_data[0]), &test_5_out_scale, 1, &test_5_out_zero_point, 1, test_5_out_scales_frac, 1);
tensor_quantizer test_5_out_fx8(test_5_out_tsr_fx8, test_5_out_fx8_frac, test_5_out_data, sizeof(test_5_out_data) / sizeof(test_5_out_data[0]));
tensor_quantizer test_5_out_fx16(test_5_out_tsr_fx16, test_5_out_fx8_frac + 8, test_5_out_data, sizeof(test_5_out_data) / sizeof(test_5_out_data[0]));
tensor_quantizer test_5_out_sa32(test_5_out_tsr_sa32, test_5_out_sa_dim, test_5_out_data, 
sizeof(test_5_out_data) / sizeof(test_5_out_data[0]), &test_5_out_scale, 1, &test_5_out_zero_point, 1, test_5_out_scales_frac, 1);

tensor_quantizer test_6_out_sa8(test_6_out_tsr_sa8, test_6_out_sa_dim, test_6_out_data, 
sizeof(test_6_out_data) / sizeof(test_6_out_data[0]), &test_6_out_scale, 1, &test_6_out_zero_point, 1, test_6_out_scales_frac, 1);
tensor_quantizer test_6_out_fx8(test_6_out_tsr_fx8, test_6_out_fx8_frac, test_6_out_data, sizeof(test_6_out_data) / sizeof(test_6_out_data[0]));
tensor_quantizer test_6_out_fx16(test_6_out_tsr_fx16, test_6_out_fx8_frac + 8, test_6_out_data, sizeof(test_6_out_data) / sizeof(test_6_out_data[0]));
tensor_quantizer test_6_out_sa32(test_6_out_tsr_sa32, test_6_out_sa_dim, test_6_out_data, 
sizeof(test_6_out_data) / sizeof(test_6_out_data[0]), &test_6_out_scale, 1, &test_6_out_zero_point, 1, test_6_out_scales_frac, 1);

