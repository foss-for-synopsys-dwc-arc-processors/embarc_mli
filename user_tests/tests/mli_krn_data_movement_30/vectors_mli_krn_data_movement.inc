/*
* Copyright 2021, Synopsys, Inc.
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

#define number_of_configs 18

static int32_t tile_sizes[number_of_configs][5] = {
    {0},    {0},    {0},          {0},          {0},          {0},
    {4, 8}, {8, 8}, {1, 1, 4, 2}, {1, 1, 4, 2}, {1, 6, 2, 0}, {1, 6, 2, 0}, {4, 8}, {4, 8}, {8, 4}, {8, 3}};

static uint32_t tile_first_increments[number_of_configs][5] = {
    {0},    {0},    {0},          {0},          {0},          {0},
    {4, 8}, {8, 8}, {1, 1, 4, 2}, {1, 1, 4, 2}, {1, 6, 2, 0}, {1, 6, 2, 0}, {3, 8}, {4, 8}, {8, 4}, {8, 2}};

static int32_t out_mem_stride_cfg[number_of_configs][5] = {
    {0},
    {0},
    {0},
    {0},
    {33, 4 ,1},
    {33, 4 ,1},
    {0},
    {0},
    {0},
    {0},
    {33, 4 ,1},
    {33, 4 ,1}, {0}, {0}, {0}, {0}};

static uint32_t offsets_cfg[number_of_configs][5] = {
    {0}, {0}, {0}, {0}, {0}, {0},
    {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};

static uint32_t sizes_cfg[number_of_configs][5] = {
    {0}, {0}, {0}, {0}, {0}, {0},
    {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};

static int32_t sub_sample[number_of_configs][5] = {
    {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};

static uint32_t out_offsets_cfg[number_of_configs][5] = {
    {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};

static int32_t tiling_order[number_of_configs][5] = {
     {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {1, 0, 2, 3, 4}, {1, 0, 2, 3, 4}};

static int32_t tiling_count[number_of_configs][5] = {
    {1, 1, 1, 1, 1},  {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, {2, 1, 1, 1, 1}, {2, 8, 1, 1, 1}, {2, 8, 1, 1, 1}, {4, 2, 2, 1, 1}, {4, 2, 2, 1, 1}, {6, 1, 1, 1, 1}, {4, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, {2, 4, 1, 1, 1}};

static int32_t tiling_last_pos_inc[number_of_configs][5] = {
    {0},    {0},    {0},          {0},          {0},          {0},
    {4, 8}, {8, 8}, {1, 1, 4, 2}, {1, 1, 4, 2}, {1, 2, 2, 0}, {1, 2, 2, 0}, {1, 8}, {4, 8}, {8, 4}, {8, 2}};

static const float input_1_data[] = {8.286820f, 2.152027f, 5.760523f, -9.209137f, -9.439082f, 2.777630f, 0.086691f, -9.923459f, -6.820004f, -1.362363f, 1.118086f, 2.201777f, 0.450819f, -3.447689f, -4.574860f, 0.236707f, 0.707713f, 5.680774f, -1.140300f, 8.814334f, 5.280867f, -2.641193f, 8.735827f, 3.382713f, -0.028107f, -1.362890f, -3.054990f, -6.572820f, 0.709137f, -7.732485f, -1.467243f, -0.805623f, 8.689197f, -2.427501f, 0.179479f, 5.904660f, 8.859151f, 1.512121f, -9.198095f, 4.772271f, 8.437635f, 3.685844f, 3.373433f, 3.015400f, -1.919783f, 0.743432f, -9.499326f, 7.141538f, 8.711848f, -9.691744f, 9.559444f, 5.302596f, 9.618321f, 9.830359f, -6.855545f, 1.136173f, -2.964916f, -5.046844f, 3.412143f, 0.712972f, -1.474889f, 7.401769f, 8.923663f, 8.486315f, -8.937352f, -6.372196f, 3.259260f, 5.393515f, -4.797753f, -4.907107f, -7.777387f, -2.044802f, -3.196458f, 0.084552f, -5.785887f, 0.653820f, 4.551648f, -6.382569f, 0.562406f, 1.477235f, 2.541792f, 0.210185f, 2.538625f, 2.979376f, -1.668876f, 3.617727f, -0.526883f, -8.592014f, 6.895158f, 3.578593f, -5.122587f, -5.243422f, -9.385137f, 3.870499f, 3.685569f, -2.745420f, -0.795467f, 2.039002f, -8.150185f, -6.949403f, 3.948650f, 1.120372f, 4.190713f, 8.637361f, 8.664962f, -5.665703f, 7.932405f, 1.841004f, 0.627123f, 9.838157f, -9.768934f, 6.639832f, -8.132678f, -8.665508f, 0.886253f, 5.747538f, -0.717055f, -0.288050f, 2.998558f, -0.232435f, 0.593799f, -7.742426f, 7.815877f, 8.667622f, -9.198464f, 5.888268f, 9.261924f, 4.176519f};

static const float input_1_scales[] = {0.1389700770f, 0.1210434511f, 0.1394459009f, 0.1419170946f, 0.1500582844f, 0.1383515149f, 0.1498492807f, 0.1461481899f};
static const float input_1_zero_points[] = {-0.1127522215f, -2.0054848194f, 0.7046295404f, -0.1974012405f, 0.0896190554f, 1.0528359413f, -0.2535052598f, -0.6430486441f};
static const int8_t input_1_scales_frac[] = {17, 18, 17, 17, 17, 17, 17, 17};
static const int input_1_sa_dim = 1;

static const int input_1_fx8_frac = 3;

#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {16, 8}, \
    /* .rank =  */ 2


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

static const float input_2_data[] = {8.286820f, 2.152027f, 5.760523f, -9.209137f, -9.439082f, 2.777630f, 0.086691f, -9.923459f, -6.820004f, -1.362363f, 1.118086f, 2.201777f, 0.450819f, -3.447689f, -4.574860f, 0.236707f, 0.707713f, 5.680774f, -1.140300f, 8.814334f, 5.280867f, -2.641193f, 8.735827f, 3.382713f, -0.028107f, -1.362890f, -3.054990f, -6.572820f, 0.709137f, -7.732485f, -1.467243f, -0.805623f, 8.689197f, -2.427501f, 0.179479f, 5.904660f, 8.859151f, 1.512121f, -9.198095f, 4.772271f, 8.437635f, 3.685844f, 3.373433f, 3.015400f, -1.919783f, 0.743432f, -9.499326f, 7.141538f, 8.711848f, -9.691744f, 9.559444f, 5.302596f, 9.618321f, 9.830359f, -6.855545f, 1.136173f, -2.964916f, -5.046844f, 3.412143f, 0.712972f, -1.474889f, 7.401769f, 8.923663f, 8.486315f, -8.937352f, -6.372196f, 3.259260f, 5.393515f, -4.797753f, -4.907107f, -7.777387f, -2.044802f, -3.196458f, 0.084552f, -5.785887f, 0.653820f, 4.551648f, -6.382569f, 0.562406f, 1.477235f, 2.541792f, 0.210185f, 2.538625f, 2.979376f, -1.668876f, 3.617727f, -0.526883f, -8.592014f, 6.895158f, 3.578593f, -5.122587f, -5.243422f, -9.385137f, 3.870499f, 3.685569f, -2.745420f, -0.795467f, 2.039002f, -8.150185f, -6.949403f, 3.948650f, 1.120372f, 4.190713f, 8.637361f, 8.664962f, -5.665703f, 7.932405f, 1.841004f, 0.627123f, 9.838157f, -9.768934f, 6.639832f, -8.132678f, -8.665508f, 0.886253f, 5.747538f, -0.717055f, -0.288050f, 2.998558f, -0.232435f, 0.593799f, -7.742426f, 7.815877f, 8.667622f, -9.198464f, 5.888268f, 9.261924f, 4.176519f};

static const float input_2_scale = 0.0774965286f;
static const float input_2_zero_point = -0.0426510200f;
static const int8_t input_2_scales_frac[] = {18};
static const int input_2_sa_dim = -1;

static const int input_2_fx8_frac = 3;

#define INPUT_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(((((2 * 1) + 1) * 4) + 3) * 8) + 2, (((2 * 1) + 1) * 4) + 3, (2 * 1) + 1, 1}, \
    /* .shape = */ {2, 8, 4, 2}, \
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

static const float input_3_data[] = {8.286820f, 2.152027f, 5.760523f, -9.209137f, -9.439082f, 2.777630f, 0.086691f, -9.923459f, -6.820004f, -1.362363f, 1.118086f, 2.201777f, 0.450819f, -3.447689f, -4.574860f, 0.236707f, 0.707713f, 5.680774f, -1.140300f, 8.814334f, 5.280867f, -2.641193f, 8.735827f, 3.382713f, -0.028107f, -1.362890f, -3.054990f, -6.572820f, 0.709137f, -7.732485f, -1.467243f, -0.805623f, 8.689197f, -2.427501f, 0.179479f, 5.904660f, 8.859151f, 1.512121f, -9.198095f, 4.772271f, 8.437635f, 3.685844f, 3.373433f, 3.015400f, -1.919783f, 0.743432f, -9.499326f, 7.141538f, 8.711848f, -9.691744f, 9.559444f, 5.302596f, 9.618321f, 9.830359f, -6.855545f, 1.136173f, -2.964916f, -5.046844f, 3.412143f, 0.712972f, -1.474889f, 7.401769f, 8.923663f, 8.486315f, -8.937352f, -6.372196f, 3.259260f, 5.393515f, -4.797753f, -4.907107f, -7.777387f, -2.044802f, -3.196458f, 0.084552f, -5.785887f, 0.653820f, 4.551648f, -6.382569f, 0.562406f, 1.477235f, 2.541792f, 0.210185f, 2.538625f, 2.979376f, -1.668876f, 3.617727f, -0.526883f, -8.592014f, 6.895158f, 3.578593f, -5.122587f, -5.243422f, -9.385137f, 3.870499f, 3.685569f, -2.745420f, -0.795467f, 2.039002f, -8.150185f, -6.949403f, 3.948650f, 1.120372f, 4.190713f, 8.637361f, 8.664962f, -5.665703f, 7.932405f, 1.841004f, 0.627123f, 9.838157f, -9.768934f, 6.639832f, -8.132678f, -8.665508f, 0.886253f, 5.747538f, -0.717055f, -0.288050f, 2.998558f, -0.232435f, 0.593799f, -7.742426f, 7.815877f, 8.667622f, -9.198464f, 5.888268f, 9.261924f, 4.176519f};

static const float input_3_scale = 0.0774965286f;
static const float input_3_zero_point = -0.0426510200f;
static const int8_t input_3_scales_frac[] = {18};
static const int input_3_sa_dim = -1;

static const int input_3_fx8_frac = 3;

#define INPUT_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {4, 8, 4}, \
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


static const float test_1_out_data[] = {8.286820f, 2.152027f, 5.760523f, -9.209137f, -9.439082f, 2.777630f, 0.086691f, -9.923459f, -6.820004f, -1.362363f, 1.118086f, 2.201777f, 0.450819f, -3.447689f, -4.574860f, 0.236707f, 0.707713f, 5.680774f, -1.140300f, 8.814334f, 5.280867f, -2.641193f, 8.735827f, 3.382713f, -0.028107f, -1.362890f, -3.054990f, -6.572820f, 0.709137f, -7.732485f, -1.467243f, -0.805623f, 8.689197f, -2.427501f, 0.179479f, 5.904660f, 8.859151f, 1.512121f, -9.198095f, 4.772271f, 8.437635f, 3.685844f, 3.373433f, 3.015400f, -1.919783f, 0.743432f, -9.499326f, 7.141538f, 8.711848f, -9.691744f, 9.559444f, 5.302596f, 9.618321f, 9.830359f, -6.855545f, 1.136173f, -2.964916f, -5.046844f, 3.412143f, 0.712972f, -1.474889f, 7.401769f, 8.923663f, 8.486315f, -8.937352f, -6.372196f, 3.259260f, 5.393515f, -4.797753f, -4.907107f, -7.777387f, -2.044802f, -3.196458f, 0.084552f, -5.785887f, 0.653820f, 4.551648f, -6.382569f, 0.562406f, 1.477235f, 2.541792f, 0.210185f, 2.538625f, 2.979376f, -1.668876f, 3.617727f, -0.526883f, -8.592014f, 6.895158f, 3.578593f, -5.122587f, -5.243422f, -9.385137f, 3.870499f, 3.685569f, -2.745420f, -0.795467f, 2.039002f, -8.150185f, -6.949403f, 3.948650f, 1.120372f, 4.190713f, 8.637361f, 8.664962f, -5.665703f, 7.932405f, 1.841004f, 0.627123f, 9.838157f, -9.768934f, 6.639832f, -8.132678f, -8.665508f, 0.886253f, 5.747538f, -0.717055f, -0.288050f, 2.998558f, -0.232435f, 0.593799f, -7.742426f, 7.815877f, 8.667622f, -9.198464f, 5.888268f, 9.261924f, 4.176519f};

static const float test_1_out_scales[] = {0.1389700770f, 0.1210434511f, 0.1394459009f, 0.1419170946f, 0.1500582844f, 0.1383515149f, 0.1498492807f, 0.1461481899f};
static const float test_1_out_zero_points[] = {-0.1127522215f, -2.0054848194f, 0.7046295404f, -0.1974012405f, 0.0896190554f, 1.0528359413f, -0.2535052598f, -0.6430486441f};
static const int8_t test_1_out_scales_frac[] = {17, 18, 17, 17, 17, 17, 17, 17};
static const int test_1_out_sa_dim = 1;

static const int test_1_out_fx8_frac = 3;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {16, 8}, \
    /* .rank =  */ 2


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

static const float test_2_out_data[] = {8.286820f, 2.152027f, 5.760523f, -9.209137f, -9.439082f, 2.777630f, 0.086691f, -9.923459f, -6.820004f, -1.362363f, 1.118086f, 2.201777f, 0.450819f, -3.447689f, -4.574860f, 0.236707f, 0.707713f, 5.680774f, -1.140300f, 8.814334f, 5.280867f, -2.641193f, 8.735827f, 3.382713f, -0.028107f, -1.362890f, -3.054990f, -6.572820f, 0.709137f, -7.732485f, -1.467243f, -0.805623f, 8.689197f, -2.427501f, 0.179479f, 5.904660f, 8.859151f, 1.512121f, -9.198095f, 4.772271f, 8.437635f, 3.685844f, 3.373433f, 3.015400f, -1.919783f, 0.743432f, -9.499326f, 7.141538f, 8.711848f, -9.691744f, 9.559444f, 5.302596f, 9.618321f, 9.830359f, -6.855545f, 1.136173f, -2.964916f, -5.046844f, 3.412143f, 0.712972f, -1.474889f, 7.401769f, 8.923663f, 8.486315f, -8.937352f, -6.372196f, 3.259260f, 5.393515f, -4.797753f, -4.907107f, -7.777387f, -2.044802f, -3.196458f, 0.084552f, -5.785887f, 0.653820f, 4.551648f, -6.382569f, 0.562406f, 1.477235f, 2.541792f, 0.210185f, 2.538625f, 2.979376f, -1.668876f, 3.617727f, -0.526883f, -8.592014f, 6.895158f, 3.578593f, -5.122587f, -5.243422f, -9.385137f, 3.870499f, 3.685569f, -2.745420f, -0.795467f, 2.039002f, -8.150185f, -6.949403f, 3.948650f, 1.120372f, 4.190713f, 8.637361f, 8.664962f, -5.665703f, 7.932405f, 1.841004f, 0.627123f, 9.838157f, -9.768934f, 6.639832f, -8.132678f, -8.665508f, 0.886253f, 5.747538f, -0.717055f, -0.288050f, 2.998558f, -0.232435f, 0.593799f, -7.742426f, 7.815877f, 8.667622f, -9.198464f, 5.888268f, 9.261924f, 4.176519f};

static const float test_2_out_scale = 0.0774965286f;
static const float test_2_out_zero_point = -0.0426510200f;
static const int8_t test_2_out_scales_frac[] = {18};
static const int test_2_out_sa_dim = -1;

static const int test_2_out_fx8_frac = 3;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {(((((2 * 1) + 2) * 4) + 5) * 8) + 4, (((2 * 1) + 2) * 4) + 5, (2 * 1) + 2, 1}, \
    /* .shape = */ {2, 8, 4, 2}, \
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

static const float test_3_out_data[] = {8.286820f, 2.152027f, 5.760523f, -9.209137f, -9.439082f, 2.777630f, 0.086691f, -9.923459f, -6.820004f, -1.362363f, 1.118086f, 2.201777f, 0.450819f, -3.447689f, -4.574860f, 0.236707f, 0.707713f, 5.680774f, -1.140300f, 8.814334f, 5.280867f, -2.641193f, 8.735827f, 3.382713f, -0.028107f, -1.362890f, -3.054990f, -6.572820f, 0.709137f, -7.732485f, -1.467243f, -0.805623f, 8.689197f, -2.427501f, 0.179479f, 5.904660f, 8.859151f, 1.512121f, -9.198095f, 4.772271f, 8.437635f, 3.685844f, 3.373433f, 3.015400f, -1.919783f, 0.743432f, -9.499326f, 7.141538f, 8.711848f, -9.691744f, 9.559444f, 5.302596f, 9.618321f, 9.830359f, -6.855545f, 1.136173f, -2.964916f, -5.046844f, 3.412143f, 0.712972f, -1.474889f, 7.401769f, 8.923663f, 8.486315f, -8.937352f, -6.372196f, 3.259260f, 5.393515f, -4.797753f, -4.907107f, -7.777387f, -2.044802f, -3.196458f, 0.084552f, -5.785887f, 0.653820f, 4.551648f, -6.382569f, 0.562406f, 1.477235f, 2.541792f, 0.210185f, 2.538625f, 2.979376f, -1.668876f, 3.617727f, -0.526883f, -8.592014f, 6.895158f, 3.578593f, -5.122587f, -5.243422f, -9.385137f, 3.870499f, 3.685569f, -2.745420f, -0.795467f, 2.039002f, -8.150185f, -6.949403f, 3.948650f, 1.120372f, 4.190713f, 8.637361f, 8.664962f, -5.665703f, 7.932405f, 1.841004f, 0.627123f, 9.838157f, -9.768934f, 6.639832f, -8.132678f, -8.665508f, 0.886253f, 5.747538f, -0.717055f, -0.288050f, 2.998558f, -0.232435f, 0.593799f, -7.742426f, 7.815877f, 8.667622f, -9.198464f, 5.888268f, 9.261924f, 4.176519f};

static const float test_3_out_scale = 0.0774965286f;
static const float test_3_out_zero_point = -0.0426510200f;
static const int8_t test_3_out_scales_frac[] = {18};
static const int test_3_out_sa_dim = -1;

static const int test_3_out_fx8_frac = 3;

#define TEST_3_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {33, 4 ,1}, \
    /* .shape = */ {4, 8, 4}, \
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
sizeof(input_1_data) / sizeof(input_1_data[0]), input_1_scales, sizeof(input_1_scales) / sizeof(input_1_scales[0]), 
input_1_zero_points, sizeof(input_1_zero_points) / sizeof(input_1_zero_points[0]), 
input_1_scales_frac, sizeof(input_1_scales_frac) / sizeof(input_1_scales_frac[0]));

tensor_quantizer input_2_fx16(input_2_tsr_fx16, input_2_fx8_frac + 8, input_2_data, sizeof(input_2_data) / sizeof(input_2_data[0]));
tensor_quantizer input_2_sa8(input_2_tsr_sa8, input_2_sa_dim, input_2_data, 
sizeof(input_2_data) / sizeof(input_2_data[0]), &input_2_scale, 1, &input_2_zero_point, 1, input_2_scales_frac, 1);

tensor_quantizer input_3_fx16(input_3_tsr_fx16, input_3_fx8_frac + 8, input_3_data, sizeof(input_3_data) / sizeof(input_3_data[0]));
tensor_quantizer input_3_sa8(input_3_tsr_sa8, input_3_sa_dim, input_3_data, 
sizeof(input_3_data) / sizeof(input_3_data[0]), &input_3_scale, 1, &input_3_zero_point, 1, input_3_scales_frac, 1);

tensor_quantizer test_1_out_fx16(test_1_out_tsr_fx16, test_1_out_fx8_frac + 8, test_1_out_data, sizeof(test_1_out_data) / sizeof(test_1_out_data[0]));
tensor_quantizer test_1_out_sa8(test_1_out_tsr_sa8, test_1_out_sa_dim, test_1_out_data, 
sizeof(test_1_out_data) / sizeof(test_1_out_data[0]), test_1_out_scales, sizeof(test_1_out_scales) / sizeof(test_1_out_scales[0]), 
test_1_out_zero_points, sizeof(test_1_out_zero_points) / sizeof(test_1_out_zero_points[0]), 
test_1_out_scales_frac, sizeof(test_1_out_scales_frac) / sizeof(test_1_out_scales_frac[0]));

tensor_quantizer test_2_out_fx16(test_2_out_tsr_fx16, test_2_out_fx8_frac + 8, test_2_out_data, sizeof(test_2_out_data) / sizeof(test_2_out_data[0]));
tensor_quantizer test_2_out_sa8(test_2_out_tsr_sa8, test_2_out_sa_dim, test_2_out_data, 
sizeof(test_2_out_data) / sizeof(test_2_out_data[0]), &test_2_out_scale, 1, &test_2_out_zero_point, 1, test_2_out_scales_frac, 1);

tensor_quantizer test_3_out_fx16(test_3_out_tsr_fx16, test_3_out_fx8_frac + 8, test_3_out_data, sizeof(test_3_out_data) / sizeof(test_3_out_data[0]));
tensor_quantizer test_3_out_sa8(test_3_out_tsr_sa8, test_3_out_sa_dim, test_3_out_data, 
sizeof(test_3_out_data) / sizeof(test_3_out_data[0]), &test_3_out_scale, 1, &test_3_out_zero_point, 1, test_3_out_scales_frac, 1);
