/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <stdint.h>

#include "mli_types.h"
#include "mli_types.hpp"
#include "test_tensor_quantizer.h"

using mli::tst::tensor_quantizer;

namespace lib_mli = ::snps_arc::metaware::mli;

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
extern mli::tst::tensor_quantizer test_7_out_sa8;
extern mli::tst::tensor_quantizer test_7_out_fx8;
extern mli::tst::tensor_quantizer test_7_out_fx16;
extern mli::tst::tensor_quantizer test_7_out_sa32;
extern mli::tst::tensor_quantizer test_8_out_sa8;
extern mli::tst::tensor_quantizer test_8_out_fx8;
extern mli::tst::tensor_quantizer test_8_out_fx16;
extern mli::tst::tensor_quantizer test_8_out_sa32;
extern const lib_mli::ReduceOpConfig test_1_cfg;
extern const lib_mli::ReduceOpConfig test_2_cfg;
extern const lib_mli::ReduceOpConfig test_3_cfg;
extern const lib_mli::ReduceOpConfig test_4_cfg;
static const float input_1_data[] = {-13.000000f, 8.000000f, -35.000000f, 7.000000f, 23.000000f, -37.000000f, 43.000000f, 32.000000f, -3.000000f, -46.000000f, -37.000000f, 12.000000f, 29.000000f, -18.000000f, 30.000000f, -52.000000f};

static const float input_1_scale = 1.0f;
static const float input_1_zero_point = 0.0f;
static const int8_t input_1_scales_frac[] = {0};
static const int input_1_sa_dim = -1;

static const int input_1_fx8_frac = 0;

#define INPUT_1_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 2, 2, 2}, \
    /* .rank =  */ 4


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

static const float input_2_data[] = {19.000000f, 18.000000f, -16.000000f, 28.000000f, -13.000000f, 24.000000f, -20.000000f, -9.000000f, -14.000000f, -1.000000f, 24.000000f, -15.000000f, 4.000000f, 28.000000f, -20.000000f, 14.000000f, -11.000000f, -6.000000f, 18.000000f, -16.000000f, 26.000000f, -11.000000f, 3.000000f, 15.000000f, -17.000000f, -12.000000f, -1.000000f, 28.000000f, -19.000000f, -20.000000f, -4.000000f, 11.000000f, 9.000000f, 24.000000f, 8.000000f, -13.000000f, 3.000000f, -10.000000f, 21.000000f, 15.000000f, -20.000000f, -13.000000f, 3.000000f, 8.000000f, 12.000000f, 1.000000f, 16.000000f, 15.000000f, 13.000000f, -17.000000f, 7.000000f, 13.000000f, -1.000000f, 7.000000f, -16.000000f, -1.000000f, 13.000000f, 10.000000f, 17.000000f, -11.000000f};

static const float input_2_scale = 1.0f;
static const float input_2_zero_point = 0.0f;
static const int8_t input_2_scales_frac[] = {0};
static const int input_2_sa_dim = -1;

static const int input_2_fx8_frac = 0;

#define INPUT_2_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 6, 5, 2}, \
    /* .rank =  */ 4


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

static const float input_3_data[] = {-4.000000f, -3.000000f, -2.000000f, -3.000000f, -3.000000f, 0.000000f, 2.000000f, -2.000000f, 6.000000f, 5.000000f, -5.000000f, 6.000000f, 4.000000f, 4.000000f, -2.000000f, -1.000000f, 0.000000f, 6.000000f, -2.000000f, -5.000000f, -1.000000f, 6.000000f, 3.000000f, 5.000000f, 5.000000f, 1.000000f, 6.000000f, 2.000000f, -3.000000f, 2.000000f, -2.000000f, 3.000000f, 1.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 6.000000f, 2.000000f, -4.000000f, 5.000000f, 0.000000f, -1.000000f, -1.000000f, 4.000000f, 4.000000f, 5.000000f, -5.000000f, 1.000000f, -3.000000f, 1.000000f, 3.000000f, -3.000000f, 6.000000f, 5.000000f, -1.000000f, -4.000000f, 1.000000f, 5.000000f, -4.000000f, 0.000000f, -4.000000f, 1.000000f, 4.000000f, 3.000000f, 1.000000f, 0.000000f, 4.000000f, 2.000000f, 0.000000f, -1.000000f, 6.000000f, 4.000000f, 6.000000f, 1.000000f, 3.000000f, -4.000000f, 0.000000f, 0.000000f, 6.000000f, 3.000000f, -2.000000f, 2.000000f, 2.000000f, 4.000000f, -1.000000f, 2.000000f, 0.000000f, 5.000000f, 4.000000f, 1.000000f, 6.000000f, 6.000000f, -3.000000f, -5.000000f, 0.000000f, -2.000000f, -5.000000f, 0.000000f, 2.000000f, -4.000000f, 3.000000f, -1.000000f, 6.000000f, 4.000000f, 5.000000f, -5.000000f, -3.000000f, -5.000000f, 2.000000f, 1.000000f, -3.000000f, 4.000000f, 4.000000f, 0.000000f, 6.000000f, -4.000000f, -5.000000f, -5.000000f, 4.000000f, -4.000000f, -4.000000f, 0.000000f, -2.000000f, -3.000000f, -5.000000f, 5.000000f, 5.000000f, -1.000000f, 5.000000f, 6.000000f, 3.000000f, 2.000000f, -4.000000f, -1.000000f, 4.000000f, -2.000000f, 1.000000f, 5.000000f, 2.000000f, -4.000000f, -5.000000f, -3.000000f, 2.000000f, -5.000000f, -5.000000f, 4.000000f, 3.000000f, -3.000000f, -5.000000f, 6.000000f, 0.000000f, 0.000000f, -5.000000f, 4.000000f, 6.000000f, 5.000000f, -5.000000f, 3.000000f, -1.000000f, 0.000000f, 6.000000f, -4.000000f, 5.000000f, -1.000000f, -5.000000f, 2.000000f, 5.000000f, -3.000000f, -5.000000f, -2.000000f, 4.000000f, 3.000000f, -2.000000f, 2.000000f, -4.000000f, 5.000000f, -5.000000f, -4.000000f, 6.000000f, 6.000000f, -1.000000f, 1.000000f, 6.000000f, -4.000000f, 6.000000f, -5.000000f, -4.000000f, 5.000000f, 5.000000f, -5.000000f, 6.000000f, -4.000000f, 3.000000f, -1.000000f, -4.000000f, 1.000000f, 3.000000f, 0.000000f, -3.000000f, -1.000000f, 2.000000f, -1.000000f, 6.000000f, -4.000000f, -3.000000f, 3.000000f, 0.000000f, -3.000000f, 4.000000f, 1.000000f, -3.000000f, 0.000000f, 2.000000f, 2.000000f, 0.000000f, 0.000000f, 1.000000f, -5.000000f, 5.000000f, 1.000000f, -1.000000f, 1.000000f, -3.000000f, 1.000000f, 3.000000f, 6.000000f, 6.000000f, 2.000000f, -5.000000f, 0.000000f, 3.000000f, 4.000000f, 2.000000f, 2.000000f, 1.000000f, -3.000000f, 0.000000f, 1.000000f, -4.000000f, 4.000000f, -2.000000f, -1.000000f, -5.000000f, -2.000000f, 0.000000f, 1.000000f, -1.000000f, -5.000000f, -1.000000f, -4.000000f, 0.000000f, -3.000000f, 6.000000f, 6.000000f, -1.000000f, 3.000000f, 6.000000f, 2.000000f, 0.000000f, 2.000000f, 0.000000f, -2.000000f, 2.000000f, 1.000000f, -3.000000f, -2.000000f, -4.000000f, -4.000000f, 3.000000f, -1.000000f, 5.000000f, -3.000000f, 6.000000f, 6.000000f, -4.000000f, 5.000000f, 2.000000f, -5.000000f, 3.000000f, 3.000000f, -4.000000f, -3.000000f, -2.000000f, -5.000000f, 5.000000f, -4.000000f, 6.000000f, -4.000000f, 0.000000f, 0.000000f, 1.000000f, -5.000000f, 4.000000f, 6.000000f, -2.000000f, 6.000000f, -3.000000f, -1.000000f, -3.000000f, 3.000000f, 5.000000f, -1.000000f, 4.000000f, -5.000000f, 5.000000f, -5.000000f, 5.000000f, -3.000000f, 1.000000f, -2.000000f, 4.000000f, 0.000000f, -1.000000f, 1.000000f, 4.000000f, 4.000000f, 4.000000f, -1.000000f, 3.000000f, -3.000000f, -1.000000f, 1.000000f, 0.000000f, 6.000000f, -2.000000f, 0.000000f, -3.000000f, 1.000000f, -1.000000f, -2.000000f, -3.000000f, 0.000000f, 4.000000f, 2.000000f, 0.000000f, 3.000000f, 6.000000f, 5.000000f, 0.000000f, -3.000000f, -5.000000f, 4.000000f, -2.000000f, -3.000000f, -3.000000f, 5.000000f, -1.000000f, -2.000000f, -1.000000f, 3.000000f, 2.000000f, 5.000000f, 5.000000f, -5.000000f, -3.000000f, 2.000000f, 3.000000f, 5.000000f, 3.000000f, -1.000000f, -5.000000f, -1.000000f, -4.000000f, 2.000000f, -1.000000f, 4.000000f, -5.000000f, -4.000000f, -5.000000f, 5.000000f, 1.000000f, -3.000000f, -5.000000f, -3.000000f, -1.000000f, -1.000000f, -4.000000f, 5.000000f, -5.000000f, -5.000000f, -3.000000f, -2.000000f, 2.000000f, 1.000000f, 4.000000f, -2.000000f, 0.000000f, 1.000000f, 3.000000f, 1.000000f, 2.000000f, -2.000000f, 6.000000f, 3.000000f, 6.000000f, 0.000000f, 1.000000f, -1.000000f, -3.000000f, 2.000000f, 4.000000f, 1.000000f, 5.000000f, 5.000000f, 3.000000f, 1.000000f, 5.000000f, -4.000000f, -5.000000f, 2.000000f, 1.000000f, 3.000000f, 0.000000f, -1.000000f, -2.000000f, -5.000000f, 2.000000f, -1.000000f, -5.000000f, 3.000000f, 5.000000f, 0.000000f, 6.000000f, -5.000000f, 2.000000f, 6.000000f, 5.000000f, 0.000000f, -4.000000f, 0.000000f, 4.000000f, -1.000000f, -4.000000f, 4.000000f, 1.000000f, 5.000000f, 0.000000f, 1.000000f, 6.000000f, 3.000000f, 0.000000f, 4.000000f, -1.000000f, 1.000000f, -5.000000f, 1.000000f, -3.000000f, 5.000000f, -2.000000f, -4.000000f, 1.000000f, 2.000000f, -5.000000f, -5.000000f, -1.000000f, 2.000000f, -5.000000f, -3.000000f, 4.000000f, 1.000000f, -4.000000f, -3.000000f, -3.000000f, 3.000000f, 4.000000f, 5.000000f, 2.000000f, -3.000000f, -1.000000f, 2.000000f, 2.000000f, 6.000000f, 4.000000f, -4.000000f, -5.000000f, -4.000000f, 3.000000f, 2.000000f, 5.000000f, -1.000000f, 5.000000f, 1.000000f, 6.000000f, -4.000000f, -1.000000f, 3.000000f, 2.000000f, -2.000000f, 6.000000f, -2.000000f, 1.000000f, -1.000000f, 4.000000f, 0.000000f, 2.000000f, 4.000000f, 5.000000f, 4.000000f, 2.000000f, 1.000000f, 4.000000f, 3.000000f, 6.000000f, 3.000000f, 1.000000f, 6.000000f, -3.000000f, -3.000000f, -3.000000f, 6.000000f, 0.000000f, -4.000000f, -1.000000f, -4.000000f, -1.000000f, 4.000000f, 6.000000f, 0.000000f, 5.000000f, -2.000000f, 0.000000f, 3.000000f, 2.000000f, -5.000000f, 5.000000f, 3.000000f, 0.000000f, -1.000000f, -5.000000f, -1.000000f, 2.000000f, 5.000000f, -5.000000f, 1.000000f, 6.000000f, 6.000000f, -4.000000f, -5.000000f, 2.000000f, 1.000000f, -5.000000f, -2.000000f, 1.000000f, 2.000000f, -1.000000f, 6.000000f, 3.000000f, 5.000000f, 2.000000f, 4.000000f, 1.000000f, -2.000000f, 5.000000f, 6.000000f, 4.000000f, -3.000000f, 0.000000f, -1.000000f, 5.000000f, -4.000000f, -4.000000f, 0.000000f, -4.000000f, -3.000000f, -4.000000f, 2.000000f, -4.000000f, -3.000000f, 3.000000f, 1.000000f, 0.000000f, 1.000000f, 3.000000f, -2.000000f, -3.000000f, -4.000000f, 0.000000f, -5.000000f, -1.000000f, -2.000000f, 2.000000f, -1.000000f, -1.000000f, -5.000000f, 0.000000f, -3.000000f, 0.000000f, 1.000000f, 4.000000f, 1.000000f, 2.000000f, 0.000000f, 2.000000f, 6.000000f, -1.000000f, 5.000000f, 3.000000f, 6.000000f, 2.000000f, 2.000000f, 3.000000f, -2.000000f, -2.000000f, 2.000000f, -5.000000f, -5.000000f, 1.000000f, 2.000000f, -5.000000f, -3.000000f, 2.000000f, -1.000000f, -3.000000f, 0.000000f, 3.000000f, -5.000000f, 0.000000f, -2.000000f, 6.000000f, 5.000000f, -1.000000f, 0.000000f, 6.000000f, 3.000000f, 5.000000f, 6.000000f, -5.000000f, 0.000000f, 4.000000f, -2.000000f, 0.000000f, -5.000000f, -4.000000f, 5.000000f, 2.000000f, 2.000000f, 2.000000f, -3.000000f, -2.000000f, 1.000000f, -1.000000f, 3.000000f, -3.000000f, 2.000000f, 0.000000f, 2.000000f, 6.000000f, -3.000000f, 4.000000f, -2.000000f, -5.000000f, -3.000000f, 4.000000f, 4.000000f, -1.000000f, 5.000000f, -2.000000f, 0.000000f, 1.000000f, -2.000000f, 3.000000f, -1.000000f, 6.000000f, 3.000000f, -4.000000f, -1.000000f, -3.000000f, 6.000000f, 3.000000f, 0.000000f, 6.000000f, -5.000000f, 5.000000f, 4.000000f, 3.000000f, -4.000000f, -4.000000f, 6.000000f, 6.000000f, -1.000000f, -4.000000f, 4.000000f, -3.000000f, 4.000000f, 6.000000f, 1.000000f, 4.000000f, 4.000000f, -4.000000f, -3.000000f, -4.000000f, 0.000000f, 6.000000f, 1.000000f, -3.000000f, 1.000000f, 3.000000f, -4.000000f, 1.000000f, -5.000000f, -5.000000f, 0.000000f, -4.000000f, 2.000000f, -1.000000f, -3.000000f, 1.000000f, -1.000000f, -4.000000f, 2.000000f, -2.000000f, -5.000000f, 3.000000f, -4.000000f, 1.000000f, 5.000000f, -5.000000f, 4.000000f, -5.000000f, 2.000000f, -2.000000f, 6.000000f, 5.000000f, -2.000000f, -3.000000f, -3.000000f, 4.000000f, -5.000000f, 4.000000f, 0.000000f, -5.000000f, 6.000000f, -3.000000f, 5.000000f, 5.000000f, -2.000000f, 5.000000f, 0.000000f, 6.000000f, 4.000000f, -5.000000f, 3.000000f, -5.000000f, 2.000000f, -5.000000f, 0.000000f, 5.000000f, 2.000000f, 6.000000f, -2.000000f, -4.000000f, -5.000000f, -3.000000f, 1.000000f, -4.000000f, 3.000000f, -4.000000f, -3.000000f, -3.000000f, 4.000000f, 2.000000f, -5.000000f, -1.000000f, -1.000000f, -5.000000f, 0.000000f, -1.000000f, 2.000000f, -4.000000f, 5.000000f, -5.000000f, 5.000000f, 6.000000f, 2.000000f, 6.000000f, -4.000000f, 0.000000f, 1.000000f, 3.000000f, 5.000000f, -1.000000f, -5.000000f, 1.000000f, 4.000000f, -2.000000f, -3.000000f, -5.000000f, -4.000000f, 5.000000f, 5.000000f, -5.000000f, 3.000000f, -3.000000f, 2.000000f, -4.000000f, 4.000000f, 1.000000f, 4.000000f, -1.000000f, 1.000000f, 3.000000f, -4.000000f, 2.000000f, 5.000000f, 4.000000f, 6.000000f, 0.000000f, 5.000000f, 3.000000f, 1.000000f, -4.000000f, -5.000000f, 2.000000f, -5.000000f, 2.000000f, -5.000000f, 6.000000f, 6.000000f, -2.000000f, 5.000000f, -5.000000f, -1.000000f, -5.000000f, 6.000000f, 3.000000f, 1.000000f, -3.000000f, 1.000000f, -3.000000f, -5.000000f, -4.000000f, 1.000000f, -1.000000f, 4.000000f, -1.000000f, -4.000000f, 2.000000f, 1.000000f, -3.000000f, 2.000000f, -5.000000f, -4.000000f, 2.000000f, 2.000000f, -3.000000f, -2.000000f, -5.000000f, -5.000000f, 0.000000f, 4.000000f, 4.000000f, 3.000000f, -5.000000f, 4.000000f, -4.000000f, -1.000000f, -5.000000f, 5.000000f, 5.000000f, 4.000000f, -2.000000f, -3.000000f, -2.000000f, 0.000000f, -3.000000f, -5.000000f, 3.000000f, 1.000000f, 6.000000f, 1.000000f, -5.000000f, 0.000000f, 6.000000f, -4.000000f, -3.000000f, 2.000000f, -2.000000f, 5.000000f, -3.000000f, 5.000000f, -4.000000f, 1.000000f, 5.000000f, 4.000000f, 5.000000f, 0.000000f, 2.000000f, 5.000000f, 3.000000f, 3.000000f, 3.000000f, 3.000000f, 4.000000f, 4.000000f, 5.000000f, 3.000000f, 4.000000f, -2.000000f, 5.000000f, -4.000000f, 5.000000f, 5.000000f, -5.000000f, 6.000000f, -4.000000f, 3.000000f, 6.000000f, -5.000000f, 4.000000f, 3.000000f, 2.000000f, -1.000000f, -1.000000f, -5.000000f, 0.000000f, 5.000000f, -2.000000f, -2.000000f, 1.000000f, -4.000000f, -4.000000f, 0.000000f, -3.000000f, -5.000000f, 0.000000f, -5.000000f, -2.000000f, -1.000000f, 5.000000f, -4.000000f, -2.000000f, 3.000000f, 0.000000f, -1.000000f, -3.000000f, -4.000000f, -2.000000f, -2.000000f, -4.000000f, 1.000000f, 1.000000f, 2.000000f, 1.000000f, 2.000000f, -4.000000f, 0.000000f, -2.000000f, 1.000000f, -1.000000f, 4.000000f, -5.000000f, -1.000000f, 6.000000f, -2.000000f, 3.000000f, -3.000000f, 3.000000f, 5.000000f, -2.000000f, 4.000000f, -3.000000f, -4.000000f, 4.000000f, 6.000000f, 1.000000f, 4.000000f, -1.000000f, 4.000000f, 1.000000f, 6.000000f, 2.000000f, 3.000000f, 5.000000f, 5.000000f, 4.000000f, 1.000000f, -1.000000f, 1.000000f, -1.000000f, 0.000000f, 2.000000f, -1.000000f, -5.000000f, 0.000000f, -3.000000f, 0.000000f, -1.000000f, 0.000000f, -4.000000f, 2.000000f, -1.000000f, -1.000000f, -1.000000f, -3.000000f, 3.000000f, 2.000000f, 1.000000f, 1.000000f, 3.000000f, -5.000000f, 1.000000f, -3.000000f, 1.000000f, 3.000000f, -1.000000f, 2.000000f, 3.000000f, 5.000000f, -2.000000f, -2.000000f, 6.000000f, -1.000000f, -5.000000f, -3.000000f, 5.000000f, 3.000000f, 3.000000f, -3.000000f, 6.000000f, 0.000000f, 1.000000f, 3.000000f, -5.000000f, 2.000000f, -2.000000f, 2.000000f, 3.000000f, 6.000000f, 0.000000f, 0.000000f, -5.000000f, 1.000000f, -1.000000f, 6.000000f, -4.000000f, 5.000000f, 5.000000f, 3.000000f, -2.000000f, 4.000000f, 0.000000f, 6.000000f, -1.000000f, 2.000000f, -2.000000f, 5.000000f, 4.000000f, 1.000000f, 1.000000f, 5.000000f, 0.000000f, 4.000000f, 6.000000f, -3.000000f, 2.000000f, 2.000000f, 1.000000f, 6.000000f, -2.000000f, -3.000000f, -5.000000f, 5.000000f, -1.000000f, 6.000000f, 4.000000f, -2.000000f, -5.000000f, -4.000000f, -5.000000f, 6.000000f, -2.000000f, -3.000000f, -5.000000f, 0.000000f, 3.000000f, -2.000000f, 4.000000f, -3.000000f, -1.000000f};

static const float input_3_scale = 1.0f;
static const float input_3_zero_point = 0.0f;
static const int8_t input_3_scales_frac[] = {0};
static const int input_3_sa_dim = -1;

static const int input_3_fx8_frac = 0;

#define INPUT_3_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 9, 4, 15}, \
    /* .rank =  */ 4


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

static const float input_4_data[] = {-2.000000f, 15.000000f, 13.000000f, 9.000000f, 5.000000f, -5.000000f, 10.000000f, -6.000000f, 3.000000f, 16.000000f, 15.000000f, -9.000000f, 1.000000f, -3.000000f, 13.000000f, 2.000000f, 1.000000f, -3.000000f, -4.000000f, -4.000000f, -6.000000f, 8.000000f, -5.000000f, 8.000000f, -4.000000f, 4.000000f, -12.000000f, 13.000000f, -9.000000f, 9.000000f, -7.000000f, 11.000000f, -11.000000f, -4.000000f, 17.000000f, 1.000000f, 3.000000f, 12.000000f, 2.000000f, 4.000000f, -5.000000f, -1.000000f, 4.000000f, -6.000000f, 17.000000f, -11.000000f, 0.000000f, 13.000000f, -4.000000f, 12.000000f, -1.000000f, -11.000000f, 9.000000f, 15.000000f, -3.000000f, 16.000000f, -11.000000f, -11.000000f, 0.000000f, 13.000000f, -6.000000f, 3.000000f, -9.000000f, 15.000000f, -2.000000f, 10.000000f, 1.000000f, 13.000000f, 17.000000f, 7.000000f, 10.000000f, 5.000000f, 4.000000f, -6.000000f, 10.000000f, 17.000000f, 15.000000f, -12.000000f, 16.000000f, -6.000000f, 13.000000f, 16.000000f, 4.000000f, 12.000000f, 1.000000f, 2.000000f, -11.000000f, 5.000000f, -10.000000f, -11.000000f, 15.000000f, 7.000000f, -12.000000f, 7.000000f, -2.000000f, 16.000000f, -7.000000f, 8.000000f, 7.000000f, -4.000000f, -11.000000f, 14.000000f, 1.000000f, 3.000000f, 3.000000f, 9.000000f, 5.000000f, -3.000000f, 13.000000f, -11.000000f, -10.000000f, 13.000000f, -6.000000f, -12.000000f, 16.000000f, -12.000000f, -4.000000f, 17.000000f, -6.000000f, 16.000000f, 4.000000f, 16.000000f, -12.000000f, 5.000000f, 9.000000f, -11.000000f};

static const float input_4_scale = 1.0f;
static const float input_4_zero_point = 0.0f;
static const int8_t input_4_scales_frac[] = {0};
static const int input_4_sa_dim = -1;

static const int input_4_fx8_frac = 0;

#define INPUT_4_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {7, 6, 3}, \
    /* .rank =  */ 3


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

static const float test_1_out_data[] = {-16.000000f, -38.000000f, -72.000000f, 19.000000f, 52.000000f, -55.000000f, 73.000000f, -20.000000f};

static const float test_1_out_scale = 1.0f;
static const float test_1_out_zero_point = 0.0f;
static const int8_t test_1_out_scales_frac[] = {0};
static const int test_1_out_sa_dim = -1;

static const int test_1_out_fx8_frac = 0;

#define TEST_1_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 2, 2, 2}, \
    /* .rank =  */ 4


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

static const float test_2_out_data[] = {10.000000f, -29.000000f, 8.000000f, 39.000000f, 26.000000f, -64.000000f, -7.000000f, -40.000000f};

static const float test_2_out_scale = 1.0f;
static const float test_2_out_zero_point = 0.0f;
static const int8_t test_2_out_scales_frac[] = {0};
static const int test_2_out_sa_dim = -1;

static const int test_2_out_fx8_frac = 0;

#define TEST_2_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 1, 2, 2}, \
    /* .rank =  */ 4


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

static const float test_3_out_data[] = {-44.000000f, 60.000000f, 15.000000f, 5.000000f, -8.000000f, 0.000000f, 37.000000f, 27.000000f, 24.000000f, -6.000000f, 20.000000f, 18.000000f};

static const float test_3_out_scale = 1.0f;
static const float test_3_out_zero_point = 0.0f;
static const int8_t test_3_out_scales_frac[] = {0};
static const int test_3_out_sa_dim = -1;

static const int test_3_out_fx8_frac = 0;

#define TEST_3_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 6, 1, 2}, \
    /* .rank =  */ 4


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

static const float test_4_out_data[] = {37.000000f, 12.000000f, 11.000000f, -29.000000f, -15.000000f, 9.000000f, 32.000000f, -6.000000f, -17.000000f, 2.000000f, 15.000000f, 18.000000f, -29.000000f, 27.000000f, -39.000000f, 7.000000f, 33.000000f, -5.000000f, -7.000000f, 36.000000f, -33.000000f, 11.000000f, 13.000000f, 31.000000f, -4.000000f, 20.000000f, 6.000000f, -17.000000f, 23.000000f, 6.000000f};

static const float test_4_out_scale = 1.0f;
static const float test_4_out_zero_point = 0.0f;
static const int8_t test_4_out_scales_frac[] = {0};
static const int test_4_out_sa_dim = -1;

static const int test_4_out_fx8_frac = 0;

#define TEST_4_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 6, 5, 1}, \
    /* .rank =  */ 4


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

static const float test_5_out_data[] = {-3.000000f, -8.000000f, -4.000000f, -2.000000f, -1.000000f, -1.000000f, 8.000000f, 1.000000f, 11.000000f, 7.000000f, -1.000000f, 7.000000f, 2.000000f, 9.000000f, 4.000000f, 3.000000f, -3.000000f, 6.000000f, -3.000000f, 0.000000f, -5.000000f, 2.000000f, 3.000000f, 1.000000f, 2.000000f, -3.000000f, 8.000000f, -2.000000f, -6.000000f, 5.000000f, -1.000000f, 3.000000f, 2.000000f, 4.000000f, -2.000000f, -2.000000f, -4.000000f, 6.000000f, -3.000000f, -5.000000f, 3.000000f, 2.000000f, -2.000000f, -2.000000f, -1.000000f, 4.000000f, 2.000000f, -5.000000f, 2.000000f, 1.000000f, 2.000000f, 5.000000f, -3.000000f, 8.000000f, 11.000000f, -2.000000f, 1.000000f, 4.000000f, 11.000000f, -2.000000f, 2.000000f, -1.000000f, -1.000000f, 2.000000f, 5.000000f, -4.000000f, -5.000000f, 5.000000f, 4.000000f, -5.000000f, -4.000000f, 8.000000f, 3.000000f, 3.000000f, 1.000000f, 6.000000f, -9.000000f, 0.000000f, -2.000000f, 12.000000f, 8.000000f, -3.000000f, 2.000000f, 8.000000f, 7.000000f, 4.000000f, 8.000000f, -5.000000f, 5.000000f, 8.000000f, -1.000000f, 6.000000f, 1.000000f, -7.000000f, 0.000000f, 2.000000f, 0.000000f, -3.000000f, -3.000000f, 0.000000f, -3.000000f, 2.000000f, 2.000000f, 3.000000f, 6.000000f, 5.000000f, -3.000000f, 3.000000f, -8.000000f, 6.000000f, -1.000000f, -8.000000f, 1.000000f, 8.000000f, 4.000000f, 5.000000f, 1.000000f, -7.000000f, -5.000000f, 5.000000f, -6.000000f, -1.000000f, -1.000000f, 4.000000f, 0.000000f, -9.000000f, 4.000000f, 2.000000f, 5.000000f, 8.000000f, 6.000000f, 9.000000f, -3.000000f, 1.000000f, 3.000000f, 7.000000f, -6.000000f, -3.000000f, 11.000000f, 8.000000f, -5.000000f, -9.000000f, 1.000000f, -1.000000f, -1.000000f, 1.000000f, 5.000000f, 7.000000f, 1.000000f, -9.000000f, 3.000000f, -4.000000f, 0.000000f, 1.000000f, 5.000000f, 3.000000f, 6.000000f, -2.000000f, -1.000000f, 0.000000f, -5.000000f, 1.000000f, -4.000000f, 1.000000f, 1.000000f, -6.000000f, -1.000000f, 6.000000f, -4.000000f, -9.000000f, 0.000000f, 2.000000f, -2.000000f, 1.000000f, -2.000000f, -3.000000f, 10.000000f, -10.000000f, 0.000000f, 1.000000f, 8.000000f, -3.000000f, 7.000000f, 11.000000f, -6.000000f, 3.000000f, -8.000000f, 0.000000f, 0.000000f, 9.000000f, -5.000000f, 1.000000f, 2.000000f, 0.000000f, 4.000000f, 1.000000f, -1.000000f, 8.000000f, 0.000000f, 3.000000f, 3.000000f, -3.000000f, 2.000000f, 1.000000f, -2.000000f, -8.000000f, 3.000000f, 5.000000f, -1.000000f, 10.000000f, -1.000000f, -7.000000f, -5.000000f, -1.000000f, 3.000000f, -4.000000f, 3.000000f, -3.000000f, -8.000000f, 2.000000f, 5.000000f, 1.000000f, -4.000000f, -4.000000f, 0.000000f, -2.000000f, 6.000000f, 5.000000f, 4.000000f, -9.000000f, 5.000000f, -2.000000f, 9.000000f, 8.000000f, 4.000000f, 7.000000f, -7.000000f, 0.000000f, 2.000000f, -1.000000f, 9.000000f, -3.000000f, -6.000000f, -4.000000f, 2.000000f, -2.000000f, -2.000000f, -6.000000f, -9.000000f, 4.000000f, 1.000000f, -5.000000f, 0.000000f, 3.000000f, 8.000000f, -5.000000f, 7.000000f, 7.000000f, 6.000000f, -1.000000f, 3.000000f, 3.000000f, -6.000000f, 4.000000f, 6.000000f, 1.000000f, 4.000000f, -4.000000f, 1.000000f, 6.000000f, 0.000000f, 1.000000f, -8.000000f, 8.000000f, 1.000000f, -2.000000f, 0.000000f, 8.000000f, 1.000000f, 1.000000f, 8.000000f, -9.000000f, -4.000000f, -7.000000f, 1.000000f, 8.000000f, -3.000000f, 3.000000f, -3.000000f, -3.000000f, -5.000000f, -3.000000f, -4.000000f, 3.000000f, 10.000000f, -3.000000f, 2.000000f, -1.000000f, 0.000000f, -6.000000f, 5.000000f, 0.000000f, -5.000000f, 6.000000f, -3.000000f, 2.000000f, -7.000000f, 0.000000f, -8.000000f, 1.000000f, 2.000000f, 8.000000f, 3.000000f, -6.000000f, 5.000000f, 0.000000f, 3.000000f, -1.000000f, 4.000000f, 8.000000f, 1.000000f, -3.000000f, -2.000000f, -2.000000f, 6.000000f, -5.000000f, -5.000000f, 0.000000f, 2.000000f, 5.000000f, -1.000000f, -8.000000f, 0.000000f, 10.000000f, -2.000000f, -3.000000f, 5.000000f, 4.000000f, 10.000000f, -3.000000f, 2.000000f, -9.000000f, 5.000000f, 3.000000f, 1.000000f, 2.000000f, 5.000000f, 1.000000f, 3.000000f, 2.000000f, 6.000000f, 5.000000f, 8.000000f, 9.000000f, -1.000000f, 2.000000f, 5.000000f, 7.000000f, 3.000000f, 8.000000f, -5.000000f, 0.000000f, 4.000000f, -9.000000f, 8.000000f, -5.000000f, 7.000000f, 1.000000f, -9.000000f, -1.000000f, 8.000000f, 3.000000f, -4.000000f, -6.000000f, -8.000000f, -1.000000f, 4.000000f, -6.000000f, 3.000000f, -4.000000f, -9.000000f, -7.000000f, -2.000000f, -1.000000f, -4.000000f, 4.000000f, -7.000000f, -2.000000f, 0.000000f, 8.000000f, -3.000000f, 0.000000f, 1.000000f, 6.000000f, 2.000000f, 3.000000f, -4.000000f, -1.000000f, -3.000000f, -7.000000f, 3.000000f, 5.000000f, 3.000000f, 6.000000f, 7.000000f, -1.000000f, 1.000000f, 3.000000f, -3.000000f, -6.000000f, 6.000000f, -4.000000f, 2.000000f, 6.000000f, -3.000000f, 1.000000f, -8.000000f, 5.000000f, 4.000000f, -7.000000f, 7.000000f, 2.000000f, -4.000000f, 10.000000f, 1.000000f, 3.000000f, 10.000000f, 4.000000f, 4.000000f, -3.000000f, 6.000000f, 6.000000f, 2.000000f, 1.000000f, 9.000000f, 5.000000f, 6.000000f, -1.000000f, 2.000000f, 5.000000f, 3.000000f, 2.000000f, 3.000000f, -6.000000f, 1.000000f, -8.000000f, 1.000000f, -4.000000f, 5.000000f, -6.000000f, -2.000000f, 0.000000f, 1.000000f, -6.000000f, -8.000000f, 2.000000f, 4.000000f, -4.000000f, -2.000000f, 7.000000f, -4.000000f, -3.000000f, -6.000000f, -2.000000f, 6.000000f, 3.000000f, 7.000000f, 5.000000f, 2.000000f, -3.000000f, 0.000000f, 8.000000f, 5.000000f, -1.000000f, -7.000000f, 0.000000f, -1.000000f, 6.000000f, -1.000000f, 11.000000f, -1.000000f, 6.000000f, 4.000000f, 1.000000f, -2.000000f, -3.000000f, 5.000000f, 5.000000f, 4.000000f, 6.000000f, -2.000000f, -4.000000f, 0.000000f, 3.000000f, 6.000000f, -2.000000f, 9.000000f, 10.000000f, 7.000000f, 0.000000f, 5.000000f, 4.000000f, 9.000000f, 5.000000f, 5.000000f, -1.000000f, 11.000000f, 1.000000f, -2.000000f, -2.000000f, 11.000000f, 0.000000f, 0.000000f, 5.000000f, -7.000000f, 1.000000f, 6.000000f, 7.000000f, 6.000000f, 3.000000f, -5.000000f, -5.000000f, 8.000000f, 1.000000f, 1.000000f, 9.000000f, 1.000000f, -5.000000f, -5.000000f, -10.000000f, 5.000000f, 0.000000f, 2.000000f, -10.000000f, 1.000000f, 9.000000f, 4.000000f, 0.000000f, -8.000000f, 1.000000f};

static const float test_5_out_scale = 1.0f;
static const float test_5_out_zero_point = 0.0f;
static const int8_t test_5_out_scales_frac[] = {0};
static const int test_5_out_sa_dim = -1;

static const int test_5_out_fx8_frac = 0;

#define TEST_5_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 9, 4, 15}, \
    /* .rank =  */ 4


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

static const float test_6_out_data[] = {6.000000f, -4.000000f, -2.000000f, 12.000000f, -21.000000f, 7.000000f, 11.000000f, 9.000000f, -2.000000f, 12.000000f, -8.000000f, 31.000000f, -2.000000f, 9.000000f, 5.000000f, 7.000000f, 10.000000f, 21.000000f, 14.000000f, 6.000000f, -5.000000f, 1.000000f, 5.000000f, 24.000000f, 12.000000f, -13.000000f, 18.000000f, -8.000000f, -4.000000f, 5.000000f, 6.000000f, 11.000000f, 0.000000f, 5.000000f, 3.000000f, 7.000000f, 17.000000f, 7.000000f, -4.000000f, 11.000000f, 3.000000f, -1.000000f, -3.000000f, 7.000000f, 3.000000f, 21.000000f, 18.000000f, 15.000000f, -19.000000f, -23.000000f, 6.000000f, 15.000000f, 22.000000f, 18.000000f, 6.000000f, -4.000000f, -1.000000f, -8.000000f, -3.000000f, 4.000000f, 10.000000f, -4.000000f, -4.000000f, 7.000000f, 25.000000f, -23.000000f, 1.000000f, 3.000000f, 4.000000f, 14.000000f, 19.000000f, 2.000000f, 7.000000f, -1.000000f, 27.000000f, 7.000000f, 0.000000f, -1.000000f, 14.000000f, 25.000000f, 9.000000f, -15.000000f, 5.000000f, -15.000000f, 8.000000f, -2.000000f, 10.000000f, 4.000000f, 9.000000f, 20.000000f, -1.000000f, -20.000000f, -8.000000f, 15.000000f, -11.000000f, -7.000000f, -1.000000f, 5.000000f, 1.000000f, -13.000000f, 1.000000f, -14.000000f, 1.000000f, -8.000000f, 19.000000f, 1.000000f, -3.000000f, 1.000000f, 7.000000f, -8.000000f, 12.000000f, -12.000000f, -4.000000f, 14.000000f, 11.000000f, 20.000000f, 8.000000f, 12.000000f, 9.000000f, 6.000000f};

static const float test_6_out_scale = 1.0f;
static const float test_6_out_zero_point = 0.0f;
static const int8_t test_6_out_scales_frac[] = {0};
static const int test_6_out_sa_dim = -1;

static const int test_6_out_fx8_frac = 0;

#define TEST_6_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {2, 1, 4, 15}, \
    /* .rank =  */ 4


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

static const float test_7_out_data[] = {26.000000f, 29.000000f, -14.000000f, 47.000000f, 2.000000f, 7.000000f, 29.000000f, -15.000000f, 15.000000f, 62.000000f, -9.000000f, 65.000000f, -3.000000f, 54.000000f, -2.000000f, 13.000000f, 41.000000f, -7.000000f};

static const float test_7_out_scale = 1.0f;
static const float test_7_out_zero_point = 0.0f;
static const int8_t test_7_out_scales_frac[] = {0};
static const int test_7_out_sa_dim = -1;

static const int test_7_out_fx8_frac = 0;

#define TEST_7_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {1, 6, 3}, \
    /* .rank =  */ 3


static const mli_tensor test_7_out_tsr_sa8 = {
    TEST_7_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_7_out_tsr_fx8 = {
    TEST_7_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_7_out_tsr_fx16 = {
    TEST_7_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_7_out_tsr_sa32 = {
    TEST_7_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_32,
    /* .el_params = */ { 0 }
};

static const float test_8_out_data[] = {26.000000f, 9.000000f, 7.000000f, 22.000000f, 11.000000f, 0.000000f, -14.000000f, 11.000000f, -12.000000f, 13.000000f, -7.000000f, 14.000000f, 17.000000f, -2.000000f, 15.000000f, 2.000000f, 7.000000f, 13.000000f, 2.000000f, 2.000000f, -12.000000f, 23.000000f, 31.000000f, 22.000000f, 8.000000f, 20.000000f, 23.000000f, 32.000000f, -8.000000f, -16.000000f, 10.000000f, 21.000000f, 8.000000f, -1.000000f, 7.000000f, 11.000000f, -8.000000f, -5.000000f, 0.000000f, 27.000000f, 8.000000f, 3.000000f};

static const float test_8_out_scale = 1.0f;
static const float test_8_out_zero_point = 0.0f;
static const int8_t test_8_out_scales_frac[] = {0};
static const int test_8_out_sa_dim = -1;

static const int test_8_out_fx8_frac = 0;

#define TEST_8_OUT_TSR_SHARED_DESCR \
    /* .data = */ { 0 },\
    /* .mem_stride = */ {0}, \
    /* .shape = */ {7, 6, 1}, \
    /* .rank =  */ 3


static const mli_tensor test_8_out_tsr_sa8 = {
    TEST_8_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_SA_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_8_out_tsr_fx8 = {
    TEST_8_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_8,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_8_out_tsr_fx16 = {
    TEST_8_OUT_TSR_SHARED_DESCR,

    /* .el_type = */ MLI_EL_FX_16,
    /* .el_params = */ { 0 }
};


static const mli_tensor test_8_out_tsr_sa32 = {
    TEST_8_OUT_TSR_SHARED_DESCR,

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

tensor_quantizer test_7_out_sa8(test_7_out_tsr_sa8, test_7_out_sa_dim, test_7_out_data, 
sizeof(test_7_out_data) / sizeof(test_7_out_data[0]), &test_7_out_scale, 1, &test_7_out_zero_point, 1, test_7_out_scales_frac, 1);
tensor_quantizer test_7_out_fx8(test_7_out_tsr_fx8, test_7_out_fx8_frac, test_7_out_data, sizeof(test_7_out_data) / sizeof(test_7_out_data[0]));
tensor_quantizer test_7_out_fx16(test_7_out_tsr_fx16, test_7_out_fx8_frac + 8, test_7_out_data, sizeof(test_7_out_data) / sizeof(test_7_out_data[0]));
tensor_quantizer test_7_out_sa32(test_7_out_tsr_sa32, test_7_out_sa_dim, test_7_out_data, 
sizeof(test_7_out_data) / sizeof(test_7_out_data[0]), &test_7_out_scale, 1, &test_7_out_zero_point, 1, test_7_out_scales_frac, 1);

tensor_quantizer test_8_out_sa8(test_8_out_tsr_sa8, test_8_out_sa_dim, test_8_out_data, 
sizeof(test_8_out_data) / sizeof(test_8_out_data[0]), &test_8_out_scale, 1, &test_8_out_zero_point, 1, test_8_out_scales_frac, 1);
tensor_quantizer test_8_out_fx8(test_8_out_tsr_fx8, test_8_out_fx8_frac, test_8_out_data, sizeof(test_8_out_data) / sizeof(test_8_out_data[0]));
tensor_quantizer test_8_out_fx16(test_8_out_tsr_fx16, test_8_out_fx8_frac + 8, test_8_out_data, sizeof(test_8_out_data) / sizeof(test_8_out_data[0]));
tensor_quantizer test_8_out_sa32(test_8_out_tsr_sa32, test_8_out_sa_dim, test_8_out_data, 
sizeof(test_8_out_data) / sizeof(test_8_out_data[0]), &test_8_out_scale, 1, &test_8_out_zero_point, 1, test_8_out_scales_frac, 1);

const lib_mli::ReduceOpConfig test_1_cfg = {/* .reduce_axis = */0};
const lib_mli::ReduceOpConfig test_2_cfg = {/* .reduce_axis = */1};
const lib_mli::ReduceOpConfig test_3_cfg = {/* .reduce_axis = */2};
const lib_mli::ReduceOpConfig test_4_cfg = {/* .reduce_axis = */3};
