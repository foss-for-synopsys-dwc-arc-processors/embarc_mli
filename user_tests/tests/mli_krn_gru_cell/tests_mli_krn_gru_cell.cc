/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_api.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "mli_types.h"
#include "test_tensor_quantizer.h"
#include "test_report.h"

#include "vectors_mli_krn_gru_cell.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

typedef mli_status(*gru_cell_func_ptr)(
    const mli_tensor* /*input*/,
    const mli_tensor* /*prev_out*/,
    const mli_tensor* /*weights_in*/,
    const mli_tensor* /*weights_out*/,
    const mli_tensor* /*bias*/,
    const mli_lut* /*tanh lut*/,
    const mli_lut* /*sigm lut*/,
    const mli_rnn_cell_cfg* /*cfg*/,
    mli_tensor* /*output*/);

struct gru_cell_test_operands {
    const char* descr;
    const gru_cell_func_ptr mli_krn_gru_cell;
    tensor_quantizer in;
    tensor_quantizer prev_out;
    tensor_quantizer weights_in;
    tensor_quantizer weights_out;
    tensor_quantizer bias;
    tensor_quantizer out;
    mli_rnn_cell_cfg cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

// Checksums of test tensors for various mli calculations mode. 
// When developer finished implementation of kernel and consider it as ok, one needs to populate
// proper checksums for tests in order to highlight any change which affects results.
#if defined(CRC_RM_UP)
const crc32_calc test_1_chksum_fx16{ 0xCA3B3621 }, test_1_chksum_fx16_fx8_fx8{ 0x7C81E8FA }, test_1_chksum_sa8{ 0xBA369AB3 },
                 test_2_chksum_fx16{ 0xCA3B3621 }, test_2_chksum_fx16_fx8_fx8{ 0x7C81E8FA }, test_2_chksum_sa8{ 0xBA369AB3 },
                 test_3_chksum_fx16{ 0x0575B7B5 }, test_3_chksum_fx16_fx8_fx8{ 0x3105731C }, test_3_chksum_sa8{ 0xD7D30910 },
                 test_4_chksum_fx16{ 0x0FF47FA3 }, test_4_chksum_fx16_fx8_fx8{ 0x86FB1945 }, test_4_chksum_sa8{ 0x82502137 },
                 test_5_chksum_fx16{ 0xA957E40B }, test_5_chksum_fx16_fx8_fx8{ 0x44D14AA8 }, test_5_chksum_sa8{ 0x551350E9 },
                 test_6_chksum_fx16{ 0xA3D62C1D }, test_6_chksum_fx16_fx8_fx8{ 0xF32F20F1 }, test_6_chksum_sa8{ 0x009078CE },
                 test_7_chksum_fx16{ 0xA9D9FC7B }, test_7_chksum_fx16_fx8_fx8{ 0xB08CE82C }, test_7_chksum_sa8{ 0xA9AED75B },
                 test_8_chksum_fx16{ 0x82B87A3D }, test_8_chksum_fx16_fx8_fx8{ 0x1D12879D }, test_8_chksum_sa8{ 0xBC364FC3 };

#elif defined(CRC_RM_CONVERGENT)
// TODO: remove after fixing mli_math_acc_ashift_fx() and supporting acc40 shift with round
#if defined(__FXAPI__)
const crc32_calc test_1_chksum_fx16{ 0xE5852A3E }, test_1_chksum_fx16_fx8_fx8{ 0xF979CA35 }, test_1_chksum_sa8{ 0xBA369AB3 },
                 test_2_chksum_fx16{ 0xE5852A3E }, test_2_chksum_fx16_fx8_fx8{ 0xF979CA35 }, test_2_chksum_sa8{ 0xBA369AB3 },
                 test_3_chksum_fx16{ 0x6F7E4D9B }, test_3_chksum_fx16_fx8_fx8{ 0xE47B56B4 }, test_3_chksum_sa8{ 0xB0B3B302 },
                 test_4_chksum_fx16{ 0x65FF858D }, test_4_chksum_fx16_fx8_fx8{ 0x53853CED }, test_4_chksum_sa8{ 0xE5309B25 },
                 test_5_chksum_fx16{ 0x3A84CF63 }, test_5_chksum_fx16_fx8_fx8{ 0x202E9565 }, test_5_chksum_sa8{ 0xE0C80764 },
                 test_6_chksum_fx16{ 0x30050775 }, test_6_chksum_fx16_fx8_fx8{ 0x97D0FF3C }, test_6_chksum_sa8{ 0xB54B2F43 },
                 test_7_chksum_fx16{ 0xD81EFB70 }, test_7_chksum_fx16_fx8_fx8{ 0x7C0CE29B }, test_7_chksum_sa8{ 0xBB1B78B5 },
                 test_8_chksum_fx16{ 0x31D77812 }, test_8_chksum_fx16_fx8_fx8{ 0x1D12879D }, test_8_chksum_sa8{ 0xBC364FC3 };
#else
const crc32_calc test_1_chksum_fx16{ 0xCA3B3621 }, test_1_chksum_fx16_fx8_fx8{ 0xF979CA35 }, test_1_chksum_sa8{ 0xBA369AB3 },
                 test_2_chksum_fx16{ 0xCA3B3621 }, test_2_chksum_fx16_fx8_fx8{ 0xF979CA35 }, test_2_chksum_sa8{ 0xBA369AB3 },
                 test_3_chksum_fx16{ 0x0575B7B5 }, test_3_chksum_fx16_fx8_fx8{ 0xE47B56B4 }, test_3_chksum_sa8{ 0xB0B3B302 },
                 test_4_chksum_fx16{ 0x0FF47FA3 }, test_4_chksum_fx16_fx8_fx8{ 0x53853CED }, test_4_chksum_sa8{ 0xE5309B25 },
                 test_5_chksum_fx16{ 0x4DEDC850 }, test_5_chksum_fx16_fx8_fx8{ 0x202E9565 }, test_5_chksum_sa8{ 0xE0C80764 },
                 test_6_chksum_fx16{ 0x476C0046 }, test_6_chksum_fx16_fx8_fx8{ 0x97D0FF3C }, test_6_chksum_sa8{ 0xB54B2F43 },
                 test_7_chksum_fx16{ 0xA9D9FC7B }, test_7_chksum_fx16_fx8_fx8{ 0x7C0CE29B }, test_7_chksum_sa8{ 0xBB1B78B5 },
                 test_8_chksum_fx16{ 0x82B87A3D }, test_8_chksum_fx16_fx8_fx8{ 0x1D12879D }, test_8_chksum_sa8{ 0xBC364FC3 };
#endif
#else // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16, test_1_chksum_fx16_fx8_fx8, test_1_chksum_sa8,
                  test_2_chksum_fx16, test_2_chksum_fx16_fx8_fx8, test_2_chksum_sa8,
                  test_3_chksum_fx16, test_3_chksum_fx16_fx8_fx8, test_3_chksum_sa8,
                  test_4_chksum_fx16, test_4_chksum_fx16_fx8_fx8, test_4_chksum_sa8,
                  test_5_chksum_fx16, test_5_chksum_fx16_fx8_fx8, test_5_chksum_sa8,
                  test_6_chksum_fx16, test_6_chksum_fx16_fx8_fx8, test_6_chksum_sa8,
                  test_7_chksum_fx16, test_7_chksum_fx16_fx8_fx8, test_7_chksum_sa8,
                  test_8_chksum_fx16, test_8_chksum_fx16_fx8_fx8, test_8_chksum_sa8;
#endif


const quality_metrics thresholds_fx16_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR_DB = */30.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_fx16_fx8_fx8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                  /* SNR_DB = */30.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                             /* SNR_DB = */20.f, quality_metrics::kPassValueQuantErrPerc };


static gru_cell_test_operands tests_list[] = {

#ifdef COMPILE_FOR_FX16
    // One-to-one, RNN_OUT_LAST, Forward processing
    {"Test 1 FX16 OtO,Forw",          mli_krn_gru_cell_fx16, 
                                      input_1_fx16, hidden_1_fx16, weights_1_in_fx16, weights_1_out_fx16, bias_1_fx16, 
                                      test_1_out_fx16, test_1_cfg, thresholds_fx16_general, test_1_chksum_fx16},
    // One-to-one, RNN_OUT_LAST, Backward processing
    {"Test 2 FX16 OtO,Back",          mli_krn_gru_cell_fx16, 
                                      input_1_fx16, hidden_1_fx16, weights_1_in_fx16, weights_1_out_fx16, bias_1_fx16, 
                                      test_1_out_fx16, test_2_cfg, thresholds_fx16_general, test_2_chksum_fx16},
    // Batch-to-batch, RNN_OUT_ALL, Forward processing
    {"Test 3 FX16 BtB,ALL,Forw",      mli_krn_gru_cell_fx16, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_fx16, weights_2_out_fx16, bias_1_fx16, 
                                      test_3_out_fx16, test_3_cfg, thresholds_fx16_general, test_3_chksum_fx16},    
    // Batch-to-batch, RNN_OUT_ALL, Forward processing, memstrides
    {"Test 4 FX16 BtB,ALL,F,ms",      mli_krn_gru_cell_fx16, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_memstr_fx16, weights_2_out_memstr_fx16, 
                                      bias_1_fx16, test_3_out_fx16, test_3_cfg, thresholds_fx16_general, 
                                      test_4_chksum_fx16},
    // Batch-to-batch, RNN_OUT_ALL, Backward processing
    {"Test 5 FX16 BtB,ALL,Back",      mli_krn_gru_cell_fx16, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_fx16, weights_2_out_fx16, bias_1_fx16, 
                                      test_4_out_fx16, test_4_cfg, thresholds_fx16_general, test_5_chksum_fx16},    
    // Batch-to-batch, RNN_OUT_ALL, Backward processing, memstrides
    {"Test 6 FX16 BtB,ALL,B,ms",      mli_krn_gru_cell_fx16, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_memstr_fx16, weights_2_out_memstr_fx16, 
                                      bias_1_fx16, test_4_out_fx16, test_4_cfg, thresholds_fx16_general, 
                                      test_6_chksum_fx16},
    // Batch-to-batch, RNN_OUT_LAST, Forward processing
    {"Test 7 FX16 BtB,LAST,Forw",     mli_krn_gru_cell_fx16, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_fx16, weights_2_out_fx16, bias_1_fx16, 
                                      test_5_out_fx16, test_5_cfg, thresholds_fx16_general, test_7_chksum_fx16},
    // Batch-to-batch, RNN_OUT_LAST, Backward processing
    {"Test 8 FX16 BtB,LAST,Back",     mli_krn_gru_cell_fx16, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_fx16, weights_2_out_fx16, bias_1_fx16, 
                                      test_6_out_fx16, test_6_cfg, thresholds_fx16_general, test_8_chksum_fx16},
    
#elif COMPILE_FOR_FX16_FX8_FX8
    {"Test 1 FX16_FX8 OtO,Forw",      mli_krn_gru_cell_fx16_fx8_fx8, 
                                      input_1_fx16, hidden_1_fx16, weights_1_in_fx8, weights_1_out_fx8, bias_1_fx8, 
                                      test_1_out_fx16, test_1_cfg, thresholds_fx16_fx8_fx8_general, 
                                      test_1_chksum_fx16_fx8_fx8},
    {"Test 2 FX16_FX8 OtO,Back",      mli_krn_gru_cell_fx16_fx8_fx8, 
                                      input_1_fx16, hidden_1_fx16, weights_1_in_fx8, weights_1_out_fx8, bias_1_fx8, 
                                      test_1_out_fx16, test_2_cfg, thresholds_fx16_fx8_fx8_general, 
                                      test_2_chksum_fx16_fx8_fx8},
    {"Test 3 FX16_FX8 BtB,ALL,Forw",  mli_krn_gru_cell_fx16_fx8_fx8, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_fx8, weights_2_out_fx8, bias_1_fx8, 
                                      test_3_out_fx16, test_3_cfg, thresholds_fx16_fx8_fx8_general, 
                                      test_3_chksum_fx16_fx8_fx8},
    {"Test 4 FX16_FX8 BtB,ALL,F,ms",  mli_krn_gru_cell_fx16_fx8_fx8, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_memstr_fx8, weights_2_out_memstr_fx8, 
                                      bias_1_fx8, test_3_out_fx16, test_3_cfg, thresholds_fx16_fx8_fx8_general, 
                                      test_4_chksum_fx16_fx8_fx8},
    {"Test 5 FX16_FX8 BtB,ALL,Back",  mli_krn_gru_cell_fx16_fx8_fx8, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_fx8, weights_2_out_fx8, bias_1_fx8, 
                                      test_4_out_fx16, test_4_cfg, thresholds_fx16_fx8_fx8_general, 
                                      test_5_chksum_fx16_fx8_fx8},
    {"Test 6 FX16_FX8 BtB,ALL,B,ms",  mli_krn_gru_cell_fx16_fx8_fx8, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_memstr_fx8, weights_2_out_memstr_fx8, 
                                      bias_1_fx8, test_4_out_fx16, test_4_cfg, thresholds_fx16_fx8_fx8_general, 
                                      test_6_chksum_fx16_fx8_fx8},
    {"Test 7 FX16_FX8 BtB,LAST,Forw", mli_krn_gru_cell_fx16_fx8_fx8, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_fx8, weights_2_out_fx8, bias_1_fx8, 
                                      test_5_out_fx16, test_5_cfg, thresholds_fx16_fx8_fx8_general, 
                                      test_7_chksum_fx16_fx8_fx8},
    {"Test 8 FX16_FX8 BtB,LAST,Back", mli_krn_gru_cell_fx16_fx8_fx8, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_fx8, weights_2_out_fx8, bias_1_fx8,
                                      test_6_out_fx16, test_6_cfg, thresholds_fx16_fx8_fx8_general, 
                                      test_8_chksum_fx16_fx8_fx8},
#elif COMPILE_FOR_SA8_SA8_SA32
    {"Test 1 SA8_SA32 OtO,Forw",      mli_krn_gru_cell_sa8_sa8_sa32,
                                      input_1_sa8, hidden_1_sa8, weights_1_in_sa8, weights_1_out_sa8, bias_1_i1_w1_sa32, 
                                      test_1_out_sa8, test_1_cfg, thresholds_sa8_general, test_1_chksum_sa8},
    {"Test 2 SA8_SA32 OtO,Back",      mli_krn_gru_cell_sa8_sa8_sa32,
                                      input_1_sa8, hidden_1_sa8, weights_1_in_sa8, weights_1_out_sa8, bias_1_i1_w1_sa32, 
                                      test_1_out_sa8, test_2_cfg, thresholds_sa8_general, test_2_chksum_sa8},
    {"Test 3 SA8_SA32 BtB,ALL,Forw",  mli_krn_gru_cell_sa8_sa8_sa32,
                                      input_2_sa8, hidden_1_sa8, weights_2_in_sa8, weights_2_out_sa8, bias_1_i2_w2_sa32, 
                                      test_3_out_sa8, test_3_cfg, thresholds_sa8_general, test_3_chksum_sa8},
    {"Test 4 SA8_SA32 BtB,ALL,F,ms",  mli_krn_gru_cell_sa8_sa8_sa32,
                                      input_2_sa8, hidden_1_sa8, weights_2_in_memstr_sa8, weights_2_out_memstr_sa8, 
                                      bias_1_i2_w2_sa32, test_3_out_sa8, test_3_cfg, thresholds_sa8_general, 
                                      test_4_chksum_sa8},
    {"Test 5 SA8_SA32 BtB,ALL,Back",  mli_krn_gru_cell_sa8_sa8_sa32,
                                      input_2_sa8, hidden_1_sa8, weights_2_in_sa8, weights_2_out_sa8, bias_1_i2_w2_sa32, 
                                      test_4_out_sa8, test_4_cfg, thresholds_sa8_general, test_5_chksum_sa8},
    {"Test 6 SA8_SA32 BtB,ALL,B,ms",  mli_krn_gru_cell_sa8_sa8_sa32,
                                      input_2_sa8, hidden_1_sa8, weights_2_in_memstr_sa8, weights_2_out_memstr_sa8, 
                                      bias_1_i2_w2_sa32, test_4_out_sa8, test_4_cfg, thresholds_sa8_general, 
                                      test_6_chksum_sa8},
    {"Test 7 SA8_SA32 BtB,LAST,Forw", mli_krn_gru_cell_sa8_sa8_sa32,
                                      input_2_sa8, hidden_1_sa8, weights_2_in_sa8, weights_2_out_sa8, bias_1_i2_w2_sa32, 
                                      test_5_out_sa8, test_5_cfg, thresholds_sa8_general, test_7_chksum_sa8},
    {"Test 8 SA8_SA32 BtB,LAST,Back", mli_krn_gru_cell_sa8_sa8_sa32,
                                      input_2_sa8, hidden_1_sa8, weights_2_in_sa8, weights_2_out_sa8, bias_1_i2_w2_sa32, 
                                      test_6_out_sa8, test_6_cfg, thresholds_sa8_general, test_8_chksum_sa8},
#else
#error incorrect make configuration
#endif
};

constexpr int kMemInputSize = 90;
constexpr int kMemOutSize = 410;
constexpr int kMemWeightSize = 2650;
constexpr int kMemIrSize = 120;
static IO_DATA_ATTR int8_t scratch_mem_in[kMemInputSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_prev_out[kMemInputSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemOutSize] = { 0 };
static W_DATA_ATTR int8_t scratch_mem_w_in[kMemWeightSize] = { 0 };
static W_DATA_ATTR int8_t scratch_mem_w_out[kMemWeightSize] = { 0 };
static W_DATA_ATTR int8_t scratch_mem_b[kMemOutSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_tanh_lut[kMemOutSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_sigm_lut[kMemOutSize] = { 0 };
static IO_DATA_ATTR int32_t scratch_mem_ir_tensor[kMemIrSize] = { 0 };

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Kernels|GRU Cell Tests");
    mli_lut tanh_lut;
    bool lut_status = true;
    int tanh_lut_size = mli_krn_tanh_get_lut_size();
    lut_status = lut_status && (tanh_lut_size < sizeof(scratch_mem_tanh_lut));
    tanh_lut.data.mem.void_p = (void*) scratch_mem_tanh_lut;
    tanh_lut.data.capacity = sizeof(scratch_mem_tanh_lut);
    lut_status = lut_status && (mli_krn_tanh_create_lut(&tanh_lut) == MLI_STATUS_OK);

    mli_lut sigm_lut;
    int sigm_lut_size = mli_krn_sigm_get_lut_size();
    lut_status = lut_status && (sigm_lut_size < sizeof(scratch_mem_sigm_lut));
    sigm_lut.data.mem.void_p = (void*) scratch_mem_sigm_lut;
    sigm_lut.data.capacity = sizeof(scratch_mem_sigm_lut);
    lut_status = lut_status && (mli_krn_sigm_create_lut(&sigm_lut) == MLI_STATUS_OK);

    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
        memory_manager mem_prev_out_keeper((int8_t*)(scratch_mem_prev_out), sizeof(scratch_mem_prev_out));
        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        memory_manager mem_w_in_keeper((int8_t*)(scratch_mem_w_in), sizeof(scratch_mem_w_in));
        memory_manager mem_w_out_keeper((int8_t*)(scratch_mem_w_out), sizeof(scratch_mem_w_out));
        memory_manager mem_b_keeper((int8_t*)(scratch_mem_b), sizeof(scratch_mem_b));
        memory_manager mem_ir_keeper((int8_t*)(scratch_mem_ir_tensor), sizeof(scratch_mem_ir_tensor));
        bool is_test_passed = true;
        gru_cell_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metrics;
        if (!(lut_status)) {
            reporter.report_message(cur_test->descr, "FAILED at init: LUT error");
            is_test_passed = false;
        }

#if defined(__Xvec_guard_bit_option) && (__Xvec_guard_bit_option == 0)
        if (strstr(cur_test->descr, "Test 1 SA8_SA32 OtO,Forw") != nullptr ||
            strstr(cur_test->descr, "Test 2 SA8_SA32 OtO,Back") != nullptr ||
            strstr(cur_test->descr, "Test 3 FX16 BtB,ALL,Forw") != nullptr ||
            strstr(cur_test->descr, "Test 3 SA8_SA32 BtB,ALL,Forw") != nullptr ||
            strstr(cur_test->descr, "Test 4 FX16 BtB,ALL,F,ms") != nullptr ||
            strstr(cur_test->descr, "Test 4 SA8_SA32 BtB,ALL,F,ms") != nullptr ||
            strstr(cur_test->descr, "Test 5 FX16 BtB,ALL,Back") != nullptr ||
            strstr(cur_test->descr, "Test 5 SA8_SA32 BtB,ALL,Back") != nullptr ||
            strstr(cur_test->descr, "Test 6 FX16 BtB,ALL,B,ms") != nullptr ||
            strstr(cur_test->descr, "Test 6 SA8_SA32 BtB,ALL,B,ms") != nullptr ||
            strstr(cur_test->descr, "Test 7 FX16 BtB,LAST,Forw") != nullptr ||
            strstr(cur_test->descr, "Test 7 SA8_SA32 BtB,LAST,Forw") != nullptr ||
            strstr(cur_test->descr, "Test 8 FX16 BtB,LAST,Back") != nullptr ||
            strstr(cur_test->descr, "Test 8 SA8_SA32 BtB,LAST,Back") != nullptr) {
            // VPX fails bitwise comparison with reference .
            reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
            continue;
        }
#endif

        if (!(cur_test->in.is_valid() && cur_test->prev_out.is_valid() && cur_test->weights_in.is_valid() && 
            cur_test->weights_out.is_valid() && cur_test->bias.is_valid() && cur_test->out.is_valid())) {
            reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
            is_test_passed = false;
        }

        mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
        mli_tensor prev_out = cur_test->prev_out.get_quantized_tensor(
            mem_prev_out_keeper.allocate_memory(cur_test->prev_out));
        mli_tensor weights_in = cur_test->weights_in.get_quantized_tensor(
            mem_w_in_keeper.allocate_memory(cur_test->weights_in));
        mli_tensor weights_out = cur_test->weights_out.get_quantized_tensor(
            mem_w_out_keeper.allocate_memory(cur_test->weights_out));
        mli_tensor bias = cur_test->bias.get_quantized_tensor(mem_b_keeper.allocate_memory(cur_test->bias));
        mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
        mli_tensor source_out_tensor = out;

        const mli_rnn_cell_cfg cur_test_cfg = {
        /* .direction = */ cur_test->cfg.direction,
        /* .results = */ cur_test->cfg.results,
        /* .act = */ cur_test->cfg.act,
        /* .scratch_data = */ mem_ir_keeper.allocate_memory(kMemIrSize)
        };

        if (is_test_passed &&
                (tensor_quantizer::validate_tensor(input) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(prev_out) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(weights_in) != tensor_quantizer::kOk||
                 tensor_quantizer::validate_tensor(weights_out) != tensor_quantizer::kOk||
                 tensor_quantizer::validate_tensor(bias) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(out) != tensor_quantizer::kOk)) {
            reporter.report_message(cur_test->descr, 
                                    "FAILED at quantization step: more memory for one of tensors might be required");
            is_test_passed = false;
        }

        if (is_test_passed &&
                (mem_in_keeper.is_memory_corrupted() || mem_prev_out_keeper.is_memory_corrupted() || 
                mem_out_keeper.is_memory_corrupted() || mem_w_in_keeper.is_memory_corrupted() || 
                mem_w_out_keeper.is_memory_corrupted() || mem_b_keeper.is_memory_corrupted() || 
                mem_ir_keeper.is_memory_corrupted())) {
            reporter.report_message(cur_test->descr,
                "FAILED at quantization step: memory beside one of operands is corrupted");
            is_test_passed = false;
        }

        // Run specific kernel for test
        if (is_test_passed &&
                cur_test->mli_krn_gru_cell
                (&input, &prev_out, &weights_in, &weights_out, &bias, &tanh_lut, &sigm_lut, &cur_test_cfg, &out) 
                != MLI_STATUS_OK) {
            reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
            is_test_passed = false;
        }

        if (is_test_passed &&
                test_metrics.calculate_metrics(out, cur_test->out) == false) {
            reporter.report_message(cur_test->descr, "FAILED at comparison output with reference");
            is_test_passed = false;
        }

        // Check that kernel didn't modify quantization parameters provided by user.
        if (is_test_passed) {
            bool is_per_tensor_quant = true;

            if (out.el_type == MLI_EL_FX_8 || out.el_type == MLI_EL_FX_16) {
                is_test_passed &= out.el_params.fx.frac_bits == source_out_tensor.el_params.fx.frac_bits;
            } else if (out.el_type == MLI_EL_SA_8 || out.el_type == MLI_EL_SA_32) {
                if (out.el_params.sa.dim < 0 || source_out_tensor.el_params.sa.dim < 0) {
                    is_test_passed &=
                        (out.el_params.sa.scale.mem.i16 == source_out_tensor.el_params.sa.scale.mem.i16) &&
                        (out.el_params.sa.zero_point.mem.i16 == source_out_tensor.el_params.sa.zero_point.mem.i16) &&
                        (out.el_params.sa.scale_frac_bits.mem.i8 ==
                            source_out_tensor.el_params.sa.scale_frac_bits.mem.i8);
                } else {
                    is_per_tensor_quant = false;
                    is_test_passed = false;
                }
            }
            if (!is_test_passed) {
                reporter.report_message(cur_test->descr,
                    is_per_tensor_quant ? "FAILED as element params of output tensor was modified"
                                        : "FAILED as per-axis quantization of output tensor isn't supported");
            }
        }

        if (is_test_passed) {
            crc32_calc data_crc;
            data_crc(input);
            data_crc(prev_out);
            data_crc(weights_in);
            data_crc(weights_out);
            data_crc(bias);
            data_crc(out);

            is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr, test_metrics, cur_test->threshold, 
                                                                data_crc, cur_test->check_sum);
        }
        final_status &= is_test_passed;
    }

    reporter.report_outline("[AUTO] Group: mli_krn_gru_cell", final_status);

    return (final_status) ? 0 : 1;
}
