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

#include "vectors_mli_krn_lstm_cell.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

typedef mli_status(*lstm_cell_func_ptr)(
    const mli_tensor* /*input*/,
    const mli_tensor* /*prev_out*/,
    const mli_tensor* /*weights_in*/,
    const mli_tensor* /*weights_out*/,
    const mli_tensor* /*bias*/,
    const mli_lut* /*tanh lut*/,
    const mli_lut* /*sigm lut*/,
    const mli_rnn_cell_cfg* /*cfg*/,
    mli_tensor* /*cell*/,
    mli_tensor* /*output*/);

struct lstm_cell_test_operands {
    const char* descr;
    const lstm_cell_func_ptr mli_krn_lstm_cell;
    tensor_quantizer in;
    tensor_quantizer prev_out;
    tensor_quantizer weights_in;
    tensor_quantizer weights_out;
    tensor_quantizer bias;
    tensor_quantizer cell;
    tensor_quantizer out;
    mli_rnn_cell_cfg cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

// Checksums of test tensors for various mli calculations mode. 
// When developer finished implementation of kernel and consider it as ok, one needs to populate
// proper checksums for tests in order to highlight any change which affects results.
#if defined(CRC_RM_UP)
const crc32_calc test_1_chksum_fx16{ 0x727F0B6E }, test_1_chksum_fx16_fx8_fx8{ 0xAFEBE943 }, test_1_chksum_sa8{ 0xF965F28D },
                 test_2_chksum_fx16{ 0x727F0B6E }, test_2_chksum_fx16_fx8_fx8{ 0xAFEBE943 }, test_2_chksum_sa8{ 0xF965F28D },
                 test_3_chksum_fx16{ 0x1BA8D258 }, test_3_chksum_fx16_fx8_fx8{ 0xBECE1810 }, test_3_chksum_sa8{ 0xC3B19248 },
                 test_4_chksum_fx16{ 0xFFFE1361 }, test_4_chksum_fx16_fx8_fx8{ 0xB0FE085C }, test_4_chksum_sa8{ 0x6124E3B6 },
                 test_5_chksum_fx16{ 0x1CC3B931 }, test_5_chksum_fx16_fx8_fx8{ 0x3861AB6F }, test_5_chksum_sa8{ 0xA677B81C },
                 test_6_chksum_fx16{ 0x50E1BD02 }, test_6_chksum_fx16_fx8_fx8{ 0x0E81E3BD }, test_6_chksum_sa8{ 0xA2779EEA };
#elif defined(CRC_RM_CONVERGENT)

// TODO: remove after fixing mli_math_acc_ashift_fx() and supporting acc40 shift with round
#if defined(__FXAPI__)
const crc32_calc test_1_chksum_fx16{ 0x58A14685 }, test_1_chksum_fx16_fx8_fx8{ 0xCA8CD205 }, test_1_chksum_sa8{ 0xC7D69291 },
                 test_2_chksum_fx16{ 0x58A14685 }, test_2_chksum_fx16_fx8_fx8{ 0xCA8CD205 }, test_2_chksum_sa8{ 0xC7D69291 },
                 test_3_chksum_fx16{ 0x87979C0D }, test_3_chksum_fx16_fx8_fx8{ 0x6B250DB5 }, test_3_chksum_sa8{ 0xE7F799DC },
                 test_4_chksum_fx16{ 0xA58D556B }, test_4_chksum_fx16_fx8_fx8{ 0x54BABBEE }, test_4_chksum_sa8{ 0x9272AF14 },
                 test_5_chksum_fx16{ 0x7AB58D77 }, test_5_chksum_fx16_fx8_fx8{ 0x9552BA41 }, test_5_chksum_sa8{ 0x60A1B278 },
                 test_6_chksum_fx16{ 0x288FC2AF }, test_6_chksum_fx16_fx8_fx8{ 0xDC558281 }, test_6_chksum_sa8{ 0x18FEB4CA };
#else
const crc32_calc test_1_chksum_fx16{ 0xD05EE688 }, test_1_chksum_fx16_fx8_fx8{ 0xCA8CD205 }, test_1_chksum_sa8{ 0xC7D69291 },
                 test_2_chksum_fx16{ 0xD05EE688 }, test_2_chksum_fx16_fx8_fx8{ 0xCA8CD205 }, test_2_chksum_sa8{ 0xC7D69291 },
                 test_3_chksum_fx16{ 0x83FE4F61 }, test_3_chksum_fx16_fx8_fx8{ 0x6B250DB5 }, test_3_chksum_sa8{ 0xE7F799DC },
                 test_4_chksum_fx16{ 0x16F93ACD }, test_4_chksum_fx16_fx8_fx8{ 0x54BABBEE }, test_4_chksum_sa8{ 0x9272AF14 },
                 test_5_chksum_fx16{ 0xB4752CB0 }, test_5_chksum_fx16_fx8_fx8{ 0x9552BA41 }, test_5_chksum_sa8{ 0x60A1B278 },
                 test_6_chksum_fx16{ 0xFDA9732E }, test_6_chksum_fx16_fx8_fx8{ 0xDC558281 }, test_6_chksum_sa8{ 0x18FEB4CA };
#endif

#else // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16, test_1_chksum_fx16_fx8_fx8, test_1_chksum_sa8,
                  test_2_chksum_fx16, test_2_chksum_fx16_fx8_fx8, test_2_chksum_sa8,
                  test_3_chksum_fx16, test_3_chksum_fx16_fx8_fx8, test_3_chksum_sa8,
                  test_4_chksum_fx16, test_4_chksum_fx16_fx8_fx8, test_4_chksum_sa8,
                  test_5_chksum_fx16, test_5_chksum_fx16_fx8_fx8, test_5_chksum_sa8,
                  test_6_chksum_fx16, test_6_chksum_fx16_fx8_fx8, test_6_chksum_sa8;
#endif


const quality_metrics thresholds_fx16_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR_DB = */40.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_fx16_fx8_fx8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                  /* SNR_DB = */30.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                             /* SNR_DB = */20.f, quality_metrics::kPassValueQuantErrPerc };


static lstm_cell_test_operands tests_list[] = {

    // One-to-one, RNN_OUT_LAST, Forward processing
#ifdef COMPILE_FOR_FX16
    {"Test 1 FX16 OtO,Forw",     mli_krn_lstm_cell_fx16, 
                                 input_1_fx16, hidden_1_fx16, weights_1_in_fx16, weights_1_out_fx16, bias_1_fx16, 
                                 cell_1_fx16, test_1_out_fx16, test_1_cfg, thresholds_fx16_general, test_1_chksum_fx16},
#elif COMPILE_FOR_FX16_FX8_FX8
    {"Test 1 FX16_FX8 OtO,Forw", mli_krn_lstm_cell_fx16_fx8_fx8, 
                                 input_1_fx16, hidden_1_fx16, weights_1_in_fx8, weights_1_out_fx8, bias_1_fx8, 
                                 cell_1_fx16, test_1_out_fx16, test_1_cfg, thresholds_fx16_fx8_fx8_general, 
                                 test_1_chksum_fx16_fx8_fx8},
#elif COMPILE_FOR_SA8_SA8_SA32
    {"Test 1 SA8_SA32 OtO,Forw", mli_krn_lstm_cell_sa8_sa8_sa32,
                                 input_1_sa8, hidden_1_sa8, weights_1_in_sa8, weights_1_out_sa8, bias_1_i1_w1_sa32, 
                                 cell_1_sa8, test_1_out_sa8, test_1_cfg, thresholds_sa8_general, test_1_chksum_sa8},
#else
#error incorrect make configuration
#endif

    // // One-to-one, RNN_OUT_LAST, Backward processing
#ifdef COMPILE_FOR_FX16
    {"Test 2 FX16 OtO,Back",     mli_krn_lstm_cell_fx16, 
                                 input_1_fx16, hidden_1_fx16, weights_1_in_fx16, weights_1_out_fx16, bias_1_fx16, 
                                 cell_1_fx16, test_1_out_fx16, test_2_cfg, thresholds_fx16_general, test_2_chksum_fx16},
#elif COMPILE_FOR_FX16_FX8_FX8
    {"Test 2 FX16_FX8 OtO,Back", mli_krn_lstm_cell_fx16_fx8_fx8, 
                                 input_1_fx16, hidden_1_fx16, weights_1_in_fx8, weights_1_out_fx8, bias_1_fx8, 
                                 cell_1_fx16, test_1_out_fx16, test_2_cfg, thresholds_fx16_fx8_fx8_general, 
                                 test_2_chksum_fx16_fx8_fx8},
#elif COMPILE_FOR_SA8_SA8_SA32
    {"Test 2 SA8_SA32 OtO,Back", mli_krn_lstm_cell_sa8_sa8_sa32,
                                 input_1_sa8, hidden_1_sa8, weights_1_in_sa8, weights_1_out_sa8, bias_1_i1_w1_sa32, 
                                 cell_1_sa8, test_1_out_sa8, test_2_cfg, thresholds_sa8_general, test_2_chksum_sa8},
#else
#error incorrect make configuration
#endif

    // // Batch-to-batch, RNN_OUT_ALL, Forward processing
#ifdef COMPILE_FOR_FX16
    {"Test 3 FX16 BtB,ALL,Forw",     mli_krn_lstm_cell_fx16, 
                                     input_2_fx16, hidden_1_fx16, weights_2_in_fx16, weights_2_out_fx16, bias_1_fx16, 
                                     cell_1_fx16, test_3_out_fx16, test_3_cfg, thresholds_fx16_general, test_3_chksum_fx16},
#elif COMPILE_FOR_FX16_FX8_FX8
    {"Test 3 FX16_FX8 BtB,ALL,Forw", mli_krn_lstm_cell_fx16_fx8_fx8, 
                                     input_2_fx16, hidden_1_fx16, weights_2_in_fx8, weights_2_out_fx8, bias_1_fx8, 
                                     cell_1_fx16, test_3_out_fx16, test_3_cfg, thresholds_fx16_fx8_fx8_general, 
                                     test_3_chksum_fx16_fx8_fx8},
#elif COMPILE_FOR_SA8_SA8_SA32
    {"Test 3 SA8_SA32 BtB,ALL,Forw", mli_krn_lstm_cell_sa8_sa8_sa32,
                                     input_2_sa8, hidden_1_sa8, weights_2_in_sa8, weights_2_out_sa8, bias_1_i2_w2_sa32, 
                                     cell_1_sa8, test_3_out_sa8, test_3_cfg, thresholds_sa8_general, test_3_chksum_sa8},
#else
#error incorrect make configuration
#endif

    // // Batch-to-batch, RNN_OUT_ALL, Backward processing
#ifdef COMPILE_FOR_FX16
    {"Test 4 FX16 BtB,ALL,Back",     mli_krn_lstm_cell_fx16, 
                                     input_2_fx16, hidden_1_fx16, weights_2_in_fx16, weights_2_out_fx16, bias_1_fx16, 
                                     cell_1_fx16, test_4_out_fx16, test_4_cfg, thresholds_fx16_general, test_4_chksum_fx16},
#elif COMPILE_FOR_FX16_FX8_FX8
    {"Test 4 FX16_FX8 BtB,ALL,Back", mli_krn_lstm_cell_fx16_fx8_fx8, 
                                     input_2_fx16, hidden_1_fx16, weights_2_in_fx8, weights_2_out_fx8, bias_1_fx8, 
                                     cell_1_fx16, test_4_out_fx16, test_4_cfg, thresholds_fx16_fx8_fx8_general, 
                                     test_4_chksum_fx16_fx8_fx8},
#elif COMPILE_FOR_SA8_SA8_SA32
    {"Test 4 SA8_SA32 BtB,ALL,Back", mli_krn_lstm_cell_sa8_sa8_sa32,
                                     input_2_sa8, hidden_1_sa8, weights_2_in_sa8, weights_2_out_sa8, bias_1_i2_w2_sa32, 
                                     cell_1_sa8, test_4_out_sa8, test_4_cfg, thresholds_sa8_general, test_4_chksum_sa8},
#else
#error incorrect make configuration
#endif

    // // Batch-to-batch, RNN_OUT_LAST, Forward processing
#ifdef COMPILE_FOR_FX16
    {"Test 5 FX16 BtB,LAST,Forw",     mli_krn_lstm_cell_fx16, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_fx16, weights_2_out_fx16, bias_1_fx16, 
                                      cell_1_fx16, test_5_out_fx16, test_5_cfg, thresholds_fx16_general, test_5_chksum_fx16},
#elif COMPILE_FOR_FX16_FX8_FX8
    {"Test 5 FX16_FX8 BtB,LAST,Forw", mli_krn_lstm_cell_fx16_fx8_fx8, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_fx8, weights_2_out_fx8, bias_1_fx8, 
                                      cell_1_fx16, test_5_out_fx16, test_5_cfg, thresholds_fx16_fx8_fx8_general, 
                                      test_5_chksum_fx16_fx8_fx8},
#elif COMPILE_FOR_SA8_SA8_SA32
    {"Test 5 SA8_SA32 BtB,LAST,Forw", mli_krn_lstm_cell_sa8_sa8_sa32,
                                      input_2_sa8, hidden_1_sa8, weights_2_in_sa8, weights_2_out_sa8, bias_1_i2_w2_sa32, 
                                      cell_1_sa8, test_5_out_sa8, test_5_cfg, thresholds_sa8_general, test_5_chksum_sa8},
#else
#error incorrect make configuration
#endif

    // // Batch-to-batch, RNN_OUT_LAST, Backward processing
#ifdef COMPILE_FOR_FX16
    {"Test 6 FX16 BtB,LAST,Back",     mli_krn_lstm_cell_fx16, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_fx16, weights_2_out_fx16, bias_1_fx16, 
                                      cell_1_fx16, test_6_out_fx16, test_6_cfg, thresholds_fx16_general, test_6_chksum_fx16},
#elif COMPILE_FOR_FX16_FX8_FX8
    {"Test 6 FX16_FX8 BtB,LAST,Back", mli_krn_lstm_cell_fx16_fx8_fx8, 
                                      input_2_fx16, hidden_1_fx16, weights_2_in_fx8, weights_2_out_fx8, bias_1_fx8,
                                      cell_1_fx16, test_6_out_fx16, test_6_cfg, thresholds_fx16_fx8_fx8_general, 
                                      test_6_chksum_fx16_fx8_fx8},
#elif COMPILE_FOR_SA8_SA8_SA32
    {"Test 6 SA8_SA32 BtB,LAST,Back", mli_krn_lstm_cell_sa8_sa8_sa32,
                                      input_2_sa8, hidden_1_sa8, weights_2_in_sa8, weights_2_out_sa8, bias_1_i2_w2_sa32, 
                                      cell_1_sa8, test_6_out_sa8, test_6_cfg, thresholds_sa8_general, test_6_chksum_sa8},
#else
#error incorrect make configuration
#endif
};

constexpr int kMemInputSize = 90;
constexpr int kMemOutSize = 410;
constexpr int kMemWeightSize = 3210;
constexpr int kMemIrSize = 160;
static IO_DATA_ATTR int8_t scratch_mem_in[kMemInputSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_prev_out[kMemInputSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_cell[kMemInputSize] = { 0 };
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

    reporter.report_header("MLI|Kernels|LSTM Cell Tests");
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
        memory_manager mem_cell_keeper((int8_t*)(scratch_mem_cell), sizeof(scratch_mem_cell));
        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        memory_manager mem_w_in_keeper((int8_t*)(scratch_mem_w_in), sizeof(scratch_mem_w_in));
        memory_manager mem_w_out_keeper((int8_t*)(scratch_mem_w_out), sizeof(scratch_mem_w_out));
        memory_manager mem_b_keeper((int8_t*)(scratch_mem_b), sizeof(scratch_mem_b));
        memory_manager mem_ir_keeper((int8_t*)(scratch_mem_ir_tensor), sizeof(scratch_mem_ir_tensor));
        bool is_test_passed = true;
        lstm_cell_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metrics;
        if (!(lut_status)) {
            reporter.report_message(cur_test->descr, "FAILED at init: LUT error");
            is_test_passed = false;
        }

#if defined(__Xvec_guard_bit_option) && (__Xvec_guard_bit_option == 0)
        if (strstr(cur_test->descr, "Test 1 FX16 OtO,Forw") != nullptr ||
                strstr(cur_test->descr, "Test 2 FX16 OtO,Back") != nullptr ||
                strstr(cur_test->descr, "Test 3 FX16 BtB,ALL,Forw") != nullptr ||
                strstr(cur_test->descr, "Test 4 FX16 BtB,ALL,Back") != nullptr ||
                strstr(cur_test->descr, "Test 5 FX16 BtB,LAST,Forw") != nullptr ||
                strstr(cur_test->descr, "Test 5 SA8_SA32 BtB,LAST,Forw") != nullptr ||
                strstr(cur_test->descr, "Test 6 FX16 BtB,LAST,Back") != nullptr) {
            // VPX fails bitwise comparison with reference .
            reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
            continue;
        }
#endif

        if (!(cur_test->in.is_valid() && cur_test->prev_out.is_valid() && cur_test->weights_in.is_valid() && 
            cur_test->weights_out.is_valid() && cur_test->bias.is_valid() && cur_test->cell.is_valid() && 
            cur_test->out.is_valid())) {
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
        mli_tensor cell = cur_test->cell.get_quantized_tensor(mem_cell_keeper.allocate_memory(cur_test->cell));
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
                 tensor_quantizer::validate_tensor(cell) != tensor_quantizer::kOk||
                 tensor_quantizer::validate_tensor(bias) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(out) != tensor_quantizer::kOk)) {
            reporter.report_message(cur_test->descr, 
                                    "FAILED at quantization step: more memory for one of tensors might be required");
            is_test_passed = false;
        }

        if (is_test_passed &&
                (mem_in_keeper.is_memory_corrupted() || mem_prev_out_keeper.is_memory_corrupted() || 
                mem_cell_keeper.is_memory_corrupted() ||  mem_out_keeper.is_memory_corrupted() ||
                mem_w_in_keeper.is_memory_corrupted() || mem_w_out_keeper.is_memory_corrupted() || 
                mem_b_keeper.is_memory_corrupted() || mem_ir_keeper.is_memory_corrupted())) {
            reporter.report_message(cur_test->descr,
                "FAILED at quantization step: memory beside one of operands is corrupted");
            is_test_passed = false;
        }

        // Run specific kernel for test
        if (is_test_passed &&
                cur_test->mli_krn_lstm_cell
                (&input, &prev_out, &weights_in, &weights_out, &bias, &tanh_lut, &sigm_lut, &cur_test_cfg, &cell, &out) 
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
            data_crc(cell);
            data_crc(out);

            is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr, test_metrics, cur_test->threshold, 
                                                                data_crc, cur_test->check_sum);
        }
        final_status &= is_test_passed;
    }

    reporter.report_outline("[AUTO] Group: mli_krn_lstm_cell", final_status);

    return (final_status) ? 0 : 1;
}
