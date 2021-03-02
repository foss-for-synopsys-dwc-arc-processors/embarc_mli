/*
* Copyright 2020-2021, Synopsys, Inc.
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

#include "vectors_mli_krn_rnn_dense.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

typedef mli_status(*rnn_dense_func_ptr)(
    const mli_tensor** /*inputs*/,
    const mli_tensor** /*weights*/,
    const mli_tensor* /*bias*/,
    const mli_rnn_dense_cfg* /*cfg*/,
    mli_tensor* /*output*/);

struct rnn_dense_test_operands {
    const char* descr;
    const rnn_dense_func_ptr mli_krn_rnn_dense;
    tensor_quantizer in[MLI_RNN_MAX_INPUT];
    tensor_quantizer weights[MLI_RNN_MAX_INPUT];
    tensor_quantizer bias;
    tensor_quantizer out;
    const mli_rnn_dense_cfg cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

// Checksums of test tensors for various mli calculations mode. 
// When developer finished implementation of kernel and consider it as ok, one needs to populate
// proper checksums for tests in order to highlight any change which affects results.
#if defined(CRC_RM_CONVERGENT) || defined(CRC_RM_UP)
// Shared CRC Results

const crc32_calc test_1_chksum_fx16{ 0x67A28725 }, test_1_chksum_fx16_fx8_fx8{ 0x39090878 }, test_1_chksum_sa8{ 0x622840E9 },
                 test_2_chksum_fx16{ 0x67A28725 }, test_2_chksum_fx16_fx8_fx8{ 0x39090878 }, test_2_chksum_sa8{ 0x622840E9 },
                 test_3_chksum_fx16{ 0x159AD436 }, test_3_chksum_fx16_fx8_fx8{ 0x5918457A }, test_3_chksum_sa8{ 0xC18BB563 },
                 test_4_chksum_fx16{ 0x159AD436 }, test_4_chksum_fx16_fx8_fx8{ 0x5918457A }, test_4_chksum_sa8{ 0xC18BB563 },
                 test_5_chksum_fx16{ 0x484FA14F }, test_5_chksum_fx16_fx8_fx8{ 0x9A6277F1 }, test_5_chksum_sa8{ 0x12A21BAD },
                 test_6_chksum_fx16{ 0x484FA14F }, test_6_chksum_fx16_fx8_fx8{ 0x9A6277F1 }, test_6_chksum_sa8{ 0x12A21BAD };
#else // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16, test_1_chksum_fx16_fx8_fx8, test_1_chksum_sa8,
                  test_2_chksum_fx16, test_2_chksum_fx16_fx8_fx8, test_2_chksum_sa8,
                  test_3_chksum_fx16, test_3_chksum_fx16_fx8_fx8, test_3_chksum_sa8,
                  test_4_chksum_fx16, test_4_chksum_fx16_fx8_fx8, test_4_chksum_sa8,
                  test_5_chksum_fx16, test_5_chksum_fx16_fx8_fx8, test_5_chksum_sa8,
                  test_6_chksum_fx16, test_6_chksum_fx16_fx8_fx8, test_6_chksum_sa8;
#endif


const quality_metrics thresholds_fx16_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR_DB = */70.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_fx16_fx8_fx8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                  /* SNR_DB = */30.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                             /* SNR_DB = */35.f, quality_metrics::kPassValueQuantErrPerc };


static const rnn_dense_test_operands tests_list[] = {
    // 2 inputs
    {"Test 1 FX16 2 inputs",         mli_krn_rnn_dense_fx16,
                                     {input_1_fx16, input_2_fx16}, {weights_1_fx16, weights_2_fx16}, bias_1_fx16, 
                                     test_1_out_fx16, test_1_cfg, thresholds_fx16_general, test_1_chksum_fx16},
    {"Test 1 FX16_FX8_FX8 2 inputs", mli_krn_rnn_dense_fx16_fx8_fx8,
                                     {input_1_fx16, input_2_fx16}, {weights_1_fx8, weights_2_fx8}, bias_1_fx8, 
                                     test_1_out_fx16, test_1_cfg, thresholds_fx16_fx8_fx8_general, test_1_chksum_fx16_fx8_fx8},
    {"Test 1 SA8_SA8_SA32 2 inputs", mli_krn_rnn_dense_sa8_sa8_sa32,
                                     {input_1_sa8, input_2_sa8}, {weights_1_sa8, weights_2_sa8}, bias_1_i1_w1_sa32, 
                                     test_1_out_sa8, test_1_cfg, thresholds_sa8_general, test_1_chksum_sa8},
    
    // 2 inputs, weight memstride
    {"Test 2 FX16 memstr 2in W_mstr",  mli_krn_rnn_dense_fx16,
                                       {input_1_fx16, input_2_fx16}, {weights_1_memstr_fx16, weights_2_memstr_fx16}, bias_1_fx16, 
                                       test_1_out_fx16, test_1_cfg, thresholds_fx16_general, test_1_chksum_fx16},
    {"Test 2 FX16_FX8_FX8 2in W_mstr", mli_krn_rnn_dense_fx16_fx8_fx8,
                                       {input_1_fx16, input_2_fx16}, {weights_1_memstr_fx8, weights_2_memstr_fx8}, bias_1_fx8, 
                                       test_1_out_fx16, test_1_cfg, thresholds_fx16_fx8_fx8_general, test_1_chksum_fx16_fx8_fx8},
    {"Test 2 SA8_SA8_SA32 2in W_mstr", mli_krn_rnn_dense_sa8_sa8_sa32,
                                       {input_1_sa8, input_2_sa8}, {weights_1_memstr_sa8, weights_2_memstr_sa8}, bias_1_i1_w1_sa32, 
                                       test_1_out_sa8, test_1_cfg, thresholds_sa8_general, test_1_chksum_sa8},

    // 3 inputs
    {"Test 3 FX16 3 inputs",         mli_krn_rnn_dense_fx16,
                                     {input_3_fx16, input_4_fx16, input_5_fx16}, {weights_3_fx16, weights_4_fx16, weights_5_fx16}, 
                                     bias_2_fx16, test_2_out_fx16, test_2_cfg, thresholds_fx16_general, test_3_chksum_fx16},
    {"Test 3 FX16_FX8_FX8 3 inputs", mli_krn_rnn_dense_fx16_fx8_fx8,
                                     {input_3_fx16, input_4_fx16, input_5_fx16}, {weights_3_fx8, weights_4_fx8, weights_5_fx8}, 
                                     bias_2_fx8, test_2_out_fx16, test_2_cfg, thresholds_fx16_fx8_fx8_general, 
                                     test_3_chksum_fx16_fx8_fx8},
    {"Test 3 SA8_SA8_SA32 3 inputs", mli_krn_rnn_dense_sa8_sa8_sa32,
                                     {input_3_sa8, input_4_sa8, input_5_sa8}, {weights_3_sa8, weights_4_sa8, weights_5_sa8}, 
                                     bias_2_i3_w3_sa32, test_2_out_sa8, test_2_cfg, thresholds_sa8_general, test_3_chksum_sa8},

    // 3 inputs, weights memstride
    {"Test 4 FX16 3in W_mstr",         mli_krn_rnn_dense_fx16,
                                       {input_3_fx16, input_4_fx16, input_5_fx16}, {weights_3_memstr_fx16, weights_4_memstr_fx16, 
                                       weights_5_memstr_fx16}, bias_2_fx16, test_2_out_fx16, test_2_cfg, thresholds_fx16_general, 
                                       test_3_chksum_fx16},
    {"Test 4 FX16_FX8_FX8 3in W_mstr", mli_krn_rnn_dense_fx16_fx8_fx8,
                                       {input_3_fx16, input_4_fx16, input_5_fx16}, {weights_3_memstr_fx8, weights_4_memstr_fx8, 
                                       weights_5_memstr_fx8}, bias_2_fx8, test_2_out_fx16, test_2_cfg, thresholds_fx16_fx8_fx8_general, 
                                       test_3_chksum_fx16_fx8_fx8},
    {"Test 4 SA8_SA8_SA32 3in W_mstr", mli_krn_rnn_dense_sa8_sa8_sa32,
                                       {input_3_sa8, input_4_sa8, input_5_sa8}, {weights_3_memstr_sa8, weights_4_memstr_sa8, 
                                       weights_5_memstr_sa8}, bias_2_i3_w3_sa32, test_2_out_sa8, test_2_cfg, thresholds_sa8_general, 
                                       test_3_chksum_sa8},

    // 4 inputs
    {"Test 5 FX16 4 inputs",         mli_krn_rnn_dense_fx16,
                                     {input_3_fx16, input_4_fx16, input_5_fx16, input_2_fx16}, {weights_6_fx16, weights_7_fx16, 
                                     weights_8_fx16, weights_9_fx16}, bias_3_fx16, test_3_out_fx16, test_3_cfg, thresholds_fx16_general, 
                                     test_5_chksum_fx16},
    {"Test 5 FX16_FX8_FX8 4 inputs", mli_krn_rnn_dense_fx16_fx8_fx8,
                                     {input_3_fx16, input_4_fx16, input_5_fx16, input_2_fx16}, {weights_6_fx8, weights_7_fx8, 
                                     weights_8_fx8, weights_9_fx8}, bias_3_fx8, test_3_out_fx16, test_3_cfg, thresholds_fx16_fx8_fx8_general, 
                                     test_5_chksum_fx16_fx8_fx8},
    {"Test 5 SA8_SA8_SA32 4 inputs", mli_krn_rnn_dense_sa8_sa8_sa32,
                                     {input_3_sa8, input_4_sa8, input_5_sa8, input_2_sa8}, {weights_6_sa8, weights_7_sa8, 
                                     weights_8_sa8, weights_9_sa8}, bias_3_i3_w6_sa32, test_3_out_sa8, test_3_cfg, thresholds_sa8_general, 
                                     test_5_chksum_sa8},

    // 4 inputs, weights memstride
    {"Test 6 FX16 4in W_mstr",         mli_krn_rnn_dense_fx16,
                                       {input_3_fx16, input_4_fx16, input_5_fx16, input_2_fx16}, {weights_6_memstr_fx16, 
                                       weights_7_memstr_fx16, weights_8_memstr_fx16, weights_9_memstr_fx16}, bias_3_fx16, 
                                       test_3_out_fx16, test_3_cfg, thresholds_fx16_general, test_6_chksum_fx16},
    {"Test 6 FX16_FX8_FX8 4in W_mstr", mli_krn_rnn_dense_fx16_fx8_fx8,
                                       {input_3_fx16, input_4_fx16, input_5_fx16, input_2_fx16}, {weights_6_memstr_fx8, 
                                       weights_7_memstr_fx8, weights_8_memstr_fx8, weights_9_memstr_fx8}, bias_3_fx8, 
                                       test_3_out_fx16, test_3_cfg, thresholds_fx16_fx8_fx8_general, test_6_chksum_fx16_fx8_fx8},
    {"Test 6 SA8_SA8_SA32 4in W_mstr", mli_krn_rnn_dense_sa8_sa8_sa32,
                                       {input_3_sa8, input_4_sa8, input_5_sa8, input_2_sa8}, {weights_6_memstr_sa8, 
                                       weights_7_memstr_sa8, weights_8_memstr_sa8, weights_9_memstr_sa8}, bias_3_i3_w6_sa32, 
                                       test_3_out_sa8, test_3_cfg, thresholds_sa8_general, test_6_chksum_sa8},
};

constexpr int kMemSize = 300;
static IO_DATA_ATTR int8_t scratch_mem_in[MLI_RNN_MAX_INPUT][kMemSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemSize] = { 0 };
static W_DATA_ATTR int8_t scratch_mem_w[MLI_RNN_MAX_INPUT][kMemSize] = { 0 };
static W_DATA_ATTR int8_t scratch_mem_b[kMemSize] = { 0 };

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Kernels|RNN Dense Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        const rnn_dense_test_operands* cur_test = &tests_list[i];
        const int inputs_num = cur_test->cfg.inputs_num;

        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        memory_manager mem_b_keeper((int8_t*)(scratch_mem_b), sizeof(scratch_mem_b));

        memory_manager mem_in_keeper[MLI_RNN_MAX_INPUT];
        memory_manager mem_w_keeper[MLI_RNN_MAX_INPUT];

        for(int input_idx = 0; input_idx < inputs_num; ++input_idx) {
            mem_in_keeper[input_idx] = memory_manager((int8_t*)(scratch_mem_in[input_idx]), sizeof(scratch_mem_in[input_idx]));
            mem_w_keeper[input_idx] = memory_manager((int8_t*)(scratch_mem_w[input_idx]), sizeof(scratch_mem_w[input_idx]));
        }

        bool is_test_passed = true;
        quality_metrics test_metrics;

#if defined(__Xvec_guard_bit_option) && (__Xvec_guard_bit_option == 0)
        if (strstr(cur_test->descr, "Test 1 FX16 2 inputs") != nullptr ||
                strstr(cur_test->descr, "Test 2 FX16 memstr 2in W_mstr") != nullptr ||
                strstr(cur_test->descr, "Test 3 FX16 3 inputs") != nullptr ||
                strstr(cur_test->descr, "Test 3 SA8_SA8_SA32 3 inputs") != nullptr ||
                strstr(cur_test->descr, "Test 4 FX16 3in W_mstr") != nullptr ||
                strstr(cur_test->descr, "Test 4 SA8_SA8_SA32 3in W_mstr") != nullptr ||
                strstr(cur_test->descr, "Test 5 FX16 4 inputs") != nullptr ||
                strstr(cur_test->descr, "Test 5 SA8_SA8_SA32 4 inputs") != nullptr ||
                strstr(cur_test->descr, "Test 6 FX16 4in W_mstr") != nullptr ||
                strstr(cur_test->descr, "Test 6 SA8_SA8_SA32 4in W_mstr") != nullptr) {
            // VPX fails bitwise comparison with reference .
            reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
            continue;
        }
#endif

        bool is_valid = true;
        for(int input_idx = 0; input_idx < inputs_num; ++input_idx) {
            is_valid |= cur_test->in[input_idx].is_valid();
            is_valid |= cur_test->weights[input_idx].is_valid();
        }

        is_valid |= cur_test->bias.is_valid();
        is_valid |= cur_test->out.is_valid();

        if (!is_valid) {
            reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
            is_test_passed = false;
        }

        mli_tensor input1 = cur_test->in[0].get_quantized_tensor(mem_in_keeper[0].allocate_memory(cur_test->in[0]));
        mli_tensor input2 = cur_test->in[1].get_quantized_tensor(mem_in_keeper[1].allocate_memory(cur_test->in[1]));
        mli_tensor input3 = cur_test->in[2].get_quantized_tensor(mem_in_keeper[2].allocate_memory(cur_test->in[2]));
        mli_tensor input4 = cur_test->in[3].get_quantized_tensor(mem_in_keeper[3].allocate_memory(cur_test->in[3]));

        const mli_tensor * inputs[] = {&input1, &input2, &input3, &input4};

        mli_tensor weights1 = cur_test->weights[0].get_quantized_tensor(mem_w_keeper[0].allocate_memory(cur_test->weights[0]));
        mli_tensor weights2 = cur_test->weights[1].get_quantized_tensor(mem_w_keeper[1].allocate_memory(cur_test->weights[1]));
        mli_tensor weights3 = cur_test->weights[2].get_quantized_tensor(mem_w_keeper[2].allocate_memory(cur_test->weights[2]));
        mli_tensor weights4 = cur_test->weights[3].get_quantized_tensor(mem_w_keeper[3].allocate_memory(cur_test->weights[3]));

        const mli_tensor * weights[] = {&weights1, &weights2, &weights3, &weights4};

        mli_tensor bias = cur_test->bias.get_quantized_tensor(mem_b_keeper.allocate_memory(cur_test->bias));
        mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
        mli_tensor source_out_tensor = out;

        tensor_quantizer::tensor_state total_state = tensor_quantizer::kOk;
        bool is_memory_corrupted = false;

        for(int input_idx = 0; input_idx < inputs_num; ++input_idx) {
            total_state = tensor_quantizer::validate_tensor(*inputs[input_idx]);
            total_state = tensor_quantizer::validate_tensor(*weights[input_idx]);

            is_memory_corrupted = mem_in_keeper[input_idx].is_memory_corrupted();
            is_memory_corrupted = mem_w_keeper[input_idx].is_memory_corrupted();            
        }

        total_state = tensor_quantizer::validate_tensor(bias);
        total_state = tensor_quantizer::validate_tensor(out);
        is_memory_corrupted = mem_out_keeper.is_memory_corrupted();
        is_memory_corrupted = mem_b_keeper.is_memory_corrupted();

        if (is_test_passed && total_state != tensor_quantizer::kOk) {
            reporter.report_message(cur_test->descr, 
                                    "FAILED at quantization step: more memory for one of tensors might be required");
            is_test_passed = false;
        }

        if (is_test_passed && is_memory_corrupted == true) {
            reporter.report_message(cur_test->descr,
                "FAILED at quantization step: memory beside one of operands is corrupted");
            is_test_passed = false;
        }

        // Run specific kernel for test
        if (is_test_passed &&
                cur_test->mli_krn_rnn_dense(inputs, weights, &bias, &cur_test->cfg, &out) != MLI_STATUS_OK) {
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
            for(int input_idx = 0; input_idx < inputs_num; ++input_idx) {
                data_crc(*inputs[input_idx]);
                data_crc(*weights[input_idx]);
            }
            data_crc(bias);
            data_crc(out);

            is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr, test_metrics, cur_test->threshold, 
                                                                data_crc, cur_test->check_sum);
        }
        final_status &= is_test_passed;
    }

    reporter.report_outline("[AUTO] Group: mli_krn_rnn_dense", final_status);

    return (final_status) ? 0 : 1;
}
