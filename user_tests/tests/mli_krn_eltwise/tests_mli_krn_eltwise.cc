/*
* Copyright 2020, Synopsys, Inc.
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

#include "vectors_mli_krn_eltwise.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

typedef mli_status(*eltwise_func_ptr)(
    const mli_tensor* /*in1*/,
    const mli_tensor* /*in2*/,
    mli_tensor* /*out*/);

struct eltwise_test_operands {
    const char* descr;
    const eltwise_func_ptr mli_krn_eltwise;
    tensor_quantizer in1;
    tensor_quantizer in2;
    tensor_quantizer out;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

// TODO Checksums of test tensors for various mli calculations mode.
// When developer finished implementation of kernel and consider it as ok, He need to populate
// proper checksums for tests in order to highlight any change which affects results.
#if defined(CRC_RM_CONVERGENT) || defined(CRC_RM_UP)

// Shared CRC Results
const crc32_calc                                    test_1_chksum_sa8{ 0xd48163e7 },
                                                    test_2_chksum_sa8{ 0x6CE064A1 },
                                                    test_3_chksum_sa8{ 0x3b9100e1 },
                                                    test_4_chksum_sa8{ 0xF22D7321 },
                                                    test_5_chksum_sa8{ 0x8ECBC7B8 },
                  test_6_chksum_fx16{ 0xfc026def }, test_6_chksum_sa8{ 0x3a54561 },
                  test_7_chksum_fx16{ 0x488ed527 }, test_7_chksum_sa8{ 0xDA50B98A },
                  test_8_chksum_fx16{ 0x68889D84 }, test_8_chksum_sa8{ 0x168B3B32 },
                  test_9_chksum_fx16{ 0x9417F3D7 }, test_9_chksum_sa8{ 0x3382BC48 },
                  test_10_chksum_fx16{ 0xD728E430 }, test_10_chksum_sa8{ 0xE34DA6B0 },
                  test_11_chksum_fx16{ 0xBF03F2E0 }, test_11_chksum_sa8{ 0xD36B7E94 };

// Platform Specific CRC Results
#if defined(CRC_RM_UP)
const crc32_calc test_1_chksum_fx16{ 0xC5BD8154 }, test_2_chksum_fx16{ 0x170065BD },
                 test_3_chksum_fx16{ 0x34f32ee0 }, test_4_chksum_fx16{ 0x0DECE100 },
                 test_5_chksum_fx16{ 0x1a678d57 };
#else
const crc32_calc test_1_chksum_fx16{ 0x80C6E2B7 }, test_2_chksum_fx16{ 0x10D03580 },
                 test_3_chksum_fx16{ 0xD6C9167D }, test_4_chksum_fx16{ 0x5B406931 },
                 test_5_chksum_fx16{ 0x1DB7DD6A };
#endif

#else  // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16, test_1_chksum_sa8,
                  test_2_chksum_fx16, test_2_chksum_sa8,
                  test_3_chksum_fx16, test_3_chksum_sa8,
                  test_4_chksum_fx16, test_4_chksum_sa8,
                  test_5_chksum_fx16, test_5_chksum_sa8,
                  test_6_chksum_fx16, test_6_chksum_sa8,
                  test_7_chksum_fx16, test_7_chksum_sa8,
                  test_8_chksum_fx16, test_8_chksum_sa8,
                  test_9_chksum_fx16, test_9_chksum_sa8,
                  test_10_chksum_fx16, test_10_chksum_sa8,
                  test_11_chksum_fx16, test_11_chksum_sa8;

#endif

const quality_metrics thresholds_fx16_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR DB = */ 60.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                               /* SNR DB = */ 30.f, quality_metrics::kPassValueQuantErrPerc };

static const eltwise_test_operands tests_list[] = {
    // Eltwise add of two vectors
    {"Test 1 FX16 Add two vectors",  mli_krn_eltwise_add_fx16,
                                     input_1_fx16, input_2_fx16, test_1_out_fx16,
                                     thresholds_fx16_general, test_1_chksum_fx16},
    {"Test 1 SA8 Add two vectors",  mli_krn_eltwise_add_sa8,
                                    input_1_sa8, input_2_sa8, test_1_out_sa8,
                                    thresholds_sa8_general, test_1_chksum_sa8},

    // Eltwise add of vector and scalar
    {"Test 2 FX16 Add vec & scalar",  mli_krn_eltwise_add_fx16,
                                      input_2_fx16, input_3_fx16, test_2_out_fx16,
                                      thresholds_fx16_general, test_2_chksum_fx16},
    {"Test 2 SA8 Add vec & scalar",  mli_krn_eltwise_add_sa8,
                                     input_2_sa8, input_3_sa8, test_2_out_sa8,
                                     thresholds_sa8_general, test_2_chksum_sa8},

    // Eltwise sub of two vectors
    {"Test 3 FX16 Sub two vectors",  mli_krn_eltwise_sub_fx16,
                                     input_1_fx16, input_2_fx16, test_3_out_fx16,
                                     thresholds_fx16_general, test_3_chksum_fx16},
    {"Test 3 SA8 Sub two vectors",  mli_krn_eltwise_sub_sa8,
                                    input_1_sa8, input_2_sa8, test_3_out_sa8,
                                    thresholds_sa8_general, test_3_chksum_sa8},

    // Eltwise sub scalar from vec
    {"Test 4 FX16 Sub scalar - vec",  mli_krn_eltwise_sub_fx16,
                                      input_1_fx16, input_3_fx16, test_4_out_fx16,
                                      thresholds_fx16_general, test_4_chksum_fx16},
    {"Test 4 SA8 Sub scalar - vec",  mli_krn_eltwise_sub_sa8,
                                     input_1_sa8, input_3_sa8, test_4_out_sa8,
                                     thresholds_sa8_general, test_4_chksum_sa8},

    // Eltwise sub vec from scalar
    {"Test 5 FX16 Sub vec - scalar",  mli_krn_eltwise_sub_fx16,
                                      input_3_fx16, input_2_fx16, test_5_out_fx16,
                                      thresholds_fx16_general, test_5_chksum_fx16},
    {"Test 5 SA8 Sub vec - scalar",  mli_krn_eltwise_sub_sa8,
                                     input_3_sa8, input_2_sa8, test_5_out_sa8,
                                     thresholds_sa8_general, test_5_chksum_sa8},

    // Eltwise Mul of two vectors
    {"Test 6 FX16 Mul two vectors",  mli_krn_eltwise_mul_fx16,
                                     input_1_fx16, input_2_fx16, test_6_out_fx16,
                                     thresholds_fx16_general, test_6_chksum_fx16},
    {"Test 6 SA8 Mul two vectors",  mli_krn_eltwise_mul_sa8,
                                    input_1_sa8, input_2_sa8, test_6_out_sa8,
                                    thresholds_sa8_general, test_6_chksum_sa8},

    // Eltwise Mul vector & scalar
    {"Test 7 FX16 Mul vec & scalar",  mli_krn_eltwise_mul_fx16,
                                      input_1_fx16, input_3_fx16, test_7_out_fx16,
                                      thresholds_fx16_general, test_7_chksum_fx16},
    {"Test 7 SA8 Mul vec & scalar",  mli_krn_eltwise_mul_sa8,
                                     input_1_sa8, input_3_sa8, test_7_out_sa8,
                                     thresholds_sa8_general, test_7_chksum_sa8},

    // Eltwise Max two vectors
    {"Test 8 FX16 Max two vectors",  mli_krn_eltwise_max_fx16,
                                   input_1_fx16_12, input_2_fx16_12, test_8_out_fx16,
                                   thresholds_fx16_general, test_8_chksum_fx16},
    {"Test 8 SA8 Max two vectors",  mli_krn_eltwise_max_sa8,
                                  input_1_sa8_12, input_2_sa8_12, test_8_out_sa8,
                                  thresholds_sa8_general, test_8_chksum_sa8},

    // Eltwise Max vector & scalar
    {"Test 9 FX16 Max vec & scalar",  mli_krn_eltwise_max_fx16,
                                    input_1_fx16_13, input_3_fx16_13, test_9_out_fx16,
                                    thresholds_fx16_general, test_9_chksum_fx16},
    {"Test 9 SA8 Max vec & scalar",  mli_krn_eltwise_max_sa8,
                                   input_1_sa8_13, input_3_sa8_13, test_9_out_sa8,
                                   thresholds_sa8_general, test_9_chksum_sa8},

    // Eltwise Min two vectors
    {"Test 10 FX16 Min two vectors",  mli_krn_eltwise_min_fx16,
                                    input_1_fx16_12, input_2_fx16_12, test_10_out_fx16,
                                    thresholds_fx16_general, test_10_chksum_fx16},
    {"Test 10 SA8 Min two vectors",  mli_krn_eltwise_min_sa8,
                                   input_1_sa8_12, input_2_sa8_12, test_10_out_sa8,
                                   thresholds_sa8_general, test_10_chksum_sa8},

    // Eltwise Min vector & scalar
    {"Test 11 FX16 Min vec & scalar",  mli_krn_eltwise_min_fx16,
                                     input_1_fx16_13, input_3_fx16_13, test_11_out_fx16,
                                     thresholds_fx16_general, test_11_chksum_fx16},
    {"Test 11 SA8 Min vec & scalar",  mli_krn_eltwise_min_sa8,
                                    input_1_sa8_13, input_3_sa8_13, test_11_out_sa8,
                                    thresholds_sa8_general, test_11_chksum_sa8},
};

constexpr int kMemSize = 2048;
static IO_DATA_ATTR int8_t scratch_mem_in1[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_in2[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemSize] = { 0 };

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Kernels|Basic Eltwise Functions Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in1_keeper((int8_t*)(scratch_mem_in1), sizeof(scratch_mem_in1));
        memory_manager mem_in2_keeper((int8_t*)(scratch_mem_in2), sizeof(scratch_mem_in2));
        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        bool is_test_passed = true;
        const eltwise_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metics;
        if (!(cur_test->in1.is_valid() && cur_test->in2.is_valid() && cur_test->out.is_valid())) {
            reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
            is_test_passed = false;
        }

        mli_tensor input1 = cur_test->in1.get_quantized_tensor(mem_in1_keeper.allocate_memory(cur_test->in1));
        mli_tensor input2 = cur_test->in2.get_quantized_tensor(mem_in2_keeper.allocate_memory(cur_test->in2));
        mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
        mli_tensor source_out_tensor = out;
        if (is_test_passed &&
                (tensor_quantizer::validate_tensor(input1) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(input2) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(out) != tensor_quantizer::kOk)) {
            reporter.report_message(cur_test->descr, 
                                    "FAILED at quantization step: more memory for one of tensors might be required");
            is_test_passed = false;
        }

        if (is_test_passed &&
                (mem_in1_keeper.is_memory_corrupted() || mem_in2_keeper.is_memory_corrupted() ||
                        mem_out_keeper.is_memory_corrupted())) {
            reporter.report_message(cur_test->descr,
                "FAILED at quantization step: memory beside one of operands is corrupted");
            is_test_passed = false;
        }

        // Run specific kernel for test 
        if (is_test_passed &&
                cur_test->mli_krn_eltwise(&input1, &input2, &out) != MLI_STATUS_OK) {
            reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
            is_test_passed = false;
        }

        if (is_test_passed &&
                (mem_in1_keeper.is_memory_corrupted() || mem_in2_keeper.is_memory_corrupted() ||
                        mem_out_keeper.is_memory_corrupted())) {
            reporter.report_message(cur_test->descr,
                "FAILED after kernel run: memory beside one of operands is corrupted");
            is_test_passed = false;
        }

        if (is_test_passed &&
                test_metics.calculate_metrics(out, cur_test->out) == false) {
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
            data_crc(input1);
            data_crc(input2);
            data_crc(out);
            is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr, test_metics, cur_test->threshold, 
                                                                data_crc, cur_test->check_sum);
        }

        final_status &= is_test_passed;
    }

    reporter.report_outline("[AUTO] Group: mli_krn_eltwise", final_status);

    return (final_status) ? 0 : 1;
}
