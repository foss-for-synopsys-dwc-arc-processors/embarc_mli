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

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "mli_types.h"
#include "test_tensor_quantizer.h"
#include "test_report.h"

#include "vectors_mli_krn_permute.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

typedef mli_status(*permute_func_ptr)(
    const mli_tensor* /*in*/,
    const mli_permute_cfg* /*cfg*/,
    mli_tensor* /*out*/);

struct permute_test_operands {
    const char* descr;
    const permute_func_ptr mli_krn_permute;
    tensor_quantizer in;
    tensor_quantizer out;
    const mli_permute_cfg cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

// Checksums of test tensors for various mli calculations mode. 
// When developer finished implementation of kernel and consider it as ok, He need to populate
// proper checksums for tests in order to highlight any change which affects results.
// TODO Make right checksums
#if defined(CRC_RM_CONVERGENT) || defined(CRC_RM_UP)

// Shared CRC Results
//TODO Fill correct checksum for sa8
const crc32_calc  test_1_chksum_fx16{ 0x7DDA59C8 }, test_1_chksum_sa8{ 0xD7062F9F },
                  test_2_chksum_fx16{ 0xDA5A558D }, test_2_chksum_sa8{ 0x157694C9 },
                  test_3_chksum_fx16{ 0xA75B8B96 }, test_3_chksum_sa8{ 0xA831F59E },
                  test_4_chksum_fx16{ 0x9C1AA0B9 }, test_4_chksum_sa8{ 0x043C46FB },
                  test_5_chksum_fx16{ 0x0E3D57F3 }, test_5_chksum_sa8{ 0xA88A700C };

// Platform Specific CRC Results
// #if defined(CRC_RM_UP)
// const crc32_calc test_3_chksum_fx16{ 0x875BA219 }, test_4_chksum_fx16{ 0xCD5A958F };
// #else 
// const crc32_calc test_3_chksum_fx16{ 0xBF9EAF0C }, test_4_chksum_fx16{ 0xC98520CF };
// #endif

#else  // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16, test_1_chksum_sa8,
                  test_2_chksum_fx16, test_2_chksum_sa8,
                  test_3_chksum_fx16, test_3_chksum_sa8,
                  test_4_chksum_fx16, test_4_chksum_sa8,
                  test_5_chksum_fx16, test_5_chksum_sa8;
#endif

const quality_metrics thresholds_fx16_general { /* MaxAbsErr = */0.0f, quality_metrics::kPassValueSnr,
                                                quality_metrics::kPassValueSnrDb, quality_metrics::kPassValueQuantErrPerc};

const quality_metrics thresholds_sa8_general{ /* MaxAbsErr = */0.0f, quality_metrics::kPassValueSnr,
                                                quality_metrics::kPassValueSnrDb, quality_metrics::kPassValueQuantErrPerc};
//TODO make normaly test description
static const permute_test_operands tests_list[] = {
    {"Test 1 FX16 matrix transpose",  mli_krn_permute_fx16,
                                   input_1_memstr_fx16, test_1_out_fx16, test_1_cfg,
                                   thresholds_fx16_general, test_1_chksum_fx16},
    {"Test 1 SA8 matrix transpose",   mli_krn_permute_sa8,
                                   input_1_memstr_sa8, test_1_out_sa8, test_1_cfg,
                                   thresholds_sa8_general, test_1_chksum_sa8},

    {"Test 2 FX16 ",  mli_krn_permute_fx16,
                            input_2_fx16, test_2_out_memstr_fx16, test_2_cfg,
                            thresholds_fx16_general, test_2_chksum_fx16},
    {"Test 2 SA8 ",   mli_krn_permute_sa8,
                            input_2_memstr_sa8, test_2_out_memstr_sa8, test_2_cfg,
                            thresholds_sa8_general, test_2_chksum_sa8},

    {"Test 3 FX16 ",  mli_krn_permute_fx16,
                            input_2_memstr_fx16, test_3_out_memstr_fx16, test_3_cfg,
                            thresholds_fx16_general, test_3_chksum_fx16},
    {"Test 3 SA8 ",   mli_krn_permute_sa8,
                            input_2_memstr_sa8, test_3_out_memstr_sa8, test_3_cfg, 
                            thresholds_sa8_general, test_3_chksum_sa8},

    {"Test 4 FX16 ",  mli_krn_permute_fx16,
                            input_3_fx16, test_4_out_fx16, test_4_cfg,
                            thresholds_fx16_general, test_4_chksum_fx16},
    {"Test 4 SA8 ",   mli_krn_permute_sa8,
                            input_3_sa8, test_4_out_sa8, test_4_cfg,
                            thresholds_sa8_general, test_4_chksum_sa8},

    {"Test 5 FX16 ",  mli_krn_permute_fx16,
                                      input_4_fx16, test_5_out_fx16, test_5_cfg,
                                      thresholds_fx16_general, test_5_chksum_fx16},
    {"Test 5 SA8 ",   mli_krn_permute_sa8,
                                      input_4_sa8, test_5_out_sa8, test_5_cfg,
                                      thresholds_sa8_general, test_5_chksum_sa8}
};

constexpr int kMemSize = 2047;
static IO_DATA_ATTR int8_t scratch_mem_in[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemSize] = { 0 };

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Kernels|Permute Activation Function Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        bool is_test_passed = true;
        const permute_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metics;
        if (!(cur_test->in.is_valid() && cur_test->out.is_valid())) {
            reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
            is_test_passed = false;
        }

        mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
        mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
        if (is_test_passed &&
                (tensor_quantizer::validate_tensor(input) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(out) != tensor_quantizer::kOk)) {
            reporter.report_message(cur_test->descr, 
                                    "FAILED at quantization step: more memory for one of tensors might be required");
            is_test_passed = false;
        }

        if (is_test_passed &&
                (mem_in_keeper.is_memory_corrupted() || mem_out_keeper.is_memory_corrupted())) {
            reporter.report_message(cur_test->descr,
                "FAILED at quantization step: memory beside one of operands is corrupted");
            is_test_passed = false;
        }

        // Run specific kernel for test 
        if (is_test_passed &&
                cur_test->mli_krn_permute(&input, &cur_test->cfg, &out) != MLI_STATUS_OK) {
            reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
            is_test_passed = false;
        }

        if (is_test_passed &&
                (mem_in_keeper.is_memory_corrupted() || mem_out_keeper.is_memory_corrupted())) {
            reporter.report_message(cur_test->descr,
                "FAILED after kernel run: memory beside one of operands is corrupted");
            is_test_passed = false;
        }

        if (is_test_passed &&
                test_metics.calculate_metrics(out, cur_test->out) == false) {
            reporter.report_message(cur_test->descr, "FAILED at comparison output with reference");
            is_test_passed = false;
        }

        if (is_test_passed) {
            crc32_calc data_crc;
            data_crc(input);
            data_crc(out);
            is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr, test_metics, cur_test->threshold, 
                                                                data_crc, cur_test->check_sum);
        }
        final_status &= is_test_passed;
    }

    reporter.report_outline("[AUTO] Group: mli_krn_permute", final_status);

    return (final_status) ? 0 : 1;
}
