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

#include "vectors_mli_krn_argmax.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

typedef mli_status(*argmax_func_ptr)(
    const mli_tensor* /*input*/,
    const mli_argmax_cfg* /*cfg*/,
    mli_tensor* /*output*/);

struct argmax_test_operands {
    const char* descr;
    const argmax_func_ptr mli_krn_argmax;
    tensor_quantizer in;
    tensor_quantizer out;
    const mli_argmax_cfg cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

// Checksums of test tensors for various mli calculations mode. 
// When developer finished implementation of kernel and consider it as ok, one needs to populate
// proper checksums for tests in order to highlight any change which affects results.
#if defined(CRC_RM_UP)
// Shared CRC Results

const crc32_calc    test_1_chksum_fx16_fx8  { 0x23A39F6B }, test_1_chksum_fx16_sa8 { 0x23A39F6B }, test_1_chksum_fx16_fx16 { 0x0841F4C2 },
                    test_1_chksum_fx16_sa32 { 0x02C59977 }, test_1_chksum_sa8_fx8  { 0x6341BBF5 }, test_1_chksum_sa8_sa8   { 0x6341BBF5 },
                    test_1_chksum_sa8_fx16  { 0x1FB6A8A5 }, test_1_chksum_sa8_sa32 { 0x4AA46E3F }, test_2_chksum_fx16_fx16 { 0xAD273965 },
                    test_2_chksum_fx16_sa32 { 0xFF8FA926 }, test_3_chksum_fx16_fx8 { 0x88B589EE }, test_3_chksum_fx16_sa8  { 0x88B589EE },
                    test_3_chksum_sa8_fx8   { 0x2D84A0F9 }, test_3_chksum_sa8_sa8  { 0x2D84A0F9 };
#elif defined(CRC_RM_CONVERGENT)

const crc32_calc    test_1_chksum_fx16_fx8  { 0x23A39F6B }, test_1_chksum_fx16_sa8 { 0x23A39F6B }, test_1_chksum_fx16_fx16 { 0x0841F4C2 },
                    test_1_chksum_fx16_sa32 { 0x02C59977 }, test_1_chksum_sa8_fx8  { 0x6341BBF5 }, test_1_chksum_sa8_sa8   { 0x6341BBF5 },
                    test_1_chksum_sa8_fx16  { 0x1FB6A8A5 }, test_1_chksum_sa8_sa32 { 0x4AA46E3F }, test_2_chksum_fx16_fx16 { 0xAD273965 },
                    test_2_chksum_fx16_sa32 { 0xFF8FA926 }, test_3_chksum_fx16_fx8 { 0x88B589EE }, test_3_chksum_fx16_sa8  { 0x88B589EE },
                    test_3_chksum_sa8_fx8   { 0x2D84A0F9 }, test_3_chksum_sa8_sa8  { 0x2D84A0F9 };
#else // Not defined CRC_*

const crc32_calc    test_1_chksum_fx16_fx8,  test_1_chksum_fx16_sa8, test_1_chksum_fx16_fx16,
                    test_1_chksum_fx16_sa32, test_1_chksum_sa8_fx8,  test_1_chksum_sa8_sa8,
                    test_1_chksum_sa8_fx16 , test_1_chksum_sa8_sa32, test_2_chksum_fx16_fx16,
                    test_2_chksum_fx16_sa32, test_3_chksum_sa32_fx8, test_3_chksum_sa32_fp32,
                    test_3_chksum_fx8_sa8,   test_3_chksum_fx8_sa32;
#endif 

const quality_metrics thresholds_test_1_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR_DB = */0.0f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_test_2_3_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                  /* SNR_DB = */227.2f, quality_metrics::kPassValueQuantErrPerc };


static const argmax_test_operands tests_list[] = {

    // Basic functionality test
    {"Test 1 FX16 - FX8  (1 elem)",             mli_krn_argmax_fx16,
                                       input_1_fx16, test_1_out_fx8, test_1_cfg,
                                       thresholds_test_1_general, test_1_chksum_fx16_fx8},
    {"Test 1 FX16 - SA8  (1 elem)",             mli_krn_argmax_fx16,
                                       input_1_fx16, test_1_out_sa8, test_1_cfg,
                                       thresholds_test_1_general, test_1_chksum_fx16_sa8},
    {"Test 1 FX16 - FX16 (1 elem)",            mli_krn_argmax_fx16,
                                       input_1_fx16, test_1_out_fx16, test_1_cfg,
                                       thresholds_test_1_general, test_1_chksum_fx16_fx16},
    {"Test 1 FX16 - SA32 (1 elem)",            mli_krn_argmax_fx16,
                                       input_1_fx16, test_1_out_sa32, test_1_cfg,
                                       thresholds_test_1_general, test_1_chksum_fx16_sa32},
    {"Test 1 SA8  - FX8  (1 elem)",             mli_krn_argmax_sa8,
                                       input_1_sa8, test_1_out_fx8, test_1_cfg,
                                       thresholds_test_1_general, test_1_chksum_sa8_fx8},
    {"Test 1 SA8  - SA8  (1 elem)",             mli_krn_argmax_sa8,
                                       input_1_sa8, test_1_out_sa8, test_1_cfg,
                                       thresholds_test_1_general, test_1_chksum_sa8_sa8},
    {"Test 1 SA8  - FX16 (1 elem)",            mli_krn_argmax_sa8,
                                       input_1_sa8, test_1_out_fx16, test_1_cfg,
                                       thresholds_test_1_general, test_1_chksum_sa8_fx16},
    {"Test 1 SA8  - SA32 (1 elem)",            mli_krn_argmax_sa8,
                                       input_1_sa8, test_1_out_sa32, test_1_cfg,
                                       thresholds_test_1_general, test_1_chksum_sa8_sa32},
    {"Test 2 FX16 - FX16 (144 elem)",            mli_krn_argmax_fx16,
                                       input_2_fx16, test_2_out_fx16, test_2_cfg,
                                       thresholds_test_2_3_general, test_2_chksum_fx16_fx16},
    {"Test 2 FX16 - SA32 (144 elem)",            mli_krn_argmax_fx16,
                                       input_2_fx16, test_2_out_sa32, test_2_cfg,
                                       thresholds_test_2_3_general, test_2_chksum_fx16_sa32},
    {"Test 3 FX16 - FX8 (axis = 2)",             mli_krn_argmax_fx16,
                                       input_3_fx16, test_3_out_fx8, test_3_cfg,
                                       thresholds_test_2_3_general, test_3_chksum_fx16_fx8},
    {"Test 3 FX16 - SA8 (axis = 2)",             mli_krn_argmax_fx16,
                                       input_3_fx16, test_3_out_sa8, test_3_cfg,
                                       thresholds_test_2_3_general, test_3_chksum_fx16_sa8},
    {"Test 3 SA8  - FX8 (axis = 2)",             mli_krn_argmax_sa8,
                                       input_3_sa8, test_3_out_fx8, test_3_cfg,
                                       thresholds_test_2_3_general, test_3_chksum_sa8_fx8},
    {"Test 3 SA8  - SA8 (axis = 2)",             mli_krn_argmax_sa8,
                                       input_3_sa8, test_3_out_sa8, test_3_cfg,
                                       thresholds_test_2_3_general, test_3_chksum_sa8_sa8}
};

constexpr int kMemSize = 10000;
static IO_DATA_ATTR int8_t scratch_mem_in[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemSize] = { 0 };

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Kernels|Argmax Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        bool is_test_passed = true;
        const argmax_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metrics;

        if (!(cur_test->in.is_valid() && cur_test->out.is_valid())) {
            reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
            is_test_passed = false;
        }

        mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
        mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
        mli_tensor source_out_tensor = out;
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
                cur_test->mli_krn_argmax(&input, &cur_test->cfg, &out) != MLI_STATUS_OK) {
            reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
            is_test_passed = false;
        }

        if (is_test_passed &&
                test_metrics.calculate_metrics(out, cur_test->out) == false) {
            reporter.report_message(cur_test->descr, "FAILED at comparison output with reference");
            is_test_passed = false;
        }

        if (is_test_passed) {
            crc32_calc data_crc;
            data_crc(input);
            data_crc(out);

            is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr, test_metrics, cur_test->threshold, 
                                                                data_crc, cur_test->check_sum);
        }
        final_status &= is_test_passed;
    }

    reporter.report_outline("[AUTO] Group: mli_krn_argmax", final_status);

    return (final_status) ? 0 : 1;
}
