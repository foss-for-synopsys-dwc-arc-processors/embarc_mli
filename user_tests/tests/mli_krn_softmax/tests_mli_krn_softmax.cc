/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_api.h"

#include <stdint.h>
#include <stdio.h>

#include "test_infra.h"
#include "mli_types.h"
#include "test_quantizer.h"
#include "test_report.h"

#include "vectors_mli_krn_softmax.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_keeper;

typedef mli_status(*softmax_func_ptr)(
    const mli_tensor* /*in*/,
    const mli_softmax_cfg* /*cfg*/,
    mli_tensor* /*out*/);

struct softmax_test_operands {
    const char* descr;
    const softmax_func_ptr mli_krn_softmax;
    tensor_quantizer in;
    tensor_quantizer out;
    const mli_softmax_cfg cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

// Checksums of test tensors for various platforms. 
// When developer finished implementation of kernel and consider it as ok, He need to populate
// proper checksums for tests in order to highlight any change which affects results.
#if defined NO_CRC_CHECK
const crc32_calc  test_1_chksum_fx16, test_1_chksum_sa8,
                  test_2_chksum_fx16, test_2_chksum_sa8,
                  test_3_chksum_fx16, test_3_chksum_sa8,
                  test_4_chksum_fx16, test_4_chksum_sa8,
                  test_5_chksum_fx16, test_5_chksum_sa8,
                  test_6_chksum_fx16, test_6_chksum_sa8;
#else
// Need to distinquish platforms and update checksums
const crc32_calc  test_1_chksum_fx16{ 0xFE566434 }, test_1_chksum_sa8{ 0xC2D91AF1 },
                  test_2_chksum_fx16{ 0x57D65150 }, test_2_chksum_sa8{ 0x426712EC },
                  test_3_chksum_fx16{ 0xBF9EAF0C }, test_3_chksum_sa8{ 0xA4C61305 },
                  test_4_chksum_fx16{ 0xC98520CF }, test_4_chksum_sa8{ 0x283A9958 },
                  test_5_chksum_fx16{ 0xCD358702 }, test_5_chksum_sa8{ 0x2866E46F },
                  test_6_chksum_fx16{ 0x767E77D1 }, test_6_chksum_sa8{ 0x4899968F };
#endif

const quality_metrics thresholds_fx16_general { /* MaxAbsErr = */0.0002, quality_metrics::kPassValueSnr,
                                                quality_metrics::kPassValueSnrDb, /*Quant Error Perc = */ 15.f };

const quality_metrics thresholds_sa8_general{ /* MaxAbsErr = */0.02, quality_metrics::kPassValueSnr,
                                                quality_metrics::kPassValueSnrDb, /*Quant Error Perc = */ 20.f };

const quality_metrics thresholds_sa8_test2{ /* MaxAbsErr = */0.015, quality_metrics::kPassValueSnr,
                                                quality_metrics::kPassValueSnrDb, /*Quant Error Perc = */ 16.f }; 

const quality_metrics thresholds_sa8_test3{ /* MaxAbsErr = */0.025, quality_metrics::kPassValueSnr,
                                                quality_metrics::kPassValueSnrDb, /*Quant Error Perc = */ 20.f };




static const softmax_test_operands tests_list[] = {
    {"Test 1 FX16 similar Probs",  mli_krn_softmax_fx16,
                                   input_1_fx16, test_1_out_fx16, test_1_cfg,
                                   thresholds_fx16_general, test_1_chksum_fx16},
    {"Test 1 SA8 similar Probs",   mli_krn_softmax_sa8,
                                   input_1_sa8, test_1_out_sa8, test_1_cfg,
                                   thresholds_sa8_general, test_1_chksum_sa8},

    {"Test 2 FX16 OneHot",  mli_krn_softmax_fx16,
                            input_2_fx16, test_2_out_fx16, test_2_cfg,
                            thresholds_fx16_general, test_2_chksum_fx16},
    {"Test 2 SA8 OneHot",   mli_krn_softmax_sa8,
                            input_2_sa8, test_2_out_sa8, test_2_cfg,
                            thresholds_sa8_test2, test_2_chksum_sa8},

    {"Test 3 FX16 Axis=0",  mli_krn_softmax_fx16,
                            input_3_fx16, test_3_out_fx16, test_3_cfg,
                            thresholds_fx16_general, test_3_chksum_fx16},
    {"Test 3 SA8 Axis=0",   mli_krn_softmax_sa8,
                            input_3_sa8, test_3_out_sa8, test_3_cfg, 
                            thresholds_sa8_test3, test_3_chksum_sa8},

    {"Test 4 FX16 Axis=2",  mli_krn_softmax_fx16,
                            input_3_fx16, test_4_out_fx16, test_4_cfg,
                            thresholds_fx16_general, test_4_chksum_fx16},
    {"Test 4 SA8 Axis=2",   mli_krn_softmax_sa8,
                            input_3_sa8, test_4_out_sa8, test_4_cfg,
                            thresholds_sa8_general, test_4_chksum_sa8},

    {"Test 5 FX16 MultiDim Axis=-1",  mli_krn_softmax_fx16,
                                      input_3_fx16, test_5_out_fx16, test_5_cfg,
                                      thresholds_fx16_general, test_5_chksum_fx16},
    {"Test 5 SA8 MultiDim Axis=-1",   mli_krn_softmax_sa8,
                                      input_3_sa8, test_5_out_sa8, test_5_cfg,
                                      thresholds_sa8_general, test_5_chksum_sa8},

    {"Test 6 FX16 Memstride",  mli_krn_softmax_fx16,
                               input_3_fx16, test_6_out_fx16, test_6_cfg,
                               thresholds_fx16_general, test_6_chksum_fx16},
    {"Test 6 SA8 Memstride",   mli_krn_softmax_sa8,
                               input_3_sa8, test_6_out_sa8, test_6_cfg,
                               thresholds_sa8_general, test_6_chksum_sa8},
};

constexpr int kMemSize = 2047;
static IO_DATA_ATTR int8_t scratch_mem_in[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemSize] = { 0 };

static const int tests_num = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Kernels|Softmax Activation Function Tests");
    for (int i = 0; i < tests_num; ++i) {
        memory_keeper mem_in_keeper(scratch_mem_in, sizeof(scratch_mem_in));
        memory_keeper mem_out_keeper(scratch_mem_out, sizeof(scratch_mem_out));
        bool is_test_passed = true;
        const softmax_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metics;
        if (!(cur_test->in.is_valid() && cur_test->out.is_valid())) {
            reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
            is_test_passed = false;
        }

        mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.afford_memory(cur_test->in));
        mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.afford_memory(cur_test->out));
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

        // Run specific for test function
        if (is_test_passed &&
                cur_test->mli_krn_softmax(&input, &cur_test->cfg, &out) != MLI_STATUS_OK) {
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

    reporter.report_outline("[AUTO] Group: mli_krn_softmax", final_status);

    return (final_status) ? 0 : 1;
}
