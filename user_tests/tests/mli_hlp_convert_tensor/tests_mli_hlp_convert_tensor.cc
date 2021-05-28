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
#include <memory>

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "mli_types.h"
#include "test_tensor_quantizer.h"
#include "test_report.h"

#include "vectors_mli_hlp_convert_tensor.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

struct hlp_convert_test_operands {
    const char* descr;
    tensor_quantizer in;
    tensor_quantizer out;
    const quality_metrics threshold;
    const crc32_calc check_sum;
    const bool use_safx_version;
    const bool in_place_comp;
};

// Checksums of test tensors for various mli calculations mode. 
// When developer finished implementation of kernel and consider it as ok, one needs to populate
// proper checksums for tests in order to highlight any change which affects results.
#if defined(CRC_RM_UP)
// Shared CRC Results

const crc32_calc    test_1_chksum_fp32_sa8 { 0x1EF88E8B }, test_1_chksum_fp32_sa32{ 0xC095E743 }, test_1_chksum_fp32_fx16{ 0x3DCC1612 },
                    test_1_chksum_fp32_fx8 { 0xB928E46F }, test_1_chksum_sa8_sa32 { 0xAD84904D }, test_1_chksum_sa8_fx16 { 0xCA7BC6A1 }, 
                    test_1_chksum_sa8_fx8  { 0xFAB36D59 }, test_1_chksum_sa8_fp32 { 0x37A02758 }, test_1_chksum_sa32_sa8 { 0x9F9222B7 },  
                    test_1_chksum_sa32_fx16{ 0x1F94737B }, test_1_chksum_sa32_fx8 { 0x5ACA54A9 }, test_1_chksum_sa32_fp32{ 0x994E7C0C },
                    test_1_chksum_fx8_sa8  { 0xA4089244 }, test_1_chksum_fx8_sa32 { 0xEDF0C491 }, test_1_chksum_fx8_fx16 { 0x6DDB257A }, 
                    test_1_chksum_fx8_fp32 { 0xDF635A22 }, test_1_chksum_fx16_sa8 { 0x31D9ADF7 }, test_1_chksum_fx16_sa32{ 0x2E4D251E },
                    test_1_chksum_fx16_fx8 { 0x9609C713 }, test_1_chksum_fx16_fp32{ 0x6ACBEA49 }, test_1_chksum_sa8_sa8  { 0x01A56298 },
                    test_1_chksum_sa32_sa32{ 0x89768420 }, test_1_chksum_fx16_fx16{ 0x68094F10 };
#elif defined(CRC_RM_CONVERGENT)

const crc32_calc    test_1_chksum_fp32_sa8 { 0x1EF88E8B }, test_1_chksum_fp32_sa32{ 0xC095E743 }, test_1_chksum_fp32_fx16{ 0x3DCC1612 },
                    test_1_chksum_fp32_fx8 { 0xB928E46F }, test_1_chksum_sa8_sa32 { 0xAD84904D }, test_1_chksum_sa8_fx16 { 0xCA7BC6A1 }, 
                    test_1_chksum_sa8_fx8  { 0xFAB36D59 }, test_1_chksum_sa8_fp32 { 0x37A02758 }, test_1_chksum_sa32_sa8 { 0x9F9222B7 },  
                    test_1_chksum_sa32_fx16{ 0x1F94737B }, test_1_chksum_sa32_fx8 { 0x5ACA54A9 }, test_1_chksum_sa32_fp32{ 0x994E7C0C },
                    test_1_chksum_fx8_sa8  { 0xA4089244 }, test_1_chksum_fx8_sa32 { 0x2FBCC4B3 }, test_1_chksum_fx8_fx16 { 0x6DDB257A },
                    test_1_chksum_fx8_fp32 { 0xDF635A22 }, test_1_chksum_fx16_sa8 { 0x31D9ADF7 }, test_1_chksum_fx16_sa32{ 0x2E4D251E },
                    test_1_chksum_fx16_fx8 { 0x9609C713 }, test_1_chksum_fx16_fp32{ 0x6ACBEA49 }, test_1_chksum_sa8_sa8  { 0x01A56298 },
                    test_1_chksum_sa32_sa32{ 0xF73AF881 }, test_1_chksum_fx16_fx16{ 0xF1AEA71D };
#else // Not defined CRC_*

const crc32_calc    test_1_chksum_fp32_sa8,  test_1_chksum_fp32_sa32, test_1_chksum_fp32_fx16,
                    test_1_chksum_fp32_fx8,  test_1_chksum_sa8_sa32,  test_1_chksum_sa8_fx16,
                    test_1_chksum_sa8_fx8,   test_1_chksum_sa8_fp32,  test_1_chksum_sa32_sa8,
                    test_1_chksum_sa32_fx16, test_1_chksum_sa32_fx8,  test_1_chksum_sa32_fp32,
                    test_1_chksum_fx8_sa8,   test_1_chksum_fx8_sa32,  test_1_chksum_fx8_fx16,
                    test_1_chksum_fx8_fp32,  test_1_chksum_fx16_sa8,  test_1_chksum_fx16_sa32,
                    test_1_chksum_fx16_fx8,  test_1_chksum_fx16_fp32, test_1_chksum_sa8_sa8,
                    test_1_chksum_sa32_sa32, test_1_chksum_fx16_fx16;
#endif 

const quality_metrics thresholds_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                         /* SNR_DB = */30.f, quality_metrics::kPassValueQuantErrPerc };


static const hlp_convert_test_operands tests_list[] = {
    {"Test FP32 --> SA8",   input_1_fp32, input_2_sa8,
                            thresholds_general, test_1_chksum_fp32_sa8,
                            false, false},
    {"Test FP32 --> SA32",  input_1_fp32, input_2_sa32,
                            thresholds_general, test_1_chksum_fp32_sa32, 
                            false, false},
    {"Test FP32 --> FX16",  input_1_fp32, input_2_fx16,
                            thresholds_general, test_1_chksum_fp32_fx16, 
                            false, false},
    {"Test FP32 --> FX8",   input_1_fp32, input_2_fx8,
                            thresholds_general, test_1_chksum_fp32_fx8, 
                            false, false},
    {"Test SA8  --> SA32",  input_1_sa8, input_2_sa32,
                            thresholds_general, test_1_chksum_sa8_sa32, 
                            false, false},
    {"Test SA8  --> FX16",  input_1_sa8, input_2_fx16,
                            thresholds_general, test_1_chksum_sa8_fx16, 
                            false, false},
    {"Test SA8  --> FX8",   input_1_sa8, input_2_fx8,
                            thresholds_general, test_1_chksum_sa8_fx8, 
                            false, false},
    {"Test SA8  --> FP32",  input_1_sa8, input_2_fp32,
                            thresholds_general, test_1_chksum_sa8_fp32, 
                            false, false},
    {"Test SA32 --> SA8",   input_1_sa32, input_2_sa8,
                            thresholds_general, test_1_chksum_sa32_sa8, 
                            false, false},
    {"Test SA32 --> FX16",  input_1_sa32, input_2_fx16,
                            thresholds_general, test_1_chksum_sa32_fx16, 
                            false, false},
    {"Test SA32 --> FX8",   input_1_sa32, input_2_fx8,
                            thresholds_general, test_1_chksum_sa32_fx8, 
                            false, false},
    {"Test SA32 --> FP32",  input_1_sa32, input_2_fp32,
                            thresholds_general, test_1_chksum_sa32_fp32, 
                            false, false},
    {"Test FX8  --> SA8",   input_1_fx8, input_2_sa8,
                            thresholds_general, test_1_chksum_fx8_sa8, 
                            false, false},
    {"Test FX8  --> SA32",  input_1_fx8, input_2_sa32,
                            thresholds_general, test_1_chksum_fx8_sa32, 
                            false, false},
    {"Test FX8  --> FX16",  input_1_fx8, input_2_fx16,
                            thresholds_general, test_1_chksum_fx8_fx16, 
                            false, false},
    {"Test FX8  --> FP32",  input_1_fx8, input_2_fp32,
                            thresholds_general, test_1_chksum_fx8_fp32, 
                            false, false},
    {"Test FX16 --> SA8",   input_1_fx16, input_2_sa8,
                            thresholds_general, test_1_chksum_fx16_sa8, 
                            false, false},
    {"Test FX16 --> SA32",  input_1_fx16, input_2_sa32,
                            thresholds_general, test_1_chksum_fx16_sa32, 
                            false, false},
    {"Test FX16 --> FX8",   input_1_fx16, input_2_fx8,
                            thresholds_general, test_1_chksum_fx16_fx8, 
                            false, false},
    {"Test FX16 --> FP32",  input_1_fx16, input_2_fp32,
                            thresholds_general, test_1_chksum_fx16_fp32,
                            false, false},
    {"Test SA8  --> SA8",   input_1_sa8, input_2_sa8,
                            thresholds_general, test_1_chksum_sa8_sa8,
                            false, false},
    {"Test SA32 --> SA32",  input_1_sa32, input_2_sa32,
                            thresholds_general, test_1_chksum_sa32_sa32,
                            false, false},
    {"Test FX16 --> FX16",  input_1_fx16, input_2_fx16,
                            thresholds_general, test_1_chksum_fx16_fx16,
                            false, false},
};

constexpr int kMemSize = 2048;
static IO_DATA_ATTR int8_t scratch_mem_in[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemSize] = { 0 };
constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

bool run_test(mli_status(*mli_hlp_convert_tensor_func)(const mli_tensor *, mli_tensor *), 
              const reporter_full & reporter,
              const hlp_convert_test_operands* cur_test) {
    memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
    memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
    bool is_test_passed = true;
    quality_metrics test_metrics;

    std::unique_ptr<char[]> description;
    const char *current_description = cur_test->descr;

    if (mli_hlp_convert_tensor_func == mli_hlp_convert_tensor_safx) {
        const char safx_label[] = " (safx)";
        description = std::unique_ptr<char[]>(new (std::nothrow) char[strlen(cur_test->descr) + strlen(safx_label) + 1]);
        if (description == nullptr) {
            reporter.report_message(current_description, "FAILED at init: Can't allocate memory for description");
            is_test_passed = false;
        }
        if (is_test_passed) {
            strcpy(description.get(), cur_test->descr);
            strcat(description.get(), safx_label);
            current_description = description.get();
        }
    }

    if (is_test_passed && 
            !(cur_test->in.is_valid() && cur_test->out.is_valid())) {
        reporter.report_message(current_description, "FAILED at init: Bad source data for one of tensors");
        is_test_passed = false;
    }

    // Quantize input tensor using mli_hlp_convert_tensor to test FP32 --> XX conversion
    mli_tensor input = cur_test->in.get_not_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
    if (is_test_passed &&
            tensor_quantizer::validate_tensor(input) != tensor_quantizer::kOk) {
        reporter.report_message(current_description,
            "FAILED at quantization step: more memory for input tensor might be required");
        is_test_passed = false;
    }
  
    mli_tensor source_input_tensor = cur_test->in.get_source_float_tensor();
    if (is_test_passed &&
            mli_hlp_convert_tensor_func(&source_input_tensor, &input) != MLI_STATUS_OK) {
        reporter.report_message(current_description, "FAILED at kernel run: kernel returned bad status");
        is_test_passed = false;
    }

    // Check if conversion from FP32 went right
    if (is_test_passed &&
            test_metrics.calculate_metrics(input, cur_test->in) == false) {
        reporter.report_message(current_description, "FAILED at comparison input with reference. Error in conversion from fp32.");
        is_test_passed = false;
    }

    mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
    if (cur_test->in_place_comp) {
        out.data = input.data;
    }
    if (is_test_passed &&
            tensor_quantizer::validate_tensor(out) != tensor_quantizer::kOk) {
        reporter.report_message(current_description,
            "FAILED at quantization step: more memory for output tensor might be required");
        is_test_passed = false;
    }
    mli_tensor source_out_tensor = out;

    // Run specific kernel for test
    if (is_test_passed &&
            mli_hlp_convert_tensor_func(&input, &out) != MLI_STATUS_OK) {
        reporter.report_message(current_description, "FAILED at kernel run: kernel returned bad status");
        is_test_passed = false;
    }

    // Check if conversion result matches initialized value
    if (is_test_passed &&
            test_metrics.calculate_metrics(out, cur_test->out) == false) {
        reporter.report_message(current_description, "FAILED at comparison output with reference");
        is_test_passed = false;
    }

    // Check that kernel didn't modify quantization parameters provided by user.
    if (is_test_passed) {

        if (out.el_type == MLI_EL_FX_8 || out.el_type == MLI_EL_FX_16) {
            is_test_passed &= out.el_params.fx.frac_bits == source_out_tensor.el_params.fx.frac_bits;
        } else if (out.el_type == MLI_EL_SA_8 || out.el_type == MLI_EL_SA_32) {
            if (out.el_params.sa.dim < 0 || source_out_tensor.el_params.sa.dim < 0) {
                is_test_passed &=
                    (out.el_params.sa.scale.mem.i16 == source_out_tensor.el_params.sa.scale.mem.i16) &&
                    (out.el_params.sa.zero_point.mem.i16 == source_out_tensor.el_params.sa.zero_point.mem.i16) &&
                    (out.el_params.sa.scale_frac_bits.mem.i8 ==
                        source_out_tensor.el_params.sa.scale_frac_bits.mem.i8);
            }
        } else if (out.el_type != MLI_EL_FP_32) {
            is_test_passed = false;
        }
    }

    if (is_test_passed) {
        crc32_calc data_crc;
        data_crc(input);
        data_crc(out);

        is_test_passed &= reporter.evaluate_and_report_case(current_description, test_metrics, cur_test->threshold,
            data_crc, cur_test->check_sum);
    }
    return is_test_passed;
}

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Helpers|Data Conversion Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        bool is_test_passed = true;
        const hlp_convert_test_operands* cur_test = &tests_list[i];

#if PLATFORM == V2DSP_XY
        if (strstr(cur_test->descr, "Test SA32 --> SA32") != nullptr) {
            // EMxD vectorized code doesn't work properly with
            // SA32 --> SA32 conversion in CONVERGENT rounding mode.
            reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
            continue;
        }

        if (strstr(cur_test->descr, "Test FX16 --> FX16") != nullptr) {
            // EMxD vectorized code doesn't work properly with
            // FX16 --> FX16 conversion in CONVERGENT rounding mode.
            reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
            continue;
        }
#endif
        is_test_passed = run_test(mli_hlp_convert_tensor, reporter, cur_test);

        if (is_test_passed &&
                cur_test->use_safx_version)
        is_test_passed = run_test(mli_hlp_convert_tensor_safx, reporter, cur_test);


        final_status &= is_test_passed;
    }

    reporter.report_outline("[AUTO] Group: mli_hlp_convert_tensor", final_status);

    return (final_status) ? 0 : 1;
}
