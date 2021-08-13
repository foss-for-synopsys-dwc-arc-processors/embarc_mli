/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

// Standard asserts should be intentionally turned-on by defenition of TEST_DEBUG.
#if !defined(TEST_DEBUG)
#define NDEBUG
#endif

#include "mli_api.h"
#include "mli_config.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <limits>

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "mli_types.h"
#include "test_tensor_quantizer.h"
#include "test_report.h"

#include "vectors_mli_hlp_tensor_struct.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_basic;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

constexpr int kMemSize = 1000;
static IO_DATA_ATTR int8_t scratch_mem_in[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemSize] = { 0 };

bool run_test_count_and_size();
bool run_test_quant_params_getters();
bool run_test_accu_bits_getters();
bool run_test_create_subtensor();

// Main entry point. Running test procedues for various helpers
//=============================================================
int main() {
    constexpr bool kSuccess = true; 
    bool final_status = run_test_count_and_size();
    
    if(final_status == kSuccess)
        final_status = run_test_quant_params_getters();

    if (final_status == kSuccess)
        final_status = run_test_accu_bits_getters();

    if (final_status == kSuccess)
        final_status = run_test_create_subtensor();

    return (final_status == kSuccess) ? 0 : 1;
}

// Tests procedure to count elements and define it's size.
//=======================================================================
struct count_and_size_test_operands {
    const char* descr;
    tensor_quantizer in;
    const uint32_t elem_num_expected[MLI_MAX_RANK];
    const uint32_t elem_size;
};

bool run_test_count_and_size() {
    bool test_status = true;
    const reporter_basic reporter;
    static const count_and_size_test_operands count_and_size_tests_list[] = {
        {"FX8 tensor ", input_1_fx8,  INPUT_1_ELEM_COUNT, /*sizeof(int8_t) =  */ 1},
        {"FX16 tensor ", input_1_fx16, INPUT_1_ELEM_COUNT, /*sizeof(int16_t) = */ 2},
        {"SA8 tensor ", input_1_sa8,  INPUT_1_ELEM_COUNT, /*sizeof(int8_t) =  */ 1},
        {"SA8 tensor per axis", input_1_sa8_per_axis,  INPUT_1_ELEM_COUNT, /*sizeof(int8_t) =  */ 1},
        {"SA32 tensor", input_1_sa32, INPUT_1_ELEM_COUNT, /*sizeof(int32_t) = */ 4},
        {"SA32 tensor per axis", input_1_sa32_per_axis, INPUT_1_ELEM_COUNT, /*sizeof(int32_t) = */ 4},
        {"FP32 tensor ", input_1_fp32, INPUT_1_ELEM_COUNT, /*sizeof(float) =   */ 4},
        {"FX8 scalar tsr", input_2_fx8,  INPUT_2_ELEM_COUNT, /*sizeof(int8_t) =  */ 1},
        {"FX16 scalar tsr", input_2_fx16, INPUT_2_ELEM_COUNT, /*sizeof(int16_t) = */ 2},
        {"SA8 scalar tsr", input_2_sa8,  INPUT_2_ELEM_COUNT, /*sizeof(int8_t) =  */ 1},
        {"SA32 scalar tsr", input_2_sa32, INPUT_2_ELEM_COUNT, /*sizeof(int32_t) = */ 4},
        {"FP32 scalar tsr", input_2_fp32, INPUT_2_ELEM_COUNT, /*sizeof(float) =   */ 4},
    };
    constexpr int kTestsNum = sizeof(count_and_size_tests_list) / sizeof(count_and_size_tests_list[0]);

    reporter.report_header("MLI|Helpers|Count Elements and Element Size Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
        bool is_test_passed = true;
        const count_and_size_test_operands* cur_test = &count_and_size_tests_list[i];

        if (!(cur_test->in.is_valid())) {
            is_test_passed = false;
            reporter.report_case(cur_test->descr, "At init: Bad source data for input tensor", is_test_passed);
        }

        const mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
        if (is_test_passed && (tensor_quantizer::validate_tensor(input) != tensor_quantizer::kOk)) {
            is_test_passed = false;
            reporter.report_case(cur_test->descr, 
                "At quantization step: more memory for in tensor is be required", is_test_passed);
        }

        if (is_test_passed && (mem_in_keeper.is_memory_corrupted())) {
            is_test_passed = false;
            reporter.report_case(cur_test->descr,
                "At quantization step: memory beside one of operands is corrupted", is_test_passed);
        }

        // Run specific kernel for test
        const uint32_t end_dim = (input.rank == 0) ? 1 : input.rank;
        uint32_t elem_num_returned[MLI_MAX_RANK] = { static_cast<uint32_t>(-1) };
        uint32_t cur_elem_size = 0;
        crc32_calc data_crc_before, data_crc_after;
        if (is_test_passed) {
            data_crc_before(input);
            for (uint32_t start_dim = 0; start_dim < end_dim; ++start_dim) {
                elem_num_returned[start_dim] = mli_hlp_count_elem_num(&input, start_dim);
            }
            cur_elem_size = mli_hlp_tensor_element_size(&input);
            data_crc_after(input);
        }

        if (is_test_passed && 
                (data_crc_before.get() != data_crc_after.get() ||
                mem_in_keeper.is_memory_corrupted())) {
            is_test_passed = false;
            reporter.report_case(cur_test->descr,
                "At function run: memory is corrupted after functions invokation", is_test_passed);
        }

        if (is_test_passed && cur_test->elem_size != cur_elem_size ) {
            is_test_passed = false;
            reporter.report_case(cur_test->descr,
                "At function run: mli_hlp_tensor_element_size returned not expected value", is_test_passed);
        }

        for (uint32_t start_dim = 0; start_dim < end_dim; ++start_dim) {
            if (is_test_passed && (elem_num_returned[start_dim] != cur_test->elem_num_expected[start_dim])) {
                is_test_passed = false;
                reporter.report_case(cur_test->descr,
                    "At DUT function run: mli_hlp_count_elem_num returned not expected value", is_test_passed);
            }
        }

        if (is_test_passed) {
            reporter.report_case(cur_test->descr, nullptr, is_test_passed);
        }
        test_status &= is_test_passed;
    }
    reporter.report_outline("[AUTO] Group: mli_hlp_count_elem_num & mli_hlp_tensor_element_size", test_status);
    return test_status;
}

// Tests procedure for various functions to get quantization parameters
//=======================================================================
struct get_quant_params_tests_operands {
    const char* descr;
    tensor_quantizer in;
    const uint32_t arr_common_size;
    const int8_t* scale_shift_expected;
    const float* scale_expected;
    const float* zero_offset_expected;
};

bool run_test_quant_params_getters() {
    bool test_status = true;
    const reporter_basic reporter;
    static const get_quant_params_tests_operands get_quant_params_tests_list[] = {
        {"FX8 tensor ",          input_1_fx8, /*arr_common_size =*/ 1,
        &input_1_fx8_frac, &input_1_fx8_exp_scale, &input_1_fx_fp_exp_zero},
        {"FX16 tensor ",         input_1_fx16,  /*arr_common_size =*/ 1,
        &input_1_fx16_frac, &input_1_fx16_exp_scale, &input_1_fx_fp_exp_zero},
        {"SA8 tensor ",          input_1_sa8, /*arr_common_size =*/ 1,
        input_1_scale_frac, &input_1_scale, &input_1_zero_point},
        {"SA8 tensor per axis",  input_1_sa8_per_axis,  sizeof(input_1_scales)/sizeof(input_1_scales[0]),
        input_1_scales_frac, input_1_scales, input_1_zero_points},
        {"SA32 tensor ",         input_1_sa32, /*arr_common_size =*/ 1,
        input_1_scale_frac, &input_1_scale, &input_1_zero_point},
        {"SA32 tensor per axis", input_1_sa32_per_axis,  sizeof(input_1_scales)/sizeof(input_1_scales[0]),
        input_1_scales_frac, input_1_scales, input_1_zero_points},
        {"FP32 tensor ",         input_1_fp32,  /*arr_common_size =*/ 1,
        &input_1_fp_exp_frac, &input_1_fp_exp_scale, &input_1_fx_fp_exp_zero}
    };
    constexpr int kTestsNum = sizeof(get_quant_params_tests_list) / sizeof(get_quant_params_tests_list[0]);

    reporter.report_header("MLI|Helpers|Get Quantization Params Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
        bool is_test_passed = true;
        const get_quant_params_tests_operands* cur_test = &get_quant_params_tests_list[i];

        if (!(cur_test->in.is_valid())) {
            is_test_passed = false;
            reporter.report_case(cur_test->descr, "FAILED at init: Bad source data for input tensor", is_test_passed);
        }

        const mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
        if (is_test_passed && (tensor_quantizer::validate_tensor(input) != tensor_quantizer::kOk)) {
            is_test_passed = false;
            reporter.report_case(cur_test->descr, 
                "FAILED at quantization step: more memory for one of tensors might be required", is_test_passed);
        }

        const int quant_params_num = cur_test->arr_common_size;
        if (is_test_passed) {
            uint32_t tensor_quant_params_num = 1;
            if (input.el_type == MLI_EL_SA_8 || input.el_type == MLI_EL_SA_32) {
                tensor_quant_params_num = (input.el_params.sa.dim < 0) ? 1 : input.shape[input.el_params.sa.dim];
            } 

            if (quant_params_num > tensor_quant_params_num) {
                is_test_passed = false;
                reporter.report_case(cur_test->descr, 
                    "FAILED as test config isn't compatible with test data", is_test_passed);
            }
        }

        // Run DUT functions for a test
        for (int idx = 0; is_test_passed && idx < quant_params_num; ++idx) {
            crc32_calc data_crc_before, data_crc_after;
            data_crc_before(input);
            const auto cur_scale_shift = mli_hlp_tensor_scale_shift(&input, idx);
            const auto cur_scale = mli_hlp_tensor_scale(&input, idx);
            const auto cur_zero_offset = mli_hlp_tensor_zero_offset(&input, idx);
            if ((data_crc_before.get() != data_crc_after(input) ||
                mem_in_keeper.is_memory_corrupted())) {
                is_test_passed = false;
                reporter.report_case(cur_test->descr,
                    "FAILED at func run: memory is corrupted after functions invokation", is_test_passed);
            }

            if (is_test_passed) {
                const int exp_scale_shift = cur_test->scale_shift_expected[idx];
                const float exp_zero_offset = cur_test->zero_offset_expected[idx] / cur_test->scale_expected[idx];
                const float mult = (exp_scale_shift >= 0) ? (float)(1ll << exp_scale_shift)
                    : (float)(1.f / (1ll << (-exp_scale_shift)));
                const float exp_scale = mult * cur_test->scale_expected[idx];

                auto are_vals_close_enough = [](float float_val, int int_val) -> bool {
                    return abs(float_val - int_val) < 1.0f;
                };
                if (exp_scale_shift != cur_scale_shift) {
                    is_test_passed = false;
                    reporter.report_case(cur_test->descr,
                        "FAILED at func run: mli_hlp_tensor_scale_shift returned not expected value", is_test_passed);
                } else if (!are_vals_close_enough(exp_zero_offset, cur_zero_offset)) {
                    is_test_passed = false;
                    reporter.report_case(cur_test->descr,
                        "FAILED at func run: mli_hlp_tensor_zero_offset returned not expected value", is_test_passed);
                } else if (!are_vals_close_enough(exp_scale, cur_scale)) {
                    is_test_passed = false;
                    reporter.report_case(cur_test->descr,
                        "FAILED at func run: mli_hlp_tensor_scale returned not expected value", is_test_passed);
                }
            }
        }
        if (is_test_passed) {
            reporter.report_case(cur_test->descr, nullptr, is_test_passed);
        }
        test_status &= is_test_passed;
    }
    reporter.report_outline("[AUTO] Group: mli_hlp_tensor_[scale|scale_shift|zero_offset]", test_status);
    return test_status;
}

// Tests procedure for mli_hlp_accu_guard_bits_* functions
//=============================================================
struct acc_bits_range {
    uint8_t min;
    uint8_t max;
};

bool run_test_accu_bits_getters() {
    bool test_status = true;
    const reporter_basic reporter;

#if (PLATFORM == V2DSP) || (PLATFORM == V2DSP_XY) || (PLATFORM == V2DSP_WIDE)
    const char* kPlatformStr = PLATFORM_STR;
    const acc_bits_range sa8_sa8_acc_bits = {/*min = */ 16, /*max = */ 16 };
    const acc_bits_range fx16_fx16_acc_bits = {/*min = */ 8, /*max = */ 8 };
    const acc_bits_range fx16_fx8_acc_bits = {/*min = */ 8, /*max = */ 8 };

#elif (PLATFORM == V2DSP_VECTOR)
    const char* kPlatformStr = PLATFORM_STR;
    const acc_bits_range sa8_sa8_acc_bits = {/*min = */ 0, /*max = */ 8 };
    const acc_bits_range fx16_fx16_acc_bits = {/*min = */ 0, /*max = */ 8 };
    const acc_bits_range fx16_fx8_acc_bits = {/*min = */ 8, /*max = */ 16 };

#elif (PLATFORM == X86_PLATFORM)
    const char* kPlatformStr = PLATFORM_STR;
    const acc_bits_range sa8_sa8_acc_bits = {/*min = */ 16, /*max = */ 16 };
    const acc_bits_range fx16_fx16_acc_bits = {/*min = */ 32, /*max = */ 32 };
    const acc_bits_range fx16_fx8_acc_bits = {/*min = */ 8, /*max = */ 8 };
#else
    const char* kPlatformStr = "Unknown";
    const acc_bits_range sa8_sa8_acc_bits = {/*min = */ 0, /*max = */ std::numeric_limits<uint8_t>::max() };
    const acc_bits_range fx16_fx16_acc_bits = {/*min = */ 0, /*max = */ std::numeric_limits<uint8_t>::max() };
    const acc_bits_range fx16_fx8_acc_bits = {/*min = */ 0, /*max = */ std::numeric_limits<uint8_t>::max() };
#endif



    reporter.report_header("MLI|Helpers|Get Accumulator Bits");
    char* message = (char *)scratch_mem_in;
    const char* message_fmt = "%s platform %d guard bits";

    uint8_t cur_acc_bits = mli_hlp_accu_guard_bits_sa8_sa8();
    bool is_value_expected = (cur_acc_bits >= sa8_sa8_acc_bits.min) && (cur_acc_bits <= sa8_sa8_acc_bits.max);
    int msg_len = sprintf(message, message_fmt, kPlatformStr, int(cur_acc_bits));
    assert(msg_len < sizeof(scratch_mem_in));
    reporter.report_case("accu_guard_bits sa8_sa8", message, is_value_expected);
    test_status &= is_value_expected;


    cur_acc_bits = mli_hlp_accu_guard_bits_fx16_fx16();
    is_value_expected = (cur_acc_bits >= fx16_fx16_acc_bits.min) && (cur_acc_bits <= fx16_fx16_acc_bits.max);
    msg_len = sprintf(message, message_fmt, kPlatformStr, int(cur_acc_bits));
    assert(msg_len < sizeof(scratch_mem_in));
    reporter.report_case("accu_guard_bits fx16_fx16", message, is_value_expected);
    test_status &= is_value_expected;


    cur_acc_bits = mli_hlp_accu_guard_bits_fx16_fx8();
    is_value_expected = (cur_acc_bits >= fx16_fx8_acc_bits.min) && (cur_acc_bits <= fx16_fx8_acc_bits.max);
    msg_len = sprintf(message, message_fmt, kPlatformStr, int(cur_acc_bits));
    assert(msg_len < sizeof(scratch_mem_in));
    reporter.report_case("accu_guard_bits fx16_fx8", message, is_value_expected);
    test_status &= is_value_expected;

    reporter.report_outline("[AUTO] Group: mli_hlp_accu_guard_bits_[sa8_sa8|fx16_fx16|fx16_fx8]", test_status);
    return test_status;
}

// Tests procedure for mli_hlp_create_subtensor function.
//=======================================================================
struct create_subtensor_test_operands {
    const char* descr;
    tensor_quantizer in;
    tensor_quantizer out;
    mli_sub_tensor_cfg cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

// Checksums of test tensors for various mli calculations mode. 
// When developer finished implementation of kernel and consider it as ok, one needs to populate
// proper checksums for tests in order to highlight any change which affects results.

const crc32_calc    test_1_chksum_fx16  { 0x418F5ED6 }, test_1_chksum_fx8       { 0x0820E5D9 },
                    test_1_chksum_sa8   { 0xBB54537D }, test_1_chksum_sa8_pa    { 0x63BAA2A1 },
                    test_1_chksum_sa32  { 0xDC93E12C }, test_1_chksum_sa32_pa   { 0x98D5A2D6 },
                    test_2_chksum_fx16  { 0xD7B05DED }, test_2_chksum_fx8       { 0x7582D890 },
                    test_2_chksum_sa8   { 0x4CB81C56 }, test_2_chksum_sa8_pa    { 0x2F0B06B8 },
                    test_2_chksum_sa32  { 0x0B379B11 }, test_2_chksum_sa32_pa   { 0x87591FC2 };

const quality_metrics thresholds_test_1_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR_DB = */35.9f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_test_2_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                  /* SNR_DB = */41.3f, quality_metrics::kPassValueQuantErrPerc };

bool run_test_create_subtensor() {
    bool test_status = true;
    const reporter_full reporter;
    static const create_subtensor_test_operands create_subtensor_tests_list[] = {
        {"FX16 tensor ",                    input_1_fx16, test_1_out_fx16, test_1_cfg, thresholds_test_1_general, test_1_chksum_fx16},
        {"FX8 tensor ",                     input_1_fx8, test_1_out_fx8, test_1_cfg, thresholds_test_1_general, test_1_chksum_fx8},
        {"SA8 tensor ",                     input_1_sa8, test_1_out_sa8, test_1_cfg, thresholds_test_1_general, test_1_chksum_sa8},
        {"SA8 tensor per axis",             input_1_sa8_per_axis, test_1_out_sa8_per_axis, test_1_cfg, thresholds_test_1_general, test_1_chksum_sa8_pa},
        {"SA32 tensor ",                    input_1_sa32, test_1_out_sa32, test_1_cfg, thresholds_test_1_general, test_1_chksum_sa32},
        {"SA32 tensor per axis",            input_1_sa32_per_axis, test_1_out_sa32_per_axis, test_1_cfg, thresholds_test_1_general, test_1_chksum_sa32_pa},
        {"FX16 tensor (rank & offset) ",    input_1_fx16, test_2_out_fx16, test_2_cfg, thresholds_test_2_general, test_2_chksum_fx16},
        {"FX8 tensor (rank & offset)",      input_1_fx8, test_2_out_fx8, test_2_cfg, thresholds_test_2_general, test_2_chksum_fx8},
        {"SA8 tensor (rank & offset)",      input_1_sa8, test_2_out_sa8, test_2_cfg, thresholds_test_2_general, test_2_chksum_sa8},
        {"SA8 per axis (rank & offset)",    input_1_sa8_per_axis, test_2_out_sa8_per_axis, test_2_cfg, thresholds_test_2_general, test_2_chksum_sa8_pa},
        {"SA32 tensor (rank & offset)",     input_1_sa32, test_2_out_sa32, test_2_cfg, thresholds_test_2_general, test_2_chksum_sa32},
        {"SA32 per axis (rank & offset)",   input_1_sa32_per_axis, test_2_out_sa32_per_axis, test_2_cfg, thresholds_test_2_general, test_2_chksum_sa32_pa}
    };
    constexpr int kTestsNum = sizeof(create_subtensor_tests_list) / sizeof(create_subtensor_tests_list[0]);

    reporter.report_header("MLI|Helpers|Create Subtensor Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        bool is_test_passed = true;
        const create_subtensor_test_operands* cur_test = &create_subtensor_tests_list[i];
        quality_metrics test_metrics;

        if (!(cur_test->in.is_valid())) {
            is_test_passed = false;
            reporter.report_message(cur_test->descr, "At init: Bad source data for input tensor");
        }

        const mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
        mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
        if (is_test_passed &&
                (tensor_quantizer::validate_tensor(input) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(out) != tensor_quantizer::kOk)) {
            is_test_passed = false;
            reporter.report_message(cur_test->descr,
                "At quantization step: more memory for in or out tensor is be required");
        }

        if (is_test_passed &&
                (mem_in_keeper.is_memory_corrupted() || mem_out_keeper.is_memory_corrupted())) {
            is_test_passed = false;
            reporter.report_message(cur_test->descr,
                "At quantization step: memory beside one of operands is corrupted");
        }

        // Run specific kernel for test

        crc32_calc data_crc_before, data_crc_after;
        if (is_test_passed &&
            mli_hlp_create_subtensor(&input, &cur_test->cfg, &out) != MLI_STATUS_OK) {
            data_crc_before(input);
            reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
            is_test_passed = false;
            data_crc_after(input);
        }

        if (is_test_passed &&
            (data_crc_before.get() != data_crc_after.get() ||
                mem_in_keeper.is_memory_corrupted())) {
            is_test_passed = false;
            reporter.report_message(cur_test->descr,
                "At function run: memory is corrupted after functions invokation");
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
        test_status &= is_test_passed;
    }
    reporter.report_outline("[AUTO] Group: mli_hlp_create_subtensor", test_status);
    return test_status;
}