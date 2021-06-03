/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_api.h"
#include "mli_config.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "mli_types.h"
#include "test_tensor_quantizer.h"
#include "test_report.h"

#include "vectors_mli_krn_depthwise_conv.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

typedef mli_status(*depthwise_conv_func_ptr)(
    const mli_tensor* /*input*/,
    const mli_tensor* /*weights*/,
    const mli_tensor* /*bias*/,
    const mli_conv2d_cfg* /*cfg*/,
    mli_tensor* /*output*/);

struct depthwise_conv_test_operands {
    const char* descr;
    const depthwise_conv_func_ptr mli_krn_depthwise_conv;
    tensor_quantizer in;
    tensor_quantizer weights;
    tensor_quantizer bias;
    tensor_quantizer out;
    const mli_conv2d_cfg cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

// Checksums of test tensors for various mli calculations mode. 
// When developer finished implementation of kernel and consider it as ok, one needs to populate
// proper checksums for tests in order to highlight any change which affects results.
#if defined(CRC_RM_CONVERGENT) || defined(CRC_RM_UP)
const crc32_calc test_1_chksum_fx16{ 0x968FB503 }, 
                 test_2_chksum_fx16{ 0xC2732A90 }, test_2_chksum_fx16_fx8_fx8{ 0x2BAB6B9C }, test_2_chksum_sa8{ 0x9EF7413F },
                 test_3_chksum_fx16{ 0x5C02F273 }, test_3_chksum_fx16_fx8_fx8{ 0x41941500 }, test_3_chksum_sa8{ 0xB3A2C34D },
                 test_4_chksum_fx16{ 0xFBE49F90 }, test_4_chksum_fx16_fx8_fx8{ 0xA37AA7C1 }, test_4_chksum_sa8{ 0x24235738 },
                 test_5_chksum_fx16{ 0xFAD19A42 }, test_5_chksum_fx16_fx8_fx8{ 0x70C8D5DD }, test_5_chksum_sa8{ 0x8DAAACEC },
                 test_6_chksum_fx16{ 0xF03253BB }, test_6_chksum_fx16_fx8_fx8{ 0x122832FF }, test_6_chksum_sa8{ 0x42F80E2D },
                 test_7_chksum_fx16{ 0xDC4EBBE7 }, test_7_chksum_fx16_fx8_fx8{ 0x10447ABF }, test_7_chksum_sa8{ 0xBC318965 },
                 test_8_chksum_fx16{ 0x4D6EFB91 },                                           test_8_chksum_sa8{ 0xF888FAB3 },
                 test_9_chksum_fx16{ 0xBF6D526A }, test_9_chksum_fx16_fx8_fx8{ 0xFB8CEA65 }, test_9_chksum_sa8{ 0xE0165E99 },
                 test_10_chksum_fx16{ 0xC02567E8 }, test_10_chksum_fx16_fx8_fx8{ 0x882F41DB }, test_10_chksum_sa8{ 0x390F7E80 };
// Platform Specific CRC Results
#if defined(CRC_RM_UP)
const crc32_calc test_1_chksum_fx16_fx8_fx8{ 0x46A964A7 }, test_1_chksum_sa8{ 0x366725E9 },
                 test_8_chksum_fx16_fx8_fx8{ 0xB73234B8 };

#else 
const crc32_calc test_1_chksum_fx16_fx8_fx8{ 0xA6CC6260 }, test_1_chksum_sa8{ 0x46D36646 },
                 test_8_chksum_fx16_fx8_fx8{ 0x55734EF0 };
#endif
#else // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16, test_1_chksum_fx16_fx8_fx8, test_1_chksum_sa8,
                  test_2_chksum_fx16, test_2_chksum_fx16_fx8_fx8, test_2_chksum_sa8,
                  test_3_chksum_fx16, test_3_chksum_fx16_fx8_fx8, test_3_chksum_sa8,
                  test_4_chksum_fx16, test_4_chksum_fx16_fx8_fx8, test_4_chksum_sa8,
                  test_5_chksum_fx16, test_5_chksum_fx16_fx8_fx8, test_5_chksum_sa8;
                  test_6_chksum_fx16, test_6_chksum_fx16_fx8_fx8, test_6_chksum_sa8,
                  test_7_chksum_fx16, test_7_chksum_fx16_fx8_fx8, test_7_chksum_sa8,
                  test_8_chksum_fx16, test_8_chksum_fx16_fx8_fx8, test_8_chksum_sa8,
                  test_9_chksum_fx16, test_9_chksum_fx16_fx8_fx8, test_9_chksum_sa8,
                  test_10_chksum_fx16, test_10_chksum_fx16_fx8_fx8, test_10_chksum_sa8;
#endif


const quality_metrics thresholds_fx16_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR_DB = */70.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_fx16_fx8_fx8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                  /* SNR_DB = */30.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                             /* SNR_DB = */35.f, quality_metrics::kPassValueQuantErrPerc };


static const depthwise_conv_test_operands tests_list[] = {
    // Basic functionality test: kernel_size=(3, 4), strides=(1, 1), with krn_padding, w/o ReLU
    {"Test 1 FX16 ",         mli_krn_depthwise_conv2d_hwcn_fx16, 
                            input_1_fx16, weights_1_fx16, bias_1_fx16, test_1_out_fx16, test_1_cfg,
                            thresholds_fx16_general, test_1_chksum_fx16},
    {"Test 1 FX16_FX8_FX8", mli_krn_depthwise_conv2d_hwcn_fx16_fx8_fx8, 
                            input_1_fx16, weights_1_fx8, bias_1_fx8, test_1_out_fx16, test_1_cfg, 
                            thresholds_fx16_fx8_fx8_general, test_1_chksum_fx16_fx8_fx8},
    {"Test 1 SA8_SA8_SA32", mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32,
                            input_1_sa8, weights_1_sa8_per_axis, bias_1_sa32_per_axis, test_1_out_sa8, test_1_cfg, 
                            thresholds_sa8_general, test_1_chksum_sa8},

    // Basic functionality test: kernel_size=(4, 3), strides=(2, 2), with krn_padding, with Gen_ReLU
    {"Test 2 FX16 ReluGen",         mli_krn_depthwise_conv2d_hwcn_fx16, 
                                    input_1_fx16, weights_2_fx16, bias_2_fx16, test_2_out_fx16, test_2_cfg, 
                                    thresholds_fx16_general, test_2_chksum_fx16},
    {"Test 2 FX16_FX8_FX8 ReluGen", mli_krn_depthwise_conv2d_hwcn_fx16_fx8_fx8,
                                    input_1_fx16, weights_2_fx8, bias_2_fx8, test_2_out_fx16, test_2_cfg,
                                    thresholds_fx16_fx8_fx8_general, test_2_chksum_fx16_fx8_fx8},
    {"Test 2 SA8_SA8_SA32 ReluGen", mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32, 
                                    input_1_sa8, weights_2_sa8, bias_2_i1_w2_sa32, test_2_out_sa8, 
                                    test_2_cfg, thresholds_sa8_general, test_2_chksum_sa8},

    // Dilation test: kernel_size=(3, 4), strides=(1, 1), w/o padding, w/o ReLU
    {"Test 3 FX16 Dil",         mli_krn_depthwise_conv2d_hwcn_fx16,
                                input_1_fx16, weights_1_fx16, bias_1_fx16, test_3_out_fx16, test_3_cfg,
                                thresholds_fx16_general, test_3_chksum_fx16},
    {"Test 3 FX16_FX8_FX8 Dil", mli_krn_depthwise_conv2d_hwcn_fx16_fx8_fx8,
                                input_1_fx16, weights_1_fx8, bias_1_fx8, test_3_out_fx16, test_3_cfg,
                                thresholds_fx16_fx8_fx8_general, test_3_chksum_fx16_fx8_fx8},
    {"Test 3 SA8_SA8_SA32 Dil", mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32,
                                input_1_sa8, weights_1_sa8_per_axis, bias_1_sa32_per_axis, test_3_out_sa8, test_3_cfg,
                                thresholds_sa8_general, test_3_chksum_sa8},

    // Input/output memstride test: kernel_size=(3, 4), strides=(3, 3), w/o padding, with ReLU_1 
    {"Test 4 FX16 Relu1 Mstr",         mli_krn_depthwise_conv2d_hwcn_fx16,
                                       input_1b_memstr_fx16, weights_1_fx16, bias_1_fx16, test_4_out_fx16, test_4_cfg,
                                       thresholds_fx16_general, test_4_chksum_fx16},
    {"Test 4 FX16_FX8_FX8 Relu1 Mstr", mli_krn_depthwise_conv2d_hwcn_fx16_fx8_fx8,
                                       input_1b_memstr_fx16, weights_1_fx8, bias_1_fx8, test_4_out_fx16, test_4_cfg,
                                       thresholds_fx16_fx8_fx8_general, test_4_chksum_fx16_fx8_fx8},
    {"Test 4 SA8_SA8_SA32 Relu1 Mstr", mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32,
                                       input_1b_memstr_sa8, weights_1_sa8_per_axis, bias_1_sa32_per_axis, test_4_out_sa8, 
                                       test_4_cfg, thresholds_sa8_general, test_4_chksum_sa8},

    // Weights memstride test: kernel_size=(8, 6), strides=(1, 1), w/o padding, with ReLU_6
    {"Test 5 FX16 Relu6 Mstr",         mli_krn_depthwise_conv2d_hwcn_fx16,
                                       input_1_fx16, weights_3_fx16, bias_2_fx16, test_5_out_fx16, test_5_cfg,
                                       thresholds_fx16_general, test_5_chksum_fx16},
    {"Test 5 FX16_FX8_FX8 Relu6 Mstr", mli_krn_depthwise_conv2d_hwcn_fx16_fx8_fx8,
                                       input_1_fx16, weights_3_fx8, bias_2_fx8, test_5_out_fx16, test_5_cfg,
                                       thresholds_fx16_fx8_fx8_general, test_5_chksum_fx16_fx8_fx8},
    {"Test 5 SA8_SA8_SA32 Relu6 Mstr", mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32,
                                       input_1_sa8, weights_3_sa8_per_axis, bias_2_i1_w3_sa32_per_axis, test_5_out_sa8, 
                                       test_5_cfg, thresholds_sa8_general, test_5_chksum_sa8},

    // k3x3 specialization test with memstride, strides=(2, 2), krn_padding and ReLU 6
    // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 6 FX16 k3x3 Spec",         mli_krn_depthwise_conv2d_hwcn_fx16_k3x3,
                                       input_1_memstr_fx16, weights_3_fx16, bias_2_fx16, test_6_out_fx16, test_6_cfg,
                                       thresholds_fx16_general, test_6_chksum_fx16},
    {"Test 6 FX16_FX8_FX8 k3x3 Spec", mli_krn_depthwise_conv2d_hwcn_fx16_fx8_fx8_k3x3,
                                       input_1_memstr_fx16, weights_3_fx8, bias_2_fx8, test_6_out_fx16, test_6_cfg,
                                       thresholds_fx16_fx8_fx8_general, test_6_chksum_fx16_fx8_fx8},
    {"Test 6 SA8_SA8_SA32 k3x3 Spec", mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32_k3x3,
                                       input_1_memstr_sa8, weights_3_sa8_per_axis, bias_2_i1_w3_sa32_per_axis, test_6_out_sa8, 
                                       test_6_cfg, thresholds_sa8_general, test_6_chksum_sa8},

    // k5x5 specialization test with memstride, strides=(2, 2), krn_padding and ReLU 6
    // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 7 FX16 k5x5 Spec",         mli_krn_depthwise_conv2d_hwcn_fx16_k5x5,
                                       input_1_memstr_fx16, weights_4_fx16, bias_2_fx16, test_7_out_fx16, test_7_cfg,
                                       thresholds_fx16_general, test_7_chksum_fx16},
    {"Test 7 FX16_FX8_FX8 k5x5 Spec", mli_krn_depthwise_conv2d_hwcn_fx16_fx8_fx8_k5x5,
                                       input_1_memstr_fx16, weights_4_fx8, bias_2_fx8, test_7_out_fx16, test_7_cfg,
                                       thresholds_fx16_fx8_fx8_general, test_7_chksum_fx16_fx8_fx8},
    {"Test 7 SA8_SA8_SA32 k5x5 Spec", mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32_k5x5,
                                       input_1_memstr_sa8, weights_4_sa8_per_axis, bias_2_i1_w4_sa32_per_axis, test_7_out_sa8, 
                                       test_7_cfg, thresholds_sa8_general, test_7_chksum_sa8},

    // Dilation test with padding for generic function, kernel_size=(3, 3), strides=(1, 1),
    // krn_padding , dilation = (2,2) and ReLU_Gen.
    // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 8-1 FX16 Dil+Pad",         mli_krn_depthwise_conv2d_hwcn_fx16,
                                       input_1_memstr_fx16, weights_3_fx16, bias_2_fx16, test_8_out_fx16, test_8_cfg,
                                       thresholds_fx16_general, test_8_chksum_fx16},
    {"Test 8-1 FX16_FX8_FX8 Dil+Pad", mli_krn_depthwise_conv2d_hwcn_fx16_fx8_fx8,
                                       input_1_memstr_fx16, weights_3_fx8, bias_2_fx8, test_8_out_fx16, test_8_cfg,
                                       thresholds_fx16_fx8_fx8_general, test_8_chksum_fx16_fx8_fx8},
    {"Test 8-1 SA8_SA8_SA32 Dil+Pad", mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32,
                                       input_1_memstr_sa8, weights_3_sa8_per_axis, bias_2_i1_w3_sa32_per_axis, test_8_out_sa8, 
                                       test_8_cfg, thresholds_sa8_general, test_8_chksum_sa8},

    // Dilation test for k3x3 specialization test, strides=(1, 1), krn_padding , dilation = (2,2) and ReLU_1.
    // Memstrides are applied on input, output and weights tensors
    {"Test 8-2 FX16 k3x3 Dil",         mli_krn_depthwise_conv2d_hwcn_fx16_k3x3,
                                       input_1_memstr_fx16, weights_3_fx16, bias_2_fx16, test_8_out_fx16, test_8_cfg,
                                       thresholds_fx16_general, test_8_chksum_fx16},
    {"Test 8-2 FX16_FX8_FX8 k3x3 Dil", mli_krn_depthwise_conv2d_hwcn_fx16_fx8_fx8_k3x3,
                                       input_1_memstr_fx16, weights_3_fx8, bias_2_fx8, test_8_out_fx16, test_8_cfg,
                                       thresholds_fx16_fx8_fx8_general, test_8_chksum_fx16_fx8_fx8},
    {"Test 8-2 SA8_SA8_SA32 k3x3 Dil", mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32_k3x3,
                                       input_1_memstr_sa8, weights_3_sa8_per_axis, bias_2_i1_w3_sa32_per_axis, test_8_out_sa8, 
                                       test_8_cfg, thresholds_sa8_general, test_8_chksum_sa8},

    // Dilation test for k5x5 specialization test, strides=(1, 1), krn_padding , dilation = (2,2) and ReLU_Gen.
    // Memstrides are applied on input, output and weights tensors
    {"Test 9 FX16 k5x5 Dil",         mli_krn_depthwise_conv2d_hwcn_fx16_k5x5,
                                       input_1_memstr_fx16, weights_4_fx16, bias_2_fx16, test_9_out_fx16, test_9_cfg,
                                       thresholds_fx16_general, test_9_chksum_fx16},
    {"Test 9 FX16_FX8_FX8 k5x5 Dil", mli_krn_depthwise_conv2d_hwcn_fx16_fx8_fx8_k5x5,
                                       input_1_memstr_fx16, weights_4_fx8, bias_2_fx8, test_9_out_fx16, test_9_cfg,
                                       thresholds_fx16_fx8_fx8_general, test_9_chksum_fx16_fx8_fx8},
    {"Test 9 SA8_SA8_SA32 k5x5 Dil", mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32_k5x5,
                                       input_1_memstr_sa8, weights_4_sa8_per_axis, bias_2_i1_w4_sa32_per_axis, test_9_out_sa8, 
                                       test_9_cfg, thresholds_sa8_general, test_9_chksum_sa8},

    // Test with huge values in operands to check negative fractional and big scales 
    {"Test 10 FX16 Huge Vals",         mli_krn_depthwise_conv2d_hwcn_fx16,
                                       input_2_fx16, weights_5_fx16, bias_3_fx16, test_10_out_fx16, test_10_cfg,
                                       thresholds_fx16_general, test_10_chksum_fx16},
    {"Test 10 FX16_FX8_FX8 Huge Vals", mli_krn_depthwise_conv2d_hwcn_fx16_fx8_fx8,
                                       input_2_fx16, weights_5_fx8, bias_3_fx8, test_10_out_fx16, test_10_cfg,
                                       thresholds_fx16_fx8_fx8_general, test_10_chksum_fx16_fx8_fx8},
    {"Test 10 SA8_SA8_SA32 Huge Vals", mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32,
                                       input_2_sa8, weights_5_sa8, bias_3_i2_w5_sa32, test_10_out_sa8, 
                                       test_10_cfg, thresholds_sa8_general, test_10_chksum_sa8},

};

constexpr int kMemSize = 2047;
static IO_DATA_ATTR int8_t scratch_mem_in[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemSize] = { 0 };
static W_DATA_ATTR int8_t scratch_mem_w[kMemSize] = { 0 };
static W_DATA_ATTR int8_t scratch_mem_b[kMemSize] = { 0 };

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Kernels|Depthwise Conv Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        memory_manager mem_w_keeper((int8_t*)(scratch_mem_w), sizeof(scratch_mem_w));
        memory_manager mem_b_keeper((int8_t*)(scratch_mem_b), sizeof(scratch_mem_b));
        bool is_test_passed = true;
        const depthwise_conv_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metrics;

#if __Xvec_guard_bit_option == 0 && defined(__Xvec_guard_bit_option)
        if (strstr(cur_test->descr, "Test 1 SA8_SA8_SA32") != nullptr ||
                strstr(cur_test->descr, "Test 2 SA8_SA8_SA32 ReluGen") != nullptr ||
                strstr(cur_test->descr, "Test 3 SA8_SA8_SA32 Dil") != nullptr ||
                strstr(cur_test->descr, "Test 4 SA8_SA8_SA32 Relu1 Mstr") != nullptr ||
                strstr(cur_test->descr, "Test 5 SA8_SA8_SA32 Relu6 Mstr") != nullptr ||
                strstr(cur_test->descr, "Test 6 SA8_SA8_SA32 k3x3 Spec") != nullptr ||
                strstr(cur_test->descr, "Test 7 SA8_SA8_SA32") != nullptr ||
                strstr(cur_test->descr, "Test 8-1 SA8") != nullptr ||
                strstr(cur_test->descr, "Test 8-2 SA8") != nullptr ||
                strstr(cur_test->descr, "Test 9 SA8_SA8_SA32 k5x5 Dil") != nullptr ||
                strstr(cur_test->descr, "Test 10 SA8_SA8_SA32 Huge Vals") != nullptr) {
            // VPX fails bitwise comparison with reference .
            reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
            continue;
        }
#endif
#if PLATFORM == V2DSP_XY && defined(CRC_RM_UP)
        if (strstr(cur_test->descr, "Test 1 SA8_SA8_SA32") != nullptr) {
            // Em9d fails comparison with reference in up rounding mode.
            reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
            continue;
        }
#endif

        if (!(cur_test->in.is_valid() && cur_test->weights.is_valid() &&
                cur_test->bias.is_valid() && cur_test->out.is_valid())) {
            reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
            is_test_passed = false;
        }

        mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
        mli_tensor weights = cur_test->weights.get_quantized_tensor(mem_w_keeper.allocate_memory(cur_test->weights));
        mli_tensor bias = cur_test->bias.get_quantized_tensor(mem_b_keeper.allocate_memory(cur_test->bias));
        mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
        mli_tensor source_out_tensor = out;
        if (is_test_passed &&
                (tensor_quantizer::validate_tensor(input) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(weights) != tensor_quantizer::kOk||
                 tensor_quantizer::validate_tensor(bias) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(out) != tensor_quantizer::kOk)) {
            reporter.report_message(cur_test->descr, 
                                    "FAILED at quantization step: more memory for one of tensors might be required");
            is_test_passed = false;
        }

        if (is_test_passed &&
                (mem_in_keeper.is_memory_corrupted() || mem_out_keeper.is_memory_corrupted() ||
                mem_w_keeper.is_memory_corrupted() || mem_b_keeper.is_memory_corrupted())) {
            reporter.report_message(cur_test->descr,
                "FAILED at quantization step: memory beside one of operands is corrupted");
            is_test_passed = false;
        }

        // Run specific kernel for test
        if (is_test_passed &&
                cur_test->mli_krn_depthwise_conv(&input, &weights, &bias, &cur_test->cfg, &out) != MLI_STATUS_OK) {
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
            data_crc(weights);
            data_crc(bias);
            data_crc(out);

            is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr, test_metrics, cur_test->threshold, 
                                                                data_crc, cur_test->check_sum);
        }
        final_status &= is_test_passed;
    }

    reporter.report_outline("[AUTO] Group: mli_krn_depthwise_conv", final_status);

    return (final_status) ? 0 : 1;
}
