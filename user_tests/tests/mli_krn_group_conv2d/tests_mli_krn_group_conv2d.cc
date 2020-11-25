/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_api.h"
#include "mli_types.h"
#include "mli_config.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"

#include "test_tensor_quantizer.h"
#include "test_report.h"

#include "vectors_mli_krn_group_conv2d.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

typedef mli_status(*group_conv2d_func_ptr)(
    const mli_tensor* /*input*/,
    const mli_tensor* /*weights*/,
    const mli_tensor* /*bias*/,
    const mli_conv2d_cfg* /*cfg*/,
    mli_tensor* /*output*/);

struct group_conv2d_test_operands {
    const char* descr;
    const group_conv2d_func_ptr mli_krn_group_conv2d;
    tensor_quantizer in;
    tensor_quantizer weights;
    tensor_quantizer bias;
    tensor_quantizer out;
    const mli_conv2d_cfg cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

// Checksums of test tensors for various mli calculations mode. 
// When developer finished implementation of kernel and consider it as ok, He need to populate
// proper checksums for tests in order to highlight any change which affects results.
#if defined(CRC_RM_CONVERGENT) || defined(CRC_RM_UP)

// Shared CRC Results
const crc32_calc test_1_chksum_fx16{ 0xB782413A }, test_1_chksum_fx16_fx8_fx8{ 0xEE4193A9 }, test_1_chksum_sa8{ 0xDC08498E },
                 test_2_chksum_fx16{ 0x65FD03D2 }, test_2_chksum_fx16_fx8_fx8{ 0xB34C4A01 }, test_2_chksum_sa8{ 0x33341D2B },
                 test_3_chksum_fx16{ 0x71B9E656 }, test_3_chksum_fx16_fx8_fx8{ 0x67578BCF }, test_3_chksum_sa8{ 0xAAD727F2 },
                 test_4_chksum_fx16{ 0x2D7FD678 }, test_4_chksum_fx16_fx8_fx8{ 0x96994E5E }, test_4_chksum_sa8{ 0xCC24BECE },
                 test_5_chksum_fx16{ 0xB719F4A0 }, test_5_chksum_fx16_fx8_fx8{ 0x985577E0 }, test_5_chksum_sa8{ 0xBD32CF0A },
                 test_6_chksum_fx16{ 0x3013D414 }, test_6_chksum_fx16_fx8_fx8{ 0xCB153206 }, test_6_chksum_sa8{ 0xC61C378D },
                 test_7_chksum_fx16{ 0x80FF4E78 }, test_7_chksum_fx16_fx8_fx8{ 0x104610B9 }, test_7_chksum_sa8{ 0x50001245 },
                 test_8_chksum_fx16{ 0xAC7AFCAE }, test_8_chksum_fx16_fx8_fx8{ 0x629A3BD0 }, test_8_chksum_sa8{ 0xE97450E4 },
                 test_9_chksum_fx16{ 0x2DDBDF54 }, test_9_chksum_fx16_fx8_fx8{ 0x575372E7 }, test_9_chksum_sa8{ 0x396D8D7A },
                 test_10_chksum_fx16{ 0x78039E74 }, test_10_chksum_fx16_fx8_fx8{ 0x1783EEEC }, test_10_chksum_sa8{ 0x079DC058 };

#else // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16, test_1_chksum_fx16_fx8_fx8, test_1_chksum_sa8,
                  test_2_chksum_fx16, test_2_chksum_fx16_fx8_fx8, test_2_chksum_sa8,
                  test_3_chksum_fx16, test_3_chksum_fx16_fx8_fx8, test_3_chksum_sa8,
                  test_4_chksum_fx16, test_4_chksum_fx16_fx8_fx8, test_4_chksum_sa8,
                  test_5_chksum_fx16, test_5_chksum_fx16_fx8_fx8, test_5_chksum_sa8,
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

const quality_metrics thresholds_fx16_fx8_fx8_test4{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR_DB = */26.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                             /* SNR_DB = */30.f, quality_metrics::kPassValueQuantErrPerc };


static const group_conv2d_test_operands tests_list[] = {
    // Basic functionality test: groups=3, kernel_size=(3, 4), strides=(1, 2), krn_padding, w/o ReLU
    {"Test 1 FX16",         mli_krn_group_conv2d_hwcn_fx16, 
                            input_1_fx16, weights_1_fx16, bias_1_fx16, test_1_out_fx16, test_1_cfg,
                            thresholds_fx16_general, test_1_chksum_fx16},
    {"Test 1 FX16_FX8_FX8", mli_krn_group_conv2d_hwcn_fx16_fx8_fx8, 
                            input_1_fx16, weights_1_fx8, bias_1_fx8, test_1_out_fx16, test_1_cfg, 
                            thresholds_fx16_fx8_fx8_general, test_1_chksum_fx16_fx8_fx8},
    {"Test 1 SA8_SA8_SA32", mli_krn_group_conv2d_hwcn_sa8_sa8_sa32,
                            input_1_sa8, weights_1_sa8, bias_1_sa32, test_1_out_sa8, test_1_cfg, 
                            thresholds_sa8_general, test_1_chksum_sa8},

    // Basic functionality test: groups=3, kernel_size=(4, 3), strides=(1, 2), krn_padding, Gen_ReLU
    {"Test 2 FX16 ReluGen",         mli_krn_group_conv2d_hwcn_fx16, 
                                    input_1_fx16, weights_2_fx16, bias_2_fx16, test_2_out_fx16, test_2_cfg, 
                                    thresholds_fx16_general, test_2_chksum_fx16},
    {"Test 2 FX16_FX8_FX8 ReluGen", mli_krn_group_conv2d_hwcn_fx16_fx8_fx8,
                                    input_1_fx16, weights_2_fx8, bias_2_fx8, test_2_out_fx16, test_2_cfg,
                                    thresholds_fx16_fx8_fx8_general, test_2_chksum_fx16_fx8_fx8},
    {"Test 2 SA8_SA8_SA32 ReluGen", mli_krn_group_conv2d_hwcn_sa8_sa8_sa32, 
                                    input_1_sa8, weights_2_sa8, bias_2_i1_w2_sa32, test_2_out_sa8, test_2_cfg,
                                    thresholds_sa8_general, test_2_chksum_sa8},

    // Multiple filters per group test: groups=3, fpg=2, kernel_size=(4, 4), strides=(1, 1), w/o padding, w/o ReLU
    {"Test 3 FX16 Mult FPG",         mli_krn_group_conv2d_hwcn_fx16,
                                     input_2_fx16, weights_3_fx16, bias_3_fx16, test_3_out_fx16, test_3_cfg,
                                     thresholds_fx16_general, test_3_chksum_fx16},
    {"Test 3 FX16_FX8_FX8 Mult FPG", mli_krn_group_conv2d_hwcn_fx16_fx8_fx8,
                                     input_2_fx16, weights_3_fx8, bias_3_fx8, test_3_out_fx16, test_3_cfg,
                                     thresholds_fx16_fx8_fx8_general, test_3_chksum_fx16_fx8_fx8},
    {"Test 3 SA8_SA8_SA32 Mult FPG", mli_krn_group_conv2d_hwcn_sa8_sa8_sa32,
                                     input_2_sa8, weights_3_sa8, bias_3_i2_w3_sa32, test_3_out_sa8, test_3_cfg,
                                     thresholds_sa8_general, test_3_chksum_sa8},

    // Depthwise kernel call test with dilation: groups=9, kernel_size=(3, 3), strides=(1, 1), dilation=(2, 2),
    // krn_padding, Gen_ReLU
    {"Test 4 FX16 DW call",         mli_krn_group_conv2d_hwcn_fx16,
                                    input_1_fx16, weights_4_fx16, bias_4_fx16, test_4_out_fx16, test_4_cfg,
                                    thresholds_fx16_general, test_4_chksum_fx16},
    {"Test 4 FX16_FX8_FX8 DW call", mli_krn_group_conv2d_hwcn_fx16_fx8_fx8,
                                    input_1_fx16, weights_4_fx8, bias_4_fx8, test_4_out_fx16, test_4_cfg,
                                    thresholds_fx16_fx8_fx8_test4, test_4_chksum_fx16_fx8_fx8},
    {"Test 4 SA8_SA8_SA32 DW call", mli_krn_group_conv2d_hwcn_sa8_sa8_sa32,
                                    input_1_sa8, weights_4_sa8, bias_4_i1_w4_sa32, test_4_out_sa8, test_4_cfg,
                                    thresholds_sa8_general, test_4_chksum_sa8},

    // Conv2D kernel call test with dilation: groups=1, kernel_size=(2, 2), strides=(2, 1), dilation=(2, 2),
    // krn_padding, w/o ReLU
    {"Test 5 FX16 Conv2D call",       mli_krn_group_conv2d_hwcn_fx16, 
                                      input_1_fx16, weights_5_fx16, bias_5_fx16, test_5_out_fx16, test_5_cfg,
                                      thresholds_fx16_general, test_5_chksum_fx16},
    {"Test 5 FX16_FX8_FX8 Conv call", mli_krn_group_conv2d_hwcn_fx16_fx8_fx8, 
                                      input_1_fx16, weights_5_fx8, bias_5_fx8, test_5_out_fx16, test_5_cfg,
                                      thresholds_fx16_fx8_fx8_general, test_5_chksum_fx16_fx8_fx8},
    {"Test 5 SA8_SA8_SA32 Conv call", mli_krn_group_conv2d_hwcn_sa8_sa8_sa32,
                                      input_1_sa8, weights_5_sa8, bias_5_i1_w5_sa32, test_5_out_sa8, test_5_cfg,
                                      thresholds_sa8_general, test_5_chksum_sa8},

    // Input/output Memstride test: groups=3, kernel_size=(3, 4), strides=(3, 3), krn_padding, ReLU_6
    {"Test 6 FX16 IO_Memstr",         mli_krn_group_conv2d_hwcn_fx16,
                                      input_1_memstr_fx16, weights_1_fx16, bias_6_fx16, test_6_out_fx16, test_6_cfg,
                                      thresholds_fx16_general, test_6_chksum_fx16},
    {"Test 6 FX16_FX8_FX8 IO_Memstr", mli_krn_group_conv2d_hwcn_fx16_fx8_fx8,
                                      input_1_memstr_fx16, weights_1_fx8, bias_6_fx8, test_6_out_fx16,
                                      test_6_cfg, thresholds_fx16_fx8_fx8_general, test_6_chksum_fx16_fx8_fx8},
    {"Test 6 SA8_SA8_SA32 IO_Memstr", mli_krn_group_conv2d_hwcn_sa8_sa8_sa32,
                                      input_1_memstr_sa8, weights_1_sa8, bias_6_i1_w1_sa32, test_6_out_sa8,
                                      test_6_cfg, thresholds_sa8_general, test_6_chksum_sa8},

    // Weights Memstride test: groups=3, kernel_size=(4, 3), strides=(3, 3), krn_padding, Gen_ReLU
    {"Test 7 FX16 W_Memstr",         mli_krn_group_conv2d_hwcn_fx16,
                                     input_1_fx16, weights_2_memstr_fx16, bias_7_fx16, test_7_out_fx16, test_7_cfg,
                                     thresholds_fx16_general, test_7_chksum_fx16},
    {"Test 7 FX16_FX8_FX8 W_Memstr", mli_krn_group_conv2d_hwcn_fx16_fx8_fx8,
                                     input_1_fx16, weights_2_memstr_fx8, bias_7_fx8, test_7_out_fx16,
                                     test_7_cfg, thresholds_fx16_fx8_fx8_general, test_7_chksum_fx16_fx8_fx8},
    {"Test 7 SA8_SA8_SA32 W_Memstr", mli_krn_group_conv2d_hwcn_sa8_sa8_sa32, 
                                     input_1_sa8, weights_2_memstr_sa8, bias_7_i1_w2_sa32, test_7_out_sa8, test_7_cfg,
                                     thresholds_sa8_general, test_7_chksum_sa8},

    // Multiple filters per group test with dilation: groups=3, fpg=2, kernel_size=(2, 2), strides=(1, 1), 
    // dilation=(2, 2), krn_padding, ReLU_1
    {"Test 8 FX16 FPG+Dil",         mli_krn_group_conv2d_hwcn_fx16,
                                    input_2_fx16, weights_6_fx16, bias_8_fx16, test_8_out_fx16, test_8_cfg,
                                    thresholds_fx16_general, test_8_chksum_fx16},
    {"Test 8 FX16_FX8_FX8 FPG+Dil", mli_krn_group_conv2d_hwcn_fx16_fx8_fx8,
                                    input_2_fx16, weights_6_fx8, bias_8_fx8, test_8_out_fx16,
                                    test_8_cfg, thresholds_fx16_fx8_fx8_general, test_8_chksum_fx16_fx8_fx8},
    {"Test 8 SA8_SA8_SA32 FPG+Dil", mli_krn_group_conv2d_hwcn_sa8_sa8_sa32, 
                                    input_2_sa8, weights_6_sa8, bias_8_i2_w6_sa32, test_8_out_sa8, test_8_cfg,
                                    thresholds_sa8_general, test_8_chksum_sa8},
    
    // k3x3 specialization test with in/out/w memstride and dilation: groups=3, kernel_size=(3, 3), strides=(1, 1), 
    // dilation=(2, 2), krn_padding, Gen_ReLU
    {"Test 9 FX16 k3x3 Mstr+Dil",     mli_krn_group_conv2d_hwcn_fx16_k3x3,
                                      input_1_memstr_fx16, weights_7_memstr_fx16, bias_9_fx16, test_9_out_fx16, 
                                      test_9_cfg, thresholds_fx16_general, test_9_chksum_fx16},
    {"Test 9 FX16_FX8 k3x3 Mstr+Dil", mli_krn_group_conv2d_hwcn_fx16_fx8_fx8_k3x3,
                                      input_1_memstr_fx16, weights_7_memstr_fx8, bias_9_fx8, test_9_out_fx16,
                                      test_9_cfg, thresholds_fx16_fx8_fx8_general, test_9_chksum_fx16_fx8_fx8},
    {"Test 9 SA8 k3x3 Mstr+Dil",      mli_krn_group_conv2d_hwcn_sa8_sa8_sa32_k3x3, 
                                      input_1_memstr_sa8, weights_7_memstr_sa8, bias_9_i1_w7_sa32, 
                                      test_9_out_sa8, test_9_cfg, thresholds_sa8_general, test_9_chksum_sa8},

    // k5x5 specialization test with in/out/w memstride and dilation: groups=3, kernel_size=(5, 5), strides=(1, 1), 
    // dilation=(2, 2), krn_padding, w/o ReLU
    {"Test 10 FX16 k5x5 Mstr+Dil",     mli_krn_group_conv2d_hwcn_fx16_k5x5,
                                       input_1_memstr_fx16, weights_8_memstr_fx16, bias_10_fx16, test_10_out_fx16, 
                                       test_10_cfg, thresholds_fx16_general, test_10_chksum_fx16},
    {"Test 10 FX16_FX8 k5x5 Mstr+Dil", mli_krn_group_conv2d_hwcn_fx16_fx8_fx8_k5x5,
                                       input_1_memstr_fx16, weights_8_memstr_fx8, bias_10_fx8, test_10_out_fx16,
                                       test_10_cfg, thresholds_fx16_fx8_fx8_general, test_10_chksum_fx16_fx8_fx8},
    {"Test 10 SA8 k5x5 Mstr+Dil",      mli_krn_group_conv2d_hwcn_sa8_sa8_sa32_k5x5, 
                                       input_1_memstr_sa8, weights_8_memstr_sa8, bias_10_i1_w8_sa32, 
                                       test_10_out_sa8, test_10_cfg, thresholds_sa8_general, test_10_chksum_sa8},
};

constexpr int kMemSize = 2247;
static IO_DATA_ATTR int8_t scratch_mem_in[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemSize] = { 0 };
static W_DATA_ATTR int8_t scratch_mem_w[kMemSize] = { 0 };
static W_DATA_ATTR int8_t scratch_mem_b[kMemSize] = { 0 };

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Kernels|Group Convolution 2D Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        memory_manager mem_w_keeper((int8_t*)(scratch_mem_w), sizeof(scratch_mem_w));
        memory_manager mem_b_keeper((int8_t*)(scratch_mem_b), sizeof(scratch_mem_b));
        bool is_test_passed = true;
        const group_conv2d_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metics;

        if (!(cur_test->in.is_valid() && cur_test->weights.is_valid() &&
                cur_test->bias.is_valid() && cur_test->out.is_valid())) {
            reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
            is_test_passed = false;
        }

        mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
        mli_tensor weights = cur_test->weights.get_quantized_tensor(mem_w_keeper.allocate_memory(cur_test->weights));
        mli_tensor bias = cur_test->bias.get_quantized_tensor(mem_b_keeper.allocate_memory(cur_test->bias));
        mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
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
                cur_test->mli_krn_group_conv2d(&input, &weights, &bias, &cur_test->cfg, &out) != MLI_STATUS_OK) {
            reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
            is_test_passed = false;
        }

        if (is_test_passed &&
                (mem_in_keeper.is_memory_corrupted() || mem_out_keeper.is_memory_corrupted() ||
                mem_w_keeper.is_memory_corrupted() || mem_b_keeper.is_memory_corrupted())) {
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
            data_crc(weights);
            data_crc(bias);
            data_crc(out);

            is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr, test_metics, cur_test->threshold, 
                                                                data_crc, cur_test->check_sum);
        }
        final_status &= is_test_passed;
    }

    reporter.report_outline("[AUTO] Group: mli_krn_group_conv2d", final_status);

    return (final_status) ? 0 : 1;
}
