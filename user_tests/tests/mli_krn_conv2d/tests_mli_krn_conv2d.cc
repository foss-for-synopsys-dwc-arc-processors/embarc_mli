/*
* Copyright 2019-2020, Synopsys, Inc.
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

#include "vectors_mli_krn_conv2d.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

typedef mli_status(*conv2d_func_ptr)(
    const mli_tensor* /*input*/,
    const mli_tensor* /*weights*/,
    const mli_tensor* /*bias*/,
    const mli_conv2d_cfg* /*cfg*/,
    mli_tensor* /*output*/);

struct conv2d_test_operands {
    const char* descr;
    const conv2d_func_ptr mli_krn_conv2d;
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
const crc32_calc test_1_chksum_fx16{ 0x3669E8DA }, test_1_chksum_fx16_fx8_fx8{ 0x627FD168 }, test_1_chksum_sa8{ 0xA3FFD976 },
                 test_2_chksum_fx16{ 0x6075722F }, test_2_chksum_fx16_fx8_fx8{ 0xBFE5DC3D }, test_2_chksum_sa8{ 0x5D288208 },
                 test_3_chksum_fx16{ 0xE2100158 }, test_3_chksum_fx16_fx8_fx8{ 0x550F135E }, test_3_chksum_sa8{ 0x9740102D },
                 test_4_chksum_fx16{ 0x5F13DD22 }, test_4_chksum_fx16_fx8_fx8{ 0xF92F0A5A }, test_4_chksum_sa8{ 0x45AA03B7 },
                 test_5_chksum_fx16{ 0xD8CA1273 }, test_5_chksum_fx16_fx8_fx8{ 0x186AA252 }, test_5_chksum_sa8{ 0x01D390FA },
                 test_6_chksum_fx16{ 0x5A7A92BB }, 
                 test_7_chksum_fx16{ 0x6EA7C12C }, test_7_chksum_fx16_fx8_fx8{ 0x0EC04486 }, test_7_chksum_sa8{ 0xEC3B6B91 },
                 test_8_chksum_fx16{ 0x2EE8436B }, test_8_chksum_fx16_fx8_fx8{ 0xA038E01B }, test_8_chksum_sa8{ 0xF0CC7CA5 },
                 test_9_chksum_fx16, test_9_chksum_fx16_fx8_fx8, test_9_chksum_sa8,
                 test_10_chksum_fx16, test_10_chksum_fx16_fx8_fx8, test_10_chksum_sa8;
// Platform Specific CRC Results
#if defined(CRC_RM_UP)
const crc32_calc test_6_chksum_fx16_fx8_fx8{ 0xB158A585 }, test_6_chksum_sa8{ 0x944BF386 };
#else 
const crc32_calc test_6_chksum_fx16_fx8_fx8{ 0x798BEF7B }, test_6_chksum_sa8{ 0x2354FD30 };
#endif
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
                                             /* SNR_DB = */35.f, /*Quant Error Perc = */40.f };


static const conv2d_test_operands tests_list[] = {
    // Basic functionality test kernel_size=(3, 4), strides=(1, 1), with krn_padding and w/o ReLU
    {"Test 1 FX16",         mli_krn_conv2d_hwcn_fx16, 
                            input_1_fx16, weights_1_fx16, bias_1_fx16, test_1_out_fx16, test_1_cfg,
                            thresholds_fx16_general, test_1_chksum_fx16},
    {"Test 1 FX16_FX8_FX8", mli_krn_conv2d_hwcn_fx16_fx8_fx8, 
                            input_1_fx16, weights_1_fx8, bias_1_fx8, test_1_out_fx16, test_1_cfg, 
                            thresholds_fx16_fx8_fx8_general, test_1_chksum_fx16_fx8_fx8},
    {"Test 1 SA8_SA8_SA32", mli_krn_conv2d_hwcn_sa8_sa8_sa32,
                            input_1_sa8, weights_1_sa8, bias_1_sa32, test_1_out_sa8, test_1_cfg, 
                            thresholds_sa8_general, test_1_chksum_sa8},

    // Basic functionality test with 7 kernels of (4, 3) size, strides = (2, 2), with krn_padding and with Gen_ReLU
    {"Test 2 FX16 ReluGen",         mli_krn_conv2d_hwcn_fx16, 
                                    input_1_fx16, weights_2_fx16, bias_1_fx16, test_2_out_fx16, test_2_cfg, 
                                    thresholds_fx16_general, test_2_chksum_fx16},
    {"Test 2 FX16_FX8_FX8 ReluGen", mli_krn_conv2d_hwcn_fx16_fx8_fx8,
                                    input_1_fx16, weights_2_fx8, bias_1_fx8, test_2_out_fx16, test_2_cfg,
                                    thresholds_fx16_fx8_fx8_general, test_2_chksum_fx16_fx8_fx8},
    {"Test 2 SA8_SA8_SA32 ReluGen", mli_krn_conv2d_hwcn_sa8_sa8_sa32, 
                                    input_1_sa8, weights_2_sa8, bias_1_w2_per_tensor_sa32, test_2_out_sa8, test_2_cfg,
                                    thresholds_sa8_general, test_2_chksum_sa8},

    // Dilation Rate Test: kernel_size=(3, 4), strides=(1, 1), w/o padding and w/o ReLU
    {"Test 3 FX16 Dilation",         mli_krn_conv2d_hwcn_fx16,
                                     input_1_fx16, weights_1_fx16, bias_1_fx16, test_3_out_fx16, test_3_cfg,
                                     thresholds_fx16_general, test_3_chksum_fx16},
    {"Test 3 FX16_FX8_FX8 Dilation", mli_krn_conv2d_hwcn_fx16_fx8_fx8,
                                     input_1_fx16, weights_1_fx8, bias_1_fx8, test_3_out_fx16, test_3_cfg,
                                     thresholds_fx16_fx8_fx8_general, test_3_chksum_fx16_fx8_fx8},
    {"Test 3 SA8_SA8_SA32 Dilation", mli_krn_conv2d_hwcn_sa8_sa8_sa32,
                                    input_1_sa8, weights_1_sa8, bias_1_sa32, test_3_out_sa8, test_3_cfg,
                                    thresholds_sa8_general, test_3_chksum_sa8},

    // Input/output Memstride test : kernel_size = (4, 3), strides = (3, 3), w / o padding and with ReLU_1
    // padded with 3 extra values on c Dim and extra 1 line. Output is also expected to have a memstride
    {"Test 4 FX16 IO_Memstr",         mli_krn_conv2d_hwcn_fx16,
                                      input_1_memstr_fx16, weights_1_fx16, bias_1_fx16, test_4_out_fx16, test_4_cfg,
                                      thresholds_fx16_general, test_4_chksum_fx16},
    {"Test 4 FX16_FX8_FX8 IO_Memstr", mli_krn_conv2d_hwcn_fx16_fx8_fx8,
                                      input_1_memstr_fx16, weights_1_fx8, bias_1_fx8, test_4_out_fx16, test_4_cfg,
                                      thresholds_fx16_fx8_fx8_test4, test_4_chksum_fx16_fx8_fx8},
    {"Test 4 SA8_SA8_SA32 IO_Memstr", mli_krn_conv2d_hwcn_sa8_sa8_sa32,
                                      input_1_memstr_sa8, weights_1_sa8, bias_1_sa32, test_4_out_sa8, test_4_cfg,
                                      thresholds_sa8_general, test_4_chksum_sa8},

    // Weights Memstride test with 7 kernels of (4, 3) size, strides = (1, 1), w / o padding and with ReLU_6
    // padded with extra channel on N dimension
    {"Test 5 FX16 W_Memstr",         mli_krn_conv2d_hwcn_fx16, 
                                     input_1_fx16, weights_2_memstr_fx16, bias_1_fx16, test_5_out_fx16, test_5_cfg,
                                     thresholds_fx16_general, test_5_chksum_fx16},
    {"Test 5 FX16_FX8_FX8 W_Memstr", mli_krn_conv2d_hwcn_fx16_fx8_fx8, 
                                     input_1_fx16, weights_2_memstr_fx8, bias_1_fx8, test_5_out_fx16, test_5_cfg,
                                     thresholds_fx16_fx8_fx8_general, test_5_chksum_fx16_fx8_fx8},
    {"Test 5 SA8_SA8_SA32 W_Memstr", mli_krn_conv2d_hwcn_sa8_sa8_sa32,
                                     input_1_sa8, weights_2_memstr_sa8, bias_1_w2_sa32, test_5_out_sa8, test_5_cfg,
                                     thresholds_sa8_general, test_5_chksum_sa8},

     // k1x1 specialization test with memstride, kernel_size=(1, 1), strides=(2, 2), krn_padding and ReLU 6
     // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 6 FX16 k1x1 Spec",          mli_krn_conv2d_hwcn_fx16_k1x1,
                                       input_1_fx16, weights_3_memstr_fx16, bias_1_fx16, test_6_out_fx16, test_6_cfg,
                                       thresholds_fx16_general, test_6_chksum_fx16},
    {"Test 6 FX16_FX8_FX8  k1x1 Spec", mli_krn_conv2d_hwcn_fx16_fx8_fx8_k1x1,
                                       input_1_fx16, weights_3_memstr_fx8, bias_1_fx8, test_6_out_fx16,
                                       test_6_cfg, thresholds_fx16_fx8_fx8_general, test_6_chksum_fx16_fx8_fx8},
    {"Test 6 SA8_SA8_SA32  k1x1 Spec", mli_krn_conv2d_hwcn_sa8_sa8_sa32_k1x1,
                                       input_1_sa8, weights_3_memstr_sa8, bias_1_w3_sa32, test_6_out_sa8,
                                       test_6_cfg, thresholds_sa8_general, test_6_chksum_sa8},

    // k3x3 specialization test with memstride, kernel_size=(3, 3), strides=(2, 2), krn_padding and ReLU 6
    // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 7 FX16 k3x3 Spec",         mli_krn_conv2d_hwcn_fx16_k3x3,
                                      input_1_fx16, weights_4_memstr_fx16, bias_1_fx16, test_7_out_fx16, test_7_cfg,
                                      thresholds_fx16_general, test_7_chksum_fx16},
    {"Test 7 FX16_FX8_FX8 k3x3 Spec", mli_krn_conv2d_hwcn_fx16_fx8_fx8_k3x3,
                                      input_1_fx16, weights_4_memstr_fx8, bias_1_fx8, test_7_out_fx16,
                                      test_7_cfg, thresholds_fx16_fx8_fx8_general, test_7_chksum_fx16_fx8_fx8},
    {"Test 7 SA8_SA8_SA32 k3x3 Spec", mli_krn_conv2d_hwcn_sa8_sa8_sa32_k3x3, 
                                      input_1_sa8, weights_4_memstr_sa8, bias_1_w4_sa32, test_7_out_sa8, test_7_cfg,
                                      thresholds_sa8_general, test_7_chksum_sa8},

    // k5x5 specialization test with memstride, kernel_size=(5, 5), strides=(2, 2), krn_padding and ReLU 6
    // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 8 FX16 k5x5 spec",         mli_krn_conv2d_hwcn_fx16_k5x5,
                                      input_1_fx16,weights_5_memstr_fx16, bias_1_fx16, test_8_out_fx16, test_8_cfg,
                                      thresholds_fx16_general, test_8_chksum_fx16},
    {"Test 8 FX16_FX8_FX8 k5x5 spec", mli_krn_conv2d_hwcn_fx16_fx8_fx8_k5x5,
                                      input_1_fx16, weights_5_memstr_fx8, bias_1_fx8, test_8_out_fx16,
                                      test_8_cfg, thresholds_fx16_fx8_fx8_general, test_8_chksum_fx16_fx8_fx8},
    {"Test 8 SA8_SA8_SA32 k5x5 spec", mli_krn_conv2d_hwcn_sa8_sa8_sa32_k5x5, 
                                      input_1_sa8, weights_5_memstr_sa8, bias_1_w5_sa32, test_8_out_sa8, test_8_cfg,
                                      thresholds_sa8_general, test_8_chksum_sa8},
    
    // Dilation test with padding for generic function, kernel_size=(3, 3), strides=(1, 1),
    // krn_padding , dilation = (2,2) and ReLU_Gen.
    // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 9-1 FX16 Dil+Pad",         mli_krn_conv2d_hwcn_fx16, 
                                      input_1_fx16, weights_4_memstr_fx16, bias_1_fx16, test_9_out_fx16, test_9_cfg,
                                      thresholds_fx16_general, test_9_chksum_fx16},
    {"Test 9-1 FX16_FX8_FX8 Dil+Pad", mli_krn_conv2d_hwcn_fx16_fx8_fx8,
                                      input_1_fx16, weights_4_memstr_fx8, bias_1_fx8, test_9_out_fx16,
                                      test_9_cfg, thresholds_fx16_fx8_fx8_general, test_9_chksum_fx16_fx8_fx8},
    {"Test 9-1 SA8_SA8_SA32 Dil+Pad", mli_krn_conv2d_hwcn_sa8_sa8_sa32,
                                      input_1_sa8, weights_4_memstr_sa8, bias_1_w4_sa32, test_9_out_sa8,
                                      test_9_cfg, thresholds_sa8_general, test_9_chksum_sa8},

    // Dilation test for k3x3 specialization test, kernel_size=(3, 3), strides=(1, 1), 
    // krn_padding , dilation = (2,2) and ReLU_Gen.
    // Memstrides are applied on input, output and weights tensors
    {"Test 9-2 FX16 k3x3 Dil",         mli_krn_conv2d_hwcn_fx16_k3x3,
                                       input_1_fx16, weights_4_memstr_fx16, bias_1_fx16, test_9_out_fx16, test_9_cfg,
                                       thresholds_fx16_general, test_9_chksum_fx16},
    {"Test 9-2 FX16_FX8_FX8 k3x3 Dil", mli_krn_conv2d_hwcn_fx16_fx8_fx8_k3x3,
                                       input_1_fx16, weights_4_memstr_fx8, bias_1_fx8, test_9_out_fx16,
                                       test_9_cfg, thresholds_fx16_fx8_fx8_general, test_9_chksum_fx16_fx8_fx8},
    {"Test 9-2 SA8_SA8_SA32 k3x3 Dil", mli_krn_conv2d_hwcn_sa8_sa8_sa32_k3x3, 
                                       input_1_sa8, weights_4_memstr_sa8, bias_1_w4_sa32, test_9_out_sa8, test_9_cfg,
                                       thresholds_sa8_general, test_9_chksum_sa8},

    // Dilation test for k5x5 specialization test, kernel_size=(5, 5), strides=(1, 1), 
    // krn_padding , dilation = (2,2) and ReLU_Gen.
    // Memstrides are applied on input, output and weights tensors
    {"Test 10 FX16 k5x5 Dil",         mli_krn_conv2d_hwcn_fx16_k5x5,
                                      input_1_fx16, weights_5_memstr_fx16, bias_1_fx16, test_10_out_fx16, test_10_cfg,
                                      thresholds_fx16_general, test_10_chksum_fx16},
    {"Test 10 FX16_FX8_FX8 k5x5 Dil", mli_krn_conv2d_hwcn_fx16_fx8_fx8_k5x5,
                                      input_1_fx16, weights_5_memstr_fx8, bias_1_fx8, test_10_out_fx16,
                                      test_10_cfg, thresholds_fx16_fx8_fx8_general, test_10_chksum_fx16_fx8_fx8},
    {"Test 10 SA8_SA8_SA32 k5x5 Dil", mli_krn_conv2d_hwcn_sa8_sa8_sa32_k5x5, 
                                      input_1_sa8, weights_5_memstr_sa8, bias_1_w5_sa32, test_10_out_sa8, test_10_cfg,
                                      thresholds_sa8_general, test_10_chksum_sa8},
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

    reporter.report_header("MLI|Kernels|Convolution 2D  Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        memory_manager mem_w_keeper((int8_t*)(scratch_mem_w), sizeof(scratch_mem_w));
        memory_manager mem_b_keeper((int8_t*)(scratch_mem_b), sizeof(scratch_mem_b));
        bool is_test_passed = true;
        const conv2d_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metics;

        if (strstr(cur_test->descr, "Test 9") != nullptr ||
                strstr(cur_test->descr, "Test 10") != nullptr) {
            // Kernel doesn't work properly with dilation ratio and padding turned on together.
            reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
            continue;
        }
        if (strstr(cur_test->descr, "Test 3") != nullptr) {
            // In debug mode with return codes checker badly handels dilation ratio.
            reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
            continue;
        }
#if PLATFORM == V2DSP_VECTOR
        if (strstr(cur_test->descr, " FX16 ") != nullptr) {
            // VPX fails bitwise comparison with reference .
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
                cur_test->mli_krn_conv2d(&input, &weights, &bias, &cur_test->cfg, &out) != MLI_STATUS_OK) {
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

    reporter.report_outline("[AUTO] Group: mli_krn_conv2d", final_status);

    return (final_status) ? 0 : 1;
}
