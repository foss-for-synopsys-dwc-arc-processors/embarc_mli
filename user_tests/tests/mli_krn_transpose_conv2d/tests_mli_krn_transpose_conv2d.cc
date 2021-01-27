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

#include "vectors_mli_krn_transpose_conv2d.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

typedef mli_status(*transpose_conv2d_func_ptr)(
    const mli_tensor* /*input*/,
    const mli_tensor* /*weights*/,
    const mli_tensor* /*bias*/,
    const mli_conv2d_cfg* /*cfg*/,
    mli_tensor* /*output*/);

struct transpose_conv2d_test_operands {
    const char* descr;
    const transpose_conv2d_func_ptr mli_krn_transpose_conv2d;
    tensor_quantizer in;
    tensor_quantizer weights;
    tensor_quantizer bias;
    tensor_quantizer out;
    const mli_conv2d_cfg cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
    const uint32_t mem_fill_pattern;
};

// Checksums of test tensors for various mli calculations mode. 
// When developer finished implementation of kernel and consider it as ok, He need to populate
// proper checksums for tests in order to highlight any change which affects results.
#if defined(CRC_RM_CONVERGENT) || defined(CRC_RM_UP)

// Shared CRC Results
const crc32_calc test_1_chksum_fx16 {0x7CD22049}, /*test_1_chksum_fx16_fx8_fx8,             */ test_1_chksum_sa8 {0x59A71A0B},
                 test_2_chksum_fx16 {0x0B88C56E}, /*test_2_chksum_fx16_fx8_fx8,                test_2_chksum_sa8,*/
                 test_3_chksum_fx16 {0x85E46A29},   test_3_chksum_fx16_fx8_fx8 {0xF9B0F692},   test_3_chksum_sa8 {0xCE83CE66},
                 test_4_chksum_fx16 {0xC724EBF9}, /*test_4_chksum_fx16_fx8_fx8,             */ test_4_chksum_sa8 {0xDEE32B04},
                 test_5_chksum_fx16 {0xE82A4691}, /*test_5_chksum_fx16_fx8_fx8,             */ test_5_chksum_sa8 {0x591EA9A4},
                 test_6_chksum_fx16 {0x6D691353}, /*test_6_chksum_fx16_fx8_fx8,                test_6_chksum_sa8,*/
                 test_7_chksum_fx16 {0x314BD269}, /*test_7_chksum_fx16_fx8_fx8,             */ test_7_chksum_sa8 {0xA422B61F},
                 test_8_chksum_fx16 {0x4CDA936B}, test_8_chksum_fx16_fx8_fx8 {0x8436810F},     test_8_chksum_sa8 {0x8BC78C83};
// Platform Specific CRC Results
#if defined(CRC_RM_UP)
const crc32_calc test_1_chksum_fx16_fx8_fx8 {0xB8EF2F73},
                 test_2_chksum_fx16_fx8_fx8 {0x2A904693}, test_2_chksum_sa8 {0x20684A32},
                 test_4_chksum_fx16_fx8_fx8 {0xF0F39D2C}, 
                 test_5_chksum_fx16_fx8_fx8 {0xA3E639A8},
                 test_6_chksum_fx16_fx8_fx8 {0x1BE42216}, test_6_chksum_sa8 {0x179FAFCC},
                 test_7_chksum_fx16_fx8_fx8 {0x91D2A974};
#else 
const crc32_calc test_1_chksum_fx16_fx8_fx8 {0x9E58234E},
                 test_2_chksum_fx16_fx8_fx8 {0xB808A08B}, test_2_chksum_sa8 {0xB6D4CCF3},
                 test_4_chksum_fx16_fx8_fx8 {0xB617F5E9},
                 test_5_chksum_fx16_fx8_fx8 {0xD261DE7C},
                 test_6_chksum_fx16_fx8_fx8 {0x069E2E0E}, test_6_chksum_sa8 {0x2CC75486},
                 test_7_chksum_fx16_fx8_fx8 {0x118C5E59};
#endif
#else // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16, test_1_chksum_fx16_fx8_fx8, test_1_chksum_sa8,
                  test_2_chksum_fx16, test_2_chksum_fx16_fx8_fx8, test_2_chksum_sa8,
                  test_3_chksum_fx16, test_3_chksum_fx16_fx8_fx8, test_3_chksum_sa8,
                  test_4_chksum_fx16, test_4_chksum_fx16_fx8_fx8, test_4_chksum_sa8,
                  test_5_chksum_fx16, test_5_chksum_fx16_fx8_fx8, test_5_chksum_sa8,
                  test_6_chksum_fx16, test_6_chksum_fx16_fx8_fx8, test_6_chksum_sa8,
                  test_7_chksum_fx16, test_7_chksum_fx16_fx8_fx8, test_7_chksum_sa8;
#endif

const quality_metrics thresholds_fx16_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR_DB = */70.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_fx16_fx8_fx8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                  /* SNR_DB = */30.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                             /* SNR_DB = */35.f, /*Quant Error Perc = */40.f };

const quality_metrics thresholds_sa8_test3_7{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
/* SNR_DB = */35.f, /*Quant Error Perc = */30.f };

const uint32_t mem_fill_pattern_general = 0xDEADBEEF;
const uint32_t mem_fill_pattern_test2_2 = 0xFEEDBEEF;


static const transpose_conv2d_test_operands tests_list[] = {
    // Basic functionality test kernel_size=(3, 4), strides=(2, 2), with krn_padding and w/o ReLU
    {"Test 1 FX16",         mli_krn_transpose_conv2d_hwcn_fx16, 
                            input_1_fx16, weights_1_fx16, bias_1_fx16, test_1_out_fx16, test_1_cfg,
                            thresholds_fx16_general, test_1_chksum_fx16, mem_fill_pattern_general},
    {"Test 1 FX16_FX8_FX8", mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8, 
                            input_1_fx16, weights_1_fx8, bias_1_fx8, test_1_out_fx16, test_1_cfg, 
                            thresholds_fx16_fx8_fx8_general, test_1_chksum_fx16_fx8_fx8, mem_fill_pattern_general},
    {"Test 1 SA8_SA8_SA32", mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32,
                            input_1_sa8, weights_1_sa8, bias_1_i1_w1_sa32, test_1_out_sa8, test_1_cfg, 
                            thresholds_sa8_general, test_1_chksum_sa8, mem_fill_pattern_general},

    // Basic functionality test with 7 kernels of (4, 3) size, strides = (4, 3), w/o padding and with Gen_ReLU
    {"Test 2-1 FX16 ReluGen",       mli_krn_transpose_conv2d_hwcn_fx16, 
                                    input_1_fx16, weights_2_fx16, bias_1_fx16, test_2_out_fx16, test_2_cfg, 
                                    thresholds_fx16_general, test_2_chksum_fx16, mem_fill_pattern_general},
    {"Test 2-1 FX16_FX8_FX8 ReluGen", mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8,
                                    input_1_fx16, weights_2_fx8, bias_1_fx8, test_2_out_fx16, test_2_cfg,
                                    thresholds_fx16_fx8_fx8_general, test_2_chksum_fx16_fx8_fx8, 
                                    mem_fill_pattern_general},
    {"Test 2-1 SA8_SA8_SA32 ReluGen", mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32, 
                                    input_1_sa8, weights_2_sa8, bias_1_i1_w2_sa32, test_2_out_sa8, test_2_cfg,
                                    thresholds_sa8_general, test_2_chksum_sa8, mem_fill_pattern_general},

    // Memory access test: The same 2-1 test with different memory fill pattern
    {"Test 2-2 FX16 Mem",         mli_krn_transpose_conv2d_hwcn_fx16, 
                                  input_1_fx16, weights_2_fx16, bias_1_fx16, test_2_out_fx16, test_2_cfg, 
                                  thresholds_fx16_general, test_2_chksum_fx16, mem_fill_pattern_test2_2},
    {"Test 2-2 FX16_FX8_FX8 Mem", mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8,
                                  input_1_fx16, weights_2_fx8, bias_1_fx8, test_2_out_fx16, test_2_cfg,
                                  thresholds_fx16_fx8_fx8_general, test_2_chksum_fx16_fx8_fx8, 
                                  mem_fill_pattern_test2_2},
    {"Test 2-2 SA8_SA8_SA32 Mem", mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32, 
                                  input_1_sa8, weights_2_sa8, bias_1_i1_w2_sa32, test_2_out_sa8, test_2_cfg,
                                  thresholds_sa8_general, test_2_chksum_sa8, mem_fill_pattern_test2_2},

    // No strides case: kernel_size=(3, 4), strides=(1, 1), w/o padding and Relu1
    {"Test 3 FX16 Str_1x1",         mli_krn_transpose_conv2d_hwcn_fx16,
                                    input_1_fx16, weights_1_fx16, bias_1_fx16, test_3_out_fx16, test_3_cfg,
                                    thresholds_fx16_general, test_3_chksum_fx16, mem_fill_pattern_general},
    {"Test 3 FX16_FX8_FX8 Str_1x1", mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8,
                                    input_1_fx16, weights_1_fx8, bias_1_fx8, test_3_out_fx16, test_3_cfg,
                                    thresholds_fx16_fx8_fx8_general, test_3_chksum_fx16_fx8_fx8, mem_fill_pattern_general},
    {"Test 3 SA8_SA8_SA32 Str_1x1", mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32,
                                    input_1_sa8, weights_1_sa8, bias_1_i1_w1_sa32, test_3_out_sa8, test_3_cfg,
                                    thresholds_sa8_test3_7, test_3_chksum_sa8, mem_fill_pattern_general},

    // Input/output Memstride test : kernel_size = (4, 3), strides = (3, 2), w / o padding and with ReLU_6
    {"Test 4 FX16 IO_Memstr",         mli_krn_transpose_conv2d_hwcn_fx16,
                                      input_1_memstr_fx16, weights_2_fx16, bias_1_fx16, test_4_out_fx16, test_4_cfg,
                                      thresholds_fx16_general, test_4_chksum_fx16, mem_fill_pattern_general},
    {"Test 4 FX16_FX8_FX8 IO_Memstr", mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8,
                                      input_1_memstr_fx16, weights_2_fx8, bias_1_fx8, test_4_out_fx16, test_4_cfg,
                                      thresholds_fx16_fx8_fx8_general, test_4_chksum_fx16_fx8_fx8, mem_fill_pattern_general},
    {"Test 4 SA8_SA8_SA32 IO_Memstr", mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32,
                                      input_1_memstr_sa8, weights_2_sa8, bias_1_i1_w2_sa32, test_4_out_sa8, test_4_cfg,
                                      thresholds_sa8_general, test_4_chksum_sa8, mem_fill_pattern_general},

    // Weights Memstride test : kernels of (3, 4) size, strides = (3, 2), krn_padding and no ReLU
    {"Test 5 FX16 IOW_Memstr",         mli_krn_transpose_conv2d_hwcn_fx16, 
                                       input_1_memstr_fx16, weights_1_memstr_fx16, bias_1_fx16, test_5_out_fx16, test_5_cfg,
                                       thresholds_fx16_general, test_5_chksum_fx16, mem_fill_pattern_general},
    {"Test 5 FX16_FX8_FX8 IOW_Memstr", mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8, 
                                       input_1_memstr_fx16, weights_1_memstr_fx8, bias_1_fx8, test_5_out_fx16, test_5_cfg,
                                       thresholds_fx16_fx8_fx8_general, test_5_chksum_fx16_fx8_fx8, mem_fill_pattern_general},
    {"Test 5 SA8_SA8_SA32 IOW_Memstr", mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32,
                                       input_1_memstr_sa8, weights_1_memstr_sa8, bias_1_i1_w1_sa32, test_5_out_sa8, test_5_cfg,
                                       thresholds_sa8_general, test_5_chksum_sa8, mem_fill_pattern_general},

    // k2x2 str2x2 specialization test with memstride, kernel_size=(2, 2), strides=(2, 2), no_padding and ReLU 6
    // Memstrides are applied on input, output and weights tensors
    {"Test 6 FX16 k2x2 str2 Spec",   mli_krn_transpose_conv2d_hwcn_fx16_k2x2_str2,
                                     input_2_memstr_fx16, weights_3_memstr_fx16, bias_2_fx16, test_6_out_fx16, test_6_cfg,
                                     thresholds_fx16_general, test_6_chksum_fx16, mem_fill_pattern_general},
    {"Test 6 FX16_FX8_FX8 k2x2 st2", mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8_k2x2_str2,
                                     input_2_memstr_fx16, weights_3_memstr_fx8, bias_2_fx8, test_6_out_fx16,
                                     test_6_cfg, thresholds_fx16_fx8_fx8_general, test_6_chksum_fx16_fx8_fx8, mem_fill_pattern_general},
    {"Test 6 SA8_SA8_SA32 k2x2 st2", mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32_k2x2_str2,
                                     input_2_memstr_sa8, weights_3_memstr_sa8, bias_2_i2_w3_sa32, test_6_out_sa8,
                                     test_6_cfg, thresholds_sa8_general, test_6_chksum_sa8, mem_fill_pattern_general},

    // k4x4 str2x2 specialization test with memstride, kernel_size=(4, 4), strides=(2, 2), krn_padding and ReLU 6
    // Memstrides are applied on input, output and weights tensors
    {"Test 7 FX16 k4x4 str2 Spec",   mli_krn_transpose_conv2d_hwcn_fx16_k4x4_str2,
                                     input_2_memstr_fx16, weights_4_memstr_fx16, bias_2_fx16, test_7_out_fx16, test_7_cfg,
                                     thresholds_fx16_general, test_7_chksum_fx16, mem_fill_pattern_general},
    {"Test 7 FX16_FX8_FX8 k4x4 st2", mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8_k4x4_str2,
                                     input_2_memstr_fx16, weights_4_memstr_fx8, bias_2_fx8, test_7_out_fx16,
                                     test_7_cfg, thresholds_fx16_fx8_fx8_general, test_7_chksum_fx16_fx8_fx8, mem_fill_pattern_general},
    {"Test 7 SA8_SA8_SA32 k4x4 st2", mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32_k4x4_str2, 
                                     input_2_memstr_sa8, weights_4_memstr_sa8, bias_2_i2_w4_sa32, test_7_out_sa8, test_7_cfg,
                                     thresholds_sa8_test3_7, test_7_chksum_sa8, mem_fill_pattern_general},

    // k3x3 str2x2 test with memstride, kernel_size=(3, 3), strides=(2, 2), krn_padding and No Relu
    // specific regression test
    {"Test 8 FX16 k3x3 str2",        mli_krn_transpose_conv2d_hwcn_fx16,
                                     input_1_memstr_fx16, weights_5_memstr_fx16, bias_1_fx16, test_8_out_fx16, test_8_cfg, 
                                     thresholds_fx16_general, test_8_chksum_fx16, mem_fill_pattern_general},
    {"Test 8 FX16_FX8_FX8 k3x3 st2", mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8,
                                     input_1_memstr_fx16, weights_5_memstr_fx8, bias_1_fx8, test_8_out_fx16,
                                     test_8_cfg, thresholds_fx16_fx8_fx8_general, test_8_chksum_fx16_fx8_fx8, mem_fill_pattern_general},
    {"Test 8 SA8_SA8_SA32 k3x3 st2", mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32, 
                                     input_1_memstr_sa8, weights_5_memstr_sa8, bias_1_i1_w5_sa32, test_8_out_sa8, test_8_cfg,
                                     thresholds_sa8_general, test_8_chksum_sa8, mem_fill_pattern_general}
};

constexpr int kMemIOSize = 3047;
constexpr int kMemWSize = 2047;
static IO_DATA_ATTR int8_t scratch_mem_in[kMemIOSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemIOSize] = { 0 };
static W_DATA_ATTR int8_t scratch_mem_w[kMemWSize] = { 0 };
static W_DATA_ATTR int8_t scratch_mem_b[kMemWSize / 2] = { 0 };

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Kernels|Transpose Convolution 2D  Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        memory_manager mem_w_keeper((int8_t*)(scratch_mem_w), sizeof(scratch_mem_w));
        memory_manager mem_b_keeper((int8_t*)(scratch_mem_b), sizeof(scratch_mem_b));
        bool is_test_passed = true;
        const transpose_conv2d_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metrics;

        if (!(cur_test->in.is_valid() && cur_test->weights.is_valid() &&
                cur_test->bias.is_valid() && cur_test->out.is_valid())) {
            reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
            is_test_passed = false;
        }

        const uint32_t fill_val = cur_test->mem_fill_pattern;
        mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in, fill_val));
        mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out, fill_val));
        mli_tensor bias = cur_test->bias.get_quantized_tensor(mem_b_keeper.allocate_memory(cur_test->bias, fill_val));
        mli_tensor weights = cur_test->weights.get_quantized_tensor(mem_w_keeper.allocate_memory(cur_test->weights,
                                                                                                 fill_val));
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
                cur_test->mli_krn_transpose_conv2d(&input, &weights, &bias, &cur_test->cfg, &out) != MLI_STATUS_OK) {
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

    reporter.report_outline("[AUTO] Group: mli_krn_transpose_conv2d", final_status);

    return (final_status) ? 0 : 1;
}
