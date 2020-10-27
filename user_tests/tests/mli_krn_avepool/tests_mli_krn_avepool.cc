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

#include "vectors_mli_krn_avepool.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

typedef mli_status(*avepool_func_ptr)(
    const mli_tensor* /*in*/,
    const mli_pool_cfg* /*cfg*/,
    mli_tensor* /*out*/);

struct avepool_test_operands {
    const char* descr;
    const avepool_func_ptr mli_krn_avepool;
    tensor_quantizer in;
    tensor_quantizer out;
    const mli_pool_cfg cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

// Checksums of test tensors for various mli calculations mode. 
// When developer finished implementation of kernel and consider it as ok, He need to populate
// proper checksums for tests in order to highlight any change which affects results.
#if defined(CRC_RM_UP) || defined(CRC_RM_CONVERGENT)

// Shared CRC Results
const crc32_calc  test_1_chksum_fx16{ 0x292DDF4E }, test_1_chksum_sa8{ 0x1564B755 },
                  test_2_chksum_fx16{ 0xA7542BBE }, test_2_chksum_sa8{ 0xD825FD74 },
                  test_3_chksum_fx16{ 0x872AD40B }, test_3_chksum_sa8{ 0x0F472106 },
                                                    test_4_chksum_sa8{ 0xFEE5E73E },
                  test_5_chksum_fx16{ 0x2F40CE76 }, test_5_chksum_sa8{ 0x999C378F },
                  test_6_chksum_fx16{ 0x4871DD9B }, test_6_chksum_sa8{ 0x784CA521 },
                  test_7_chksum_fx16{ 0x56FC93D9 }, test_7_chksum_sa8{ 0x67A9C0DA };

// Platform Specific CRC Results
#if defined(CRC_RM_UP)
const crc32_calc  test_4_chksum_fx16{ 0x128DE247 };
#else
const crc32_calc  test_4_chksum_fx16{ 0x8F820331 };
#endif

#else // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16, test_1_chksum_sa8,
                  test_2_chksum_fx16, test_2_chksum_sa8,
                  test_3_chksum_fx16, test_3_chksum_sa8,
                  test_4_chksum_fx16, test_4_chksum_sa8,
                  test_5_chksum_fx16, test_5_chksum_sa8,
                  test_6_chksum_fx16, test_6_chksum_sa8,
                  test_7_chksum_fx16, test_7_chksum_sa8;
#endif

const quality_metrics thresholds_fx16_general { /* MaxAbsErr = */0.0003, quality_metrics::kPassValueSnr,
                                                /* SNR_DB = */80.f, /*Quant Error Perc = */ 27.f };

const quality_metrics thresholds_sa8_general{ /* MaxAbsErr = */0.04, quality_metrics::kPassValueSnr,
                                              /* SNR_DB = */37.f, /*Quant Error Perc = */ 30.f };


static const avepool_test_operands tests_list[] = {
    // Basic functionality test kernel_size=(4, 3), strides=(1, 1), w/o padding
    {"Test 1 FX16",  mli_krn_avepool_hwc_fx16,
                     input_1_fx16, test_1_out_fx16, test_1_cfg,
                     thresholds_fx16_general, test_1_chksum_fx16},
    {"Test 1 SA8",   mli_krn_avepool_hwc_sa8,
                     input_1_sa8, test_1_out_sa8, test_1_cfg,
                     thresholds_sa8_general, test_1_chksum_sa8},

    // Basic functionality test kernel_size=(3, 4), strides=(2, 2), with krn_padding
    {"Test 2 FX16",  mli_krn_avepool_hwc_fx16,
                     input_1_fx16, test_2_out_fx16, test_2_cfg,
                     thresholds_fx16_general, test_2_chksum_fx16},
    {"Test 2 SA8",   mli_krn_avepool_hwc_sa8,
                     input_1_sa8, test_2_out_sa8, test_2_cfg,
                     thresholds_sa8_general, test_2_chksum_sa8},
    
    // Memstride test kernel_size=(3, 4), strides=(3, 3), with krn_padding
    {"Test 3 FX16 Memstr",  mli_krn_avepool_hwc_fx16,
                            input_1_memstr_fx16, test_3_out_fx16, test_3_cfg,
                            thresholds_fx16_general, test_3_chksum_fx16},
    {"Test 3 SA8 Memstr",   mli_krn_avepool_hwc_sa8,
                            input_1_memstr_sa8, test_3_out_sa8, test_3_cfg, 
                            thresholds_sa8_general, test_3_chksum_sa8},

    // Global Pooling test with memstride
    {"Test 4 FX16 GlobalPool",  mli_krn_avepool_hwc_fx16,
                                input_2_memstr_fx16, test_4_out_fx16, test_4_cfg,
                                thresholds_fx16_general, test_4_chksum_fx16},
    {"Test 4 SA8 GlobalPool",   mli_krn_avepool_hwc_sa8,
                                input_2_memstr_sa8, test_4_out_sa8, test_4_cfg,
                                thresholds_sa8_general, test_4_chksum_sa8},
                    
    // Padding only areas test with memstride, kernel_size=(4, 4), strides=(2, 2), with krn_padding
    {"Test 5 FX16 Pad areas only",  mli_krn_avepool_hwc_fx16,
                                    input_2_memstr_fx16, test_5_out_fx16, test_5_cfg,
                                    thresholds_fx16_general, test_5_chksum_fx16},
    {"Test 5 SA8 pad areas only",   mli_krn_avepool_hwc_sa8,
                                    input_2_memstr_sa8, test_5_out_sa8, test_5_cfg,
                                    thresholds_sa8_general, test_5_chksum_sa8},

    // k2x2 specialization test with memstride, kernel_size=(2, 2), strides=(2, 2), krn_padding
    {"Test 6 FX16 k2x2 spec",  mli_krn_avepool_hwc_fx16_k2x2,
                               input_1_memstr_fx16, test_6_out_fx16, test_6_cfg,
                               thresholds_fx16_general, test_6_chksum_fx16},
    {"Test 6 SA8 k2x2 spec",   mli_krn_avepool_hwc_sa8_k2x2,
                               input_1_memstr_sa8, test_6_out_sa8, test_6_cfg,
                               thresholds_sa8_general, test_6_chksum_sa8},

    // k3x3 specialization test with memstride, kernel_size=(3, 3), strides=(2, 2), krn_padding
    {"Test 7 FX16 k3x3 spec",  mli_krn_avepool_hwc_fx16_k3x3,
                               input_1_memstr_fx16, test_7_out_fx16, test_7_cfg,
                               thresholds_fx16_general, test_7_chksum_fx16},
    {"Test 7 SA8 k3x3 spec",   mli_krn_avepool_hwc_sa8_k3x3,
                               input_1_memstr_sa8, test_7_out_sa8, test_7_cfg,
                               thresholds_sa8_general, test_7_chksum_sa8},
};

constexpr int kMemSize = 2047;
static IO_DATA_ATTR int8_t scratch_mem_in[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemSize] = { 0 };

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Kernels|Average Pooling Function Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        bool is_test_passed = true;
        const avepool_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metics;
        if (!(cur_test->in.is_valid() && cur_test->out.is_valid())) {
            reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
            is_test_passed = false;
        }

        mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
        mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
        // Workaround until the next FIXME fixed
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
                cur_test->mli_krn_avepool(&input, &cur_test->cfg, &out) != MLI_STATUS_OK) {
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

        // Check that kernel didn't modify quantization parameters provided by user.
        if (is_test_passed) {
            // FIXME: SA8 parameters are not filled
            // const mli_data_container empty_container{ 0 };
            // mli_tensor source_out_tensor = cur_test->out.get_not_quantized_tensor(empty_container);
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
            data_crc(out);
            is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr, test_metics, cur_test->threshold, 
                                                                data_crc, cur_test->check_sum);
        }
        final_status &= is_test_passed;
    }

    reporter.report_outline("[AUTO] Group: mli_krn_avepool", final_status);

    return (final_status) ? 0 : 1;
}
