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
const crc32_calc  test_1_chksum_fx16{ 0x7DDA59C8 }, test_1_chksum_fx8{ 0xBFE26CDD }, test_1_chksum_sa8{ 0xB555AC5A },
                  test_2_chksum_fx16{ 0x48A78608 }, test_2_chksum_fx8{ 0x13275726 }, test_2_chksum_sa8{ 0x1C3CD42E },
                  test_3_chksum_fx16{ 0x8670B52E }, test_3_chksum_fx8{ 0xEDBB5DD4 }, test_3_chksum_sa8{ 0xC7AB060F },
                  test_4_chksum_fx16{ 0x9C1AA0B9 }, test_4_chksum_fx8{ 0xD6A1C316 }, test_4_chksum_sa8{ 0x043C46FB },
                  test_5_chksum_fx16{ 0x638D962C }, test_5_chksum_fx8{ 0x0FDF29A8 }, test_5_chksum_sa8{ 0x3441D826 },
                  test_6_chksum_fx16{ 0x14ECC2F0 }, test_6_chksum_fx8{ 0x5A43F7EB }, test_6_chksum_sa8{ 0x75D4A41F },
                  test_7_chksum_fx16{ 0xF4951D21 }, test_7_chksum_fx8{ 0xE25FDC5A }, test_7_chksum_sa8{ 0xA8E5DD39 },
                  test_8_chksum_fx16{ 0x3307AC05 }, test_8_chksum_fx8{ 0x598491BD }, test_8_chksum_sa8{ 0xA496E075 },
                  test_9_chksum_fx16{ 0xCF6F71BA }, test_9_chksum_fx8{ 0x254C2F3E }, test_9_chksum_sa8{ 0xBA173619 };

#else  // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16, test_1_chksum_fx8, test_1_chksum_sa8,
                  test_2_chksum_fx16, test_2_chksum_fx8, test_2_chksum_sa8,
                  test_3_chksum_fx16, test_3_chksum_fx8, test_3_chksum_sa8,
                  test_4_chksum_fx16, test_4_chksum_fx8, test_4_chksum_sa8,
                  test_5_chksum_fx16, test_5_chksum_fx8, test_5_chksum_sa8,
                  test_6_chksum_fx16, test_6_chksum_fx8, test_6_chksum_sa8,
                  test_7_chksum_fx16, test_7_chksum_fx8, test_7_chksum_sa8,
                  test_8_chksum_fx16, test_8_chksum_fx8, test_8_chksum_sa8,
                  test_9_chksum_fx16, test_9_chksum_fx8, test_9_chksum_sa8;
#endif

const quality_metrics thresholds_fx_general {/* MaxAbsErr = */0.0f, quality_metrics::kPassValueSnr,
                                              /* SNR_DB = */84.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general{quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                              /* SNR_DB = */40.f, /*Quant Error Perc = */ 99.9f };
                                                
//TODO make normaly test description
static const permute_test_operands tests_list[] = {
    {"Test 1 FX16 I_m_str",  mli_krn_permute_fx16,
                            input_1_memstr_fx16, test_1_out_fx16, test_1_cfg,
                            thresholds_fx_general, test_1_chksum_fx16},
    {"Test 1 FX8 I_m_str",  mli_krn_permute_fx8,
                            input_1_memstr_fx8, test_1_out_fx8, test_1_cfg,
                            thresholds_fx_general, test_1_chksum_fx8},
    {"Test 1 SA8 I_m_str 0-axis",   mli_krn_permute_sa8,
                            input_1_memstr_sa8, test_1_out_sa8, test_1_cfg,
                            thresholds_sa8_general, test_1_chksum_sa8},

    {"Test 2 FX16 O_m_str",  mli_krn_permute_fx16,
                            input_2_fx16, test_2_out_memstr_fx16, test_2_cfg,
                            thresholds_fx_general, test_2_chksum_fx16},
    {"Test 2 FX8 O_m_str",  mli_krn_permute_fx8,
                            input_2_fx8, test_2_out_memstr_fx8, test_2_cfg,
                            thresholds_fx_general, test_2_chksum_fx8},
    {"Test 2 SA8 O_m_str per-tensor",   mli_krn_permute_sa8,
                            input_2_memstr_sa8, test_2_out_memstr_sa8, test_2_cfg,
                            thresholds_sa8_general, test_2_chksum_sa8},

    {"Test 3 FX16 IO_m_str",  mli_krn_permute_fx16,
                            input_2_memstr_fx16, test_3_out_memstr_fx16, test_3_cfg,
                            thresholds_fx_general, test_3_chksum_fx16},
    {"Test 3 FX8 IO_m_str",  mli_krn_permute_fx8,
                            input_2_memstr_fx8, test_3_out_memstr_fx8, test_3_cfg,
                            thresholds_fx_general, test_3_chksum_fx8},
    {"Test 3 SA8 IO_m_str per-tensor",   mli_krn_permute_sa8,
                            input_2_memstr_sa8, test_3_out_memstr_sa8, test_3_cfg, 
                            thresholds_sa8_general, test_3_chksum_sa8},

    {"Test 4 FX16 ",  mli_krn_permute_fx16,
                            input_3_fx16, test_4_out_fx16, test_4_cfg,
                            thresholds_fx_general, test_4_chksum_fx16},
    {"Test 4 FX8 ",  mli_krn_permute_fx8,
                            input_3_fx8, test_4_out_fx8, test_4_cfg,
                            thresholds_fx_general, test_4_chksum_fx8},
    {"Test 4 SA8 per-tensor",   mli_krn_permute_sa8,
                            input_3_sa8, test_4_out_sa8, test_4_cfg,
                            thresholds_sa8_general, test_4_chksum_sa8},

    {"Test 5 FX16",  mli_krn_permute_fx16,
                            input_4_fx16, test_5_out_fx16, test_5_cfg,
                            thresholds_fx_general, test_5_chksum_fx16},
    {"Test 5 FX8",  mli_krn_permute_fx8,
                            input_4_fx8, test_5_out_fx8, test_5_cfg,
                            thresholds_fx_general, test_5_chksum_fx8},
    {"Test 5 SA8 1-axis",   mli_krn_permute_sa8,
                            input_4_sa8, test_5_out_sa8, test_5_cfg,
                            thresholds_sa8_general, test_5_chksum_sa8},

    {"Test 6 FX16",  mli_krn_permute_fx16,
                           input_2_fx16, test_6_out_fx16, test_6_cfg,
                           thresholds_fx_general, test_6_chksum_fx16},
    {"Test 6 FX8",  mli_krn_permute_fx8,
                           input_2_fx8, test_6_out_fx8, test_6_cfg,
                           thresholds_fx_general, test_6_chksum_fx8},
    {"Test 6 SA8 per-tensor",   mli_krn_permute_sa8,
                            input_2_sa8, test_6_out_sa8, test_6_cfg,
                            thresholds_sa8_general, test_6_chksum_sa8},

    {"Test 7 FX16",  mli_krn_permute_fx16,
                            input_2_fx16, test_7_out_fx16, test_7_cfg,
                            thresholds_fx_general, test_7_chksum_fx16},
    {"Test 7 FX8",  mli_krn_permute_fx8,
                            input_2_fx8, test_7_out_fx8, test_7_cfg,
                            thresholds_fx_general, test_7_chksum_fx8},
    {"Test 7 SA8 per-tensor",   mli_krn_permute_sa8,
                            input_2_sa8, test_7_out_sa8, test_7_cfg,
                            thresholds_sa8_general, test_7_chksum_sa8},

    {"Test 8 FX16",  mli_krn_permute_fx16,
                            input_2_fx16, test_8_out_fx16, test_8_cfg,
                            thresholds_fx_general, test_8_chksum_fx16},
    {"Test 8 FX8",  mli_krn_permute_fx8,
                            input_2_fx8, test_8_out_fx8, test_8_cfg,
                            thresholds_fx_general, test_8_chksum_fx8},
    {"Test 8 SA8 per-tensor",   mli_krn_permute_sa8,
                            input_2_sa8, test_8_out_sa8, test_8_cfg,
                            thresholds_sa8_general, test_8_chksum_sa8},

    {"Test 9 FX16",  mli_krn_permute_fx16,
                            input_2_fx16, test_9_out_fx16, test_9_cfg,
                            thresholds_fx_general, test_9_chksum_fx16},
    {"Test 9 FX8",  mli_krn_permute_fx8,
                            input_2_fx8, test_9_out_fx8, test_9_cfg,
                            thresholds_fx_general, test_9_chksum_fx8},
    {"Test 9 SA8 per-tensor",   mli_krn_permute_sa8,
                            input_2_sa8, test_9_out_sa8, test_9_cfg,
                            thresholds_sa8_general, test_9_chksum_sa8}
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

        mli_tensor in = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
        mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));

        if (out.el_type == MLI_EL_SA_8) {
            if (out.el_params.sa.dim >= 0) {
                if (out.el_params.sa.zero_point.capacity < in.el_params.sa.zero_point.capacity &&
                        out.el_params.sa.scale.capacity < in.el_params.sa.scale.capacity &&
                        out.el_params.sa.scale_frac_bits.capacity < in.el_params.sa.scale_frac_bits.capacity) {
                    reporter.report_message(cur_test->descr, 
                        "FAILED at init: not enough memory allocated for quantization parameters");
                }
            }
        }
        
        if (is_test_passed &&
                (tensor_quantizer::validate_tensor(in) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(out) != tensor_quantizer::kOk)) {
            reporter.report_message(cur_test->descr, 
                                    "FAILED at quantization step: more memory for one of tensors might be required");
            is_test_passed = false;
        }

        //Fill all fields in out tensor as -1 except data field and capacity field
        out.rank = -1;
        for(int i = 0; i < MLI_MAX_RANK; i++) {
            out.shape[i] = 0;
            out.mem_stride[i] = 0;
        }

        if (is_test_passed &&
                (mem_in_keeper.is_memory_corrupted() || mem_out_keeper.is_memory_corrupted())) {
            reporter.report_message(cur_test->descr,
                "FAILED at quantization step: memory beside one of operands is corrupted");
            is_test_passed = false;
        }

        // Run specific kernel for test 
        if (is_test_passed &&
                cur_test->mli_krn_permute(&in, &cur_test->cfg, &out) != MLI_STATUS_OK) {
            reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
            is_test_passed = false;
        }

        if(is_test_passed && in.rank != out.rank) {
            reporter.report_message(cur_test->descr,
                "FAILED after kernel run: rank input and output tensors are different");
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
            data_crc(in);
            data_crc(out);
            is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr, test_metics, cur_test->threshold, 
                                                                data_crc, cur_test->check_sum);
        }
        final_status &= is_test_passed;
    }

    reporter.report_outline("[AUTO] Group: mli_krn_permute", final_status);

    return (final_status) ? 0 : 1;
}
