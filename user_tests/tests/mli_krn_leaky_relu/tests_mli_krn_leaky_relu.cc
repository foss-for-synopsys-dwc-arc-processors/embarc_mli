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

#include "vectors_mli_krn_leaky_relu.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

typedef mli_status (*leaky_relu_func_ptr)(
    const mli_tensor* /*in*/, 
    const mli_tensor* /*slope_coeff*/, 
    mli_tensor* /*out*/);

struct leaky_relu_test_operands {
    const char* descr;
    const leaky_relu_func_ptr mli_krn_leaky_relu;
    tensor_quantizer in;
    tensor_quantizer slope_coeff;
    tensor_quantizer out;
    const quality_metrics threshold;
    const crc32_calc check_sum;
    const bool in_place_comp;
};

#if defined(CRC_RM_CONVERGENT) || defined(CRC_RM_UP)

// Shared CRC Results
const crc32_calc  test_1_chksum_fx16{ 0x7695FBF8 }, test_1_chksum_sa8{ 0x97AACEC4 },
                  test_2_chksum_fx16{ 0x0E245804 }, test_2_chksum_sa8{ 0x6A2A3EB5 },
                  test_3_chksum_fx16{ 0xB4A6991D }, test_3_chksum_sa8{ 0x6FEF833F };

#else  // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16, test_1_chksum_sa8,
                  test_2_chksum_fx16, test_2_chksum_sa8,
                  test_3_chksum_fx16, test_3_chksum_sa8;

#endif

const quality_metrics thresholds_fx16_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR DB = */ 60.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                               /* SNR DB = */ 30.f, quality_metrics::kPassValueQuantErrPerc };

static const leaky_relu_test_operands tests_list[] = {
    // Leaky Relu input range [-3:3], slope < 1, MemStr
    {"Test 1 FX16 (slope < 1)",  mli_krn_leaky_relu_fx16,
                                    input_1_fx16, input_2_fx16, test_1_out_fx16,
                                    thresholds_fx16_general, test_1_chksum_fx16, false},
    {"Test 1 SA8  (slope < 1)",  mli_krn_leaky_relu_sa8,
                                    input_1_sa8, input_2_sa8, test_1_out_sa8,
                                    thresholds_sa8_general, test_1_chksum_sa8, false},

    // Leaky Relu input range [-3:3], slope > 1, MemStr
    {"Test 2 FX16 (slope > 1)",  mli_krn_leaky_relu_fx16, 
                                    input_1_fx16, input_3_fx16, test_2_out_fx16,
                                    thresholds_fx16_general, test_2_chksum_fx16, false},
    {"Test 2 SA8  (slope > 1)",  mli_krn_leaky_relu_sa8, 
                                    input_1_sa8, input_3_sa8, test_2_out_sa8,
                                    thresholds_sa8_general, test_2_chksum_sa8, false},

    // Leaky Relu input range [-3:3], slope > 1, MemStr, In Place Computation
    {"Test 3 FX16 (slope > 1) IPC",  mli_krn_leaky_relu_fx16, 
                                    input_1_fx16, input_3_fx16, test_2_out_fx16,
                                    thresholds_fx16_general, test_3_chksum_fx16, true},
    {"Test 3 SA8  (slope > 1) IPC",  mli_krn_leaky_relu_sa8, 
                                    input_1_sa8, input_3_sa8, test_2_out_sa8,
                                    thresholds_sa8_general, test_3_chksum_sa8, true},
};

constexpr int kMemSize = 2048;
static IO_DATA_ATTR int8_t scratch_mem_in[kMemSize]  = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_slope_coeff[kMemSize]  = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemSize] = { 0 };

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Kernels|Leaky Relu Functions Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
        memory_manager mem_slope_coeff_keeper((int8_t*)(scratch_mem_slope_coeff), sizeof(scratch_mem_slope_coeff));
        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        bool is_test_passed = true;
        const leaky_relu_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metics;
        if (!(cur_test->in.is_valid() && cur_test->slope_coeff.is_valid() && cur_test->out.is_valid())) {
            reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
            is_test_passed = false;
        }

        mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
        mli_tensor slope_coeff = cur_test->slope_coeff.get_quantized_tensor(
                                           mem_slope_coeff_keeper.allocate_memory(cur_test->slope_coeff));
        mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
        if (cur_test->in_place_comp) {
            mli_element_params params = out.el_params;
			/* Reuse Input Tensor */
            out = input;
            /* Output Params Provided by User */
            out.el_params = params;
        }

        mli_tensor source_out_tensor = out;
        if (is_test_passed &&
                (tensor_quantizer::validate_tensor(input) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(slope_coeff) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(out) != tensor_quantizer::kOk)) {
            reporter.report_message(cur_test->descr, 
                                    "FAILED at quantization step: more memory for one of tensors might be required");
            is_test_passed = false;
        }

        if (is_test_passed &&
                (mem_in_keeper.is_memory_corrupted() || 
                 mem_slope_coeff_keeper.is_memory_corrupted() || 
                 mem_out_keeper.is_memory_corrupted())) {
            reporter.report_message(cur_test->descr,
                "FAILED at quantization step: memory beside one of operands is corrupted");
            is_test_passed = false;
        }

        // Run specific kernel for test 
        if (is_test_passed &&
                cur_test->mli_krn_leaky_relu(&input, &slope_coeff, &out) != MLI_STATUS_OK) {
            reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
            is_test_passed = false;
        }

        if (is_test_passed &&
                (mem_in_keeper.is_memory_corrupted() || 
                 mem_slope_coeff_keeper.is_memory_corrupted() || 
                 mem_out_keeper.is_memory_corrupted())) {
            reporter.report_message(cur_test->descr,
                "FAILED after kernel run: memory beside one of operands is corrupted");
            is_test_passed = false;
        }

        if (is_test_passed && cur_test->in_place_comp &&
                !mli_hlp_tensor_data_ptr_cmp(&input, &out)) {
            reporter.report_message(cur_test->descr,
                "FAILED after kernel run: memory corrupted for In Place Computation");
            is_test_passed = false;
        }

        if (is_test_passed &&
                test_metics.calculate_metrics(out, cur_test->out) == false) {
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
            data_crc(out);
            is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr, test_metics, cur_test->threshold, 
                                                                data_crc, cur_test->check_sum);
        }
        final_status &= is_test_passed;
    }

    reporter.report_outline("[AUTO] Group: mli_krn_leaky_relu", final_status);

    return (final_status) ? 0 : 1;
}
