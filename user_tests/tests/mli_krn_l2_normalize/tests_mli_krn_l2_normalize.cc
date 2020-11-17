/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_api.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "mli_types.h"
#include "test_tensor_quantizer.h"
#include "test_report.h"

#include "vectors_mli_krn_l2_normalize.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

typedef mli_status (*l2_normalize_func_ptr)(
    const mli_tensor* /*in*/, 
    const mli_tensor* /*epsilon*/, 
    const mli_l2_normalize_cfg *cfg,
    mli_tensor* /*out*/);

struct l2_normalize_test_operands {
    const char* descr;
    const l2_normalize_func_ptr mli_krn_l2_normalize;
    tensor_quantizer in;
    // tensor_quantizer epsilon;
    const mli_l2_normalize_cfg cfg;
    tensor_quantizer out;
    const quality_metrics threshold;
    const crc32_calc check_sum;
    const bool in_place_comp;
};

static constexpr int kOutFx16FracBits = 15;
static constexpr int kOutSa8Scale = 1;
static constexpr int kOutSa8ZeroPoint = 0;
static constexpr int kOutSa8ScaleFracBits = 7;

#if defined(CRC_RM_CONVERGENT) || defined(CRC_RM_UP)

// Shared CRC Results
const crc32_calc                                    test_1_chksum_sa8{ 0x5827AA71 },
                                                    test_2_chksum_sa8{ 0x6AB2B7E2 },
                                                    test_3_chksum_sa8{ 0xF7C243B9 },
                  test_4_chksum_fx16{ 0xBD6B3431 }, test_4_chksum_sa8{ 0xC8856D4A },
                  test_5_chksum_fx16{ 0xCB86AA5F }, test_5_chksum_sa8{ 0x73701BB8 },
                                                    test_6_chksum_sa8{ 0xCE87C84D },
                  test_7_chksum_fx16{ 0x5FB149F8 }, test_7_chksum_sa8{ 0x735D468E },
                  test_8_chksum_fx16{ 0x7F2E302F }, test_8_chksum_sa8{ 0x082A5005 };

// Platform Specific CRC Results
#if defined(CRC_RM_CONVERGENT)
const crc32_calc  test_1_chksum_fx16{ 0xA0D2FBE9 },
                  test_2_chksum_fx16{ 0xFC8B4CA8 },
                  test_3_chksum_fx16{ 0x06A11B7C },
                  test_6_chksum_fx16{ 0xAC710778 };
#else
const crc32_calc  test_1_chksum_fx16{ 0xA0D2FBE9 },
                  test_2_chksum_fx16{ 0xFC8B4CA8 },
                  test_3_chksum_fx16{ 0x629B7DA6 },
                  test_6_chksum_fx16{ 0x94E25844 };
#endif

#else  // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16{}, test_1_chksum_sa8{},
                  test_2_chksum_fx16{}, test_2_chksum_sa8{},
                  test_3_chksum_fx16{}, test_3_chksum_sa8{},
                  test_4_chksum_fx16{}, test_4_chksum_sa8{},
                  test_5_chksum_fx16{}, test_5_chksum_sa8{},
                  test_6_chksum_fx16{}, test_6_chksum_sa8{},
                  test_7_chksum_fx16{}, test_7_chksum_sa8{},
                  test_8_chksum_fx16{}, test_8_chksum_sa8{};

#endif

const quality_metrics thresholds_fx16_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR DB = */ 50.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                               /* SNR DB = */ 28.f, quality_metrics::kPassValueQuantErrPerc };

static const l2_normalize_test_operands tests_list[] = {
    /* Unifrom Distrbution, per tensor and per axis */
    {"Test 1 FX16 3D",  mli_krn_l2_normalize_fx16,
                                    input_1_fx16, test_1_cfg, test_1_out_fx16,
                                    thresholds_fx16_general, test_1_chksum_fx16, false},
    {"Test 1 SA8  3D",  mli_krn_l2_normalize_sa8,
                                    input_1_sa8, test_1_cfg, test_1_out_sa8,
                                    thresholds_sa8_general, test_1_chksum_sa8, false},
    {"Test 2 FX16 3D, Axis=2",  mli_krn_l2_normalize_fx16,
                                    input_1_fx16, test_2_cfg, test_2_out_fx16,
                                    thresholds_fx16_general, test_2_chksum_fx16, false},
    {"Test 2 SA8  3D, Axis=2",  mli_krn_l2_normalize_sa8,
                                    input_1_sa8, test_2_cfg, test_2_out_sa8,
                                    thresholds_sa8_general, test_2_chksum_sa8, false},
    {"Test 3 FX16 3D, Axis=0",  mli_krn_l2_normalize_fx16,
                                    input_1_fx16, test_3_cfg, test_3_out_fx16,
                                    thresholds_fx16_general, test_3_chksum_fx16, false},
    {"Test 3 SA8  3D, Axis=0",  mli_krn_l2_normalize_sa8,
                                    input_1_sa8, test_3_cfg, test_3_out_sa8,
                                    thresholds_sa8_general, test_3_chksum_sa8, false},
    /* Normal Distrbution with LeakyRELU Emulation , per tensor and per axis,  Memory Strides  */
    {"Test 4 FX16 4D, Memstr",  mli_krn_l2_normalize_fx16,
                                    input_2_fx16, test_4_cfg, test_4_out_fx16,
                                    thresholds_fx16_general, test_4_chksum_fx16, false},
    {"Test 4 SA8  4D, Memstr",  mli_krn_l2_normalize_sa8,
                                    input_2_sa8, test_4_cfg, test_4_out_sa8,
                                    thresholds_sa8_general, test_4_chksum_sa8, false},
    {"Test 5 FX16 4D, Axis=3, Memstr",  mli_krn_l2_normalize_fx16,
                                    input_2_fx16, test_5_cfg, test_5_out_fx16,
                                    thresholds_fx16_general, test_5_chksum_fx16, false},
    {"Test 5 SA8  4D, Axis=3, Memstr",  mli_krn_l2_normalize_sa8,
                                    input_2_sa8, test_5_cfg, test_5_out_sa8,
                                    thresholds_sa8_general, test_5_chksum_sa8, false},
    {"Test 6 FX16 4D, Axis=0, Memstr",  mli_krn_l2_normalize_fx16,
                                    input_2_fx16, test_6_cfg, test_6_out_fx16,
                                    thresholds_fx16_general, test_6_chksum_fx16, false},
    {"Test 6 SA8  4D, Axis=0, Memstr",  mli_krn_l2_normalize_sa8,
                                    input_2_sa8, test_6_cfg, test_6_out_sa8,
                                    thresholds_sa8_general, test_6_chksum_sa8, false},
    /* Special Case: Zeros, Equal Values, In Place Computation */
    {"Test 7 FX16 2D, Axis=1, IPC",  mli_krn_l2_normalize_fx16,
                                    input_3_fx16, test_7_cfg, test_7_out_fx16,
                                    thresholds_fx16_general, test_7_chksum_fx16, true},
    {"Test 7 SA8  2D, Axis=1, IPC",  mli_krn_l2_normalize_sa8,
                                    input_3_sa8, test_7_cfg, test_7_out_sa8,
                                    thresholds_sa8_general, test_7_chksum_sa8, true},
    /* Special Case: One Hot Value, In Place Computation */
    {"Test 8 FX16 2D, Axis=0, IPC",  mli_krn_l2_normalize_fx16,
                                    input_3_fx16, test_8_cfg, test_8_out_fx16,
                                    thresholds_fx16_general, test_8_chksum_fx16, true},
    {"Test 8 SA8  2D, Axis=0, IPC",  mli_krn_l2_normalize_sa8,
                                    input_3_sa8, test_8_cfg, test_8_out_sa8,
                                    thresholds_sa8_general, test_8_chksum_sa8, true},
};

constexpr int kMemSize = 2048;
static IO_DATA_ATTR int8_t scratch_mem_in[kMemSize]  = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemSize] = { 0 };

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Kernels|L2 Normalize Functions Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        bool is_test_passed = true;
        const l2_normalize_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metics;
        if (!(cur_test->in.is_valid() && cur_test->out.is_valid())) {
            reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
            is_test_passed = false;
        }

        mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
        mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
        if (cur_test->in_place_comp) {
            mli_element_params params = out.el_params;
			/* Reuse Input Tensor */
            out = input;
            /* Output Params Provided by User */
            out.el_params = params;
        }

        if (is_test_passed &&
                (tensor_quantizer::validate_tensor(input) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(out) != tensor_quantizer::kOk)) {
            reporter.report_message(cur_test->descr, 
                                    "FAILED at quantization step: more memory for one of tensors might be required");
            is_test_passed = false;
        }

        if (is_test_passed &&
                (mem_in_keeper.is_memory_corrupted() ||
                 mem_out_keeper.is_memory_corrupted())) {
            reporter.report_message(cur_test->descr,
                "FAILED at quantization step: memory beside one of operands is corrupted");
            is_test_passed = false;
        }

        // Run specific kernel for test 
        mli_tensor epsilon;
        if (is_test_passed &&
                cur_test->mli_krn_l2_normalize(&input, &epsilon, &cur_test->cfg, &out) != MLI_STATUS_OK) {
            reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
            is_test_passed = false;
        }

        if (is_test_passed &&
                (mem_in_keeper.is_memory_corrupted() || 
                 mem_out_keeper.is_memory_corrupted())) {
            reporter.report_message(cur_test->descr,
                "FAILED after kernel run: memory beside one of operands is corrupted");
            is_test_passed = false;
        }

        if (is_test_passed && cur_test->in_place_comp &&
                (input.data.mem.void_p != out.data.mem.void_p)) {
            reporter.report_message(cur_test->descr,
                "FAILED after kernel run: memory corrupted for In Place Computation");
            is_test_passed = false;
        }

        if (is_test_passed &&
                test_metics.calculate_metrics(out, cur_test->out) == false) {
            reporter.report_message(cur_test->descr, "FAILED at comparison output with reference");
            is_test_passed = false;
        }

        // Check that kernel output quantization parameters are set by kernel (according to spec).
        if (is_test_passed) {
            bool is_per_tensor_quant = true;

            if (out.el_type == MLI_EL_FX_16) {
                is_test_passed &= out.el_params.fx.frac_bits == kOutFx16FracBits;
            } else if (out.el_type == MLI_EL_SA_8) {
                if (out.el_params.sa.dim < 0 || input.el_params.sa.dim < 0) {
                    is_test_passed &=
                        (out.el_params.sa.scale.mem.i16 == kOutSa8Scale) &&
                        (out.el_params.sa.zero_point.mem.i16 == kOutSa8ZeroPoint) &&
                        (out.el_params.sa.scale_frac_bits.mem.i8 == kOutSa8ScaleFracBits);
                } else {
                    is_per_tensor_quant = false;
                    is_test_passed = false;
                }
            } else {
                assert(0);
            }
            if (!is_test_passed) {
                reporter.report_message(cur_test->descr,
                    is_per_tensor_quant ? "FAILED as element params of output tensor are incorrect"
                                        : "FAILED as per-axis quantization of tensors isn't supported by kernel");
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

    reporter.report_outline("[AUTO] Group: mli_krn_l2_normalize", final_status);

    return (final_status) ? 0 : 1;
}
