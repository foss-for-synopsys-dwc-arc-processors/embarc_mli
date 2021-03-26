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

#include "vectors_mli_krn_tanh.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;

typedef mli_status (*tanh_func_ptr)(
    const mli_tensor* /*in*/, 
    const mli_lut* /*lut*/,
    mli_tensor* /*out*/);

struct tanh_test_operands {
    const char* descr;
    const tanh_func_ptr mli_krn_tanh;
    tensor_quantizer in;
    tensor_quantizer out;
    const quality_metrics threshold;
    const crc32_calc check_sum;
    const bool in_place_comp;
};

static constexpr int kOutFx16FracBits = 15;
static constexpr int kOutFx8FracBits = 7;
static constexpr int kOutSa8Scale = 1;
static constexpr int kOutSa8ZeroPoint = 0;
static constexpr int kOutSa8ScaleFracBits = 7;

#if defined(CRC_RM_UP)
static const crc32_calc test_1_chksum_fx16{ 0x2A705B8D }, test_1_chksum_sa8{ 0x13FF84FA },
                        test_2_chksum_fx16{ 0x64F8E918 }, test_2_chksum_sa8{ 0x06F6413A },
                        test_3_chksum_fx16{ 0x5385A814 }, test_3_chksum_sa8{ 0x1A539A5D },
                        test_4_chksum_fx16{ 0x5131F06A }, test_4_chksum_sa8{ 0xF77795E6 };

#elif defined(CRC_RM_CONVERGENT)
static const crc32_calc test_1_chksum_fx16{ 0x2A705B8D }, test_1_chksum_sa8{ 0x7E3E4789 },
                        test_2_chksum_fx16{ 0x64F8E918 }, test_2_chksum_sa8{ 0xFD31C67A },
                        test_3_chksum_fx16{ 0x5385A814 }, test_3_chksum_sa8{ 0x1A539A5D },
                        test_4_chksum_fx16{ 0x5131F06A }, test_4_chksum_sa8{ 0x1913F4D4 };

#else  // Not defined CRC_*
static const crc32_calc test_1_chksum_fx16, test_1_chksum_sa8,
                        test_2_chksum_fx16, test_2_chksum_sa8,
                        test_3_chksum_fx16, test_3_chksum_sa8,
                        test_4_chksum_fx16, test_4_chksum_sa8,
                        test_5_chksum_fx16, test_5_chksum_sa8;

#endif

static const quality_metrics thresholds_fx16_general { /* MaxAbsErr = */ 0.0004f, quality_metrics::kPassValueSnr,
                                                       /* SNR DB = */ 60.f, quality_metrics::kPassValueQuantErrPerc };

static const quality_metrics thresholds_sa8_general { /* MaxAbsErr = */ 0.008f, quality_metrics::kPassValueSnr,
                                                      /* SNR DB = */ 30.f, quality_metrics::kPassValueQuantErrPerc };

static const tanh_test_operands tests_list[] = {
    // tanh input range [-3:3]
    {"Test 1 FX16 tanh", mli_krn_tanh_fx16,
                                input_1_fx16, test_1_out_fx16,
                                thresholds_fx16_general, test_1_chksum_fx16, false},
    {"Test 1 SA8  tanh", mli_krn_tanh_sa8,
                                input_1_sa8, test_1_out_sa8,
                                thresholds_sa8_general, test_1_chksum_sa8, false},

    // tanh input range [-3:3], memstr
    {"Test 2 FX16 tanh, memstr", mli_krn_tanh_fx16,
                                 input_1_memstr_fx16, test_1_out_memstr_fx16,
                                 thresholds_fx16_general, test_2_chksum_fx16, false},
    {"Test 2 SA8  tanh, memstr", mli_krn_tanh_sa8,
                                 input_1_memstr_sa8, test_1_out_memstr_sa8,
                                 thresholds_sa8_general, test_2_chksum_sa8, false},

    // tanh input range [3:8]
    {"Test 3 FX16 tanh", mli_krn_tanh_fx16,
                                input_2_fx16, test_2_out_fx16,
                                thresholds_fx16_general, test_3_chksum_fx16, false},
    {"Test 3 SA8  tanh", mli_krn_tanh_sa8,
                                input_2_sa8, test_2_out_sa8,
                                thresholds_sa8_general, test_3_chksum_sa8, false},

    // tanh input range [-3:3], In Place Computation
    {"Test 4 FX16 tanh IPC", mli_krn_tanh_fx16,
                                    input_1_fx16, test_1_out_fx16,
                                    thresholds_fx16_general, test_4_chksum_fx16, true},
    {"Test 4 SA8  tanh IPC", mli_krn_tanh_sa8,
                                    input_1_sa8, test_1_out_sa8,
                                    thresholds_sa8_general, test_4_chksum_sa8, true},
};

static constexpr int kMemSize = 2048;
static IO_DATA_ATTR int8_t scratch_mem_in[kMemSize]  = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_lut[kMemSize] = { 0 };

static constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Kernels|Basic tanh Functions Tests");
    mli_lut lut;
    bool lut_status = true;
    int lut_size = mli_krn_tanh_get_lut_size();
    lut_status = lut_status && (lut_size < sizeof(scratch_mem_lut));
    lut.data.mem.void_p = (void*) scratch_mem_lut;
    lut.data.capacity = sizeof(scratch_mem_lut);
    lut_status = lut_status && (mli_krn_tanh_create_lut(&lut) == MLI_STATUS_OK);
    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        bool is_test_passed = true;
        const tanh_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metics;
        if (!(lut_status)) {
            reporter.report_message(cur_test->descr, "FAILED at init: LUT error");
            is_test_passed = false;
        }

        if (!(cur_test->in.is_valid() && cur_test->out.is_valid())) {
            reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
            is_test_passed = false;
        }

        mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
        mli_tensor out;
        if (cur_test->in_place_comp) {
            out = input;
        } else {
            out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
        }

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
                cur_test->mli_krn_tanh(&input, &lut, &out) != MLI_STATUS_OK) {
            reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
            is_test_passed = false;
        }

        if (is_test_passed &&
                (mem_in_keeper.is_memory_corrupted() || mem_out_keeper.is_memory_corrupted())) {
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

        // Check that kernel output quantization parameters are same as for input (according spec).
        if (is_test_passed) {
            bool is_per_tensor_quant = true;

            if (out.el_type == MLI_EL_FX_16) {
                is_test_passed &= out.el_params.fx.frac_bits == kOutFx16FracBits;
            } else if (out.el_type == MLI_EL_FX_8) {
                is_test_passed &= out.el_params.fx.frac_bits == kOutFx8FracBits;
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

    reporter.report_outline("[AUTO] Group: mli_krn_tanh", final_status);

    return (final_status) ? 0 : 1;
}
