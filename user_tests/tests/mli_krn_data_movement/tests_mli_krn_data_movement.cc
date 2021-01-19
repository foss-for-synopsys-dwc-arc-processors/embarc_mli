/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_api.h"

#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "mli_types.h"
#include "mli_api.h"

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "test_report.h"
#include "test_tensor_quantizer.h"

#include "vectors_mli_krn_data_movement.inc"
#include "mli_check.h"

using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;


typedef mli_status (*mov_tensor_sync_ptr)(const mli_tensor* src,
    const mli_mov_cfg_t* cfg,
    mli_tensor* dst
    );


struct data_movement_test_operands {
    const char* descr;
    const mov_tensor_sync_ptr mli_krn_data_movement;
    tensor_quantizer in;
    tensor_quantizer out;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

#if defined(CRC_RM_CONVERGENT) || defined(CRC_RM_UP)
// Shared CRC Results
const crc32_calc  test_1_chksum_fx16{0x967CB076},  test_1_chksum_sa8{0x68021AE1},
                  test_2_chksum_fx16{0xCA635039},  test_2_chksum_sa8{0xC8771FF1},
                  test_3_chksum_fx16{0xFDE1DFCD},  test_3_chksum_sa8{0x58B44570},
                  test_4_chksum_fx16{0x2003D08A},  test_4_chksum_sa8{},
                  test_5_chksum_fx16{0x761AC768},  test_5_chksum_sa8{0xF97A618B},
                  test_6_chksum_fx16{0x9F1D8CD6},  test_6_chksum_sa8{0xBA7A7305},
                  test_7_chksum_fx16{0x5EA45492},  test_7_chksum_sa8{},
                  test_8_chksum_fx16{0x7C4438D5},  test_8_chksum_sa8{0x459A61E0},
                  test_9_chksum_fx16{0xAF45E89C},  test_9_chksum_sa8{},
                  test_10_chksum_fx16{0xD2412339}, test_10_chksum_sa8{0xBC0A3692},
                  test_11_chksum_fx16{0x1E454291}, test_11_chksum_sa8{0xFC387CD3},
                  test_12_chksum_fx16{0x4BE8FE1C}, test_12_chksum_sa8{0x85CA59DE},
                  test_13_chksum_sa32{0x274A232D}, test_14_chksum_sa32{0x8638EA84};

#else // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16{},  test_1_chksum_sa8{},
                  test_2_chksum_fx16{},  test_2_chksum_sa8{},
                  test_3_chksum_fx16{},  test_3_chksum_sa8{},
                  test_4_chksum_fx16{},  test_4_chksum_sa8{},
                  test_5_chksum_fx16{},  test_5_chksum_sa8{},
                  test_6_chksum_fx16{},  test_6_chksum_sa8{},
                  test_7_chksum_fx16{},  test_7_chksum_sa8{},
                  test_8_chksum_fx16{},  test_8_chksum_sa8{},
                  test_9_chksum_fx16{},  test_9_chksum_sa8{},
                  test_10_chksum_fx16{}, test_10_chksum_sa8{},
                  test_11_chksum_fx16{}, test_11_chksum_sa8{},
                  test_12_chksum_fx16{}, test_12_chksum_sa8{},
                  test_13_chksum_sa32{}, test_14_chksum_sa32{};
#endif

const quality_metrics thresholds_fx16_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR DB = */ 30.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                               /* SNR DB = */ 30.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa32_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                               /* SNR DB = */ 30.f, quality_metrics::kPassValueQuantErrPerc };

static const data_movement_test_operands tests_list[] = {
    {"Test 1 FX16 Copy same MemStirde ", mli_mov_tensor_sync, input_1_fx16,
                                        test_1_out_fx16,
                                        thresholds_fx16_general, test_1_chksum_fx16},
    {"Test 1 SA8  Copy same MemStride AXIS = 1", mli_mov_tensor_sync, input_1_sa8,
                                        test_1_out_sa8,
                                        thresholds_sa8_general, test_1_chksum_sa8},
    {"Test 2 FX16 Copy different MemStride", mli_mov_tensor_sync, input_2_fx16,
                                        test_2_out_fx16,
                                        thresholds_fx16_general, test_2_chksum_fx16},
    {"Test 2 SA8 Copy different MemStride", mli_mov_tensor_sync, input_2_sa8,
                                        test_2_out_sa8,
                                        thresholds_sa8_general, test_2_chksum_sa8},
    {"Test 3 FX16 Slice different MemStride", mli_mov_tensor_sync, input_3_fx16,
                                        test_3_out_fx16,
                                        thresholds_fx16_general, test_3_chksum_fx16},
    {"Test 3 SA8 Slice different MemStride", mli_mov_tensor_sync, input_3_sa8,
                                        test_3_out_sa8,
                                        thresholds_sa8_general, test_3_chksum_sa8},
    {"Test 4 FX16 Concat same MemStride", mli_mov_tensor_sync, input_1_fx16,
                                        test_4_out_fx16,
                                        thresholds_fx16_general, test_4_chksum_fx16},
    {"Test 5 FX16 Padding same MemStride", mli_mov_tensor_sync, input_3_fx16,
                                        test_5_out_fx16,
                                        thresholds_fx16_general, test_5_chksum_fx16},
    {"Test 5 SA8 Padding same MemStride", mli_mov_tensor_sync, input_3_sa8,
                                        test_5_out_sa8,
                                        thresholds_sa8_general, test_5_chksum_sa8},
    {"Test 6 FX16 Permute same MemStride", mli_mov_tensor_sync, input_4_fx16,
                                        test_6_out_fx16,
                                        thresholds_fx16_general, test_6_chksum_fx16},
    {"Test 6 SA8 Permute same MemStride", mli_mov_tensor_sync, input_4_sa8,
                                        test_6_out_sa8,
                                        thresholds_sa8_general, test_6_chksum_sa8},
    {"Test 7 FX16 Subsample same MemStride", mli_mov_tensor_sync, input_1_fx16,
                                        test_7_out_fx16,
                                        thresholds_fx16_general, test_7_chksum_fx16},
    {"Test 8 FX16 slice and Padding same MemStr", mli_mov_tensor_sync, input_3_fx16,
                                        test_8_out_fx16,
                                        thresholds_fx16_general, test_8_chksum_fx16},
    {"Test 8 SA8 slice and Padding same MemStr", mli_mov_tensor_sync, input_3_sa8,
                                        test_8_out_sa8,
                                        thresholds_sa8_general, test_8_chksum_sa8},
    {"Test 9 FX16 Permute and Subsample different MemStr", mli_mov_tensor_sync, input_1_fx16,
                                        test_9_out_fx16,
                                        thresholds_fx16_general, test_9_chksum_fx16},
    {"Test 10 FX16 Permute and Subsample and Padding same MemStr", mli_mov_tensor_sync, input_3_fx16,
                                        test_10_out_fx16,
                                        thresholds_fx16_general, test_10_chksum_fx16},
    {"Test 10 SA8 Permute and Subsample and Padding same MemStr", mli_mov_tensor_sync, input_3_sa8,
                                        test_10_out_sa8,
                                        thresholds_sa8_general, test_10_chksum_sa8},
    {"Test 11 FX16 padding and Subsample and slice same MemStr", mli_mov_tensor_sync, input_5_fx16,
                                        test_11_out_fx16,
                                        thresholds_fx16_general, test_11_chksum_fx16},
    {"Test 11 SA8 padding and Subsample and slice same MemStr", mli_mov_tensor_sync, input_5_sa8,
                                        test_11_out_sa8,
                                        thresholds_sa8_general, test_11_chksum_sa8},
    {"Test 12 FX16 padding and Subsample and slice different MemStr in inner dimension", mli_mov_tensor_sync, input_6_fx16,
                                        test_12_out_fx16,
                                        thresholds_fx16_general, test_12_chksum_fx16},
    {"Test 12 SA8 padding and Subsample and slice different MemStr in inner dimension", mli_mov_tensor_sync, input_6_sa8,
                                        test_12_out_sa8,
                                        thresholds_sa8_general, test_12_chksum_sa8},
    {"Test 13 SA32 slice and Subsample same MemStr", mli_mov_tensor_sync, input_7_sa32,
                                        test_13_out_sa32,
                                        thresholds_sa32_general, test_13_chksum_sa32},
    {"Test 14 SA32 slice and Subsample and padding different MemStr", mli_mov_tensor_sync, input_8_sa32,
                                        test_14_out_sa32,
                                        thresholds_sa32_general, test_14_chksum_sa32},

};

constexpr int kMemSize = 2048;
static int8_t scratch_mem_in_outside[kMemSize]  = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_in_inside[kMemSize] = { 0 };
static int8_t scratch_mem_out_outside[kMemSize]  = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out_inside[kMemSize] = { 0 };

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Kernels|Data Movement Functions Tests");
    for (int i = 0; i < kTestsNum; ++i) {

        memory_manager mem_in_keeper(mem[i].in_mem == CCM_MEM ? (int8_t*)scratch_mem_in_inside : (int8_t*)scratch_mem_in_outside,
                mem[i].in_mem == CCM_MEM ? sizeof(scratch_mem_in_inside) : sizeof(scratch_mem_in_outside));
        memory_manager mem_out_keeper(mem[i].out_mem == CCM_MEM ? (int8_t*)scratch_mem_out_inside : (int8_t*)scratch_mem_out_outside,
                mem[i].out_mem == CCM_MEM ? sizeof(scratch_mem_out_inside) : sizeof(scratch_mem_out_outside));

        bool is_test_passed = true;
        const data_movement_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metics;

        if (!(cur_test->in.is_valid() && cur_test->out.is_valid())) {

            reporter.report_message(cur_test->descr, "FAILED at init: bad source data for one of tensors");
            is_test_passed = false;
        }

        mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
        mli_tensor out;
        if (i == 6) {
            out = cur_test->out.get_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));

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
                (mem_in_keeper.is_memory_corrupted() ||
                 mem_out_keeper.is_memory_corrupted())) {
            reporter.report_message(cur_test->descr,
                "FAILED at quantization step: memory beside one of operands is corrupted");
            is_test_passed = false;
        }

        mli_mov_cfg_t cfg;
        mli_mov_cfg_all(&cfg,offsets_cfg[i],sizes_cfg[i],sub_sample[i],out_offsets_cfg[i],out_mem_stride_cfg[i],
                perm_dim[i],padd_left[i],padd_right[i],padd_top[i],padd_bottom[i]);
        // Run specific kernel for test
        mli_status stat = cur_test->mli_krn_data_movement(&input,&cfg, &out);

        if (is_test_passed &&
                stat != MLI_STATUS_OK) {
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

        if (is_test_passed &&
                test_metics.calculate_metrics(out, cur_test->out) == false) {
            reporter.report_message(cur_test->descr, "FAILED at comparison output with reference");
            is_test_passed = false;
        }

        // Check that input and output have the same quantization parameters provided by user.
        if (is_test_passed) {
            bool is_per_tensor_quant = true;

            if (out.el_type == MLI_EL_FX_8 || out.el_type == MLI_EL_FX_16) {
                is_test_passed &= out.el_params.fx.frac_bits == input.el_params.fx.frac_bits;
            } else if (out.el_type == MLI_EL_SA_8 || out.el_type == MLI_EL_SA_32) {
                is_test_passed &= out.el_params.sa.dim == input.el_params.sa.dim;
                if (out.el_params.sa.dim < 0) {
                    is_test_passed &=
                        (out.el_params.sa.scale.mem.i16 == input.el_params.sa.scale.mem.i16) &&
                        (out.el_params.sa.zero_point.mem.i16 == input.el_params.sa.zero_point.mem.i16) &&
                        (out.el_params.sa.scale_frac_bits.mem.i8 ==
                            input.el_params.sa.scale_frac_bits.mem.i8);
                } else {
                    is_test_passed &=
                            !(memcmp(out.el_params.sa.scale.mem.pi16, input.el_params.sa.scale.mem.pi16, input.el_params.sa.scale.capacity) ||
                            memcmp(out.el_params.sa.scale.mem.pi16, input.el_params.sa.scale.mem.pi16, input.el_params.sa.scale.capacity) ||
                            memcmp(out.el_params.sa.scale.mem.pi16, input.el_params.sa.scale.mem.pi16, input.el_params.sa.scale.capacity));

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

        final_status &= (is_test_passed);
    }

    reporter.report_outline("[AUTO] Group: Data Movement", final_status);

    return (final_status) ? 0 : 1;
}
