/*
* Copyright 2022, Synopsys, Inc.
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
#include "test_rescale_utility.h"
#include "mli_types.h"
#include "test_tensor_quantizer.h"
#include "test_report.h"

#include "vectors_mli_krn3_depthwise_conv.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;
using mli::tst::scales_calc;
using mli::tst::bias_folder;

typedef mli_status(*depthwise_conv_func_ptr)(
    const mli_tensor* /*input*/,
    const mli_tensor* /*weights*/,
    const mli_conv2d_cfg* /*cfg*/,
    mli_tensor* /*output*/);

typedef mli_status(*rescale_func_ptr)(
    const mli_tensor* /*input*/,
    const mli_tensor* /*bias_in*/,
    const mli_tensor* /*scale*/,
    const mli_tensor* /*shift*/,
    const mli_tensor* /*bias_out*/,
    mli_tensor* /*output*/);

struct depthwise_conv_test_operands {
    const char* descr;
    const depthwise_conv_func_ptr mli_krn_depthwise_conv;
    const rescale_func_ptr mli_krn_rescale;
    tensor_quantizer in;
    tensor_quantizer weights;
    tensor_quantizer bias_in;
    tensor_quantizer bias_out;
    //tensor_quantizer scale;
    //tensor_quantizer shift;
    tensor_quantizer out_acc;
    tensor_quantizer out;
    const float in_scale;
    const float out_scale;
    const float* w_scales;
    const size_t w_scales_size;
    const mli_conv2d_cfg cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

const crc32_calc test_1_chksum_w8_i8{ 0x697F637B }, test_2_chksum_w8_i8{ 0x98D3DFBE }, test_3_chksum_w8_i8{ 0x4AC0F277 },
                 test_4_chksum_w8_i8{ 0xFF81EADE }, test_5_chksum_w8_i8{ 0x42BA3C5D }, test_8_chksum_w8_i8{ 0x221436B9 },
                 test_10_chksum_w8_i8{ 0x1564A1D1 };

const quality_metrics thresholds_sa8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                             /* SNR_DB = */35.f, quality_metrics::kPassValueQuantErrPerc };


static const depthwise_conv_test_operands tests_list[] = {
    // Basic functionality test: kernel_size=(3, 4), strides=(1, 1), with krn_padding, w/o ReLU
    {"Test 1 8x8 SA ",      mli3_krn_depthwise_conv2d_hwcn_w8_i8_o32, mli_krn_rescale_i32_o8,
                            input_1_sa8, weights_1_sa8_per_axis, bias_1_sa32_per_axis,
                            test_1_bias_out_sa8, test_1_out_acc_sa32, test_1_out_sa8, input_1_scale, test_1_out_scale,
                            weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                            test_1_cfg, thresholds_sa8_general, test_1_chksum_w8_i8},

    // Basic functionality test: kernel_size=(4, 3), strides=(2, 2), with krn_padding, with Gen_ReLU
    {"Test 2 8x8 SA ReluGen", mli3_krn_depthwise_conv2d_hwcn_w8_i8_o32, mli_krn_rescale_i32_o8,
                              input_1_sa8, weights_2_sa8_per_axis, bias_2_i1_w2_sa32_per_axis, test_2_bias_out_sa8,
                              test_2_out_acc_sa32, test_2_out_sa8, input_1_scale, test_2_out_scale,
                              weights_2_scales, sizeof(weights_2_scales) / sizeof(weights_2_scales[0]),
                              test_2_cfg, thresholds_sa8_general, test_2_chksum_w8_i8},

    // Dilation test: kernel_size=(3, 4), strides=(1, 1), w/o padding, w/o ReLU
    {"Test 3 8x8 SA Dil", mli3_krn_depthwise_conv2d_hwcn_w8_i8_o32, mli_krn_rescale_i32_o8,
                          input_1_sa8, weights_1_sa8_per_axis, bias_1_sa32_per_axis, test_3_bias_out_sa8,
                          test_3_out_acc_sa32, test_3_out_sa8, input_1_scale, test_3_out_scale,
                          weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                          test_3_cfg, thresholds_sa8_general, test_3_chksum_w8_i8},
    
    // Input/output memstride test: kernel_size=(3, 4), strides=(3, 3), w/o padding, with ReLU_1 
    {"Test 4 8x8 SA Relu1 Mstr", mli3_krn_depthwise_conv2d_hwcn_w8_i8_o32, mli_krn_rescale_i32_o8,
                                 input_1b_memstr_sa8, weights_1_sa8_per_axis, bias_1_sa32_per_axis, test_4_bias_out_sa8,
                                 test_4_out_acc_sa32, test_4_out_sa8, input_1_scale, test_4_out_scale,
                                 weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                                 test_4_cfg, thresholds_sa8_general, test_4_chksum_w8_i8},
    
    // Weights memstride test: kernel_size=(8, 6), strides=(1, 1), w/o padding, with ReLU_6
    {"Test 5 8x8 SA Relu6 Mstr", mli3_krn_depthwise_conv2d_hwcn_w8_i8_o32, mli_krn_rescale_i32_o8,
                                 input_1_sa8, weights_3_sa8_per_axis, bias_2_i1_w3_sa32_per_axis, test_5_bias_out_sa8,
                                 test_5_out_acc_sa32, test_5_out_sa8, input_1_scale, test_5_out_scale,
                                 weights_3_scales, sizeof(weights_3_scales) / sizeof(weights_3_scales[0]),
                                 test_5_cfg, thresholds_sa8_general, test_5_chksum_w8_i8},

    // Dilation test with padding for generic function, kernel_size=(3, 3), strides=(1, 1),
    // krn_padding , dilation = (2,2) and ReLU_Gen.
    // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 8 8x8 SA Dil+Pad", mli3_krn_depthwise_conv2d_hwcn_w8_i8_o32, mli_krn_rescale_i32_o8,
                              input_1_memstr_sa8, weights_3_sa8_per_axis, bias_2_i1_w3_sa32_per_axis, test_8_bias_out_sa8,
                              test_8_out_acc_sa32, test_8_out_sa8, input_1_scale, test_8_out_scale,
                              weights_3_scales, sizeof(weights_3_scales) / sizeof(weights_3_scales[0]),
                              test_8_cfg, thresholds_sa8_general, test_8_chksum_w8_i8},

    // Test with huge values in operands to check negative fractional and big scales 
    {"Test 10 8x8 SA Huge Vals", mli3_krn_depthwise_conv2d_hwcn_w8_i8_o32, mli_krn_rescale_i32_o8,
                                 input_2_sa8, weights_5_sa8, bias_3_i2_w5_sa32, test_10_bias_out_sa8,
                                 test_10_out_acc_sa32, test_10_out_sa8, input_2_scale, test_10_out_scale,
                                 weights_5_scales, sizeof(weights_5_scales) / sizeof(weights_5_scales[0]),
                                 test_10_cfg, thresholds_sa8_general, test_10_chksum_w8_i8},
};

constexpr int kMemSize = 2047;
constexpr int kMemAccSize = kMemSize*sizeof(int32_t); // TODO: for double wide accu, more space might be required
static IO_DATA_ATTR int8_t scratch_mem_in[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_acc_out[kMemAccSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_out[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t scratch_mem_bias_out[kMemSize] = { 0 };
static W_DATA_ATTR int8_t scratch_mem_w[kMemSize] = { 0 };
static W_DATA_ATTR int8_t scratch_mem_b[kMemSize] = { 0 };

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI_3|Kernels|Depthwise Conv Tests");
    for (int i = 0; i < kTestsNum; ++i) {
        memory_manager mem_in_keeper((int8_t*)(scratch_mem_in), sizeof(scratch_mem_in));
        memory_manager mem_out_acc_keeper((int8_t*)(scratch_mem_acc_out), sizeof(scratch_mem_acc_out));
        memory_manager mem_out_keeper((int8_t*)(scratch_mem_out), sizeof(scratch_mem_out));
        memory_manager mem_bias_out_keeper((int8_t*)(scratch_mem_bias_out), sizeof(scratch_mem_bias_out));
        memory_manager mem_w_keeper((int8_t*)(scratch_mem_w), sizeof(scratch_mem_w));
        memory_manager mem_b_keeper((int8_t*)(scratch_mem_b), sizeof(scratch_mem_b));
        bool is_test_passed = true;
        const depthwise_conv_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metrics;

        if (!(cur_test->in.is_valid() && cur_test->weights.is_valid() &&
                cur_test->bias_in.is_valid() && cur_test->bias_out.is_valid() && cur_test->out.is_valid())) {
            reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
            is_test_passed = false;
        }

        mli_tensor input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
        mli_tensor weights = cur_test->weights.get_quantized_tensor(mem_w_keeper.allocate_memory(cur_test->weights));
        // mli_tensor bias_in = cur_test->bias_in.get_quantized_tensor(mem_b_keeper.allocate_memory(cur_test->bias));
        mli_tensor bias_out = cur_test->bias_out.get_quantized_tensor(mem_bias_out_keeper.allocate_memory(cur_test->bias_out));
        mli_tensor out_acc = cur_test->out_acc.get_not_quantized_tensor(mem_out_acc_keeper.allocate_memory(cur_test->out_acc));
        mli_tensor out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));

        // additional params for MLI3 Symantic
        auto mli3_bias = bias_folder(cur_test->bias_in.get_quantized_tensor(mem_b_keeper.allocate_memory(cur_test->bias_in)),
                                     input, weights);
        auto mli3_scales_keeper = scales_calc(cur_test->in_scale, cur_test->out_scale, 
                                              cur_test->w_scales,cur_test->w_scales_size);
        mli_tensor source_out_tensor = out;
        if (is_test_passed &&
                (tensor_quantizer::validate_tensor(input) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(weights) != tensor_quantizer::kOk||
                 tensor_quantizer::validate_tensor(bias_out) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(mli3_bias.get_bias_tsr()) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(mli3_scales_keeper.get_scales_tsr()) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(mli3_scales_keeper.get_shift_tsr()) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(out_acc) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(out) != tensor_quantizer::kOk)) {
            reporter.report_message(cur_test->descr, 
                                    "FAILED at quantization step: more memory for one of tensors might be required");
            is_test_passed = false;
        }

        if (is_test_passed &&
                (mem_in_keeper.is_memory_corrupted() || mem_out_keeper.is_memory_corrupted() ||
                 mem_out_acc_keeper.is_memory_corrupted() || mem_bias_out_keeper.is_memory_corrupted() ||
                 mem_w_keeper.is_memory_corrupted() || mem_b_keeper.is_memory_corrupted())) {
            reporter.report_message(cur_test->descr,
                "FAILED at quantization step: memory beside one of operands is corrupted");
            is_test_passed = false;
        }

        // Run specific kernel for test
        if (is_test_passed &&
                cur_test->mli_krn_depthwise_conv(&input, &weights, &cur_test->cfg, &out_acc) != MLI_STATUS_OK) {
            reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
            is_test_passed = false;
        }

        if (is_test_passed &&
                cur_test->mli_krn_rescale(&out_acc, &mli3_bias.get_bias_tsr(), &mli3_scales_keeper.get_scales_tsr(),
                                          &mli3_scales_keeper.get_shift_tsr(), &bias_out, &out) != MLI_STATUS_OK) {
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
            data_crc(bias_out);
            // Consider: Adding other tensors (scales/shifts/bias_in, etc). But this test is assumed to be temporary.
            data_crc(out);

            is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr, test_metrics, cur_test->threshold, 
                                                                data_crc, cur_test->check_sum);
        }
        final_status &= is_test_passed;
    }

    reporter.report_outline("[AUTO] Group: mli_krn_depthwise_conv", final_status);

    return (final_status) ? 0 : 1;
}
