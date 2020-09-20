/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_api.h"

#include <stdint.h>
#include <stdio.h>

#include "mli_types.h"
#include "test_metrics.h"
#include "test_quantizer.h"
#include "test_report.h"

#include "vectors_mli_krn_conv2d.inc"

using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;

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

#if defined NO_CRC_CHECk
crc32_calc  test_1_chksum_fx16, test_1_chksum_fx8w16d, test_1_chksum_sa8,
            test_2_chksum_fx16, test_2_chksum_fx8w16d, test_2_chksum_sa8,
            test_3_chksum_fx16, test_3_chksum_fx8w16d, test_3_chksum_sa8,
            test_4_chksum_fx16, test_4_chksum_fx8w16d, test_4_chksum_sa8,
            test_5_chksum_fx16, test_5_chksum_fx8w16d, test_5_chksum_sa8;
#else
// Need to distinquish platforms and update checksums
crc32_calc test_1_chksum_fx16{ 0x3669E8DA }, test_1_chksum_fx8w16d{ 0x627FD168 }, test_1_chksum_sa8{ 0xCDBA5893 },
            test_2_chksum_fx16{ 0x6075722F}, test_2_chksum_fx8w16d{ 0xBFE5DC3D }, test_2_chksum_sa8{ 0x8748D3A9 },
            test_3_chksum_fx16{ 0xE2100158 }, test_3_chksum_fx8w16d{ 0x550F135E }, test_3_chksum_sa8{ 0x53FABFE8 },
            test_4_chksum_fx16, test_4_chksum_fx8w16d, test_4_chksum_sa8,
            test_5_chksum_fx16, test_5_chksum_fx8w16d, test_5_chksum_sa8;
#endif
static const conv2d_test_operands tests_list[] = {
    {"Test 1 FX16",         mli_krn_conv2d_hwcn_fx16, 
                            input_1_fx16, weights_1_fx16, bias_1_fx16, test_1_out_fx16, test_1_cfg,
                            quality_metrics(), test_1_chksum_fx16},
    {"Test 1 FX16_FX8_FX8", mli_krn_conv2d_hwcn_fx16_fx8_fx8, 
                            input_1_fx16, weights_1_fx8, bias_1_fx8, test_1_out_fx16, test_1_cfg, 
                            quality_metrics(), test_1_chksum_fx8w16d},
    {"Test 1 SA8_SA8_SA32", mli_krn_conv2d_hwcn_sa8_sa8_sa32,
                            input_1_sa8, weights_1_sa8, bias_1_sa32, test_1_out_sa8, test_1_cfg, 
                            quality_metrics(quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr, 
                                            /* SNR_DB = */30.f, quality_metrics::kPassValueQuantErrPerc),
                            test_1_chksum_sa8},

    {"Test 2 FX16 ReluGen",         mli_krn_conv2d_hwcn_fx16, input_1_fx16, weights_2_fx16, bias_1_fx16,
                                    test_2_out_fx16, test_2_cfg, quality_metrics(), test_2_chksum_fx16},
    {"Test 2 FX16_FX8_FX8 ReluGen", mli_krn_conv2d_hwcn_fx16_fx8_fx8, input_1_fx16, weights_2_fx8, bias_1_fx8,
                                    test_2_out_fx16, test_2_cfg, quality_metrics(), test_2_chksum_fx8w16d},
    {"Test 2 SA8_SA8_SA32 ReluGen", mli_krn_conv2d_hwcn_sa8_sa8_sa32, input_1_sa8, weights_2_sa8, bias_1_w2_sa32, test_2_out_sa8, test_2_cfg,
                                    quality_metrics(quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr, 30.f, quality_metrics::kPassValueQuantErrPerc),
                                    test_2_chksum_sa8},

    {"Test 3 FX16 Dilation",         mli_krn_conv2d_hwcn_fx16, input_1_fx16, weights_1_fx16, bias_1_fx16, test_3_out_fx16, test_3_cfg,
                                    quality_metrics(), test_3_chksum_fx16},
    {"Test 3 FX16_FX8_FX8 Dilation", mli_krn_conv2d_hwcn_fx16_fx8_fx8, input_1_fx16, weights_1_fx8, bias_1_fx8, test_3_out_fx16, test_3_cfg,
                                    quality_metrics(), test_3_chksum_fx8w16d},
    {"Test 3 SA8_SA8_SA32 Dilation", mli_krn_conv2d_hwcn_sa8_sa8_sa32, input_1_sa8, weights_1_sa8, bias_1_sa32, test_3_out_sa8, test_3_cfg,
                                    quality_metrics(quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr, 30.f, quality_metrics::kPassValueQuantErrPerc),
                                    test_3_chksum_sa8},

    {"Test 4 FX16 InMemstr",         mli_krn_conv2d_hwcn_fx16, input_1_memstr_fx16, weights_1_fx16, bias_1_fx16, test_4_out_fx16, test_4_cfg,
                                    quality_metrics(), test_4_chksum_fx16},
    {"Test 4 FX16_FX8_FX8 InMemstr", mli_krn_conv2d_hwcn_fx16_fx8_fx8, input_1_memstr_fx16, weights_1_fx8, bias_1_fx8, test_4_out_fx16, test_4_cfg,
                                    quality_metrics(), test_4_chksum_fx8w16d},
    {"Test 4 SA8_SA8_SA32 InMemstr", mli_krn_conv2d_hwcn_sa8_sa8_sa32, input_1_memstr_sa8, weights_1_sa8, bias_1_sa32, test_4_out_sa8, test_4_cfg,
                                    quality_metrics(quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr, 30.f, quality_metrics::kPassValueQuantErrPerc),
                                    test_4_chksum_sa8},

    {"Test 5 FX16 InMemstr",         mli_krn_conv2d_hwcn_fx16, input_1_fx16, weights_2_memstr_fx16, bias_1_fx16, test_5_out_fx16, test_5_cfg,
                                    quality_metrics(), test_5_chksum_fx16},
    {"Test 5 FX16_FX8_FX8 InMemstr", mli_krn_conv2d_hwcn_fx16_fx8_fx8, input_1_fx16, weights_2_memstr_fx8, bias_1_fx8, test_5_out_fx16, test_5_cfg,
                                    quality_metrics(), test_5_chksum_fx8w16d},
    {"Test 5 SA8_SA8_SA32 InMemstr", mli_krn_conv2d_hwcn_sa8_sa8_sa32, input_1_sa8, weights_2_memstr_sa8, bias_1_w2_sa32, test_5_out_sa8, test_5_cfg,
                                    quality_metrics(quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr, 30.f, quality_metrics::kPassValueQuantErrPerc),
                                    test_5_chksum_sa8},
};

constexpr int kMemSize = 2047;
static int8_t scratch_mem_in[kMemSize] = { 0 };
static int8_t scratch_mem_out[kMemSize] = { 0 };
static int8_t scratch_mem_w[kMemSize] = { 0 };
static int8_t scratch_mem_b[kMemSize] = { 0 };

static mli_data_container container_in = { kMemSize, {.pi8 = scratch_mem_in + 1} };
static mli_data_container container_out = { kMemSize, {.pi8 = scratch_mem_out + 3} };
static mli_data_container container_w = { kMemSize, {.pi8 = scratch_mem_w + 1} };
static mli_data_container container_b = { kMemSize, {.pi8 = scratch_mem_b + 3} };

static const int tests_num = sizeof(tests_list) / sizeof(tests_list[0]);

int main() {
    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI|Kernels|Convolution 2D  Tests");
    for (int i = 0; i < tests_num; ++i) {
        bool is_test_passed = true;
        const conv2d_test_operands* cur_test = &tests_list[i];
        quality_metrics test_metics;
        if (!(cur_test->in.is_valid() && cur_test->weights.is_valid() &&
                cur_test->bias.is_valid() && cur_test->out.is_valid())) {
            reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
            is_test_passed = false;
        }

        mli_tensor input = cur_test->in.get_quantized_tensor(container_in);
        mli_tensor weights = cur_test->weights.get_quantized_tensor(container_w);
        mli_tensor bias = cur_test->bias.get_quantized_tensor(container_b);
        mli_tensor out = cur_test->out.get_not_quantized_tensor(container_out);
        if (is_test_passed &&
                (tensor_quantizer::validate_tensor(input) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(weights) != tensor_quantizer::kOk||
                 tensor_quantizer::validate_tensor(bias) != tensor_quantizer::kOk ||
                 tensor_quantizer::validate_tensor(out) != tensor_quantizer::kOk)) {
            reporter.report_message(cur_test->descr, 
                                    "FAILED at quantization step: more memory for one of tensors might be required");
            is_test_passed = false;
        }

        // Run specific for test function
        if (is_test_passed &&
                cur_test->mli_krn_conv2d(&input, &weights, &bias, &cur_test->cfg, &out) != MLI_STATUS_OK) {
            reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
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
