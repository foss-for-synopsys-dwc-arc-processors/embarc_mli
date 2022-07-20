/*
 * Copyright 2022, Synopsys, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-3-Clause license found in
 * the LICENSE file in the root directory of this source tree.
 *
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <iostream>

#include "../mli_krn_depthwise_conv_30/vectors_mli_krn3_depthwise_conv.inc"
#include "mli_api.h"
#include "mli_config.h"
#include "mli_types.h"
#include "mli_types.hpp"
#include "mli_kernels_factory_ref.hpp"
#include "mli_runtime_api.hpp"

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "test_rescale_utility.h"
#include "test_tensor_quantizer.h"
#include "test_report.h"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;
using mli::tst::scales_calc;
using mli::tst::bias_folder;
using mli::tst::vectorize_single_elem_tensor;

namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;

struct depthwise_conv2d_test_operands {
    const char* descr;
    tensor_quantizer in;
    tensor_quantizer weights;
    tensor_quantizer bias_in;
    tensor_quantizer bias_out;
    // tensor_quantizer scale;
    // tensor_quantizer shift;
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

// Checksums of test tensors for various mli calculations mode.
// When developer finished implementation of kernel and consider it as ok, one needs to populate
// proper checksums for tests in order to highlight any change which affects results.
//
//Updated CRC values based on Rescale new vectorized tensors (The metrics are the same)
const crc32_calc test_1_chksum_w8_i8 { 0x06BE7BE2 }, test_2_chksum_w8_i8 {
    0xF7EFEE19 }, test_3_chksum_w8_i8 { 0x42B97CC0 }, test_4_chksum_w8_i8 {
        0x637A54CF }, test_5_chksum_w8_i8 { 0x90010A42 }, test_8_chksum_w8_i8 {
            0x519EBD35 }, test_10_chksum_w8_i8 { 0x0FD7DDFE };

const quality_metrics thresholds_sa8_general {
        quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
        /* SNR_DB = */33.f, quality_metrics::kPassValueQuantErrPerc };

// Test Cases
//==================================================================
static const depthwise_conv2d_test_operands tests_list[] = {
// Basic functionality test: kernel_size=(3, 4), strides=(1, 1), with krn_padding, w/o ReLU
// input layout: HWCN, kernel layout: HWCo, output layout: HWCoN
        { "Test 1 SA8_SA8_SA32", input_1_sa8,
                weights_1_sa8_per_axis, bias_1_sa32_per_axis,
                test_1_bias_out_sa8, test_1_out_acc_sa32, test_1_out_sa8,
                input_1_scale, test_1_out_scale, weights_1_scales,
                sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                test_1_cfg, thresholds_sa8_general, test_1_chksum_w8_i8 },

        // Basic functionality test: kernel_size=(4, 3), strides=(2, 2), with krn_padding, with Gen_ReLU
        { "Test 2 SA8_SA8_SA32 ReluGen", input_1_sa8,
                weights_2_sa8_per_axis, bias_2_i1_w2_sa32_per_axis,
                test_2_bias_out_sa8, test_2_out_acc_sa32, test_2_out_sa8,
                input_1_scale, test_2_out_scale, weights_2_scales,
                sizeof(weights_2_scales) / sizeof(weights_2_scales[0]),
                test_2_cfg, thresholds_sa8_general, test_2_chksum_w8_i8 },

        // Dilation test: kernel_size=(3, 4), strides=(1, 1), w/o padding, w/o ReLU
        { "Test 3 SA8_SA8_SA32 Dil", input_1_sa8,
                weights_1_sa8_per_axis, bias_1_sa32_per_axis,
                test_3_bias_out_sa8, test_3_out_acc_sa32, test_3_out_sa8,
                input_1_scale, test_3_out_scale, weights_1_scales,
                sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                test_3_cfg, thresholds_sa8_general, test_3_chksum_w8_i8 },

        // Input/output memstride test: kernel_size=(3, 4), strides=(3, 3), w/o padding, with ReLU_1
        { "Test 4 SA8_SA8_SA32 Relu1 Mstr",
                input_1b_memstr_sa8, weights_1_sa8_per_axis,
                bias_1_sa32_per_axis, test_4_bias_out_sa8, test_4_out_acc_sa32,
                test_4_out_sa8, input_1_scale, test_4_out_scale,
                weights_1_scales, sizeof(weights_1_scales)
                        / sizeof(weights_1_scales[0]), test_4_cfg,
                thresholds_sa8_general, test_4_chksum_w8_i8 },

        // Weights memstride test: kernel_size=(8, 6), strides=(1, 1), w/o padding, with ReLU_6
        { "Test 5 SA8_SA8_SA32 Relu6 Mstr", input_1_sa8,
                weights_3_sa8_per_axis, bias_2_i1_w3_sa32_per_axis,
                test_5_bias_out_sa8, test_5_out_acc_sa32, test_5_out_sa8,
                input_1_scale, test_5_out_scale, weights_3_scales,
                sizeof(weights_3_scales) / sizeof(weights_3_scales[0]),
                test_5_cfg, thresholds_sa8_general, test_5_chksum_w8_i8 },

        // Dilation test with padding for generic function, kernel_size=(3, 3), strides=(1, 1),
        // krn_padding , dilation = (2,2) and ReLU_Gen.
        // No Dilation ratio. Memstrides are applied on input, output and weights tensors
        { "Test 8-1 SA8_SA8_SA32 Dil+Pad", input_1_memstr_sa8,
                weights_3_sa8_per_axis, bias_2_i1_w3_sa32_per_axis,
                test_8_bias_out_sa8, test_8_out_acc_sa32, test_8_out_sa8,
                input_1_scale, test_8_out_scale, weights_3_scales,
                sizeof(weights_3_scales) / sizeof(weights_3_scales[0]),
                test_8_cfg, thresholds_sa8_general, test_8_chksum_w8_i8 },

        // Test with huge values in operands to check negative fractional and big scales
        { "Test 10 SA8_SA8_SA32 Huge Vals", input_2_sa8,
                weights_5_sa8, bias_3_i2_w5_sa32, test_10_bias_out_sa8,
                test_10_out_acc_sa32, test_10_out_sa8, input_2_scale,
                test_10_out_scale, weights_5_scales, sizeof(weights_5_scales)
                        / sizeof(weights_5_scales[0]), test_10_cfg,
                thresholds_sa8_general, test_10_chksum_w8_i8 }, };
constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

// Global Memory Memagement
//==================================================================
constexpr uint32_t kMemSize = 2047;
constexpr int kMemAccSize = kMemSize * sizeof(int32_t); // TODO: for double wide accu, more space might be required
static int8_t g_scratch_mem_in[kMemSize] = { 0 };
static int8_t g_scratch_mem_acc_out[kMemAccSize] = { 0 };
static int8_t g_scratch_mem_out[kMemSize] = { 0 };
static int8_t g_scratch_mem_bias_out[kMemSize] = { 0 };
static int8_t g_scratch_mem_w[kMemSize] = { 0 };
static int8_t g_scratch_mem_b[kMemSize] = { 0 };
constexpr uint32_t kMemPoolSize = 4096;
static IO_DATA_ATTR int8_t g_mem_pool[kMemPoolSize] = { 0 };

struct DepthwiseConvOp {
    // Depthwise Conv2d Kernel
    DepthwiseConvOp(const depthwise_conv2d_test_operands* cur_test) {
        mem_in_keeper = memory_manager((int8_t*) (g_scratch_mem_in),
                sizeof(g_scratch_mem_in));
        mem_w_keeper = memory_manager((int8_t*) (g_scratch_mem_w),
                sizeof(g_scratch_mem_w));
        mem_out_acc_keeper = memory_manager((int8_t*) (g_scratch_mem_acc_out),
                sizeof(g_scratch_mem_acc_out));

        input = cur_test->in.get_quantized_tensor(
                mem_in_keeper.allocate_memory(cur_test->in));
        weights = cur_test->weights.get_quantized_tensor(
                mem_w_keeper.allocate_memory(cur_test->weights));
        out_acc = cur_test->out_acc.get_not_quantized_tensor(
                mem_out_acc_keeper.allocate_memory(cur_test->out_acc));
    }

    // memory memagement for ins & outs tensors
    memory_manager mem_in_keeper;
    memory_manager mem_w_keeper;
    memory_manager mem_out_acc_keeper;

    // ins & outs tensors
    mli_tensor input;
    mli_tensor weights;
    mli_tensor out_acc;

    // the offset of output buffer
    uint32_t out_mem_offset { 0 };

    // depthwise conv runtime instnace
    void* depthwise_conv2d_instance { nullptr};
    uint32_t depthwise_conv2d_instance_size {0};

    // depthwise conv private data
    void* depthwise_conv2d_conf_private {nullptr};
    uint32_t depthwise_conv2d_conf_private_size {0};
};

struct RescaleOp {
    // Rescale Kernel
    RescaleOp(const depthwise_conv2d_test_operands* cur_test,
            const mli_tensor& input, const mli_tensor& weights) {
        mem_b_keeper = memory_manager((int8_t*) (g_scratch_mem_b),
                sizeof(g_scratch_mem_b));
        mem_bias_out_keeper = memory_manager((int8_t*) (g_scratch_mem_bias_out),
                sizeof(g_scratch_mem_bias_out));
        mem_out_keeper = memory_manager((int8_t*) (g_scratch_mem_out),
                sizeof(g_scratch_mem_out));

        bias_in = cur_test->bias_in.get_quantized_tensor(
                mem_b_keeper.allocate_memory(cur_test->bias_in));
        bias_out = cur_test->bias_out.get_quantized_tensor(
                mem_bias_out_keeper.allocate_memory(cur_test->bias_out));
        out = cur_test->out.get_not_quantized_tensor(
                mem_out_keeper.allocate_memory(cur_test->out));

        original_bias_out = bias_out;
        original_out = out;

        // additional params for MLI3 Symantic
        mli3_bias = bias_folder(bias_in, input, weights);
        mli3_scales_keeper = scales_calc(cur_test->in_scale,
                cur_test->out_scale, cur_test->w_scales,
                cur_test->w_scales_size);

    }

    // memory memagement for ins & outs tensors
    memory_manager mem_b_keeper;
    memory_manager mem_bias_out_keeper;
    memory_manager mem_out_keeper;

    // ins & outs tensors
    mli_tensor bias_in;
    mli_tensor bias_out;
    mli_tensor out;

    // original tensors
    mli_tensor original_out;
    mli_tensor original_bias_out;

    // additional params for MLI3 semantic
    bias_folder mli3_bias;
    scales_calc mli3_scales_keeper;

    // additional params for MLI3 runtime
    void* rescale_instance;
    uint32_t rescale_instance_size;
    void* rescale_conf_private;
    uint32_t rescale_conf_private_size;
};

bool preprocess_phase(const reporter_full& reporter,
        const depthwise_conv2d_test_operands* cur_test,
        const DepthwiseConvOp& dwc_op, const RescaleOp& rs_op) {
    bool is_test_passed = true;

    if (!(cur_test->in.is_valid() && cur_test->weights.is_valid()
            && cur_test->bias_in.is_valid() && cur_test->bias_out.is_valid()
            && cur_test->out.is_valid())) {
        reporter.report_message(cur_test->descr,
                "FAILED at init: Bad source data for one of tensors");
        is_test_passed = false;
    }

    if (is_test_passed
            && (tensor_quantizer::validate_tensor(dwc_op.input)
                    != tensor_quantizer::kOk
                    || tensor_quantizer::validate_tensor(dwc_op.weights)
                            != tensor_quantizer::kOk
                    || tensor_quantizer::validate_tensor(dwc_op.out_acc)
                            != tensor_quantizer::kOk
                    || tensor_quantizer::validate_tensor(rs_op.bias_out)
                            != tensor_quantizer::kOk
                    || tensor_quantizer::validate_tensor(
                            rs_op.mli3_bias.get_bias_tsr())
                            != tensor_quantizer::kOk
                    || tensor_quantizer::validate_tensor(
                            rs_op.mli3_scales_keeper.get_scales_tsr())
                            != tensor_quantizer::kOk
                    || tensor_quantizer::validate_tensor(
                            rs_op.mli3_scales_keeper.get_shift_tsr())
                            != tensor_quantizer::kOk
                    || tensor_quantizer::validate_tensor(rs_op.out)
                            != tensor_quantizer::kOk)) {
        reporter.report_message(cur_test->descr,
                "FAILED at quantization step: more memory for one of tensors might be required");
        is_test_passed = false;
    }

    if (is_test_passed
            && (dwc_op.mem_in_keeper.is_memory_corrupted()
                    || rs_op.mem_out_keeper.is_memory_corrupted()
                    || dwc_op.mem_out_acc_keeper.is_memory_corrupted()
                    || rs_op.mem_bias_out_keeper.is_memory_corrupted()
                    || dwc_op.mem_w_keeper.is_memory_corrupted()
                    || rs_op.mem_b_keeper.is_memory_corrupted())) {
        reporter.report_message(cur_test->descr,
                "FAILED at quantization step: memory beside one of operands is corrupted");
        is_test_passed = false;
    }

    return is_test_passed;
}

void prepare_phase(const depthwise_conv2d_test_operands* cur_test,
                    DepthwiseConvOp& dwc_op, RescaleOp &rs_op,
                    uint32_t& dwc_out_mem_offset, uint32_t& rs_out_mem_offset) {
    // STEP 1.1: Construct [Depthwise_Conv2d] as a specific ExecutionInterface successor
    //==================================================================

    // NHWCin vs. HWCin
    uint32_t input_shape[4] = { 1, dwc_op.input.shape[0], dwc_op.input.shape[1],
            dwc_op.input.shape[2] };
    int32_t input_stride[4] = { int32_t(dwc_op.input.shape[0])
            * dwc_op.input.mem_stride[0], dwc_op.input.mem_stride[0],
            dwc_op.input.mem_stride[1], dwc_op.input.mem_stride[2] };

    // HWCo vs. HW1Co
    uint32_t weight_shape[3] = { dwc_op.weights.shape[0],
            dwc_op.weights.shape[1], dwc_op.weights.shape[2]
                    * dwc_op.weights.shape[3] };
    int32_t weight_stride[3] = { dwc_op.weights.mem_stride[0],
            dwc_op.weights.mem_stride[1], dwc_op.weights.mem_stride[2]
                    * dwc_op.weights.mem_stride[3] };

    // NHWCo vs. HWCo
    uint32_t output_shape[4] = { 1, dwc_op.out_acc.shape[0],
            dwc_op.out_acc.shape[1], dwc_op.out_acc.shape[2] };

    int32_t output_stride[4] = { int32_t(dwc_op.out_acc.shape[0])
            * dwc_op.out_acc.mem_stride[0], dwc_op.out_acc.mem_stride[0],
            dwc_op.out_acc.mem_stride[1], dwc_op.out_acc.mem_stride[2] };

    // Cin=Co
    assert(input_shape[3] == output_shape[3]);
    assert(weight_shape[2] == output_shape[3]);

    const lib_mli::Tensor<lib_mli::NoBuffer, 4> in_tensor(input_shape,
                                                            input_stride);
    const lib_mli::Tensor<lib_mli::NoBuffer, 4> out_tensor(output_shape,
                                                            output_stride);
    const lib_mli::Tensor<lib_mli::NoBuffer, 3> wt_tensor(weight_shape,
                                                            weight_stride);

    lib_mli::PlatformDescription pd;
    lib_ref::KernelsFactory kernel_factory(pd);
    uint32_t dw_conv2d_cs_size = kernel_factory.DepthwiseConv2d_CS_GetSize();
    void* dw_conv2d_cs_buffer = malloc(dw_conv2d_cs_size);

    lib_mli::DwConv2DConfig dwc_cfg(cur_test->cfg.stride_height,
            cur_test->cfg.stride_width, cur_test->cfg.padding_top,
            cur_test->cfg.padding_left, cur_test->cfg.padding_bottom,
            cur_test->cfg.padding_right, cur_test->cfg.dilation_height,
            cur_test->cfg.dilation_width);

    auto dw_conv2d_op = kernel_factory.DepthwiseConv2d_CS(dw_conv2d_cs_buffer,
            in_tensor, wt_tensor, dwc_cfg, out_tensor);

    // STEP 1.1: Construct [Rescale] as a specific ExecutionInterface successor
    //==================================================================

    mli_tensor &rs_input_tsr = dwc_op.out_acc;
    const mli_tensor &rs_inbias_tsr = rs_op.mli3_bias.get_bias_tsr();
    const mli_tensor &rs_scale_tsr = rs_op.mli3_scales_keeper.get_scales_tsr();
    const mli_tensor &rs_shift_tsr = rs_op.mli3_scales_keeper.get_shift_tsr();
    mli_tensor &rs_outbias_tsr = rs_op.bias_out;
    mli_tensor &rs_output_tsr = rs_op.out;

    void* &rescale_instance = rs_op.rescale_instance;
    uint32_t &rescale_instance_size = rs_op.rescale_instance_size;
    void* &rescale_conf_private = rs_op.rescale_conf_private;
    uint32_t &rescale_conf_private_size = rs_op.rescale_conf_private_size;

    uint32_t io_rank = rs_input_tsr.rank;
    uint32_t innermost_axis = io_rank - 1;

    const lib_mli::Tensor<lib_mli::NoBuffer, 4> input_tensor(rs_input_tsr.shape,
            rs_input_tsr.mem_stride, io_rank);
    const lib_mli::Tensor<lib_mli::NoBuffer, 4> output_tensor(
                        rs_output_tsr.shape, rs_output_tsr.mem_stride, io_rank);

    uint32_t rescale_cs_size = kernel_factory.Rescale_CS_GetSize();
    void* rescale_cs_buffer = malloc(rescale_cs_size);

    lib_mli::RescaleConfig rs_cfg;
    if (mli_hlp_count_elem_num(&rs_scale_tsr, 0) == 1) {
        rs_cfg.axis = -1;
    } else {
        rs_cfg.axis = innermost_axis;
    }

    auto rescale_op = kernel_factory.Rescale_CS(rescale_cs_buffer, input_tensor,
                                                             rs_cfg, output_tensor);

    // STEP 1.2: [Depthwise_Conv2d] Memory management (Up to user on how to deal with it)
    //==================================================================
    uint32_t in_mem_offset = 0;
    uint32_t w_mem_offset = 0;
    uint32_t inpzp_mem_offset = 0;
    uint32_t wtszp_mem_offset = 0;
    uint32_t offsets[1] = { 0 };

    // NOTE: Currently, only supoort these data types.
    assert(dwc_op.input.el_type == MLI_EL_SA_8);
    assert(dwc_op.weights.el_type == MLI_EL_SA_8);
    assert(dwc_op.out_acc.el_type == MLI_EL_SA_32);

    // Define buffers for in\out tensors
    // Leave space for runtime object
    uint32_t* dwc_offset = &offsets[0];
    uint32_t dwc_runtime_obj_size = dw_conv2d_op->GetRuntimeObjectSize();
    *dwc_offset += dwc_runtime_obj_size;

    // Leave space for private data buffer
    dwc_offset = &offsets[0];
    uint32_t dwc_private_buffer_size = dw_conv2d_op->GetKernelPrivateDataSize();
    *dwc_offset += dwc_private_buffer_size;

    // depthwise conv2d input
    dwc_offset = &offsets[0];
    uint32_t dwc_i_elem_size = mli_hlp_tensor_element_size(&dwc_op.input);
    uint32_t dwc_in_size = dw_conv2d_op->GetInputBufferSize() * dwc_i_elem_size;
    lib_mli::OffsetBuffer dw_conv2d_in_buf { *dwc_offset, 0, dwc_in_size,
                                                dwc_i_elem_size };
    lib_mli::Tensor<lib_mli::OffsetBuffer, 4>
                                dw_conv2d_in_tensor(dw_conv2d_in_buf, input_shape);
    in_mem_offset = *dwc_offset;
    *dwc_offset += dwc_in_size;

    // depthwise conv2d weight
    dwc_offset = &offsets[0];
    uint32_t dwc_w_elem_size = mli_hlp_tensor_element_size(&dwc_op.weights);
    uint32_t w_size = dw_conv2d_op->GetWeightsBufferSize() * dwc_w_elem_size;
    lib_mli::OffsetBuffer dw_conv2d_w_buf { *dwc_offset, 0, w_size,
                                                dwc_w_elem_size };
    w_mem_offset = *dwc_offset;
    *dwc_offset += w_size;

    // depthwise conv2d output
    dwc_offset = &offsets[0];
    // NOTE: The output should be aligned, otherwise, it will cause `vvst` crash.
    //       For example, offset is 4 byts aligned if output is int32_t.
    uint32_t dwc_o_elem_size = mli_hlp_tensor_element_size(&dwc_op.out_acc);
    *dwc_offset = CEIL_RND(*dwc_offset, dwc_o_elem_size);
    uint32_t dwc_out_size = dw_conv2d_op->GetOutputBufferSize()
            * dwc_o_elem_size;
    lib_mli::OffsetBuffer dw_conv2d_out_buf { *dwc_offset, 0, dwc_out_size,
                                                    dwc_o_elem_size };
    lib_mli::Tensor<lib_mli::OffsetBuffer, 4>
                      dw_conv2d_out_tensor(dw_conv2d_out_buf, output_shape);
    dwc_out_mem_offset = *dwc_offset;
    *dwc_offset += dwc_out_size;

    // depthwise conv2d input zero point
    dwc_offset = &offsets[0];
    uint32_t zp_elem_size = sizeof(int16_t);
    uint32_t inpzp_size =
            dw_conv2d_op->GetEncodedInpZeroPtsSize() * zp_elem_size;
    lib_mli::OffsetBuffer dw_inpzp_buf { *dwc_offset, 0, inpzp_size,
                                            zp_elem_size };
    inpzp_mem_offset = *dwc_offset;
    *dwc_offset += inpzp_size;

    // depthwise conv2d weights zero point
    dwc_offset = &offsets[0];
    uint32_t wtszp_size = dw_conv2d_op->GetEncodedWtsZeroPtsSize()
            * zp_elem_size;
    lib_mli::OffsetBuffer dw_wtszp_buf { *dwc_offset, 0, wtszp_size,
                                            zp_elem_size };
    wtszp_mem_offset = *dwc_offset;
    *dwc_offset += wtszp_size;

    // DataBuffer size is 0 for reference kernel
    dwc_offset = &offsets[0];
    uint32_t dwc_data_buffer_size = dw_conv2d_op->GetDataBufferSize();
    lib_mli::OffsetBuffer dw_conv2d_descr_buf { *dwc_offset, 0,
                                    dwc_data_buffer_size, sizeof(char) };
    *dwc_offset += dwc_data_buffer_size;

    // Attaching buffer (descriptors) to the operation
    mli_status status = MLI_STATUS_OK;

    status = dw_conv2d_op->AttachBufferOffsets(dw_conv2d_in_tensor,
                                                dw_conv2d_out_tensor,
                                                dw_conv2d_w_buf,
                                                dw_inpzp_buf,
                                                dw_wtszp_buf,
                                                dw_conv2d_descr_buf);
    assert(status == MLI_STATUS_OK);

    // STEP 1.2: [Rescale] Memory management (Up to user on how to deal with it)
    //==================================================================
    uint32_t encoded_params_mem_offset = 0;

    // Define buffers for in\out tensors
    // Leave space for runtime object
    uint32_t* rs_offset = &offsets[0];
    int8_t* rs_runtime_obj_addr = (int8_t*)g_mem_pool + offsets[0];
    uint32_t rs_runtime_obj_size = rescale_op->GetRuntimeObjectSize();
    *rs_offset += rs_runtime_obj_size;

    // Leave space for private data buffer
    rs_offset = &offsets[0];
    uint32_t rs_private_buffer_size = rescale_op->GetKernelPrivateDataSize();
    *rs_offset += rs_private_buffer_size;

    // rescale input
    uint32_t input_elem_size = mli_hlp_tensor_element_size(&rs_input_tsr);
    uint32_t rs_in_size = rescale_op->GetInputBufferSize() * input_elem_size;
    lib_mli::OffsetBuffer rescale_in_buf { dwc_out_mem_offset, 0, rs_in_size,
                                            input_elem_size };
    lib_mli::Tensor<lib_mli::OffsetBuffer, 4>
                        rescale_in_tensor(rescale_in_buf, rs_input_tsr.shape);

    // rescale output
    rs_offset = &offsets[0];
    uint32_t output_elem_size = mli_hlp_tensor_element_size(&rs_output_tsr);
    uint32_t rs_out_size = rescale_op->GetOutputBufferSize() * output_elem_size;
    lib_mli::OffsetBuffer rescale_out_buf { *rs_offset, 0, rs_out_size,
                                             output_elem_size };
    lib_mli::Tensor<lib_mli::OffsetBuffer, 4>
                       rescale_out_tensor(rescale_out_buf, rs_output_tsr.shape);
    rs_out_mem_offset = *rs_offset;
    *rs_offset += rs_out_size;

    // rescale params
    rs_offset = &offsets[0];
    uint32_t encoded_params_size = rescale_op->GetEncodedParamsSize();
    lib_mli::OffsetBuffer encoded_params_buf { *rs_offset, 0, encoded_params_size,
                                                sizeof(int8_t) };
    encoded_params_mem_offset = *rs_offset;
    *rs_offset += encoded_params_size;

    // DataBuffer size is 0 for reference kernel
    rs_offset = &offsets[0];
    uint32_t rs_data_buffer_size = rescale_op->GetDataBufferSize();
    lib_mli::OffsetBuffer rescale_descr_buf { *rs_offset, 0,
                                            rs_data_buffer_size, sizeof(char) };
    *rs_offset += rs_data_buffer_size;

    // Attaching buffer (descriptors) to the operation
    status = rescale_op->AttachBufferOffsets(rescale_in_tensor,
                                             rescale_out_tensor,
                                             encoded_params_buf,
                                             rescale_descr_buf);
    assert(status == MLI_STATUS_OK);

    // STEP 1.3: [Depthwise_Conv2d] Copy dataset from scratch buffer to the global shared memory pool
    //==================================================================
    // Copy input data from scratch buffer to the shared memory pool
    for (uint32_t i = 0; i < dwc_op.input.data.capacity; ++i) {
        const uint32_t idx = in_mem_offset + i;
        g_mem_pool[idx] = dwc_op.input.data.mem.pi8[i];
    }
    // Copy weights from scratch buffer to the shaped memory pool (EncodeWeights is not supported)
    for (uint32_t i = 0; i < dwc_op.weights.data.capacity; ++i) {
        const uint32_t idx = w_mem_offset + i;
        g_mem_pool[idx] = dwc_op.weights.data.mem.pi8[i];
    }

    // Copy input zero points and weights zero points to the temp host buffers
    //==================================================================
    size_t shared_buf_size = MAX(inpzp_size, wtszp_size);
    char * host_buf_a = (char *) malloc(shared_buf_size);
    char * host_buf_b = (char *) malloc(shared_buf_size);
    lib_mli::Buffer src_inpzp_buf(host_buf_a, inpzp_size, dwc_i_elem_size);
    lib_mli::Buffer dst_inpzp_buf(host_buf_b, inpzp_size, zp_elem_size);
    lib_mli::Buffer src_wtszp_buf(host_buf_a, wtszp_size, dwc_w_elem_size);
    lib_mli::Buffer dst_wtszp_buf(host_buf_b, wtszp_size, zp_elem_size);
    // NOTE: Current the input and weights are int8_t, and zp is int16_t.
    //       Later, we will support other types.
    assert(src_inpzp_buf.get_size() == dw_inpzp_buf.get_size());
    assert(src_inpzp_buf.get_elem_size() * 2 == dw_inpzp_buf.get_elem_size());
    assert(src_wtszp_buf.get_size() == dw_wtszp_buf.get_size());
    assert(src_wtszp_buf.get_elem_size() * 2 == dw_wtszp_buf.get_elem_size());

    uint32_t inpzp_shape[1] = {1};
    lib_mli::Tensor<lib_mli::Buffer, 1> inpzp_tensor(src_inpzp_buf, inpzp_shape);

    uint32_t wtszp_shape[1] = {weight_shape[2]};
    lib_mli::Tensor<lib_mli::Buffer, 1> wtszp_tensor(src_wtszp_buf, wtszp_shape);

    // input zero points: mli_tensor -> host tensor
    // NOTE: Zero Points should have the same size as the tensor they belong to.
    //       Since ZP is 16b in `mli_tensor`, so we should cast it to the same type as input.
    if (dwc_op.input.el_params.sa.dim == -1) {
        assert(dwc_op.input.el_params.sa.zero_point.capacity == 0);
        inpzp_tensor.write(0, static_cast<int8_t>(dwc_op.input.el_params.sa.zero_point.mem.i16));
    } else {
        assert(dwc_op.input.el_params.sa.zero_point.capacity == src_inpzp_buf.get_size());
        for (size_t i = 0; i < inpzp_size / zp_elem_size; ++i) {
          inpzp_tensor.write(int(i), static_cast<int8_t>(dwc_op.input.el_params.sa.zero_point.mem.pi16[i]));
        }
    }
    // host tensor 8bit -> encoded host buffer 16bit
    status = dw_conv2d_op->EncodeInpZeroPts(inpzp_tensor, dst_inpzp_buf);
    assert(status == MLI_STATUS_OK);
    // encoded host buffer -> global mem pool
    auto inpzp_mem = reinterpret_cast<int16_t*>((int8_t*)g_mem_pool + inpzp_mem_offset);
    for (size_t i = 0; i < inpzp_size / zp_elem_size; ++i) {
        inpzp_mem[i] = dst_inpzp_buf.read<int16_t>(i);
    }

    // weights zero points: mli_tensor -> host buffer
    if (dwc_op.weights.el_params.sa.dim == -1) {
        assert(dwc_op.weights.el_params.sa.zero_point.capacity == 0);
        wtszp_tensor.write(0, static_cast<int8_t>(dwc_op.weights.el_params.sa.zero_point.mem.i16));
    } else {
        assert(dwc_op.weights.el_params.sa.zero_point.capacity == src_wtszp_buf.get_size());
        for (size_t i = 0; i < wtszp_size / zp_elem_size; ++i) {
            wtszp_tensor.write(int(i), static_cast<int8_t>(dwc_op.weights.el_params.sa.zero_point.mem.pi16[i]));
        }
    }
    // host tensor -> encoded host buffer
    status = dw_conv2d_op->EncodeWtsZeroPts(wtszp_tensor, dst_wtszp_buf);
    assert(status == MLI_STATUS_OK);
    auto wtszp_mem = reinterpret_cast<int16_t*>((int8_t*)g_mem_pool + wtszp_mem_offset);
    // encoded host buffer -> global mem pool
    for (size_t i = 0; i < wtszp_size / zp_elem_size; ++i) {
        wtszp_mem[i] = dst_wtszp_buf.read<int16_t>(i);
    }

    // Compile depthwise conv2d into the binary data
    //==================================================================
    dwc_op.depthwise_conv2d_instance = (int8_t*)g_mem_pool;
    dwc_op.depthwise_conv2d_instance_size =
            dw_conv2d_op->GetRuntimeObjectSize();

    status = dw_conv2d_op->GetKernelPrivateData(
            (int8_t*)g_mem_pool + dwc_op.depthwise_conv2d_instance_size);
    assert(status == MLI_STATUS_OK);

    dwc_op.depthwise_conv2d_conf_private = (int8_t*)g_mem_pool
                    + dwc_op.depthwise_conv2d_instance_size;
    dwc_op.depthwise_conv2d_conf_private_size =
            dw_conv2d_op->GetKernelPrivateDataSize();

    // STEP 1.3: [Rescale] Copy dataset from tensors to the global shared memory pool
    //==================================================================
    int8_t * host_src_buf = (int8_t *) malloc(encoded_params_size);
    int8_t * host_dst_buf = (int8_t *) malloc(encoded_params_size);
    uint32_t params_shape[1] = {rs_inbias_tsr.shape[0]};

    uint32_t inbias_elem_size = mli_hlp_tensor_element_size(&rs_inbias_tsr);
    uint32_t scale_elem_size = mli_hlp_tensor_element_size(&rs_scale_tsr);
    uint32_t shift_elem_size = mli_hlp_tensor_element_size(&rs_shift_tsr);
    uint32_t outbias_elem_size = mli_hlp_tensor_element_size(&rs_outbias_tsr);
    uint32_t inbias_size = inbias_elem_size * mli_hlp_count_elem_num(&rs_inbias_tsr, 0);
    uint32_t scale_size = scale_elem_size * mli_hlp_count_elem_num(&rs_scale_tsr, 0);
    uint32_t shift_size = shift_elem_size * mli_hlp_count_elem_num(&rs_shift_tsr, 0);
    uint32_t outbias_size = outbias_elem_size * mli_hlp_count_elem_num(&rs_outbias_tsr, 0);

    lib_mli::Buffer src_inbias_buf(host_src_buf,
            inbias_size, inbias_elem_size);
    lib_mli::Buffer src_scale_buf(host_src_buf + inbias_size,
            scale_size, scale_elem_size);
    lib_mli::Buffer src_shift_buf(host_src_buf + inbias_size + scale_size,
            shift_size, shift_elem_size);
    lib_mli::Buffer src_outbias_buf(host_src_buf + inbias_size + scale_size + shift_size,
            outbias_size, outbias_elem_size);

    lib_mli::Buffer encoded_params_buffer(host_dst_buf, encoded_params_size, sizeof(int8_t));

    lib_mli::Tensor<lib_mli::Buffer,1> inbias_tensor(src_inbias_buf, params_shape);
    lib_mli::Tensor<lib_mli::Buffer,1> scale_tensor(src_scale_buf, params_shape);
    lib_mli::Tensor<lib_mli::Buffer,1> shift_tensor(src_shift_buf, params_shape);
    lib_mli::Tensor<lib_mli::Buffer,1> outbias_tensor(src_outbias_buf, params_shape);

    if(rs_cfg.axis < 0) { // per-tensor
        assert(rs_inbias_tsr.rank == 0);
        assert(rs_scale_tsr.rank == 0);
        assert(rs_shift_tsr.rank == 0);
        assert(rs_outbias_tsr.rank == 0);
        inbias_tensor.write<int32_t>(0, rs_inbias_tsr.data.mem.i32);
        scale_tensor.write<int16_t>(0, rs_scale_tsr.data.mem.i16);
        shift_tensor.write<int8_t>(0, rs_shift_tsr.data.mem.i8);
        outbias_tensor.write<int8_t>(0, rs_outbias_tsr.data.mem.i8);
    } else { // per-axis
        for(uint32_t i = 0; i< (inbias_size/inbias_elem_size); i++) {
            inbias_tensor.write<int32_t>(i, rs_inbias_tsr.data.mem.pi32[i]);
        }
        for(uint32_t i = 0; i< (scale_size/scale_elem_size); i++) {
            scale_tensor.write<int16_t>(i, rs_scale_tsr.data.mem.pi16[i]);
        }
        for(uint32_t i = 0; i< (shift_size/shift_elem_size); i++) {
            shift_tensor.write<int8_t>(i, rs_shift_tsr.data.mem.pi8[i]);
        }
        for(uint32_t i = 0; i< (outbias_size/outbias_elem_size); i++) {
            outbias_tensor.write<int8_t>(i, rs_outbias_tsr.data.mem.pi8[i]);
        }
    }

    // host tensors -> encoded host buffer
    status = rescale_op->EncodeParams(inbias_tensor,
                                      scale_tensor,
                                      shift_tensor,
                                      outbias_tensor,
                                      encoded_params_buffer);
    assert(status == MLI_STATUS_OK);

    // encoded host buffer -> global mem pool
    for (uint32_t i = 0; i < encoded_params_size; ++i) {
        const uint32_t idx = encoded_params_mem_offset + i;
        g_mem_pool[idx] = encoded_params_buffer.read<int8_t>(i);
    }

    // Copy output data(including holes due to CRC calculation) to the shared memory pool
    for (uint32_t i = 0; i < rs_out_size; ++i) {
        const uint32_t idx = rs_out_mem_offset + i;
        g_mem_pool[idx] = rs_output_tsr.data.mem.pi8[i];
    }

    // Compile Rescale into the binary data
    //==================================================================
    rescale_instance = rs_runtime_obj_addr;
    rescale_instance_size = rescale_op->GetRuntimeObjectSize();
    rescale_conf_private = rs_runtime_obj_addr + rescale_instance_size;
    rescale_conf_private_size = rescale_op->GetKernelPrivateDataSize();

    status = rescale_op->GetKernelPrivateData(rescale_conf_private);
    assert(status == MLI_STATUS_OK);

    free(dw_conv2d_cs_buffer);
    free(rescale_cs_buffer);
    free(host_buf_a);
    free(host_buf_b);
    free(host_src_buf);
    free(host_dst_buf);
}

void execution_phase(DepthwiseConvOp &dwc_op, RescaleOp &rs_op) {
    // STEP 4: Execution phase
    //==================================================================
    uint32_t tiles_num = 1;

    uint64_t membasis[] = { reinterpret_cast<uint64_t>(g_mem_pool) };

    auto mli_depthwise_conv = lib_mli::ExecutionInterface::Create(
                                dwc_op.depthwise_conv2d_instance,
                                dwc_op.depthwise_conv2d_instance_size,
                                dwc_op.depthwise_conv2d_conf_private,
                                dwc_op.depthwise_conv2d_conf_private_size,
                                membasis, sizeof(membasis) / sizeof(membasis[0]));

    auto mli_rescale = lib_mli::ExecutionInterface::Create(
                        rs_op.rescale_instance,
                        rs_op.rescale_instance_size,
                        rs_op.rescale_conf_private,
                        rs_op.rescale_conf_private_size,
                        membasis, sizeof(membasis) / sizeof(membasis[0]));

    assert(mli_depthwise_conv != nullptr);
    assert(mli_rescale != nullptr);

    mli_status status = MLI_STATUS_OK;

    for (int i = 0; i < tiles_num; ++i) {
        status = mli_depthwise_conv->Prefetch();
        assert(status == MLI_STATUS_OK);
        status = mli_depthwise_conv->Issue();
        assert(status == MLI_STATUS_OK);
        status = mli_depthwise_conv->Update();
        assert(status == MLI_STATUS_OK);

        status = mli_rescale->Prefetch();
        assert(status == MLI_STATUS_OK);
        status = mli_rescale->Issue();
        assert(status == MLI_STATUS_OK);
        status = mli_rescale->Update();
        assert(status == MLI_STATUS_OK);
    }
}

bool postprocess_phase(const reporter_full& reporter,
        const depthwise_conv2d_test_operands* cur_test, DepthwiseConvOp& dwc_op,
        RescaleOp& rs_op) {
    quality_metrics test_metrics;
    bool is_test_passed = true;

    auto& out = rs_op.out;
    mli_tensor source_out_tensor = rs_op.original_out;

    if (is_test_passed
            && test_metrics.calculate_metrics(out, cur_test->out) == false) {
        reporter.report_message(cur_test->descr,
                "FAILED at comparison output with reference");
        is_test_passed = false;
    }

    // Check that kernel didn't modify quantization parameters provided by user.
    if (is_test_passed) {
        bool is_per_tensor_quant = true;

        if (out.el_type == MLI_EL_FX_8 || out.el_type == MLI_EL_FX_16) {
            is_test_passed &= out.el_params.fx.frac_bits
                    == source_out_tensor.el_params.fx.frac_bits;
        } else if (out.el_type == MLI_EL_SA_8 || out.el_type == MLI_EL_SA_32) {
            if (out.el_params.sa.dim < 0
                    || source_out_tensor.el_params.sa.dim < 0) {
                is_test_passed &=
                        (out.el_params.sa.scale.mem.i16
                                == source_out_tensor.el_params.sa.scale.mem.i16)
                                && (out.el_params.sa.zero_point.mem.i16
                                        == source_out_tensor.el_params.sa.zero_point.mem.i16)
                                && (out.el_params.sa.scale_frac_bits.mem.i8
                                        == source_out_tensor.el_params.sa.scale_frac_bits.mem.i8);
            } else {
                is_per_tensor_quant = false;
                is_test_passed = false;
            }
        }
        if (!is_test_passed) {
            reporter.report_message(cur_test->descr,
                    is_per_tensor_quant ?
                            "FAILED as element params of output tensor was modified" :
                            "FAILED as per-axis quantization of output tensor isn't supported");
        }
    }

    if (is_test_passed) {
        crc32_calc data_crc;
        data_crc(dwc_op.input);
        data_crc(dwc_op.weights);
        data_crc(rs_op.bias_out);
        // Consider: Adding other tensors (scales/shifts/bias_in, etc). But this test is assumed to be temporary.
        data_crc(out);

        is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr,
                test_metrics, cur_test->threshold, data_crc,
                cur_test->check_sum);
    }

    return is_test_passed;
}

int main() {
    const reporter_full reporter;
    reporter.report_header("MLI3.0|Kernels|Depthwise Conv Tests");

    bool final_status = true;

    for (int i = 0; i < kTestsNum; ++i) {
        // get the current test case
        const depthwise_conv2d_test_operands* cur_test = &tests_list[i];

// NOTE: Copied from `test_mli_krn_depthwise_conv.cc`, since using the same tect vectors.
#if __Xvec_guard_bit_option == 0 && defined(__Xvec_guard_bit_option)
        if (strstr(cur_test->descr, "Test 1 SA8_SA8_SA32") != nullptr ||
                strstr(cur_test->descr, "Test 2 SA8_SA8_SA32 ReluGen") != nullptr ||
                strstr(cur_test->descr, "Test 5 SA8_SA8_SA32 Relu6 Mstr") != nullptr ||
                strstr(cur_test->descr, "Test 6 SA8_SA8_SA32 k3x3 Spec") != nullptr ||
                strstr(cur_test->descr, "Test 7 SA8_SA8_SA32") != nullptr ||
                strstr(cur_test->descr, "Test 8-1 SA8") != nullptr ||
                strstr(cur_test->descr, "Test 8-2 SA8") != nullptr ||
                strstr(cur_test->descr, "Test 9 SA8_SA8_SA32 k5x5 Dil") != nullptr ||
                strstr(cur_test->descr, "Test 10 SA8_SA8_SA32 Huge Vals") != nullptr) {
            // VPX fails bitwise comparison with reference .
            reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
            continue;
        }
#endif
#if PLATFORM == V2DSP_XY && defined(CRC_RM_UP)
        if (strstr(cur_test->descr, "Test 1 SA8_SA8_SA32") != nullptr) {
            // Em9d fails comparison with reference in up rounding mode.
            reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
            continue;
        }
#endif

#if PLATFORM == V2DSP_XY && defined(CRC_RM_CONVERGENT)
        if (strstr(cur_test->descr, "Test 9 SA8_SA8_SA32 k5x5 Dil") != nullptr) {
            // Em9d fails bitwise comparison with reference .
            reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
            continue;
        }
#endif

        // STEP 0: Preprocessing phase
        //==================================================================
        DepthwiseConvOp dwc_op(cur_test);
        RescaleOp rs_op(cur_test, dwc_op.input, dwc_op.weights);

        bool is_test_passed = preprocess_phase(reporter, cur_test, dwc_op,
                rs_op);

        //
        // Solution to vectorize Rescale params tensors in case of per-axis
        // computation.
        //
        // All params tensors that have one element, should have rank of 0
        // (including out_bias).
        //
        const mli_tensor& inbias_tsr = rs_op.mli3_bias.get_bias_tsr();
        auto& outbias_tsr = rs_op.bias_out;
        auto& shift_tsr  = (mli_tensor&)rs_op.mli3_scales_keeper.get_shift_tsr();
        auto& scale_tsr  = (mli_tensor&)rs_op.mli3_scales_keeper.get_scales_tsr();
        void *outbias_data = NULL, *shift_data = NULL, *scale_data = NULL;
        {
            int32_t rescale_axis;
            if (mli_hlp_count_elem_num(&scale_tsr, 0) == 1) {
                rescale_axis = -1;
            } else {
                rescale_axis = dwc_op.out_acc.rank - 1;
            }

            // If per-axis computation && out_bias is one element,
            // so construct out_bias tensor as vector the same as other params.
            if((rescale_axis != -1) && (mli_hlp_count_elem_num(&outbias_tsr, 0) == 1)) {
                outbias_data = vectorize_single_elem_tensor(outbias_tsr, inbias_tsr);
            }

            // If per-tensor computation && in_bias is vector,
            // so construct out_bias, shift and scale tensors as vectors the same as in_bias.
            if((rescale_axis == -1) && (mli_hlp_count_elem_num(&inbias_tsr, 0) != 1)) {
                outbias_data = vectorize_single_elem_tensor(outbias_tsr, inbias_tsr);
                shift_data = vectorize_single_elem_tensor(shift_tsr, inbias_tsr);
                scale_data = vectorize_single_elem_tensor(scale_tsr, inbias_tsr);
            }
        }

        // STEP 1: Preparing phase
        //==================================================================
        uint32_t dwc_out_mem_offset = 0;
        uint32_t rs_out_mem_offset = 0;

        prepare_phase(cur_test, dwc_op, rs_op, dwc_out_mem_offset,
                rs_out_mem_offset);

        // STEP 2: Executing phase
        //==================================================================
        // Run depthwise conv2d and Rescale MLI3.0 kernels
        execution_phase(dwc_op, rs_op);

        // Get the output of Rescale and copy it to rs_op.out
        for (uint32_t j = 0; j < rs_op.out.data.capacity; ++j) {
            rs_op.out.data.mem.pi8[j] = *((int8_t*)g_mem_pool + rs_out_mem_offset + j);
        }

        // STEP 3: Postprocessing phase
        //==================================================================
        is_test_passed &= postprocess_phase(reporter, cur_test, dwc_op, rs_op);

        final_status &= is_test_passed;

        // Free buffers for Rescale params
        free(outbias_data);
        free(shift_data);
        free(scale_data);
    }

    reporter.report_outline("[AUTO] Group: mli_krn_depthwise_conv_30",
            final_status);

    return (final_status) ? 0 : 1;
}
