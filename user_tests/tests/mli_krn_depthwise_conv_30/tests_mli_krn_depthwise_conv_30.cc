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

#include "vectors_mli_krn3_depthwise_conv.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;
using mli::tst::scales_calc;
using mli::tst::bias_folder;

namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;

typedef mli_status(*depthwise_conv2d_func_ptr)(
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

struct depthwise_conv2d_test_operands {
  const char* descr;
  const rescale_func_ptr mli_krn_rescale;
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
const crc32_calc test_1_chksum_w8_i8{ 0xE003BF6F }, test_2_chksum_w8_i8{ 0x83E6D68F },
                 test_3_chksum_w8_i8{ 0xF60E84E7 }, test_4_chksum_w8_i8{ 0xF91C5358 },
                 test_5_chksum_w8_i8{ 0xE5D1757A }, test_8_chksum_w8_i8{ 0x36C5E229 },
                 test_10_chksum_w8_i8{ 0xD95AFE49 };

const quality_metrics thresholds_sa8_general{quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                       /* SNR_DB = */33.f, quality_metrics::kPassValueQuantErrPerc };


// Test Cases
//==================================================================
static const depthwise_conv2d_test_operands tests_list[] = {
  // Basic functionality test: kernel_size=(3, 4), strides=(1, 1), with krn_padding, w/o ReLU
  // input layout: HWCN, kernel layout: HWCo, output layout: HWCoN
  {"Test 1 8x8 SA ", mli_krn_rescale_i32_o8, input_1_sa8, weights_1_sa8_per_axis, bias_1_sa32_per_axis,
                     test_1_bias_out_sa8, test_1_out_acc_sa32, test_1_out_sa8, input_1_scale, test_1_out_scale,
                     weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                     test_1_cfg, thresholds_sa8_general, test_1_chksum_w8_i8},

  // Basic functionality test: kernel_size=(4, 3), strides=(2, 2), with krn_padding, with Gen_ReLU
  {"Test 2 8x8 SA ReluGen", mli_krn_rescale_i32_o8, input_1_sa8, weights_2_sa8_per_axis, bias_2_i1_w2_sa32_per_axis,
                            test_2_bias_out_sa8, test_2_out_acc_sa32, test_2_out_sa8, input_1_scale, test_2_out_scale,
                            weights_2_scales, sizeof(weights_2_scales) / sizeof(weights_2_scales[0]),
                            test_2_cfg, thresholds_sa8_general, test_2_chksum_w8_i8},

  // Dilation test: kernel_size=(3, 4), strides=(1, 1), w/o padding, w/o ReLU
  {"Test 3 8x8 SA Dil", mli_krn_rescale_i32_o8, input_1_sa8, weights_1_sa8_per_axis, bias_1_sa32_per_axis,
                        test_3_bias_out_sa8, test_3_out_acc_sa32, test_3_out_sa8, input_1_scale, test_3_out_scale,
                        weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                        test_3_cfg, thresholds_sa8_general, test_3_chksum_w8_i8},

  // Input/output memstride test: kernel_size=(3, 4), strides=(3, 3), w/o padding, with ReLU_1
  {"Test 4 8x8 SA Relu1 Mstr", mli_krn_rescale_i32_o8, input_1b_memstr_sa8, weights_1_sa8_per_axis, bias_1_sa32_per_axis,
                               test_4_bias_out_sa8, test_4_out_acc_sa32, test_4_out_sa8, input_1_scale, test_4_out_scale,
                               weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                               test_4_cfg, thresholds_sa8_general, test_4_chksum_w8_i8},

  // Weights memstride test: kernel_size=(8, 6), strides=(1, 1), w/o padding, with ReLU_6
  {"Test 5 8x8 SA Relu6 Mstr", mli_krn_rescale_i32_o8, input_1_sa8, weights_3_sa8_per_axis, bias_2_i1_w3_sa32_per_axis,
                               test_5_bias_out_sa8, test_5_out_acc_sa32, test_5_out_sa8, input_1_scale, test_5_out_scale,
                               weights_3_scales, sizeof(weights_3_scales) / sizeof(weights_3_scales[0]),
                               test_5_cfg, thresholds_sa8_general, test_5_chksum_w8_i8},

  // Dilation test with padding for generic function, kernel_size=(3, 3), strides=(1, 1),
  // krn_padding , dilation = (2,2) and ReLU_Gen.
  // No Dilation ratio. Memstrides are applied on input, output and weights tensors
  {"Test 8 8x8 SA Dil+Pad", mli_krn_rescale_i32_o8, input_1_memstr_sa8, weights_3_sa8_per_axis, bias_2_i1_w3_sa32_per_axis,
                            test_8_bias_out_sa8, test_8_out_acc_sa32, test_8_out_sa8, input_1_scale, test_8_out_scale,
                            weights_3_scales, sizeof(weights_3_scales) / sizeof(weights_3_scales[0]),
                            test_8_cfg, thresholds_sa8_general, test_8_chksum_w8_i8},

  // Test with huge values in operands to check negative fractional and big scales
  {"Test 10 8x8 SA Huge Vals", mli_krn_rescale_i32_o8, input_2_sa8, weights_5_sa8, bias_3_i2_w5_sa32,
                               test_10_bias_out_sa8, test_10_out_acc_sa32, test_10_out_sa8, input_2_scale, test_10_out_scale,
                               weights_5_scales, sizeof(weights_5_scales) / sizeof(weights_5_scales[0]),
                               test_10_cfg, thresholds_sa8_general, test_10_chksum_w8_i8},
};
constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

// Global Memory Memagement
//==================================================================
constexpr uint32_t kMemSize = 2047;
constexpr int kMemAccSize = kMemSize*sizeof(int32_t); // TODO: for double wide accu, more space might be required
static IO_DATA_ATTR int8_t g_scratch_mem_in[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t g_scratch_mem_acc_out[kMemAccSize] = { 0 };
static IO_DATA_ATTR int8_t g_scratch_mem_out[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t g_scratch_mem_bias_out[kMemSize] = { 0 };
static W_DATA_ATTR int8_t g_scratch_mem_w[kMemSize] = { 0 };
static W_DATA_ATTR int8_t g_scratch_mem_b[kMemSize] = { 0 };
static int8_t g_mem_pool[kMemSize] = {0};

struct DepthwiseConvOp {
  // Depthwise Conv2d Kernel
  DepthwiseConvOp(const depthwise_conv2d_test_operands* cur_test) {
    mem_in_keeper = memory_manager((int8_t*)(g_scratch_mem_in), sizeof(g_scratch_mem_in));
    mem_w_keeper = memory_manager((int8_t*)(g_scratch_mem_w), sizeof(g_scratch_mem_w));
    mem_out_acc_keeper = memory_manager((int8_t*)(g_scratch_mem_acc_out), sizeof(g_scratch_mem_acc_out));

    input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
    weights = cur_test->weights.get_quantized_tensor(mem_w_keeper.allocate_memory(cur_test->weights));
    out_acc = cur_test->out_acc.get_not_quantized_tensor(mem_out_acc_keeper.allocate_memory(cur_test->out_acc));
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
  uint32_t out_mem_offset{0};

  // depthwise conv runtime instnace
  void* depthwise_conv2d_instance{nullptr};
  uint32_t depthwise_conv2d_instance_size{0};

  // depthwise conv private data
  lib_mli::PrivateData* depthwise_conv2d_conf_private{nullptr};
  uint32_t depthwise_conv2d_conf_private_size{0};
};

struct RescaleOp {
  // Rescale Kernel
  RescaleOp(const depthwise_conv2d_test_operands* cur_test, const mli_tensor& input, const mli_tensor& weights) {
    mem_b_keeper = memory_manager((int8_t*)(g_scratch_mem_b), sizeof(g_scratch_mem_b));
    mem_bias_out_keeper =  memory_manager ((int8_t*)(g_scratch_mem_bias_out), sizeof(g_scratch_mem_bias_out));
    mem_out_keeper =  memory_manager ((int8_t*)(g_scratch_mem_out), sizeof(g_scratch_mem_out));

    bias_in = cur_test->bias_in.get_quantized_tensor(mem_b_keeper.allocate_memory(cur_test->bias_in));
    bias_out = cur_test->bias_out.get_quantized_tensor(mem_bias_out_keeper.allocate_memory(cur_test->bias_out));
    out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));

    original_bias_out = bias_out;
    original_out = out;

    // additional params for MLI3 Symantic
    mli3_bias = bias_folder(bias_in, input, weights);
    mli3_scales_keeper = scales_calc(cur_test->in_scale, cur_test->out_scale,
                                     cur_test->w_scales, cur_test->w_scales_size);

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

  // additional params for MLI3 Symantic
  bias_folder mli3_bias;
  scales_calc mli3_scales_keeper;
};

bool preprocess_phase(const reporter_full& reporter,
                      const depthwise_conv2d_test_operands* cur_test,
                      const DepthwiseConvOp& dwc_op, const RescaleOp& rs_op) {
    bool is_test_passed = true;

    if (!(cur_test->in.is_valid() && cur_test->weights.is_valid() &&
        cur_test->bias_in.is_valid() && cur_test->bias_out.is_valid() && cur_test->out.is_valid())) {
      reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
      is_test_passed = false;
    }

    if (is_test_passed &&
        (tensor_quantizer::validate_tensor(dwc_op.input) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(dwc_op.weights) != tensor_quantizer::kOk||
         tensor_quantizer::validate_tensor(dwc_op.out_acc) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(rs_op.bias_out) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(rs_op.mli3_bias.get_bias_tsr()) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(rs_op.mli3_scales_keeper.get_scales_tsr()) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(rs_op.mli3_scales_keeper.get_shift_tsr()) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(rs_op.out) != tensor_quantizer::kOk)) {
      reporter.report_message(cur_test->descr,
                  "FAILED at quantization step: more memory for one of tensors might be required");
      is_test_passed = false;
    }

    if (is_test_passed &&
        (dwc_op.mem_in_keeper.is_memory_corrupted() || rs_op.mem_out_keeper.is_memory_corrupted() ||
         dwc_op.mem_out_acc_keeper.is_memory_corrupted() || rs_op.mem_bias_out_keeper.is_memory_corrupted() ||
         dwc_op.mem_w_keeper.is_memory_corrupted() || rs_op.mem_b_keeper.is_memory_corrupted())) {
      reporter.report_message(cur_test->descr,
        "FAILED at quantization step: memory beside one of operands is corrupted");
      is_test_passed = false;
    }

    return is_test_passed;
}

void prepare_phase(const depthwise_conv2d_test_operands* cur_test,
                   uint32_t& out_mem_offset, DepthwiseConvOp& op) {
  // STEP 1.1: Construct Depthwise Conv2d as a specific ExecutionInterface successor
  //==================================================================

  // NHWCin vs. HWCin
  uint32_t input_shape[4] = {1, op.input.shape[0], op.input.shape[1], op.input.shape[2]};
  int32_t input_stride[4] = {int32_t(op.input.shape[0]) * op.input.mem_stride[0],
                             op.input.mem_stride[0],
                             op.input.mem_stride[1],
                             op.input.mem_stride[2]};

  // HWCo vs. HW1Co
  uint32_t weight_shape[3] = {op.weights.shape[0], op.weights.shape[1], op.weights.shape[2] * op.weights.shape[3]};
  int32_t weight_stride[3] = {op.weights.mem_stride[0], op.weights.mem_stride[1],
                              op.weights.mem_stride[2] * op.weights.mem_stride[3]};

  // NHWCo vs. HWCo
  uint32_t output_shape[4] = {1, op.out_acc.shape[0], op.out_acc.shape[1], op.out_acc.shape[2]};

  int32_t output_stride[4] = {int32_t(op.out_acc.shape[0]) * op.out_acc.mem_stride[0],
                              op.out_acc.mem_stride[0],
                              op.out_acc.mem_stride[1],
                              op.out_acc.mem_stride[2]};

  // Cin=Co
  assert(input_shape[3] == output_shape[3]);
  assert(weight_shape[2] == output_shape[3]);

  const lib_mli::Tensor<lib_mli::NoBuffer, 4> in_tensor(input_shape, input_stride);
  const lib_mli::Tensor<lib_mli::NoBuffer, 4> out_tensor(output_shape, output_stride);
  const lib_mli::Tensor<lib_mli::NoBuffer, 3> wt_tensor(weight_shape, weight_stride);

  lib_mli::PlatformDescription pd;
  lib_ref::KernelsFactory kernel_factory(pd);
  uint32_t dw_conv2d_cs_size = kernel_factory.DepthwiseConv2d_CS_GetSize();
  void* dw_conv2d_cs_buffer = malloc(dw_conv2d_cs_size);

  lib_mli::DwConv2DConfig cfg(
    cur_test->cfg.stride_height, cur_test->cfg.stride_width,
    cur_test->cfg.padding_top, cur_test->cfg.padding_left,
    cur_test->cfg.padding_bottom, cur_test->cfg.padding_right,
    cur_test->cfg.dilation_height, cur_test->cfg.dilation_width
  );

  auto dw_conv2d_op = kernel_factory.DepthwiseConv2d_CS(
    dw_conv2d_cs_buffer, in_tensor, wt_tensor, cfg, out_tensor);

  // STEP 1.2: Memory management (Up to user on how to deal with it)
  //==================================================================
  uint32_t in_mem_offset = 0;
  uint32_t w_mem_offset = 0;
  uint32_t inpzp_mem_offset = 0;
  uint32_t wtszp_mem_offset = 0;
  uint32_t offsets[1] = {0};

  // Define buffers for in\out tensors
  // Leave space for runtime object
  uint32_t* offset = &offsets[0];
  uint32_t runtime_obj_size = dw_conv2d_op->GetRuntimeObjectSize();
  *offset += runtime_obj_size;

  // Leave space for private data buffer
  offset = &offsets[0];
  uint32_t private_buffer_size = dw_conv2d_op->GetKernelPrivateDataSize();
  *offset += private_buffer_size;

  // depthwise conv2d input
  offset = &offsets[0];
  uint32_t in_size = dw_conv2d_op->GetInputBufferSize() * sizeof(int8_t);
  lib_mli::OffsetBuffer dw_conv2d_in_buf{*offset, 0, in_size, sizeof(int8_t)};
  lib_mli::Tensor<lib_mli::OffsetBuffer, 4> dw_conv2d_in_tensor(dw_conv2d_in_buf, input_shape);
  in_mem_offset = *offset;
  *offset += in_size;

  // depthwise conv2d weight
  offset = &offsets[0];
  uint32_t w_size = dw_conv2d_op->GetWeightsBufferSize() * sizeof(int8_t);
  lib_mli::OffsetBuffer dw_conv2d_w_buf{*offset, 0, w_size, sizeof(int8_t)};
  w_mem_offset = *offset;
  *offset += w_size;

  // depthwise conv2d output
  offset = &offsets[0];
  uint32_t out_size = dw_conv2d_op->GetOutputBufferSize() * sizeof(int32_t);
  lib_mli::OffsetBuffer dw_conv2d_out_buf{*offset, 0, out_size, sizeof(int32_t)};
  lib_mli::Tensor<lib_mli::OffsetBuffer, 4> dw_conv2d_out_tensor(dw_conv2d_out_buf, output_shape);
  out_mem_offset = *offset;
  *offset += out_size;

  // depthwise conv2d input zero point
  offset = &offsets[0];
  uint32_t inpzp_size = dw_conv2d_op->GetEncodedInpZeroPtsSize() * sizeof(int16_t);
  lib_mli::OffsetBuffer dw_inpzp_buf{*offset, 0, inpzp_size, sizeof(int16_t)};
  inpzp_mem_offset = *offset;
  *offset += inpzp_size;

  // depthwise conv2d weights zero point
  offset = &offsets[0];
  uint32_t wtszp_size = dw_conv2d_op->GetEncodedWtsZeroPtsSize() * sizeof(int16_t);
  lib_mli::OffsetBuffer dw_wtszp_buf{*offset, 0, wtszp_size, sizeof(int16_t)};
  wtszp_mem_offset = *offset;
  *offset += wtszp_size;

  // MLI tensor structures and depthwise conv2d configuration
  offset = &offsets[0];
  uint32_t data_buffer_size = dw_conv2d_op->GetDataBufferSize();
  lib_mli::OffsetBuffer dw_conv2d_descr_buf{*offset, 0, data_buffer_size, sizeof(char)};
  *offset += data_buffer_size;

  // Attaching buffer (descriptors) to the operation
  mli_status status = MLI_STATUS_OK;

  status = dw_conv2d_op->AttachBufferOffsets(dw_conv2d_in_tensor,
                                             dw_conv2d_out_tensor,
                                             dw_conv2d_w_buf,
                                             dw_inpzp_buf,
                                             dw_wtszp_buf,
                                             dw_conv2d_descr_buf);
  assert(status == MLI_STATUS_OK);

  // STEP 1.3: Copy dataset from scratch buffer to the global shared memory pool
  //==================================================================
  // Copy input data from scratch buffer to the shared memory pool
  for (uint32_t i = 0; i < op.input.data.capacity; ++i) {
    const uint32_t idx = in_mem_offset + i;
    g_mem_pool[idx] = op.input.data.mem.pi8[i];
  }
  // Copy weights from scratch buffer to the shaped memory pool (EncodeWeights is not supported)
  for (uint32_t i = 0; i < op.weights.data.capacity; ++i) {
    const uint32_t idx = w_mem_offset + i;
    g_mem_pool[idx] = op.weights.data.mem.pi8[i];
  }

  // Copy input zero points and weights zero points to the temp host buffers
  //==================================================================
  size_t shared_buf_size = std::max(inpzp_size, wtszp_size) * sizeof(int16_t);
  char host_buf_a[shared_buf_size];
  char host_buf_b[shared_buf_size];
  lib_mli::Buffer src_inpzp_buf(host_buf_a, inpzp_size, sizeof(int16_t));
  lib_mli::Buffer dst_inpzp_buf(host_buf_b, inpzp_size, sizeof(int16_t));
  lib_mli::Buffer src_wtszp_buf(host_buf_a, wtszp_size, sizeof(int16_t));
  lib_mli::Buffer dst_wtszp_buf(host_buf_b, wtszp_size, sizeof(int16_t));
  assert(src_inpzp_buf.get_size() == dw_inpzp_buf.get_size());
  assert(src_inpzp_buf.get_elem_size() == dw_inpzp_buf.get_elem_size());
  assert(src_wtszp_buf.get_size() == dw_wtszp_buf.get_size());
  assert(src_wtszp_buf.get_elem_size() == dw_wtszp_buf.get_elem_size());

  uint32_t inpzp_shape[1] = {1};
  lib_mli::Tensor<lib_mli::Buffer, 1> inpzp_tensor(src_inpzp_buf, inpzp_shape);

  uint32_t wtszp_shape[1] = {weight_shape[2]};
  lib_mli::Tensor<lib_mli::Buffer, 1> wtszp_tensor(dst_wtszp_buf, wtszp_shape);

  // input zero points: mli_tensor -> host tensor
  if (op.input.el_params.sa.dim == -1) {
    assert(op.input.el_params.sa.zero_point.capacity == 0);
    inpzp_tensor.write(0, op.input.el_params.sa.zero_point.mem.i16);
  } else {
    assert(op.input.el_params.sa.zero_point.capacity == src_inpzp_buf.get_size());
    for (size_t i = 0; i < inpzp_size; ++i) {
      inpzp_tensor.write(int(i), op.input.el_params.sa.zero_point.mem.pi16[i]);
    }
  }
  // host tensor -> encoded host buffer
  status = dw_conv2d_op->EncodeInpZeroPts(inpzp_tensor, dst_inpzp_buf);
  assert(status == MLI_STATUS_OK);
  // encoded host buffer -> global mem pool
  auto inpzp_mem = reinterpret_cast<int16_t*>(g_mem_pool + inpzp_mem_offset);
  for (size_t i = 0; i < inpzp_size; ++i) {
    inpzp_mem[i] = dst_inpzp_buf.read<int16_t>(i);
  }

  // weights zero points: mli_tensor -> host buffer
  if (op.weights.el_params.sa.dim == -1) {
    assert(op.weights.el_params.sa.zero_point.capacity == 0);
    wtszp_tensor.write(0, op.weights.el_params.sa.zero_point.mem.i16);
  } else {
    assert(op.weights.el_params.sa.zero_point.capacity == src_wtszp_buf.get_size());
    for (size_t i = 0; i < wtszp_size; ++i) {
      wtszp_tensor.write(int(i), op.weights.el_params.sa.zero_point.mem.pi16[i]);
    }
  }
  // host tensor -> encoded host buffer
  status = dw_conv2d_op->EncodeWtsZeroPts(wtszp_tensor, dst_wtszp_buf);
  assert(status == MLI_STATUS_OK);
  auto wtszp_mem = reinterpret_cast<int16_t*>(g_mem_pool + wtszp_mem_offset);
  // encoded host buffer -> global mem pool
  for (size_t i = 0; i < wtszp_size; ++i) {
    wtszp_mem[i] = dst_wtszp_buf.read<int16_t>(i);
  }

  // STEP 1.4: Compile depthwise conv2d into the binary data
  //==================================================================
  op.depthwise_conv2d_instance = g_mem_pool;
  op.depthwise_conv2d_instance_size = dw_conv2d_op->GetRuntimeObjectSize();

  status =
      dw_conv2d_op->GetKernelPrivateData(g_mem_pool + op.depthwise_conv2d_instance_size);
  assert(status == MLI_STATUS_OK);
  op.depthwise_conv2d_conf_private = reinterpret_cast<lib_mli::PrivateData*>(
      g_mem_pool + op.depthwise_conv2d_instance_size);
  op.depthwise_conv2d_conf_private_size = dw_conv2d_op->GetKernelPrivateDataSize();
}

void execution_phase(DepthwiseConvOp& op) {
  // STEP 3: Execution phase
  //==================================================================
  uint32_t tiles_num = 1;

  uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool)};

  auto mli_depthwise_conv = lib_mli::ExecutionInterface::Create(
    op.depthwise_conv2d_instance, op.depthwise_conv2d_instance_size,
    op.depthwise_conv2d_conf_private, op.depthwise_conv2d_conf_private_size,
    membasis, sizeof(membasis) / sizeof(membasis[0]));

  assert(mli_depthwise_conv != nullptr);

  mli_status status = MLI_STATUS_OK;
  for (int i = 0; i < tiles_num; ++i) {
    status = mli_depthwise_conv->Prefetch();
    assert(status == MLI_STATUS_OK);

    status = mli_depthwise_conv->Issue();
    assert(status == MLI_STATUS_OK);

    status = mli_depthwise_conv->Update();
    assert(status == MLI_STATUS_OK);
  }
}

bool postprocess_phase(const reporter_full& reporter,
                       const depthwise_conv2d_test_operands* cur_test,
                       DepthwiseConvOp& dwc_op, RescaleOp& rs_op) {
  quality_metrics test_metrics;
  bool is_test_passed = true;

  auto& out = rs_op.out;
  mli_tensor source_out_tensor = rs_op.original_out;

  if (is_test_passed &&
      (dwc_op.mem_in_keeper.is_memory_corrupted() || rs_op.mem_out_keeper.is_memory_corrupted() ||
       dwc_op.mem_out_acc_keeper.is_memory_corrupted() || rs_op.mem_bias_out_keeper.is_memory_corrupted() ||
       dwc_op.mem_w_keeper.is_memory_corrupted() || rs_op.mem_b_keeper.is_memory_corrupted())) {
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
          (out.el_params.sa.scale_frac_bits.mem.i8 == source_out_tensor.el_params.sa.scale_frac_bits.mem.i8);
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
    data_crc(dwc_op.input);
    data_crc(dwc_op.weights);
    data_crc(rs_op.bias_out);
    // Consider: Adding other tensors (scales/shifts/bias_in, etc). But this test is assumed to be temporary.
    data_crc(out);

    is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr, test_metrics, cur_test->threshold,
                              data_crc, cur_test->check_sum);
  }

  return is_test_passed;
}

int main() {
  const reporter_full reporter;
  reporter.report_header("MLI_3|Kernels|Depthwise Conv Tests");

  bool final_status = true;

  for (int i = 0; i < kTestsNum; ++i) {
    // get the current test case
    const depthwise_conv2d_test_operands* cur_test = &tests_list[i];

#if defined(__Xvec_guard_bit_option)
    // VPX code needs to be debugged
    reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
    continue;
#endif

    // STEP 0: Preprocessing phase
    //==================================================================
    DepthwiseConvOp dwc_op(cur_test);
    RescaleOp rs_op(cur_test, dwc_op.input, dwc_op.weights);

    bool is_test_passed = preprocess_phase(reporter, cur_test, dwc_op, rs_op);

    // STEP 1: Preparing phase
    //==================================================================
    uint32_t out_mem_offset = 0;
    prepare_phase(cur_test, out_mem_offset, dwc_op);

    // STEP 2: Executing phase
    //==================================================================
    // Run depthwise conv2d MLI3.0 kernel
    execution_phase(dwc_op);

    // Get the output of Depthwise Conv2d and copy it to dwc_op.out_acc
    for (uint32_t i = 0; i < dwc_op.out_acc.data.capacity; ++i) {
      dwc_op.out_acc.data.mem.pi8[i] = *(g_mem_pool + out_mem_offset + i);
    }

    // Run rescale kernel
    if (is_test_passed &&
        cur_test->mli_krn_rescale(&dwc_op.out_acc,
                                  &rs_op.mli3_bias.get_bias_tsr(),
                                  &rs_op.mli3_scales_keeper.get_scales_tsr(),
                                  &rs_op.mli3_scales_keeper.get_shift_tsr(),
                                  &rs_op.bias_out, &rs_op.out) != MLI_STATUS_OK) {
      reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
      is_test_passed = false;
    }

    // STEP 3: Postprocessing phase
    //==================================================================
    is_test_passed &= postprocess_phase(reporter, cur_test, dwc_op, rs_op);

    final_status &= is_test_passed;
  }

  reporter.report_outline("[AUTO] Group: mli_krn_depthwise_conv", final_status);

  return (final_status) ? 0 : 1;
}
