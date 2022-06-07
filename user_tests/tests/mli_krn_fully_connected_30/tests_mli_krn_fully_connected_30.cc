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
#include "mli_private_types.h"

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "test_rescale_utility.h"
#include "test_tensor_quantizer.h"
#include "test_report.h"
#include "vectors_mli_krn_fully_connected.inc"

// #include "common/mli_krn_fully_connected.h"

using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;
using mli::tst::scales_calc;
using mli::tst::bias_folder;

namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;

typedef mli_status(*fully_connected_func_ptr)(
    const mli_tensor* /*input*/,
    const mli_tensor* /*weights*/,
    const mli_tensor* /*bias*/,
    const mli_fully_connected_cfg* /*cfg*/,
    mli_tensor* /*output*/);


typedef mli_status(*rescale_func_ptr)(
  const mli_tensor* /*input*/,
  const mli_tensor* /*bias_in*/,
  const mli_tensor* /*scale*/,
  const mli_tensor* /*shift*/,
  const mli_tensor* /*bias_out*/,
  mli_tensor* /*output*/);

struct fully_connected_test_operands {
  const char* descr;
  const fully_connected_func_ptr mli_krn_fully_connected;
  const rescale_func_ptr mli_krn_rescale;
  tensor_quantizer in;
  tensor_quantizer weights;
  tensor_quantizer bias_in;
  tensor_quantizer bias_out;
  tensor_quantizer out_acc;
  tensor_quantizer out;
  const float in_scale;
  const float out_scale;
  const float* w_scales;
  const size_t w_scales_size;
  const mli_fully_connected_cfg cfg;
  const quality_metrics threshold;
  const crc32_calc check_sum;
  bool is_spec{false};
};

// Checksums of test tensors for various mli calculations mode.
// When developer finished implementation of kernel and consider it as ok, one needs to populate
// proper checksums for tests in order to highlight any change which affects results.
#if defined(CRC_RM_CONVERGENT) || defined(CRC_RM_UP)
// Shared CRC Results

const crc32_calc test_1_chksum_fx16{ 0x933AC67B }, test_1_chksum_fx16_fx8_fx8{ 0x73D433B0 }, test_1_chksum_sa8{ 0x313DB9AC },
                 test_2_chksum_fx16{ 0xDD365A8B }, test_2_chksum_fx16_fx8_fx8{ 0xE7CFF930 }, test_2_chksum_sa8{ 0xC60B29FF },
                 test_3_chksum_fx16{ 0xB5E17BAF }, test_3_chksum_fx16_fx8_fx8{ 0xD1D009B6 }, test_3_chksum_sa8{ 0xDA985432 },
                 test_4_chksum_fx16{ 0x4BCFDBF2 }, test_4_chksum_fx16_fx8_fx8{ 0x923FDE15 }, test_4_chksum_sa8{ 0x33950BC3 },
                 test_5_chksum_fx16{ 0x0231B226 }, test_5_chksum_fx16_fx8_fx8{ 0x0EC859C8 }, test_5_chksum_sa8{ 0xCBDD6577 };

const crc32_calc test_1_chksum_sa8_spec{ 0xD33291C2 }, test_2_chksum_sa8_spec{ 0xF39F7D6F },
                 test_3_chksum_sa8_spec{ 0x5E436805 }, test_4_chksum_sa8_spec{ 0x686E0B8E },
                 test_5_chksum_sa8_spec{ 0x7CDD8ED7 };

#else // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16, test_1_chksum_fx16_fx8_fx8, test_1_chksum_sa8,
                  test_2_chksum_fx16, test_2_chksum_fx16_fx8_fx8, test_2_chksum_sa8,
                  test_3_chksum_fx16, test_3_chksum_fx16_fx8_fx8, test_3_chksum_sa8,
                  test_4_chksum_fx16, test_4_chksum_fx16_fx8_fx8, test_4_chksum_sa8,
                  test_5_chksum_fx16, test_5_chksum_fx16_fx8_fx8, test_5_chksum_sa8;

const crc32_calc test_1_chksum_sa8_spec, test_2_chksum_sa8_spec,
                 test_3_chksum_sa8_spec, test_4_chksum_sa8_spec,
                 test_5_chksum_sa8_spec{ 0x5E436805 };
#endif


const quality_metrics thresholds_fx16_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR_DB = */70.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_fx16_fx8_fx8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                  /* SNR_DB = */30.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                             /* SNR_DB = */35.f, quality_metrics::kPassValueQuantErrPerc };

static const fully_connected_test_operands tests_list[] = {

    // Basic functionality test: with ReLU
    {"Test 1 SA8_SA8_SA32",   mli_krn_fully_connected_sa8_sa8_sa32, mli_krn_rescale_i32_o8,
                              input_1_sa8, weights_1_sa8_per_axis, bias_1_sa32_per_axis,
                              test_1_bias_out_sa8, test_1_out_acc_sa32, test_1_out_sa8,
                              input_1_scale, test_1_out_scale, weights_1_scales_1,
                              sizeof(weights_1_scales_1) / sizeof(weights_1_scales_1[0]),
                              test_1_cfg, thresholds_sa8_general, test_1_chksum_sa8},
    // Basic functionality test: with Gen_ReLU
    {"Test 2 SA8_SA8_SA32 ReluGen", mli_krn_fully_connected_sa8_sa8_sa32, mli_krn_rescale_i32_o8,
                                    input_1_sa8, weights_1_sa8, bias_1_sa32,
                                    test_2_bias_out_sa8, test_2_out_acc_sa32,
                                    test_2_out_sa8,
                                    input_1_scale, test_2_out_scale, &weights_1_scale, 1,
                                    test_2_cfg, thresholds_sa8_general, test_2_chksum_sa8},
    {"Test 2 SA8_SA8_SA32 Spec",    mli_krn_fully_connected_sa8_sa8_sa32_ext_bias, mli_krn_rescale_i32_o8,
                                    input_1_sa8, weights_1_sa8, bias_1_sa32_spec,
                                    test_2_bias_out_sa8, test_2_out_acc_sa32,
                                    test_2_out_sa8,
                                    input_1_scale, test_2_out_scale, &weights_1_scale, 1,
                                    test_2_cfg, thresholds_sa8_general, test_2_chksum_sa8_spec, true},

    // Weights memstride test: with ReLU_1
    {"Test 3 SA8_SA8_SA32 Relu1 Mstr", mli_krn_fully_connected_sa8_sa8_sa32, mli_krn_rescale_i32_o8,
                                       input_1_sa8, weights_2_memstr_sa8_per_axis, bias_2_i1_w2_sa32_per_axis,
                                       test_3_bias_out_sa8, test_3_out_acc_sa32,
                                       test_3_out_sa8,
                                       input_1_scale, test_3_out_scale,
                                       weights_2_scales_2, sizeof(weights_2_scales_2) / sizeof(weights_2_scales_2[0]),
                                       test_3_cfg, thresholds_sa8_general, test_3_chksum_sa8},
    {"Test 3 SA8_SA8_SA32 Spec",       mli_krn_fully_connected_sa8_sa8_sa32_ext_bias, mli_krn_rescale_i32_o8,
                                       input_1_sa8, weights_2_memstr_sa8_per_axis, bias_2_i1_w2_sa32_per_axis_spec,
                                       test_3_bias_out_sa8, test_3_out_acc_sa32,
                                       test_3_out_sa8,
                                       input_1_scale, test_3_out_scale,
                                       weights_2_scales_2, sizeof(weights_2_scales_2) / sizeof(weights_2_scales_2[0]),
                                       test_3_cfg, thresholds_sa8_general, test_3_chksum_sa8_spec, true},

    // Multidimensional input test: with ReLU_6
    {"Test 4 SA8_SA8_SA32 Relu6", mli_krn_fully_connected_sa8_sa8_sa32, mli_krn_rescale_i32_o8,
                                  input_2_sa8, weights_3_sa8_per_axis, bias_3_i2_w3_sa32_per_axis,
                                  test_4_bias_out_sa8, test_4_out_acc_sa32,
                                  test_4_out_sa8,
                                  input_2_scale, test_4_out_scale,
                                  weights_3_scales, sizeof(weights_3_scales) / sizeof(weights_3_scales[0]),
                                  test_4_cfg, thresholds_sa8_general, test_4_chksum_sa8},
    {"Test 4 SA8_SA8_SA32 Spec",  mli_krn_fully_connected_sa8_sa8_sa32_ext_bias, mli_krn_rescale_i32_o8,
                                  input_2_sa8, weights_3_sa8_per_axis, bias_3_i2_w3_sa32_per_axis_spec,
                                  test_4_bias_out_sa8, test_4_out_acc_sa32,
                                  test_4_out_sa8,
                                  input_2_scale, test_4_out_scale,
                                  weights_3_scales, sizeof(weights_3_scales) / sizeof(weights_3_scales[0]),
                                  test_4_cfg, thresholds_sa8_general, test_4_chksum_sa8_spec, true},

    // Test with huge values in operands to check negative fractional and big scales
    {"Test 5 SA8_SA8_SA32 Huge Vals", mli_krn_fully_connected_sa8_sa8_sa32, mli_krn_rescale_i32_o8,
                                      input_3_sa8, weights_4_sa8, bias_4_i3_w4_sa32,
                                      test_5_bias_out_sa8, test_5_out_acc_sa32,
                                      test_5_out_sa8,
                                      input_3_scale, test_5_out_scale,
                                      weights_4_scales, sizeof(weights_4_scales) / sizeof(weights_4_scales[0]),
                                      test_5_cfg, thresholds_sa8_general, test_5_chksum_sa8},
    {"Test 5 SA8_SA8_SA32 Spec", mli_krn_fully_connected_sa8_sa8_sa32_ext_bias, mli_krn_rescale_i32_o8,
                                 input_3_sa8, weights_4_sa8, bias_4_i3_w4_sa32_spec,
                                 test_5_bias_out_sa8, test_5_out_acc_sa32,
                                 test_5_out_sa8,
                                 input_3_scale, test_5_out_scale,
                                 weights_4_scales, sizeof(weights_4_scales) / sizeof(weights_4_scales[0]),
                                 test_5_cfg, thresholds_sa8_general, test_5_chksum_sa8_spec, true},
};
constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

// Global Memory Memagement
//==================================================================
constexpr uint32_t kMemSize = 2047;
constexpr int kMemAccSize = kMemSize*sizeof(int32_t); // TODO: for double wide accu, more space might be required
static int8_t g_scratch_mem_in[kMemSize] = { 0 };
static int8_t g_scratch_mem_acc_out[kMemAccSize] = { 0 };
static int8_t g_scratch_mem_out[kMemSize] = { 0 };
static int8_t g_scratch_mem_bias_out[kMemSize] = { 0 };
static int8_t g_scratch_mem_w[kMemSize] = { 0 };
static int8_t g_scratch_mem_b[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t g_mem_pool[kMemSize] = {0};

struct FullyConnectedOp {
  // Fully Connected Kernel
  FullyConnectedOp(const fully_connected_test_operands* cur_test) {
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

  // fully conncted runtime instnace
  void* fully_connected_instance{nullptr};
  uint32_t fully_connected_instance_size{0};

  // fully conncted private data
  lib_mli::PrivateData* fully_connected_conf_private{nullptr};
  uint32_t fully_connected_conf_private_size{0};
};

struct RescaleOp {
  // Rescale Kernel
  RescaleOp(const fully_connected_test_operands* cur_test, const mli_tensor& input, const mli_tensor& weights) {
    mem_b_keeper = memory_manager((int8_t*)(g_scratch_mem_b), sizeof(g_scratch_mem_b));
    mem_bias_out_keeper =  memory_manager ((int8_t*)(g_scratch_mem_bias_out), sizeof(g_scratch_mem_bias_out));
    mem_out_keeper =  memory_manager ((int8_t*)(g_scratch_mem_out), sizeof(g_scratch_mem_out));

    bias_in = cur_test->bias_in.get_quantized_tensor(mem_b_keeper.allocate_memory(cur_test->bias_in));
    bias_out = cur_test->bias_out.get_quantized_tensor(mem_bias_out_keeper.allocate_memory(cur_test->bias_out));
    out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));

    // Note: The data of bias_out is the zero point of output.
    //  Due to the differenct quantization rules (round) in test compenents, bias_out.data is not the same as output.zero_point.
    //  This issue will causes result comparaison with test vectors failed. So, as a workaround,
    //  here we just assign the zero point of output to bias_out. And it is always a scalar.
    assert(out.el_params.sa.zero_point.capacity == 0);
    assert(bias_out.rank == 0);
    bias_out.data.mem.i8 = out.el_params.sa.zero_point.mem.i8;

    original_bias_out = bias_out;
    original_out = out;

    // additional params for MLI3 Symantic
    if (cur_test->is_spec) {
      mli3_bias = bias_folder(bias_in);
    } else {
      mli3_bias = bias_folder(bias_in, input, weights);
    }
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

void relu(mli_tensor *out, const mli_relu_cfg *cfg) {
    mli_minmax_t val_limit;

    int min_val, max_val;
    int zero, one, neg_one, six;
    int16_t scale;
    int shift;
    {
        MLI_ASSERT((out->el_type == MLI_EL_SA_8 || out->el_type == MLI_EL_SA_32));
        // only per tensor quantization for output tensor supported.
        MLI_ASSERT(out->el_params.sa.dim < 0);
        zero = out->el_params.sa.zero_point.mem.i16;
        scale = out->el_params.sa.scale.mem.i16;
        shift = out->el_params.sa.scale_frac_bits.mem.i8;
    }

    min_val = std::numeric_limits<int8_t>::min();
    max_val = std::numeric_limits<int8_t>::max();

    // In theory it is possible that scale of input is really small value and shift might be bigger than 16 bit to
    // represent six and one in such format before int div (may exceed 32 bits).
    // One and six are not casted to 16bit directly, only after comparison with min_val and max_val and all of them are int.
    // Min val and max val always fit to container range, while six and one don't have to.
    // when six doesn't fit in the container range, it will be clipped to the container range.

    switch (cfg->type) {
    case MLI_RELU_GEN:
        val_limit.min = (int16_t) MAX(zero, min_val);
        val_limit.max = (int16_t) max_val;
        break;
    case MLI_RELU_6:
        if (shift >= 0) {
            six = (shift < 28) ? ((int32_t)6 << shift) / scale : max_val;
        }
        else {
            six = (shift > -3) ?((int32_t)6 >> (-shift)) / scale : 0;
        }

        six = six + zero;
        val_limit.min = (int16_t) MAX(zero, min_val);
        val_limit.max = (int16_t) MIN (six, max_val);
        break;
    case MLI_RELU_1:
        if (shift >= 0) {
            one = (shift < 30) ? ((int32_t)1 << shift) / scale : max_val;
        }
        else {
            one = 0;
        }

        neg_one = -one + zero;
        one = one + zero;
        val_limit.min = (int16_t) MAX(neg_one, min_val);
        val_limit.max = (int16_t) MIN(one, max_val);
        break;
    default:
        // For leaky and param relu there is no saturation in the function domain.
        // only container type limitations (8bit or 16 bit)
        val_limit.min = (int16_t) min_val;
        val_limit.max = (int16_t) max_val;
    }

    int8_t val_min_limit = (int8_t)val_limit.min;
    int8_t val_max_limit = (int8_t)val_limit.max;
    for (size_t i = 0; i < out->data.capacity; ++i) {
      auto result = out->data.mem.pi8[i];
      result = MIN(result, val_max_limit);
      result = MAX(result, val_min_limit);
      out->data.mem.pi8[i] = result;
    }
}

bool preprocess_phase(const reporter_full& reporter,
                      const fully_connected_test_operands* cur_test,
                      const FullyConnectedOp& fc_op, const RescaleOp& rs_op) {
    bool is_test_passed = true;

    if (!(cur_test->in.is_valid() && cur_test->weights.is_valid() &&
        cur_test->bias_in.is_valid() && cur_test->bias_out.is_valid() && cur_test->out.is_valid())) {
      reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
      is_test_passed = false;
    }

    if (is_test_passed &&
        (tensor_quantizer::validate_tensor(fc_op.input) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(fc_op.weights) != tensor_quantizer::kOk||
         tensor_quantizer::validate_tensor(fc_op.out_acc) != tensor_quantizer::kOk ||
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
        (fc_op.mem_in_keeper.is_memory_corrupted() || rs_op.mem_out_keeper.is_memory_corrupted() ||
         fc_op.mem_out_acc_keeper.is_memory_corrupted() || rs_op.mem_bias_out_keeper.is_memory_corrupted() ||
         fc_op.mem_w_keeper.is_memory_corrupted() || rs_op.mem_b_keeper.is_memory_corrupted())) {
      reporter.report_message(cur_test->descr,
        "FAILED at quantization step: memory beside one of operands is corrupted");
      is_test_passed = false;
    }

    return is_test_passed;
}

void prepare_phase(const fully_connected_test_operands* cur_test,
                   uint32_t& out_mem_offset, FullyConnectedOp& op) {
  // STEP 1.1: Construct Fully Connected as a specific ExecutionInterface successor
  //==================================================================

  // NCin vs. Cin'
  uint32_t in_shape = 1;
  for (size_t i = 0; i < op.input.rank; ++i) {
    in_shape *= op.input.shape[i];
  }
  uint32_t input_shape[2] = {1, in_shape};
  int32_t input_stride[2] = {int32_t(op.input.shape[0]) * op.input.mem_stride[0],
                             op.input.mem_stride[0]};

  // CinCo vs. CinCo
  assert(op.weights.rank == 2);
  uint32_t weight_shape[2] = {op.weights.shape[0], op.weights.shape[1]};
  int32_t weight_stride[2] = {op.weights.mem_stride[0], op.weights.mem_stride[1]};

  // NCo vs. Co
  assert(op.out_acc.rank == 1);
  uint32_t output_shape[2] = {1, op.out_acc.shape[0]};

  int32_t output_stride[2] = {int32_t(op.out_acc.shape[0]) * op.out_acc.mem_stride[0],
                              op.out_acc.mem_stride[0]};

  assert(input_shape[0] == output_shape[0]);
  assert(input_shape[1] == weight_shape[0]);
  assert(weight_shape[1] == output_shape[1]);

  const lib_mli::Tensor<lib_mli::NoBuffer, 2> in_tensor(input_shape, input_stride);
  const lib_mli::Tensor<lib_mli::NoBuffer, 2> out_tensor(output_shape, output_stride);
  const lib_mli::Tensor<lib_mli::NoBuffer, 2> wt_tensor(weight_shape, weight_stride);

  lib_mli::PlatformDescription pd;
  lib_ref::KernelsFactory kernel_factory(pd);
  uint32_t fully_connected_cs_size = kernel_factory.FullyConnected_CS_GetSize();
  void* fully_connected_cs_buffer = malloc(fully_connected_cs_size);

  auto fc_op = kernel_factory.FullyConnected_CS(
    fully_connected_cs_buffer, in_tensor, wt_tensor, out_tensor);

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
  uint32_t runtime_obj_size = fc_op->GetRuntimeObjectSize();
  *offset += runtime_obj_size;

  // Leave space for private data buffer
  offset = &offsets[0];
  uint32_t private_buffer_size = fc_op->GetKernelPrivateDataSize();
  *offset += private_buffer_size;

  // fully connected input
  offset = &offsets[0];
  uint32_t in_size = fc_op->GetInputBufferSize() * sizeof(int8_t);
  lib_mli::OffsetBuffer fully_connected_in_buf{*offset, 0, in_size, sizeof(int8_t)};
  lib_mli::Tensor<lib_mli::OffsetBuffer, 2> fully_connected_in_tensor(fully_connected_in_buf, input_shape);
  in_mem_offset = *offset;
  *offset += in_size;

  // fully connected weight
  offset = &offsets[0];
  uint32_t w_size = fc_op->GetWeightsBufferSize() * sizeof(int8_t);
  lib_mli::OffsetBuffer fully_connected_w_buf{*offset, 0, w_size, sizeof(int8_t)};
  w_mem_offset = *offset;
  *offset += w_size;

  // fully connected output
  offset = &offsets[0];
  // The output should be 4 bytes aligned for int32_t, otherwise, it will cause `vvst` crash.
  *offset = (*offsets + 4 - 1) / 4 * 4;
  uint32_t out_size = fc_op->GetOutputBufferSize() * sizeof(int32_t);
  lib_mli::OffsetBuffer fully_connected_out_buf{*offset, 0, out_size, sizeof(int32_t)};
  lib_mli::Tensor<lib_mli::OffsetBuffer, 2> fully_connected_out_tensor(fully_connected_out_buf, output_shape);
  out_mem_offset = *offset;
  *offset += out_size;

  // fully connected input zero point
  offset = &offsets[0];
  uint32_t inpzp_size = fc_op->GetEncodedInpZeroPtsSize() * sizeof(int16_t);
  lib_mli::OffsetBuffer fc_inpzp_buf{*offset, 0, inpzp_size, sizeof(int16_t)};
  inpzp_mem_offset = *offset;
  *offset += inpzp_size;

  // fully connected weights zero point
  offset = &offsets[0];
  uint32_t wtszp_size = fc_op->GetEncodedWtsZeroPtsSize() * sizeof(int16_t);
  lib_mli::OffsetBuffer fc_wtszp_buf{*offset, 0, wtszp_size, sizeof(int16_t)};
  wtszp_mem_offset = *offset;
  *offset += wtszp_size;

  // MLI tensor structures and fully connected configuration
  offset = &offsets[0];
  uint32_t data_buffer_size = fc_op->GetDataBufferSize();
  lib_mli::OffsetBuffer fully_connected_descr_buf{*offset, 0, data_buffer_size, sizeof(char)};
  *offset += data_buffer_size;

  // Attaching buffer (descriptors) to the operation
  mli_status status = MLI_STATUS_OK;

  status = fc_op->AttachBufferOffsets(fully_connected_in_tensor,
                                             fully_connected_out_tensor,
                                             fully_connected_w_buf,
                                             fc_inpzp_buf,
                                             fc_wtszp_buf,
                                             fully_connected_descr_buf);
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

  // Compile fully connected into the binary data
  //==================================================================
  op.fully_connected_instance = (int8_t*)g_mem_pool;
  op.fully_connected_instance_size = fc_op->GetRuntimeObjectSize();

  status =
      fc_op->GetKernelPrivateData((int8_t*)g_mem_pool + op.fully_connected_instance_size);
  assert(status == MLI_STATUS_OK);
  op.fully_connected_conf_private = reinterpret_cast<lib_mli::PrivateData*>(
      (int8_t*)g_mem_pool + op.fully_connected_instance_size);
  op.fully_connected_conf_private_size = fc_op->GetKernelPrivateDataSize();
}

void execution_phase(FullyConnectedOp& op) {
  // STEP 3: Execution phase
  //==================================================================
  uint32_t tiles_num = 1;

  uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool)};

  auto mli_fully_connected = lib_mli::ExecutionInterface::Create(
    op.fully_connected_instance, op.fully_connected_instance_size,
    op.fully_connected_conf_private, op.fully_connected_conf_private_size,
    membasis, sizeof(membasis) / sizeof(membasis[0]));

  assert(mli_fully_connected != nullptr);

  mli_status status = MLI_STATUS_OK;
  for (int i = 0; i < tiles_num; ++i) {
    status = mli_fully_connected->Prefetch();
    assert(status == MLI_STATUS_OK);

    status = mli_fully_connected->Issue();
    assert(status == MLI_STATUS_OK);

    status = mli_fully_connected->Update();
    assert(status == MLI_STATUS_OK);
  }
}

bool postprocess_phase(const reporter_full& reporter,
                       const fully_connected_test_operands* cur_test,
                       FullyConnectedOp& fc_op, RescaleOp& rs_op) {
  quality_metrics test_metrics;
  bool is_test_passed = true;

  auto& out = rs_op.out;
  mli_tensor source_out_tensor = rs_op.original_out;

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
    data_crc(fc_op.input);
    data_crc(fc_op.weights);
    data_crc(rs_op.bias_in);
    // Consider: Adding other tensors (scales/shifts/bias_in, etc). But this test is assumed to be temporary.
    data_crc(out);

    is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr, test_metrics, cur_test->threshold,
                              data_crc, cur_test->check_sum);
  }

  return is_test_passed;
}

int main() {
  const reporter_full reporter;
  reporter.report_header("MLI_3|Kernels|Fully Connected Tests");

  bool final_status = true;

  for (int i = 0; i < kTestsNum; ++i) {
    // get the current test case
    const fully_connected_test_operands* cur_test = &tests_list[i];

#if defined(__Xvec_guard_bit_option) && (__Xvec_guard_bit_option == 0)
        if (strstr(cur_test->descr, "Test 3 SA8_SA8_SA32 Relu1 Mstr") != nullptr ||
                strstr(cur_test->descr, "Test 3 SA8_SA8_SA32 Spec") != nullptr ||
                strstr(cur_test->descr, "Test 4 SA8_SA8_SA32 Relu6") != nullptr ||
                strstr(cur_test->descr, "Test 4 SA8_SA8_SA32 Spec") != nullptr ||
                strstr(cur_test->descr, "Test 5 SA8_SA8_SA32 Huge Vals") != nullptr ||
                strstr(cur_test->descr, "Test 5 SA8_SA8_SA32 Spec") != nullptr) {
            // VPX fails bitwise comparison with reference .
            reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
            continue;
        }
#endif
    // STEP 0: Preprocessing phase
    //==================================================================
    FullyConnectedOp fc_op(cur_test);
    RescaleOp rs_op(cur_test, fc_op.input, fc_op.weights);

    bool is_test_passed = preprocess_phase(reporter, cur_test, fc_op, rs_op);

    // STEP 1: Preparing phase
    //==================================================================
    uint32_t out_mem_offset = 0;
    prepare_phase(cur_test, out_mem_offset, fc_op);

    // STEP 2: Executing phase
    //==================================================================
    // Run fully connected MLI3.0 kernel
    execution_phase(fc_op);

    // Get the output of Fully Connected and copy it to fc_op.out_acc
    for (uint32_t i = 0; i < fc_op.out_acc.data.capacity; ++i) {
      fc_op.out_acc.data.mem.pi8[i] = *((int8_t*)g_mem_pool + out_mem_offset + i);
    }

    // Run rescale kernel
    if (is_test_passed &&
        cur_test->mli_krn_rescale(&fc_op.out_acc,
                                  &rs_op.mli3_bias.get_bias_tsr(),
                                  &rs_op.mli3_scales_keeper.get_scales_tsr(),
                                  &rs_op.mli3_scales_keeper.get_shift_tsr(),
                                  &rs_op.bias_out, &rs_op.out) != MLI_STATUS_OK) {
      reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
      is_test_passed = false;
    }
    if (is_test_passed) {
      // TODO: refactor, reuse relu code
      relu(&rs_op.out, &cur_test->cfg.relu);
    }

#if 0
    // Run Fully Connected MLI2.0 kernel, for debug purpose
    if (is_test_passed && cur_test->mli_krn_fully_connected(
                              &fc_op.input, &fc_op.weights, &rs_op.bias_in,
                              &cur_test->cfg, &rs_op.out) != MLI_STATUS_OK) {
        reporter.report_message(
            cur_test->descr,
            "FAILED at kernel run: kernel returned bad status");
        is_test_passed = false;
    }
    for (size_t i = 0; i < rs_op.out.data.capacity; ++i) {
        std::cout << static_cast<int16_t>(rs_op.out.data.mem.pi8[i]) << ",";
    }
#endif
    // STEP 3: Postprocessing phase
    //==================================================================
    is_test_passed &= postprocess_phase(reporter, cur_test, fc_op, rs_op);

    final_status &= is_test_passed;
  }

  reporter.report_outline("[AUTO] Group: mli_krn_fully_connected", final_status);

  return (final_status) ? 0 : 1;
}
