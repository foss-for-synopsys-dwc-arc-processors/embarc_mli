/*
* Copyright 2019-2021, Synopsys, Inc.
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

#include "vectors_mli_krn_conv2d.inc"


using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;
using mli::tst::scales_calc;
using mli::tst::bias_folder;

namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;

typedef mli_status(*conv2d_func_ptr)(
    const mli_tensor* /*input*/,
    const mli_tensor* /*weights*/,
    const mli_tensor* /*bias*/,
    const mli_conv2d_cfg* /*cfg*/,
    mli_tensor* /*output*/);

typedef mli_status(*rescale_func_ptr)(
  const mli_tensor* /*input*/,
  const mli_tensor* /*bias_in*/,
  const mli_tensor* /*scale*/,
  const mli_tensor* /*shift*/,
  const mli_tensor* /*bias_out*/,
  mli_tensor* /*output*/);

struct conv2d_test_operands {
    const char* descr;
    const conv2d_func_ptr mli_krn_conv2d;
    tensor_quantizer in;
    tensor_quantizer weights;
    tensor_quantizer bias_in;
    tensor_quantizer out;
    tensor_quantizer out_acc;
    tensor_quantizer bias_out;
    const float in_scale;
    const float out_scale;
    const float* w_scales;
    const size_t w_scales_size;
    const mli_conv2d_cfg cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

// Checksums of test tensors for various mli calculations mode.
// When developer finished implementation of kernel and consider it as ok, He need to populate
// proper checksums for tests in order to highlight any change which affects results.

// Shared CRC Results
const crc32_calc test_1_chksum_fx16{ 0x3669E8DA }, test_1_chksum_fx16_fx8_fx8{ 0x627FD168 },
                 test_2_chksum_fx16{ 0x6075722F }, test_2_chksum_fx16_fx8_fx8{ 0xBFE5DC3D }, test_2_chksum_sa8{ 0x5D288208 },
                 test_3_chksum_fx16{ 0xE2100158 }, test_3_chksum_fx16_fx8_fx8{ 0x550F135E }, test_3_chksum_sa8{ 0x9740102D },
                 test_4_chksum_fx16{ 0x987AC0A8 }, test_4_chksum_fx16_fx8_fx8{ 0x21C772CE }, test_4_chksum_sa8{ 0x056EDB56 },
                 test_5_chksum_fx16{ 0xD8CA1273 }, test_5_chksum_fx16_fx8_fx8{ 0x186AA252 }, test_5_chksum_sa8{ 0x01D390FA },
                 test_6_chksum_fx16{ 0x150A5D20 },
                 test_7_chksum_fx16{ 0x05737544 }, test_7_chksum_fx16_fx8_fx8{ 0x7FFA25C2 }, test_7_chksum_sa8{ 0x5E7CF172 },
                 test_8_chksum_fx16{ 0x69862892 }, test_8_chksum_fx16_fx8_fx8{ 0xA124C817 }, test_8_chksum_sa8{ 0x99E3EE1D },
                 test_9_chksum_fx16{ 0x3B2662E7 }, test_9_chksum_fx16_fx8_fx8{ 0x5C4D2278 },
                 test_10_chksum_fx16{ 0x0AD3FF47 }, test_10_chksum_fx16_fx8_fx8{ 0x0CDE9B47 }, test_10_chksum_sa8{ 0xA4EB24F1 },
                 test_11_chksum_fx16{ 0xEE754246 }, test_11_chksum_fx16_fx8_fx8{ 0x77A6F1AD }, test_11_chksum_sa8{ 0x10AA2F03 };

const crc32_calc test_1_chksum_sa8{ 0x63A6B2EC }, test_9_chksum_sa8{ 0x3DB4B9EF };
const crc32_calc test_6_chksum_fx16_fx8_fx8{ 0x8C24C65A }, test_6_chksum_sa8{ 0x2BA3EA5D };


const quality_metrics thresholds_fx16_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR_DB = */70.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_fx16_fx8_fx8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                  /* SNR_DB = */30.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_fx16_fx8_fx8_test4{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR_DB = */26.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                             /* SNR_DB = */35.f, /*Quant Error Perc = */40.f };


static const conv2d_test_operands tests_list[] = {
    // Basic functionality test kernel_size=(3, 4), strides=(1, 1), with krn_padding and w/o ReLU
    {"Test 1 SA8_SA8_SA32", mli_krn_conv2d_hwcn_sa8_sa8_sa32,
                            input_1_sa8, weights_1_sa8, bias_1_sa32, test_1_out_sa8,
                            test_1_out_acc_sa32, test_1_bias_out_sa8,
                            input_1_scale, test_1_out_scale,
                            weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                            test_1_cfg, thresholds_sa8_general, test_1_chksum_sa8},

    // Basic functionality test with 7 kernels of (4, 3) size, strides = (2, 2), with krn_padding and with Gen_ReLU
    {"Test 2 SA8_SA8_SA32 ReluGen", mli_krn_conv2d_hwcn_sa8_sa8_sa32,
                                    input_1_sa8, weights_2_sa8, bias_1_w2_per_tensor_sa32, test_2_out_sa8,
                                    test_2_out_acc_sa32, test_2_bias_out_sa8,
                                    input_1_scale, test_2_out_scale,
                                    weights_2_per_tensor_scales, sizeof(weights_2_per_tensor_scales) / sizeof(weights_2_per_tensor_scales[0]),
                                    test_2_cfg, thresholds_sa8_general, test_2_chksum_sa8},

    // Dilation Rate Test: kernel_size=(3, 4), strides=(1, 1), w/o padding and w/o ReLU
    {"Test 3 SA8_SA8_SA32 Dilation", mli_krn_conv2d_hwcn_sa8_sa8_sa32,
                                    input_1_sa8, weights_1_sa8, bias_1_sa32, test_3_out_sa8,
                                    test_3_out_acc_sa32, test_3_bias_out_sa8,
                                    input_1_scale, test_3_out_scale,
                                    weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                                    test_3_cfg, thresholds_sa8_general, test_3_chksum_sa8},

    // Input/output Memstride test : kernel_size = (4, 3), strides = (3, 3), w / o padding and with ReLU_1
    // padded with 3 extra values on c Dim and extra 1 line. Output is also expected to have a memstride
    {"Test 4 SA8_SA8_SA32 IO_Memstr", mli_krn_conv2d_hwcn_sa8_sa8_sa32,
                                      input_1_memstr_sa8, weights_1_sa8, bias_1_sa32, test_4_out_sa8,
                                      test_4_out_acc_sa32, test_4_bias_out_sa8,
                                      input_1_scale, test_4_out_scale,
                                      weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                                      test_4_cfg, thresholds_sa8_general, test_4_chksum_sa8},

    // Weights Memstride test with 7 kernels of (4, 3) size, strides = (1, 1), w / o padding and with ReLU_6
    // padded with extra channel on N dimension
    {"Test 5 SA8_SA8_SA32 W_Memstr", mli_krn_conv2d_hwcn_sa8_sa8_sa32,
                                     input_1_sa8, weights_2_memstr_sa8, bias_1_w2_sa32, test_5_out_sa8,
                                     test_5_out_acc_sa32, test_5_bias_out_sa8,
                                     input_1_scale, test_5_out_scale,
                                     weights_2_scales, sizeof(weights_2_scales) / sizeof(weights_2_scales[0]),
                                     test_5_cfg, thresholds_sa8_general, test_5_chksum_sa8},

    // k1x1 specialization test with memstride, kernel_size=(1, 1), strides=(2, 2), krn_padding and ReLU 6
    // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 6 SA8_SA8_SA32  k1x1 Spec", mli_krn_conv2d_hwcn_sa8_sa8_sa32_k1x1,
                                       input_1_sa8, weights_3_memstr_sa8, bias_1_w3_sa32, test_6_out_sa8,
                                       test_6_out_acc_sa32, test_6_bias_out_sa8,
                                       input_1_scale, test_6_out_scale,
                                       weights_3_scales, sizeof(weights_3_scales) / sizeof(weights_3_scales[0]),
                                       test_6_cfg, thresholds_sa8_general, test_6_chksum_sa8},

    // k3x3 specialization test with memstride, kernel_size=(3, 3), strides=(2, 2), krn_padding and ReLU 6
    // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 7 SA8_SA8_SA32 k3x3 Spec", mli_krn_conv2d_hwcn_sa8_sa8_sa32_k3x3,
                                      input_1_sa8, weights_4_memstr_sa8, bias_1_w4_sa32, test_7_out_sa8,
                                      test_7_out_acc_sa32, test_7_bias_out_sa8,
                                      input_1_scale, test_7_out_scale,
                                      weights_4_scales, sizeof(weights_4_scales) / sizeof(weights_4_scales[0]),
                                      test_7_cfg, thresholds_sa8_general, test_7_chksum_sa8},

    // k5x5 specialization test with memstride, kernel_size=(5, 5), strides=(2, 2), krn_padding and ReLU 6
    // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 8 SA8_SA8_SA32 k5x5 spec", mli_krn_conv2d_hwcn_sa8_sa8_sa32_k5x5,
                                      input_1_sa8, weights_5_memstr_sa8, bias_1_w5_sa32, test_8_out_sa8,
                                      test_8_out_acc_sa32, test_8_bias_out_sa8,
                                      input_1_scale, test_8_out_scale,
                                      weights_5_scales, sizeof(weights_5_scales) / sizeof(weights_5_scales[0]),
                                      test_8_cfg, thresholds_sa8_general, test_8_chksum_sa8},

    // Dilation test with padding for generic function, kernel_size=(3, 3), strides=(1, 1),
    // krn_padding , dilation = (2,2) and ReLU_Gen.
    // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 9-1 SA8_SA8_SA32 Dil+Pad", mli_krn_conv2d_hwcn_sa8_sa8_sa32,
                                      input_1_sa8, weights_4_memstr_sa8, bias_1_w4_sa32, test_9_out_sa8,
                                      test_9_out_acc_sa32, test_9_bias_out_sa8,
                                      input_1_scale, test_9_out_scale,
                                      weights_4_scales, sizeof(weights_4_scales) / sizeof(weights_4_scales[0]),
                                      test_9_cfg, thresholds_sa8_general, test_9_chksum_sa8},

    // Dilation test for k3x3 specialization test, kernel_size=(3, 3), strides=(1, 1),
    // krn_padding , dilation = (2,2) and ReLU_Gen.
    // Memstrides are applied on input, output and weights tensors
    {"Test 9-2 SA8_SA8_SA32 k3x3 Dil", mli_krn_conv2d_hwcn_sa8_sa8_sa32_k3x3,
                                       input_1_sa8, weights_4_memstr_sa8, bias_1_w4_sa32, test_9_out_sa8,
                                       test_9_out_acc_sa32, test_9_bias_out_sa8,
                                       input_1_scale, test_9_out_scale,
                                       weights_4_scales, sizeof(weights_4_scales) / sizeof(weights_4_scales[0]),
                                       test_9_cfg, thresholds_sa8_general, test_9_chksum_sa8},

    // Dilation test for k5x5 specialization test, kernel_size=(5, 5), strides=(1, 1),
    // krn_padding , dilation = (2,2) and ReLU_Gen.
    // Memstrides are applied on input, output and weights tensors
    {"Test 10 SA8_SA8_SA32 k5x5 Dil", mli_krn_conv2d_hwcn_sa8_sa8_sa32_k5x5,
                                      input_1_sa8, weights_5_memstr_sa8, bias_1_w5_sa32, test_10_out_sa8,
                                      test_10_out_acc_sa32, test_10_bias_out_sa8,
                                      input_1_scale, test_10_out_scale,
                                      weights_5_scales, sizeof(weights_5_scales) / sizeof(weights_5_scales[0]),
                                      test_10_cfg, thresholds_sa8_general, test_10_chksum_sa8},

    // Test with huge values in operands to check negative fractional and big scales
    {"Test 11 SA8_SA8_SA32 Huge Vals", mli_krn_conv2d_hwcn_sa8_sa8_sa32,
                                       input_2_sa8, weights_6_sa8, bias_2_i2_w6_sa32, test_11_out_sa8,
                                       test_11_out_acc_sa32, test_11_bias_out_sa8,
                                       input_2_scale, test_11_out_scale,
                                       weights_6_scales, sizeof(weights_6_scales) / sizeof(weights_6_scales[0]),
                                       test_11_cfg, thresholds_sa8_general, test_11_chksum_sa8},
};
constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

// Global Memory Memagement
//==================================================================
constexpr uint32_t kMemSize = 2247;
constexpr int kMemAccSize = kMemSize*sizeof(int32_t); // TODO: for double wide accu, more space might be required
static IO_DATA_ATTR int8_t g_scratch_mem_in[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t g_scratch_mem_acc_out[kMemAccSize] = { 0 };
static IO_DATA_ATTR int8_t g_scratch_mem_out[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t g_scratch_mem_bias_out[kMemSize] = { 0 };
static W_DATA_ATTR int8_t g_scratch_mem_w[kMemSize] = { 0 };
static W_DATA_ATTR int8_t g_scratch_mem_b[kMemSize] = { 0 };
static int8_t g_mem_pool[kMemSize] = {0};


struct Conv2dOp {
  // Conv2d Kernel
  Conv2dOp(const conv2d_test_operands* cur_test) {
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

  // conv runtime instnace
  void* conv2d_instance{nullptr};
  uint32_t conv2d_instance_size{0};

  // conv private data
  lib_mli::PrivateData* conv2d_conf_private{nullptr};
  uint32_t conv2d_conf_private_size{0};
};

struct RescaleOp {
  // Rescale Kernel
  RescaleOp(const conv2d_test_operands* cur_test, const mli_tensor& input, const mli_tensor& weights) {
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
                      const conv2d_test_operands* cur_test,
                      const Conv2dOp& conv2d_op, const RescaleOp& rs_op) {
    bool is_test_passed = true;

    if (!(cur_test->in.is_valid() && cur_test->weights.is_valid() &&
        cur_test->bias_in.is_valid() && cur_test->bias_out.is_valid() && cur_test->out.is_valid())) {
      reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
      is_test_passed = false;
    }

    if (is_test_passed &&
        (tensor_quantizer::validate_tensor(conv2d_op.input) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(conv2d_op.weights) != tensor_quantizer::kOk||
         tensor_quantizer::validate_tensor(conv2d_op.out_acc) != tensor_quantizer::kOk ||
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
        (conv2d_op.mem_in_keeper.is_memory_corrupted() || rs_op.mem_out_keeper.is_memory_corrupted() ||
         conv2d_op.mem_out_acc_keeper.is_memory_corrupted() || rs_op.mem_bias_out_keeper.is_memory_corrupted() ||
         conv2d_op.mem_w_keeper.is_memory_corrupted() || rs_op.mem_b_keeper.is_memory_corrupted())) {
      reporter.report_message(cur_test->descr,
        "FAILED at quantization step: memory beside one of operands is corrupted");
      is_test_passed = false;
    }

    return is_test_passed;
}

void prepare_phase(const conv2d_test_operands* cur_test,
                   uint32_t& out_mem_offset, Conv2dOp& op) {
  // STEP 1.1: Construct Conv2d as a specific ExecutionInterface successor
  //==================================================================

  // NHWCin vs. HWCin
  uint32_t input_shape[4] = {1, op.input.shape[0], op.input.shape[1], op.input.shape[2]};
  int32_t input_stride[4] = {int32_t(op.input.shape[0]) * op.input.mem_stride[0],
                             op.input.mem_stride[0],
                             op.input.mem_stride[1],
                             op.input.mem_stride[2]};

  // GHWCinCo vs. HWCinCo
  uint32_t weight_shape[5] = {1, op.weights.shape[0], op.weights.shape[1], op.weights.shape[2], op.weights.shape[3]};
  int32_t weight_stride[5] = {int32_t(op.weights.shape[0] * op.weights.mem_stride[0]),
                              op.weights.mem_stride[0], op.weights.mem_stride[1],
                              op.weights.mem_stride[2], op.weights.mem_stride[3]};

  // GHWCo vs. HWCo
  uint32_t output_shape[4] = {1, op.out_acc.shape[0], op.out_acc.shape[1], op.out_acc.shape[2]};

  int32_t output_stride[4] = {int32_t(op.out_acc.shape[0]) * op.out_acc.mem_stride[0],
                              op.out_acc.mem_stride[0],
                              op.out_acc.mem_stride[1],
                              op.out_acc.mem_stride[2]};

  // G == 1
  assert(input_shape[0] == 1 && output_shape[0] == 1);
  // Input Cin == Weight Cin
  assert(input_shape[3] == weight_shape[3]);
  // Weight Co = Output Co
  assert(weight_shape[4] == output_shape[3]);

  const lib_mli::Tensor<lib_mli::NoBuffer, 4> in_tensor(input_shape, input_stride);
  const lib_mli::Tensor<lib_mli::NoBuffer, 4> out_tensor(output_shape, output_stride);
  const lib_mli::Tensor<lib_mli::NoBuffer, 5> wt_tensor(weight_shape, weight_stride);

  lib_mli::PlatformDescription pd;
  lib_ref::KernelsFactory kernel_factory(pd);
  uint32_t conv2d_cs_size = kernel_factory.Conv2d_CS_GetSize();
  void* conv2d_cs_buffer = malloc(conv2d_cs_size);

  lib_mli::Conv2DConfig cfg(
    cur_test->cfg.stride_height, cur_test->cfg.stride_width,
    cur_test->cfg.padding_top, cur_test->cfg.padding_left,
    cur_test->cfg.padding_bottom, cur_test->cfg.padding_right,
    cur_test->cfg.dilation_height, cur_test->cfg.dilation_width,
    /* groups=1 */ input_shape[0]
  );

  auto conv2d_op = kernel_factory.Conv2d_CS(
    conv2d_cs_buffer, in_tensor, wt_tensor, cfg, out_tensor);

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
  uint32_t runtime_obj_size = conv2d_op->GetRuntimeObjectSize();
  *offset += runtime_obj_size;

  // Leave space for private data buffer
  offset = &offsets[0];
  uint32_t private_buffer_size = conv2d_op->GetKernelPrivateDataSize();
  *offset += private_buffer_size;

  // conv2d input
  offset = &offsets[0];
  uint32_t in_size = conv2d_op->GetInputBufferSize() * sizeof(int8_t);
  lib_mli::OffsetBuffer conv2d_in_buf{*offset, 0, in_size, sizeof(int8_t)};
  lib_mli::Tensor<lib_mli::OffsetBuffer, 4> conv2d_in_tensor(conv2d_in_buf, input_shape);
  in_mem_offset = *offset;
  *offset += in_size;

  // conv2d weight
  offset = &offsets[0];
  uint32_t w_size = conv2d_op->GetWeightsBufferSize() * sizeof(int8_t);
  lib_mli::OffsetBuffer conv2d_w_buf{*offset, 0, w_size, sizeof(int8_t)};
  w_mem_offset = *offset;
  *offset += w_size;

  // conv2d output
  offset = &offsets[0];
  uint32_t out_size = conv2d_op->GetOutputBufferSize() * sizeof(int32_t);
  lib_mli::OffsetBuffer conv2d_out_buf{*offset, 0, out_size, sizeof(int32_t)};
  lib_mli::Tensor<lib_mli::OffsetBuffer, 4> conv2d_out_tensor(conv2d_out_buf, output_shape);
  out_mem_offset = *offset;
  *offset += out_size;

  // conv2d input zero point
  offset = &offsets[0];
  uint32_t inpzp_size = conv2d_op->GetEncodedInpZeroPtsSize() * sizeof(int16_t);
  lib_mli::OffsetBuffer inpzp_buf{*offset, 0, inpzp_size, sizeof(int16_t)};
  inpzp_mem_offset = *offset;
  *offset += inpzp_size;

  // conv2d weights zero point
  offset = &offsets[0];
  uint32_t wtszp_size = conv2d_op->GetEncodedWtsZeroPtsSize() * sizeof(int16_t);
  lib_mli::OffsetBuffer wtszp_buf{*offset, 0, wtszp_size, sizeof(int16_t)};
  wtszp_mem_offset = *offset;
  *offset += wtszp_size;

  // MLI tensor structures and conv2d configuration
  offset = &offsets[0];
  uint32_t data_buffer_size = conv2d_op->GetDataBufferSize();
  lib_mli::OffsetBuffer conv2d_descr_buf{*offset, 0, data_buffer_size, sizeof(char)};
  *offset += data_buffer_size;

  // Attaching buffer (descriptors) to the operation
  mli_status status = MLI_STATUS_OK;

  status = conv2d_op->AttachBufferOffsets(conv2d_in_tensor,
                                          conv2d_out_tensor,
                                          conv2d_w_buf,
                                          inpzp_buf,
                                          wtszp_buf,
                                          conv2d_descr_buf);
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
  size_t shared_buf_size = std::max(inpzp_size, wtszp_size);
  char host_buf_a[shared_buf_size];
  char host_buf_b[shared_buf_size];
  lib_mli::Buffer src_inpzp_buf(host_buf_a, inpzp_size, sizeof(int8_t));
  lib_mli::Buffer dst_inpzp_buf(host_buf_b, inpzp_size, sizeof(int16_t));
  lib_mli::Buffer src_wtszp_buf(host_buf_a, wtszp_size, sizeof(int8_t));
  lib_mli::Buffer dst_wtszp_buf(host_buf_b, wtszp_size, sizeof(int16_t));
  assert(src_inpzp_buf.get_size() == inpzp_buf.get_size());
  assert(src_inpzp_buf.get_elem_size() * 2 == inpzp_buf.get_elem_size());
  assert(src_wtszp_buf.get_size() == wtszp_buf.get_size());
  assert(src_wtszp_buf.get_elem_size() * 2 == wtszp_buf.get_elem_size());

  uint32_t inpzp_shape[1] = {1};
  lib_mli::Tensor<lib_mli::Buffer, 1> inpzp_tensor(src_inpzp_buf, inpzp_shape);

  uint32_t wtszp_shape[1] = {weight_shape[4]};
  lib_mli::Tensor<lib_mli::Buffer, 1> wtszp_tensor(src_wtszp_buf, wtszp_shape);

  // NOTE: Zero Points should have the same size as the tensor they belong to.
  // input zero points: mli_tensor -> host tensor
  if (op.input.el_params.sa.dim == -1) {
    assert(op.input.el_params.sa.zero_point.capacity == 0);
    inpzp_tensor.write(0, static_cast<int8_t>(op.input.el_params.sa.zero_point.mem.i16));
  } else {
    assert(op.input.el_params.sa.zero_point.capacity == src_inpzp_buf.get_size());
    for (size_t i = 0; i < inpzp_size / sizeof(int16_t); ++i) {
      inpzp_tensor.write(int(i), static_cast<int8_t>(op.input.el_params.sa.zero_point.mem.pi16[i]));
    }
  }
  // host tensor 8bit -> encoded host buffer 16bit
  status = conv2d_op->EncodeInpZeroPts(inpzp_tensor, dst_inpzp_buf);
  assert(status == MLI_STATUS_OK);
  // encoded host buffer -> global mem pool
  auto inpzp_mem = reinterpret_cast<int16_t*>(g_mem_pool + inpzp_mem_offset);
  for (size_t i = 0; i < inpzp_size / sizeof(int16_t); ++i) {
    inpzp_mem[i] = dst_inpzp_buf.read<int16_t>(i);
  }

  // weights zero points: mli_tensor -> host buffer
  if (op.weights.el_params.sa.dim == -1) {
    assert(op.weights.el_params.sa.zero_point.capacity == 0);
    wtszp_tensor.write(0, static_cast<int8_t>(op.weights.el_params.sa.zero_point.mem.i16));
  } else {
    assert(op.weights.el_params.sa.zero_point.capacity == src_wtszp_buf.get_size());
    for (size_t i = 0; i < wtszp_size / sizeof(int16_t); ++i) {
      wtszp_tensor.write(int(i), static_cast<int8_t>(op.weights.el_params.sa.zero_point.mem.pi16[i]));
    }
  }
  // host tensor -> encoded host buffer
  status = conv2d_op->EncodeWtsZeroPts(wtszp_tensor, dst_wtszp_buf);
  assert(status == MLI_STATUS_OK);
  auto wtszp_mem = reinterpret_cast<int16_t*>(g_mem_pool + wtszp_mem_offset);
  // encoded host buffer -> global mem pool
  for (size_t i = 0; i < wtszp_size / sizeof(int16_t); ++i) {
    wtszp_mem[i] = dst_wtszp_buf.read<int16_t>(i);
  }

  // STEP 1.4: Compile conv2d into the binary data
  //==================================================================
  op.conv2d_instance = g_mem_pool;
  op.conv2d_instance_size = conv2d_op->GetRuntimeObjectSize();

  status =
      conv2d_op->GetKernelPrivateData(g_mem_pool + op.conv2d_instance_size);
  assert(status == MLI_STATUS_OK);
  op.conv2d_conf_private = reinterpret_cast<lib_mli::PrivateData*>(
      g_mem_pool + op.conv2d_instance_size);
  op.conv2d_conf_private_size = conv2d_op->GetKernelPrivateDataSize();
}

void execution_phase(Conv2dOp& op) {
  // STEP 3: Execution phase
  //==================================================================
  uint32_t tiles_num = 1;

  uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool)};

  auto mli_conv = lib_mli::ExecutionInterface::Create(
    op.conv2d_instance, op.conv2d_instance_size,
    op.conv2d_conf_private, op.conv2d_conf_private_size,
    membasis, sizeof(membasis) / sizeof(membasis[0]));

  assert(mli_conv != nullptr);

  mli_status status = MLI_STATUS_OK;
  for (int i = 0; i < tiles_num; ++i) {
    status = mli_conv->Prefetch();
    assert(status == MLI_STATUS_OK);

    status = mli_conv->Issue();
    assert(status == MLI_STATUS_OK);

    status = mli_conv->Update();
    assert(status == MLI_STATUS_OK);
  }
}

bool postprocess_phase(const reporter_full& reporter,
                       const conv2d_test_operands* cur_test,
                       Conv2dOp& conv2d_op, RescaleOp& rs_op) {
  quality_metrics test_metrics;
  bool is_test_passed = true;

  auto& out = rs_op.out;
  mli_tensor source_out_tensor = rs_op.original_out;

  if (is_test_passed &&
      (conv2d_op.mem_in_keeper.is_memory_corrupted() || rs_op.mem_out_keeper.is_memory_corrupted() ||
       conv2d_op.mem_out_acc_keeper.is_memory_corrupted() || rs_op.mem_bias_out_keeper.is_memory_corrupted() ||
       conv2d_op.mem_w_keeper.is_memory_corrupted() || rs_op.mem_b_keeper.is_memory_corrupted())) {
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
    data_crc(conv2d_op.input);
    data_crc(conv2d_op.weights);
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
  bool final_status = true;

  reporter.report_header("MLI|Kernels|Convolution 2D  Tests");
  for (int i = 0; i < kTestsNum; ++i) {
    // get the current test case
    const conv2d_test_operands* cur_test = &tests_list[i];

#if defined(__Xvec_guard_bit_option)
    // VPX code needs to be debugged
    reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
    continue;
#endif

    // STEP 0: Preprocessing phase
    //==================================================================
    Conv2dOp conv2d_op(cur_test);
    RescaleOp rs_op(cur_test, conv2d_op.input, conv2d_op.weights);

    bool is_test_passed = preprocess_phase(reporter, cur_test, conv2d_op, rs_op);

    // STEP 1: Preparing phase
    //==================================================================
    uint32_t out_mem_offset = 0;
    prepare_phase(cur_test, out_mem_offset, conv2d_op);

    // STEP 2: Executing phase
    //==================================================================
    // Run conv2d MLI3.0 kernel
    execution_phase(conv2d_op);

    // Get the output of Conv2d and copy it to conv2d_op.out_acc
    for (uint32_t i = 0; i < conv2d_op.out_acc.data.capacity; ++i) {
      conv2d_op.out_acc.data.mem.pi8[i] = *(g_mem_pool + out_mem_offset + i);
    }

    // Run rescale kernel
    if (is_test_passed &&
        // i32->i8
        mli_krn_rescale_i32_o8(&conv2d_op.out_acc,
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

    // Run conv2d MLI2.0 kernel, for debug purpose
    // if (is_test_passed &&
    //     cur_test->mli_krn_conv2d(&conv2d_op.input, &conv2d_op.weights, &rs_op.bias_in,
    //                              &cur_test->cfg, &rs_op.out) != MLI_STATUS_OK) {
    //   reporter.report_message(cur_test->descr, "FAILED at kernel run: kernel returned bad status");
    //   is_test_passed = false;
    // }

    // STEP 3: Postprocessing phase
    //==================================================================
    is_test_passed &= postprocess_phase(reporter, cur_test, conv2d_op, rs_op);

    final_status &= is_test_passed;
  }
  reporter.report_outline("[AUTO] Group: mli_krn_conv2d", final_status);

  return (final_status) ? 0 : 1;
}
