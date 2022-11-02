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
#include "mli_ref_runtime_api.hpp"
#include "mli_private_types.h"
#include "mli_service_functions.hpp"

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "test_rescale_utility.h"
#include "test_tensor_quantizer.h"
#include "test_report.h"
#include "test_tiling.hpp"

#include "vectors_mli_krn_conv2d.inc"

 /**
  * Comment USE_TILING if you want to use single tile (tile size = input size).
  */
#define USE_TILING

#define NUM_GROUPS 1  // don't change this
#define BATCH_SIZE 1  // don't change this

using namespace snps_arc::metaware::mli::service;

using lib_mli::kMliAlignment;

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

using lib_mli::kPerTensorQuantDim;
using lib_mli::kInpZPRank;
using lib_mli::kWZPRank;

using lib_mli::kConvIORank;
using lib_mli::kConvIOIterRank;
using lib_mli::kConvWRank;
using lib_mli::kConvWIterRank;
using lib_mli::kConvZPRank;
using lib_mli::kConvZPIterRank;
using lib_mli::kConvIterRank;

using lib_mli::kGroupTensorBatchDim;
using lib_mli::kGroupTensorHeightDim;
using lib_mli::kGroupTensorWidthDim;
using lib_mli::kGroupTensorGroupDim;
using lib_mli::kGroupTensorChannelDim;

using lib_mli::kKernelChannelInDim;
using lib_mli::kKernelChannelOutDim;

using lib_mli::kClipRank;
using lib_mli::kClipIterRank;
using lib_mli::kPreluRank;
using lib_mli::kPreluIterRank;

using lib_mli::kPreluParamRank;
using lib_mli::kTensorChannelDim;

using lib_mli::kSkipIterDim;

struct conv2d_test_operands {
    const char* descr;
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
    {"Test 1 SA8_SA8_SA32", input_1_sa8, weights_1_sa8, bias_1_sa32, test_1_out_sa8,
                            test_1_out_acc_sa32, test_1_bias_out_sa8,
                            input_1_scale, test_1_out_scale,
                            weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                            test_1_cfg, thresholds_sa8_general, test_1_chksum_sa8},

    // Basic functionality test with 7 kernels of (4, 3) size, strides = (2, 2), with krn_padding and with Gen_ReLU
    {"Test 2 SA8_SA8_SA32 ReluGen", input_1_sa8, weights_2_sa8, bias_1_w2_per_tensor_sa32, test_2_out_sa8,
                                    test_2_out_acc_sa32, test_2_bias_out_sa8,
                                    input_1_scale, test_2_out_scale,
                                    weights_2_per_tensor_scales, sizeof(weights_2_per_tensor_scales) / sizeof(weights_2_per_tensor_scales[0]),
                                    test_2_cfg, thresholds_sa8_general, test_2_chksum_sa8},

    // Dilation Rate Test: kernel_size=(3, 4), strides=(1, 1), w/o padding and w/o ReLU
    {"Test 3 SA8_SA8_SA32 Dilation", input_1_sa8, weights_1_sa8, bias_1_sa32, test_3_out_sa8,
                                     test_3_out_acc_sa32, test_3_bias_out_sa8,
                                     input_1_scale, test_3_out_scale,
                                     weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                                     test_3_cfg, thresholds_sa8_general, test_3_chksum_sa8},

    // Input/output Memstride test : kernel_size = (4, 3), strides = (3, 3), w / o padding and with ReLU_1
    // padded with 3 extra values on c Dim and extra 1 line. Output is also expected to have a memstride
    {"Test 4 SA8_SA8_SA32 IO_Memstr", input_1_memstr_sa8, weights_1_sa8, bias_1_sa32, test_4_out_sa8,
                                      test_4_out_acc_sa32, test_4_bias_out_sa8,
                                      input_1_scale, test_4_out_scale,
                                      weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                                      test_4_cfg, thresholds_sa8_general, test_4_chksum_sa8},

    // Weights Memstride test with 7 kernels of (4, 3) size, strides = (1, 1), w / o padding and with ReLU_6
    // padded with extra channel on N dimension
    {"Test 5 SA8_SA8_SA32 W_Memstr", input_1_sa8, weights_2_memstr_sa8, bias_1_w2_sa32, test_5_out_sa8,
                                     test_5_out_acc_sa32, test_5_bias_out_sa8,
                                     input_1_scale, test_5_out_scale,
                                     weights_2_scales, sizeof(weights_2_scales) / sizeof(weights_2_scales[0]),
                                     test_5_cfg, thresholds_sa8_general, test_5_chksum_sa8},

    // k1x1 specialization test with memstride, kernel_size=(1, 1), strides=(2, 2), krn_padding and ReLU 6
    // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 6 SA8_SA8_SA32  k1x1 Spec", input_1_sa8, weights_3_memstr_sa8, bias_1_w3_sa32, test_6_out_sa8,
                                       test_6_out_acc_sa32, test_6_bias_out_sa8,
                                       input_1_scale, test_6_out_scale,
                                       weights_3_scales, sizeof(weights_3_scales) / sizeof(weights_3_scales[0]),
                                       test_6_cfg, thresholds_sa8_general, test_6_chksum_sa8},

    // k3x3 specialization test with memstride, kernel_size=(3, 3), strides=(2, 2), krn_padding and ReLU 6
    // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 7 SA8_SA8_SA32 k3x3 Spec", input_1_sa8, weights_4_memstr_sa8, bias_1_w4_sa32, test_7_out_sa8,
                                      test_7_out_acc_sa32, test_7_bias_out_sa8,
                                      input_1_scale, test_7_out_scale,
                                      weights_4_scales, sizeof(weights_4_scales) / sizeof(weights_4_scales[0]),
                                      test_7_cfg, thresholds_sa8_general, test_7_chksum_sa8},

    // k5x5 specialization test with memstride, kernel_size=(5, 5), strides=(2, 2), krn_padding and ReLU 6
    // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 8 SA8_SA8_SA32 k5x5 spec", input_1_sa8, weights_5_memstr_sa8, bias_1_w5_sa32, test_8_out_sa8,
                                      test_8_out_acc_sa32, test_8_bias_out_sa8,
                                      input_1_scale, test_8_out_scale,
                                      weights_5_scales, sizeof(weights_5_scales) / sizeof(weights_5_scales[0]),
                                      test_8_cfg, thresholds_sa8_general, test_8_chksum_sa8},

    // Dilation test with padding for generic function, kernel_size=(3, 3), strides=(1, 1),
    // krn_padding , dilation = (2,2) and ReLU_Gen.
    // No Dilation ratio. Memstrides are applied on input, output and weights tensors
    {"Test 9-1 SA8_SA8_SA32 Dil+Pad", input_1_sa8, weights_4_memstr_sa8, bias_1_w4_sa32, test_9_out_sa8,
                                      test_9_out_acc_sa32, test_9_bias_out_sa8,
                                      input_1_scale, test_9_out_scale,
                                      weights_4_scales, sizeof(weights_4_scales) / sizeof(weights_4_scales[0]),
                                      test_9_cfg, thresholds_sa8_general, test_9_chksum_sa8},

    // Dilation test for k3x3 specialization test, kernel_size=(3, 3), strides=(1, 1),
    // krn_padding , dilation = (2,2) and ReLU_Gen.
    // Memstrides are applied on input, output and weights tensors
    {"Test 9-2 SA8_SA8_SA32 k3x3 Dil", input_1_sa8, weights_4_memstr_sa8, bias_1_w4_sa32, test_9_out_sa8,
                                       test_9_out_acc_sa32, test_9_bias_out_sa8,
                                       input_1_scale, test_9_out_scale,
                                       weights_4_scales, sizeof(weights_4_scales) / sizeof(weights_4_scales[0]),
                                       test_9_cfg, thresholds_sa8_general, test_9_chksum_sa8},

    // Dilation test for k5x5 specialization test, kernel_size=(5, 5), strides=(1, 1),
    // krn_padding , dilation = (2,2) and ReLU_Gen.
    // Memstrides are applied on input, output and weights tensors
    {"Test 10 SA8_SA8_SA32 k5x5 Dil", input_1_sa8, weights_5_memstr_sa8, bias_1_w5_sa32, test_10_out_sa8,
                                      test_10_out_acc_sa32, test_10_bias_out_sa8,
                                      input_1_scale, test_10_out_scale,
                                      weights_5_scales, sizeof(weights_5_scales) / sizeof(weights_5_scales[0]),
                                      test_10_cfg, thresholds_sa8_general, test_10_chksum_sa8},

    // Test with huge values in operands to check negative fractional and big scales
    {"Test 11 SA8_SA8_SA32 Huge Vals", input_2_sa8, weights_6_sa8, bias_2_i2_w6_sa32, test_11_out_sa8,
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
static int8_t g_scratch_mem_in[kMemSize] = { 0 };
static int8_t g_scratch_mem_acc_out[kMemAccSize] = { 0 };
static int8_t g_scratch_mem_out[kMemSize] = { 0 };
static int8_t g_scratch_mem_bias_out[kMemSize] = { 0 };
static int8_t g_scratch_mem_w[kMemSize] = { 0 };
static int8_t g_scratch_mem_b[kMemSize] = { 0 };
constexpr uint32_t kMemPoolSize = 16384;
static IO_DATA_ATTR int8_t g_mem_pool[kMemPoolSize] = {0};
constexpr uint32_t kWeightsAndWeightsZPBufferSize = 1112;
static int8_t g_weights_buf_mem[kWeightsAndWeightsZPBufferSize] = { 0 };
constexpr uint32_t kPreluEncodedParamBufSize = 77;
static int8_t g_prelu_buf_mem[kPreluEncodedParamBufSize];

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
  void* conv2d_conf_private{nullptr};
  uint32_t conv2d_conf_private_size{0};
};

struct PreluOp {
  // Prelu Kernel
  PreluOp(const conv2d_test_operands* cur_test, const mli_tensor& input, const mli_tensor& weights) {
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

  // additional params for MLI3 semantic
  bias_folder mli3_bias;
  scales_calc mli3_scales_keeper;

  // additional params for MLI3 runtime
  void* prelu_instance;
  uint32_t prelu_instance_size;
  void* prelu_conf_private;
  uint32_t prelu_conf_private_size;
};

struct ClipOp {
  // Clip Kernel
  ClipOp(const conv2d_test_operands* cur_test, const mli_tensor& out) {
      original_out = out;
  }

  // original tensors
  mli_tensor original_out;

  // additional params for MLI3 runtime
  void* clip_instance;
  uint32_t clip_instance_size;
  lib_mli::PrivateData* clip_conf_private;
  uint32_t clip_conf_private_size;
};

mli_minmax_t get_val_limit(mli_tensor *out, const mli_relu_cfg *cfg) {
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
    return val_limit;
}

bool preprocess_phase(const reporter_full& reporter,
                      const conv2d_test_operands* cur_test,
                      const Conv2dOp& conv2d_op, const PreluOp& pr_op,
                      const ClipOp& clp_op) {
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
         tensor_quantizer::validate_tensor(pr_op.bias_out) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(pr_op.mli3_bias.get_bias_tsr()) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(pr_op.mli3_scales_keeper.get_scales_tsr()) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(pr_op.mli3_scales_keeper.get_shift_tsr()) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(pr_op.out) != tensor_quantizer::kOk)) {
      reporter.report_message(cur_test->descr,
                  "FAILED at quantization step: more memory for one of tensors might be required");
      is_test_passed = false;
    }

    if (is_test_passed &&
        (conv2d_op.mem_in_keeper.is_memory_corrupted() || pr_op.mem_out_keeper.is_memory_corrupted() ||
         conv2d_op.mem_out_acc_keeper.is_memory_corrupted() || pr_op.mem_bias_out_keeper.is_memory_corrupted() ||
         conv2d_op.mem_w_keeper.is_memory_corrupted() || pr_op.mem_b_keeper.is_memory_corrupted())) {
      reporter.report_message(cur_test->descr,
        "FAILED at quantization step: memory beside one of operands is corrupted");
      is_test_passed = false;
    }

    return is_test_passed;
}


void prepare_phase(const conv2d_test_operands* cur_test, uint32_t& num_tiles,
                  Conv2dOp& cnv_op, PreluOp &pr_op, ClipOp &clp_op, lib_mli::Buffer &encoded_params_buffer) {

  static_assert(BATCH_SIZE == 1 && NUM_GROUPS == 1);

  int32_t iteration_order[kConvIOIterRank]{ 0, 1, 2, 3, 4 };
  uint32_t total_input_size[kConvIORank]{ BATCH_SIZE, cnv_op.input.shape[0], cnv_op.input.shape[1], NUM_GROUPS, cnv_op.input.shape[2] };
  uint32_t total_output_size[kConvIORank]{ BATCH_SIZE, cnv_op.out_acc.shape[0], cnv_op.out_acc.shape[1], NUM_GROUPS, cnv_op.out_acc.shape[2] };

#ifdef USE_TILING
  /**
    TODO: investigate why test case 10 hit assert in vpx dbg and pass without errors in vpx rel
    lib/src/private\mli_prv_layout.h: Line 70: 
    assert(pad_left >= 0 && pad_top >= 0 && out_h_idx >= 0 && out_w_idx >= 0) failed.
  */
  uint32_t tile_output_size[kConvIORank]{ BATCH_SIZE, 4, 4, NUM_GROUPS, 2 };
  tile_output_size[kGroupTensorHeightDim] = MIN(tile_output_size[kGroupTensorHeightDim], total_output_size[kGroupTensorHeightDim]);
  tile_output_size[kGroupTensorWidthDim] = MIN(tile_output_size[kGroupTensorWidthDim], total_output_size[kGroupTensorWidthDim]);
  // TODO: smaller H/W tile sizes will fail on 9-2, because of MLI_ASSERT I put inside IteratorCfg ctor - do something with it

#else
  uint32_t tile_output_size[kConvIORank]{ BATCH_SIZE, total_output_size[1], total_output_size[2], NUM_GROUPS, total_output_size[3] };
#endif

 
  int32_t output_stride[kConvIORank] = { int32_t(total_output_size[1]) * cnv_op.out_acc.mem_stride[0],
                                        cnv_op.out_acc.mem_stride[0],
                                        cnv_op.out_acc.mem_stride[1],
                                        cnv_op.out_acc.mem_stride[1],
                                        cnv_op.out_acc.mem_stride[2] };

  const lib_mli::Tensor<lib_mli::NoBuffer, kConvIORank> full_out_tensor(total_output_size, output_stride);
  lib_mli::TensorIterator<lib_mli::NoBuffer, kConvIORank, kConvIOIterRank> out_tensor_it(full_out_tensor, tile_output_size, iteration_order);

  num_tiles = out_tensor_it.GetTotalCount();
#ifndef USE_TILING
  assert(num_tiles == 1);
#endif

  uint32_t effective_kernel_size[kConvIORank]{
    1, lib_mli::service::get_effective_kernel_size(cnv_op.weights.shape[0], cur_test->cfg.dilation_height),
    lib_mli::service::get_effective_kernel_size(cnv_op.weights.shape[1], cur_test->cfg.dilation_width),
    1, total_input_size[kGroupTensorChannelDim]
  };
  uint32_t stride[kConvIORank]{ 1, cur_test->cfg.stride_height, cur_test->cfg.stride_width, 1, 0 };
  uint32_t pre_padding[kConvIORank]{ 0, cur_test->cfg.padding_top, cur_test->cfg.padding_left, 0, 0 };
  int32_t input_stride[kConvIORank] = { int32_t(cnv_op.input.shape[0]) * cnv_op.input.mem_stride[0],
                                      cnv_op.input.mem_stride[0],
                                      cnv_op.input.mem_stride[1],
                                      cnv_op.input.mem_stride[1],
                                      cnv_op.input.mem_stride[2] };
  const lib_mli::Tensor<lib_mli::NoBuffer, kConvIORank> full_in_tensor(total_input_size, input_stride);
  lib_mli::TensorIterator<lib_mli::NoBuffer, kConvIORank, kConvIOIterRank> in_tensor_it(full_in_tensor, out_tensor_it,
                                                                                        effective_kernel_size, stride, pre_padding);

  uint32_t tile_input_shape[kConvIORank];
  uint32_t tile_output_shape[kConvIORank];
  const auto& input_it_config = in_tensor_it.get_config();
  for (unsigned i = 0; i < kConvIORank; i++) {
    tile_input_shape[i] = (uint32_t)MAX(input_it_config.get_first_size(i), input_it_config.get_size(i));
    tile_output_shape[i] = (uint32_t)MAX(out_tensor_it.get_config().get_first_size(i), out_tensor_it.get_config().get_size(i));
  }

  // GHWCinCo vs. HWCinCo
  uint32_t weight_shape[kConvWRank] = { 1, cnv_op.weights.shape[0], cnv_op.weights.shape[1],
                               cnv_op.weights.shape[2], cnv_op.weights.shape[3]};
  int32_t weight_stride[kConvWRank] = {int32_t(cnv_op.weights.shape[0] * cnv_op.weights.mem_stride[0]),
                              cnv_op.weights.mem_stride[0], cnv_op.weights.mem_stride[1],
                              cnv_op.weights.mem_stride[2], cnv_op.weights.mem_stride[3]};

  uint32_t tile_weights_shape[kConvWRank];
  for (unsigned i = 0; i < kConvWRank - 1; i++) {
    tile_weights_shape[i] = weight_shape[i];
  }
  tile_weights_shape[kKernelChannelOutDim] = tile_output_shape[kGroupTensorChannelDim];


  assert(total_input_size[kGroupTensorBatchDim] == BATCH_SIZE && tile_output_shape[kGroupTensorBatchDim] == BATCH_SIZE);
  assert(total_input_size[kGroupTensorGroupDim] == NUM_GROUPS && tile_output_shape[kGroupTensorGroupDim] == NUM_GROUPS);

  assert(total_input_size[kGroupTensorChannelDim] == weight_shape[kKernelChannelInDim]);
  assert(weight_shape[kKernelChannelOutDim] == total_output_size[kGroupTensorChannelDim]);

  lib_mli::Conv2DConfig cfg(
    cur_test->cfg.stride_height, cur_test->cfg.stride_width,
    cur_test->cfg.padding_top, cur_test->cfg.padding_left,
    cur_test->cfg.padding_bottom, cur_test->cfg.padding_right,
    cur_test->cfg.dilation_height, cur_test->cfg.dilation_width,
    /* groups=1 */ total_input_size[0]
  );

  // STEP 1.1: Construct [Conv2d] as a specific ExecutionInterface successor
  //==================================================================
  lib_mli::PlatformDescription pd;
  lib_ref::KernelsFactory kernel_factory(pd);
  uint32_t conv2d_cs_size = kernel_factory.Conv2d_CS_GetSize();
  void* conv2d_cs_buffer = malloc(conv2d_cs_size);

  const lib_mli::Tensor<lib_mli::NoBuffer, kConvWRank> wt_tensor(weight_shape, weight_stride);
  
  const int32_t zero_inc_mask[kConvWRank]{1, 1, 1, 1, 0};
  lib_mli::TensorIterator<lib_mli::NoBuffer, kConvWRank, kConvWIterRank> w_tensor_it(wt_tensor, out_tensor_it, nullptr, zero_inc_mask);

  uint32_t wzp_shape[kConvZPRank]{total_output_size[kGroupTensorChannelDim]};
  lib_mli::Tensor<lib_mli::NoBuffer, kConvZPRank> wzp_tensor(wzp_shape);
  const int32_t wzp_it_order[kConvWRank]{ -1, -1, -1, -1, 0 };
  lib_mli::TensorIterator<lib_mli::NoBuffer, kConvZPRank, kConvZPIterRank> wzp_tensor_it(wzp_tensor, out_tensor_it, wzp_it_order, zero_inc_mask);

  uint32_t izp_shape[kInpZPRank]{ 1 };
  lib_mli::Tensor<lib_mli::NoBuffer, kInpZPRank> izp_tensor(izp_shape);
  lib_mli::TensorIterator<lib_mli::NoBuffer, kInpZPRank, kConvIterRank> izp_tensor_it(izp_tensor);
  auto conv2d_op = kernel_factory.Conv2d_CS(conv2d_cs_buffer, in_tensor_it, izp_tensor_it, w_tensor_it, wzp_tensor_it, cfg, out_tensor_it);

  // STEP 1.1.2: Construct [Prelu] as a specific ExecutionInterface successor
  //==================================================================

  mli_tensor &pr_input_tsr = cnv_op.out_acc;
  mli_tensor &pr_output_tsr = pr_op.out;

  void* &prelu_instance = pr_op.prelu_instance;
  uint32_t &prelu_instance_size = pr_op.prelu_instance_size;
  void* &prelu_conf_private = pr_op.prelu_conf_private;
  uint32_t &prelu_conf_private_size = pr_op.prelu_conf_private_size;

  for (int i = 0; i < 2; i++) {
    assert(pr_input_tsr.shape[i] == total_output_size[1 + i]);
    assert(pr_input_tsr.mem_stride[i] == output_stride[1 + i]);
    assert(pr_input_tsr.shape[i] == total_output_size[1 + i]);
    assert(pr_input_tsr.mem_stride[i] == output_stride[1 + i]);
  }
  assert(pr_input_tsr.shape[2] == total_output_size[4]);
  assert(pr_input_tsr.mem_stride[2] == output_stride[4]);
  assert(pr_input_tsr.shape[2] == total_output_size[4]);
  assert(pr_input_tsr.mem_stride[2] == output_stride[4]);


  uint32_t prelu_cs_size = kernel_factory.Prelu_CS_GetSize();
  void* prelu_cs_buffer = malloc(prelu_cs_size);

  lib_mli::PreluOpConfig pr_cfg;
  if (mli_hlp_count_elem_num(&pr_op.mli3_scales_keeper.get_scales_tsr(), 0) == 1) {
      pr_cfg.axis = -1;
  } else {
      pr_cfg.axis = kGroupTensorChannelDim;
  }

  assert(kPreluRank == kConvIORank);
  assert(kPreluIterRank == kConvIOIterRank);
  auto prelu_op = kernel_factory.Prelu_CS(prelu_cs_buffer, out_tensor_it, pr_cfg, out_tensor_it);

  // STEP 1.1.3: Construct [Clip] as a specific ExecutionInterface successor
  //==================================================================

  mli_tensor &clip_input_tsr = pr_op.original_out;
  mli_tensor &clip_output_tsr = pr_op.original_out;
  void* &clip_instance = clp_op.clip_instance;
  uint32_t &clip_instance_size = clp_op.clip_instance_size;
  lib_mli::PrivateData* &clip_conf_private = clp_op.clip_conf_private;
  uint32_t &clip_conf_private_size = clp_op.clip_conf_private_size;

  for (int i = 0; i < 2; i++) {
    assert(clip_input_tsr.shape[i] == total_output_size[1 + i]);
    assert(clip_input_tsr.mem_stride[i] == output_stride[1 + i]);
    assert(clip_output_tsr.shape[i] == total_output_size[1 + i]);
    assert(clip_output_tsr.mem_stride[i] == output_stride[1 + i]);
  }
  assert(clip_input_tsr.shape[2] == total_output_size[4]);
  assert(clip_input_tsr.mem_stride[2] == output_stride[4]);
  assert(clip_output_tsr.shape[2] == total_output_size[4]);
  assert(clip_output_tsr.mem_stride[2] == output_stride[4]);

  uint32_t clip_cs_size = kernel_factory.Clip_CS_GetSize();
  void* clip_cs_buffer = malloc(clip_cs_size);

  assert(kPreluRank - 1 == kClipRank);
  assert(kPreluIterRank - 1 == kClipIterRank);
  uint32_t io_output_shape[kClipRank]{ total_output_size[0], total_output_size[1], total_output_size[2], total_output_size[4] };
  int32_t io_output_strides[kClipRank]{ output_stride[0], output_stride[1], output_stride[2], output_stride[4] };
  lib_mli::Tensor<lib_mli::NoBuffer, kClipRank> io_tensor(io_output_shape, io_output_strides);
  // TODO: Remove IteratorCfg and use only TensorIterator when kClipRank becomes 5
  lib_mli::IteratorCfg<kClipRank> io_it_cfg(out_tensor_it.get_config(), kGroupTensorGroupDim);
  lib_mli::TensorIterator<lib_mli::NoBuffer, kClipRank, kClipIterRank> io_tensor_it(io_tensor, io_it_cfg);
  auto clip_op = kernel_factory.Clip_CS(clip_cs_buffer, io_tensor_it, io_tensor_it);
  // STEP 1.2.1: [Conv2D] Memory management (Up to user on how to deal with it)
  //==================================================================
  uint32_t offsets[1] = {0};

  // NOTE: Currently, only supoort these data types.
  assert(cnv_op.input.el_type == MLI_EL_SA_8);
  assert(cnv_op.weights.el_type == MLI_EL_SA_8);
  assert(cnv_op.out_acc.el_type == MLI_EL_SA_32);

  // Leave space for runtime object
  uint32_t* cnv_offset = &offsets[0];
  uint32_t runtime_obj_size = conv2d_op->GetRuntimeObjectSize();
  *cnv_offset += runtime_obj_size;

  // Leave space for private data buffer
  uint32_t private_buffer_size = conv2d_op->GetKernelPrivateDataSize();
  *cnv_offset += private_buffer_size;

  // conv2d input
  uint32_t cnv_i_elem_size = mli_hlp_tensor_element_size(&cnv_op.input);
  uint32_t in_size = GetBufferSize(kConvIORank, tile_input_shape, input_stride) * cnv_i_elem_size;
  lib_mli::OffsetBuffer conv2d_in_buf{*cnv_offset, 0, in_size, cnv_i_elem_size};
  *cnv_offset += in_size;

  // conv2d weight
  uint32_t cnv_w_elem_size = mli_hlp_tensor_element_size(&cnv_op.weights);
  uint32_t w_size = GetBufferSize(kConvWRank, tile_weights_shape, weight_stride) * cnv_w_elem_size;
  lib_mli::OffsetBuffer conv2d_w_buf{*cnv_offset, 0, w_size, cnv_w_elem_size};
  *cnv_offset += w_size;

  // conv2d output
  // NOTE: The output should be aligned, otherwise, it will cause `vvst` crash.
  //       For example, offset is 4 byts aligned if output is int32_t.
  uint32_t cnv_o_elem_size = mli_hlp_tensor_element_size(&cnv_op.out_acc);
  *cnv_offset = CEIL_RND(*cnv_offset, cnv_o_elem_size);
  uint32_t conv_out_size_in_elements = GetBufferSize(kConvIORank, tile_output_shape, output_stride);
  uint32_t out_size = conv_out_size_in_elements * cnv_o_elem_size;
  lib_mli::OffsetBuffer conv2d_out_buf{*cnv_offset, 0, out_size, cnv_o_elem_size};
  *cnv_offset += out_size;

  // conv2d input zero point
  uint32_t inpzp_size = conv2d_op->GetEncodedInpZeroPtsSize() * cnv_i_elem_size;
  lib_mli::OffsetBuffer inpzp_buf{*cnv_offset, 0, inpzp_size, cnv_i_elem_size};
  uint32_t inpzp_mem_offset = *cnv_offset;
  *cnv_offset += inpzp_size;

  // conv2d weights zero point
  uint32_t wtszp_size = tile_output_shape[kGroupTensorChannelDim] * cnv_w_elem_size;
  lib_mli::OffsetBuffer wtszp_buf{*cnv_offset, 0, wtszp_size, cnv_w_elem_size};
  *cnv_offset += wtszp_size;

  // MLI tensor structures and conv2d configuration
  assert(conv2d_op->GetCtrlBufferSize() == 0);
  lib_mli::OffsetBuffer conv2d_ctrl_buf{*cnv_offset, 0, 0, sizeof(char)};
  assert(*cnv_offset <= kMemPoolSize);

  // DataBuffer size is 0 for reference kernel
  mli_status status = MLI_STATUS_OK;
  status = conv2d_op->AttachBufferOffsets(conv2d_in_buf,
                                          conv2d_out_buf,
                                          conv2d_w_buf,
                                          inpzp_buf,
                                          wtszp_buf,
                                          conv2d_ctrl_buf);
  assert(status == MLI_STATUS_OK);

  // STEP 1.2.2: [Prelu] Memory management (Up to user on how to deal with it)
  //==================================================================

  // Define buffers for in\out tensors
  // Leave space for runtime object
  uint32_t* pr_offset = &offsets[0];
  *pr_offset = CEIL_RND(*pr_offset, kMliAlignment);
  int8_t* pr_runtime_obj_addr = (int8_t*)g_mem_pool + offsets[0];
  uint32_t pr_runtime_obj_size = prelu_op->GetRuntimeObjectSize();
  *pr_offset += pr_runtime_obj_size;

  // Leave space for private data buffer
  uint32_t pr_private_buffer_size = prelu_op->GetKernelPrivateDataSize();
  *pr_offset += pr_private_buffer_size;

  // prelu input = conv output
  uint32_t input_elem_size = mli_hlp_tensor_element_size(&pr_input_tsr);
  uint32_t pr_in_size = conv_out_size_in_elements * input_elem_size;
  lib_mli::OffsetBuffer prelu_in_buf { conv2d_out_buf.get_offset(), 0, pr_in_size, input_elem_size };

  // prelu output
  uint32_t output_elem_size = mli_hlp_tensor_element_size(&pr_output_tsr);
  uint32_t pr_out_size = conv_out_size_in_elements * output_elem_size;
  lib_mli::OffsetBuffer prelu_out_buf { *pr_offset, 0, pr_out_size, output_elem_size };
  *pr_offset += pr_out_size;

  // prelu params
  uint32_t encoded_params_size = prelu_op->GetEncodedParamsSize();
  lib_mli::OffsetBuffer encoded_params_buf { *pr_offset, 0, encoded_params_size,
                                              sizeof(int8_t) };
  *pr_offset += encoded_params_size;

  // DataBuffer size is 0 for reference kernel
  assert(prelu_op->GetCtrlBufferSize() == 0);
  lib_mli::OffsetBuffer prelu_ctrl_buf { *pr_offset, 0, 0, sizeof(char) };
  assert(*pr_offset <= kMemPoolSize);

  status = prelu_op->AttachBufferOffsets(prelu_in_buf,
                                         prelu_out_buf,
                                         encoded_params_buf,
                                         prelu_ctrl_buf);
  assert(status == MLI_STATUS_OK);

  // STEP 1.2.3: [Clip] Memory management (Up to user on how to deal with it)
  //==================================================================

  // Leave space for runtime object
  uint32_t* clip_offset = &offsets[0];
  *clip_offset = CEIL_RND(*clip_offset, kMliAlignment);
  int8_t* clip_runtime_obj_addr = (int8_t*)g_mem_pool + offsets[0];
  uint32_t clip_runtime_obj_size = clip_op->GetRuntimeObjectSize();
  *clip_offset += clip_runtime_obj_size;

  // Leave space for private data buffer
  uint32_t clip_private_buffer_size = clip_op->GetKernelPrivateDataSize();
  *clip_offset += clip_private_buffer_size;

  // clip input = prelu output
  uint32_t clip_input_elem_size = mli_hlp_tensor_element_size(&clip_input_tsr);
  uint32_t clip_in_size = conv_out_size_in_elements * clip_input_elem_size;
  lib_mli::OffsetBuffer clip_in_buf{prelu_out_buf.get_offset(), 0, clip_in_size, clip_input_elem_size};

  // clip output
  uint32_t clip_output_elem_size = mli_hlp_tensor_element_size(&clip_output_tsr);
  uint32_t clip_out_size = conv_out_size_in_elements * clip_output_elem_size;
  lib_mli::OffsetBuffer clip_out_buf{ *clip_offset, 0, clip_out_size, clip_output_elem_size};
  *clip_offset += clip_in_size;

  // clip min
  uint32_t clip_encoded_params_size = clip_op->GetEncodedParamsSize();
  lib_mli::OffsetBuffer clip_encoded_params_buf {*clip_offset, 0, clip_encoded_params_size,
                                 sizeof(int8_t)};
  uint32_t clip_encoded_params_mem_offset = *clip_offset;
  *clip_offset += clip_encoded_params_size;;

  // DataBuffer size is 0 for reference kernel
  assert(clip_op->GetCtrlBufferSize() == 0);
  lib_mli::OffsetBuffer clip_ctrl_buf{*clip_offset, 0, 0, sizeof(char)};
  assert(*clip_offset <= kMemPoolSize);

  status = clip_op->AttachBufferOffsets(clip_in_buf,
                                        clip_out_buf,
                                        clip_encoded_params_buf,
                                        clip_ctrl_buf);
  assert(status == MLI_STATUS_OK);



  // encode input zero points to the global mem pool
  //==================================================================
  assert(cnv_i_elem_size == sizeof(int8_t) && cnv_w_elem_size == sizeof(int8_t));
  uint32_t full_weights_size = conv2d_op->GetEncodedWeightsSize();
  assert(full_weights_size == lib_mli::service::GetBufferSize(kConvWRank, weight_shape, weight_stride) * cnv_w_elem_size);
  uint32_t full_wtszp_size = conv2d_op->GetEncodedWtsZeroPtsSize();
  uint32_t full_weights_and_wzp_size = full_wtszp_size + full_weights_size;
  uint32_t max_dst_encoded_buffer_size = MAX(full_weights_and_wzp_size, inpzp_size);
  assert(max_dst_encoded_buffer_size <= kWeightsAndWeightsZPBufferSize);

  // copy input zero point into inpzp_tensor
  void* src_izp_mem = malloc(inpzp_size);
  lib_mli::Buffer src_inpzp_buf(src_izp_mem, inpzp_size, cnv_i_elem_size);
  uint32_t inpzp_shape[kInpZPRank] = {1};
  lib_mli::Tensor<lib_mli::Buffer, kInpZPRank> inpzp_tensor(src_inpzp_buf, inpzp_shape);
  if (cnv_op.input.el_params.sa.dim == kPerTensorQuantDim) {
    assert(cnv_op.input.el_params.sa.zero_point.capacity == 0);
    inpzp_tensor.write(0, static_cast<int8_t>(cnv_op.input.el_params.sa.zero_point.mem.i16));
  } else {
    assert(cnv_op.input.el_params.sa.zero_point.capacity / sizeof(int16_t) == src_inpzp_buf.get_size());
    for (uint32_t i = 0; i < inpzp_size / cnv_i_elem_size; ++i) {
      inpzp_tensor.write(int(i), static_cast<int8_t>(cnv_op.input.el_params.sa.zero_point.mem.pi16[i]));
    }
  }

  // encode input zero point into dst_inpzp_buf
  lib_mli::Buffer dst_inpzp_buf((void*)g_weights_buf_mem, inpzp_size, cnv_i_elem_size);
  auto izp_tensor_it_with_buf = lib_mli::TensorIterator<lib_mli::Buffer, kInpZPRank, kConvIterRank>(inpzp_tensor);
  status = conv2d_op->EncodeInpZeroPts(izp_tensor_it_with_buf, dst_inpzp_buf);
  assert(status == MLI_STATUS_OK);

  // copy encoded input zero point from dst_inpzp_buf into g_mem_pool
  auto inpzp_mem = (int8_t*)g_mem_pool + inpzp_mem_offset;
  for (uint32_t i = 0; i < inpzp_size / cnv_i_elem_size; ++i) {
    inpzp_mem[i] = dst_inpzp_buf.read<int8_t>(i);
  }
  free(src_izp_mem);

  // copy weights zero point(s) into wtszp_tensor
  void* src_wzp_mem = malloc(full_wtszp_size);
  lib_mli::Buffer src_wtszp_buf(src_wzp_mem, full_wtszp_size, cnv_w_elem_size);
  uint32_t wtszp_shape[kWZPRank] = { weight_shape[kKernelChannelOutDim] };
  lib_mli::Tensor<lib_mli::Buffer, kWZPRank> wtszp_tensor(src_wtszp_buf, wtszp_shape);
  if (cnv_op.weights.el_params.sa.dim == kPerTensorQuantDim) {
    assert(cnv_op.weights.el_params.sa.zero_point.capacity == 0);
    wtszp_tensor.write(0, static_cast<int8_t>(cnv_op.weights.el_params.sa.zero_point.mem.i16));
  }
  else {
    assert(cnv_op.weights.el_params.sa.zero_point.capacity / sizeof(int16_t) == full_wtszp_size);
    for (size_t i = 0; i < full_wtszp_size / cnv_w_elem_size; ++i) {
      wtszp_tensor.write(int(i), static_cast<int8_t>(cnv_op.weights.el_params.sa.zero_point.mem.pi16[i]));
    }
  }

  // copy weights into src_weights_buf
  void* src_w_mem = malloc(full_weights_size);
  lib_mli::Buffer src_weights_buf(src_w_mem, full_weights_size, cnv_w_elem_size);
  lib_mli::Tensor<lib_mli::Buffer, kConvWRank> weights_tensor(src_weights_buf, weight_shape);
  int32_t zero_offsets[kConvWRank]{};
  strided_copy_with_offsets(kConvWRank, cnv_w_elem_size,
                            cnv_op.weights.data.mem.pi8,
                            zero_offsets, zero_offsets, weight_stride,
                            weight_shape, weights_tensor.get_buf().get_ptr<int8_t>());

  // encode weights and weights zero point(s) in dst_weights_and_weights_zp_buf
  lib_mli::Buffer dst_w_wzp_encoded_buffer((void*)g_weights_buf_mem, full_weights_and_wzp_size, cnv_w_elem_size);
  auto w_tensor_it_with_buf = lib_mli::TensorIterator<lib_mli::Buffer, kConvWRank, kConvIterRank>(weights_tensor);
  auto wzp_tensor_it_with_buf = lib_mli::TensorIterator<lib_mli::Buffer, kWZPRank, kConvIterRank>(wtszp_tensor);
  status = conv2d_op->EncodeWeightsAndZeroPts(w_tensor_it_with_buf, wzp_tensor_it_with_buf, dst_w_wzp_encoded_buffer);
  assert(status == MLI_STATUS_OK);

  free(src_wzp_mem);
  free(src_w_mem);

  // Compile conv2d into the binary data
  //==================================================================
  cnv_op.conv2d_instance = (int8_t*)g_mem_pool;
  cnv_op.conv2d_instance_size = conv2d_op->GetRuntimeObjectSize();

  status =
      conv2d_op->GetKernelPrivateData((int8_t*)g_mem_pool + cnv_op.conv2d_instance_size);
  assert(status == MLI_STATUS_OK);
  cnv_op.conv2d_conf_private = (int8_t*)g_mem_pool + cnv_op.conv2d_instance_size;
  cnv_op.conv2d_conf_private_size = conv2d_op->GetKernelPrivateDataSize();
  assert(status == MLI_STATUS_OK);

  // Compile prelu into the binary data
  //==================================================================
  prelu_instance = pr_runtime_obj_addr;
  prelu_instance_size = prelu_op->GetRuntimeObjectSize();
  prelu_conf_private = pr_runtime_obj_addr + prelu_instance_size;
  prelu_conf_private_size = prelu_op->GetKernelPrivateDataSize();

  status = prelu_op->GetKernelPrivateData(prelu_conf_private);

  assert(status == MLI_STATUS_OK);
  // [PReLU] encode params into special buffer
  //==================================================================
  const mli_tensor& pr_inbias_tsr = pr_op.mli3_bias.get_bias_tsr();
  const mli_tensor& pr_posshift_tsr = pr_op.mli3_scales_keeper.get_shift_tsr();
  const mli_tensor& pr_negshift_tsr = pr_op.mli3_scales_keeper.get_shift_tsr();
  const mli_tensor& pr_posscale_tsr = pr_op.mli3_scales_keeper.get_scales_tsr();
  const mli_tensor& pr_negscale_tsr = pr_op.mli3_scales_keeper.get_scales_tsr();
  const mli_tensor& pr_outbias_tsr = pr_op.bias_out;
  uint32_t inbias_elem_size = mli_hlp_tensor_element_size(&pr_inbias_tsr);
  uint32_t posscale_elem_size = mli_hlp_tensor_element_size(&pr_posscale_tsr);
  uint32_t posshift_elem_size = mli_hlp_tensor_element_size(&pr_posshift_tsr);
  uint32_t negscale_elem_size = mli_hlp_tensor_element_size(&pr_negscale_tsr);
  uint32_t negshift_elem_size = mli_hlp_tensor_element_size(&pr_negshift_tsr);
  uint32_t outbias_elem_size = mli_hlp_tensor_element_size(&pr_outbias_tsr);
  uint32_t inbias_size = inbias_elem_size * mli_hlp_count_elem_num(&pr_inbias_tsr, 0);
  uint32_t posscale_size = posscale_elem_size * mli_hlp_count_elem_num(&pr_posscale_tsr, 0);
  uint32_t posshift_size = posshift_elem_size * mli_hlp_count_elem_num(&pr_posshift_tsr, 0);
  uint32_t negscale_size = negscale_elem_size * mli_hlp_count_elem_num(&pr_negscale_tsr, 0);
  uint32_t negshift_size = negshift_elem_size * mli_hlp_count_elem_num(&pr_negshift_tsr, 0);
  uint32_t outbias_size = outbias_elem_size * mli_hlp_count_elem_num(&pr_outbias_tsr, 0);
  
  int8_t* host_src_buf = (int8_t*)malloc(encoded_params_size);
  lib_mli::Buffer src_inbias_buf(host_src_buf,
    inbias_size, inbias_elem_size);
  lib_mli::Buffer src_posscale_buf(host_src_buf + inbias_size,
    posscale_size, posscale_elem_size);
  lib_mli::Buffer src_negscale_buf(host_src_buf + inbias_size + posscale_size,
    negscale_size, negscale_elem_size);
  lib_mli::Buffer src_posshift_buf(host_src_buf + inbias_size + posscale_size + negscale_size,
    posshift_size, posshift_elem_size);
  lib_mli::Buffer src_negshift_buf(host_src_buf + inbias_size + posscale_size + negscale_size + posshift_size,
    negshift_size, negshift_elem_size);
  lib_mli::Buffer src_outbias_buf(host_src_buf + inbias_size + posscale_size + negscale_size + posshift_size + negshift_size,
    outbias_size, outbias_elem_size);
  assert(encoded_params_size <= kPreluEncodedParamBufSize);
  encoded_params_buffer = lib_mli::Buffer(g_prelu_buf_mem, encoded_params_size, sizeof(int8_t));
  uint32_t params_shape[kPreluParamRank] = { pr_inbias_tsr.shape[0], 0};
  lib_mli::Tensor<lib_mli::Buffer, kPreluParamRank> inbias_tensor(src_inbias_buf, params_shape);
  lib_mli::Tensor<lib_mli::Buffer, kPreluParamRank> posscale_tensor(src_posscale_buf, params_shape);
  lib_mli::Tensor<lib_mli::Buffer, kPreluParamRank> negscale_tensor(src_negscale_buf, params_shape);
  lib_mli::Tensor<lib_mli::Buffer, kPreluParamRank> posshift_tensor(src_posshift_buf, params_shape);
  lib_mli::Tensor<lib_mli::Buffer, kPreluParamRank> negshift_tensor(src_negshift_buf, params_shape);
  lib_mli::Tensor<lib_mli::Buffer, kPreluParamRank> outbias_tensor(src_outbias_buf, params_shape);
  if (pr_cfg.axis == kPerTensorQuantDim) {
    assert(pr_inbias_tsr.rank == 0);
    assert(pr_posscale_tsr.rank == 0);
    assert(pr_posshift_tsr.rank == 0);
    assert(pr_negscale_tsr.rank == 0);
    assert(pr_negshift_tsr.rank == 0);
    assert(pr_outbias_tsr.rank == 0);
    inbias_tensor.write<int32_t>(0, pr_inbias_tsr.data.mem.i32);
    posscale_tensor.write<int16_t>(0, pr_posscale_tsr.data.mem.i16);
    negscale_tensor.write<int16_t>(0, pr_negscale_tsr.data.mem.i16);
    posshift_tensor.write<int8_t>(0, pr_posshift_tsr.data.mem.i8);
    negshift_tensor.write<int8_t>(0, pr_negshift_tsr.data.mem.i8);
    outbias_tensor.write<int8_t>(0, pr_outbias_tsr.data.mem.i8);
  } else { // per-axis
    for (uint32_t i = 0; i < (inbias_size / inbias_elem_size); i++) {
      inbias_tensor.write<int32_t>(i, pr_inbias_tsr.data.mem.pi32[i]);
    }
    for (uint32_t i = 0; i < (posscale_size / posscale_elem_size); i++) {
      posscale_tensor.write<int16_t>(i, pr_posscale_tsr.data.mem.pi16[i]);
    }
    for (uint32_t i = 0; i < (negscale_size / negscale_elem_size); i++) {
      negscale_tensor.write<int16_t>(i, pr_negscale_tsr.data.mem.pi16[i]);
    }
    for (uint32_t i = 0; i < (posshift_size / posshift_elem_size); i++) {
      posshift_tensor.write<int8_t>(i, pr_posshift_tsr.data.mem.pi8[i]);
    }
    for (uint32_t i = 0; i < (negshift_size / negshift_elem_size); i++) {
      negshift_tensor.write<int8_t>(i, pr_negshift_tsr.data.mem.pi8[i]);
    }
    for (uint32_t i = 0; i < (outbias_size / outbias_elem_size); i++) {
      outbias_tensor.write<int8_t>(i, pr_outbias_tsr.data.mem.pi8[i]);
    }
  }
  // host tensors -> encoded host buffer
  status = prelu_op->EncodeParams(inbias_tensor,
                                  posscale_tensor,
                                  negscale_tensor,
                                  posshift_tensor,
                                  negshift_tensor,
                                  outbias_tensor,
                                  encoded_params_buffer);
  assert(status == MLI_STATUS_OK);

  // STEP 1.3.3: [clip] encode params to the global shared memory pool
  //==================================================================
  // Copy min data to the shared memory pool
  mli_minmax_t val_limit;
  val_limit = get_val_limit(&pr_op.out, &cur_test->cfg.relu);

  int8_t * clp_host_src_buf = (int8_t*) malloc(clip_encoded_params_size);
  int8_t * clp_host_dst_buf = (int8_t*) malloc(clip_encoded_params_size);
  uint32_t clp_params_shape[1] = {1};
  uint32_t limit_min_size = 1;
  uint32_t limit_max_size = 1;

  lib_mli::Buffer src_min_buf(clp_host_src_buf, limit_min_size, sizeof(int8_t));
  lib_mli::Buffer src_max_buf(clp_host_src_buf + limit_min_size, limit_max_size, sizeof(int8_t));
  lib_mli::Buffer clip_encoded_params_buffer(clp_host_dst_buf, clip_encoded_params_size, sizeof(int8_t));
  lib_mli::Tensor<lib_mli::Buffer, 1> min_tensor(src_min_buf,
                                                 clp_params_shape);

  lib_mli::Tensor<lib_mli::Buffer, 1> max_tensor(src_max_buf,
                                                 clp_params_shape);

  for (uint32_t i = 0; i < limit_min_size; ++i) {
      min_tensor.write<int8_t>(i, (uint8_t)val_limit.min);
  }
  for (uint32_t i = 0; i < limit_max_size; ++i) {
      max_tensor.write<int8_t>(i, (uint8_t)val_limit.max);
  }
  // host tensors -> encoded host buffer
  status = clip_op->EncodeParams(min_tensor,
                                 max_tensor,
                                 clip_encoded_params_buffer);
  assert(status == MLI_STATUS_OK);

  // encoded host buffer -> global mem pool
  for (uint32_t i = 0; i < clip_encoded_params_size; ++i) {
      const uint32_t idx = clip_encoded_params_mem_offset + i;
      g_mem_pool[idx] = clip_encoded_params_buffer.read<int8_t>(i);
  }

  // Compile Clip into the binary data
  //==================================================================
  clip_instance = clip_runtime_obj_addr;
  clip_instance_size = clip_op->GetRuntimeObjectSize();
  clip_conf_private =
          reinterpret_cast<lib_mli::PrivateData*>(clip_runtime_obj_addr
                  + clip_instance_size);
  clip_conf_private_size = clip_op->GetKernelPrivateDataSize();

  status = clip_op->GetKernelPrivateData(clip_conf_private);
  assert(status == MLI_STATUS_OK);

  free(conv2d_cs_buffer);
  free(prelu_cs_buffer);
  free(host_src_buf);
  free(clip_cs_buffer);
  free(clp_host_src_buf);
  free(clp_host_dst_buf);
}


void execution_phase(Conv2dOp& cnv_op, PreluOp &pr_op, ClipOp &clp_op, uint32_t tiles_num, lib_mli::Buffer &encoded_params_buffer) {
  // STEP 3: Execution phase
  //==================================================================

  uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool)};

  auto mli_conv = lib_mli::ExecutionInterface::Create(
                    cnv_op.conv2d_instance,
                    cnv_op.conv2d_instance_size,
                    cnv_op.conv2d_conf_private,
                    cnv_op.conv2d_conf_private_size,
                    membasis, sizeof(membasis) / sizeof(membasis[0]));

  auto mli_prelu = lib_mli::ExecutionInterface::Create(
                        pr_op.prelu_instance,
                        pr_op.prelu_instance_size,
                        pr_op.prelu_conf_private,
                        pr_op.prelu_conf_private_size,
                        membasis, sizeof(membasis) / sizeof(membasis[0]));

  auto mli_clip = lib_mli::ExecutionInterface::Create(
                      clp_op.clip_instance,
                      clp_op.clip_instance_size,
                      clp_op.clip_conf_private,
                      clp_op.clip_conf_private_size,
                      membasis, sizeof(membasis) / sizeof(membasis[0]));

  assert(mli_conv != nullptr);
  assert(mli_prelu != nullptr);
  assert(mli_clip != nullptr);


  lib_ref::Conv2d* mli_conv2d_pimpl = dynamic_cast<lib_ref::Conv2d*>(mli_conv);
  lib_ref::Prelu* prelu_pimpl = dynamic_cast<lib_ref::Prelu*>(mli_prelu);
  lib_ref::Conv2DPrivateData * conv2d_private = (lib_ref::Conv2DPrivateData*)(cnv_op.conv2d_conf_private);
  lib_ref::PreluPrivateData * prelu_private = (lib_ref::PreluPrivateData*)(pr_op.prelu_conf_private);
  lib_ref::ClipPrivateData * clip_private = (lib_ref::ClipPrivateData*)(clp_op.clip_conf_private);
 
  int32_t tile_input_strides[kConvIORank]{};
  conv2d_private->input.get_mem_strides(tile_input_strides);
  int32_t tile_output_strides[kConvIORank]{};
  conv2d_private->output.get_mem_strides(tile_output_strides);
  int32_t tile_weights_strides[kConvWRank]{};
  conv2d_private->weights.get_mem_strides(tile_weights_strides);
  
  uint32_t input_tile_size[kConvIOIterRank];
  uint32_t output_tile_size[kConvIOIterRank];
  uint32_t weights_tile_size[kConvWIterRank];
  int32_t input_tile_offsets[kConvIOIterRank];
  int32_t output_tile_offsets[kConvIOIterRank];
  int32_t weights_tile_offsets[kConvWIterRank];
  const int32_t zero_offsets[kConvWIterRank]{};
  uint32_t enc_param_size = 0, inp_bias_offset = 0, posscale_offset = 0, negscale_offset = 0, posshift_offset = 0, negshift_offset = 0, out_bias_offset = 0;

  int32_t output_tile_offsets_4d[kClipRank];
  int32_t tile_output_strides_4d[kClipRank];
  uint32_t output_tile_size_4d[kClipRank];

  mli_status status = MLI_STATUS_OK;
  for (uint32_t i = 0; i < tiles_num; ++i) {
    mli_conv2d_pimpl->GetIOSizesAndOffsets(input_tile_size, output_tile_size, weights_tile_size,
                                           input_tile_offsets, output_tile_offsets, weights_tile_offsets);
    for (int i = 0; i < 3; i++) {
      output_tile_offsets_4d[i] = output_tile_offsets[i];
      tile_output_strides_4d[i] = tile_output_strides[i];
      output_tile_size_4d[i] = output_tile_size[i];
    }
    output_tile_offsets_4d[3] = output_tile_offsets[4];
    tile_output_strides_4d[3] = tile_output_strides[4];
    output_tile_size_4d[3] = output_tile_size[4];

    // copy input from global to local buffer
    strided_copy_with_offsets(kConvIORank, conv2d_private->input.get_buf().get_elem_size(),
                              cnv_op.input.data.mem.pi8,
                              input_tile_offsets, zero_offsets, tile_input_strides,
                              input_tile_size, (int8_t*)(g_mem_pool + conv2d_private->input.get_buf().get_offset()));

    // copy weights from global to local buffer
    strided_copy_with_offsets(kConvWRank, conv2d_private->weights.get_buf().get_elem_size(),
                              cnv_op.weights.data.mem.pi8,
                              weights_tile_offsets, zero_offsets, tile_weights_strides,
                              weights_tile_size, (int8_t*)(g_mem_pool + conv2d_private->weights.get_buf().get_offset()));

    // copy weights zps from global to local buffer
    int8_t* wtszp_tile_buf = (int8_t*)(g_mem_pool + conv2d_private->weights_zp.get_buf().get_offset());
    for (uint32_t j = 0; j < weights_tile_size[kKernelChannelOutDim]; j++) {
      if (cnv_op.weights.el_params.sa.dim == kPerTensorQuantDim) {
        wtszp_tile_buf[j] = (int8_t)cnv_op.weights.el_params.sa.zero_point.mem.i16;
      }
      else {
        wtszp_tile_buf[j] = (int8_t)cnv_op.weights.el_params.sa.zero_point.mem.pi16[j];
      }
    }


    prelu_pimpl->GetIOSizesAndOffsets(enc_param_size, inp_bias_offset, posscale_offset, negscale_offset, posshift_offset, negshift_offset, out_bias_offset);
    const uint32_t enc_param_buf_offset = prelu_private->encoded_params_buffer.get_offset();
    const uint32_t num_enc_param = cnv_op.out_acc.shape[2];

    // copy prelu input bias from global to local buffer
    uint32_t enc_param_offset = output_tile_offsets_4d[3] * sizeof(int32_t);
    memcpy((void*)(g_mem_pool + enc_param_buf_offset + inp_bias_offset),
           (void*)encoded_params_buffer.get_ptr<int8_t>(enc_param_offset), sizeof(int32_t) * enc_param_size);

    // copy prelu posscale from global to local buffer
    enc_param_offset = num_enc_param * sizeof(int32_t) + output_tile_offsets_4d[3] * sizeof(int16_t);
    memcpy((void*)(g_mem_pool + enc_param_buf_offset + posscale_offset),
           (void*)encoded_params_buffer.get_ptr<int8_t>(enc_param_offset), sizeof(int16_t) * enc_param_size);

   // copy prelu negscale from global to local buffer
    enc_param_offset = num_enc_param * (sizeof(int32_t) + sizeof(int16_t)) + output_tile_offsets_4d[3] * sizeof(int16_t);
    memcpy((void*)(g_mem_pool + enc_param_buf_offset + negscale_offset),
           (void*)encoded_params_buffer.get_ptr<int8_t>(enc_param_offset), sizeof(int16_t) * enc_param_size);

    // copy prelu posshift from global to local buffer
    enc_param_offset = num_enc_param * (sizeof(int32_t) + sizeof(int16_t) + sizeof(int16_t)) + output_tile_offsets_4d[3] * sizeof(int8_t);
    memcpy((void*)(g_mem_pool + enc_param_buf_offset + posshift_offset),
           (void*)encoded_params_buffer.get_ptr<int8_t>(enc_param_offset), sizeof(int8_t) * enc_param_size);

    // copy prelu negshift from global to local buffer
    enc_param_offset = num_enc_param * (sizeof(int32_t) + sizeof(int16_t) + sizeof(int16_t) + sizeof(int8_t)) + output_tile_offsets_4d[3] * sizeof(int8_t);
    memcpy((void*)(g_mem_pool + enc_param_buf_offset + negshift_offset),
           (void*)encoded_params_buffer.get_ptr<int8_t>(enc_param_offset), sizeof(int8_t) * enc_param_size);

    // copy prelu output bias from global to local buffer
    enc_param_offset = num_enc_param * (sizeof(int32_t) + sizeof(int16_t) + sizeof(int16_t) + sizeof(int8_t) + sizeof(int8_t)) + output_tile_offsets_4d[3] * sizeof(int8_t);
    memcpy((void*)(g_mem_pool + enc_param_buf_offset + out_bias_offset),
           (void*)encoded_params_buffer.get_ptr<int8_t>(enc_param_offset), sizeof(int8_t) * enc_param_size);


    status = mli_conv->Prefetch();
    assert(status == MLI_STATUS_OK);
    status = mli_conv->Issue();
    assert(status == MLI_STATUS_OK);
    status = mli_conv->Update();
    assert(status == MLI_STATUS_OK);

    status = mli_prelu->Prefetch();
    assert(status == MLI_STATUS_OK);
    status = mli_prelu->Issue();
    assert(status == MLI_STATUS_OK);
    status = mli_prelu->Update();
    assert(status == MLI_STATUS_OK);

    status = mli_clip->Prefetch();
    assert(status == MLI_STATUS_OK);
    status = mli_clip->Issue();
    assert(status == MLI_STATUS_OK);
    status = mli_clip->Update();
    assert(status == MLI_STATUS_OK);

    // copy results from prelu output tile to the global buffer
    strided_copy_with_offsets(kClipRank, clip_private->output.get_buf().get_elem_size(),
                              (int8_t*)g_mem_pool + clip_private->output.get_buf().get_offset(),
                              zero_offsets, output_tile_offsets_4d, tile_output_strides_4d,
                              output_tile_size_4d, clp_op.original_out.data.mem.pi8);
  }
}

bool postprocess_phase(const reporter_full& reporter,
                       const conv2d_test_operands* cur_test,
                       Conv2dOp& conv2d_op, PreluOp& pr_op, ClipOp& clp_op) {
  quality_metrics test_metrics;
  bool is_test_passed = true;

  auto& out = clp_op.original_out;
  mli_tensor source_out_tensor = clp_op.original_out;

  if (is_test_passed &&
      (conv2d_op.mem_in_keeper.is_memory_corrupted() || pr_op.mem_out_keeper.is_memory_corrupted() ||
       conv2d_op.mem_out_acc_keeper.is_memory_corrupted() || pr_op.mem_bias_out_keeper.is_memory_corrupted() ||
       conv2d_op.mem_w_keeper.is_memory_corrupted() || pr_op.mem_b_keeper.is_memory_corrupted())) {
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
    data_crc(pr_op.bias_in);
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

  reporter.report_header("MLI3.0|Kernels|Convolution 2D  Tests");
  for (int i = 0; i < kTestsNum; ++i) {
    // get the current test case
    const conv2d_test_operands* cur_test = &tests_list[i];

// NOTE: Copied from `test_mli_krn_conv2d.cc`, since using the same tect vectors.
#if defined(__Xvec_guard_bit_option) && (__Xvec_guard_bit_option == 0)
        if (strstr(cur_test->descr, "Test 1 SA8_SA8_SA32") != nullptr ||
                strstr(cur_test->descr, "Test 2 SA8_SA8_SA32 ReluGen") != nullptr ||
                strstr(cur_test->descr, "Test 3 SA8_SA8_SA32 Dilation") != nullptr ||
                strstr(cur_test->descr, "Test 4 SA8_SA8_SA32 IO_Memstr") != nullptr ||
                strstr(cur_test->descr, "Test 5 SA8_SA8_SA32 W_Memstr") != nullptr ||
                strstr(cur_test->descr, "Test 6 SA8_SA8_SA32  k1x1 Spec") != nullptr ||
                strstr(cur_test->descr, "Test 7 SA8_SA8_SA32 k3x3 Spec") != nullptr ||
                strstr(cur_test->descr, "Test 8 FX16 k5x5 spec") != nullptr ||
                strstr(cur_test->descr, "Test 8 SA8_SA8_SA32 k5x5 spec") != nullptr ||
                strstr(cur_test->descr, "Test 9-1 SA8_SA8_SA32 Dil+Pad") != nullptr ||
                strstr(cur_test->descr, "Test 9-2 SA8_SA8_SA32 k3x3 Dil") != nullptr ||
                strstr(cur_test->descr, "Test 10 FX16 k5x5 Dil") != nullptr ||
                strstr(cur_test->descr, "Test 10 SA8_SA8_SA32 k5x5 Dil") != nullptr ||
                strstr(cur_test->descr, "Test 11 FX16 Huge Vals") != nullptr ||
                strstr(cur_test->descr, "Test 11 SA8_SA8_SA32 Huge Vals") != nullptr) {
            // VPX fails bitwise comparison with reference .
            reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
            continue;
        }
#endif

#if PLATFORM == V2DSP_XY && defined(CRC_RM_CONVERGENT)
        if (strstr(cur_test->descr, "Test 1 SA8_SA8_SA32") != nullptr ||
                strstr(cur_test->descr, "Test 9-1 SA8_SA8_SA32 Dil+Pad") != nullptr ||
                strstr(cur_test->descr, "Test 9-2 SA8_SA8_SA32 k3x3 Dil") != nullptr) {
            // Em9d fails bitwise comparison with reference .
            reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
            continue;
        }
#endif

    // STEP 0: Preprocessing phase
    //==================================================================
    Conv2dOp conv2d_op(cur_test);
    PreluOp pr_op(cur_test, conv2d_op.input, conv2d_op.weights);
    ClipOp clp_op(cur_test,pr_op.out);

    bool is_test_passed = preprocess_phase(reporter, cur_test, conv2d_op, pr_op, clp_op);

    //
    // Solution to vectorize Prelu params tensors in case of per-axis
    // computation.
    //
    // All params tensors that have one element, should have rank of 0
    // (including out_bias).
    //
    const mli_tensor& inbias_tsr = pr_op.mli3_bias.get_bias_tsr();
    auto& outbias_tsr = pr_op.bias_out;
    auto& posshift_tsr  = (mli_tensor&)pr_op.mli3_scales_keeper.get_shift_tsr();
    auto& posscale_tsr  = (mli_tensor&)pr_op.mli3_scales_keeper.get_scales_tsr();
    auto& negshift_tsr  = (mli_tensor&)pr_op.mli3_scales_keeper.get_shift_tsr();
    auto& negscale_tsr  = (mli_tensor&)pr_op.mli3_scales_keeper.get_scales_tsr();
    void *outbias_data = NULL, *posshift_data = NULL, *negshift_data = NULL, *posscale_data = NULL, *negscale_data = NULL;
    {
        int32_t prelu_axis;
        if (mli_hlp_count_elem_num(&posscale_tsr, 0) == 1) {
            prelu_axis = -1;
        } else {
            prelu_axis = conv2d_op.out_acc.rank;
        }

        // If per-axis computation && out_bias is one element,
        // so construct out_bias tensor as vector the same as other params.
        if((prelu_axis != -1) && (mli_hlp_count_elem_num(&outbias_tsr, 0) == 1)) {
            outbias_data = vectorize_single_elem_tensor(outbias_tsr, inbias_tsr);
        }

        // If per-tensor computation && in_bias is vector,
        // so construct out_bias, shift and scale tensors as vectors the same as in_bias.
        if((prelu_axis == -1) && (mli_hlp_count_elem_num(&inbias_tsr, 0) != 1)) {
            outbias_data = vectorize_single_elem_tensor(outbias_tsr, inbias_tsr);
            posshift_data = vectorize_single_elem_tensor(posshift_tsr, inbias_tsr);
            posscale_data = vectorize_single_elem_tensor(posscale_tsr, inbias_tsr);
            negshift_data = vectorize_single_elem_tensor(negshift_tsr, inbias_tsr);
            negscale_data = vectorize_single_elem_tensor(negscale_tsr, inbias_tsr);
        }
    }

    // STEP 1: Preparing phase
    //==================================================================
    uint32_t num_tiles = 0; // num_tiles calculated inside prepare_phase
    lib_mli::Buffer encoded_params_buffer; // initialized inside prepare_phase
    prepare_phase(cur_test, num_tiles, conv2d_op, pr_op, clp_op, encoded_params_buffer);

    // STEP 2: Executing phase
    //==================================================================
    // Run conv2d, prelu and clip MLI3.0 kernels
    
    execution_phase(conv2d_op, pr_op, clp_op, num_tiles, encoded_params_buffer);

    // STEP 3: Postprocessing phase
    //==================================================================
    is_test_passed &= postprocess_phase(reporter, cur_test, conv2d_op, pr_op, clp_op);

    final_status &= is_test_passed;

    // Free buffers for prelu params
    free(outbias_data);
    free(posshift_data);
    free(posscale_data);
    free(negshift_data);
    free(negscale_data);
  }
  reporter.report_outline("[AUTO] Group: mli_krn_conv2d_30", final_status);

  return (final_status) ? 0 : 1;
}
