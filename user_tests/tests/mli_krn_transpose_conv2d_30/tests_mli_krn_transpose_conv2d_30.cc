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

#include "mli_api.h"
#include "mli_config.h"
#include "mli_types.h"
#include "mli_types.hpp"
#include "mli_kernels_factory_ref.hpp"
#include "mli_runtime_api.hpp"
#include "mli_private_types.h"
#include "mli_service_functions.hpp"
#include "mli_ref_private_types.hpp"
#include "mli_ref_runtime_api.hpp"
#include "mli_helpers_api.hpp"

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "test_rescale_utility.h"
#include "test_tensor_quantizer.h"
#include "test_report.h"
#include "test_tiling.hpp"

#include "vectors_mli_krn_transpose_conv2d.inc"

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


using lib_mli::kTransposeConvIORank;
using lib_mli::kTransposeConvIOIterRank;
using lib_mli::kTransposeConvWRank;
using lib_mli::kTransposeConvWIterRank;
using lib_mli::kTransposeConvZPRank;
using lib_mli::kTransposeConvZPIterRank;
using lib_mli::kTransposeConvIterRank;
using lib_mli::kWZPRank;
using lib_mli::kClipRank;
using lib_mli::kClipIterRank;
using lib_mli::kRescaleRank;
using lib_mli::kRescaleIterRank;
using lib_mli::kRescaleParamRank;
using lib_mli::kPerTensorQuantDim;
using lib_mli::kTensorBatchDim;
using lib_mli::kTensorChannelDim;
using lib_mli::kGroupTensorBatchDim;
using lib_mli::kGroupTensorHeightDim;
using lib_mli::kGroupTensorWidthDim;
using lib_mli::kGroupTensorGroupDim;
using lib_mli::kGroupTensorChannelDim;
using lib_mli::kInpZPRank;
using lib_mli::kClipParamRank;
using lib_mli::kKernelGroupDim;
using lib_mli::kKernelHeightDim;
using lib_mli::kKernelWidthDim;
using lib_mli::kKernelChannelInDim;
using lib_mli::kKernelChannelOutDim;
using lib_mli::kSkipIterDim;


struct transpose_conv2d_test_operands {
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
#if defined(CRC_RM_CONVERGENT) || defined(CRC_RM_UP)

// Shared CRC Results
const crc32_calc test_1_chksum_fx16 {0x7CD22049}, /*test_1_chksum_fx16_fx8_fx8,             */
                 test_2_chksum_fx16 {0x0B88C56E}, /*test_2_chksum_fx16_fx8_fx8,                test_2_chksum_sa8,*/
                 test_3_chksum_fx16 {0x85E46A29},   test_3_chksum_fx16_fx8_fx8 {0xF9B0F692},   test_3_chksum_sa8 {0xCE83CE66},
                 test_4_chksum_fx16 {0xC724EBF9}, /*test_4_chksum_fx16_fx8_fx8,             */ test_4_chksum_sa8 {0xDEE32B04},
                 test_5_chksum_fx16 {0xE82A4691}, /*test_5_chksum_fx16_fx8_fx8,             */ test_5_chksum_sa8 {0x591EA9A4},
                 test_6_chksum_fx16 {0x6D691353}, /*test_6_chksum_fx16_fx8_fx8,                test_6_chksum_sa8,*/
                 test_7_chksum_fx16 {0x314BD269}, /*test_7_chksum_fx16_fx8_fx8,             */ test_7_chksum_sa8 {0xA422B61F},
                 test_8_chksum_fx16 {0x4CDA936B}, test_8_chksum_fx16_fx8_fx8 {0x8436810F};
// Platform Specific CRC Results
#if defined(FULL_ACCU_ON)
const crc32_calc test_1_chksum_sa8 {0xD1E36355};
#else
const crc32_calc test_1_chksum_sa8 {0xD1E36355};
#endif

#if defined(CRC_RM_UP) && defined(FULL_ACCU_ON)
const crc32_calc test_1_chksum_fx16_fx8_fx8 {0xB8EF2F73},
                 test_2_chksum_fx16_fx8_fx8 {0x2A904693}, test_2_chksum_sa8 {0x591979F2},
                 test_4_chksum_fx16_fx8_fx8 {0xF0F39D2C}, 
                 test_5_chksum_fx16_fx8_fx8 {0xA3E639A8},
                 test_6_chksum_fx16_fx8_fx8 {0x1BE42216}, test_6_chksum_sa8 {0x179FAFCC},
                 test_7_chksum_fx16_fx8_fx8 {0x91D2A974}, test_8_chksum_sa8 {0x8BC78C83};
#elif defined(CRC_RM_UP) && !defined(FULL_ACCU_ON)
const crc32_calc test_1_chksum_fx16_fx8_fx8 {0xB8EF2F73},
                 test_2_chksum_fx16_fx8_fx8 {0x2A904693}, test_2_chksum_sa8 {0x591979F2},
                 test_4_chksum_fx16_fx8_fx8 {0xF0F39D2C}, 
                 test_5_chksum_fx16_fx8_fx8 {0xA3E639A8},
                 test_6_chksum_fx16_fx8_fx8 {0x1BE42216}, test_6_chksum_sa8 {0x179FAFCC},
                 test_7_chksum_fx16_fx8_fx8 {0x91D2A974}, test_8_chksum_sa8 {0x8BC78C83};
#elif defined(CRC_RM_CONVERGENT) && defined(FULL_ACCU_ON)
const crc32_calc test_1_chksum_fx16_fx8_fx8 {0x9E58234E},
                 test_2_chksum_fx16_fx8_fx8 {0xB808A08B}, test_2_chksum_sa8 {0x591979F2},
                 test_4_chksum_fx16_fx8_fx8 {0xB617F5E9},
                 test_5_chksum_fx16_fx8_fx8 {0xD261DE7C},
                 test_6_chksum_fx16_fx8_fx8 {0x069E2E0E}, test_6_chksum_sa8 {0x179FAFCC},
                 test_7_chksum_fx16_fx8_fx8 {0x118C5E59}, test_8_chksum_sa8 {0x7AAD7CC6},
#elif defined(CRC_RM_CONVERGENT) && !defined(FULL_ACCU_ON)
const crc32_calc test_1_chksum_fx16_fx8_fx8 {0x9E58234E},
                 test_2_chksum_fx16_fx8_fx8 {0xB808A08B}, test_2_chksum_sa8 {0x591979F2},
                 test_4_chksum_fx16_fx8_fx8 {0xB617F5E9},
                 test_5_chksum_fx16_fx8_fx8 {0xD261DE7C},
                 test_6_chksum_fx16_fx8_fx8 {0x069E2E0E}, test_6_chksum_sa8 {0x179FAFCC},
                 test_7_chksum_fx16_fx8_fx8 {0x118C5E59}, test_8_chksum_sa8 {0x8BC78C83};
#endif
#else // Not defined CRC_*
const crc32_calc  test_1_chksum_fx16, test_1_chksum_fx16_fx8_fx8, test_1_chksum_sa8,
                  test_2_chksum_fx16, test_2_chksum_fx16_fx8_fx8, test_2_chksum_sa8,
                  test_3_chksum_fx16, test_3_chksum_fx16_fx8_fx8, test_3_chksum_sa8,
                  test_4_chksum_fx16, test_4_chksum_fx16_fx8_fx8, test_4_chksum_sa8,
                  test_5_chksum_fx16, test_5_chksum_fx16_fx8_fx8, test_5_chksum_sa8,
                  test_6_chksum_fx16, test_6_chksum_fx16_fx8_fx8, test_6_chksum_sa8,
                  test_7_chksum_fx16, test_7_chksum_fx16_fx8_fx8, test_7_chksum_sa8,
                  test_8_chksum_fx16, test_8_chksum_fx16_fx8_fx8, test_8_chksum_sa8;
#endif

const quality_metrics thresholds_fx16_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR_DB = */70.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_fx16_fx8_fx8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                  /* SNR_DB = */30.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                             /* SNR_DB = */35.f, /*Quant Error Perc = */40.f };

const quality_metrics thresholds_sa8_test3_7{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
/* SNR_DB = */35.f, /*Quant Error Perc = */30.f };


static const transpose_conv2d_test_operands tests_list[] = {
    // Basic functionality test kernel_size=(3, 4), strides=(2, 2), with krn_padding and w/o ReLU
    {"Test 1 SA8_SA8_SA32", input_1_sa8, weights_1_sa8, bias_1_i1_w1_sa32, test_1_out_sa8,
                            test_1_out_acc_sa32, test_1_bias_out_sa8,
                            input_1_scale, test_1_out_scale,
                            weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                            test_1_cfg, thresholds_sa8_general, test_1_chksum_sa8},
    // // Basic functionality test with 7 kernels of (4, 3) size, strides = (2, 2), with krn_padding and with Gen_ReLU
    {"Test 2 SA8_SA8_SA32 ReluGen", input_1_sa8, weights_2_sa8, bias_1_i1_w2_sa32, test_2_out_sa8, 
                                    test_2_out_acc_sa32, test_2_bias_out_sa8,
                                    input_1_scale, test_2_out_scale,
                                    weights_2_scales, sizeof(weights_2_scales) / sizeof(weights_2_scales[0]),
                                    test_2_cfg, thresholds_sa8_general, test_2_chksum_sa8},
    // No strides case: kernel_size=(3, 4), strides=(1, 1), w/o padding and Relu1
    {"Test 3 SA8_SA8_SA32 Str_1x1", input_1_sa8, weights_1_sa8, bias_1_i1_w1_sa32, test_3_out_sa8, 
                                    test_3_out_acc_sa32, test_3_bias_out_sa8,
                                    input_1_scale, test_3_out_scale,
                                    weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                                    test_3_cfg, thresholds_sa8_test3_7, test_3_chksum_sa8},
    // Input/output Memstride test : kernel_size = (4, 3), strides = (3, 2), w / o padding and with ReLU_6
    {"Test 4 SA8_SA8_SA32 IO_Memstr", input_1_memstr_sa8, weights_2_sa8, bias_1_i1_w2_sa32, test_4_out_sa8,
                                      test_4_out_acc_sa32, test_4_bias_out_sa8,
                                      input_1_scale, test_4_out_scale,
                                      weights_2_scales, sizeof(weights_2_scales) / sizeof(weights_2_scales[0]),
                                      test_4_cfg, thresholds_sa8_general, test_4_chksum_sa8},
    // Weights Memstride test : kernels of (3, 4) size, strides = (3, 2), krn_padding and no ReLU
    {"Test 5 SA8_SA8_SA32 IOW_Memstr", input_1_memstr_sa8, weights_1_memstr_sa8, bias_1_i1_w1_sa32, test_5_out_sa8,
                                       test_5_out_acc_sa32, test_5_bias_out_sa8,
                                       input_1_scale, test_5_out_scale,
                                       weights_1_scales, sizeof(weights_1_scales) / sizeof(weights_1_scales[0]),
                                       test_5_cfg, thresholds_sa8_general, test_5_chksum_sa8},
    // k2x2 str2x2 specialization test with memstride, kernel_size=(2, 2), strides=(2, 2), no_padding and ReLU 6
    // Memstrides are applied on input, output and weights tensors
    {"Test 6 SA8_SA8_SA32 k2x2 st2", input_2_memstr_sa8, weights_3_memstr_sa8, bias_2_i2_w3_sa32, test_6_out_sa8,
                                     test_6_out_acc_sa32, test_6_bias_out_sa8,
                                     input_2_scale, test_6_out_scale,
                                     weights_3_scales, sizeof(weights_3_scales) / sizeof(weights_3_scales[0]),
                                     test_6_cfg, thresholds_sa8_general, test_6_chksum_sa8},
    // k4x4 str2x2 specialization test with memstride, kernel_size=(4, 4), strides=(2, 2), krn_padding and ReLU 6
    // Memstrides are applied on input, output and weights tensors
    {"Test 7 SA8_SA8_SA32 k4x4 st2", input_2_memstr_sa8, weights_4_memstr_sa8, bias_2_i2_w4_sa32, test_7_out_sa8, 
                                     test_7_out_acc_sa32, test_7_bias_out_sa8,
                                     input_2_scale, test_7_out_scale,
                                     weights_4_scales, sizeof(weights_4_scales) / sizeof(weights_4_scales[0]),
                                     test_7_cfg, thresholds_sa8_test3_7, test_7_chksum_sa8},
    // k3x3 str2x2 test with memstride, kernel_size=(3, 3), strides=(2, 2), krn_padding and No Relu
    // specific regression test
    {"Test 8 SA8_SA8_SA32 k3x3 st2", input_1_memstr_sa8, weights_5_memstr_sa8, bias_1_i1_w5_sa32, test_8_out_sa8,
                                     test_8_out_acc_sa32, test_8_bias_out_sa8,
                                     input_1_scale, test_8_out_scale,
                                     weights_5_scales, sizeof(weights_5_scales) / sizeof(weights_5_scales[0]),
                                     test_8_cfg, thresholds_sa8_general, test_8_chksum_sa8}

};
constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

// Global Memory Management
//==================================================================
constexpr uint32_t kMemSize = 3047;
constexpr int kMemAccSize = kMemSize*sizeof(int32_t); // TODO: for double wide accu, more space might be required
static int8_t g_scratch_mem_in[kMemSize] = {0};
static int8_t g_scratch_mem_acc_out[kMemAccSize] = {0};
static int8_t g_scratch_mem_out[kMemSize] = {0};
static int8_t g_scratch_mem_bias_out[kMemSize] = {0};
static int8_t g_scratch_mem_w[kMemSize] = {0};
static int8_t g_scratch_mem_b[kMemSize] = {0};
constexpr uint32_t kMemPoolSize = 20480;
static IO_DATA_ATTR int8_t g_mem_pool[kMemPoolSize] = {0};
constexpr uint32_t kRescaleEncodedParamBufSize = 55;
static int8_t g_rescale_buf_mem[kRescaleEncodedParamBufSize];
constexpr uint32_t kWeightsAndWeightsZPBufferSize = 1018;
static int8_t g_weights_buf_mem[kWeightsAndWeightsZPBufferSize] = { 0 };

struct TransposeConv2DOp {
  // TransposeConv2D Kernel
  TransposeConv2DOp(const transpose_conv2d_test_operands* cur_test) {
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
  void* transpose_conv2d_instance{nullptr};
  uint32_t transpose_conv2d_instance_size{0};

  // conv private data
  void* transpose_conv2d_conf_private{nullptr};
  uint32_t transpose_conv2d_conf_private_size{0};
};

struct RescaleOp {
  // Rescale Kernel
  RescaleOp(const transpose_conv2d_test_operands* cur_test, const mli_tensor& input, const mli_tensor& weights) {
    mem_b_keeper = memory_manager((int8_t*)(g_scratch_mem_b), sizeof(g_scratch_mem_b));
    mem_bias_out_keeper =  memory_manager ((int8_t*)(g_scratch_mem_bias_out), sizeof(g_scratch_mem_bias_out));
    mem_out_keeper =  memory_manager ((int8_t*)(g_scratch_mem_out), sizeof(g_scratch_mem_out));

    bias_in = cur_test->bias_in.get_quantized_tensor(mem_b_keeper.allocate_memory(cur_test->bias_in));
    bias_out = cur_test->bias_out.get_quantized_tensor(mem_bias_out_keeper.allocate_memory(cur_test->bias_out));
    out = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));

    // Note: The data of bias_out is the zero point of output.
    //  Due to the different quantization rules (round) in test components, bias_out.data is not the same as output.zero_point.
    //  This issue will causes result comparaison with test vectors failed. So, as a workaround,
    //  here we just assign the zero point of output to bias_out. And it is always a scalar.
    assert(out.el_params.sa.zero_point.capacity == 0);
    assert(bias_out.rank == 0);
    bias_out.data.mem.i8 = out.el_params.sa.zero_point.mem.i8;

    original_bias_out = bias_out;
    original_out = out;

    // additional params for MLI3 semantic
    mli3_bias = bias_folder(bias_in, input, weights, /* mirror_weights = */ true);
    mli3_scales_keeper = scales_calc(cur_test->in_scale, cur_test->out_scale,
                                     cur_test->w_scales, cur_test->w_scales_size);

  }

  // memory management for ins & outs tensors
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

struct ClipOp {
  // Clip Kernel
  ClipOp(const transpose_conv2d_test_operands* cur_test, const mli_tensor& out) {
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
                      const transpose_conv2d_test_operands* cur_test,
                      const TransposeConv2DOp& transpose_conv2d_op, const RescaleOp& rs_op,
                      const ClipOp& clp_op) {
    bool is_test_passed = true;

    if (!(cur_test->in.is_valid() && cur_test->weights.is_valid() &&
        cur_test->bias_in.is_valid() && cur_test->bias_out.is_valid() && cur_test->out.is_valid())) {
      reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
      is_test_passed = false;
    }

    if (is_test_passed &&
        (tensor_quantizer::validate_tensor(transpose_conv2d_op.input) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(transpose_conv2d_op.weights) != tensor_quantizer::kOk||
         tensor_quantizer::validate_tensor(transpose_conv2d_op.out_acc) != tensor_quantizer::kOk ||
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
        (transpose_conv2d_op.mem_in_keeper.is_memory_corrupted() || rs_op.mem_out_keeper.is_memory_corrupted() ||
         transpose_conv2d_op.mem_out_acc_keeper.is_memory_corrupted() || rs_op.mem_bias_out_keeper.is_memory_corrupted() ||
         transpose_conv2d_op.mem_w_keeper.is_memory_corrupted() || rs_op.mem_b_keeper.is_memory_corrupted())) {
      reporter.report_message(cur_test->descr,
        "FAILED at quantization step: memory beside one of operands is corrupted");
      is_test_passed = false;
    }

    return is_test_passed;
}

void prepare_phase(const transpose_conv2d_test_operands* cur_test,
                   TransposeConv2DOp& cnv_op, RescaleOp &rs_op, ClipOp &clp_op,
                   uint32_t& num_tiles, lib_mli::Buffer& encoded_params_buffer) {
 
  int32_t iteration_order[kTransposeConvIOIterRank]{ 0, 1, 2, 3, 4};
  uint32_t input_shape[kTransposeConvIORank] = { BATCH_SIZE,
                                                 cnv_op.input.shape[FMAP_H_DIM_HWC],
                                                 cnv_op.input.shape[FMAP_W_DIM_HWC],
                                                 NUM_GROUPS, 
                                                 cnv_op.input.shape[FMAP_C_DIM_HWC] };

  uint32_t weight_shape[kTransposeConvWRank] = { NUM_GROUPS,
                                                 cnv_op.weights.shape[KRNL_H_DIM_HWCN],
                                                 cnv_op.weights.shape[KRNL_W_DIM_HWCN],
                                                 cnv_op.weights.shape[KRNL_D_DIM_HWCN],
                                                 cnv_op.weights.shape[KRNL_C_DIM_HWCN] };

  uint32_t tile_oc = weight_shape[kKernelChannelOutDim];
  uint32_t tile_input_size[kTransposeConvIORank];
  for (uint32_t i = 0; i < kTransposeConvIORank; i++) {
    tile_input_size[i] = input_shape[i];
  }

#ifdef USE_TILING
  tile_oc = 2;
  // TODO: now if input last_tile_size is smaller than kernel H/W - tiling will be disabled in this dimension! - some better idea needed
  const uint32_t tile_hw = 2;
  tile_input_size[kGroupTensorHeightDim] = MAX(tile_hw, weight_shape[kKernelHeightDim]);
  tile_input_size[kGroupTensorWidthDim] = MAX(tile_hw, weight_shape[kKernelWidthDim]);
#endif
 
  int32_t input_stride[kTransposeConvIORank] = { BATCH_SIZE * NUM_GROUPS * int32_t(cnv_op.input.shape[FMAP_H_DIM_HWC]) * \
                                                cnv_op.input.mem_stride[FMAP_H_DIM_HWC],
                                                NUM_GROUPS * cnv_op.input.mem_stride[FMAP_H_DIM_HWC],
                                                NUM_GROUPS * cnv_op.input.mem_stride[FMAP_W_DIM_HWC],
                                                cnv_op.input.mem_stride[FMAP_W_DIM_HWC],
                                                cnv_op.input.mem_stride[FMAP_C_DIM_HWC]};

  int32_t weight_stride[kTransposeConvWRank] = { NUM_GROUPS * int32_t(cnv_op.weights.shape[KRNL_H_DIM_HWCN]) * \
                                                 cnv_op.weights.mem_stride[KRNL_H_DIM_HWCN],
                                                 cnv_op.weights.mem_stride[KRNL_H_DIM_HWCN],
                                                 cnv_op.weights.mem_stride[KRNL_W_DIM_HWCN],
                                                 cnv_op.weights.mem_stride[KRNL_D_DIM_HWCN],
                                                 cnv_op.weights.mem_stride[KRNL_C_DIM_HWCN] };

  uint32_t output_shape[kTransposeConvIORank] = { BATCH_SIZE,
                                                  cnv_op.out_acc.shape[FMAP_H_DIM_HWC],
                                                  cnv_op.out_acc.shape[FMAP_W_DIM_HWC],
                                                  NUM_GROUPS,
                                                  cnv_op.out_acc.shape[FMAP_C_DIM_HWC] };

  int32_t output_stride[kTransposeConvIORank] = { BATCH_SIZE * NUM_GROUPS * int32_t(cnv_op.out_acc.shape[FMAP_H_DIM_HWC]) * \
                                                  cnv_op.out_acc.mem_stride[FMAP_H_DIM_HWC],
                                                  NUM_GROUPS * cnv_op.out_acc.mem_stride[FMAP_H_DIM_HWC],
                                                  NUM_GROUPS * cnv_op.out_acc.mem_stride[FMAP_W_DIM_HWC],
                                                  cnv_op.out_acc.mem_stride[FMAP_W_DIM_HWC],
                                                  cnv_op.out_acc.mem_stride[FMAP_C_DIM_HWC]};

  assert(input_shape[kGroupTensorBatchDim] == BATCH_SIZE && output_shape[kGroupTensorBatchDim] == BATCH_SIZE);
  assert(input_shape[kGroupTensorChannelDim] == weight_shape[kKernelChannelInDim]);
  assert(weight_shape[kKernelChannelOutDim] == output_shape[kGroupTensorChannelDim]);

  const lib_mli::Tensor<lib_mli::NoBuffer, kTransposeConvIORank> full_in_tensor(input_shape, input_stride);

  const lib_mli::Tensor<lib_mli::NoBuffer, kTransposeConvIORank> full_out_tensor(output_shape, output_stride);
  uint32_t effective_kernel_size[kTransposeConvWRank]{1, weight_shape[kKernelHeightDim], weight_shape[kKernelWidthDim],
                                                      1, weight_shape[kKernelChannelOutDim]};

  uint32_t stride[kTransposeConvIORank]{ 1, cur_test->cfg.stride_height, cur_test->cfg.stride_width, 1, 1};
  uint32_t pre_padding[kTransposeConvIORank]{ 0, cur_test->cfg.padding_top, cur_test->cfg.padding_left, 0, 0};

  const lib_mli::Tensor<lib_mli::NoBuffer, kTransposeConvWRank> wt_tensor(weight_shape, weight_stride);

  auto i_w_wzp_o_tensor_its = lib_mli::GetDeconvTensorIterators(full_in_tensor, tile_input_size,
                                                                full_out_tensor, tile_oc,
                                                                wt_tensor, effective_kernel_size, stride, pre_padding, iteration_order);

  auto in_tensor_it = std::get<0>(i_w_wzp_o_tensor_its);
  auto w_tensor_it = std::get<1>(i_w_wzp_o_tensor_its);
  auto wzp_tensor_it = std::get<2>(i_w_wzp_o_tensor_its);
  auto out_tensor_it = std::get<3>(i_w_wzp_o_tensor_its);

  num_tiles = in_tensor_it.GetTotalCount();
#ifndef USE_TILING
  assert(num_tiles == BATCH_SIZE);
#endif

  // STEP 1.1.1: Construct [TransposeConv2D] as a specific ExecutionInterface successor
  //==================================================================


  lib_mli::PlatformDescription pd;
  lib_ref::KernelsFactory kernel_factory(pd);
  uint32_t transpose_conv2d_cs_size = kernel_factory.TransposeConv2D_CS_GetSize();
  void* transpose_conv2d_cs_buffer = malloc(transpose_conv2d_cs_size);

  lib_mli::TransposeConv2DConfig cfg(
    cur_test->cfg.stride_height, cur_test->cfg.stride_width,
    cur_test->cfg.padding_top, cur_test->cfg.padding_left,
    cur_test->cfg.padding_bottom, cur_test->cfg.padding_right);

  uint32_t izp_shape[kInpZPRank]{ 1 };
  lib_mli::Tensor<lib_mli::NoBuffer, kInpZPRank> izp_tensor(izp_shape);
  lib_mli::TensorIterator<lib_mli::NoBuffer, kInpZPRank, kTransposeConvIterRank> izp_tensor_it(izp_tensor);
  auto transpose_conv2d_op = kernel_factory.TransposeConv2D_CS(transpose_conv2d_cs_buffer, in_tensor_it, izp_tensor_it, w_tensor_it, wzp_tensor_it, cfg, out_tensor_it);

  // STEP 1.1.2: Construct [Rescale] as a specific ExecutionInterface successor
  //==================================================================

  mli_tensor &rs_input_tsr = cnv_op.out_acc;
  const mli_tensor &rs_inbias_tsr = rs_op.mli3_bias.get_bias_tsr();
  const mli_tensor &rs_scale_tsr = rs_op.mli3_scales_keeper.get_scales_tsr();
  const mli_tensor &rs_shift_tsr = rs_op.mli3_scales_keeper.get_shift_tsr();
  mli_tensor &rs_outbias_tsr = rs_op.bias_out;
  mli_tensor &rs_output_tsr = rs_op.out;

  void* &rescale_instance = rs_op.rescale_instance;
  uint32_t &rescale_instance_size = rs_op.rescale_instance_size;
  void* &rescale_conf_private = rs_op.rescale_conf_private;
  uint32_t &rescale_conf_private_size = rs_op.rescale_conf_private_size;

  static_assert(kRescaleRank == kTransposeConvIORank - 1 && kTransposeConvIORank > 0);
  static_assert(kRescaleIterRank == kTransposeConvIOIterRank - 1 && kTransposeConvIOIterRank > 0);

  uint32_t io_output_shape[kRescaleRank]{ output_shape[kGroupTensorBatchDim],
                                          output_shape[kGroupTensorHeightDim],
                                          output_shape[kGroupTensorWidthDim],
                                          output_shape[kGroupTensorChannelDim] };
  int32_t io_output_strides[kRescaleRank]{ output_stride[kGroupTensorBatchDim],
                                           output_stride[kGroupTensorHeightDim],
                                           output_stride[kGroupTensorWidthDim],
                                           output_stride[kGroupTensorChannelDim] };
  lib_mli::Tensor<lib_mli::NoBuffer, kRescaleRank> io_tensor(io_output_shape, io_output_strides);
  lib_mli::IteratorCfg<kRescaleRank> io_it_cfg(out_tensor_it.get_config(), kGroupTensorGroupDim);
  lib_mli::TensorIterator<lib_mli::NoBuffer, kRescaleRank, kRescaleIterRank> io_tensor_it(io_tensor, io_it_cfg);

  uint32_t ep_shape[kRescaleParamRank]{ output_shape[kGroupTensorChannelDim] };
  lib_mli::Tensor<lib_mli::NoBuffer, kRescaleParamRank> ep_tensor(ep_shape);
  const int32_t ep_it_order[kRescaleIterRank]{ kSkipIterDim, kSkipIterDim, kSkipIterDim, 0};
  const int32_t ep_zero_inc_mask[kRescaleIterRank]{ 1, 1, 1, 0};
  lib_mli::TensorIterator<lib_mli::NoBuffer, kRescaleParamRank, kRescaleIterRank> ep_tensor_it(ep_tensor, io_tensor_it, ep_it_order, ep_zero_inc_mask);

  uint32_t rescale_cs_size = kernel_factory.Rescale_CS_GetSize();
  void* rescale_cs_buffer = malloc(rescale_cs_size);

  lib_mli::RescaleConfig rs_cfg;
  if (mli_hlp_count_elem_num(&rs_scale_tsr, 0) == 1) {
      rs_cfg.axis = kPerTensorQuantDim;
  } else {
      rs_cfg.axis = kRescaleRank - 1;
  }

  auto rescale_op = kernel_factory.Rescale_CS(rescale_cs_buffer, io_tensor_it, rs_cfg, ep_tensor_it, io_tensor_it);

  // STEP 1.1.3: Construct [Clip] as a specific ExecutionInterface successor
  //==================================================================

  mli_tensor &clip_input_tsr = rs_op.original_out;
  mli_tensor &clip_output_tsr = rs_op.original_out;
  void* &clip_instance = clp_op.clip_instance;
  uint32_t &clip_instance_size = clp_op.clip_instance_size;
  lib_mli::PrivateData* &clip_conf_private = clp_op.clip_conf_private;
  uint32_t &clip_conf_private_size = clp_op.clip_conf_private_size;

  uint32_t clip_cs_size = kernel_factory.Clip_CS_GetSize();
  void* clip_cs_buffer = malloc(clip_cs_size);

  static_assert(kRescaleRank == kClipRank);
  static_assert(kRescaleIterRank == kClipIterRank);
  auto clip_op = kernel_factory.Clip_CS(clip_cs_buffer, io_tensor_it, io_tensor_it);

  // STEP 1.2.1: [TransposeConv2D] Memory management (Up to user on how to deal with it)
  //==================================================================
  uint32_t in_mem_offset = 0;
  uint32_t w_mem_offset = 0;
  uint32_t inpzp_mem_offset = 0;
  uint32_t wtszp_mem_offset = 0;
  uint32_t offsets[1] = {0};

  // NOTE: Currently, only supoort these data types.
  assert(cnv_op.input.el_type == MLI_EL_SA_8);
  assert(cnv_op.weights.el_type == MLI_EL_SA_8);
  assert(cnv_op.out_acc.el_type == MLI_EL_SA_32);

  // Define buffers for in\out tensors
  // Leave space for runtime object
  uint32_t* cnv_offset = &offsets[0];
  uint32_t runtime_obj_size = transpose_conv2d_op->GetRuntimeObjectSize();
  *cnv_offset += runtime_obj_size;

  // Leave space for private data buffer
  cnv_offset = &offsets[0];
  uint32_t private_buffer_size = transpose_conv2d_op->GetKernelPrivateDataSize();
  *cnv_offset += private_buffer_size;

  // TransposeConv2D input
  cnv_offset = &offsets[0];
  uint32_t cnv_i_elem_size = mli_hlp_tensor_element_size(&cnv_op.input);
  uint32_t in_size = GetBufferSize(lib_mli::kTransposeConvIORank, input_shape, input_stride) * cnv_i_elem_size;
  lib_mli::OffsetBuffer transpose_conv2d_in_buf{*cnv_offset, 0, in_size, cnv_i_elem_size};
  in_mem_offset = *cnv_offset;
  *cnv_offset += in_size;

  // TransposeConv2D weight
  cnv_offset = &offsets[0];
  uint32_t cnv_w_elem_size = mli_hlp_tensor_element_size(&cnv_op.weights);
  uint32_t w_size = GetBufferSize(lib_mli::kTransposeConvWRank, weight_shape, weight_stride) * cnv_w_elem_size;
  lib_mli::OffsetBuffer transpose_conv2d_w_buf{*cnv_offset, 0, w_size, cnv_w_elem_size};
  w_mem_offset = *cnv_offset;
  *cnv_offset += w_size;

  // TransposeConv2D output
  cnv_offset = &offsets[0];
  // NOTE: The output should be aligned, otherwise, it will cause `vvst` crash.
  //       For example, offset is 4 byts aligned if output is int32_t.
  uint32_t cnv_o_elem_size = mli_hlp_tensor_element_size(&cnv_op.out_acc);
  *cnv_offset = CEIL_RND(*cnv_offset, cnv_o_elem_size);
  uint32_t out_size = GetBufferSize(lib_mli::kTransposeConvIORank, output_shape, output_stride) * cnv_o_elem_size;
  lib_mli::OffsetBuffer transpose_conv2d_out_buf{*cnv_offset, 0, out_size, cnv_o_elem_size};
  uint32_t cnv_out_mem_offset = *cnv_offset;
  *cnv_offset += out_size;

  // TransposeConv2D input zero point
  cnv_offset = &offsets[0];
  uint32_t inpzp_size = transpose_conv2d_op->GetEncodedInpZeroPtsSize() * cnv_i_elem_size;
  lib_mli::OffsetBuffer inpzp_buf{*cnv_offset, 0, inpzp_size, cnv_i_elem_size};
  inpzp_mem_offset = *cnv_offset;
  *cnv_offset += inpzp_size;

  // TransposeConv2D weights zero point
  cnv_offset = &offsets[0];
  uint32_t wtszp_size = tile_oc * cnv_w_elem_size;
  lib_mli::OffsetBuffer wtszp_buf{*cnv_offset, 0, wtszp_size, cnv_w_elem_size};
  wtszp_mem_offset = *cnv_offset;
  *cnv_offset += wtszp_size;

  // MLI tensor structures and conv2d configuration
  cnv_offset = &offsets[0];
  uint32_t ctrl_buffer_size = transpose_conv2d_op->GetCtrlBufferSize();
  lib_mli::OffsetBuffer transpose_conv2d_descr_buf{*cnv_offset, 0, ctrl_buffer_size, sizeof(char)};
  *cnv_offset += ctrl_buffer_size;

  assert(ctrl_buffer_size == 0);
  assert(*cnv_offset <= kMemPoolSize);

  // DataBuffer size is 0 for reference kernel
  mli_status status = MLI_STATUS_OK;

  status = transpose_conv2d_op->AttachBufferOffsets(transpose_conv2d_in_buf,
                                                    transpose_conv2d_out_buf,
                                                    transpose_conv2d_w_buf,
                                                    inpzp_buf,
                                                    wtszp_buf,
                                                    transpose_conv2d_descr_buf);
  assert(status == MLI_STATUS_OK);

  // STEP 1.2.2: [Rescale] Memory management (Up to user on how to deal with it)
  //==================================================================
  uint32_t encoded_params_mem_offset = 0;

  // Define buffers for in\out tensors
  // Leave space for runtime object
  uint32_t* rs_offset = &offsets[0];
  *rs_offset = CEIL_RND(*rs_offset, kMliAlignment);
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
  lib_mli::OffsetBuffer rescale_in_buf { cnv_out_mem_offset, 0, rs_in_size,
                                          input_elem_size };

  // rescale output
  rs_offset = &offsets[0];
  uint32_t output_elem_size = mli_hlp_tensor_element_size(&rs_output_tsr);
  uint32_t rs_out_size = rescale_op->GetOutputBufferSize() * output_elem_size;
  lib_mli::OffsetBuffer rescale_out_buf { *rs_offset, 0, rs_out_size,
                                           output_elem_size };
  uint32_t rs_out_mem_offset = *rs_offset;
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
  uint32_t rs_ctrl_buffer_size = rescale_op->GetCtrlBufferSize();
  lib_mli::OffsetBuffer rescale_descr_buf { *rs_offset, 0,
                                          rs_ctrl_buffer_size, sizeof(char) };
  *rs_offset += rs_ctrl_buffer_size;
  assert(*rs_offset < kMemPoolSize);

  // Attaching buffer (descriptors) to the operation
  status = rescale_op->AttachBufferOffsets(rescale_in_buf,
                                           rescale_out_buf,
                                           encoded_params_buf,
                                           rescale_descr_buf);
  assert(status == MLI_STATUS_OK);

  // STEP 1.2.3: [Clip] Memory management (Up to user on how to deal with it)
  //==================================================================

  // Define buffers for in\out tensors
  // Leave space for runtime object
  uint32_t* clip_offset = &offsets[0];
  *rs_offset = CEIL_RND(*rs_offset, kMliAlignment);
  int8_t* clip_runtime_obj_addr = (int8_t*)g_mem_pool + offsets[0];
  uint32_t clip_runtime_obj_size = clip_op->GetRuntimeObjectSize();
  *clip_offset += clip_runtime_obj_size;

  // Leave space for private data buffer
  clip_offset = &offsets[0];
  uint32_t clip_private_buffer_size = clip_op->GetKernelPrivateDataSize();
  *clip_offset += clip_private_buffer_size;

  // clip input
  uint32_t clip_input_elem_size = mli_hlp_tensor_element_size(&clip_input_tsr);
  uint32_t clip_in_size = clip_op->GetInputBufferSize() * clip_input_elem_size;
  lib_mli::OffsetBuffer clip_in_buf{rs_out_mem_offset, 0, clip_in_size,
                                    clip_input_elem_size};

  // clip output
  clip_offset = &offsets[0];
  uint32_t clip_output_elem_size = mli_hlp_tensor_element_size(&clip_output_tsr);
  uint32_t clip_out_size = clip_op->GetOutputBufferSize() * clip_output_elem_size;
  lib_mli::OffsetBuffer clip_out_buf{rs_out_mem_offset, 0, clip_out_size,
                                     clip_output_elem_size};

  // clip min
  clip_offset = &offsets[0];
  uint32_t clip_encoded_params_size = clip_op->GetEncodedParamsSize();
  lib_mli::OffsetBuffer clip_encoded_params_buf {*clip_offset, 0, clip_encoded_params_size,
                                 sizeof(int8_t)};
  *clip_offset += clip_encoded_params_size;;

  // DataBuffer size is 0 for reference kernel
  clip_offset = &offsets[0];
  uint32_t clip_ctrl_buffer_size = clip_op->GetCtrlBufferSize();
  lib_mli::OffsetBuffer clip_descr_buf{*clip_offset, 0,
                                       clip_ctrl_buffer_size, sizeof(char)};
  *clip_offset += clip_ctrl_buffer_size;
  assert(*clip_offset < kMemPoolSize);

  // Attaching buffer (descriptors) to the operation
  status = clip_op->AttachBufferOffsets(clip_in_buf,
                                        clip_out_buf,
                                        clip_encoded_params_buf,
                                        clip_descr_buf);
  assert(status == MLI_STATUS_OK);

  // STEP 1.3.1: [TransposeConv2D] Copy dataset from scratch buffer to the global shared memory pool
  //==================================================================

  // Copy input zero points and weights zero points to the temp host buffers
  //==================================================================
  assert(cnv_i_elem_size == sizeof(int8_t) && cnv_w_elem_size == sizeof(int8_t));
  uint32_t full_weights_size = transpose_conv2d_op->GetEncodedWeightsSize();
  assert(full_weights_size == lib_mli::service::GetBufferSize(kTransposeConvWRank, weight_shape, weight_stride) * cnv_w_elem_size);
  uint32_t full_wtszp_size = transpose_conv2d_op->GetEncodedWtsZeroPtsSize();
  uint32_t full_weights_and_wzp_size = full_wtszp_size + full_weights_size;
  uint32_t max_dst_encoded_buffer_size = MAX(full_weights_and_wzp_size, inpzp_size);
  assert(max_dst_encoded_buffer_size <= kWeightsAndWeightsZPBufferSize);

  // copy input zero point into inpzp_tensor
  void* src_izp_mem = malloc(inpzp_size);
  lib_mli::Buffer src_inpzp_buf(src_izp_mem, inpzp_size, cnv_i_elem_size);
  uint32_t inpzp_shape[kInpZPRank] = { 1 };
  lib_mli::Tensor<lib_mli::Buffer, kInpZPRank> inpzp_tensor(src_inpzp_buf, inpzp_shape);
  if (cnv_op.input.el_params.sa.dim == kPerTensorQuantDim) {
    assert(cnv_op.input.el_params.sa.zero_point.capacity == 0);
    inpzp_tensor.write(0, static_cast<int8_t>(cnv_op.input.el_params.sa.zero_point.mem.i16));
  }
  else {
    assert(cnv_op.input.el_params.sa.zero_point.capacity / sizeof(int16_t) == src_inpzp_buf.get_size());
    for (uint32_t i = 0; i < inpzp_size / cnv_i_elem_size; ++i) {
      inpzp_tensor.write(int(i), static_cast<int8_t>(cnv_op.input.el_params.sa.zero_point.mem.pi16[i]));
    }
  }

  // encode input zero point into dst_inpzp_buf
  lib_mli::Buffer dst_inpzp_buf((void*)g_weights_buf_mem, inpzp_size, cnv_i_elem_size);
  auto izp_tensor_it_with_buf = lib_mli::TensorIterator<lib_mli::Buffer, kInpZPRank, kTransposeConvIterRank>(inpzp_tensor);
  status = transpose_conv2d_op->EncodeInpZeroPts(izp_tensor_it_with_buf, dst_inpzp_buf);
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
    assert(full_wtszp_size == src_wtszp_buf.get_size());
    for (size_t i = 0; i < full_wtszp_size / cnv_w_elem_size; ++i) {
      wtszp_tensor.write(int(i), static_cast<int8_t>(cnv_op.weights.el_params.sa.zero_point.mem.pi16[i]));
    }
  }

  // copy weights into src_weights_buf
  void* src_w_mem = malloc(full_weights_size);
  lib_mli::Buffer src_weights_buf(src_w_mem, full_weights_size, cnv_w_elem_size);
  lib_mli::Tensor<lib_mli::Buffer, kTransposeConvWRank> weights_tensor(src_weights_buf, weight_shape);
  int32_t zero_offsets[kTransposeConvWRank]{};
  strided_copy_with_offsets(kTransposeConvWRank, cnv_w_elem_size,
                            cnv_op.weights.data.mem.pi8,
                            zero_offsets, zero_offsets, weight_stride,
                            weight_shape, weights_tensor.get_buf().get_ptr<int8_t>());

  // encode weights and weights zero point(s) in dst_w_wzp_encoded_buffer
  lib_mli::Buffer dst_w_wzp_encoded_buffer((void*)g_weights_buf_mem, full_weights_and_wzp_size, cnv_w_elem_size);
  auto w_tensor_it_with_buf = lib_mli::TensorIterator<lib_mli::Buffer, kTransposeConvWRank, kTransposeConvIterRank>(weights_tensor);
  auto wzp_tensor_it_with_buf = lib_mli::TensorIterator<lib_mli::Buffer, kWZPRank, kTransposeConvIterRank>(wtszp_tensor);
  status = transpose_conv2d_op->EncodeWeightsAndZeroPts(w_tensor_it_with_buf, wzp_tensor_it_with_buf, dst_w_wzp_encoded_buffer);
  assert(status == MLI_STATUS_OK);

  free(src_wzp_mem);
  free(src_w_mem);

  // Compile TransposeConv2D into the binary data
  //==================================================================
  cnv_op.transpose_conv2d_instance = (int8_t*)g_mem_pool;
  cnv_op.transpose_conv2d_instance_size = transpose_conv2d_op->GetRuntimeObjectSize();

  status =
      transpose_conv2d_op->GetKernelPrivateData((int8_t*)g_mem_pool + cnv_op.transpose_conv2d_instance_size);
  assert(status == MLI_STATUS_OK);
  cnv_op.transpose_conv2d_conf_private = (int8_t*)g_mem_pool + cnv_op.transpose_conv2d_instance_size;
  cnv_op.transpose_conv2d_conf_private_size = transpose_conv2d_op->GetKernelPrivateDataSize();

  // STEP 1.3.2: [Rescale] encode params into special buffer
  //==================================================================
  int8_t * host_src_buf = (int8_t *) malloc(encoded_params_size);
  uint32_t params_shape[kRescaleParamRank] = {rs_inbias_tsr.shape[0]};

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

  assert(encoded_params_size <= kRescaleEncodedParamBufSize);
  encoded_params_buffer = lib_mli::Buffer(g_rescale_buf_mem, encoded_params_size, sizeof(int8_t));

  lib_mli::Tensor<lib_mli::Buffer, kRescaleParamRank> inbias_tensor(src_inbias_buf, params_shape);
  lib_mli::Tensor<lib_mli::Buffer, kRescaleParamRank> scale_tensor(src_scale_buf, params_shape);
  lib_mli::Tensor<lib_mli::Buffer, kRescaleParamRank> shift_tensor(src_shift_buf, params_shape);
  lib_mli::Tensor<lib_mli::Buffer, kRescaleParamRank> outbias_tensor(src_outbias_buf, params_shape);

  if(rs_cfg.axis == kPerTensorQuantDim) {
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

  // Compile Rescale into the binary data
  //==================================================================
  rescale_instance = rs_runtime_obj_addr;
  rescale_instance_size = rescale_op->GetRuntimeObjectSize();
  rescale_conf_private = rs_runtime_obj_addr + rescale_instance_size;
  rescale_conf_private_size = rescale_op->GetKernelPrivateDataSize();

  status = rescale_op->GetKernelPrivateData(rescale_conf_private);

  assert(status == MLI_STATUS_OK);
  // STEP 1.3.3: [clip] Copy dataset from tensors to the global shared memory pool
  //==================================================================
  // Copy min data to the shared memory pool
  mli_minmax_t val_limit;
  val_limit = get_val_limit(&rs_op.out, &cur_test->cfg.relu);

  int8_t * clp_host_src_buf = (int8_t*) malloc(clip_encoded_params_size);
  int8_t * clp_host_dst_buf = (int8_t*) malloc(clip_encoded_params_size);
  uint32_t clp_params_shape[1] = {1};
  uint32_t limit_min_size = 1;
  uint32_t limit_max_size = 1;

  lib_mli::Buffer src_min_buf(clp_host_src_buf, limit_min_size, sizeof(int8_t));
  lib_mli::Buffer src_max_buf(clp_host_src_buf + limit_min_size, limit_max_size, sizeof(int8_t));
  lib_mli::Buffer clip_encoded_params_buffer(clp_host_dst_buf, clip_encoded_params_size, sizeof(int8_t));
  lib_mli::Tensor<lib_mli::Buffer, kClipParamRank> min_tensor(src_min_buf,
                                                              clp_params_shape);

  lib_mli::Tensor<lib_mli::Buffer, kClipParamRank> max_tensor(src_max_buf,
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
      const uint32_t idx = clip_encoded_params_buf.get_offset() + i;
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

  free(transpose_conv2d_cs_buffer);
  free(rescale_cs_buffer);
  free(clip_cs_buffer);
  free(host_src_buf);
  free(clp_host_src_buf);
  free(clp_host_dst_buf);

  /*
   * It will be overritten later output of tconv -> rescale -> clip chain,
   * but for some reason this is needed for correct CRC claculation
   * and tests 4, 5, 6, 7, 8 pass. If you delete this - tests 4, 5, 6, 7, 8 would fail.
   * TODO: In my opinion this is bug and these tests cases should be fixed to work without this workaround.
   */ 
  for (uint32_t i = 0; i < clip_out_size; ++i) {
    const uint32_t idx = clip_out_buf.get_offset() + i;
    g_mem_pool[idx] = clip_output_tsr.data.mem.pi8[i];
  }

}


void execution_phase(TransposeConv2DOp& cnv_op, RescaleOp &rs_op, ClipOp &clp_op,
                     uint32_t tiles_num, const lib_mli::Buffer& encoded_params_buffer) {
  // STEP 3: Execution phase
  //==================================================================

  uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool)};

  auto mli_conv = lib_mli::ExecutionInterface::Create(
                    cnv_op.transpose_conv2d_instance,
                    cnv_op.transpose_conv2d_instance_size,
                    cnv_op.transpose_conv2d_conf_private,
                    cnv_op.transpose_conv2d_conf_private_size,
                    membasis, sizeof(membasis) / sizeof(membasis[0]));

  auto mli_rescale = lib_mli::ExecutionInterface::Create(
                        rs_op.rescale_instance,
                        rs_op.rescale_instance_size,
                        rs_op.rescale_conf_private,
                        rs_op.rescale_conf_private_size,
                        membasis, sizeof(membasis) / sizeof(membasis[0]));

  auto mli_clip = lib_mli::ExecutionInterface::Create(
                      clp_op.clip_instance,
                      clp_op.clip_instance_size,
                      clp_op.clip_conf_private,
                      clp_op.clip_conf_private_size,
                      membasis, sizeof(membasis) / sizeof(membasis[0]));

  assert(mli_conv != nullptr);
  assert(mli_rescale != nullptr);
  assert(mli_clip != nullptr);


  auto mli_tconv2d_pimpl = dynamic_cast<lib_ref::TransposeConv2D*>(mli_conv);
  auto tconv2d_private = (lib_ref::TransposeConv2DPrivateData*)(cnv_op.transpose_conv2d_conf_private);
  auto rescale_pimpl = dynamic_cast<lib_ref::Rescale*>(mli_rescale);
  auto rescale_private = (lib_ref::RescalePrivateData*)(rs_op.rescale_conf_private);
  auto clip_private = (lib_ref::ClipPrivateData*)(clp_op.clip_conf_private);

  int32_t tile_input_strides[kTransposeConvIORank]{};
  tconv2d_private->input.get_mem_strides(tile_input_strides);
  int32_t tile_output_strides[kTransposeConvIORank]{};
  tconv2d_private->output.get_mem_strides(tile_output_strides);
  int32_t tile_weights_strides[kTransposeConvWRank]{};
  tconv2d_private->weights.get_mem_strides(tile_weights_strides);

  uint32_t input_tile_size[kTransposeConvIOIterRank]{};
  uint32_t output_tile_size[kTransposeConvIOIterRank]{};
  uint32_t weights_tile_size[kTransposeConvWIterRank]{};
  int32_t input_tile_offsets[kTransposeConvIOIterRank]{};
  int32_t output_tile_offsets[kTransposeConvIOIterRank]{};
  int32_t weights_tile_offsets[kTransposeConvWIterRank]{};
  const int32_t zero_offsets[kTransposeConvWIterRank]{};
  uint32_t enc_param_size = 0, inp_bias_offset = 0, scale_offset = 0, shift_offset = 0, out_bias_offset = 0;
  int32_t output_tile_offsets_4d[kClipRank];
  int32_t tile_output_strides_4d[kClipRank];
  uint32_t output_tile_size_4d[kClipRank];

  mli_status status = MLI_STATUS_OK;
  for (uint32_t i = 0; i < tiles_num; ++i) {

    mli_tconv2d_pimpl->GetIOSizesAndOffsets(input_tile_size, output_tile_size, weights_tile_size,
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
    strided_copy_with_offsets(kTransposeConvIORank, tconv2d_private->input.get_buf().get_elem_size(),
      cnv_op.input.data.mem.pi8,
      input_tile_offsets, zero_offsets, tile_input_strides,
      input_tile_size, (int8_t*)(g_mem_pool + tconv2d_private->input.get_buf().get_offset()));

    // copy weights from global to local buffer
    strided_copy_with_offsets(kTransposeConvWRank, tconv2d_private->weights.get_buf().get_elem_size(),
      cnv_op.weights.data.mem.pi8,
      weights_tile_offsets, zero_offsets, tile_weights_strides,
      weights_tile_size, (int8_t*)(g_mem_pool + tconv2d_private->weights.get_buf().get_offset()));

    // copy weights zps from global to local buffer
    int8_t* wtszp_tile_buf = (int8_t*)(g_mem_pool + tconv2d_private->weights_zp.get_buf().get_offset());
    for (uint32_t j = 0; j < weights_tile_size[kKernelChannelOutDim]; j++) {
      if (cnv_op.weights.el_params.sa.dim == kPerTensorQuantDim) {
        wtszp_tile_buf[j] = (int8_t)cnv_op.weights.el_params.sa.zero_point.mem.i16;
      }
      else {
        wtszp_tile_buf[j] = (int8_t)cnv_op.weights.el_params.sa.zero_point.mem.pi16[j];
      }
    }

    rescale_pimpl->GetIOSizesAndOffsets(enc_param_size, inp_bias_offset, scale_offset, shift_offset, out_bias_offset);

    const uint32_t enc_param_buf_offset = rescale_private->enc_param.get_buf().get_offset();
    const uint32_t num_enc_param = rescale_private->enc_param.get_dim(0);

    // copy rescale input bias from global to local buffer
    uint32_t enc_param_offset = output_tile_offsets_4d[3] * sizeof(int32_t);
    memcpy((void*)(g_mem_pool + enc_param_buf_offset + inp_bias_offset),
           (void*)encoded_params_buffer.get_ptr<int8_t>(enc_param_offset), sizeof(int32_t) * enc_param_size);

    // copy rescale scale from global to local buffer
    enc_param_offset = num_enc_param * sizeof(int32_t) + output_tile_offsets_4d[3] * sizeof(int16_t);
    memcpy((void*)(g_mem_pool + enc_param_buf_offset + scale_offset),
           (void*)encoded_params_buffer.get_ptr<int8_t>(enc_param_offset), sizeof(int16_t) * enc_param_size);

    // copy rescale shift from global to local buffer
    enc_param_offset = num_enc_param * ( sizeof(int32_t) + sizeof(int16_t) ) + output_tile_offsets_4d[3] * sizeof(int8_t);
    memcpy((void*)(g_mem_pool + enc_param_buf_offset + shift_offset),
           (void*)encoded_params_buffer.get_ptr<int8_t>(enc_param_offset), sizeof(int8_t) * enc_param_size);

    // copy rescale output bias from global to local buffer
    enc_param_offset = num_enc_param * (sizeof(int32_t) + sizeof(int16_t) + sizeof(int8_t) ) + output_tile_offsets_4d[3] * sizeof(int8_t);
    memcpy((void*)(g_mem_pool + enc_param_buf_offset + out_bias_offset),
           (void*)encoded_params_buffer.get_ptr<int8_t>(enc_param_offset), sizeof(int8_t) * enc_param_size);

    status = mli_conv->Prefetch();
    assert(status == MLI_STATUS_OK);
    status = mli_conv->Issue();
    assert(status == MLI_STATUS_OK);
    status = mli_conv->Update();
    assert(status == MLI_STATUS_OK);

    status = mli_rescale->Prefetch();
    assert(status == MLI_STATUS_OK);
    status = mli_rescale->Issue();
    assert(status == MLI_STATUS_OK);
    status = mli_rescale->Update();
    assert(status == MLI_STATUS_OK);

    status = mli_clip->Prefetch();
    assert(status == MLI_STATUS_OK);
    status = mli_clip->Issue();
    assert(status == MLI_STATUS_OK);
    status = mli_clip->Update();
    assert(status == MLI_STATUS_OK);

    // copy results from clip output tile to the global buffer
    strided_copy_with_offsets(kClipRank, clip_private->output.get_buf().get_elem_size(),
                              (int8_t*)g_mem_pool + clip_private->output.get_buf().get_offset(),
                              zero_offsets, output_tile_offsets_4d, tile_output_strides_4d,
                              output_tile_size_4d, clp_op.original_out.data.mem.pi8);
  }
}

bool postprocess_phase(const reporter_full& reporter,
                       const transpose_conv2d_test_operands* cur_test,
                       TransposeConv2DOp& transpose_conv2d_op, RescaleOp& rs_op, ClipOp& clp_op) {
  quality_metrics test_metrics;
  bool is_test_passed = true;

  auto& out = clp_op.original_out;
  mli_tensor source_out_tensor = clp_op.original_out;

  if (is_test_passed &&
      (transpose_conv2d_op.mem_in_keeper.is_memory_corrupted() || rs_op.mem_out_keeper.is_memory_corrupted() ||
       transpose_conv2d_op.mem_out_acc_keeper.is_memory_corrupted() || rs_op.mem_bias_out_keeper.is_memory_corrupted() ||
       transpose_conv2d_op.mem_w_keeper.is_memory_corrupted() || rs_op.mem_b_keeper.is_memory_corrupted())) {
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
    data_crc(transpose_conv2d_op.input);
    data_crc(transpose_conv2d_op.weights);
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

  reporter.report_header("MLI3.0|Kernels|Transpose Convolution 2D Tests");
  for (int i = 0; i < kTestsNum; ++i) {
    // get the current test case
    const transpose_conv2d_test_operands* cur_test = &tests_list[i];

// NOTE: MLI 3.0 kernel won't work for VDSP because optimized kernel requires modifications
#if PLATFORM != X86_PLATFORM && PLATFORM != GENERIC_ARC
    reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
    continue;
#endif

// NOTE: Copied from `test_mli_krn_transpose_conv2d.cc`, since using the same tect vectors.
#if defined(__Xvec_guard_bit_option) && (__Xvec_guard_bit_option == 0)
    if (strstr(cur_test->descr, "Test 1 SA8_SA8_SA32") != nullptr ||
        strstr(cur_test->descr, "Test 2-1 SA8_SA8_SA32 ReluGen") != nullptr ||
        strstr(cur_test->descr, "Test 2-2 SA8_SA8_SA32 Mem") != nullptr ||
        strstr(cur_test->descr, "Test 3 FX16 Str_1x1") != nullptr ||
        strstr(cur_test->descr, "Test 3 SA8_SA8_SA32 Str_1x1") != nullptr ||
        strstr(cur_test->descr, "Test 4 SA8_SA8_SA32 IO_Memstr") != nullptr ||
        strstr(cur_test->descr, "Test 5 SA8_SA8_SA32 IOW_Memstr") != nullptr ||
        strstr(cur_test->descr, "Test 6 SA8_SA8_SA32 k2x2 st2") != nullptr ||
        strstr(cur_test->descr, "Test 7 SA8_SA8_SA32 k4x4 st2") != nullptr ||
        strstr(cur_test->descr, "Test 8 FX16 k3x3 str2") != nullptr ||
        strstr(cur_test->descr, "Test 8 SA8_SA8_SA32 k3x3 st2") != nullptr) {
      // VPX fails bitwise comparison with reference .
      reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
      continue;
    }
#endif

#if PLATFORM == V2DSP_XY && defined(CRC_RM_CONVERGENT)
    if (strstr(cur_test->descr, "Test 1 SA8_SA8_SA32") != nullptr ||
        strstr(cur_test->descr, "Test 2-1 SA8_SA8_SA32 ReluGen") != nullptr ||
        strstr(cur_test->descr, "Test 2-2 SA8_SA8_SA32 Mem") != nullptr ||
        strstr(cur_test->descr, "Test 6 SA8_SA8_SA32 k2x2 st2") != nullptr ||
        strstr(cur_test->descr, "SA8_SA8_SA32 k3x3 st2") != nullptr) {
      // Em9d fails bitwise comparison with reference .
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
    TransposeConv2DOp transpose_conv2d_op(cur_test);
    RescaleOp rs_op(cur_test, transpose_conv2d_op.input, transpose_conv2d_op.weights);
    ClipOp clp_op(cur_test,rs_op.out);

    bool is_test_passed = preprocess_phase(reporter, cur_test, transpose_conv2d_op, rs_op, clp_op);

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
            rescale_axis = kPerTensorQuantDim;
        } else {
            rescale_axis = transpose_conv2d_op.out_acc.rank - 1;
        }

        // If per-axis computation && out_bias is one element,
        // so construct out_bias tensor as vector the same as other params.
        if((rescale_axis != kPerTensorQuantDim) && (mli_hlp_count_elem_num(&outbias_tsr, 0) == 1)) {
            outbias_data = vectorize_single_elem_tensor(outbias_tsr, inbias_tsr);
        }

        // If per-tensor computation && in_bias is vector,
        // so construct out_bias, shift and scale tensors as vectors the same as in_bias.
        if((rescale_axis == kPerTensorQuantDim) && (mli_hlp_count_elem_num(&inbias_tsr, 0) != 1)) {
            outbias_data = vectorize_single_elem_tensor(outbias_tsr, inbias_tsr);
            shift_data = vectorize_single_elem_tensor(shift_tsr, inbias_tsr);
            scale_data = vectorize_single_elem_tensor(scale_tsr, inbias_tsr);
        }
    }

    // STEP 1: Preparing phase
    //==================================================================
    uint32_t num_tiles = 0; // num_tiles calculated inside prepare_phase
    lib_mli::Buffer encoded_params_buffer; // initialized inside prepare_phase
    prepare_phase(cur_test, transpose_conv2d_op, rs_op, clp_op, num_tiles, encoded_params_buffer);

    // STEP 2: Executing phase
    //==================================================================
    // Run conv2d, rescale and clip MLI3.0 kernels
    execution_phase(transpose_conv2d_op, rs_op, clp_op, num_tiles, encoded_params_buffer);

    // STEP 3: Postprocessing phase
    //==================================================================
    is_test_passed &= postprocess_phase(reporter, cur_test, transpose_conv2d_op, rs_op, clp_op);

    final_status &= is_test_passed;

    // Free buffers for Rescale params
    free(outbias_data);
    free(shift_data);
    free(scale_data);
  }
  reporter.report_outline("[AUTO] Group: mli_krn_transpose_conv2d_30", final_status);

  return (final_status) ? 0 : 1;
}
