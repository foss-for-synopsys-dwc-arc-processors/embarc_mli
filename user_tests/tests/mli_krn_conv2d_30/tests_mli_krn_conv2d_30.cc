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
 * Uncomment USE_DEPRECTED_CONV_CS_CONSTRUCTOR if you want to call a combination of
 * deprecated Conv2D_CS constructor + SetIterators + AttachBufferOffsets methods.
 */
// #define USE_DEPRECTED_CONV_CS_CONSTRUCTOR

 /**
  * Comment USE_TILING if you want to use single tile.
  * In case of using together with USE_DEPRECTED_CONV_CS_CONSTRUCTOR - no SetIterators will be called.
  */
#define USE_TILING

using namespace snps_arc::metaware::mli::service;

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

using lib_mli::KConvIORank;
using lib_mli::KConvIOIterRank;
using lib_mli::KConvWRank;
using lib_mli::KConvWIterRank;
using lib_mli::kConvZPRank;
using lib_mli::kConvZPIterRank;
using lib_mli::kTensorChannelDim;
using lib_mli::kKernelChannelOutDim;

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
                      const Conv2dOp& conv2d_op, const RescaleOp& rs_op,
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

template<typename T>
static void hwc_to_bhwc(const T src[4], T dst[4], T b_val) {
  dst[0] = b_val;
  for (int i = 0; i < 3; i++) {
    dst[1 + i] = src[i];
  }
}

void prepare_phase(const conv2d_test_operands* cur_test, const Tiling& tiling,
                  uint32_t iteration_order[KConvIOIterRank],
                  Conv2dOp& cnv_op, RescaleOp &rs_op, ClipOp &clp_op) {

  // BHWC layout
  uint32_t total_input_size[KConvIORank];
  uint32_t total_output_size[KConvIORank];
  uint32_t first_tile_size[KConvIORank];
  uint32_t tile_size[KConvIORank];
  uint32_t input_tile_first_inc[KConvIORank];
  uint32_t output_tile_first_inc[KConvIORank];
  uint32_t input_tile_inc[KConvIORank];
  uint32_t output_tile_inc[KConvIORank];
  tiling.get_io_tiles_parameters(total_input_size, total_output_size,
                                 first_tile_size, tile_size,
                                 input_tile_first_inc, output_tile_first_inc,
                                 input_tile_inc, output_tile_inc);

  uint32_t tile_input_shape[KConvIORank];
  uint32_t tile_output_shape[KConvIORank];
  for (unsigned i = 0; i < KConvIORank; i++) {
    tile_input_shape[i] = MAX(first_tile_size[i], tile_size[i]);
    tile_output_shape[i] = MAX(output_tile_first_inc[i], output_tile_inc[i]);
  }

  // GHWCinCo vs. HWCinCo
  uint32_t weight_shape[KConvWRank] = { 1, cnv_op.weights.shape[0], cnv_op.weights.shape[1],
                               cnv_op.weights.shape[2], cnv_op.weights.shape[3]};
  int32_t weight_stride[KConvWRank] = {int32_t(cnv_op.weights.shape[0] * cnv_op.weights.mem_stride[0]),
                              cnv_op.weights.mem_stride[0], cnv_op.weights.mem_stride[1],
                              cnv_op.weights.mem_stride[2], cnv_op.weights.mem_stride[3]};

  uint32_t tile_weights_shape[KConvWRank];
  for (unsigned i = 0; i < KConvWRank - 1; i++) {
    tile_weights_shape[i] = weight_shape[i];
  }
  tile_weights_shape[4] = output_tile_inc[kTensorChannelDim];
  

  int32_t input_stride[KConvIORank] = { int32_t(cnv_op.input.shape[0]) * cnv_op.input.mem_stride[0],
                                        cnv_op.input.mem_stride[0],
                                        cnv_op.input.mem_stride[1],
                                        cnv_op.input.mem_stride[2] };

  int32_t output_stride[KConvIORank] = {int32_t(tile_output_shape[1]) * cnv_op.out_acc.mem_stride[0],
                                        cnv_op.out_acc.mem_stride[0],
                                        cnv_op.out_acc.mem_stride[1],
                                        cnv_op.out_acc.mem_stride[2]};

  // G == 1
  assert(total_input_size[0] == 1 && tile_output_shape[0] == 1);
  // Input Cin == Weight Cin
  assert(total_input_size[3] == weight_shape[3]);
  // Weight Co = Output Co
  assert(weight_shape[4] == total_output_size[3]);

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

  const lib_mli::Tensor<lib_mli::NoBuffer, KConvIORank> in_tensor(total_input_size, input_stride);
  const lib_mli::Tensor<lib_mli::NoBuffer, KConvWRank> wt_tensor(weight_shape, weight_stride);

#ifdef USE_DEPRECTED_CONV_CS_CONSTRUCTOR
  const lib_mli::Tensor<lib_mli::NoBuffer, 4> out_tensor(tile_output_shape, output_stride);
  auto conv2d_op = kernel_factory.Conv2d_CS(conv2d_cs_buffer, in_tensor, wt_tensor, cfg, out_tensor);
#else
  int32_t count[KConvIOIterRank];
  int32_t input_first_increment[KConvIOIterRank];
  int32_t input_increment[KConvIOIterRank];
  int32_t input_last_increment[KConvIOIterRank];
  int32_t input_first_size[KConvIOIterRank];
  int32_t input_size[KConvIOIterRank];
  int32_t input_last_size[KConvIOIterRank];
  int32_t output_first_increment[KConvIOIterRank];
  int32_t output_increment[KConvIOIterRank];
  int32_t output_last_increment[KConvIOIterRank];
  int32_t output_first_size[KConvIOIterRank];
  int32_t output_size[KConvIOIterRank];
  int32_t output_last_size[KConvIOIterRank];
  tiling.get_io_parameters_for_tensor_iterator(count, true,
                                               input_first_increment, input_increment, input_last_increment,
                                               input_first_size, input_size, input_last_size,
                                               output_first_increment, output_increment, output_last_increment,
                                               output_first_size, output_size, output_last_size);
  int32_t iteration_order_signed[KConvIOIterRank];
  for (unsigned i = 0; i < KConvIOIterRank; i++) {
    iteration_order_signed[i] = (int32_t)iteration_order[i];
  }

  int32_t weights_iteration_order_signed[KConvWIterRank]{ 0, 1, 2, 3, 4 };  // TODO: maybe add some connection between i/o and w orders
  int32_t weights_count[KConvWIterRank]{ 1, 1, 1, 1, count[kTensorChannelDim]};
  int32_t weights_first_increment[KConvWIterRank]{ 0, 0, 0, 0, output_first_increment[kTensorChannelDim] };
  int32_t weights_increment[KConvWIterRank]{ 0, 0, 0, 0, output_increment[kTensorChannelDim] };
  int32_t weights_last_increment[KConvWIterRank]{ 0, 0, 0, 0, output_last_increment[kTensorChannelDim]};
  int32_t weights_first_size[KConvWIterRank];
  int32_t weights_size[KConvWIterRank];
  int32_t weights_last_size[KConvWIterRank];
  for (unsigned i = 0; i < KConvWIterRank - 1; i++) {
    weights_first_size[i] = tile_weights_shape[i];
    weights_size[i] = tile_weights_shape[i];
    weights_last_size[i] = tile_weights_shape[i];
  }
  weights_first_size[kKernelChannelOutDim] = output_first_size[kTensorChannelDim];
  weights_size[kKernelChannelOutDim] = output_size[kTensorChannelDim];
  weights_last_size[kKernelChannelOutDim] = output_last_size[kTensorChannelDim];

  int32_t wzp_iteration_order[kConvZPIterRank]{ 0 };
  int32_t wzp_count[kConvZPIterRank]{ count[kTensorChannelDim] };
  int32_t wzp_first_increment[kConvZPIterRank]{ output_first_increment[kTensorChannelDim] };
  int32_t wzp_increment[kConvZPIterRank]{ output_increment[kTensorChannelDim] };
  int32_t wzp_last_increment[kConvZPIterRank]{ output_last_increment[kTensorChannelDim] };
  int32_t wzp_first_size[kConvZPIterRank]{ output_first_size[kTensorChannelDim] };
  int32_t wzp_size[kConvZPIterRank]{ output_size[kTensorChannelDim] };
  int32_t wzp_last_size[kConvZPIterRank]{ output_last_size[kTensorChannelDim] };

  lib_mli::IteratorCfg<KConvIOIterRank> input_it_config(
    iteration_order_signed, count,
    input_first_increment, input_increment, input_last_increment,
    input_first_size, input_size, input_last_size
  );
  lib_mli::TensorIterator<lib_mli::NoBuffer, KConvIORank, KConvIOIterRank> in_tensor_it(in_tensor, input_it_config);

  lib_mli::IteratorCfg<KConvWIterRank> weights_it_config(
    weights_iteration_order_signed, weights_count,
    weights_first_increment, weights_increment, weights_last_increment,
    weights_first_size, weights_size, weights_last_size
  );
  lib_mli::TensorIterator<lib_mli::NoBuffer, KConvWRank, KConvWIterRank> w_tensor_it(wt_tensor, weights_it_config);

  lib_mli::IteratorCfg<KConvIOIterRank> output_it_config(
    iteration_order_signed, count,
    output_first_increment, output_increment, output_last_increment,
    output_first_size, output_size, output_last_size
  );
  lib_mli::Tensor<lib_mli::NoBuffer, KConvIORank> full_out_tensor(total_output_size, output_stride);
  lib_mli::TensorIterator<lib_mli::NoBuffer, KConvIORank, KConvIOIterRank> out_tensor_it(full_out_tensor, output_it_config);

  uint32_t wzp_shape[kConvZPRank]{tile_output_shape[kTensorChannelDim]};
  lib_mli::Tensor<lib_mli::NoBuffer, kConvZPIterRank> wzp_tensor(wzp_shape);
  lib_mli::IteratorCfg<kConvZPIterRank> wzp_it_config(
    wzp_iteration_order, wzp_count,
    wzp_first_increment, wzp_increment, wzp_last_increment,
    wzp_first_size, wzp_size, wzp_last_size
  );
  lib_mli::TensorIterator<lib_mli::NoBuffer, kConvZPRank, kConvZPIterRank> wzp_tensor_it(wzp_tensor, wzp_it_config);
  auto conv2d_op = kernel_factory.Conv2d_CS(conv2d_cs_buffer, in_tensor_it, w_tensor_it, wzp_tensor_it, cfg, out_tensor_it);
#endif

  // STEP 1.1.2: Construct [Rescale] as a specific ExecutionInterface successor
  //==================================================================

  mli_tensor &rs_input_tsr = cnv_op.out_acc;
  mli_tensor &rs_output_tsr = rs_op.out;

  void* &rescale_instance = rs_op.rescale_instance;
  uint32_t &rescale_instance_size = rs_op.rescale_instance_size;
  void* &rescale_conf_private = rs_op.rescale_conf_private;
  uint32_t &rescale_conf_private_size = rs_op.rescale_conf_private_size;

  for (int i = 0; i < 3; i++) {
    assert(rs_input_tsr.shape[i] == total_output_size[1 + i]);
    assert(rs_input_tsr.mem_stride[i] == output_stride[1 + i]);
    assert(rs_output_tsr.shape[i] == total_output_size[1 + i]);
    assert(rs_output_tsr.mem_stride[i] == output_stride[1 + i]);
  }

  const lib_mli::Tensor<lib_mli::NoBuffer, 4> input_tensor(tile_output_shape, output_stride);
  const lib_mli::Tensor<lib_mli::NoBuffer, 4> output_tensor(tile_output_shape, output_stride);

  uint32_t rescale_cs_size = kernel_factory.Rescale_CS_GetSize();
  void* rescale_cs_buffer = malloc(rescale_cs_size);

  lib_mli::RescaleConfig rs_cfg;
  if (mli_hlp_count_elem_num(&rs_op.mli3_scales_keeper.get_scales_tsr(), 0) == 1) {
      rs_cfg.axis = -1;
  } else {
      rs_cfg.axis = kTensorChannelDim;
  }

  auto rescale_op = kernel_factory.Rescale_CS(rescale_cs_buffer, input_tensor,
                                                           rs_cfg, output_tensor);

  // STEP 1.1.3: Construct [Clip] as a specific ExecutionInterface successor
  //==================================================================

  mli_tensor &clip_input_tsr = rs_op.original_out;
  mli_tensor &clip_output_tsr = rs_op.original_out;
  void* &clip_instance = clp_op.clip_instance;
  uint32_t &clip_instance_size = clp_op.clip_instance_size;
  lib_mli::PrivateData* &clip_conf_private = clp_op.clip_conf_private;
  uint32_t &clip_conf_private_size = clp_op.clip_conf_private_size;

  for (int i = 0; i < 3; i++) {
    assert(clip_input_tsr.shape[i] == total_output_size[1 + i]);
    assert(clip_input_tsr.mem_stride[i] == output_stride[1 + i]);
    assert(clip_output_tsr.shape[i] == total_output_size[1 + i]);
    assert(clip_output_tsr.mem_stride[i] == output_stride[1 + i]);
  }

  const lib_mli::Tensor<lib_mli::NoBuffer, 4> clip_input_tensor(total_output_size, output_stride);
  const lib_mli::Tensor<lib_mli::NoBuffer, 4> clip_output_tensor(total_output_size, output_stride);

  uint32_t clip_cs_size = kernel_factory.Clip_CS_GetSize();
  void* clip_cs_buffer = malloc(clip_cs_size);

  auto clip_op = kernel_factory.Clip_CS(clip_cs_buffer, clip_input_tensor, clip_output_tensor);

  // STEP 1.2.1: [Conv2D] Memory management (Up to user on how to deal with it)
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

  // Leave space for runtime object
  uint32_t* cnv_offset = &offsets[0];
  uint32_t runtime_obj_size = conv2d_op->GetRuntimeObjectSize();
  *cnv_offset += runtime_obj_size;

  // Leave space for private data buffer
  cnv_offset = &offsets[0];
  uint32_t private_buffer_size = conv2d_op->GetKernelPrivateDataSize();
  *cnv_offset += private_buffer_size;

  // conv2d input
  cnv_offset = &offsets[0];
  uint32_t cnv_i_elem_size = mli_hlp_tensor_element_size(&cnv_op.input);
  uint32_t in_size = GetBufferSize(KConvIORank, tile_input_shape, input_stride) * cnv_i_elem_size;
  lib_mli::OffsetBuffer conv2d_in_buf{*cnv_offset, 0, in_size, cnv_i_elem_size};
  lib_mli::Tensor<lib_mli::OffsetBuffer, KConvIORank> conv2d_in_tensor(conv2d_in_buf, total_input_size);
  in_mem_offset = *cnv_offset;
  *cnv_offset += in_size;

  // conv2d weight
  cnv_offset = &offsets[0];
  uint32_t cnv_w_elem_size = mli_hlp_tensor_element_size(&cnv_op.weights);
  uint32_t w_size = GetBufferSize(KConvWRank, tile_weights_shape, weight_stride) * cnv_w_elem_size;
  lib_mli::OffsetBuffer conv2d_w_buf{*cnv_offset, 0, w_size, cnv_w_elem_size};
  w_mem_offset = *cnv_offset;
  *cnv_offset += w_size;

  // conv2d output
  cnv_offset = &offsets[0];
  // NOTE: The output should be aligned, otherwise, it will cause `vvst` crash.
  //       For example, offset is 4 byts aligned if output is int32_t.
  uint32_t cnv_o_elem_size = mli_hlp_tensor_element_size(&cnv_op.out_acc);
  *cnv_offset = CEIL_RND(*cnv_offset, cnv_o_elem_size);
  uint32_t out_size = conv2d_op->GetOutputBufferSize() * cnv_o_elem_size;
  lib_mli::OffsetBuffer conv2d_out_buf{*cnv_offset, 0, out_size, cnv_o_elem_size};
  lib_mli::Tensor<lib_mli::OffsetBuffer, KConvIORank> conv2d_out_tensor(conv2d_out_buf, tile_output_shape);
  *cnv_offset += out_size;

  // conv2d input zero point
  cnv_offset = &offsets[0];
  // NOTE: ZP has fixed 16 bit in MLI internal
  uint32_t zp_elem_size = sizeof(int16_t);
  uint32_t inpzp_size = conv2d_op->GetEncodedInpZeroPtsSize() * zp_elem_size;
  lib_mli::OffsetBuffer inpzp_buf{*cnv_offset, 0, inpzp_size, zp_elem_size};
  inpzp_mem_offset = *cnv_offset;
  *cnv_offset += inpzp_size;

  // conv2d weights zero point
  cnv_offset = &offsets[0];
  uint32_t wtszp_size = tile_output_shape[kTensorChannelDim] * zp_elem_size;
  lib_mli::OffsetBuffer wtszp_buf{*cnv_offset, 0, wtszp_size, zp_elem_size};
  wtszp_mem_offset = *cnv_offset;
  *cnv_offset += wtszp_size;

  // MLI tensor structures and conv2d configuration
  cnv_offset = &offsets[0];
  uint32_t ctrl_buffer_size = conv2d_op->GetCtrlBufferSize();
  lib_mli::OffsetBuffer conv2d_ctrl_buf{*cnv_offset, 0, ctrl_buffer_size, sizeof(char)};
  *cnv_offset += ctrl_buffer_size;

  assert(ctrl_buffer_size == 0);
  assert(*cnv_offset <= kMemPoolSize);

  // DataBuffer size is 0 for reference kernel
  mli_status status = MLI_STATUS_OK;

#ifdef USE_DEPRECTED_CONV_CS_CONSTRUCTOR
  status = conv2d_op->AttachBufferOffsets(conv2d_in_tensor,
                                          conv2d_out_tensor,
                                          conv2d_w_buf,
                                          inpzp_buf,
                                          wtszp_buf,
                                          conv2d_ctrl_buf);
  assert(status == MLI_STATUS_OK);
#else
  status = conv2d_op->AttachBufferOffsets(conv2d_in_buf,
                                          conv2d_out_buf,
                                          conv2d_w_buf,
                                          inpzp_buf,
                                          wtszp_buf,
                                          conv2d_ctrl_buf);
  assert(status == MLI_STATUS_OK);
#endif

#ifdef USE_DEPRECTED_CONV_CS_CONSTRUCTOR
#ifdef USE_TILING
  uint32_t weights_inc[KConvWRank] = { 1, weight_shape[1], weight_shape[2], weight_shape[3], output_tile_inc[3] };
  status = conv2d_op->SetIterators(total_output_size, iteration_order,
                                   input_tile_first_inc, input_tile_inc,
                                   output_tile_first_inc, output_tile_inc,
                                   weights_inc);
  assert(status == MLI_STATUS_OK);
#endif
#endif

  // STEP 1.2.2: [Rescale] Memory management (Up to user on how to deal with it)
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

  // rescale input = conv output
  uint32_t input_elem_size = mli_hlp_tensor_element_size(&rs_input_tsr);
  uint32_t rs_in_size = GetBufferSize(4, tile_output_shape, output_stride) * input_elem_size;
  lib_mli::OffsetBuffer rescale_in_buf { conv2d_out_buf.get_offset(), 0, rs_in_size, input_elem_size };
  lib_mli::Tensor<lib_mli::OffsetBuffer, 4> rescale_in_tensor(rescale_in_buf, total_output_size);

  // rescale output
  rs_offset = &offsets[0];
  uint32_t output_elem_size = mli_hlp_tensor_element_size(&rs_output_tsr);
  uint32_t rs_out_size = rescale_op->GetOutputBufferSize() * output_elem_size;
  lib_mli::OffsetBuffer rescale_out_buf { *rs_offset, 0, rs_out_size, output_elem_size };
  lib_mli::Tensor<lib_mli::OffsetBuffer, 4> rescale_out_tensor(rescale_out_buf, tile_output_shape);
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
  lib_mli::OffsetBuffer rescale_ctrl_buf { *rs_offset, 0,
                                          rs_ctrl_buffer_size, sizeof(char) };
  *rs_offset += rs_ctrl_buffer_size;
  assert(*rs_offset <= kMemPoolSize);

  // Attaching buffer (descriptors) to the operation
  status = rescale_op->AttachBufferOffsets(rescale_in_tensor,
                                           rescale_out_tensor,
                                           encoded_params_buf,
                                           rescale_ctrl_buf);
  assert(status == MLI_STATUS_OK);

#ifdef USE_TILING
  status = rescale_op->SetIterators(total_output_size, iteration_order,
                                    output_tile_first_inc, output_tile_inc);
  assert(status == MLI_STATUS_OK);
#endif

  // STEP 1.2.3: [Clip] Memory management (Up to user on how to deal with it)
  //==================================================================

  // Leave space for runtime object
  uint32_t* clip_offset = &offsets[0];
  int8_t* clip_runtime_obj_addr = (int8_t*)g_mem_pool + offsets[0];
  uint32_t clip_runtime_obj_size = clip_op->GetRuntimeObjectSize();
  *clip_offset += clip_runtime_obj_size;

  // Leave space for private data buffer
  clip_offset = &offsets[0];
  uint32_t clip_private_buffer_size = clip_op->GetKernelPrivateDataSize();
  *clip_offset += clip_private_buffer_size;

  // clip input = rescale output
  uint32_t clip_input_elem_size = mli_hlp_tensor_element_size(&clip_input_tsr);
  uint32_t clip_in_size = GetBufferSize(4, tile_output_shape, output_stride) * clip_input_elem_size;
  lib_mli::OffsetBuffer clip_in_buf{rescale_out_buf.get_offset(), 0, clip_in_size, clip_input_elem_size};
  lib_mli::Tensor<lib_mli::OffsetBuffer, 4> clip_in_tensor(clip_in_buf, total_output_size);

  // clip output
  clip_offset = &offsets[0];
  uint32_t clip_output_elem_size = mli_hlp_tensor_element_size(&clip_output_tsr);
  uint32_t clip_out_size = clip_op->GetOutputBufferSize() * clip_output_elem_size;
  lib_mli::OffsetBuffer clip_out_buf{ *clip_offset, 0, clip_out_size, clip_output_elem_size};
  lib_mli::Tensor<lib_mli::OffsetBuffer, 4> clip_out_tensor(clip_out_buf, total_output_size);

  *clip_offset += clip_in_size;

  // clip min
  clip_offset = &offsets[0];
  uint32_t clip_encoded_params_size = clip_op->GetEncodedParamsSize();
  lib_mli::OffsetBuffer clip_encoded_params_buf {*clip_offset, 0, clip_encoded_params_size,
                                 sizeof(int8_t)};
  uint32_t clip_encoded_params_mem_offset = *clip_offset;
  *clip_offset += clip_encoded_params_size;;

  // DataBuffer size is 0 for reference kernel
  clip_offset = &offsets[0];
  uint32_t clip_ctrl_buffer_size = clip_op->GetCtrlBufferSize();
  lib_mli::OffsetBuffer clip_ctrl_buf{*clip_offset, 0,
                                       clip_ctrl_buffer_size, sizeof(char)};
  *clip_offset += clip_ctrl_buffer_size;
  assert(*clip_offset <= kMemPoolSize);

  // Attaching buffer (descriptors) to the operation
  status = clip_op->AttachBufferOffsets(clip_in_tensor,
                                        clip_out_tensor,
                                        clip_encoded_params_buf,
                                        clip_ctrl_buf);
  assert(status == MLI_STATUS_OK);
#ifdef USE_TILING
  status = clip_op->SetIterators(total_output_size, iteration_order,
                                 output_tile_first_inc, output_tile_inc);
  assert(status == MLI_STATUS_OK);
#endif

  // encode input zero points to the global mem pool
  //==================================================================
  size_t shared_buf_size = MAX(inpzp_size, wtszp_size);
  char * host_buf_a = (char*) malloc(shared_buf_size);
  char * host_buf_b = (char*) malloc(shared_buf_size);
  lib_mli::Buffer src_inpzp_buf(host_buf_a, inpzp_size, cnv_i_elem_size);
  lib_mli::Buffer dst_inpzp_buf(host_buf_b, inpzp_size, zp_elem_size);
  // NOTE: Current the input and weights are int8_t, and zp is int16_t.
  //       Later, we will support other types.
  assert(src_inpzp_buf.get_size() == inpzp_buf.get_size());
  assert(src_inpzp_buf.get_elem_size() * 2 == inpzp_buf.get_elem_size());

  uint32_t inpzp_shape[1] = {1};
  lib_mli::Tensor<lib_mli::Buffer, 1> inpzp_tensor(src_inpzp_buf, inpzp_shape);

  // input zero points: mli_tensor -> host tensor
  // NOTE: Zero Points should have the same size as the tensor they belong to.
  //       Since ZP is 16b in `mli_tensor`, so we should cast it to the same type as input.
  if (cnv_op.input.el_params.sa.dim == -1) {
    assert(cnv_op.input.el_params.sa.zero_point.capacity == 0);
    inpzp_tensor.write(0, static_cast<int8_t>(cnv_op.input.el_params.sa.zero_point.mem.i16));
  } else {
    assert(cnv_op.input.el_params.sa.zero_point.capacity == src_inpzp_buf.get_size());
    for (uint32_t i = 0; i < inpzp_size / zp_elem_size; ++i) {
      inpzp_tensor.write(int(i), static_cast<int8_t>(cnv_op.input.el_params.sa.zero_point.mem.pi16[i]));
    }
  }
  // host tensor 8bit -> encoded host buffer 16bit
  status = conv2d_op->EncodeInpZeroPts(inpzp_tensor, dst_inpzp_buf);
  assert(status == MLI_STATUS_OK);
  // encoded host buffer -> global mem pool
  auto inpzp_mem = reinterpret_cast<int16_t*>((int8_t*)g_mem_pool + inpzp_mem_offset);
  for (uint32_t i = 0; i < inpzp_size / zp_elem_size; ++i) {
    inpzp_mem[i] = dst_inpzp_buf.read<int16_t>(i);
  }

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

  // Compile Rescale into the binary data
  //==================================================================
  rescale_instance = rs_runtime_obj_addr;
  rescale_instance_size = rescale_op->GetRuntimeObjectSize();
  rescale_conf_private = rs_runtime_obj_addr + rescale_instance_size;
  rescale_conf_private_size = rescale_op->GetKernelPrivateDataSize();

  status = rescale_op->GetKernelPrivateData(rescale_conf_private);

  assert(status == MLI_STATUS_OK);

  // STEP 1.3.3: [clip] encode params to the global shared memory pool
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
  free(rescale_cs_buffer);
  free(clip_cs_buffer);
  free(host_buf_a);
  free(host_buf_b);
  free(clp_host_src_buf);
  free(clp_host_dst_buf);
}


void execution_phase(Conv2dOp& cnv_op, RescaleOp &rs_op, ClipOp &clp_op, uint32_t tiles_num) {
  // STEP 3: Execution phase
  //==================================================================

  uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool)};

  auto mli_conv = lib_mli::ExecutionInterface::Create(
                    cnv_op.conv2d_instance,
                    cnv_op.conv2d_instance_size,
                    cnv_op.conv2d_conf_private,
                    cnv_op.conv2d_conf_private_size,
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


  lib_ref::Conv2d* mli_conv2d_pimpl = dynamic_cast<lib_ref::Conv2d*>(mli_conv);
  lib_ref::Rescale* rescale_pimpl = dynamic_cast<lib_ref::Rescale*>(mli_rescale);
  lib_ref::Conv2DPrivateData * conv2d_private = (lib_ref::Conv2DPrivateData*)(cnv_op.conv2d_conf_private);
  lib_ref::RescalePrivateData * rescale_private = (lib_ref::RescalePrivateData*)(rs_op.rescale_conf_private);
  lib_ref::ClipPrivateData * clip_private = (lib_ref::ClipPrivateData*)(clp_op.clip_conf_private);
 
  int32_t tile_input_strides[KConvIORank]{};
  conv2d_private->input.get_mem_strides(tile_input_strides);
  int32_t tile_output_strides[KConvIORank]{};
  conv2d_private->output.get_mem_strides(tile_output_strides);
  int32_t tile_weights_strides[KConvWRank]{};
  conv2d_private->weights.get_mem_strides(tile_weights_strides);
  
  uint32_t input_tile_size[KConvIOIterRank];
  uint32_t output_tile_size[KConvIOIterRank];
  uint32_t weights_tile_size[KConvWIterRank];
  int32_t input_tile_offsets[KConvIOIterRank];
  int32_t output_tile_offsets[KConvIOIterRank];
  int32_t weights_tile_offsets[KConvWIterRank];
  const int32_t zero_offsets[KConvWIterRank]{};
  uint32_t enc_param_size = 0, inp_bias_offset = 0, scale_offset = 0, shift_offset = 0, out_bias_offset = 0;

  mli_status status = MLI_STATUS_OK;
  for (uint32_t i = 0; i < tiles_num; ++i) {
    mli_conv2d_pimpl->GetIOSizesAndOffsets(input_tile_size, output_tile_size, weights_tile_size,
                                           input_tile_offsets, output_tile_offsets, weights_tile_offsets);

    // copy input from global to local buffer
    strided_copy_with_offsets(KConvIORank, conv2d_private->input.get_buf().get_elem_size(),
                              cnv_op.input.data.mem.pi8,
                              input_tile_offsets, zero_offsets, tile_input_strides,
                              input_tile_size, (int8_t*)(g_mem_pool + conv2d_private->input.get_buf().get_offset()));

    // copy weights from global to local buffer
    strided_copy_with_offsets(KConvWRank, conv2d_private->weights.get_buf().get_elem_size(),
                              cnv_op.weights.data.mem.pi8,
                              weights_tile_offsets, zero_offsets, tile_weights_strides,
                              weights_tile_size, (int8_t*)(g_mem_pool + conv2d_private->weights.get_buf().get_offset()));

    // copy weights zps from global to local buffer
    int16_t* wtszp_tile_buf = (int16_t*)(g_mem_pool + conv2d_private->weights_zp.get_buf().get_offset());
    uint32_t wzp_tile_buf_size = weights_tile_size[4] * sizeof(int16_t);
    if (cnv_op.weights.el_params.sa.dim != -1) {
      memcpy(wtszp_tile_buf, cnv_op.weights.el_params.sa.zero_point.mem.pi16 + weights_tile_offsets[4], wzp_tile_buf_size);
    }
    else std::fill_n(wtszp_tile_buf, wzp_tile_buf_size / sizeof(int16_t), cnv_op.weights.el_params.sa.zero_point.mem.i16);

    rescale_pimpl->GetIOSizesAndOffsets(enc_param_size, inp_bias_offset, scale_offset, shift_offset, out_bias_offset);

    // copy rescale input bias from global to local buffer
    memcpy((void*)(g_mem_pool + rescale_private->encoded_params_buffer.get_offset() + inp_bias_offset),
           (void*)(rs_op.mli3_bias.get_bias_tsr().data.mem.pi32 + output_tile_offsets[3]), sizeof(int32_t) * enc_param_size);

    // copy rescale scale from global to local buffer
    memcpy((void*)(g_mem_pool + rescale_private->encoded_params_buffer.get_offset() + scale_offset),
           (void*)(rs_op.mli3_scales_keeper.get_scales_tsr().data.mem.pi16 + output_tile_offsets[3]), sizeof(int16_t) * enc_param_size);

    // copy rescale shift from global to local buffer
    memcpy((void*)(g_mem_pool + rescale_private->encoded_params_buffer.get_offset() + shift_offset),
           (void*)(rs_op.mli3_scales_keeper.get_shift_tsr().data.mem.pi8 + output_tile_offsets[3]), sizeof(int8_t) * enc_param_size);

    // copy rescale output bias from global to local buffer
    memcpy((void*)(g_mem_pool + rescale_private->encoded_params_buffer.get_offset() + out_bias_offset),
           (void*)(rs_op.bias_out.data.mem.pi8 + output_tile_offsets[3]), sizeof(int8_t) * enc_param_size);

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

    // copy results from rescale output tile to the global buffer
    strided_copy_with_offsets(4, clip_private->output_buffer.get_elem_size(),
                              (int8_t*)g_mem_pool + clip_private->output_buffer.get_offset(),
                              zero_offsets, output_tile_offsets, tile_output_strides,
                              output_tile_size, clp_op.original_out.data.mem.pi8);
  }
}

bool postprocess_phase(const reporter_full& reporter,
                       const conv2d_test_operands* cur_test,
                       Conv2dOp& conv2d_op, RescaleOp& rs_op, ClipOp& clp_op) {
  quality_metrics test_metrics;
  bool is_test_passed = true;

  auto& out = clp_op.original_out;
  mli_tensor source_out_tensor = clp_op.original_out;

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

  reporter.report_header("MLI3.0|Kernels|Convolution 2D  Tests");
  uint32_t iteration_order[4]{ 0, 1, 2, 3 };
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
    RescaleOp rs_op(cur_test, conv2d_op.input, conv2d_op.weights);
    ClipOp clp_op(cur_test,rs_op.out);

    bool is_test_passed = preprocess_phase(reporter, cur_test, conv2d_op, rs_op, clp_op);

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
            rescale_axis = conv2d_op.out_acc.rank - 1;
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
    uint32_t input_size[4]{ 1, conv2d_op.input.shape[0], conv2d_op.input.shape[1], conv2d_op.input.shape[2] };
    auto cfg = cur_test->cfg;
    const KernelInfo kernel_info{
      conv2d_op.weights.shape[0], conv2d_op.weights.shape[1], cfg.stride_height, cfg.stride_width,
      cfg.dilation_height, cfg.dilation_width,
      cfg.padding_top, cfg.padding_bottom, cfg.padding_left, cfg.padding_right
    };

#ifdef USE_TILING
    const uint32_t tile_oc = 2;
    const uint32_t tile_hw = 2;
    uint32_t tile_input_size[4]{ input_size[0], tile_hw, tile_hw, input_size[3] };
    Tiling tiling(input_size, tile_input_size, kernel_info, tile_oc, conv2d_op.weights.shape[3]);
#else
    Tiling tiling(input_size, input_size, kernel_info, conv2d_op.weights.shape[3], conv2d_op.weights.shape[3]);
#endif

    prepare_phase(cur_test, tiling, iteration_order, conv2d_op, rs_op, clp_op);

    // STEP 2: Executing phase
    //==================================================================
    // Run conv2d, rescale and clip MLI3.0 kernels
    uint32_t num_tiles = tiling.get_num_tiles();
    execution_phase(conv2d_op, rs_op, clp_op, num_tiles);

    // STEP 3: Postprocessing phase
    //==================================================================
    is_test_passed &= postprocess_phase(reporter, cur_test, conv2d_op, rs_op, clp_op);

    final_status &= is_test_passed;

    // Free buffers for Rescale params
    free(outbias_data);
    free(shift_data);
    free(scale_data);
  }
  reporter.report_outline("[AUTO] Group: mli_krn_conv2d_30", final_status);

  return (final_status) ? 0 : 1;
}
