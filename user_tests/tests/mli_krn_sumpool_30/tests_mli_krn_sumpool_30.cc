/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#include <cstdlib>


#include "mli_api.h"
#include "mli_config.h"
#include "mli_types.h"
#include "mli_types.hpp"
#include "mli_kernels_factory_ref.hpp"
#include "mli_ref_private_types.hpp"
#include "mli_runtime_api.hpp"
#include "mli_private_types.h"
#include "mli_compiler_api.hpp"
#include "mli_ref_runtime_api.hpp"
#include "mli_iterator.hpp"

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "test_rescale_utility.h"
#include "test_tensor_quantizer.h"
#include "test_report.h"
#include "test_tiling.hpp"

// reuse test vectors of avepool2d
#include "vectors_mli_krn_avepool.inc"


/**
 * Comment USE_TILING if you want to use single tile (tile size = input size).
 */
#define USE_TILING

/**
  * Initally this user test i/o data was prepared for single batch.
  * To test batching/tiling in batch dimension same single real batch of data run NUM_VIRTUAL_BATCHES times.
  * You can change NUM_VIRTUAL_BATCHES, don't change NUM_REAL_BATCHES or you will get "out of the array boarders" errors.
  */
#define NUM_VIRTUAL_BATCHES 5
#define NUM_REAL_BATCHES 1

using namespace snps_arc::metaware::mli::service;

using mli::tst::tensor_quantizer;
using mli::tst::quality_metrics;
using mli::tst::crc32_calc;
using mli::tst::reporter_full;
using mli::tst::memory_manager;
using mli::tst::scales_calc;
using mli::tst::bias_folder;

namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;

using lib_mli::kPoolRank;
using lib_mli::kPoolIterRank;
using lib_mli::kTensorBatchDim;
using lib_mli::kTensorChannelDim;

typedef mli_status(*rescale_func_ptr)(
  const mli_tensor* /*input*/,
  const mli_tensor* /*bias_in*/,
  const mli_tensor* /*scale*/,
  const mli_tensor* /*shift*/,
  const mli_tensor* /*bias_out*/,
  mli_tensor* /*output*/);

typedef mli_status(*avepool_func_ptr)(
    const mli_tensor* /*in*/,
    const mli_pool_cfg* /*cfg*/,
    mli_tensor* /*out*/);

struct sumpool2d_test_operands {
    const char* descr;
    const avepool_func_ptr mli_krn_avepool;
    tensor_quantizer in;
    tensor_quantizer out;
    tensor_quantizer out_acc;
    tensor_quantizer bias_in;
    tensor_quantizer bias_out;
    const float in_scale;
    const float out_scale;
    const mli_pool_cfg cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

// Checksums of test tensors for various mli calculations mode.
// When developer finished implementation of kernel and consider it as ok, He need to populate
// proper checksums for tests in order to highlight any change which affects results.
const crc32_calc test_1_chksum_sa8{ 0x760E03C5 },
                 test_2_chksum_sa8{ 0x4CAD1DF4 },
                 test_3_chksum_sa8{ 0x5B174EB5 },
                 test_4_chksum_sa8{ 0x7A3068C0 },
                 test_5_chksum_sa8{ 0x64504A2A },
                 test_6_chksum_sa8{ 0x0E760FC6 },
                 test_7_chksum_sa8{ 0x67A9C0DA },
                 test_8_chksum_sa8{ 0x9FDA9994 };

const quality_metrics thresholds_fx16_general { /* MaxAbsErr = */0.0003f, quality_metrics::kPassValueSnr,
                                                /* SNR_DB = */80.f, /*Quant Error Perc = */ 27.f };

const quality_metrics thresholds_sa8_general{ /* MaxAbsErr = */0.06f, quality_metrics::kPassValueSnr,
                                              /* SNR_DB = */30.f, /*Quant Error Perc = */ 13.f };

const quality_metrics thresholds_fx16_test_huge_vals {
    quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
    /* SNR_DB = */90.f, /*Quant Error Perc = */ 91.f };

const quality_metrics thresholds_sa8_test_huge_vals {
    quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
    /* SNR_DB = */47.1f, /*Quant Error Perc = */ 47.15f };


static const sumpool2d_test_operands tests_list[] = {
    // Basic functionality test kernel_size=(4, 3), strides=(1, 1), w/o padding
    {"Test 1 SA8",   mli_krn_avepool_hwc_sa8,
                     input_1_sa8, test_1_out_sa8,
                     test_1_out_acc_sa32, test_1_bias_in_sa32, test_1_bias_out_sa8,
                     input_1_scale, test_1_out_scale,
                     test_1_cfg, thresholds_sa8_general, test_1_chksum_sa8},

    // Basic functionality test kernel_size=(3, 4), strides=(2, 2), with krn_padding
    {"Test 2 SA8",   mli_krn_avepool_hwc_sa8,
                     input_1_sa8, test_2_out_sa8,
                     test_2_out_acc_sa32, test_2_bias_in_sa32, test_2_bias_out_sa8,
                     input_1_scale, test_2_out_scale,
                     test_2_cfg, thresholds_sa8_general, test_2_chksum_sa8},

    // Memstride test kernel_size=(3, 4), strides=(3, 3), with krn_padding
    {"Test 3 SA8 Memstr",   mli_krn_avepool_hwc_sa8,
                            input_1_memstr_sa8, test_3_out_sa8,
                            test_3_out_acc_sa32, test_3_bias_in_sa32, test_3_bias_out_sa8,
                            input_1_scale, test_3_out_scale,
                            test_3_cfg, thresholds_sa8_general, test_3_chksum_sa8},

    // Global Pooling test with memstride
    {"Test 4 SA8 GlobalPool",   mli_krn_avepool_hwc_sa8,
                                input_2_memstr_sa8, test_4_out_sa8,
                                test_4_out_acc_sa32, test_4_bias_in_sa32, test_4_bias_out_sa8,
                                input_2_scale, test_4_out_scale,
                                test_4_cfg, thresholds_sa8_general, test_4_chksum_sa8},

    // Padding only areas test with memstride, kernel_size=(4, 4), strides=(2, 2), with krn_padding
    {"Test 5 SA8 pad areas only",   mli_krn_avepool_hwc_sa8,
                                    input_2_memstr_sa8, test_5_out_sa8,
                                    test_5_out_acc_sa32, test_5_bias_in_sa32, test_5_bias_out_sa8,
                                    input_2_scale, test_5_out_scale,
                                    test_5_cfg, thresholds_sa8_general, test_5_chksum_sa8},

    // k2x2 specialization test with memstride, kernel_size=(2, 2), strides=(2, 2), krn_padding
    {"Test 6 SA8 k2x2 spec",   mli_krn_avepool_hwc_sa8_k2x2,
                               input_1_memstr_sa8, test_6_out_sa8,
                               test_6_out_acc_sa32, test_6_bias_in_sa32, test_6_bias_out_sa8,
                               input_1_scale, test_6_out_scale,
                               test_6_cfg, thresholds_sa8_general, test_6_chksum_sa8},

    // k3x3 specialization test with memstride, kernel_size=(3, 3), strides=(2, 2), krn_padding
    {"Test 7 SA8 k3x3 spec",   mli_krn_avepool_hwc_sa8_k3x3,
                               input_1_memstr_sa8, test_7_out_sa8,
                               test_7_out_acc_sa32, test_7_bias_in_sa32, test_7_bias_out_sa8,
                               input_1_scale, test_7_out_scale,
                               test_7_cfg, thresholds_sa8_general, test_7_chksum_sa8},

    // Test with huge values in operands to check negative fractional and big scales
    {"Test 8 SA8 Huge Vals",   mli_krn_avepool_hwc_sa8,
                               input_3_sa8, test_8_out_sa8,
                               test_8_out_acc_sa32, test_8_bias_in_sa32, test_8_bias_out_sa8,
                               input_3_scale, test_8_out_scale,
                               test_8_cfg, thresholds_sa8_test_huge_vals, test_8_chksum_sa8},
};

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);


// Global Memory Memagement
//==================================================================
constexpr uint32_t kMemSize = 8192;
constexpr int kMemAccSize = kMemSize*sizeof(int32_t); // TODO: for double wide accu, more space might be required
static int8_t g_scratch_mem_in[kMemSize] = { 0 };
static int8_t g_scratch_mem_acc_out[kMemAccSize] = { 0 };
static int8_t g_scratch_mem_out[kMemSize] = { 0 };
static int8_t g_scratch_mem_bias_out[kMemSize] = { 0 };
static int8_t g_scratch_mem_b[kMemSize] = { 0 };
static IO_DATA_ATTR int8_t g_mem_pool[kMemSize] = {0};


struct SumPool2dOp {
  // SumPool2d Kernel
  SumPool2dOp(const sumpool2d_test_operands* cur_test) {
    mem_in_keeper = memory_manager((int8_t*)(g_scratch_mem_in), sizeof(g_scratch_mem_in));
    mem_out_acc_keeper = memory_manager((int8_t*)(g_scratch_mem_acc_out), sizeof(g_scratch_mem_acc_out));

    input = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
    out_acc = cur_test->out_acc.get_not_quantized_tensor(mem_out_acc_keeper.allocate_memory(cur_test->out_acc));

    // layout: HWC
    height = input.shape[0];
    width = input.shape[1];
    channel = input.shape[2];
  }

  // memory memagement for ins & outs tensors
  memory_manager mem_in_keeper;
  memory_manager mem_out_acc_keeper;

  // ins & outs tensors
  mli_tensor input;
  mli_tensor out_acc;

  int32_t height;
  int32_t width;
  int32_t channel;

  // the offset of output buffer
  uint32_t out_mem_offset{0};

  // conv runtime instnace
  void* sumpool2d_instance{nullptr};
  uint32_t sumpool2d_instance_size{0};

  // conv private data
  lib_mli::PrivateData* sumpool2d_conf_private{nullptr};
  uint32_t sumpool2d_conf_private_size{0};

  uint32_t input_offsets{0};
	uint32_t output_offsets{0};

  int32_t input_stride[kPoolRank]{0};
  int32_t output_stride[kPoolRank]{0};
};

struct RescaleOp {
  // Rescale Kernel
  RescaleOp(const sumpool2d_test_operands* cur_test, const mli_tensor& input) {
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

    // s_o * (o - o_zp) = s_x * sum_k(x - x_zp) / pool_size
    // o = s_x / (s_o * pool_size) * sum_k(x - x_zp) + o_zp
    //   = s_x / (s_o * pool_size) * (sum_k(x) - x_zp * pool_size) + o_zp
    //   = scale * (sum_pool_out_acc - bias_in) + bias_out
    pool_size = cur_test->cfg.kernel_width * cur_test->cfg.kernel_height;

    mli3_bias = bias_folder();
    mli3_scales_keeper = scales_calc(cur_test->in_scale, cur_test->out_scale, 1.0f / pool_size);
  }

  bias_folder GetBiasFolder(const mli_tensor& input, int32_t pool_size) const {
    return bias_folder();
  }

  scales_calc GetScaleCalc(const sumpool2d_test_operands* cur_test, int32_t pool_size) const {
    return scales_calc(cur_test->in_scale, cur_test->out_scale, 1.0f / pool_size);
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

  // pool size unsed in averaging
  uint32_t pool_size{0};
};

bool preprocess_phase(const reporter_full& reporter,
                      const sumpool2d_test_operands* cur_test,
                      const SumPool2dOp& sumpool2d_op, const RescaleOp& rs_op) {
    bool is_test_passed = true;

    if (!(cur_test->in.is_valid() && cur_test->bias_in.is_valid() && cur_test->bias_out.is_valid() && cur_test->out.is_valid())) {
      reporter.report_message(cur_test->descr, "FAILED at init: Bad source data for one of tensors");
      is_test_passed = false;
    }

    if (is_test_passed &&
        (tensor_quantizer::validate_tensor(sumpool2d_op.input) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(sumpool2d_op.out_acc) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(rs_op.bias_out) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(rs_op.mli3_scales_keeper.get_scales_tsr()) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(rs_op.mli3_scales_keeper.get_shift_tsr()) != tensor_quantizer::kOk ||
         tensor_quantizer::validate_tensor(rs_op.out) != tensor_quantizer::kOk)) {
      reporter.report_message(cur_test->descr,
                  "FAILED at quantization step: more memory for one of tensors might be required");
      is_test_passed = false;
    }

    if (is_test_passed &&
        (sumpool2d_op.mem_in_keeper.is_memory_corrupted() || rs_op.mem_out_keeper.is_memory_corrupted() ||
         sumpool2d_op.mem_out_acc_keeper.is_memory_corrupted() || rs_op.mem_bias_out_keeper.is_memory_corrupted() ||
         rs_op.mem_b_keeper.is_memory_corrupted())) {
      reporter.report_message(cur_test->descr,
        "FAILED at quantization step: memory beside one of operands is corrupted");
      is_test_passed = false;
    }

    return is_test_passed;
}

void prepare_phase(const sumpool2d_test_operands* cur_test,
                   uint32_t& num_tiles, SumPool2dOp& op) {
 

  assert(op.input.rank == kTensorChannelDim);
  assert(op.out_acc.rank == kTensorChannelDim);

  int32_t iteration_order[kPoolIterRank]{ 0, 1, 2, 3 };
  uint32_t total_input_size[kPoolRank]{ NUM_VIRTUAL_BATCHES, op.input.shape[KRNL_H_DIM_HWCN], op.input.shape[KRNL_W_DIM_HWCN],
                                          op.input.shape[KRNL_D_DIM_HWCN]};
  uint32_t total_output_size[kPoolRank]{ NUM_VIRTUAL_BATCHES, op.out_acc.shape[KRNL_H_DIM_HWCN], op.out_acc.shape[KRNL_W_DIM_HWCN],
                                           op.out_acc.shape[KRNL_D_DIM_HWCN] };
  
  // N == NUM_VIRTUAL_BATCHES
  assert(total_input_size[kTensorBatchDim] == NUM_VIRTUAL_BATCHES && total_output_size[kTensorBatchDim] == NUM_VIRTUAL_BATCHES);
  // Cin == Co
  assert(total_input_size[kTensorChannelDim] == total_output_size[kTensorChannelDim]);


  for (unsigned i = KRNL_W_DIM_HWCN; i < kPoolRank; i++) {
    op.input_stride[i] = op.input.mem_stride[i - KRNL_W_DIM_HWCN];
    op.output_stride[i] = op.out_acc.mem_stride[i - KRNL_W_DIM_HWCN];
  }
  op.input_stride[KRNL_H_DIM_HWCN] = total_input_size[KRNL_W_DIM_HWCN] * op.input_stride[KRNL_W_DIM_HWCN];
  op.output_stride[KRNL_H_DIM_HWCN] = total_output_size[KRNL_W_DIM_HWCN] * op.output_stride[KRNL_W_DIM_HWCN];

  const uint32_t batch_tile_size = 1;
#ifdef USE_TILING
  uint32_t tile_output_size[kPoolRank]{ batch_tile_size, 2, 3, 3 };
#else
  uint32_t tile_output_size[kPoolRank]{ batch_tile_size, total_output_size[1], total_output_size[2], total_output_size[3] };
#endif
  
  const lib_mli::Tensor<lib_mli::NoBuffer, kPoolRank> out_tensor(total_output_size, op.output_stride);
  lib_mli::TensorIterator<lib_mli::NoBuffer, kPoolRank, kPoolIterRank> out_tensor_it(out_tensor, tile_output_size, iteration_order);
  
  uint32_t effective_kernel_size[kPoolIterRank]{ 1, cur_test->cfg.kernel_height, cur_test->cfg.kernel_width, 1 };
  uint32_t stride[kPoolIterRank]{ 1, cur_test->cfg.stride_height, cur_test->cfg.stride_width, 1 };
  uint32_t pre_padding[kPoolIterRank]{ 0, cur_test->cfg.padding_top, cur_test->cfg.padding_left, 0 };
  const lib_mli::Tensor<lib_mli::NoBuffer, kPoolRank> full_in_tensor(total_input_size, op.input_stride);
  lib_mli::TensorIterator<lib_mli::NoBuffer, kPoolRank, kPoolIterRank> in_tensor_it(full_in_tensor, out_tensor_it,
                                                                                    effective_kernel_size, stride, pre_padding);

  //calculate number of tiles
  num_tiles = out_tensor_it.GetTotalCount();
  #ifndef USE_TILING
    assert(num_tiles == CEIL_DIV(NUM_VIRTUAL_BATCHES, batch_tile_size));
  #endif

  uint32_t input_tile_shape[kPoolIterRank];
  uint32_t output_tile_shape[kPoolIterRank];
  const auto& input_it_config = in_tensor_it.get_config();
  const auto& output_it_config = out_tensor_it.get_config();
  for (unsigned i = 0; i < kPoolIterRank; i++) {
      input_tile_shape[i] = (uint32_t)MAX(input_it_config.get_first_size(i), input_it_config.get_size(i));
      output_tile_shape[i] = (uint32_t)MAX(output_it_config.get_first_size(i), output_it_config.get_size(i));
  }

  // STEP 1: Construct SumPool2d as a specific ExecutionInterface successor
  //==================================================================
  lib_mli::PlatformDescription pd;
  lib_ref::KernelsFactory kernel_factory(pd);
  uint32_t sumpool2d_cs_size = kernel_factory.SumPool2D_CS_GetSize();
  void* sumpool2d_cs_buffer = malloc(sumpool2d_cs_size);

  lib_mli::PoolOpConfig cfg(
    cur_test->cfg.kernel_height, cur_test->cfg.kernel_width,
    cur_test->cfg.stride_height, cur_test->cfg.stride_width,
    cur_test->cfg.padding_top, cur_test->cfg.padding_left,
    cur_test->cfg.padding_bottom, cur_test->cfg.padding_right
  );

  auto sumpool2d_op = kernel_factory.SumPool2D_CS(sumpool2d_cs_buffer, in_tensor_it, cfg, out_tensor_it);

  // STEP 1.2: Memory management (Up to user on how to deal with it)
  //==================================================================
  uint32_t offsets[1] = {0};

  uint32_t i_elem_size = mli_hlp_tensor_element_size(&op.input);
  assert(op.input.el_type == MLI_EL_SA_8);
  uint32_t o_elem_size = mli_hlp_tensor_element_size(&op.out_acc);
  assert(op.out_acc.el_type == MLI_EL_SA_32);

  // Define buffers for in\out tensors
  // Leave space for runtime object
  uint32_t* offset = &offsets[0];
  uint32_t runtime_obj_size = sumpool2d_op->GetRuntimeObjectSize();
  *offset += runtime_obj_size;

  // Leave space for private data buffer
  offset = &offsets[0];
  uint32_t private_buffer_size = sumpool2d_op->GetKernelPrivateDataSize();
  *offset += private_buffer_size;

  // sumpool2d input
  offset = &offsets[0];
  op.input_offsets = *offset;
  uint32_t in_size = GetBufferSize(kPoolRank, input_tile_shape, op.input_stride) * i_elem_size;
  lib_mli::OffsetBuffer sumpool2d_in_buf{*offset, 0, in_size, i_elem_size};
  *offset += in_size;

  // sumpool2d output
  offset = &offsets[0];
  // NOTE: The output should be 4 bytes aligned for int32_t, otherwise, it will cause `vvst` crash.
  *offset = (*offset + 4 - 1) / 4 * 4;
  op.output_offsets = *offset;
  uint32_t out_size = GetBufferSize(kPoolRank, output_tile_shape, op.output_stride) * o_elem_size;
  lib_mli::OffsetBuffer sumpool2d_out_buf{*offset, 0, out_size, o_elem_size};
  *offset += out_size;

  // MLI tensor structures and sumpool2d configuration
  offset = &offsets[0];
  uint32_t ctrl_buffer_size = sumpool2d_op->GetCtrlBufferSize();
  lib_mli::OffsetBuffer sumpool2d_ctrl_buf{*offset, 0, ctrl_buffer_size, sizeof(char)};
  *offset += ctrl_buffer_size;

  assert(ctrl_buffer_size == 0);
  assert(*offset < kMemSize);

  // Attaching buffer (descriptors) to the operation
  mli_status status = MLI_STATUS_OK;

  status = sumpool2d_op->AttachBufferOffsets(sumpool2d_in_buf,
                                             sumpool2d_out_buf,
                                             sumpool2d_ctrl_buf);
  assert(status == MLI_STATUS_OK);

  // STEP 1.4: Compile sumpool2d into the binary data
  //==================================================================
  op.sumpool2d_instance = (int8_t*)g_mem_pool;
  op.sumpool2d_instance_size = sumpool2d_op->GetRuntimeObjectSize();

  status = sumpool2d_op->GetKernelPrivateData((int8_t*)g_mem_pool + op.sumpool2d_instance_size);
  assert(status == MLI_STATUS_OK);
  op.sumpool2d_conf_private = reinterpret_cast<lib_mli::PrivateData*>(
      (int8_t*)g_mem_pool + op.sumpool2d_instance_size);
  op.sumpool2d_conf_private_size = sumpool2d_op->GetKernelPrivateDataSize();
}

void execution_phase(uint32_t num_tiles, SumPool2dOp& op) {
  // STEP 3: Execution phase
  //==================================================================
  uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool)};

  auto mli_op = lib_mli::ExecutionInterface::Create(
    op.sumpool2d_instance, op.sumpool2d_instance_size,
    op.sumpool2d_conf_private, op.sumpool2d_conf_private_size,
    membasis, sizeof(membasis) / sizeof(membasis[0]));

  assert(mli_op != nullptr);

  mli_status status = MLI_STATUS_OK;

  lib_ref::SumPool2D* mli_sumoool2_pimpl = dynamic_cast<lib_ref::SumPool2D*>(mli_op);
  uint32_t input_tile_size[kPoolRank];
  uint32_t output_tile_size[kPoolRank];
  int32_t input_tile_offsets[kPoolRank];
  int32_t output_tile_offsets[kPoolRank];
  const int32_t zero_offsets[kPoolRank]{};

  for (uint32_t n_tile = 0; n_tile < num_tiles; ++n_tile) {
    status = mli_op->Prefetch();
    assert(status == MLI_STATUS_OK);

    mli_sumoool2_pimpl->GetIOSizesAndOffsets(input_tile_size, output_tile_size, input_tile_offsets, output_tile_offsets);
    input_tile_offsets[kTensorBatchDim] = NUM_REAL_BATCHES - 1;
    output_tile_offsets[kTensorBatchDim] = NUM_REAL_BATCHES - 1;

    // copy input from global to local buffer
    strided_copy_with_offsets(kPoolRank, sizeof(int8_t),
                              op.input.data.mem.pi8,
                              input_tile_offsets, zero_offsets, op.input_stride,
                              input_tile_size, (int8_t*)(g_mem_pool + op.input_offsets));

    status = mli_op->Issue();
    assert(status == MLI_STATUS_OK);

    status = mli_op->Update();
    assert(status == MLI_STATUS_OK);

    // copy output from local tile buffer to global buffer
    strided_copy_with_offsets(kPoolRank, sizeof(int32_t),
                              (int8_t*)(g_mem_pool + op.output_offsets),
                              zero_offsets, output_tile_offsets, op.output_stride,
                              output_tile_size, op.out_acc.data.mem.pi8);
  }
}

bool postprocess_phase(const reporter_full& reporter,
                       const sumpool2d_test_operands* cur_test,
                       SumPool2dOp& sumpool2d_op, RescaleOp& rs_op) {
  quality_metrics test_metrics;
  bool is_test_passed = true;

  auto& out = rs_op.out;
  mli_tensor source_out_tensor = rs_op.original_out;

  if (is_test_passed &&
      (sumpool2d_op.mem_in_keeper.is_memory_corrupted() || rs_op.mem_out_keeper.is_memory_corrupted() ||
       sumpool2d_op.mem_out_acc_keeper.is_memory_corrupted() || rs_op.mem_bias_out_keeper.is_memory_corrupted() ||
       rs_op.mem_b_keeper.is_memory_corrupted())) 
       {
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
    // FIXME: SA8 parameters are not filled
    // const mli_data_container empty_container{ 0 };
    // mli_tensor source_out_tensor = cur_test->out.get_not_quantized_tensor(empty_container);
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
    data_crc(sumpool2d_op.input);
    data_crc(out);

    is_test_passed &= reporter.evaluate_and_report_case(cur_test->descr, test_metrics, cur_test->threshold,
                              data_crc, cur_test->check_sum);
  }

  return is_test_passed;
}

std::vector<int32_t> CaclPoolSize(int32_t height, int32_t width, const mli_pool_cfg& cfg) {
  int32_t kernel_h = cfg.kernel_height;
  int32_t kernel_w = cfg.kernel_width;
  int32_t stride_h = cfg.stride_height;
  int32_t stride_w = cfg.stride_width;
  int32_t pad_l = cfg.padding_left;
  int32_t pad_r = cfg.padding_right;
  int32_t pad_t = cfg.padding_top;
  int32_t pad_b = cfg.padding_bottom;

  int32_t col_begin = 0 - pad_t;
  int32_t col_end = height + pad_b;
  int32_t row_begin = 0 - pad_l;
  int32_t row_end = width + pad_r;

  std::vector<int32_t> pool_size;

  for (int32_t h = col_begin; h + kernel_h <= col_end; h += stride_h) {
    for (int32_t w = row_begin; w + kernel_w <= row_end; w += stride_w) {
      int32_t comp_h = 0;
      int32_t comp_w = 0;
      if (h < 0) {
        comp_h = h;
      } else if (h + kernel_h > height) {
        comp_h = height - (h + kernel_h);
      }
      if (w < 0) {
        comp_w = w;
      } else if (w + kernel_w > width) {
        comp_w = width - (w + kernel_w);
      }
      pool_size.push_back((kernel_h + comp_h) * (kernel_w + comp_w));
    }
  }

  return std::move(pool_size);
}

void avg_and_rescale(const sumpool2d_test_operands* cur_test,
                   const SumPool2dOp& sumpool2d_op, RescaleOp& rs_op) {
  const int32_t inp_zp = sumpool2d_op.input.el_params.sa.zero_point.mem.i8;
  int32_t in_bias = 0; // rs_op.GetBiasFolder(pool);
  const int8_t out_bias = rs_op.bias_out.data.mem.i8;
  std::vector<int32_t> pool_size = CaclPoolSize(sumpool2d_op.height, sumpool2d_op.width, cur_test->cfg);

  constexpr int kMaxSupportedRank = 3;
  assert(sumpool2d_op.out_acc.rank == kMaxSupportedRank);
  assert(rs_op.out.rank == kMaxSupportedRank);

  // Average and Rescale
  for (size_t h = 0; h < sumpool2d_op.out_acc.shape[0]; h++) {
    for (size_t w = 0; w < sumpool2d_op.out_acc.shape[1]; w++) {
      for (size_t c = 0; c < sumpool2d_op.out_acc.shape[2]; c++) {
        int32_t idx = h * sumpool2d_op.out_acc.shape[1] + w;
        assert(idx < pool_size.size());
        int32_t pool = pool_size[idx];
        auto scalc_calc = rs_op.GetScaleCalc(cur_test, pool);
        const int8_t shift = scalc_calc.get_shift_tsr().data.mem.pi8[0];
        const int16_t scale = scalc_calc.get_scales_tsr().data.mem.pi16[0];

        int32_t in_val = sumpool2d_op.out_acc.data.mem.pi32[h * sumpool2d_op.out_acc.mem_stride[0] +
          w * sumpool2d_op.out_acc.mem_stride[1] + c * sumpool2d_op.out_acc.mem_stride[2]];

        in_bias = pool * inp_zp;
        int8_t out_val = (static_cast<int64_t>((in_val - in_bias) * static_cast<int32_t>(scale)) >> shift) + out_bias;

        rs_op.out.data.mem.pi8[h * rs_op.out.mem_stride[0] +
          w * rs_op.out.mem_stride[1] + c * rs_op.out.mem_stride[2]] = out_val;
      }
    }
  }
}

int main() {
  const reporter_full reporter;
  bool final_status = true;

  reporter.report_header("MLI3.0|Kernels|SumPool2D  Tests");
  for (int i = 0; i < kTestsNum; ++i) {
    // get the current test case
    const sumpool2d_test_operands* cur_test = &tests_list[i];

    // STEP 0: Preprocessing phase
    //==================================================================
    SumPool2dOp sumpool2d_op(cur_test);
    RescaleOp rs_op(cur_test, sumpool2d_op.input);

    bool is_test_passed = preprocess_phase(reporter, cur_test, sumpool2d_op, rs_op);

    // STEP 1: Preparing phase
    //==================================================================
    uint32_t num_tiles = 0; // num_tiles calculated inside prepare_phase
    
    prepare_phase(cur_test, num_tiles, sumpool2d_op);

    // STEP 2: Executing phase
    //==================================================================
    // Run sumpool2d MLI3.0 kernel
    execution_phase(num_tiles, sumpool2d_op);

    avg_and_rescale(cur_test, sumpool2d_op, rs_op);

    // STEP 3: Postprocessing phase
    //==================================================================
    is_test_passed &= postprocess_phase(reporter, cur_test, sumpool2d_op, rs_op);

    final_status &= is_test_passed;
  }
  reporter.report_outline("[AUTO] Group: mli_krn_sumpool2d_30", final_status);

  return (final_status) ? 0 : 1;
}
