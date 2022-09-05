/*
 * Copyright 2022, Synopsys, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-3-Clause license found in
 * the LICENSE file in the root directory of this source tree.
 *
 */
#include <cstdlib>

#include "mli_kernels_factory_ref.hpp"
#include "mli_ref_private_types.hpp"
#include "mli_compiler_api.hpp"
#include "mli_runtime_api.hpp"
#include "mli_types.h"
#include "mli_types.hpp"
#include "mli_ref_runtime_api.hpp"
#include "mli_iterator.hpp"

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "test_report.h"
#include "test_tensor_quantizer.h"
#include "test_tiling.hpp"

#include "vectors_mli_krn_maxpool.inc"

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

using mli::tst::crc32_calc;
using mli::tst::quality_metrics;
using mli::tst::reporter_full;
using mli::tst::tensor_quantizer;

namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;

using lib_mli::kMaxpoolRank;
using lib_mli::kMaxpoolIterRank;
using lib_mli::kTensorBatchDim;
using lib_mli::kTensorChannelDim;

struct maxpool_test_operands {
  const char* descr;
  tensor_quantizer in;
  tensor_quantizer out;
  uint32_t data_size;
  const lib_mli::PoolOpConfig cfg;
  const quality_metrics threshold;
  const crc32_calc check_sum;
};

const crc32_calc test_1_chksum_fx16, test_1_chksum_sa8, test_2_chksum_fx16,
    test_2_chksum_sa8, test_3_chksum_fx16, test_3_chksum_sa8,
    test_4_chksum_fx16, test_4_chksum_sa8, test_5_chksum_fx16,
    test_5_chksum_sa8, test_6_chksum_fx16, test_6_chksum_sa8,
    test_7_chksum_fx16, test_7_chksum_sa8;

const quality_metrics thresholds_fx16_general{
    quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
    /* SNR_DB = */ 84.f, /*Quant Error Perc = */ 99.9f};
const quality_metrics thresholds_sa8_general{
    quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
    /* SNR_DB = */ 40.f, /*Quant Error Perc = */ 99.9f};

 static const maxpool_test_operands tests_list[] = {
    // Basic functionality test kernel_size=(4, 3), strides=(1, 1), w/o padding
    {"Test 1 SA8", input_1_sa8, test_1_out_sa8, sizeof(int8_t), test_1_cfg,
     thresholds_sa8_general, test_1_chksum_sa8},
    {"Test 1 FX16", input_1_fx16, test_1_out_fx16, sizeof(int16_t), test_1_cfg,
     thresholds_fx16_general, test_1_chksum_fx16},
    {"Test 2 FX16", input_1_fx16, test_2_out_fx16, sizeof(int16_t), test_2_cfg,
     thresholds_fx16_general, test_2_chksum_fx16},
    {"Test 2 SA8", input_1_sa8, test_2_out_sa8, sizeof(int8_t), test_2_cfg,
     thresholds_sa8_general, test_2_chksum_sa8},
    {"Test 3 FX16 Memstr", input_1_memstr_fx16, test_3_out_fx16,
     sizeof(int16_t), test_3_cfg, thresholds_fx16_general, test_3_chksum_fx16},
    {"Test 3 SA8 Memstr", input_1_memstr_sa8, test_3_out_sa8, sizeof(int8_t),
     test_3_cfg, thresholds_sa8_general, test_3_chksum_sa8},
    {"Test 4 FX16 GlobalPool", input_2_fx16, test_4_out_fx16, sizeof(int16_t),
     test_4_cfg, thresholds_fx16_general, test_4_chksum_fx16},
    {"Test 5 FX16 Pad areas only", input_2_memstr_fx16, test_5_out_fx16,
     sizeof(int16_t), test_5_cfg, thresholds_fx16_general, test_5_chksum_fx16},
    {"Test 5 SA8 pad areas only", input_2_memstr_sa8, test_5_out_sa8,
     sizeof(int8_t), test_5_cfg, thresholds_sa8_general, test_5_chksum_sa8},
    {"Test 6 FX16 k2x2 spec", input_1_memstr_fx16, test_6_out_fx16,
     sizeof(int16_t), test_6_cfg, thresholds_fx16_general, test_6_chksum_fx16},
    {"Test 6 SA8 k2x2 spec", input_1_memstr_sa8, test_6_out_sa8, sizeof(int8_t),
     test_6_cfg, thresholds_sa8_general, test_6_chksum_sa8},
    {"Test 7 FX16 k3x3 spec", input_1_memstr_fx16, test_7_out_fx16,
     sizeof(int16_t), test_7_cfg, thresholds_fx16_general, test_7_chksum_fx16},
    {"Test 7 SA8 k3x3 spec", input_1_memstr_sa8, test_7_out_sa8, sizeof(int8_t),
     test_7_cfg, thresholds_sa8_general, test_7_chksum_sa8},
};

constexpr uint32_t kMemSize = 8192;
static int8_t g_scratch_mem_in[kMemSize] = {0};
static int8_t g_scratch_mem_ref[kMemSize] = {0};
static int8_t g_scratch_mem_out[kMemSize] = {0};
static IO_DATA_ATTR int8_t g_mem_pool[kMemSize] = {0};

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

void prepare_phase(const maxpool_test_operands* cur_test,
                   uint32_t& num_tiles,
                   void*& maxpool2d_instance,
                   uint32_t& maxpool2d_instance_size,
                   lib_mli::PrivateData*& maxpool2d_conf_private,
                   uint32_t& maxpool2d_conf_private_size) {
  mli_data_container temp_in_container{0};
  mli_data_container temp_out_container{0};
  temp_in_container.capacity =
      cur_test->in.get_not_quantized_tensor(temp_in_container).data.capacity;
  temp_in_container.mem.pi8 = g_scratch_mem_in;
  temp_out_container.capacity =
      cur_test->out.get_not_quantized_tensor(temp_out_container).data.capacity;
  temp_out_container.mem.pi8 = g_scratch_mem_ref;

  mli_tensor temp_input_tensor =
      cur_test->in.get_quantized_tensor(temp_in_container);
  mli_tensor temp_output_tensor =
      cur_test->out.get_not_quantized_tensor(temp_out_container);


  int32_t iteration_order[kMaxpoolIterRank]{ 0, 1, 2, 3 };
  uint32_t total_input_size[kMaxpoolRank]{ NUM_VIRTUAL_BATCHES, temp_input_tensor.shape[0], temp_input_tensor.shape[1],
                                          temp_input_tensor.shape[2]};
  uint32_t total_output_size[kMaxpoolRank]{ NUM_VIRTUAL_BATCHES, temp_output_tensor.shape[0], temp_output_tensor.shape[1],
                                           temp_output_tensor.shape[2] };
  assert(total_input_size[kTensorChannelDim] == total_output_size[kTensorChannelDim]);

  const uint32_t batch_tile_size = 2;
#ifdef USE_TILING
  uint32_t tile_output_size[kMaxpoolRank]{ batch_tile_size, 1, 1, 2 };
#else
  uint32_t tile_output_size[kMaxpoolRank]{ batch_tile_size, total_output_size[1], total_output_size[2], total_output_size[3] };
#endif

  int32_t input_stride[kMaxpoolRank];
  int32_t output_stride[kMaxpoolRank];
  for (unsigned i = 1; i < kMaxpoolRank; i++) {
    input_stride[i] = temp_input_tensor.mem_stride[i - 1];
    output_stride[i] = temp_output_tensor.mem_stride[i - 1];
  }
  input_stride[0] = total_input_size[1] * input_stride[1];
  output_stride[0] = total_output_size[1] * output_stride[1];

  const lib_mli::Tensor<lib_mli::NoBuffer, kMaxpoolRank> full_out_tensor(total_output_size, output_stride);
  lib_mli::TensorIterator<lib_mli::NoBuffer, kMaxpoolRank, kMaxpoolIterRank> out_tensor_it(full_out_tensor, tile_output_size, iteration_order);

  uint32_t effective_kernel_size[kMaxpoolIterRank]{ 1, cur_test->cfg.kernel_size[0], cur_test->cfg.kernel_size[1] , 1 };
  uint32_t stride[kMaxpoolIterRank]{ 1, cur_test->cfg.stride[0], cur_test->cfg.stride[1] , 1 };
  uint32_t pre_padding[kMaxpoolIterRank]{ 0, cur_test->cfg.padding_begin[0], cur_test->cfg.padding_begin[1], 0 };
  const lib_mli::Tensor<lib_mli::NoBuffer, kMaxpoolRank> full_in_tensor(total_input_size, input_stride);
  lib_mli::TensorIterator<lib_mli::NoBuffer, kMaxpoolRank, kMaxpoolIterRank> in_tensor_it(full_in_tensor, out_tensor_it,
    effective_kernel_size, stride, pre_padding);

  num_tiles = out_tensor_it.GetTotalCount();
#ifndef USE_TILING
  assert(num_tiles == CEIL_DIV(NUM_VIRTUAL_BATCHES, batch_tile_size));
#endif

  uint32_t input_tile_shape[kMaxpoolIterRank];
  uint32_t output_tile_shape[kMaxpoolIterRank];
  const auto& input_it_config = in_tensor_it.get_config();
  const auto& output_it_config = out_tensor_it.get_config();
  for (unsigned i = 0; i < kMaxpoolIterRank; i++) {
      input_tile_shape[i] = (uint32_t)MAX(input_it_config.get_first_size(i), input_it_config.get_size(i));
      output_tile_shape[i] = (uint32_t)MAX(output_it_config.get_first_size(i), output_it_config.get_size(i));
  }

  // STEP 1: Construct MaxPool2D as a specific ExecutionInterface successor
  //==================================================================
  lib_mli::PlatformDescription pd;
  lib_ref::KernelsFactory kernel_factory(pd);
  uint32_t maxpool2d_cs_size = kernel_factory.MaxPool2D_CS_GetSize();
  void* maxpool2d_cs_buffer = malloc(maxpool2d_cs_size);

  auto maxpool2d_op = kernel_factory.MaxPool2D_CS(maxpool2d_cs_buffer, in_tensor_it, cur_test->cfg, out_tensor_it);

  // STEP 2: Memory management (Up to user on how to deal with it)
  //==================================================================

  uint32_t offset = 0;
  uint32_t elem_size = cur_test->data_size;

  // Leave space for runtime object
  uint32_t runtime_obj_size = maxpool2d_op->GetRuntimeObjectSize();
  offset += runtime_obj_size;

  // Leave space for private data buffer
  uint32_t private_buffer_size = maxpool2d_op->GetKernelPrivateDataSize();
  offset += private_buffer_size;

  // MaxPool2D Input
  uint32_t in_size = GetBufferSize(kMaxpoolRank, input_tile_shape, input_stride) * elem_size;
  lib_mli::OffsetBuffer maxpool2d_in_buf{offset, 0, in_size, elem_size};
  offset += in_size;

  // MaxPool2D Output
  uint32_t out_size = GetBufferSize(kMaxpoolRank, output_tile_shape, output_stride)  * elem_size;
  lib_mli::OffsetBuffer maxpool2d_out_buf{offset, 0, out_size, elem_size};
  offset += out_size;
  assert(offset < kMemSize);

  assert(maxpool2d_op->GetCtrlBufferSize() == 0);
  lib_mli::OffsetBuffer maxpool2d_ctrl_buf{offset, 0, 0, sizeof(char)};

  // Attaching buffer (descriptors) to the operation
  mli_status status = MLI_STATUS_OK;
  status = maxpool2d_op->AttachBufferOffsets(maxpool2d_in_buf, maxpool2d_out_buf, maxpool2d_ctrl_buf);
  assert(status == MLI_STATUS_OK);

  maxpool2d_instance = (int8_t*)g_mem_pool;
  maxpool2d_instance_size = runtime_obj_size;

  status =
      maxpool2d_op->GetKernelPrivateData((int8_t*)g_mem_pool + maxpool2d_instance_size);
  assert(status == MLI_STATUS_OK);
  maxpool2d_conf_private = reinterpret_cast<lib_mli::PrivateData*>(
      (int8_t*)g_mem_pool + maxpool2d_instance_size);
  maxpool2d_conf_private_size = private_buffer_size;
}


void execution_phase(uint32_t num_tiles,
                     void* maxpool2d_instance, uint32_t maxpool2d_instance_size,
                     lib_mli::PrivateData* maxpool2d_conf_private,
                     uint32_t maxpool2d_conf_private_size) {
  // STEP 3: Execution phase
  //==================================================================
  uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool)};

  auto mli_maxpool2d = lib_mli::ExecutionInterface::Create(
      maxpool2d_instance, maxpool2d_instance_size, maxpool2d_conf_private,
      maxpool2d_conf_private_size, membasis,
      sizeof(membasis) / sizeof(membasis[0]));

  assert(mli_maxpool2d != nullptr);

  mli_status status = MLI_STATUS_OK;
  lib_ref::MaxPool2DPrivateData* maxpool2d_private = (lib_ref::MaxPool2DPrivateData*)(maxpool2d_conf_private);
  int32_t tile_input_strides[kMaxpoolRank];
  int32_t tile_output_strides[kMaxpoolRank];
  maxpool2d_private->input.get_mem_strides(tile_input_strides);
  maxpool2d_private->output.get_mem_strides(tile_output_strides);

  lib_ref::MaxPool2D* mli_maxpool2d_pimpl = dynamic_cast<lib_ref::MaxPool2D*>(mli_maxpool2d);
  uint32_t input_tile_size[kMaxpoolRank];
  uint32_t output_tile_size[kMaxpoolRank];
  int32_t input_tile_offsets[kMaxpoolRank];
  int32_t output_tile_offsets[kMaxpoolRank];
  const int32_t zero_offsets[kMaxpoolRank]{};
  for (uint32_t n_tile = 0; n_tile < num_tiles; n_tile++) {
    status = mli_maxpool2d->Prefetch();
    assert(status == MLI_STATUS_OK);

    mli_maxpool2d_pimpl->GetIOSizesAndOffsets(input_tile_size, output_tile_size, input_tile_offsets, output_tile_offsets);
    input_tile_offsets[kTensorBatchDim] = NUM_REAL_BATCHES - 1;
    output_tile_offsets[kTensorBatchDim] = NUM_REAL_BATCHES - 1;

    // copy input from global buffer to local tile buffer
    strided_copy_with_offsets(kMaxpoolRank, maxpool2d_private->input.get_buf().get_elem_size(),
                              g_scratch_mem_in, input_tile_offsets, zero_offsets, tile_input_strides,
                              input_tile_size, (int8_t*) (g_mem_pool + maxpool2d_private->input.get_buf().get_offset()) );


    status = mli_maxpool2d->Issue();
    assert(status == MLI_STATUS_OK);

    status = mli_maxpool2d->Update();
    assert(status == MLI_STATUS_OK);

    // copy output from local tile buffer to global buffer
    strided_copy_with_offsets(kMaxpoolRank, maxpool2d_private->input.get_buf().get_elem_size(),
                              (int8_t*)(g_mem_pool + maxpool2d_private->output.get_buf().get_offset()),
                              zero_offsets, output_tile_offsets, tile_output_strides,
                              output_tile_size, (int8_t*) g_scratch_mem_out);
  }
}

bool postprocess_phase(const reporter_full* reporter,
                       const maxpool_test_operands* cur_test) {
  quality_metrics test_metics;
  bool is_test_passed = false;

  mli_data_container temp_in_container{0};
  mli_data_container temp_out_container{0};
  temp_in_container.capacity =
      cur_test->in.get_not_quantized_tensor(temp_in_container).data.capacity;
  temp_in_container.mem.pi8 = g_scratch_mem_in;
  temp_out_container.capacity =
      cur_test->out.get_not_quantized_tensor(temp_out_container).data.capacity;
  temp_out_container.mem.pi8 = g_scratch_mem_ref;

  mli_tensor temp_input_tensor =
      cur_test->in.get_quantized_tensor(temp_in_container);
  mli_tensor temp_output_tensor =
      cur_test->out.get_not_quantized_tensor(temp_out_container);
  temp_output_tensor.data.mem.pi8 = g_scratch_mem_out;

  test_metics.calculate_metrics(temp_output_tensor, cur_test->out);

  crc32_calc data_crc;
  data_crc(temp_input_tensor);
  data_crc(temp_output_tensor);
  is_test_passed = reporter->evaluate_and_report_case(cur_test->descr, test_metics,
                                     cur_test->threshold, data_crc,
                                     cur_test->check_sum);
  return is_test_passed;
}

int main() {
  const reporter_full reporter;
  bool final_status = true;

  reporter.report_header("MLI3.0|Kernels|Max Pooling Function Tests");
  for (int i = 0; i < kTestsNum; ++i) {
    bool is_test_passed = true;
    const maxpool_test_operands* cur_test = &tests_list[i];

    if (!(cur_test->in.is_valid() && cur_test->out.is_valid())) {
      reporter.report_message(
          cur_test->descr,
          "FAILED at init: Bad source data for one of tensors");
      is_test_passed = false;
    }

#if defined(__Xvec_guard_bit_option)
    // VPX code needs to be debugged
    reporter.report_message(cur_test->descr, "SKIPPED due to a known issue");
    continue;
#endif

    /**************************************************************************************************************/
    void* maxpool2d_instance = nullptr;
    uint32_t maxpool2d_instance_size = 0;

    lib_mli::PrivateData* maxpool2d_conf_private = nullptr;
    uint32_t maxpool2d_conf_private_size = 0;
    /**************************************************************************************************************/
    uint32_t num_tiles = 0; // num_tiles calculated inside prepare_phase
    prepare_phase(cur_test, num_tiles,
                  maxpool2d_instance,
                  maxpool2d_instance_size, maxpool2d_conf_private,
                  maxpool2d_conf_private_size);

    execution_phase(num_tiles, maxpool2d_instance, maxpool2d_instance_size,
                    maxpool2d_conf_private, maxpool2d_conf_private_size);

    is_test_passed &= postprocess_phase(&reporter, cur_test);

    final_status &= is_test_passed;
  }
  reporter.report_outline("[AUTO] Group: mli_krn_maxpool_30", final_status);

  return 0;
}