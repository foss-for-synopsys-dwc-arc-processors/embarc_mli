/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#include <cstdlib>

#include "mli_compiler_api.hpp"
#include "mli_iterator.hpp"
#include "mli_kernels_factory_ref.hpp"
#include "mli_runtime_api.hpp"
#include "mli_types.h"
#include "mli_types.hpp"

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "test_report.h"
#include "test_tensor_quantizer.h"

#include "vectors_mli_krn_data_movement.inc"

using mli::tst::crc32_calc;
using mli::tst::quality_metrics;
using mli::tst::reporter_full;
using mli::tst::tensor_quantizer;
using mli::tst::memory_manager;

namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;

struct data_movement_test_operands {
  const char* descr;
  tensor_quantizer in;
  tensor_quantizer out;
  uint32_t data_size;
  const quality_metrics threshold;
  const crc32_calc check_sum;
};

struct DataMoveConfig {
    uint32_t* offset;
    uint32_t* size;
    int32_t* sub_sample_step;
    uint32_t* dst_offset;
    int32_t * dst_mem_stride;
};

const crc32_calc test_1_chksum_fx16{}, test_1_chksum_sa8{},
    test_2_chksum_fx16{}, test_2_chksum_sa8{}, test_3_chksum_fx16{},
    test_3_chksum_sa8{}, test_4_chksum_fx16{}, test_4_chksum_sa8{},
    test_5_chksum_fx16{}, test_5_chksum_sa8{}, test_6_chksum_fx16{},
    test_6_chksum_sa8{}, test_7_chksum_fx16{}, test_7_chksum_sa8{},
    test_8_chksum_fx16{}, test_8_chksum_sa8{}, test_9_chksum_fx16{},
    test_9_chksum_sa8{}, test_10_chksum_fx16{}, test_10_chksum_sa8{},
    test_11_chksum_fx16{}, test_11_chksum_sa8{}, test_12_chksum_fx16{},
    test_12_chksum_sa8{}, test_13_chksum_sa32{}, test_14_chksum_sa32{};

const quality_metrics thresholds_fx16_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                                /* SNR DB = */ 30.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                               /* SNR DB = */ 30.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa32_general { quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                               /* SNR DB = */ 30.f, quality_metrics::kPassValueQuantErrPerc };

static const data_movement_test_operands tests_list[] = {
     {"Test 1 FX16 Copy", input_1_fx16, test_1_out_fx16, sizeof(int16_t),
                                         thresholds_fx16_general, test_1_chksum_fx16},
     {"Test 1 SA8  Copy AXIS = 1", input_1_sa8,
                                         test_1_out_sa8, sizeof(int8_t),
                                         thresholds_sa8_general, test_1_chksum_sa8},
    {"Test 2 FX16 Copy MemStr", input_2_fx16,
                                        test_2_out_fx16, sizeof(int16_t),
                                        thresholds_fx16_general, test_2_chksum_fx16},
    {"Test 2 SA8 Copy MemStr",  input_2_sa8,
                                        test_2_out_sa8, sizeof(int8_t),
                                        thresholds_sa8_general, test_2_chksum_sa8},
     {"Test 3 FX16 Slice MemStr",  input_3_fx16,
                                         test_3_out_fx16, sizeof(int16_t),
                                         thresholds_fx16_general, test_3_chksum_fx16},
     {"Test 3 SA8 Slice MemStr",  input_3_sa8,
                                         test_3_out_sa8, sizeof(int8_t),
                                         thresholds_sa8_general, test_3_chksum_sa8},
     {"Test 4 FX16 Concat",  input_1_fx16,
                                         test_4_out_fx16, sizeof(int16_t),
                                         thresholds_fx16_general, test_4_chksum_fx16},
     {"Test 7 FX16 Subsample",  input_1_fx16,
                                         test_7_out_fx16, sizeof(int16_t),
                                         thresholds_fx16_general, test_7_chksum_fx16},
};

DataMoveConfig current_test_cfg(int n) {
  return {offsets_cfg[n],
          sizes_cfg[n],
          sub_sample[n],
          out_offsets_cfg[n],
          out_mem_stride_cfg[n]
          };
}

constexpr unsigned kMaxRank = 4;
constexpr int kMemSize = 2048;
static int8_t g_scratch_mem_in[kMemSize] = {0};
static int8_t g_scratch_mem_out[kMemSize] = {0};
static int8_t g_mem_pool[kMemSize] = {0};

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

void prepare_phase(int i, memory_manager& mem_in_keeper,
                   memory_manager& mem_out_keeper,
                   const data_movement_test_operands* cur_test,
                   uint32_t& dst_mem_offset, void*& move_instance,
                   uint32_t& move_instance_size,
                   lib_mli::PrivateData*& move_conf_private,
                   uint32_t& move_conf_private_size) {

  mli_tensor temp_input_tensor = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
  mli_tensor temp_output_tensor;

  if (i == 6) {
    temp_output_tensor = cur_test->out.get_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
  } else {
    temp_output_tensor = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
  }

  int8_t* input = temp_input_tensor.data.mem.pi8;

    // CWHB layot
    uint32_t input_shape[kMaxRank] = { 0 };
    uint32_t output_shape[kMaxRank] = { 0 };
    int32_t input_stride[kMaxRank]= { 0 };
    int32_t output_stride[kMaxRank]= { 0 };
    
    for (int i = 0; i < temp_input_tensor.rank; i++) {
      input_shape[i] = temp_input_tensor.shape[i];
      input_stride[i] = temp_input_tensor.mem_stride[i];
      output_shape[i] = temp_output_tensor.shape[i];
      output_stride[i] = temp_output_tensor.mem_stride[i];
    }

    DataMoveConfig temp_move_conf = current_test_cfg(i);

    const lib_mli::Tensor<lib_mli::NoBuffer, kMaxRank> src_shape(input_shape, input_stride);
    const lib_mli::Tensor<lib_mli::NoBuffer, kMaxRank> dst_shape(output_shape, output_stride);

    lib_mli::IteratorCfg<4> src_it_cfg(temp_move_conf.sub_sample_step, input_shape);
    int32_t increments[4] = { 1, 1, 1, 1};
    lib_mli::IteratorCfg<4> dst_it_cfg(increments, output_shape);

    lib_mli::PlatformDescription pd;
    lib_ref::KernelsFactory kernel_factory(pd);
    uint32_t move_cs_size = kernel_factory.Move_CS_GetSize();
    void* move_cs_buffer = malloc(move_cs_size);
    auto move_op = kernel_factory.Move_CS(move_cs_buffer, src_shape, src_it_cfg, dst_shape, dst_it_cfg);

    // STEP 2: Memory management (Up to user on how to deal with it)
    //==================================================================

    uint32_t src_mem_offset;

    uint32_t offsets[1] = { 0 };

    uint32_t elem_size = cur_test->data_size;

    // Define buffers for in\out tensors
    // Leave space for runtime object
    uint32_t* offset = &offsets[0];
    uint32_t runtime_obj_size = move_op->GetRuntimeObjectSize();
    *offset += runtime_obj_size;

    // Leave space for private data buffer
    offset = &offsets[0];
    uint32_t private_buffer_size = move_op->GetKernelPrivateDataSize();
    *offset += private_buffer_size;

    // Move Source
    offset = &offsets[0];
    uint32_t src_size = move_op->GetInputBufferSize() * elem_size;
    lib_mli::OffsetBuffer move_in_buf{*offset, 0, src_size, elem_size};
    // Need to push stride manually otherwise it will be calculated wrong for reference
    lib_mli::Tensor<lib_mli::OffsetBuffer, 4> move_in_tensor (move_in_buf, input_shape, input_stride);
    src_mem_offset = *offset;
    *offset += src_size;

    // Move Destination
    offset = &offsets[0];
    uint32_t dst_size = move_op->GetOutputBufferSize() * elem_size;
    lib_mli::OffsetBuffer move_out_buf{*offset, 0, dst_size, elem_size};
    lib_mli::Tensor<lib_mli::OffsetBuffer, 4> move_out_tensor (move_out_buf, output_shape, output_stride);
    dst_mem_offset = *offset;
    *offset += dst_size;


   // Filling output for special test case called "Concat" with pre-filled
   // output tensor Data from src tensor copied to dst tensor with offset
   if (i == 6) {
    int8_t* local_output = temp_output_tensor.data.mem.pi8;
    for (uint32_t i = 0; i < temp_output_tensor.data.capacity; ++i) {
      const uint32_t idx = dst_mem_offset + i;
      g_mem_pool[idx] = local_output[i];
    }
  }

    // Attaching buffer (descriptors) to the operation
    mli_status status = MLI_STATUS_OK;

    // Getting slices for specific tests. Others will use full tensor
    if (i == 4 || i == 5 || i == 6) {
      move_in_tensor = move_in_tensor.slice(temp_move_conf.offset, temp_move_conf.size);
      move_out_tensor = move_out_tensor.slice(temp_move_conf.dst_offset, temp_move_conf.size);
    }

    status = move_op->AttachBufferOffsets(move_in_tensor,
                                         move_out_tensor);

    // Copy input data from scratch buffer to the shared memory pool
    for (uint32_t i = 0; i < temp_input_tensor.data.capacity; ++i) {
      const uint32_t idx = src_mem_offset + i;
      g_mem_pool[idx] = input[i];
    }

    move_instance = g_mem_pool;
    move_instance_size = move_op->GetRuntimeObjectSize();

    status = move_op->GetKernelPrivateData(g_mem_pool + move_instance_size);
    assert(status == MLI_STATUS_OK);
    move_conf_private = reinterpret_cast<lib_mli::PrivateData*>(g_mem_pool + move_instance_size);
    move_conf_private_size = move_op->GetKernelPrivateDataSize();

    // Will allocate it once more for post process stage
    mem_in_keeper.return_memory();
    mem_out_keeper.return_memory();
}

void execution_phase(int i, void* move_instance, uint32_t move_instance_size,
                     lib_mli::PrivateData* move_conf_private,
                     uint32_t move_conf_private_size) {

  uint32_t tiles_num = 1;

  uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool)};

  auto mli_move = lib_mli::ExecutionInterface::Create(
      static_cast<void*>(move_instance), move_instance_size, move_conf_private,
      move_conf_private_size, membasis, sizeof(membasis) / sizeof(membasis[0]));

  assert(mli_move != nullptr);

  mli_status status = MLI_STATUS_OK;
  for (int i = 0; i < tiles_num; ++i) {
    status = mli_move->Prefetch();
    assert(status == MLI_STATUS_OK);

    status = mli_move->Issue();
    assert(status == MLI_STATUS_OK);

    status = mli_move->Update();
    assert(status == MLI_STATUS_OK);
  }
}

bool postprocess_phase(int i, memory_manager& mem_in_keeper,
                       memory_manager& mem_out_keeper,
                       const reporter_full* reporter,
                       const data_movement_test_operands* cur_test,
                       const uint32_t dst_mem_offset) {
  quality_metrics test_metics;
  bool is_test_passed = false;

  mli_tensor temp_input_tensor = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
  mli_tensor temp_output_tensor;

  if (i == 6) {
    temp_output_tensor = cur_test->out.get_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
  } else {
    temp_output_tensor = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));
  }

  int8_t output[kMemSize];
  for (uint32_t i = 0; i < temp_output_tensor.data.capacity; ++i) {
    const uint32_t idx = dst_mem_offset + i;
    output[i] = g_mem_pool[idx];
  }

  temp_output_tensor.data.mem.pi8 = output;
  test_metics.calculate_metrics(temp_output_tensor, cur_test->out);

  crc32_calc data_crc;
  data_crc(temp_input_tensor);
  data_crc(temp_output_tensor);
  is_test_passed = reporter->evaluate_and_report_case(
      cur_test->descr, test_metics, cur_test->threshold, data_crc,
      cur_test->check_sum);
  return is_test_passed;
}

int main() {
  const reporter_full reporter;
  bool final_status = true;

  reporter.report_header("MLI|Kernels|Data Movement Functions Tests");
  for (int i = 0; i < kTestsNum; ++i) {
    memory_manager mem_in_keeper((int8_t*)g_scratch_mem_in, sizeof(g_scratch_mem_in));
    memory_manager mem_out_keeper((int8_t*)g_scratch_mem_out, sizeof(g_scratch_mem_out));


    bool is_test_passed = true;
    const data_movement_test_operands* cur_test = &tests_list[i];

    if (!(cur_test->in.is_valid() && cur_test->out.is_valid())) {
      reporter.report_message(cur_test->descr, "FAILED at init: bad source data for one of tensors");
      is_test_passed = false;
    }

    /**************************************************************************************************************/
    uint32_t dst_mem_offset = 0;

    void* move_instance = nullptr;
    uint32_t move_instance_size = 0;

    lib_mli::PrivateData* move_conf_private = nullptr;
    uint32_t move_conf_private_size = 0;
    /**************************************************************************************************************/

    prepare_phase(i, mem_in_keeper, mem_out_keeper, cur_test, dst_mem_offset,
                  move_instance, move_instance_size, move_conf_private,
                  move_conf_private_size);
    execution_phase(i, move_instance, move_instance_size, move_conf_private,
                    move_conf_private_size);
    
    is_test_passed &= postprocess_phase(i, mem_in_keeper, mem_out_keeper, &reporter, cur_test, dst_mem_offset);

    final_status &= is_test_passed;
  }
  
  reporter.report_outline("[AUTO] Group: mli_krn_move_30", final_status);
  return 0;
}
