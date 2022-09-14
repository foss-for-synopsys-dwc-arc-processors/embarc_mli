/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#include <cstdlib>
#include <cstring>

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

using lib_mli::kMoveRank;
using lib_mli::kMoveIterRank;

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
    int32_t* tile_size;
    uint32_t* first_increment;
    int32_t* sub_sample_step;
    uint32_t* dst_offset;
    int32_t * dst_mem_stride;
    int32_t *tiling_order;
    int32_t *tiling_count;
    int32_t *tiling_last_pos_inc;
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
     {"Test 1 Copy",                     input_1_fx16, test_1_out_fx16, sizeof(int16_t),
                                         thresholds_fx16_general, test_1_chksum_fx16},
     {"Test 1 Copy AXIS = 1",            input_1_sa8,
                                         test_1_out_sa8, sizeof(int8_t),
                                         thresholds_sa8_general, test_1_chksum_sa8},
     {"Test 2 Copy MemStr",             input_2_fx16,
                                        test_2_out_fx16, sizeof(int16_t),
                                        thresholds_fx16_general, test_2_chksum_fx16},
     {"Test 2 Copy MemStr",             input_2_sa8,
                                        test_2_out_sa8, sizeof(int8_t),
                                        thresholds_sa8_general, test_2_chksum_sa8},
     {"Test 3 Copy MemStr",              input_3_fx16,
                                         test_3_out_fx16, sizeof(int16_t),
                                         thresholds_fx16_general, test_3_chksum_fx16},
     {"Test 3 Copy MemStr",              input_3_sa8,
                                         test_3_out_sa8, sizeof(int8_t),
                                         thresholds_sa8_general, test_3_chksum_sa8},
// Tiling tests
     {"Test 1 Copy (tiling)",            input_1_fx16, test_1_out_fx16, sizeof(int16_t),
                                         thresholds_fx16_general, test_1_chksum_fx16},
     {"Test 1 Copy AXIS = 1 (tiling)",   input_1_sa8,
                                         test_1_out_sa8, sizeof(int8_t),
                                         thresholds_sa8_general, test_1_chksum_sa8},
     {"Test 2 Copy MemStr (tiling)",    input_2_fx16,
                                        test_2_out_fx16, sizeof(int16_t),
                                        thresholds_fx16_general, test_2_chksum_fx16},
     {"Test 2 Copy MemStr (tiling)",    input_2_sa8,
                                        test_2_out_sa8, sizeof(int8_t),
                                        thresholds_sa8_general, test_2_chksum_sa8},
     {"Test 3 Copy MemStr (tiling)",     input_3_fx16,
                                         test_3_out_fx16, sizeof(int16_t),
                                         thresholds_fx16_general, test_3_chksum_fx16},
     {"Test 3 Copy MemStr (tiling)",     input_3_sa8,
                                         test_3_out_sa8, sizeof(int8_t),
                                         thresholds_sa8_general, test_3_chksum_sa8},
     {"Test 1 Copy (overlap)",           input_1_fx16, test_1_out_fx16, sizeof(int16_t),
                                         thresholds_fx16_general, test_1_chksum_fx16},
     {"Test 1 Copy (tiling, double)",    input_1_fx16, test_1_out_fx16, sizeof(int16_t),
                                         thresholds_fx16_general, test_1_chksum_fx16},
     {"Test 1 Copy (c-order, double)",   input_1_fx16, test_1_out_fx16, sizeof(int16_t),
                                         thresholds_fx16_general, test_1_chksum_fx16},
     {"Test 1 Copy (c-order, overlap)",  input_1_fx16, test_1_out_fx16, sizeof(int16_t),
                                         thresholds_fx16_general, test_1_chksum_fx16}
};

DataMoveConfig current_test_cfg(int n) {
  return {offsets_cfg[n],
          sizes_cfg[n],
          tile_sizes[n],
          tile_first_increments[n],
          sub_sample[n],
          out_offsets_cfg[n],
          out_mem_stride_cfg[n],
          tiling_order[n],
          tiling_count[n],
          tiling_last_pos_inc[n]
          };
}


constexpr uint32_t kMemSize =  2048;
constexpr uint32_t kMemOutSize = 64;
static int8_t g_scratch_mem_in[kMemSize] = {0};
static int8_t g_scratch_mem_out[kMemSize] = {0};
static int8_t g_mem_pool[kMemSize] = {0};
static int8_t g_mem_out_pool[2 * kMemOutSize] = {0};
static int8_t g_mem_out_big_pool[kMemSize] = {0};

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

void prepare_phase(int test_num, memory_manager& mem_in_keeper,
                   memory_manager& mem_out_keeper,
                   const data_movement_test_operands* cur_test, void*& move_instance,
                   uint32_t& move_instance_size,
                   lib_mli::PrivateData*& move_conf_private,
                   uint32_t& move_conf_private_size,
                   uint32_t & dst_capacity,
                   uint32_t & rank,
                   uint32_t &elem_size,
                   uint32_t *input_shape,
                   uint32_t *output_shape,
                   int32_t * output_stride,
                   int32_t*& dst_tensor_shape, 
                   int32_t*& dst_tensor_iterator,
                   int32_t*& last_pos_inc_t,
                   int32_t*& tiling_order_t,
                   int32_t*& tiling_count_t
                   ) {

    mli_tensor temp_input_tensor = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
    mli_tensor temp_output_tensor = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));

    int8_t* input = temp_input_tensor.data.mem.pi8;

    rank = temp_input_tensor.rank;

    // CWHB layot
    int32_t input_stride[kMoveRank]= { 0 };
    
    for (int i = 0; i < rank; i++) {
      input_shape[i] = temp_input_tensor.shape[i];
      input_stride[i] = temp_input_tensor.mem_stride[i];
      output_shape[i] = temp_output_tensor.shape[i];
      output_stride[i] = temp_output_tensor.mem_stride[i];
    }

    DataMoveConfig temp_move_conf = current_test_cfg(test_num);
    
    const lib_mli::Tensor<lib_mli::NoBuffer, kMoveRank> src_shape(input_shape, input_stride);
    lib_mli::MoveDataDirection data_dir = lib_mli::MoveDataDirection::kMoveDataDirectionInput;
    const lib_mli::Tensor<lib_mli::NoBuffer, kMoveRank> dst_shape(output_shape, output_stride);

    dst_tensor_shape = (*temp_move_conf.tile_size == 0)
                           ? (int32_t*)input_shape
                           : temp_move_conf.tile_size;
    dst_tensor_iterator = (*(int32_t*)temp_move_conf.first_increment == 0)
                              ? (int32_t*)output_shape
                              : (int32_t*)temp_move_conf.first_increment;
      
    for (int i = 0; i < kMoveRank; i++) {
        if (dst_tensor_shape[i] == 0) {
            dst_tensor_shape[i] = 1;
        }
    }
    
    last_pos_inc_t=temp_move_conf.tiling_last_pos_inc;
    tiling_order_t=temp_move_conf.tiling_order;
    tiling_count_t=temp_move_conf.tiling_count;
    
    
    // Setting here tile size for IteratorCfg
    uint32_t tiles_number = test_num < 6 ? 0 : 1;
    tiles_number = test_num > 12 ? 2 : tiles_number;
    const int32_t * itr_order = tiling_order_t;

    //TODO: Check if size of slice is less or equal to the size of tensor
    int32_t * src_tile_mem_stride = test_num > 12 ? nullptr : input_stride;
    lib_mli::IteratorCfg<kMoveRank> src_it_cfg(tiling_order_t, tiling_count_t, dst_tensor_iterator, dst_tensor_iterator, last_pos_inc_t,
                                       dst_tensor_shape, dst_tensor_shape, dst_tensor_shape,0);
    int32_t * dst_tile_mem_stride = test_num > 12 ? nullptr : output_stride;
    lib_mli::IteratorCfg<kMoveRank> dst_it_cfg(tiling_order_t, tiling_count_t, dst_tensor_iterator, dst_tensor_iterator, last_pos_inc_t,
                                       dst_tensor_shape, dst_tensor_shape, dst_tensor_shape,tiles_number);

    lib_mli::TensorIterator<lib_mli::NoBuffer, kMoveRank, kMoveIterRank> src_it(src_shape, src_it_cfg);
    lib_mli::TensorIterator<lib_mli::NoBuffer, kMoveRank, kMoveIterRank> dst_it(dst_shape, dst_it_cfg);
    
    lib_mli::PlatformDescription pd;
    lib_ref::KernelsFactory kernel_factory(pd);
    uint32_t move_cs_size = kernel_factory.Move_CS_GetSize();
    void* move_cs_buffer = malloc(move_cs_size);
    auto move_op = kernel_factory.Move_CS(move_cs_buffer, src_it, dst_it, data_dir);

    // STEP 2: Memory management (Up to user on how to deal with it)
    //==================================================================

    uint32_t src_mem_offset;
    uint32_t dst_mem_offset;

    // Two buffers - separated for src and dst
    uint32_t offsets[2] = { 0, 0 };

    elem_size = cur_test->data_size;

    // Define buffers for in\out tensors
    uint32_t* input_offset = &offsets[0];
    uint32_t* output_offset = &offsets[1];

    // Leave space for runtime object
    uint32_t runtime_obj_size = move_op->GetRuntimeObjectSize();
    *input_offset += runtime_obj_size;

    // Leave space for private data buffer
    input_offset = &offsets[0];
    uint32_t private_buffer_size = move_op->GetKernelPrivateDataSize();
    *input_offset += private_buffer_size;

    // Move Source
    input_offset = &offsets[0];
    uint32_t temp_src_buffer_size = kMemSize;
    uint32_t src_size = temp_src_buffer_size;
    lib_mli::OffsetBuffer move_in_buf{*input_offset, 0, src_size, elem_size};
    // Need to push stride manually otherwise it will be calculated wrong for reference
    // lib_mli::Tensor<lib_mli::OffsetBuffer, kMoveRank> move_in_tensor (move_in_buf, input_shape, input_stride);
    src_mem_offset = *input_offset;
    *input_offset += src_size;

    // Move Destination
    output_offset = &offsets[1];
    uint32_t temp_dst_buffer_size = test_num < 6 ? kMemSize : kMemOutSize;
    temp_dst_buffer_size = test_num > 12 ? kMemOutSize * 2 : temp_dst_buffer_size;
    uint32_t dst_size = temp_dst_buffer_size;
    lib_mli::OffsetBuffer move_out_buf{*output_offset, 1, dst_size, elem_size};
    // lib_mli::Tensor<lib_mli::OffsetBuffer, kMoveRank> move_out_tensor (move_out_buf, output_shape, output_stride);
    dst_mem_offset = *output_offset;
    *output_offset += dst_size;

    // Control buffer size for ref is 0
    input_offset = &offsets[0];
    uint32_t temp_ctrl_buffer_size = move_op->GetCtrlBufferSize();
    lib_mli::OffsetBuffer move_ctrl_buf{*input_offset, 0, temp_ctrl_buffer_size, elem_size};
    // lib_mli::Tensor<lib_mli::OffsetBuffer, kMoveRank> move_out_tensor (move_out_buf, output_shape, output_stride);
    // dst_mem_offset = *output_offset;
    *input_offset += temp_ctrl_buffer_size;


    // Attaching buffer (descriptors) to the operation
    mli_status status = MLI_STATUS_OK;
    // status = move_op->AttachBufferOffsets(move_in_tensor, move_out_tensor);
    status = move_op->AttachBufferOffsets(move_in_buf, move_out_buf, move_ctrl_buf);

    // Copy input data from scratch buffer to the shared memory pool
    for (uint32_t i = 0; i < temp_input_tensor.data.capacity; ++i) {
      const uint32_t idx = src_mem_offset + i;
      g_mem_pool[idx] = input[i];
    }

    // Setting parameters to share with execution phase

    dst_capacity = temp_output_tensor.data.capacity;

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

void execution_phase(int test_num, void* move_instance, uint32_t move_instance_size,
                     lib_mli::PrivateData* move_conf_private,
                     uint32_t move_conf_private_size, uint32_t dst_capacity, uint32_t rank,
                     uint32_t dst_elem_size, uint32_t* dst_shape,
                     int32_t* dst_mem_stride, int32_t* tile_shape, int32_t* tile_iterator,
                     int32_t*& last_pos_inc_t, int32_t*& tiling_order_t, int32_t*& tiling_count_t, int8_t* output) {

  int8_t * p_mem_out_pool; 
  if (test_num <= 6) {
    p_mem_out_pool = g_mem_out_big_pool;
  } else {
    p_mem_out_pool = g_mem_out_pool;
  }
  
  uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool), reinterpret_cast<uint64_t>(p_mem_out_pool)};
  
  auto mli_move = lib_mli::ExecutionInterface::Create(move_instance, move_instance_size, move_conf_private,
      move_conf_private_size, membasis, sizeof(membasis) / sizeof(membasis[0]));

  assert(mli_move != nullptr);

  mli_status status = MLI_STATUS_OK;

  const int32_t * itr_order = tiling_order_t;
  int32_t * tile_mem_stride = test_num > 12 ? nullptr : dst_mem_stride;

  // Setting up result buffer tensor and iterator
  lib_mli::Buffer output_result_buffer(output, dst_capacity, dst_elem_size);
  lib_mli::Tensor<lib_mli::Buffer, kMoveRank> output_result_tensor(output_result_buffer, dst_shape, dst_mem_stride);
  lib_mli::IteratorCfg<kMoveRank> output_result_it_cfg(tiling_order_t, tiling_count_t, tile_iterator, tile_iterator, last_pos_inc_t, 
                                              tile_shape, tile_shape, tile_shape, 0);
  lib_mli::TensorIterator output_result_itr(output_result_tensor, output_result_it_cfg);
  
  // Setting up kernel output tensor and iterator
  uint32_t buf_size = (test_num < 6 ? kMemSize : kMemOutSize);
  buf_size = test_num > 12 ? kMemOutSize * 2 : buf_size;
  lib_mli::Buffer output_kernel_buffer((void*)membasis[1], buf_size, dst_elem_size);
  lib_mli::Tensor<lib_mli::Buffer, kMoveRank> output_kernel_tensor(output_kernel_buffer, dst_shape, dst_mem_stride);
  uint32_t tiles_number = test_num < 6 ? 0 : 1;
  tiles_number = test_num > 12 ? 2 : tiles_number;
  lib_mli::IteratorCfg<kMoveRank> output_kernel_it_cfg(tiling_order_t, tiling_count_t, tile_iterator, tile_iterator, last_pos_inc_t,tile_shape, 
                                              tile_shape, tile_shape, tiles_number);
  lib_mli::TensorIterator output_kernel_itr(output_kernel_tensor, output_kernel_it_cfg);


  bool out_done = false;
  while(!out_done) {
    status = mli_move->Prefetch();
    assert(status == MLI_STATUS_OK);

    status = mli_move->Issue();
    assert(status == MLI_STATUS_OK);

    lib_mli::TensorIterator<lib_mli::Buffer, kMoveRank, kMoveIterRank> src_it = output_kernel_itr.GetSubTensorIterator();
    lib_mli::TensorIterator<lib_mli::Buffer, kMoveRank, kMoveIterRank> dst_it = output_result_itr.GetSubTensorIterator();
    int32_t count_[5]={0,0,0,0};
        for (uint32_t i = 0; i < kMoveRank; i++)
        {
          count_[i]=src_it.GetTensorShape(i);
          if (count_[i]<=0)count_[i]=1;
        }
    src_it.SetCount(count_);
    dst_it.SetCount(count_);
    bool done = false;
    while (!done) {
      switch (src_it.get_elem_size()) {
        case 1:
          dst_it.write(src_it.template read<uint8_t>());
          break;
        case 2:
          dst_it.write(src_it.template read<uint16_t>());
          break;
        case 4:
          dst_it.write(src_it.template read<uint32_t>());
          break;
        default:
          MLI_ASSERT(false);
      }
      done = src_it.Next();
      done |= dst_it.Next();
    }
    out_done = output_result_itr.Next();
    out_done |= output_kernel_itr.Next();
    
    status = mli_move->Update();
    
    assert(status == MLI_STATUS_OK);
  }
}

bool postprocess_phase(int test_num, memory_manager& mem_in_keeper,
                       memory_manager& mem_out_keeper,
                       const reporter_full* reporter,
                       const data_movement_test_operands* cur_test, int8_t* output) {
  quality_metrics test_metics;
  bool is_test_passed = false;

  mli_tensor temp_input_tensor = cur_test->in.get_quantized_tensor(mem_in_keeper.allocate_memory(cur_test->in));
  mli_tensor temp_output_tensor = cur_test->out.get_not_quantized_tensor(mem_out_keeper.allocate_memory(cur_test->out));

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

  reporter.report_header("MLI3.0|Kernels|Data Movement Functions Tests");
  for (int test_num =0; test_num < kTestsNum; ++test_num) {
    memory_manager mem_in_keeper((int8_t*)g_scratch_mem_in, sizeof(g_scratch_mem_in));
    memory_manager mem_out_keeper((int8_t*)g_scratch_mem_out, sizeof(g_scratch_mem_out));


    bool is_test_passed = true;
    const data_movement_test_operands* cur_test = &tests_list[test_num];

    if (!(cur_test->in.is_valid() && cur_test->out.is_valid())) {
      reporter.report_message(cur_test->descr, "FAILED at init: bad source data for one of tensors");
      is_test_passed = false;
    }

    /**************************************************************************************************************/

    void* move_instance = nullptr;
    uint32_t move_instance_size = 0;

    lib_mli::PrivateData* move_conf_private = nullptr;
    uint32_t move_conf_private_size = 0;

    uint32_t rank;
    uint32_t dst_capacity;
    uint32_t dst_elem_size;
    uint32_t src_shape[kMoveRank] = {1, 1, 1, 1, 1};
    uint32_t dst_shape[kMoveRank] = {1, 1, 1, 1,1};
    int32_t dst_mem_stride[kMoveRank] = {0};
    int32_t *tile_shape;
    int32_t *tile_iterator;
    int32_t *last_pos_inc_t;
    int32_t *tiling_order_t;
    int32_t *tiling_count_t;
    int8_t output[kMemSize];


    /**************************************************************************************************************/

    prepare_phase(test_num, mem_in_keeper, mem_out_keeper, cur_test,
                  move_instance, move_instance_size, move_conf_private,
                  move_conf_private_size, dst_capacity, rank, dst_elem_size, src_shape,
                  dst_shape, dst_mem_stride, tile_shape, tile_iterator, last_pos_inc_t,
                  tiling_order_t, tiling_count_t);
    for (int i = 0; i < kMemSize; i++) {
      output[i] = 0;
    }
    for (int i = 0; i < kMemOutSize; i++) {
      g_mem_out_pool[i] = 0;
    }
    for (int i = 0; i < kMemSize; i++) {
      g_mem_out_big_pool[i] = 0;
    }
    execution_phase(test_num, move_instance, move_instance_size, move_conf_private,
                    move_conf_private_size, dst_capacity, rank,
                    dst_elem_size, dst_shape, dst_mem_stride, tile_shape, tile_iterator,
                    last_pos_inc_t, tiling_order_t, tiling_count_t, output);

    is_test_passed &= postprocess_phase(test_num, mem_in_keeper, mem_out_keeper, &reporter, cur_test, output);

    final_status &= is_test_passed;
  }

  reporter.report_outline("[AUTO] Group: mli_krn_data_movement_30", final_status);
  return 0;
}
