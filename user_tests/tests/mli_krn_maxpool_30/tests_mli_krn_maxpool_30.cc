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
#include "mli_compiler_api.hpp"
#include "mli_runtime_api.hpp"
#include "mli_types.h"
#include "mli_types.hpp"

#include "test_crc32_calc.h"
#include "test_quality_metrics.h"
#include "test_report.h"
#include "test_tensor_quantizer.h"

#include "vectors_mli_krn_maxpool.inc"

using mli::tst::crc32_calc;
using mli::tst::quality_metrics;
using mli::tst::reporter_full;
using mli::tst::tensor_quantizer;

namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;

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

constexpr uint32_t kMemSize = 2048;
static int8_t g_scratch_mem_in[kMemSize] = {0};
static int8_t g_scratch_mem_out[kMemSize] = {0};
static int8_t g_mem_pool[kMemSize] = {0};

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

void prepare_phase(const maxpool_test_operands* cur_test,
                   uint32_t& out_mem_offset, void*& maxpool2d_instance,
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
  temp_out_container.mem.pi8 = g_scratch_mem_out;

  mli_tensor temp_input_tensor =
      cur_test->in.get_quantized_tensor(temp_in_container);
  mli_tensor temp_output_tensor =
      cur_test->out.get_not_quantized_tensor(temp_out_container);

  int8_t* input = temp_input_tensor.data.mem.pi8;

  // STEP 1: Construct MaxPool2D as a specific ExecutionInterface successor
  //==================================================================

  // CWHB layot
  uint32_t input_shape[4];
  uint32_t output_shape[4];
  int32_t input_stride[4];
  int32_t output_stride[4];

  // TODO: Do not recalculated quantized tensor everytime - need to calculate it
  // once to use in all cases.

for (int i = 1; i < 4; i++) {
    input_shape[i] = temp_input_tensor.shape[i-1];
    output_shape[i] = temp_output_tensor.shape[i-1];
    input_stride[i] = temp_input_tensor.mem_stride[i-1];
    output_stride[i] = temp_output_tensor.mem_stride[i-1];
  }
  input_shape[0] = output_shape[0] = 1;
  input_stride[0] = output_stride[0] = 0;


  const lib_mli::Tensor<lib_mli::NoBuffer, 4> in_tensor(
      input_shape, input_stride);
  const lib_mli::Tensor<lib_mli::NoBuffer, 4> out_tensor(
      output_shape, output_stride);

  lib_mli::PlatformDescription pd;
  lib_ref::KernelsFactory kernel_factory(pd);
  uint32_t maxpool2d_cs_size = kernel_factory.MaxPool2D_CS_GetSize();
  void* maxpool2d_cs_buffer = malloc(maxpool2d_cs_size);
  auto maxpool2d_op = kernel_factory.MaxPool2D_CS(maxpool2d_cs_buffer, in_tensor, cur_test->cfg, out_tensor);

  // STEP 2: Memory management (Up to user on how to deal with it)
  //==================================================================
  uint32_t in_mem_offset;

  uint32_t offsets[1] = {0};

  uint32_t elem_size = cur_test->data_size;

  // Define buffers for in\out tensors
  // Leave space for runtime object
  uint32_t* offset = &offsets[0];
  uint32_t runtime_obj_size = maxpool2d_op->GetRuntimeObjectSize();
  *offset += runtime_obj_size;

  // Leave space for private data buffer
  offset = &offsets[0];
  uint32_t private_buffer_size = maxpool2d_op->GetKernelPrivateDataSize();
  *offset += private_buffer_size;

  // MaxPool2D Input
  offset = &offsets[0];
  uint32_t in_size = maxpool2d_op->GetInputBufferSize() * elem_size;
  lib_mli::OffsetBuffer maxpool2d_in_buf{*offset, 0, in_size, elem_size};
  lib_mli::Tensor<lib_mli::OffsetBuffer, 4> maxpool2d_in_tensor(maxpool2d_in_buf, input_shape);
  in_mem_offset = *offset;
  *offset += in_size;

  // MaxPool2D Output
  offset = &offsets[0];
  uint32_t out_size = maxpool2d_op->GetOutputBufferSize() * elem_size;
  lib_mli::OffsetBuffer maxpool2d_out_buf{*offset, 0, out_size, elem_size};
  lib_mli::Tensor<lib_mli::OffsetBuffer, 4> maxpool2d_out_tensor(maxpool2d_out_buf, input_shape);
  out_mem_offset = *offset;
  *offset += out_size;

  // DataBuffer size is 0 for reference kernel
  offset = &offsets[0];
  uint32_t data_buffer_size = maxpool2d_op->GetDataBufferSize();
  lib_mli::OffsetBuffer maxpool2d_descr_buf{*offset, 0, data_buffer_size,
                                            sizeof(char)};
  *offset += data_buffer_size;

  // Attaching buffer (descriptors) to the operation
  mli_status status = MLI_STATUS_OK;

  status = maxpool2d_op->AttachBufferOffsets(maxpool2d_in_tensor, maxpool2d_out_tensor,
                                            maxpool2d_descr_buf);
  assert(status == MLI_STATUS_OK);

  // Copy input data from scratch buffer to the shared memory pool
  for (uint32_t i = 0; i < temp_in_container.capacity; ++i) {
    const uint32_t idx = in_mem_offset + i;
    g_mem_pool[idx] = input[i];
  }

  maxpool2d_instance = g_mem_pool;
  maxpool2d_instance_size = maxpool2d_op->GetRuntimeObjectSize();

  status =
      maxpool2d_op->GetKernelPrivateData(g_mem_pool + maxpool2d_instance_size);
  assert(status == MLI_STATUS_OK);
  maxpool2d_conf_private = reinterpret_cast<lib_mli::PrivateData*>(
      g_mem_pool + maxpool2d_instance_size);
  maxpool2d_conf_private_size = maxpool2d_op->GetKernelPrivateDataSize();
}

void execution_phase(void* maxpool2d_instance, uint32_t maxpool2d_instance_size,
                     lib_mli::PrivateData* maxpool2d_conf_private,
                     uint32_t maxpool2d_conf_private_size) {
  // STEP 3: Execution phase
  //==================================================================
  uint32_t tiles_num = 1;

  uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool)};

  auto mli_maxpool2d = lib_mli::ExecutionInterface::Create(
      maxpool2d_instance, maxpool2d_instance_size, maxpool2d_conf_private,
      maxpool2d_conf_private_size, membasis,
      sizeof(membasis) / sizeof(membasis[0]));

  assert(mli_maxpool2d != nullptr);

  mli_status status = MLI_STATUS_OK;
  for (int i = 0; i < tiles_num; ++i) {
    status = mli_maxpool2d->Prefetch();
    assert(status == MLI_STATUS_OK);

    status = mli_maxpool2d->Issue();
    assert(status == MLI_STATUS_OK);

    status = mli_maxpool2d->Update();
    assert(status == MLI_STATUS_OK);
  }
}

bool postprocess_phase(const reporter_full* reporter,
                       const maxpool_test_operands* cur_test,
                       const uint32_t out_mem_offset) {
  quality_metrics test_metics;
  bool is_test_passed = false;

  mli_data_container temp_in_container{0};
  mli_data_container temp_out_container{0};
  temp_in_container.capacity =
      cur_test->in.get_not_quantized_tensor(temp_in_container).data.capacity;
  temp_in_container.mem.pi8 = g_scratch_mem_in;
  temp_out_container.capacity =
      cur_test->out.get_not_quantized_tensor(temp_out_container).data.capacity;
  temp_out_container.mem.pi8 = g_scratch_mem_out;

  mli_tensor temp_input_tensor =
      cur_test->in.get_quantized_tensor(temp_in_container);
  mli_tensor temp_output_tensor =
      cur_test->out.get_not_quantized_tensor(temp_out_container);

  // Copy output data from shared memory pool to the local scratch buffer
  int8_t output[1024];
  for (uint32_t i = 0; i < temp_out_container.capacity; ++i) {
    const uint32_t idx = out_mem_offset + i;
    output[i] = g_mem_pool[idx];
  }

  temp_output_tensor.data.mem.pi8 = output;
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

  reporter.report_header("MLI|Kernels|Max Pooling Function Tests");
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
    uint32_t out_mem_offset = 0;

    void* maxpool2d_instance = nullptr;
    uint32_t maxpool2d_instance_size = 0;

    lib_mli::PrivateData* maxpool2d_conf_private = nullptr;
    uint32_t maxpool2d_conf_private_size = 0;
    /**************************************************************************************************************/

    prepare_phase(cur_test, out_mem_offset, maxpool2d_instance,
                  maxpool2d_instance_size, maxpool2d_conf_private,
                  maxpool2d_conf_private_size);

    execution_phase(maxpool2d_instance, maxpool2d_instance_size,
                    maxpool2d_conf_private, maxpool2d_conf_private_size);

    is_test_passed &= postprocess_phase(&reporter, cur_test, out_mem_offset);

    final_status &= is_test_passed;
  }
  reporter.report_outline("[AUTO] Group: mli_krn_maxpool_30", final_status);

  return 0;
}