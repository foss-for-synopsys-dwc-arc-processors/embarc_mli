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
#include "vectors_mli_krn_matmul.inc"



using namespace snps_arc::metaware::mli::service;

using mli::tst::crc32_calc;
using mli::tst::quality_metrics;
using mli::tst::reporter_full;
using mli::tst::tensor_quantizer;
using snps_arc::metaware::mli::InternalBuffer;
using snps_arc::metaware::mli::Tensor;
using snps_arc::metaware::mli::OffsetBuffer;
using snps_arc::metaware::mli::kMatMulRank;
using snps_arc::metaware::mli::kMatMulIterRank;


namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;

struct MatMul_test_operands {
  const char* descr;
  tensor_quantizer in1;
  tensor_quantizer in2;
  int8_t in1_zp;
  int8_t in2_zp;
  tensor_quantizer out;
  uint32_t data_size;
  const quality_metrics threshold;
  const crc32_calc check_sum;
};

const crc32_calc test_1_chksum_sa8{ 0x9AAA87CA }, test_2_chksum_sa8{ 0x387DBA3E }, test_3_chksum_sa8{ 0x80E93591 },
                 test_4_chksum_sa8{ 0xFE711D0A }, test_5_chksum_sa8{ 0x387DBA3E }, test_6_chksum_sa8{ 0xD6CED655 }; 

const quality_metrics thresholds_fx16_general{
    quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
    /* SNR_DB = */ 84.f, /*Quant Error Perc = */ 0.0f};
const quality_metrics thresholds_sa8_general{
    quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
    /* SNR_DB = */ 40.f, /*Quant Error Perc = */ 0.0f};

static MatMul_test_operands tests_list[] = {
    // (4, 1) * (1, 6) = (4, 6)
    {"Test 1 SA8 (4, 1) * (1, 6)",  input_1_sa8, input_2_sa8, (int8_t)input_1_zero_point, (int8_t)input_2_zero_point, test_1_out_sa32, sizeof(int8_t), thresholds_sa8_general, test_1_chksum_sa8},
    // (2, 9) * (9, 5) = (2, 5)
    {"Test 2 SA8 (2, 9) * (9, 5)", input_3_sa8, input_4_sa8, (int8_t)input_3_zero_point, (int8_t)input_4_zero_point, test_2_out_sa32, sizeof(int8_t), thresholds_sa8_general, test_2_chksum_sa8},   
    // (3, 4) * (4, 5) = (3, 5)
    {"Test 3 SA8 (3, 4) * (4, 5)", input_5_sa8, input_6_sa8, (int8_t)input_5_zero_point, (int8_t)input_6_zero_point, test_3_out_sa32, sizeof(int8_t), thresholds_sa8_general, test_3_chksum_sa8},    
    // (4, 9) * (9, 5)  = (4, 5)     
    {"Test 4 SA8 (4, 9) * (9, 5)", input_7_sa8, input_8_sa8, (int8_t)input_7_zero_point, (int8_t)input_8_zero_point, test_4_out_sa32, sizeof(int8_t), thresholds_sa8_general, test_4_chksum_sa8}, 
    // (2, 9) * (9, 5) = (2, 5)
    {"Test 5 SA8 (2, 9) * (9, 5)", input_3_sa8, input_8_sa8, (int8_t)input_3_zero_point, (int8_t)input_8_zero_point, test_5_out_sa32, sizeof(int8_t), thresholds_sa8_general, test_5_chksum_sa8},  
    // (3, 4) * (4, 9) = (3, 9) 
    {"Test 6 SA8 (3, 4) * (4, 9)", input_5_sa8, input_7_sa8, (int8_t)input_5_zero_point, (int8_t)input_7_zero_point, test_6_out_sa32, sizeof(int8_t), thresholds_sa8_general, test_6_chksum_sa8}
  };

constexpr uint32_t kMemSize = 8192;
static int8_t g_scratch_mem_in1[kMemSize] = {0};
static int8_t g_scratch_mem_in2[kMemSize] = {0};
static int8_t g_scratch_mem_ref[kMemSize] = {0};
static int8_t g_scratch_mem_out[kMemSize] = {0};
static IO_DATA_ATTR int8_t g_mem_pool[kMemSize] = {0};

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);


void prepare_phase(MatMul_test_operands* cur_test,
                   void*& MatMul_instance,
                   uint32_t& MatMul_instance_size,
                   void*& MatMul_conf_private,
                   uint32_t& MatMul_conf_private_size,
                   lib_mli::TensorIterator<lib_mli::OffsetBuffer, kMatMulRank, kMatMulIterRank> &input1_tensor,
                   lib_mli::TensorIterator<lib_mli::OffsetBuffer, kMatMulRank, kMatMulIterRank> &input2_tensor,
                   lib_mli::TensorIterator<lib_mli::OffsetBuffer, kMatMulRank, kMatMulIterRank> &output_tensor,
                   uint32_t *input_tile_shape,
                   uint32_t *output_tile_shape,
                   int32_t *iteration_order ) {

  
    mli_data_container temp_in1_container{0};
    mli_data_container temp_in2_container{0};
    mli_data_container temp_out_container{0};
    temp_in1_container.capacity = cur_test->in1.get_not_quantized_tensor(temp_in1_container).data.capacity;
    temp_in1_container.mem.pi8 = g_scratch_mem_in1;
    temp_in2_container.capacity = cur_test->in2.get_not_quantized_tensor(temp_in2_container).data.capacity;
    temp_in2_container.mem.pi8 = g_scratch_mem_in2;
    temp_out_container.capacity = cur_test->out.get_not_quantized_tensor(temp_out_container).data.capacity;
    temp_out_container.mem.pi8 = g_scratch_mem_ref;

    // STEP 1: Construct MatMul as a specific ExecutionInterface successor
    //==================================================================

    mli_tensor temp_input1_tensor = cur_test->in1.get_quantized_tensor(temp_in1_container);
    mli_tensor temp_input2_tensor = cur_test->in2.get_quantized_tensor(temp_in2_container);
    mli_tensor temp_output_tensor = cur_test->out.get_quantized_tensor(temp_out_container);

    const lib_mli::Tensor<lib_mli::NoBuffer, kMatMulRank> in1_tensor(temp_input1_tensor.shape, temp_input1_tensor.mem_stride);
    const lib_mli::Tensor<lib_mli::NoBuffer, kMatMulRank> in2_tensor(temp_input2_tensor.shape, temp_input2_tensor.mem_stride);
    const lib_mli::Tensor<lib_mli::NoBuffer, kMatMulRank> out_tensor(temp_output_tensor.shape, temp_output_tensor.mem_stride);

    lib_mli::TensorIterator<lib_mli::NoBuffer, kMatMulRank, kMatMulIterRank> in1_tensor_it(in1_tensor, input_tile_shape, iteration_order);
    lib_mli::TensorIterator<lib_mli::NoBuffer, kMatMulRank, kMatMulIterRank> in2_tensor_it(in2_tensor, temp_input2_tensor.shape, iteration_order);
    lib_mli::TensorIterator<lib_mli::NoBuffer, kMatMulRank, kMatMulIterRank> out_tensor_it(out_tensor, output_tile_shape, iteration_order);

    input1_tensor = in1_tensor_it;
    input2_tensor = in2_tensor_it;
    output_tensor = out_tensor_it;

    lib_mli::PlatformDescription pd;
    lib_ref::KernelsFactory kernel_factory(pd);
    uint32_t MatMul_cs_size = kernel_factory.MatMul_CS_GetSize();
    void* MatMul_cs_buffer = malloc(MatMul_cs_size);
    auto MatMul_op = kernel_factory.MatMul_CS(MatMul_cs_buffer, in1_tensor_it, in2_tensor_it, out_tensor_it);

    // STEP 2: Memory management (Up to user on how to deal with it)
    //==================================================================

    uint32_t offsets[1] = {0};
    uint32_t elem_size = cur_test->data_size;
    mli_status status = MLI_STATUS_OK;

    // Define buffers for in/out tensors
    // Leave space for runtime object
    uint32_t* offset = &offsets[0];
    uint32_t runtime_obj_size = MatMul_op->GetRuntimeObjectSize();
    *offset += runtime_obj_size;

    // Leave space for private data buffer
    offset = &offsets[0];
    uint32_t private_buffer_size = MatMul_op->GetKernelPrivateDataSize();
    uint32_t privateData_mem_offset = *offset;
    *offset += private_buffer_size;

    // MatMul Input1
    offset = &offsets[0];

    uint32_t in1_size = lib_mli::service::GetBufferSize(lib_mli::kMatMulRank, input_tile_shape, temp_input1_tensor.mem_stride) * elem_size;
    lib_mli::OffsetBuffer MatMul_in1_buf{*offset, 0, in1_size, elem_size};
    input1_tensor.set_buf(MatMul_in1_buf);
    uint32_t in1_mem_offset = *offset;
    *offset += in1_size;

    // MatMul Input2
    offset = &offsets[0];
    uint32_t in2_size = lib_mli::service::GetBufferSize(lib_mli::kMatMulRank, temp_input2_tensor.shape, temp_input2_tensor.mem_stride) * elem_size;
    lib_mli::OffsetBuffer MatMul_in2_buf{*offset, 0, in2_size, elem_size};
    input2_tensor.set_buf(MatMul_in2_buf);
    uint32_t in2_mem_offset = *offset;
    *offset += in2_size;

    // MatMul Output
    offset = &offsets[0];

    uint32_t out_size = lib_mli::service::GetBufferSize(lib_mli::kMatMulRank, output_tile_shape, temp_output_tensor.mem_stride) * sizeof(int32_t);
    lib_mli::OffsetBuffer MatMul_out_buf{*offset, 0, out_size, sizeof(int32_t)};
    output_tensor.set_buf(MatMul_out_buf);
    uint32_t out_mem_offset = *offset;
    *offset += out_size;

    // MatMul input zero point
    uint32_t inpzp_size = MatMul_op->GetEncodedParamsSize() * elem_size;
    lib_mli::OffsetBuffer MatMul_encoded_params_buf{*offset, 0, inpzp_size, elem_size};
    uint32_t inpzp_mem_offset = *offset;
    *offset += inpzp_size;

    // CtrlBuffer size is 0 for reference kernel
    offset = &offsets[0];
    assert(MatMul_op->GetCtrlBufferSize() == 0);
    uint32_t ctrl_buffer_size = 0;
    lib_mli::OffsetBuffer MatMul_ctrl_buf{*offset, 0, ctrl_buffer_size, sizeof(char)};
    *offset += ctrl_buffer_size;

    assert(*offset <= kMemSize);
    // Attaching buffer (descriptors) to the operation
    status = MatMul_op->AttachBufferOffsets(MatMul_in1_buf, 
                                            MatMul_in2_buf,
                                            MatMul_out_buf,
                                            MatMul_encoded_params_buf,
                                            MatMul_ctrl_buf);
    assert(status == MLI_STATUS_OK);

   /*Encode inputs zp*/
    uint8_t dst[2]{0};
    uint8_t dst_size = 2;
    const lib_mli::Buffer in1_zp_buf(&cur_test->in1_zp, sizeof(cur_test->in1_zp), sizeof(int8_t));
    const lib_mli::Buffer in2_zp_buf(&cur_test->in1_zp, sizeof(cur_test->in2_zp), sizeof(int8_t));
    lib_mli::Buffer encoded_zp_buf(&dst, dst_size, sizeof(int8_t));

    MatMul_op->EncodeParams(in1_zp_buf, 
                            in2_zp_buf,
                            encoded_zp_buf);

   /*copy zp from scratch memory to g_mem_pool*/
    for (uint32_t i = 0; i < inpzp_size; ++i) {
      const uint32_t idx = inpzp_mem_offset + i;
      g_mem_pool[idx] = encoded_zp_buf.read<int8_t>(i);
    }

    MatMul_instance = (int8_t*)g_mem_pool;
    MatMul_instance_size = MatMul_op->GetRuntimeObjectSize();

    status = MatMul_op->GetKernelPrivateData((int8_t*)g_mem_pool + privateData_mem_offset);
    assert(status == MLI_STATUS_OK);
    MatMul_conf_private = (int8_t*)g_mem_pool + privateData_mem_offset;
    MatMul_conf_private_size = MatMul_op->GetKernelPrivateDataSize();

}

void execution_phase(const MatMul_test_operands* cur_test, 
                     void* MatMul_instance, 
                     uint32_t MatMul_instance_size,
                     void*& MatMul_conf_private,
                     uint32_t& MatMul_conf_private_size,
                     lib_mli::TensorIterator<lib_mli::OffsetBuffer, kMatMulRank, kMatMulIterRank> &input1_tensor,
                     lib_mli::TensorIterator<lib_mli::OffsetBuffer, kMatMulRank, kMatMulIterRank> &input2_tensor,
                     lib_mli::TensorIterator<lib_mli::OffsetBuffer, kMatMulRank, kMatMulIterRank> &output_tensor,
                     uint32_t& num_tiles,
                     int32_t* iteration_order) {
  // STEP 3: Execution phase
  //==================================================================

  uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool)};

  auto MatMul_run_op = lib_mli::ExecutionInterface::Create(MatMul_instance,
                                                           MatMul_instance_size,
                                                           MatMul_conf_private,
                                                           MatMul_conf_private_size,
                                                           membasis, sizeof(membasis) / sizeof(membasis[0]));


    assert(MatMul_run_op != nullptr);
    mli_status status = MLI_STATUS_OK;

    uint32_t input1_tile_size[kMatMulRank]{};
    uint32_t input2_tile_size[kMatMulRank]{};
    uint32_t output_tile_size[kMatMulRank]{};
    int32_t input1_tile_offsets[kMatMulRank]{};
    int32_t input2_tile_offsets[kMatMulRank]{};
    int32_t output_tile_offsets[kMatMulRank]{};
    int32_t tile_input1_strides[kMatMulRank]{};
    int32_t tile_input2_strides[kMatMulRank]{};
    int32_t tile_output_strides[kMatMulRank]{};
    const int32_t zero_offsets[kMatMulRank]{};

    input1_tensor.get_mem_strides(tile_input1_strides);
    input2_tensor.get_mem_strides(tile_input2_strides);
    output_tensor.get_mem_strides(tile_output_strides);

    for (size_t i = 0; i < num_tiles; i++)
    {
    lib_ref::MatMul* pimpl = dynamic_cast<lib_ref::MatMul*>(MatMul_run_op);
    pimpl->GetIOSizesAndOffsets(input1_tile_size, input2_tile_size, output_tile_size,
                                input1_tile_offsets, input2_tile_offsets, output_tile_offsets);
    input2_tile_offsets[0] = input2_tile_offsets[1] = 0;

    status = MatMul_run_op->Prefetch();
    assert(status == MLI_STATUS_OK);
    
     // copy inputs from global buffer to local tile buffer 
    strided_copy_with_offsets(kMatMulRank, input1_tensor.get_buf().get_elem_size(),
                            g_scratch_mem_in1, input1_tile_offsets, zero_offsets, tile_input1_strides,
                            input1_tile_size, (int8_t*)(g_mem_pool + input1_tensor.get_buf().get_offset()));

    strided_copy_with_offsets(kMatMulRank, input2_tensor.get_buf().get_elem_size(),
                            g_scratch_mem_in2, input2_tile_offsets, zero_offsets, tile_input2_strides,
                            input2_tile_size, (int8_t*)(g_mem_pool + input2_tensor.get_buf().get_offset()));
    
    
    status = MatMul_run_op->Issue();
    assert(status == MLI_STATUS_OK);

    // copy output from local tile buffer to global buffer
    strided_copy_with_offsets(kMatMulRank,  output_tensor.get_buf().get_elem_size(),
                              (int8_t*)(g_mem_pool + output_tensor.get_buf().get_offset()),
                              zero_offsets, output_tile_offsets, tile_output_strides,
                              output_tile_size, (int8_t*)g_scratch_mem_out);

    status = MatMul_run_op->Update();
    assert(status == MLI_STATUS_OK);    
    }
                     }

bool postprocess_phase(const reporter_full* reporter,
                       const MatMul_test_operands* cur_test) {

    quality_metrics test_metics;
    bool is_test_passed = false;

    mli_data_container temp_in1_container{0};
    mli_data_container temp_in2_container{0};
    mli_data_container temp_out_container{0};
    mli_data_container temp_ref_container{0};
    
    temp_in1_container.capacity = cur_test->in1.get_not_quantized_tensor(temp_in1_container).data.capacity;
    temp_in2_container.capacity = cur_test->in2.get_not_quantized_tensor(temp_in2_container).data.capacity;
    temp_out_container.capacity = cur_test->out.get_not_quantized_tensor(temp_out_container).data.capacity;
    temp_ref_container.capacity = cur_test->out.get_not_quantized_tensor(temp_ref_container).data.capacity;

    temp_in1_container.mem.pi8 = g_scratch_mem_in1;
    temp_in2_container.mem.pi8 = g_scratch_mem_in2;
    temp_out_container.mem.pi8 = g_scratch_mem_out;
    temp_ref_container.mem.pi8 = g_scratch_mem_ref;

    mli_tensor temp_input1_tensor = cur_test->in1.get_quantized_tensor(temp_in1_container);
    mli_tensor temp_input2_tensor = cur_test->in2.get_quantized_tensor(temp_in2_container);
    mli_tensor temp_output_tensor = cur_test->out.get_not_quantized_tensor(temp_out_container);
    mli_tensor temp_ref_tensor = cur_test->out.get_not_quantized_tensor(temp_ref_container);

    temp_output_tensor.data.mem.pi8 = g_scratch_mem_out;
    test_metics.calculate_metrics(temp_output_tensor, cur_test->out);

    crc32_calc data_crc;
    data_crc(temp_input1_tensor);
    data_crc(temp_input2_tensor);
    data_crc(temp_output_tensor);
    is_test_passed = reporter->evaluate_and_report_case(cur_test->descr, test_metics,
                                                        cur_test->threshold, data_crc,
                                                        cur_test->check_sum);

    return is_test_passed;
}

int main() {
     const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI3.0|Kernels|MatMul Function Tests");

    for (int i = 0; i< kTestsNum; ++i) {

        bool is_test_passed = true;
        MatMul_test_operands* cur_test = &tests_list[i];

        // validate quantization
        if (!(cur_test->in1.is_valid() && cur_test->in2.is_valid() && cur_test->out.is_valid())) {
        reporter.report_message(
            cur_test->descr,
            "FAILED at init: Bad source data for one of tensors");
        is_test_passed = false;
        }

        /**************************************************************************************************************/
        void* MatMul_instance = nullptr;
        uint32_t MatMul_instance_size = 0;

        void* MatMul_conf_private = nullptr;
        uint32_t MatMul_conf_private_size = 0;

        lib_mli::TensorIterator<lib_mli::OffsetBuffer, kMatMulRank, kMatMulIterRank> in1_tensor_iter;
        lib_mli::TensorIterator<lib_mli::OffsetBuffer, kMatMulRank, kMatMulIterRank> in2_tensor_iter;
        lib_mli::TensorIterator<lib_mli::OffsetBuffer, kMatMulRank, kMatMulIterRank> out_tensor_iter;


        mli_data_container temp_in1_container{0};
        mli_data_container temp_in2_container{0};
        temp_in1_container.capacity = cur_test->in1.get_not_quantized_tensor(temp_in1_container).data.capacity;
        temp_in2_container.capacity = cur_test->in2.get_not_quantized_tensor(temp_in2_container).data.capacity;

        mli_tensor temp_input1_tensor = cur_test->in1.get_quantized_tensor(temp_in1_container);
        mli_tensor temp_input2_tensor = cur_test->in2.get_quantized_tensor(temp_in2_container);

        uint32_t input_tile_size[kMatMulRank] =  {1, temp_input1_tensor.shape[1]};
        uint32_t output_tile_size[kMatMulRank] =  {1, temp_input2_tensor.shape[1]};
        int32_t iteration_order[kMatMulRank] = {0, 1};
        uint32_t shape[kMatMulRank] = {temp_input1_tensor.shape[0], temp_input1_tensor.shape[1]};
        
        //tiling the Height
        assert(input_tile_size[1] == temp_input1_tensor.shape[1]);
        assert(output_tile_size[1] == temp_input2_tensor.shape[1]);

        // calculate number of tiles needed
        uint32_t num_tiles = 1;
        for(int i = 0; i < kMatMulRank; i++) {
            uint32_t tiles_per_dim = 1 + CEIL_DIV(shape[i] - input_tile_size[i], input_tile_size[i]);
            num_tiles *= tiles_per_dim;
        }


        /************ Prepare Phase *************/
        prepare_phase(cur_test, MatMul_instance, MatMul_instance_size, MatMul_conf_private, MatMul_conf_private_size,
                      in1_tensor_iter, in2_tensor_iter, out_tensor_iter, input_tile_size, output_tile_size, iteration_order);


        /************ Execution Phase *************/
        execution_phase(cur_test, MatMul_instance, MatMul_instance_size, MatMul_conf_private, MatMul_conf_private_size,
                        in1_tensor_iter, in2_tensor_iter, out_tensor_iter, num_tiles, iteration_order);

    
        /************ Postprocess Phase *************/
        final_status &= postprocess_phase(&reporter, cur_test);

    }

    reporter.report_outline("[AUTO] Group: mli_krn_MatMul_30", final_status);
    
    return 0;
}
