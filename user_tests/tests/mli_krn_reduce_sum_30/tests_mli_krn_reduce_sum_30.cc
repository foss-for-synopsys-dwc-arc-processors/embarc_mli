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
#include <iostream>

#include "mli_api.h"
#include "mli_config.h"
#include "mli_types.h"
#include "mli_types.hpp"
#include "mli_kernels_factory_ref.hpp"
#include "mli_ref_private_types.hpp"
#include "mli_runtime_api.hpp"
#include "mli_private_types.h"
#include "mli_ref_runtime_api.hpp"
#include "mli_helpers_api.h"

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "test_report.h"
#include "test_tensor_quantizer.h"
#include "test_tiling.hpp"

#include "vectors_mli_krn_reduce_sum.inc"

using mli::tst::crc32_calc;
using mli::tst::quality_metrics;
using mli::tst::reporter_full;
using mli::tst::tensor_quantizer;

namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;

using lib_mli::kReduceSumRank;
using lib_mli::kReduceSumIterRank;

struct reduce_sum_test_operands {
    const char* descr;
    tensor_quantizer in;
    tensor_quantizer out;
    uint32_t data_size;
    const lib_mli::ReduceOpConfig cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

const crc32_calc test_1_chksum_int{ 0x1C392777 }, test_2_chksum_int{ 0xA05D9BDD }, test_3_chksum_int{ 0xF9C52E4A },
    test_4_chksum_int{ 0xB5CA1CC0 }, test_5_chksum_int{ 0x8ED095EB }, test_6_chksum_int{ 0x0852577B },
    test_7_chksum_int{ 0xFD3D4B6C }, test_8_chksum_int{ 0x546091D5 };

const quality_metrics thresholds_fx16_general{
    quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
    /* SNR_DB = */ 84.f, /*Quant Error Perc = */ 99.9f};
const quality_metrics thresholds_sa8_general{
    quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
    /* SNR_DB = */ 20.f, /*Quant Error Perc = */ 99.9f};
const quality_metrics thresholds_sa32_general{
    quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
    /* SNR_DB = */ 20.f, /*Quant Error Perc = */ 20.0f};
const quality_metrics thresholds_int_general{
    quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
    /* SNR_DB = */ 84.f, /*Quant Error Perc = */ 0.0f};

// test vectors are all integers data
static const reduce_sum_test_operands tests_list[] = {
    {"Test 1 int", input_1_sa8, test_1_out_sa8, sizeof(int8_t), test_1_cfg, thresholds_int_general, test_1_chksum_int},
    {"Test 2 int", input_1_fx16, test_2_out_fx16, sizeof(int16_t), test_2_cfg, thresholds_int_general, test_2_chksum_int},
    {"Test 3 int", input_2_sa32, test_3_out_sa32, sizeof(int32_t), test_3_cfg, thresholds_int_general, test_3_chksum_int},
    {"Test 4 int", input_2_sa8, test_4_out_sa8, sizeof(int8_t), test_4_cfg, thresholds_int_general, test_4_chksum_int},
    {"Test 5 int", input_3_fx16, test_5_out_fx16, sizeof(int16_t), test_1_cfg, thresholds_int_general, test_5_chksum_int},
    {"Test 6 int", input_3_sa32, test_6_out_sa32, sizeof(int32_t), test_2_cfg, thresholds_int_general, test_6_chksum_int},
    {"Test 7 int", input_4_sa8, test_7_out_sa8, sizeof(int8_t), test_1_cfg, thresholds_int_general, test_7_chksum_int},
    {"Test 8 int", input_4_sa32, test_8_out_sa32, sizeof(int32_t), test_3_cfg, thresholds_int_general, test_8_chksum_int},
};

constexpr uint32_t kMemSize = 4*2048;
static int8_t g_scratch_mem_in[kMemSize] = {0};
static int8_t g_scratch_mem_ref[kMemSize] = {0};
static int8_t g_scratch_mem_out[kMemSize] = {0};
static IO_DATA_ATTR int8_t g_mem_pool[kMemSize] = {0};

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

void prepare_phase(const reduce_sum_test_operands* cur_test,
                   int32_t iteration_order[kReduceSumIterRank],
                   uint32_t input_tile_size[kReduceSumIterRank],
                   uint32_t output_tile_size[kReduceSumIterRank],
                   void*& reduce_sum_instance,
                   uint32_t& reduce_sum_instance_size,
                   lib_mli::PrivateData*& reduce_sum_conf_private,
                   uint32_t& reduce_sum_conf_private_size) {

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
        cur_test->out.get_quantized_tensor(temp_out_container);


    // STEP 1: Construct ReduceSum as a specific ExecutionInterface successor
    //==================================================================

    for (uint32_t i = 0; i<kReduceSumRank; i++){
        // in case of input rank less than 4
        if (temp_input_tensor.shape[i] == 0) {
            temp_input_tensor.shape[i] = 1;
            temp_output_tensor.shape[i] = 1;
        }
    }
    
    const lib_mli::Tensor<lib_mli::NoBuffer, kReduceSumRank> in_tensor(
        temp_input_tensor.shape, temp_input_tensor.mem_stride);
    const lib_mli::Tensor<lib_mli::NoBuffer, kReduceSumRank> out_tensor(
        temp_output_tensor.shape, temp_output_tensor.mem_stride);

    lib_mli::PlatformDescription pd;
    lib_ref::KernelsFactory kernel_factory(pd);
    uint32_t reduce_sum_cs_size = kernel_factory.ReduceSum_CS_GetSize();
    void* reduce_sum_cs_buffer = malloc(reduce_sum_cs_size);

    lib_mli::IteratorCfg<kReduceSumIterRank> input_it_config(in_tensor, input_tile_size, iteration_order);
    lib_mli::IteratorCfg<kReduceSumIterRank> output_it_config(out_tensor, output_tile_size, iteration_order);

    lib_mli::Tensor<lib_mli::NoBuffer, kReduceSumRank> full_in_tensor(temp_input_tensor.shape, temp_input_tensor.mem_stride);
    lib_mli::TensorIterator<lib_mli::NoBuffer, kReduceSumRank, kReduceSumIterRank> in_tensor_it(full_in_tensor, input_it_config);
    lib_mli::Tensor<lib_mli::NoBuffer, kReduceSumRank> full_out_tensor(temp_output_tensor.shape, temp_output_tensor.mem_stride);
    lib_mli::TensorIterator<lib_mli::NoBuffer, kReduceSumRank, kReduceSumIterRank> out_tensor_it(full_out_tensor, output_it_config);

    auto reduce_sum_op = kernel_factory.ReduceSum_CS(reduce_sum_cs_buffer, in_tensor_it, cur_test->cfg, out_tensor_it);

    // STEP 2: Memory management (Up to user on how to deal with it)
    //==================================================================
    mli_status status = MLI_STATUS_OK;
    uint32_t offsets[1] = {0};

    uint32_t elem_size = cur_test->data_size;

    // Define buffers for in\out tensors
    // Leave space for runtime object
    uint32_t* offset = &offsets[0];
    uint32_t runtime_obj_size = reduce_sum_op->GetRuntimeObjectSize();
    *offset += runtime_obj_size;

    // Leave space for private data buffer
    offset = &offsets[0];
    uint32_t private_buffer_size = reduce_sum_op->GetKernelPrivateDataSize();
    *offset += private_buffer_size;

    // ReduceSum Input
    offset = &offsets[0];
    uint32_t in_size = lib_mli::service::GetBufferSize(kReduceSumRank, input_tile_size, temp_input_tensor.mem_stride) * elem_size;
    lib_mli::OffsetBuffer reduce_sum_in_buf{*offset, 0, in_size, elem_size};
    *offset += in_size;

    // ReduceSum Output
    offset = &offsets[0];
    uint32_t out_size = lib_mli::service::GetBufferSize(kReduceSumRank, output_tile_size, temp_output_tensor.mem_stride) * elem_size; //(out_tensor.get_rank(), temp_output_tensor.shape, temp_output_tensor.mem_stride) * elem_size;
    lib_mli::OffsetBuffer reduce_sum_out_buf{*offset, 0, out_size, elem_size};
    *offset += out_size;

    // DataBuffer size is 0 for reference kernel
    offset = &offsets[0];
    uint32_t ctrl_buffer_size = reduce_sum_op->GetCtrlBufferSize();
    lib_mli::OffsetBuffer reduce_sum_ctrl_buf{*offset, 0, ctrl_buffer_size, sizeof(char)};
    *offset += ctrl_buffer_size;

    // Attaching buffer (descriptors) to the operation

    status = reduce_sum_op->AttachBufferOffsets(reduce_sum_in_buf, reduce_sum_out_buf,
                                                reduce_sum_ctrl_buf);
    assert(status == MLI_STATUS_OK);

    reduce_sum_instance = (int8_t*)g_mem_pool;
    reduce_sum_instance_size = reduce_sum_op->GetRuntimeObjectSize();

    status =
        reduce_sum_op->GetKernelPrivateData((int8_t*)g_mem_pool + reduce_sum_instance_size);
    assert(status == MLI_STATUS_OK);
    reduce_sum_conf_private = reinterpret_cast<lib_mli::PrivateData*>(
        (int8_t*)g_mem_pool + reduce_sum_instance_size);
    reduce_sum_conf_private_size = reduce_sum_op->GetKernelPrivateDataSize();

}


void execution_phase(uint32_t num_tiles,
                     void* reduce_sum_instance, uint32_t reduce_sum_instance_size,
                     lib_mli::PrivateData* reduce_sum_conf_private, uint32_t reduce_sum_conf_private_size) {
    // STEP 3: Execution phase
    //==================================================================
    
    uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool)};

    auto reduce_sum_run_op = lib_mli::ExecutionInterface::Create(reduce_sum_instance,
                                                                 reduce_sum_instance_size,
                                                                 reduce_sum_conf_private,
                                                                 reduce_sum_conf_private_size,
                                                                 membasis, sizeof(membasis) / sizeof(membasis[0]));
    assert(reduce_sum_run_op != nullptr);
    mli_status status = MLI_STATUS_OK;

    lib_ref::ReduceSumPrivateData* reduce_sum_private = (lib_ref::ReduceSumPrivateData*)(reduce_sum_conf_private);

    int32_t tile_input_strides[kReduceSumRank];
    int32_t tile_output_strides[kReduceSumRank];
    reduce_sum_private->input.get_mem_strides(tile_input_strides);
    reduce_sum_private->output.get_mem_strides(tile_output_strides);

    lib_ref::ReduceSum* reduceSum_pimpl  = dynamic_cast<lib_ref::ReduceSum*>(reduce_sum_run_op);
    uint32_t input_tile_size[kReduceSumRank];
    uint32_t output_tile_size[kReduceSumRank];
    int32_t input_tile_offsets[kReduceSumRank];
    int32_t output_tile_offsets[kReduceSumRank];
    const int32_t zero_offsets[kReduceSumRank]{};
    
    for(uint32_t n_tile = 0; n_tile < num_tiles; n_tile++) {
        
        status = reduce_sum_run_op->Prefetch();
        assert(status == MLI_STATUS_OK);

        reduceSum_pimpl->GetIOSizesAndOffsets(input_tile_size, output_tile_size,
                                              input_tile_offsets, output_tile_offsets);

        // copy input from global buffer to local tile buffer
        strided_copy_with_offsets(kReduceSumRank, reduce_sum_private->input.get_buf().get_elem_size(),
                                  g_scratch_mem_in, input_tile_offsets, zero_offsets, tile_input_strides,
                                  input_tile_size, (int8_t*) (g_mem_pool + reduce_sum_private->input.get_buf().get_offset()) );

        
        status = reduce_sum_run_op->Issue();
        assert(status == MLI_STATUS_OK);

        // copy output from local tile buffer to global buffer
        strided_copy_with_offsets(kReduceSumRank, reduce_sum_private->input.get_buf().get_elem_size(),
                                  (int8_t*)(g_mem_pool + reduce_sum_private->output.get_buf().get_offset()),
                                  zero_offsets, output_tile_offsets, tile_output_strides,
                                  output_tile_size, (int8_t*) g_scratch_mem_out);

        status = reduce_sum_run_op->Update();
        assert(status == MLI_STATUS_OK);
    }

}

bool postprocess_phase(const reporter_full* reporter,
                       const reduce_sum_test_operands* cur_test) {
    
    //////////////// for debugging ////////////////
    // int16_t* temp_in_mem = (int16_t*)g_scratch_mem_in;
    // mli_tensor temp_tensor = cur_test->in.get_source_float_tensor();
    // int num_in_elem = mli_hlp_count_elem_num(&temp_tensor, 0);
    // printf("input data \n");
    // for (int idx = 0; idx < num_in_elem; idx++) {
    //     if (idx % 15 == 0) {
    //         printf("\n");
    //         if (idx % 4 == 0) {
    //         printf("\n");
    //         }
    //     }
        
    //     printf("%d\t", temp_in_mem[idx]);
        
    // }
    // printf("\n________________________\n");



    // int16_t* temp_out_mem = (int16_t*)g_scratch_mem_out;
    // temp_tensor = cur_test->out.get_source_float_tensor();
    // int num_out_elem = mli_hlp_count_elem_num(&temp_tensor, 0);
    // printf("kernel output data \n");
    // for (int idx = 0; idx < num_out_elem; idx++) {
    //     printf("%d\t", temp_out_mem[idx]);
    // }
    // printf("\n________________________\n");


    // temp_out_mem = (int16_t*)g_scratch_mem_ref;
    // temp_tensor = cur_test->out.get_source_float_tensor();
    // num_out_elem = mli_hlp_count_elem_num(&temp_tensor, 0);
    // printf("refernece output data \n");
    // for (int idx = 0; idx < num_out_elem; idx++) {
    //     printf("%d\t", temp_out_mem[idx]);
    // }
    // printf("\n________________________\n");

    ////////////////// end of debugging part ///////////////
    
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

int main(){

    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI3.0|Kernels|Reduce Sum Function Tests");
    int32_t iteration_order[kReduceSumIterRank]{ 0, 1, 2, 3 };

    for(int i = 0; i< kTestsNum; ++i) {

        bool is_test_passed = true;
        const reduce_sum_test_operands* cur_test = &tests_list[i];

        // validate quantization
        if(!(cur_test->in.is_valid() && cur_test->out.is_valid())) {
        reporter.report_message(
            cur_test->descr,
            "FAILED at init: Bad source data for one of tensors");
        is_test_passed = false;
        }

        uint32_t input_tile_size[kReduceSumRank] = {1, 4, 4, 3};
        uint32_t output_tile_size[kReduceSumRank] = {1, 4, 4, 3};

        mli_data_container temp_container{ 0 };
        mli_tensor temp_input_tensor = cur_test->in.get_not_quantized_tensor(temp_container);

        for(uint8_t i = 0; i < kReduceSumRank; i++) {
            // in case of input rank less than 4
            if(temp_input_tensor.shape[i] == 0) {
                temp_input_tensor.shape[i] = 1;
                input_tile_size[i] = 1;
                output_tile_size[i] = 1;
            }
            // no tiling along the reduce axis (make the tiling size same as tensor size for the axis of reduction)
            if(cur_test->cfg.axis == i) {
                input_tile_size[i] = temp_input_tensor.shape[i];
                output_tile_size[i] = 1;
            }
        }

        // calculate number of tiles needed
        uint32_t num_tiles = 1;
        for(int i = 0; i < 4; i++) {
            uint32_t tiles_per_dim = 1 + CEIL_DIV(temp_input_tensor.shape[i] - input_tile_size[i], input_tile_size[i]);
            num_tiles *= tiles_per_dim;
        }


        void* reduce_sum_instance = nullptr;
        uint32_t reduce_sum_instance_size = 0;

        lib_mli::PrivateData* reduce_sum_conf_private = nullptr;
        uint32_t reduce_sum_conf_private_size = 0;
        /**************************************************************************************************************/

        /************ Prepare Phase *************/
        prepare_phase(cur_test, iteration_order, input_tile_size, output_tile_size,
                      reduce_sum_instance, reduce_sum_instance_size,
                      reduce_sum_conf_private, reduce_sum_conf_private_size);



        /************ Execution Phase *************/
        execution_phase(num_tiles,
                        reduce_sum_instance, reduce_sum_instance_size,
                        reduce_sum_conf_private, reduce_sum_conf_private_size);



        /************ Postprocess Phase *************/
        final_status &= postprocess_phase(&reporter, cur_test);



    }
    

    reporter.report_outline("[AUTO] Group: mli_krn_reduce_sum_30", final_status);
    
    return 0;
}
