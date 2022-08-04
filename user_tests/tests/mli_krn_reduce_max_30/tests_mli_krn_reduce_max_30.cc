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
#include "mli_helpers_api.h"

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "test_report.h"
#include "test_tensor_quantizer.h"

#include "vectors_mli_krn_reduce_max.inc"

using mli::tst::crc32_calc;
using mli::tst::quality_metrics;
using mli::tst::reporter_full;
using mli::tst::tensor_quantizer;

namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;



struct reduce_max_test_operands {
    const char* descr;
    tensor_quantizer in;
    tensor_quantizer out;
    uint32_t data_size;
    const lib_mli::ReduceOpConfig cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
};

const crc32_calc test_1_chksum_sa8{ 0x38BABDA2 }, test_2_chksum_fx16{ 0x4EDABBB7 }, test_3_chksum_sa32{ 0x7EF74B21 },
    test_4_chksum_sa8{ 0x25791C37 }, test_5_chksum_fx16{ 0x79A4BD6D }, test_6_chksum_sa32{ 0xF699CD03 },
    test_7_chksum_sa8{ 0x741FACD2 }, test_8_chksum_fx16{ 0xC43D8A28 };

const quality_metrics thresholds_fx16_general{
    quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
    /* SNR_DB = */ 84.f, /*Quant Error Perc = */ 99.9f};
const quality_metrics thresholds_sa8_general{
    quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
    /* SNR_DB = */ 20.f, /*Quant Error Perc = */ 99.9f};
const quality_metrics thresholds_sa32_general{
    quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
    /* SNR_DB = */ 30.f, /*Quant Error Perc = */ 99.9f};

static const reduce_max_test_operands tests_list[] = {
    {"Test 1 SA8", input_1_sa8, test_1_out_sa8, sizeof(int8_t), test_1_cfg, thresholds_sa8_general, test_1_chksum_sa8},
    {"Test 2 FX16", input_1_fx16, test_2_out_fx16, sizeof(int16_t), test_2_cfg, thresholds_fx16_general, test_2_chksum_fx16},
    {"Test 3 SA32", input_2_sa32, test_3_out_sa32, sizeof(int32_t), test_3_cfg, thresholds_sa32_general, test_3_chksum_sa32},
    {"Test 4 SA8", input_2_sa8, test_4_out_sa8, sizeof(int8_t), test_4_cfg, thresholds_sa8_general, test_4_chksum_sa8},
    {"Test 5 FX16", input_3_fx16, test_5_out_fx16, sizeof(int16_t), test_1_cfg, thresholds_fx16_general, test_5_chksum_fx16},
    {"Test 6 SA32", input_3_sa32, test_6_out_sa32, sizeof(int32_t), test_2_cfg, thresholds_sa32_general, test_6_chksum_sa32},
    {"Test 7 SA8", input_4_sa8, test_7_out_sa8, sizeof(int8_t), test_1_cfg, thresholds_sa8_general, test_7_chksum_sa8},
    {"Test 8 FX16", input_4_fx16, test_8_out_fx16, sizeof(int16_t), test_3_cfg, thresholds_fx16_general, test_8_chksum_fx16},
};

constexpr uint32_t kMemSize = 4*2048;
static int8_t g_scratch_mem_in[kMemSize] = {0};
static int8_t g_scratch_mem_ref[kMemSize] = {0};
static int8_t g_scratch_mem_out[kMemSize] = {0};
static IO_DATA_ATTR int8_t g_mem_pool[kMemSize] = {0};

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

void prepare_phase(const reduce_max_test_operands* cur_test,
                   void*& reduce_max_instance,
                   uint32_t& reduce_max_instance_size,
                   lib_mli::PrivateData*& reduce_max_conf_private,
                   uint32_t& reduce_max_conf_private_size) {

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


    // STEP 1: Construct ReduceMax as a specific ExecutionInterface successor
    //==================================================================

    const lib_mli::Tensor<lib_mli::NoBuffer, 4> in_tensor(
        temp_input_tensor.shape, temp_input_tensor.mem_stride);
    const lib_mli::Tensor<lib_mli::NoBuffer, 4> out_tensor(
        temp_output_tensor.shape, temp_output_tensor.mem_stride);

    lib_mli::PlatformDescription pd;
    lib_ref::KernelsFactory kernel_factory(pd);
    uint32_t reduce_max_cs_size = kernel_factory.ReduceMax_CS_GetSize();
    void* reduce_max_cs_buffer = malloc(reduce_max_cs_size);
    auto reduce_max_op = kernel_factory.ReduceMax_CS(reduce_max_cs_buffer, in_tensor, cur_test->cfg, out_tensor);

    // STEP 2: Memory management (Up to user on how to deal with it)
    //==================================================================

    uint32_t offsets[1] = {0};

    uint32_t elem_size = cur_test->data_size;

    // Define buffers for in\out tensors
    // Leave space for runtime object
    uint32_t* offset = &offsets[0];
    uint32_t runtime_obj_size = reduce_max_op->GetRuntimeObjectSize();
    *offset += runtime_obj_size;

    // Leave space for private data buffer
    offset = &offsets[0];
    uint32_t private_buffer_size = reduce_max_op->GetKernelPrivateDataSize();
    *offset += private_buffer_size;

    // ReduceMax Input
    offset = &offsets[0];
    uint32_t in_size = reduce_max_op->GetInputBufferSize() * elem_size;
    lib_mli::OffsetBuffer reduce_max_in_buf{*offset, 0, in_size, elem_size};
    lib_mli::Tensor<lib_mli::OffsetBuffer, 4> reduce_max_in_tensor(reduce_max_in_buf, temp_input_tensor.shape);
    *offset += in_size;

    // ReduceMax Output
    offset = &offsets[0];
    uint32_t out_size = reduce_max_op->GetOutputBufferSize() * elem_size;
    lib_mli::OffsetBuffer reduce_max_out_buf{*offset, 0, out_size, elem_size};
    lib_mli::Tensor<lib_mli::OffsetBuffer, 4> reduce_max_out_tensor(reduce_max_out_buf, temp_output_tensor.shape);
    *offset += out_size;

    // DataBuffer size is 0 for reference kernel
    offset = &offsets[0];
    uint32_t data_buffer_size = reduce_max_op->GetDataBufferSize();
    lib_mli::OffsetBuffer reduce_max_descr_buf{*offset, 0, data_buffer_size,
                                                sizeof(char)};
    *offset += data_buffer_size;

    // Attaching buffer (descriptors) to the operation
    mli_status status = MLI_STATUS_OK;

    status = reduce_max_op->AttachBufferOffsets(reduce_max_in_tensor, reduce_max_out_tensor,
                                                reduce_max_descr_buf);
    assert(status == MLI_STATUS_OK);

    reduce_max_instance = (int8_t*)g_mem_pool;
    reduce_max_instance_size = reduce_max_op->GetRuntimeObjectSize();

    status =
        reduce_max_op->GetKernelPrivateData((int8_t*)g_mem_pool + reduce_max_instance_size);
    assert(status == MLI_STATUS_OK);
    reduce_max_conf_private = reinterpret_cast<lib_mli::PrivateData*>(
        (int8_t*)g_mem_pool + reduce_max_instance_size);
    reduce_max_conf_private_size = reduce_max_op->GetKernelPrivateDataSize();

}

template <typename T>
static void strided_copy_with_offsets(const T * src, const uint32_t src_offsets[4], const uint32_t dst_offsets[4], const uint32_t strides[4], const uint32_t size[4], T * dst) {
    uint32_t src_ind = 0, dst_ind = 0;
    for (uint32_t b = 0; b < size[0]; b++) {
        for (uint32_t h = 0; h < size[1]; h++) {
            for (uint32_t w = 0; w < size[2]; w++) {
                for (uint32_t c = 0; c < size[3]; c++) {
                    src_ind = (c + src_offsets[3]) * strides[3] + (w + src_offsets[2]) * strides[2] + (h + src_offsets[1])  * strides[1] + (b + src_offsets[0])  * strides[0];
                    dst_ind = (c + dst_offsets[3]) * strides[3] + (w + dst_offsets[2]) * strides[2] + (h + dst_offsets[1])  * strides[1] + (b + dst_offsets[0])  * strides[0];
                    dst[dst_ind] = src[src_ind];
                }
            }
        }
    }
}

void execution_phase(void* reduce_max_instance, uint32_t reduce_max_instance_size,
                     lib_mli::PrivateData* reduce_max_conf_private, uint32_t reduce_max_conf_private_size) {
    // STEP 3: Execution phase
    //==================================================================
    
    uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool)};

    auto reduce_max_run_op = lib_mli::ExecutionInterface::Create(reduce_max_instance,
                                                                 reduce_max_instance_size,
                                                                 reduce_max_conf_private,
                                                                 reduce_max_conf_private_size,
                                                                 membasis, sizeof(membasis) / sizeof(membasis[0]));
    assert(reduce_max_run_op != nullptr);
    mli_status status = MLI_STATUS_OK;

    lib_ref::ReduceMaxPrivateData* reduce_max_private = (lib_ref::ReduceMaxPrivateData*)(reduce_max_conf_private);

    status = reduce_max_run_op->Prefetch();
    assert(status == MLI_STATUS_OK);

    if(1 == reduce_max_private->input.get_buf().get_elem_size()) {
        int8_t* temp_in_mem = (int8_t*)g_scratch_mem_in;
        int8_t* dst_in_buffer = (int8_t*) (g_mem_pool + reduce_max_private->input.get_buf().get_offset());
        int num_in_elem = reduce_max_private->input.get_buf().get_size() / reduce_max_private->input.get_buf().get_elem_size();
        for (int idx = 0; idx < num_in_elem; idx++) {
            dst_in_buffer[idx] = temp_in_mem[idx];
        }
    }
    else if(2 == reduce_max_private->input.get_buf().get_elem_size()) {
        int16_t* temp_in_mem = (int16_t*)g_scratch_mem_in;
        int16_t* dst_in_buffer = (int16_t*) (g_mem_pool + reduce_max_private->input.get_buf().get_offset());
        int num_in_elem = reduce_max_private->input.get_buf().get_size() / reduce_max_private->input.get_buf().get_elem_size();
        for (int idx = 0; idx < num_in_elem; idx++) {
            dst_in_buffer[idx] = temp_in_mem[idx];
        }
    }
    else if(4 == reduce_max_private->input.get_buf().get_elem_size()) {
        int32_t* temp_in_mem = (int32_t*)g_scratch_mem_in;
        int32_t* dst_in_buffer = (int32_t*) (g_mem_pool + reduce_max_private->input.get_buf().get_offset());
        int num_in_elem = reduce_max_private->input.get_buf().get_size() / reduce_max_private->input.get_buf().get_elem_size();
        for (int idx = 0; idx < num_in_elem; idx++) {
            dst_in_buffer[idx] = temp_in_mem[idx];
        }
    }
    

    status = reduce_max_run_op->Issue();
    assert(status == MLI_STATUS_OK);

    if(1 == reduce_max_private->output.get_buf().get_elem_size()) {
        int8_t* temp_out_mem = (int8_t*)g_scratch_mem_out;
        int8_t* src_out_buffer = (int8_t*) (g_mem_pool + reduce_max_private->output.get_buf().get_offset());
        int num_out_elem = reduce_max_private->output.get_buf().get_size() / reduce_max_private->output.get_buf().get_elem_size();
        for (int idx = 0; idx < num_out_elem; idx++) {
            temp_out_mem[idx] = src_out_buffer[idx];
        }
    }
    else if(2 == reduce_max_private->output.get_buf().get_elem_size()) {
        int16_t* temp_out_mem = (int16_t*)g_scratch_mem_out;
        int16_t* src_out_buffer = (int16_t*) (g_mem_pool + reduce_max_private->output.get_buf().get_offset());
        int num_out_elem = reduce_max_private->output.get_buf().get_size() / reduce_max_private->output.get_buf().get_elem_size();
        for (int idx = 0; idx < num_out_elem; idx++) {
            temp_out_mem[idx] = src_out_buffer[idx];
        }
    }
    else if(4 == reduce_max_private->output.get_buf().get_elem_size()) {
        int32_t* temp_out_mem = (int32_t*)g_scratch_mem_out;
        int32_t* src_out_buffer = (int32_t*) (g_mem_pool + reduce_max_private->output.get_buf().get_offset());
        int num_out_elem = reduce_max_private->output.get_buf().get_size() / reduce_max_private->output.get_buf().get_elem_size();
        for (int idx = 0; idx < num_out_elem; idx++) {
            temp_out_mem[idx] = src_out_buffer[idx];
        }
    }
    

    status = reduce_max_run_op->Update();
    assert(status == MLI_STATUS_OK);

}

bool postprocess_phase(const reporter_full* reporter,
                       const reduce_max_test_operands* cur_test) {
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

    reporter.report_header("MLI3.0|Kernels|Reduce Max Function Tests");

    for (int i = 0; i< kTestsNum; ++i) {

        bool is_test_passed = true;
        const reduce_max_test_operands* cur_test = &tests_list[i];

        // validate quantization
        if (!(cur_test->in.is_valid() && cur_test->out.is_valid())) {
        reporter.report_message(
            cur_test->descr,
            "FAILED at init: Bad source data for one of tensors");
        is_test_passed = false;
        }

        /**************************************************************************************************************/
        // auto cfg = cur_test->cfg;
        // const KernelInfo kernel_info{
        //     cfg.kernel_size[0], cfg.kernel_size[1], cfg.stride[0], cfg.stride[1], 1, 1,
        //     cfg.padding_begin[0], cfg.padding_end[0], cfg.padding_begin[1], cfg.padding_end[1]
        // };
        // mli_data_container temp_container{ 0 };
        // mli_tensor temp_input_tensor = cur_test->in.get_not_quantized_tensor(temp_container);
        // uint32_t input_shape[4]{};
        // hwc_to_bhwc(temp_input_tensor.shape, input_shape);
        // uint32_t tile_input_size[4]{ 1, 4, 4, input_shape[3] };
        // Tiling tiling(input_shape, tile_input_size, kernel_info);
        /**************************************************************************************************************/

        void* reduce_max_instance = nullptr;
        uint32_t reduce_max_instance_size = 0;

        lib_mli::PrivateData* reduce_max_conf_private = nullptr;
        uint32_t reduce_max_conf_private_size = 0;
        /**************************************************************************************************************/

        /************ Prepare Phase *************/
        prepare_phase(cur_test, reduce_max_instance, reduce_max_instance_size,
                      reduce_max_conf_private, reduce_max_conf_private_size);



        /************ Execution Phase *************/
        execution_phase(reduce_max_instance, reduce_max_instance_size,
                        reduce_max_conf_private, reduce_max_conf_private_size);



        /************ Postprocess Phase *************/
        final_status &= postprocess_phase(&reporter, cur_test);



    }
    

    reporter.report_outline("[AUTO] Group: mli_krn_reduce_max_30", final_status);
    
    return 0;
}
