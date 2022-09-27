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
#include "mli_private_types.h"
#include "mli_ref_runtime_api.hpp"
#include "mli_runtime_api.hpp"
#include "mli_helpers_api.h"

#include "test_crc32_calc.h"
#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "test_report.h"
#include "test_tensor_quantizer.h"
#include "test_tiling.hpp"

#include "vectors_mli_krn_move_broadcast_30.inc"

namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;

using mli::tst::crc32_calc;
using mli::tst::quality_metrics;
using mli::tst::reporter_full;
using mli::tst::tensor_quantizer;

using lib_mli::kMoveBroadcastRank;
using lib_mli::kMoveBroadcastIterRank;

struct move_broadcast_test_operands {
    const char* descr;
    tensor_quantizer in;
    tensor_quantizer out;
    uint32_t data_size;
    const quality_metrics threshold;
    const crc32_calc check_sum;
    uint32_t src_offsets;
    uint32_t dst_offsets;
    uint32_t src_size;
    uint32_t dst_size;
};

const crc32_calc test_1_chksum_fx16{ 0x942C9DC4 }, test_2_chksum_sa32{ 0x4C375B1B }, test_3_chksum_sa8{ 0xB75ADC0E },
                 test_5_chksum_fx16{ 0x1D310788 }, test_4_chksum_sa32{ 0xD7A3F343 }, test_7_chksum_sa8{ 0x5D9077C5 },
                 test_8_chksum_fx16{ 0xEDFC03D6 }, test_6_chksum_sa32{ 0x41CBCEF6 }; 

const quality_metrics thresholds_fx16_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                               /* SNR_DB = */ 84.f, /*Quant Error Perc = */ 0.0f};
const quality_metrics thresholds_sa8_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                              /* SNR_DB = */ 20.f, /*Quant Error Perc = */ 0.0f};
const quality_metrics thresholds_sa32_general{ quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                               /* SNR_DB = */ 30.f, /*Quant Error Perc = */ 0.0f};

static move_broadcast_test_operands tests_list[] = {
    {"Test 1 FX16 Broadcast 2-axis", input_1_fx16, test_1_out_fx16, sizeof(int16_t), thresholds_fx16_general, test_1_chksum_fx16},   
    {"Test 2 SA32 MoveBroadcast", input_1_sa32, test_2_out_sa32, sizeof(int32_t), thresholds_sa32_general, test_2_chksum_sa32},   
    {"Test 3 SA8  MoveBroadcast", input_2_sa8, test_3_out_sa8, sizeof(int8_t), thresholds_sa8_general, test_3_chksum_sa8},         
    {"Test 4 SA32 Move", input_2_sa32, test_4_out_sa32, sizeof(int32_t), thresholds_sa32_general, test_4_chksum_sa32},   
    {"Test 5 FX16 MoveBroadcast", input_3_fx16, test_5_out_fx16, sizeof(int16_t), thresholds_fx16_general, test_5_chksum_fx16},   
    {"Test 6 SA32 Move", input_3_sa32, test_6_out_sa32, sizeof(int32_t), thresholds_sa32_general, test_6_chksum_sa32},  
    {"Test 7 SA8  Move rank<max", input_4_sa8, test_7_out_sa8, sizeof(int8_t), thresholds_sa8_general, test_7_chksum_sa8},         
    {"Test 8 FX16 Broadcast rank<max", input_4_fx16, test_8_out_fx16, sizeof(int16_t), thresholds_fx16_general, test_8_chksum_fx16},   
};

constexpr uint32_t kMemSize = 4*2048;
static int8_t g_scratch_mem_in[kMemSize] = {0};
static int8_t g_scratch_mem_ref[kMemSize] = {0};
static int8_t g_scratch_mem_out[kMemSize] = {0};
static IO_DATA_ATTR int8_t g_mem_pool[kMemSize] = {0};

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

/**************************************************************************************************************/

void prepare_phase(move_broadcast_test_operands* cur_test,
                   void*& move_broadcast_instance,
                   uint32_t& move_broadcast_instance_size,
                   void*& move_broadcast_conf_private,
                   uint32_t& move_broadcast_conf_private_size) {

    mli_data_container temp_in_container{0};
    mli_data_container temp_out_container{0};
    temp_in_container.capacity = cur_test->in.get_not_quantized_tensor(temp_in_container).data.capacity;
    temp_in_container.mem.pi8 = g_scratch_mem_in;
    temp_out_container.capacity = cur_test->out.get_not_quantized_tensor(temp_out_container).data.capacity;
    temp_out_container.mem.pi8 = g_scratch_mem_ref;

    // STEP 1: Construct MoveBroadcast as a specific ExecutionInterface successor
    //==================================================================

    mli_tensor temp_input_tensor = cur_test->in.get_quantized_tensor(temp_in_container);
    mli_tensor temp_output_tensor = cur_test->out.get_quantized_tensor(temp_out_container);

    // input rank less than Max_rank handling
    // ToDo: when mli_tensor takes [rank=5] -> change condition to iterate to (kMoveBroadcastRank) instead of (kMoveBroadcastRank-1)
    for(uint32_t i = 0; i < kMoveBroadcastRank-1; i++) {
        if (temp_input_tensor.shape[i] == 0) {
            temp_input_tensor.shape[i] = 1;
        }
        if (temp_output_tensor.shape[i] == 0) {
            temp_output_tensor.shape[i] = 1;
        }
    }

    lib_mli::MoveDataDirection data_dir = lib_mli::MoveDataDirection::kMoveDataDirectionInput;

    const lib_mli::Tensor<lib_mli::NoBuffer, kMoveBroadcastRank> in_tensor(temp_input_tensor.shape, temp_input_tensor.mem_stride);
    const lib_mli::Tensor<lib_mli::NoBuffer, kMoveBroadcastRank> out_tensor(temp_output_tensor.shape, temp_output_tensor.mem_stride);

    lib_mli::TensorIterator<lib_mli::NoBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank> in_tensor_it(in_tensor);
    lib_mli::TensorIterator<lib_mli::NoBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank> out_tensor_it(out_tensor);

    lib_mli::PlatformDescription pd;
    lib_ref::KernelsFactory kernel_factory(pd);
    uint32_t move_broadcast_cs_size = kernel_factory.MoveBroadcast_CS_GetSize();
    void* move_broadcast_cs_buffer = malloc(move_broadcast_cs_size);
    auto move_broadcast_op = kernel_factory.MoveBroadcast_CS(move_broadcast_cs_buffer, in_tensor_it, out_tensor_it, data_dir);

    // STEP 2: Memory management (Up to user on how to deal with it)
    //==================================================================

    uint32_t offsets[1] = {0};
    uint32_t elem_size = cur_test->data_size;
    mli_status status = MLI_STATUS_OK;

    // Define buffers for in/out tensors
    // Leave space for runtime object
    uint32_t* offset = &offsets[0];
    uint32_t runtime_obj_size = move_broadcast_op->GetRuntimeObjectSize();
    *offset += runtime_obj_size;

    // Leave space for private data buffer
    offset = &offsets[0];
    uint32_t private_buffer_size = move_broadcast_op->GetKernelPrivateDataSize();
    *offset += private_buffer_size;

    // MoveBroadcast Input
    offset = &offsets[0];
    cur_test->src_offsets = *offset;
    uint32_t in_size = lib_mli::service::GetBufferSize(lib_mli::kMoveBroadcastRank, temp_input_tensor.shape, temp_input_tensor.mem_stride) * elem_size;
    cur_test->src_size = in_size;
    lib_mli::OffsetBuffer move_broadcast_in_buf{*offset, 0, in_size, elem_size};
    *offset += in_size;

    // MoveBroadcast Output
    offset = &offsets[0];
    cur_test->dst_offsets = *offset;
    uint32_t out_size = lib_mli::service::GetBufferSize(lib_mli::kMoveBroadcastRank, temp_output_tensor.shape, temp_output_tensor.mem_stride) * elem_size;
    cur_test->dst_size = out_size;
    lib_mli::OffsetBuffer move_broadcast_out_buf{*offset, 0, out_size, elem_size};
    *offset += out_size;

    // CtrlBuffer size is 0 for reference kernel
    offset = &offsets[0];
    uint32_t ctrl_buffer_size = move_broadcast_op->GetCtrlBufferSize();
    lib_mli::OffsetBuffer move_broadcast_ctrl_buf{*offset, 0, ctrl_buffer_size, sizeof(char)};
    *offset += ctrl_buffer_size;

    // Attaching buffer (descriptors) to the operation
    status = move_broadcast_op->AttachBufferOffsets(move_broadcast_in_buf, 
                                                    move_broadcast_out_buf,
                                                    move_broadcast_ctrl_buf);
    assert(status == MLI_STATUS_OK);

    move_broadcast_instance = (int8_t*)g_mem_pool;
    move_broadcast_instance_size = move_broadcast_op->GetRuntimeObjectSize();

    status = move_broadcast_op->GetKernelPrivateData((int8_t*)g_mem_pool + move_broadcast_instance_size);
    assert(status == MLI_STATUS_OK);
    move_broadcast_conf_private = (int8_t*)g_mem_pool + move_broadcast_instance_size;
    move_broadcast_conf_private_size = move_broadcast_op->GetKernelPrivateDataSize();

}

/**************************************************************************************************************/

void execution_phase(const move_broadcast_test_operands* cur_test, 
                     void* move_broadcast_instance, 
                     uint32_t move_broadcast_instance_size,
                     void* move_broadcast_conf_private, 
                     uint32_t move_broadcast_conf_private_size) {
    // STEP 3: Execution phase
    //==================================================================
    
    uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool)};

    auto move_broadcast_run_op = lib_mli::ExecutionInterface::Create(move_broadcast_instance,
                                                                     move_broadcast_instance_size,
                                                                     move_broadcast_conf_private,
                                                                     move_broadcast_conf_private_size,
                                                                     membasis, sizeof(membasis) / sizeof(membasis[0]));
    assert(move_broadcast_run_op != nullptr);
    mli_status status = MLI_STATUS_OK;

    status = move_broadcast_run_op->Prefetch();
    assert(status == MLI_STATUS_OK);

    if(sizeof(int8_t) == cur_test->data_size) {
        int8_t* temp_in_mem = (int8_t*)g_scratch_mem_in;
        int8_t* dst_in_buffer = (int8_t*) (g_mem_pool + cur_test->src_offsets);
        int num_in_elem = cur_test->src_size / cur_test->data_size;
        for (int idx = 0; idx < num_in_elem; idx++) {
            dst_in_buffer[idx] = temp_in_mem[idx];
        }
    }
    else if(sizeof(int16_t) == cur_test->data_size) {
        int16_t* temp_in_mem = (int16_t*)g_scratch_mem_in;
        int16_t* dst_in_buffer = (int16_t*) (g_mem_pool + cur_test->src_offsets);
        int num_in_elem = cur_test->src_size / cur_test->data_size;
        for (int idx = 0; idx < num_in_elem; idx++) {
            dst_in_buffer[idx] = temp_in_mem[idx];
        }
    }
    else if(sizeof(int32_t) == cur_test->data_size) {
        int32_t* temp_in_mem = (int32_t*)g_scratch_mem_in;
        int32_t* dst_in_buffer = (int32_t*) (g_mem_pool + cur_test->src_offsets);
        int num_in_elem = cur_test->src_size / cur_test->data_size;
        for (int idx = 0; idx < num_in_elem; idx++) {
            dst_in_buffer[idx] = temp_in_mem[idx];
        }
    }
    

    status = move_broadcast_run_op->Issue();
    assert(status == MLI_STATUS_OK);

    if(sizeof(int8_t) == cur_test->data_size) {
        int8_t* temp_out_mem = (int8_t*)g_scratch_mem_out;
        int8_t* src_out_buffer = (int8_t*) (g_mem_pool + cur_test->dst_offsets);
        int num_out_elem = cur_test->dst_size / cur_test->data_size;
        for (int idx = 0; idx < num_out_elem; idx++) {
            temp_out_mem[idx] = src_out_buffer[idx];
        }
    }
    else if(sizeof(int16_t) == cur_test->data_size) {
        int16_t* temp_out_mem = (int16_t*)g_scratch_mem_out;
        int16_t* src_out_buffer = (int16_t*) (g_mem_pool + cur_test->dst_offsets);
        int num_out_elem = cur_test->dst_size / cur_test->data_size;
        for (int idx = 0; idx < num_out_elem; idx++) {
            temp_out_mem[idx] = src_out_buffer[idx];
        }
    }
    else if(sizeof(int32_t) == cur_test->data_size) {
        int32_t* temp_out_mem = (int32_t*)g_scratch_mem_out;
        int32_t* src_out_buffer = (int32_t*) (g_mem_pool + cur_test->dst_offsets);
        int num_out_elem = cur_test->dst_size / cur_test->data_size;
        for (int idx = 0; idx < num_out_elem; idx++) {
            temp_out_mem[idx] = src_out_buffer[idx];
        }
    }

    status = move_broadcast_run_op->Update();
    assert(status == MLI_STATUS_OK);
}

/**************************************************************************************************************/

bool postprocess_phase(const reporter_full* reporter,
                       const move_broadcast_test_operands* cur_test) {

    quality_metrics test_metics;
    bool is_test_passed = false;

    mli_data_container temp_in_container{0};
    mli_data_container temp_out_container{0};
    mli_data_container temp_ref_container{0};
    
    temp_in_container.capacity = cur_test->in.get_not_quantized_tensor(temp_in_container).data.capacity;
    temp_out_container.capacity = cur_test->out.get_not_quantized_tensor(temp_out_container).data.capacity;
    temp_ref_container.capacity = cur_test->out.get_not_quantized_tensor(temp_ref_container).data.capacity;

    temp_in_container.mem.pi8 = g_scratch_mem_in;
    temp_out_container.mem.pi8 = g_scratch_mem_out;
    temp_ref_container.mem.pi8 = g_scratch_mem_ref;

    mli_tensor temp_input_tensor = cur_test->in.get_quantized_tensor(temp_in_container);
    mli_tensor temp_output_tensor = cur_test->out.get_not_quantized_tensor(temp_out_container);
    mli_tensor temp_ref_tensor = cur_test->out.get_not_quantized_tensor(temp_ref_container);

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

/**************************************************************************************************************/

int main() {

    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI3.0|Kernels|Move Broadcast Function Tests");

    for (int i = 0; i< kTestsNum; ++i) {

        bool is_test_passed = true;
        move_broadcast_test_operands* cur_test = &tests_list[i];

        // validate quantization
        if (!(cur_test->in.is_valid() && cur_test->out.is_valid())) {
        reporter.report_message(
            cur_test->descr,
            "FAILED at init: Bad source data for one of tensors");
        is_test_passed = false;
        }

        /**************************************************************************************************************/
        void* move_broadcast_instance = nullptr;
        uint32_t move_broadcast_instance_size = 0;

        void* move_broadcast_conf_private = nullptr;
        uint32_t move_broadcast_conf_private_size = 0;

        /************ Prepare Phase *************/
        prepare_phase(cur_test, move_broadcast_instance, move_broadcast_instance_size,
                      move_broadcast_conf_private, move_broadcast_conf_private_size);


        /************ Execution Phase *************/
        execution_phase(cur_test, move_broadcast_instance, move_broadcast_instance_size,
                        move_broadcast_conf_private, move_broadcast_conf_private_size);

    
        /************ Postprocess Phase *************/
        final_status &= postprocess_phase(&reporter, cur_test);

    }

    reporter.report_outline("[AUTO] Group: mli_krn_move_broadcast_30", final_status);
    
    return 0;
}
