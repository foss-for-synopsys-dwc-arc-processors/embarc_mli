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

#include "test_memory_manager.h"
#include "test_quality_metrics.h"
#include "test_report.h"

#include "impl/mli_reduce_max_ref.hpp"

// #include "mli_krn_rescale.hpp"

// namespace lib_ref = ::snps_arc::metaware::mli::krn::ref;

using mli::tst::crc32_calc;
using mli::tst::quality_metrics;
using mli::tst::reporter_full;

namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;


static int8_t input_1_data[16] = {(3), (2), (4), (1), (9), (5), (7), (6), (2), (57), (35), (20), (7), (4), (0), (90)};

static mli_tensor input_1_tsr_sa8 = {{0U}, {0}, {2U, 2U, 2U, 2U}, 4U, (mli_element_type)264, {{(int8_t)0}}};

static int8_t output_1_data[8] = {0U};

static mli_tensor output_1_tsr_sa8 = {{0U}, {0}, {2U, 1U, 2U, 2U}, 4U, (mli_element_type)264, {{(int8_t)0}}};

static int8_t test_output_1_data[8] = {9, 5, 7, 6, 7, 57, 35, 90};

static mli_tensor test_1_out_sa8 = {{0U}, {0}, {2U, 1U, 2U, 2U}, 4U, (mli_element_type)264, {{(int8_t)0}}};

///================================///

static int16_t input_2_data[16] = {(3), (2), (4), (1), (9), (5), (7), (6), (2), (57), (35), (20), (7), (4), (0), (90)};

static mli_tensor input_2_tsr_sa8 = {{0U}, {0}, {2U, 2U, 2U, 2U}, 4U, (mli_element_type)16, {{(int16_t)0}}};

static int16_t output_2_data[8] = {0U};

static mli_tensor output_2_tsr_sa8 = {{0U}, {0}, {2U, 2U, 1U, 2U}, 4U, (mli_element_type)16, {{(int16_t)0}}};

static int16_t test_output_2_data[8] = {4, 2, 9, 6, 35, 57, 7, 90};

static mli_tensor test_2_out_sa8 = {{0U}, {0}, {2U, 2U, 1U, 2U}, 4U, (mli_element_type)16, {{(int16_t)0}}};

///===============================///

static int32_t input_3_data[16] = {(3), (2), (4), (1), (9), (5), (7), (6), (2), (57), (35), (20), (7), (4), (0), (90)};

static mli_tensor input_3_tsr_sa8 = {{0U}, {0}, {2U, 2U, 2U, 2U}, 4U, (mli_element_type)288, {{(int32_t)0}}};

static int32_t output_3_data[8] = {0U};

static mli_tensor output_3_tsr_sa8 = {{0U}, {0}, {2U, 2U, 2U, 1U}, 4U, (mli_element_type)288, {{(int32_t)0}}};

static int32_t test_output_3_data[8] = {3, 4, 9, 7, 57, 35, 7, 90};

static mli_tensor test_3_out_sa8 = {{0U}, {0}, {2U, 2U, 2U, 1U}, 4U, (mli_element_type)288, {{(int32_t)0}}};


const lib_mli::ReduceOpConfig test_1_cfg = {1};
const lib_mli::ReduceOpConfig test_2_cfg = {2};
const lib_mli::ReduceOpConfig test_3_cfg = {3};

// int test_core_function(){

//     input_1_tsr_sa8.data.mem.pi8 = input_1_data;
//     input_1_tsr_sa8.data.capacity = sizeof(int8_t) * 16;
//     mli_hlp_set_tensor_mem_strides(&input_1_tsr_sa8);

//     output_1_tsr_sa8.data.mem.pi8 = output_1_data;
//     output_1_tsr_sa8.data.capacity = sizeof(int8_t) * 8;
//     mli_hlp_set_tensor_mem_strides(&output_1_tsr_sa8);

//     snps_arc::metaware::mli::krn::ref::mli_reduce_max<int8_t>(&input_1_tsr_sa8, 0, &output_1_tsr_sa8);
//     printf("hello to reduce max mli3\n");
//     for(int i = 0; i< 8 ; i++){
//         printf(" %d ",output_1_data[i]);
//         printf("\n");
//     }
//     return 0;
// }

// int test_core_function(){

//     input_2_tsr_sa8.data.mem.pi16 = input_2_data;
//     input_2_tsr_sa8.data.capacity = sizeof(int16_t) * 16;
//     mli_hlp_set_tensor_mem_strides(&input_2_tsr_sa8);

//     output_2_tsr_sa8.data.mem.pi16 = output_2_data;
//     output_2_tsr_sa8.data.capacity = sizeof(int16_t) * 8;
//     mli_hlp_set_tensor_mem_strides(&output_2_tsr_sa8);

//     snps_arc::metaware::mli::krn::ref::mli_reduce_max<int16_t>(&input_2_tsr_sa8, 0, &output_2_tsr_sa8);
//     printf("hello to reduce max mli3\n");
//     for(int i = 0; i< 8 ; i++){
//         printf(" %d ",output_2_data[i]);
//         printf("\n");
//     }
//     return 0;
// }

int test_core_function(){

    input_3_tsr_sa8.data.mem.pi32 = input_3_data;
    input_3_tsr_sa8.data.capacity = sizeof(int32_t) * 16;
    mli_hlp_set_tensor_mem_strides(&input_3_tsr_sa8);

    output_3_tsr_sa8.data.mem.pi32 = output_3_data;
    output_3_tsr_sa8.data.capacity = sizeof(int32_t) * 8;
    mli_hlp_set_tensor_mem_strides(&output_3_tsr_sa8);

    snps_arc::metaware::mli::krn::ref::mli_reduce_max<int32_t>(&input_3_tsr_sa8, 0, &output_3_tsr_sa8);
    printf("hello to reduce max mli3\n");
    for(int i = 0; i< 8 ; i++){
        printf(" %d ",output_3_data[i]);
        printf("\n");
    }
    return 0;
}

struct reduce_max_test_operands {
    const char* descr;
    mli_tensor in;
    mli_tensor out;
    uint32_t data_size;
    const lib_mli::ReduceOpConfig cfg;
};

// TODO: make it const
static reduce_max_test_operands tests_list[] = {
    {"Test 1 int8", input_1_tsr_sa8, test_1_out_sa8, sizeof(int8_t), test_1_cfg},
    {"Test 2 int16", input_2_tsr_sa8, test_2_out_sa8, sizeof(int16_t), test_2_cfg},
    {"Test 3 int32", input_3_tsr_sa8, test_3_out_sa8, sizeof(int32_t), test_3_cfg},
};

constexpr uint32_t kMemSize = 2048;
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
    
    mli_tensor temp_input_tensor = cur_test->in;
    int num_elem_in = cur_test->in.data.capacity / cur_test->data_size ;
    if (1 == cur_test->data_size) {
        int8_t* temp_in_mem = (int8_t*) g_scratch_mem_in;
        for (int idx = 0; idx < num_elem_in; idx++) {
            temp_in_mem[idx] = cur_test->in.data.mem.pi8[idx];
        }
    }
    else if (2 == cur_test->data_size) {
        int16_t* temp_in_mem = (int16_t*) g_scratch_mem_in;
        for (int idx = 0; idx < num_elem_in; idx++) {
            temp_in_mem[idx] = cur_test->in.data.mem.pi16[idx];
        }
    }
    else if (4 == cur_test->data_size) {
        int32_t* temp_in_mem = (int32_t*) g_scratch_mem_in;
        for (int idx = 0; idx < num_elem_in; idx++) {
            temp_in_mem[idx] = cur_test->in.data.mem.pi32[idx];
        }
    }

    mli_tensor temp_output_tensor = cur_test->out;
    int num_elem_out = cur_test->out.data.capacity / cur_test->data_size ;
        if (1 == cur_test->data_size) {
        int8_t* temp_out_mem = (int8_t*) g_scratch_mem_ref;
        for (int idx = 0; idx < num_elem_out; idx++) {
            temp_out_mem[idx] = cur_test->out.data.mem.pi8[idx];
        }
    }
    else if (2 == cur_test->data_size) {
        int16_t* temp_out_mem = (int16_t*) g_scratch_mem_ref;
        for (int idx = 0; idx < num_elem_out; idx++) {
            temp_out_mem[idx] = cur_test->out.data.mem.pi16[idx];
        }
    }
    else if (1 == cur_test->data_size) {
        int32_t* temp_out_mem = (int32_t*) g_scratch_mem_ref;
        for (int idx = 0; idx < num_elem_out; idx++) {
            temp_out_mem[idx] = cur_test->out.data.mem.pi32[idx];
        }
    }

    // STEP 1: Construct MaxPool2D as a specific ExecutionInterface successor
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
    uint32_t ctrl_buffer_size = reduce_max_op->GetCtrlBufferSize();
    lib_mli::OffsetBuffer reduce_max_ctrl_buf{*offset, 0, ctrl_buffer_size,
                                                sizeof(char)};
    *offset += ctrl_buffer_size;

    // Attaching buffer (descriptors) to the operation
    mli_status status = MLI_STATUS_OK;

    status = reduce_max_op->AttachBufferOffsets(reduce_max_in_tensor, reduce_max_out_tensor,
                                                reduce_max_ctrl_buf);
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

bool postprocess_phase(const reduce_max_test_operands* cur_test,
                       lib_mli::PrivateData* reduce_max_conf_private) {
    bool is_test_passed = true;

    lib_ref::ReduceMaxPrivateData* reduce_max_private = (lib_ref::ReduceMaxPrivateData*)(reduce_max_conf_private);
    
    printf("| ");
    printf(cur_test->descr);
    if(1 == reduce_max_private->output.get_buf().get_elem_size()) {
        int8_t* temp_out_mem = (int8_t*)g_scratch_mem_out;
        int num_out_elem = reduce_max_private->output.get_buf().get_size() / reduce_max_private->output.get_buf().get_elem_size();
        for (int idx = 0; idx < num_out_elem; idx++) {
            if (temp_out_mem[idx] != cur_test->out.data.mem.pi8[idx]) {
                printf("\t\t\tFAILED\n");
                return false;
            }
        }
    }
    else if(2 == reduce_max_private->output.get_buf().get_elem_size()) {
        int16_t* temp_out_mem = (int16_t*)g_scratch_mem_out;
        int num_out_elem = reduce_max_private->output.get_buf().get_size() / reduce_max_private->output.get_buf().get_elem_size();
        for (int idx = 0; idx < num_out_elem; idx++) {
            if (temp_out_mem[idx] != cur_test->out.data.mem.pi16[idx]) {
                printf("\t\t\tFAILED\n");
                return false;
            }
        }
    }
    else if(4 == reduce_max_private->output.get_buf().get_elem_size()) {
        int32_t* temp_out_mem = (int32_t*)g_scratch_mem_out;
        int num_out_elem = reduce_max_private->output.get_buf().get_size() / reduce_max_private->output.get_buf().get_elem_size();
        for (int idx = 0; idx < num_out_elem; idx++) {
            if (temp_out_mem[idx] != cur_test->out.data.mem.pi32[idx]) {
                printf("\t\t\tFAILED\n");
                return false;
            }
        }
    }
    printf("\t\t\tPASSED\n");
    return is_test_passed;
}

int main(){
    // // tests only the reduce max core functionality
    // test_core_function();

    tests_list[0].in.data.mem.pi8 = input_1_data;
    tests_list[0].in.data.capacity = sizeof(int8_t) * 16;
    mli_hlp_set_tensor_mem_strides(&(tests_list[0].in));

    output_1_tsr_sa8.data.mem.pi8 = output_1_data;
    output_1_tsr_sa8.data.capacity = sizeof(int8_t) * 8;
    mli_hlp_set_tensor_mem_strides(&output_1_tsr_sa8);

    tests_list[0].out.data.mem.pi8 = test_output_1_data;
    tests_list[0].out.data.capacity = sizeof(int8_t) * 8;
    mli_hlp_set_tensor_mem_strides(&(tests_list[0].out));

    ///========================///

    tests_list[1].in.data.mem.pi16 = input_2_data;
    tests_list[1].in.data.capacity = sizeof(int16_t) * 16;
    mli_hlp_set_tensor_mem_strides(&(tests_list[1].in));

    output_2_tsr_sa8.data.mem.pi16 = output_2_data;
    output_2_tsr_sa8.data.capacity = sizeof(int16_t) * 8;
    mli_hlp_set_tensor_mem_strides(&output_2_tsr_sa8);

    tests_list[1].out.data.mem.pi16 = test_output_2_data;
    tests_list[1].out.data.capacity = sizeof(int16_t) * 8;
    mli_hlp_set_tensor_mem_strides(&(tests_list[1].out));

    ///========================///

    tests_list[2].in.data.mem.pi32 = input_3_data;
    tests_list[2].in.data.capacity = sizeof(int32_t) * 16;
    mli_hlp_set_tensor_mem_strides(&(tests_list[2].in));

    output_2_tsr_sa8.data.mem.pi32 = output_3_data;
    output_2_tsr_sa8.data.capacity = sizeof(int32_t) * 8;
    mli_hlp_set_tensor_mem_strides(&output_1_tsr_sa8);

    tests_list[2].out.data.mem.pi32 = test_output_3_data;
    tests_list[2].out.data.capacity = sizeof(int32_t) * 8;
    mli_hlp_set_tensor_mem_strides(&(tests_list[2].out));    

    

    const reporter_full reporter;
    bool final_status = true;

    reporter.report_header("MLI3.0|Kernels|Reduce Max Function Tests");

    for (int i = 0; i< kTestsNum; ++i) {

        bool is_test_passed = true;
        const reduce_max_test_operands* cur_test = &tests_list[i];

        // validate quantization
        // if (!(cur_test->in.is_valid() && cur_test->out.is_valid())) {
        // reporter.report_message(
        //     cur_test->descr,
        //     "FAILED at init: Bad source data for one of tensors");
        // is_test_passed = false;
        // }

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
        final_status &= postprocess_phase(cur_test, reduce_max_conf_private);



    }
    

    reporter.report_outline("[AUTO] Group: mli_krn_reduce_max_30", final_status);
    
    return 0;
}
