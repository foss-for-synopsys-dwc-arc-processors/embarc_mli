
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

#include "vectors_mli_krn_permute_30.inc"

using mli::tst::crc32_calc;
using mli::tst::quality_metrics;
using mli::tst::reporter_full;
using mli::tst::tensor_quantizer;

namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;

using lib_mli::kPermuteRank;
using lib_mli::kPermuteIterRank;

struct permute_test_operands {
    const char* descr;
    tensor_quantizer in;
    tensor_quantizer out;
    uint32_t data_size;
    const lib_mli::PermuteOpConfig cfg;
    const quality_metrics threshold;
    const crc32_calc check_sum;
    uint32_t input_tile_size[kPermuteIterRank];
};

#if defined(CRC_RM_CONVERGENT) || defined(CRC_RM_UP)
// Shared CRC Results
const crc32_calc  test_1_chksum_fx8{ 0xC7AE8D46 }, test_1_chksum_sa8{ 0x461599A8 }, test_1_chksum_sa32{ 0x4B5A8506 }, 
                  test_2_chksum_fx8{ 0xD90894DA }, test_2_chksum_sa8{ 0x7A9D9932 }, test_2_chksum_sa32{ 0xDBD0813E }, 
                  test_3_chksum_fx8{ 0xADA1B465 }, test_3_chksum_sa8{ 0xC7523FA5 }, test_3_chksum_sa32{ 0x0A48FA41 },  
                  test_4_chksum_fx8{ 0xD6A1C316 }, test_4_chksum_sa8{ 0x043C46FB }, test_4_chksum_sa32{ 0x9E20747F }, 
                  test_5_chksum_fx8{ 0x0FDF29A8 }, test_5_chksum_sa8{ 0x3441D826 }, test_5_chksum_sa32{ 0x47C727F2 }, 
                  test_6_chksum_fx8{ 0x5A43F7EB }, test_6_chksum_sa8{ 0x75D4A41F }, test_6_chksum_sa32{ 0x768D898E }, 
                  test_7_chksum_fx8{ 0xE25FDC5A }, test_7_chksum_sa8{ 0xA8E5DD39 }, test_7_chksum_sa32{ 0x61D3FD74 }, 
                  test_8_chksum_fx8{ 0x598491BD }, test_8_chksum_sa8{ 0xA496E075 }, test_8_chksum_sa32{ 0x1E88E1FD }, 
                  test_9_chksum_fx8{ 0x254C2F3E }, test_9_chksum_sa8{ 0xBA173619 }, test_9_chksum_sa32{ 0xCDB7D98E }; 

#else  // Not defined CRC_*
const crc32_calc  test_1_chksum_fx8, test_1_chksum_sa8, test_1_chksum_sa32,
                  test_2_chksum_fx8, test_2_chksum_sa8, test_2_chksum_sa32,
                  test_3_chksum_fx8, test_3_chksum_sa8, test_3_chksum_sa32,
                  test_4_chksum_fx8, test_4_chksum_sa8, test_4_chksum_sa32,
                  test_5_chksum_fx8, test_5_chksum_sa8, test_5_chksum_sa32,
                  test_6_chksum_fx8, test_6_chksum_sa8, test_6_chksum_sa32,
                  test_7_chksum_fx8, test_7_chksum_sa8, test_7_chksum_sa32,
                  test_8_chksum_fx8, test_8_chksum_sa8, test_8_chksum_sa32,
                  test_9_chksum_fx8, test_9_chksum_sa8, test_9_chksum_sa32;
#endif

const quality_metrics thresholds_fx_general {/* MaxAbsErr = */0.0f, quality_metrics::kPassValueSnr,
                                             /* SNR_DB = */84.f, quality_metrics::kPassValueQuantErrPerc };

const quality_metrics thresholds_sa8_general {quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                              /* SNR_DB = */40.f, /*Quant Error Perc = */ 99.9f };

const quality_metrics thresholds_sa32_general {quality_metrics::kPassValueMaxAbsErr, quality_metrics::kPassValueSnr,
                                              /* SNR_DB = */40.f, /*Quant Error Perc = */ 99.9f };

static const permute_test_operands tests_list[] = {
    /* Per-axis quantization, input shape = {16, 32}, input memory stride = {32, 1},
     * output shape = {32, 16}, perm_dim = {1, 0} */
    {"Test 1 FX8 I_mstr", input_1_memstr_fx8, test_1_out_fx8, sizeof(int8_t), test_1_cfg,
                          thresholds_fx_general, test_1_chksum_fx8, INPUT_1_TILE_IN},                                        
    {"Test 1 SA8 I_mstr 0-axis", input_1_memstr_sa8, test_1_out_sa8, sizeof(int8_t), test_1_cfg,
                                 thresholds_sa8_general, test_1_chksum_sa8, INPUT_1_TILE_IN},
    {"Test 1 SA32 I_mstr 0-axis", input_1_memstr_sa32, test_1_out_sa32, sizeof(int32_t), test_1_cfg,
                                  thresholds_sa32_general, test_1_chksum_sa32, INPUT_1_TILE_IN},

    /* Per-tensor quantization, input shape = {2, 3, 4, 5}, output shape = {5, 4, 3, 2},
     * output memstride = {96, 12, 4, 1}, perm_dim = {3, 2, 1, 0} */
    {"Test 2 FX8 O_mstr",  input_2_fx8, test_2_out_memstr_fx8, sizeof(int8_t), test_2_cfg,
                           thresholds_fx_general, test_2_chksum_fx8, INPUT_2_TILE_IN},
    {"Test 2 SA8 O_mstr per-tensor", input_2_memstr_sa8, test_2_out_memstr_sa8, sizeof(int8_t), test_2_cfg,
                                     thresholds_sa8_general, test_2_chksum_sa8, INPUT_2_TILE_IN},
    {"Test 2 SA32 O_mstr per-tensor", input_2_memstr_sa32, test_2_out_memstr_sa32, sizeof(int32_t), test_2_cfg,
                                      thresholds_sa32_general, test_2_chksum_sa32, INPUT_2_TILE_IN},                                 

    /* Per-tensor quantization, input shape = {2, 3, 4, 5}, input memory stride = {192, 32, 8, 1}, 
     * output shape = {3, 2, 5, 4}, output memory stride = {80, 40, 4, 1}, perm_dim = {1, 0, 3, 2} */
    {"Test 3 FX8 IO_mstr", input_2_memstr_fx8, test_3_out_memstr_fx8, sizeof(int8_t), test_3_cfg,
                           thresholds_fx_general, test_3_chksum_fx8, INPUT_2_TILE_IN},
    {"Test 3 SA8 IO_mstr per-tensor",input_2_memstr_sa8, test_3_out_memstr_sa8, sizeof(int8_t),test_3_cfg, 
                                     thresholds_sa8_general, test_3_chksum_sa8, INPUT_2_TILE_IN},
    {"Test 3 SA32 IO_mstr per-tensor",input_2_memstr_sa32, test_3_out_memstr_sa32, sizeof(int32_t),test_3_cfg, 
                                      thresholds_sa32_general, test_3_chksum_sa32, INPUT_2_TILE_IN},

    /* Per-tensor quantization, input shape = {10, 1, 1, 1}, output shape = {1, 1, 1, 10}, perm_dim = {1, 2, 3, 0} */
    {"Test 4 FX8 ", input_3_fx8, test_4_out_fx8, sizeof(int8_t), test_4_cfg,
                    thresholds_fx_general, test_4_chksum_fx8, INPUT_3_TILE_IN},
    {"Test 4 SA8 per-tensor", input_3_sa8, test_4_out_sa8, sizeof(int8_t), test_4_cfg,
                              thresholds_sa8_general, test_4_chksum_sa8, INPUT_3_TILE_IN},
    {"Test 4 SA32 per-tensor", input_3_sa32, test_4_out_sa32, sizeof(int32_t), test_4_cfg,
                               thresholds_sa32_general, test_4_chksum_sa32, INPUT_3_TILE_IN},

    /* Per-axis quantization, input shape = {3, 4, 5}, output shape = {4, 3, 5}, perm_dim = {1, 0, 2} */
    {"Test 5 FX8",  input_4_fx8, test_5_out_fx8, sizeof(int8_t), test_5_cfg,
                    thresholds_fx_general, test_5_chksum_fx8, INPUT_4_TILE_IN},
    {"Test 5 SA8 1-axis", input_4_sa8, test_5_out_sa8, sizeof(int8_t), test_5_cfg,
                          thresholds_sa8_general, test_5_chksum_sa8, INPUT_4_TILE_IN},

    /* Per-tensor quantization, input shape = {2, 3, 4, 5}, output shape = {2, 5, 3, 4}, perm_dim = {0, 3, 1, 2} */
    {"Test 6 FX8", input_2_fx8, test_6_out_fx8, sizeof(int8_t), test_6_cfg,
                   thresholds_fx_general, test_6_chksum_fx8, INPUT_2_TILE_IN},
    {"Test 6 SA8 per-tensor", input_2_sa8, test_6_out_sa8, sizeof(int8_t), test_6_cfg,
                              thresholds_sa8_general, test_6_chksum_sa8, INPUT_2_TILE_IN},
    {"Test 6 SA32 per-tensor", input_2_sa32, test_6_out_sa32, sizeof(int32_t), test_6_cfg,
                               thresholds_sa32_general, test_6_chksum_sa32, INPUT_2_TILE_IN},

    /* Per-tensor quantization, input shape = {2, 3, 4, 5}, output shape = {2, 4, 5, 3}, perm_dim = {0, 2, 3, 1} */
    {"Test 7 FX8",  input_2_fx8, test_7_out_fx8, sizeof(int8_t), test_7_cfg,
                    thresholds_fx_general, test_7_chksum_fx8, INPUT_2_TILE_IN},
    {"Test 7 SA8 per-tensor", input_2_sa8, test_7_out_sa8, sizeof(int8_t), test_7_cfg,
                              thresholds_sa8_general, test_7_chksum_sa8, INPUT_2_TILE_IN},
    {"Test 7 SA32 per-tensor", input_2_sa32, test_7_out_sa32, sizeof(int32_t), test_7_cfg,
                               thresholds_sa32_general, test_7_chksum_sa32, INPUT_2_TILE_IN},

    /* Per-tensor quantization, input shape = {2, 3, 4, 5}, output shape = {2, 4, 3, 5}, perm_dim = {0, 2, 1, 3} */
    {"Test 8 FX8",  input_2_fx8, test_8_out_fx8, sizeof(int8_t), test_8_cfg,
                    thresholds_fx_general, test_8_chksum_fx8, INPUT_2_TILE_IN},
    {"Test 8 SA8 per-tensor", input_2_sa8, test_8_out_sa8, sizeof(int8_t), test_8_cfg,
                              thresholds_sa8_general, test_8_chksum_sa8, INPUT_2_TILE_IN},
    {"Test 8 SA32 per-tensor", input_2_sa32, test_8_out_sa32, sizeof(int32_t), test_8_cfg,
                               thresholds_sa32_general, test_8_chksum_sa32, INPUT_2_TILE_IN},

    /* Per-tensor quantization, input shape = {2, 3, 4, 5}, output shape = {3, 2, 4, 5}, perm_dim = {1, 0, 2, 3} */
    {"Test 9 FX8",  input_2_fx8, test_9_out_fx8, sizeof(int8_t), test_9_cfg,
                    thresholds_fx_general, test_9_chksum_fx8, INPUT_2_TILE_IN},
    {"Test 9 SA8 per-tensor", input_2_sa8, test_9_out_sa8, sizeof(int8_t), test_9_cfg,
                              thresholds_sa8_general, test_9_chksum_sa8, INPUT_2_TILE_IN},
    {"Test 9 SA32 per-tensor", input_2_sa32, test_9_out_sa32, sizeof(int32_t), test_9_cfg,
                               thresholds_sa32_general, test_9_chksum_sa32, INPUT_2_TILE_IN}

};

constexpr uint32_t kMemSize = 4*2048;
static int8_t g_scratch_mem_in[kMemSize] = {0};
static int8_t g_scratch_mem_ref[kMemSize] = {0};
static int8_t g_scratch_mem_out[kMemSize] = {0};
static IO_DATA_ATTR int8_t g_mem_pool[kMemSize] = {0};

constexpr int kTestsNum = sizeof(tests_list) / sizeof(tests_list[0]);

void prepare_phase(const permute_test_operands* cur_test,
                   int32_t iteration_order[kPermuteIterRank],
                   uint32_t input_tile_size[kPermuteIterRank],
                   uint32_t output_tile_size[kPermuteIterRank],
                   void*& permute_instance,
                   uint32_t& permute_instance_size,
                   lib_mli::PrivateData*& permute_conf_private,
                   uint32_t& permute_conf_private_size) {

    mli_data_container temp_in_container{0};
    mli_data_container temp_out_container{0};
    temp_in_container.capacity = kMemSize;
    temp_in_container.mem.pi8 = g_scratch_mem_in;
    temp_out_container.capacity = kMemSize;
    temp_out_container.mem.pi8 = g_scratch_mem_ref;
    
    mli_tensor temp_input_tensor = cur_test->in.get_quantized_tensor(temp_in_container);
    mli_tensor temp_output_tensor = cur_test->out.get_quantized_tensor(temp_out_container);
    

    // STEP 1: Construct Permute as a specific ExecutionInterface successor
    //==================================================================
    const lib_mli::Tensor<lib_mli::NoBuffer, kPermuteRank> in_tensor(temp_input_tensor.shape, temp_input_tensor.mem_stride);
    const lib_mli::Tensor<lib_mli::NoBuffer, kPermuteRank> out_tensor(temp_output_tensor.shape, temp_output_tensor.mem_stride);

    lib_mli::PlatformDescription pd;
    lib_ref::KernelsFactory kernel_factory(pd);
    uint32_t permute_cs_size = kernel_factory.Permute_CS_GetSize();
    void* permute_cs_buffer = malloc(permute_cs_size);

    lib_mli::IteratorCfg<kPermuteIterRank> input_it_config(in_tensor, input_tile_size, iteration_order);
    lib_mli::IteratorCfg<kPermuteIterRank> output_it_config(out_tensor, output_tile_size, iteration_order);

    lib_mli::Tensor<lib_mli::NoBuffer, kPermuteRank> full_in_tensor(temp_input_tensor.shape, temp_input_tensor.mem_stride);
    lib_mli::TensorIterator<lib_mli::NoBuffer, kPermuteRank, kPermuteIterRank> in_tensor_it(full_in_tensor, input_it_config);

    lib_mli::Tensor<lib_mli::NoBuffer, kPermuteRank> full_out_tensor(temp_output_tensor.shape, temp_output_tensor.mem_stride);
    lib_mli::TensorIterator<lib_mli::NoBuffer, kPermuteRank, kPermuteIterRank> out_tensor_it(full_out_tensor, output_it_config);
    
    auto permute_op = kernel_factory.Permute_CS(permute_cs_buffer, in_tensor_it, cur_test->cfg, out_tensor_it);


    // STEP 2: Memory management (Up to user on how to deal with it)
    //==================================================================
    mli_status status = MLI_STATUS_OK;
    uint32_t offsets[1] = {0};

    uint32_t elem_size = cur_test->data_size;

    // Define buffers for in\out tensors
    // Leave space for runtime object
    uint32_t* offset = &offsets[0];
    uint32_t runtime_obj_size = permute_op->GetRuntimeObjectSize();
    *offset += runtime_obj_size;

    // Leave space for private data buffer
    offset = &offsets[0];
    uint32_t private_buffer_size = permute_op->GetKernelPrivateDataSize();
    *offset += private_buffer_size;

    // Permute Input
    offset = &offsets[0];
    uint32_t in_size = lib_mli::service::GetBufferSize(kPermuteRank, input_tile_size, temp_input_tensor.mem_stride) * elem_size;
    lib_mli::OffsetBuffer permute_in_buf{*offset, 0, in_size, elem_size};
    *offset += in_size;

    // Permute Output
    offset = &offsets[0];
    uint32_t out_size = lib_mli::service::GetBufferSize(kPermuteRank, output_tile_size, temp_output_tensor.mem_stride) * elem_size;
    lib_mli::OffsetBuffer permute_out_buf{*offset, 0, out_size, elem_size};
    *offset += out_size;

    // DataBuffer size is 0 for reference kernel
    offset = &offsets[0];
    uint32_t ctrl_buffer_size = permute_op->GetCtrlBufferSize();
    lib_mli::OffsetBuffer permute_ctrl_buf{*offset, 0, ctrl_buffer_size, sizeof(char)};
    *offset += ctrl_buffer_size;

    // Attaching buffer (descriptors) to the operation
    status = permute_op->AttachBufferOffsets(permute_in_buf, permute_out_buf, permute_ctrl_buf);
    assert(status == MLI_STATUS_OK);

    permute_instance = (int8_t*)g_mem_pool;
    permute_instance_size = permute_op->GetRuntimeObjectSize();

    status = permute_op->GetKernelPrivateData((int8_t*)g_mem_pool + permute_instance_size);
    assert(status == MLI_STATUS_OK);
    permute_conf_private = reinterpret_cast<lib_mli::PrivateData*>((int8_t*)g_mem_pool + permute_instance_size);
    permute_conf_private_size = permute_op->GetKernelPrivateDataSize();

}


void execution_phase(const permute_test_operands* cur_test, uint32_t num_tiles, void* permute_instance, uint32_t permute_instance_size,
                    lib_mli::PrivateData* permute_conf_private, uint32_t permute_conf_private_size) {

    // STEP 3: Execution phase
    //==================================================================

    uint64_t membasis[] = {reinterpret_cast<uint64_t>(g_mem_pool)};

    auto permute_run_op = lib_mli::ExecutionInterface::Create(permute_instance,
                                                              permute_instance_size,
                                                              permute_conf_private,
                                                              permute_conf_private_size,
                                                              membasis, sizeof(membasis) / sizeof(membasis[0]));
    assert(permute_run_op != nullptr);
    mli_status status = MLI_STATUS_OK;

    lib_ref::PermutePrivateData* permute_private = (lib_ref::PermutePrivateData*)(permute_conf_private);
    int32_t tile_input_strides[kPermuteIterRank];
    int32_t tile_output_strides[kPermuteIterRank];
    permute_private->input.get_mem_strides(tile_input_strides);
    permute_private->output.get_mem_strides(tile_output_strides);

    lib_ref::Permute* permute_impl  = dynamic_cast<lib_ref::Permute*>(permute_run_op);
    uint32_t input_tile_size[kPermuteIterRank];
    uint32_t output_tile_size[kPermuteIterRank];
    int32_t input_tile_offsets[kPermuteIterRank];
    int32_t output_tile_offsets[kPermuteIterRank];
    const int32_t zero_offsets[kPermuteIterRank]{};

    // Handling SA quantization     
    mli_data_container temp_in_container{0};
    temp_in_container.capacity = kMemSize;
    temp_in_container.mem.pi8 = g_scratch_mem_in;
    mli_tensor temp_input_tensor = cur_test->in.get_quantized_tensor(temp_in_container);
    uint32_t temp_in_offest = 0;
    if((temp_input_tensor.el_type == MLI_EL_SA_8)||(temp_input_tensor.el_type == MLI_EL_SA_32))
    {
        if(temp_input_tensor.el_params.sa.dim >= 0 )
        {
            temp_in_offest += (temp_input_tensor.shape[temp_input_tensor.el_params.sa.dim] * 5) ;
        }
    }

    for(uint32_t n_tile = 0; n_tile < num_tiles; n_tile++) {
        
        status = permute_run_op->Prefetch();
        assert(status == MLI_STATUS_OK);

        permute_impl->GetIOSizesAndOffsets(input_tile_size, output_tile_size,
                                           input_tile_offsets, output_tile_offsets);

        // copy input from global buffer to local tile buffer
        strided_copy_with_offsets(kPermuteRank, permute_private->input.get_buf().get_elem_size(),
                                  (g_scratch_mem_in + temp_in_offest), input_tile_offsets, zero_offsets, tile_input_strides,
                                  input_tile_size, (int8_t*)(g_mem_pool + permute_private->input.get_buf().get_offset()) );

        
        status = permute_run_op->Issue();
        assert(status == MLI_STATUS_OK);

        // copy output from local tile buffer to global buffer
        strided_copy_with_offsets(kPermuteRank, permute_private->input.get_buf().get_elem_size(),
                                  (int8_t*)(g_mem_pool + permute_private->output.get_buf().get_offset()),
                                  zero_offsets, output_tile_offsets, tile_output_strides,
                                  output_tile_size, (int8_t*) g_scratch_mem_out);

        status = permute_run_op->Update();
        assert(status == MLI_STATUS_OK);
    }
}

bool postprocess_phase(const reporter_full* reporter,
                       const permute_test_operands* cur_test) {
    quality_metrics test_metics;
    bool is_test_passed = false;

    mli_data_container temp_in_container{0};
    mli_data_container temp_out_container{0};
    temp_in_container.capacity = kMemSize;
    temp_in_container.mem.pi8 = g_scratch_mem_in;
    temp_out_container.capacity = kMemSize;
    temp_out_container.mem.pi8 = g_scratch_mem_ref;

    mli_tensor temp_input_tensor = cur_test->in.get_quantized_tensor(temp_in_container);
    mli_tensor temp_output_tensor = cur_test->out.get_not_quantized_tensor(temp_out_container);
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

    reporter.report_header("MLI3.0|Kernels|Permute Function Tests");

    for (int i = 0; i< kTestsNum; ++i) {
        
        bool is_test_passed = true;
        const permute_test_operands* cur_test = &tests_list[i];

#if PLATFORM == V2DSP_VECTOR
        if (strstr(cur_test->descr, "Test 8 SA8 per-tensor") != nullptr ||
            strstr(cur_test->descr, "Test 8 SA32 per-tensor") != nullptr ||
            strstr(cur_test->descr, "Test 9 FX8") != nullptr ||
            strstr(cur_test->descr, "Test 9 SA8 per-tensor") != nullptr ||
            strstr(cur_test->descr, "Test 9 SA32 per-tensor") != nullptr) {
        
            reporter.report_message(cur_test->descr, "SKIPPED due to vpx issue");
            continue;
        }
#endif

        // validate quantization
        if (!(cur_test->in.is_valid() && cur_test->out.is_valid())) {
        reporter.report_message(
            cur_test->descr,
            "FAILED at init: Bad source data for one of tensors");
        is_test_passed = false;
        }

        /**************************************************************************************************************/
        mli_data_container temp_container{ 0 };
        mli_tensor temp_input_tensor = cur_test->in.get_not_quantized_tensor(temp_container);

        for(uint32_t i = 0; i < kPermuteRank; i++) {
            // in case of input rank less than 4
            if (temp_input_tensor.shape[i] == 0) {
                temp_input_tensor.shape[i] = 1;
            }
        }

        int32_t iteration_order[kPermuteRank] = {0, 1, 2, 3};
        uint32_t input_tile_size[kPermuteIterRank] = {cur_test->input_tile_size[0], cur_test->input_tile_size[1], cur_test->input_tile_size[2], cur_test->input_tile_size[3]};
        uint32_t output_tile_size[kPermuteRank];

        // output tile size is generated using input tile size and reordered by perm_dim.
        for(uint32_t i = 0; i < kPermuteIterRank; i++) {
            output_tile_size[i] = input_tile_size[cur_test->cfg.perm_dim[i]];
        }

        // calculate number of tiles needed
        uint32_t num_tiles = 1;
        for (uint32_t i = 0; i < kPermuteIterRank; i++) {
            uint32_t tiles_per_dim = 1 + CEIL_DIV(temp_input_tensor.shape[i] - input_tile_size[i], input_tile_size[i]);
            num_tiles *= tiles_per_dim;
        }

        /**************************************************************************************************************/
        void* permute_instance = nullptr;
        uint32_t permute_instance_size = 0;

        lib_mli::PrivateData* permute_conf_private = nullptr;
        uint32_t permute_conf_private_size = 0;

        /**************************************************************************************************************/

                /************ Prepare Phase *************/
        prepare_phase(cur_test, iteration_order, input_tile_size, output_tile_size,
                      permute_instance, permute_instance_size,
                      permute_conf_private, permute_conf_private_size);



        /************ Execution Phase *************/
        execution_phase(cur_test, num_tiles,
                        permute_instance, permute_instance_size,
                        permute_conf_private, permute_conf_private_size);
        


        /************ Postprocess Phase *************/
        final_status &= postprocess_phase(&reporter, cur_test);

    }

    reporter.report_outline("[AUTO] Group: mli_krn_permute_30", final_status);
    
    return 0;
}
