/*
 * Copyright 2022, Synopsys, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-3-Clause license found in
 * the LICENSE file in the root directory of this source tree.
 *
 */

#include <cstring>

#include "mli_debug.h"
#include "mli_ref_runtime_api.hpp"
#include "mli_reduce_sum.hpp"
#include "mli_ref_private_types.hpp"

namespace snps_arc::metaware::mli::ref {

namespace mli_krn = ::snps_arc::metaware::mli::krn;

ReduceSum::ReduceSum(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems){
    
    MLI_ASSERT(size == sizeof(ReduceSumPrivateData));
    ReduceSumPrivateData private_buffer;
    memcpy(&private_buffer, kernel_private_data_buffer, sizeof(ReduceSumPrivateData));
    MLI_ASSERT(private_buffer.size == sizeof(ReduceSumPrivateData));

    m_in_elem_size = private_buffer.input.get_tensor().get_elem_size();
    m_out_elem_size = private_buffer.output.get_tensor().get_elem_size();
    
    // construct configurations
    m_reduce_axis = private_buffer.reduce_axis;

    // construct Input tensor
    m_input = private_buffer.input;
    const auto input_tile_tensor = m_input.GetSubTensor();
    InternalBuffer input_internal(private_buffer.input.get_buf(), membases, num_mems);
    m_tile_input.rank = private_buffer.io_rank;
    if(m_in_elem_size == sizeof(int32_t)){
        m_tile_input.el_type = MLI_EL_SA_32;
        mli_prv_tensor_set_data_ptr(&m_tile_input, input_internal.get_ptr<int32_t>());
    }else if(m_in_elem_size == sizeof(int16_t)){
        m_tile_input.el_type = MLI_EL_FX_16;
        mli_prv_tensor_set_data_ptr(&m_tile_input, input_internal.get_ptr<int16_t>());
    }else if(m_in_elem_size == sizeof(int8_t)){
        m_tile_input.el_type = MLI_EL_SA_8;
        mli_prv_tensor_set_data_ptr(&m_tile_input, input_internal.get_ptr<int8_t>());
    }else{
        MLI_ASSERT(0);
    }
    
    for(unsigned int i = 0; i < m_tile_input.rank ; i++){
        m_tile_input.shape[i] = input_tile_tensor.get_dim(i);
        m_tile_input.mem_stride[i] = private_buffer.input.get_mem_stride(i);
    }


    // construct Output tensor
    m_output = private_buffer.output;
    const auto output_tile_tensor = m_output.GetSubTensor();
    InternalBuffer output_internal(private_buffer.output.get_buf(), membases, num_mems);
    m_tile_output.rank = private_buffer.io_rank;
    if(m_out_elem_size == sizeof(int32_t)){
        m_tile_output.el_type = MLI_EL_SA_32;
        mli_prv_tensor_set_data_ptr(&m_tile_output, output_internal.get_ptr<int32_t>());
    }else if(m_out_elem_size == sizeof(int16_t)){
        m_tile_output.el_type = MLI_EL_FX_16;
        mli_prv_tensor_set_data_ptr(&m_tile_output, output_internal.get_ptr<int16_t>());
    }else if(m_out_elem_size == sizeof(int8_t)){
        m_tile_output.el_type = MLI_EL_SA_8;
        mli_prv_tensor_set_data_ptr(&m_tile_output, output_internal.get_ptr<int8_t>());
    }else{
        MLI_ASSERT(0);
    }
    
    for(unsigned int i = 0; i < m_tile_output.rank ; i++){
        m_tile_output.shape[i] = output_tile_tensor.get_dim(i);
        m_tile_output.mem_stride[i] = private_buffer.output.get_mem_stride(i);
    }
}


mli_status ReduceSum::Issue() {
    switch(m_in_elem_size){
        case (sizeof(int32_t)):
            mli_krn::mli_reduce_sum<int32_t>(&m_tile_input, m_reduce_axis, &m_tile_output);
            break;
        case (sizeof(int16_t)):
            mli_krn::mli_reduce_sum<int16_t>(&m_tile_input, m_reduce_axis, &m_tile_output);
            break;
        case (sizeof(int8_t)):
            mli_krn::mli_reduce_sum<int8_t>(&m_tile_input, m_reduce_axis, &m_tile_output);
            break;
        default:
            MLI_ASSERT(0);
            break;
    }
    return MLI_STATUS_OK;
}

mli_status ReduceSum::Prefetch() { return MLI_STATUS_OK; }

mli_status ReduceSum::Update() {
    m_input.Next();
    m_output.Next();

    m_input.GetSubTensor().get_dims(m_tile_input.shape);
    m_output.GetSubTensor().get_dims(m_tile_output.shape);

    return MLI_STATUS_OK;
}

void ReduceSum::GetIOSizesAndOffsets(uint32_t input_size[kReduceSumRank], uint32_t output_size[kReduceSumRank],
                                     int32_t input_offsets[kReduceSumRank], int32_t output_offsets[kReduceSumRank]){
    m_input.get_pos(input_offsets);

    m_output.get_pos(output_offsets);

    const auto input_tile_tensor = m_input.GetSubTensor();
    input_tile_tensor.get_dims(input_size);

    const auto output_tile_tensor = m_output.GetSubTensor();
    output_tile_tensor.get_dims(output_size);
}


}  // namespace snps_arc::metaware::mli::ref