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
#include "mli_reduce_max.hpp"
#include "mli_ref_private_types.hpp"

namespace snps_arc::metaware::mli::ref {

namespace mli_krn = ::snps_arc::metaware::mli::krn;

ReduceMax::ReduceMax(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems){
    
    MLI_ASSERT(size == sizeof(ReduceMaxPrivateData));
    ReduceMaxPrivateData private_buffer;
    memcpy(&private_buffer, kernel_private_data_buffer, sizeof(ReduceMaxPrivateData));
    MLI_ASSERT(private_buffer.size == sizeof(ReduceMaxPrivateData));

    m_in_elem_size = private_buffer.input.get_elem_size();
    m_out_elem_size = private_buffer.output.get_elem_size();
    
    // construct configurations
    m_reduce_axis = private_buffer.reduce_axis;

    // construct Input tensor
    InternalBuffer input_internal(private_buffer.input.get_buf(), membases, num_mems);
    if(m_in_elem_size == sizeof(int32_t)){
        m_input.data.mem.pi32 = input_internal.get_ptr<int32_t>();
    }else if(m_in_elem_size == sizeof(int16_t)){
        m_input.data.mem.pi16 = input_internal.get_ptr<int16_t>();
    }else if(m_in_elem_size == sizeof(int8_t)){
        m_input.data.mem.pi8 = input_internal.get_ptr<int8_t>();
    }else{
        MLI_ASSERT(0);
    }
    m_input.rank = private_buffer.io_rank;
    for(unsigned int i = 0; i < m_input.rank ; i++){
        m_input.shape[i] = private_buffer.input.get_dim(i);
        m_input.mem_stride[i] = private_buffer.input.get_mem_stride(i);
    }


    // construct Output tensor
    InternalBuffer output_internal(private_buffer.output.get_buf(), membases, num_mems);
    if(m_out_elem_size == sizeof(int32_t)){
        m_output.data.mem.pi32 = output_internal.get_ptr<int32_t>();
    }else if(m_out_elem_size == sizeof(int16_t)){
        m_output.data.mem.pi16 = output_internal.get_ptr<int16_t>();
    }else if(m_out_elem_size == sizeof(int8_t)){
        m_output.data.mem.pi8 = output_internal.get_ptr<int8_t>();
    }else{
        MLI_ASSERT(0);
    }
    m_output.rank = private_buffer.io_rank;
    for(unsigned int i = 0; i < m_output.rank ; i++){
        m_output.shape[i] = private_buffer.output.get_dim(i);
        m_output.mem_stride[i] = private_buffer.output.get_mem_stride(i);
    }
}

mli_status ReduceMax::Issue() {
    switch(m_in_elem_size){
        case (sizeof(int32_t)):
            mli_krn::mli_reduce_max<int32_t>(&m_input, m_reduce_axis, &m_output);
            break;
        case (sizeof(int16_t)):
            mli_krn::mli_reduce_max<int16_t>(&m_input, m_reduce_axis, &m_output);
            break;
        case (sizeof(int8_t)):
            mli_krn::mli_reduce_max<int8_t>(&m_input, m_reduce_axis, &m_output);
            break;
        default:
            MLI_ASSERT(0);
            break;
    }
    return MLI_STATUS_OK;
}

mli_status ReduceMax::Prefetch() {return MLI_STATUS_OK;}

mli_status ReduceMax::Update() {return MLI_STATUS_OK;}

}  // namespace snps_arc::metaware::mli::ref