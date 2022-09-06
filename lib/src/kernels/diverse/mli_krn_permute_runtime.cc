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
#include "mli_krn_permute.h"
#include "mli_ref_private_types.hpp"

namespace snps_arc::metaware::mli::ref {

namespace mli_ref = ::mli::krn::ref;

Permute::Permute(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems) {
    
    MLI_ASSERT(size == sizeof(PermutePrivateData));
    PermutePrivateData private_buffer;
    memcpy(&private_buffer, kernel_private_data_buffer, sizeof(PermutePrivateData));
    MLI_ASSERT(private_buffer.size == sizeof(PermutePrivateData));

    m_in_elem_size = private_buffer.input.get_tensor().get_elem_size();
    m_out_elem_size = private_buffer.output.get_tensor().get_elem_size();
    
    // construct configurations
    for(uint32_t i = 0; i < kPermuteRank; i++) {
        m_perm_dim[i] = private_buffer.perm_dim[i];
    }

    // construct Input tensor
    m_metadata.m_input = private_buffer.input;
    // check if dim less than rank.
    for(uint32_t i = 0; i < kPermuteRank; i++) {
        m_metadata.m_input.get_tensor().set_dim(i, (m_metadata.m_input.get_tensor().get_dim(i) == 0) ? 1 : m_metadata.m_input.get_tensor().get_dim(i));
    }
    const auto input_tile_tensor = m_metadata.m_input.GetSubTensor();
    InternalBuffer input_internal(private_buffer.input.get_buf(), membases, num_mems);
    m_metadata.m_tile_input.rank = private_buffer.io_rank;
    if(m_in_elem_size == sizeof(int32_t)) {
        m_metadata.m_tile_input.el_type = MLI_EL_SA_32;
        mli_prv_tensor_set_data_ptr(&m_metadata.m_tile_input, input_internal.get_ptr<int32_t>());
    }
	else if(m_in_elem_size == sizeof(int16_t)) {
        m_metadata.m_tile_input.el_type = MLI_EL_FX_16;
        mli_prv_tensor_set_data_ptr(&m_metadata.m_tile_input, input_internal.get_ptr<int16_t>());
    }
	else if(m_in_elem_size == sizeof(int8_t)) {
        m_metadata.m_tile_input.el_type = MLI_EL_SA_8;
        mli_prv_tensor_set_data_ptr(&m_metadata.m_tile_input, input_internal.get_ptr<int8_t>());
    }
	else {
        MLI_ASSERT(0);
    }
    for(uint32_t i = 0; i < m_metadata.m_tile_input.rank ; i++) {
        m_metadata.m_tile_input.shape[i] = input_tile_tensor.get_dim(i);
        m_metadata.m_tile_input.mem_stride[i] = private_buffer.input.get_mem_stride(i);
    }

    // construct Output tensor
    m_metadata.m_output = private_buffer.output;
    // check if dim less than rank.
    for(uint32_t i = 0; i < kPermuteRank; i++) {
        m_metadata.m_output.get_tensor().set_dim(i, (m_metadata.m_output.get_tensor().get_dim(i) == 0) ? 1 : m_metadata.m_output.get_tensor().get_dim(i));
    }
    const auto output_tile_tensor = m_metadata.m_output.GetSubTensor();
    InternalBuffer output_internal(private_buffer.output.get_buf(), membases, num_mems);
    m_metadata.m_tile_output.rank = private_buffer.io_rank;
    if(m_out_elem_size == sizeof(int32_t)) {
        m_metadata.m_tile_output.el_type = MLI_EL_SA_32;
        mli_prv_tensor_set_data_ptr(&m_metadata.m_tile_output, output_internal.get_ptr<int32_t>());
    }
	else if(m_out_elem_size == sizeof(int16_t)) {
        m_metadata.m_tile_output.el_type = MLI_EL_FX_16;
        mli_prv_tensor_set_data_ptr(&m_metadata.m_tile_output, output_internal.get_ptr<int16_t>());
    }
	else if(m_out_elem_size == sizeof(int8_t)) {
        m_metadata.m_tile_output.el_type = MLI_EL_SA_8;
        mli_prv_tensor_set_data_ptr(&m_metadata.m_tile_output, output_internal.get_ptr<int8_t>());
    }
	else {
        MLI_ASSERT(0);
    }
    for(uint32_t i = 0; i < m_metadata.m_tile_output.rank ; i++) {
        m_metadata.m_tile_output.shape[i] = output_tile_tensor.get_dim(i);
        m_metadata.m_tile_output.mem_stride[i] = private_buffer.output.get_mem_stride(i);
    }
}

mli_status Permute::Issue() {
    mli_permute_cfg cfg ;
    bool sa_el_type = true;
    if((m_metadata.m_tile_input.el_type == MLI_EL_SA_8) || (m_metadata.m_tile_input.el_type == MLI_EL_SA_32)) {
        sa_el_type = true;
    }
    else {
        sa_el_type = false;
    }
    switch(m_in_elem_size) {
        case (sizeof(int32_t)):
            for(uint32_t i = 0; i < kPermuteRank; i++) {
                cfg.perm_dim[i] = m_perm_dim[i];
            }
            if(sa_el_type == true) {
                mli_ref::mli_krn_permute_run<int32_t, true>(&m_metadata.m_tile_input, &cfg, &m_metadata.m_tile_output);
            }
            else {
                mli_ref::mli_krn_permute_run<int32_t, false>(&m_metadata.m_tile_input, &cfg, &m_metadata.m_tile_output);
            }
            break;

        case (sizeof(int8_t)):
            for(uint32_t i = 0; i < kPermuteRank; i++) {
                cfg.perm_dim[i] = m_perm_dim[i];
            }
            if(sa_el_type == true) {
                mli_ref::mli_krn_permute_run<int8_t, true>(&m_metadata.m_tile_input, &cfg, &m_metadata.m_tile_output);
            }
            else {
                mli_ref::mli_krn_permute_run<int8_t, false>(&m_metadata.m_tile_input, &cfg, &m_metadata.m_tile_output);
            }
            break;

        default:
            MLI_ASSERT(0);
            break;
    }
    return MLI_STATUS_OK;
}

mli_status Permute::Prefetch() {return MLI_STATUS_OK;}

mli_status Permute::Update() {
    m_metadata.m_input.Next();
    m_metadata.m_output.Next();

    m_metadata.m_input.GetSubTensor().get_dims(m_metadata.m_tile_input.shape);
    m_metadata.m_output.GetSubTensor().get_dims(m_metadata.m_tile_output.shape);

    return MLI_STATUS_OK;
}

void Permute::GetIOSizesAndOffsets(uint32_t input_size[kPermuteRank], uint32_t output_size[kPermuteRank],
                                   int32_t input_offsets[kPermuteRank], int32_t output_offsets[kPermuteRank]) {
    m_metadata.m_input.get_pos(input_offsets);

    m_metadata.m_output.get_pos(output_offsets);

    const auto input_tile_tensor = m_metadata.m_input.GetSubTensor();
    input_tile_tensor.get_dims(input_size);

    const auto output_tile_tensor = m_metadata.m_output.GetSubTensor();
    output_tile_tensor.get_dims(output_size);
}

}  // /*namespace snps_arc::metaware::*/mli::krn::ref
