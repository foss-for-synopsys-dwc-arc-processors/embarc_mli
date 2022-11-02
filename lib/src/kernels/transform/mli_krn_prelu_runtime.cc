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
#include "mli_krn_prelu.h"
#include "mli_ref_private_types.hpp"


namespace snps_arc::metaware::mli::ref {

namespace mli_krn = ::mli::krn::ref;

Prelu::Prelu(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems) {

    MLI_ASSERT(size == sizeof(PreluPrivateData));
    PreluPrivateData private_buffer;
    memcpy(&private_buffer, kernel_private_data_buffer, sizeof(PreluPrivateData));
    MLI_ASSERT(private_buffer.size == sizeof(PreluPrivateData));

    m_input = private_buffer.input;
    m_output = private_buffer.output;

    m_tile_metadata.input = Tensor<InternalBuffer, kPreluRank>(m_input.GetSubTensor(), membases, num_mems);
    m_tile_metadata.output = Tensor<InternalBuffer, kPreluRank>(m_output.GetSubTensor(), membases, num_mems);

    m_tile_param_max_size = private_buffer.tile_params_max_elem_num;
    private_buffer.encoded_params_buffer.set_elem_size(sizeof(int32_t));
    m_tile_metadata.in_bias = InternalBuffer(private_buffer.encoded_params_buffer, membases, num_mems);

    private_buffer.encoded_params_buffer.inc(m_tile_param_max_size);
    private_buffer.encoded_params_buffer.set_elem_size(sizeof(int16_t));
    m_tile_metadata.posscale = InternalBuffer(private_buffer.encoded_params_buffer, membases, num_mems);
   
    private_buffer.encoded_params_buffer.inc(m_tile_param_max_size);
    private_buffer.encoded_params_buffer.set_elem_size(sizeof(int16_t));
    m_tile_metadata.negscale = InternalBuffer(private_buffer.encoded_params_buffer, membases, num_mems);
  
    private_buffer.encoded_params_buffer.inc(m_tile_param_max_size);
    private_buffer.encoded_params_buffer.set_elem_size(sizeof(int8_t));
    m_tile_metadata.posshift = InternalBuffer(private_buffer.encoded_params_buffer, membases, num_mems);

    private_buffer.encoded_params_buffer.inc(m_tile_param_max_size);
    private_buffer.encoded_params_buffer.set_elem_size(sizeof(int8_t));
    m_tile_metadata.negshift = InternalBuffer(private_buffer.encoded_params_buffer, membases, num_mems);

    private_buffer.encoded_params_buffer.inc(m_tile_param_max_size);
    private_buffer.encoded_params_buffer.set_elem_size(sizeof(int8_t));
    m_tile_metadata.out_bias = InternalBuffer(private_buffer.encoded_params_buffer, membases, num_mems);


    m_tile_metadata.prelu_axis = private_buffer.prelu_axis;

}

mli_status Prelu::Issue() {

    uint32_t out_elem_size = m_output.get_tensor().get_elem_size();
    switch(m_input.get_tensor().get_elem_size()) {
        case (sizeof(int32_t)):
            if (out_elem_size == sizeof(int8_t)) {
                mli_krn::mli_krn_prelu<int32_t, int8_t>(m_tile_metadata.input,
                                                        m_tile_metadata.in_bias,
                                                        m_tile_metadata.posscale,
                                                        m_tile_metadata.negscale,
                                                        m_tile_metadata.posshift,
                                                        m_tile_metadata.negshift,
                                                        m_tile_metadata.out_bias,
                                                        m_tile_metadata.prelu_axis,
                                                        m_tile_metadata.output);
            }
            else if (out_elem_size == sizeof(int16_t)) {
                MLI_ASSERT(0);
            }
            break;
        case (sizeof(int16_t)):
            if (out_elem_size == sizeof(int8_t)) {
                MLI_ASSERT(0);
            }
            else if (out_elem_size == sizeof(int32_t)) {
                MLI_ASSERT(0);
            }
            break;
        case (sizeof(int8_t)):
            if (out_elem_size == sizeof(int8_t)) {
                MLI_ASSERT(0);
            }
            else if (out_elem_size == sizeof(int16_t)) {
                MLI_ASSERT(0);
            }
            else if (out_elem_size == sizeof(int32_t)) {
                MLI_ASSERT(0);
            }
            break;
        default:
            MLI_ASSERT(0);
            break;
    }

    return MLI_STATUS_OK;
}

mli_status Prelu::Prefetch() { return MLI_STATUS_OK; }

mli_status Prelu::Update() {

    m_input.Next();
    m_output.Next();
    
    uint32_t ishape[kPreluRank] = {0};
    uint32_t oshape[kPreluRank] = {0};

    m_input.GetSubTensor().get_dims(ishape);
    m_tile_metadata.input.set_dims(ishape);
    m_output.GetSubTensor().get_dims(oshape);
    m_tile_metadata.output.set_dims(oshape);
  
    return MLI_STATUS_OK;
}

void Prelu::GetIOSizesAndOffsets(uint32_t &enc_param_size, uint32_t &inp_bias_offset, uint32_t &posscale_offset, uint32_t &negscale_offset,
                                 uint32_t &posshift_offset, uint32_t &negshift_offset, uint32_t &out_bias_offset) {
  enc_param_size = m_tile_metadata.input.get_dim(kGroupTensorChannelDim);
  inp_bias_offset = 0;
  posscale_offset = m_tile_param_max_size * sizeof(int32_t);
  negscale_offset = posscale_offset + m_tile_param_max_size * sizeof(int16_t);
  posshift_offset = negscale_offset + m_tile_param_max_size * sizeof(int16_t);
  negshift_offset = posshift_offset + m_tile_param_max_size * sizeof(int8_t);
  out_bias_offset = negshift_offset + m_tile_param_max_size * sizeof(int8_t);
}

}