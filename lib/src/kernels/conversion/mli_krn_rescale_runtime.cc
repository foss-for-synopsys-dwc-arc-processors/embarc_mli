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
#include "mli_krn_rescale.hpp"
#include "mli_ref_private_types.hpp"

namespace snps_arc::metaware::mli::ref {

namespace mli_krn = ::snps_arc::metaware::mli::krn;

Rescale::Rescale(void* kernel_private_data_buffer, size_t size,
                 uint64_t membases[], int num_mems) {

    MLI_ASSERT(size == sizeof(RescalePrivateData));
    RescalePrivateData private_buffer;
    memcpy(&private_buffer, kernel_private_data_buffer, sizeof(RescalePrivateData));
    MLI_ASSERT(private_buffer.size == sizeof(RescalePrivateData));

    m_input = private_buffer.input;
    m_output = private_buffer.output;

    m_tile_metadata.rescale_axis = private_buffer.rescale_axis;

    // set input tile
    const auto input = m_input.GetSubTensor();
    InternalBuffer input_internal(input.get_buf(), membases, num_mems);
    if (input.get_elem_size() == sizeof(int32_t)) {
        m_tile_metadata.input.data.mem.pi32 = input_internal.get_ptr<int32_t>();
        m_tile_metadata.input.el_type = MLI_EL_SA_32;
    } else {
        MLI_ASSERT(0);
    }
    m_tile_metadata.input.rank = input.get_rank();
    input.get_dims(m_tile_metadata.input.shape);
    input.get_mem_strides(m_tile_metadata.input.mem_stride);

    // set output tile
    const auto output = m_output.GetSubTensor();
    InternalBuffer output_internal(output.get_buf(), membases, num_mems);
    if (output.get_elem_size() == sizeof(int8_t)) {
        m_tile_metadata.output.data.mem.pi8 = output_internal.get_ptr<int8_t>();
        m_tile_metadata.output.el_type = MLI_EL_FX_8;
    } else {
        MLI_ASSERT(0);
    }
    m_tile_metadata.output.rank = output.get_rank();
    output.get_dims(m_tile_metadata.output.shape);
    output.get_mem_strides(m_tile_metadata.output.mem_stride);

    // set in_bias
    InternalBuffer encoded_params_internal(private_buffer.encoded_params_buffer, membases, num_mems);
    m_tile_param_max_size = private_buffer.tile_params_max_elem_num;
    if (input.get_elem_size() == sizeof(int32_t)) {
        encoded_params_internal.set_elem_size(sizeof(int32_t));
        if(m_tile_metadata.rescale_axis < 0) { // per-tensor
            m_tile_metadata.in_bias.rank = 0;
            m_tile_metadata.in_bias.data.mem.i32 = encoded_params_internal.read<int32_t>(0);
        } else { // per-axis
            m_tile_metadata.in_bias.rank = 1;
            m_tile_metadata.in_bias.data.mem.pi32 = encoded_params_internal.get_ptr<int32_t>();
        }
        encoded_params_internal.inc(m_tile_param_max_size);

        m_tile_metadata.in_bias.el_type = MLI_EL_SA_32;
    }
    else {
        MLI_ASSERT(0);
    }

    // set scale
    encoded_params_internal.set_elem_size(sizeof(int16_t));
    if(m_tile_metadata.rescale_axis < 0) {
        m_tile_metadata.scale.rank = 0;
        m_tile_metadata.scale.data.mem.i16 = encoded_params_internal.read<int16_t>(0);
    } else {
        m_tile_metadata.scale.rank = 1;
        m_tile_metadata.scale.data.mem.pi16 = encoded_params_internal.get_ptr<int16_t>();
    }
    encoded_params_internal.inc(m_tile_param_max_size);
    m_tile_metadata.scale.el_type = MLI_EL_FX_16;

    // set shift
    encoded_params_internal.set_elem_size(sizeof(int8_t));
    if (m_tile_metadata.rescale_axis < 0) {
        m_tile_metadata.shift.rank = 0;
        m_tile_metadata.shift.data.mem.i8 = encoded_params_internal.read<int8_t>(0);
    } else {
        m_tile_metadata.shift.rank = 1;
        m_tile_metadata.shift.data.mem.pi8 = encoded_params_internal.get_ptr<int8_t>();
    }
    encoded_params_internal.inc(m_tile_param_max_size);

    m_tile_metadata.shift.el_type = MLI_EL_FX_8;

    // set out_bias
    if (output.get_elem_size() == sizeof(int8_t)) {
        encoded_params_internal.set_elem_size(sizeof(int8_t));
        if(m_tile_metadata.rescale_axis < 0) {
            m_tile_metadata.out_bias.rank = 0;
            m_tile_metadata.out_bias.data.mem.i8 = encoded_params_internal.read<int8_t>(0);
        } else {
            m_tile_metadata.out_bias.rank = 1;
            m_tile_metadata.out_bias.data.mem.pi8 = encoded_params_internal.get_ptr<int8_t>();
        }
            
        m_tile_metadata.out_bias.el_type = MLI_EL_FX_8;
    }
    else {
        MLI_ASSERT(0);
    }

}

mli_status Rescale::Issue() {

    uint32_t out_elem_size = m_output.get_tensor().get_elem_size();
    switch(m_input.get_tensor().get_elem_size()) {
        // TODO: To be implemented for all configurations
        case (sizeof(int32_t)):
            if (out_elem_size == sizeof(int8_t)) {
                mli_krn::mli_krn_rescale<int32_t, int8_t>(&m_tile_metadata.input,
                                                          &m_tile_metadata.in_bias,
                                                          &m_tile_metadata.scale,
                                                          &m_tile_metadata.shift,
                                                          &m_tile_metadata.out_bias,
                                                          m_tile_metadata.rescale_axis,
                                                          &m_tile_metadata.output);
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

mli_status Rescale::Prefetch() {return MLI_STATUS_OK;}

mli_status Rescale::Update() {

  m_input.Next();
  m_output.Next();

  m_input.GetSubTensor().get_dims(m_tile_metadata.input.shape);
  m_output.GetSubTensor().get_dims(m_tile_metadata.output.shape);

  return MLI_STATUS_OK;
}

void Rescale::GetIOSizesAndOffsets(uint32_t& enc_param_size, uint32_t& inp_bias_offset, uint32_t& scale_offset,
                                   uint32_t& shift_offset, uint32_t& out_bias_offset) const {
  enc_param_size = m_tile_metadata.input.shape[kTensorChannelDim];
  inp_bias_offset = 0;
  scale_offset = m_tile_param_max_size * sizeof(int32_t);
  shift_offset = m_tile_param_max_size * (sizeof(int32_t) + sizeof(int16_t));
  out_bias_offset = m_tile_param_max_size * (sizeof(int32_t) + sizeof(int16_t) + sizeof(int8_t));
}

}  // namespace snps_arc::metaware::mli::ref
