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

    m_in_elem_size = private_buffer.input_buffer.get_elem_size();
    m_out_elem_size = private_buffer.output_buffer.get_elem_size();

    // Reconstruct configuration
    m_metadata.rescale_axis = private_buffer.rescale_axis;

    // TODO: Move partly or all into a helper function and use for each tensor
    {
        // Reconstruct Input Tensor
        InternalBuffer input_internal(private_buffer.input_buffer, membases, num_mems);

        auto& tsr = m_metadata.input;
        if (m_in_elem_size == sizeof(int32_t)) {
            tsr.data.mem.pi32 = input_internal.get_ptr<int32_t>();
            tsr.el_type = MLI_EL_SA_32;
        } else {
            MLI_ASSERT(0);
        }

        tsr.rank = private_buffer.io_rank;
        tsr.shape[0] = private_buffer.input_b;
        tsr.shape[1] = private_buffer.input_h;
        tsr.shape[2] = private_buffer.input_w;
        tsr.shape[3] = private_buffer.input_c;
        tsr.mem_stride[0] = private_buffer.input_b_stride;
        tsr.mem_stride[1] = private_buffer.input_h_stride;
        tsr.mem_stride[2] = private_buffer.input_w_stride;
        tsr.mem_stride[3] = private_buffer.input_c_stride;
    }

    {
        // Reconstruct Output Tensor
        InternalBuffer output_internal(private_buffer.output_buffer, membases, num_mems);

        auto& tsr = m_metadata.output;
        if (m_out_elem_size == sizeof(int8_t)) {
            tsr.data.mem.pi8 = output_internal.get_ptr<int8_t>();
            tsr.el_type = MLI_EL_FX_8;
        } else {
            MLI_ASSERT(0);
        }

        tsr.rank = private_buffer.io_rank;
        tsr.shape[0] = private_buffer.output_b;
        tsr.shape[1] = private_buffer.output_h;
        tsr.shape[2] = private_buffer.output_w;
        tsr.shape[3] = private_buffer.output_c;
        tsr.mem_stride[0] = private_buffer.output_b_stride;
        tsr.mem_stride[1] = private_buffer.output_h_stride;
        tsr.mem_stride[2] = private_buffer.output_w_stride;
        tsr.mem_stride[3] = private_buffer.output_c_stride;
    }

    InternalBuffer encoded_params_internal(private_buffer.encoded_params_buffer, membases, num_mems);
    m_tile_param_max_size = private_buffer.m_use_tiling ? MAX(private_buffer.m_tile_output_first_inc[kTensorChannelDim],
                                                              private_buffer.m_tile_output_inc[kTensorChannelDim]) :
                            private_buffer.params_elem_num;

    {
        // Reconstruct in_bias Tensor
        auto& tsr = m_metadata.in_bias;
        if (m_in_elem_size == sizeof(int32_t)) {
            encoded_params_internal.set_elem_size(sizeof(int32_t));
            if(m_metadata.rescale_axis < 0) { // per-tensor
                tsr.rank = 0;
                tsr.data.mem.i32 = encoded_params_internal.read<int32_t>(0);
            } else { // per-axis
                tsr.rank = 1;
                tsr.data.mem.pi32 = encoded_params_internal.get_ptr<int32_t>();
            }
            encoded_params_internal.inc(m_tile_param_max_size);

            tsr.el_type = MLI_EL_SA_32;
        }
        else {
            MLI_ASSERT(0);
        }
    }

    {
        // Reconstruct scale Tensor
        auto& tsr = m_metadata.scale;
        encoded_params_internal.set_elem_size(sizeof(int16_t));
        if(m_metadata.rescale_axis < 0) {
            tsr.rank = 0;
            tsr.data.mem.i16 = encoded_params_internal.read<int16_t>(0);
        } else {
            tsr.rank = 1;
            tsr.data.mem.pi16 = encoded_params_internal.get_ptr<int16_t>();
        }
        encoded_params_internal.inc(m_tile_param_max_size);

        tsr.el_type = MLI_EL_FX_16;
    }

    {
        // Reconstruct shift Tensor
        auto& tsr = m_metadata.shift;
        encoded_params_internal.set_elem_size(sizeof(int8_t));
        if(m_metadata.rescale_axis < 0) {
            tsr.rank = 0;
            tsr.data.mem.i8 = encoded_params_internal.read<int8_t>(0);
        } else {
            tsr.rank = 1;
            tsr.data.mem.pi8 = encoded_params_internal.get_ptr<int8_t>();
        }
        encoded_params_internal.inc(m_tile_param_max_size);

        tsr.el_type = MLI_EL_FX_8;
    }

    {
        // Reconstruct out_bias Tensor
        auto& tsr = m_metadata.out_bias;
        if (m_out_elem_size == sizeof(int8_t)) {
            encoded_params_internal.set_elem_size(sizeof(int8_t));
            if(m_metadata.rescale_axis < 0) {
                tsr.rank = 0;
                tsr.data.mem.i8 = encoded_params_internal.read<int8_t>(0);
            } else {
                tsr.rank = 1;
                tsr.data.mem.pi8 = encoded_params_internal.get_ptr<int8_t>();
            }
            
            tsr.el_type = MLI_EL_FX_8;
        }
        else {
            MLI_ASSERT(0);
        }
    }

    m_tile_metadata = m_metadata;
    if (private_buffer.m_use_tiling) {
      m_use_tiling = private_buffer.m_use_tiling;
      for (int i = 0; i < 4; i++) {
        m_tile_total_output_size[i] = private_buffer.m_tile_total_output_size[i];
        m_tile_iteration_order[i] = private_buffer.m_tile_iteration_order[i];
        m_tile_output_first_inc[i] = private_buffer.m_tile_output_first_inc[i];
        m_tile_output_inc[i] = private_buffer.m_tile_output_inc[i];
        m_tile_metadata.input.shape[i] = private_buffer.m_tile_output_first_inc[i];
        m_tile_metadata.output.shape[i] = private_buffer.m_tile_output_first_inc[i];
        m_tile_io_offsets[i] = 0;
      }
    }
    else m_use_tiling = false;
}

mli_status Rescale::Issue() {

    switch(m_in_elem_size) {
        //TODO: To be implemented for all configurations
        case (sizeof(int32_t)):
            if (m_out_elem_size == sizeof(int8_t)) {
                mli_krn::mli_krn_rescale<int32_t, int8_t>(&m_tile_metadata.input,
                                                          &m_tile_metadata.in_bias,
                                                          &m_tile_metadata.scale,
                                                          &m_tile_metadata.shift,
                                                          &m_tile_metadata.out_bias,
                                                          m_tile_metadata.rescale_axis,
                                                          &m_tile_metadata.output);
            }
            else if (m_out_elem_size == sizeof(int16_t)) {
                MLI_ASSERT(0);
            }
            break;
        case (sizeof(int16_t)):
            if (m_out_elem_size == sizeof(int8_t)) {
                MLI_ASSERT(0);
            }
            else if (m_out_elem_size == sizeof(int32_t)) {
                MLI_ASSERT(0);
            }
            break;
        case (sizeof(int8_t)):
            if (m_out_elem_size == sizeof(int8_t)) {
                MLI_ASSERT(0);
            }
            else if (m_out_elem_size == sizeof(int16_t)) {
                MLI_ASSERT(0);
            }
            else if (m_out_elem_size == sizeof(int32_t)) {
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
  if (!m_use_tiling) return MLI_STATUS_OK;

  uint32_t rank = m_tile_metadata.input.rank;

  // update state with i/o tile increments, that were used in Issue()
  for (uint32_t i = 0; i < rank; i++) {
    uint32_t axis = m_tile_iteration_order[i];
    bool first_tile = !m_tile_io_offsets[axis];
    m_tile_io_offsets[axis] += (first_tile ? m_tile_output_first_inc[axis] : m_tile_output_inc[axis]);

    if (m_tile_io_offsets[axis] >= m_tile_total_output_size[axis]) {
      // end of this axis
      m_tile_io_offsets[axis] = 0;
    }
    else {
      // not end of this axis
      break;
    }
  }

  // set i/o sizes for next call of Issue()
  for (uint32_t i = 0; i < rank; i++) {
    bool first_tile = !m_tile_io_offsets[i];
    uint32_t tile_size = MIN(first_tile ? m_tile_output_first_inc[i] : m_tile_output_inc[i],
                             m_tile_total_output_size[i] - m_tile_io_offsets[i]);
    m_tile_metadata.input.shape[i] = tile_size;
    m_tile_metadata.output.shape[i] = tile_size;
  }

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
