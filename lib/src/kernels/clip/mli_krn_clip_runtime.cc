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
#include "mli_krn_clip.hpp"
#include "mli_ref_private_types.hpp"

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
typedef vNx4accshort_t mli_8x8_accu_t;
#else
typedef mli_acc32_t mli_8x8_accu_t;
#endif

namespace snps_arc::metaware::mli::ref {

namespace mli_krn = ::snps_arc::metaware::mli::krn;

Clip::Clip(void* kernel_private_data_buffer,
                               size_t size,
                               uint64_t membases[], int num_mems) {
  MLI_ASSERT(size == sizeof(ClipPrivateData));
  ClipPrivateData  private_data;
  memcpy(&private_data, kernel_private_data_buffer, sizeof(ClipPrivateData));
  MLI_ASSERT(private_data.kernel_id == kClipId);
  MLI_ASSERT(private_data.size == sizeof(ClipPrivateData));

  // element size for input, output in bytes
  m_in_elem_size = private_data.input_buffer.get_elem_size();
  m_out_elem_size = private_data.output_buffer.get_elem_size();
  assert(num_mems > 0);

  // 0. Reconstruct Input Tensor
  // TODO: Move partly or all into a helper function and use for each tensor
  {
    if (m_in_elem_size == sizeof(int8_t)) {
    // assign data type and pointer
      InternalBuffer input_internal(private_data.input_buffer, membases, num_mems);
      m_input.el_type = MLI_EL_SA_8;
      m_input.data.mem.pi8 = input_internal.get_ptr<int8_t>();
    } else {
      MLI_ASSERT(false);
    }

    m_input.rank = private_data.io_rank;
    m_input.shape[0] = private_data.input_b;
    m_input.shape[1] = private_data.input_h;
    m_input.shape[2] = private_data.input_w;
    m_input.shape[3] = private_data.input_c;
    m_input.mem_stride[0] = private_data.input_b_stride;
    m_input.mem_stride[1] = private_data.input_h_stride;
    m_input.mem_stride[2] = private_data.input_w_stride;
    m_input.mem_stride[3] = private_data.input_c_stride;
  }

  // 1. Reconstruct Output Tensor
  {
    if (m_out_elem_size == sizeof(int8_t)) {
    // assign data type and pointer
      InternalBuffer output_internal(private_data.output_buffer, membases, num_mems);
      m_output.el_type = MLI_EL_SA_8;
      m_output.data.mem.pi8 = output_internal.get_ptr<int8_t>();

    } else {
        MLI_ASSERT(false);
    }

    m_output.rank = private_data.io_rank;
    m_output.shape[0] = private_data.output_b;
    m_output.shape[1] = private_data.output_h;
    m_output.shape[2] = private_data.output_w;
    m_output.shape[3] = private_data.output_c;
    m_output.mem_stride[0] = private_data.output_b_stride;
    m_output.mem_stride[1] = private_data.output_h_stride;
    m_output.mem_stride[2] = private_data.output_w_stride;
    m_output.mem_stride[3] = private_data.output_c_stride;

  }

  InternalBuffer encoded_params_internal(private_data.encoded_params_buffer, membases, num_mems);

  {
      // Reconstruct min Tensor
      if (m_in_elem_size == sizeof(int8_t)) {
          encoded_params_internal.set_elem_size(sizeof(int8_t));
          m_min.data.mem.pi8 = encoded_params_internal.get_ptr<int8_t>();
          m_min.el_type = MLI_EL_SA_8;
          encoded_params_internal.inc(private_data.params_elem_num);
      }
      else {
          MLI_ASSERT(0);
      }
      m_min.rank = 1;
  }

  {
      // Reconstruct max Tensor
      if (m_in_elem_size == sizeof(int8_t)) {
          encoded_params_internal.set_elem_size(sizeof(int8_t));
          m_max.data.mem.pi8 = encoded_params_internal.get_ptr<int8_t>();
          m_max.el_type = MLI_EL_SA_8;
      }
      else {
          MLI_ASSERT(0);
      }
      m_max.rank = 1;
  }


  if (private_data.m_use_tiling) {
    m_use_tiling = private_data.m_use_tiling;
    for (int i = 0; i < 4; i++) {
      m_tile_total_output_size[i] = private_data.m_tile_total_output_size[i];
      m_tile_iteration_order[i] = private_data.m_tile_iteration_order[i];
      m_tile_output_first_inc[i] = private_data.m_tile_output_first_inc[i];
      m_tile_output_inc[i] = private_data.m_tile_output_inc[i];
      m_input.shape[i] = private_data.m_tile_output_first_inc[i];
      m_output.shape[i] = private_data.m_tile_output_first_inc[i];
      m_tile_io_offsets[i] = 0;
    }
  }
  else m_use_tiling = false;
}

mli_status Clip::Issue() {

  if (m_out_elem_size == sizeof(int8_t) && (m_out_elem_size == sizeof(int8_t)) ) {
      mli_krn::mli_krn_clip<int8_t, int8_t>(&m_input,
                                            &m_min,
                                            &m_max,
                                            &m_output);
  } else if (m_out_elem_size == sizeof(int16_t)) {
      assert(0);
  }
  return MLI_STATUS_OK;
}

mli_status Clip::Prefetch() { return MLI_STATUS_OK; }

mli_status Clip::Update() { 
  if (!m_use_tiling) return MLI_STATUS_OK;

  uint32_t rank = m_input.rank;

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
    m_input.shape[i] = tile_size;
    m_output.shape[i] = tile_size;
  }

  return MLI_STATUS_OK; 
}

}  // namespace snps_arc::metaware::mli::ref
