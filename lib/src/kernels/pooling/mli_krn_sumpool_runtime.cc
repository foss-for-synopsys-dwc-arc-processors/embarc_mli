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
#include "mli_ref_private_types.hpp"
#include "mli_ref_runtime_api.hpp"
#include "mli_krn_pool_hwc.h"

namespace snps_arc::metaware::mli::ref {

namespace mli_krn = ::mli::krn;

SumPool2D::SumPool2D(void* kernel_private_data_buffer, size_t size,
                     uint64_t membases[], int num_mems) {
  MLI_ASSERT(size == sizeof(Pool2DPrivateData));
  Pool2DPrivateData private_data(kSumPool2DId);
  memcpy(&private_data, kernel_private_data_buffer, sizeof(Pool2DPrivateData));
  MLI_ASSERT(private_data.kernel_id == kSumPool2DId);
  MLI_ASSERT(private_data.size == sizeof(Pool2DPrivateData));

  m_i_elem_size = private_data.input_buffer.get_elem_size();
  m_o_elem_size = private_data.output_buffer.get_elem_size();

  MLI_ASSERT(private_data.input_b > 0);
  m_batch_number = private_data.input_b;
  m_input_batch_offset = private_data.input_b_stride;
  m_output_batch_offset = private_data.output_b_stride;

  // SumPool2D configuration construction
  m_cfg.kernel_width = private_data.kernel_width;
  m_cfg.kernel_height = private_data.kernel_height;
  m_cfg.stride_width = private_data.stride_width;
  m_cfg.stride_height = private_data.stride_height;
  m_cfg.padding_left = private_data.padding_left;
  m_cfg.padding_right = private_data.padding_right;
  m_cfg.padding_top = private_data.padding_top;
  m_cfg.padding_bottom = private_data.padding_bottom;

  {
    if (m_i_elem_size == sizeof(int8_t)) {
      InternalBuffer input_internal(private_data.input_buffer, membases, num_mems);
      m_input.el_type = MLI_EL_SA_8;
      m_input.data.mem.pi8 = input_internal.get_ptr<int8_t>();
    } else {
      MLI_ASSERT(false);
    }
    m_input.rank = 3;
    m_input.mem_stride[0] = private_data.input_h_stride;
    m_input.mem_stride[1] = private_data.input_w_stride;
    m_input.mem_stride[2] = private_data.input_c_stride;
    m_input.mem_stride[3] = 0;
    m_input.shape[0] = private_data.input_h;
    m_input.shape[1] = private_data.input_w;
    m_input.shape[2] = private_data.input_c;
    m_input.shape[3] = 0;
  }

  {
    if (m_o_elem_size == sizeof(int32_t)) {
      InternalBuffer output_internal(private_data.output_buffer, membases, num_mems);
      m_output.el_type = MLI_EL_SA_32;
      m_output.data.mem.pi32 = output_internal.get_ptr<int32_t>();
    } else {
      MLI_ASSERT(false);
    }
    m_output.rank = 3;
    m_output.mem_stride[0] = private_data.output_h_stride;
    m_output.mem_stride[1] = private_data.output_w_stride;
    m_output.mem_stride[2] = private_data.output_c_stride;
    m_output.mem_stride[3] = 0;
    m_output.shape[0] = private_data.output_h;
    m_output.shape[1] = private_data.output_w;
    m_output.shape[2] = private_data.output_c;
    m_output.shape[3] = 0;
  }
}

mli_status SumPool2D::Issue() {
  if (m_i_elem_size == sizeof(int8_t) && m_o_elem_size == sizeof(int32_t)) {
    int8_t* in_ptr = m_input.data.mem.pi8;
    int32_t* out_ptr = m_output.data.mem.pi32;
    for (uint32_t i = 0; i < m_batch_number; i++) {
      mli_krn::mli_krn_pool_hwc
        <mli_krn::SUMPOOL, int8_t, int32_t, POOL_NO_FIXED_KRN_SIZE>(
          &m_input, &m_cfg, &m_output);
      m_input.data.mem.pi8 += m_input_batch_offset;
      m_output.data.mem.pi32 += m_output_batch_offset;
    }
    m_input.data.mem.pi8 = in_ptr;
    m_output.data.mem.pi32 = out_ptr;
  } else {
    return MLI_STATUS_NOT_SUPPORTED;
  }
  return MLI_STATUS_OK;
}

mli_status SumPool2D::Prefetch() { return MLI_STATUS_OK; }

mli_status SumPool2D::Update() { return MLI_STATUS_OK; }

}  // namespace snps_arc::metaware::mli::ref