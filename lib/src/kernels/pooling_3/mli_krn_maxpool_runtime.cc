/*
 * Copyright 2022, Synopsys, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-3-Clause license found in
 * the LICENSE file in the root directory of this source tree.
 *
 */

#include <cstring>
#include <new>

#include "mli_debug.h"
#include "mli_ref_private_types.hpp"
#include "mli_ref_runtime_api.hpp"
#include "mli_krn_pool_hwc.h"

namespace snps_arc::metaware::mli::ref {

namespace mli_krn = ::mli::krn;

MaxPool2D::MaxPool2D(PrivateData* kernel_private_data_buffer, size_t size,
                     uint64_t membases[], int num_mems) {
  MLI_ASSERT(size == sizeof(MaxPool2DPrivateData));
  MLI_ASSERT(kernel_private_data_buffer->size == sizeof(MaxPool2DPrivateData));

  MaxPool2DPrivateData* maxpool2d_private_buffer =
      static_cast<MaxPool2DPrivateData*>(kernel_private_data_buffer);

  m_io_elem_size = maxpool2d_private_buffer->io_elem_size;

  // MaxPool2D configuration construction
  m_cfg.kernel_width = maxpool2d_private_buffer->kernel_width;
  m_cfg.kernel_height = maxpool2d_private_buffer->kernel_height;
  m_cfg.stride_width = maxpool2d_private_buffer->stride_width;
  m_cfg.stride_height = maxpool2d_private_buffer->stride_height;
  m_cfg.padding_left = maxpool2d_private_buffer->padding_left;
  m_cfg.padding_right = maxpool2d_private_buffer->padding_right;
  m_cfg.padding_top = maxpool2d_private_buffer->padding_top;
  m_cfg.padding_bottom = maxpool2d_private_buffer->padding_bottom;

  assert(maxpool2d_private_buffer->input_b > 0);
  m_batch_number = maxpool2d_private_buffer->input_b;
  m_input_batch_offset = maxpool2d_private_buffer->input_b_stride;
  m_output_batch_offset = maxpool2d_private_buffer->output_b_stride;

  assert(maxpool2d_private_buffer->input_mem_id < num_mems);
  int input_mem_id = maxpool2d_private_buffer->input_mem_id;

  uint32_t input_offset = maxpool2d_private_buffer->input_offset;
  if (maxpool2d_private_buffer->io_elem_size == sizeof(int16_t)) {
    m_input.el_type = MLI_EL_FX_16;
    m_input.data.mem.pi16 =
        reinterpret_cast<int16_t*>(membases[input_mem_id] + input_offset);
  } else if (maxpool2d_private_buffer->io_elem_size == sizeof(int8_t)) {
    m_input.el_type = MLI_EL_FX_8;
    m_input.data.mem.pi8 =
        reinterpret_cast<int8_t*>(membases[input_mem_id] + input_offset);
  } else {
    assert(0);
  }

  m_input.rank = 3;  // TODO: Maybe need to get from compiler.
  m_input.mem_stride[0] = maxpool2d_private_buffer->input_h_stride;
  m_input.mem_stride[1] = maxpool2d_private_buffer->input_w_stride;
  m_input.mem_stride[2] = maxpool2d_private_buffer->input_c_stride;
  m_input.mem_stride[3] = 0;
  m_input.shape[0] = maxpool2d_private_buffer->input_h;
  m_input.shape[1] = maxpool2d_private_buffer->input_w;
  m_input.shape[2] = maxpool2d_private_buffer->input_c;
  m_input.shape[3] = 0;

  assert(maxpool2d_private_buffer->output_mem_id < num_mems);
  int output_mem_id = maxpool2d_private_buffer->output_mem_id;

  uint32_t output_offset = maxpool2d_private_buffer->output_offset;
  if (maxpool2d_private_buffer->io_elem_size == sizeof(int16_t)) {
    m_output.el_type = MLI_EL_FX_16;
    m_output.data.mem.pi16 =
        reinterpret_cast<int16_t*>(membases[output_mem_id] + output_offset);
  } else if (maxpool2d_private_buffer->io_elem_size == sizeof(int8_t)) {
    m_output.el_type = MLI_EL_FX_8;
    m_output.data.mem.pi8 =
        reinterpret_cast<int8_t*>(membases[output_mem_id] + output_offset);
  } else {
    assert(0);
  }

  m_output.rank = 3;  // TODO: Maybe need to get from compiler.
  m_output.mem_stride[0] = maxpool2d_private_buffer->output_h_stride;
  m_output.mem_stride[1] = maxpool2d_private_buffer->output_w_stride;
  m_output.mem_stride[2] = maxpool2d_private_buffer->output_c_stride;
  m_output.mem_stride[3] = 0;
  m_output.shape[0] = maxpool2d_private_buffer->output_h;
  m_output.shape[1] = maxpool2d_private_buffer->output_w;
  m_output.shape[2] = maxpool2d_private_buffer->output_c;
  m_output.shape[3] = 0;
}

mli_status MaxPool2D::Issue() {
  if (m_io_elem_size == sizeof(int16_t)) {
    int16_t* in_ptr = m_input.data.mem.pi16;
    int16_t* out_ptr = m_output.data.mem.pi16;
    for (uint32_t i = 0; i < m_batch_number; i++) {
      mli_krn::mli_krn_pool_hwc<mli_krn::MAXPOOL, int16_t, POOL_NO_FIXED_KRN_SIZE>(&m_input, &m_cfg, &m_output);
      m_input.data.mem.pi16 += m_input_batch_offset;
      m_output.data.mem.pi16 += m_output_batch_offset;
    }
    m_input.data.mem.pi16 = in_ptr;
    m_output.data.mem.pi16 = out_ptr;
  } else if (m_io_elem_size == sizeof(int8_t)) {
    int8_t* in_ptr = m_input.data.mem.pi8;
    int8_t* out_ptr = m_output.data.mem.pi8;
    for (uint32_t i = 0; i < m_batch_number; i++) {
      mli_krn::mli_krn_pool_hwc<mli_krn::MAXPOOL, int8_t, POOL_NO_FIXED_KRN_SIZE>(&m_input, &m_cfg, &m_output);
      m_input.data.mem.pi8 += m_input_batch_offset;
      m_output.data.mem.pi8 += m_output_batch_offset;
    }
    m_input.data.mem.pi8 = in_ptr;
    m_output.data.mem.pi8 = out_ptr;
  }
  return MLI_STATUS_OK;
}

mli_status MaxPool2D::Prefetch() { return MLI_STATUS_OK; }

mli_status MaxPool2D::Update() { return MLI_STATUS_OK; }

}  // namespace snps_arc::metaware::mli::ref