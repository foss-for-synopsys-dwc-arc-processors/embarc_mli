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
#include "mli_compiler_api.hpp"
#include "mli_krn_pool_hwc.h"
#include "mli_prv_tensor.h"


namespace snps_arc::metaware::mli::ref {

namespace mli_krn = ::mli::krn;


MaxPool2D::MaxPool2D(void* kernel_private_data_buffer, size_t size,
                     uint64_t membases[], int num_mems) {
  MLI_ASSERT(size == sizeof(MaxPool2DPrivateData));
  MaxPool2DPrivateData maxpool2d_private_buffer;
  memcpy(&maxpool2d_private_buffer, kernel_private_data_buffer, sizeof(MaxPool2DPrivateData));
  MLI_ASSERT(maxpool2d_private_buffer.size == sizeof(MaxPool2DPrivateData));

  m_io_elem_size = maxpool2d_private_buffer.input_buffer.get_elem_size();

  // MaxPool2D configuration construction
  m_cfg.kernel_width = maxpool2d_private_buffer.kernel_width;
  m_cfg.kernel_height = maxpool2d_private_buffer.kernel_height;
  m_cfg.stride_width = maxpool2d_private_buffer.stride_width;
  m_cfg.stride_height = maxpool2d_private_buffer.stride_height;
  m_cfg.padding_left = maxpool2d_private_buffer.padding_left;
  m_cfg.padding_right = maxpool2d_private_buffer.padding_right;
  m_cfg.padding_top = maxpool2d_private_buffer.padding_top;
  m_cfg.padding_bottom = maxpool2d_private_buffer.padding_bottom;

  assert(maxpool2d_private_buffer.input_b > 0);
  m_batch_number = maxpool2d_private_buffer.input_b;
  m_input_batch_offset = maxpool2d_private_buffer.input_b_stride;
  m_output_batch_offset = maxpool2d_private_buffer.output_b_stride;

  m_input.rank = 3;
  InternalBuffer input_internal(maxpool2d_private_buffer.input_buffer, membases, num_mems);
  if (m_io_elem_size == sizeof(int16_t)) {
    m_input.el_type = MLI_EL_FX_16;
    mli_prv_tensor_set_data_ptr(&m_input, input_internal.get_ptr<int16_t>());
  } else if (m_io_elem_size == sizeof(int8_t)) {
    m_input.el_type = MLI_EL_FX_8;
    mli_prv_tensor_set_data_ptr(&m_input, input_internal.get_ptr<int8_t>());
  } else {
    assert(0);
  }

  m_input.mem_stride[0] = maxpool2d_private_buffer.input_h_stride;
  m_input.mem_stride[1] = maxpool2d_private_buffer.input_w_stride;
  m_input.mem_stride[2] = maxpool2d_private_buffer.input_c_stride;
  m_input.mem_stride[3] = 0;
  
  m_use_tiling = maxpool2d_private_buffer.m_tile_first_size[kTensorBatchDim] > 0;
  if (m_use_tiling) {
    m_input.shape[0] = maxpool2d_private_buffer.m_tile_first_size[kTensorHeightDim];
    m_input.shape[1] = maxpool2d_private_buffer.m_tile_first_size[kTensorWidthDim];
    m_input.shape[2] = maxpool2d_private_buffer.m_tile_first_size[kTensorChannelDim];
  }
  else {
    m_input.shape[0] = maxpool2d_private_buffer.input_h;
    m_input.shape[1] = maxpool2d_private_buffer.input_w;
    m_input.shape[2] = maxpool2d_private_buffer.input_c;
  }
  m_input.shape[3] = 0;

  m_output.rank = m_input.rank;
  InternalBuffer output_internal(maxpool2d_private_buffer.output_buffer, membases, num_mems);
  if (m_io_elem_size == sizeof(int16_t)) {
    m_output.el_type = MLI_EL_FX_16;
    mli_prv_tensor_set_data_ptr(&m_output, output_internal.get_ptr<int16_t>());
  } else if (m_io_elem_size == sizeof(int8_t)) {
    m_output.el_type = MLI_EL_FX_8;
    mli_prv_tensor_set_data_ptr(&m_output, output_internal.get_ptr<int8_t>());
  } else {
    assert(0);
  }

  m_output.mem_stride[0] = maxpool2d_private_buffer.output_h_stride;
  m_output.mem_stride[1] = maxpool2d_private_buffer.output_w_stride;
  m_output.mem_stride[2] = maxpool2d_private_buffer.output_c_stride;
  m_output.mem_stride[3] = 0;
  m_output.shape[0] = maxpool2d_private_buffer.output_h;
  m_output.shape[1] = maxpool2d_private_buffer.output_w;
  m_output.shape[2] = maxpool2d_private_buffer.output_c;
  m_output.shape[3] = 0;

  if (m_use_tiling) {
    m_tile_total_input_size[kTensorBatchDim] = maxpool2d_private_buffer.input_b;
    m_tile_total_input_size[kTensorHeightDim] = maxpool2d_private_buffer.input_h;
    m_tile_total_input_size[kTensorWidthDim] = maxpool2d_private_buffer.input_w;
    m_tile_total_input_size[kTensorChannelDim] = maxpool2d_private_buffer.input_c;
    for (int i = 0; i < 4; i++) {
      m_tile_total_output_size[i] = maxpool2d_private_buffer.m_tile_total_output_size[i];
      m_tile_iteration_order[i] = maxpool2d_private_buffer.m_tile_iteration_order[i];
      m_tile_first_size[i] = maxpool2d_private_buffer.m_tile_first_size[i];
      m_tile_size[i] = maxpool2d_private_buffer.m_tile_size[i];
      m_tile_input_first_inc[i] = maxpool2d_private_buffer.m_tile_input_first_inc[i];
      m_tile_input_inc[i] = maxpool2d_private_buffer.m_tile_input_inc[i];
      m_tile_output_first_inc[i] = maxpool2d_private_buffer.m_tile_output_first_inc[i];
      m_tile_output_inc[i] = maxpool2d_private_buffer.m_tile_output_inc[i];
      m_tile_input_offsets[i] = 0;
      m_tile_output_offsets[i] = 0;
    }
  }
  UpdateTilePaddings();
}

mli_status MaxPool2D::Issue() {
  uint32_t cur_batch_size = m_batch_number;
  if (m_use_tiling) {
    cur_batch_size = !m_tile_input_offsets[kTensorBatchDim] ? m_tile_first_size[kTensorBatchDim] : m_tile_size[kTensorBatchDim];
  }

  if (m_io_elem_size == sizeof(int16_t)) {
    int16_t* in_ptr = m_input.data.mem.pi16;
    int16_t* out_ptr = m_output.data.mem.pi16;
    for (uint32_t i = 0; i < cur_batch_size; i++) {
      mli_krn::mli_krn_pool_hwc<mli_krn::MAXPOOL, int16_t, int16_t, POOL_NO_FIXED_KRN_SIZE>(&m_input, &m_tile_cfg, &m_output);
      m_input.data.mem.pi16 += m_input_batch_offset;
      m_output.data.mem.pi16 += m_output_batch_offset;
    }
    mli_prv_tensor_set_data_ptr(&m_input, in_ptr);
    mli_prv_tensor_set_data_ptr(&m_output, out_ptr);
  } else if (m_io_elem_size == sizeof(int8_t)) {
    int8_t* in_ptr = m_input.data.mem.pi8;
    int8_t* out_ptr = m_output.data.mem.pi8;
    for (uint32_t i = 0; i < cur_batch_size; i++) {
      mli_krn::mli_krn_pool_hwc<mli_krn::MAXPOOL, int8_t, int8_t, POOL_NO_FIXED_KRN_SIZE>(&m_input, &m_tile_cfg, &m_output);
      m_input.data.mem.pi8 += m_input_batch_offset;
      m_output.data.mem.pi8 += m_output_batch_offset;
    }
    mli_prv_tensor_set_data_ptr(&m_input, in_ptr);
    mli_prv_tensor_set_data_ptr(&m_output, out_ptr);
  }
  return MLI_STATUS_OK;
}

mli_status MaxPool2D::Prefetch() { return MLI_STATUS_OK; }

mli_status MaxPool2D::Update() {

  if (!m_use_tiling) return MLI_STATUS_OK;

  // update state with current tile sizes
  for (int i = 0; i < 4; i++) {
    int axis = m_tile_iteration_order[i];
    bool first_tile = !m_tile_input_offsets[axis];
    m_tile_input_offsets[axis] += (first_tile ? m_tile_input_first_inc[axis] : m_tile_input_inc[axis]);
    m_tile_output_offsets[axis] += (first_tile ? m_tile_output_first_inc[axis] : m_tile_output_inc[axis]);

    if (m_tile_output_offsets[axis] >= m_tile_total_output_size[axis]) {
      // end of this axis, reset this axis iterator
      m_tile_input_offsets[axis] = 0;
      m_tile_output_offsets[axis] = 0;
    }
    else {
      // not end of this axis
      break;
    }
  }

  // update next tile sizes
  // m_input and m_output are mli_tensor HWC layout
  // m_tile_input_offsets, m_tile_size, .. are BHWC layout
  for (int i = 0; i < 3; i++) {
    int axis = i + 1;
    bool first_tile = !m_tile_input_offsets[axis];
    uint32_t input_tile_size = first_tile ? m_tile_first_size[axis] : m_tile_size[axis];
    m_input.shape[i] = MIN(input_tile_size, m_tile_total_input_size[axis] - m_tile_input_offsets[axis]);
    uint32_t output_tile_size = first_tile ? m_tile_output_first_inc[axis] : m_tile_output_inc[axis];
    m_output.shape[i] = MIN(output_tile_size, m_tile_total_output_size[axis] - m_tile_output_offsets[axis]);
  }
  UpdateTilePaddings();
  return MLI_STATUS_OK;
}

void MaxPool2D::UpdateTilePaddings() {
  memcpy(&m_tile_cfg, &m_cfg, sizeof(mli_pool_cfg));
  if (!m_use_tiling) return;

  if (m_tile_input_offsets[kTensorHeightDim]) m_tile_cfg.padding_top = 0;
  if (m_tile_input_offsets[kTensorHeightDim] + m_tile_size[kTensorHeightDim] < m_tile_total_input_size[kTensorHeightDim]) {
    m_tile_cfg.padding_bottom = 0;
  }
  if (m_tile_input_offsets[kTensorWidthDim]) m_tile_cfg.padding_left = 0;
  if (m_tile_input_offsets[kTensorWidthDim] + m_tile_size[kTensorWidthDim] < m_tile_total_input_size[kTensorWidthDim]) {
    m_tile_cfg.padding_right = 0;
  }
}

void MaxPool2D::get_io_sizes_and_offsets(uint32_t input_size[4], uint32_t output_size[4],
                                         uint32_t input_offsets[4], uint32_t output_offsets[4]) const {
    input_size[0] = 1;
    output_size[0] = 1;
    input_offsets[0] = 0;
    output_offsets[0] = 0;
    for (int i = 1; i < 4; i++) {
        input_size[i] = m_input.shape[i - 1];
        output_size[i] = m_output.shape[i - 1];
        input_offsets[i] = m_tile_input_offsets[i];
        output_offsets[i] = m_tile_output_offsets[i];
    }
}

}  // namespace snps_arc::metaware::mli::ref