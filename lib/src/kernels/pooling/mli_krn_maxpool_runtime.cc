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
#include "mli_iterator.hpp"


namespace snps_arc::metaware::mli::ref {

namespace mli_krn = ::mli::krn;

MaxPool2D::MaxPool2D(void* kernel_private_data_buffer, size_t size,
                     uint64_t membases[], int num_mems) {
  MLI_ASSERT(size == sizeof(MaxPool2DPrivateData));
  MaxPool2DPrivateData maxpool2d_private_buffer(kMaxPool2DId);
  memcpy(&maxpool2d_private_buffer, kernel_private_data_buffer, sizeof(MaxPool2DPrivateData));
  MLI_ASSERT(maxpool2d_private_buffer.size == sizeof(MaxPool2DPrivateData));

  m_io_elem_size = maxpool2d_private_buffer.input.get_buf().get_elem_size();
  MLI_ASSERT(m_io_elem_size == maxpool2d_private_buffer.output.get_buf().get_elem_size());

  // MaxPool2D configuration construction
  m_cfg.kernel_width = maxpool2d_private_buffer.config.kernel_size[1];
  m_cfg.kernel_height = maxpool2d_private_buffer.config.kernel_size[0];
  m_cfg.stride_width = maxpool2d_private_buffer.config.stride[1];
  m_cfg.stride_height = maxpool2d_private_buffer.config.stride[0];
  m_cfg.padding_left = maxpool2d_private_buffer.config.padding_begin[1];
  m_cfg.padding_right = maxpool2d_private_buffer.config.padding_end[1];
  m_cfg.padding_top = maxpool2d_private_buffer.config.padding_begin[0];
  m_cfg.padding_bottom = maxpool2d_private_buffer.config.padding_end[0];

  m_input_batch_offset = maxpool2d_private_buffer.input.get_mem_stride(kTensorBatchDim);
  m_output_batch_offset = maxpool2d_private_buffer.output.get_mem_stride(kTensorBatchDim);;

  m_input = maxpool2d_private_buffer.input;
  m_output = maxpool2d_private_buffer.output;
  const auto input_tile_tensor = m_input.GetSubTensor();
  const auto output_tile_tensor = m_output.GetSubTensor();

  m_tile_input.rank = 3;
  InternalBuffer input_internal(maxpool2d_private_buffer.input.get_buf(), membases, num_mems);
  if (m_io_elem_size == sizeof(int16_t)) {
    m_tile_input.el_type = MLI_EL_FX_16;
    mli_prv_tensor_set_data_ptr(&m_tile_input, input_internal.get_ptr<int16_t>());
  } else if (m_io_elem_size == sizeof(int8_t)) {
    m_tile_input.el_type = MLI_EL_FX_8;
    mli_prv_tensor_set_data_ptr(&m_tile_input, input_internal.get_ptr<int8_t>());
  } else {
    MLI_ASSERT(0);
  }

  int32_t input_strides[4];
  maxpool2d_private_buffer.input.get_mem_strides(input_strides);
  for (int i = 0; i < 3; i++) {
    m_tile_input.mem_stride[i] = input_strides[i + 1];         // BHWC -> HWC
    m_tile_input.shape[i] = input_tile_tensor.get_dim(i + 1);  // BHWC -> HWC
  }
  m_tile_input.mem_stride[3] = 0;
  m_tile_input.shape[3] = 0;

  m_tile_output.rank = m_tile_input.rank;
  InternalBuffer output_internal(maxpool2d_private_buffer.output.get_buf(), membases, num_mems);
  if (m_io_elem_size == sizeof(int16_t)) {
    m_tile_output.el_type = MLI_EL_FX_16;
    mli_prv_tensor_set_data_ptr(&m_tile_output, output_internal.get_ptr<int16_t>());
  } else if (m_io_elem_size == sizeof(int8_t)) {
    m_tile_output.el_type = MLI_EL_FX_8;
    mli_prv_tensor_set_data_ptr(&m_tile_output, output_internal.get_ptr<int8_t>());
  } else {
    MLI_ASSERT(0);
  }

  int32_t output_strides[4];
  maxpool2d_private_buffer.output.get_mem_strides(output_strides);
  for (int i = 0; i < 3; i++) {
    m_tile_output.mem_stride[i] = output_strides[i + 1];         // BHWC -> HWC
    m_tile_output.shape[i] = output_tile_tensor.get_dim(i + 1);  // BHWC -> HWC
  }
  m_tile_output.mem_stride[3] = 0;
  m_tile_output.shape[3] = 0;
  m_tile_batch_size = input_tile_tensor.get_dim(kTensorBatchDim);

  UpdateTilePaddings();
}

mli_status MaxPool2D::Issue() {
  if (m_io_elem_size == sizeof(int16_t)) {
    int16_t* in_ptr = m_tile_input.data.mem.pi16;
    int16_t* out_ptr = m_tile_output.data.mem.pi16;
    for (uint32_t i = 0; i < m_tile_batch_size; i++) {
      mli_krn::mli_krn_pool_hwc<mli_krn::MAXPOOL, int16_t, int16_t, POOL_NO_FIXED_KRN_SIZE>(&m_tile_input, &m_tile_cfg, &m_tile_output);
      m_tile_input.data.mem.pi16 += m_input_batch_offset;
      m_tile_output.data.mem.pi16 += m_output_batch_offset;
    }
    mli_prv_tensor_set_data_ptr(&m_tile_input, in_ptr);
    mli_prv_tensor_set_data_ptr(&m_tile_output, out_ptr);
  } else if (m_io_elem_size == sizeof(int8_t)) {
    int8_t* in_ptr = m_tile_input.data.mem.pi8;
    int8_t* out_ptr = m_tile_output.data.mem.pi8;
    for (uint32_t i = 0; i < m_tile_batch_size; i++) {
      mli_krn::mli_krn_pool_hwc<mli_krn::MAXPOOL, int8_t, int8_t, POOL_NO_FIXED_KRN_SIZE>(&m_tile_input, &m_tile_cfg, &m_tile_output);
      m_tile_input.data.mem.pi8 += m_input_batch_offset;
      m_tile_output.data.mem.pi8 += m_output_batch_offset;
    }
    mli_prv_tensor_set_data_ptr(&m_tile_input, in_ptr);
    mli_prv_tensor_set_data_ptr(&m_tile_output, out_ptr);
  }
  return MLI_STATUS_OK;
}

mli_status MaxPool2D::Prefetch() { return MLI_STATUS_OK; }

mli_status MaxPool2D::Update() {

  m_input.Next();
  m_output.Next();

  const auto input_tile_tensor = m_input.GetSubTensor();
  const auto output_tile_tensor = m_output.GetSubTensor();
  for (int i = 0; i < 3; i++) {
    m_tile_input.shape[i] = input_tile_tensor.get_dim(i + 1);    // BHWC -> HWC
    m_tile_output.shape[i] = output_tile_tensor.get_dim(i + 1);  // BHWC -> HWC
  }
  m_tile_batch_size = input_tile_tensor.get_dim(kTensorBatchDim);
  UpdateTilePaddings();

  return MLI_STATUS_OK;
}

void MaxPool2D::UpdateTilePaddings() {
  memcpy(&m_tile_cfg, &m_cfg, sizeof(mli_pool_cfg));

  int32_t tile_input_offsets[4];
  int32_t tile_output_offsets[4];
  m_input.get_pos(tile_input_offsets);
  m_output.get_pos(tile_output_offsets);

  const auto& input_it_cfg = m_input.get_config();
  if (tile_input_offsets[kTensorHeightDim]) m_tile_cfg.padding_top = 0;
  if (tile_input_offsets[kTensorHeightDim] + (int32_t) input_it_cfg.get_size(kTensorHeightDim) < (int32_t)m_input.get_dim(kTensorHeightDim)) {
    m_tile_cfg.padding_bottom = 0;
  }
  if (tile_input_offsets[kTensorWidthDim]) m_tile_cfg.padding_left = 0;
  if (tile_input_offsets[kTensorWidthDim] + (int32_t) input_it_cfg.get_size(kTensorWidthDim) < (int32_t)m_input.get_dim(kTensorWidthDim)) {
    m_tile_cfg.padding_right = 0;
  }
}

void MaxPool2D::GetIOSizesAndOffsets(uint32_t input_size[4], uint32_t output_size[4],
                                     int32_t input_offsets[4], int32_t output_offsets[4]){
  m_input.get_pos(input_offsets);
  input_size[kTensorBatchDim] = m_tile_batch_size;

  m_output.get_pos(output_offsets);
  output_size[kTensorBatchDim] = m_tile_batch_size;

  const auto input_tile_tensor = m_input.GetSubTensor();
  input_tile_tensor.get_dims(input_size);

  const auto output_tile_tensor = m_output.GetSubTensor();
  output_tile_tensor.get_dims(output_size);
}

}  // namespace snps_arc::metaware::mli::ref