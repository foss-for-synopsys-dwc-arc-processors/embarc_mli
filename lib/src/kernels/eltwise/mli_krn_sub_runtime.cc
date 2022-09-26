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
#include "mli_krn_eltwise.h"
#include "mli_ref_runtime_api.hpp"

namespace snps_arc::metaware::mli::ref {

Sub::Sub(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems) {
  MLI_ASSERT(size == sizeof(EltwisePrivateData));
  EltwisePrivateData private_buffer(kSubId);
  memcpy(&private_buffer, kernel_private_data_buffer, sizeof(EltwisePrivateData));
  MLI_ASSERT(private_buffer.kernel_id == kSubId);
  MLI_ASSERT(private_buffer.size == sizeof(EltwisePrivateData));

  m_i_elem_size = private_buffer.m_in_left_buffer.get_elem_size();
  m_o_elem_size = private_buffer.m_output_buffer.get_elem_size();
  // left and right input have the same type
  MLI_ASSERT(private_buffer.m_in_right_buffer.get_elem_size() == m_i_elem_size);

  m_input_left = private_buffer.m_in_left_buffer;
  m_input_right = private_buffer.m_in_right_buffer;
  m_output = private_buffer.m_output_buffer;

  // reconstruct input tensors
  InternalBuffer left_buffer(private_buffer.m_in_left_buffer.get_buf(), membases, num_mems);
  InternalBuffer right_buffer(private_buffer.m_in_right_buffer.get_buf(), membases, num_mems);
  uint32_t left_shape[kEltwiseRank];
  uint32_t right_shape[kEltwiseRank];
  uint32_t output_shape[kEltwiseRank];
  const auto input_right_tile_tensor = private_buffer.m_in_right_buffer.GetSubTensor();
  const auto input_left_tile_tensor = private_buffer.m_in_left_buffer.GetSubTensor();
  const auto output_tile_tensor = private_buffer.m_output_buffer.GetSubTensor();

  for (uint32_t i = 0; i < private_buffer.m_output_buffer.get_rank(); ++i) {
      left_shape[i] = input_left_tile_tensor.get_dim(i);
      right_shape[i] = input_right_tile_tensor.get_dim(i);
      output_shape[i] = output_tile_tensor.get_dim(i);
  }

  int32_t left_stride[kEltwiseRank];
  int32_t right_stride[kEltwiseRank];
  int32_t output_stride[kEltwiseRank];
  private_buffer.m_in_left_buffer.get_mem_strides(left_stride);
  private_buffer.m_in_right_buffer.get_mem_strides(right_stride);
  private_buffer.m_output_buffer.get_mem_strides(output_stride);

  uint32_t m_in_left_rank = 0;
  uint32_t m_in_right_rank = 0;
  if( private_buffer.is_in_left_scalar == false )
  {
    m_in_left_rank = private_buffer.m_in_left_buffer.get_rank();
  }
  if( private_buffer.is_in_right_scalar == false )
  {
    m_in_right_rank = private_buffer.m_in_right_buffer.get_rank();
  }

  if (m_i_elem_size == sizeof(int8_t)) {
    m_tile_input_left.el_type = MLI_EL_SA_8;
    service::ReconstructTensor<int8_t>(left_buffer, m_tile_input_left, m_in_left_rank,
      left_shape, left_stride);

    m_tile_input_right.el_type = MLI_EL_SA_8;
    service::ReconstructTensor<int8_t>(right_buffer, m_tile_input_right, m_in_right_rank,
      right_shape, right_stride);
  } else if (m_i_elem_size == sizeof(int16_t)) {
    m_tile_input_left.el_type = MLI_EL_FX_16;
    service::ReconstructTensor<int16_t>(left_buffer, m_tile_input_left, m_in_left_rank,
      left_shape, left_stride);

    m_tile_input_right.el_type = MLI_EL_FX_16;
    service::ReconstructTensor<int16_t>(right_buffer, m_tile_input_right, m_in_right_rank,
      right_shape, right_stride);
  }
  else if (m_i_elem_size == sizeof(int32_t)) {
    m_tile_input_left.el_type = MLI_EL_SA_32;
    service::ReconstructTensor<int32_t>(left_buffer, m_tile_input_left, m_in_left_rank,
      left_shape, left_stride);

    m_tile_input_right.el_type = MLI_EL_SA_32;
    service::ReconstructTensor<int32_t>(right_buffer, m_tile_input_right, m_in_right_rank,
      right_shape, right_stride);
  }
   else {
    // not support yet
    MLI_ASSERT(false);
  }
  // reconstruct output tensor
  InternalBuffer output_buffer(private_buffer.m_output_buffer.get_buf(), membases, num_mems);
  if (m_o_elem_size == sizeof(int32_t)) {
    // assign data type and pointer
    m_tile_output.el_type = MLI_EL_SA_32;
    service::ReconstructTensor<int32_t>(output_buffer, m_tile_output, private_buffer.m_output_buffer.get_rank(),
      output_shape, output_stride);
  } 
  else if (m_o_elem_size == sizeof(int16_t)) {
     // assign data type and pointer
    m_tile_output.el_type = MLI_EL_FX_16;
    service::ReconstructTensor<int16_t>(output_buffer, m_tile_output, private_buffer.m_output_buffer.get_rank(),
      output_shape, output_stride);
    }
  else if (m_o_elem_size == sizeof(int8_t)) {
     // assign data type and pointer
    m_tile_output.el_type = MLI_EL_SA_8;
    service::ReconstructTensor<int8_t>(output_buffer, m_tile_output, private_buffer.m_output_buffer.get_rank(),
      output_shape, output_stride);
  }
  else {
    // not support yet
    MLI_ASSERT(false);
  }
}

mli_status Sub::Issue() {
   if (m_i_elem_size == sizeof(int8_t) && m_o_elem_size == sizeof(int8_t)) {
    ::mli::krn::eltwise_prepare_and_run
      <int8_t, int8_t, ::mli::ELTWISE_SUB, false>(&m_tile_input_left, &m_tile_input_right, &m_tile_output);
  }
  else if (m_i_elem_size == sizeof(int16_t) && m_o_elem_size == sizeof(int16_t)) {
    ::mli::krn::eltwise_prepare_and_run
      <int16_t, int16_t, ::mli::ELTWISE_SUB, false>(&m_tile_input_left, &m_tile_input_right, &m_tile_output);
  }
  else if (m_i_elem_size == sizeof(int32_t) && m_o_elem_size == sizeof(int32_t)) {
    ::mli::krn::eltwise_prepare_and_run
      <int32_t, int32_t, ::mli::ELTWISE_SUB, false>(&m_tile_input_left, &m_tile_input_right, &m_tile_output);
  } else {
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}

mli_status Sub::Prefetch() {
  return MLI_STATUS_OK;
}

mli_status Sub::Update() {
  m_input_right.Next();
  m_input_left.Next();
  m_output.Next();

  const auto input_right_tile_tensor = m_input_right.GetSubTensor();
  const auto input_left_tile_tensor = m_input_left.GetSubTensor();
  const auto output_tile_tensor = m_output.GetSubTensor();
  for (uint32_t i = 0; i < m_tile_output.rank; ++i) {
      m_tile_input_left.shape[i] = input_left_tile_tensor.get_dim(i);
      m_tile_input_right.shape[i] = input_right_tile_tensor.get_dim(i);
      m_tile_output.shape[i] = output_tile_tensor.get_dim(i);
    }
  return MLI_STATUS_OK;
}
void Sub::GetIOSizesAndOffsets(uint32_t input_left_size[kEltwiseRank], uint32_t input_right_size[kEltwiseRank], uint32_t output_size[kEltwiseRank],
                              int32_t input_left_offsets[kEltwiseRank], int32_t input_right_offsets[kEltwiseRank], int32_t output_offsets[kEltwiseRank]){
    
    m_input_left.get_pos(input_left_offsets);
    m_input_right.get_pos(input_right_offsets);
    m_output.get_pos(output_offsets);

    const auto input1_tile_tensor = m_input_left.GetSubTensor();
    input1_tile_tensor.get_dims(input_left_size);

    const auto input2_tile_tensor = m_input_right.GetSubTensor();
    input2_tile_tensor.get_dims(input_right_size);

    const auto output_tile_tensor = m_output.GetSubTensor();
    output_tile_tensor.get_dims(output_size);
}

}  // namespace snps_arc::metaware::mli::ref