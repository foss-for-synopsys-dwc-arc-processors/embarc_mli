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

template <typename T>
void ReconstructTensor(InternalBuffer& internal_buffer, mli_tensor& tensor,
                         uint32_t rank, uint32_t* shape, int32_t* stride) {
  tensor.rank = rank;
  // assign pointer or val
  if (tensor.rank != 0) {
    MLI_ASSERT(internal_buffer.get_ptr<T>() != nullptr);
    mli_prv_tensor_set_data_ptr(&tensor, internal_buffer.get_ptr<T>());
  } else {
    if constexpr(sizeof(T) == sizeof(int8_t)) {
      tensor.data.mem.i8 = internal_buffer.read<T>(0);
    } else if constexpr(sizeof(T) == sizeof(int16_t)) {
      tensor.data.mem.i16 = internal_buffer.read<T>(0);
    } else if constexpr(sizeof(T) == sizeof(int32_t)) {
      tensor.data.mem.i32 = internal_buffer.read<T>(0);
    } else {
      MLI_ASSERT(false);
    }
  }
  // assgin shape and stride
  for (uint32_t i = 0; i < tensor.rank; ++i) {
    tensor.shape[i] = shape[i];
    tensor.mem_stride[i] = stride[i];
  }
}

Mul::Mul(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems) {
  MLI_ASSERT(size == sizeof(EltwisePrivateData));
  EltwisePrivateData private_buffer(kMulId);
  memcpy(&private_buffer, kernel_private_data_buffer, sizeof(EltwisePrivateData));
  MLI_ASSERT(private_buffer.kernel_id == kMulId);
  MLI_ASSERT(private_buffer.size == sizeof(EltwisePrivateData));

  m_i_elem_size = private_buffer.m_in_left_buffer.get_elem_size();
  m_o_elem_size = private_buffer.m_output_buffer.get_elem_size();
  // left and right input have the same type
  MLI_ASSERT(private_buffer.m_in_right_buffer.get_elem_size() == m_i_elem_size);

  // reconstruct input tensors
  InternalBuffer left_buffer(private_buffer.m_in_left_buffer, membases, num_mems);
  InternalBuffer right_buffer(private_buffer.m_in_right_buffer, membases, num_mems);
  if (m_i_elem_size == sizeof(int8_t)) {
    m_input_left.el_type = MLI_EL_SA_8;
    ReconstructTensor<int8_t>(left_buffer, m_input_left, private_buffer.m_in_left_rank,
      private_buffer.m_in_left_shape, private_buffer.m_in_left_stride);

    m_input_right.el_type = MLI_EL_SA_8;
    ReconstructTensor<int8_t>(right_buffer, m_input_right, private_buffer.m_in_right_rank,
      private_buffer.m_in_right_shape, private_buffer.m_in_right_stride);
  } else if (m_i_elem_size == sizeof(int16_t)) {
    m_input_left.el_type = MLI_EL_FX_16;
    ReconstructTensor<int16_t>(left_buffer, m_input_left, private_buffer.m_in_left_rank,
      private_buffer.m_in_left_shape, private_buffer.m_in_left_stride);

    m_input_right.el_type = MLI_EL_FX_16;
    ReconstructTensor<int16_t>(right_buffer, m_input_right, private_buffer.m_in_right_rank,
      private_buffer.m_in_right_shape, private_buffer.m_in_right_stride);
  } else {
    // not support yet
    MLI_ASSERT(false);
  }

  // reconstruct output tensor
  InternalBuffer output_buffer(private_buffer.m_output_buffer, membases, num_mems);
  if (m_o_elem_size == sizeof(int32_t)) {
    // assign data type and pointer
    m_output.el_type = MLI_EL_SA_32;
    ReconstructTensor<int32_t>(output_buffer, m_output, private_buffer.m_output_rank,
      private_buffer.m_output_shape, private_buffer.m_output_stride);
  } else {
    // not support yet
    MLI_ASSERT(false);
  }
}

mli_status Mul::Issue() {
  if (m_i_elem_size == sizeof(int8_t) && m_o_elem_size == sizeof(int32_t)) {
    ::mli::krn::eltwise_prepare_and_run
      <int8_t, int32_t, ::mli::ELTWISE_MUL, false>(&m_input_left, &m_input_right, &m_output);
  } else if (m_i_elem_size == sizeof(int16_t) && m_o_elem_size == sizeof(int32_t)) {
    ::mli::krn::eltwise_prepare_and_run
      <int16_t, int32_t, ::mli::ELTWISE_MUL, false>(&m_input_left, &m_input_right, &m_output);
  } else {
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}

mli_status Mul::Prefetch() {
  return MLI_STATUS_OK;
}

mli_status Mul::Update() {
  return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref