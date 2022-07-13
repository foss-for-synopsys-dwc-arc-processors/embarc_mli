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

  // reconstruct input tensors
  if (m_i_elem_size == sizeof(int32_t)) {
    // assign data type and pointer
    m_input_left.el_type = MLI_EL_SA_32;
    m_input_left.rank = private_buffer.m_in_left_rank;
    InternalBuffer left_buffer(private_buffer.m_in_left_buffer, membases, num_mems);
    MLI_ASSERT(left_buffer.get_ptr<int32_t>() != nullptr);
    if (m_input_left.rank != 0) {
      m_input_left.data.mem.pi32 = left_buffer.get_ptr<int32_t>();
    } else {
      m_input_left.data.mem.i32 = left_buffer.read<int32_t>(0);
    }
    // assgin shape and stride
    for (uint32_t i = 0; i < m_input_left.rank; ++i) {
      m_input_left.shape[i] = private_buffer.m_in_left_shape[i];
      m_input_left.mem_stride[i] = private_buffer.m_in_left_stride[i];
    }

    // assign data type and pointer
    m_input_right.el_type = MLI_EL_SA_32;
    m_input_right.rank = private_buffer.m_in_right_rank;
    InternalBuffer right_buffer(private_buffer.m_in_right_buffer, membases, num_mems);
    MLI_ASSERT(right_buffer.get_ptr<int32_t>() != nullptr);
    if (m_input_right.rank != 0) {
      m_input_right.data.mem.pi32 = right_buffer.get_ptr<int32_t>();
    } else {
      m_input_right.data.mem.i32 = right_buffer.read<int32_t>(0);
    }
    // assgin shape and stride
    for (uint32_t i = 0; i < m_input_right.rank; ++i) {
      m_input_right.shape[i] = private_buffer.m_in_right_shape[i];
      m_input_right.mem_stride[i] = private_buffer.m_in_right_stride[i];
    }
  } else {
    // not support yet
    MLI_ASSERT(false);
  }

  // reconstruct output tensor
  if (m_o_elem_size == sizeof(int32_t)) {
    // assign data type and pointer
    m_output.el_type = MLI_EL_SA_32;
    m_output.rank = private_buffer.m_output_rank;
    InternalBuffer output_buffer(private_buffer.m_output_buffer, membases, num_mems);
    m_output.data.mem.pi32 = output_buffer.get_ptr<int32_t>();
    // assgin shape and stride
    for (uint32_t i = 0; i < m_output.rank; ++i) {
      m_output.shape[i] = private_buffer.m_output_shape[i];
      m_output.mem_stride[i] = private_buffer.m_output_stride[i];
    }
  } else {
    // not support yet
    MLI_ASSERT(false);
  }
}

mli_status Sub::Issue() {
  if (m_i_elem_size == sizeof(int32_t) && m_o_elem_size == sizeof(int32_t)) {
    ::mli::krn::eltwise_prepare_and_run
      <int32_t, int32_t, ::mli::ELTWISE_SUB, false>(&m_input_left, &m_input_right, &m_output);
  } else {
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}

mli_status Sub::Prefetch() {
  return MLI_STATUS_OK;
}

mli_status Sub::Update() {
  return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref