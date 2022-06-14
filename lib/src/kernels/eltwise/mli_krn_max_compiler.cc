/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#include <cstring>

#include "mli_ref_compiler_api.hpp"
#include "mli_ref_runtime_api.hpp"
#include "mli_ref_private_types.hpp"

namespace snps_arc::metaware::mli::ref {

Max_CS::Max_CS(const lib_mli::PlatformDescription pd,
               const Tensor<NoBuffer, 4> in_left,
               const Tensor<NoBuffer, 4> in_right,
               const Tensor<NoBuffer, 4> output_tile_shape) : m_pd(pd) {
  uint32_t in_left_shape[4];
  uint32_t in_right_shape[4];
  uint32_t output_shape[4];
  int32_t in_left_stride[4];
  int32_t in_right_stride[4];
  int32_t output_stride[4];

  for (size_t i = 0; i < 4; ++i) {
    in_left_shape[i] = in_left.get_dim(i);
    in_left_stride[i] = in_left.get_mem_stride(i);

    in_right_shape[i] = in_right.get_dim(i);
    in_right_stride[i] = in_right.get_mem_stride(i);

    output_shape[i] = output_tile_shape.get_dim(i);
    output_stride[i] = output_tile_shape.get_mem_stride(i);

    m_is_left_scalar &= (in_left_shape[i] == 1);
    m_is_right_scalar &= (in_right_shape[i] == 1);

    // verify broadcasting
    MLI_ASSERT(m_is_left_scalar ?
      output_shape[i] == in_right_shape[i] : in_left_shape[i] == in_right_shape[i]);
    MLI_ASSERT(m_is_right_scalar ?
      output_shape[i] == in_left_shape[i] : in_left_shape[i] == in_right_shape[i]);
  }

  m_in_left = Tensor<OffsetBuffer, 4>(OffsetBuffer(), in_left_shape, in_left_stride);
  m_in_right = Tensor<OffsetBuffer, 4>(OffsetBuffer(), in_right_shape, in_right_stride);
  m_output = Tensor<OffsetBuffer, 4>(OffsetBuffer(), output_shape, output_stride);

  m_in_left_buffer_size =
      service::GetBufferSize(in_left.get_rank(), in_left_shape, in_left_stride);
  m_in_right_buffer_size =
      service::GetBufferSize(in_right.get_rank(), in_right_shape, in_right_stride);
  m_output_buffer_size
      = service::GetBufferSize(output_tile_shape.get_rank(), output_shape, output_stride);
}

unsigned Max_CS::GetKernelPrivateDataSize() const {
  return sizeof(EltwisePrivateData);
}

unsigned Max_CS::GetRuntimeObjectSize() const {
  return sizeof(Max);
}

mli_status Max_CS::GetKernelPrivateData(void* kernel_private_data_buffer) {
  EltwisePrivateData obj(kMaxId);

  obj.size = sizeof(EltwisePrivateData);

  obj.m_in_left_buffer = m_in_left.get_buf();
  obj.m_in_right_buffer = m_in_right.get_buf();
  obj.m_output_buffer = m_output.get_buf();
  obj.m_metadata = m_metadata;

  MLI_ASSERT(m_in_left.get_rank() == 4);
  if (m_is_left_scalar) {
    obj.m_in_left_rank = 0;
  } else {
    obj.m_in_left_rank = m_in_left.get_rank();
  }
  for (uint32_t i = 0; i < obj.m_in_left_rank; ++i) {
    obj.m_in_left_shape[i] = m_in_left.get_dim(i);
    obj.m_in_left_stride[i] = m_in_left.get_mem_stride(i);
  }

  MLI_ASSERT(m_in_right.get_rank() == 4);
  if (m_is_right_scalar) {
    obj.m_in_right_rank = 0;
  } else {
    obj.m_in_right_rank = m_in_right.get_rank();
  }
  for (uint32_t i = 0; i < obj.m_in_right_rank; ++i) {
    obj.m_in_right_shape[i] = m_in_right.get_dim(i);
    obj.m_in_right_stride[i] = m_in_right.get_mem_stride(i);
  }

  obj.m_output_rank = m_output.get_rank();
  MLI_ASSERT(obj.m_output_rank == 4);
  for (uint32_t i = 0; i < obj.m_output_rank; ++i) {
    obj.m_output_shape[i] = m_output.get_dim(i);
    obj.m_output_stride[i] = m_output.get_mem_stride(i);
  }

  std::memcpy(kernel_private_data_buffer, (void *)&obj, sizeof(obj));

  return MLI_STATUS_OK;
}

mli_status Max_CS::AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input_left,
                                       const Tensor<OffsetBuffer, 4> &input_right,
                                       const Tensor<OffsetBuffer, 4> &output,
                                       const OffsetBuffer &data) {
  MLI_ASSERT(input_left.get_buf().get_size() >= m_in_left_buffer_size * input_left.get_elem_size());
  MLI_ASSERT(input_right.get_buf().get_size() >= m_in_right_buffer_size * input_right.get_elem_size());
  MLI_ASSERT(output.get_buf().get_size() >= m_output_buffer_size * output.get_elem_size());

  m_in_left.set_buf(input_left.get_buf());
  m_in_right.set_buf(input_right.get_buf());
  m_output.set_buf(output.get_buf());
  m_metadata = data;

  return MLI_STATUS_OK;
}

unsigned Max_CS::GetInputLeftBufferSize() {
  return m_in_left_buffer_size;
}
unsigned Max_CS::GetInputRightBufferSize() {
  return m_in_right_buffer_size;
}
unsigned Max_CS::GetOutputBufferSize() {
  return m_output_buffer_size;
}
unsigned Max_CS::GetDataBufferSize() {
  return 0;
}

}  // namespace snps_arc::metaware::mli::ref