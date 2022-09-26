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

Add_CS::Add_CS(const lib_mli::PlatformDescription pd,
               const Tensor<NoBuffer, kEltwiseRank> in_left,
               const Tensor<NoBuffer, kEltwiseRank> in_right,
               const Tensor<NoBuffer, kEltwiseRank> output) : m_pd(pd) {
  uint32_t in_left_shape[kEltwiseRank];
  uint32_t in_right_shape[kEltwiseRank];
  uint32_t output_shape[kEltwiseRank];
  int32_t in_left_stride[kEltwiseRank];
  int32_t in_right_stride[kEltwiseRank];
  int32_t output_stride[kEltwiseRank];

  for (uint32_t i = 0; i < kEltwiseRank; ++i) {
    in_left_shape[i] = in_left.get_dim(i);
    in_left_stride[i] = in_left.get_mem_stride(i);

    in_right_shape[i] = in_right.get_dim(i);
    in_right_stride[i] = in_right.get_mem_stride(i);

    output_shape[i] = output.get_dim(i);
    output_stride[i] = output.get_mem_stride(i);

    bool is_left_bcast = (in_left_shape[i] == 1);
    m_is_left_scalar &= is_left_bcast;
    bool is_right_bcast = (in_right_shape[i] == 1);
    m_is_right_scalar &= is_right_bcast;

    // verify broadcasting
    MLI_ASSERT(is_left_bcast ?
      output_shape[i] == in_right_shape[i] : output_shape[i] == in_left_shape[i]);
    MLI_ASSERT(is_right_bcast ?
      output_shape[i] == in_left_shape[i] : output_shape[i] == in_right_shape[i]);
  }

  m_in_left  = Tensor<OffsetBuffer, kEltwiseRank>(OffsetBuffer(), in_left_shape, in_left_stride);
  m_in_right = Tensor<OffsetBuffer, kEltwiseRank>(OffsetBuffer(), in_right_shape, in_right_stride);
  m_output   = Tensor<OffsetBuffer, kEltwiseRank>(OffsetBuffer(), output_shape, output_stride);

  m_in_left_buffer_size  = service::GetBufferSize(in_left.get_rank(), in_left_shape, in_left_stride);
  m_in_right_buffer_size = service::GetBufferSize(in_right.get_rank(), in_right_shape, in_right_stride);
  m_output_buffer_size   = service::GetBufferSize(output.get_rank(), output_shape, output_stride);
}

Add_CS::Add_CS(const lib_mli::PlatformDescription pd,
               const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> in_left,
               const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> in_right,
               const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> output) : 
               m_in_left(in_left),
               m_in_right(in_right),
               m_output(output),
               m_pd(pd) {
  uint32_t in_left_shape[kEltwiseRank];
  uint32_t in_right_shape[kEltwiseRank];
  uint32_t output_shape[kEltwiseRank];
  int32_t in_left_stride[kEltwiseRank];
  int32_t in_right_stride[kEltwiseRank];
  int32_t output_stride[kEltwiseRank];
 
  in_left.get_full_shape(in_left_shape);
  in_left.get_mem_strides(in_left_stride);
  in_right.get_full_shape(in_right_shape);
  in_right.get_mem_strides(in_right_stride);
  output.get_full_shape(output_shape);
  output.get_mem_strides(output_stride);
  
  for(uint32_t i = 0; i < kEltwiseRank; ++i)
  {
    bool is_left_bcast = (in_left_shape[i] == 1);
    m_is_left_scalar &= is_left_bcast;
    bool is_right_bcast = (in_right_shape[i] == 1);
    m_is_right_scalar &= is_right_bcast;

    // verify broadcasting
    MLI_ASSERT(is_left_bcast ?
      output_shape[i] == in_right_shape[i] : output_shape[i] == in_left_shape[i]);
    MLI_ASSERT(is_right_bcast ?
      output_shape[i] == in_left_shape[i] : output_shape[i] == in_right_shape[i]);
  }
  m_in_left_buffer_size  = service::GetBufferSize(in_left.get_rank(), in_left_shape, in_left_stride);
  m_in_right_buffer_size = service::GetBufferSize(in_right.get_rank(), in_right_shape, in_right_stride);
  m_output_buffer_size   = service::GetBufferSize(output.get_rank(), output_shape, output_stride);
}

unsigned Add_CS::GetKernelPrivateDataSize() const {
  return sizeof(EltwisePrivateData);
}

unsigned Add_CS::GetRuntimeObjectSize() const {
  return sizeof(Add);
}

mli_status Add_CS::GetKernelPrivateData(void* kernel_private_data_buffer) {
  EltwisePrivateData obj(kAddId);

  obj.m_in_left_buffer =  m_in_left;
  obj.m_in_right_buffer = m_in_right;
  obj.m_output_buffer =  m_output;

  MLI_ASSERT(m_in_left.get_rank() == kEltwiseRank);
  obj.is_in_left_scalar = m_is_left_scalar;
  
  MLI_ASSERT(m_in_right.get_rank() == kEltwiseRank);
  obj.is_in_right_scalar = m_is_right_scalar;

  std::memcpy(kernel_private_data_buffer, (void *)&obj, sizeof(obj));

  return MLI_STATUS_OK;
}

mli_status Add_CS::AttachBufferOffsets(const Tensor<OffsetBuffer, kEltwiseRank> &input_left,
                                       const Tensor<OffsetBuffer, kEltwiseRank> &input_right,
                                       const Tensor<OffsetBuffer, kEltwiseRank> &output,
                                       const OffsetBuffer &ctrl_buffer) {
  MLI_ASSERT(input_left.get_buf().get_size() >= m_in_left_buffer_size * input_left.get_elem_size());
  MLI_ASSERT(input_right.get_buf().get_size() >= m_in_right_buffer_size * input_right.get_elem_size());
  MLI_ASSERT(output.get_buf().get_size() >= m_output_buffer_size * output.get_elem_size());

  m_in_left.set_buf(input_left.get_buf());
  m_in_right.set_buf(input_right.get_buf());
  m_output.set_buf(output.get_buf());

  return MLI_STATUS_OK;
}


mli_status Add_CS::AttachBufferOffsets(const OffsetBuffer &input_left,
                                       const OffsetBuffer &input_right,
                                       const OffsetBuffer &output,
                                       const OffsetBuffer &ctrl_buffer) {
  m_in_left.set_buf(input_left);
  m_in_right.set_buf(input_right);
  m_output.set_buf(output);

  return MLI_STATUS_OK;
}

unsigned Add_CS::GetInputLeftBufferSize() {
  return m_in_left_buffer_size;
}
unsigned Add_CS::GetInputRightBufferSize() {
  return m_in_right_buffer_size;
}
unsigned Add_CS::GetOutputBufferSize() {
  return m_output_buffer_size;
}


}  // namespace snps_arc::metaware::mli::ref