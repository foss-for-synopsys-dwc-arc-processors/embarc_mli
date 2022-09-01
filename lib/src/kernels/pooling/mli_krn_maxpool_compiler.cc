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
#include "mli_ref_private_types.hpp"
#include "mli_ref_runtime_api.hpp"
#include "mli_service_functions.hpp"


namespace snps_arc::metaware::mli::ref {

MaxPool2D_CS::MaxPool2D_CS(const lib_mli::PlatformDescription pd,
                           const Tensor<NoBuffer, kMaxpoolRank> in,
                           const PoolOpConfig &cfg,
                           const Tensor<NoBuffer, kMaxpoolRank> output_tile_shape)
    : m_config(cfg),
      m_pd(pd) {

  DEPRECATED_METHOD

  uint32_t input_shape[kMaxpoolRank];
  int32_t input_stride[kMaxpoolRank];
  uint32_t output_shape[kMaxpoolRank];
  int32_t output_stride[kMaxpoolRank];
  for (unsigned dim = 0; dim < kMaxpoolRank; dim++) {
    input_shape[dim] = in.get_dim(dim);
    input_stride[dim] = in.get_mem_stride(dim);
    output_shape[dim] = output_tile_shape.get_dim(dim);
    output_stride[dim] = output_tile_shape.get_mem_stride(dim);
  }
  Tensor<OffsetBuffer, kMaxpoolRank> input_tensor(OffsetBuffer(), input_shape, input_stride);
  m_input = TensorIterator<OffsetBuffer, kMaxpoolRank, kMaxpoolIterRank>(input_tensor);

  Tensor<OffsetBuffer, kMaxpoolRank> output_tensor(OffsetBuffer(), output_shape, output_stride);
  m_output = TensorIterator<OffsetBuffer, kMaxpoolRank, kMaxpoolIterRank>(output_tensor);

  m_input_buffer_size =
    service::GetBufferSize(kMaxpoolRank, input_shape, input_stride);
  m_output_buffer_size =
    service::GetBufferSize(kMaxpoolRank, output_shape, output_stride);
};

MaxPool2D_CS::MaxPool2D_CS(const lib_mli::PlatformDescription pd,
                          const TensorIterator<NoBuffer, kMaxpoolRank, kMaxpoolIterRank> in,
                          const PoolOpConfig& cfg,
                          const TensorIterator<NoBuffer, kMaxpoolRank, kMaxpoolIterRank> out)
    : m_input(in),
      m_output(out),
      m_config(cfg),
      m_pd(pd){

  uint32_t input_shape[kMaxpoolRank];
  int32_t input_stride[kMaxpoolRank];
  uint32_t output_shape[kMaxpoolRank];
  int32_t output_stride[kMaxpoolRank];
  in.get_full_shape(input_shape);
  in.get_mem_strides(input_stride);
  out.get_full_shape(output_shape);
  out.get_mem_strides(output_stride);

  m_input_buffer_size =
    service::GetBufferSize(kMaxpoolRank, input_shape, input_stride);
  m_output_buffer_size =
    service::GetBufferSize(kMaxpoolRank, output_shape, output_stride);
}


unsigned MaxPool2D_CS::GetKernelPrivateDataSize() const {
  return sizeof(MaxPool2DPrivateData);
}

unsigned MaxPool2D_CS::GetRuntimeObjectSize() const {
  return sizeof(MaxPool2D);
}

mli_status MaxPool2D_CS::GetKernelPrivateData(
    void *kernel_private_data_buffer) {

  MaxPool2DPrivateData obj(kMaxPool2DId);
  obj.input = m_input;
  obj.output = m_output;
  obj.config = m_config;
  std::memcpy(kernel_private_data_buffer, (void *)&obj, sizeof(obj));

  return MLI_STATUS_OK;
}

mli_status MaxPool2D_CS::AttachBufferOffsets(const Tensor<OffsetBuffer, kMaxpoolRank> &input,
                                             const Tensor<OffsetBuffer, kMaxpoolRank> &output,
                                             const OffsetBuffer &data) {
  DEPRECATED_METHOD

  m_input.set_buf(input.get_buf());
  m_output.set_buf(output.get_buf());

  return MLI_STATUS_OK;
}

mli_status MaxPool2D_CS::AttachBufferOffsets(const OffsetBuffer& input,
                                             const OffsetBuffer& output,
                                             const OffsetBuffer& ctrl_buffer) {
  m_input.set_buf(input);
  m_output.set_buf(output);

  return MLI_STATUS_OK;
}

unsigned MaxPool2D_CS::GetInputBufferSize() const {
  return m_input_buffer_size;
}
unsigned MaxPool2D_CS::GetOutputBufferSize() const {
  return m_output_buffer_size;
}

}  // namespace snps_arc::metaware::mli::ref