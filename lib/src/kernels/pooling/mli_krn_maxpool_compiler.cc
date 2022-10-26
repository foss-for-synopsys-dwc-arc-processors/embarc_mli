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
                           const Tensor<NoBuffer, kPoolRank> in,
                           const PoolOpConfig &cfg,
                           const Tensor<NoBuffer, kPoolRank> output_tile_shape)
    : m_config(cfg),
      m_pd(pd) {

  DEPRECATED_METHOD

  Tensor<OffsetBuffer, kPoolRank> input_tensor(OffsetBuffer(), in);
  m_input = TensorIterator<OffsetBuffer, kPoolRank, kPoolIterRank>(input_tensor);

  Tensor<OffsetBuffer, kPoolRank> output_tensor(OffsetBuffer(),output_tile_shape);
  m_output = TensorIterator<OffsetBuffer, kPoolRank, kPoolIterRank>(output_tensor);
};

MaxPool2D_CS::MaxPool2D_CS(const lib_mli::PlatformDescription pd,
                          const TensorIterator<NoBuffer, kPoolRank, kPoolIterRank> in,
                          const PoolOpConfig& cfg,
                          const TensorIterator<NoBuffer, kPoolRank, kPoolIterRank> out)
    : m_input(in),
      m_output(out),
      m_config(cfg),
      m_pd(pd){}


unsigned MaxPool2D_CS::GetKernelPrivateDataSize() const {
  return sizeof(Pool2DPrivateData);
}

unsigned MaxPool2D_CS::GetRuntimeObjectSize() const {
  return sizeof(MaxPool2D);
}

mli_status MaxPool2D_CS::GetKernelPrivateData(
    void *kernel_private_data_buffer) {

  Pool2DPrivateData obj(kMaxPool2DId);
  obj.input = m_input;
  obj.output = m_output;
  obj.config = m_config;
  std::memcpy(kernel_private_data_buffer, (void *)&obj, sizeof(obj));

  return MLI_STATUS_OK;
}

mli_status MaxPool2D_CS::AttachBufferOffsets(const Tensor<OffsetBuffer, kPoolRank> &input,
                                             const Tensor<OffsetBuffer, kPoolRank> &output,
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
  uint32_t input_shape[kPoolRank];
  int32_t input_stride[kPoolRank];
  m_input.get_full_shape(input_shape);
  m_input.get_mem_strides(input_stride);

  return service::GetBufferSize(m_input.get_rank(), input_shape, input_stride);
}
unsigned MaxPool2D_CS::GetOutputBufferSize() const {
  uint32_t output_shape[kPoolRank];
  int32_t output_stride[kPoolRank];
  m_output.get_full_shape(output_shape);
  m_output.get_mem_strides(output_stride);

  return service::GetBufferSize(m_output.get_rank(), output_shape, output_stride);
}

}  // namespace snps_arc::metaware::mli::ref