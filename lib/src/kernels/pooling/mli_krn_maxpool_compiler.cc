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

using namespace snps_arc::metaware::mli::service;

namespace snps_arc::metaware::mli::ref {

MaxPool2D_CS::MaxPool2D_CS(const lib_mli::PlatformDescription pd,
                           const Tensor<NoBuffer, KMaxpoolRank> in,
                           const PoolOpConfig &cfg,
                           const Tensor<NoBuffer, KMaxpoolRank> output_tile_shape)
    : m_config(cfg),
      m_pd(pd) {

  DEPRICATED_METHOD

  uint32_t input_shape[KMaxpoolRank];
  int32_t input_stride[KMaxpoolRank];
  uint32_t output_shape[KMaxpoolRank];
  int32_t output_stride[KMaxpoolRank];
  for (unsigned dim = 0; dim < KMaxpoolRank; dim++) {
    input_shape[dim] = in.get_dim(dim);
    input_stride[dim] = in.get_mem_stride(dim);
    output_shape[dim] = output_tile_shape.get_dim(dim);
    output_stride[dim] = output_tile_shape.get_mem_stride(dim);
  }
  Tensor<OffsetBuffer, KMaxpoolRank> input_tensor(OffsetBuffer(), input_shape, input_stride);
  m_input = TensorIterator<OffsetBuffer, KMaxpoolRank, KMaxpoolIterRank>(input_tensor);

  Tensor<OffsetBuffer, KMaxpoolRank> output_tensor(OffsetBuffer(), output_shape, output_stride);
  m_output = TensorIterator<OffsetBuffer, KMaxpoolRank, KMaxpoolIterRank>(output_tensor);

  m_input_buffer_size =
    service::GetBufferSize(KMaxpoolRank, input_shape, input_stride);
  m_output_buffer_size =
    service::GetBufferSize(KMaxpoolRank, output_shape, output_stride);
};

MaxPool2D_CS::MaxPool2D_CS(const lib_mli::PlatformDescription pd,
                          const TensorIterator<NoBuffer, KMaxpoolRank, KMaxpoolIterRank> in,
                          const PoolOpConfig& cfg,
                          const TensorIterator<NoBuffer, KMaxpoolRank, KMaxpoolIterRank> out)
    : m_input(in),
      m_output(out),
      m_config(cfg),
      m_pd(pd){

  uint32_t input_shape[KMaxpoolRank];
  int32_t input_stride[KMaxpoolRank];
  uint32_t output_shape[KMaxpoolRank];
  int32_t output_stride[KMaxpoolRank];
  in.get_full_shape(input_shape);
  in.get_mem_strides(input_stride);
  out.get_full_shape(output_shape);
  out.get_mem_strides(output_stride);

  m_input_buffer_size =
    service::GetBufferSize(KMaxpoolRank, input_shape, input_stride);
  m_output_buffer_size =
    service::GetBufferSize(KMaxpoolRank, output_shape, output_stride);
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

mli_status MaxPool2D_CS::AttachBufferOffsets(const Tensor<OffsetBuffer, KMaxpoolRank> &input,
                                             const Tensor<OffsetBuffer, KMaxpoolRank> &output,
                                             const OffsetBuffer &data) {
  DEPRICATED_METHOD

  m_input.set_buf(input.get_buf());
  m_output.set_buf(output.get_buf());

  return MLI_STATUS_OK;
}

mli_status MaxPool2D_CS::AttachBufferOffsets(const OffsetBuffer& input,
                                             const OffsetBuffer& output,
                                             const OffsetBuffer& data) {
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
unsigned MaxPool2D_CS::GetDataBufferSize() const {
  return 0;
}

mli_status MaxPool2D_CS::SetIterators(uint32_t output_total_size[KMaxpoolIterRank],
                                      uint32_t iteration_order[KMaxpoolIterRank],
                                      uint32_t input_first_inc[KMaxpoolIterRank],
                                      uint32_t input_inc[KMaxpoolIterRank],
                                      uint32_t output_first_inc[KMaxpoolIterRank],
                                      uint32_t output_inc[KMaxpoolIterRank]) {
  DEPRICATED_METHOD

  int32_t output_mem_stride[KMaxpoolRank];
  m_output.get_mem_strides(output_mem_stride);
  Tensor<OffsetBuffer, KMaxpoolRank> output_tensor(m_output.get_buf(), output_total_size, output_mem_stride);
  m_output = TensorIterator<OffsetBuffer, KMaxpoolRank, KMaxpoolIterRank>(output_tensor);

  // set common for input and output IteratorCfg parameters
  int32_t iteration_order_signed[KMaxpoolIterRank];
  int32_t count[KMaxpoolIterRank];
  for (unsigned i = 0; i < KMaxpoolIterRank; i++) {
    iteration_order_signed[i] = (int32_t)iteration_order[i];

    if (output_total_size[i] == output_first_inc[i]) count[i] = 1;
    else count[i] = 1 + (int32_t)CEIL_DIV(output_total_size[i] - output_first_inc[i], output_inc[i]);
  }

  // set part of input IteratorCfg parameters
  int32_t input_first_increment_signed[KMaxpoolIterRank];
  int32_t input_increment_signed[KMaxpoolIterRank];
  int32_t input_last_increment_signed[KMaxpoolIterRank];
  int32_t input_first_size_signed[KMaxpoolIterRank];
  int32_t input_size_signed[KMaxpoolIterRank];
  int32_t input_last_size_signed[KMaxpoolIterRank];
  for (unsigned i = 0; i < KMaxpoolIterRank; i++) {
    input_first_increment_signed[i] = (int32_t)input_first_inc[i];
    input_increment_signed[i] = (int32_t)input_inc[i];
    if (count[i] == 1) input_last_increment_signed[i] = 0;
    else {
      input_last_increment_signed[i] = -(input_increment_signed[i] * (count[i] - 2) + input_first_increment_signed[i]);
    }  
  }

  // calculate rest part of input IteratorCfg parameters
  // B
  input_first_size_signed[kTensorBatchDim] = (int32_t) input_first_inc[kTensorBatchDim];
  input_size_signed[kTensorBatchDim] = (int32_t) input_inc[kTensorBatchDim];

  // H
  uint32_t padding_y = m_config.padding_begin[0];
  bool single_tile_y = output_first_inc[kTensorHeightDim] == output_total_size[kTensorHeightDim];
  if (single_tile_y) padding_y += m_config.padding_end[0];
  input_first_size_signed[kTensorHeightDim] = (int32_t) get_conv_input_size(output_first_inc[kTensorHeightDim], padding_y,
                                                                            m_config.kernel_size[0], 1, m_config.stride[0]);
  input_first_size_signed[kTensorHeightDim] = MIN(input_first_size_signed[kTensorHeightDim], (int32_t) m_input.get_dim(kTensorHeightDim));
  if (single_tile_y) input_size_signed[kTensorHeightDim] = input_first_size_signed[kTensorHeightDim];
  else {
    input_size_signed[kTensorHeightDim] = (int32_t) get_conv_input_size(output_inc[kTensorHeightDim], 0,
                                                                        m_config.kernel_size[0], 1, m_config.stride[0]);
    input_size_signed[kTensorHeightDim] = MIN(input_size_signed[kTensorHeightDim], (int32_t)m_input.get_dim(kTensorHeightDim));
  }

  // W
  uint32_t padding_x = m_config.padding_begin[1];
  bool single_tile_x = output_first_inc[kTensorWidthDim] == output_total_size[kTensorWidthDim];
  if (single_tile_x) padding_x += m_config.padding_end[1];
  input_first_size_signed[kTensorWidthDim] = (int32_t) get_conv_input_size(output_first_inc[kTensorWidthDim], padding_x,
                                                                           m_config.kernel_size[1], 1, m_config.stride[1]);
  input_first_size_signed[kTensorWidthDim] = MIN(input_first_size_signed[kTensorWidthDim], (int32_t)m_input.get_dim(kTensorWidthDim));
  if (single_tile_x) input_size_signed[kTensorWidthDim] = input_first_size_signed[kTensorWidthDim];
  else {
    input_size_signed[kTensorWidthDim] = (int32_t) get_conv_input_size(output_inc[kTensorWidthDim], 0,
                                                                       m_config.kernel_size[1], 1, m_config.stride[1]);
    input_size_signed[kTensorWidthDim] = MIN(input_size_signed[kTensorWidthDim], (int32_t) m_input.get_dim(kTensorWidthDim));
  }

  // C
  input_first_size_signed[kTensorChannelDim] = (int32_t)input_first_inc[kTensorChannelDim];
  input_size_signed[kTensorChannelDim] = (int32_t)input_inc[kTensorChannelDim];

  for (unsigned i = 0; i < KMaxpoolIterRank; i++) {
    input_last_size_signed[i] = (int32_t)m_input.get_dim(i) + input_last_increment_signed[i];
  }

  // set part of output IteratorCfg parameters
  int32_t output_first_increment_signed[KMaxpoolIterRank];
  int32_t output_increment_signed[KMaxpoolIterRank];
  int32_t output_last_increment_signed[KMaxpoolIterRank];
  int32_t output_first_size_signed[KMaxpoolIterRank];
  int32_t output_size_signed[KMaxpoolIterRank];
  int32_t output_last_size_signed[KMaxpoolIterRank];
  for (unsigned i = 0; i < KMaxpoolIterRank; i++) {
    output_first_increment_signed[i] = (int32_t)output_first_inc[i];
    output_increment_signed[i] = (int32_t)output_inc[i];
    if (count[i] == 1) output_last_increment_signed[i] = 0;
    else {
      output_last_increment_signed[i] = -(output_increment_signed[i] * (count[i] - 2) + output_first_increment_signed[i]);
    }
    output_first_size_signed[i] = (int32_t)output_first_inc[i];
    output_size_signed[i] = (int32_t)output_inc[i];
    output_last_size_signed[i] = (int32_t) output_total_size[i] + output_last_increment_signed[i];
  }

  // create input IteratorCfg and set in to the input TensorIterator
  IteratorCfg<KMaxpoolIterRank> input_config(
    iteration_order_signed,
    count,
    input_first_increment_signed,
    input_increment_signed,
    input_last_increment_signed,
    input_first_size_signed,
    input_size_signed,
    input_last_size_signed
  );
  m_input.set_config(input_config);
  
  // create output IteratorCfg and set in to the output TensorIterator
  IteratorCfg<KMaxpoolIterRank> output_config(
    iteration_order_signed,
    count,
    output_first_increment_signed,
    output_increment_signed,
    output_last_increment_signed,
    output_first_size_signed,
    output_size_signed,
    output_last_size_signed
  );
  m_output.set_config(output_config);

  return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref