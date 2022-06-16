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

namespace snps_arc::metaware::mli::ref {

MaxPool2D_CS::MaxPool2D_CS(const lib_mli::PlatformDescription pd,
                           const Tensor<NoBuffer, 4> in,
                           const PoolOpConfig &cfg,
                           const Tensor<NoBuffer, 4> output_tile_shape)
    : m_kernel_width(cfg.kernel_size[1]),
      m_kernel_height(cfg.kernel_size[0]),
      m_stride_width(cfg.stride[1]),
      m_stride_height(cfg.stride[0]),
      m_padding_left(cfg.padding_begin[1]),
      m_padding_right(cfg.padding_end[1]),
      m_padding_top(cfg.padding_begin[0]),
      m_padding_bottom(cfg.padding_end[0]),
      m_pd(pd) {
  
  uint32_t input_shape[4];
  int32_t input_stride[4];
  uint32_t output_shape[4];
  int32_t output_stride[4];
  for (int dim = 0; dim < 4; dim++) {
    input_shape[dim] = in.get_dim(dim);
    input_stride[dim] = in.get_mem_stride(dim);
    output_shape[dim] = output_tile_shape.get_dim(dim);
    output_stride[dim] = output_tile_shape.get_mem_stride(dim);
  }

  m_in = Tensor<OffsetBuffer, 4>(OffsetBuffer(), input_shape, input_stride);
  m_output = Tensor<OffsetBuffer, 4>(OffsetBuffer(), output_shape, output_stride);

  m_input_buffer_size =
      service::GetBufferSize(4, input_shape, input_stride);
  m_output_buffer_size =
      service::GetBufferSize(4, output_shape, output_stride);

  for (int i = 0; i < 4; i++) {
    m_tile_first_size[i] = 0;
    m_tile_size[i] = 0;
    m_tile_total_output_size[i] = 0;
    m_tile_iteration_order[i] = 0;
    m_tile_input_first_inc[i] = 0;
    m_tile_input_inc[i] = 0;
    m_tile_output_first_inc[i] = 0;
    m_tile_output_inc[i] = 0;
  };

};

unsigned MaxPool2D_CS::GetKernelPrivateDataSize() const {
  return sizeof(MaxPool2DPrivateData);
}

unsigned MaxPool2D_CS::GetRuntimeObjectSize() const {
  return sizeof(MaxPool2D);
}

mli_status MaxPool2D_CS::GetKernelPrivateData(
    void *kernel_private_data_buffer) {
  MaxPool2DPrivateData obj;

  obj.size = sizeof(MaxPool2DPrivateData);
  
  obj.input_buffer = m_in.get_buf();
  obj.output_buffer = m_output.get_buf();

  obj.input_c = m_in.get_dim(kTensorChannelDim);
  obj.input_w = m_in.get_dim(kTensorWidthDim);
  obj.input_h = m_in.get_dim(kTensorHeightDim);
  obj.input_b = m_in.get_dim(kTensorBatchDim);

  obj.input_c_stride = m_in.get_mem_stride(kTensorChannelDim);
  obj.input_w_stride = m_in.get_mem_stride(kTensorWidthDim);
  obj.input_h_stride = m_in.get_mem_stride(kTensorHeightDim);
  obj.input_b_stride = m_in.get_mem_stride(kTensorBatchDim);

  obj.output_c = m_output.get_dim(kTensorChannelDim);
  obj.output_w = m_output.get_dim(kTensorWidthDim);
  obj.output_h = m_output.get_dim(kTensorHeightDim);
  obj.output_b = m_output.get_dim(kTensorBatchDim);

  obj.output_c_stride = m_output.get_mem_stride(kTensorChannelDim);
  obj.output_w_stride = m_output.get_mem_stride(kTensorWidthDim);
  obj.output_h_stride = m_output.get_mem_stride(kTensorHeightDim);
  obj.output_b_stride = m_output.get_mem_stride(kTensorBatchDim);

  obj.kernel_width = m_kernel_width;
  obj.kernel_height = m_kernel_height;
  obj.stride_width = m_stride_width;
  obj.stride_height = m_stride_height;
  obj.padding_left = m_padding_left;
  obj.padding_right = m_padding_right;
  obj.padding_top = m_padding_top;
  obj.padding_bottom = m_padding_bottom;

  for (int i = 0; i < 4; i++) {
    obj.m_tile_total_output_size[i] = m_tile_total_output_size[i];
    obj.m_tile_iteration_order[i] = m_tile_iteration_order[i];
    obj.m_tile_input_first_inc[i] = m_tile_input_first_inc[i];
    obj.m_tile_first_size[i] = m_tile_first_size[i];
    obj.m_tile_size[i] = m_tile_size[i];
    obj.m_tile_input_inc[i] = m_tile_input_inc[i];
    obj.m_tile_output_first_inc[i] = m_tile_output_first_inc[i];
    obj.m_tile_output_inc[i] = m_tile_output_inc[i];
  }


  std::memcpy(kernel_private_data_buffer, (void *)&obj, sizeof(obj));

  return MLI_STATUS_OK;
}

mli_status MaxPool2D_CS::AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input,
                                             const Tensor<OffsetBuffer, 4> &output,
                                             const OffsetBuffer &data) {

  assert(input.get_buf().get_size() == m_input_buffer_size * input.get_elem_size());
  assert(output.get_buf().get_size() == m_output_buffer_size * output.get_elem_size());

  m_in.set_buf(input.get_buf());
  m_output.set_buf(output.get_buf());

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

mli_status MaxPool2D_CS::SetIterators(uint32_t total_output_size[4], uint32_t iteration_order[4],
                                      uint32_t first_tile_size[4], uint32_t tile_size[4],
                                      uint32_t input_first_inc[4], uint32_t input_inc[4],
                                      uint32_t output_first_inc[4], uint32_t output_inc[4]) {
  for (int i = 0; i < 4; i++) {
    m_tile_first_size[i] = first_tile_size[i];
    m_tile_size[i] = tile_size[i];
    m_tile_total_output_size[i] = total_output_size[i];
    m_tile_iteration_order[i] = iteration_order[i];
    m_tile_input_first_inc[i] = input_first_inc[i];
    m_tile_input_inc[i] = input_inc[i];
    m_tile_output_first_inc[i] = output_first_inc[i];
    m_tile_output_inc[i] = output_inc[i];
  }

  return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref