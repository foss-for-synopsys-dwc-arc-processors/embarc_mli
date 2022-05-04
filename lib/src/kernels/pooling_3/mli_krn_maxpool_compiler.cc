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
                           const Tensor<OffsetBuffer, 4> in,
                           const PoolOpConfig &cfg,
                           const Tensor<OffsetBuffer, 4> output_tile_shape)
    : m_kernel_width(cfg.kernel_size[1]),
      m_kernel_height(cfg.kernel_size[0]),
      m_stride_width(cfg.stride[1]),
      m_stride_height(cfg.stride[0]),
      m_padding_left(cfg.padding_begin[1]),
      m_padding_right(cfg.padding_end[1]),
      m_padding_top(cfg.padding_begin[0]),
      m_padding_bottom(cfg.padding_end[0]),
      m_pd(pd) {
  for (int dim = 0; dim < 4; dim++) {
    m_input_shape[dim] = in.get_dim(dim);
    m_input_stride[dim] = in.get_mem_stride(dim);
    m_output_shape[dim] = output_tile_shape.get_dim(dim);
    m_output_stride[dim] = output_tile_shape.get_mem_stride(dim);
  }

  m_input_buffer_size =
      service::GetBufferSize(4, m_input_shape, m_input_stride);
  m_output_buffer_size =
      service::GetBufferSize(4, m_output_shape, m_output_stride);
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

  obj.io_elem_size = m_io_elem_size;
  obj.input_c = m_input_shape[0];
  obj.input_w = m_input_shape[1];
  obj.input_h = m_input_shape[2];
  obj.input_b = m_input_shape[3];
  obj.input_c_stride = m_input_stride[0];
  obj.input_w_stride = m_input_stride[1];
  obj.input_h_stride = m_input_stride[2];
  obj.input_b_stride = m_input_stride[3];
  obj.output_c = m_output_shape[0];
  obj.output_w = m_output_shape[1];
  obj.output_h = m_output_shape[2];
  obj.output_b = m_output_shape[3];
  obj.output_c_stride = m_output_stride[0];
  obj.output_w_stride = m_output_stride[1];
  obj.output_h_stride = m_output_stride[2];
  obj.output_b_stride = m_output_stride[3];
  obj.kernel_width = m_kernel_width;
  obj.kernel_height = m_kernel_height;
  obj.stride_width = m_stride_width;
  obj.stride_height = m_stride_height;
  obj.padding_left = m_padding_left;
  obj.padding_right = m_padding_right;
  obj.padding_top = m_padding_top;
  obj.padding_bottom = m_padding_bottom;
  obj.input_offset = m_input_offset;
  obj.output_offset = m_output_offset;
  obj.tensor_data_offset = m_descr_offset;
  obj.input_mem_id = m_input_mem_id;
  obj.output_mem_id = m_output_mem_id;
  obj.descr_mem_id = m_descr_mem_id;

  std::memcpy(kernel_private_data_buffer, (void *)&obj, sizeof(obj));

  return MLI_STATUS_OK;
}

mli_status MaxPool2D_CS::AttachBufferOffsets(const OffsetBuffer &input,
                                             const OffsetBuffer &output,
                                             const OffsetBuffer &data) {
  assert(input.get_size() == m_input_buffer_size * input.get_elem_size());
  assert(output.get_size() == m_output_buffer_size * output.get_elem_size());

  assert(input.get_elem_size() == output.get_elem_size());
  m_io_elem_size = input.get_elem_size();

  m_input_offset = input.get_offset();
  m_output_offset = output.get_offset();
  m_descr_offset = data.get_offset();

  m_input_mem_id = input.get_mem_idx();
  m_output_mem_id = output.get_mem_idx();
  m_descr_mem_id = data.get_mem_idx();

  return MLI_STATUS_OK;
}

unsigned MaxPool2D_CS::GetInputBufferSize() const {
  return m_input_buffer_size;
}
unsigned MaxPool2D_CS::GetOutputBufferSize() const {
  return m_output_buffer_size;
}
unsigned MaxPool2D_CS::GetDataBufferSize() const {
  return (sizeof(mli_pool_cfg) + sizeof(mli_tensor) * 2);
}

}  // namespace snps_arc::metaware::mli::ref