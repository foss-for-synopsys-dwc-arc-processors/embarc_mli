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

MaxPool2D_CS::MaxPool2D_CS(const Tensor<OffsetBuffer, 4> in, const PoolOpConfig *cfg,
                           const Tensor<OffsetBuffer, 4> output_tile_shape)
    : kernel_height(cfg->kernel_size[0]),
      kernel_width(cfg->kernel_size[1]),
      stride_height(cfg->stride[0]),
      stride_width(cfg->stride[1]),
      padding_top(cfg->padding_begin[0]),
      padding_bottom(cfg->padding_end[0]),
      padding_left(cfg->padding_begin[1]),
      padding_right(cfg->padding_end[1]) {


  for (int dim = 0; dim < 4; dim++) {
    input_shape[dim] = in.get_dim(dim);
    input_stride[dim] = in.get_mem_stride(dim);
    output_shape[dim] = output_tile_shape.get_dim(dim);
    output_stride[dim] = output_tile_shape.get_mem_stride(dim);
  }

  input_buffer_size_ = service::GetBufferSize(4, input_shape, input_stride);
  output_buffer_size_ = service::GetBufferSize(4, output_shape, output_stride);
};

unsigned MaxPool2D_CS::GetKernelPrivateDataSize() const { 
  return sizeof(MaxPool2DPrivateData);
}

unsigned MaxPool2D_CS::GetRuntimeObjectSize() const {
  return sizeof(MaxPool2D);
}

mli_status MaxPool2D_CS::GetKernelPrivateData(void* kernel_private_data_buffer) {
  MaxPool2DPrivateData obj;

  obj.io_elem_size = io_elem_size;
  obj.input_c = input_shape[0];
  obj.input_w = input_shape[1];
  obj.input_h = input_shape[2];
  obj.input_b = input_shape[3];
  obj.input_c_stride = input_stride[0];
  obj.input_w_stride = input_stride[1];
  obj.input_h_stride = input_stride[2];
  obj.input_b_stride = input_stride[3];
  obj.output_c = output_shape[0];
  obj.output_w = output_shape[1];
  obj.output_h = output_shape[2];
  obj.output_b = output_shape[3];
  obj.output_c_stride = output_stride[0];
  obj.output_w_stride = output_stride[1];
  obj.output_h_stride = output_stride[2];
  obj.output_b_stride = output_stride[3];
  obj.kernel_width = kernel_width;
  obj.kernel_height = kernel_height;
  obj.stride_width = stride_width;
  obj.stride_height = stride_height;
  obj.padding_left = padding_left;
  obj.padding_right = padding_right;
  obj.padding_top = padding_top;
  obj.padding_bottom = padding_bottom;
  obj.input_offset = input_offset_;
  obj.output_offset = output_offset_;
  obj.tensor_data_offset = descr_offset_;
  obj.input_mem_id = input_mem_id_;
  obj.output_mem_id = output_mem_id_;
  obj.descr_mem_id = descr_mem_id_;

  std::memcpy(kernel_private_data_buffer, (void *)&obj, sizeof(obj));

  return MLI_STATUS_OK;
}

mli_status MaxPool2D_CS::AttachBufferOffsets(const OffsetBuffer &input,
                                             const OffsetBuffer &output,
                                             const OffsetBuffer &data) {
  assert(input.get_size() == input_buffer_size_);
  assert(output.get_size() == output_buffer_size_);
  
  assert(input.get_elem_size() == output.get_elem_size());
  io_elem_size = input.get_elem_size();

  input_offset_ = input.get_offset();
  output_offset_ = output.get_offset();
  descr_offset_ = data.get_offset();

  input_mem_id_ = input.get_mem_idx();
  output_mem_id_ = output.get_mem_idx();
  descr_mem_id_ = data.get_mem_idx();

  return MLI_STATUS_OK;
}

unsigned MaxPool2D_CS::GetInputBufferSize() const {
  return input_buffer_size_;
}
unsigned MaxPool2D_CS::GetOutputBufferSize() const {
  return output_buffer_size_;
}
unsigned MaxPool2D_CS::GetDataBufferSize() const {
  return (sizeof(mli_pool_cfg) + sizeof(mli_tensor) * 2);
}

}  // namespace snps_arc::metaware::mli::ref