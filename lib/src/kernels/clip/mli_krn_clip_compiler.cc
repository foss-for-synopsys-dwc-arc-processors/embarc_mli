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

Clip_CS::Clip_CS(const lib_mli::PlatformDescription pd,
                 const Tensor<NoBuffer, 4> &input_shape,
                 const Tensor<NoBuffer, 4> &output_tile_shape
                 )
    : m_pd{pd}
{
  uint32_t in_shape[4];
  uint32_t out_shape[4];
  int32_t in_stride[4];
  int32_t out_stride[4];

  for (uint32_t i = 0; i < 4; ++i) {
      in_shape[i] = input_shape.get_dim(i);
      in_stride[i] = input_shape.get_mem_stride(i);
      out_shape[i] = output_tile_shape.get_dim(i);
      out_stride[i] = output_tile_shape.get_mem_stride(i);
  }

  m_input = Tensor<OffsetBuffer, 4>(OffsetBuffer(), in_shape, in_stride, input_shape.get_rank());
  m_output = Tensor<OffsetBuffer, 4>(OffsetBuffer(), out_shape, out_stride, output_tile_shape.get_rank());

  m_input_buffer_size =
  service::GetBufferSize(input_shape.get_rank(), in_shape, in_stride);
  m_output_buffer_size =
  service::GetBufferSize(output_tile_shape.get_rank(), out_shape, out_stride);

  // 2 == min_value_size() + max_value_size(); each contains one element.
  m_encoded_params_buffer_size = 2 * sizeof(int8_t);

  m_use_tiling = false;
  for (int i = 0; i < 4; i++) {
    m_tile_total_output_size[i] = 0;
    m_tile_iteration_order[i] = 0;
    m_tile_output_first_inc[i] = 0;
    m_tile_output_inc[i] = 0;
  };
}

unsigned Clip_CS::GetKernelPrivateDataSize() const {
  return sizeof(ClipPrivateData);
}

unsigned Clip_CS::GetRuntimeObjectSize() const {
  return sizeof(Clip);
}

mli_status Clip_CS::GetKernelPrivateData(void* kernel_private_data_buffer) {
  ClipPrivateData clip_opaque_obj;

  clip_opaque_obj.size = sizeof(ClipPrivateData);

  assert(m_input.get_rank() == m_output.get_rank());
  clip_opaque_obj.io_rank = m_input.get_rank();

  clip_opaque_obj.input_buffer = m_input.get_buf();
  clip_opaque_obj.output_buffer = m_output.get_buf();
  clip_opaque_obj.encoded_params_buffer = m_encoded_params;
  clip_opaque_obj.params_elem_num = m_params_elem_num;

  clip_opaque_obj.input_b = m_input.get_dim(mli::kTensorBatchDim);
  clip_opaque_obj.input_h = m_input.get_dim(mli::kTensorHeightDim);
  clip_opaque_obj.input_w = m_input.get_dim(mli::kTensorWidthDim);
  clip_opaque_obj.input_c = m_input.get_dim(mli::kTensorChannelDim);

  clip_opaque_obj.output_b = m_output.get_dim(mli::kTensorBatchDim);
  clip_opaque_obj.output_h = m_output.get_dim(mli::kTensorHeightDim);
  clip_opaque_obj.output_w = m_output.get_dim(mli::kTensorWidthDim);
  clip_opaque_obj.output_c = m_output.get_dim(mli::kTensorChannelDim);

  clip_opaque_obj.input_b_stride = m_input.get_mem_stride(mli::kTensorBatchDim);
  clip_opaque_obj.input_h_stride = m_input.get_mem_stride(mli::kTensorHeightDim);
  clip_opaque_obj.input_w_stride = m_input.get_mem_stride(mli::kTensorWidthDim);
  clip_opaque_obj.input_c_stride = m_input.get_mem_stride(mli::kTensorChannelDim);

  clip_opaque_obj.output_b_stride = m_output.get_mem_stride(mli::kTensorBatchDim);
  clip_opaque_obj.output_h_stride = m_output.get_mem_stride(mli::kTensorHeightDim);
  clip_opaque_obj.output_w_stride = m_output.get_mem_stride(mli::kTensorWidthDim);
  clip_opaque_obj.output_c_stride = m_output.get_mem_stride(mli::kTensorChannelDim);

  clip_opaque_obj.m_use_tiling = m_use_tiling;
  for (int i = 0; i < 4; i++) {
    clip_opaque_obj.m_tile_total_output_size[i] = m_tile_total_output_size[i];
    clip_opaque_obj.m_tile_iteration_order[i] = m_tile_iteration_order[i];
    clip_opaque_obj.m_tile_output_first_inc[i] = m_tile_output_first_inc[i];
    clip_opaque_obj.m_tile_output_inc[i] = m_tile_output_inc[i];
  }

  std::memcpy(kernel_private_data_buffer, (void *)&clip_opaque_obj, sizeof(clip_opaque_obj));

  return MLI_STATUS_OK;
}

mli_status Clip_CS::AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input,
                                        const Tensor<OffsetBuffer, 4> &output,
                                        const OffsetBuffer &encoded_params,
                                        const OffsetBuffer &metadata) {
  MLI_ASSERT(output.get_buf().get_size() == m_output_buffer_size * output.get_elem_size());

  m_input.set_buf(input.get_buf());
  m_output.set_buf(output.get_buf());
  m_encoded_params = encoded_params;

  return MLI_STATUS_OK;

}

mli_status Clip_CS::EncodeParams(Tensor<Buffer, 1> &min_value,
                                 Tensor<Buffer, 1> &max_value,
                                 Buffer &encoded_params) {
  // the element size of source should eqaul to the encoded one's
  assert(min_value.get_buf().get_size() + max_value.get_buf().get_size() == encoded_params.get_size());
  assert(min_value.get_buf().get_size() == max_value.get_buf().get_size());
  // TODO: support other data types
  m_params_elem_num =  min_value.get_buf().get_size();

  assert(min_value.get_elem_size() == 1);
  encoded_params.write<int8_t>(0, min_value.read<int8_t>(0));
  encoded_params.write<int8_t>(1, max_value.read<int8_t>(0));

  return MLI_STATUS_OK;
}

unsigned Clip_CS::GetEncodedParamsSize() const {
    return m_encoded_params_buffer_size;
}

unsigned Clip_CS::GetInputBufferSize() const {
  return m_input_buffer_size;
}

unsigned Clip_CS::GetOutputBufferSize() const {
  return m_output_buffer_size;
}

unsigned Clip_CS::GetParamsBufferSize() const {
    return m_encoded_params_buffer_size;
}

unsigned Clip_CS::GetDataBufferSize() const {
  return 0;
}

mli_status Clip_CS::SetIterators(uint32_t output_total_size[4],
                                 uint32_t iteration_order[4],
                                 uint32_t output_first_inc[4],
                                 uint32_t output_inc[4]) {
  m_use_tiling = true;
  for (int i = 0; i < 4; i++) {
    m_tile_total_output_size[i] = output_total_size[i];
    m_tile_iteration_order[i] = iteration_order[i];
    m_tile_output_first_inc[i] = output_first_inc[i];
    m_tile_output_inc[i] = output_inc[i];
  }
  return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref
