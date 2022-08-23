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
                 const Tensor<NoBuffer, kClipRank> &input_shape,
                 const Tensor<NoBuffer, kClipRank> &output_tile_shape
                 )
    : m_pd(pd) {

  DEPRECATED_METHOD

  uint32_t io_rank = input_shape.get_rank();
  MLI_ASSERT(io_rank == output_tile_shape.get_rank());
  MLI_ASSERT(io_rank <= kClipRank);

  uint32_t in_shape[kClipRank];
  uint32_t out_shape[kClipRank];
  int32_t in_stride[kClipRank];
  int32_t out_stride[kClipRank];
  input_shape.get_dims(in_shape);
  input_shape.get_mem_strides(in_stride);
  output_tile_shape.get_dims(out_shape);
  output_tile_shape.get_mem_strides(out_stride);

  m_input_buffer_size =
  service::GetBufferSize(input_shape.get_rank(), in_shape, in_stride);
  m_output_buffer_size =
  service::GetBufferSize(output_tile_shape.get_rank(), out_shape, out_stride);

  Tensor<OffsetBuffer, kClipRank> input_tensor(OffsetBuffer(), in_shape, in_stride, io_rank);
  m_input = TensorIterator<OffsetBuffer, kClipRank, kClipIterRank>(input_tensor);

  Tensor<OffsetBuffer, kClipRank> output_tensor(OffsetBuffer(), out_shape, out_stride, io_rank);
  m_output = TensorIterator<OffsetBuffer, kClipRank, kClipIterRank>(output_tensor);

}

Clip_CS::Clip_CS(const lib_mli::PlatformDescription pd,
                 const TensorIterator<NoBuffer, kClipRank, kClipIterRank>& input,
                 const TensorIterator<NoBuffer, kClipRank, kClipIterRank>& output) 
  : m_input(input),
    m_output(output),
    m_pd(pd) {

  uint32_t io_rank = input.get_tensor().get_rank();
  MLI_ASSERT(io_rank == output.get_tensor().get_rank());
  MLI_ASSERT(io_rank <= kClipRank);

  uint32_t in_shape[kClipRank];
  uint32_t out_shape[kClipRank];
  int32_t in_stride[kClipRank];
  int32_t out_stride[kClipRank];
  input.get_full_shape(in_shape);
  input.get_mem_strides(in_stride);
  output.get_full_shape(out_shape);
  output.get_mem_strides(out_stride);


  m_input_buffer_size =
    service::GetBufferSize(io_rank, in_shape, in_stride);
  m_output_buffer_size =
    service::GetBufferSize(io_rank, out_shape, out_stride);
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

  assert(m_input.get_tensor().get_rank() == m_output.get_tensor().get_rank());

  clip_opaque_obj.input = m_input;
  clip_opaque_obj.encoded_params_buf = m_encoded_params;
  clip_opaque_obj.output = m_output;

  std::memcpy(kernel_private_data_buffer, (void *)&clip_opaque_obj, sizeof(clip_opaque_obj));

  return MLI_STATUS_OK;
}

mli_status Clip_CS::AttachBufferOffsets(const Tensor<OffsetBuffer, kClipRank> &input,
                                        const Tensor<OffsetBuffer, kClipRank> &output,
                                        const OffsetBuffer &encoded_params,
                                        const OffsetBuffer &ctrl_buffer) {
  DEPRECATED_METHOD
  
  MLI_ASSERT(output.get_buf().get_size() == input.get_buf().get_size());
  m_input.set_buf(input.get_buf());
  m_output.set_buf(output.get_buf());
  m_encoded_params = encoded_params;

  return MLI_STATUS_OK;

}

mli_status Clip_CS::EncodeParams(Tensor<Buffer, kClipParamRank> &min_value,
                                 Tensor<Buffer, kClipParamRank> &max_value,
                                 Buffer &encoded_params) {
  // the element size of source should eqaul to the encoded one's
  assert(min_value.get_buf().get_size() + max_value.get_buf().get_size() == encoded_params.get_size());
  assert(min_value.get_buf().get_size() == max_value.get_buf().get_size());
  // TODO: support other data types

  assert(min_value.get_elem_size() == 1);
  encoded_params.write<int8_t>(0, min_value.read<int8_t>(0));
  encoded_params.write<int8_t>(1, max_value.read<int8_t>(0));

  return MLI_STATUS_OK;
}

mli_status Clip_CS::AttachBufferOffsets(const OffsetBuffer& input,
                                        const OffsetBuffer& output,
                                        const OffsetBuffer& encoded_params,
                                        const OffsetBuffer& ctrl_buffer) {
  MLI_ASSERT(output.get_size() == input.get_size());

  m_input.set_buf(input);
  m_output.set_buf(output);
  m_encoded_params = encoded_params;

  return MLI_STATUS_OK;
}

unsigned Clip_CS::GetEncodedParamsSize() const {
  return sizeof(int8_t) + sizeof(int8_t);  // min param + max param;
}

unsigned Clip_CS::GetInputBufferSize() const {
  return m_input_buffer_size;
}

unsigned Clip_CS::GetOutputBufferSize() const {
  return m_output_buffer_size;
}

unsigned Clip_CS::GetParamsBufferSize() const {
  return GetEncodedParamsSize();
}

mli_status Clip_CS::SetIterators(uint32_t output_total_size[kClipIterRank],
                                 uint32_t iteration_order[kClipIterRank],
                                 uint32_t output_first_inc[kClipIterRank],
                                 uint32_t output_inc[kClipIterRank]) {
  DEPRECATED_METHOD

  int32_t output_mem_stride[kClipRank];
  m_output.get_mem_strides(output_mem_stride);
  Tensor<OffsetBuffer, kClipRank> output_tensor(m_output.get_buf(), output_total_size, output_mem_stride);
  m_output = TensorIterator<OffsetBuffer, kClipRank, kClipIterRank>(output_tensor);

  int32_t iteration_order_signed[kClipIterRank];
  int32_t count[kClipIterRank];
  for (unsigned i = 0; i < kClipIterRank; i++) {
    iteration_order_signed[i] = (int32_t)iteration_order[i];

    if (output_total_size[i] == output_first_inc[i]) count[i] = 1;
    else count[i] = 1 + (int32_t)CEIL_DIV(output_total_size[i] - output_first_inc[i], output_inc[i]);
  }

  int32_t output_first_increment_signed[kClipIterRank];
  int32_t output_increment_signed[kClipIterRank];
  int32_t output_last_increment_signed[kClipIterRank];
  int32_t output_first_size_signed[kClipIterRank];
  int32_t output_size_signed[kClipIterRank];
  int32_t output_last_size_signed[kClipIterRank];
  for (unsigned i = 0; i < kClipIterRank; i++) {
    output_first_increment_signed[i] = (int32_t)output_first_inc[i];
    output_increment_signed[i] = (int32_t)output_inc[i];
    if (count[i] == 1) output_last_increment_signed[i] = 0;
    else {
      output_last_increment_signed[i] = service::get_last_increment(count[i], output_first_increment_signed[i], output_increment_signed[i]);
    }
    output_first_size_signed[i] = (int32_t)output_first_inc[i];
    output_size_signed[i] = (int32_t)output_inc[i];
    output_last_size_signed[i] = (int32_t)output_total_size[i] + output_last_increment_signed[i];
  }

  IteratorCfg<kClipIterRank> io_config(
    iteration_order_signed,
    count,
    output_first_increment_signed,
    output_increment_signed,
    output_last_increment_signed,
    output_first_size_signed,
    output_size_signed,
    output_last_size_signed
  );
  m_input.set_config(io_config);
  m_output.set_config(io_config);

  return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref
