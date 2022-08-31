/*
 * Copyright 2022, Synopsys, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-3-Clause license found in
 * the LICENSE file in the root directory of this source tree.
 *
 */

#include <cstring>

#include "mli_ref_runtime_api.hpp"
#include "mli_ref_compiler_api.hpp"
#include "mli_ref_private_types.hpp"


namespace snps_arc::metaware::mli::ref {

Rescale_CS::Rescale_CS(const lib_mli::PlatformDescription pd,
                       const Tensor<NoBuffer, kRescaleRank>& input_shape,
                       const RescaleConfig &cfg,
                       const Tensor<NoBuffer, kRescaleRank>& output_tile_shape)
    : m_config (cfg),
      m_pd (pd) {


  DEPRECATED_METHOD

  uint32_t io_rank = input_shape.get_rank();
  MLI_ASSERT(io_rank == output_tile_shape.get_rank());
  MLI_ASSERT(io_rank <= kRescaleRank);

  uint32_t in_shape[kRescaleRank];
  uint32_t out_shape[kRescaleRank];
  int32_t in_stride[kRescaleRank];
  int32_t out_stride[kRescaleRank];
  input_shape.get_dims(in_shape);
  input_shape.get_mem_strides(in_stride);
  output_tile_shape.get_dims(out_shape);
  output_tile_shape.get_mem_strides(out_stride);

  uint32_t params_elem_num;
  if (cfg.axis < 0) {
    params_elem_num = 1;
  } else {
    params_elem_num = in_shape[cfg.axis];
  }

  // size_in_elements
  m_params_elem_num = params_elem_num;
  // size_in_bytes = No.of elements multplied by params elements' sizes
  m_encoded_params_buffer_size = params_elem_num *
          (sizeof(int32_t) + sizeof(int16_t) + sizeof(int8_t) + sizeof(int8_t));

  m_input_buffer_size = service::GetBufferSize(io_rank, in_shape, in_stride);
  m_output_buffer_size = service::GetBufferSize(io_rank, out_shape, out_stride);

  Tensor<OffsetBuffer, kRescaleRank> input_tensor(OffsetBuffer(), in_shape, in_stride, io_rank);
  m_input = TensorIterator<OffsetBuffer, kRescaleRank, kRescaleIterRank>(input_tensor);

  Tensor<OffsetBuffer, kRescaleRank> output_tensor(OffsetBuffer(), out_shape, out_stride, io_rank);
  m_output = TensorIterator<OffsetBuffer, kRescaleRank, kRescaleIterRank>(output_tensor);
}

Rescale_CS::Rescale_CS(const lib_mli::PlatformDescription pd,
                       const TensorIterator<NoBuffer, kRescaleRank, kRescaleIterRank>& input,
                       const RescaleConfig& cfg,
                       const TensorIterator<NoBuffer, kRescaleRank, kRescaleIterRank>& output)
  : m_config(cfg),
    m_input(input),
    m_output(output),
    m_pd(pd) {

    uint32_t io_rank = input.get_tensor().get_rank();
    MLI_ASSERT(io_rank == output.get_tensor().get_rank());
    MLI_ASSERT(io_rank <= kRescaleRank);

    uint32_t in_shape[kRescaleRank];
    uint32_t out_shape[kRescaleRank];
    int32_t in_stride[kRescaleRank];
    int32_t out_stride[kRescaleRank];
    input.get_full_shape(in_shape);
    input.get_mem_strides(in_stride);
    output.get_full_shape(out_shape);
    output.get_mem_strides(out_stride);

    m_input_buffer_size =
      service::GetBufferSize(io_rank, in_shape, in_stride);
    m_output_buffer_size =
      service::GetBufferSize(io_rank, out_shape, out_stride);

    m_params_elem_num = in_shape[io_rank - 1];
    m_encoded_params_buffer_size = m_params_elem_num *
      (sizeof(int32_t) + sizeof(int16_t) + sizeof(int8_t) + sizeof(int8_t));
}

mli_status Rescale_CS::AttachBufferOffsets(const Tensor<OffsetBuffer, kRescaleRank> &input,
                                           const Tensor<OffsetBuffer, kRescaleRank> &output,
                                           const OffsetBuffer &encoded_params,
                                           const OffsetBuffer &ctrl_buffer) {
    DEPRECATED_METHOD
  
    MLI_ASSERT(output.get_buf().get_size() == m_output_buffer_size * output.get_elem_size());

    m_input.set_buf(input.get_buf());
    m_output.set_buf(output.get_buf());
    m_encoded_params = encoded_params;

    return MLI_STATUS_OK;
}

mli_status Rescale_CS::AttachBufferOffsets(const OffsetBuffer& input,
                                           const OffsetBuffer& output,
                                           const OffsetBuffer& encoded_params,
                                           const OffsetBuffer& metadata) {

    MLI_ASSERT(input.get_size() / input.get_elem_size() == output.get_size() / output.get_elem_size());
    m_input.set_buf(input);
    m_output.set_buf(output);
    m_encoded_params = encoded_params;

    return MLI_STATUS_OK;
}

mli_status Rescale_CS::GetKernelPrivateData( void *kernel_private_data_buffer ) {

    RescalePrivateData opaque_obj;
    opaque_obj.size = sizeof(RescalePrivateData);

    assert(m_input.get_tensor().get_rank() == m_output.get_tensor().get_rank());
    
    for (uint32_t i = 0; i < m_input.get_tensor().get_rank(); i++) {
      MLI_ASSERT(m_output.get_dim(i) == m_input.get_dim(i));
    }
    uint32_t io_rank = m_input.get_tensor().get_rank();
    opaque_obj.input = m_input;
    opaque_obj.output = m_output;
    opaque_obj.encoded_params_buffer = m_encoded_params;
    opaque_obj.params_elem_num = m_params_elem_num;
    opaque_obj.rescale_axis = m_config.axis;
    opaque_obj.tile_params_max_elem_num = (uint32_t) MAX(m_input.get_config().get_first_inc(io_rank - 1),
                                                         m_input.get_config().get_inc(io_rank - 1));
    if (!opaque_obj.tile_params_max_elem_num) opaque_obj.tile_params_max_elem_num = m_params_elem_num;
    MLI_ASSERT(opaque_obj.tile_params_max_elem_num > 0);

    std::memcpy(kernel_private_data_buffer, (void *)&opaque_obj, sizeof(opaque_obj));

    return MLI_STATUS_OK;
}

mli_status Rescale_CS::EncodeParams(const Tensor<Buffer, kRescaleParamRank> &in_bias,
                                    const Tensor<Buffer, kRescaleParamRank> &scale,
                                    const Tensor<Buffer, kRescaleParamRank> &shift,
                                    const Tensor<Buffer, kRescaleParamRank> &out_bias,
                                    Buffer &encoded_params) {

    uint32_t i, j, last_count;
    uint32_t inbias_size = in_bias.get_buf().get_size();
    uint32_t scale_size = scale.get_buf().get_size();
    uint32_t shift_size = shift.get_buf().get_size();
    uint32_t outbias_size = out_bias.get_buf().get_size();

    m_params_elem_num = in_bias.get_buf().get_size() / in_bias.get_buf().get_elem_size();

    for (i = 0, j = 0; i < inbias_size; i+=4, j++) {
        int32_t val = in_bias.read<int32_t>(j);
        int8_t* pval = reinterpret_cast<int8_t*>(&val);
        encoded_params.write<int8_t>(i + 0, *(pval + 0));
        encoded_params.write<int8_t>(i + 1, *(pval + 1));
        encoded_params.write<int8_t>(i + 2, *(pval + 2));
        encoded_params.write<int8_t>(i + 3, *(pval + 3));
    }
    for (last_count = i, j = 0; i < (scale_size + last_count); i+=2, j++) {
        int16_t val = scale.read<int16_t>(j);
        int8_t* pval = reinterpret_cast<int8_t*>(&val);
        encoded_params.write<int8_t>(i + 0, *(pval + 0));
        encoded_params.write<int8_t>(i + 1, *(pval + 1));
    }
    for (last_count = i, j = 0; i < (shift_size + last_count); i+=1, j++) {
        encoded_params.write<int8_t>(i, shift.read<int8_t>(j));
    }
    for (last_count = i, j = 0; i < (outbias_size + last_count); i+=1, j++) {
        encoded_params.write<int8_t>(i, out_bias.read<int8_t>(j));
    }

    return MLI_STATUS_OK;
}

unsigned Rescale_CS::GetKernelPrivateDataSize() const {
    return sizeof(RescalePrivateData);
}

unsigned Rescale_CS::GetRuntimeObjectSize() const {
    return sizeof(Rescale);
}

unsigned Rescale_CS::GetInputBufferSize() const {
    return m_input_buffer_size;
}

unsigned Rescale_CS::GetOutputBufferSize() const {
    return m_output_buffer_size;
}

unsigned Rescale_CS::GetEncodedParamsSize() const {
    return m_encoded_params_buffer_size; // in bytes
}

mli_status Rescale_CS::SetIterators(uint32_t output_total_size[kRescaleIterRank],
                                    uint32_t iteration_order[kRescaleIterRank],
                                    uint32_t output_first_inc[kRescaleIterRank],
                                    uint32_t output_inc[kRescaleIterRank]) {
  
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
