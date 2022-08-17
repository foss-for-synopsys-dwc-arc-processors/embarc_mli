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

DepthwiseConv2d_CS::DepthwiseConv2d_CS(const lib_mli::PlatformDescription pd,
                                       const Tensor<NoBuffer, 4> &in,
                                       const Tensor<NoBuffer, 3> &weights,
                                       const DwConv2DConfig &cfg,
                                       const Tensor<NoBuffer, 4> &output_tile_shape)
    : m_pd{pd} {
  uint32_t input_shape[4];
  uint32_t output_shape[4];
  int32_t input_stride[4];
  int32_t output_stride[4];
  for (uint32_t i = 0; i < 4; ++i) {
    input_shape[i] = in.get_dim(i);
    input_stride[i] = in.get_mem_stride(i);
    output_shape[i] = output_tile_shape.get_dim(i);
    output_stride[i] = output_tile_shape.get_mem_stride(i);
  }

  uint32_t weights_shape[3];
  int32_t weights_stride[3];
  for (uint32_t i = 0; i < 3; ++i) {
    weights_shape[i] = weights.get_dim(i);
    weights_stride[i] = weights.get_mem_stride(i);
  }

  m_input_buffer_size =
      service::GetBufferSize(in.get_rank(), input_shape, input_stride);
  m_weights_buffer_size
      = service::GetBufferSize(weights.get_rank(), weights_shape, weights_stride);
  m_output_buffer_size
      = service::GetBufferSize(output_tile_shape.get_rank(), output_shape, output_stride);

  // Init in and out tensors with empty offset buffer
  // Layout: NHWCin
  m_input = Tensor<OffsetBuffer, 4>(OffsetBuffer(), input_shape, input_stride);
  // Layout: HWCo
  m_weights = Tensor<OffsetBuffer, 3>(OffsetBuffer(), weights_shape, weights_stride);
  // Layout: NHWCo (Cin=Co)
  m_output = Tensor<OffsetBuffer, 4>(OffsetBuffer(), output_shape, output_stride);

  // Init depthwise conv2d config
  m_config = cfg;
}

unsigned DepthwiseConv2d_CS::GetKernelPrivateDataSize() const {
  return sizeof(DepthwiseConv2DPrivateData);
}

unsigned DepthwiseConv2d_CS::GetRuntimeObjectSize() const {
  return sizeof(DepthwiseConv2d);
}

mli_status DepthwiseConv2d_CS::GetKernelPrivateData(void* kernel_private_data_buffer) {
  MLI_ASSERT(kernel_private_data_buffer != nullptr);

  // Batch checking
  MLI_ASSERT(m_input.get_dim(mli::kTensorBatchDim) == 1);

  // Channel checking
  MLI_ASSERT(m_input.get_dim(mli::kTensorChannelDim) == m_output.get_dim(mli::kTileChannelDim));
  MLI_ASSERT(m_weights.get_dim(mli::kKernelDWChannelInDim) == m_output.get_dim(mli::kTileChannelDim));

  DepthwiseConv2DPrivateData prv_data;
  prv_data.input = m_input;
  prv_data.weights = m_weights;
  prv_data.output = m_output;
  prv_data.inpzp_buffer = m_inpzp_buffer;
  prv_data.wtszp_buffer = m_wtszp_buffer;
  prv_data.inp_quant_axis = m_inp_quant_axis;
  prv_data.wts_quant_axis = m_wts_quant_axis;
  prv_data.config = m_config;
  prv_data.layout = LAYOUT_HWC;

  std::memcpy(kernel_private_data_buffer, (void *)&prv_data, prv_data.size);

  return MLI_STATUS_OK;
}

mli_status DepthwiseConv2d_CS::AttachBufferOffsets(Tensor<OffsetBuffer, 4> &input,
                                                   Tensor<OffsetBuffer, 4> &output,
                                                   OffsetBuffer &weights,
                                                   OffsetBuffer &inpzeropts,
                                                   OffsetBuffer &wtszeropts,
                                                   OffsetBuffer &ctrl_buffer) {
  MLI_ASSERT(input.get_buf().get_size() >= m_input_buffer_size * input.get_elem_size());
  MLI_ASSERT(output.get_buf().get_size() >= m_output_buffer_size * output.get_elem_size());
  MLI_ASSERT(weights.get_size() >= m_weights_buffer_size * weights.get_elem_size());

  // The metadata or descriptor is not required for ref kernel
  m_input.set_buf(input.get_buf());
  m_output.set_buf(output.get_buf());
  m_weights.set_buf(weights);
  // Zero Points maybe empty
  m_inpzp_buffer = inpzeropts;
  m_wtszp_buffer = wtszeropts;

  return MLI_STATUS_OK;
}

mli_status DepthwiseConv2d_CS::EncodeWeights(Tensor<Buffer, 3> &weights,
                                             Buffer &encoded_weights,
                                             compression_mode_t mode){
  return service::EncodeWeights(weights, encoded_weights);
}

unsigned DepthwiseConv2d_CS::GetEncodedWeightsSize() {
  return m_weights_buffer_size;
}

mli_status DepthwiseConv2d_CS::EncodeInpZeroPts(Tensor<Buffer, 1> &inpzeropts,
                                                Buffer &encoded_inpzeropts) {
  constexpr int channel_axis = mli::kTensorChannelDim;
  uint32_t channel_length = m_input.get_dim(channel_axis);
  return service::EncodeZeroPts<channel_axis>(
    inpzeropts, encoded_inpzeropts, m_inp_quant_axis, channel_length);
}

unsigned DepthwiseConv2d_CS::GetEncodedInpZeroPtsSize() {
  // per-tensor quantization
  return 1;
}

mli_status DepthwiseConv2d_CS::EncodeWtsZeroPts(Tensor<Buffer, 1> &wtszeropts,
                                                Buffer &encoded_wtszeropts) {
  constexpr int channel_axis = mli::kKernelDWChannelInDim;
  uint32_t channel_length = m_weights.get_dim(channel_axis);
  return service::EncodeZeroPts<channel_axis>(
    wtszeropts, encoded_wtszeropts, m_wts_quant_axis, channel_length);
}

unsigned DepthwiseConv2d_CS::GetEncodedWtsZeroPtsSize() {
  // per-channel quantization
  return m_weights.get_dim(mli::kKernelDWChannelInDim) ;
}

unsigned DepthwiseConv2d_CS::GetInputBufferSize() {
  return m_input_buffer_size;
}

unsigned DepthwiseConv2d_CS::GetWeightsBufferSize() {
  return m_weights_buffer_size;
}

unsigned DepthwiseConv2d_CS::GetOutputBufferSize() {
  return m_output_buffer_size;
}

}  // namespace snps_arc::metaware::mli::ref
