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

/**
  * @deprecated
  * Be carefull - this ctor doesn't support tiling - only single tile size of provided tensors
  * Be carefull - depthwise conv2d I/O tensors of rank 4 are deprecated - new interfaces use rank 5
  */
DepthwiseConv2d_CS::DepthwiseConv2d_CS(const lib_mli::PlatformDescription pd,
                                       const Tensor<NoBuffer, 4> &in,
                                       const Tensor<NoBuffer, 3> &weights,
                                       const DwConv2DConfig &cfg,
                                       const Tensor<NoBuffer, 4> &output)
    : m_config(cfg),
      m_pd(pd) {

  DEPRECATED_METHOD

  uint32_t input_shape[kDepthwiseIORank];
  uint32_t output_shape[kDepthwiseIORank];
  int32_t input_stride[kDepthwiseIORank];
  int32_t output_stride[kDepthwiseIORank];
  in.get_dims(input_shape);
  in.get_mem_strides(input_stride);
  output.get_dims(output_shape);
  output.get_mem_strides(output_stride);

  // B, H, W, C -> B, H, W, G=1, C
  input_shape[kGroupTensorChannelDim] = input_shape[kGroupTensorGroupDim];
  input_shape[kGroupTensorGroupDim] = 1;
  input_stride[kGroupTensorChannelDim] = input_stride[kGroupTensorGroupDim];
  input_stride[kGroupTensorGroupDim] = input_stride[kGroupTensorWidthDim];
  output_shape[kGroupTensorChannelDim] = output_shape[kGroupTensorGroupDim];
  output_shape[kGroupTensorGroupDim] = 1;
  output_stride[kGroupTensorChannelDim] = output_stride[kGroupTensorGroupDim];
  output_stride[kGroupTensorGroupDim] = output_stride[kGroupTensorWidthDim];

  uint32_t weights_shape[kDepthwiseWRank];
  int32_t weights_stride[kDepthwiseWRank];
  weights.get_dims(weights_shape);
  weights.get_mem_strides(weights_stride);

  m_input_buffer_size =
      service::GetBufferSize(kDepthwiseIORank, input_shape, input_stride);
  m_weights_buffer_size
      = service::GetBufferSize(kDepthwiseWRank, weights_shape, weights_stride);
  m_output_buffer_size
      = service::GetBufferSize(kDepthwiseIORank, output_shape, output_stride);

  m_inp_quant_axis = kPerTensorQuantDim;
  m_wts_quant_axis = kKernelDWChannelInDim;

  Tensor<OffsetBuffer, kDepthwiseIORank> input_tensor(OffsetBuffer(), input_shape, input_stride);
  m_input = TensorIterator<OffsetBuffer, kDepthwiseIORank, kDepthwiseIterRank>(input_tensor);

  Tensor<NoBuffer, kDepthwiseIORank> output_tensor(output_shape, output_stride);
  TensorIterator<NoBuffer, kDepthwiseIORank, kDepthwiseIterRank> output_tensor_it(output_tensor);
  m_output = TensorIterator<OffsetBuffer, kDepthwiseIORank, kDepthwiseIterRank>(output_tensor_it);

  Tensor<NoBuffer, kDepthwiseWRank> weights_tensor(weights_shape, weights_stride);
  const int32_t zero_inc_mask[kDepthwiseIterRank]{ 1, 1, 1, 1, 0 };
  const int32_t weights_it_order[kDepthwiseIterRank]{ kSkipIterDim, 0, 1, kSkipIterDim, 2 };
  lib_mli::TensorIterator<lib_mli::NoBuffer, kDepthwiseWRank, kDepthwiseIterRank> w_tensor_it(weights_tensor, output_tensor_it,
                                                                                               weights_it_order, zero_inc_mask);
  m_weights = TensorIterator<OffsetBuffer, kDepthwiseWRank, kDepthwiseIterRank>(w_tensor_it);

  uint32_t wzp_shape[kDepthwiseZPRank]{ output_tensor.get_dim(kGroupTensorChannelDim) };
  lib_mli::Tensor<lib_mli::NoBuffer, kDepthwiseZPRank> wzp_tensor(wzp_shape);  
  const int32_t wzp_it_order[kDepthwiseIterRank]{ kSkipIterDim, kSkipIterDim, kSkipIterDim, kSkipIterDim, 0 };
  lib_mli::TensorIterator<lib_mli::NoBuffer, kDepthwiseZPRank, kDepthwiseIterRank> wzp_tensor_it(wzp_tensor, output_tensor_it,
                                                                                                   wzp_it_order, zero_inc_mask);
  m_weights_zp = TensorIterator<OffsetBuffer, kDepthwiseZPRank, kDepthwiseIterRank>(wzp_tensor_it);
}


DepthwiseConv2d_CS::DepthwiseConv2d_CS(const lib_mli::PlatformDescription pd,
                                       const TensorIterator<NoBuffer, kDepthwiseIORank, kDepthwiseIterRank>& input,
                                       const TensorIterator<NoBuffer, kDepthwiseWRank, kDepthwiseIterRank>& weights,
                                       const TensorIterator<NoBuffer, kDepthwiseZPRank, kDepthwiseIterRank>& weights_zp,
                                       const DwConv2DConfig& cfg,
                                       const TensorIterator<NoBuffer, kDepthwiseIORank, kDepthwiseIterRank>& output)
  : m_input(input),
    m_weights(weights),
    m_weights_zp(weights_zp),
    m_output(output),
    m_config(cfg),
    m_pd(pd) {

  uint32_t input_shape[kDepthwiseIORank];
  int32_t input_stride[kDepthwiseIORank];
  uint32_t output_shape[kDepthwiseIORank];
  int32_t output_stride[kDepthwiseIORank];
  input.get_full_shape(input_shape);
  input.get_mem_strides(input_stride);
  output.get_full_shape(output_shape);
  output.get_mem_strides(output_stride);

  uint32_t weights_shape[kDepthwiseWRank];
  int32_t weights_stride[kDepthwiseWRank];
  weights.get_full_shape(weights_shape);
  weights.get_mem_strides(weights_stride);

  m_input_buffer_size =
    service::GetBufferSize(kDepthwiseIORank, input_shape, input_stride);
  m_weights_buffer_size
    = service::GetBufferSize(kDepthwiseWRank, weights_shape, weights_stride);
  m_output_buffer_size =
    service::GetBufferSize(kDepthwiseIORank, output_shape, output_stride);

  m_inp_quant_axis = kPerTensorQuantDim;
  m_wts_quant_axis = kKernelDWChannelInDim;
}


unsigned DepthwiseConv2d_CS::GetKernelPrivateDataSize() const {
  return sizeof(DepthwiseConv2DPrivateData);
}

unsigned DepthwiseConv2d_CS::GetRuntimeObjectSize() const {
  return sizeof(DepthwiseConv2d);
}

mli_status DepthwiseConv2d_CS::GetKernelPrivateData(void* kernel_private_data_buffer) {
  MLI_ASSERT(kernel_private_data_buffer != nullptr);

  // Channel checking
  MLI_ASSERT(m_input.get_dim(kGroupTensorChannelDim) == m_output.get_dim(kGroupTensorChannelDim));
  MLI_ASSERT(m_weights.get_dim(kKernelDWChannelInDim) == m_output.get_dim(kGroupTensorChannelDim));

  DepthwiseConv2DPrivateData prv_data;
  prv_data.input = m_input;
  prv_data.weights = m_weights;
  prv_data.weights_zp = m_weights_zp;
  prv_data.output = m_output;
  prv_data.inpzp_buffer = m_inpzp_buffer;
  prv_data.inp_quant_axis = m_inp_quant_axis;
  prv_data.wts_quant_axis = m_wts_quant_axis;
  prv_data.config = m_config;
  prv_data.layout = LAYOUT_HWC;

  std::memcpy(kernel_private_data_buffer, (void *)&prv_data, prv_data.size);

  return MLI_STATUS_OK;
}


/**
  * @deprecated
  * Be carefull - depthwise conv2d I/O tensors of rank 4 are deprecated - new interfaces use rank 5
  */
mli_status DepthwiseConv2d_CS::AttachBufferOffsets(Tensor<OffsetBuffer, 4> &input,
                                                   Tensor<OffsetBuffer, 4> &output,
                                                   OffsetBuffer &weights,
                                                   OffsetBuffer &inpzeropts,
                                                   OffsetBuffer &wtszeropts,
                                                   OffsetBuffer &ctrl_buffer) {
  DEPRECATED_METHOD

  MLI_ASSERT(input.get_buf().get_size() >= m_input_buffer_size * input.get_elem_size());
  MLI_ASSERT(output.get_buf().get_size() >= m_output_buffer_size * output.get_elem_size());
  MLI_ASSERT(weights.get_size() >= m_weights_buffer_size * weights.get_elem_size());

  // The metadata or descriptor is not required for ref kernel
  m_input.set_buf(input.get_buf());
  m_output.set_buf(output.get_buf());
  m_weights.set_buf(weights);
  m_weights_zp.set_buf(wtszeropts);
  // Zero Points maybe empty
  m_inpzp_buffer = inpzeropts;

  return MLI_STATUS_OK;
}

mli_status DepthwiseConv2d_CS::AttachBufferOffsets(const OffsetBuffer& input,
                                                   const OffsetBuffer& output,
                                                   const OffsetBuffer& weights,
                                                   const OffsetBuffer& inpzeropts,
                                                   const OffsetBuffer& wtszeropts,
                                                   const OffsetBuffer& ctrl_buffer) {
  m_input.set_buf(input);
  m_output.set_buf(output);
  m_weights.set_buf(weights);
  m_weights_zp.set_buf(wtszeropts);
  m_inpzp_buffer = inpzeropts;
  return MLI_STATUS_OK;
}

mli_status DepthwiseConv2d_CS::EncodeWeights(Tensor<Buffer, kDepthwiseWRank> &weights,
                                             Buffer &encoded_weights,
                                             compression_mode_t mode){
  return service::EncodeWeights(weights, encoded_weights);
}

unsigned DepthwiseConv2d_CS::GetEncodedWeightsSize() {
  return m_weights_buffer_size;
}

mli_status DepthwiseConv2d_CS::EncodeInpZeroPts(Tensor<Buffer, kDepthwiseZPRank> &inpzeropts,
                                                Buffer &encoded_inpzeropts) {
  constexpr int channel_axis = mli::kGroupTensorChannelDim;
  uint32_t channel_length = m_input.get_dim(channel_axis);
  return service::EncodeZeroPts<channel_axis>(
    inpzeropts, encoded_inpzeropts, m_inp_quant_axis, channel_length);
}

unsigned DepthwiseConv2d_CS::GetEncodedInpZeroPtsSize() {
  // per-tensor quantization
  return 1;
}

mli_status DepthwiseConv2d_CS::EncodeWtsZeroPts(Tensor<Buffer, kDepthwiseZPRank> &wtszeropts,
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
