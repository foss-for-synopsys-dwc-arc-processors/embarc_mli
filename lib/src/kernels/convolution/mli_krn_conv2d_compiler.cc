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
#include "mli_service_functions.hpp"


namespace snps_arc::metaware::mli::ref {

/**
  * @deprecated
  * Be carefull - you need to use another deprected method to set tiling - SetIterators
  * Be carefull - conv2d I/O tensors of rank 4 are deprecated - new interfaces use rank 5
  * Be carefull - this is the most deprecated Constructor
  */
Conv2d_CS::Conv2d_CS(const PlatformDescription pd,
                     const Tensor<NoBuffer, 4> &in,                // B, H, W, Ci
                     const Tensor<NoBuffer, 5> &weights,           // G, H, W, Ci, Co
                     const Conv2DConfig &cfg,
                     const Tensor<NoBuffer, 4> &output_tile_shape) // G, H, W, Co
  : m_config(cfg),
    m_pd(pd) {
  
  DEPRECATED_METHOD

  uint32_t input_shape[kConvIORank];
  uint32_t output_shape[kConvIORank];
  int32_t input_stride[kConvIORank];
  int32_t output_stride[kConvIORank];
  in.get_dims(input_shape);
  in.get_mem_strides(input_stride);
  output_tile_shape.get_dims(output_shape);
  output_tile_shape.get_mem_strides(output_stride);

  // B, H, W, C -> B, H, W, G=1, C
  input_shape[kGroupTensorChannelDim] = input_shape[kGroupTensorGroupDim];
  input_shape[kGroupTensorGroupDim] = 1;
  input_stride[kGroupTensorChannelDim] = input_stride[kGroupTensorGroupDim];
  input_stride[kGroupTensorGroupDim] = input_stride[kGroupTensorWidthDim];
  output_shape[kGroupTensorChannelDim] = output_shape[kGroupTensorGroupDim];
  output_shape[kGroupTensorGroupDim] = 1;
  output_stride[kGroupTensorChannelDim] = output_stride[kGroupTensorGroupDim];
  output_stride[kGroupTensorGroupDim] = output_stride[kGroupTensorWidthDim];

  uint32_t weights_shape[kConvWRank];
  int32_t weights_stride[kConvWRank];
  weights.get_dims(weights_shape);
  weights.get_mem_strides(weights_stride);

  m_input_buffer_size =
      service::GetBufferSize(kConvIORank, input_shape, input_stride);
  m_weights_buffer_size
      = service::GetBufferSize(weights.get_rank(), weights_shape, weights_stride);
  m_output_buffer_size
      = service::GetBufferSize(kConvIORank, output_shape, output_stride);

  // Init in and out tensors with empty offset buffer
  Tensor<OffsetBuffer, kConvIORank> input_tensor(OffsetBuffer(), input_shape, input_stride);
  m_input = TensorIterator<OffsetBuffer, kConvIORank, kConvIOIterRank>(input_tensor);

  Tensor<OffsetBuffer, kConvWRank> weights_tensor(OffsetBuffer(), weights_shape, weights_stride);
  m_weights = TensorIterator<OffsetBuffer, kConvWRank, kConvWIterRank>(weights_tensor);

  Tensor<NoBuffer, kConvIORank> output_tensor(output_shape, output_stride);
  TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank> output_tensor_it(output_tensor);
  m_output = TensorIterator<OffsetBuffer, kConvIORank, kConvIOIterRank>(output_tensor_it);

  uint32_t wzp_shape[kConvZPRank]{ weights.get_dim(kKernelChannelOutDim)};
  const Tensor<NoBuffer, kConvZPRank> wzp_tensor(wzp_shape);
  const int32_t wzp_it_order[kConvWRank]{ -1, -1, -1, -1, 0 };
  const int32_t zero_inc_mask[kConvWRank]{ 1, 1, 1, 1, 0 };
  TensorIterator<NoBuffer, kConvZPRank, kConvZPIterRank> wzp_tensor_it(wzp_tensor, output_tensor_it, wzp_it_order, zero_inc_mask);
  m_weights_zp = TensorIterator<NoBuffer, kConvZPRank, kConvZPIterRank>(wzp_tensor_it);

  m_inp_quant_axis = kPerTensorQuantDim;
  m_wts_quant_axis = kKernelChannelOutDim;
}

/**
 * @deprecated
 */
Conv2d_CS::Conv2d_CS(const PlatformDescription pd,
                     const TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank>& input,
                     const TensorIterator<NoBuffer, kConvWRank, kConvWIterRank>& weights,
                     const TensorIterator<NoBuffer, kConvZPRank, kConvZPIterRank>& weights_zp,
                     const Conv2DConfig& cfg,
                     const TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank>& output)
  : m_input(input),
    m_weights(weights),
    m_weights_zp(weights_zp),
    m_output(output),
    m_config(cfg),
    m_pd(pd) {

  DEPRECATED_METHOD

  m_input_buffer_size = service::GetBufferSize(input.get_tensor());
  m_weights_buffer_size = service::GetBufferSize(weights.get_tensor());
  m_output_buffer_size = service::GetBufferSize(output.get_tensor());

  m_inp_quant_axis = kPerTensorQuantDim;
  m_wts_quant_axis = kKernelChannelOutDim;
}

Conv2d_CS::Conv2d_CS(const PlatformDescription pd,
                     const TensorIterator<NoBuffer, kConvIORank, kConvIterRank>& input,
                     const TensorIterator<NoBuffer, kConvZPRank, kConvIterRank>& input_zp,
                     const TensorIterator<NoBuffer, kConvWRank, kConvIterRank>& weights,
                     const TensorIterator<NoBuffer, kConvZPRank, kConvIterRank>& weights_zp,
                     const Conv2DConfig& cfg,
                     const TensorIterator<NoBuffer, kConvIORank, kConvIterRank>& output) 
  : m_input(input),
    m_weights(weights),
    m_weights_zp(weights_zp),
    m_output(output),
    m_config(cfg),
    m_pd(pd) {

  m_input_buffer_size = service::GetBufferSize(input.get_tensor());
  m_weights_buffer_size = service::GetBufferSize(weights.get_tensor());
  m_output_buffer_size = service::GetBufferSize(output.get_tensor());

  m_inp_quant_axis = kPerTensorQuantDim;
  m_wts_quant_axis = kKernelChannelOutDim;
}

unsigned Conv2d_CS::GetKernelPrivateDataSize() const {
  return sizeof(Conv2DPrivateData);
}

unsigned Conv2d_CS::GetRuntimeObjectSize() const {
  return sizeof(Conv2d);
}

mli_status Conv2d_CS::GetKernelPrivateData(void* kernel_private_data_buffer) {

  MLI_ASSERT(kernel_private_data_buffer != nullptr);

  MLI_ASSERT(m_input.get_dim(kGroupTensorBatchDim) == 1);

  MLI_ASSERT(m_weights.get_dim(kKernelChannelOutDim) ==
      m_output.get_dim(kGroupTensorChannelDim));

  MLI_ASSERT(m_weights.get_dim(kKernelChannelInDim) ==
      m_input.get_dim(kGroupTensorChannelDim));

  MLI_ASSERT(m_weights.get_dim(kKernelGroupDim) == 1);

  MLI_ASSERT(m_weights.get_dim(kKernelGroupDim) ==
      m_output.get_dim(kGroupTensorGroupDim));

  Conv2DPrivateData prv_data;
  prv_data.input = m_input;
  prv_data.weights = m_weights;
  prv_data.output = m_output;
  prv_data.weights_zp = m_weights_zp;
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
  * Be carefull - conv2d I/O tensors of rank 4 are deprecated - new interfaces use rank 5 
  */
mli_status Conv2d_CS::AttachBufferOffsets(Tensor<OffsetBuffer, 4> &input,
                                          Tensor<OffsetBuffer, 4> &output,
                                          OffsetBuffer &weights,
                                          OffsetBuffer &inpzeropts,
                                          OffsetBuffer &wtszeropts,
                                          OffsetBuffer &ctrl_buffer) {
  DEPRECATED_METHOD

  MLI_ASSERT(output.get_buf().get_size() >= m_output_buffer_size * output.get_elem_size());

  // The metadata or descriptor is not required for ref kernel
  m_input.set_buf(input.get_buf());
  m_output.set_buf(output.get_buf());
  m_weights.set_buf(weights);
  m_weights_zp.set_buf(wtszeropts);

  // Zero Points maybe empty
  m_inpzp_buffer = inpzeropts;

  return MLI_STATUS_OK;
}

mli_status Conv2d_CS::AttachBufferOffsets(const OffsetBuffer& input,
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

/**
 * @deprecated
 */
mli_status Conv2d_CS::EncodeWeights(Tensor<Buffer, kConvWRank> &weights,
                                    Buffer &encoded_weights,
                                    compression_mode_t mode){
  DEPRECATED_METHOD
  return service::EncodeWeights(weights, encoded_weights);
}


mli_status Conv2d_CS::EncodeWeightsAndZeroPts(TensorIterator<Buffer, kConvWRank, kConvIterRank>& weights,
                                              TensorIterator<Buffer, kConvZPRank, kConvIterRank>& weights_zp,
                                              Buffer& encoded_weights) {
  return service::EncodeWeightsAndZeroPts(weights.get_tensor(), weights_zp.get_tensor(), encoded_weights);
}

unsigned Conv2d_CS::GetEncodedWeightsSize() {
  return m_weights_buffer_size;
}

/**
 * @deprecated
 */
mli_status Conv2d_CS::EncodeInpZeroPts(Tensor<Buffer, kInpZPRank> &inpzeropts,
                                       Buffer &encoded_inpzeropts) {
  DEPRECATED_METHOD
  constexpr int channel_axis = kTensorChannelDim;
  uint32_t channel_length = m_input.get_dim(channel_axis);
  return service::EncodeZeroPts<channel_axis>(
    inpzeropts, encoded_inpzeropts, m_inp_quant_axis, channel_length);
}


mli_status Conv2d_CS::EncodeInpZeroPts(TensorIterator<Buffer, kConvZPRank, kConvZPIterRank>& input_zp,
                                       Buffer& encoded_inpzeropts) {
  constexpr int channel_axis = kTensorChannelDim;
  uint32_t channel_length = m_input.get_dim(channel_axis);
  return service::EncodeZeroPts<channel_axis>(
    input_zp.get_tensor(), encoded_inpzeropts, m_inp_quant_axis, channel_length);
}

unsigned Conv2d_CS::GetEncodedInpZeroPtsSize() {
  // per-tensor quantization
  return 1;
}

/**
 * @deprecated
 */
mli_status Conv2d_CS::EncodeWtsZeroPts(Tensor<Buffer, kConvZPRank> &wtszeropts,
                                       Buffer &encoded_wtszeropts) {
  DEPRECATED_METHOD
  constexpr int channel_axis = kKernelChannelOutDim;
  uint32_t channel_length = m_weights.get_dim(channel_axis);
  return service::EncodeZeroPts<channel_axis>(
    wtszeropts, encoded_wtszeropts, m_wts_quant_axis, channel_length);
}

unsigned Conv2d_CS::GetEncodedWtsZeroPtsSize() {
  // per-channel quantization
  return m_weights.get_dim(kKernelChannelOutDim) ;
}

unsigned Conv2d_CS::GetInputBufferSize() {
  return m_input_buffer_size;
}

unsigned Conv2d_CS::GetWeightsBufferSize() {
  return m_weights_buffer_size;
}

unsigned Conv2d_CS::GetOutputBufferSize() {
  return m_output_buffer_size;
}

unsigned Conv2d_CS::GetZeroPointBufferSize() {
  return 0;
}

/**
  * @deprecated
  * Be carefull - conv2d I/O tensors of rank 4 are deprecated - new interfaces use rank 5 
  */
mli_status Conv2d_CS::SetIterators(uint32_t output_total_size[4],
                                   uint32_t iteration_order[4],
                                   uint32_t input_first_inc[4],
                                   uint32_t input_inc[4],
                                   uint32_t output_first_inc[4],
                                   uint32_t output_inc[4],
                                   uint32_t weights_inc[4]) {

  DEPRECATED_METHOD

  // set output tensor iterator
  int32_t output_mem_stride[kConvIORank];
  m_output.get_mem_strides(output_mem_stride);
  uint32_t output_total_size_5d[kConvIORank]; // B, H, W, C -> B, H, W, G=1, C
  for (int i = 0; i < 4; i++) {
    output_total_size_5d[i] = output_total_size[i];
  }
  output_total_size_5d[kGroupTensorChannelDim] = output_total_size_5d[kGroupTensorGroupDim];
  output_total_size_5d[kGroupTensorGroupDim] = 1;

  const Tensor<NoBuffer, kConvIORank> output_tensor(output_total_size_5d, output_mem_stride);
  uint32_t output_inc_5d[kConvIOIterRank]; // B, H, W, C -> B, H, W, G=1, C
  for (int i = 0; i < 4; i++) {
    output_inc_5d[i] = output_inc[i];
  }
  output_inc_5d[kGroupTensorChannelDim] = output_inc_5d[kGroupTensorGroupDim];
  output_inc_5d[kGroupTensorGroupDim] = 1;

  int32_t iteration_order_signed[kConvIOIterRank];
  for (int i = 0; i < 4; i++) {
    iteration_order_signed[i] = (int32_t)iteration_order[i];
  }
  iteration_order_signed[4] = 4;

  TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank> output_tensor_it(output_tensor, output_inc_5d, iteration_order_signed);
  m_output = TensorIterator<OffsetBuffer, kConvIORank, kConvIOIterRank>(output_tensor_it);

  // set input tensor iterator
  uint32_t effective_kernel_size[kConvIORank]{
    1, service::get_effective_kernel_size(m_weights.get_dim(kKernelHeightDim), m_config.dilation[0]),
    service::get_effective_kernel_size(m_weights.get_dim(kKernelWidthDim), m_config.dilation[1]),
    1, m_input.get_dim(kGroupTensorChannelDim)
  };
  uint32_t stride[kConvIORank]{ 1, m_config.stride[0], m_config.stride[1], 1, 0};
  uint32_t pre_padding[kConvIORank]{ 0, m_config.padding_begin[0], m_config.padding_begin[1], 0, 0};
  int32_t input_mem_stride[kConvIORank];
  m_input.get_mem_strides(input_mem_stride);
  uint32_t input_shape[kConvIORank];
  m_input.get_full_shape(input_shape);
  const Tensor<NoBuffer, kConvIORank> full_in_tensor(input_shape, input_mem_stride);
  TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank> in_tensor_it(full_in_tensor, output_tensor_it,
                                                                      effective_kernel_size, stride, pre_padding);
  // set weights tensor iterator
  int32_t weights_mem_stride[kConvIORank];
  m_weights.get_mem_strides(weights_mem_stride);
  uint32_t weights_shape[kConvWRank];
  m_weights.get_full_shape(weights_shape);
  const Tensor<NoBuffer, kConvWRank> wt_tensor(weights_shape, weights_mem_stride);
  const int32_t zero_inc_mask[kConvWRank]{ 1, 1, 1, 1, 0 };
  TensorIterator<NoBuffer, kConvWRank, kConvWIterRank> w_tensor_it(wt_tensor, output_tensor_it, nullptr, zero_inc_mask);
  m_weights = TensorIterator<OffsetBuffer, kConvWRank, kConvWIterRank>(w_tensor_it);

  // set weights ZPs tensor iterator
  uint32_t wzp_shape[kConvZPRank]{ output_total_size_5d[kGroupTensorChannelDim] };
  Tensor<NoBuffer, kConvZPRank> wzp_tensor(wzp_shape);
  const int32_t wzp_it_order[kConvWRank]{ -1, -1, -1, -1, 0 };
  TensorIterator<NoBuffer, kConvZPRank, kConvZPIterRank> wzp_tensor_it(wzp_tensor, output_tensor_it, wzp_it_order, zero_inc_mask);
  m_weights_zp = TensorIterator<OffsetBuffer, kConvZPRank, kConvZPIterRank>(wzp_tensor_it);

  return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref