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

using namespace snps_arc::metaware::mli::service;

namespace snps_arc::metaware::mli::ref {

Conv2d_CS::Conv2d_CS(const lib_mli::PlatformDescription pd,
                     const Tensor<NoBuffer, KConvIORank> &in, // B, H, W, Cin
                     const Tensor<NoBuffer, KConvWRank> &weights,  // G, H, W, Cin, Co
                     const Conv2DConfig &cfg,
                     const Tensor<NoBuffer, KConvIORank> &output_tile_shape) // G, H, W, Co
  : m_config(cfg),
    m_pd(pd) {
  
  DEPRECATED_METHOD

  uint32_t input_shape[KConvIORank];
  uint32_t output_shape[KConvIORank];
  int32_t input_stride[KConvIORank];
  int32_t output_stride[KConvIORank];
  in.get_dims(input_shape);
  in.get_mem_strides(input_stride);
  output_tile_shape.get_dims(output_shape);
  output_tile_shape.get_mem_strides(output_stride);

  uint32_t weights_shape[KConvWRank];
  int32_t weights_stride[KConvWRank];
  weights.get_dims(weights_shape);
  weights.get_mem_strides(weights_stride);

  m_input_buffer_size =
      service::GetBufferSize(in.get_rank(), input_shape, input_stride);
  m_weights_buffer_size
      = service::GetBufferSize(weights.get_rank(), weights_shape, weights_stride);
  m_output_buffer_size
      = service::GetBufferSize(output_tile_shape.get_rank(), output_shape, output_stride);

  // Init in and out tensors with empty offset buffer
  Tensor<OffsetBuffer, KConvIORank> input_tensor(OffsetBuffer(), input_shape, input_stride);
  m_input = TensorIterator<OffsetBuffer, KConvIORank, KConvIOIterRank>(input_tensor);

  Tensor<OffsetBuffer, KConvWRank> weights_tensor(OffsetBuffer(), weights_shape, weights_stride);
  m_weights = TensorIterator<OffsetBuffer, KConvWRank, KConvWIterRank>(weights_tensor);

  uint32_t wzp_shape[kConvZPRank]{ weights_shape[kKernelChannelOutDim] };
  Tensor<OffsetBuffer, kConvZPIterRank> wzp_tensor(OffsetBuffer(), wzp_shape);
  m_weights_zp = TensorIterator<OffsetBuffer, kConvZPRank, kConvZPIterRank>(wzp_tensor);

  Tensor<OffsetBuffer, KConvIORank> output_tensor(OffsetBuffer(), output_shape, output_stride);
  m_output = TensorIterator<OffsetBuffer, KConvIORank, KConvIOIterRank>(output_tensor);

  m_inp_quant_axis = kPerTensorQuantDim;
  m_wts_quant_axis = kKernelChannelOutDim;
}

Conv2d_CS::Conv2d_CS(const lib_mli::PlatformDescription pd,
                     const TensorIterator<NoBuffer, KConvIORank, KConvIOIterRank>& input,
                     const TensorIterator<NoBuffer, KConvWRank, KConvWIterRank>& weights,
                     const TensorIterator<NoBuffer, kConvZPRank, kConvZPIterRank>& weights_zp,
                     const Conv2DConfig& cfg,
                     const TensorIterator<NoBuffer, KConvIORank, KConvIOIterRank>& output)
  : m_input(input),
    m_weights(weights),
    m_weights_zp(weights_zp),
    m_output(output),
    m_config(cfg),
    m_pd(pd) {

  uint32_t input_shape[KConvIORank];
  int32_t input_stride[KConvIORank];
  uint32_t output_shape[KConvIORank];
  int32_t output_stride[KConvIORank];
  input.get_full_shape(input_shape);
  input.get_mem_strides(input_stride);
  output.get_full_shape(output_shape);
  output.get_mem_strides(output_stride);

  uint32_t weights_shape[KConvWRank];
  int32_t weights_stride[KConvWRank];
  weights.get_full_shape(weights_shape);
  weights.get_mem_strides(weights_stride);

  m_input_buffer_size =
    service::GetBufferSize(KConvIORank, input_shape, input_stride);
  m_weights_buffer_size
    = service::GetBufferSize(KConvWRank, weights_shape, weights_stride);
  m_output_buffer_size =
    service::GetBufferSize(KConvIORank, output_shape, output_stride);

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

  // Batch checking
  MLI_ASSERT(m_input.get_dim(mli::kTensorBatchDim) == 1);

  // Channel checking
  MLI_ASSERT(m_weights.get_dim(mli::kKernelChannelOutDim) ==
      m_output.get_dim(mli::kTileChannelDim));

  MLI_ASSERT(m_weights.get_dim(mli::kKernelChannelInDim) ==
      m_input.get_dim(mli::kTensorChannelDim));

  MLI_ASSERT(m_input.get_dim(mli::kTensorChannelDim) == m_weights.get_dim(mli::kKernelChannelInDim));

  // Group checking
  MLI_ASSERT(m_weights.get_dim(mli::kKernelGroupDim) == 1);

  MLI_ASSERT(m_weights.get_dim(mli::kKernelGroupDim) ==
      m_output.get_dim(mli::kTileGroupDim));

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

mli_status Conv2d_CS::AttachBufferOffsets(Tensor<OffsetBuffer, KConvIORank> &input,
                                          Tensor<OffsetBuffer, KConvIORank> &output,
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

mli_status Conv2d_CS::EncodeWeights(Tensor<Buffer, KConvWRank> &weights,
                                    Buffer &encoded_weights,
                                    compression_mode_t mode){
  return service::EncodeWeights(weights, encoded_weights);
}

unsigned Conv2d_CS::GetEncodedWeightsSize() {
  return m_weights_buffer_size;
}

mli_status Conv2d_CS::EncodeInpZeroPts(Tensor<Buffer, kConvZPRank> &inpzeropts,
                                       Buffer &encoded_inpzeropts) {
  constexpr int channel_axis = mli::kTensorChannelDim;
  uint32_t channel_length = m_input.get_dim(channel_axis);
  return service::EncodeZeroPts<channel_axis>(
    inpzeropts, encoded_inpzeropts, m_inp_quant_axis, channel_length);
}

unsigned Conv2d_CS::GetEncodedInpZeroPtsSize() {
  // per-tensor quantization
  return 1;
}

mli_status Conv2d_CS::EncodeWtsZeroPts(Tensor<Buffer, kConvZPRank> &wtszeropts,
                                       Buffer &encoded_wtszeropts) {
  constexpr int channel_axis = mli::kKernelChannelOutDim;
  uint32_t channel_length = m_weights.get_dim(channel_axis);
  return service::EncodeZeroPts<channel_axis>(
    wtszeropts, encoded_wtszeropts, m_wts_quant_axis, channel_length);
}

unsigned Conv2d_CS::GetEncodedWtsZeroPtsSize() {
  // per-channel quantization
  return m_weights.get_dim(mli::kKernelChannelOutDim) ;
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

mli_status Conv2d_CS::SetIterators(uint32_t output_total_size[KConvIORank],
                                   uint32_t iteration_order[KConvIORank],
                                   uint32_t input_first_inc[KConvIORank],
                                   uint32_t input_inc[KConvIORank],
                                   uint32_t output_first_inc[KConvIORank],
                                   uint32_t output_inc[KConvIORank],
                                   uint32_t weights_inc[KConvWRank]) {

  DEPRECATED_METHOD

  int32_t output_mem_stride[KConvIORank];
  m_output.get_mem_strides(output_mem_stride);
  Tensor<OffsetBuffer, KConvIORank> output_tensor(m_output.get_buf(), output_total_size, output_mem_stride);
  m_output = TensorIterator<OffsetBuffer, KConvIORank, KConvIOIterRank>(output_tensor);

  // set common for input and output IteratorCfg parameters
  int32_t iteration_order_signed[KConvIOIterRank];
  int32_t count[KConvIOIterRank];
  for (unsigned i = 0; i < KConvIOIterRank; i++) {
    iteration_order_signed[i] = (int32_t)iteration_order[i];

    if (output_total_size[i] == output_first_inc[i]) count[i] = 1;
    else count[i] = 1 + (int32_t)CEIL_DIV(output_total_size[i] - output_first_inc[i], output_inc[i]);
  }

  // set part of input IteratorCfg parameters
  int32_t input_first_increment_signed[KConvIOIterRank];
  int32_t input_increment_signed[KConvIOIterRank];
  int32_t input_last_increment_signed[KConvIOIterRank];
  int32_t input_first_size_signed[KConvIOIterRank];
  int32_t input_size_signed[KConvIOIterRank];
  int32_t input_last_size_signed[KConvIOIterRank];
  for (unsigned i = 0; i < KConvIOIterRank; i++) {
    input_first_increment_signed[i] = (int32_t)input_first_inc[i];
    input_increment_signed[i] = (int32_t)input_inc[i];
    if (count[i] == 1) input_last_increment_signed[i] = 0;
    else if (i == kTensorChannelDim) {
      input_first_increment_signed[kTensorChannelDim] = 0;
      input_increment_signed[kTensorChannelDim] = 0;
      input_last_increment_signed[kTensorChannelDim] = 0;
    }
    else {
      input_last_increment_signed[i] = get_last_increment(count[i], input_first_increment_signed[i], input_increment_signed[i]);
    }
  }

  // calculate rest part of input IteratorCfg parameters
  // B
  input_first_size_signed[kTensorBatchDim] = (int32_t)input_first_inc[kTensorBatchDim];
  input_size_signed[kTensorBatchDim] = (int32_t)input_inc[kTensorBatchDim];

  // H
  uint32_t padding_y = m_config.padding_begin[0];
  bool single_tile_y = output_first_inc[kTensorHeightDim] == output_total_size[kTensorHeightDim];
  if (single_tile_y) padding_y += m_config.padding_end[0];
  input_first_size_signed[kTensorHeightDim] = (int32_t)get_conv_input_size(output_first_inc[kTensorHeightDim], padding_y,
                                                                           m_weights.get_dim(kKernelHeightDim),
                                                                           m_config.dilation[0], m_config.stride[0]);
  input_first_size_signed[kTensorHeightDim] = MIN(input_first_size_signed[kTensorHeightDim],
                                                  (int32_t)m_input.get_dim(kTensorHeightDim));
  if (single_tile_y) input_size_signed[kTensorHeightDim] = input_first_size_signed[kTensorHeightDim];
  else {
    input_size_signed[kTensorHeightDim] = (int32_t)get_conv_input_size(output_inc[kTensorHeightDim], 0,
                                                                       m_weights.get_dim(kKernelHeightDim),
                                                                       m_config.dilation[0], m_config.stride[0]);
    input_size_signed[kTensorHeightDim] = MIN(input_size_signed[kTensorHeightDim],
                                              (int32_t)m_input.get_dim(kTensorHeightDim));
  }

  // W
  uint32_t padding_x = m_config.padding_begin[1];
  bool single_tile_x = output_first_inc[kTensorWidthDim] == output_total_size[kTensorWidthDim];
  if (single_tile_x) padding_x += m_config.padding_end[1];
  input_first_size_signed[kTensorWidthDim] = (int32_t)get_conv_input_size(output_first_inc[kTensorWidthDim], padding_x,
                                                                          m_weights.get_dim(kKernelWidthDim),
                                                                          m_config.dilation[1], m_config.stride[1]);
  input_first_size_signed[kTensorWidthDim] = MIN(input_first_size_signed[kTensorWidthDim],
                                                 (int32_t)m_input.get_dim(kTensorWidthDim));
  if (single_tile_x) input_size_signed[kTensorWidthDim] = input_first_size_signed[kTensorWidthDim];
  else {
    input_size_signed[kTensorWidthDim] = (int32_t)get_conv_input_size(output_inc[kTensorWidthDim], 0,
                                                                      m_weights.get_dim(kKernelWidthDim),
                                                                      m_config.dilation[1], m_config.stride[1]);
    input_size_signed[kTensorWidthDim] = MIN(input_size_signed[kTensorWidthDim],
                                             (int32_t)m_input.get_dim(kTensorWidthDim));
  }

  // C
  input_first_size_signed[kTensorChannelDim] = (int32_t)input_first_inc[kTensorChannelDim];
  input_size_signed[kTensorChannelDim] = (int32_t)input_inc[kTensorChannelDim];

  for (unsigned i = 0; i < KConvIOIterRank; i++) {
    input_last_size_signed[i] = (int32_t)m_input.get_dim(i) + input_last_increment_signed[i];
  }

  // set output IteratorCfg parameters
  int32_t output_first_increment_signed[KConvIOIterRank];
  int32_t output_increment_signed[KConvIOIterRank];
  int32_t output_last_increment_signed[KConvIOIterRank];
  int32_t output_first_size_signed[KConvIOIterRank];
  int32_t output_size_signed[KConvIOIterRank];
  int32_t output_last_size_signed[KConvIOIterRank];
  for (unsigned i = 0; i < KConvIOIterRank; i++) {
    output_first_increment_signed[i] = (int32_t)output_first_inc[i];
    output_increment_signed[i] = (int32_t)output_inc[i];
    if (count[i] == 1) output_last_increment_signed[i] = 0;
    else {
      output_last_increment_signed[i] = get_last_increment(count[i], output_first_increment_signed[i], output_increment_signed[i]);
    }
    output_first_size_signed[i] = (int32_t)output_first_inc[i];
    output_size_signed[i] = (int32_t)output_inc[i];
    output_last_size_signed[i] = (int32_t)output_total_size[i] + output_last_increment_signed[i];
  }

  // set weights IteratorCfg parameters
  int32_t weights_iteration_order_signed[KConvWIterRank]{ 0, 1, 2, 3, 4 };  // TODO: maybe add some connection between i/o and w orders
  int32_t weights_count[KConvWIterRank]{ 1, 1, 1, 1, count[kTensorChannelDim] };
  int32_t weights_first_increment[KConvWIterRank]{ 0, 0, 0, 0, output_first_increment_signed[kTensorChannelDim] };
  int32_t weights_increment[KConvWIterRank]{ 0, 0, 0, 0, output_increment_signed[kTensorChannelDim] };
  int32_t weights_last_increment[KConvWIterRank]{ 0, 0, 0, 0, output_last_increment_signed[kTensorChannelDim] };
  int32_t weights_first_size[KConvWIterRank];
  int32_t weights_size[KConvWIterRank];
  int32_t weights_last_size[KConvWIterRank];
  for (unsigned i = 0; i < KConvWIterRank - 1; i++) {
    weights_first_size[i] = (int32_t) weights_inc[i];
    weights_size[i] = (int32_t)weights_inc[i];
    weights_last_size[i] = (int32_t)weights_inc[i];
  }
  weights_first_size[kKernelChannelOutDim] = output_first_size_signed[kTensorChannelDim];
  weights_size[kKernelChannelOutDim] = output_size_signed[kTensorChannelDim];
  weights_last_size[kKernelChannelOutDim] = output_last_size_signed[kTensorChannelDim];

  // set weights zp(s) IteratorCfg parameters
  int32_t wzp_iteration_order[kConvZPIterRank]{ 0 };
  int32_t wzp_count[kConvZPIterRank]{ count[kTensorChannelDim] };
  int32_t wzp_first_increment[kConvZPIterRank]{ output_first_increment_signed[kTensorChannelDim] };
  int32_t wzp_increment[kConvZPIterRank]{ output_increment_signed[kTensorChannelDim] };
  int32_t wzp_last_increment[kConvZPIterRank]{ output_last_increment_signed[kTensorChannelDim] };
  int32_t wzp_first_size[kConvZPIterRank]{ output_first_size_signed[kTensorChannelDim] };
  int32_t wzp_size[kConvZPIterRank]{ output_size_signed[kTensorChannelDim] };
  int32_t wzp_last_size[kConvZPIterRank]{ output_last_size_signed[kTensorChannelDim] };

  IteratorCfg<KConvIOIterRank> input_config(
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

  IteratorCfg<KConvIOIterRank> output_config(
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

  lib_mli::IteratorCfg<KConvWIterRank> weights_it_config(
    weights_iteration_order_signed, weights_count,
    weights_first_increment, weights_increment, weights_last_increment,
    weights_first_size, weights_size, weights_last_size
  );
  m_weights.set_config(weights_it_config);

  lib_mli::IteratorCfg<kConvZPIterRank> wzp_it_config(
    wzp_iteration_order, wzp_count,
    wzp_first_increment, wzp_increment, wzp_last_increment,
    wzp_first_size, wzp_size, wzp_last_size
  );
  m_weights_zp.set_config(wzp_it_config);

  return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref