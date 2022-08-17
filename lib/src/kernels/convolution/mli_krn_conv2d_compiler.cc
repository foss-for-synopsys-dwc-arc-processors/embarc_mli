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
                     const Tensor<NoBuffer, KConvIORank> &output_tile_shape // G, H, W, Co
                    ) : m_pd{pd} {
  
  // TODO: uncomment DEPRICATED_METHOD after implementation of new version
  // DEPRICATED_METHOD

  uint32_t input_shape[KConvIORank];
  uint32_t output_shape[KConvIORank];
  int32_t input_stride[KConvIORank];
  int32_t output_stride[KConvIORank];
  for (uint32_t i = 0; i < KConvIORank; ++i) {
    input_shape[i] = in.get_dim(i);
    input_stride[i] = in.get_mem_stride(i);
    output_shape[i] = output_tile_shape.get_dim(i);
    output_stride[i] = output_tile_shape.get_mem_stride(i);
  }

  uint32_t weights_shape[KConvWRank];
  int32_t weights_stride[KConvWRank];
  for (uint32_t i = 0; i < KConvWRank; ++i) {
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
  m_input = Tensor<OffsetBuffer, KConvIORank>(OffsetBuffer(), in);
  m_weights = Tensor<OffsetBuffer, KConvWRank>(OffsetBuffer(), weights);
  m_output = Tensor<OffsetBuffer, KConvIORank>(OffsetBuffer(), output_tile_shape);

  m_inp_quant_axis = kPerTensorQuantDim;
  m_wts_quant_axis = kKernelChannelOutDim;

  // Init convolution config
  m_config = cfg;

  // TODO: remove these code and replace with TensorIterator usage
  m_use_tiling = false;
  for (unsigned i = 0; i < KConvIORank; i++) {
    m_tile_total_input_size[i] = 0;
    m_tile_total_output_size[i] = 0;
    m_tile_total_weights_size[i] = 0;
    m_tile_iteration_order[i] = 0;
    m_tile_input_first_inc[i] = 0;
    m_tile_input_inc[i] = 0;
    m_tile_output_first_inc[i] = 0;
    m_tile_output_inc[i] = 0;
    m_tile_weights_inc[i] = 0;
  };
}

Conv2d_CS::Conv2d_CS(const lib_mli::PlatformDescription pd,
                     const TensorIterator<NoBuffer, KConvIORank, KConvIOIterRank>& input,
                     const TensorIterator<NoBuffer, KConvWRank, KConvWIterRank>& weights,
                     const TensorIterator<NoBuffer, kConvZPRank, kConvZPIterRank>& weights_zp,
                     const Conv2DConfig& cfg,
                     const TensorIterator<NoBuffer, KConvIORank, KConvIOIterRank>& output) {
  // TODO: implementation
}

unsigned Conv2d_CS::GetKernelPrivateDataSize() const {
  return sizeof(Conv2DPrivateData);
}

unsigned Conv2d_CS::GetRuntimeObjectSize() const {
  return sizeof(Conv2d);
}

void Conv2d_CS::FillTilingParams(Conv2DPrivateData& pdata) {

  pdata.m_use_tiling = m_use_tiling;

  // B
  pdata.m_tile_first_size[kTensorBatchDim] = m_tile_input_first_inc[kTensorBatchDim];
  pdata.m_tile_size[kTensorBatchDim] = m_tile_input_inc[kTensorBatchDim];

  // H
  uint32_t padding_y = m_config.padding_begin[0];
  bool single_tile_y = m_tile_output_first_inc[kTensorHeightDim] == m_tile_total_output_size[kTensorHeightDim];
  if (single_tile_y) padding_y += m_config.padding_end[0];
  pdata.m_tile_first_size[kTensorHeightDim] = get_conv_input_size(m_tile_output_first_inc[kTensorHeightDim], padding_y,
                                                                  m_weights.get_dim(kTensorHeightDim), m_config.dilation[0], m_config.stride[0]);
  pdata.m_tile_first_size[kTensorHeightDim] = MIN(pdata.m_tile_first_size[kTensorHeightDim], m_input.get_dim(kTensorHeightDim));
  if (single_tile_y) pdata.m_tile_size[kTensorHeightDim] = pdata.m_tile_first_size[kTensorHeightDim];
  else {
    pdata.m_tile_size[kTensorHeightDim] = get_conv_input_size(m_tile_output_inc[kTensorHeightDim], 0,
                                                              m_weights.get_dim(kTensorHeightDim), m_config.dilation[0], m_config.stride[0]);
    pdata.m_tile_size[kTensorHeightDim] = MIN(pdata.m_tile_size[kTensorHeightDim], m_input.get_dim(kTensorHeightDim));
  }

  // W
  uint32_t padding_x = m_config.padding_begin[1];
  bool single_tile_x = m_tile_output_first_inc[kTensorWidthDim] == m_tile_total_output_size[kTensorWidthDim];
  if (single_tile_x) padding_x += m_config.padding_end[1];
  pdata.m_tile_first_size[kTensorWidthDim] = get_conv_input_size(m_tile_output_first_inc[kTensorWidthDim], padding_x,
                                                                 m_weights.get_dim(kTensorWidthDim), m_config.dilation[1], m_config.stride[1]);
  pdata.m_tile_first_size[kTensorWidthDim] = MIN(pdata.m_tile_first_size[kTensorWidthDim], m_input.get_dim(kTensorWidthDim));
  if (single_tile_x) pdata.m_tile_size[kTensorWidthDim] = pdata.m_tile_first_size[kTensorWidthDim];
  else {
    pdata.m_tile_size[kTensorWidthDim] = get_conv_input_size(m_tile_output_inc[kTensorWidthDim], 0,
                                                             m_weights.get_dim(kTensorWidthDim), m_config.dilation[1], m_config.stride[1]);
    pdata.m_tile_size[kTensorWidthDim] = MIN(pdata.m_tile_size[kTensorWidthDim], m_input.get_dim(kTensorWidthDim));
  }

  // C
  pdata.m_tile_first_size[kTensorChannelDim] = m_tile_input_first_inc[kTensorChannelDim];
  pdata.m_tile_size[kTensorChannelDim] = m_tile_input_inc[kTensorChannelDim];

  // TODO: remove these code and replace with TensorIterator usage
  for (unsigned i = 0; i < KConvIORank; i++) {
    pdata.m_tile_total_output_size[i] = m_tile_total_output_size[i];
    pdata.m_tile_iteration_order[i] = m_tile_iteration_order[i];
    pdata.m_tile_input_first_inc[i] = m_tile_input_first_inc[i];
    pdata.m_tile_input_inc[i] = m_tile_input_inc[i];
    pdata.m_tile_output_first_inc[i] = m_tile_output_first_inc[i];
    pdata.m_tile_output_inc[i] = m_tile_output_inc[i];
    pdata.m_tile_weights_inc[i] = m_tile_weights_inc[i];
    pdata.m_tile_total_weights_size[i] = m_weights.get_dim(i + 1);
    pdata.m_tile_total_input_size[i] = m_input.get_dim(i);
  }
}

mli_status Conv2d_CS::GetKernelPrivateData(void* kernel_private_data_buffer) {

  MLI_ASSERT(kernel_private_data_buffer != nullptr);

  // Batch checking
  MLI_ASSERT(m_input.get_dim(mli::kTensorBatchDim) == 1);

  // Channel checking
  MLI_ASSERT(m_weights.get_dim(mli::kKernelChannelOutDim) ==
      m_tile_total_output_size[mli::kTileChannelDim]);
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
  prv_data.inpzp_buffer = m_inpzp_buffer;
  prv_data.wtszp_buffer = m_wtszp_buffer;
  prv_data.inp_quant_axis = m_inp_quant_axis;
  prv_data.wts_quant_axis = m_wts_quant_axis;
  prv_data.config = m_config;
  prv_data.layout = LAYOUT_HWC;

  FillTilingParams(prv_data);

  std::memcpy(kernel_private_data_buffer, (void *)&prv_data, prv_data.size);

  return MLI_STATUS_OK;
}

mli_status Conv2d_CS::AttachBufferOffsets(Tensor<OffsetBuffer, KConvIORank> &input,
                                          Tensor<OffsetBuffer, KConvIORank> &output,
                                          OffsetBuffer &weights,
                                          OffsetBuffer &inpzeropts,
                                          OffsetBuffer &wtszeropts,
                                          OffsetBuffer &ctrl_buffer) {

  // TODO: uncomment DEPRICATED_METHOD after implementation of new version
  // DEPRICATED_METHOD

  MLI_ASSERT(output.get_buf().get_size() >= m_output_buffer_size * output.get_elem_size());

  // The metadata or descriptor is not required for ref kernel
  m_input.set_buf(input.get_buf());
  m_output.set_buf(output.get_buf());
  m_weights.set_buf(weights);

  // Zero Points maybe empty
  m_inpzp_buffer = inpzeropts;
  m_wtszp_buffer = wtszeropts;

  return MLI_STATUS_OK;
}

mli_status Conv2d_CS::AttachBufferOffsets(const OffsetBuffer& input,
                                          const OffsetBuffer& output,
                                          const OffsetBuffer& weights,
                                          const OffsetBuffer& inpzeropts,
                                          const OffsetBuffer& wtszeropts,
                                          const OffsetBuffer& ctrl_buffer) {
  // TODO: implementation
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
                                   uint32_t weights_inc[4]) {
  
  // TODO: uncomment DEPRICATED_METHOD after implementation of new version
  // DEPRICATED_METHOD
  
  m_use_tiling = true;
  for (unsigned i = 0; i < KConvIORank; i++) {
    m_tile_total_output_size[i] = output_total_size[i];
    m_tile_iteration_order[i] = iteration_order[i];
    m_tile_input_first_inc[i] = input_first_inc[i];
    m_tile_input_inc[i] = input_inc[i];
    m_tile_output_first_inc[i] = output_first_inc[i];
    m_tile_output_inc[i] = output_inc[i];
    m_tile_weights_inc[i] = weights_inc[i];
  }
  return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref