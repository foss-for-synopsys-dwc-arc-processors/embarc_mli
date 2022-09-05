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

TransposeConv2D_CS::TransposeConv2D_CS(
    const lib_mli::PlatformDescription pd,
    const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> &input,
    const TensorIterator<NoBuffer, kTransposeConvWRank, kTransposeConvWIterRank> &weights,
    const TensorIterator<NoBuffer, kTransposeConvZPRank, kTransposeConvZPIterRank> &weights_zp,
    const TransposeConv2DConfig &cfg,
    const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> &output)
    : m_input(input),
      m_weights(weights),
      m_weights_zp(weights_zp),
      m_output(output),
      m_config(cfg),
      m_pd(pd) {
  uint32_t input_shape[kConvIORank];
  int32_t input_stride[kConvIORank];
  uint32_t output_shape[kConvIORank];
  int32_t output_stride[kConvIORank];
  input.get_full_shape(input_shape);
  input.get_mem_strides(input_stride);
  output.get_full_shape(output_shape);
  output.get_mem_strides(output_stride);

  uint32_t weights_shape[kConvWRank];
  int32_t weights_stride[kConvWRank];
  weights.get_full_shape(weights_shape);
  weights.get_mem_strides(weights_stride);

  m_inp_quant_axis = kPerTensorQuantDim;
  m_wts_quant_axis = kKernelChannelOutDim;
}

unsigned TransposeConv2D_CS::GetKernelPrivateDataSize() const {
  return sizeof(TransposeConv2DPrivateData);
}

unsigned TransposeConv2D_CS::GetRuntimeObjectSize() const {
  return sizeof(Conv2d);
}

mli_status TransposeConv2D_CS::GetKernelPrivateData(
    void *kernel_private_data_buffer) {
  MLI_ASSERT(kernel_private_data_buffer != nullptr);

  // Batch checking
  MLI_ASSERT(m_input.get_dim(mli::kTensorBatchDim) == 1);

  // Channel checking
  MLI_ASSERT(m_weights.get_dim(mli::kKernelChannelOutDim) ==
      m_output.get_dim(mli::kTileChannelDim));

  MLI_ASSERT(m_weights.get_dim(mli::kKernelChannelInDim) ==
      m_input.get_dim(mli::kTensorChannelDim));

  MLI_ASSERT(m_input.get_dim(mli::kTensorChannelDim) ==
             m_weights.get_dim(mli::kKernelChannelInDim));

  // Group checking
  MLI_ASSERT(m_weights.get_dim(mli::kKernelGroupDim) == 1);

  MLI_ASSERT(m_weights.get_dim(mli::kKernelGroupDim) ==
      m_output.get_dim(mli::kTileGroupDim));

  TransposeConv2DPrivateData prv_data;
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

mli_status TransposeConv2D_CS::AttachBufferOffsets(
    const OffsetBuffer &input, const OffsetBuffer &output,
    const OffsetBuffer &weights, const OffsetBuffer &inpzeropts,
    const OffsetBuffer &wtszeropts, const OffsetBuffer &ctrl_buffer) {
  m_input.set_buf(input);
  m_output.set_buf(output);
  m_weights.set_buf(weights);
  m_weights_zp.set_buf(wtszeropts);
  m_inpzp_buffer = inpzeropts;
  return MLI_STATUS_OK;
}

mli_status TransposeConv2D_CS::EncodeWeights(Tensor<Buffer, kTransposeConvWRank> &weights,
                                             Buffer &encoded_weights,
                                             compression_mode_t mode) {
  return service::EncodeWeights(weights, encoded_weights);
}

mli_status TransposeConv2D_CS::EncodeInpZeroPts(Tensor<Buffer, kConvZPRank> &inpzeropts,
                                                Buffer &encoded_inpzeropts) {
  constexpr int channel_axis = mli::kTensorChannelDim;
  uint32_t channel_length = m_input.get_dim(channel_axis);
  return service::EncodeZeroPts<channel_axis>(
    inpzeropts, encoded_inpzeropts, m_inp_quant_axis, channel_length);
}

mli_status TransposeConv2D_CS::EncodeWtsZeroPts(Tensor<Buffer, kConvZPRank> &wtszeropts,
                                                Buffer &encoded_wtszeropts) {
  constexpr int channel_axis = mli::kKernelChannelOutDim;
  uint32_t channel_length = m_weights.get_dim(channel_axis);
  return service::EncodeZeroPts<channel_axis>(wtszeropts, encoded_wtszeropts,
                                              m_wts_quant_axis, channel_length);
}

unsigned TransposeConv2D_CS::GetEncodedWtsZeroPtsSize() const {
  // per-channel quantization
  return m_weights.get_dim(mli::kKernelChannelOutDim) ;
}

unsigned TransposeConv2D_CS::GetEncodedInpZeroPtsSize() const {
  // per-tensor quantization
  return 1;
}

}  // namespace snps_arc::metaware::mli::ref