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
#include "mli_helpers_api.hpp"


namespace snps_arc::metaware::mli::ref {

/**
 * @deprecated
 */
TransposeConv2D_CS::TransposeConv2D_CS(
    const PlatformDescription pd,
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
  DEPRECATED_METHOD
  m_inp_quant_axis = kPerTensorQuantDim;
  m_wts_quant_axis = kKernelChannelOutDim;
  m_weights_buffer_size = service::GetBufferSize(weights.get_tensor());
}

TransposeConv2D_CS::TransposeConv2D_CS(const PlatformDescription pd,
                                       const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIterRank>& input,
                                       const TensorIterator<NoBuffer, kTransposeConvZPRank, kTransposeConvIterRank>& input_zp,
                                       const TensorIterator<NoBuffer, kTransposeConvWRank, kTransposeConvIterRank>& weights,
                                       const TensorIterator<NoBuffer, kTransposeConvZPRank, kTransposeConvIterRank>& weights_zp,
                                       const TransposeConv2DConfig& cfg,
                                       const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank>& output)
  : m_input(input),
    m_weights(weights),
    m_weights_zp(weights_zp),
    m_output(output),
    m_config(cfg),
    m_pd(pd) {
  m_inp_quant_axis = kPerTensorQuantDim;
  m_wts_quant_axis = kKernelChannelOutDim;
  m_weights_buffer_size = service::GetBufferSize(weights.get_tensor());
}

unsigned TransposeConv2D_CS::GetKernelPrivateDataSize() const {
  return sizeof(TransposeConv2DPrivateData);
}

unsigned TransposeConv2D_CS::GetRuntimeObjectSize() const {
  return sizeof(TransposeConv2D);
}

mli_status TransposeConv2D_CS::GetKernelPrivateData(
    void *kernel_private_data_buffer) {

  MLI_ASSERT(kernel_private_data_buffer != nullptr);
  MLI_ASSERT(m_input.get_dim(kGroupTensorBatchDim) == 1);
  MLI_ASSERT(m_weights.get_dim(kKernelChannelOutDim) ==
      m_output.get_dim(kGroupTensorChannelDim));
  MLI_ASSERT(m_weights.get_dim(kKernelChannelInDim) ==
      m_input.get_dim(kGroupTensorChannelDim));
  MLI_ASSERT(m_weights.get_dim(kKernelGroupDim) == 1);

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

/**
 * @deprecated
 */
mli_status TransposeConv2D_CS::EncodeWeights(Tensor<Buffer, kTransposeConvWRank> &weights,
                                             Buffer &encoded_weights,
                                             compression_mode_t mode) {
  DEPRECATED_METHOD
  return service::EncodeWeights(weights, encoded_weights);
}

mli_status TransposeConv2D_CS::EncodeWeightsAndZeroPts(TensorIterator<Buffer, kTransposeConvWRank, kTransposeConvIterRank>& weights,
                                                       TensorIterator<Buffer, kTransposeConvZPRank, kTransposeConvIterRank>& weights_zp,
                                                       Buffer& encoded_weights) {
  return service::EncodeWeightsAndZeroPts(weights.get_tensor(), weights_zp.get_tensor(), encoded_weights);
};

unsigned TransposeConv2D_CS::GetEncodedWeightsSize() const {
  return m_weights_buffer_size;
};

/**
 * @deprecated
 */
mli_status TransposeConv2D_CS::EncodeInpZeroPts(Tensor<Buffer, kTransposeConvZPRank> &inpzeropts,
                                                Buffer &encoded_inpzeropts) {
  DEPRECATED_METHOD
  constexpr int channel_axis = kTensorChannelDim;
  uint32_t channel_length = m_input.get_dim(channel_axis);
  return service::EncodeZeroPts<channel_axis>(
    inpzeropts, encoded_inpzeropts, m_inp_quant_axis, channel_length);
}

mli_status TransposeConv2D_CS::EncodeInpZeroPts(TensorIterator<Buffer, kTransposeConvZPRank, kTransposeConvZPIterRank>& input_zp,
                                                Buffer& encoded_input_zp) {
  constexpr int channel_axis = kTensorChannelDim;
  uint32_t channel_length = m_input.get_dim(channel_axis);
  return service::EncodeZeroPts<channel_axis>(input_zp.get_tensor(), encoded_input_zp, m_inp_quant_axis, channel_length);
}

/**
 * @deprecated
 */
mli_status TransposeConv2D_CS::EncodeWtsZeroPts(Tensor<Buffer, kTransposeConvZPRank> &wtszeropts,
                                                Buffer &encoded_wtszeropts) {
  DEPRECATED_METHOD
  constexpr int channel_axis = kKernelChannelOutDim;
  uint32_t channel_length = m_weights.get_dim(channel_axis);
  return service::EncodeZeroPts<channel_axis>(wtszeropts, encoded_wtszeropts,
                                              m_wts_quant_axis, channel_length);
}

unsigned TransposeConv2D_CS::GetEncodedWtsZeroPtsSize() const {
  // per-channel quantization
  return m_weights.get_dim(kKernelChannelOutDim);
}

unsigned TransposeConv2D_CS::GetEncodedInpZeroPtsSize() const {
  // per-tensor quantization
  return 1;
}

}  // namespace snps_arc::metaware::mli::ref
