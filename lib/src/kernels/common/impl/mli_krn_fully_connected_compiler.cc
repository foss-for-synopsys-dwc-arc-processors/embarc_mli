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
 */
FullyConnected_CS::FullyConnected_CS(const lib_mli::PlatformDescription pd,
                                     const Tensor<NoBuffer, kFullyConnectedIORank> &in,
                                     const Tensor<NoBuffer, kFullyConnectedWRank>  &weights,
                                     const Tensor<NoBuffer, kFullyConnectedZPRank> &wtszp,
                                     const Tensor<NoBuffer, kFullyConnectedIORank> &output_tile_shape)
    : m_pd{pd},
      m_in{Tensor<OffsetBuffer, kFullyConnectedIORank>(OffsetBuffer(), in)},
      m_weights{Tensor<OffsetBuffer, kFullyConnectedWRank>(OffsetBuffer(), weights)},
      m_wtszp{Tensor<OffsetBuffer, kFullyConnectedZPRank>(OffsetBuffer(), wtszp)},
      m_output{Tensor<OffsetBuffer, kFullyConnectedIORank>(OffsetBuffer(), output_tile_shape)} {
  DEPRECATED_METHOD
}

/**
 * @deprecated
 */
FullyConnected_CS::FullyConnected_CS(const lib_mli::PlatformDescription pd,
                                     const Tensor<NoBuffer, kFullyConnectedIORank> &in,
                                     const Tensor<NoBuffer, kFullyConnectedWRank>  &weights,
                                     const Tensor<NoBuffer, kFullyConnectedIORank> &output_tile_shape)
    : FullyConnected_CS(pd, in, weights, Tensor<NoBuffer, kFullyConnectedZPRank>(), output_tile_shape) {
  DEPRECATED_METHOD
}

FullyConnected_CS::FullyConnected_CS(const PlatformDescription pd,
                                     const TensorIterator<NoBuffer, kFullyConnectedIORank, kFullyConnectedIterRank>& input,
                                     const TensorIterator<NoBuffer, kFullyConnectedWRank, kFullyConnectedIterRank>& weights,
                                     const TensorIterator<NoBuffer, kFullyConnectedZPRank, kFullyConnectedIterRank>& weights_zp,
                                     const FullyConnectedConfig& cfg,
                                     const TensorIterator<NoBuffer, kFullyConnectedIORank, kFullyConnectedIterRank>& output)
  : m_pd(pd),
    m_in(OffsetBuffer(), input.get_tensor()),
    m_weights(OffsetBuffer(), weights.get_tensor()),
    m_wtszp(OffsetBuffer(), weights_zp.get_tensor()),
    m_output(OffsetBuffer(), output.get_tensor()) {
}


unsigned FullyConnected_CS::GetKernelPrivateDataSize() const {
  return sizeof(FullyConnectedPrivateData);
}

unsigned FullyConnected_CS::GetRuntimeObjectSize() const {
  return sizeof(FullyConnected);
}

mli_status FullyConnected_CS::GetKernelPrivateData(void* kernel_private_data_buffer) {
  FullyConnectedPrivateData fc_opaque_obj;

  fc_opaque_obj.input_buffer = m_in.get_buf();
  fc_opaque_obj.weights_buffer = m_weights.get_buf();
  fc_opaque_obj.output_buffer = m_output.get_buf();
  fc_opaque_obj.wtszp_buffer = m_weights_zp;
  // Only two types of weights zero point quantization are supported, per-tensor or per-channel.
  // -1 indicates per-tensor, 1 indicates per-channel.
  // Potential bug if m_weights.shape[1] equals 1
  fc_opaque_obj.qt_wtszp_axis = (GetEncodedWtsZeroPtsSize() == 1) ? -1 : kKernelFCChannelOutDim;
  MLI_ASSERT(m_in.get_dim(mli::kTensorBatchDim) == m_output.get_dim(mli::kTensorBatchDim));
  MLI_ASSERT(m_in.get_dim(1) == m_weights.get_dim(mli::kKernelFCChannelInDim));
  MLI_ASSERT(m_weights.get_dim(mli::kKernelFCChannelOutDim) == m_output.get_dim(mli::kKernelFCChannelOutDim));

  // TODO: support batch processing. Here we ignor batch dim for now.
  MLI_ASSERT(m_in.get_dim(mli::kTensorBatchDim) == 1);
  fc_opaque_obj.input_n = m_in.get_dim(mli::kTensorBatchDim);
  fc_opaque_obj.input_ic = m_in.get_dim(mli::kKernelFCChannelOutDim);

  fc_opaque_obj.output_n = m_output.get_dim(mli::kTensorBatchDim);
  fc_opaque_obj.output_oc = m_output.get_dim(mli::kKernelFCChannelOutDim);

  fc_opaque_obj.weights_ic = m_weights.get_dim(mli::kKernelFCChannelInDim);
  fc_opaque_obj.weights_oc = m_weights.get_dim(mli::kKernelFCChannelOutDim);

  fc_opaque_obj.input_n_stride = m_in.get_mem_stride(mli::kTensorBatchDim);
  fc_opaque_obj.input_ic_stride = m_in.get_mem_stride(mli::kKernelFCChannelOutDim);

  fc_opaque_obj.output_n_stride = m_output.get_mem_stride(mli::kTensorBatchDim);
  fc_opaque_obj.output_oc_stride = m_output.get_mem_stride(mli::kKernelFCChannelOutDim);

  fc_opaque_obj.weights_ic_stride = m_weights.get_mem_stride(mli::kKernelFCChannelInDim);
  fc_opaque_obj.weights_oc_stride = m_weights.get_mem_stride(mli::kKernelFCChannelOutDim);

  std::memcpy(kernel_private_data_buffer, (void *)&fc_opaque_obj, sizeof(fc_opaque_obj));

  return MLI_STATUS_OK;
}

mli_status FullyConnected_CS::AttachBufferOffsets(const Tensor<OffsetBuffer, kFullyConnectedIORank> &input,
                                                  const Tensor<OffsetBuffer, kFullyConnectedIORank> &output,
                                                  const OffsetBuffer &weights,
                                                  const OffsetBuffer &wtszeropts,
                                                  const OffsetBuffer &ctrl_buffer) {
  DEPRECATED_METHOD
  MLI_ASSERT(input.get_buf().get_size() >= GetInputBufferSize());
  MLI_ASSERT(output.get_buf().get_size() >= GetOutputBufferSize());
  MLI_ASSERT(weights.get_size() >= GetWeightsBufferSize());
  MLI_ASSERT(wtszeropts.get_elem_size() == 2);
  m_in.set_buf(input.get_buf());
  m_output.set_buf(output.get_buf());
  m_weights.set_buf(weights);
  m_weights_zp = wtszeropts;

  return MLI_STATUS_OK;
}

mli_status FullyConnected_CS::AttachBufferOffsets(const OffsetBuffer& input,
                                                  const OffsetBuffer& output,
                                                  const OffsetBuffer& weights_and_zeropts,
                                                  const OffsetBuffer& ctrl_buffer) {
  m_in.set_buf(input);
  m_output.set_buf(output);
  m_weights.set_buf(weights_and_zeropts);
  m_weights_zp = OffsetBuffer(weights_and_zeropts, GetWeightsBufferSize() * weights_and_zeropts.get_elem_size());
  return MLI_STATUS_OK;
};

/**
 * @deprecated
 */
mli_status FullyConnected_CS::EncodeWeights(const Tensor<Buffer, kFullyConnectedWRank> &weights,
                                            Buffer &encoded_weights) {
  DEPRECATED_METHOD
  // the element size of source should eqaul to the encoded one's
  MLI_ASSERT(weights.get_buf().get_size() == encoded_weights.get_size());
  // TODO: support other data types
  MLI_ASSERT(weights.get_elem_size() == 1);

  if (weights.get_elem_size() == sizeof(int8_t)) {
    for (uint32_t i = 0; i < weights.get_buf().get_size(); ++i) {
      encoded_weights.write(i, weights.read<int8_t>(i));
    }
  } else {
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}



mli_status FullyConnected_CS::EncodeWeightsAndZeroPts(TensorIterator<Buffer, kFullyConnectedWRank, kFullyConnectedIterRank>& weights,
                                                      TensorIterator<Buffer, kFullyConnectedZPRank, kFullyConnectedIterRank>& weights_zp,
                                                      Buffer& encoded_weights) {
  MLI_ASSERT(weights.get_buf().get_size() + weights_zp.get_buf().get_size() == encoded_weights.get_size());
  MLI_ASSERT(weights.get_elem_size() == sizeof(int8_t));
  MLI_ASSERT(weights_zp.get_elem_size() == sizeof(int8_t));
  return service::EncodeWeightsAndZeroPts(weights.get_tensor(), weights_zp.get_tensor(), encoded_weights);
}

unsigned FullyConnected_CS::GetEncodedWeightsSize() const {
  return GetWeightsBufferSize();
}

/**
 * @deprecated
 */
mli_status FullyConnected_CS::EncodeWtsZeroPts(const Tensor<Buffer, 1> &wtszeropts,
                                               Buffer &encoded_wtszeropts) {
  DEPRECATED_METHOD
  MLI_ASSERT(wtszeropts.get_buf().get_size() == encoded_wtszeropts.get_size());
  // the element size of source less than or equal to the encoded one's
  MLI_ASSERT(wtszeropts.get_elem_size() <= encoded_wtszeropts.get_elem_size());

  if (wtszeropts.get_elem_size() == sizeof(int8_t)) {
    // per-tensor quantization
    if(1 == m_wtszp.get_dim(0)) {
       encoded_wtszeropts.write(0, static_cast<int16_t>(wtszeropts.read<int8_t>(0)));
    }
    else {
      MLI_ASSERT(encoded_wtszeropts.get_size() / encoded_wtszeropts.get_elem_size() ==
      m_weights.get_dim(kKernelFCChannelOutDim));
      for (uint32_t i = 0; i < wtszeropts.get_dim(kKernelFCChannelOutDim); ++i) {
        encoded_wtszeropts.write(i, static_cast<int16_t>(wtszeropts.read<int8_t>(i)));
      }
    }

  } else {
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}

unsigned FullyConnected_CS::GetEncodedWtsZeroPtsSize() const {
  return service::GetBufferSize(m_wtszp);
}

unsigned FullyConnected_CS::GetInputBufferSize() const {
  return service::GetBufferSize(m_in);
}

unsigned FullyConnected_CS::GetWeightsBufferSize() const {
  return service::GetBufferSize(m_weights);
}

unsigned FullyConnected_CS::GetOutputBufferSize() const {
  return service::GetBufferSize(m_output);
}
unsigned FullyConnected_CS::GetZeroPointBufferSize() const {
  return 0;
}

}  // namespace snps_arc::metaware::mli::ref