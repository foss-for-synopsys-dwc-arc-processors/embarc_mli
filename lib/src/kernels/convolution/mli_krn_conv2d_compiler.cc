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

namespace snps_arc::metaware::mli::ref {

Conv2d_CS::Conv2d_CS(const lib_mli::PlatformDescription pd,
                     const Tensor<NoBuffer, 4> &in, // B, H, W, Cin
                     const Tensor<NoBuffer, 5> &weights,  // G, H, W, Cin, Co
                     const Conv2DConfig &cfg,
                     const Tensor<NoBuffer, 4> &output_tile_shape // G, H, W, Co
                    ) : m_pd{pd} {
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

  uint32_t weights_shape[5];
  int32_t weights_stride[5];
  for (uint32_t i = 0; i < 5; ++i) {
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
  m_input = Tensor<OffsetBuffer, 4>(OffsetBuffer(), in);
  m_weights = Tensor<OffsetBuffer, 5>(OffsetBuffer(), weights);
  m_output = Tensor<OffsetBuffer, 4>(OffsetBuffer(), output_tile_shape);

  // Init convolution config
  m_config = cfg;
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

  std::memcpy(kernel_private_data_buffer, (void *)&prv_data, prv_data.size);

  return MLI_STATUS_OK;
}

mli_status Conv2d_CS::AttachBufferOffsets(Tensor<OffsetBuffer, 4> &input,
                                          Tensor<OffsetBuffer, 4> &output,
                                          OffsetBuffer &weights,
                                          OffsetBuffer &inpzeropts,
                                          OffsetBuffer &wtszeropts,
                                          OffsetBuffer &metadata) {
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

mli_status Conv2d_CS::EncodeWeights(Tensor<Buffer, 5> &weights,
                                    Buffer &encoded_weights,
                                    compression_mode_t mode){
  // the element size of source should eqaul to the encoded one's
  MLI_ASSERT(weights.get_buf().get_size() == encoded_weights.get_size());

  // TODO: support other data types
  if (weights.get_elem_size() == sizeof(int8_t)) {
    for (uint32_t i = 0; i < weights.get_buf().get_size(); ++i) {
      encoded_weights.write(i, weights.read<int8_t>(i));
    }
  } else {
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}

unsigned Conv2d_CS::GetEncodedWeightsSize() {
  return m_weights_buffer_size;
}

template <int channel_axis>
mli_status EncodeZeroPts(const Tensor<Buffer, 1>& zeropts,
                         Buffer& encoded_zeropts,
                         int& quant_axis,
                         uint32_t channel_length) {
  // should have the same total size
  MLI_ASSERT(zeropts.get_buf().get_size() == encoded_zeropts.get_size());
  // the element size of source should less than or equal to the encoded one's
  MLI_ASSERT(zeropts.get_elem_size() <= encoded_zeropts.get_elem_size());
  // should have the same number of elements
  MLI_ASSERT(zeropts.get_dim(0) ==
    encoded_zeropts.get_size() / encoded_zeropts.get_elem_size());

  if (zeropts.get_dim(0) == 1) {
    // per-tensor quantization
    quant_axis = -1;
  } else if (zeropts.get_dim(0) == channel_length) {
    // per-channel quantization
    quant_axis = channel_axis;
  } else {
    return MLI_STATUS_SHAPE_MISMATCH;
  }

  if (zeropts.get_elem_size() == sizeof(int8_t)) {
    for (uint32_t i = 0; i < zeropts.get_dim(0); ++i) {
      encoded_zeropts.write(i, static_cast<int16_t>(zeropts.read<int8_t>(i)));
    }
  } else {
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}

mli_status Conv2d_CS::EncodeInpZeroPts(Tensor<Buffer, 1> &inpzeropts,
                                       Buffer &encoded_inpzeropts) {
  constexpr int channel_axis = mli::kTensorChannelDim;
  uint32_t channel_length = m_input.get_dim(channel_axis);
  return EncodeZeroPts<channel_axis>(inpzeropts, encoded_inpzeropts,
                                     m_inp_quant_axis,
                                     channel_length);
}

unsigned Conv2d_CS::GetEncodedInpZeroPtsSize() {
  // per-tensor quantization
  return 1;
}

mli_status Conv2d_CS::EncodeWtsZeroPts(Tensor<Buffer, 1> &wtszeropts,
                                       Buffer &encoded_wtszeropts) {
  constexpr int channel_axis = mli::kKernelChannelOutDim;
  uint32_t channel_length = m_weights.get_dim(channel_axis);
  return EncodeZeroPts<channel_axis>(wtszeropts, encoded_wtszeropts,
                                     m_wts_quant_axis,
                                     channel_length);
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

unsigned Conv2d_CS::GetDataBufferSize() {
  return 0;
}

mli_status Conv2d_CS::SetIterators(uint32_t output_total_size[4],
                                   uint32_t iteration_order[4],
                                   uint32_t input_first_inc[4],
                                   uint32_t input_inc[4],
                                   uint32_t output_first_inc[4],
                                   uint32_t output_inc[4],
                                   uint32_t weights_inc[4]) {
    // TODO: Store iterator params
    return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref