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

FullyConnected_CS::FullyConnected_CS(const lib_mli::PlatformDescription pd,
                                     const Tensor<NoBuffer, 2> &in,
                                     const Tensor<NoBuffer, 2> &weights,
                                     const Tensor<NoBuffer, 1> &wtszp,
                                     const Tensor<NoBuffer, 2> &output_tile_shape)
    : m_pd{pd},
      m_in{Tensor<OffsetBuffer, 2>(OffsetBuffer(), in)},
      m_weights{Tensor<OffsetBuffer, 2>(OffsetBuffer(), weights)},
      m_wtszp{Tensor<OffsetBuffer, 1>(OffsetBuffer(), wtszp)},
      m_output{Tensor<OffsetBuffer, 2>(OffsetBuffer(), output_tile_shape)} {
  uint32_t input_shape[2];
  uint32_t output_shape[2];
  int32_t input_stride[2];
  int32_t output_stride[2];
  uint32_t wtzp_shape[1];
  int32_t wtzp_stride[1];
  for (uint32_t i = 0; i < 2; ++i) {
    input_shape[i] = in.get_dim(i);
    input_stride[i] = in.get_mem_stride(i);
    output_shape[i] = output_tile_shape.get_dim(i);
    output_stride[i] = output_tile_shape.get_mem_stride(i);
  }
  uint32_t weights_shape[2];
  int32_t weights_stride[2];
  for (uint32_t i = 0; i < 2; ++i) {
    weights_shape[i] = weights.get_dim(i);
    weights_stride[i] = weights.get_mem_stride(i);
  }
  wtzp_shape[0] = wtszp.get_dim(0);
  wtzp_stride[0] = wtszp.get_mem_stride(0);

  m_input_buffer_size =
      service::GetBufferSize(in.get_rank(), input_shape, input_stride);
  m_weights_buffer_size
      = service::GetBufferSize(weights.get_rank(), weights_shape, weights_stride);
  m_wtszp_buffer_size =
      service::GetBufferSize(wtszp.get_rank(), wtzp_shape, wtzp_stride);
  m_output_buffer_size = service::GetBufferSize(output_tile_shape.get_rank(),
                                                output_shape, output_stride);
}

FullyConnected_CS::FullyConnected_CS(const lib_mli::PlatformDescription pd,
                                     const Tensor<NoBuffer, 2> &in,
                                     const Tensor<NoBuffer, 2> &weights,
                                     const Tensor<NoBuffer, 2> &output_tile_shape)
    : FullyConnected_CS(pd, in, weights, Tensor<NoBuffer, 1>(), output_tile_shape) {
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
  fc_opaque_obj.qt_wtszp_axis = (m_wtszp_buffer_size == 1) ? -1 : kKernelFCChannelOutDim;
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

mli_status FullyConnected_CS::AttachBufferOffsets(const Tensor<OffsetBuffer, 2> &input,
                                                  const Tensor<OffsetBuffer, 2> &output,
                                                  const OffsetBuffer &weights,
                                                  const OffsetBuffer &wtszeropts,
                                                  const OffsetBuffer &metadata) {
  MLI_ASSERT(input.get_buf().get_size() >= m_input_buffer_size * input.get_elem_size());
  MLI_ASSERT(output.get_buf().get_size() >= m_output_buffer_size * output.get_elem_size());
  MLI_ASSERT(weights.get_size() >= m_weights_buffer_size * weights.get_elem_size());
  MLI_ASSERT(wtszeropts.get_elem_size() == 2);
  m_in.set_buf(input.get_buf());
  m_output.set_buf(output.get_buf());
  m_weights.set_buf(weights);
  m_weights_zp = wtszeropts;

  return MLI_STATUS_OK;
}

mli_status FullyConnected_CS::EncodeWeights(const Tensor<Buffer, 2> &weights,
                                            Buffer &encoded_weights) {
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

unsigned FullyConnected_CS::GetEncodedWeightsSize() const {
  return m_weights_buffer_size;
}

mli_status FullyConnected_CS::EncodeWtsZeroPts(const Tensor<Buffer, 1> &wtszeropts,
                                               Buffer &encoded_wtszeropts) {
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
  return m_wtszp_buffer_size;
}

unsigned FullyConnected_CS::GetInputBufferSize() const {
  return m_input_buffer_size;
}

unsigned FullyConnected_CS::GetWeightsBufferSize() const {
  return m_weights_buffer_size;
}

unsigned FullyConnected_CS::GetOutputBufferSize() const {
  return m_output_buffer_size;
}
unsigned FullyConnected_CS::GetZeroPointBufferSize() const {
  return 0;
}

unsigned FullyConnected_CS::GetDataBufferSize() const {
  return 0;
}

}  // namespace snps_arc::metaware::mli::ref