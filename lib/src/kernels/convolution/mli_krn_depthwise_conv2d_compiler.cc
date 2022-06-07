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
    : m_config{cfg}
    , m_input_zp{}
    , m_metadata{}
    , m_pd{pd}
{
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

  m_in = Tensor<OffsetBuffer, 4>(OffsetBuffer(), input_shape, input_stride);
  m_weights = Tensor<OffsetBuffer, 3>(OffsetBuffer(), weights_shape, weights_stride);
  m_output = Tensor<OffsetBuffer, 4>(OffsetBuffer(), output_shape, output_stride);

  m_input_buffer_size =
      service::GetBufferSize(in.get_rank(), input_shape, input_stride);
  m_weights_buffer_size
      = service::GetBufferSize(weights.get_rank(), weights_shape, weights_stride);
  m_output_buffer_size
      = service::GetBufferSize(output_tile_shape.get_rank(), output_shape, output_stride);
}

unsigned DepthwiseConv2d_CS::GetKernelPrivateDataSize() const {
  return sizeof(DepthwiseConv2DPrivateData);
}

unsigned DepthwiseConv2d_CS::GetRuntimeObjectSize() const {
  return sizeof(DepthwiseConv2d);
}

mli_status DepthwiseConv2d_CS::GetKernelPrivateData(void* kernel_private_data_buffer) {
  DepthwiseConv2DPrivateData dw_opaque_obj;

  dw_opaque_obj.size = sizeof(DepthwiseConv2DPrivateData);

  dw_opaque_obj.input_buffer = m_in.get_buf();
  dw_opaque_obj.weights_buffer = m_weights.get_buf();
  dw_opaque_obj.output_buffer = m_output.get_buf();
  dw_opaque_obj.inpzp_buffer = m_input_zp;
  dw_opaque_obj.wtszp_buffer = m_weights_zp;

  assert(m_in.get_dim(mli::kTensorChannelDim) == m_output.get_dim(mli::kTileChannelDim));
  assert(m_weights.get_dim(mli::kKernelDWChannelInDim) == m_output.get_dim(mli::kTileChannelDim));

  // TODO: support batch processing. Here we ignor batch dim  for now.
  MLI_ASSERT(m_in.get_dim(mli::kTensorBatchDim) == 1);
  dw_opaque_obj.input_h = m_in.get_dim(mli::kTensorHeightDim);
  dw_opaque_obj.input_w = m_in.get_dim(mli::kTensorWidthDim);
  dw_opaque_obj.input_output_c = m_in.get_dim(mli::kTensorChannelDim);

  dw_opaque_obj.output_h = m_output.get_dim(mli::kTileHeightDim);
  dw_opaque_obj.output_w = m_output.get_dim(mli::kTileWidthDim);

  dw_opaque_obj.weights_h = m_weights.get_dim(mli::kKernelDWHeightDim);
  dw_opaque_obj.weights_w = m_weights.get_dim(mli::kKernelDWWidthDim);

  dw_opaque_obj.input_h_stride = m_in.get_mem_stride(mli::kTensorHeightDim);
  dw_opaque_obj.input_w_stride = m_in.get_mem_stride(mli::kTensorWidthDim);

  dw_opaque_obj.output_h_stride = m_output.get_mem_stride(mli::kTileHeightDim);
  dw_opaque_obj.output_w_stride = m_output.get_mem_stride(mli::kTileWidthDim);

  dw_opaque_obj.weights_h_stride = m_weights.get_mem_stride(mli::kKernelDWHeightDim);
  dw_opaque_obj.weights_w_stride = m_weights.get_mem_stride(mli::kKernelDWWidthDim);

  // depthwise conv2d configuration
  dw_opaque_obj.stride_height = m_config.stride[0];
  dw_opaque_obj.stride_width = m_config.stride[1];
  dw_opaque_obj.padding_top = m_config.padding_begin[0];
  dw_opaque_obj.padding_left = m_config.padding_begin[1];
  dw_opaque_obj.padding_bottom = m_config.padding_end[0];
  dw_opaque_obj.padding_right = m_config.padding_end[1];
  dw_opaque_obj.dilation_height = m_config.dilation[0];
  dw_opaque_obj.dilation_width = m_config.dilation[1];

  std::memcpy(kernel_private_data_buffer, (void *)&dw_opaque_obj, sizeof(dw_opaque_obj));

  return MLI_STATUS_OK;
}

mli_status DepthwiseConv2d_CS::AttachBufferOffsets(Tensor<OffsetBuffer, 4> &input,
                                                   Tensor<OffsetBuffer, 4> &output,
                                                   OffsetBuffer &weights,
                                                   OffsetBuffer &inpzeropts,
                                                   OffsetBuffer &wtszeropts,
                                                   OffsetBuffer &metadata) {
  assert(input.get_buf().get_size() == m_input_buffer_size * input.get_elem_size());
  assert(output.get_buf().get_size() == m_output_buffer_size * output.get_elem_size());
  assert(weights.get_size() == m_weights_buffer_size * weights.get_elem_size());

  // The metadata or descriptor is not required for ref kernel
  m_in.set_buf(input.get_buf());
  m_output.set_buf(output.get_buf());
  m_weights.set_buf(weights);
  m_input_zp = inpzeropts;
  m_weights_zp = wtszeropts;

  return MLI_STATUS_OK;
}

mli_status DepthwiseConv2d_CS::EncodeWeights(Tensor<Buffer, 3> &weights,
                                             Buffer &encoded_weights,
                                             compression_mode_t mode){
  // the element size of source should eqaul to the encoded one's
  assert(weights.get_buf().get_size() == encoded_weights.get_size());
  // TODO: support other data types
  assert(weights.get_elem_size() == 1);

  for (uint32_t i = 0; i < weights.get_dim(0); ++i) {
    encoded_weights.write(i, weights.read<int8_t>(i));
  }

  return MLI_STATUS_OK;
}

unsigned DepthwiseConv2d_CS::GetEncodedWeightsSize() {
  return m_weights_buffer_size;
}

mli_status DepthwiseConv2d_CS::EncodeInpZeroPts(Tensor<Buffer, 1> &inpzeropts,
                                                Buffer &encoded_inpzeropts) {
  // only supports per-tensor quantization
  assert(inpzeropts.get_buf().get_size() / inpzeropts.get_elem_size() == 1);
  assert(inpzeropts.get_buf().get_size() == encoded_inpzeropts.get_size());
  // the element size of source should eqaul to the encoded one's
  assert(inpzeropts.get_elem_size() == encoded_inpzeropts.get_elem_size());
  // only supports 2 bytes value
  assert(inpzeropts.get_elem_size() == 2);

  for (uint32_t i = 0; i < inpzeropts.get_dim(0); ++i) {
    encoded_inpzeropts.write(i, inpzeropts.read<int16_t>(i));
  }

  return MLI_STATUS_OK;
}

unsigned DepthwiseConv2d_CS::GetEncodedInpZeroPtsSize() {
  // per-tensor quantization
  return 1;
}

mli_status DepthwiseConv2d_CS::EncodeWtsZeroPts(Tensor<Buffer, 1> &wtszeropts,
                                                Buffer &encoded_wtszeropts) {
  // only supports per-channel quantization
  assert(wtszeropts.get_buf().get_size() / wtszeropts.get_elem_size() ==
      m_in.get_dim(mli::kTensorChannelDim));
  assert(wtszeropts.get_buf().get_size() == encoded_wtszeropts.get_size());
  // the element size of source should eqaul to the encoded one's
  assert(wtszeropts.get_elem_size() == encoded_wtszeropts.get_elem_size());
  // only supports 2 bytes value
  assert(wtszeropts.get_elem_size() == 2);

  for (uint32_t i = 0; i < wtszeropts.get_dim(0); ++i) {
    encoded_wtszeropts.write(i, wtszeropts.read<int16_t>(i));
  }

  return MLI_STATUS_OK;
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

unsigned DepthwiseConv2d_CS::GetDataBufferSize() {
  return 0;
}

}  // namespace snps_arc::metaware::mli::ref