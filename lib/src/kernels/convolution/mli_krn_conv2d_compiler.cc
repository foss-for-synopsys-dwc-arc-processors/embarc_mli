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

Conv2d_CS::Conv2d_CS(const lib_mli::PlatformDescription pd,
                     const Tensor<NoBuffer, 4> &in, // B, H, W, Cin
                     const Tensor<NoBuffer, 5> &weights,  // G, H, W, Cin, Co
                     const Conv2DConfig &cfg,
                     const Tensor<NoBuffer, 4> &output_tile_shape // G, H, W, Co
                     )
    : m_config{cfg}
    , m_input_zp{}
    , m_weights_zp{}
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

  uint32_t weights_shape[5];
  int32_t weights_stride[5];
  for (uint32_t i = 0; i < 5; ++i) {
    weights_shape[i] = weights.get_dim(i);
    weights_stride[i] = weights.get_mem_stride(i);
  }

  m_in = Tensor<OffsetBuffer, 4>(OffsetBuffer(), input_shape, input_stride);
  m_weights = Tensor<OffsetBuffer, 5>(OffsetBuffer(), weights_shape, weights_stride);
  m_output = Tensor<OffsetBuffer, 4>(OffsetBuffer(), output_shape, output_stride);

  m_input_buffer_size =
      service::GetBufferSize(in.get_rank(), input_shape, input_stride);
  m_weights_buffer_size
      = service::GetBufferSize(weights.get_rank(), weights_shape, weights_stride);
  m_output_buffer_size
      = service::GetBufferSize(output_tile_shape.get_rank(), output_shape, output_stride);
}

unsigned Conv2d_CS::GetKernelPrivateDataSize() const {
  return sizeof(Conv2DPrivateData);
}

unsigned Conv2d_CS::GetRuntimeObjectSize() const {
  return sizeof(Conv2d);
}

mli_status Conv2d_CS::GetKernelPrivateData(void* kernel_private_data_buffer) {
  Conv2DPrivateData conv_opaque_obj;

  conv_opaque_obj.input_buffer = m_in.get_buf();
  conv_opaque_obj.weights_buffer = m_weights.get_buf();
  conv_opaque_obj.output_buffer = m_output.get_buf();
  conv_opaque_obj.inpzp_buffer = m_input_zp;
  conv_opaque_obj.wtszp_buffer = m_weights_zp;

  MLI_ASSERT(m_in.get_dim(mli::kTensorChannelDim) == m_weights.get_dim(mli::kKernelChannelInDim));
  MLI_ASSERT(m_weights.get_dim(mli::kKernelChannelOutDim) == m_output.get_dim(mli::kTileChannelDim));

  // TODO: support batch processing. Here we ignor batch dim for now.
  MLI_ASSERT(m_in.get_dim(mli::kTensorBatchDim) == 1);
  conv_opaque_obj.input_h = m_in.get_dim(mli::kTensorHeightDim);
  conv_opaque_obj.input_w = m_in.get_dim(mli::kTensorWidthDim);
  conv_opaque_obj.input_c = m_in.get_dim(mli::kTensorChannelDim);

  MLI_ASSERT(m_output.get_dim(mli::kTileGroupDim) == 1);
  conv_opaque_obj.output_h = m_output.get_dim(mli::kTileHeightDim);
  conv_opaque_obj.output_w = m_output.get_dim(mli::kTileWidthDim);
  conv_opaque_obj.output_c = m_output.get_dim(mli::kTileChannelDim);

  // TODO: support group conv2d
  MLI_ASSERT(m_weights.get_dim(mli::kKernelGroupDim) == m_output.get_dim(mli::kTileGroupDim));
  MLI_ASSERT(m_weights.get_dim(mli::kKernelGroupDim) == 1);
  conv_opaque_obj.weights_h = m_weights.get_dim(mli::kKernelHeightDim);
  conv_opaque_obj.weights_w = m_weights.get_dim(mli::kKernelWidthDim);
  MLI_ASSERT(m_weights.get_dim(mli::kKernelChannelInDim) == conv_opaque_obj.input_c);
  MLI_ASSERT(m_weights.get_dim(mli::kKernelChannelOutDim) == conv_opaque_obj.output_c);

  conv_opaque_obj.input_h_stride = m_in.get_mem_stride(mli::kTensorHeightDim);
  conv_opaque_obj.input_w_stride = m_in.get_mem_stride(mli::kTensorWidthDim);

  conv_opaque_obj.output_h_stride = m_output.get_mem_stride(mli::kTileHeightDim);
  conv_opaque_obj.output_w_stride = m_output.get_mem_stride(mli::kTileWidthDim);

  conv_opaque_obj.weights_h_stride = m_weights.get_mem_stride(mli::kKernelHeightDim);
  conv_opaque_obj.weights_w_stride = m_weights.get_mem_stride(mli::kKernelWidthDim);
  conv_opaque_obj.weights_c_stride = m_weights.get_mem_stride(mli::kKernelChannelInDim);

  // depthwise conv2d configuration
  conv_opaque_obj.stride_height = m_config.stride[0];
  conv_opaque_obj.stride_width = m_config.stride[1];
  conv_opaque_obj.padding_top = m_config.padding_begin[0];
  conv_opaque_obj.padding_left = m_config.padding_begin[1];
  conv_opaque_obj.padding_bottom = m_config.padding_end[0];
  conv_opaque_obj.padding_right = m_config.padding_end[1];
  conv_opaque_obj.dilation_height = m_config.dilation[0];
  conv_opaque_obj.dilation_width = m_config.dilation[1];
  conv_opaque_obj.groups = m_weights.get_dim(mli::kKernelGroupDim);

  std::memcpy(kernel_private_data_buffer, (void *)&conv_opaque_obj, sizeof(conv_opaque_obj));

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
  m_in.set_buf(input.get_buf());
  m_output.set_buf(output.get_buf());
  m_weights.set_buf(weights);
  // Zero Points maybe empty
  m_input_zp = inpzeropts;
  m_weights_zp = wtszeropts;

  return MLI_STATUS_OK;
}

mli_status Conv2d_CS::EncodeWeights(Tensor<Buffer, 5> &weights,
                                    Buffer &encoded_weights,
                                    compression_mode_t mode){
  // the element size of source should eqaul to the encoded one's
  MLI_ASSERT(weights.get_buf().get_size() == encoded_weights.get_size());
  // TODO: support other data types
  MLI_ASSERT(weights.get_elem_size() == sizeof(int8_t));

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

mli_status Conv2d_CS::EncodeInpZeroPts(Tensor<Buffer, 1> &inpzeropts,
                                       Buffer &encoded_inpzeropts) {
  // only supports per-tensor quantization
  MLI_ASSERT(encoded_inpzeropts.get_size() / encoded_inpzeropts.get_elem_size() == 1);
  MLI_ASSERT(inpzeropts.get_buf().get_size() == encoded_inpzeropts.get_size());
  // the element size of source should less than or equal to the encoded one's
  MLI_ASSERT(inpzeropts.get_elem_size() <= encoded_inpzeropts.get_elem_size());

  if (inpzeropts.get_elem_size() == sizeof(int8_t)) {
    for (uint32_t i = 0; i < inpzeropts.get_dim(0); ++i) {
      encoded_inpzeropts.write(i, static_cast<int16_t>(inpzeropts.read<int8_t>(i)));
    }
  } else {
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}

unsigned Conv2d_CS::GetEncodedInpZeroPtsSize() {
  // per-tensor quantization
  return 1;
}

mli_status Conv2d_CS::EncodeWtsZeroPts(Tensor<Buffer, 1> &wtszeropts,
                                       Buffer &encoded_wtszeropts) {
  // only supports per-channel quantization
  MLI_ASSERT(encoded_wtszeropts.get_size() / encoded_wtszeropts.get_elem_size() ==
      m_weights.get_dim(mli::kKernelChannelOutDim));
  MLI_ASSERT(wtszeropts.get_buf().get_size() == encoded_wtszeropts.get_size());
  // the element size of source less than or equal to the encoded one's
  MLI_ASSERT(wtszeropts.get_elem_size() <= encoded_wtszeropts.get_elem_size());

  if (wtszeropts.get_elem_size() == sizeof(int8_t)) {
    for (uint32_t i = 0; i < wtszeropts.get_dim(0); ++i) {
      encoded_wtszeropts.write(i, static_cast<int16_t>(wtszeropts.read<int8_t>(i)));
    }
  } else {
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
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