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

#include "mli_ref_krn_conv_private_types.hpp"



namespace snps_arc::metaware::mli::ref {

DepthwiseConv2d_CS::DepthwiseConv2d_CS(const Tensor<OffsetBuffer, 4> &in,
                                       const Tensor<OffsetBuffer, 3> &weights,
                                       const DwConv2DConfig &cfg,
                                       const Tensor<OffsetBuffer, 4> &output_tile_shape) 
    : m_in{in}
    , m_weights{weights}
    , m_output{output_tile_shape}
    , m_config{cfg}
    , m_input_zp{}
    , m_metadata{}
{  
  // TODO: Check data compatibility 
};

unsigned DepthwiseConv2d_CS::GetKernelPrivateDataSize() { 
  return sizeof(DepthwiseConv2DPrivateData);
}

unsigned DepthwiseConv2d_CS::GetRuntimeObjectSize() {
  return sizeof(DepthwiseConv2d);
}

mli_status DepthwiseConv2d_CS::GetKernelPrivateData(void* kernel_private_data_buffer) {
  DepthwiseConv2DPrivateData dw_opaque_obj;

  dw_opaque_obj.metadata_mem_id = m_metadata.get_mem_idx();
  dw_opaque_obj.input_mem_id = m_in.get_buf().get_mem_idx();
  dw_opaque_obj.weights_mem_id = m_weights.get_buf().get_mem_idx();
  dw_opaque_obj.output_mem_id = m_output.get_buf().get_mem_idx();
  dw_opaque_obj.input_zp_mem_id = m_input_zp.get_mem_idx();

  dw_opaque_obj.metadata_offset = m_metadata.get_offset();
  dw_opaque_obj.input_mem_offset = m_in.get_buf().get_offset();
  dw_opaque_obj.weights_mem_offset = m_in.get_buf().get_offset();
  dw_opaque_obj.output_mem_offset = m_in.get_buf().get_offset();
  dw_opaque_obj.input_zp_mem_offset = m_input_zp.get_offset();

  // TODO: support batch processing. Here we ignor batch dim  for now. 
  // TODO: use named constants provided in parallel branch to define indexes instead of magic numbers
  dw_opaque_obj.input_h = m_in.get_dim(1);
  dw_opaque_obj.input_w = m_in.get_dim(2);
  dw_opaque_obj.input_output_c = m_in.get_dim(3);

  dw_opaque_obj.output_h = m_output.get_dim(1);
  dw_opaque_obj.output_w = m_output.get_dim(2);

  dw_opaque_obj.weights_h = m_weights.get_dim(0);
  dw_opaque_obj.weights_w = m_weights.get_dim(1);

  dw_opaque_obj.input_h_stride = m_in.get_mem_stride(1);
  dw_opaque_obj.input_w_stride = m_in.get_mem_stride(2);

  dw_opaque_obj.output_h_stride = m_output.get_mem_stride(1);
  dw_opaque_obj.output_w_stride = m_output.get_mem_stride(2);

  dw_opaque_obj.weights_h_stride = m_weights.get_mem_stride(0);
  dw_opaque_obj.weights_w_stride = m_weights.get_mem_stride(1);
  
  dw_opaque_obj.stride_height = m_config.stride[0];
  dw_opaque_obj.stride_width = m_config.stride[1];
  dw_opaque_obj.dilation_height = m_config.dilation[0];
  dw_opaque_obj.dilation_width = m_config.dilation[1];

  dw_opaque_obj.padding_left = m_config.padding_begin[1];
  dw_opaque_obj.padding_right = m_config.padding_end[1];
  dw_opaque_obj.padding_top = m_config.padding_begin[0];
  dw_opaque_obj.padding_bottom = m_config.padding_end[0];

  std::memcpy(kernel_private_data_buffer, (void *)&dw_opaque_obj, sizeof(dw_opaque_obj));

  return MLI_STATUS_OK;
}

mli_status DepthwiseConv2d_CS::AttachBufferOffsets(OffsetBuffer &input,
                                                   OffsetBuffer &output,
                                                   OffsetBuffer &weights,
                                                   OffsetBuffer &inpzeropts,
                                                   OffsetBuffer &metadata) {
  // TODO: Check buffers compatibility
  m_in.set_buf(input);
  m_output.set_buf(output);
  m_weights.set_buf(weights);
  m_input_zp = inpzeropts;
  m_metadata = metadata;
  return MLI_STATUS_OK;
}

// TODO: Populate functions below 
//=======================================================
mli_status DepthwiseConv2d_CS::EncodeWeights(Tensor<Buffer, 3> &weights, 
                                             Buffer &encoded_weights,
                                             compression_mode_t mode){
  return MLI_STATUS_NOT_SUPPORTED;
};

mli_status DepthwiseConv2d_CS::EncodeInpZeroPts(Tensor<Buffer, 1> &inpzeropts, 
                                                Buffer &encoded_inpzeropts) {
  return MLI_STATUS_NOT_SUPPORTED;
};

unsigned DepthwiseConv2d_CS::GetEncodedWeightsSize() {return 0;};
unsigned DepthwiseConv2d_CS::GetEncodedInpZeroPtsSize() {return 0;};
unsigned DepthwiseConv2d_CS::GetWeightsBufferSize() {return 0;};

unsigned DepthwiseConv2d_CS::GetInputBufferSize() {
  return 0; // Reuse GetBufferSize in a paralle branch
}
unsigned DepthwiseConv2d_CS::GetOutputBufferSize() {
  return 0;//return output_buffer_size_;
}
unsigned DepthwiseConv2d_CS::GetDataBufferSize() {
  // contains pointer which size might differs depending on platform
  // Complement metadata struct itself with a helper method??
  return sizeof(DepthwiseConv2dMetadata); 
}

}  // namespace snps_arc::metaware::mli::ref