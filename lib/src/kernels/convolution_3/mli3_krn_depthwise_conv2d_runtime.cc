/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

//#include <cstring>

#include <new>

#include "mli_ref_runtime_api.hpp"
#include "mli_debug.h"

#include "mli_ref_krn_conv_private_types.hpp"


namespace snps_arc::metaware::mli::ref {

DepthwiseConv2d::DepthwiseConv2d(PrivateData* kernel_private_data_buffer,
                                 size_t size,
                                 uint64_t membases[], int num_mems) {
  MLI_ASSERT(kernel_private_data_buffer->kernel_id == kDWConv2dId);
  MLI_ASSERT(size == sizeof(DepthwiseConv2DPrivateData));
  MLI_ASSERT(kernel_private_data_buffer->size == sizeof(DepthwiseConv2DPrivateData)); //Remark: Ambigous

  auto private_data = static_cast<DepthwiseConv2DPrivateData*>(kernel_private_data_buffer);
  
  MLI_ASSERT(private_data->metadata_mem_id < num_mems);
  auto address = membases[private_data->metadata_mem_id]; 
  address += private_data->metadata_offset;
  m_metadata = new(reinterpret_cast<void*>(address)) DepthwiseConv2dMetadata{};

  // Reconstruct configuration
  auto krn_cfg = &m_metadata->cfg;
  krn_cfg->dilation_height = private_data->dilation_height;
  krn_cfg->dilation_width = private_data->dilation_width;
  krn_cfg->padding_bottom = private_data->padding_bottom;
  krn_cfg->padding_top = private_data->padding_top;
  krn_cfg->padding_left = private_data->padding_left;
  krn_cfg->padding_right = private_data->padding_right;
  krn_cfg->stride_height = private_data->stride_height;
  krn_cfg->stride_width = private_data->stride_width;
  krn_cfg->relu.type = MLI_RELU_NONE;

  // Reconstruct Input Tensor
  MLI_ASSERT(private_data->input_mem_id < num_mems);
  address = membases[private_data->input_mem_id];
  address += private_data->input_mem_offset;
  {
    // TODO: Move partly or all into a helper function and use for each tensor
    mli_tensor tsr{0};
    tsr.data.mem.pi8 = reinterpret_cast<int8_t*>(address);
    tsr.rank = 3;
    tsr.shape[0] = private_data->input_h;
    tsr.shape[1] = private_data->input_w;
    tsr.shape[2] = private_data->input_output_c;
    tsr.mem_stride[0] = private_data->input_h_stride;
    tsr.mem_stride[1] = private_data->input_w_stride;
    tsr.mem_stride[2] = 1;
    tsr.data.capacity = 0;
    for (int i = 0; i < tsr.rank; ++i) {
      tsr.data.capacity += (tsr.shape[i] - 1) * tsr.mem_stride[i];
    }
    tsr.data.capacity += sizeof(int8_t);
    tsr.el_type = MLI_EL_SA_8;
    tsr.el_params.sa.dim = -1;
    tsr.el_params.sa.scale.capacity = 0;
    tsr.el_params.sa.scale.mem.i16 = 1;
    tsr.el_params.sa.type = MLI_EL_PARAM_SC16_ZP16;
    tsr.el_params.sa.scale_frac_bits.capacity = 0;
    tsr.el_params.sa.scale_frac_bits.mem.i8 = 0;
    tsr.el_params.sa.zero_point.capacity = 0;
    tsr.el_params.sa.zero_point.mem.i16 = 0;
    m_metadata->input = tsr;
  }

  // Reconstruct Output Tensor
  MLI_ASSERT(private_data->output_mem_id < num_mems);
  address = membases[private_data->output_mem_id];
  address += private_data->output_mem_offset;
  {
    mli_tensor tsr{0};
    tsr.data.mem.pi32 = reinterpret_cast<int32_t *>(address);
    tsr.rank = 3;
    tsr.shape[0] = private_data->output_h;
    tsr.shape[1] = private_data->output_w;
    tsr.shape[2] = private_data->input_output_c;
    tsr.mem_stride[0] = private_data->output_h_stride;
    tsr.mem_stride[1] = private_data->output_w_stride;
    tsr.mem_stride[2] = 1;
    tsr.data.capacity = 0;
    for (int i = 0; i < tsr.rank; ++i) {
      tsr.data.capacity += (tsr.shape[i] - 1) * tsr.mem_stride[i];
    }
    tsr.data.capacity += sizeof(int32_t);
    tsr.el_type = MLI_EL_SA_32;
    tsr.el_params.sa.dim = -1;
    tsr.el_params.sa.scale.capacity = 0;
    tsr.el_params.sa.scale.mem.i16 = 1;
    tsr.el_params.sa.type = MLI_EL_PARAM_SC16_ZP16;
    tsr.el_params.sa.scale_frac_bits.capacity = 0;
    tsr.el_params.sa.scale_frac_bits.mem.i8 = 0;
    tsr.el_params.sa.zero_point.capacity = 0;
    tsr.el_params.sa.zero_point.mem.i16 = 0;
    m_metadata->output = tsr;
  }

  // Reconstruct Weights Tensor
  MLI_ASSERT(private_data->weights_mem_id < num_mems);
  address = membases[private_data->weights_mem_id];
  address += private_data->weights_mem_offset;
  {
    mli_tensor tsr{0};
    tsr.data.mem.pi8 = reinterpret_cast<int8_t*>(address);
    tsr.rank = 4;
    tsr.shape[0] = private_data->weights_h;
    tsr.shape[1] = private_data->weights_w;
    tsr.shape[2] = 1;
    tsr.shape[3] = private_data->input_output_c;
    tsr.mem_stride[0] = private_data->weights_h_stride;
    tsr.mem_stride[1] = private_data->weights_w_stride;
    tsr.mem_stride[2] = private_data->input_output_c; // derived from the fact that memstride of innermost is 1
    tsr.mem_stride[3] = 1;
    tsr.data.capacity = 0;
    for (int i = 0; i < tsr.rank; ++i) {
      tsr.data.capacity += (tsr.shape[i] - 1) * tsr.mem_stride[i];
    }
    tsr.data.capacity += sizeof(int8_t);
    tsr.el_type = MLI_EL_SA_8;
    tsr.el_params.sa.dim = -1;
    tsr.el_params.sa.scale.capacity = 0;
    tsr.el_params.sa.scale.mem.i16 = 1;
    tsr.el_params.sa.type = MLI_EL_PARAM_SC16_ZP16;
    tsr.el_params.sa.scale_frac_bits.capacity = 0;
    tsr.el_params.sa.scale_frac_bits.mem.i8 = 0;
    tsr.el_params.sa.zero_point.capacity = 0;
    tsr.el_params.sa.zero_point.mem.i16 = 0;
    m_metadata->weights = tsr;
  }

  MLI_ASSERT(private_data->input_zp_mem_id < num_mems);
  address = membases[private_data->input_zp_mem_id];
  address += private_data->input_zp_mem_offset;
  m_metadata->input_zero_point = reinterpret_cast<int16_t*>(address);
}

mli_status DepthwiseConv2d::Issue() {
  //TODO: Support Batch processing
  // We don't support per-channel zero point for feature maps at the moment.
  m_metadata->input.el_params.sa.zero_point.mem.i16 = 
      *m_metadata->input_zero_point;
  return mli3_krn_depthwise_conv2d_hwcn_w8_i8_o32(&m_metadata->input,
                                                  &m_metadata->weights,
                                                  &m_metadata->cfg,
                                                  &m_metadata->output);
  return MLI_STATUS_OK;
}

mli_status DepthwiseConv2d::Init(PrivateData* kernel_private_data_buffer, 
                                 int private_data_size, uint64_t membases[],
                                 int num_mems) {
  // Same functionality as constructor, but constructor shall not call virtual functions.
  // Hence construction functionality should be moved to the third non-virtual methods
  // and used together in constructor and this init function. 
  return MLI_STATUS_OK;
}

mli_status DepthwiseConv2d::Prefetch() { return MLI_STATUS_OK; }

mli_status DepthwiseConv2d::Update() { return MLI_STATUS_OK; }

}  // namespace snps_arc::metaware::mli::ref