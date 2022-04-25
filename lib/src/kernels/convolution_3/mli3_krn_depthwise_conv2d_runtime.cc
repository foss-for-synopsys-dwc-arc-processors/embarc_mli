/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <cstring>

// #include "mli_ref_compiler_api.h"

#include "mli_ref_runtime_api.hpp"
//#include "mli_ref_private_types.hpp"
#include "mli_debug.h"

#include "mli_ref_krn_conv_private_types.hpp"

//Temporary
#include <iostream>

namespace snps_arc::metaware::mli::ref {


static constexpr mli_tensor flat_tensor_common(size_t tsr_size) {
    mli_tensor ret_tsr{0};
    ret_tsr.el_params.fx.frac_bits = 0;
    ret_tsr.rank = 1;
    ret_tsr.shape[0] = tsr_size;
    ret_tsr.mem_stride[0] = 1;
    return ret_tsr;
}

static constexpr mli_tensor flat_tensor(int16_t* data, size_t tsr_size) {
    assert(data != nullptr);
    mli_tensor ret_tsr = flat_tensor_common(tsr_size);
    ret_tsr.data.mem.pi16 = data;
    ret_tsr.data.capacity = tsr_size * sizeof(data[0]);
    ret_tsr.el_type = MLI_EL_FX_16;
    return ret_tsr;
}

static constexpr mli_tensor flat_tensor(int8_t* data, size_t tsr_size) {
    assert(data != nullptr);
    mli_tensor ret_tsr = flat_tensor_common(tsr_size);
    ret_tsr.data.capacity = tsr_size * sizeof(data[0]);
    ret_tsr.data.mem.pi8 = data;
    ret_tsr.el_type = MLI_EL_FX_8;
    return ret_tsr;
}

static constexpr mli_tensor flat_tensor(int32_t* data, size_t tsr_size) {
    assert(data != nullptr);
    mli_tensor ret_tsr = flat_tensor_common(tsr_size);
    ret_tsr.data.capacity = tsr_size * sizeof(data[0]);
    ret_tsr.data.mem.pi32 = data;
    ret_tsr.el_type = MLI_EL_SA_32;
    ret_tsr.el_params.sa.dim = -1;
    ret_tsr.el_params.sa.scale.capacity = 0;
    ret_tsr.el_params.sa.scale.mem.i16 = 1;
    ret_tsr.el_params.sa.type = MLI_EL_PARAM_SC16_ZP16;
    ret_tsr.el_params.sa.scale_frac_bits.capacity = 0;
    ret_tsr.el_params.sa.scale_frac_bits.mem.i8 = 0;
    ret_tsr.el_params.sa.zero_point.capacity = 0;
    ret_tsr.el_params.sa.zero_point.mem.i16 = 0;
    return ret_tsr;
}

//constexpr mli_tensor construct_tensor
DepthwiseConv2d::DepthwiseConv2d(PrivateData* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems) {
  MLI_ASSERT(kernel_private_data_buffer->kernel_id == kDWConv2dId);
  MLI_ASSERT(size == sizeof(DepthwiseConv2DPrivateData));
  MLI_ASSERT(kernel_private_data_buffer->size == sizeof(DepthwiseConv2DPrivateData)); //Remark: Ambigous

  auto private_data = static_cast<DepthwiseConv2DPrivateData*>(kernel_private_data_buffer);
  
  MLI_ASSERT(private_data->metadata_mem_id < num_mems);
  int8_t* address = reinterpret_cast<int8_t*>(membases[private_data->metadata_mem_id]);
  address += private_data->metadata_offset;
  m_metadata = new(address) DepthwiseConv2dMetadata{};

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
  address = reinterpret_cast<int8_t*>(membases[private_data->input_mem_id]);
  address += private_data->input_mem_offset;
  {
    mli_tensor tsr{0};
    tsr.data.mem.pi8 = address;
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
  address = reinterpret_cast<int8_t*>(membases[private_data->output_mem_id]);
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
  address = reinterpret_cast<int8_t*>(membases[private_data->weights_mem_id]);
  address += private_data->weights_mem_offset;
  {
    mli_tensor tsr{0};
    tsr.data.mem.pi8 = address;
    tsr.rank = 4;
    tsr.shape[0] = private_data->weights_h;
    tsr.shape[1] = private_data->weights_w;
    tsr.shape[2] = 1;
    tsr.shape[3] = private_data->input_output_c;
    tsr.mem_stride[0] = private_data->weights_h_stride;
    tsr.mem_stride[1] = private_data->weights_w_stride;
    tsr.mem_stride[2] = private_data->input_output_c; // derived from the fact tat memstride of innermost is 1
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
  address = reinterpret_cast<int8_t*>(membases[private_data->input_zp_mem_id]);
  address += private_data->input_zp_mem_offset;
  m_metadata->input_zero_point = reinterpret_cast<int16_t*>(address);
}

mli_status DepthwiseConv2d::Issue() {
  //TODO: Support Batch processing
  // We don't support per-channel zero point for feature maps at the moment.
  m_metadata->input.el_params.sa.zero_point.mem.i16 = *m_metadata->input_zero_point;
  return mli3_krn_depthwise_conv2d_hwcn_w8_i8_o32(&m_metadata->input,
                                                  &m_metadata->weights,
                                                  &m_metadata->cfg,
                                                  &m_metadata->output);

  // mli3_krn_depthwise_conv2d_hwcn_w8_i8_o32(&input, &weights, &cur_test->cfg, &out_acc); != MLI_STATUS_OK
  // if (io_elem_size_ == sizeof(int16_t)) {
  //   for (uint32_t i = 0; i < batch_number_; i++) {
  //     input_->data.mem.pi16 += input_batch_offset_ * i;
  //     output_->data.mem.pi16 += output_batch_offset_* i;
  //     mli3_krn_maxpool_hwc_io16(input_, cfg_, output_);
  //   }
  // } else if (io_elem_size_ == sizeof(int8_t)) {
  //   for (uint32_t i = 0; i < batch_number_; i++) {
  //     input_->data.mem.pi8 += input_batch_offset_ * i;
  //     output_->data.mem.pi8 += output_batch_offset_ * i;
  //     mli3_krn_maxpool_hwc_io8(input_, cfg_, output_);
  //   }
  // }
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

mli_status DepthwiseConv2d::Prefetch() {
  return MLI_STATUS_OK;
}

mli_status DepthwiseConv2d::Update() {
  return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref