/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <cstring>
#include <new>

#include "mli_debug.h"
#include "mli_krn_convolution.h"
#include "mli_ref_runtime_api.hpp"

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
typedef vNx4accshort_t mli_8x8_accu_t;
#else
typedef mli_acc32_t mli_8x8_accu_t;
#endif

namespace snps_arc::metaware::mli::ref {

DepthwiseConv2d::DepthwiseConv2d(void* kernel_private_data_buffer,
                                 size_t size,
                                 uint64_t membases[], int num_mems) {
  MLI_ASSERT(size == sizeof(DepthwiseConv2DPrivateData));
  DepthwiseConv2DPrivateData private_data;
  memcpy(&private_data, kernel_private_data_buffer, sizeof(DepthwiseConv2DPrivateData));
  MLI_ASSERT(private_data.kernel_id == kDWConv2dId);
  MLI_ASSERT(private_data.size == sizeof(DepthwiseConv2DPrivateData));

  MLI_ASSERT(private_data.layout == LAYOUT_HWC);

  m_metadata.input = private_data.input;
  m_metadata.weights = private_data.weights;
  m_metadata.weights_zp = private_data.weights_zp;
  m_metadata.output = private_data.output;

  m_metadata.inpzp_buffer = InternalBuffer(private_data.inpzp_buffer, membases, num_mems);

  m_metadata.inp_quant_axis = private_data.inp_quant_axis;
  m_metadata.wts_quant_axis = private_data.wts_quant_axis;
  m_metadata.config = private_data.config;

  auto input_tile_tensor = m_metadata.input.GetSubTensor();
  m_tile_batch_size = input_tile_tensor.get_dim(kGroupTensorBatchDim);

  // setup m_tile_input to execute batch by batch m_tile_batch_size times
  input_tile_tensor.set_dim(kGroupTensorBatchDim, 1);
  m_tile_input = Tensor<InternalBuffer, kDepthwiseIORank>(input_tile_tensor, membases, num_mems);
  InternalBuffer inp_buf = m_tile_input.get_buf();
  InternalBuffer single_batch_inp_buf(inp_buf.get_ptr<int8_t>(), inp_buf.get_size() / m_tile_batch_size);
  m_tile_input.set_buf(single_batch_inp_buf);

  m_tile_weights = Tensor<InternalBuffer, kDepthwiseWRank>(m_metadata.weights.GetSubTensor(), membases, num_mems);
  m_tile_wzp = Tensor<InternalBuffer, kDepthwiseZPRank>(m_metadata.weights_zp.GetSubTensor(), membases, num_mems);

  // setup m_tile_output to execute batch by batch m_tile_batch_size times
  auto output_tile_tensor = m_metadata.output.GetSubTensor();
  output_tile_tensor.set_dim(kGroupTensorBatchDim, 1);
  m_tile_output = Tensor<InternalBuffer, kDepthwiseIORank>(output_tile_tensor, membases, num_mems);
  InternalBuffer out_buf = m_tile_output.get_buf();
  InternalBuffer single_batch_out_buf(out_buf.get_ptr<int32_t>(), out_buf.get_size() / m_tile_batch_size);
  m_tile_output.set_buf(single_batch_out_buf);
  
  UpdateTilePaddings();
}

mli_status DepthwiseConv2d::Issue() {
  // element size for input, output and weights in bytes
  uint32_t i_elem_size = m_metadata.input.get_buf().get_elem_size();
  uint32_t o_elem_size = m_metadata.output.get_buf().get_elem_size();
  uint32_t w_elem_size = m_metadata.weights.get_buf().get_elem_size();

  if (i_elem_size == sizeof(int8_t) &&
      w_elem_size == sizeof(int8_t) &&
      o_elem_size == sizeof(int32_t)) {

    InternalBuffer inp_buf = m_tile_input.get_buf();
    InternalBuffer out_buf = m_tile_output.get_buf();
    InternalBuffer curr_inp_buf(inp_buf);
    InternalBuffer curr_out_buf(out_buf);
    m_tile_input.set_buf(curr_inp_buf);
    m_tile_output.set_buf(curr_out_buf);

    QTensor<InternalBuffer, kDepthwiseIORank> qinput{
      m_tile_input, m_metadata.inpzp_buffer, m_metadata.inp_quant_axis};
    QTensor<InternalBuffer, kDepthwiseWRank> qweights{
      m_tile_weights, m_tile_wzp.get_buf(), m_metadata.wts_quant_axis};


    for (uint32_t i = 0; i < m_tile_batch_size; i++) {
      conv2d_prepare_and_run<int8_t, int8_t, int32_t, mli_8x8_accu_t, LAYOUT_HWC,
                            ::mli::CONV_DEPTHWISE, kDepthwiseIORank,
                            kDepthwiseWRank, DwConv2DConfig>(
                              qinput, qweights, m_tile_output, m_tile_cfg);
        
      curr_inp_buf.inc(m_metadata.input.get_mem_stride(kGroupTensorBatchDim));
      curr_out_buf.inc(m_metadata.output.get_mem_stride(kGroupTensorBatchDim));
    }
    m_tile_input.set_buf(inp_buf);
    m_tile_output.set_buf(out_buf);
  } else {
    // datatype is not supported yet
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}

mli_status DepthwiseConv2d::Prefetch() { return MLI_STATUS_OK; }

mli_status DepthwiseConv2d::Update() {   
  m_metadata.input.Next();
  m_metadata.output.Next();
  m_metadata.weights.Next();
  m_metadata.weights_zp.Next();

  const auto input_tile_tensor = m_metadata.input.GetSubTensor();
  uint32_t input_tile_shape[kDepthwiseIORank];
  input_tile_tensor.get_dims(input_tile_shape);
  input_tile_shape[kGroupTensorBatchDim] = 1;
  m_tile_input = Tensor<InternalBuffer, kDepthwiseIORank>(m_tile_input, input_tile_shape);

  const auto weights_tile_tensor = m_metadata.weights.GetSubTensor();
  uint32_t weights_tile_shape[kDepthwiseWRank];
  weights_tile_tensor.get_dims(weights_tile_shape);
  m_tile_weights = Tensor<InternalBuffer, kDepthwiseWRank>(m_tile_weights, weights_tile_shape);

  const auto wzp_tile_tensor = m_metadata.weights_zp.GetSubTensor();
  uint32_t wzp_tile_shape[kDepthwiseZPRank];
  wzp_tile_tensor.get_dims(wzp_tile_shape);
  m_tile_wzp = Tensor<InternalBuffer, kDepthwiseZPRank>(m_tile_wzp, wzp_tile_shape);
  // TODO: maybe some method to handle instead of this code (last tile can be smaller than others)
  if (wzp_tile_shape[0] != m_tile_wzp.get_buf().get_size()) {
    InternalBuffer buf = m_tile_wzp.get_buf();
    buf.set_buffer(buf.get_ptr<int8_t>(), wzp_tile_shape[0]);
    m_tile_wzp.set_buf(buf);
  }

  const auto output_tile_tensor = m_metadata.output.GetSubTensor();
  uint32_t output_tile_shape[kDepthwiseIORank];
  output_tile_tensor.get_dims(output_tile_shape);
  output_tile_shape[kGroupTensorBatchDim] = 1;
  m_tile_output = Tensor<InternalBuffer, kDepthwiseIORank>(m_tile_output, output_tile_shape);

  UpdateTilePaddings();

  return MLI_STATUS_OK;
}

void DepthwiseConv2d::UpdateTilePaddings() {
  memcpy(&m_tile_cfg, &m_metadata.config, sizeof(m_metadata.config));

  int32_t tile_input_offsets[kDepthwiseIORank];
  const auto& input = m_metadata.input;
  input.get_pos(tile_input_offsets);
  const auto& input_it_cfg = input.get_config();

  // top padding
  if (!input.is_first_tile(kGroupTensorHeightDim)) m_tile_cfg.padding_begin[0] = 0;

  // left padding
  if (!input.is_first_tile(kGroupTensorWidthDim)) m_tile_cfg.padding_begin[1] = 0;

  // bottom padding
  int32_t pad_bot = tile_input_offsets[kGroupTensorHeightDim] + (int32_t)input_it_cfg.get_size(kGroupTensorHeightDim) - (int32_t)input.get_dim(kGroupTensorHeightDim);
  if (pad_bot > 0) {
    m_tile_cfg.padding_end[0] = MIN((uint32_t)pad_bot, m_metadata.config.padding_end[0]);
  }
  else {
    m_tile_cfg.padding_end[0] = 0;
  }

  // right padding
  int32_t pad_right = tile_input_offsets[kGroupTensorWidthDim] + (int32_t)input_it_cfg.get_size(kGroupTensorWidthDim) - (int32_t)input.get_dim(kGroupTensorWidthDim);
  if (pad_right > 0) {
    m_tile_cfg.padding_end[1] = MIN((uint32_t)pad_right, m_metadata.config.padding_end[1]);
  }
  else {
    m_tile_cfg.padding_end[1] = 0;
  }
}

void DepthwiseConv2d::GetIOSizesAndOffsets(uint32_t input_size[kDepthwiseIORank], uint32_t output_size[kDepthwiseIORank],
                                           uint32_t weights_size[kDepthwiseWRank],
                                           int32_t input_offsets[kDepthwiseIORank], int32_t output_offsets[kDepthwiseIORank],
                                           int32_t weights_offsets[kDepthwiseWRank]) {

  m_metadata.input.get_pos(input_offsets);
  weights_offsets[kKernelDWHeightDim] = m_metadata.weights.GetPos(kKernelHeightDim);
  weights_offsets[kKernelDWWidthDim] = m_metadata.weights.GetPos(kKernelWidthDim);
  weights_offsets[kKernelDWChannelInDim] = m_metadata.weights.GetPos(kKernelChannelOutDim);
  m_metadata.output.get_pos(output_offsets);

  m_tile_input.get_dims(input_size);
  m_tile_weights.get_dims(weights_size);
  m_tile_output.get_dims(output_size);
}

}  // namespace snps_arc::metaware::mli::ref