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
#include "mli_krn_transpose_conv.h"
#include "mli_ref_runtime_api.hpp"

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
typedef vNx4accshort_t mli_8x8_accu_t;
#else
typedef mli_acc32_t mli_8x8_accu_t;
#endif


namespace snps_arc::metaware::mli::ref {

TransposeConv2D::TransposeConv2D(void* kernel_private_data_buffer, size_t size,
                                 uint64_t membases[], int num_mems) {
    MLI_ASSERT(size == sizeof(TransposeConv2DPrivateData));
    
    TransposeConv2DPrivateData private_data;
    memcpy(&private_data, kernel_private_data_buffer, sizeof(TransposeConv2DPrivateData));
    MLI_ASSERT(private_data.kernel_id == kTransConv2DId);
    MLI_ASSERT(private_data.size == sizeof(TransposeConv2DPrivateData));
    MLI_ASSERT(private_data.layout == LAYOUT_HWC);

    m_metadata.input = private_data.input;
    m_metadata.weights = private_data.weights;
    m_metadata.weights_zp = private_data.weights_zp;
    m_metadata.output = private_data.output;
    m_metadata.inpzp_buffer = InternalBuffer(private_data.inpzp_buffer, membases, num_mems);

    m_metadata.inp_quant_axis = private_data.inp_quant_axis;
    m_metadata.wts_quant_axis = private_data.wts_quant_axis;
    m_metadata.cfg = private_data.config;

    m_tile_input = Tensor<InternalBuffer, kTransposeConvIORank>(m_metadata.input.GetSubTensor(true), membases, num_mems);
    m_tile_weights = Tensor<InternalBuffer, kTransposeConvWRank>(m_metadata.weights.GetSubTensor(), membases, num_mems);
    m_tile_wzp = Tensor<InternalBuffer, kTransposeConvZPRank>(m_metadata.weights_zp.GetSubTensor(), membases, num_mems);
    m_tile_output = Tensor<InternalBuffer, kTransposeConvIORank>(m_metadata.output.GetSubTensor(true), membases, num_mems);

    UpdateTilePaddings();
}

mli_status TransposeConv2D::Issue() {
  // element size for input, output and weights in bytes
  uint32_t i_elem_size = m_metadata.input.get_buf().get_elem_size();
  uint32_t o_elem_size = m_metadata.output.get_buf().get_elem_size();
  uint32_t w_elem_size = m_metadata.weights.get_buf().get_elem_size();
  
  if (i_elem_size == sizeof(int8_t) && w_elem_size == sizeof(int8_t) &&
      o_elem_size == sizeof(int32_t)) {
      QTensor<InternalBuffer, kTransposeConvIORank> qinput{ m_tile_input,
                                                            m_metadata.inpzp_buffer,
                                                            m_metadata.inp_quant_axis};
      QTensor<InternalBuffer, kTransposeConvWRank> qweights{ m_tile_weights,
                                                             m_tile_wzp.get_buf(),
                                                             m_metadata.wts_quant_axis};

      transpose_conv2d_prepare_and_run<int8_t, int8_t, int32_t, mli_8x8_accu_t,
                                       LAYOUT_HWC, kTransposeConvIORank,
                                       kTransposeConvWRank>(qinput, qweights, m_tile_output, m_tile_cfg);

  } else {
      // datatype is not supported yet
      return MLI_STATUS_NOT_SUPPORTED;
  }
  
  return MLI_STATUS_OK;
}

mli_status TransposeConv2D::Prefetch() { return MLI_STATUS_OK; }

mli_status TransposeConv2D::Update() {

  m_metadata.input.Next();
  m_metadata.output.Next();
  m_metadata.weights.Next();
  m_metadata.weights_zp.Next();

  const auto input_tile_tensor = m_metadata.input.GetSubTensor(true);
  uint32_t input_tile_shape[kTransposeConvIORank];
  input_tile_tensor.get_dims(input_tile_shape);
  m_tile_input = Tensor<InternalBuffer, kTransposeConvIORank>(m_tile_input, input_tile_shape);

  const auto weights_tile_tensor = m_metadata.weights.GetSubTensor();
  uint32_t weights_tile_shape[kTransposeConvWRank];
  weights_tile_tensor.get_dims(weights_tile_shape);
  m_tile_weights = Tensor<InternalBuffer, kTransposeConvWRank>(m_tile_weights, weights_tile_shape);

  const auto wzp_tile_tensor = m_metadata.weights_zp.GetSubTensor();
  uint32_t wzp_tile_shape[kTransposeConvZPRank];
  wzp_tile_tensor.get_dims(wzp_tile_shape);
  m_tile_wzp = Tensor<InternalBuffer, kTransposeConvZPRank>(m_tile_wzp, wzp_tile_shape);
  // TODO: maybe some method to handle instead of this code (last tile can be smaller than others)
  if (wzp_tile_shape[0] != m_tile_wzp.get_buf().get_size()) {
    InternalBuffer buf = m_tile_wzp.get_buf();
    buf.set_buffer(buf.get_ptr<int8_t>(), wzp_tile_shape[0] * sizeof(int8_t));
    m_tile_wzp.set_buf(buf);
  }

  const auto output_tile_tensor = m_metadata.output.GetSubTensor(true);
  uint32_t output_tile_shape[kTransposeConvIORank];
  output_tile_tensor.get_dims(output_tile_shape);
  m_tile_output = Tensor<InternalBuffer, kTransposeConvIORank>(m_tile_output, output_tile_shape);

  UpdateTilePaddings();

  return MLI_STATUS_OK;

}

void TransposeConv2D::GetIOSizesAndOffsets(uint32_t input_size[kTransposeConvIORank],
                                           uint32_t output_size[kTransposeConvIORank],
                                           uint32_t weights_size[kTransposeConvWRank],
                                           int32_t input_offsets[kTransposeConvIORank],
                                           int32_t output_offsets[kTransposeConvIORank],
                                           int32_t weights_offsets[kTransposeConvWRank]) {
  m_metadata.input.get_pos(input_offsets);
  m_metadata.weights.get_pos(weights_offsets);
  m_metadata.output.get_pos(output_offsets);

  m_tile_input.get_dims(input_size);
  m_tile_weights.get_dims(weights_size);
  m_tile_output.get_dims(output_size);
}

void TransposeConv2D::UpdateTilePaddings() {
  memcpy(&m_tile_cfg, &m_metadata.cfg, sizeof(m_metadata.cfg));

  int32_t tile_output_offsets[kTransposeConvIORank];
  const auto& output = m_metadata.output;
  output.get_pos(tile_output_offsets);
  const auto& output_it_cfg = output.get_config();

  int32_t tile_idx[kConvIORank];
  output.get_tile_idx(tile_idx);

  // top padding
  if (!output.is_first_tile(kGroupTensorHeightDim)) {
    //TODO: if first_inc != 0 get_first_inc + get_inc * (idx - 1)
    uint32_t pad_used = output_it_cfg.get_inc(kGroupTensorHeightDim) * tile_idx[kGroupTensorHeightDim];
    if (pad_used < m_tile_cfg.padding_begin[0]) {
      m_tile_cfg.padding_begin[0] -= pad_used;
    }
    else {
      m_tile_cfg.padding_begin[0] = 0;
    }
  }

  // left padding
  if (!output.is_first_tile(kGroupTensorWidthDim)) {
    // TODO: if first_inc != 0 get_first_inc + get_inc * (idx - 1)
    uint32_t pad_used = output_it_cfg.get_inc(kGroupTensorWidthDim) * tile_idx[kGroupTensorWidthDim];
    if (pad_used < m_tile_cfg.padding_begin[1]) {
      m_tile_cfg.padding_begin[1] -= pad_used;
    }
    else {
      m_tile_cfg.padding_begin[1] = 0;
    }
  }

  // bottom padding
  uint32_t used_size_y = (int32_t)(output.get_dim(kGroupTensorHeightDim) + m_tile_cfg.padding_begin[0]);
  int32_t pad_bot = tile_output_offsets[kGroupTensorHeightDim] + (int32_t)output_it_cfg.get_size(kGroupTensorHeightDim) - used_size_y;
  if (pad_bot > 0) {
    m_tile_cfg.padding_end[0] = MIN((uint32_t)pad_bot, m_metadata.cfg.padding_end[0]);
  }
  else {
    m_tile_cfg.padding_end[0] = 0;
  }

  // right padding
  uint32_t used_size_x = (int32_t)(output.get_dim(kGroupTensorWidthDim) + m_tile_cfg.padding_begin[1]);
  int32_t pad_right = tile_output_offsets[kGroupTensorWidthDim] + (int32_t)output_it_cfg.get_size(kGroupTensorWidthDim) - used_size_x;
  if (pad_right > 0) {
    m_tile_cfg.padding_end[1] = MIN((uint32_t)pad_right, m_metadata.cfg.padding_end[1]);
  }
  else {
    m_tile_cfg.padding_end[1] = 0;
  }
}


}  // namespace snps_arc::metaware::mli::ref