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

Conv2d::Conv2d(void* kernel_private_data_buffer,
               size_t size,
               uint64_t membases[], int num_mems) {
  MLI_ASSERT(size == sizeof(Conv2DPrivateData));
  Conv2DPrivateData private_data;
  memcpy(&private_data, kernel_private_data_buffer, sizeof(Conv2DPrivateData));
  MLI_ASSERT(private_data.kernel_id == kConv2dId);
  MLI_ASSERT(private_data.size == sizeof(Conv2DPrivateData));

  MLI_ASSERT(private_data.layout == LAYOUT_HWC);

  // full size
  m_metadata.input = Tensor<InternalBuffer, 4>(private_data.input, membases, num_mems);
  m_metadata.output = Tensor<InternalBuffer, 4>(private_data.output, membases, num_mems);
  m_metadata.weights = Tensor<InternalBuffer, 5>(private_data.weights, membases, num_mems);
  m_metadata.inpzp_buffer = InternalBuffer(private_data.inpzp_buffer, membases, num_mems);
  m_metadata.wtszp_buffer = InternalBuffer(private_data.wtszp_buffer, membases, num_mems);
  m_metadata.inp_quant_axis = private_data.inp_quant_axis;
  m_metadata.wts_quant_axis = private_data.wts_quant_axis;
  m_metadata.cfg = private_data.config;

  // tiling
  if (!private_data.m_use_tiling) m_tile_metadata = m_metadata;
  else {
    m_use_tiling = private_data.m_use_tiling;
    for (int i = 0; i < 4; i++) {
      m_tile_total_input_size[i] = private_data.m_tile_total_input_size[i];
      m_tile_total_output_size[i] = private_data.m_tile_total_output_size[i];
      m_tile_total_weights_size[i] = private_data.m_tile_total_weights_size[i];
      m_tile_iteration_order[i] = private_data.m_tile_iteration_order[i];
      m_tile_first_size[i] = private_data.m_tile_first_size[i];
      m_tile_size[i] = private_data.m_tile_size[i];
      m_tile_input_first_inc[i] = private_data.m_tile_input_first_inc[i];
      m_tile_input_inc[i] = private_data.m_tile_input_inc[i];
      m_tile_output_first_inc[i] = private_data.m_tile_output_first_inc[i];
      m_tile_output_inc[i] = private_data.m_tile_output_inc[i];
      m_tile_weights_inc[i] = private_data.m_tile_weights_inc[i];
      m_tile_input_offsets[i] = 0;
      m_tile_output_offsets[i] = 0;
    }
    m_tile_metadata.input = Tensor<InternalBuffer, 4>(m_metadata.input, m_tile_first_size);
    m_tile_metadata.output = Tensor<InternalBuffer, 4>(m_metadata.output, m_tile_output_first_inc);
    uint32_t weight_tile_size[5]{};
    weight_tile_size[kKernelGroupDim] = 1;
    for (int i = 0; i < 4; i++) {
      weight_tile_size[i + 1] = m_tile_weights_inc[i];
    }
    m_tile_metadata.weights = Tensor<InternalBuffer, 5>(m_metadata.weights, weight_tile_size);
    m_tile_metadata.inpzp_buffer = m_metadata.inpzp_buffer;
    m_tile_metadata.wtszp_buffer = InternalBuffer();
    m_tile_metadata.wtszp_buffer.set_buffer(m_metadata.wtszp_buffer.get_ptr<int16_t>(),
                                            private_data.m_tile_weights_inc[3] * sizeof(int16_t) );
    m_tile_metadata.inp_quant_axis = private_data.inp_quant_axis;
    m_tile_metadata.wts_quant_axis = private_data.wts_quant_axis;
  }

  UpdateTilePaddings();
}

mli_status Conv2d::Issue() {
  // element size for input, output and weights in bytes
  uint32_t i_elem_size = m_tile_metadata.input.get_elem_size();
  uint32_t o_elem_size = m_tile_metadata.output.get_elem_size();
  uint32_t w_elem_size = m_tile_metadata.weights.get_elem_size();

  if (i_elem_size == sizeof(int8_t) &&
      w_elem_size == sizeof(int8_t) &&
      o_elem_size == sizeof(int32_t)) {

    QTensor<InternalBuffer, 4> qinput{
      m_tile_metadata.input, m_tile_metadata.inpzp_buffer, m_tile_metadata.inp_quant_axis};
    QTensor<InternalBuffer, 5> qweights{
      m_tile_metadata.weights, m_tile_metadata.wtszp_buffer, m_tile_metadata.wts_quant_axis};

    conv2d_prepare_and_run<int8_t, int8_t, int32_t, mli_8x8_accu_t, LAYOUT_HWC,
                           ::mli::CONV_GENERAL, /* io_rank */ 4, /* w_rank */ 5,
                           Conv2DConfig>(
        qinput, qweights, m_tile_metadata.output, m_tile_metadata.cfg);
  } else {
    // datatype is not supported yet
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}

mli_status Conv2d::Prefetch() { return MLI_STATUS_OK; }

mli_status Conv2d::Update() {
  if (!m_use_tiling) return MLI_STATUS_OK;

  // update state with i/o tile increments, that was used in Issue()
  for (int i = 0; i < 4; i++) {
    int axis = m_tile_iteration_order[i];
    bool first_tile = !m_tile_input_offsets[axis];
    m_tile_input_offsets[axis] += (first_tile ? m_tile_input_first_inc[axis] : m_tile_input_inc[axis]);
    m_tile_output_offsets[axis] += (first_tile ? m_tile_output_first_inc[axis] : m_tile_output_inc[axis]);

    if (m_tile_input_offsets[axis] >= m_tile_total_input_size[axis]) m_tile_input_offsets[axis] = 0;

    if (m_tile_output_offsets[axis] >= m_tile_total_output_size[axis]) {
      // end of this axis
      m_tile_input_offsets[axis] = 0;
      m_tile_output_offsets[axis] = 0;
    }
    else {
      // not end of this axis
      break;
    }
  }

  // set i/o/w/w_zp sizes for next call of Issue()
  uint32_t input_tile_size[4]{};
  uint32_t output_tile_size[4]{};
  for (int i = 0; i < 4; i++) {
    bool first_tile = !m_tile_input_offsets[i];
    input_tile_size[i] = MIN(first_tile ? m_tile_first_size[i] : m_tile_size[i],
                             m_tile_total_input_size[i] - m_tile_input_offsets[i]);
    output_tile_size[i] = MIN(first_tile ? m_tile_output_first_inc[i] : m_tile_output_inc[i],
                              m_tile_total_output_size[i] - m_tile_output_offsets[i]);
  }
  m_tile_metadata.input = Tensor<InternalBuffer, 4>(m_metadata.input, input_tile_size);
  m_tile_metadata.output = Tensor<InternalBuffer, 4>(m_metadata.output, output_tile_size);

  uint32_t weight_tile_size[5]{};
  weight_tile_size[kKernelGroupDim] = 1;
  for (int i = 0; i < 3; i++) {
    weight_tile_size[i + 1] = m_tile_weights_inc[i];
  }
  weight_tile_size[kKernelChannelOutDim] = MIN(m_tile_weights_inc[3],
                                               m_tile_total_weights_size[3] - m_tile_output_offsets[kTensorChannelDim]);

  m_tile_metadata.weights = Tensor<InternalBuffer, 5>(m_metadata.weights, weight_tile_size);
  m_tile_metadata.inpzp_buffer = m_metadata.inpzp_buffer;
  m_tile_metadata.wtszp_buffer = InternalBuffer(); ;
  m_tile_metadata.wtszp_buffer.set_buffer(m_metadata.wtszp_buffer.get_ptr<int16_t>(),
                                          weight_tile_size[kKernelChannelOutDim] * sizeof(int16_t));

  UpdateTilePaddings();
  return MLI_STATUS_OK;

}

void Conv2d::UpdateTilePaddings() {
  memcpy(&m_tile_metadata.cfg, &m_metadata.cfg, sizeof(m_metadata.cfg));
  if (!m_use_tiling) return;

  Conv2DConfig& cfg = m_tile_metadata.cfg;
  if (m_tile_input_offsets[kTensorHeightDim]) cfg.padding_begin[0] = 0;
  if (m_tile_input_offsets[kTensorHeightDim] + m_tile_size[kTensorHeightDim] < m_tile_total_input_size[kTensorHeightDim]) {
    cfg.padding_end[0] = 0;
  }
  if (m_tile_input_offsets[kTensorWidthDim]) cfg.padding_begin[1] = 0;
  if (m_tile_input_offsets[kTensorWidthDim] + m_tile_size[kTensorWidthDim] < m_tile_total_input_size[kTensorWidthDim]) {
    cfg.padding_end[1] = 0;
  }
}

void Conv2d::GetIOSizesAndOffsets(uint32_t input_size[4], uint32_t output_size[4], uint32_t weights_size[5],
                                  uint32_t input_offsets[4], uint32_t output_offsets[4], uint32_t weights_offsets[5]) const {
  for (int i = 0; i < 4; i++) {
    input_size[i] = m_tile_metadata.input.get_dim(i);
    output_size[i] = m_tile_metadata.output.get_dim(i);
    weights_size[i] = m_tile_metadata.weights.get_dim(i);
    input_offsets[i] = m_tile_input_offsets[i];
    output_offsets[i] = m_tile_output_offsets[i];
    weights_offsets[i] = 0;
  }
  weights_size[kKernelChannelOutDim] = m_tile_metadata.weights.get_dim(kKernelChannelOutDim);
  weights_offsets[kKernelChannelOutDim] = output_offsets[kTensorChannelDim];
}

}  // namespace snps_arc::metaware::mli::ref