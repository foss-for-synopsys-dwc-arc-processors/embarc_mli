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
#include "mli_ref_private_types.hpp"
#include "mli_ref_runtime_api.hpp"
#include "mli_compiler_api.hpp"
#include "mli_krn_pool_hwc.h"
#include "mli_prv_tensor.h"
#include "mli_iterator.hpp"

namespace snps_arc::metaware::mli::ref {

namespace mli_krn = ::mli::krn;

SumPool2D::SumPool2D(void* kernel_private_data_buffer, size_t size,
                     uint64_t membases[], int num_mems) {
  MLI_ASSERT(size == sizeof(Pool2DPrivateData));
  Pool2DPrivateData private_data(kSumPool2DId);
  memcpy(&private_data, kernel_private_data_buffer, sizeof(Pool2DPrivateData));
  MLI_ASSERT(private_data.kernel_id == kSumPool2DId);
  MLI_ASSERT(private_data.size == sizeof(Pool2DPrivateData));

  m_i_elem_size = private_data.input.get_buf().get_elem_size();
  m_o_elem_size = private_data.output.get_buf().get_elem_size();

  m_input_batch_offset = private_data.input.get_mem_stride(kTensorBatchDim);
  m_output_batch_offset = private_data.output.get_mem_stride(kTensorBatchDim);

  // SumPool2D configuration construction
  m_cfg.kernel_width = private_data.config.kernel_size[KRNL_W_DIM_HWCN];
  m_cfg.kernel_height = private_data.config.kernel_size[KRNL_H_DIM_HWCN];
  m_cfg.stride_width = private_data.config.stride[KRNL_W_DIM_HWCN];
  m_cfg.stride_height = private_data.config.stride[KRNL_H_DIM_HWCN];
  m_cfg.padding_left = private_data.config.padding_begin[KRNL_W_DIM_HWCN];
  m_cfg.padding_right = private_data.config.padding_end[KRNL_W_DIM_HWCN];
  m_cfg.padding_top = private_data.config.padding_begin[KRNL_H_DIM_HWCN];
  m_cfg.padding_bottom = private_data.config.padding_end[KRNL_H_DIM_HWCN];

  m_input = private_data.input;
  m_output = private_data.output;
  const auto input_tile_tensor = m_input.GetSubTensor();
  const auto output_tile_tensor = m_output.GetSubTensor();

  //input
  m_tile_input.rank = 3;
  InternalBuffer input_internal(private_data.input.get_buf(), membases, num_mems);
  if (m_i_elem_size == sizeof(int8_t)) {
    m_tile_input.el_type = MLI_EL_SA_8;
    mli_prv_tensor_set_data_ptr(&m_tile_input, input_internal.get_ptr<int8_t>());
  } else {
    MLI_ASSERT(0);
  }

  int32_t input_strides[kPoolRank];
  private_data.input.get_mem_strides(input_strides);
  for (int i = 0; i < 3; i++) {
    m_tile_input.mem_stride[i] = input_strides[i + 1];         // BHWC -> HWC
    m_tile_input.shape[i] = input_tile_tensor.get_dim(i + 1);  // BHWC -> HWC
  }
  m_tile_input.mem_stride[3] = 0;
  m_tile_input.shape[3] = 0;
  
  //output
  m_tile_output.rank = m_tile_input.rank;
  InternalBuffer output_internal(private_data.output.get_buf(), membases, num_mems);
  if (m_o_elem_size == sizeof(int32_t)) {
    m_tile_output.el_type = MLI_EL_SA_32;
    mli_prv_tensor_set_data_ptr(&m_tile_output, output_internal.get_ptr<int32_t>());
  }  else {
    MLI_ASSERT(0);
  }

  int32_t output_strides[kPoolRank];
  private_data.output.get_mem_strides(output_strides);
  for (int i = 0; i < 3; i++) {
    m_tile_output.mem_stride[i] = output_strides[i + 1];         // BHWC -> HWC
    m_tile_output.shape[i] = output_tile_tensor.get_dim(i + 1);  // BHWC -> HWC
  }
  m_tile_output.mem_stride[3] = 0;
  m_tile_output.shape[3] = 0;
  m_tile_batch_size = input_tile_tensor.get_dim(kTensorBatchDim);

  UpdateTilePaddings();
}

mli_status SumPool2D::Issue() {
  if (m_i_elem_size == sizeof(int8_t) && m_o_elem_size == sizeof(int32_t)) {
    int8_t* in_ptr = m_tile_input.data.mem.pi8;
    int32_t* out_ptr = m_tile_output.data.mem.pi32;
    for (uint32_t i = 0; i < m_tile_batch_size; i++) {
      mli_krn::mli_krn_pool_hwc
        <mli_krn::SUMPOOL, int8_t, int32_t, POOL_NO_FIXED_KRN_SIZE>(
          &m_tile_input, &m_tile_cfg, &m_tile_output);
      m_tile_input.data.mem.pi8 += m_input_batch_offset;
      m_tile_output.data.mem.pi32 += m_output_batch_offset;
    }
    mli_prv_tensor_set_data_ptr(&m_tile_input, in_ptr);
    mli_prv_tensor_set_data_ptr(&m_tile_output, out_ptr);
  } else {
    return MLI_STATUS_NOT_SUPPORTED;
  }
  return MLI_STATUS_OK;
}

mli_status SumPool2D::Prefetch() { return MLI_STATUS_OK; }

mli_status SumPool2D::Update() { 
  
  m_input.Next();
  m_output.Next();

  const auto input_tile_tensor = m_input.GetSubTensor();
  const auto output_tile_tensor = m_output.GetSubTensor();
  for (int i = 0; i < 3; i++) {
    m_tile_input.shape[i] = input_tile_tensor.get_dim(i + 1);    // BHWC -> HWC
    m_tile_output.shape[i] = output_tile_tensor.get_dim(i + 1);  // BHWC -> HWC
  }
  m_tile_batch_size = input_tile_tensor.get_dim(kTensorBatchDim);
  UpdateTilePaddings();
  return MLI_STATUS_OK;   
}

void SumPool2D::UpdateTilePaddings() {
  memcpy(&m_tile_cfg, &m_cfg, sizeof(mli_pool_cfg));

  int32_t tile_input_offsets[kPoolRank];
  m_input.get_pos(tile_input_offsets);

  const auto& input_it_cfg = m_input.get_config();
  if (tile_input_offsets[kTensorHeightDim]) m_tile_cfg.padding_top = 0;
  if (tile_input_offsets[kTensorHeightDim] + (int32_t) input_it_cfg.get_size(kTensorHeightDim) < (int32_t)m_input.get_dim(kTensorHeightDim)) {
    m_tile_cfg.padding_bottom = 0;
  }
  if (tile_input_offsets[kTensorWidthDim]) m_tile_cfg.padding_left = 0;
  if (tile_input_offsets[kTensorWidthDim] + (int32_t) input_it_cfg.get_size(kTensorWidthDim) < (int32_t)m_input.get_dim(kTensorWidthDim)) {
    m_tile_cfg.padding_right = 0;
  }
}

void SumPool2D::GetIOSizesAndOffsets(uint32_t input_size[kPoolRank], uint32_t output_size[kPoolRank],
                                     int32_t input_offsets[kPoolRank], int32_t output_offsets[kPoolRank]){
  m_input.get_pos(input_offsets);
  m_output.get_pos(output_offsets);

  const auto input_tile_tensor = m_input.GetSubTensor();
  input_tile_tensor.get_dims(input_size);

  const auto output_tile_tensor = m_output.GetSubTensor();
  output_tile_tensor.get_dims(output_size);
}

}  // namespace snps_arc::metaware::mli::ref