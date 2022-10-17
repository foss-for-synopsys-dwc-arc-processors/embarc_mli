/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_krn_matmul_ref.hpp"

using mli::krn::ref::MatMul_prepare_and_run;
namespace snps_arc::metaware::mli::ref {

MatMul::MatMul(void* kernel_private_data_buffer,
               size_t size,
               uint64_t membases[], int num_mems) {
  MLI_ASSERT(size == sizeof(MatMulPrivateData));
  MatMulPrivateData private_data;
  memcpy(&private_data, kernel_private_data_buffer, sizeof(MatMulPrivateData));
  MLI_ASSERT(private_data.kernel_id == kMatMulId);
  MLI_ASSERT(private_data.size == sizeof(MatMulPrivateData));

  m_i_elem_size = private_data.m_in_left.get_elem_size();
  m_o_elem_size = private_data.m_output.get_elem_size();

  MLI_ASSERT(sizeof(int8_t) == m_i_elem_size);
  MLI_ASSERT(sizeof(int32_t) == m_o_elem_size);

  // left and right input have the same type
  MLI_ASSERT(private_data.m_in_right.get_elem_size() == m_i_elem_size);

  m_input_left = private_data.m_in_left;
  m_input_right = private_data.m_in_right;
  m_output = private_data.m_output;

  m_encoded_params = InternalBuffer(private_data.encoded_params, membases, num_mems);

  m_tile_input_left = Tensor<InternalBuffer, kMatMulRank>(m_input_left.GetSubTensor(), membases, num_mems);
  m_tile_input_right = Tensor<InternalBuffer, kMatMulRank>(m_input_right.GetSubTensor(), membases, num_mems);
  m_tile_output = Tensor<InternalBuffer, kMatMulRank>(m_output.GetSubTensor(), membases, num_mems);


}

mli_status MatMul::Issue() {

  if (m_i_elem_size == sizeof(int8_t) &&
      m_o_elem_size == sizeof(int32_t)) {

    MatMul_prepare_and_run<int8_t, int8_t, int32_t, kMatMulRank>
                          (m_tile_input_left, m_tile_input_right, m_tile_output, m_encoded_params);
  } else {
    // not supported yet
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}

mli_status MatMul::Prefetch() { return MLI_STATUS_OK; }

mli_status MatMul::Update() {

  m_input_left.Next();
  m_input_right.Next();
  m_output.Next();
  
  const auto input_left_tile_tensor = m_input_left.GetSubTensor();
  uint32_t input_left_tile_shape[kMatMulRank];
  input_left_tile_tensor.get_dims(input_left_tile_shape);
  m_tile_input_left = Tensor<InternalBuffer, kMatMulRank>(m_tile_input_left, input_left_tile_shape);

  const auto input_right_tile_tensor = m_input_right.GetSubTensor();
  uint32_t input_right_tile_shape[kMatMulRank];
  input_right_tile_tensor.get_dims(input_right_tile_shape);
  m_tile_input_right = Tensor<InternalBuffer, kMatMulRank>(m_tile_input_right, input_right_tile_shape);
  
  const auto output_tile_tensor = m_output.GetSubTensor();
  uint32_t output_tile_shape[kMatMulRank];
  output_tile_tensor.get_dims(output_tile_shape);
  m_tile_output = Tensor<InternalBuffer, kMatMulRank>(m_tile_output, output_tile_shape);
    

  return MLI_STATUS_OK;

}

void MatMul::GetIOSizesAndOffsets(uint32_t input_left_size[kMatMulRank], uint32_t input_right_size[kMatMulRank],
                                  uint32_t output_size[kMatMulRank], 
                                  int32_t input_left_offsets[kMatMulRank], int32_t input_right_offsets[kMatMulRank],
                                  int32_t output_offsets[kMatMulRank]) const{
  
  m_input_left.get_pos(input_left_offsets);
  m_input_left.get_pos(input_right_offsets);
  m_output.get_pos(output_offsets);
  
  m_tile_input_left.get_dims(input_left_size);
  m_tile_input_right.get_dims(input_right_size);
  m_tile_output.get_dims(output_size);
}
}  // namespace snps_arc::metaware::mli::ref
