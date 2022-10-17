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
#include "mli_service_functions.hpp"
namespace snps_arc::metaware::mli::ref {

MatMul_CS::MatMul_CS(const lib_mli::PlatformDescription &pd,
                     const TensorIterator<NoBuffer, kMatMulRank, kMatMulIterRank> &in_left,
                     const TensorIterator<NoBuffer, kMatMulRank, kMatMulIterRank> &in_right,
                     const TensorIterator<NoBuffer, kMatMulRank, kMatMulIterRank> &output)
  : m_in_left(in_left),
    m_in_right(in_right),
    m_output(output),
    m_pd(pd) {
      m_encoded_params_buffer_size = sizeof(int8_t) * kMatMulRank;
}

unsigned MatMul_CS::GetKernelPrivateDataSize() const {
  return sizeof(MatMulPrivateData);
}

unsigned MatMul_CS::GetRuntimeObjectSize() const {
  return sizeof(MatMul);
}

mli_status MatMul_CS::GetKernelPrivateData(void* kernel_private_data_buffer) {

  MLI_ASSERT(kernel_private_data_buffer != nullptr);

  MatMulPrivateData prv_data;

  prv_data.m_in_left = m_in_left;
  prv_data.m_in_right = m_in_right;
  prv_data.m_output = m_output;
  prv_data.encoded_params = m_encoded_params;

  std::memcpy(kernel_private_data_buffer, (void *)&prv_data, sizeof(prv_data));

  return MLI_STATUS_OK;
}

mli_status MatMul_CS::AttachBufferOffsets(const OffsetBuffer &input_left,
                                          const OffsetBuffer &input_right,
                                          const OffsetBuffer &output,
                                          const OffsetBuffer &encoded_params,
                                          const OffsetBuffer &ctrl_buffer) {

  m_in_left.set_buf(input_left);
  m_in_right.set_buf(input_right);
  m_output.set_buf(output);
  m_encoded_params = encoded_params;

  return MLI_STATUS_OK;
}

mli_status MatMul_CS::EncodeParams(const Buffer &in_bias1, 
                                   const Buffer &in_bias2,
                                   Buffer &encoded_params) {
  // the element size of source should eqaul to the encoded one's
  assert(in_bias1.get_size() + in_bias2.get_size() == encoded_params.get_size());
  assert(in_bias1.get_size() == in_bias2.get_size() == 1);

  // in_zp must be int8_t
  assert(in_bias1.get_elem_size() == sizeof(int8_t));
  encoded_params.write<int8_t>(0, in_bias1.read<int8_t>(0));
  encoded_params.write<int8_t>(1, in_bias2.read<int8_t>(0));

  return MLI_STATUS_OK;
}

unsigned MatMul_CS::GetEncodedParamsSize() const {
  return m_encoded_params_buffer_size;
}

}
