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

    m_metadata.input = Tensor<InternalBuffer, kTransposeConvIORank>(
        private_data.input.GetSubTensor(), membases, num_mems);
    m_metadata.weights = Tensor<InternalBuffer, kTransposeConvWRank>(
        private_data.weights.GetSubTensor(), membases, num_mems);
    m_metadata.weights_zp = Tensor<InternalBuffer, kTransposeConvZPRank>(
        private_data.weights_zp.GetSubTensor(), membases, num_mems);
    m_metadata.output = Tensor<InternalBuffer, kTransposeConvIORank>(
        private_data.output.GetSubTensor(), membases, num_mems);
    m_metadata.inpzp_buffer =
        InternalBuffer(private_data.inpzp_buffer, membases, num_mems);

    m_metadata.inp_quant_axis = private_data.inp_quant_axis;
    m_metadata.wts_quant_axis = private_data.wts_quant_axis;
    m_metadata.cfg = private_data.config;
}

mli_status TransposeConv2D::Issue() {
  // element size for input, output and weights in bytes
  uint32_t i_elem_size = m_metadata.input.get_buf().get_elem_size();
  uint32_t o_elem_size = m_metadata.output.get_buf().get_elem_size();
  uint32_t w_elem_size = m_metadata.weights.get_buf().get_elem_size();
  
  if (i_elem_size == sizeof(int8_t) && w_elem_size == sizeof(int8_t) &&
      o_elem_size == sizeof(int32_t)) {
      QTensor<InternalBuffer, kTransposeConvIORank> qinput{m_metadata.input,
                                                  m_metadata.inpzp_buffer,
                                                  m_metadata.inp_quant_axis};
      QTensor<InternalBuffer, kTransposeConvWRank> qweights{m_metadata.weights, 
                                                   m_metadata.weights_zp.get_buf(),
                                                   m_metadata.wts_quant_axis};

      transpose_conv2d_prepare_and_run<int8_t, int8_t, int32_t, mli_8x8_accu_t,
                                       LAYOUT_HWC, kTransposeConvIORank,
                                       kTransposeConvWRank>(
          qinput, qweights, m_metadata.output, m_metadata.cfg);

  } else {
      // datatype is not supported yet
      return MLI_STATUS_NOT_SUPPORTED;
  }
  
  return MLI_STATUS_OK;
}

mli_status TransposeConv2D::Prefetch() { return MLI_STATUS_OK; }

mli_status TransposeConv2D::Update() { return MLI_STATUS_OK; }

}  // namespace snps_arc::metaware::mli::ref