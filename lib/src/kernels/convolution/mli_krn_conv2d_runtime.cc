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

  m_metadata.input = Tensor<InternalBuffer, 4>(private_data.input, membases, num_mems);
  m_metadata.output = Tensor<InternalBuffer, 4>(private_data.output, membases, num_mems);
  m_metadata.weights = Tensor<InternalBuffer, 5>(private_data.weights, membases, num_mems);

  m_metadata.inpzp_buffer = InternalBuffer(private_data.inpzp_buffer, membases, num_mems);
  m_metadata.wtszp_buffer = InternalBuffer(private_data.wtszp_buffer, membases, num_mems);

  m_metadata.inp_quant_axis = private_data.inp_quant_axis;
  m_metadata.wts_quant_axis = private_data.wts_quant_axis;

  m_metadata.cfg = private_data.config;
}

mli_status Conv2d::Issue() {
  // element size for input, output and weights in bytes
  uint32_t i_elem_size = m_metadata.input.get_elem_size();
  uint32_t o_elem_size = m_metadata.output.get_elem_size();
  uint32_t w_elem_size = m_metadata.weights.get_elem_size();

  if (i_elem_size == sizeof(int8_t) &&
      w_elem_size == sizeof(int8_t) &&
      o_elem_size == sizeof(int32_t)) {

    QTensor<InternalBuffer, 4> qinput{
      m_metadata.input, m_metadata.inpzp_buffer, m_metadata.inp_quant_axis};
    QTensor<InternalBuffer, 5> qweights{
      m_metadata.weights, m_metadata.wtszp_buffer, m_metadata.wts_quant_axis};

    conv2d_prepare_and_run<int8_t, int8_t, int32_t, mli_8x8_accu_t, LAYOUT_HWC,
                           ::mli::CONV_GENERAL, /* io_rank */4, /* w_rank */5>(
        qinput, qweights, m_metadata.output, m_metadata.cfg);
  } else {
    // datatype is not supported yet
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}

mli_status Conv2d::Prefetch() { return MLI_STATUS_OK; }

mli_status Conv2d::Update() { return MLI_STATUS_OK; }

}  // namespace snps_arc::metaware::mli::ref