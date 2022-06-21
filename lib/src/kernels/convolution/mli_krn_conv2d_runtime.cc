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

  // element size for input, output and weights in bytes
  m_i_elem_size = private_data.input_buffer.get_elem_size();
  m_o_elem_size = private_data.output_buffer.get_elem_size();
  m_w_elem_size = private_data.weights_buffer.get_elem_size();

  m_metadata = Conv2dMetadata();

  // Reconstruct configuration
  auto krn_cfg = &m_metadata.cfg;
  krn_cfg->dilation_height = private_data.dilation_height;
  krn_cfg->dilation_width = private_data.dilation_width;
  krn_cfg->padding_bottom = private_data.padding_bottom;
  krn_cfg->padding_top = private_data.padding_top;
  krn_cfg->padding_left = private_data.padding_left;
  krn_cfg->padding_right = private_data.padding_right;
  krn_cfg->stride_height = private_data.stride_height;
  krn_cfg->stride_width = private_data.stride_width;
  krn_cfg->relu.type = MLI_RELU_NONE;

  // TODO: Move partly or all into a helper function and use for each tensor
  {
    // Reconstruct Input Tensor
    auto& tsr = m_metadata.input;
    if (m_i_elem_size == sizeof(int8_t)) {
      InternalBuffer input_internal(private_data.input_buffer, membases, num_mems);
      tsr.el_type = MLI_EL_SA_8;
      tsr.data.mem.pi8 = input_internal.get_ptr<int8_t>();
    } else {
      MLI_ASSERT(false);
    }
    // HWCi
    tsr.rank = 3;
    tsr.shape[0] = private_data.input_h;
    tsr.shape[1] = private_data.input_w;
    tsr.shape[2] = private_data.input_c;
    tsr.mem_stride[0] = private_data.input_h_stride;
    tsr.mem_stride[1] = private_data.input_w_stride;
    tsr.mem_stride[2] = 1;

    // input zero points
    uint32_t inpzp_elem_size = private_data.inpzp_buffer.get_elem_size();
    MLI_ASSERT(inpzp_elem_size == sizeof(int16_t));
    if (private_data.inpzp_buffer.get_size() / inpzp_elem_size == 1) {
      // per-tensor quantization
      MLI_ASSERT(inpzp_elem_size == sizeof(int16_t));
      tsr.el_params.sa.dim = -1;
      tsr.el_params.sa.zero_point.capacity = 0;
      InternalBuffer inpzp_internal(private_data.inpzp_buffer, membases, num_mems);
      tsr.el_params.sa.zero_point.mem.i16 = inpzp_internal.read<int16_t>(0);
    } else {
      // not support yet
      MLI_ASSERT(false);
    }
  }

  {
    // Reconstruct Output Tensor
    auto& tsr = m_metadata.output;
    if (m_o_elem_size == sizeof(int32_t)) {
      InternalBuffer output_internal(private_data.output_buffer, membases, num_mems);
      tsr.el_type = MLI_EL_SA_32;
      tsr.data.mem.pi32 = output_internal.get_ptr<int32_t>();
    } else {
      MLI_ASSERT(false);
    }
    // HWCo
    tsr.rank = 3;
    tsr.shape[0] = private_data.output_h;
    tsr.shape[1] = private_data.output_w;
    tsr.shape[2] = private_data.output_c;
    tsr.mem_stride[0] = private_data.output_h_stride;
    tsr.mem_stride[1] = private_data.output_w_stride;
    tsr.mem_stride[2] = 1;
  }

  {
    // Reconstruct Weights Tensor
    auto& tsr = m_metadata.weights;
    if (m_w_elem_size == sizeof(int8_t)) {
      InternalBuffer weights_internal(private_data.weights_buffer, membases, num_mems);
      tsr.el_type = MLI_EL_SA_8;
      tsr.data.mem.pi8 = weights_internal.get_ptr<int8_t>();
    } else {
      MLI_ASSERT(false);
    }
    // HWCinCo
    tsr.rank = 4;
    tsr.shape[0] = private_data.weights_h;
    tsr.shape[1] = private_data.weights_w;
    tsr.shape[2] = private_data.input_c;
    tsr.shape[3] = private_data.output_c;
    tsr.mem_stride[0] = private_data.weights_h_stride;
    tsr.mem_stride[1] = private_data.weights_w_stride;
    tsr.mem_stride[2] = private_data.weights_c_stride;
    tsr.mem_stride[3] = 1;

    // weights zero point should have the same size as the tensor they belong to.
    uint32_t wtszp_elem_size = private_data.wtszp_buffer.get_elem_size();
    MLI_ASSERT(wtszp_elem_size == sizeof(int16_t));
    uint32_t wtszp_size = private_data.wtszp_buffer.get_size();
    if (wtszp_size / wtszp_elem_size > 1) {
      // per-channel quantization
      MLI_ASSERT(private_data.output_c == wtszp_size / wtszp_elem_size);
      MLI_ASSERT(wtszp_elem_size == sizeof(int16_t));
      tsr.el_params.sa.dim = 3; // channel dim
      tsr.el_params.sa.zero_point.capacity = wtszp_size;
      InternalBuffer wtszp_internal(private_data.wtszp_buffer, membases, num_mems);
      tsr.el_params.sa.zero_point.mem.pi16 = wtszp_internal.get_ptr<int16_t>();
    } else {
      // not support yet
      MLI_ASSERT(false);
    }
  }
}

mli_status Conv2d::Issue() {
  if (m_i_elem_size == sizeof(int8_t) &&
      m_w_elem_size == sizeof(int8_t) &&
      m_o_elem_size == sizeof(int32_t)) {
    ::mli::krn::conv2d_prepare_and_run
        <int8_t, int8_t, int32_t, int32_t, mli_8x8_accu_t,
         ::mli::krn::int_quant_specific_params, LAYOUT_HWCN,
         ::mli::CONV_GENERAL, KRN_SZ_VAR, KRN_SZ_VAR>(&m_metadata.input,
                                                      &m_metadata.weights,
                                                      /* bias */ nullptr,
                                                      &m_metadata.cfg,
                                                      &m_metadata.output);
  } else {
    // datatype is not supported yet
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}

mli_status Conv2d::Prefetch() { return MLI_STATUS_OK; }

mli_status Conv2d::Update() { return MLI_STATUS_OK; }

}  // namespace snps_arc::metaware::mli::ref