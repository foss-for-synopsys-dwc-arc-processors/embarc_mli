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
#include "common/mli_krn_fully_connected.h"
#include "mli_ref_runtime_api.hpp"

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
typedef vNx4accshort_t mli_8x8_accu_t;
#else
typedef mli_acc32_t mli_8x8_accu_t;
#endif

namespace snps_arc::metaware::mli::ref {

FullyConnected::FullyConnected(void* kernel_private_data_buffer,
                               size_t size,
                               uint64_t membases[], int num_mems) {
  MLI_ASSERT(size == sizeof(FullyConnectedPrivateData));
  FullyConnectedPrivateData  private_data;
  memcpy(&private_data, kernel_private_data_buffer, sizeof(FullyConnectedPrivateData));
  MLI_ASSERT(private_data.kernel_id == kFullyConnectedId);
  MLI_ASSERT(private_data.size == sizeof(FullyConnectedPrivateData));

  // element size for input, output and weights in bytes
  m_i_elem_size = private_data.input_buffer.get_elem_size();
  m_o_elem_size = private_data.output_buffer.get_elem_size();
  m_w_elem_size = private_data.weights_buffer.get_elem_size();

  assert(num_mems > 0);
  m_metadata = FullyConnectedMetadata();

  // TODO: Move partly or all into a helper function and use for each tensor
  {
    // Reconstruct Input Tensor
    auto& tsr = m_metadata.input;
    if (m_i_elem_size == sizeof(int8_t)) {

      InternalBuffer input_internal(private_data.input_buffer, membases, num_mems);
      tsr.el_type = MLI_EL_SA_8;
      tsr.data.mem.pi8 = input_internal.get_ptr<int8_t>();
    } else {
      assert(false);
    }
    // [N,IC]
    tsr.rank = 2;
    tsr.shape[0] = private_data.input_n;
    tsr.shape[1] = private_data.input_ic;

    tsr.mem_stride[0] = private_data.input_n_stride;
    tsr.mem_stride[1] = private_data.input_ic_stride;

    // input zero points per tensor
    uint32_t inpzp_elem_size = private_data.inpzp_buffer.get_elem_size();
    assert(inpzp_elem_size == sizeof(int16_t));
    if (private_data.inpzp_buffer.get_size() / inpzp_elem_size == 1) {
      // per-tensor quantization
      MLI_ASSERT(inpzp_elem_size == sizeof(int16_t));
      tsr.el_params.sa.dim = -1;
      tsr.el_params.sa.zero_point.capacity = 0;

      InternalBuffer inpzp_internal(private_data.inpzp_buffer, membases, num_mems);
      tsr.el_params.sa.zero_point.mem.i16 = inpzp_internal.read<int16_t>(0);
    } else {
      // not support yet
      assert(false);
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
      assert(false);
    }
    //[N, OC] = [N,IC] * [IC, OC]
    tsr.rank = 2;
    tsr.shape[0] = private_data.output_n;
    tsr.shape[1] = private_data.output_oc;
    tsr.mem_stride[0] = private_data.output_n_stride;
    tsr.mem_stride[1] = private_data.output_oc_stride;
  }

  {
    // Reconstruct Weights Tensor
    auto& tsr = m_metadata.weights;
    if (m_w_elem_size == sizeof(int8_t)) {
      InternalBuffer weights_internal(private_data.weights_buffer, membases, num_mems);
      tsr.el_type = MLI_EL_SA_8;
      tsr.data.mem.pi8 = weights_internal.get_ptr<int8_t>();
      } else {
      assert(false);
    }
    // weights [IC,OC]
    tsr.rank = 2;
    tsr.shape[0] = private_data.weights_ic;
    tsr.shape[1] = private_data.weights_oc;
    tsr.mem_stride[0] = private_data.weights_ic_stride;
    tsr.mem_stride[1] = private_data.weights_oc_stride;

    // weights zero point should have the same size as the tensor they belong to.
    uint32_t wtszp_elem_size = private_data.wtszp_buffer.get_elem_size();
    assert(wtszp_elem_size == sizeof(int16_t));
    uint32_t wtszp_size = private_data.wtszp_buffer.get_size();

    // per-channel quantization
    if (wtszp_size / wtszp_elem_size > 1) {
      MLI_ASSERT(private_data.weights_oc == wtszp_size / wtszp_elem_size);
      MLI_ASSERT(wtszp_elem_size == sizeof(int16_t));
      tsr.el_params.sa.dim = 1; // channel dim

      tsr.el_params.sa.zero_point.capacity = wtszp_size;
      InternalBuffer wtszp_internal(private_data.wtszp_buffer, membases, num_mems);
      tsr.el_params.sa.zero_point.mem.pi16 = wtszp_internal.get_ptr<int16_t>();
    } else {
      // not support yet
      assert(false);
    }
  }
}

mli_status FullyConnected::Issue() {
  if (m_i_elem_size == sizeof(int8_t) &&
      m_w_elem_size == sizeof(int8_t) &&
      m_o_elem_size == sizeof(int32_t)) {
        mli_fully_connected_cfg cfg = { MLI_RELU_NONE };
        MLI_ASSERT(cfg.relu.type == MLI_RELU_NONE);
        ::mli::krn::fully_connected_prepare_and_run<
            int8_t, int8_t, int32_t, int32_t, mli_8x8_accu_t,
            ::mli::krn::int_quant_specific_params, true>(
            &m_metadata.input, &m_metadata.weights,
            /* bias */ nullptr, &cfg, &m_metadata.output);
  } else {
    // datatype is not supported yet
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}

mli_status FullyConnected::Prefetch() { return MLI_STATUS_OK; }

mli_status FullyConnected::Update() { return MLI_STATUS_OK; }

}  // namespace snps_arc::metaware::mli::ref