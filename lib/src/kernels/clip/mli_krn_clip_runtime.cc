/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#include <cstring>

#include "mli_debug.h"
#include "mli_ref_runtime_api.hpp"
#include "mli_krn_clip.hpp"
#include "mli_ref_private_types.hpp"

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
typedef vNx4accshort_t mli_8x8_accu_t;
#else
typedef mli_acc32_t mli_8x8_accu_t;
#endif

namespace snps_arc::metaware::mli::ref {

namespace mli_krn = ::snps_arc::metaware::mli::krn;

Clip::Clip(void* kernel_private_data_buffer,
                               size_t size,
                               uint64_t membases[], int num_mems) {
  MLI_ASSERT(size == sizeof(ClipPrivateData));
  ClipPrivateData  private_data;
  memcpy(&private_data, kernel_private_data_buffer, sizeof(ClipPrivateData));
  MLI_ASSERT(private_data.kernel_id == kClipId);
  MLI_ASSERT(private_data.size == sizeof(ClipPrivateData));
  assert(num_mems > 0);

  m_input = private_data.input;
  m_output = private_data.output;

  // set input tile
  if (private_data.input.get_buf().get_elem_size() == sizeof(int8_t)) {
    InternalBuffer input_internal(private_data.input.get_buf(), membases, num_mems);
    m_tile_input.el_type = MLI_EL_SA_8;
    m_tile_input.data.mem.pi8 = input_internal.get_ptr<int8_t>();
  } else {
    MLI_ASSERT(false);
  }
  const auto input = m_input.GetSubTensor();
  m_tile_input.rank = input.get_rank();
  input.get_dims(m_tile_input.shape);
  input.get_mem_strides(m_tile_input.mem_stride);

  // set output tile
  if (private_data.output.get_buf().get_elem_size() == sizeof(int8_t)) {
    InternalBuffer output_internal(private_data.output.get_buf(), membases, num_mems);
    m_tile_output.el_type = MLI_EL_SA_8;
    m_tile_output.data.mem.pi8 = output_internal.get_ptr<int8_t>();

  } else {
      MLI_ASSERT(false);
  }
  const auto output = m_output.GetSubTensor();
  m_tile_output.rank = output.get_rank();
  output.get_dims(m_tile_output.shape);
  output.get_mem_strides(m_tile_output.mem_stride);

  // set min param
  InternalBuffer encoded_params_internal(private_data.encoded_params_buf, membases, num_mems);
  if (private_data.input.get_buf().get_elem_size() == sizeof(int8_t)) {
      encoded_params_internal.set_elem_size(sizeof(int8_t));
      m_min.data.mem.pi8 = encoded_params_internal.get_ptr<int8_t>();
      m_min.el_type = MLI_EL_SA_8;
      encoded_params_internal.inc(1);
  }
  else {
      MLI_ASSERT(0);
  }
  m_min.rank = kClipParamRank;

  // set max param
  if (private_data.input.get_buf().get_elem_size() == sizeof(int8_t)) {
      encoded_params_internal.set_elem_size(sizeof(int8_t));
      m_max.data.mem.pi8 = encoded_params_internal.get_ptr<int8_t>();
      m_max.el_type = MLI_EL_SA_8;
  }
  else {
      MLI_ASSERT(0);
  }
  m_max.rank = kClipParamRank;
}

mli_status Clip::Issue() {

  if (m_input.get_buf().get_elem_size() == sizeof(int8_t) && (m_output.get_buf().get_elem_size() == sizeof(int8_t))) {
      mli_krn::mli_krn_clip<int8_t, int8_t>(&m_tile_input,
                                            &m_min,
                                            &m_max,
                                            &m_tile_output);
  } else {
      assert(0);
  }
  return MLI_STATUS_OK;
}

mli_status Clip::Prefetch() { return MLI_STATUS_OK; }

mli_status Clip::Update() {

  m_input.Next();
  m_output.Next();

  m_input.GetSubTensor().get_dims(m_tile_input.shape);
  m_output.GetSubTensor().get_dims(m_tile_output.shape);

  return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref
