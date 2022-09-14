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
#include "mli_ref_compiler_api.hpp"
#include "mli_ref_private_types.hpp"
#include "mli_ref_runtime_api.hpp"

namespace snps_arc::metaware::mli::ref {

Move::Move(void* kernel_private_data_buffer, size_t size, uint64_t membases[],
           int num_mems) {
  MLI_ASSERT(size == sizeof(MovePrivateData));
  MovePrivateData private_data;
  memcpy(&private_data, kernel_private_data_buffer, sizeof(MovePrivateData));
  MLI_ASSERT(private_data.kernel_id == kMoveId);
  MLI_ASSERT(private_data.size == sizeof(MovePrivateData));

  m_src_it = TensorIterator<InternalBuffer, kMoveRank, kMoveIterRank>(private_data.src_it, membases, num_mems);

  m_dst_it = TensorIterator<InternalBuffer, kMoveRank, kMoveIterRank>(private_data.dst_it, membases, num_mems);

  m_src_it.Reset();
  m_dst_it.Reset();
}

mli_status Move::Issue() {
  TensorIterator<InternalBuffer, kMoveRank, kMoveIterRank> src_it =
      m_src_it.GetSubTensorIterator();
  TensorIterator<InternalBuffer, kMoveRank, kMoveIterRank> dst_it =
      m_dst_it.GetSubTensorIterator();
  int32_t count[kMoveRank] = {0};
  for (uint32_t i = 0; i < kMoveRank; i++) {
    count[i] = src_it.GetTensorShape(i);
    if (count[i] <= 0) count[i] = 1;
  }
  src_it.SetCount(count);
  dst_it.SetCount(count);

  bool src_done = false;
  bool dst_done = false;
  while (!src_done) {
    switch (src_it.get_elem_size()) {
      case 1:
        dst_it.write(src_it.template read<uint8_t>());
        break;
      case 2:
        dst_it.write(src_it.template read<uint16_t>());
        break;
      case 4:
        dst_it.write(src_it.template read<uint32_t>());
        break;
      default:
        MLI_ASSERT(false);
    }
    src_done = src_it.Next();
    dst_done = dst_it.Next();
    MLI_ASSERT(src_done == dst_done);
  }
  return MLI_STATUS_OK;
}

mli_status Move::Prefetch() {
  return MLI_STATUS_OK; 
}

mli_status Move::Update() {
  m_src_it.Next();
  m_dst_it.Next();
  return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref