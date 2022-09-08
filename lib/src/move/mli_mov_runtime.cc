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
#include "mli_ref_compiler_api.hpp"
#include "mli_ref_private_types.hpp"
#include "mli_ref_runtime_api.hpp"

namespace snps_arc::metaware::mli::ref {

Move::Move(void* kernel_private_data_buffer, size_t size,
           uint64_t membases[], int num_mems)
    : m_src_it(
          GetSrcTensorTileItr(kernel_private_data_buffer, membases, num_mems)),
      m_dst_it(
          GetDstTensorTileItr(kernel_private_data_buffer, membases, num_mems)) {

  MLI_ASSERT(size == sizeof(MovePrivateData));
  MovePrivateData private_data;
  memcpy(&private_data, kernel_private_data_buffer, sizeof(MovePrivateData));
  MLI_ASSERT(private_data.kernel_id == kMoveId);
  MLI_ASSERT(private_data.size == sizeof(MovePrivateData));

  m_src_it_cfg = private_data.src_cfg;
  m_dst_it_cfg = private_data.dst_cfg;
  m_src_it.Reset();
  m_dst_it.Reset();
}

template <typename buf_T, unsigned N>
void Move::CopySrcToDst(Tensor<buf_T, N> src, Tensor<buf_T, N> dst) {
  TensorIterator<buf_T, N, N> src_it(src, m_src_it_cfg);
  TensorIterator<buf_T, N, N> dst_it(dst, m_dst_it_cfg);
  bool done = false;
  while (!done) {
    switch (src.get_elem_size()) {
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
    done = src_it.Next();
    // TODO: Better to check one (src or dst) and if second is not done - throw
    // exception
    done |= dst_it.Next();
  }
}

mli_status Move::Issue() {
  Tensor<InternalBuffer, kMoveRank> src;
  Tensor<InternalBuffer, kMoveRank> dst;
  src = m_src_it.GetSubTensor();
  dst = m_dst_it.GetSubTensor();
  CopySrcToDst(src, dst);
  return MLI_STATUS_OK;
}

mli_status Move::Prefetch() {
  return MLI_STATUS_OK; 
}

mli_status Move::Update() {
  m_dst_it.Next();
  m_src_it.Next();
  return MLI_STATUS_OK;
}

TensorIterator<InternalBuffer, kMoveRank, kMoveIterRank> Move::GetSrcTensorTileItr(
    void* kernel_private_data_buffer, uint64_t membases[],
    int num_mems) {
  MovePrivateData private_data;
  memcpy(&private_data, kernel_private_data_buffer, sizeof(MovePrivateData));
  MLI_ASSERT(private_data.kernel_id == kMoveId);
  MLI_ASSERT(private_data.size == sizeof(MovePrivateData));

  return TensorIterator<InternalBuffer, kMoveRank, kMoveIterRank>(
      Tensor<InternalBuffer, kMoveRank>(private_data.src, membases,
                                                num_mems),
      private_data.src_cfg);
}

TensorIterator<InternalBuffer, kMoveRank, kMoveIterRank> Move::GetDstTensorTileItr(
    void* kernel_private_data_buffer, uint64_t membases[],
    int num_mems) {
  MovePrivateData private_data;
  memcpy(&private_data, kernel_private_data_buffer, sizeof(MovePrivateData));
  MLI_ASSERT(private_data.kernel_id == kMoveId);
  MLI_ASSERT(private_data.size == sizeof(MovePrivateData));

  return TensorIterator<InternalBuffer, kMoveRank, kMoveIterRank>(
      Tensor<InternalBuffer, kMoveIterRank>(private_data.dst, membases,
                                                num_mems),
      private_data.dst_cfg);
}

}  // namespace snps_arc::metaware::mli::ref