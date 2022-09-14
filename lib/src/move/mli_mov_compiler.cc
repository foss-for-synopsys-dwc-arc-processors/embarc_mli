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
#include "mli_ref_private_types.hpp"

namespace snps_arc::metaware::mli::ref {

Move_CS::Move_CS(const lib_mli::PlatformDescription pd,
                 const Tensor<NoBuffer, kMoveRank> src,
                 const Tensor<NoBuffer, kMoveRank> dst,
                 const IteratorCfg<kMoveRank> src_it_cfg,
                 const IteratorCfg<kMoveRank> dst_it_cfg) {

  m_src_it = TensorIterator<NoBuffer, kMoveRank, kMoveIterRank>(src, src_it_cfg);
  m_dst_it = TensorIterator<NoBuffer, kMoveRank, kMoveIterRank>(dst, dst_it_cfg);
};

Move_CS::Move_CS(const lib_mli::PlatformDescription pd,
                 const TensorIterator<NoBuffer, kMoveRank, kMoveIterRank> &src,
                 const TensorIterator<NoBuffer, kMoveRank, kMoveIterRank> &dst)
                 : m_pd(pd), m_src_it(src), m_dst_it(dst) {
    // no body is needed (all is done in the initializer list)
}

unsigned Move_CS::GetKernelPrivateDataSize() const {
  return sizeof(MovePrivateData);
}

unsigned Move_CS::GetRuntimeObjectSize() const {
  return sizeof(Move);
}

mli_status Move_CS::GetKernelPrivateData(void *kernel_private_data_buffer) {
  MovePrivateData obj;

  obj.src_it = m_src_it;
  obj.dst_it = m_dst_it;

  std::memcpy(kernel_private_data_buffer, (void *)&obj, sizeof(obj));

  return MLI_STATUS_OK;
}

mli_status Move_CS::AttachBufferOffsets(const Tensor<OffsetBuffer, kMoveRank> &src,
                                        const Tensor<OffsetBuffer, kMoveRank> &dst) {
  m_src_offset_buf = src.get_buf();
  m_dst_offset_buf = dst.get_buf();

  assert(src.get_elem_size() == dst.get_elem_size());

  return MLI_STATUS_OK;
}

mli_status Move_CS::AttachBufferOffsets(const OffsetBuffer &src,
                                        const OffsetBuffer &dst,
                                        const OffsetBuffer &ctrl_buffer) {
  m_src_it.set_buf(src);
  m_dst_it.set_buf(dst);

  assert(src.get_elem_size() == dst.get_elem_size());

  return MLI_STATUS_OK;
}


}  // namespace snps_arc::metaware::mli::ref