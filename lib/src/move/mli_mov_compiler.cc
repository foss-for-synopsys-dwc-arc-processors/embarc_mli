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
                 const Tensor<NoBuffer, kMaxRank> src,
                 const Tensor<NoBuffer, kMaxRank> dst,
                 const IteratorCfg<kMaxRank> src_it_cfg,
                 const IteratorCfg<kMaxRank> dst_it_cfg)
    : m_src_cfg(src_it_cfg), m_dst_cfg(dst_it_cfg) {
  m_src_rank = 0;
  m_dst_rank = 0;

  for (unsigned dim = 0; dim < kMaxRank; dim++) {
    m_src_shape[dim] = src.get_dim(dim);
    if (m_src_shape[dim] > 0) m_src_rank += 1;
    m_src_stride[dim] = src.get_mem_stride(dim);
  }

  for (unsigned dim = 0; dim < kMaxRank; dim++) {
    m_dst_shape[dim] = dst.get_dim(dim);
    if (m_dst_shape[dim] > 0) m_dst_rank += 1;
    m_dst_stride[dim] = dst.get_mem_stride(dim);
  }
};

unsigned Move_CS::GetKernelPrivateDataSize() const {
  return sizeof(MovePrivateData);
}

unsigned Move_CS::GetRuntimeObjectSize() const { 
  return sizeof(Move); 
}

mli_status Move_CS::GetKernelPrivateData(void *kernel_private_data_buffer) {
  MovePrivateData obj;

  obj.src = m_src;
  obj.dst = m_dst;
  obj.src_cfg = m_src_cfg;
  obj.dst_cfg = m_dst_cfg;

  std::memcpy(kernel_private_data_buffer, (void *)&obj, sizeof(obj));

  return MLI_STATUS_OK;
}

mli_status Move_CS::AttachBufferOffsets(const Tensor<OffsetBuffer, kMaxRank> &src,
                                        const Tensor<OffsetBuffer, kMaxRank> &dst) {
  m_src = src;
  m_dst = dst;
  return MLI_STATUS_OK;
}

unsigned Move_CS::GetInputBufferSize() const {
  return service::GetBufferSize(m_src_rank, m_src_shape, m_src_stride);
}
unsigned Move_CS::GetOutputBufferSize() const {
  return service::GetBufferSize(m_dst_rank, m_dst_shape, m_dst_stride);
}

unsigned Move_CS::GetDataBufferSize() const { return 0; }

}  // namespace snps_arc::metaware::mli::ref