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

MoveBroadcast_CS::MoveBroadcast_CS(const lib_mli::PlatformDescription pd,
                                   const TensorIterator<NoBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank> &src,
                                   const TensorIterator<NoBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank> &dst) 
                                   : m_pd(pd), m_src(src), m_dst(dst) {
}

unsigned MoveBroadcast_CS::GetKernelPrivateDataSize() const {
    return sizeof(MoveBroadcastPrivateData);
}

unsigned MoveBroadcast_CS::GetRuntimeObjectSize() const {
    return sizeof(MoveBroadcast);
}

mli_status MoveBroadcast_CS::GetKernelPrivateData(void *kernel_private_data_buffer) {
    MoveBroadcastPrivateData obj;
    obj.src = m_src;
    obj.dst = m_dst;

    std::memcpy(kernel_private_data_buffer, (void *)&obj, sizeof(obj));
    MLI_ASSERT(obj.src.get_tensor().get_rank() == obj.dst.get_tensor().get_rank());
    return MLI_STATUS_OK;
}

mli_status MoveBroadcast_CS::AttachBufferOffsets(const OffsetBuffer &src,
                                                 const OffsetBuffer &dst,
                                                 const OffsetBuffer &ctrl_buffer) {
    m_src.set_buf(src);
    m_dst.set_buf(dst);

    MLI_ASSERT(src.get_elem_size() == dst.get_elem_size());
    return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref