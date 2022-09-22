/*
 * Copyright 2022, Synopsys, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-3-Clause license found in
 * the LICENSE file in the root directory of this source tree.
 *
 */
#include <cstring>

#include "mli_ref_runtime_api.hpp"
#include "mli_ref_compiler_api.hpp"
#include "mli_ref_private_types.hpp"

namespace snps_arc::metaware::mli::ref {

ReduceMax_CS::ReduceMax_CS(const lib_mli::PlatformDescription pd,
                           const TensorIterator<NoBuffer, kReduceMaxRank, kReduceMaxIterRank> &in,
                           const ReduceOpConfig &cfg,
                           const TensorIterator<NoBuffer, kReduceMaxRank, kReduceMaxIterRank> &out)
                           : m_cfg(cfg),
                             m_in(in),
                             m_out(out),
                             m_pd(pd) {
    uint32_t in_shape[4];
    uint32_t out_shape[4];
    int32_t in_stride[4];
    int32_t out_stride[4];

    in.get_full_shape(in_shape);
    in.get_mem_strides(in_stride);
    out.get_full_shape(out_shape);
    out.get_mem_strides(out_stride);
    
}

mli_status ReduceMax_CS::AttachBufferOffsets(const OffsetBuffer &input,
                                             const OffsetBuffer &output,
                                             const OffsetBuffer &ctrl_buffer) {

    m_in.set_buf(input);
    m_out.set_buf(output);

    return MLI_STATUS_OK;
}

mli_status ReduceMax_CS::GetKernelPrivateData(void *kernel_private_data_buffer ) {
    ReduceMaxPrivateData opaque_obj;

    MLI_ASSERT(m_in.get_tensor().get_rank() == m_out.get_tensor().get_rank());
    opaque_obj.io_rank = m_in.get_tensor().get_rank();
    opaque_obj.input = m_in;
    opaque_obj.output = m_out;

    MLI_ASSERT( (0 <= m_cfg.axis) && ((int32_t)(opaque_obj.io_rank) > m_cfg.axis) );
    opaque_obj.reduce_axis = m_cfg.axis;

    for(int32_t i = 0; i < (int32_t)(opaque_obj.io_rank); i++) {
        if( i != opaque_obj.reduce_axis ){
            MLI_ASSERT(m_in.get_dim(i) == m_out.get_dim(i));
        }
        else{
            MLI_ASSERT(1 == m_out.get_dim(i));
        }
    }

    std::memcpy(kernel_private_data_buffer, (void *)&opaque_obj, sizeof(opaque_obj));

    return MLI_STATUS_OK;    

}

unsigned ReduceMax_CS::GetKernelPrivateDataSize() const {
    return sizeof(ReduceMaxPrivateData);
}

unsigned ReduceMax_CS::GetRuntimeObjectSize() const {
    return sizeof(ReduceMax);
}

}  // namespace snps_arc::metaware::mli::ref
