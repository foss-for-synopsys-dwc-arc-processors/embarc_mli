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

Permute_CS::Permute_CS(const lib_mli::PlatformDescription pd,
                       const TensorIterator<NoBuffer, kPermuteRank, kPermuteIterRank> input,
                       const PermuteOpConfig &cfg,
                       const TensorIterator<NoBuffer, kPermuteRank, kPermuteIterRank> output) : 
                       m_cfg(cfg), m_in(input), m_out(output), m_pd(pd) {
    uint32_t in_shape[kPermuteRank];
    uint32_t out_shape[kPermuteRank];
    int32_t in_stride[kPermuteRank];
    int32_t out_stride[kPermuteRank];

    input.get_full_shape(in_shape);
    input.get_mem_strides(in_stride);
    output.get_full_shape(out_shape);
    output.get_mem_strides(out_stride);

    m_input_buffer_size = service::GetBufferSize(input.get_tensor().get_rank(), in_shape, in_stride);
    m_output_buffer_size = service::GetBufferSize(output.get_tensor().get_rank(), out_shape, out_stride);
}

mli_status Permute_CS::AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &ctrl_buffer) {

    m_in.set_buf(input);
    m_out.set_buf(output);

    return MLI_STATUS_OK;
}

mli_status Permute_CS::GetKernelPrivateData(void *kernel_private_data_buffer) {
    PermutePrivateData opaque_obj;
    opaque_obj.io_rank = m_in.get_tensor().get_rank();
    uint8_t temp[kPermuteRank] = {0};

    // 1- Check that input and output have same rank.
    MLI_ASSERT(m_in.get_tensor().get_rank() == m_out.get_tensor().get_rank());
    
    opaque_obj.input = m_in;
    opaque_obj.output = m_out;
    
    for (uint32_t k = 0; k < opaque_obj.io_rank; k++) {

        // 2- Check that values must be valid dimensions within input shape.
        MLI_ASSERT((0 <= m_cfg.perm_dim[k]) && (m_cfg.perm_dim[k] < (int32_t)(opaque_obj.io_rank)));

        // 3- Check that values shouldn't be repeated.
        MLI_ASSERT(++temp[m_cfg.perm_dim[k]] == 1);

        opaque_obj.perm_dim[k] = m_cfg.perm_dim[k];
    }

    std::memcpy(kernel_private_data_buffer, (void *)&opaque_obj, sizeof(opaque_obj));

    return MLI_STATUS_OK;    

}

unsigned Permute_CS::GetKernelPrivateDataSize() const {
    return sizeof(PermutePrivateData);
}

unsigned Permute_CS::GetRuntimeObjectSize() const {
    return sizeof(Permute);
}

}  // namespace snps_arc::metaware::mli::krn::ref