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

MoveBroadcast::MoveBroadcast(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems) {
    MLI_ASSERT(size == sizeof(MoveBroadcastPrivateData));
    MoveBroadcastPrivateData private_buffer;
    memcpy(&private_buffer, kernel_private_data_buffer, sizeof(MoveBroadcastPrivateData));
    MLI_ASSERT(private_buffer.kernel_id == kMoveBroadcastId);
    MLI_ASSERT(private_buffer.size == sizeof(MoveBroadcastPrivateData));

    m_src = private_buffer.src;
    m_dst = private_buffer.dst;
    m_tile_src = Tensor<InternalBuffer, kMoveBroadcastRank>(m_src.GetSubTensor(), membases, num_mems);
    m_tile_dst = Tensor<InternalBuffer, kMoveBroadcastRank>(m_dst.GetSubTensor(), membases, num_mems);
}

template <typename buf_T, unsigned N>
int32_t tensor_read(Tensor<buf_T, N> tsr, uint32_t *index) {
    int32_t result = 0;
    int32_t offset = tsr.get_offset(index);
    switch (tsr.get_elem_size()) {
        case sizeof(int8_t):
            result = tsr.template read<int8_t>(offset);
            break;
        case sizeof(int16_t):
            result = tsr.template read<int16_t>(offset);
            break;
        case sizeof(int32_t):
            result = tsr.template read<int32_t>(offset);
            break;
        default:
            MLI_ASSERT(false);
    }
    return result;
}

template <typename buf_T, unsigned N>
void tensor_write(Tensor<buf_T, N> tsr, uint32_t *index, int32_t value) {
    int32_t offset = tsr.get_offset(index);
    switch (tsr.get_elem_size()) {
        case sizeof(int8_t):
            tsr.template write<int8_t>(offset, value);
            break;
        case sizeof(int16_t):
            tsr.template write<int16_t>(offset, value);
            break;
        case sizeof(int32_t):
            tsr.template write<int32_t>(offset, value);
            break;
        default:
            MLI_ASSERT(false);
    }
}

// Move Broadcast Core Function
template <typename buf_T, unsigned N>
void MoveBroadcast::MoveBroadcastRun(Tensor<buf_T, N> &src, Tensor<buf_T, N> &dst) {
    uint32_t src_idx[N] = {0};
    uint32_t dst_idx[N] = {0};
    uint32_t src_shape[N] = {0};
    uint32_t dst_shape[N] = {0};
    uint32_t src_rank = src.get_rank();
    uint32_t dst_rank = dst.get_rank();

    MLI_ASSERT(src_rank == dst_rank);

    // get shapes
    src.get_dims(src_shape);
    dst.get_dims(dst_shape);

    // Tensors with rank less than MLI_MAX_RANK, the tensor is automatically filled with 1's
    for (uint32_t i = src_rank; i < kMoveBroadcastRank; i++) {
        src_shape[i] = 1;
    }
    for (uint32_t i = dst_rank; i < kMoveBroadcastRank; i++) {
        dst_shape[i] = 1;
    }

    for (int d0_cnt = 0; d0_cnt < (int)dst_shape[0]; d0_cnt++) {
        for (int d1_cnt = 0; d1_cnt < (int)dst_shape[1]; d1_cnt++) {
            for (int d2_cnt = 0; d2_cnt < (int)dst_shape[2]; d2_cnt++) {
                for (int d3_cnt = 0; d3_cnt < (int)dst_shape[3]; d3_cnt++) {
                    // ToDo: when mli_tensor takes [rank=5] 
                    // for (int d4_cnt = 0; d4_cnt < (int)dst_shape[4]; d4_cnt++) {
                        dst_idx[0] = d0_cnt;
                        dst_idx[1] = d1_cnt;
                        dst_idx[2] = d2_cnt;
                        dst_idx[3] = d3_cnt;
                        // dst_idx[4] = d4_cnt;
                        
                        // inner loop for move broad cast.
                        for (uint32_t i = 0; i < dst_rank; i++) {
                            if(src_shape[i] != dst_shape[i]) {
                                MLI_ASSERT(src_shape[i] == 1);
                                src_idx[i] = 0;
                            }
                            else {
                                src_idx[i] = dst_idx[i];
                            }
                        }
                        int32_t value = tensor_read<buf_T, N>(src, src_idx);
                        tensor_write<buf_T, N>(dst, dst_idx, value);
                    // }
                }
            }
        }
    }
}

mli_status MoveBroadcast::Issue() {    
    MoveBroadcastRun<InternalBuffer, kMoveBroadcastRank>(m_tile_src, m_tile_dst);
    return MLI_STATUS_OK;
}

mli_status MoveBroadcast::Prefetch() {
    return MLI_STATUS_OK; 
}

mli_status MoveBroadcast::Update() {
    m_src.Next();
    m_dst.Next();

    const auto src_tile_tensor = m_src.GetSubTensor();
    uint32_t src_tile_shape[kMoveBroadcastRank];
    src_tile_tensor.get_dims(src_tile_shape);
    m_tile_src = Tensor<InternalBuffer, kMoveBroadcastRank>(m_tile_src, src_tile_shape);

    const auto dst_tile_tensor = m_dst.GetSubTensor();
    uint32_t dst_tile_shape[kMoveBroadcastRank];
    dst_tile_tensor.get_dims(dst_tile_shape);
    m_tile_dst = Tensor<InternalBuffer, kMoveBroadcastRank>(m_tile_dst, dst_tile_shape);

    return MLI_STATUS_OK;
}

void MoveBroadcast::GetIOSizesAndOffsets(uint32_t src_size[kMoveBroadcastRank], uint32_t dst_size[kMoveBroadcastRank],
                                         int32_t src_offsets[kMoveBroadcastRank], int32_t dst_offsets[kMoveBroadcastRank]) {
    
    m_src.get_pos(src_offsets);
    m_dst.get_pos(dst_offsets);

    const auto src_tile_tensor = m_src.GetSubTensor();
    src_tile_tensor.get_dims(src_size);

    const auto dst_tile_tensor = m_dst.GetSubTensor();
    dst_tile_tensor.get_dims(dst_size);
}

}  // namespace snps_arc::metaware::mli::ref