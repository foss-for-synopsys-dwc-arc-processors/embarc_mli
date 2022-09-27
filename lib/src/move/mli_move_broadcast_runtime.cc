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
    MoveBroadcastPrivateData private_data;
    memcpy(&private_data, kernel_private_data_buffer, sizeof(MoveBroadcastPrivateData));
    MLI_ASSERT(private_data.kernel_id == kMoveBroadcastId);
    MLI_ASSERT(private_data.size == sizeof(MoveBroadcastPrivateData));

    m_src = TensorIterator<InternalBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank>(private_data.src, membases, num_mems);
    m_dst = TensorIterator<InternalBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank>(private_data.dst, membases, num_mems);
    m_src.Reset();
    m_dst.Reset();
}

template <typename buf_T, unsigned N>
int32_t tensor_read(TensorIterator<buf_T, N, N> tsr, uint32_t *index) {
    int32_t result = 0;
    int32_t offset = tsr.get_tensor().get_offset(index);
    switch (tsr.get_tensor().get_elem_size()) {
        case sizeof(int8_t):
            result = tsr.get_tensor().template read<int8_t>(offset);
            break;
        case sizeof(int16_t):
            result = tsr.get_tensor().template read<int16_t>(offset);
            break;
        case sizeof(int32_t):
            result = tsr.get_tensor().template read<int32_t>(offset);
            break;
        default:
            MLI_ASSERT(false);
    }
    return result;
}

template <typename buf_T, unsigned N>
void tensor_write(TensorIterator<buf_T, N, N> tsr, uint32_t *index, int32_t value) {
    int32_t offset = tsr.get_tensor().get_offset(index);
    switch (tsr.get_tensor().get_elem_size()) {
        case sizeof(int8_t):
            tsr.get_tensor().template write<int8_t>(offset, value);
            break;
        case sizeof(int16_t):
            tsr.get_tensor().template write<int16_t>(offset, value);
            break;
        case sizeof(int32_t):
            tsr.get_tensor().template write<int32_t>(offset, value);
            break;
        default:
            MLI_ASSERT(false);
    }
}

// Move Broadcast Core Function
template <typename buf_T, unsigned N>
void MoveBroadcast::MoveBroadcastRun(TensorIterator<buf_T, N, N> src, TensorIterator<buf_T, N, N> dst) {
    uint32_t src_idx[N] = {0};
    uint32_t dst_idx[N] = {0};
    uint32_t src_shape[N] = {0};
    uint32_t dst_shape[N] = {0};
    uint32_t src_rank = src.get_tensor().get_rank();
    uint32_t dst_rank = dst.get_tensor().get_rank();

    MLI_ASSERT(src_rank == dst_rank);

    // get shapes
    src.get_full_shape(src_shape);
    dst.get_full_shape(dst_shape);

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
                    for (int d4_cnt = 0; d4_cnt < (int)dst_shape[4]; d4_cnt++) {
                        dst_idx[0] = d0_cnt;
                        dst_idx[1] = d1_cnt;
                        dst_idx[2] = d2_cnt;
                        dst_idx[3] = d3_cnt;
                        dst_idx[4] = d4_cnt;
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
                    }
                }
            }
        }
    }
}

mli_status MoveBroadcast::Issue() {
    MoveBroadcastRun<InternalBuffer, kMoveBroadcastRank>(m_src, m_dst);
    return MLI_STATUS_OK;
}

mli_status MoveBroadcast::Prefetch() {
    return MLI_STATUS_OK; 
}

mli_status MoveBroadcast::Update() {
    return MLI_STATUS_OK;
}

}  // namespace snps_arc::metaware::mli::ref