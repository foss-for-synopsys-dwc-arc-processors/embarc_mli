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
#include "mli_ref_runtime_api.hpp"
#include "mli_sync_interface.hpp"

namespace snps_arc::metaware::mli {
using ref::Nop;
using ref::MaxPool2D;
using ref::FullyConnected;
using ref::SumPool2D;
using ref::Add;
using ref::Sub;
using ref::Mul;
using ref::Max;
using ref::Min;
using ref::Move;
using ref::Conv2d;
using ref::DepthwiseConv2d;
using ref::Rescale;
using ref::Clip;
using ref::TransposeConv2D;
using ref::ReduceMax;
using ref::Permute;
using ref::ArgMax;
using ref::TableBuiltin;
using ref::MatMul;
using ref::ReduceSum;
using ref::Prelu;
using ref::MoveBroadcast;

ExecutionInterface* ExecutionInterface::Create(
        void* allocation_memory_buffer,
        uint32_t alloc_buf_size,
        void* kernel_private_data_buffer,
        uint32_t private_data_size,
        uint64_t* membases,
        int num_mems) {

    /*
     * The MLI classes need to be 32 bit aligned
     */
    assert(allocation_memory_buffer != nullptr);
    assert(((size_t) allocation_memory_buffer % kMliAlignment) == 0);
    MLI_ASSERT(private_data_size >= sizeof(PrivateData));
    PrivateData private_data;
    memcpy(&private_data, kernel_private_data_buffer, sizeof(PrivateData)); // only copy the base class in order to inspect the kernel_id
    MLI_ASSERT(private_data.size == private_data_size);
    kernel_id_t kernel_id = private_data.kernel_id;
    ExecutionInterface *obj = nullptr;
    //  TODO: Update it with MLI REF/EM/VPX Classes and remove [[maybe_unused]] attr of the kernel_id
    switch (kernel_id) {
        //  TODO: Update it with MLI REF/EM/VPX Classes
        case kInvalidId:
            MLI_ASSERT(0);
            break;
        case kNopId:
            if(alloc_buf_size >= sizeof(Nop)) {
                obj = new (allocation_memory_buffer) Nop(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [Nop] runtime object\n");
            }
            break;
        case kConv2dId:
            if(alloc_buf_size >= sizeof(Conv2d)) {
                obj = new (allocation_memory_buffer) Conv2d(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [Conv2d] runtime object\n");
            }
            break;
        case kFullyConnectedId:
            if(alloc_buf_size >= sizeof(FullyConnected)) {
                obj = new (allocation_memory_buffer) FullyConnected(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [FullyConnected] runtime object\n");
            }
            break;
        case kPreluId:
            if(alloc_buf_size >= sizeof(Prelu)) {
                obj = new (allocation_memory_buffer) Prelu(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [Prelu] runtime object\n");
            }
            break;
        case kAddId:
            if(alloc_buf_size >= sizeof(Add)) {
                obj = new (allocation_memory_buffer) Add(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [Add] runtime object\n");
            }
            break;
        case kSubId:
            if(alloc_buf_size >= sizeof(Sub)) {
                obj = new (allocation_memory_buffer) Sub(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [Sub] runtime object\n");
            }
            break;
        case kMulId:
            if(alloc_buf_size >= sizeof(Mul)) {
                obj = new (allocation_memory_buffer) Mul(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [Mul] runtime object\n");
            }
            break;
        case kMaxId:
            if(alloc_buf_size >= sizeof(Max)) {
                obj = new (allocation_memory_buffer) Max(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [Max] runtime object\n");
            }
            break;
        case kMinId:
            if(alloc_buf_size >= sizeof(Min)) {
                obj = new (allocation_memory_buffer) Min(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [Min] runtime object\n");
            }
            break;
        case kMoveId:
            if(alloc_buf_size >= sizeof(Move)) {
                obj = new (allocation_memory_buffer) Move(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [Move] runtime object\n");
            }
            break;
        case kDWConv2dId:
            if(alloc_buf_size >= sizeof(DepthwiseConv2d)) {
                obj = new (allocation_memory_buffer) DepthwiseConv2d(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [DepthwiseConv2d] runtime object\n");
            }
            break;
        case kMaxPool2DId:
            if(alloc_buf_size >= sizeof(MaxPool2D)) {
                obj = new (allocation_memory_buffer) MaxPool2D(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [MaxPool2D] runtime object\n");
            }
            break;
        case kSumPool2DId:
            if(alloc_buf_size >= sizeof(SumPool2D)) {
                obj = new (allocation_memory_buffer) SumPool2D(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [SumPool2D] runtime object\n");
            }
            break;
        case kRescaleId:
            if(alloc_buf_size >= sizeof(Rescale)) {
                obj = new (allocation_memory_buffer) Rescale(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [Rescale] runtime object\n");
            }
            break;
        case kClipId:
            if(alloc_buf_size >= sizeof(Clip)) {
                obj = new (allocation_memory_buffer) Clip(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [Clip] runtime object\n");
            }
            break;
        case kReduceMaxId:
            if(alloc_buf_size >= sizeof(ReduceMax)) {
                obj = new (allocation_memory_buffer) ReduceMax(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [ReduceMax] runtime object\n");
            }
            break;
        case kTransConv2DId:
            if(alloc_buf_size >= sizeof(TransposeConv2D)) {
                obj = new (allocation_memory_buffer) TransposeConv2D(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [TransposeConv2D] runtime object\n");
            }
            break;
        case kReduceSumId:
            if(alloc_buf_size >= sizeof(ReduceSum)) {
                obj = new (allocation_memory_buffer) ReduceSum(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [ReduceSum] runtime object\n");
            }
            break;
        case kPermuteId:
            if(alloc_buf_size >= sizeof(Permute)) {
                obj = new (allocation_memory_buffer) Permute(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [Permute] runtime object\n");
            }
            break;
        case kMatMulId:
            if(alloc_buf_size >= sizeof(MatMul)) {
              obj = new (allocation_memory_buffer) MatMul(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [MatMul] runtime object\n");
            }
            break;
        case kMoveBroadcastId:
            if(alloc_buf_size >= sizeof(MoveBroadcast)) {
                obj = new (allocation_memory_buffer) MoveBroadcast(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nMLI_ERROR: Insufficient space for [MoveBroadcast] runtime object\n");
            }
            break;
        default:
            MLI_ASSERT(0);
            break;
    }

    return obj;
}

mli_status SynchronizationInterface::WaitEvent(int32_t mask) {
    return MLI_STATUS_OK;
}

} // namespace mli
