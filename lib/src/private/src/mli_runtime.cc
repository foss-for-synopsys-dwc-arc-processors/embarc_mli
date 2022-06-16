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

namespace snps_arc::metaware::mli {
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

ExecutionInterface* ExecutionInterface::Create(
        void* allocation_memory_buffer,
        uint32_t alloc_buf_size,
        void* kernel_private_data_buffer,
        uint32_t private_data_size,
        uint64_t* membases,
        int num_mems) {

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
            MLI_ASSERT(0);
            break;
        case kAddId:
            if(alloc_buf_size >= sizeof(Add)) {
                obj = new (allocation_memory_buffer) Add(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nASSERT: Insufficient space for [Add] runtime object\n");
            }
            break;
        case kSubId:
            if(alloc_buf_size >= sizeof(Sub)) {
                obj = new (allocation_memory_buffer) Sub(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nASSERT: Insufficient space for [Sub] runtime object\n");
            }
            break;
        case kMulId:
            if(alloc_buf_size >= sizeof(Mul)) {
                obj = new (allocation_memory_buffer) Mul(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nASSERT: Insufficient space for [Mul] runtime object\n");
            }
            break;
        case kMaxId:
            if(alloc_buf_size >= sizeof(Max)) {
                obj = new (allocation_memory_buffer) Max(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nASSERT: Insufficient space for [Max] runtime object\n");
            }
            break;
        case kMinId:
            if(alloc_buf_size >= sizeof(Min)) {
                obj = new (allocation_memory_buffer) Min(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nASSERT: Insufficient space for [Min] runtime object\n");
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
        case kSomeOtherKernelId:
            MLI_ASSERT(0);
            break;
        default:
            MLI_ASSERT(0);
            break;
    }

    return obj;
}

} // namespace mli
