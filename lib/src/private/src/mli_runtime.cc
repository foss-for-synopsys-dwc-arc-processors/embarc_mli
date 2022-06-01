/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <new>

#include "mli_ref_runtime_api.hpp"

#include "mli_debug.h"

namespace snps_arc::metaware::mli {
using ref::MaxPool2D;
using ref::Move;
using ref::Conv2d;
using ref::DepthwiseConv2d;

ExecutionInterface* ExecutionInterface::Create(
        void* allocation_memory_buffer,
        uint32_t alloc_buf_size,
        PrivateData* kernel_private_data_buffer,
        uint32_t private_data_size,
        uint64_t* membases,
        int num_mems) {

    MLI_ASSERT(private_data_size >= sizeof(PrivateData));
    [[maybe_unused]] kernel_id_t kernel_id = kernel_private_data_buffer->kernel_id;
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
                MLI_PRINTF("\nASSERT: Insufficient space for [Conv2d] runtime object\n");
            }
            break;
        case kPreluId:
            MLI_ASSERT(0);
            break;
        case kMoveId:
            if(alloc_buf_size >= sizeof(Move)) {
                obj = new (allocation_memory_buffer) Move(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nASSERT: Insufficient space for [Move] runtime object\n");
            }
            break;
        case kDWConv2dId:
            if(alloc_buf_size >= sizeof(DepthwiseConv2d)) {
                obj = new (allocation_memory_buffer) DepthwiseConv2d(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nASSERT: Insufficient space for [DepthwiseConv2d] runtime object\n");
            }
            break;
        case kMaxPool2DId:
            if(alloc_buf_size >= sizeof(MaxPool2D)) {
                obj = new (allocation_memory_buffer) MaxPool2D(kernel_private_data_buffer, private_data_size, membases, num_mems);
            } else {
                MLI_PRINTF("\nASSERT: Insufficient space for [MaxPool2D] runtime object\n");
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
