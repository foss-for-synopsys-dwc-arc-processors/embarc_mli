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
        MLI_ASSERT(0);
        break;
      case kPreluId:
        MLI_ASSERT(0);
        break;
      case kMoveId:
        MLI_ASSERT(0);
        break;
      case kDWConv2dId:
        MLI_ASSERT(0);
        break;
      case kMaxPool2DId:
        MLI_ASSERT(sizeof(MaxPool2D) == alloc_buf_size);
        obj = new (allocation_memory_buffer) MaxPool2D(kernel_private_data_buffer, private_data_size, membases, num_mems);
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
