/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_debug.h"
#include "mli_runtime_api.hpp"
#include "mli_runtime_kernels.hpp"


namespace snps_arc::metaware::mli {

ExecutionInterface* ExecutionInterface::Create(
        void* allocation_memory_buffer,
        uint32_t alloc_buf_size,
        PrivateData* kernel_private_data_buffer,
        uint32_t private_data_size,
        uint64_t* membases,
        int num_mems) {

    MLI_ASSERT(private_data_size >= sizeof(PrivateData));
    kernel_id_t kernel_id = kernel_private_data_buffer->kernel_id;
    ExecutionInterface *obj = nullptr;
    switch (kernel_id) {
        //  TODO: Update it with MLI REF/EM/VPX Classes
        /*
        case kConv2dId:
            MLI_ASSERT(sizeof(Conv2d) == alloc_buf_size);
            obj = new(allocation_memory_buffer) Conv2d(kernel_private_data_buffer, private_data_size, membases, num_mems);
            break;
        case kMoveId:
            MLI_ASSERT(sizeof(Move) == alloc_buf_size);
            obj = new(allocation_memory_buffer) Move(kernel_private_data_buffer, private_data_size, membases, num_mems);
            break;
        case kPreluId:
            MLI_ASSERT(sizeof(Prelu) == alloc_buf_size);
            obj = new(allocation_memory_buffer) Prelu(kernel_private_data_buffer, private_data_size, membases, num_mems);
            break;
        */
	default:
		assert(0);
    }

    return obj;
}

} // namespace mli
