/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_perf_estim.hpp"

namespace snps_arc::metaware::mli {
    
PerfEstimator* PerfEstimator::Create(void* allocation_memory_buffer,
        lib_mli::PlatformDescription& desc, 
        lib_mli::ExecutionInterface& rt_kernel, 
        int num_tiles){
       kernel_id_t kernel_id = rt_kernel.GetKernelId();
    PerfEstimator *obj = nullptr;
    switch (kernel_id) {
      /*case kConv2dId:
          obj = new(allocation_memory_buffer) Conv2dPerfEstimator(desc, rt_kernel, num_tiles);
          break;
      case kMoveId:
          obj = new(allocation_memory_buffer) MovePerfEstimator(desc, rt_kernel, num_tiles);
          break;
      case kPreluId:
          obj = new(allocation_memory_buffer) PreluPerfEstimator(desc, rt_kernel, num_tiles);
          break;*/
    }

    return obj;
}
int PerfEstimator::KernelPerf_GetSize(lib_mli::ExecutionInterface& rt_kernel){
    uint32_t perf_kernel_size = 0;
    kernel_id_t kernel_id = rt_kernel.GetKernelId();
    switch (kernel_id) {
      /*case kConv2dId:
          perf_kernel_size = sizeof(Conv2dPerfEstimator);
          break;
      case kMoveId:
          perf_kernel_size = sizeof(MovePerfEstimator);
          break;
      case kPreluId:
          perf_kernel_size = sizeof(PreluPerfEstimator);
          break;
    }*/
    return perf_kernel_size;
}

}