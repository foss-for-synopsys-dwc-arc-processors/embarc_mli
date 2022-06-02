/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_RUNTIME_API_HPP_
#define _MLI_RUNTIME_API_HPP_

#include "mli_types.h"
#include "mli_types.hpp"

namespace snps_arc::metaware::mli {

/**
 * @brief This is the base class for Execution side of MLI kernels
 *
 *
 */

class ExecutionInterface;
class ExecutionInterface {

  public:
    /**
     * @brief Method to create a ML-ISA operation
     *
     * This method can be used to create a ML-ISA run-time operation object, it will be initialized
     * using the information stored in the kernel_private_data_buffer  that has been computed at compile time
     * by the get_kernel_private_data() method.
     * The object is created in the memory pointed to by the 'allocation_memory_buffer' argument
     *
     * @param allocation_memory_buffer [I] memory buffer where the object should be created.
     * @param alloc_buf_size [I] Size of the above memory buffer.
     *
     * @param kernel_private_data_buffer [I] pointer to the compiletime computed initialization data
     * @param private_data_size [I] Size of the data is used to check for coding errors
     * @param membases[]  [I] The kernel private data may contain offsets inside a (vector) memory.
     *                        At run-time specific locations in memory are allocated for
     *                        the graph, the membase array contains the is the start of
     *                        each memory region.
     *                        The init method will add this base to all the memory offsets
     *                        inside the descriptor according to the memory number associated
     *                        with that offset.
     *                        Each platform can have different (number of) memories. For mli
     *                        this is completely transparant. Compiler needs to use the same
     *                        memory id's when attaching the buffers as are used by the
     *                        xop-interpreter to set the membases.
     * @param num_mems    [I] Number of elements in the membases array.
     *
     * @return This function return a pointer to a ML-ISA run-time object
     */
    static ExecutionInterface* Create(void* allocation_memory_buffer,
                                     uint32_t alloc_buf_size,
                                     void* kernel_private_data_buffer,
                                     uint32_t private_data_size,
                                     uint64_t* membases, int num_mems);

    kernel_id_t GetKernelId();                                 

    /**
     * @brief Method to issue a ML-ISA operation
     *
     * In case of a HW accelerator this method will trigger the HW to start its compute
     * and directly return.
     * In case of SW implementation of this kernel, this method will perform the compute
     * and return after the job is completed.
     * 
     * If predication is needed, it should be handled at the caller
     */
    virtual mli_status Issue() = 0;

    /**
     * @brief Method to prefetch the next ML-ISA operation
     *
     * In case of a HW accelerator this method can be used to load the descriptor into
     * the HW in order to make it available for the SW to update it for the next operation
     * In case of a SW implementation of the kernel, this method will be empty
     */
    virtual mli_status Prefetch() = 0;

    /**
     * @brief Method to update the internal data structures for the next ML-ISA operation
     *
     * This method will increment the iterator(s) and update the internal data structures
     * (like the descriptor in case of HW acceleration) to the next operation
     */
    virtual mli_status Update() = 0;
private:
    kernel_id_t m_kernel_id;
};

} // namespace mli

#endif // _MLI_RUNTIME_API_HPP_
