/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_USER_TESTS_TEST_MEMORY_MANAGER_H_
#define _MLI_USER_TESTS_TEST_MEMORY_MANAGER_H_

#include "mli_api.h"
#include "mli_config.h"
#include "test_crc32_calc.h"
#include "test_tensor_quantizer.h"

// Attributes for data arrays allocation. 
// const W_DATA_ATTR int arr[] will be placed in a HW specific memory according to MLI requirements
// W_DATA_ATTR for weights and IO_DATA_ATTR for activations (feature maps).
// Allocation works only with CCAC copiler. For others #else branch will be used
#if (PLATFORM == V2DSP_XY)
#define W_DATA_ATTR __xy __attribute__((section(".Xdata")))
#define IO_DATA_ATTR __xy __attribute__((section(".Ydata")))

#elif (PLATFORM == V2DSP_VECTOR)
#define W_DATA_ATTR __vccm __attribute__((section(".vecmem_data")))
#define IO_DATA_ATTR __vccm __attribute__((section(".vecmem_data")))

#else
#define W_DATA_ATTR 
#define IO_DATA_ATTR 
#endif

namespace mli {
namespace tst {
//=======================================================================
// Module to handle and check externally allocated memory for test needs
//
// This memory_manager intend to handle some memory region for test needs. 
// - It returns container with requested memory size 
// - It can return the only one container. return_memory() method should 
//   be called to mark memory as unused and able to be allocated again.
// - it initialize all the memory with some pre-defined pattern
// - It checks that memory beside the allocated region is not corrupted
//=======================================================================
class memory_manager {
public:
    // Parametrized constructors to initialize memory_manager.
    memory_manager(int8_t* memory, uint32_t mem_size);

    // Methods to get (allocate) memory from internally handled area by memory_manager.
    // The whole memory is initialized with fill_pattern. if requested size less than internal memory,
    // returns container that points to the middle of the region, e.g.:
    // /*                   All handled memory region                           */
    // /*------head_region----*------memory_to_return------*-----tail_region----*/
    // CRC32 sum for head and tail regions are kept to check that it wasn't modifyed after a while
    //
    // If requested memory is bigger, then returns empty container (nullptr and 0 capacity)
    
    // Method to allocate memory of exact size
    // params:
    // [IN] size - uint32_t requested size of memory
    // [IN] fill_pattern - optional uint32_t value to initialize all memory with
    mli_data_container allocate_memory(uint32_t size, uint32_t fill_pattern = 0xDEADBEEF);
    // Method to allocate memory according to tensor_quantizer requirements
    // params:
    // [IN] quant_unit - valid tensor_quantizer Which will be analyzed to return memory container 
    //                   to exactly fit it's requirements
    // [IN] fill_pattern - optional uint32_t value to initialize all memory with
    mli_data_container allocate_memory(const tensor_quantizer& quant_unit, uint32_t fill_pattern = 0xBEADED37);
    
    // Reset module to mark memory as unused and get an opportunity to allocate again
    // No return
    void return_memory();

    // Check that head and tail regions are not corrupted using pre-calculated CRC32 checksums. 
    // Return false (not corrupted) if regions aren't corrupted OR memory wasn't properly allocated before OR 
    // memory manager don't handle a valid memory region
    bool is_memory_corrupted() const;

private:
    int8_t* source_memory_;
    int8_t* allocated_memory_start_;
    uint32_t source_mem_size_;
    uint32_t allocated_mem_size_;
    crc32_calc head_mem_crc_;
    crc32_calc tail_mem_crc_;
};

} // namespace tst 
} // namespace mli

#endif //_MLI_USER_TESTS_TEST_MEMORY_MANAGER_H_

