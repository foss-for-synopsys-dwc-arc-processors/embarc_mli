/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_PERF_ESTIM_HPP_
#define _MLI_PERF_ESTIM_HPP_

#include "mli_platform_desc.hpp"
#include "mli_runtime_api.hpp"

namespace lib_mli = ::snps_arc::metaware::mli;

namespace snps_arc::metaware::mli {

/**
 * @brief This is the base class of Performance estimation for kernels
 *
 * This class defines a set of functions which can be used to query platform capabilities
 * like execution time per kernel, Cycles and bytes for input tensor, etc.
 */

class PerfEstimator;
class PerfEstimator {
  public:
    /**
     * @brief Method to create a kernel perfEstimator object
     *
     * This method can be used to create a kernel perfEstimator object based on the kernel_id 
     * extracted from rt_kernel object
     * It will be initialized using the platform description object and number of tiles
     * The object is created in the memory pointed to by the 'allocation_memory_buffer' argument
     *
     * @param allocation_memory_buffer [I] memory buffer where the object should be created.
     * @param alloc_buf_size [I] Size of the above memory buffer.
     * @param pd [I] platform description class object.
     *
     * @param rt_kernel [I] run time kernel object.
     * @param num_tiles [I] Number of tiles.
     *
     * @return This function return a pointer to a kernel perfEstimator object.
     */
    static PerfEstimator* Create(void* allocation_memory_buffer,
        uint32_t alloc_buf_size,
        lib_mli::PlatformDescription& pd, 
        lib_mli::ExecutionInterface& rt_kernel, 
        int num_tiles);

    /**
     * @brief Method to get the size of kernel perfEstimator class
     *
     * This method can be used to get the size of kernel perfEstimator class 
     * based on kernel_id parameter stored in the private data of the run time kernel
     * to then allocate memory with the correct size 
     * and pass it as 'allocation_memory_buffer' in the Create function.
     *
     * @param rt_kernel [I] run time kernel object..
     *
     * @return This function returns an int which is the class size of the intended kernel.
     */    
    static int KernelPerf_GetSize(lib_mli::ExecutionInterface& rt_kernel);

    /**
     * @brief Method to return the cost of the function in terms of cycles to execute for all tiles
     *
     * This method will make a dry run over tiles and calculate the execution cycles for each tile
     * and accumulate them.
     *
     */     
    virtual int GetTotalCycles() = 0;

    /**
     * @brief Method to return the cost of the function in terms of cycles to execute for a specific tile
     *
     * This method will calculate the execution cycles for a specific tile passed as a parameter.
     * 
     * @param tile_idx [I] tile index
     * 
     */ 
    virtual int GetTileCycles(int tile_idx) = 0;

    /**
     * @brief Method to return the Bytes used for input tensors(reading) for all tiles
     *
     * This method will make a dry run over tiles and calculate the Bytes used for 
     * input tensors reading
     * This is done for each tile then accumulate them.
     *
     */
    virtual int GetTotalReadBytes() = 0;

    /**
     * @brief Method to return the Bytes used for output tensors(writing) for all tiles
     *
     * This method will make a dry run over tiles and calculate the Bytes used for 
     * writing output tensors
     * This is done for each tile then accumulate them.
     *
     */
    virtual int GetTotalWriteBytes() = 0;

    /**
     * @brief Method to return the Bytes used for input tensors(reading) for a specific tile
     *
     * This method will calculate the Bytes used for reading the input tensors
     * This is done for a specific tile passed as a parameter.
     *
     * @param tile_idx [I] tile index
     *
     */
    virtual int GetTileReadBytes(int tile_idx) = 0;

    /**
     * @brief Method to return the Bytes used for output tensors(writing) for a specific tile
     *
     * This method will calculate the Bytes used to write output tensors
     * this is done for a specific tile passed as a parameter.
     *
     * @param tile_idx [I] tile index
     */
    virtual int GetTileWriteBytes(int tile_idx) = 0;

    /**
     * @brief Method to return the MAC operations used for all tiles
     *
     * This method will make a dry run over tiles and calculate MAC operations 
     * used for each tile and accumulate them.
     *
     */
    virtual int GetTotalMacs(){
      return 0;
    }

    /**
     * @brief Method to return the MAC operations used for specific tile
     *
     * This method will calculate MAC operations used for a specific tile passed as a parameter.
     *
     * @param tile_idx [I] tile index
     */
    virtual int GetTileMacs(int tile_idx){
      return 0;
    }
    PerfEstimator(lib_mli::PlatformDescription& pd,  
                  int num_tiles) : m_pd(pd), m_num_tiles(num_tiles){}
protected:
    lib_mli::PlatformDescription m_pd;
    int m_num_tiles;  
};
}
#endif