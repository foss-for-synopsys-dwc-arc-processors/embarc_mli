/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_COMPILER_API_HPP_
#define _MLI_COMPILER_API_HPP_

#include "mli_debug.h"
#include "mli_iterator.hpp"
#include "mli_types.h"
#include "mli_types.hpp"

namespace snps_arc::metaware::mli {

/**
 * @brief This is the base class for compiler side of MLI kernels
 *
 * This base class contains the virtual functions that are required for all the mli kernels
 * This allows the compiler to traverse over all kernels in the graph, and call these methods
 */

class CompilerGenericInterface {

  public:
    /**
     * @brief Method to query the size of the kernel private data
     *
     * This function returns the size of the buffer that is needed for the private configuration data for this kernel
     */
    virtual unsigned GetKernelPrivateDataSize() = 0;

    /**
     * @brief Method to fill the allocated memory with the private configuration data
     *
     * All information that is computed compile time and is needed during run-time
     * needs to be transfered to the init function using this 'box of bits' that is
     * being filled by this function
     */
    virtual mli_status GetKernelPrivateData(void* kernel_private_data_buffer) = 0;

    /**
     * @brief Method to get the size of the runtime object
     *
     * This method is used query the size of the run-time object
     * The compiler needs this to create a memory map for the various run-time mli objects
     *
     */
    virtual unsigned GetRuntimeObjectSize() = 0;

// TODO add virtual destructor
};

/**
 * @brief This class implements the Conv2d kernel Compiler Support interface
 *
 *
 */
class Conv2d_CS : public CompilerGenericInterface {
public:
    /**
     * @brief Method to encode the weights (coefficients)
     *
     * This method will read the weights buffer in a platform independend layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * This transformation may include compression
     * The content of the encode_weights buffer is opaque for the compiler.
     *
     * @param weights [I] tensor with the weights
     * @param buffer_t[I] buffer pointer where the encode function can write the encoded weights
     *
     * TODO: how to handle sliding in the output channel dimension? is this weights encoding for the complete 'thing' or just for this slide?
     */
    virtual mli_status EncodeWeights(Tensor<Buffer, 5> weights, Buffer encoded_weights, compression_mode_t mode = compression_mode_t::Uncompressed) = 0;

    /**
     * @brief Method to query the size of the encoded weights buffer
     *
     * This function returns the size of the buffer that is needed by the EncodeWeights method
     */
    virtual unsigned GetEncodedWeightsSize() = 0;

    /**
     * @brief Method to encode input zero-points (padding values)
     *
     * This method will read the input zero-points buffer in a platform independend layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * The content of the encode_inpzerpts buffer is opaque for the compiler.
     *
     */
    virtual mli_status EncodeInpZeroPts(Tensor<Buffer, 1> inpzeropts, Buffer encoded_inpzeropts) = 0;

    /**
     * @brief Method to query the size of the encoded input zero-points buffer
     *
     * This function returns the size of the buffer that is needed by the EncodeInpZeroPts method
     */
    virtual unsigned GetEncodedInpZeroPtsSize() = 0;

    /**
     * @brief Methods to get buffer sizes
     *
     * Tensor buffer sizes could depend on the platform and/or parameters
     * This / these functions can be used to query how much memory needs to be allocated for
     * the input and output tensors
     *
     * TODO: should we have a function per buffer, or one function that returns an array of sizes for all buffers?
     */

    virtual unsigned GetInputBufferSize() = 0;
    virtual unsigned GetOutputBufferSize() = 0;
    virtual unsigned GetWeightsBufferSize() = 0;
    virtual unsigned GetPaddingBufferSize() = 0;

    /**
     * @brief Methods to set buffer offsets
     *
     * Compiler computes a memory map and buffer offsets are set using this method.
     * Compiler also needs to indicate in which memory the buffers reside.
     * These id's need to match the array of memory bases that the xop-interpreter passes to
     * the init function.
     *
     * Note that the weights buffer offset in this function is in local memory, where it will be copied by a dma task
     * the weights buffer passed to the encode_weigths function is in compiler memoryspace because the
     * encode function will write the encoded weights data there.
     */
    virtual mli_status AttachBufferOffsets(OffsetBuffer input,
                                   OffsetBuffer output,
                                   OffsetBuffer weights,
                                   OffsetBuffer padding,
                                   OffsetBuffer descr) = 0;

    // mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override ;
    // unsigned GetKernelPrivateDataSize() override ;
    // unsigned GetRuntimeObjectSize() override ;

};

/**
 * @brief This class implements the Prelu kernel Compiler Support interface
 *
 *
 */
class Prelu_CS : public CompilerGenericInterface {
public:

    /**
     * @brief Method to encode the weights (coefficients)
     *
     * This method will read the weights buffer in a platform independend layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * This transformation may include compression
     * The content of the encode_weights buffer is opaque for the compiler.
     *
     * @param weights [I] tensor with the weights
     * @param buffer_t[I] buffer pointer where the encode function can write the encoded weights
     *
     * TODO: how to handle sliding in the output channel dimension? is this weights encoding for the complete 'thing' or just for this slide?
     */
    virtual mli_status EncodeParams(Tensor<Buffer, 2> bias,
                            Tensor<Buffer, 2> posscale,
                            Tensor<Buffer, 2> negscale,
                            Tensor<Buffer, 2> posshift,
                            Tensor<Buffer, 2> negshift,
                            Tensor<Buffer, 2> asymm,
                            Buffer encoded_params) = 0;

    /**
     * @brief Method to query the size of the encoded weights buffer
     *
     * This function returns the size of the buffer that is needed by the encodeweights method
     */
    virtual unsigned GetEncodedParamsSize() = 0;

    /**
     * @brief Methods to get buffer sizes
     *
     * Tensor buffer sizes could depend on the platform and/or parameters
     * This / these functions can be used to query how much memory needs to be allocated for
     * the input and output tensors
     *
     * TODO: should we have a function per buffer, or one function that returns an array of sizes for all buffers?
     */

    virtual unsigned GetInputBufferSize() = 0;
    virtual unsigned GetOutputBufferSize() = 0;
    virtual unsigned GetParamsBufferSize() = 0;

    /**
     * @brief Methods to set buffer offsets
     *
     * Compiler computes a memory map and buffer offsets are set using this method.
     * Compiler also needs to indicate in which memory the buffers reside.
     * These id's need to match the array of memory bases that the xop-interpreter passes to
     * the init function.
     *
     * Note that the weights buffer offset in this function is in local memory, where it will be copied by a dma task
     * the weights buffer passed to the encode_weigths function is in compiler memoryspace because the
     * encode function will write the encoded weights data there.
     */
    virtual mli_status AttachBufferOffsets(OffsetBuffer input,
                                   OffsetBuffer output,
                                   OffsetBuffer params,
                                   OffsetBuffer descr) = 0;

    // mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override ;
    // unsigned GetKernelPrivateDataSize() override ;
    // unsigned GetRuntimeObjectSize() override ;
};

} // namespace mli

#endif // _MLI_COMPILER_API_HPP_
