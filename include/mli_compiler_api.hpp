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

#include <cstring>

#include "mli_debug.h"
#include "mli_iterator.hpp"
#include "mli_types.h"
#include "mli_types.hpp"
#include "mli_service_functions.hpp"

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
    virtual unsigned GetKernelPrivateDataSize() const = 0;

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
    virtual unsigned GetRuntimeObjectSize() const = 0;

    /**
     * @brief Method to get the platform-specific descriptor ctrl buffer size
     *
     * CtrlBuffer requires allocation in closely coupled data memory (CCM) on
     * some platforms
     *
     * @return Size of platform-specific descriptor ctrl buffer in bytes
     */

    virtual unsigned GetCtrlBufferSize() const { return 0; }
    
    virtual int32_t GetEventPrefetchMask() const { return 0; }

    virtual int32_t GetEventIssueMask() const { return 0; }

    mli_status SetEventPrefetch(bool enable) { 
        m_prefetch_enable = enable;
        return MLI_STATUS_OK;
    }

    mli_status SetEventIssue(bool enable) { 
        m_issue_enable = enable;
        return MLI_STATUS_OK;
    }

    /**
     * @brief this function will return the vectorization in the input channel
     *        dimension that is used by the platform.
     */
    virtual unsigned GetInputChannelMultiple() { return 1; };

    /**
     * @brief this function will return the vectorization in the output channel
     *        dimension that is used by the platform.
     */
    virtual unsigned GetOutputChannelMultiple() { return 1; };

// TODO add virtual destructor
protected:
    bool m_issue_enable{false};
    bool m_prefetch_enable{false};
};

/**
 * @brief This class provides an interface for the no-op kernel Compiler Support
 */
class Nop_CS : public CompilerGenericInterface {
public:
  Nop_CS() {}
  unsigned GetKernelPrivateDataSize() const override {
      return sizeof(PrivateData);
  }
  mli_status GetKernelPrivateData(void *kernel_private_data_buffer) override {
      PrivateData data = PrivateData(kNopId, sizeof(PrivateData));
      std::memcpy(
          kernel_private_data_buffer,
          (void *)&data,
          sizeof(data)
      );
      return MLI_STATUS_OK;
  }
  unsigned GetRuntimeObjectSize() const override {return 0;}
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
     * This method will read the weights buffer in a platform independent layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * This transformation may include compression
     * The content of the encode_weights buffer is opaque for the user.
     *
     * @param weights [I] tensor with the weights
     * @param buffer_t[I] buffer pointer where the encode function can write the encoded weights
     *
     * TODO: how to handle sliding in the output channel dimension? is this weights encoding for the complete 'thing' or just for this slide?
     */
    virtual mli_status EncodeWeights(Tensor<Buffer, kConvWRank>& weights,
                                     Buffer& encoded_weights,
                                     compression_mode_t mode = compression_mode_t::Uncompressed) = 0;

    /**
     * @brief Method to encode the weights (coefficients) with weights zero points
     *
     * This method will read the weights and weights zp buffers in a platform independent layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * This transformation may include compression
     * The content of the encode_weights buffer is opaque for the user.
     *
     * @param weights          [I] TensorIterator with the weights
     * @param weights_zp       [I] TensorIterator with the weights_zp
     * @param encoded_weights  [I] buffer pointer where the encode function can write the encoded weights/weights_zp
     */
    virtual mli_status EncodeWeightsAndZeroPts(TensorIterator<Buffer, kConvWRank,  kConvIterRank> &weights,
                                               TensorIterator<Buffer, kConvZPRank, kConvIterRank> &weights_zp,
                                               Buffer &encoded_weights)
                                               { NOT_IMPLEMENTED_METHOD;
                                                 return MLI_STATUS_OK; };

    /**
     * @brief Method to query the size of the encoded weights buffer
     *
     * This function returns the size of the full weights buffer in bytes that
     * is needed by the EncodeWeights method EncodeWeights method
     */
    virtual unsigned GetEncodedWeightsSize() = 0;

    /**
     * @brief Method to encode input zero-points (padding values)
     *
     * This method will read the input zero-points buffer in a platform independent layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * The content of the encoded_inpzeropts buffer is opaque for the user.
     *
     */
    virtual mli_status EncodeInpZeroPts(Tensor<Buffer, kInpZPRank>& inpzeropts, Buffer& encoded_inpzeropts) = 0;

    /**
     * @brief Method to encode input zero-points (padding values)
     *
     * This method will read the input zero-points buffer in a platform independent layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * The content of the encoded_inpzeropts buffer is opaque for the user.
     *
     * @param input_zp          [I] Input_zp TensorIterator
     * @param encoded_input_zp  [I] buffer pointer where the encode function can write the encoded input zero points
     * 
     */
    virtual mli_status EncodeInpZeroPts(TensorIterator<Buffer, kConvZPRank, kConvZPIterRank> &input_zp,
                                        Buffer& encoded_input_zp)
                                        { NOT_IMPLEMENTED_METHOD;
                                          return MLI_STATUS_OK; };
    /**
     * @brief Method to query the size of the encoded input zero-points buffer
     *
     * This function returns the size of the buffer that is needed by the EncodeInpZeroPts method
     */
    virtual unsigned GetEncodedInpZeroPtsSize() = 0;

    /**
     * @brief Method to encode weights zero-points (padding values)
     *
     * This method will read the weights zero-points buffer in a platform independent layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * The content of the encode_wtszeropts buffer is opaque for the user.
     *
     */
    virtual mli_status EncodeWtsZeroPts(Tensor<Buffer, kConvZPRank>& wtszeropts, Buffer& encoded_wtszeropts) = 0;

    /**
     * @brief Method to query the size of the encoded input zero-points buffer
     *
     * This function returns the size of the buffer that is needed by the EncodeWtsZeroPts method
     */
    virtual unsigned GetEncodedWtsZeroPtsSize() = 0;

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
    virtual unsigned GetZeroPointBufferSize() = 0;
    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     *
     * Compiler computes a memory map and buffer offsets are set using this method.
     * Compiler also needs to indicate in which memory the buffers reside.
     * These ID's need to match the array of memory bases that the xop-interpreter passes to
     * the init function.
     *
     * In this method you specify offsets for tensors passed to the constructor
     *
     * @deprecated
     * @param input [I] Tensor descriptor containing input OffsetBuffer and tensor shape and memory strides
     * @param output [I] Tensor descriptor containing output OffsetBuffer and tensor shape and memory strides
     * @param weights [I] Tensor descriptor containing weights OffsetBuffer and tensor shape and memory strides
     * @param inpzeropts [I] Tensor descriptor containing input zero point(s) OffsetBuffer
     * @param wtszeropts [I] Tensor descriptor containing weights zero points OffsetBuffer
     * @param ctrl_buffer [I] Tensor descriptor containing descriptor data OffsetBuffer
     *
     * @return MLI status code
     */
    virtual mli_status AttachBufferOffsets(Tensor<OffsetBuffer, 4> &input,
                                           Tensor<OffsetBuffer, 4> &output,
                                           OffsetBuffer &weights,
                                           OffsetBuffer &inpzeropts,
                                           OffsetBuffer &wtszeropts,
                                           OffsetBuffer &ctrl_buffer) = 0;

    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     *
     * Compiler computes a memory map and buffer offsets are set using this method.
     * Compiler also needs to indicate in which memory the buffers reside.
     * These ID's need to match the array of memory bases that the xop-interpreter passes to
     * the init function.
     *
     * In this method you specify offsets for tensors passed to the constructor
     *
     * @param input       [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param output      [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param weights     [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param inpzeropts  [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param wtszeropts  [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param ctrl_buffer [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     *
     * @return MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer& input,
                                           const OffsetBuffer& output,
                                           const OffsetBuffer& weights,
                                           const OffsetBuffer& inpzeropts,
                                           const OffsetBuffer& wtszeropts,
                                           const OffsetBuffer& ctrl_buffer) = 0;

    // mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override ;
    // unsigned GetKernelPrivateDataSize() override ;
    // unsigned GetRuntimeObjectSize() override ;

    /**
     * @brief Method to set iteration information used in the .Update()
     *
     * NOTE: the use of this method is optional. if there is a single tile, and the .Update() is not used,
     *       this data doesn't need to be set.     
     * All the increments are following the output tile iterator.
     * 
     * @deprecated
     * @param output_total_size[4] [I] total size in each dimension
     * @param iteration_order[4] [I] which dimension of the output to iterate first.
     * @param input_first_inc[4] [I] increment of the input buffer pointer for the first iteration in each dimension
     * @param input_inc[4] [I] increment of the input buffer pointer for the other iterations in each dimension
     * @param output_first_inc[4] [I] increment of the output buffer pointer for the first iteration in each dimension
     * @param output_inc[4] [I] increment of the output buffer pointer for the other iterations in each dimension
     * @param weights_inc[4] [I] increment of the weights buffer pointer for the other iterations in each dimension of the output iterator
     */
    virtual mli_status SetIterators(uint32_t output_total_size[4],
                                    uint32_t iteration_order[4],
                                    uint32_t input_first_inc[4],
                                    uint32_t input_inc[4],
                                    uint32_t output_first_inc[4],
                                    uint32_t output_inc[4],
                                    uint32_t weights_inc[4]) = 0;

};

/**
 * @brief This class implements the Prelu kernel Compiler Support interface
 *
 *
 */
class Prelu_CS : public CompilerGenericInterface {
public:

    /**
     * @brief Method to encode the parameters
     *
     * This method will read the different parameters buffer in a platform independent layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * This transformation may include compression
     * The content of the encode_params buffer is opaque for the user.
     *
     * @param bias [I] tensor with the input bias
     * @param posscale[I] tensor with the positive scale
     * @param negscale[I] tensor with the negative scale
     * @param posshift[I] tensor with the positive shift
     * @param negshift[I] tensor with the negative shift
     * @param asymm[I] tensor with the output bias
     * @param encoded_params[I] encoded parameters buffer
     */
    virtual mli_status EncodeParams(Tensor<Buffer, kPreluParamRank> &bias,
                                    Tensor<Buffer, kPreluParamRank> &posscale,
                                    Tensor<Buffer, kPreluParamRank> &negscale,
                                    Tensor<Buffer, kPreluParamRank> &posshift,
                                    Tensor<Buffer, kPreluParamRank> &negshift,
                                    Tensor<Buffer, kPreluParamRank> &asymm,
                                    Buffer &encoded_params) = 0;

    /**
     * @brief Method to query the size of the encoded parameters buffer
     *
     * This function returns the size of the buffer that is needed by the EncodeParams method
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
    /**
     * @deprecated
     */
    virtual unsigned GetInputBufferSize() { return 0; }
    virtual unsigned GetOutputBufferSize() { return 0; }
    virtual unsigned GetParamsBufferSize() = 0;

    /**
     * @brief Methods to set buffer offsets
     * @deprecated
     *
     * Compiler computes a memory map and buffer offsets are set using this method.
     * Compiler also needs to indicate in which memory the buffers reside.
     * These id's need to match the array of memory bases that the xop-interpreter passes to
     * the init function.
     *
     * Note that the weights buffer offset in this function is in local memory, where it will be copied by a dma task
     * the weights buffer passed to the encode_weights function is in compiler memoryspace because the
     * encode function will write the encoded weights data there.
     */
    virtual mli_status AttachBufferOffsets(Tensor<OffsetBuffer, kPreluRank> &input,
                                           Tensor<OffsetBuffer, kPreluRank> &output,
                                           OffsetBuffer &params,
                                           OffsetBuffer &ctrl_buffer) { return MLI_STATUS_OK; }

    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     * 
     * The memory ID's are used to index the membases array that will be passed
     * to the constructor of the runtime class. The offsets will added to the base
     * addresses provided in the membase array during runtime.
     *
     * @param input       [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param output      [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param params      [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param ctrl_buffer [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * 
     * @return MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &params,
                                           const OffsetBuffer &ctrl_buffer) { return MLI_STATUS_OK; }

    /**
     * @brief Method to set iteration information used in the .Update()
     * @deprecated
     *
     * NOTE: the use of this method is optional. if there is a single tile, and the .Update() is not used,
     *       this data doesn't need to be set.     
     * All the increments are following the output tile iterator.
     * @param output_total_size[4] [I] total size in each dimension
     * @param iteration_order[4] [I] which dimension of the output to iterate first.
     * @param input_first_inc[4] [I] increment of the input buffer pointer for the first iteration in each dimension
     * @param input_inc[4] [I] increment of the input buffer pointer for the other iterations in each dimension
     * @param output_first_inc[4] [I] increment of the output buffer pointer for the first iteration in each dimension
     * @param output_inc[4] [I] increment of the output buffer pointer for the other iterations in each dimension
     */
    virtual mli_status SetIterators(uint32_t output_total_size[4],
                                    uint32_t iteration_order[4],
                                    uint32_t input_first_inc[4],
                                    uint32_t input_inc[4],
                                    uint32_t output_first_inc[4],
                                    uint32_t output_inc[4]) { return MLI_STATUS_OK; }
};


/**
 * @brief This class implements the Depthwise Convolution 2D Compiler Support kernel interface
 *
 */
class DepthwiseConv2d_CS : public CompilerGenericInterface {
public:
    virtual ~DepthwiseConv2d_CS() = default;

    /**
     * @brief Method to encode the weights (coefficients).
     * TODO: add description using conv2d_cs as a starting point
     */
    virtual mli_status EncodeWeights(Tensor<Buffer, kDepthwiseWRank> &weights,
                                     Buffer &encoded_weights,
                                     compression_mode_t mode) = 0;

    /**
     * @brief Method to encode the weights (coefficients) with weights zero points
     *
     * This method will read the weights and weights zp buffers in a platform independent layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * This transformation may include compression
     * The content of the encode_weights buffer is opaque for the user.
     *
     * @param weights          [I] TensorIterator with the weights
     * @param weights_zp       [I] TensorIterator with the weights_zp
     * @param encoded_weights  [I] buffer pointer where the encode function can write the encoded weights/weights_zp
     */
    virtual mli_status EncodeWeightsAndZeroPts(TensorIterator<Buffer, kDepthwiseWRank,  kDepthwiseIterRank> &weights,
                                               TensorIterator<Buffer, kDepthwiseZPRank, kDepthwiseIterRank> &weights_zp,
                                               Buffer &encoded_weights)
                                               { NOT_IMPLEMENTED_METHOD;
                                                 return MLI_STATUS_OK; };
    /**
     * @brief Method to query the size of the encoded weights buffer
     *
     */
    virtual unsigned GetEncodedWeightsSize() = 0;

    /**
     * @brief Method to encode input zero-points (padding values)
     *
     */
    virtual mli_status EncodeInpZeroPts(Tensor<Buffer, kDepthwiseZPRank> &inpzeropts,
                                        Buffer &encoded_inpzeropts) = 0;

    /**
     * @brief Method to encode input zero-points (padding values)
     *
     * This method will read the input zero-points buffer in a platform independent layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * The content of the encoded_inpzeropts buffer is opaque for the user.
     *
     * @param input_zp          [I] Input_zp TensorIterator
     * @param encoded_input_zp  [I] buffer pointer where the encode function can write the encoded input zero points
     * 
     */
    virtual mli_status EncodeInpZeroPts(TensorIterator<Buffer, kDepthwiseZPRank, kDepthwiseIterRank> &input_zp,
                                        Buffer& encoded_input_zp)
                                        { NOT_IMPLEMENTED_METHOD;
                                          return MLI_STATUS_OK; };
    /**
     * @brief Method to query the size of the encoded input zero-points buffer
     *
     */
    virtual unsigned GetEncodedInpZeroPtsSize() = 0;

    /**
     * @brief Method to encode weights zero-points
     *
     */
    virtual mli_status EncodeWtsZeroPts(Tensor<Buffer, kDepthwiseZPRank> &wtszeropts,
                                        Buffer &encoded_wtszeropts) { return MLI_STATUS_OK; }

    /**
     * @brief Method to query the size of the encoded weights zero-points buffer
     *
     */
    virtual unsigned GetEncodedWtsZeroPtsSize() { return 0; }

    /**
     * @brief Methods to get buffer sizes
     * TODO: add description using conv2d_cs as a starting point
     */

    virtual unsigned GetInputBufferSize() = 0;
    virtual unsigned GetOutputBufferSize() = 0;
    virtual unsigned GetWeightsBufferSize() = 0;   
    virtual unsigned GetInputZeroPtsBufferSize() { return 0; }

    /**
     * @brief Methods to set buffer offsets
     * @deprecated
     * Be carefull - depthwise conv2d I/O tensors of rank 4 are deprecated - new interfaces use rank 5
     */
    virtual mli_status AttachBufferOffsets(Tensor<OffsetBuffer, 4> &input,
                                           Tensor<OffsetBuffer, 4> &output,
                                           OffsetBuffer &weights,
                                           OffsetBuffer &inpzeropts,
                                           OffsetBuffer &wtszeropts,
                                           OffsetBuffer &ctrl_buffer) = 0;

    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     *
     * Compiler computes a memory map and buffer offsets are set using this method.
     * Compiler also needs to indicate in which memory the buffers reside.
     * These ID's need to match the array of memory bases that the xop-interpreter passes to
     * the init function.
     *
     * In this method you specify offsets for tensors passed to the constructor
     *
     * @param input       [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param output      [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param weights     [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param inpzeropts  [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param wtszeropts  [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param ctrl_buffer [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     *
     * @return MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer& input,
                                           const OffsetBuffer& output,
                                           const OffsetBuffer& weights,
                                           const OffsetBuffer& inpzeropts,
                                           const OffsetBuffer& wtszeropts,
                                           const OffsetBuffer& ctrl_buffer)  = 0;
};

/**
 * @brief This class implements the Fully Connected Compiler Support kernel interface
 *
 */
class FullyConnected_CS : public CompilerGenericInterface {
public:
    virtual ~FullyConnected_CS() = default;

    /**
     * @brief Method to encode the weights (coefficients).
     * TODO: add description using conv2d_cs as a starting point
     */
    virtual mli_status EncodeWeights(const Tensor<Buffer, kFullyConnectedWRank> &weights,
                                     Buffer &encoded_weights) = 0;

    /**
     * @brief Method to encode the weights (coefficients) with weights zero points
     *
     * This method will read the weights and weights zp buffers in a platform independent layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * This transformation may include compression
     * The content of the encode_weights buffer is opaque for the user.
     *
     * @param weights          [I] TensorIterator with the weights
     * @param weights_zp       [I] TensorIterator with the weights_zp
     * @param encoded_weights  [I] buffer pointer where the encode function can write the encoded weights/weights_zp
     */
    virtual mli_status EncodeWeightsAndZeroPts(TensorIterator<Buffer, kFullyConnectedWRank,  kFullyConnectedIterRank> &weights,
                                               TensorIterator<Buffer, kFullyConnectedZPRank, kFullyConnectedIterRank> &weights_zp,
                                               Buffer &encoded_weights)
                                               { NOT_IMPLEMENTED_METHOD;
                                                 return MLI_STATUS_OK; };
    /**
     * @brief Method to query the size of the encoded weights buffer
     *
     */
    virtual unsigned GetEncodedWeightsSize() const = 0;

    /**
     * @brief Method to encode weights zero-points
     *
     */
    virtual mli_status EncodeWtsZeroPts(const Tensor<Buffer, kFullyConnectedZPRank> &wtszeropts,
                                        Buffer &encoded_wtszeropts) = 0;
    /**
     * @brief Method to query the size of the encoded weights zero-points buffer
     *
     */
    virtual unsigned GetEncodedWtsZeroPtsSize() const { return 0;}

    /**
     * @brief Methods to get buffer sizes
     * TODO: add description using conv2d_cs as a starting point
     */

    virtual unsigned GetInputBufferSize() const = 0;
    virtual unsigned GetOutputBufferSize() const = 0;
    virtual unsigned GetWeightsBufferSize() const = 0;
    virtual unsigned GetZeroPointBufferSize() const = 0;
    /**

     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kFullyConnectedIORank> &input,
                                           const Tensor<OffsetBuffer, kFullyConnectedIORank> &output,
                                           const OffsetBuffer &weights,
                                           const OffsetBuffer &wtszeropts,
                                           const OffsetBuffer &ctrl_buffer) = 0;
};

/**
 * @brief This class implements the Max Pooling 2D Compiler Support kernel interface
 *
 */
class MaxPool2D_CS : public CompilerGenericInterface {
public:
    virtual ~MaxPool2D_CS() = default;

    /**
     * @brief Method to get the input buffer size
     *
     * @return Size of the input buffer in bytes
     */
    virtual unsigned GetInputBufferSize() const = 0;

    /**
     * @brief Method to get the output buffer size
     *
     * @return Size of the output buffer in bytes
     */
    virtual unsigned GetOutputBufferSize() const = 0;
    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     * 
     * Compiler computes a memory map and buffer offsets are set using this method.
     * Compiler also needs to indicate in which memory the buffers reside.
     * These ID's need to match the array of memory bases that the xop-interpreter passes to
     * the Create function.
     *
     * In this method you specify offsets for tensors passed to the constructor.
     * 
     * @deprecated
     * @param input [I] Tensor descriptor containing input OffsetBuffer and tensor shape and memory strides
     * @param output [I] Tensor descriptor containing output OffsetBuffer and tensor shape and memory strides
     * @param ctrl_buffer [I] data OffsetBuffer
     * 
     * @return MLI status code
     */
    virtual mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kMaxpoolRank> &input,
                                           const Tensor<OffsetBuffer, kMaxpoolRank> &output,
                                           const OffsetBuffer &ctrl_buffer) = 0;

    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     *
     * Compiler computes a memory map and buffer offsets are set using this method.
     * Compiler also needs to indicate in which memory the buffers reside.
     * These ID's need to match the array of memory bases that the xop-interpreter passes to
     * the Create function.
     *
     * In this method you specify offsets for tensors passed to the constructor.
     *
     * @param input       [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param output      [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param ctrl_buffer [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     *
     * @return MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer& input,
                                           const OffsetBuffer& output,
                                           const OffsetBuffer& ctrl_buffer)  = 0;
};

/**
 * @brief This class implements the Summation Pooling 2D Compiler Support kernel interface
 * Summation pooling is a first phase of average pooling which accumulates all values
 * across perception areas of the kernel size. The following multiplication of result with reciprocal
 * value is required to get average pooling behavior.
 */
class SumPool2D_CS : public CompilerGenericInterface {
public:
    virtual ~SumPool2D_CS() = default;

    /**
     * @brief Methods to get buffer sizes
     */

    virtual unsigned GetInputBufferSize() const = 0;
    virtual unsigned GetOutputBufferSize() const = 0;
    /**

     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input,
                                           const Tensor<OffsetBuffer, 4> &output,
                                           const OffsetBuffer &ctrl_buffer) = 0;
};


 /**
 * @brief This class implements the Rescale Compiler Support kernel interface
 *
 */
class Rescale_CS : public CompilerGenericInterface {
public:
    virtual ~Rescale_CS() = default;

    /**
     * @brief Method to encode the parameters
     *
     * This method will read the different parameters buffer in a platform independent layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * This transformation may include compression
     * The content of the encode_params buffer is opaque for the user.
     *
     * @param in_bias        [I] tensor with the input bias
     * @param out_bias       [I] tensor with the output bias
     * @param scale          [I] tensor with the scale
     * @param shift          [I] tensor with the shift
     * @param encoded_params [O] encoded parameters buffer
     */
    virtual mli_status EncodeParams(const Tensor<Buffer, kRescaleParamRank> &in_bias,
                                    const Tensor<Buffer, kRescaleParamRank> &out_bias,
                                    const Tensor<Buffer, kRescaleParamRank> &scale,
                                    const Tensor<Buffer, kRescaleParamRank> &shift,
                                    Buffer &encoded_params) = 0;

    /**
     * @brief Method to query the size of the encoded weights buffer
     *
     */
    virtual unsigned GetEncodedParamsSize() const = 0;

    /**
     * @brief Methods to get buffer sizes
     *
     */

    virtual unsigned GetInputBufferSize() const = 0;
    virtual unsigned GetOutputBufferSize() const = 0;

    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     *
     * Compiler computes a memory map and buffer offsets are set using this method.
     * Compiler also needs to indicate in which memory the buffers reside.
     * These ID's need to match the array of memory bases that the xop-interpreter passes to
     * the init function.
     *
     * In this method you specify offsets for tensors passed to the constructor
     *
     * @param input          [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param output         [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param encoded_params [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param ctrl_buffer    [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     *
     * @return MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer& input,
                                           const OffsetBuffer& output,
                                           const OffsetBuffer& encoded_params,
                                           const OffsetBuffer& ctrl_buffer) = 0;
};

/**
 * @brief This class implements the Clip Compiler Support kernel interface
 *
 */
class Clip_CS : public CompilerGenericInterface {
public:
    virtual ~Clip_CS() = default;

    /**
     * @brief Method to encode parameters (coefficients)
     *
     */
    virtual mli_status EncodeParams(Tensor<Buffer, kClipParamRank> &min_val,
                                    Tensor<Buffer, kClipParamRank> &max_val,
                                    Buffer &encoded_params) = 0;

    /**
     * @brief Method to query the size of the encoded weights buffer
     *
     */
    virtual unsigned GetEncodedParamsSize() const = 0;

    /**
     * @brief Methods to get buffer sizes
     *
     */

    virtual unsigned GetInputBufferSize() const = 0;
    virtual unsigned GetOutputBufferSize() const = 0;
    virtual unsigned GetParamsBufferSize() const = 0;
    /**
     * @brief Methods to set buffer offsets
     * @deprecated
     */
    virtual mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kClipRank> &input,
                                           const Tensor<OffsetBuffer, kClipRank> &output,
                                           const OffsetBuffer &encoded_params,
                                           const OffsetBuffer &ctrl_buffer) = 0;

    /**
     * @brief Methods to set buffer offsets
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer& input,
                                           const OffsetBuffer& output,
                                           const OffsetBuffer& encoded_params,
                                           const OffsetBuffer& descr) = 0;

    /**
     * @brief Method to set iteration information used in the .Update()
     *
     * NOTE: the use of this method is optional. if there is a single tile, and the .Update() is not used,
     *       this data doesn't need to be set.
     * All the increments are following the output tile iterator.
     * 
     * @deprecated
     * @param output_total_size[4] [I] total size in each dimension
     * @param iteration_order[4] [I] which dimension of the output to iterate first.
     * @param output_first_inc[4] [I] increment of the output buffer pointer for the first iteration in each dimension
     * @param output_inc[4] [I] increment of the output buffer pointer for the other iterations in each dimension
     */
    virtual mli_status SetIterators(uint32_t output_total_size[kClipIterRank],
                                    uint32_t iteration_order[kClipIterRank],
                                    uint32_t output_first_inc[kClipIterRank],
                                    uint32_t output_inc[kClipIterRank]) = 0;

};

/**
 * @brief This class implements the Eltwise Addition Compiler Support kernel interface
 *
 */
class Add_CS : public CompilerGenericInterface {
public:
    virtual ~Add_CS() = default;

    /**
     * @brief Methods to get buffer sizes
     */

    virtual unsigned GetInputLeftBufferSize() = 0;
    virtual unsigned GetInputRightBufferSize() = 0;
    virtual unsigned GetOutputBufferSize() = 0;
    /**
     * @brief Methods to set buffer offsets
     *
     * @deprecated
     */
    virtual mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kEltwiseRank> &input_l,
                                           const Tensor<OffsetBuffer, kEltwiseRank> &input_r,
                                           const Tensor<OffsetBuffer, kEltwiseRank> &output,
                                           const OffsetBuffer &ctrl_buffer) = 0;
    
    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     *
     * Compiler computes a memory map and buffer offsets are set using this method.
     * Compiler also needs to indicate in which memory the buffers reside.
     * These ID's need to match the array of memory bases that the xop-interpreter passes to
     * the Create function.
     *
     * In this method you specify offsets for tensors passed to the constructor.
     *
     * @param input_l         [I] input OffsetBuffer 
     * @param input_r         [I] input OffsetBuffer 
     * @param output          [I] output OffsetBuffer
     * @param ctrl_buffer     [I] descriptor data OffsetBuffer
     *
     * @return    MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input_l,
                                           const OffsetBuffer &input_r,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &ctrl_buffer) { return MLI_STATUS_OK; }
};

/**
 * @brief This class implements the Eltwise Subtraction Compiler Support kernel interface
 *
 */
class Sub_CS : public CompilerGenericInterface {
public:
    virtual ~Sub_CS() = default;

    /**
     * @brief Methods to get buffer sizes
     */

    virtual unsigned GetInputLeftBufferSize() = 0;
    virtual unsigned GetInputRightBufferSize() = 0;
    virtual unsigned GetOutputBufferSize() = 0;
    /**
     * @brief Methods to set buffer offsets
     * @deprecated
     */
    virtual mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input_left,
                                           const Tensor<OffsetBuffer, 4> &input_right,
                                           const Tensor<OffsetBuffer, 4> &output,
                                           const OffsetBuffer &ctrl_buffer) = 0;

    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     *
     * Compiler computes a memory map and buffer offsets are set using this method.
     * Compiler also needs to indicate in which memory the buffers reside.
     * These ID's need to match the array of memory bases that the xop-interpreter passes to
     * the Create function.
     *
     * In this method you specify offsets for tensors passed to the constructor.
     *
     * @param input_l         [I] input OffsetBuffer 
     * @param input_r         [I] input OffsetBuffer 
     * @param output          [I] output OffsetBuffer
     * @param ctrl_buffer     [I] descriptor data OffsetBuffer
     *
     * @return    MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input_l,
                                           const OffsetBuffer &input_r,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &ctrl_buffer){ return MLI_STATUS_OK; }
};


/**
 * @brief This class implements the Eltwise Multiply Compiler Support kernel interface
 *
 */
class Mul_CS : public CompilerGenericInterface {
public:
    virtual ~Mul_CS() = default;

    /**
     * @brief Methods to get buffer sizes
     */

    virtual unsigned GetInputLeftBufferSize() = 0;
    virtual unsigned GetInputRightBufferSize() = 0;
    virtual unsigned GetOutputBufferSize() = 0;
    /**
     * @brief Methods to set buffer offsets
     * @deprecated
     */
    virtual mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input_left,
                                           const Tensor<OffsetBuffer, 4> &input_right,
                                           const Tensor<OffsetBuffer, 4> &output,
                                           const OffsetBuffer &ctrl_buffer) = 0;

    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     *
     * Compiler computes a memory map and buffer offsets are set using this method.
     * Compiler also needs to indicate in which memory the buffers reside.
     * These ID's need to match the array of memory bases that the xop-interpreter passes to
     * the Create function.
     *
     * In this method you specify offsets for tensors passed to the constructor.
     *
     * @param input_l         [I] input OffsetBuffer 
     * @param input_r         [I] input OffsetBuffer 
     * @param output          [I] output OffsetBuffer
     * @param ctrl_buffer     [I] descriptor data OffsetBuffer
     *
     * @return    MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input_l,
                                           const OffsetBuffer &input_r,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &ctrl_buffer){ return MLI_STATUS_OK; }                                   
};

/**
 * @brief This class implements the Eltwise Max Compiler Support kernel interface
 *
 */
class Max_CS : public CompilerGenericInterface {
public:
    virtual ~Max_CS() = default;

    /**
     * @brief Methods to get buffer sizes
     */

    virtual unsigned GetInputLeftBufferSize() = 0;
    virtual unsigned GetInputRightBufferSize() = 0;
    virtual unsigned GetOutputBufferSize() = 0;
    /**

     * @brief Methods to set buffer offsets
     * @deprecated
     */
    virtual mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input_left,
                                           const Tensor<OffsetBuffer, 4> &input_right,
                                           const Tensor<OffsetBuffer, 4> &output,
                                           const OffsetBuffer &ctrl_buffer) = 0;

    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     *
     * Compiler computes a memory map and buffer offsets are set using this method.
     * Compiler also needs to indicate in which memory the buffers reside.
     * These ID's need to match the array of memory bases that the xop-interpreter passes to
     * the Create function.
     *
     * In this method you specify offsets for tensors passed to the constructor.
     *
     * @param input_l         [I] input OffsetBuffer 
     * @param input_r         [I] input OffsetBuffer 
     * @param output          [I] output OffsetBuffer
     * @param ctrl_buffer     [I] descriptor data OffsetBuffer
     *
     * @return    MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input_l,
                                           const OffsetBuffer &input_r,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &ctrl_buffer){ return MLI_STATUS_OK; } 
};

/**
 * @brief This class implements the Eltwise Min Compiler Support kernel interface
 *
 */
class Min_CS : public CompilerGenericInterface {
public:
    virtual ~Min_CS() = default;

    /**
     * @brief Methods to get buffer sizes
     */

    virtual unsigned GetInputLeftBufferSize() = 0;
    virtual unsigned GetInputRightBufferSize() = 0;
    virtual unsigned GetOutputBufferSize() = 0;
    /**

     * @brief Methods to set buffer offsets
     * @deprecated
     */
    virtual mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input_left,
                                           const Tensor<OffsetBuffer, 4> &input_right,
                                           const Tensor<OffsetBuffer, 4> &output,
                                           const OffsetBuffer &ctrl_buffer) = 0;

    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     *
     * Compiler computes a memory map and buffer offsets are set using this method.
     * Compiler also needs to indicate in which memory the buffers reside.
     * These ID's need to match the array of memory bases that the xop-interpreter passes to
     * the Create function.
     *
     * In this method you specify offsets for tensors passed to the constructor.
     *
     * @param input_l         [I] input OffsetBuffer 
     * @param input_r         [I] input OffsetBuffer 
     * @param output          [I] output OffsetBuffer
     * @param ctrl_buffer     [I] descriptor data OffsetBuffer
     *
     * @return    MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input_l,
                                           const OffsetBuffer &input_r,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &ctrl_buffer){ return MLI_STATUS_OK; }

};

/**
 * @brief This class implements the Table BuiltIn Compiler Support kernel interface
 *
 */
class TableBuiltin_CS : public CompilerGenericInterface {
public:
    virtual ~TableBuiltin_CS() = default;

    /**
     * @brief Method to encode parameters (coefficients)
     *
     */
    virtual mli_status EncodeParams(const TensorIterator<Buffer, kBiasRank, kBiasIterRank> &in_bias,
                                    Buffer &encoded_params) = 0;

    /**
     * @brief Method to query the size of the encoded weights buffer
     *
     */
    virtual unsigned GetEncodedParamsSize() = 0;

    /**
     * @brief Methods to get buffer sizes
     *
     */
    virtual unsigned GetParamsBufferSize() = 0;
    /**
     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &params,
                                           const OffsetBuffer &ctrl_buffer) = 0;
};

/**
 * @brief This class implements the ArgMax Compiler Support kernel interface
 *
 */
class ArgMax_CS : public CompilerGenericInterface {
public:

    virtual ~ArgMax_CS() = default;

    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     *
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &ctrl_buffer) = 0;
};

/**
 * @brief This class implements the ReduceMax Compiler Support kernel interface
 *
 */
class ReduceMax_CS : public CompilerGenericInterface {
public:
    virtual ~ReduceMax_CS() = default;


    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     * 
     * The memory ID's are used to index the membases array that will be passed
     * to the constructor of the runtime class. The offsets will added to the base
     * addresses provided in the membase array during runtime.
     *
     * In this method you specify offsets for tensors passed to the constructor.
     *
     * @param input       [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param output      [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param ctrl_buffer [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * 
     * @return MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &ctrl_buffer) = 0;
};


/**
 * @brief This class implements the ReduceSum Compiler Support kernel interface
 *
 */
class ReduceSum_CS : public CompilerGenericInterface {
public:

    virtual ~ReduceSum_CS() = default;
    
    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     * 
     * The memory ID's are used to index the membases array that will be passed
     * to the constructor of the runtime class. The offsets will added to the base
     * addresses provided in the membase array during runtime.
     *
     * @param input       [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param output      [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param ctrl_buffer [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * 
     * @return MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &ctrl_buffer) = 0;
};

/**
 * @brief This class implements the Move Compiler Support kernel interface
 *
 */
class Move_CS : public CompilerGenericInterface {
public:

    /**
     * @brief Methods to get buffer sizes
     * TODO: add description using conv2d_cs as a starting point
     */

    // Temporary non-pure virtual functions, need to be implemented for other platforms.
    virtual unsigned GetInputBufferSize() const { return 0; };
    virtual unsigned GetOutputBufferSize() const { return 0; };
   
    /**
     * @brief Methods to set buffer offsets
     * @deprecated
     */
    virtual mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kMoveRank> &src,
                                           const Tensor<OffsetBuffer, kMoveRank> &dst) {
      return MLI_STATUS_NOT_SUPPORTED;
    };

    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     *
     * The memory ID's are used to index the membases array that will be passed
     * to the constructor of the runtime class. The offsets will be added to
     * the base addresses provided in the membase array during runtime.
     *
     * @param src         [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param dst         [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param ctrl_buffer [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     *
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &src,
                                           const OffsetBuffer &dst,
                                           const OffsetBuffer &ctrl_buffer) {
      return MLI_STATUS_NOT_SUPPORTED;
    };
};

/**
 * @brief This class implements the Transpose Convolution 2D kernel Compiler
 * Support interface
 *
 *
 */
class TransposeConv2D_CS : public CompilerGenericInterface {
public:
    /**
     * @brief Method to encode the weights (coefficients)
     *
     * This method will read the weights buffer in a platform independent layout
     * and translate it into a buffer that can be easily read by the platform
     * specific kernel implementation. This transformation may include
     * compression. The content of the encode_weights buffer is opaque for the
     * user.
     *
     * @param weights [I] tensor with the weights
     * @param buffer_t [I] buffer pointer where the encode function can write
     * the encoded weights
     * 
     * @return MLI status code
     */
    virtual mli_status EncodeWeights(
        Tensor<Buffer, kTransposeConvWRank> &weights, Buffer &encoded_weights,
        compression_mode_t mode = compression_mode_t::Uncompressed) = 0;

    /**
     * @brief Method to encode the weights (coefficients) with weights zero points
     *
     * This method will read the weights and weights zp buffers in a platform independent layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * This transformation may include compression
     * The content of the encode_weights buffer is opaque for the user.
     *
     * @param weights          [I] TensorIterator with the weights
     * @param weights_zp       [I] TensorIterator with the weights_zp
     * @param encoded_weights  [I] buffer pointer where the encode function can write the encoded weights/weights_zp
     */
    virtual mli_status EncodeWeightsAndZeroPts(TensorIterator<Buffer, kTransposeConvWRank,  kTransposeConvIterRank> &weights,
                                               TensorIterator<Buffer, kTransposeConvZPRank, kTransposeConvIterRank> &weights_zp,
                                               Buffer &encoded_weights)
                                               { NOT_IMPLEMENTED_METHOD;
                                                 return MLI_STATUS_OK; };
    // /**
    //  * @brief Method to query the size of the encoded weights buffer
    //  *
    //  * This function returns the size of the full weights buffer that
    //  * is needed by the EncodeWeights method.
    //  *
    //  * @return Size of encoded weights buffer in bytes
    //  */
    // virtual unsigned GetEncodedWeightsSize() const = 0;

    /**
     * @brief Method to encode input zero-points (padding values)
     *
     * This method will read the input zero-points buffer in a platform
     * independent layout and translate it into a buffer that can be easily read
     * by the platform specific kernel implementation. The content of the
     * encoded_inpzeropts buffer is opaque for the user.
     *
     * @return MLI status code
     */
    virtual mli_status EncodeInpZeroPts(Tensor<Buffer, kTransposeConvZPRank> &inpzeropts,
                                        Buffer &encoded_inpzeropts) = 0;

    /**
     * @brief Method to encode input zero-points (padding values)
     *
     * This method will read the input zero-points buffer in a platform independent layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * The content of the encoded_inpzeropts buffer is opaque for the user.
     *
     * @param input_zp          [I] Input_zp TensorIterator
     * @param encoded_input_zp  [I] buffer pointer where the encode function can write the encoded input zero points
     * 
     */
    virtual mli_status EncodeInpZeroPts(TensorIterator<Buffer, kTransposeConvZPRank, kTransposeConvZPIterRank> &input_zp,
                                        Buffer& encoded_input_zp)
                                        { NOT_IMPLEMENTED_METHOD;
                                          return MLI_STATUS_OK; };

    /**
     * @brief Method to query the size of the encoded input zero-points buffer
     *
     * This function returns the size of the full buffer that is needed by the
     * EncodeInpZeroPts method
     *
     * @return Size of input zero-points buffer in bytes
     */
    virtual unsigned GetEncodedInpZeroPtsSize() const = 0;

    /**
     * @brief Method to encode weights zero-points (padding values)
     *
     * This method will read the weights zero-points buffer in a platform
     * independent layout and translate it into a buffer that can be easily read
     * by the platform specific kernel implementation. The content of the
     * encode_wtszeropts buffer is opaque for the user.
     *
     * @return MLI status code
     */
    virtual mli_status EncodeWtsZeroPts(Tensor<Buffer, kTransposeConvZPRank> &wtszeropts,
                                        Buffer &encoded_wtszeropts) = 0;

    /**
     * @brief Method to query the size of the encoded input zero-points buffer
     *
     * This function returns the size of the full buffer that is needed by the
     * EncodeWtsZeroPts method
     *
     * @return Size of input zero-points buffer in bytes
     */
    virtual unsigned GetEncodedWtsZeroPtsSize() const = 0;

    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     * 
     * The memory ID's are used to index the membases array that will be passed
     * to the constructor of the runtime class. The offsets will added to the base
     * addresses provided in the membase array during runtime.
     *
     * @param input       [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param output      [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param weights     [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param inpzeropts  [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param wtszeropts  [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param ctrl_buffer [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * 
     * @return MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &weights,
                                           const OffsetBuffer &inpzeropts,
                                           const OffsetBuffer &wtszeropts,
                                           const OffsetBuffer &ctrl_buffer) = 0;
};

/**
 * @brief This class implements the Permute Compiler Support kernel interface
 *
 */
class Permute_CS : public CompilerGenericInterface {
public:
    virtual ~Permute_CS() = default;

     /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     * 
     * The memory ID's are used to index the membases array that will be passed
     * to the constructor of the runtime class. The offsets will added to the base
     * addresses provided in the membase array during runtime.
     *
     * @param input       [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param output      [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param ctrl_buffer [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * 
     * @return MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &ctrl_buffer) = 0;
};

/**
 * @brief This class implements the Matrix Multiply Compiler Support kernel interface
 *
 */
class MatMul_CS : public CompilerGenericInterface {
public:
    virtual ~MatMul_CS() = default;

    /**
     * @brief Method to encode parameters (coefficients)
     *
     */
    virtual mli_status EncodeParams(const Buffer &in_bias1, 
                                    const Buffer &in_bias2,
                                    const Buffer &encoded_params) = 0;

    /**
     * @brief Methods to set buffer offsets
     * 
     * The memory ID's are used to index the membases array that will be passed
     * to the constructor of the runtime class. The offsets will added to the base
     * addresses provided in the membase array during runtime.
     * 
     * @param input_left     [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param input_right    [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param output         [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param encoded_params [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param ctrl_buffer    [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input_left,
                                           const OffsetBuffer &input_right,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &encoded_params,
                                           const OffsetBuffer &ctrl_buffer) = 0;
};


/**
 * @brief This class implements the MoveBroadcast Compiler Support kernel interface
 *
 */
class MoveBroadcast_CS : public CompilerGenericInterface {
public:
    virtual ~MoveBroadcast_CS() = default;
    
    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     *
     * @param src         [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param dst         [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param ctrl_buffer [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * 
     * @return MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &src,
                                           const OffsetBuffer &dst,
                                           const OffsetBuffer &ctrl_buffer) = 0;
};

/**
 * @brief This class implements the ResizeBilinear Compiler Support kernel interface
 *
 */
class ResizeBilinear_CS : public CompilerGenericInterface {
public:

    virtual ~ResizeBilinear_CS() = default;

    /**
     * @brief Method to set buffer memory offsets and memory IDs for the kernel
     * 
     * The memory ID's are used to index the membases array that will be passed
     * to the constructor of the runtime class. The offsets will added to the base
     * addresses provided in the membase array during runtime.
     *
     * @param input       [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param output      [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * @param ctrl_buffer [I] OffsetBuffer containing Memory Identifier and Offset in that memory
     * 
     * @return MLI status code
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &ctrl_buffer) = 0;
};

} // namespace mli

#endif // _MLI_COMPILER_API_HPP_




