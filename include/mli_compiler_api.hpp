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
#include "mli_service_functions.hpp"

namespace snps_arc::metaware::mli {

    constexpr short int kTensorBatchDim = 0;
    constexpr short int kTensorHeightDim = 1;
    constexpr short int kTensorWidthDim = 2;
    constexpr short int kTensorChannelDim = 3;

    constexpr short int kTileGroupDim = 0;
    constexpr short int kTileHeightDim = 1;
    constexpr short int kTileWidthDim = 2;
    constexpr short int kTileChannelDim = 3;

    constexpr short int kKernelGroupDim = 0;
    constexpr short int kKernelHeightDim = 1;
    constexpr short int kKernelWidthDim = 2;
    constexpr short int kKernelChannelInDim = 3;
    constexpr short int kKernelChannelOutDim = 4;

    constexpr short int kKernelDWHeightDim = 0;
    constexpr short int kKernelDWWidthDim = 1;
    constexpr short int kKernelDWChannelInDim = 2;


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
    virtual mli_status EncodeWeights(Tensor<Buffer, 5>& weights,
                                     Buffer& encoded_weights,
                                     compression_mode_t mode = compression_mode_t::Uncompressed) = 0;

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
     * The content of the encode_inpzeropts buffer is opaque for the compiler.
     *
     */
    virtual mli_status EncodeInpZeroPts(Tensor<Buffer, 1>& inpzeropts, Buffer& encoded_inpzeropts) = 0;

    /**
     * @brief Method to query the size of the encoded input zero-points buffer
     *
     * This function returns the size of the buffer that is needed by the EncodeInpZeroPts method
     */
    virtual unsigned GetEncodedInpZeroPtsSize() = 0;

    /**
     * @brief Method to encode weights zero-points (padding values)
     *
     * This method will read the weights zero-points buffer in a platform independend layout
     * and translate it into a buffer that can be easily read by the platform specific
     * kernel implementation.
     * The content of the encode_wtszeropts buffer is opaque for the compiler.
     *
     */
    virtual mli_status EncodeWtsZeroPts(Tensor<Buffer, 1>& inpzeropts, Buffer& encoded_wtszeropts) = 0;

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
    virtual unsigned GetDataBufferSize() = 0;

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
    virtual mli_status AttachBufferOffsets(Tensor<OffsetBuffer, 4> &input,
                                           Tensor<OffsetBuffer, 4> &output,
                                           OffsetBuffer &weights,
                                           OffsetBuffer &inpzeropts,
                                           OffsetBuffer &wtszeropts,
                                           OffsetBuffer &descr) = 0;

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
    virtual unsigned GetDataBufferSize() = 0;

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
    virtual mli_status AttachBufferOffsets(Tensor<OffsetBuffer, 4> &input,
                                           Tensor<OffsetBuffer, 4> &output,
                                           OffsetBuffer &params,
                                           OffsetBuffer &descr) = 0;

    // mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override ;
    // unsigned GetKernelPrivateDataSize() override ;
    // unsigned GetRuntimeObjectSize() override ;
};

/* **********************************************************
 *
 * Add New Kernels Below
 * 
 * *********************************************************/


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
    virtual mli_status EncodeWeights(Tensor<Buffer, 3> &weights,
                                     Buffer &encoded_weights,
                                     compression_mode_t mode) = 0;

    /**
     * @brief Method to query the size of the encoded weights buffer
     *
     */
    virtual unsigned GetEncodedWeightsSize() = 0;

    /**
     * @brief Method to encode input zero-points (padding values)
     *
     */
    virtual mli_status EncodeInpZeroPts(Tensor<Buffer, 1> &inpzeropts,
                                        Buffer &encoded_inpzeropts) = 0;

    /**
     * @brief Method to query the size of the encoded input zero-points buffer
     *
     */
    virtual unsigned GetEncodedInpZeroPtsSize() = 0;

    /**
     * @brief Method to encode weights zero-points
     *
     */
    virtual mli_status EncodeWtsZeroPts(Tensor<Buffer, 1> &wtszeropts,
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
    virtual unsigned GetDataBufferSize() = 0;
    virtual unsigned GetInputZeroPtsBufferSize() {return 0;}
    /**
     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(Tensor<OffsetBuffer, 4> &input,
                                           Tensor<OffsetBuffer, 4> &output,
                                           OffsetBuffer &weights,
                                           OffsetBuffer &inpzeropts,
                                           OffsetBuffer &wtszeropts,
                                           OffsetBuffer &metadata) = 0;
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
    virtual mli_status EncodeWeights(const Tensor<Buffer, 2> &weights, 
                                     Buffer &encoded_weights) = 0;

    /**
     * @brief Method to query the size of the encoded weights buffer
     *
     */
    virtual unsigned GetEncodedWeightsSize() = 0;

    /**
     * @brief Methods to get buffer sizes
     * TODO: add description using conv2d_cs as a starting point
     */

    virtual unsigned GetInputBufferSize() = 0;
    virtual unsigned GetOutputBufferSize() = 0;
    virtual unsigned GetWeightsBufferSize() = 0;
    virtual unsigned GetDataBufferSize() = 0;

    /**
     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &weights,
                                           const OffsetBuffer &data) = 0;
};

/**
 * @brief This class implements the Max Pooling 2D Compiler Support kernel interface
 *
 */
class MaxPool2D_CS : public CompilerGenericInterface {
public:
    virtual ~MaxPool2D_CS() = default;

    /**
     * @brief Methods to get buffer sizes
     * TODO: add description using conv2d_cs as a starting point
     */

    virtual unsigned GetInputBufferSize() const = 0;
    virtual unsigned GetOutputBufferSize() const = 0;
    virtual unsigned GetDataBufferSize() const = 0;


    /**
     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input,
                                           const Tensor<OffsetBuffer, 4> &output,
                                           const OffsetBuffer &data) = 0;
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

    virtual unsigned GetInputBufferSize() = 0;
    virtual unsigned GetOutputBufferSize() = 0;
    virtual unsigned GetDataBufferSize() = 0;

    /**
     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &data) = 0;
};


 /**
 * @brief This class implements the Rescale Compiler Support kernel interface
 *
 */
class Rescale_CS : public CompilerGenericInterface {
public:
    virtual ~Rescale_CS() = default;

    /**
     * @brief Method to encode parameters (scales)
     *
     */
    virtual mli_status EncodeParams(const Tensor<Buffer, 1> &in_bias,
                                    const Tensor<Buffer, 1> &out_bias,
                                    const Tensor<Buffer, 1> &scale,
                                    const Tensor<Buffer, 1> &shift,
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

    virtual unsigned GetInputBufferSize() = 0;
    virtual unsigned GetOutputBufferSize() = 0;
    virtual unsigned GetParamsBufferSize() = 0;
    virtual unsigned GetDataBufferSize() = 0;

    /**
     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &params,
                                           const OffsetBuffer &data) = 0;
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
    virtual mli_status EncodeParams(Tensor<Buffer, 1> &min_val,
                                    Tensor<Buffer, 1> &max_val,
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

    virtual unsigned GetInputBufferSize() = 0;
    virtual unsigned GetOutputBufferSize() = 0;
    virtual unsigned GetParamsBufferSize() = 0;
    virtual unsigned GetDataBufferSize() = 0;

    /**
     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &params,
                                           const OffsetBuffer &data) = 0;
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
    virtual unsigned GetDataBufferSize() = 0;

    /**
     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input_l,
                                           const Tensor<OffsetBuffer, 4> &input_r,
                                           const Tensor<OffsetBuffer, 4> &output,
                                           const OffsetBuffer &descr) = 0;
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
    virtual unsigned GetDataBufferSize() = 0;

    /**
     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input_left,
                                           const Tensor<OffsetBuffer, 4> &input_right,
                                           const Tensor<OffsetBuffer, 4> &output,
                                           const OffsetBuffer &descr) = 0;
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
    virtual unsigned GetDataBufferSize() = 0;

    /**
     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input_left,
                                           const Tensor<OffsetBuffer, 4> &input_right,
                                           const Tensor<OffsetBuffer, 4> &output,
                                           const OffsetBuffer &descr) = 0;
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
    virtual mli_status EncodeParams(const Tensor<Buffer, 1> &in_bias,
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

    virtual unsigned GetInputBufferSize() = 0;
    virtual unsigned GetOutputBufferSize() = 0;
    virtual unsigned GetParamsBufferSize() = 0;
    virtual unsigned GetDataBufferSize() = 0;

    /**
     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &params,
                                           const OffsetBuffer &data) = 0;
};


/**
 * @brief This class implements the ReduceMax Compiler Support kernel interface
 *
 */
class ReduceMax_CS : public CompilerGenericInterface {
public:
    virtual ~ReduceMax_CS() = default;

    /**
     * @brief Methods to get buffer sizes
     */

    virtual unsigned GetInputBufferSize() = 0;
    virtual unsigned GetOutputBufferSize() = 0;
    virtual unsigned GetDataBufferSize() = 0;

    /**
     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &data) = 0;
};


/**
 * @brief This class implements the ReduceSum Compiler Support kernel interface
 *
 */
class ReduceSum_CS : public CompilerGenericInterface {
public:
    virtual ~ReduceSum_CS() = default;

    /**
     * @brief Methods to get buffer sizes
     */

    virtual unsigned GetInputBufferSize() = 0;
    virtual unsigned GetOutputBufferSize() = 0;
    virtual unsigned GetDataBufferSize() = 0;

    /**
     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                           const OffsetBuffer &output,
                                           const OffsetBuffer &data) = 0;
};

/**
 * @brief This class implements the Move Compiler Support kernel interface
 *
 */
class Move_CS : public CompilerGenericInterface {
public:
    static constexpr unsigned kMaxRank = 4;
    /**
     * @brief Methods to get buffer sizes
     * TODO: add description using conv2d_cs as a starting point
     */
    
    // Temporary non-pure virtual functions, need to be implemented for other platforms.
    virtual unsigned GetInputBufferSize() const { return 0; };
    virtual unsigned GetOutputBufferSize() const { return 0; };
    virtual unsigned GetDataBufferSize() const { return 0; };

    /**
     * @brief Methods to set buffer offsets
     *
     */
    virtual mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kMaxRank> &src,
                                           const Tensor<OffsetBuffer, kMaxRank> &dst) {
      return MLI_STATUS_NOT_SUPPORTED;
    };
};

} // namespace mli

#endif // _MLI_COMPILER_API_HPP_
