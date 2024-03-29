/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_REF_COMPILER_API_HPP_
#define _MLI_REF_COMPILER_API_HPP_

#include "mli_compiler_api.hpp"
#include "mli_platform_desc.hpp"

namespace lib_mli = ::snps_arc::metaware::mli;

namespace snps_arc::metaware::mli::ref {

using lib_mli::Tensor;
using lib_mli::Buffer;
using lib_mli::OffsetBuffer;
using lib_mli::NoBuffer;
class Conv2DPrivateData;
class Pool2DPrivateData;

class Nop_CS : public lib_mli::Nop_CS {};

class Conv2d_CS : public lib_mli::Conv2d_CS {
public:
    /**
     * @brief Constructor to create an Conv2d_CS compiler support object.
     *
     * This constructor can be used to create a Convolution 2D compiler support
     * object. This kernel computes each value of the output tensor as the result of convolution operation 
     * of all values in the related perception area of all channels of the input tensor.
     *
     * @deprected
     * Be carefull - conv2d I/O tensors of rank 4 are deprecated - new interfaces use rank 5 
     * Be carefull - this is the most deprecated Constructor
     *
     * @param pd [IN] Platform description
     * @param in [IN] input tensor (full shape, BHWCi layout)
     * @param weights [IN] weights tensor (full shape, GKyKxCiCo layout)
     * @param cfg [IN] Conv2DConfig structure
     * @param output_tile_shape [OUT] output tensor (tile shape, BHWCo layout)
     */
    Conv2d_CS(const lib_mli::PlatformDescription pd,
              const Tensor<NoBuffer, 4>& in,
              const Tensor<NoBuffer, 5>& weights,
              const Conv2DConfig& cfg,
              const Tensor<NoBuffer, 4>& output_tile_shape);

    /**
      * @brief Constructor to create an Conv2d_CS compiler support object.
      * @deprecated
      *
      * This constructor can be used to create a Convolution 2D compiler support
      * object. This kernel computes each value of the output tensor as the result of convolution operation
      * of all values in the related perception area of all channels of the input tensor.
      * of all values in the related perception area of a single channel of the input tensor.
      *
      * @param pd [IN] Platform description
      * @param input [IN] input TensorIterator (BHWGCi layout)
      * @param weights [IN] weights TensorIterator (GKyKxCiCo layout)
      * @param weights [IN] weights_zp TensorIterator
      * @param cfg [IN] Conv2DConfig structure
      * @param output [OUT] output TensorIterator (BHWGCo layout)
      */
    Conv2d_CS(const lib_mli::PlatformDescription pd,
              const TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank>& input,
              const TensorIterator<NoBuffer, kConvWRank, kConvWIterRank>& weights,
              const TensorIterator<NoBuffer, kConvZPRank, kConvZPIterRank>& weights_zp,
              const Conv2DConfig& cfg,
              const TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank>& output);

    /**
      * @brief Constructor to create an Conv2d_CS compiler support object.
      *
      * This constructor can be used to create a Convolution 2D compiler support
      * object. This kernel computes each value of the output as the result of convolution operation
      * of input with weights.
      *
      * @param pd          [I] Platform description
      * @param input       [I] Input TensorIterator (BHWGCi layout)
      * @param input_zp    [I] Input zero point(s) TensorIterator
      * @param weights     [I] Weights TensorIterator (GKyKxCiCo layout)
      * @param weights_zp  [I] Weights zero point(s) TensorIterator
      * @param cfg         [I] Conv2DConfig structure with conv parameters (stride, dilation, paddings)
      * @param output      [I] Output TensorIterator (BHWGCo layout)
      */
    Conv2d_CS(const lib_mli::PlatformDescription pd,
              const TensorIterator<NoBuffer, kConvIORank, kConvIterRank> &input,
              const TensorIterator<NoBuffer, kConvZPRank, kConvIterRank> &input_zp,
              const TensorIterator<NoBuffer, kConvWRank,  kConvIterRank> &weights,
              const TensorIterator<NoBuffer, kConvZPRank, kConvIterRank> &weights_zp,
              const Conv2DConfig &cfg,
              const TensorIterator<NoBuffer, kConvIORank, kConvIterRank> &output);

    /**
     * @deprecated
     */
    mli_status EncodeWeights(Tensor<Buffer, kConvWRank> &weights,
                             Buffer &encoded_weights,
                             compression_mode_t mode = compression_mode_t::Uncompressed) override;

    mli_status EncodeWeightsAndZeroPts(TensorIterator<Buffer, kConvWRank, kConvIterRank>& weights,
                                       TensorIterator<Buffer, kConvZPRank, kConvIterRank>& weights_zp,
                                       Buffer& encoded_weights) override;

    unsigned GetEncodedWeightsSize() override;

    /**
     * @deprecated
     */
    mli_status EncodeInpZeroPts(Tensor<Buffer, kInpZPRank> &inpzeropts,
                                Buffer &encoded_inpzeropts) override;

    mli_status EncodeInpZeroPts(TensorIterator<Buffer, kConvZPRank, kConvZPIterRank>& input_zp,
                                Buffer& encoded_inpzeropts) override;

    unsigned GetEncodedInpZeroPtsSize() override;

    /**
     * @deprecated
     */
    mli_status EncodeWtsZeroPts(Tensor<Buffer, kConvZPRank> &wtszeropts,
                                Buffer &encoded_wtszeropts) override;

    unsigned GetEncodedWtsZeroPtsSize() override;

    /**
     * Tensor buffer sizes could depend on the platform and/or parameters.
     * These functions can be used to query how much memory needs to be allocated for
     * the input, weights and output tensors.
     * Note, that these sizes are for full tensors, not tiles. 
     */
    unsigned GetInputBufferSize() override;
    unsigned GetOutputBufferSize() override;
    unsigned GetWeightsBufferSize() override;
    unsigned GetZeroPointBufferSize() override;

    /**
     * @deprecated
     * Be carefull - conv2d I/O tensors of rank 4 are deprecated - new interfaces use rank 5 
     */
    mli_status AttachBufferOffsets(Tensor<OffsetBuffer, 4> &input,
                                   Tensor<OffsetBuffer, 4> &output,
                                   OffsetBuffer &weights,
                                   OffsetBuffer &inpzeropts,
                                   OffsetBuffer &wtszeropts,
                                   OffsetBuffer &ctrl_buffer) override;

    mli_status AttachBufferOffsets(const OffsetBuffer& input,
                                   const OffsetBuffer& output,
                                   const OffsetBuffer& weights,
                                   const OffsetBuffer& inpzeropts,
                                   const OffsetBuffer& wtszeropts,
                                   const OffsetBuffer& ctrl_buffer) override;


    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;

    unsigned GetKernelPrivateDataSize() const override;

    unsigned GetRuntimeObjectSize() const override;

private:

    // Input, weights, weights zp(s), output tensors with offset buffer attached
    TensorIterator<OffsetBuffer, kConvIORank, kConvIOIterRank> m_input;
    TensorIterator<OffsetBuffer, kConvWRank, kConvWIterRank> m_weights;
    TensorIterator<OffsetBuffer, kConvZPRank, kConvZPIterRank> m_weights_zp;
    TensorIterator<OffsetBuffer, kConvIORank, kConvIOIterRank> m_output;

    // encoded zp buffers for input (optional for FX type)
    OffsetBuffer m_inpzp_buffer;

    // the axis to represent the quantization granularity (optional for FX type)
    int m_inp_quant_axis;
    int m_wts_quant_axis;

    // Configuration for Conv2d
    Conv2DConfig m_config;

    // The size of input, weights and output buffers used in `GetXX` methods
    uint32_t m_input_buffer_size;
    uint32_t m_weights_buffer_size;
    uint32_t m_output_buffer_size;

    // Platform descriptor
    lib_mli::PlatformDescription m_pd;
};

class DepthwiseConv2d_CS : public lib_mli::DepthwiseConv2d_CS {
public:
    /**
     * @brief Constructor of the DepthwiseConv2d_CS object
     * @deprecated
     * Be carefull - this ctor doesn't support tiling - only single tile size of provided tensors
     * Be carefull - depthwise conv2d I/O tensors of rank 4 are deprecated - new interfaces use rank 5
     * Be carefull - this is the most deprecated Constructor
     */
    DepthwiseConv2d_CS(const lib_mli::PlatformDescription pd,
                       const Tensor<NoBuffer, 4> &in,
                       const Tensor<NoBuffer, 3> &weights,
                       const DwConv2DConfig &cfg,
                       const Tensor<NoBuffer, 4> &output);

    /**
     * @brief Constructor of the DepthwiseConv2d_CS object
     * @deprecated
     */
    DepthwiseConv2d_CS(const lib_mli::PlatformDescription pd,
                       const TensorIterator<NoBuffer, kDepthwiseIORank, kDepthwiseIterRank>& input,
                       const TensorIterator<NoBuffer, kDepthwiseWRank, kDepthwiseIterRank>& weights,
                       const TensorIterator<NoBuffer, kDepthwiseZPRank, kDepthwiseIterRank>& weights_zp,
                       const DwConv2DConfig& cfg,
                       const TensorIterator<NoBuffer, kDepthwiseIORank, kDepthwiseIterRank>& output);

    /**
      * @brief Constructor to create an DepthwiseConv2d_CS compiler support object.
      *
      * This constructor can be used to create a Depthwise Convolution 2D compiler support
      * object. This kernel computes each value of the output tensor as the result of convolution operation
      * of input with weights.
      *
      * @param pd          [I] Platform description
      * @param input       [I] Input TensorIterator (BHWGCi layout)
      * @param input_zp    [I] input_zp TensorIterator
      * @param weights     [I] weights TensorIterator (GKyKxCiCo layout)
      * @param weights_zp  [I] weights_zp TensorIterator
      * @param cfg         [I] DwConv2DConfig structure
      * @param output      [I] output TensorIterator (BHWGCo layout)
      */
    DepthwiseConv2d_CS(const lib_mli::PlatformDescription pd,
                       const TensorIterator<NoBuffer, kDepthwiseIORank, kDepthwiseIterRank> &input,
                       const TensorIterator<NoBuffer, kDepthwiseZPRank, kDepthwiseIterRank> &input_zp,
                       const TensorIterator<NoBuffer, kDepthwiseWRank,  kDepthwiseIterRank> &weights,
                       const TensorIterator<NoBuffer, kDepthwiseZPRank, kDepthwiseIterRank> &weights_zp,
                       const DwConv2DConfig &cfg,
                       const TensorIterator<NoBuffer, kDepthwiseIORank, kDepthwiseIterRank> &output);

    /**
      * @deprecated
      */
    mli_status EncodeWeights(Tensor<Buffer, kDepthwiseWRank> &weights,
                             Buffer &encoded_weights,
                             compression_mode_t mode = compression_mode_t::Uncompressed) override;

    mli_status EncodeWeightsAndZeroPts(TensorIterator<Buffer, kDepthwiseWRank, kDepthwiseIterRank>& weights,
                                       TensorIterator<Buffer, kDepthwiseZPRank, kDepthwiseIterRank>& weights_zp,
                                       Buffer& encoded_weights) override;

    unsigned GetEncodedWeightsSize() override;

    /**
      * @deprecated
      */
    mli_status EncodeInpZeroPts(Tensor<Buffer, kDepthwiseZPRank> &inpzeropts,
                                Buffer &encoded_inpzeropts) override;

    mli_status EncodeInpZeroPts(TensorIterator<Buffer, kDepthwiseZPRank, kDepthwiseIterRank>& input_zp,
                                Buffer& encoded_input_zp) override;

    unsigned GetEncodedInpZeroPtsSize() override;

    /**
      * @deprecated
      */
    mli_status EncodeWtsZeroPts(Tensor<Buffer, kDepthwiseZPRank> &wtszeropts,
                                Buffer &encoded_wtszeropts) override;

    unsigned GetEncodedWtsZeroPtsSize() override;

    unsigned GetInputBufferSize() override;
    unsigned GetOutputBufferSize() override;
    unsigned GetWeightsBufferSize() override;

    /**
     * @deprecated
     * Be carefull - depthwise conv2d I/O tensors of rank 4 are deprecated - new interfaces use rank 5
     */
    mli_status AttachBufferOffsets(Tensor<OffsetBuffer, 4> &input,
                                   Tensor<OffsetBuffer, 4> &output,
                                   OffsetBuffer &weights,
                                   OffsetBuffer &inpzeropts,
                                   OffsetBuffer &wtszeropts,
                                   OffsetBuffer &ctrl_buffer) override;

    mli_status AttachBufferOffsets(const OffsetBuffer& input,
                                   const OffsetBuffer& output,
                                   const OffsetBuffer& weights,
                                   const OffsetBuffer& inpzeropts,
                                   const OffsetBuffer& wtszeropts,
                                   const OffsetBuffer& descr) override;

    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

private:
    // Input, weights, output tensors with offset buffer attached
    TensorIterator<OffsetBuffer, kDepthwiseIORank, kDepthwiseIterRank> m_input;
    TensorIterator<OffsetBuffer, kDepthwiseWRank, kDepthwiseIterRank> m_weights;
    TensorIterator<OffsetBuffer, kDepthwiseZPRank, kDepthwiseIterRank> m_weights_zp;
    TensorIterator<OffsetBuffer, kDepthwiseIORank, kDepthwiseIterRank> m_output;

    // encoded zp buffers for input and weights (optional for FX type)
    OffsetBuffer m_inpzp_buffer;

    // the axis to represent the quantization granularity (optional for FX type)
    int m_inp_quant_axis;
    int m_wts_quant_axis;

    // Configuration for Conv2d
    DwConv2DConfig m_config;

    // The size of input, weights and output buffers used in `GetXX` methods
    uint32_t m_input_buffer_size;
    uint32_t m_weights_buffer_size;
    uint32_t m_output_buffer_size;

    // Platform descriptor
    lib_mli::PlatformDescription m_pd;
};

class TransposeConv2D_CS : public lib_mli::TransposeConv2D_CS {
public:
    /**
     * @brief Constructor of the TransposeConv2D_CS object
     * @deprecated
     */
    TransposeConv2D_CS(const lib_mli::PlatformDescription pd,
                       const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> &input,
                       const TensorIterator<NoBuffer, kTransposeConvWRank, kTransposeConvWIterRank> &weights,
                       const TensorIterator<NoBuffer, kTransposeConvZPRank, kTransposeConvZPIterRank> &weights_zp,
                       const TransposeConv2DConfig &cfg,
                       const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> &output);

    /**
      * @brief Constructor to create an TransposeConv2d_CS compiler support object.
      *
      * This constructor can be used to create a Transpose Convolution 2D compiler support
      * object. This kernel computes each value of the output tensor as the result of deconvolution operation
      * of input with weights.
      *
      * @param pd          [I] Platform description
      * @param input       [I] Input TensorIterator (BHWGCi layout)
      * @param input_zp    [I] input_zp TensorIterator
      * @param weights     [I] weights TensorIterator (GKyKxCiCo layout)
      * @param weights_zp  [I] weights_zp TensorIterator
      * @param cfg         [I] TransposeConv2DConfig structure
      * @param output      [I] output TensorIterator (BHWGCo layout)
      */
    TransposeConv2D_CS(const lib_mli::PlatformDescription pd,
                       const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIterRank> &input,
                       const TensorIterator<NoBuffer, kTransposeConvZPRank, kTransposeConvIterRank> &input_zp,
                       const TensorIterator<NoBuffer, kTransposeConvWRank,  kTransposeConvIterRank> &weights,
                       const TensorIterator<NoBuffer, kTransposeConvZPRank, kTransposeConvIterRank> &weights_zp,
                       const TransposeConv2DConfig &cfg,
                       const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> &output);

    /**
     * @deprecated
     */
    mli_status EncodeWeights(Tensor<Buffer, kTransposeConvWRank> &weights, Buffer &encoded_weights,
                             compression_mode_t mode = compression_mode_t::Uncompressed) override;

    mli_status EncodeWeightsAndZeroPts(TensorIterator<Buffer, kTransposeConvWRank, kTransposeConvIterRank>& weights,
                                       TensorIterator<Buffer, kTransposeConvZPRank, kTransposeConvIterRank>& weights_zp,
                                       Buffer& encoded_weights) override;

    unsigned GetEncodedWeightsSize() const override;

    /**
     * @deprecated
     */
    mli_status EncodeInpZeroPts(Tensor<Buffer, kTransposeConvZPRank> &inpzeropts,
                                Buffer &encoded_inpzeropts) override;

    mli_status EncodeInpZeroPts(TensorIterator<Buffer, kTransposeConvZPRank, kTransposeConvZPIterRank>& input_zp,
                                Buffer& encoded_input_zp) override;

    unsigned GetEncodedInpZeroPtsSize() const override;

    /**
     * @deprecated
     */
    mli_status EncodeWtsZeroPts(Tensor<Buffer, kTransposeConvZPRank> &wtszeropts,
                                Buffer &encoded_wtszeropts) override;

    unsigned GetEncodedWtsZeroPtsSize() const override;

    mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                   const OffsetBuffer &output,
                                   const OffsetBuffer &weights,
                                   const OffsetBuffer &inpzeropts,
                                   const OffsetBuffer &wtszeropts,
                                   const OffsetBuffer &ctrl_buffer) override;

    mli_status GetKernelPrivateData(void *kernel_private_data_buffer) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

private:
    // Input, weights, output tensors with offset buffer attached
    TensorIterator<OffsetBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> m_input;
    TensorIterator<OffsetBuffer, kTransposeConvWRank, kTransposeConvWIterRank> m_weights;
    TensorIterator<OffsetBuffer, kTransposeConvZPRank, kTransposeConvZPIterRank> m_weights_zp;
    TensorIterator<OffsetBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> m_output;

    // Encoded zp buffers for input and weights (optional for FX type)
    OffsetBuffer m_inpzp_buffer;
    OffsetBuffer m_wtszp_buffer;

    uint32_t m_weights_buffer_size;

    // The axis to represent the quantization granularity (optional for FX type)
    int m_inp_quant_axis;
    int m_wts_quant_axis;

    // Configuration for TransposeConv2DConfig
    TransposeConv2DConfig m_config;

    // Platform descriptor
    lib_mli::PlatformDescription m_pd;
};

class MaxPool2D_CS : public lib_mli::MaxPool2D_CS {
public:
    /**
     * @brief Constructor to create a MaxPool2D compiler support object.
     *
     * This constructor can be used to create a Max Pooling 2D compiler support
     * object. This kernel computes each value of the output tensor as the maximum 
     * of all values in the related perception area of a single channel of the input tensor.
     *
     * @deprecated
     * @param pd [I] Platform description
     * @param in [I] Input tensor (full shape, BHWC layout)
     * @param cfg [I] PoolOpConfig structure
     * @param output_tile_shape [O] Output tensor (tile shape, BHWC layout)
     */
    MaxPool2D_CS(const lib_mli::PlatformDescription pd,
                 const Tensor<NoBuffer, kPoolRank> in,
                 const PoolOpConfig &cfg,
                 const Tensor<NoBuffer, kPoolRank> output_tile_shape);

     /**
     * @brief Constructor to create a MaxPool2D compiler support object.
     *
     * This constructor can be used to create a Max Pooling 2D compiler support
     * object. This kernel computes each value of the output tensor as the maximum
     * of all values in the related perception area of a single channel of the input tensor.
     *
     * @param pd [I] Platform description
     * @param in [I] Input tensor iterator (BHWC layout)
     * @param cfg [I] PoolOpConfig structure
     * @param out [O] Output tensor iterator (BHWC layout)
     */
    MaxPool2D_CS(const lib_mli::PlatformDescription pd,
                 const TensorIterator<NoBuffer, kPoolRank, kPoolIterRank> in,
                 const PoolOpConfig& cfg,
                 const TensorIterator<NoBuffer, kPoolRank, kPoolIterRank> out);

    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

    unsigned GetInputBufferSize() const override;
    unsigned GetOutputBufferSize() const override;

    /**
     * @return Always returns zero for reference kernel.
     */

    mli_status GetKernelPrivateData(void *kernel_private_data_buffer) override;

    /**
     * @deprecated
     */
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kPoolRank> &input,
                                   const Tensor<OffsetBuffer, kPoolRank> &output,
                                   const OffsetBuffer &ctrl_buffer) override;

    mli_status AttachBufferOffsets(const OffsetBuffer& input,
                                   const OffsetBuffer& output,
                                   const OffsetBuffer& ctrl_buffer) override;

private:
    TensorIterator<OffsetBuffer, kPoolRank, kPoolIterRank> m_input;
    TensorIterator<OffsetBuffer, kPoolRank, kPoolIterRank> m_output;

    PoolOpConfig m_config;

    lib_mli::PlatformDescription m_pd;
};

class SumPool2D_CS : public lib_mli::SumPool2D_CS {
public:
    SumPool2D_CS(const lib_mli::PlatformDescription pd,
                 const Tensor<NoBuffer, kPoolRank> in,
                 const PoolOpConfig &cfg,
                 const Tensor<NoBuffer, kPoolRank> output_tile_shape);
    /**
     * @brief Constructor to create a SumPool2D compiler support object.
     *
     * This constructor can be used to create a Sum Pooling 2D compiler support
     * object. This kernel computes each value of the output tensor as the sum 
     * of all values in the related perception area of a single channel of the input tensor.
     *
     * @param pd [I] Platform description
     * @param in [I] Input tensor iterator (BHWC layout)
     * @param cfg [I] PoolOpConfig structure
     * @param out [I] Output tensor iterator (BHWC layout)
     */
    SumPool2D_CS(const lib_mli::PlatformDescription pd,
                 const TensorIterator<NoBuffer, kPoolRank, kPoolIterRank> &in,
                 const PoolOpConfig &cfg,
                 const TensorIterator<NoBuffer, kPoolRank, kPoolIterRank> &out);

    // From CompilerGenericInterface
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

    unsigned GetInputBufferSize() const override;
    unsigned GetOutputBufferSize() const override;

    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;

    /**
     * @deprecated
    */
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kPoolRank> &input,
                                   const Tensor<OffsetBuffer, kPoolRank> &output,
                                   const OffsetBuffer &ctrl_buffer) override;

    mli_status AttachBufferOffsets(const OffsetBuffer& input,
                                   const OffsetBuffer& output,
                                   const OffsetBuffer& ctrl_buffer) override;

private:
    TensorIterator<OffsetBuffer, kPoolRank, kPoolIterRank> m_input;
    TensorIterator<OffsetBuffer, kPoolRank, kPoolIterRank> m_output;

    PoolOpConfig m_config;

    lib_mli::PlatformDescription m_pd;
};

class FullyConnected_CS : public lib_mli::FullyConnected_CS {
public:
    /**
     * @brief Constructor of the  FullyConnected_CS object
     * @deprecated
     */
    FullyConnected_CS(const lib_mli::PlatformDescription pd,
                      const Tensor<NoBuffer, kFullyConnectedIORank> &in,
                      const Tensor<NoBuffer, kFullyConnectedWRank>  &weights,
                      const Tensor<NoBuffer, kFullyConnectedIORank> &output_tile_shape);

    /**
     * @brief Constructor of the  FullyConnected_CS object
     * @deprecated
     */
    FullyConnected_CS(const lib_mli::PlatformDescription pd,
                      const Tensor<NoBuffer, kFullyConnectedIORank> &in,
                      const Tensor<NoBuffer, kFullyConnectedWRank> &weights,
                      const Tensor<NoBuffer, kFullyConnectedZPRank> &wtszp,
                      const Tensor<NoBuffer, kFullyConnectedIORank> &output_tile_shape);

    /**
      * @brief Constructor to create an FullyConnected_CS compiler support object.
      *
      * This constructor can be used to create a FullyConnected compiler support
      * object. This kernel computes each value of the output tensor as the result of convolution operation
      * for input with weights.
      *
      * @param pd          [I] Platform description
      * @param input       [I] Input TensorIterator (NCi layout)
      * @param weights     [I] weights TensorIterator (CiCo layout)
      * @param weights_zp  [I] weights_zp TensorIterator
      * @param cfg         [I] FullyConnectedConfig structure
      * @param output      [I] output TensorIterator (NCo layout)
      */
    FullyConnected_CS(const PlatformDescription pd,
                      const TensorIterator<NoBuffer, kFullyConnectedIORank, kFullyConnectedIterRank> &input,
                      const TensorIterator<NoBuffer, kFullyConnectedWRank,  kFullyConnectedIterRank> &weights,
                      const TensorIterator<NoBuffer, kFullyConnectedZPRank, kFullyConnectedIterRank> &weights_zp,
                      const FullyConnectedConfig &cfg,
                      const TensorIterator<NoBuffer, kFullyConnectedIORank, kFullyConnectedIterRank> &output);

    /**
      * @deprecated
      */
    mli_status EncodeWeights(const Tensor<Buffer, kFullyConnectedWRank> &weights,
                             Buffer &encoded_weights) override;


    mli_status EncodeWeightsAndZeroPts(TensorIterator<Buffer, kFullyConnectedWRank, kFullyConnectedIterRank>& weights,
                                       TensorIterator<Buffer, kFullyConnectedZPRank, kFullyConnectedIterRank>& weights_zp,
                                       Buffer& encoded_weights) override;

    unsigned GetEncodedWeightsSize() const override;

    /**
      * @deprecated
      */
    mli_status EncodeWtsZeroPts(const Tensor<Buffer, kFullyConnectedZPRank> &wtszeropts,
                                Buffer &encoded_wtszeropts) override;

    unsigned GetEncodedWtsZeroPtsSize() const override;
    unsigned GetInputBufferSize() const override;
    unsigned GetOutputBufferSize() const override;
    unsigned GetWeightsBufferSize() const override;
    unsigned GetZeroPointBufferSize() const override;

    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kFullyConnectedIORank> &input,
                                   const Tensor<OffsetBuffer, kFullyConnectedIORank> &output,
                                   const OffsetBuffer &weights,
                                   const OffsetBuffer &wtszeropts,
                                   const OffsetBuffer &ctrl_buffer) override;

    mli_status AttachBufferOffsets(const OffsetBuffer& input,
                                   const OffsetBuffer& output,
                                   const OffsetBuffer& weights_and_zeropts,
                                   const OffsetBuffer& ctrl_buffer) override;

    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

private:
    PlatformDescription m_pd;
    Tensor<OffsetBuffer, kFullyConnectedIORank> m_in;
    Tensor<OffsetBuffer, kFullyConnectedWRank>  m_weights;
    Tensor<OffsetBuffer, kFullyConnectedZPRank> m_wtszp;
    Tensor<OffsetBuffer, kFullyConnectedIORank> m_output;
    OffsetBuffer m_weights_zp;
};

class TableBuiltin_CS : public lib_mli::TableBuiltin_CS {
public:

    TableBuiltin_CS(const lib_mli::PlatformDescription pd,
                    const TensorIterator<NoBuffer, kTableBuiltinIORank, kTableBuiltinIOIterRank> &in,
                    const TableBuiltinConfig &cfg,
                    const TensorIterator<NoBuffer, kTableBuiltinIORank, kTableBuiltinIOIterRank> &out);

    // From TableBuiltin_CS
    unsigned GetParamsBufferSize() override;
    unsigned GetEncodedParamsSize() override;

    mli_status EncodeParams(const TensorIterator<Buffer, kBiasRank ,kBiasIterRank> &in_bias,
                            Buffer &encoded_params) override;

    mli_status GetKernelPrivateData(void *kernel_private_data_buffer) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

    mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                   const OffsetBuffer &output,
                                   const OffsetBuffer &params,
                                   const OffsetBuffer &ctrl_buffer) override;

private:
    TableBuiltinConfig m_config;
    TensorIterator<OffsetBuffer, kTableBuiltinIORank, kTableBuiltinIOIterRank> m_input;
    TensorIterator<OffsetBuffer, kTableBuiltinIORank, kTableBuiltinIOIterRank> m_output;
    OffsetBuffer m_encoded_params;

    uint32_t m_input_buffer_size;
    uint32_t m_output_buffer_size;
};

class ReduceMax_CS : public lib_mli::ReduceMax_CS {
public:

    /**
     * @brief Constructor to create a ReduceMax compiler support object.
     *
     * This constructor can be used to create a ReduceMax compiler support
     * object. This kernel computes each value of the output tensor as the maximum
     * of all values in the reduction axis of the input tensor.
     *
     * @param pd [I] Platform description
     * @param in [I] Input tensor iterator (BHWC layout)
     * @param cfg [I] ReduceOpConfig structure
     * @param out [O] Output tensor iterator (BHWC layout)
     */
    ReduceMax_CS(const lib_mli::PlatformDescription pd,
                 const TensorIterator<NoBuffer, kReduceMaxRank, kReduceMaxIterRank> &in,
                 const ReduceOpConfig &cfg,
                 const TensorIterator<NoBuffer, kReduceMaxRank, kReduceMaxIterRank> &out);
    
    /**
     * @brief Constructor to create a ReduceMax compiler support object.
     * ( // TODO: to be removed after support TensorIterator).
     *
     * This constructor can be used to create a ReduceMax compiler support
     * object. This kernel computes each value of the output tensor as the maximum
     * of all values in the reduction axis of the input tensor.
     *
     * @param pd [I] Platform description
     * @param in [I] Input tensor (BHWC layout)
     * @param cfg [I] ReduceOpConfig structure
     * @param out [O] Output tensor (BHWC layout)
     */
    ReduceMax_CS(const lib_mli::PlatformDescription pd,
                 const Tensor<NoBuffer, kReduceMaxRank> &input_shape,
                 const ReduceOpConfig &cfg,
                 const Tensor<NoBuffer, kReduceMaxRank> &out_tile_shape);
    
    mli_status GetKernelPrivateData(void *kernel_private_data_buffer ) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

    mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                   const OffsetBuffer &output,
                                   const OffsetBuffer &ctrl_buffer) override;

private:
    ReduceOpConfig m_cfg;

    TensorIterator<OffsetBuffer, kReduceMaxRank, kReduceMaxIterRank> m_in;
    TensorIterator<OffsetBuffer, kReduceMaxRank, kReduceMaxIterRank> m_out;
    uint32_t m_input_buffer_size;
    uint32_t m_output_buffer_size;

    lib_mli::PlatformDescription m_pd;
};

class ArgMax_CS : public lib_mli::ArgMax_CS {
public:
    /**
     * @brief Constructor to create a ArgMax compiler support object.
     *
     * This constructor can be used to create a ArgMax compiler support object.
     * This kernel returns the indexes of maximum values across whole Tensor, or for each slice across a dimension.  
     *
     * @param pd [IN] Platform description
     * @param in [IN] Input tensor (full shape)
     * @param cfg [IN] ArgMaxConfig structure
     * @param out [OUT] Output tensor (tile shape)
     */
    ArgMax_CS(const lib_mli::PlatformDescription pd,
              const TensorIterator<NoBuffer, kArgMaxInRank, kArgMaxInIterRank> in,
              const ArgMaxConfig &cfg,
              const TensorIterator<NoBuffer, kArgMaxOutRank, kArgMaxOutIterRank> out);

    mli_status GetKernelPrivateData(void *kernel_private_data_buffer ) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

    mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                   const OffsetBuffer &output,
                                   const OffsetBuffer &ctrl_buffer) override;

private:
    ArgMaxConfig m_cfg;
    TensorIterator<OffsetBuffer, kArgMaxInRank, kArgMaxInIterRank> m_in;
    TensorIterator<OffsetBuffer, kArgMaxOutRank, kArgMaxOutIterRank> m_out;

    uint32_t m_input_buffer_size;
    uint32_t m_output_buffer_size;

    lib_mli::PlatformDescription m_pd;
};

class ReduceSum_CS : public lib_mli::ReduceSum_CS {
public:
    /**
     * @brief Constructor to create a ReduceSum compiler support object.
     *
     * This constructor can be used to create a Reduce Sum compiler support
     * object. This kernel computes each value of the output tensor as the summation 
     * of all values in the reduction axis of the input tensor 
     *
     * @param pd [IN] Platform description
     * @param input_shape [IN] Input tensor (full shape)
     * @param cfg [IN] ReduceOpConfig structure
     * @param output_tile_shape [OUT] Output tensor (tile shape)
     */
    ReduceSum_CS(const lib_mli::PlatformDescription pd,
                 const TensorIterator<NoBuffer, kReduceSumRank, kReduceSumIterRank> &in,
                 const ReduceOpConfig &cfg,
                 const TensorIterator<NoBuffer, kReduceSumRank, kReduceSumIterRank> &out);


    mli_status GetKernelPrivateData(void *kernel_private_data_buffer ) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

    mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                   const OffsetBuffer &output,
                                   const OffsetBuffer &ctrl_buffer) override;

private:
    ReduceOpConfig m_cfg;
    TensorIterator<OffsetBuffer, kReduceSumRank, kReduceSumIterRank> m_in;
    TensorIterator<OffsetBuffer, kReduceSumRank, kReduceSumIterRank> m_out;

    lib_mli::PlatformDescription m_pd;
};

class Rescale_CS : public lib_mli::Rescale_CS {
public:
    Rescale_CS(const lib_mli::PlatformDescription pd,
               const TensorIterator<NoBuffer, kRescaleRank, kRescaleIterRank>& input,
               const RescaleConfig& cfg,
               const TensorIterator<NoBuffer, kRescaleParamRank, kRescaleIterRank>& enc_param,
               const TensorIterator<NoBuffer, kRescaleRank, kRescaleIterRank>& output);

    // From Rescale_CS
    unsigned GetInputBufferSize() const override;
    unsigned GetOutputBufferSize() const override;
    unsigned GetEncodedParamsSize() const override;
    mli_status EncodeParams(const Tensor<Buffer, kRescaleParamRank> &in_bias,
                            const Tensor<Buffer, kRescaleParamRank> &out_bias,
                            const Tensor<Buffer, kRescaleParamRank> &scale,
                            const Tensor<Buffer, kRescaleParamRank> &shift,
                            Buffer &encoded_params) override;
    mli_status GetKernelPrivateData( void *kernel_private_data_buffer ) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

    mli_status AttachBufferOffsets(const OffsetBuffer& input,
                                   const OffsetBuffer& output,
                                   const OffsetBuffer& encoded_params,
                                   const OffsetBuffer& ctrl_buffer) override;

private:
    RescaleConfig m_config;

    TensorIterator<OffsetBuffer, kRescaleRank, kRescaleIterRank> m_input;
    TensorIterator<OffsetBuffer, kRescaleParamRank, kRescaleIterRank> m_enc_param;
    TensorIterator<OffsetBuffer, kRescaleRank, kRescaleIterRank> m_output;

    uint32_t m_input_buffer_size;
    uint32_t m_output_buffer_size;
    uint32_t m_params_elem_num;

    // sizes in bytes
    uint32_t m_encoded_params_buffer_size;

    lib_mli::PlatformDescription m_pd;
};

class Move_CS : public lib_mli::Move_CS {
 public:
  /**
   * @brief Constructor to create a Move compiler support object.
   *
   * This constructor can be used to create a Move compiler object. This kernel
   * can transfer the data from one tensor to another in a tiled manner
   * The shapes of both src and dst tensors must match. However, memory strides
   * can be different for src and dst.
   *
   * The function accepts tensors with a templated rank up to kMoveRank
   *
   * IteratorCfg must be configured with the same tiles sizes for the src and
   * dst tensors while increments can be different.
   * @deprecated
   *
   * @param pd      [I] Platform description
   * @param src     [I] Source tensor shape and memory strides
   * @param dst     [O] Destination tensor shape and memory strides
   * @param src_cfg [I] Source iterator configuration
   * @param dst_cfg [I] Destination iterator configuration
   */
  Move_CS(const lib_mli::PlatformDescription pd,
          const Tensor<NoBuffer, kMoveRank> src,
          const Tensor<NoBuffer, kMoveRank> dst,
          const IteratorCfg<kMoveIterRank> src_it_cfg = IteratorCfg<kMoveIterRank>(),
          const IteratorCfg<kMoveIterRank> dst_it_cfg = IteratorCfg<kMoveIterRank>());

  /**
   * @brief Constructor to create a Move compiler support object.
   *
   * This constructor can be used to create a Move compiler support object. This kernel
   * can transfer the data from one tensor to another in a tiled manner
   * The shapes of both src and dst tensors must match. However, memory strides
   * can be different for src and dst.
   *
   * The function accepts TensorIterators with a templated rank up to kMoveRank
   *
   * @param pd  [I] Platform description
   * @param src [I] Source TensorIterator with shape and memory strides
   * @param dst [I] Destination TensorIterator with shape and memory strides
   */
  Move_CS(const lib_mli::PlatformDescription pd,
          const TensorIterator<NoBuffer, kMoveRank, kMoveIterRank> &src,
          const TensorIterator<NoBuffer, kMoveRank, kMoveIterRank> &dst);
  
  unsigned GetKernelPrivateDataSize() const override;
  unsigned GetRuntimeObjectSize() const override;
  mli_status GetKernelPrivateData(void *kernel_private_data_buffer) override;
  /*
  * @deprecated
  */
  mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kMoveRank> &src,
                                 const Tensor<OffsetBuffer, kMoveRank> &dst) override;
  
  mli_status AttachBufferOffsets(const OffsetBuffer &src,
                                 const OffsetBuffer &dst,
                                 const OffsetBuffer &ctrl_buffer) override;

private:
  lib_mli::PlatformDescription m_pd;

  TensorIterator<OffsetBuffer, kMoveRank, kMoveIterRank> m_src_it;
  TensorIterator<OffsetBuffer, kMoveRank, kMoveIterRank> m_dst_it;

  OffsetBuffer m_src_offset_buf;
  OffsetBuffer m_dst_offset_buf;

};

class Add_CS : public lib_mli::Add_CS {
public:

    Add_CS();

    /**
     * @brief Constructor to create an Add compiler support object.
     *
     * This constructor can be used to create Add compiler support
     * object. This kernel computes each value of the output tensor as the result of addition operation 
     * of the two input tensors element wise.
     *
     * @param pd                [I] Platform description
     * @param in_left           [I] First Input tensor (full shape)
     * @param in_right          [I] Second Input tensor (full shape)
     * @param output            [I] Output tensor (full shape)
     *
     * @deprecated
     */
    Add_CS(const lib_mli::PlatformDescription pd,
           const Tensor<NoBuffer, kEltwiseRank> in_left,
           const Tensor<NoBuffer, kEltwiseRank> in_right,
           const Tensor<NoBuffer, kEltwiseRank> output);

    /**
     * @brief Constructor to create an Add compiler support object.
     *
     * This constructor can be used to create Add compiler support
     * object. This kernel computes each value of the output tensor as the result of addition operation 
     * of the two input tensors element wise.
     *
     * @param pd                [I] Platform description
     * @param in_left           [I] First Input tensorIterator (full shape)
     * @param in_right          [I] Second Input tensorIterator (full shape)
     * @param output            [I] Output tensorIterator (full shape)
     */       

    Add_CS(const lib_mli::PlatformDescription pd,
           const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> in_left,
           const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> in_right,
           const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> output);

    // From CompilerGenericInterface
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kEltwiseRank> &input_left,
                                   const Tensor<OffsetBuffer, kEltwiseRank> &input_right,
                                   const Tensor<OffsetBuffer, kEltwiseRank> &output,
                                   const OffsetBuffer &ctrl_buffer) override;

    mli_status AttachBufferOffsets(const OffsetBuffer &input_left,
                                   const OffsetBuffer &input_right,
                                   const OffsetBuffer &output,
                                   const OffsetBuffer &ctrl_buffer) override;    

    // From Add_CS
    unsigned GetInputLeftBufferSize() override;
    unsigned GetInputRightBufferSize() override;
    unsigned GetOutputBufferSize() override;

private:
    
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_in_left;
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_in_right;
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_output;

    uint32_t m_in_left_buffer_size;
    uint32_t m_in_right_buffer_size;
    uint32_t m_output_buffer_size;

    bool m_is_left_scalar{true};
    bool m_is_right_scalar{true};

    lib_mli::PlatformDescription m_pd;
};

class Sub_CS : public lib_mli::Sub_CS {
public:

    Sub_CS();

    /**
     * @brief Constructor to create an Sub compiler support object.
     *
     * This constructor can be used to create Sub compiler support
     * object. This kernel computes each value of the output tensor as the subtraction 
     * of corresponding values in the two input tensors.
     *
     * @param pd                [I] Platform description
     * @param in_left           [I] First Input tensor (full shape)
     * @param in_right          [I] Second Input tensor (full shape)
     * @param output            [I] Output tensor (tile shape)
     * 
     * @deprecated
     */
    Sub_CS(const lib_mli::PlatformDescription pd,
           const Tensor<NoBuffer, kEltwiseRank> in_left,
           const Tensor<NoBuffer, kEltwiseRank> in_right,
           const Tensor<NoBuffer, kEltwiseRank> output);

    /**
     * @brief Constructor to create an Sub compiler support object.
     *
     * This constructor can be used to create Sub compiler support
     * object. This kernel computes each value of the output tensor as the result of subtraction operation 
     * of the two input tensors element wise.
     *
     * @param pd                [I] Platform description
     * @param in_left           [I] First Input tensor (full shape)
     * @param in_right          [I] Second Input tensor (full shape)
     * @param output            [I] Output tensor (full shape)
     */
    Sub_CS(const lib_mli::PlatformDescription pd,
           const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> in_left,
           const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> in_right,
           const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> output);

    // From CompilerGenericInterface
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input_left,
                                   const Tensor<OffsetBuffer, 4> &input_right,
                                   const Tensor<OffsetBuffer, 4> &output,
                                   const OffsetBuffer &ctrl_buffer) override;
                                   
    mli_status AttachBufferOffsets(const OffsetBuffer &input_left,
                                   const OffsetBuffer &input_right,
                                   const OffsetBuffer &output,
                                   const OffsetBuffer &ctrl_buffer) override;

    // From Sub_CS
    unsigned GetInputLeftBufferSize() override;
    unsigned GetInputRightBufferSize() override;
    unsigned GetOutputBufferSize() override;

private:
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_in_left;
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_in_right;
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_output;
   
    uint32_t m_in_left_buffer_size;
    uint32_t m_in_right_buffer_size;
    uint32_t m_output_buffer_size;

    bool m_is_left_scalar{true};
    bool m_is_right_scalar{true};

    lib_mli::PlatformDescription m_pd;
};

class Mul_CS : public lib_mli::Mul_CS {
public:

    Mul_CS();
    /**
     * @brief Constructor to create an Mul compiler support object.
     *
     * This constructor can be used to create Mul compiler support
     * object. This kernel computes each value of the output tensor as the result of multiplication operation 
     * of the two input tensors element wise.
     *
     * @param pd                [I] Platform description
     * @param in_left           [I] First Input tensor (full shape)
     * @param in_right          [I] Second Input tensor (full shape)
     * @param output            [I] Output tensor (full shape)
     *
     * @deprecated
     */
    Mul_CS(const lib_mli::PlatformDescription pd,
           const Tensor<NoBuffer, kEltwiseRank> in_left,
           const Tensor<NoBuffer, kEltwiseRank> in_right,
           const Tensor<NoBuffer, kEltwiseRank> output);
    
    /**
     * @brief Constructor to create an Mul compiler support object.
     *
     * This constructor can be used to create Mul compiler support
     * object. This kernel computes each value of the output tensor as the result of multiplication operation 
     * of the two input tensors element wise.
     *
     * @param pd                [I] Platform description
     * @param in_left           [I] First Input tensorIterator (full shape)
     * @param in_right          [I] Second Input tensorIterator (full shape)
     * @param output            [I] Output tensorIterator (full shape)
     */
    Mul_CS(const lib_mli::PlatformDescription pd,
           const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> in_left,
           const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> in_right,
           const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> output);

    // From CompilerGenericInterface
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kEltwiseRank> &input_left,
                                   const Tensor<OffsetBuffer, kEltwiseRank> &input_right,
                                   const Tensor<OffsetBuffer, kEltwiseRank> &output,
                                   const OffsetBuffer &ctrl_buffer) override;
                                   
    mli_status AttachBufferOffsets(const OffsetBuffer &input_left,
                                   const OffsetBuffer &input_right,
                                   const OffsetBuffer &output,
                                   const OffsetBuffer &ctrl_buffer) override;

    // From Mul_CS
    unsigned GetInputLeftBufferSize() override;
    unsigned GetInputRightBufferSize() override;
    unsigned GetOutputBufferSize() override;

private:
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_in_left;
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_in_right;
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_output;

    uint32_t m_in_left_buffer_size;
    uint32_t m_in_right_buffer_size;
    uint32_t m_output_buffer_size;

    bool m_is_left_scalar{true};
    bool m_is_right_scalar{true};

    lib_mli::PlatformDescription m_pd;
};

class Max_CS : public lib_mli::Max_CS {
public:

    Max_CS();
    /**
     * @brief Constructor to create an Max compiler support object.
     *
     * This constructor can be used to create Max compiler support
     * object. This kernel computes each value of the output tensor as the result of Max operation 
     * of the two input tensors element wise.
     *
     * @param pd                [I] Platform description
     * @param in_left           [I] First Input tensor (full shape)
     * @param in_right          [I] Second Input tensor (full shape)
     * @param output            [I] Output tensor (full shape)
     * 
     * @deprecated
     */
    Max_CS(const lib_mli::PlatformDescription pd,
           const Tensor<NoBuffer, kEltwiseRank> in_left,
           const Tensor<NoBuffer, kEltwiseRank> in_right,
           const Tensor<NoBuffer, kEltwiseRank> output);
    /**
     * @brief Constructor to create an Max compiler support object.
     *
     * This constructor can be used to create Max compiler support
     * object. This kernel computes each value of the output tensor as the result of Max operation 
     * of the two input tensors element wise.
     *
     * @param pd                [I] Platform description
     * @param in_left           [I] First Input tensorIterator (full shape)
     * @param in_right          [I] Second Input tensorIterator (full shape)
     * @param output            [I] Output tensorIterator (full shape)
     */
    Max_CS(const lib_mli::PlatformDescription pd,
           const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> in_left,
           const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> in_right,
           const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> output);

    // From CompilerGenericInterface
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kEltwiseRank> &input_left,
                                   const Tensor<OffsetBuffer, kEltwiseRank> &input_right,
                                   const Tensor<OffsetBuffer, kEltwiseRank> &output,
                                   const OffsetBuffer &ctrl_buffer) override;
                                  
    mli_status AttachBufferOffsets(const OffsetBuffer &input_left,
                                   const OffsetBuffer &input_right,
                                   const OffsetBuffer &output,
                                   const OffsetBuffer &ctrl_buffer) override;

    // From Max_CS
    unsigned GetInputLeftBufferSize() override;
    unsigned GetInputRightBufferSize() override;
    unsigned GetOutputBufferSize() override;

private:
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_in_left;
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_in_right;
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_output;

    uint32_t m_in_left_buffer_size;
    uint32_t m_in_right_buffer_size;
    uint32_t m_output_buffer_size;

    bool m_is_left_scalar{true};
    bool m_is_right_scalar{true};

    lib_mli::PlatformDescription m_pd;
};

class Min_CS : public lib_mli::Min_CS {
public:

    Min_CS();
    /**
     * @brief Constructor to create an Min compiler support object.
     *
     * This constructor can be used to create Min compiler support
     * object. This kernel computes each value of the output tensor as the result of Min operation 
     * of the two input tensors element wise.
     *
     * @param pd                [I] Platform description
     * @param in_left           [I] First Input tensorIterator (full shape)
     * @param in_right          [I] Second Input tensorIterator (full shape)
     * @param output            [I] Output tensorIterator (full shape)
     *
     * @deprecated
     */
    Min_CS(const lib_mli::PlatformDescription pd,
           const Tensor<NoBuffer, kEltwiseRank> in_left,
           const Tensor<NoBuffer, kEltwiseRank> in_right,
           const Tensor<NoBuffer, kEltwiseRank> output);
    /**
     * @brief Constructor to create an Min compiler support object.
     *
     * This constructor can be used to create Min compiler support
     * object. This kernel computes each value of the output tensor as the result of Min operation 
     * of the two input tensors element wise.
     *
     * @param pd                [I] Platform description
     * @param in_left           [I] First Input tensorIterator (full shape)
     * @param in_right          [I] Second Input tensorIterator (full shape)
     * @param output            [I] Output tensorIterator (full shape)
     */
    Min_CS(const lib_mli::PlatformDescription pd,
           const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> in_left,
           const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> in_right,
           const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> output);

    // From CompilerGenericInterface
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input_left,
                                   const Tensor<OffsetBuffer, 4> &input_right,
                                   const Tensor<OffsetBuffer, 4> &output,
                                   const OffsetBuffer &ctrl_buffer) override;
                                   
    mli_status AttachBufferOffsets(const OffsetBuffer &input_left,
                                   const OffsetBuffer &input_right,
                                   const OffsetBuffer &output,
                                   const OffsetBuffer &ctrl_buffer) override;

    // From Min_CS
    unsigned GetInputLeftBufferSize() override;
    unsigned GetInputRightBufferSize() override;
    unsigned GetOutputBufferSize() override;

private:
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_in_left;
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_in_right;
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_output;

    uint32_t m_in_left_buffer_size;
    uint32_t m_in_right_buffer_size;
    uint32_t m_output_buffer_size;

    bool m_is_left_scalar{true};
    bool m_is_right_scalar{true};

    lib_mli::PlatformDescription m_pd;
};

class Clip_CS : public lib_mli::Clip_CS {

 public:
   /**
     * @deprecated
     */
    Clip_CS(const lib_mli::PlatformDescription pd,
            const Tensor<NoBuffer, kClipRank> &input,
            const Tensor<NoBuffer, kClipRank> &output);

    Clip_CS(const lib_mli::PlatformDescription pd,
            const TensorIterator<NoBuffer, kClipRank, kClipIterRank>& input,
            const TensorIterator<NoBuffer, kClipRank, kClipIterRank>& output);

    unsigned GetRuntimeObjectSize() const override;

    mli_status GetKernelPrivateData(void *kernel_private_data_buffer) override;
    unsigned GetKernelPrivateDataSize() const override;

    mli_status EncodeParams(Tensor<Buffer, kClipParamRank> &min_val,
                            Tensor<Buffer, kClipParamRank> &max_val,
                            Buffer &encoded_params) override;

    unsigned GetEncodedParamsSize() const override;

    unsigned GetInputBufferSize() const override;
    unsigned GetOutputBufferSize() const override;
    unsigned GetParamsBufferSize() const override;

    /**
     * @deprecated
     */
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kClipRank> &input,
                                   const Tensor<OffsetBuffer, kClipRank> &output,
                                   const OffsetBuffer &encoded_params,
                                   const OffsetBuffer &ctrl_buffer)  override;

    mli_status AttachBufferOffsets(const OffsetBuffer& input,
                                   const OffsetBuffer& output,
                                   const OffsetBuffer& encoded_params,
                                   const OffsetBuffer& descr) override;

private:
    TensorIterator<OffsetBuffer, kClipRank, kClipIterRank> m_input;
    TensorIterator<OffsetBuffer, kClipRank, kClipIterRank> m_output;
    OffsetBuffer m_encoded_params;

    uint32_t m_input_buffer_size;
    uint32_t m_output_buffer_size;

    lib_mli::PlatformDescription m_pd;
};

class Permute_CS : public lib_mli::Permute_CS {
public:
    Permute_CS(const lib_mli::PlatformDescription pd,
               const TensorIterator<NoBuffer, kPermuteRank, kPermuteIterRank> in,
               const PermuteOpConfig &cfg,
               const TensorIterator<NoBuffer, kPermuteRank, kPermuteIterRank> out);

    mli_status GetKernelPrivateData(void *kernel_private_data_buffer) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

    mli_status AttachBufferOffsets( const OffsetBuffer &input,
                                    const OffsetBuffer &output,
                                    const OffsetBuffer &ctrl_buffer) override;

private:
    PermuteOpConfig m_cfg;
    TensorIterator<OffsetBuffer, kPermuteRank, kPermuteIterRank> m_in;
    TensorIterator<OffsetBuffer, kPermuteRank, kPermuteIterRank> m_out;

    uint32_t m_input_buffer_size;
    uint32_t m_output_buffer_size;

    lib_mli::PlatformDescription m_pd;
};

class Prelu_CS : public lib_mli::Prelu_CS {
public:
    /**
     * @brief Constructor to create a PReLU compiler support object.
     *
     * This constructor can be used to create a PReLU compiler support
     * object. This kernel computes values of the output tensor scaled by 
     * positive scale and shifted by positive shift if the input value is 
     * greater than the input bias,
     * Otherwise It will apply negative scale and negative shift
     * for all values in the desired axis of the input tensor 
     *
     * @param pd [IN] Platform description
     * @param input [IN] Input tensor (full shape)
     * @param cfg [IN] PreluOpConfig structure
     * @param output [OUT] Output tensor (tile shape)
     */
    Prelu_CS(const lib_mli::PlatformDescription pd,
             const TensorIterator<NoBuffer, 4, 4> &input,
             const PreluOpConfig &cfg,
             const TensorIterator<NoBuffer, 4, 4> &output);

    /**
     * @brief Constructor to create a PReLU compiler support object.
     *
     * This constructor can be used to create a PReLU compiler support
     * object. This kernel computes values of the output tensor scaled by 
     * positive scale and shifted by positive shift if the input value is 
     * greater than the input bias,
     * Otherwise It will apply negative scale and negative shift
     * for all values in the desired axis of the input tensor 
     *
     * @param pd        [IN]  Platform description
     * @param input     [IN]  Input tensor (full shape)
     * @param cfg       [IN]  PreluOpConfig structure
     * @param enc_param [IN]  Encoded parameters tensor
     * @param output    [OUT] Output tensor (tile shape)
     */
    Prelu_CS(const lib_mli::PlatformDescription pd,
             const TensorIterator<NoBuffer, kPreluRank, kPreluIterRank> &input,
             const PreluOpConfig &cfg,
             const TensorIterator<NoBuffer, kPreluRank, kPreluIterRank> &output);

    mli_status EncodeParams(Tensor<Buffer, kPreluParamRank> &bias,
                            Tensor<Buffer, kPreluParamRank> &posscale,
                            Tensor<Buffer, kPreluParamRank> &negscale,
                            Tensor<Buffer, kPreluParamRank> &posshift,
                            Tensor<Buffer, kPreluParamRank> &negshift,
                            Tensor<Buffer, kPreluParamRank> &asymm,
                            Buffer &encoded_params) override;

    unsigned GetEncodedParamsSize() override;
    unsigned GetParamsBufferSize() override;

    // From CompilerGenericInterface
    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

    mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                   const OffsetBuffer &output,
                                   const OffsetBuffer &params,
                                   const OffsetBuffer &ctrl_buffer) override;
private:
    PreluOpConfig m_config;
    TensorIterator<OffsetBuffer, kPreluRank, kPreluIterRank> m_input;
    TensorIterator<OffsetBuffer, kPreluRank, kPreluIterRank> m_output;

    OffsetBuffer m_encoded_params;
    uint32_t m_encoded_params_buffer_size;

    lib_mli::PlatformDescription m_pd;
};

class MatMul_CS : public lib_mli::MatMul_CS {
public:

    MatMul_CS();

    /**
     * @brief Constructor to create an MatMul compiler support object.
     *
     * This constructor can be used to create MAtMul compiler support
     * object. This kernel computes matrix multiplication 
     * of the two input tensors, inputs must be 2D.
     *
     * @param pd                [I] Platform description
     * @param in_left           [I] First Input tensorIterator
     * @param in_right          [I] Second Input tensorIterator
     * @param output            [O] Output tensorIterator
     */
    MatMul_CS(const lib_mli::PlatformDescription &pd,
              const TensorIterator<NoBuffer, kMatMulRank, kMatMulIterRank> &in_left,
              const TensorIterator<NoBuffer, kMatMulRank, kMatMulIterRank> &in_right,
              const TensorIterator<NoBuffer, kMatMulRank, kMatMulIterRank> &output);

    // From CompilerGenericInterface
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    mli_status AttachBufferOffsets(const OffsetBuffer &input_left,
                                   const OffsetBuffer &input_right,
                                   const OffsetBuffer &output,
                                   const OffsetBuffer &encoded_params,
                                   const OffsetBuffer &ctrl_buffer) override;
    // From MatMul_CS
    mli_status EncodeParams(const Buffer &in_bias1, 
                            const Buffer &in_bias2,
                            Buffer &encoded_params) override; 
    unsigned GetEncodedParamsSize() const override;

private:

    TensorIterator<OffsetBuffer, kMatMulRank, kMatMulIterRank> m_in_left;
    TensorIterator<OffsetBuffer, kMatMulRank, kMatMulIterRank> m_in_right;
    TensorIterator<OffsetBuffer, kMatMulRank, kMatMulIterRank> m_output;

    OffsetBuffer  m_encoded_params;
    uint32_t m_encoded_params_buffer_size;

    lib_mli::PlatformDescription m_pd;
};

class MoveBroadcast_CS : public lib_mli::MoveBroadcast_CS {
 public:
    /**
     * @brief constructor to create an iterating move object and then broadcast its pixels
     *
     * This constructor can be used to create a data movement operations which 
     * allows copy data from one location to another with broadcasting. 
     * e.g duplicating data to fill wider shape data with narrowed shape data. 
     *
     *
     * @param src [I] source tensor iterator
     * @param dst [I] destination tensor iterator
     */
    MoveBroadcast_CS(const lib_mli::PlatformDescription pd,
                     const TensorIterator<NoBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank> &src,
                     const TensorIterator<NoBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank> &dst);

    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    mli_status GetKernelPrivateData(void *kernel_private_data_buffer) override;

    mli_status AttachBufferOffsets(const OffsetBuffer &src,
                                   const OffsetBuffer &dst,
                                   const OffsetBuffer &ctrl_buffer) override;

private:
    lib_mli::PlatformDescription m_pd;

    TensorIterator<OffsetBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank> m_src;
    TensorIterator<OffsetBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank> m_dst;
};

class ResizeBilinear_CS : public lib_mli::ResizeBilinear_CS {
public:
    /**
     * @brief Constructor to create a ResizeBilinear compiler support object.
     *
     * This constructor can be used to create a ResizeBilinear compiler support
     * object. This kernel computes each value of the output tensor as the interpolation
     * of the nearest 4 values of the input tensor for each (H * W) plane using bilinear method
     *
     * @param pd  [I] Platform description
     * @param in  [I] Input tensor iterator
     * @param cfg [I] ResizeOpConfig structure
     * @param out [I] Output tensor iterator
     */
    ResizeBilinear_CS(const lib_mli::PlatformDescription pd,
                      const TensorIterator<NoBuffer, kResizeBilinearRank, kResizeBilinearIterRank> &in,
                      const ResizeOpConfig &cfg,
                      const TensorIterator<NoBuffer, kResizeBilinearRank, kResizeBilinearIterRank> &out);


    mli_status GetKernelPrivateData(void *kernel_private_data_buffer ) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

    mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                   const OffsetBuffer &output,
                                   const OffsetBuffer &ctrl_buffer) override;

private:
    ResizeOpConfig m_cfg;
    TensorIterator<OffsetBuffer, kResizeBilinearRank, kResizeBilinearIterRank> m_in;
    TensorIterator<OffsetBuffer, kResizeBilinearRank, kResizeBilinearIterRank> m_out;

    lib_mli::PlatformDescription m_pd;
};

} // namespace ref

#endif // _MLI_REF_COMPILER_API_HPP_
