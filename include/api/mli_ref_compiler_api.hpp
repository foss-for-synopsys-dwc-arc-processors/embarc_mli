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
     * @param pd [IN] Platform description
     * @param in [IN] input tensor (full shape, BHWC layout)
     * @param weights [IN] weights tensor (full shape, GKyKxCiCo layout)
     * @param cfg [IN] Conv2DConfig structure
     * @param output_tile_shape [OUT] output tensor (tile shape, BHWC layout)
     */
    Conv2d_CS(const lib_mli::PlatformDescription pd,
              const Tensor<NoBuffer, kConvIORank>& in,
              const Tensor<NoBuffer, kConvWRank>& weights,
              const Conv2DConfig& cfg,
              const Tensor<NoBuffer, kConvIORank>& output_tile_shape);

    /**
      * @brief Constructor to create an Conv2d_CS compiler support object.
      *
      * This constructor can be used to create a Convolution 2D compiler support
      * object. This kernel computes each value of the output tensor as the result of convolution operation
      * of all values in the related perception area of all channels of the input tensor.
      * of all values in the related perception area of a single channel of the input tensor.
      *
      * @param pd [IN] Platform description
      * @param input [IN] input TensorIterator (BHWC layout)
      * @param weights [IN] weights TensorIterator (GKyKxCiCo layout)
      * @param weights [IN] weights_zp TensorIterator
      * @param cfg [IN] Conv2DConfig structure
      * @param output [OUT] output TensorIterator (BHWC layout)
      */
    Conv2d_CS(const lib_mli::PlatformDescription pd,
              const TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank>& input,
              const TensorIterator<NoBuffer, kConvWRank, kConvWIterRank>& weights,
              const TensorIterator<NoBuffer, kConvZPRank, kConvZPIterRank>& weights_zp,
              const Conv2DConfig& cfg,
              const TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank>& output);

    mli_status EncodeWeights(Tensor<Buffer, kConvWRank> &weights,
                             Buffer &encoded_weights,
                             compression_mode_t mode = compression_mode_t::Uncompressed) override;

    unsigned GetEncodedWeightsSize() override;

    mli_status EncodeInpZeroPts(Tensor<Buffer, kConvZPRank> &inpzeropts,
                                Buffer &encoded_inpzeropts) override;

    unsigned GetEncodedInpZeroPtsSize() override;

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
     * @return Always returns zero for reference kernel.
     */

    /**
     * @deprecated
     */
    mli_status AttachBufferOffsets(Tensor<OffsetBuffer, kConvIORank> &input,
                                   Tensor<OffsetBuffer, kConvIORank> &output,
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

    /**
     * @deprecated
     */
    mli_status SetIterators(uint32_t output_total_size[kConvIORank],
                            uint32_t iteration_order[kConvIORank],
                            uint32_t input_first_inc[kConvIORank],
                            uint32_t input_inc[kConvIORank],
                            uint32_t output_first_inc[kConvIORank],
                            uint32_t output_inc[kConvIORank],
                            uint32_t weights_inc[kConvWRank]) override;
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
     *
     */
    DepthwiseConv2d_CS(const lib_mli::PlatformDescription pd,
                       const Tensor<NoBuffer, 4> &in,
                       const Tensor<NoBuffer, 3> &weights,
                       const DwConv2DConfig &cfg,
                       const Tensor<NoBuffer, 4> &output_tile_shape);

    mli_status EncodeWeights(Tensor<Buffer, 3> &weights,
                             Buffer &encoded_weights,
                             compression_mode_t mode = compression_mode_t::Uncompressed) override;

    unsigned GetEncodedWeightsSize() override;

    mli_status EncodeInpZeroPts(Tensor<Buffer, 1> &inpzeropts,
                                Buffer &encoded_inpzeropts) override;

    unsigned GetEncodedInpZeroPtsSize() override;

    mli_status EncodeWtsZeroPts(Tensor<Buffer, 1> &wtszeropts,
                                Buffer &encoded_wtszeropts) override;

    unsigned GetEncodedWtsZeroPtsSize() override;

    unsigned GetInputBufferSize() override;
    unsigned GetOutputBufferSize() override;
    unsigned GetWeightsBufferSize() override;

    mli_status AttachBufferOffsets(Tensor<OffsetBuffer, 4> &input,
                                   Tensor<OffsetBuffer, 4> &output,
                                   OffsetBuffer &weights,
                                   OffsetBuffer &inpzeropts,
                                   OffsetBuffer &wtszeropts,
                                   OffsetBuffer &ctrl_buffer) override;

    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

private:
    // Input, weights, output tensors with offset buffer attached
    Tensor<OffsetBuffer, 4> m_input;
    Tensor<OffsetBuffer, 3> m_weights;
    Tensor<OffsetBuffer, 4> m_output;

    // encoded zp buffers for input and weights (optional for FX type)
    OffsetBuffer m_inpzp_buffer;
    OffsetBuffer m_wtszp_buffer;

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
     *
     */
    TransposeConv2D_CS(const lib_mli::PlatformDescription pd,
                       const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> &input,
                       const TensorIterator<NoBuffer, kTransposeConvWRank, kTransposeConvWIterRank> &weights,
                       const TransposeConv2DConfig &cfg,
                       const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> &output);

    mli_status EncodeWeights(Tensor<Buffer, kTransposeConvWRank> &weights, Buffer &encoded_weights,
                             compression_mode_t mode = compression_mode_t::Uncompressed) override;

    unsigned GetEncodedWeightsSize() const override;

    mli_status EncodeInpZeroPts(Tensor<Buffer, kTransposeConvZPRank> &inpzeropts,
                                Buffer &encoded_inpzeropts) override;

    unsigned GetEncodedInpZeroPtsSize() const override;

    mli_status EncodeWtsZeroPts(Tensor<Buffer, kTransposeConvZPRank> &wtszeropts,
                                Buffer &encoded_wtszeropts) override;

    unsigned GetEncodedWtsZeroPtsSize() const override;

    mli_status AttachBufferOffsets(OffsetBuffer &input,
                                   OffsetBuffer &output,
                                   OffsetBuffer &weights,
                                   OffsetBuffer &inpzeropts,
                                   OffsetBuffer &wtszeropts,
                                   OffsetBuffer &ctrl_buffer) override;

    mli_status GetKernelPrivateData(void *kernel_private_data_buffer) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

private:
    // Input, weights, output tensors with offset buffer attached
    TensorIterator<OffsetBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> m_input;
    TensorIterator<OffsetBuffer, kTransposeConvWRank, kTransposeConvWIterRank> m_weights;
    TensorIterator<OffsetBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> m_output;

    // Encoded zp buffers for input and weights (optional for FX type)
    OffsetBuffer m_inpzp_buffer;
    OffsetBuffer m_wtszp_buffer;

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
                 const Tensor<NoBuffer, kMaxpoolRank> in,
                 const PoolOpConfig &cfg,
                 const Tensor<NoBuffer, kMaxpoolRank> output_tile_shape);

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
                 const TensorIterator<NoBuffer, kMaxpoolRank, kMaxpoolIterRank> in,
                 const PoolOpConfig& cfg,
                 const TensorIterator<NoBuffer, kMaxpoolRank, kMaxpoolIterRank> out);

    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    
    /**
     * Tensor buffer sizes could depend on the platform and/or parameters.
     * These functions can be used to query how much memory needs to be allocated for
     * the input, weights and output tensors.
     * Note, that these sizes are for full tensors, not tiles. 
     */
    unsigned GetInputBufferSize() const override;
    unsigned GetOutputBufferSize() const override;
    /**
     * @return Always returns zero for reference kernel.
     */

    mli_status GetKernelPrivateData(void *kernel_private_data_buffer) override;

    /**
     * @deprecated
     */
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kMaxpoolRank> &input,
                                   const Tensor<OffsetBuffer, kMaxpoolRank> &output,
                                   const OffsetBuffer &ctrl_buffer) override;

    mli_status AttachBufferOffsets(const OffsetBuffer& input,
                                   const OffsetBuffer& output,
                                   const OffsetBuffer& ctrl_buffer) override;

    /**
      * @deprecated
      */
    mli_status SetIterators(uint32_t output_total_size[kMaxpoolIterRank],
                            uint32_t iteration_order[kMaxpoolIterRank],
                            uint32_t input_first_inc[kMaxpoolIterRank],
                            uint32_t input_inc[kMaxpoolIterRank],
                            uint32_t output_first_inc[kMaxpoolIterRank],
                            uint32_t output_inc[kMaxpoolIterRank]) override;

private:
    TensorIterator<OffsetBuffer, kMaxpoolRank, kMaxpoolIterRank> m_input;
    TensorIterator<OffsetBuffer, kMaxpoolRank, kMaxpoolIterRank> m_output;

    PoolOpConfig m_config;

    uint32_t m_input_buffer_size;
    uint32_t m_output_buffer_size;

    lib_mli::PlatformDescription m_pd;
};

class SumPool2D_CS : public lib_mli::SumPool2D_CS {
public:

    SumPool2D_CS(const lib_mli::PlatformDescription pd,
                 const Tensor<NoBuffer, 4> in,
                 const PoolOpConfig &cfg,
                 const Tensor<NoBuffer, 4> output_tile_shape);

    // From CompilerGenericInterface
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input,
                                   const Tensor<OffsetBuffer, 4> &output,
                                   const OffsetBuffer &ctrl_buffer) override;

    // From SumPool2D_CS
    unsigned GetInputBufferSize() const override;
    unsigned GetOutputBufferSize() const override;

private:
    Tensor<OffsetBuffer, 4> m_in;
    Tensor<OffsetBuffer, 4> m_output;

    PoolOpConfig m_config;

    uint32_t m_input_buffer_size;
    uint32_t m_output_buffer_size;

    lib_mli::PlatformDescription m_pd;
};

class FullyConnected_CS : public lib_mli::FullyConnected_CS {
public:
    /**
     * @brief Constructor of the  FullyConnected_CS object
     *
     */
    FullyConnected_CS(const lib_mli::PlatformDescription pd,
                      const Tensor<NoBuffer, 2> &in,
                      const Tensor<NoBuffer, 2> &weights,
                      const Tensor<NoBuffer, 2> &output_tile_shape);

    FullyConnected_CS(const lib_mli::PlatformDescription pd,
                      const Tensor<NoBuffer, 2> &in,
                      const Tensor<NoBuffer, 2> &weights,
                      const Tensor<NoBuffer, 1> &wtszp,
                      const Tensor<NoBuffer, 2> &output_tile_shape);

    mli_status EncodeWeights(const Tensor<Buffer, 2> &weights,
                             Buffer &encoded_weights) override;

    unsigned GetEncodedWeightsSize() const override;

    mli_status EncodeWtsZeroPts(const Tensor<Buffer, 1> &wtszeropts,
                                Buffer &encoded_wtszeropts) override;

    unsigned GetEncodedWtsZeroPtsSize() const override;

    unsigned GetInputBufferSize() const override;
    unsigned GetOutputBufferSize() const override;
    unsigned GetWeightsBufferSize() const override;
    unsigned GetZeroPointBufferSize() const override;

    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 2> &input,
                                   const Tensor<OffsetBuffer, 2> &output,
                                   const OffsetBuffer &weights,
                                   const OffsetBuffer &wtszeropts,
                                   const OffsetBuffer &ctrl_buffer) override;

    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

private:
    lib_mli::PlatformDescription m_pd;
    Tensor<OffsetBuffer, 2> m_in;
    Tensor<OffsetBuffer, 2> m_weights;
    Tensor<OffsetBuffer, 1> m_wtszp;
    Tensor<OffsetBuffer, 2> m_output;

    OffsetBuffer m_weights_zp;

    uint32_t m_input_buffer_size;
    uint32_t m_weights_buffer_size;
    uint32_t m_wtszp_buffer_size;
    uint32_t m_output_buffer_size;

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
    ReduceMax_CS(const lib_mli::PlatformDescription pd,
                 const Tensor<NoBuffer, 4> input_shape,
                 const ReduceOpConfig &cfg,
                 const Tensor<NoBuffer, 4> output_tile_shape);

    // From ReduceMax_CS
    unsigned GetInputBufferSize() const override;
    unsigned GetOutputBufferSize() const override;
    
    mli_status GetKernelPrivateData(void *kernel_private_data_buffer ) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

    mli_status AttachBufferOffsets( const Tensor<OffsetBuffer, 4> &input,
                                    const Tensor<OffsetBuffer, 4> &output,
                                    const OffsetBuffer &ctrl_buffer) override;

private:
    ReduceOpConfig m_cfg;
    Tensor<OffsetBuffer, 4> m_in;
    Tensor<OffsetBuffer, 4> m_out;

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
                 const TensorIterator<NoBuffer, kReduceSumRank, kReduceSumIterRank> in,
                 const ReduceOpConfig &cfg,
                 const TensorIterator<NoBuffer, kReduceSumRank, kReduceSumIterRank> out);


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

    uint32_t m_input_buffer_size;
    uint32_t m_output_buffer_size;

    lib_mli::PlatformDescription m_pd;
};

class Rescale_CS : public lib_mli::Rescale_CS {
public:
    /**
     * @deprecated
     */
    Rescale_CS(const lib_mli::PlatformDescription pd,
               const Tensor<NoBuffer, kRescaleRank>& input_shape,
               const RescaleConfig &cfg,
               const Tensor<NoBuffer, kRescaleRank>& output_tile_shape);

    Rescale_CS(const lib_mli::PlatformDescription pd,
               const TensorIterator<NoBuffer, kRescaleRank, kRescaleIterRank>& input,
               const RescaleConfig& cfg,
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

    /**
     * @deprecated
     */
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kRescaleRank> &input,
                                   const Tensor<OffsetBuffer, kRescaleRank> &output,
                                   const OffsetBuffer &encoded_params,
                                   const OffsetBuffer &ctrl_buffer) override;

    mli_status AttachBufferOffsets(const OffsetBuffer& input,
                                   const OffsetBuffer& output,
                                   const OffsetBuffer& encoded_params,
                                   const OffsetBuffer& metadata) override;

    /**
     * @deprecated
     */
    mli_status SetIterators(uint32_t output_total_size[kRescaleIterRank],
                            uint32_t iteration_order[kRescaleIterRank],
                            uint32_t output_first_inc[kRescaleIterRank],
                            uint32_t output_inc[kRescaleIterRank]) override;

private:
    RescaleConfig m_config;

    TensorIterator<OffsetBuffer, kRescaleRank, kRescaleIterRank> m_input;
    TensorIterator<OffsetBuffer, kRescaleRank, kRescaleIterRank> m_output;

    OffsetBuffer m_encoded_params;

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
   * @brief constructor to create an iterating move object
   *
   * This constructor can be used to create a move object that can transfer the
   * data from one tensor to the other, in a tiled manner The shapes of both src
   * and dst tensors need to match. the offsets (aka mem_strides) can be
   * different.
   *
   * The function accepts tensors with a templated rank up to kMaxRank
   *
   * Separate iterator configs are needed for src and dst because the increments
   * can be different.
   * The cfg.size needs to be the same for src and dst. (as wel as the
   * cfg.start_size)
   *
   * TODO: decide if we want to put the IteratorCfg on the interface, or the
   * individual fields
   *
   * @param src [I] source tensor
   * @param dst [I] destination tensor
   * @param src_cfg [I] source iterator configuration
   * @param dst_cfg [I] destination iterator configuration
   */

  Move_CS(const lib_mli::PlatformDescription pd,
          const Tensor<NoBuffer, kMaxRank> src,
          const Tensor<NoBuffer, kMaxRank> dst,
          const IteratorCfg<kMaxRank> src_it_cfg = IteratorCfg<kMaxRank>(),
          const IteratorCfg<kMaxRank> dst_it_cfg = IteratorCfg<kMaxRank>());

  unsigned GetKernelPrivateDataSize() const override;
  unsigned GetRuntimeObjectSize() const override;
  mli_status GetKernelPrivateData(void *kernel_private_data_buffer) override;
  mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kMaxRank> &src,
                                 const Tensor<OffsetBuffer, kMaxRank> &dst) override;

  unsigned GetInputBufferSize() const override;
  unsigned GetOutputBufferSize() const override;

private:
  IteratorCfg<kMaxRank> m_src_cfg;
  IteratorCfg<kMaxRank> m_dst_cfg;

  lib_mli::PlatformDescription m_pd;

  Tensor<OffsetBuffer, kMaxRank> m_src;
  Tensor<OffsetBuffer, kMaxRank> m_dst;

  uint32_t m_src_rank;
  uint32_t m_dst_rank;

  uint32_t m_src_shape[kMaxRank];
  uint32_t m_dst_shape[kMaxRank];

  int32_t m_src_stride[kMaxRank];
  int32_t m_dst_stride[kMaxRank];
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
     * @param pd [I] Platform description
     * @param in_left [I] first Input tensor (full shape)
     * @param in_right [I] second Input tensor (full shape)
     * @param output_tile_shape [O] Output tensor (tile shape)
     */

    Add_CS(const lib_mli::PlatformDescription pd,
           const Tensor<NoBuffer, 4> in_left,
           const Tensor<NoBuffer, 4> in_right,
           const Tensor<NoBuffer, 4> output_tile_shape);

    // From CompilerGenericInterface
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input_left,
                                   const Tensor<OffsetBuffer, 4> &input_right,
                                   const Tensor<OffsetBuffer, 4> &output,
                                   const OffsetBuffer &ctrl_buffer) override;

    // From Add_CS
    unsigned GetInputLeftBufferSize() override;
    unsigned GetInputRightBufferSize() override;
    unsigned GetOutputBufferSize() override;

private:
    Tensor<OffsetBuffer, 4> m_in_left;
    Tensor<OffsetBuffer, 4> m_in_right;
    Tensor<OffsetBuffer, 4> m_output;

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
     * @param output_tile_shape [O] Output tensor (tile shape)
     */

    Sub_CS(const lib_mli::PlatformDescription pd,
           const Tensor<NoBuffer, 4> in_left,
           const Tensor<NoBuffer, 4> in_right,
           const Tensor<NoBuffer, 4> output_tile_shape);

    // From CompilerGenericInterface
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input_left,
                                   const Tensor<OffsetBuffer, 4> &input_right,
                                   const Tensor<OffsetBuffer, 4> &output,
                                   const OffsetBuffer &ctrl_buffer) override;

    // From Sub_CS
    unsigned GetInputLeftBufferSize() override;
    unsigned GetInputRightBufferSize() override;
    unsigned GetOutputBufferSize() override;

private:
    Tensor<OffsetBuffer, 4> m_in_left;
    Tensor<OffsetBuffer, 4> m_in_right;
    Tensor<OffsetBuffer, 4> m_output;

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

    Mul_CS(const lib_mli::PlatformDescription pd,
           const Tensor<NoBuffer, 4> in_left,
           const Tensor<NoBuffer, 4> in_right,
           const Tensor<NoBuffer, 4> output_tile_shape);

    // From CompilerGenericInterface
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input_left,
                                   const Tensor<OffsetBuffer, 4> &input_right,
                                   const Tensor<OffsetBuffer, 4> &output,
                                   const OffsetBuffer &ctrl_buffer) override;

    // From Mul_CS
    unsigned GetInputLeftBufferSize() override;
    unsigned GetInputRightBufferSize() override;
    unsigned GetOutputBufferSize() override;

private:
    Tensor<OffsetBuffer, 4> m_in_left;
    Tensor<OffsetBuffer, 4> m_in_right;
    Tensor<OffsetBuffer, 4> m_output;

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

    Max_CS(const lib_mli::PlatformDescription pd,
           const Tensor<NoBuffer, 4> in_left,
           const Tensor<NoBuffer, 4> in_right,
           const Tensor<NoBuffer, 4> output_tile_shape);

    // From CompilerGenericInterface
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input_left,
                                   const Tensor<OffsetBuffer, 4> &input_right,
                                   const Tensor<OffsetBuffer, 4> &output,
                                   const OffsetBuffer &ctrl_buffer) override;

    // From Max_CS
    unsigned GetInputLeftBufferSize() override;
    unsigned GetInputRightBufferSize() override;
    unsigned GetOutputBufferSize() override;

private:
    Tensor<OffsetBuffer, 4> m_in_left;
    Tensor<OffsetBuffer, 4> m_in_right;
    Tensor<OffsetBuffer, 4> m_output;

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

    Min_CS(const lib_mli::PlatformDescription pd,
           const Tensor<NoBuffer, 4> in_left,
           const Tensor<NoBuffer, 4> in_right,
           const Tensor<NoBuffer, 4> output_tile_shape);

    // From CompilerGenericInterface
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input_left,
                                   const Tensor<OffsetBuffer, 4> &input_right,
                                   const Tensor<OffsetBuffer, 4> &output,
                                   const OffsetBuffer &ctrl_buffer) override;

    // From Min_CS
    unsigned GetInputLeftBufferSize() override;
    unsigned GetInputRightBufferSize() override;
    unsigned GetOutputBufferSize() override;

private:
    Tensor<OffsetBuffer, 4> m_in_left;
    Tensor<OffsetBuffer, 4> m_in_right;
    Tensor<OffsetBuffer, 4> m_output;

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

    /**
     * @deprecated
     */
    mli_status SetIterators(uint32_t output_total_size[kClipIterRank],
                            uint32_t iteration_order[kClipIterRank],
                            uint32_t output_first_inc[kClipIterRank],
                            uint32_t output_inc[kClipIterRank]) override;

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


} // namespace ref

#endif // _MLI_REF_COMPILER_API_HPP_
