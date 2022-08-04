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
     * @param pd [IN] Platform description
     * @param in [IN] input tensor (full shape)
     * @param weights [IN] weights tensor (full shape)
     * @param cfg [IN] Conv2DConfig structure
     * @param output_tile_shape [OUT] output tensor (tile shape)
     */
    Conv2d_CS(const lib_mli::PlatformDescription pd,
              const Tensor<NoBuffer, 4> &in, /**< layout: BHWC */
              const Tensor<NoBuffer, 5> &weights, /**< layout: GKyKxCiCo */
              const Conv2DConfig &cfg,
              const Tensor<NoBuffer, 4> &output_tile_shape /**< layout: BHWC */);

    mli_status EncodeWeights(Tensor<Buffer, 5> &weights,
                             Buffer &encoded_weights,
                             compression_mode_t mode = compression_mode_t::Uncompressed) override;

    unsigned GetEncodedWeightsSize() override;

    mli_status EncodeInpZeroPts(Tensor<Buffer, 1> &inpzeropts,
                                Buffer &encoded_inpzeropts) override;

    unsigned GetEncodedInpZeroPtsSize() override;

    mli_status EncodeWtsZeroPts(Tensor<Buffer, 1> &wtszeropts,
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
    unsigned GetDataBufferSize() override;

    mli_status AttachBufferOffsets(Tensor<OffsetBuffer, 4> &input,
                                   Tensor<OffsetBuffer, 4> &output,
                                   OffsetBuffer &weights,
                                   OffsetBuffer &inpzeropts,
                                   OffsetBuffer &wtszeropts,
                                   OffsetBuffer &metadata) override;

    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

    mli_status SetIterators(uint32_t output_total_size[4],
                            uint32_t iteration_order[4],
                            uint32_t input_first_inc[4],
                            uint32_t input_inc[4],
                            uint32_t output_first_inc[4],
                            uint32_t output_inc[4],
                            uint32_t weights_inc[4]) override;
private:
    void FillTilingParams(Conv2DPrivateData& pdata);

    // Input, weights, output tensors with offset buffer attached
    Tensor<OffsetBuffer, 4> m_input;

    Tensor<OffsetBuffer, 5> m_weights;
    Tensor<OffsetBuffer, 4> m_output;

    // encoded zp buffers for input and weights (optional for FX type)
    OffsetBuffer m_inpzp_buffer;
    OffsetBuffer m_wtszp_buffer;

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

    // Tile Parameters BHWC
    bool m_use_tiling;
    uint32_t m_tile_total_input_size[4];
    uint32_t m_tile_total_output_size[4];
    uint32_t m_tile_total_weights_size[4];  // KyKxCiCo
    uint32_t m_tile_iteration_order[4];
    uint32_t m_tile_input_first_inc[4];
    uint32_t m_tile_input_inc[4];
    uint32_t m_tile_output_first_inc[4];
    uint32_t m_tile_output_inc[4];
    uint32_t m_tile_weights_inc[4];
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
    unsigned GetDataBufferSize() override;

    mli_status AttachBufferOffsets(Tensor<OffsetBuffer, 4> &input,
                                   Tensor<OffsetBuffer, 4> &output,
                                   OffsetBuffer &weights,
                                   OffsetBuffer &inpzeropts,
                                   OffsetBuffer &wtszeropts,
                                   OffsetBuffer &descr) override;

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
                       const TensorIterator<NoBuffer, /* tensorRank = */ 4, /* iterRank = */ 4> &input,
                       const TensorIterator<NoBuffer, /* tensorRank = */ 4, /* iterRank = */ 5> &weights,
                       const TransposeConv2DConfig &cfg,
                       const TensorIterator<NoBuffer, /* tensorRank = */ 4, /* iterRank = */ 4> &output);

    mli_status EncodeWeights(Tensor<Buffer, 5> &weights, Buffer &encoded_weights,
                             compression_mode_t mode = compression_mode_t::Uncompressed) override;

    unsigned GetEncodedWeightsSize() const override;

    mli_status EncodeInpZeroPts(Tensor<Buffer, 1> &inpzeropts,
                                Buffer &encoded_inpzeropts) override;

    unsigned GetEncodedInpZeroPtsSize() const override;

    mli_status EncodeWtsZeroPts(Tensor<Buffer, 1> &wtszeropts,
                                Buffer &encoded_wtszeropts) override;

    unsigned GetEncodedWtsZeroPtsSize() const override;

    unsigned GetDataBufferSize() const override;

    mli_status AttachBufferOffsets(OffsetBuffer &input,
                                   OffsetBuffer &output,
                                   OffsetBuffer &weights,
                                   OffsetBuffer &inpzeropts,
                                   OffsetBuffer &wtszeropts,
                                   OffsetBuffer &descr) override;

    mli_status GetKernelPrivateData(void *kernel_private_data_buffer) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

private:
    // Input, weights, output tensors with offset buffer attached
    TensorIterator<OffsetBuffer, 4, 4> m_input;
    TensorIterator<OffsetBuffer, 4, 5> m_weights;
    TensorIterator<OffsetBuffer, 4, 4> m_output;

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
     * @brief Constructor to create an MaxPool2D compiler support object.
     *
     * This constructor can be used to create a Max Pooling 2D compiler support
     * object. This kernel computes each value of the output tensor as the maximum 
     * of all values in the related perception area of a single channel of the input tensor.
     *
     * @param pd [I] Platform description
     * @param in [I] Input tensor (full shape)
     * @param cfg [I] PoolOpConfig structure
     * @param output_tile_shape [O] Output tensor (tile shape)
     */
    MaxPool2D_CS(const lib_mli::PlatformDescription pd,
                 const Tensor<NoBuffer, 4> in, /**< layout: BHWC */
                 const PoolOpConfig &cfg,
                 const Tensor<NoBuffer, 4> output_tile_shape); /**< layout: BHWC */

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
    unsigned GetDataBufferSize() const override;

    mli_status GetKernelPrivateData(void *kernel_private_data_buffer) override;

    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input,
                                   const Tensor<OffsetBuffer, 4> &output,
                                   const OffsetBuffer &data) override;

    mli_status SetIterators(uint32_t output_total_size[4],
                            uint32_t iteration_order[4],
                            uint32_t input_first_inc[4],
                            uint32_t input_inc[4],
                            uint32_t output_first_inc[4],
                            uint32_t output_inc[4]) override;

private:
    void FillTilingParams(Pool2DPrivateData& private_data);

    Tensor<OffsetBuffer, 4> m_in;
    Tensor<OffsetBuffer, 4> m_output;

    PoolOpConfig m_config;

    uint32_t m_input_buffer_size;
    uint32_t m_output_buffer_size;

    lib_mli::PlatformDescription m_pd;

    // Tile Parameters BHWC
    bool m_use_tiling;
    uint32_t m_tile_total_output_size[4];
    uint32_t m_tile_iteration_order[4];
    uint32_t m_tile_input_first_inc[4];
    uint32_t m_tile_input_inc[4];
    uint32_t m_tile_output_first_inc[4];
    uint32_t m_tile_output_inc[4];
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
                                   const OffsetBuffer &data) override;

    // From SumPool2D_CS
    unsigned GetInputBufferSize() const override;
    unsigned GetOutputBufferSize() const override;
    unsigned GetDataBufferSize() const override;

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
    unsigned GetDataBufferSize() const override;

    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 2> &input,
                                   const Tensor<OffsetBuffer, 2> &output,
                                   const OffsetBuffer &weights,
                                   const OffsetBuffer &wtszeropts,
                                   const OffsetBuffer &descr) override;

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

class ReduceMax_CS : public lib_mli::ReduceMax_CS {
public:
    ReduceMax_CS(const lib_mli::PlatformDescription pd,
                 const Tensor<NoBuffer, 4> input_shape,
                 const ReduceOpConfig &cfg,
                 const Tensor<NoBuffer, 4> output_tile_shape);

    // From ReduceMax_CS
    unsigned GetInputBufferSize() const override;
    unsigned GetOutputBufferSize() const override;
    unsigned GetDataBufferSize() const override;

    mli_status GetKernelPrivateData(void *kernel_private_data_buffer ) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

    mli_status AttachBufferOffsets( const Tensor<OffsetBuffer, 4> &input,
                                    const Tensor<OffsetBuffer, 4> &output,
                                    const OffsetBuffer &metadata) override;

private:
    ReduceOpConfig m_cfg;
    Tensor<OffsetBuffer, 4> m_in;
    Tensor<OffsetBuffer, 4> m_out;

    uint32_t m_input_buffer_size;
    uint32_t m_output_buffer_size;

    lib_mli::PlatformDescription m_pd;
};

class Rescale_CS : public lib_mli::Rescale_CS {
public:
    Rescale_CS(const lib_mli::PlatformDescription pd,
               const Tensor<NoBuffer, 4> input_shape,
               const RescaleConfig &cfg,
               const Tensor<NoBuffer, 4> output_tile_shape);
    // From Rescale_CS
    unsigned GetInputBufferSize() const override;
    unsigned GetOutputBufferSize() const override;
    unsigned GetDataBufferSize() const override;
    unsigned GetEncodedParamsSize() const override;
    mli_status EncodeParams(const Tensor<Buffer, 1> &in_bias,
                            const Tensor<Buffer, 1> &out_bias,
                            const Tensor<Buffer, 1> &scale,
                            const Tensor<Buffer, 1> &shift,
                            Buffer &encoded_params) override;
    mli_status GetKernelPrivateData( void *kernel_private_data_buffer ) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input,
                                   const Tensor<OffsetBuffer, 4> &output,
                                   const OffsetBuffer &encoded_params,
                                   const OffsetBuffer &metadata) override;

    mli_status SetIterators(uint32_t output_total_size[4],
                            uint32_t iteration_order[4],
                            uint32_t output_first_inc[4],
                            uint32_t output_inc[4]) override;

private:
    RescaleConfig m_config;

    Tensor<OffsetBuffer, 4> m_input;
    Tensor<OffsetBuffer, 4> m_output;

    OffsetBuffer m_encoded_params;

    uint32_t m_input_buffer_size;
    uint32_t m_output_buffer_size;
    uint32_t m_params_elem_num;

    // sizes in bytes
    uint32_t m_encoded_params_buffer_size;

    lib_mli::PlatformDescription m_pd;

    // Tile Parameters BHWC
    bool m_use_tiling;
    uint32_t m_tile_total_output_size[4];
    uint32_t m_tile_iteration_order[4];
    uint32_t m_tile_output_first_inc[4];
    uint32_t m_tile_output_inc[4];
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
  unsigned GetDataBufferSize() const override;

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
                                   const OffsetBuffer &data) override;

    // From Add_CS
    unsigned GetInputLeftBufferSize() override;
    unsigned GetInputRightBufferSize() override;
    unsigned GetOutputBufferSize() override;
    unsigned GetDataBufferSize() override;

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
                                   const OffsetBuffer &data) override;

    // From Sub_CS
    unsigned GetInputLeftBufferSize() override;
    unsigned GetInputRightBufferSize() override;
    unsigned GetOutputBufferSize() override;
    unsigned GetDataBufferSize() override;

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
                                   const OffsetBuffer &data) override;

    // From Mul_CS
    unsigned GetInputLeftBufferSize() override;
    unsigned GetInputRightBufferSize() override;
    unsigned GetOutputBufferSize() override;
    unsigned GetDataBufferSize() override;

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
                                   const OffsetBuffer &data) override;

    // From Max_CS
    unsigned GetInputLeftBufferSize() override;
    unsigned GetInputRightBufferSize() override;
    unsigned GetOutputBufferSize() override;
    unsigned GetDataBufferSize() override;

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
                                   const OffsetBuffer &data) override;

    // From Min_CS
    unsigned GetInputLeftBufferSize() override;
    unsigned GetInputRightBufferSize() override;
    unsigned GetOutputBufferSize() override;
    unsigned GetDataBufferSize() override;

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
    Clip_CS(const lib_mli::PlatformDescription pd,
            const Tensor<NoBuffer, kMaxRank> &input,
            const Tensor<NoBuffer, kMaxRank> &output);

    unsigned GetRuntimeObjectSize() const override;

    mli_status GetKernelPrivateData(void *kernel_private_data_buffer) override;
    unsigned GetKernelPrivateDataSize() const override;

    mli_status EncodeParams(Tensor<Buffer, 1> &min_val,
                            Tensor<Buffer, 1> &max_val,
                            Buffer &encoded_params) override;

    unsigned GetEncodedParamsSize() const override;

    unsigned GetInputBufferSize() const override;
    unsigned GetOutputBufferSize() const override;
    unsigned GetParamsBufferSize() const override;
    unsigned GetDataBufferSize() const override;

    /**
     * @brief Methods to set buffer offsets
     *
     */
    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, kMaxRank> &input,
                                   const Tensor<OffsetBuffer, kMaxRank> &output,
                                   const OffsetBuffer &encoded_params,
                                   const OffsetBuffer &metadata)  override;


    mli_status SetIterators(uint32_t output_total_size[4],
                            uint32_t iteration_order[4],
                            uint32_t output_first_inc[4],
                            uint32_t output_inc[4]) override;

private:
    Tensor<OffsetBuffer, kMaxRank> m_input;
    Tensor<OffsetBuffer, kMaxRank> m_output;

    OffsetBuffer m_encoded_params;

    OffsetBuffer m_min;
    OffsetBuffer m_max;

    uint32_t m_input_buffer_size;
    uint32_t m_output_buffer_size;
    uint32_t m_encoded_params_buffer_size;

    uint32_t m_params_elem_num;

    lib_mli::PlatformDescription m_pd;

    // Tile Parameters BHWC
    bool m_use_tiling;
    uint32_t m_tile_total_output_size[4];
    uint32_t m_tile_iteration_order[4];
    uint32_t m_tile_output_first_inc[4];
    uint32_t m_tile_output_inc[4];
};

} // namespace ref

#endif // _MLI_REF_COMPILER_API_HPP_
