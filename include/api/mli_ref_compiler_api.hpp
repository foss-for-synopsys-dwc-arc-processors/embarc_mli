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

class Conv2d_CS : public lib_mli::Conv2d_CS {
public:
    /**
     * @brief Constructor of the Conv2d_CS object
     *
     */
    Conv2d_CS(const lib_mli::PlatformDescription pd,
              const Tensor<NoBuffer, 4> &in,
              const Tensor<NoBuffer, 5> &weights,
              const Conv2DConfig &cfg,
              const Tensor<NoBuffer, 4> &output_tile_shape);

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

    unsigned GetInputBufferSize() override;
    unsigned GetOutputBufferSize() override;
    unsigned GetWeightsBufferSize() override;
    unsigned GetZeroPointBufferSize() override;
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
    Tensor<OffsetBuffer, 4> m_in;
    Tensor<OffsetBuffer, 5> m_weights;
    Tensor<OffsetBuffer, 4> m_output;

    Conv2DConfig m_config;

    OffsetBuffer m_input_zp;
    OffsetBuffer m_weights_zp;
    OffsetBuffer m_metadata;

    uint32_t m_input_buffer_size;
    uint32_t m_weights_buffer_size;
    uint32_t m_output_buffer_size;

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
    Tensor<OffsetBuffer, 4> m_in;
    Tensor<OffsetBuffer, 3> m_weights;
    Tensor<OffsetBuffer, 4> m_output;

    DwConv2DConfig m_config;

    OffsetBuffer m_input_zp;
    OffsetBuffer m_weights_zp;

    uint32_t m_input_buffer_size;
    uint32_t m_weights_buffer_size;
    uint32_t m_output_buffer_size;

    lib_mli::PlatformDescription m_pd;
};

class MaxPool2D_CS : public lib_mli::MaxPool2D_CS {
public:

    MaxPool2D_CS(const lib_mli::PlatformDescription pd,
                 const Tensor<NoBuffer, 4> in, // input fmap width, height, channels, batch size
                 const PoolOpConfig &cfg,
                 const Tensor<NoBuffer, 4> output_tile_shape); // output tile width, height, ch, groups

    // From CompilerGenericInterface
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;

    mli_status AttachBufferOffsets(const Tensor<OffsetBuffer, 4> &input,
                                   const Tensor<OffsetBuffer, 4> &output,
                                   const OffsetBuffer &data) override;

    // From MaxPool2D_CS
    unsigned GetInputBufferSize() const override;
    unsigned GetOutputBufferSize() const override;
    unsigned GetDataBufferSize() const override;

    mli_status SetIterators(uint32_t total_output_size[4], uint32_t iteration_order[4],
                            uint32_t first_tile_size[4], uint32_t tile_size[4],
                            uint32_t input_first_inc[4], uint32_t input_inc[4],
                            uint32_t output_first_inc[4], uint32_t output_inc[4]) override;
    
    //TODO: add destructor if need


private:
    Tensor<OffsetBuffer, 4> m_in;
    Tensor<OffsetBuffer, 4> m_output;

    uint32_t m_input_buffer_size;
    uint32_t m_output_buffer_size;

    uint8_t m_kernel_width;
    uint8_t m_kernel_height;
    uint8_t m_stride_width;
    uint8_t m_stride_height;
    uint8_t m_padding_left;
    uint8_t m_padding_right;
    uint8_t m_padding_top;
    uint8_t m_padding_bottom;

    lib_mli::PlatformDescription m_pd;

    // Tile Parameters BHWC
    uint32_t m_tile_total_output_size[4];
    uint32_t m_tile_iteration_order[4];
    uint32_t m_tile_first_size[4];
    uint32_t m_tile_size[4];
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
    OffsetBuffer m_metadata;

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

    mli_status EncodeWeights(const Tensor<Buffer, 2> &weights,
                             Buffer &encoded_weights) override;

    unsigned GetEncodedWeightsSize() const override;

    mli_status EncodeInpZeroPts(const Tensor<Buffer, 1> &inpzeropts,
                                Buffer &encoded_inpzeropts) override;

    unsigned GetEncodedInpZeroPtsSize() const override;

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
                                   const OffsetBuffer &inpzeropts,
                                   const OffsetBuffer &wtszeropts,
                                   const OffsetBuffer &descr) override;

    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

private:
    Tensor<OffsetBuffer, 2> m_in;
    Tensor<OffsetBuffer, 2> m_weights;
    Tensor<OffsetBuffer, 2> m_output;

    OffsetBuffer m_input_zp;
    OffsetBuffer m_weights_zp;

    uint32_t m_input_buffer_size;
    uint32_t m_weights_buffer_size;
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
    unsigned GetParamsBufferSize() const override;
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
    OffsetBuffer m_metadata;

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
    OffsetBuffer m_metadata;

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
    OffsetBuffer m_metadata;

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
    OffsetBuffer m_metadata;

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
    OffsetBuffer m_metadata;

    uint32_t m_in_left_buffer_size;
    uint32_t m_in_right_buffer_size;
    uint32_t m_output_buffer_size;

    bool m_is_left_scalar{true};
    bool m_is_right_scalar{true};

    lib_mli::PlatformDescription m_pd;
};

} // namespace ref

#endif // _MLI_REF_COMPILER_API_HPP_
