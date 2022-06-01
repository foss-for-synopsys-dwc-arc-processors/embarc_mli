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
    OffsetBuffer m_metadata;

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

    // TODO: add destructor if need

private:
    uint32_t m_io_elem_size;

    uint32_t m_input_buffer_size;
    uint32_t m_output_buffer_size;

    uint32_t m_input_offset;
    uint32_t m_output_offset;
    uint32_t m_descr_offset;

    uint32_t m_input_mem_id;
    uint32_t m_output_mem_id;
    uint32_t m_descr_mem_id;

    uint32_t m_input_shape[4];
    uint32_t m_output_shape[4];

    int32_t m_input_stride[4];
    int32_t m_output_stride[4];

    uint8_t m_kernel_width;
    uint8_t m_kernel_height;
    uint8_t m_stride_width;
    uint8_t m_stride_height;
    uint8_t m_padding_left;
    uint8_t m_padding_right;
    uint8_t m_padding_top;
    uint8_t m_padding_bottom;

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

} // namespace ref

#endif // _MLI_REF_COMPILER_API_HPP_
