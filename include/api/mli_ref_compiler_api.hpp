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

class DepthwiseConv2d_CS : public lib_mli::DepthwiseConv2d_CS {
public:
    /**
     * @brief Constructor of the DepthwiseConv2d_CS object
     *
     */
    DepthwiseConv2d_CS(const Tensor<OffsetBuffer, 4> &in,
                       const Tensor<OffsetBuffer, 3> &weights,
                       const mli_conv2d_cfg &cfg,
                       const Tensor<OffsetBuffer, 4> &output_tile_shape);

    mli_status EncodeWeights(Tensor<Buffer, 3> &weights,
                            Buffer &encoded_weights, 
                            compression_mode_t mode = compression_mode_t::Uncompressed) override;

    unsigned GetEncodedWeightsSize() override;
    mli_status EncodeInpZeroPts(Tensor<Buffer, 1> &inpzeropts, 
                                Buffer &encoded_inpzeropts) override;
    unsigned GetEncodedInpZeroPtsSize() override;
    unsigned GetInputBufferSize() override;
    unsigned GetOutputBufferSize() override;
    unsigned GetWeightsBufferSize() override;
    unsigned GetDataBufferSize() override;

    mli_status AttachBufferOffsets(OffsetBuffer &input,
                                   OffsetBuffer &output,
                                   OffsetBuffer &weights,
                                   OffsetBuffer &padding,
                                   OffsetBuffer &descr) override;

    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;

};

class MaxPool2D_CS : public lib_mli::MaxPool2D_CS {
public:

    MaxPool2D_CS(const lib_mli::PlatformDescription pd,
                 const Tensor<OffsetBuffer, 4> in, // input fmap width, height, channels, batch size
                 const PoolOpConfig &cfg,
                 const Tensor<OffsetBuffer, 4> output_tile_shape); // output tile width, height, ch, groups

    // From CompilerGenericInterface
    unsigned GetKernelPrivateDataSize() const override;
    unsigned GetRuntimeObjectSize() const override;
    mli_status GetKernelPrivateData(void* kernel_private_data_buffer) override;
    mli_status AttachBufferOffsets(const OffsetBuffer &input,
                                   const OffsetBuffer &output,
                                   const OffsetBuffer &data) override;

    // From MaxPool2D_CS
    unsigned GetInputBufferSize() const override;
    unsigned GetOutputBufferSize() const override;
    unsigned GetDataBufferSize() const override;

    
    //TODO: add destructor if need

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

} // namespace ref

#endif // _MLI_REF_COMPILER_API_HPP_
