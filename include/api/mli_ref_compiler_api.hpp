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
    unsigned GetKernelPrivateDataSize() override;
    unsigned GetRuntimeObjectSize() override;

};

} // namespace ref

#endif // _MLI_REF_COMPILER_API_HPP_
