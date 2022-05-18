/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KERNELS_FACTORY_HPP_
#define _MLI_KERNELS_FACTORY_HPP_

#include "mli_compiler_api.hpp"

namespace lib_mli = ::snps_arc::metaware::mli;

namespace snps_arc::metaware::mli {

using lib_mli::Tensor;
using lib_mli::NoBuffer;

class KernelsFactory {
public:
    virtual uint32_t Conv2d_CS_GetSize() const = 0;

    virtual uint32_t Prelu_CS_GetSize() const = 0;

    virtual uint32_t Move_CS_GetSize() const = 0;

    virtual uint32_t MaxPool2D_CS_GetSize() const { return 0; }

    virtual uint32_t DepthwiseConv2d_CS_GetSize() const { return 0; }

    virtual lib_mli::Conv2d_CS* Conv2d_CS(void *kernel_buffer,
                                          const Tensor<NoBuffer, 4> input_shape,
                                          const Tensor<NoBuffer, 5> weights,
                                          const mli_conv2d_cfg *cfg,
                                          const Tensor<NoBuffer, 4> output_tile_shape) = 0;

    virtual lib_mli::Prelu_CS* Prelu_CS(void *kernel_buffer,
                                        const Tensor<NoBuffer, 4> input_shape,
                                        const Tensor<NoBuffer, 4> output_tile_shape,
                                        int groups) = 0;

    virtual lib_mli::Move_CS* Move_CS(void *kernel_buffer,
                                      const Tensor<NoBuffer, lib_mli::Move_CS::kMaxRank> src,
                                      const Tensor<NoBuffer, lib_mli::Move_CS::kMaxRank> dst) = 0;

    virtual lib_mli::Move_CS* Move_CS(void *kernel_buffer,
                                      const Tensor<NoBuffer, lib_mli::Move_CS::kMaxRank> src,
                                      const IteratorCfg<lib_mli::Move_CS::kMaxRank> src_cfg,
                                      const Tensor<NoBuffer, lib_mli::Move_CS::kMaxRank> dst,
                                      const IteratorCfg<lib_mli::Move_CS::kMaxRank> dst_cfg) = 0;

    virtual lib_mli::MaxPool2D_CS* MaxPool2D_CS(void *kernel_buffer,
                                                const Tensor<NoBuffer, 4> in, // input fmap width, height, channels, batch size
                                                const PoolOpConfig &cfg,
                                                const Tensor<NoBuffer, 4> output_tile_shape) {return nullptr;};// output tile width, height, ch, groups

    virtual lib_mli::DepthwiseConv2d_CS* DepthwiseConv2d_CS(void *kernel_buffer,
                                                            const Tensor<NoBuffer, 4> in,
                                                            const Tensor<NoBuffer, 3> weights,
                                                            const DwConv2DConfig &cfg,
                                                            const Tensor<NoBuffer, 4> output_tile_shape) { return nullptr; }

};

} // namespace snps_arc::metaware::mli

#endif
