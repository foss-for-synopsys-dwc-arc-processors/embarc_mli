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
    virtual uint32_t Conv2d_CS_GetSize() const { return 0; }

    virtual uint32_t Prelu_CS_GetSize() const { return 0; }

    virtual uint32_t Move_CS_GetSize() const { return 0; }

    virtual uint32_t MaxPool2D_CS_GetSize() const { return 0; }

    virtual uint32_t SumPool2D_CS_GetSize() const { return 0; }

    virtual uint32_t DepthwiseConv2d_CS_GetSize() const { return 0; }

    virtual uint32_t FullyConnected_CS_GetSize() const { return 0; }

    virtual uint32_t Rescale_CS_GetSize() const { return 0; }

    virtual uint32_t Clip_CS_GetSize() const { return 0; }

    virtual uint32_t Add_CS_GetSize() const { return 0; }

    virtual uint32_t Sub_CS_GetSize() const { return 0; }

    virtual uint32_t Mul_CS_GetSize() const { return 0; }

    virtual uint32_t Max_CS_GetSize() const { return 0; }

    virtual uint32_t Min_CS_GetSize() const { return 0; }

    virtual lib_mli::Conv2d_CS* Conv2d_CS(void *kernel_buffer,
                                          const Tensor<NoBuffer, 4> input_shape,
                                          const Tensor<NoBuffer, 5> weights,
                                          const Conv2DConfig &cfg,
                                          const Tensor<NoBuffer, 4> output_tile_shape) { return nullptr; }

    virtual lib_mli::Prelu_CS* Prelu_CS(void *kernel_buffer,
                                        const Tensor<NoBuffer, 4> input_shape,
                                        const Tensor<NoBuffer, 4> output_tile_shape,
                                        int groups) { return nullptr; }

    virtual lib_mli::Move_CS *Move_CS(void *kernel_buffer,
                                      const Tensor<NoBuffer, lib_mli::Move_CS::kMaxRank> src,
                                      const Tensor<NoBuffer, lib_mli::Move_CS::kMaxRank> dst,
                                      const lib_mli::MoveDataDirection data_dir) {
      return nullptr;
    }

    virtual lib_mli::Move_CS *Move_CS(void *kernel_buffer,
                                      const Tensor<NoBuffer, lib_mli::Move_CS::kMaxRank> src,
                                      const IteratorCfg<lib_mli::Move_CS::kMaxRank> src_cfg,
                                      const Tensor<NoBuffer, lib_mli::Move_CS::kMaxRank> dst,
                                      const IteratorCfg<lib_mli::Move_CS::kMaxRank> dst_cfg,
                                      const lib_mli::MoveDataDirection data_dir) {
      return nullptr;
    }

    virtual lib_mli::MaxPool2D_CS* MaxPool2D_CS(void *kernel_buffer,
                                        const Tensor<NoBuffer, 4> in, // input fmap width, height, channels, batch size
                                        const PoolOpConfig &cfg,
                                        const Tensor<NoBuffer, 4> output_tile_shape) {
      return nullptr;
    } // output tile width, height, ch, groups

    virtual lib_mli::SumPool2D_CS* SumPool2D_CS(void *kernel_buffer,
                                                const Tensor<NoBuffer, 4> in,
                                                const PoolOpConfig &cfg,
                                                const Tensor<NoBuffer, 4> output_tile_shape) { return nullptr; }

    virtual lib_mli::DepthwiseConv2d_CS* DepthwiseConv2d_CS(void *kernel_buffer,
                                                            const Tensor<NoBuffer, 4> in,
                                                            const Tensor<NoBuffer, 3> weights,
                                                            const DwConv2DConfig &cfg,
                                                            const Tensor<NoBuffer, 4> output_tile_shape) { return nullptr; }

    virtual lib_mli::FullyConnected_CS* FullyConnected_CS(void *kernel_buffer,
                                                          const Tensor<NoBuffer, 2> in,
                                                          const Tensor<NoBuffer, 2> weights,
                                                          const Tensor<NoBuffer, 2> output_tile_shape) { return nullptr; }

    virtual lib_mli::Rescale_CS* Rescale_CS(void *kernel_buffer,
                                            const Tensor<NoBuffer, 4> input,
                                            const Tensor<NoBuffer, 4> output) { return nullptr; }

    virtual lib_mli::Clip_CS* Clip_CS(void *kernel_buffer,
                                      const Tensor<NoBuffer, 4> input,
                                      const Tensor<NoBuffer, 4> output) { return nullptr; }

    virtual lib_mli::Add_CS* Add_CS(void *kernel_buffer,
                                    const Tensor<NoBuffer, 4> input_left,
                                    const Tensor<NoBuffer, 4> input_right,
                                    const Tensor<NoBuffer, 4> output_tile_shape) { return nullptr; }

    virtual lib_mli::Sub_CS* Sub_CS(void *kernel_buffer,
                                    const Tensor<NoBuffer, 4> input_left,
                                    const Tensor<NoBuffer, 4> input_right,
                                    const Tensor<NoBuffer, 4> output) { return nullptr; }

    virtual lib_mli::Mul_CS* Mul_CS(void *kernel_buffer,
                                    const Tensor<NoBuffer, 4> input_left,
                                    const Tensor<NoBuffer, 4> input_right,
                                    const Tensor<NoBuffer, 4> output) { return nullptr; }

    virtual lib_mli::Max_CS* Max_CS(void *kernel_buffer,
                                    const Tensor<NoBuffer, 4> input_left,
                                    const Tensor<NoBuffer, 4> input_right,
                                    const Tensor<NoBuffer, 4> output) { return nullptr; }

    virtual lib_mli::Min_CS* Min_CS(void *kernel_buffer,
                                    const Tensor<NoBuffer, 4> input_left,
                                    const Tensor<NoBuffer, 4> input_right,
                                    const Tensor<NoBuffer, 4> output) { return nullptr; }
};

} // namespace snps_arc::metaware::mli

#endif
