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

namespace snps_arc::metaware::mli{

using lib_mli::Tensor;
using lib_mli::OffsetBuffer;

class KernelsFactory{
public:
    virtual uint32_t Conv2d_CS_GetSize() const = 0;

    virtual uint32_t Prelu_CS_GetSize() const = 0;

    virtual uint32_t Move_CS_GetSize() const = 0;

    virtual lib_mli::Conv2d_CS* Conv2d_CS(void *kernel_buffer,
                                          const Tensor<OffsetBuffer, 4> input_shape,
                                          const Tensor<OffsetBuffer, 5> weights,
                                          const mli_conv2d_cfg *cfg,
                                          const Tensor<OffsetBuffer, 4> output_tile_shape) = 0;
    
    virtual lib_mli::Prelu_CS* Prelu_CS(void *kernel_buffer,
                                        const Tensor<OffsetBuffer, 4> input_shape,
                                        const Tensor<OffsetBuffer, 4> output_tile_shape,
                                        int groups) = 0;

    virtual lib_mli::Move_CS* Move_CS(void *kernel_buffer,
                                      const Tensor<OffsetBuffer, lib_mli::Move_CS::kMaxRank> src,
                                      const Tensor<OffsetBuffer, lib_mli::Move_CS::kMaxRank> dst) = 0;

    virtual lib_mli::Move_CS* Move_CS(void *kernel_buffer,
                                      const Tensor<OffsetBuffer, lib_mli::Move_CS::kMaxRank> src,
                                      const IteratorCfg<lib_mli::Move_CS::kMaxRank> src_cfg,
                                      const Tensor<OffsetBuffer, lib_mli::Move_CS::kMaxRank> dst,
                                      const IteratorCfg<lib_mli::Move_CS::kMaxRank> dst_cfg) = 0;                 
};
}
#endif
