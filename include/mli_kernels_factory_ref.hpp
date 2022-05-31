/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KERNELS_FACTORY_REF_HPP_
#define _MLI_KERNELS_FACTORY_REF_HPP_

#include <new>

#include "mli_kernels_factory.hpp"
#include "mli_platform_desc.hpp"
#include "mli_ref_compiler_api.hpp"

namespace lib_mli = ::snps_arc::metaware::mli;
namespace lib_ref = ::snps_arc::metaware::mli::ref;

namespace snps_arc::metaware::mli::ref {

using lib_mli::Tensor;
using lib_mli::NoBuffer;

class KernelsFactory : public lib_mli::KernelsFactory{
public:

    KernelsFactory(const lib_mli::PlatformDescription pd): m_pd(pd) {}

    uint32_t Conv2d_CS_GetSize() const override { return 0; /*sizeof(lib_ref::Conv2d_CS);*/ }

    lib_mli::Conv2d_CS* Conv2d_CS(void *kernel_buffer,
                                  const Tensor<NoBuffer, 4> input_shape,
                                  const Tensor<NoBuffer, 5> weights,
                                  const Conv2DConfig &cfg,
                                  const Tensor<NoBuffer, 4> output_tile_shape) override {
        //return new(kernel_buffer) lib_ref::Conv2d_CS(m_pd, input_shape, weights, cfg, output_tile_shape);
        return nullptr;
    }

    uint32_t Prelu_CS_GetSize() const override { return 0 /*sizeof(lib_ref::Prelu_CS)*/; }

    lib_mli::Prelu_CS* Prelu_CS(void *kernel_buffer,
                                const Tensor<NoBuffer, 4> input_shape,
                                const Tensor<NoBuffer, 4> output_tile_shape,
                                int groups) override {
        //return new(kernel_buffer) lib_ref::Prelu_CS(m_pd, input_shape, output_tile_shape, groups);
        return nullptr;
    }

    uint32_t Move_CS_GetSize() const override { return sizeof(lib_ref::Move_CS); }


    lib_mli::Move_CS* Move_CS(void *kernel_buffer,
                              const Tensor<NoBuffer, lib_mli::Move_CS::kMaxRank> src,
                              const Tensor<NoBuffer, lib_mli::Move_CS::kMaxRank> dst) 
                              override {
        return new(kernel_buffer) lib_ref::Move_CS(m_pd, src, dst);
    }

    lib_mli::Move_CS* Move_CS(void *kernel_buffer,
                              const Tensor<NoBuffer, lib_mli::Move_CS::kMaxRank> src,
                              const IteratorCfg<lib_mli::Move_CS::kMaxRank> src_cfg,
                              const Tensor<NoBuffer, lib_mli::Move_CS::kMaxRank> dst,
                              const IteratorCfg<lib_mli::Move_CS::kMaxRank> dst_cfg) 
                              override {
      return new (kernel_buffer) lib_ref::Move_CS(m_pd, src, dst, src_cfg, dst_cfg);
    }

    uint32_t MaxPool2D_CS_GetSize() const override { return sizeof(lib_ref::MaxPool2D_CS); }

    lib_mli::MaxPool2D_CS* MaxPool2D_CS(void *kernel_buffer,
                                        const Tensor<NoBuffer, 4> in, // input fmap width, height, channels, batch size
                                        const PoolOpConfig &cfg,
                                        const Tensor<NoBuffer, 4> output_tile_shape) // output tile width, height, ch, groups
                                        override {
        return new(kernel_buffer) lib_ref::MaxPool2D_CS(m_pd, in, cfg, output_tile_shape);
    }

    uint32_t DepthwiseConv2d_CS_GetSize() const override { return sizeof(lib_ref::DepthwiseConv2d_CS); }

    lib_mli::DepthwiseConv2d_CS* DepthwiseConv2d_CS(void *kernel_buffer,
                                                    const Tensor<NoBuffer, 4> in,
                                                    const Tensor<NoBuffer, 3> weights,
                                                    const DwConv2DConfig &cfg,
                                                    const Tensor<NoBuffer, 4> output_tile_shape) override {
        return new(kernel_buffer) lib_ref::DepthwiseConv2d_CS(m_pd, in, weights, cfg, output_tile_shape);
    }

private:
    lib_mli::PlatformDescription m_pd;

};

} // namespace snps_arc::metaware::mli::ref

#endif
