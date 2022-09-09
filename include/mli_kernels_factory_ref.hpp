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

class KernelsFactory : public lib_mli::KernelsFactory {
public:

    KernelsFactory(const lib_mli::PlatformDescription pd): m_pd(pd) {}

    uint32_t Conv2d_CS_GetSize() const override { return sizeof(lib_ref::Conv2d_CS); }

    /**
     * @deprecated
     * Be carefull - conv2d I/O tensors of rank 4 are deprecated - new interfaces use rank 5 
     */
    lib_mli::Conv2d_CS* Conv2d_CS(void *kernel_buffer,
                                  const Tensor<NoBuffer, 4> input_shape,
                                  const Tensor<NoBuffer, 5> weights,
                                  const Conv2DConfig &cfg,
                                  const Tensor<NoBuffer, 4> output_tile_shape) override {
        return new(kernel_buffer) lib_ref::Conv2d_CS(m_pd, input_shape, weights, cfg, output_tile_shape);
    }

    lib_mli::Conv2d_CS* Conv2d_CS(void* kernel_buffer,
                                  const TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank>& input,
                                  const TensorIterator<NoBuffer, kConvWRank, kConvWIterRank>& weights,
                                  const TensorIterator<NoBuffer, kConvZPRank, kConvZPIterRank>& weights_zp,
                                  const Conv2DConfig& cfg,
                                  const TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank>& output) override {
        return new(kernel_buffer) lib_ref::Conv2d_CS(m_pd, input, weights, weights_zp, cfg, output);
    }

    uint32_t Prelu_CS_GetSize() const override { return sizeof(lib_ref::Prelu_CS); }

    lib_mli::Prelu_CS* Prelu_CS(void *kernel_buffer,
                                const TensorIterator<NoBuffer, kPreluRank, kPreluIterRank> &input,
                                const PreluOpConfig &cfg,
                                const TensorIterator<NoBuffer, kPreluRank, kPreluIterRank> &output,
                                int groups) override {
        return new(kernel_buffer) lib_ref::Prelu_CS(m_pd, input, cfg, output);
    }

    uint32_t Move_CS_GetSize() const override { return sizeof(lib_ref::Move_CS); }

    lib_mli::Move_CS* Move_CS(void *kernel_buffer,
                              const Tensor<NoBuffer, kMoveRank> src,
                              const Tensor<NoBuffer, kMoveRank> dst,
                              const lib_mli::MoveDataDirection data_dir)
                              override {
        return new(kernel_buffer) lib_ref::Move_CS(m_pd, src, dst);
    }

    lib_mli::Move_CS* Move_CS(void *kernel_buffer,
                              const Tensor<NoBuffer, kMoveRank> src,
                              const IteratorCfg<kMoveIterRank> src_cfg,
                              const Tensor<NoBuffer, kMoveRank> dst,
                              const IteratorCfg<kMoveIterRank> dst_cfg,
                              const lib_mli::MoveDataDirection data_dir)
                              override {
      return new (kernel_buffer) lib_ref::Move_CS(m_pd, src, dst, src_cfg, dst_cfg);
    }

    lib_mli::Move_CS* Move_CS(void *kernel_buffer,
                              const TensorIterator<NoBuffer, kMoveRank, kMoveIterRank> &src,
                              const TensorIterator<NoBuffer, kMoveRank, kMoveIterRank> &dst,
                              const lib_mli::MoveDataDirection data_dir)
                              override {
      return nullptr; /* new (kernel_buffer) lib_ref::Move_CS(m_pd, src, dst); */
    }

    uint32_t Add_CS_GetSize() const override { return sizeof(lib_ref::Add_CS); }

    lib_mli::Add_CS* Add_CS(void *kernel_buffer,
                            const Tensor<NoBuffer, 4> in_left,
                            const Tensor<NoBuffer, 4> in_right,
                            const Tensor<NoBuffer, 4> output_tile_shape) override {
        return new(kernel_buffer) lib_ref::Add_CS(m_pd, in_left, in_right, output_tile_shape);
    }

    uint32_t Sub_CS_GetSize() const override { return sizeof(lib_ref::Sub_CS); }

    lib_mli::Sub_CS* Sub_CS(void *kernel_buffer,
                            const Tensor<NoBuffer, 4> in_left,
                            const Tensor<NoBuffer, 4> in_right,
                            const Tensor<NoBuffer, 4> output_tile_shape) override {
        return new(kernel_buffer) lib_ref::Sub_CS(m_pd, in_left, in_right, output_tile_shape);
    }

    uint32_t Mul_CS_GetSize() const override { return sizeof(lib_ref::Mul_CS); }

    lib_mli::Mul_CS* Mul_CS(void *kernel_buffer,
                            const Tensor<NoBuffer, 4> in_left,
                            const Tensor<NoBuffer, 4> in_right,
                            const Tensor<NoBuffer, 4> output_tile_shape) override {
        return new(kernel_buffer) lib_ref::Mul_CS(m_pd, in_left, in_right, output_tile_shape);
    }

    uint32_t Max_CS_GetSize() const override { return sizeof(lib_ref::Max_CS); }

    lib_mli::Max_CS* Max_CS(void *kernel_buffer,
                            const Tensor<NoBuffer, 4> in_left,
                            const Tensor<NoBuffer, 4> in_right,
                            const Tensor<NoBuffer, 4> output_tile_shape) override {
        return new(kernel_buffer) lib_ref::Max_CS(m_pd, in_left, in_right, output_tile_shape);
    }

    uint32_t Min_CS_GetSize() const override { return sizeof(lib_ref::Min_CS); }

    lib_mli::Min_CS* Min_CS(void *kernel_buffer,
                            const Tensor<NoBuffer, 4> in_left,
                            const Tensor<NoBuffer, 4> in_right,
                            const Tensor<NoBuffer, 4> output_tile_shape) override {
        return new(kernel_buffer) lib_ref::Min_CS(m_pd, in_left, in_right, output_tile_shape);
    }

    uint32_t MaxPool2D_CS_GetSize() const override { return sizeof(lib_ref::MaxPool2D_CS); }

    /**
     * @deprecated
     */
    lib_mli::MaxPool2D_CS* MaxPool2D_CS(void *kernel_buffer,
                                        const Tensor<NoBuffer, kMaxpoolRank> in,
                                        const PoolOpConfig &cfg,
                                        const Tensor<NoBuffer, kMaxpoolRank> output_tile_shape)
                                        override {
        return new(kernel_buffer) lib_ref::MaxPool2D_CS(m_pd, in, cfg, output_tile_shape);
    }

    lib_mli::MaxPool2D_CS* MaxPool2D_CS(void* kernel_buffer,
                                        const TensorIterator<NoBuffer, kMaxpoolRank, kMaxpoolIterRank>& in,
                                        const PoolOpConfig& cfg,
                                        const TensorIterator<NoBuffer, kMaxpoolRank, kMaxpoolIterRank>& out)
                                        override {
        return new(kernel_buffer) lib_ref::MaxPool2D_CS(m_pd, in, cfg, out);
    }

    uint32_t SumPool2D_CS_GetSize() const override { return sizeof(lib_ref::SumPool2D_CS); }

    lib_mli::SumPool2D_CS* SumPool2D_CS(void *kernel_buffer,
                                        const Tensor<NoBuffer, 4> in,
                                        const PoolOpConfig &cfg,
                                        const Tensor<NoBuffer, 4> output_tile_shape)
                                        override {
        return new(kernel_buffer) lib_ref::SumPool2D_CS(m_pd, in, cfg, output_tile_shape);
    }

    uint32_t DepthwiseConv2d_CS_GetSize() const override { return sizeof(lib_ref::DepthwiseConv2d_CS); }

    /**
     * @deprecated
     */
    lib_mli::DepthwiseConv2d_CS* DepthwiseConv2d_CS(void *kernel_buffer,
                                                    const Tensor<NoBuffer, kDepthwiseIORank> in,
                                                    const Tensor<NoBuffer, kDepthwiseWRank> weights,
                                                    const DwConv2DConfig &cfg,
                                                    const Tensor<NoBuffer, kDepthwiseIORank> output_tile_shape) override {
        return new(kernel_buffer) lib_ref::DepthwiseConv2d_CS(m_pd, in, weights, cfg, output_tile_shape);
    }

    lib_mli::DepthwiseConv2d_CS* DepthwiseConv2d_CS(void* kernel_buffer,
                                                    const TensorIterator<NoBuffer, kDepthwiseIORank, kDepthwiseIOIterRank>& input,
                                                    const TensorIterator<NoBuffer, kDepthwiseWRank, kDepthwiseWIterRank>& weights,
                                                    const TensorIterator<NoBuffer, kDepthwiseZPRank, kDepthwiseZPIterRank>& weights_zp,
                                                    const DwConv2DConfig& cfg,
                                                    const TensorIterator<NoBuffer, kDepthwiseIORank, kDepthwiseIOIterRank>& output) override {
        return new(kernel_buffer) lib_ref::DepthwiseConv2d_CS(m_pd, input, weights, weights_zp, cfg, output);
    }

     uint32_t FullyConnected_CS_GetSize() const override { return sizeof(lib_ref:: FullyConnected_CS); }

    lib_mli:: FullyConnected_CS* FullyConnected_CS(void *kernel_buffer,
                                                   const Tensor<NoBuffer, 2> in,
                                                   const Tensor<NoBuffer, 2> weights,
                                                   const Tensor<NoBuffer, 2> output_tile_shape) override {
        return new(kernel_buffer) lib_ref::FullyConnected_CS(m_pd, in, weights, output_tile_shape);
    }
    lib_mli:: FullyConnected_CS* FullyConnected_CS(void *kernel_buffer,
                                                   const Tensor<NoBuffer, 2> in,
                                                   const Tensor<NoBuffer, 2> weights,
                                                   const Tensor<NoBuffer, 1> wtszp,
                                                   const Tensor<NoBuffer, 2> output_tile_shape) override {
        return new(kernel_buffer) lib_ref::FullyConnected_CS(m_pd, in, weights, wtszp, output_tile_shape);
    }


    uint32_t Rescale_CS_GetSize() const override { return sizeof(lib_ref::Rescale_CS); }

    /**
     * @deprecated
     */
    lib_mli::Rescale_CS* Rescale_CS(void *kernel_buffer,
                                    const Tensor<NoBuffer, kRescaleRank>& input_shape,
                                    const RescaleConfig &cfg,
                                    const Tensor<NoBuffer, kRescaleRank>& output_tile_shape) override {
        return new(kernel_buffer) lib_ref::Rescale_CS(m_pd, input_shape, cfg, output_tile_shape);
    }
    uint32_t TableBuiltin_CS_GetSize() const override { return 0;/*return sizeof(lib_ref::TableBuiltin_CS);*/ }

    lib_mli::TableBuiltin_CS* TableBuiltin_CS(void *kernel_buffer,
                                              const TensorIterator<NoBuffer, kTableBuiltinIORank, kTableBuiltinIOIterRank> &in,
                                              const TableBuiltinConfig &cfg,
                                              const TensorIterator<NoBuffer, kTableBuiltinIORank, kTableBuiltinIOIterRank> &out) override { return nullptr;
        /*return new(kernel_buffer) lib_ref::TableBuiltin_CS(m_pd, input_shape, cfg,   );*/
    }

    lib_mli::Rescale_CS* Rescale_CS(void* kernel_buffer,
                                    const TensorIterator<NoBuffer, kRescaleRank, kRescaleIterRank>& input,
                                    const RescaleConfig& cfg,
                                    const TensorIterator<NoBuffer, kRescaleRank, kRescaleIterRank>& output) override {
      return new(kernel_buffer) lib_ref::Rescale_CS(m_pd, input, cfg, output);
    }

    uint32_t ReduceMax_CS_GetSize() const override { return sizeof(lib_ref::ReduceMax_CS); }

    lib_mli::ReduceMax_CS* ReduceMax_CS(void *kernel_buffer,
                                        const TensorIterator<NoBuffer, kReduceMaxRank, kReduceMaxIterRank> &in,
                                        const ReduceOpConfig &cfg,
                                        const TensorIterator<NoBuffer, kReduceMaxRank, kReduceMaxIterRank> &out) override {
        return new(kernel_buffer) lib_ref::ReduceMax_CS(m_pd, in, cfg, out);
    }

    // TODO: to be removed after support IensorIterator
    lib_mli::ReduceMax_CS* ReduceMax_CS(void *kernel_buffer,
                                        const Tensor<NoBuffer, kReduceMaxRank> &input_shape,
                                        const ReduceOpConfig &cfg,
                                        const Tensor<NoBuffer, kReduceMaxRank> &output_tile_shape) override {
        return new(kernel_buffer) lib_ref::ReduceMax_CS(m_pd, input_shape, cfg, output_tile_shape);
    }

    uint32_t ReduceSum_CS_GetSize() const override { return sizeof(lib_ref::ReduceSum_CS); }

    lib_mli::ReduceSum_CS* ReduceSum_CS(void *kernel_buffer,
                                        const TensorIterator<NoBuffer, kReduceSumRank, kReduceSumIterRank> &in,
                                        const ReduceOpConfig &cfg,
                                        const TensorIterator<NoBuffer, kReduceSumRank, kReduceSumIterRank> &out) override {
        return new(kernel_buffer) lib_ref::ReduceSum_CS(m_pd, in, cfg, out);
    }

    uint32_t Clip_CS_GetSize() const override { return sizeof(lib_ref::Clip_CS); }

    lib_mli::Clip_CS* Clip_CS(void *kernel_buffer,
                              const Tensor<NoBuffer, kClipRank>& input_shape,
                              const Tensor<NoBuffer, kClipRank>& output_tile_shape) override {
        return new(kernel_buffer) lib_ref::Clip_CS(m_pd, input_shape, output_tile_shape);
    }

    lib_mli::Clip_CS* Clip_CS(void* kernel_buffer,
                              const TensorIterator<NoBuffer, kClipRank, kClipIterRank>& input,
                              const TensorIterator<NoBuffer, kClipRank, kClipIterRank>& output) override {
        return new(kernel_buffer) lib_ref::Clip_CS(m_pd, input, output);
    }
    
    uint32_t ArgMax_CS_GetSize() const override { return 0; /*sizeof(lib_ref::ArgMax_CS);*/ }

    lib_mli::ArgMax_CS* ArgMax_CS(void *kernel_buffer,
                                  const TensorIterator<NoBuffer, kArgMaxInRank, kArgMaxInIterRank> in,
                                  const ArgMaxConfig &cfg,
                                  const TensorIterator<NoBuffer, kArgMaxOutRank, kArgMaxOutIterRank> out) override {
        return nullptr;/*new(kernel_buffer) lib_ref::ArgMax_CS(m_pd, input_shape, cfg, output_tile_shape);*/
    }

    uint32_t TransposeConv2D_CS_GetSize() const override { return sizeof(lib_ref::TransposeConv2D_CS); }

    lib_mli::TransposeConv2D_CS* TransposeConv2D_CS(
        void *kernel_buffer,
        const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> &input,
        const TensorIterator<NoBuffer, kTransposeConvWRank, kTransposeConvWIterRank> &weights,
        const TensorIterator<NoBuffer, kTransposeConvZPRank, kTransposeConvZPIterRank>& weights_zp,
        const TransposeConv2DConfig &cfg,
        const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> &output) override {
        return new (kernel_buffer)lib_ref::TransposeConv2D_CS(m_pd, input, weights, weights_zp, cfg, output);
    }

    uint32_t Permute_CS_GetSize() const override { return sizeof(lib_ref::Permute_CS); }

    lib_mli::Permute_CS* Permute_CS(void *kernel_buffer,
                                    const TensorIterator<NoBuffer, kPermuteRank, kPermuteIterRank> in,
                                    const PermuteOpConfig &cfg,
                                    const TensorIterator<NoBuffer, kPermuteRank, kPermuteIterRank> out) override { 
        return new(kernel_buffer) lib_ref::Permute_CS(m_pd, in, cfg, out); 
    }
    
    uint32_t MatMul_CS_GetSize() const override { return 0 /*sizeof(lib_ref::MatMul_CS)*/; }

    lib_mli::MatMul_CS* MatMul_CS(void *kernel_buffer,
                            const TensorIterator<NoBuffer, kMatMulRank, kMatMulIterRank> &in_left,
                            const TensorIterator<NoBuffer, kMatMulRank, kMatMulIterRank> &in_right,
                            const TensorIterator<NoBuffer, kMatMulRank, kMatMulIterRank> &output) override {
       /* return new(kernel_buffer) lib_ref::MatMul_CS(m_pd, in_left, in_right, output);*/
       return nullptr;
    }

    uint32_t MoveBroadcast_CS_GetSize() const override { return 0 /* sizeof(lib_ref::MoveBroadcast_CS)*/; }

    lib_mli::MoveBroadcast_CS* MoveBroadcast_CS(void *kernel_buffer,
                                                const TensorIterator<NoBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank> &src,
                                                const TensorIterator<NoBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank> &dst,
                                                const lib_mli::MoveDataDirection data_dir) override {
      return nullptr; // new (kernel_buffer) lib_ref::MoveBroadcast_CS(m_pd, src, dst);
    }


    uint32_t ResizeBilinear_CS_GetSize() const override { return 0 /*sizeof(lib_ref::ResizeBilinear_CS) */; }

    lib_mli::ResizeBilinear_CS* ResizeBilinear_CS(void *kernel_buffer,
                                                  const TensorIterator<NoBuffer, kResizeBilinearRank, kResizeBilinearIterRank> &in,
                                                  const ResizeOpConfig &cfg,
                                                  const TensorIterator<NoBuffer, kResizeBilinearRank, kResizeBilinearIterRank> &out) override { 
        /* return new(kernel_buffer) lib_ref::ResizeBilinear_CS(m_pd, in, cfg, out); */
        return nullptr;
    }

private:
    lib_mli::PlatformDescription m_pd;
};

} // namespace snps_arc::metaware::mli::ref

#endif
