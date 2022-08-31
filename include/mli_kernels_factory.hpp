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

    virtual uint32_t ReduceMax_CS_GetSize() const { return 0; }
    
    virtual uint32_t TableBuiltin_CS_GetSize() const { return 0; }

    virtual uint32_t TransposeConv2D_CS_GetSize() const { return 0; }
    
    virtual uint32_t ReduceSum_CS_GetSize() const { return 0; }

    virtual uint32_t Permute_CS_GetSize() const { return 0; }

    /**
     * @brief Convolution 2D kernel Compiler Support interface factory
     * method
     *
     * @deprecated
     * 
     * @param kernel_buffer       [I] Pointer to the pre-allocated memory to store
     *                                kernel Compiler Support object
     * @param input_shape         [I] Tensor object containing input Tensor shape and
     *                                memory strides
     * @param weights             [I] Tensor object containing weights Tensor shape
     *                                and memory strides
     * @param cfg                 [I] Kernel configuration structure
     * @param output_tile_shape   [I] Tensor object containing output Tensor shape
     *                                and memory strides
     *
     * @return Convolution 2D kernel Compiler Support interface object
     */
   

    virtual uint32_t ArgMax_CS_GetSize() const { return 0; }

    virtual lib_mli::Conv2d_CS* Conv2d_CS(void *kernel_buffer,
                                          const Tensor<NoBuffer, kConvIORank> input_shape,          // BHWC        
                                          const Tensor<NoBuffer, kConvWRank> weights,               // GKyKxCiCo
                                          const Conv2DConfig &cfg,
                                          const Tensor<NoBuffer, kConvIORank> output_tile_shape) {  // BHWC
      return nullptr;
    }

    /**
     * @brief Convolution 2D kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param input         [I] TensorIterator object containing input Tensor shape and
     *                          memory strides and IteratorCfg
     * @param weights       [I] TensorIterator object containing weights Tensor shape
     *                          and memory strides and IteratorCfg
     * @param weights_zp    [I] TensorIterator object containing weight zp(s) array
     * @param cfg           [I] Kernel configuration structure
     * @param output        [I] TensorIterator object containing output Tensor shape
     *                          and memory strides and IteratorCfg
     *
     * @return Convolution 2D kernel Compiler Support interface object
     */
    virtual lib_mli::Conv2d_CS* Conv2d_CS(void* kernel_buffer,
                                          const TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank>& input,      // BHWC
                                          const TensorIterator<NoBuffer, kConvWRank, kConvWIterRank>& weights,      // GKyKxCiCo
                                          const TensorIterator<NoBuffer, kConvZPRank, kConvZPIterRank>& weights_zp,
                                          const Conv2DConfig& cfg,
                                          const TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank>& output) {   // BHWC
      return nullptr;
    }

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

    /**
     * @brief Maxpool 2D kernel Compiler Support interface factory
     * method
     *
     * @deprecated
     * 
     * @param kernel_buffer       [I] Pointer to the pre-allocated memory to store
     *                                kernel Compiler Support object
     * @param in                  [I] Tensor object containing input Tensor shape and
     *                                memory strides
     * @param cfg                 [I] Kernel configuration structure
     * @param output_tile_shape   [I] Tensor object containing output Tensor shape
     *                                and memory strides
     *
     * @return Maxpool 2D kernel Compiler Support interface object
     */
    virtual lib_mli::MaxPool2D_CS* MaxPool2D_CS(void *kernel_buffer,
                                                const Tensor<NoBuffer, kMaxpoolRank> in,                  // BHWC
                                                const PoolOpConfig &cfg,
                                                const Tensor<NoBuffer, kMaxpoolRank> output_tile_shape) { // BHWC
      return nullptr;
    } 

    /**
     * @brief Maxpool 2D kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param input         [I] TensorIterator object containing input Tensor shape and
     *                          memory strides and IteratorCfg
     * @param cfg           [I] Kernel configuration structure
     * @param output        [I] TensorIterator object containing output Tensor shape
     *                          and memory strides and IteratorCfg
     *
     * @return Maxpool 2D kernel Compiler Support interface object
     */
    virtual lib_mli::MaxPool2D_CS* MaxPool2D_CS(void* kernel_buffer,
                                                const TensorIterator<NoBuffer, kMaxpoolRank, kMaxpoolIterRank>& in,      // BHWC
                                                const PoolOpConfig& cfg,
                                                const TensorIterator<NoBuffer, kMaxpoolRank, kMaxpoolIterRank>& out) {   // BHWC
      return nullptr;
    } 

    virtual lib_mli::SumPool2D_CS* SumPool2D_CS(void *kernel_buffer,
                                                const Tensor<NoBuffer, 4> in,
                                                const PoolOpConfig &cfg,
                                                const Tensor<NoBuffer, 4> output_tile_shape) { return nullptr; }


    /**
      * @brief Depthwise Convolution 2D kernel Compiler Support interface factory
      * method
      *
      * @deprecated
      *
      * @param kernel_buffer       [I] Pointer to the pre-allocated memory to store
      *                                kernel Compiler Support object
      * @param in                  [I] Tensor object containing input Tensor shape and
      *                                memory strides
      * @param weights             [I] Tensor object containing weights Tensor shape
      *                                and memory strides
      * @param cfg                 [I] Kernel configuration structure
      * @param output_tile_shape   [I] Tensor object containing output Tensor shape
      *                                and memory strides
      *
      * @return Depthwise Convolution 2D kernel Compiler Support interface object
      */
    virtual lib_mli::DepthwiseConv2d_CS* DepthwiseConv2d_CS(void *kernel_buffer,
                                                            const Tensor<NoBuffer, kDepthwiseIORank> in,                  // BHWC
                                                            const Tensor<NoBuffer, kDepthwiseWRank> weights,                  // KiKoC
                                                            const DwConv2DConfig &cfg,
                                                            const Tensor<NoBuffer, kDepthwiseIORank> output_tile_shape) { // BHWC
      return nullptr;
    }

    /**
     * @brief Depthwise Convolution 2D kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param input         [I] TensorIterator object containing input Tensor shape and
     *                          memory strides and IteratorCfg
     * @param weights       [I] TensorIterator object containing weights Tensor shape
     *                          and memory strides and IteratorCfg
     * @param weights_zp    [I] TensorIterator object containing weight zp(s) array
     * @param cfg           [I] Kernel configuration structure
     * @param output        [I] TensorIterator object containing output Tensor shape
     *                          and memory strides and IteratorCfg
     *
     * @return Depthwise Convolution 2D kernel Compiler Support interface object
     */
    virtual lib_mli::DepthwiseConv2d_CS* DepthwiseConv2d_CS(void* kernel_buffer,
                                                            const TensorIterator<NoBuffer, kDepthwiseIORank, kDepthwiseIOIterRank>& input,    // BHWC
                                                            const TensorIterator<NoBuffer, kDepthwiseWRank, kDepthwiseWIterRank>& weights,            // KiKoC
                                                            const TensorIterator<NoBuffer, kDepthwiseZPRank, kDepthwiseZPIterRank>& weights_zp,       // C
                                                            const DwConv2DConfig& cfg,
                                                            const TensorIterator<NoBuffer, kDepthwiseIORank, kDepthwiseIOIterRank>& output) { // BHWC
      return nullptr;
    }

    virtual lib_mli::FullyConnected_CS* FullyConnected_CS(void *kernel_buffer,
                                                          const Tensor<NoBuffer, 2> in,
                                                          const Tensor<NoBuffer, 2> weights,
                                                          const Tensor<NoBuffer, 2> output_tile_shape) { return nullptr; }

    virtual lib_mli::FullyConnected_CS* FullyConnected_CS(void *kernel_buffer,
                                                          const Tensor<NoBuffer, 2> in,
                                                          const Tensor<NoBuffer, 2> weights,
                                                          const Tensor<NoBuffer, 1> wtszp,
                                                          const Tensor<NoBuffer, 2> output_tile_shape) { return nullptr; }

    /**
     * @brief Clip kernel Compiler Support interface factory
     * method
     *
     * @deprecated
     *
     * @param kernel_buffer       [I] Pointer to the pre-allocated memory to store
     *                                kernel Compiler Support object
     * @param input               [I] Tensor object containing input Tensor shape and
     *                                memory strides
     * @param output_tile_shape   [I] Tensor object containing output tile Tensor shape
     *                                and memory strides
     *
     * @return Clip kernel Compiler Support interface object
     */
    virtual lib_mli::Clip_CS* Clip_CS(void *kernel_buffer,
                                      const Tensor<NoBuffer, kClipRank>& input,
                                      const Tensor<NoBuffer, kClipRank>& output_tile_shape) {
      return nullptr;
    }

    /**
     * @brief Clip kernel Compiler Support interface factory
     * method
     * 
     * @param kernel_buffer       [I] Pointer to the pre-allocated memory to store
     *                                kernel Compiler Support object
     * @param input               [I] TensorIterator object containing input Tensor shape and
     *                                memory strides
     * @param output              [I] TensorIterator object containing output Tensor shape
     *                                and memory strides
     *
     * @return Clip kernel Compiler Support interface object
     */
    virtual lib_mli::Clip_CS* Clip_CS(void* kernel_buffer,
                                      const TensorIterator<NoBuffer, kClipRank, kClipIterRank>& input,
                                      const TensorIterator<NoBuffer, kClipRank, kClipIterRank>& output) {
      return nullptr;
    }

    /**
     * @brief Add kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param input_left    [I] Tensor object containing input1 Tensor shape and
     *                          memory strides
     * @param input_right   [I] Tensor object containing input2 Tensor shape and
     *                          memory strides
     * @param output        [I] Tensor object containing output Tensor shape
     *                          and memory strides
     *
     * @return Add kernel Compiler Support interface object
     */                                  

    virtual lib_mli::Add_CS* Add_CS(void *kernel_buffer,
                                    const Tensor<NoBuffer, 4> input_left,
                                    const Tensor<NoBuffer, 4> input_right,
                                    const Tensor<NoBuffer, 4> output_tile_shape) { return nullptr; }


    /**
     * @brief Sub kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param input_left    [I] Tensor object containing input1 Tensor shape and
     *                          memory strides
     * @param input_right    [I] Tensor object containing input2 Tensor shape and
     *                          memory strides
     * @param output        [I] Tensor object containing output Tensor shape
     *                          and memory strides
     *
     * @return Sub kernel Compiler Support interface object
     */                                

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

    /**
     * @brief Rescale kernel Compiler Support interface factory
     * method
     *
     * @deprecated
     *
     * @param kernel_buffer       [I] Pointer to the pre-allocated memory to store
     *                                kernel Compiler Support object
     * @param input_shape         [I] Tensor object containing input Tensor shape and
     *                                memory strides
     * @param cfg                 [I] Kernel configuration structure
     * @param output_tile_shape   [I] Tensor object containing output tile Tensor shape
     *                                and memory strides
     *
     * @return Rescale kernel Compiler Support interface object
     */
     virtual lib_mli::Rescale_CS* Rescale_CS(void *kernel_buffer,
                                            const Tensor<NoBuffer, kRescaleRank>& input_shape,
                                            const RescaleConfig &cfg,
                                            const Tensor<NoBuffer, kRescaleRank>& output_tile_shape) {
       return nullptr;
     }

    /**
     * @brief Clip kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer       [I] Pointer to the pre-allocated memory to store
     *                                kernel Compiler Support object
     * @param input               [I] TensorIterator object containing input Tensor shape and
     *                                memory strides
     * @param cfg                 [I] Kernel configuration structure
     * @param output              [I] TensorIterator object containing output Tensor shape
     *                                and memory strides
     *
     * @return Clip kernel Compiler Support interface object
     */
     virtual lib_mli::Rescale_CS* Rescale_CS(void* kernel_buffer,
                                            const TensorIterator<NoBuffer, kRescaleRank, kRescaleIterRank>& input,
                                            const RescaleConfig& cfg,
                                            const TensorIterator<NoBuffer, kRescaleRank, kRescaleIterRank>& output) {
       return nullptr;
     }

    virtual lib_mli::ReduceMax_CS* ReduceMax_CS(void *kernel_buffer,
                                                const Tensor<NoBuffer, 4> input_shape,
                                                const ReduceOpConfig &cfg,
                                                const Tensor<NoBuffer, 4> output_tile_shape) { return nullptr; }
    
    virtual lib_mli::ReduceSum_CS* ReduceSum_CS(void *kernel_buffer,
                                                const TensorIterator<NoBuffer, kReduceSumRank, kReduceSumIterRank> in,
                                                const ReduceOpConfig &cfg,
                                                const TensorIterator<NoBuffer, kReduceSumRank, kReduceSumIterRank> out) {return nullptr; }

    virtual lib_mli::ArgMax_CS* ArgMax_CS(void *kernel_buffer,
                                          const TensorIterator<NoBuffer, kArgMaxInRank, kArgMaxInIterRank> in,
                                          const ArgMaxConfig &cfg,
                                          const TensorIterator<NoBuffer, kArgMaxOutRank, kArgMaxOutIterRank> out) { return nullptr; }
                                          
    virtual lib_mli::TableBuiltin_CS* TableBuiltin_CS(void *kernel_buffer,
                                                      const TensorIterator<NoBuffer, kTableBuiltinIORank, kTableBuiltinIOIterRank> &in,
                                                      const TableBuiltinConfig &cfg,
                                                      const TensorIterator<NoBuffer, kTableBuiltinIORank, kTableBuiltinIOIterRank> &out) { return nullptr; }

    /**
     * @brief Transpose Convolution 2D kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param input         [I] TensorIterator object containing input Tensor shape and
     *                          memory strides and IteratorCfg
     * @param weights       [I] TensorIterator object containing weights Tensor shape
     *                          and memory strides and IteratorCfg
     * @param cfg           [I] Kernel configuration structure
     * @param output        [I] TensorIterator object containing output Tensor shape
     *                          and memory strides and IteratorCfg
     *
     * @return Transpose Convolution 2D kernel Compiler Support interface object
     */
    virtual lib_mli::TransposeConv2D_CS* TransposeConv2D_CS(
        void *kernel_buffer,
        const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> input,    // BHWC
        const TensorIterator<NoBuffer, kTransposeConvWRank, kTransposeConvWIterRank> weights,    // GHWCiCo
        const TransposeConv2DConfig &cfg,
        const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> output) { // BHWC
        return nullptr;
    }

    /**
     * @brief Permute kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param input         [I] TensorIterator object containing input Tensor shape and
     *                          memory strides and IteratorCfg
     * @param cfg           [I] Kernel configuration structure
     * @param output        [I] TensorIterator object containing output Tensor shape
     *                          and memory strides and IteratorCfg
     *
     * @return Permute kernel Compiler Support interface object
     */
    virtual lib_mli::Permute_CS* Permute_CS(
        void *kernel_buffer,
        const TensorIterator<NoBuffer, kPermuteRank, kPermuteIterRank> in,
        const PermuteOpConfig &cfg,
        const TensorIterator<NoBuffer, kPermuteRank, kPermuteIterRank> out) { 
        return nullptr; 
      }

};

} // namespace snps_arc::metaware::mli

#endif
