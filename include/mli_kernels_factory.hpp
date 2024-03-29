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
using lib_mli::service::set_default_align;

class KernelsFactory {
public:
    virtual uint32_t Nop_CS_GetSize() const { return 0; }

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

    virtual uint32_t MatMul_CS_GetSize() const { return 0; }

    virtual uint32_t MoveBroadcast_CS_GetSize() const { return 0; }
    
    virtual uint32_t ResizeBilinear_CS_GetSize() const { return 0; }

    /**
     * @brief No-op kernel Compiler Support interface factory
     *
     * @return No-op kernel Compiler Support interface object
     */
    virtual lib_mli::Nop_CS *Nop_CS(void *kernel_buffer) { return nullptr; }

    /**
     * @brief Convolution 2D kernel Compiler Support interface factory
     * method
     *
     * @deprecated
     * Be carefull - conv2d I/O tensors of rank 4 are deprecated - new interfaces use rank 5 
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
    virtual lib_mli::Conv2d_CS* Conv2d_CS(void *kernel_buffer,
                                          const Tensor<NoBuffer, 4> input_shape,          // BHWCi       
                                          const Tensor<NoBuffer, 5> weights,              // GKyKxCiCo
                                          const Conv2DConfig &cfg,
                                          const Tensor<NoBuffer, 4> output_tile_shape) {  // BHWCo
      return nullptr;
    }

    /**
     * @brief Convolution 2D kernel Compiler Support interface factory
     * method
     * @deprecated
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
                                          const TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank>& input,      // BHWGCi
                                          const TensorIterator<NoBuffer, kConvWRank, kConvWIterRank>& weights,      // GKyKxCiCo
                                          const TensorIterator<NoBuffer, kConvZPRank, kConvZPIterRank>& weights_zp, // Co tensor, GKyKxCiCo iterator
                                          const Conv2DConfig& cfg,
                                          const TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank>& output) {   // BHWGCo
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
      * @param input_zp      [I] TensorIterator object containing input zp(s) array
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
                                          const TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank>& input,      // BHWGCi
                                          const TensorIterator<NoBuffer, kConvZPRank, kConvZPIterRank>& input_zp,
                                          const TensorIterator<NoBuffer, kConvWRank, kConvWIterRank>& weights,      // GKyKxCiCo
                                          const TensorIterator<NoBuffer, kConvZPRank, kConvZPIterRank>& weights_zp, // Co tensor, GKyKxCiCo iterator
                                          const Conv2DConfig& cfg,
                                          const TensorIterator<NoBuffer, kConvIORank, kConvIOIterRank>& output) {   // BHWGCo
      return nullptr;
    }

    /**
     * @brief Convolution 2D kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     * 
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void Conv2d_CS_GetInputAlign(uint32_t input_align[kConvIORank]) {
      set_default_align<kConvIORank>(input_align);
    }

    /**
     * @brief Convolution 2D kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void Conv2d_CS_GetOutputAlign(uint32_t output_align[kConvIORank]) {
      set_default_align<kConvIORank>(output_align);
    }

    /**
     * @deprecated
     */
    virtual lib_mli::Prelu_CS* Prelu_CS(void *kernel_buffer,
                                        const Tensor<NoBuffer, 4> input_shape,
                                        const PreluOpConfig &cfg,
                                        const Tensor<NoBuffer, 4> output_tile_shape,
                                        int groups) { return nullptr; }

    /**
     * @deprecated
     * @brief PReLU kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param input         [I] TensorIterator object containing input Tensor shape and
     *                          memory strides and IteratorCfg
     * @param cfg           [I] Kernel configuration structure
     * @param output        [I] TensorIterator object containing output Tensor shape
     *                          and memory strides and IteratorCfg
     * @param groups        [I] Number of groups @deprecated
     *
     * @return PReLU kernel Compiler Support interface object
     */
    virtual lib_mli::Prelu_CS* Prelu_CS(void *kernel_buffer,
                                        const TensorIterator<NoBuffer, 4, 4> &input,
                                        const PreluOpConfig &cfg,
                                        const TensorIterator<NoBuffer, 4, 4> &output,
                                        int groups) { return nullptr; }

    /**
     * @brief PReLU kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param input         [I] TensorIterator object containing input Tensor shape and
     *                          memory strides and IteratorCfg
     * @param cfg           [I] Kernel configuration structure
     * @param enc_param     [I] TensorIterator object containing encoded parameters Tensor shape
     *                          and memory strides and IteratorCfg
     * @param output        [I] TensorIterator object containing output Tensor shape
     *                          and memory strides and IteratorCfg
     *
     * @return PReLU kernel Compiler Support interface object
     */
    virtual lib_mli::Prelu_CS* Prelu_CS(void *kernel_buffer,
                                        const TensorIterator<NoBuffer, kPreluRank, kPreluIterRank> &input,
                                        const PreluOpConfig &cfg,
                                        const TensorIterator<NoBuffer, kPreluRank, kPreluIterRank> &output) { return nullptr; }                                                                 

    /**
     * @brief Prelu kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void Prelu_CS_GetInputAlign(uint32_t input_align[kPreluRank]) {
      set_default_align<kPreluRank>(input_align);
    }

    /**
     * @brief Prelu kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void Prelu_CS_GetOutputAlign(uint32_t output_align[kPreluRank]) {
      set_default_align<kPreluRank>(output_align);
    }

    virtual lib_mli::Move_CS *Move_CS(void *kernel_buffer,
                                      const Tensor<NoBuffer, kMoveRank> src,
                                      const Tensor<NoBuffer, kMoveRank> dst,
                                      const lib_mli::MoveDataDirection data_dir) {
      return nullptr;
    }

    virtual lib_mli::Move_CS *Move_CS(void *kernel_buffer,
                                      const Tensor<NoBuffer, kMoveRank> src,
                                      const IteratorCfg<kMoveIterRank> src_cfg,
                                      const Tensor<NoBuffer, kMoveRank> dst,
                                      const IteratorCfg<kMoveIterRank> dst_cfg,
                                      const lib_mli::MoveDataDirection data_dir) {
      return nullptr;
    }

    /**
     * @brief Move kernel Compiler Support interface factory method
     * 
     * @param kernel_buffer       [I] Pointer to the pre-allocated memory to
     *                                store kernel Compiler Support object
     * @param src                 [I] TensorIterator object containing input
     *                                Tensor shape and memory strides and
     *                                IteratorCfg
     * @param dst                 [I] TensorIterator object containing output
     *                                Tensor shape and memory strides and
     *                                IteratorCfg
     * @param data_dir            [I] Define Move Data Direction
     *
     * @return Move kernel Compiler Support interface object
     */
    virtual lib_mli::Move_CS *Move_CS(void *kernel_buffer,
                                      const TensorIterator<NoBuffer, kMoveRank, kMoveIterRank> &src,
                                      const TensorIterator<NoBuffer, kMoveRank, kMoveIterRank> &dst,
                                      const lib_mli::MoveDataDirection data_dir) {
      return nullptr;
    }

    /**
     * @brief Move kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void Move_CS_GetInputAlign(uint32_t input_align[kMoveRank], const lib_mli::MoveDataDirection data_dir) {
      set_default_align<kMoveRank>(input_align);
    }

    /**
     * @brief Move kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void Move_CS_GetOutputAlign(uint32_t output_align[kMoveRank], const lib_mli::MoveDataDirection data_dir) {
      set_default_align<kMoveRank>(output_align);
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
                                                const Tensor<NoBuffer, kPoolRank> in,                  // BHWC
                                                const PoolOpConfig &cfg,
                                                const Tensor<NoBuffer, kPoolRank> output_tile_shape) { // BHWC
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
                                                const TensorIterator<NoBuffer, kPoolRank, kPoolIterRank>& in,      // BHWC
                                                const PoolOpConfig& cfg,
                                                const TensorIterator<NoBuffer, kPoolRank, kPoolIterRank>& out) {   // BHWC
      return nullptr;
    } 

    /**
     * @brief Maxpool 2D kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void MaxPool2D_CS_GetInputAlign(uint32_t input_align[kPoolRank]) {
      set_default_align<kPoolRank>(input_align);
    }

    /**
     * @brief Maxpool 2D kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
     virtual void MaxPool2D_CS_GetOutputAlign(uint32_t output_align[kPoolRank]) {
      set_default_align<kPoolRank>(output_align);
    }

/**
     * @brief Sumpool 2D kernel Compiler Support interface factory method
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
     * @return Sumpool 2D kernel Compiler Support interface object
     */
    virtual lib_mli::SumPool2D_CS* SumPool2D_CS(void *kernel_buffer,
                                                const Tensor<NoBuffer, kPoolRank> &in,
                                                const PoolOpConfig &cfg,
                                                const Tensor<NoBuffer, kPoolRank> &output_tile_shape) { 
      return nullptr; 
    }
    
    /**
     * @brief Sumpool 2D kernel Compiler Support interface factory method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param in            [I] TensorIterator object containing input Tensor shape and
     *                          memory strides and IteratorCfg
     * @param cfg           [I] Kernel configuration structure
     * @param out           [I] TensorIterator object containing output Tensor shape
     *                          and memory strides and IteratorCfg
     *
     * @return Sumpool 2D kernel Compiler Support interface object
     */
    virtual lib_mli::SumPool2D_CS* SumPool2D_CS(void *kernel_buffer,
                                                const TensorIterator<NoBuffer, kPoolRank, kPoolIterRank> &in,
                                                const PoolOpConfig &cfg,
                                                const TensorIterator<NoBuffer, kPoolRank, kPoolIterRank> &out) { 
      return nullptr; 
    }
    
    /**
     * @brief Sumpool 2D kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void SumPool2D_CS_GetInputAlign(uint32_t input_align[4]) {
      set_default_align<4>(input_align);
    }

    /**
     * @brief Sumpool 2D kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void SumPool2D_CS_GetOutputAlign(uint32_t output_align[4]) {
      set_default_align<4>(output_align);
    }

    /**
      * @brief Depthwise Convolution 2D kernel Compiler Support interface factory
      * method
      *
      * @deprecated
      * Be carefull - this factory method doesn't support tiling - only single tile size of provided tensors
      * Be carefull - depthwise conv2d I/O tensors of rank 4 are deprecated - new interfaces use rank 5 
      * Be carefull - this is the most deprecated factory method for DepthwiseConv2d_CS
      * @param kernel_buffer       [I] Pointer to the pre-allocated memory to store
      *                                kernel Compiler Support object
      * @param in                  [I] Tensor object containing input Tensor shape and
      *                                memory strides
      * @param weights             [I] Tensor object containing weights Tensor shape
      *                                and memory strides
      * @param cfg                 [I] Kernel configuration structure
      * @param output              [I] Tensor object containing output Tensor shape
      *                                and memory strides
      *
      * @return Depthwise Convolution 2D kernel Compiler Support interface object
      */
    virtual lib_mli::DepthwiseConv2d_CS* DepthwiseConv2d_CS(void *kernel_buffer,
                                                            const Tensor<NoBuffer, 4> in,       // BHWC
                                                            const Tensor<NoBuffer, 3> weights,  // KiKoC
                                                            const DwConv2DConfig &cfg,
                                                            const Tensor<NoBuffer, 4> output) { // BHWC
      return nullptr;
    }

    /**
     * @brief Depthwise Convolution 2D kernel Compiler Support interface factory
     * method
     *
     * @deprecated
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
                                                            const TensorIterator<NoBuffer, kDepthwiseIORank, kDepthwiseIterRank>& input,      // BHWGC
                                                            const TensorIterator<NoBuffer, kDepthwiseWRank, kDepthwiseIterRank>& weights,      // KiKoC
                                                            const TensorIterator<NoBuffer, kDepthwiseZPRank, kDepthwiseIterRank>& weights_zp, // C
                                                            const DwConv2DConfig& cfg,
                                                            const TensorIterator<NoBuffer, kDepthwiseIORank, kDepthwiseIterRank>& output) {   // BHWGC
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
     * @param input_zp      [I] TensorIterator object containing input zp(s) Tensor shape and
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
                                                            const TensorIterator<NoBuffer, kDepthwiseIORank, kDepthwiseIterRank>& input,      // BHWGC
                                                            const TensorIterator<NoBuffer, kDepthwiseZPRank, kDepthwiseIterRank>& input_zp,
                                                            const TensorIterator<NoBuffer, kDepthwiseWRank, kDepthwiseIterRank>& weights,     // KiKoC
                                                            const TensorIterator<NoBuffer, kDepthwiseZPRank, kDepthwiseIterRank>& weights_zp, // C
                                                            const DwConv2DConfig& cfg,
                                                            const TensorIterator<NoBuffer, kDepthwiseIORank, kDepthwiseIterRank>& output) {   // BHWGC
      return nullptr;
    }

    /**
     * @brief Depthwise Convolution 2D kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void DepthwiseConv2d_CS_GetInputAlign(uint32_t input_align[kDepthwiseIORank]) {
      set_default_align<kDepthwiseIORank>(input_align);
    }

    /**
     * @brief Depthwise Convolution 2D kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void DepthwiseConv2d_CS_GetOutputAlign(uint32_t output_align[kDepthwiseIORank]) {
      set_default_align<kDepthwiseIORank>(output_align);
    }

    /**
     * @deprecated
     */
    virtual lib_mli::FullyConnected_CS* FullyConnected_CS(void *kernel_buffer,
                                                          const Tensor<NoBuffer, 2> in,
                                                          const Tensor<NoBuffer, 2> weights,
                                                          const Tensor<NoBuffer, 2> output_tile_shape) { return nullptr; }

    /**
     * @deprecated
     */
    virtual lib_mli::FullyConnected_CS* FullyConnected_CS(void *kernel_buffer,
                                                          const Tensor<NoBuffer, 2> in,
                                                          const Tensor<NoBuffer, 2> weights,
                                                          const Tensor<NoBuffer, 1> wtszp,
                                                          const Tensor<NoBuffer, 2> output_tile_shape) { return nullptr; }

    /**
     * @brief Fully Connected kernel Compiler Support interface factory
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
     * @return Fully Connected kernel Compiler Support interface object
     */
    virtual lib_mli::FullyConnected_CS* FullyConnected_CS(void* kernel_buffer,
                                                          const TensorIterator<NoBuffer, kFullyConnectedIORank, kFullyConnectedIterRank>& input,
                                                          const TensorIterator<NoBuffer, kFullyConnectedWRank, kFullyConnectedIterRank>& weights,
                                                          const TensorIterator<NoBuffer, kFullyConnectedZPRank, kFullyConnectedIterRank>& weights_zp,
                                                          const FullyConnectedConfig& cfg,
                                                          const TensorIterator<NoBuffer, kFullyConnectedIORank, kFullyConnectedIterRank>& output) {
      return nullptr;
    }

    /**
     * @brief FullyConnected kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void FullyConnected_CS_GetInputAlign(uint32_t input_align[2]) {
      set_default_align<2>(input_align);
    }

    /**
     * @brief FullyConnected kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void FullyConnected_CS_GetOutputAlign(uint32_t output_align[2]) {
      set_default_align<2>(output_align);
    }

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
     * @brief Clip kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void Clip_CS_GetInputAlign(uint32_t input_align[kClipRank]) {
      set_default_align<kClipRank>(input_align);
    }

    /**
     * @brief Clip kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void Clip_CS_GetOutputAlign(uint32_t output_align[kClipRank]) {
      set_default_align<kClipRank>(output_align);
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
     *
     * @deprecated
     */
    virtual lib_mli::Add_CS* Add_CS(void *kernel_buffer,
                                    const Tensor<NoBuffer, kEltwiseRank> input_left,
                                    const Tensor<NoBuffer, kEltwiseRank> input_right,
                                    const Tensor<NoBuffer, kEltwiseRank> output) { return nullptr; }
    /**
     * @brief Add kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param input_left    [I] TensorIterator object containing input1 Tensor and
     *                          tile configuration parameters
     * @param input_right   [I] TensorIterator object containing input2 Tensor and
     *                          tile configuration parameters
     * @param output        [I] TensorIterator object containing output Tensor and
     *                          tile configuration parameters
     *
     * @return Add kernel Compiler Support interface object
     */ 
    virtual lib_mli::Add_CS* Add_CS(void *kernel_buffer,
                                    const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> input_left,
                                    const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> input_right,
                                    const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> output) { return nullptr; }

    /**
     * @brief Add kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void Add_CS_GetInputAlign(uint32_t input_align[kEltwiseRank]) {
      set_default_align<kEltwiseRank>(input_align);
    }

    /**
     * @brief Add kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void Add_CS_GetOutputAlign(uint32_t output_align[kEltwiseRank]) {
      set_default_align<kEltwiseRank>(output_align);
    }

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
     *
     * @deprecated
     */
    virtual lib_mli::Sub_CS* Sub_CS(void *kernel_buffer,
                                    const Tensor<NoBuffer, 4> input_left,
                                    const Tensor<NoBuffer, 4> input_right,
                                    const Tensor<NoBuffer, 4> output) { return nullptr; }

    /**
     * @brief Sub kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param input_left    [I] TensorIterator object containing input1 Tensor and
     *                          tile configuration parameters
     * @param input_right   [I] TensorIterator object containing input2 Tensor and
     *                          tile configuration parameters
     * @param output        [I] TensorIterator object containing output Tensor and
     *                          tile configuration parameters
     *
     * @return Sub kernel Compiler Support interface object
     */
    virtual lib_mli::Sub_CS* Sub_CS(void *kernel_buffer,
                                    const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> input_left,
                                    const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> input_right,
                                    const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> output) { return nullptr; }

    /**
     * @brief Sub kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void Sub_CS_GetInputAlign(uint32_t input_align[kEltwiseRank]) {
      set_default_align<kEltwiseRank>(input_align);
    }

    /**
     * @brief Sub kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void Sub_CS_GetOutputAlign(uint32_t output_align[kEltwiseRank]) {
      set_default_align<kEltwiseRank>(output_align);
    }

    /**
     * @brief Mul kernel Compiler Support interface factory
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
     * @return Mul kernel Compiler Support interface object
     *
     * @deprecated
     */
    virtual lib_mli::Mul_CS* Mul_CS(void *kernel_buffer,
                                    const Tensor<NoBuffer, 4> input_left,
                                    const Tensor<NoBuffer, 4> input_right,
                                    const Tensor<NoBuffer, 4> output) { return nullptr; }
    /**
     * @brief Mul kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param input_left    [I] TensorIterator object containing input1 Tensor and
     *                          tile configuration parameters
     * @param input_right   [I] TensorIterator object containing input2 Tensor and
     *                          tile configuration parameters
     * @param output        [I] TensorIterator object containing output Tensor and
     *                          tile configuration parameters
     *
     * @return Mul kernel Compiler Support interface object
     */
    virtual lib_mli::Mul_CS* Mul_CS(void *kernel_buffer,
                                    const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> input_left,
                                    const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> input_right,
                                    const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> output) { return nullptr; }

    /**
     * @brief Mul kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void Mul_CS_GetInputAlign(uint32_t input_align[kEltwiseRank]) {
      set_default_align<kEltwiseRank>(input_align);
    }

    /**
     * @brief Mul kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void Mul_CS_GetOutputAlign(uint32_t output_align[kEltwiseRank]) {
      set_default_align<kEltwiseRank>(output_align);
    }

    /**
     * @brief Max kernel Compiler Support interface factory
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
     * @return Max kernel Compiler Support interface object
     *
     * @deprecated
     */
    virtual lib_mli::Max_CS* Max_CS(void *kernel_buffer,
                                    const Tensor<NoBuffer, 4> input_left,
                                    const Tensor<NoBuffer, 4> input_right,
                                    const Tensor<NoBuffer, 4> output) { return nullptr; }
    /**
     * @brief Max kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param input_left    [I] TensorIterator object containing input1 Tensor and
     *                          tile configuration parameters
     * @param input_right   [I] TensorIterator object containing input2 Tensor and
     *                          tile configuration parameters
     * @param output        [I] TensorIterator object containing output Tensor and
     *                          tile configuration parameters
     *
     * @return Max kernel Compiler Support interface object
     */
    virtual lib_mli::Max_CS* Max_CS(void *kernel_buffer,
                                    const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> input_left,
                                    const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> input_right,
                                    const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> output) { return nullptr; }

    /**
     * @brief Max kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void Max_CS_GetInputAlign(uint32_t input_align[kEltwiseRank]) {
      set_default_align<kEltwiseRank>(input_align);
    }

    /**
     * @brief Max kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void Max_CS_GetOutputAlign(uint32_t output_align[kEltwiseRank]) {
      set_default_align<kEltwiseRank>(output_align);
    }

    /**
     * @brief Min kernel Compiler Support interface factory
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
     * @return Min kernel Compiler Support interface object
     *
     * @deprecated
     */
    virtual lib_mli::Min_CS* Min_CS(void *kernel_buffer,
                                    const Tensor<NoBuffer, 4> input_left,
                                    const Tensor<NoBuffer, 4> input_right,
                                    const Tensor<NoBuffer, 4> output) { return nullptr; }
    /**
     * @brief Min kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param input_left    [I] TensorIterator object containing input1 Tensor and
     *                          tile configuration parameters
     * @param input_right   [I] TensorIterator object containing input2 Tensor and
     *                          tile configuration parameters
     * @param output        [I] TensorIterator object containing output Tensor and
     *                          tile configuration parameters
     *
     * @return Min kernel Compiler Support interface object
     */
    virtual lib_mli::Min_CS* Min_CS(void *kernel_buffer,
                                    const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> input_left,
                                    const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> input_right,
                                    const TensorIterator<NoBuffer, kEltwiseRank, kEltwiseIterRank> output) { return nullptr; }

    /**
     * @brief Min kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void Min_CS_GetInputAlign(uint32_t input_align[kEltwiseRank]) {
      set_default_align<kEltwiseRank>(input_align);
    }

    /**
     * @brief Min kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void Min_CS_GetOutputAlign(uint32_t output_align[kEltwiseRank]) {
      set_default_align<kEltwiseRank>(output_align);
    }

    /**
     * @brief Rescale kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer       [I] Pointer to the pre-allocated memory to store
     *                                kernel Compiler Support object
     * @param input               [I] TensorIterator object containing input Tensor shape and
     *                                memory strides
     * @param cfg                 [I] Kernel configuration structure
     * @param enc_param           [I] TensorIterator object containing encode params Tensor shape and
     *                                memory strides
     * @param output              [I] TensorIterator object containing output Tensor shape
     *                                and memory strides
     *
     * @return Rescale kernel Compiler Support interface object
     */
    virtual lib_mli::Rescale_CS* Rescale_CS(void* kernel_buffer,
                                            const TensorIterator<NoBuffer, kRescaleRank, kRescaleIterRank>& input,
                                            const RescaleConfig& cfg,
                                            const TensorIterator<NoBuffer, kRescaleParamRank, kRescaleIterRank>& enc_param,
                                            const TensorIterator<NoBuffer, kRescaleRank, kRescaleIterRank>& output) {
      return nullptr;
    }

    /**
     * @brief Rescale kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void Rescale_CS_GetInputAlign(uint32_t input_align[kRescaleRank]) {
      set_default_align<kRescaleRank>(input_align);
    }

    /**
     * @brief Rescale kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void Rescale_CS_GetOutputAlign(uint32_t output_align[kRescaleRank]) {
      set_default_align<kRescaleRank>(output_align);
    }

    virtual lib_mli::ReduceMax_CS* ReduceMax_CS(void *kernel_buffer,
                                                const TensorIterator<NoBuffer, kReduceMaxRank, kReduceMaxIterRank> &in,
                                                const ReduceOpConfig &cfg,
                                                const TensorIterator<NoBuffer, kReduceMaxRank, kReduceMaxIterRank> &out) { return nullptr; }

    /**
     * @brief ReduceMax kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void ReduceMax_CS_GetInputAlign(uint32_t input_align[kReduceMaxRank]) {
      set_default_align<kReduceMaxRank>(input_align);
    }

    /**
     * @brief ReduceMax kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void ReduceMax_CS_GetOutputAlign(uint32_t output_align[kReduceMaxRank]) {
      set_default_align<kReduceMaxRank>(output_align);
    }
    
    virtual lib_mli::ReduceSum_CS* ReduceSum_CS(void *kernel_buffer,
                                                const TensorIterator<NoBuffer, kReduceSumRank, kReduceSumIterRank> &in,
                                                const ReduceOpConfig &cfg,
                                                const TensorIterator<NoBuffer, kReduceSumRank, kReduceSumIterRank> &out) { return nullptr; }

    /**
     * @brief ReduceSum kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void ReduceSum_CS_GetInputAlign(uint32_t input_align[kReduceSumRank]) {
      set_default_align<kReduceSumRank>(input_align);
    }

    /**
     * @brief ReduceSum kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void ReduceSum_CS_GetOutputAlign(uint32_t output_align[kReduceSumRank]) {
      set_default_align<kReduceSumRank>(output_align);
    }

    virtual lib_mli::ArgMax_CS* ArgMax_CS(void *kernel_buffer,
                                          const TensorIterator<NoBuffer, kArgMaxInRank, kArgMaxInIterRank> in,
                                          const ArgMaxConfig &cfg,
                                          const TensorIterator<NoBuffer, kArgMaxOutRank, kArgMaxOutIterRank> out) { return nullptr; }

    virtual uint32_t ArgMax_CS_GetSize() const { return 0; }

    /**
     * @brief ArgMax kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void ArgMax_CS_GetInputAlign(uint32_t input_align[kArgMaxInRank]) {
      set_default_align<kArgMaxInRank>(input_align);
    }

    /**
     * @brief ArgMax kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void ArgMax_CS_GetOutputAlign(uint32_t output_align[kArgMaxOutRank]) {
      set_default_align<kArgMaxOutRank>(output_align);
    }

    virtual lib_mli::TableBuiltin_CS* TableBuiltin_CS(void *kernel_buffer,
                                                      const TensorIterator<NoBuffer, kTableBuiltinIORank, kTableBuiltinIOIterRank> &in,
                                                      const TableBuiltinConfig &cfg,
                                                      const TensorIterator<NoBuffer, kTableBuiltinIORank, kTableBuiltinIOIterRank> &out) { return nullptr; }

    /**
     * @brief TableBuiltin kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void TableBuiltin_CS_GetInputAlign(uint32_t input_align[kTableBuiltinIORank]) {
      set_default_align<kTableBuiltinIORank>(input_align);
    }

    /**
     * @brief TableBuiltin kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void TableBuiltin_CS_GetOutputAlign(uint32_t output_align[kTableBuiltinIORank]) {
      set_default_align<kTableBuiltinIORank>(output_align);
    }

    virtual lib_mli::MatMul_CS* MatMul_CS(void *kernel_buffer,
                                          const TensorIterator<NoBuffer, kMatMulRank, kMatMulIterRank> &input_left,
                                          const TensorIterator<NoBuffer, kMatMulRank, kMatMulIterRank> &input_right,
                                          const TensorIterator<NoBuffer, kMatMulRank, kMatMulIterRank> &output) { return nullptr; }

    /**
     * @brief MatMul kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void MatMul_CS_GetInputAlign(uint32_t input_align[kMatMulRank]) {
      set_default_align<kMatMulRank>(input_align);
    }

    /**
     * @brief MatMul kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void MatMul_CS_GetOutputAlign(uint32_t output_align[kMatMulRank]) {
      set_default_align<kMatMulRank>(output_align);
    }

    /**
     * @brief Transpose Convolution 2D kernel Compiler Support interface factory
     * method
     * @deprecated
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
     * @return Transpose Convolution 2D kernel Compiler Support interface object
     */
    virtual lib_mli::TransposeConv2D_CS* TransposeConv2D_CS(
        void *kernel_buffer,
        const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank>& input,    // BHWGCi
        const TensorIterator<NoBuffer, kTransposeConvWRank, kTransposeConvWIterRank>& weights,    // GHWCiCo
        const TensorIterator<NoBuffer, kTransposeConvZPRank, kTransposeConvZPIterRank>& weights_zp,
        const TransposeConv2DConfig &cfg,
        const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank>& output) { // BHWGCo
        return nullptr;
    }

    /**
     * @brief Transpose Convolution 2D kernel Compiler Support interface factory
     * method
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param input         [I] TensorIterator object containing input Tensor shape and
     *                          memory strides and IteratorCfg
     * @param input_zp      [I] TensorIterator object containing input zp(s) array
     * @param weights       [I] TensorIterator object containing weights Tensor shape
     *                          and memory strides and IteratorCfg
     * @param weights_zp    [I] TensorIterator object containing weight zp(s) array
     * @param cfg           [I] Kernel configuration structure
     * @param output        [I] TensorIterator object containing output Tensor shape
     *                          and memory strides and IteratorCfg
     *
     * @return Transpose Convolution 2D kernel Compiler Support interface object
     */
    virtual lib_mli::TransposeConv2D_CS* TransposeConv2D_CS(void* kernel_buffer,
                                                            const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank>& input,      // BHWGCi
                                                            const TensorIterator<NoBuffer, kTransposeConvZPRank, kTransposeConvIterRank>& input_zp,
                                                            const TensorIterator<NoBuffer, kTransposeConvWRank, kTransposeConvWIterRank>& weights,      // GHWCiCo
                                                            const TensorIterator<NoBuffer, kTransposeConvZPRank, kTransposeConvZPIterRank>& weights_zp,
                                                            const TransposeConv2DConfig& cfg,
                                                            const TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank>& output) {   // BHWGCo
        return nullptr;
    }

    /**
     * @brief Transpose Convolution 2D kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void TransposeConv2D_CS_GetInputAlign(uint32_t input_align[kTransposeConvIORank]) {
      set_default_align<kTransposeConvIORank>(input_align);
    }

    /**
     * @brief Transpose Convolution 2D kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void TransposeConv2D_CS_GetOutputAlign(uint32_t output_align[kTransposeConvIORank]) {
      set_default_align<kTransposeConvIORank>(output_align);
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

    /**
     * @brief Permute kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void Permute_CS_GetInputAlign(uint32_t input_align[kPermuteRank]) {
      set_default_align<kPermuteRank>(input_align);
    }

    /**
     * @brief Permute kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void Permute_CS_GetOutputAlign(uint32_t output_align[kPermuteRank]) {
      set_default_align<kPermuteRank>(output_align);
    }

    virtual lib_mli::ResizeBilinear_CS* ResizeBilinear_CS(void *kernel_buffer,
                                                          const TensorIterator<NoBuffer, kResizeBilinearRank, kResizeBilinearIterRank> &in,
                                                          const ResizeOpConfig &cfg,
                                                          const TensorIterator<NoBuffer, kResizeBilinearRank, kResizeBilinearIterRank> &out) { return nullptr; }

    /**
     * @brief ResizeBilinear kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void ResizeBilinear_CS_GetInputAlign(uint32_t input_align[kResizeBilinearRank]) {
      set_default_align<kResizeBilinearRank>(input_align);
    }

    /**
     * @brief ResizeBilinear kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void ResizeBilinear_CS_GetOutputAlign(uint32_t output_align[kResizeBilinearRank]) {
      set_default_align<kResizeBilinearRank>(output_align);
    }

    /**
     * @brief MoveBroadcast kernel Compiler Support interface factory
     * method
     *
     * @param kernel_buffer [I] Pointer to the pre-allocated memory to store
     *                          kernel Compiler Support object
     * @param src           [I] TensorIterator object containing input tensor 
     *                          shape and memory strides and IteratorCfg
     * @param dst           [I] TensorIterator object containing output tensor 
     *                          shape and memory strides and IteratorCfg
     * @param data_dir      [I] Define Move Data Direction
     *
     * @return MoveBroadcast kernel Compiler Support interface object
     */
    virtual lib_mli::MoveBroadcast_CS *MoveBroadcast_CS(void *kernel_buffer,
                                                        const TensorIterator<NoBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank> &src,
                                                        const TensorIterator<NoBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank> &dst,
                                                        const lib_mli::MoveDataDirection data_dir) {
      return nullptr;
    }

    /**
     * @brief MoveBroadcast kernel Compiler Support interface
     *        to get the Alignment Restrictions in Input Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param input_align  [O] Array to be filled with the Input Alignment Restrictions
     */
    virtual void MoveBroadcast_CS_GetInputAlign(uint32_t input_align[kMoveBroadcastRank]) {
      set_default_align<kMoveBroadcastRank>(input_align);
    }

    /**
     * @brief MoveBroadcast kernel Compiler Support interface
     *        to get the Alignment Restrictions in Output Tensor.
     *
     *        Alignment restriction can be used for
     *          - Buffer Size Calculation
     *          - Tile Size Selection
     *          - Shape Adjustment
     *          - Pointer Alignment (Pointer should be aligned on the product of alignment)
     * 
     * @param output_align  [O] Array to be filled with the Output Alignment Restrictions
     */
    virtual void MoveBroadcast_CS_GetOutputAlign(uint32_t output_align[kMoveBroadcastRank]) {
      set_default_align<kMoveBroadcastRank>(output_align);
    }

};

} // namespace snps_arc::metaware::mli

#endif
