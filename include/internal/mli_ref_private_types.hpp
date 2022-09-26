/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_REF_PRIVATE_TYPES_HPP_
#define _MLI_REF_PRIVATE_TYPES_HPP_

#include "mli_types.h"
#include "mli_types.hpp"
#include "mli_prv_layout.h"
#include "mli_compiler_api.hpp"

namespace lib_mli = ::snps_arc::metaware::mli;

namespace snps_arc::metaware::mli::ref {

// the layout of tensor
using Layout = mli_layout_type;

class Conv2DPrivateData : public PrivateData {
public:
    Conv2DPrivateData() : PrivateData(kConv2dId, sizeof(Conv2DPrivateData)) {}

    // In/Out/weights/weights zp(s) tensor iterators with attached offset buffers
    TensorIterator<OffsetBuffer, kConvIORank, kConvIOIterRank> input;
    TensorIterator<OffsetBuffer, kConvWRank, kConvWIterRank> weights;
    TensorIterator<OffsetBuffer, kConvZPRank, kConvZPIterRank> weights_zp;
    TensorIterator<OffsetBuffer, kConvIORank, kConvIOIterRank> output;

    // The layout of input
    Layout layout;

    // Encoded input zero point
    OffsetBuffer inpzp_buffer;

    // the index of quantization axis
    int inp_quant_axis;
    int wts_quant_axis;

    // Convolution config
    Conv2DConfig config;
};

struct Conv2dMetadata {
    TensorIterator<OffsetBuffer, kConvIORank, kConvIOIterRank> input;
    TensorIterator<OffsetBuffer, kConvWRank, kConvWIterRank> weights;
    TensorIterator<OffsetBuffer, kConvZPRank, kConvZPIterRank> weights_zp;
    TensorIterator<OffsetBuffer, kConvIORank, kConvIOIterRank> output;

    InternalBuffer inpzp_buffer;
    int inp_quant_axis;
    int wts_quant_axis;

    Conv2DConfig cfg;
};

class DepthwiseConv2DPrivateData : public PrivateData {

public:
    DepthwiseConv2DPrivateData()
        : PrivateData(kDWConv2dId, sizeof(DepthwiseConv2DPrivateData)) {}

    // In/Out Tensor attached with offset buffer
    TensorIterator<OffsetBuffer, kDepthwiseIORank, kDepthwiseIOIterRank> input;
    TensorIterator<OffsetBuffer, kDepthwiseWRank, kDepthwiseWIterRank> weights;
    TensorIterator<OffsetBuffer, kDepthwiseZPRank, kDepthwiseZPIterRank> weights_zp;
    TensorIterator<OffsetBuffer, kDepthwiseIORank, kDepthwiseIOIterRank> output;

    // The layout of input
    Layout layout;

    // Encoded input zero pointers
    OffsetBuffer inpzp_buffer;

    // the index of quantization axis
    int inp_quant_axis;
    int wts_quant_axis;

    // Convolution config
    DwConv2DConfig config;
};

struct DepthwiseConv2dMetadata {
    TensorIterator<OffsetBuffer, kDepthwiseIORank, kDepthwiseIOIterRank> input;
    TensorIterator<OffsetBuffer, kDepthwiseWRank, kDepthwiseWIterRank> weights;
    TensorIterator<OffsetBuffer, kDepthwiseZPRank, kDepthwiseZPIterRank> weights_zp;
    TensorIterator<OffsetBuffer, kDepthwiseIORank, kDepthwiseIOIterRank> output;

    InternalBuffer inpzp_buffer;
    int inp_quant_axis;
    int wts_quant_axis;

    DwConv2DConfig config;
};

class TransposeConv2DPrivateData : public PrivateData {
public:
    TransposeConv2DPrivateData() : PrivateData(kTransConv2DId, sizeof(TransposeConv2DPrivateData)) {}

    // In/Out/weights/weights zp(s) tensor iterators with attached offset buffers
    TensorIterator<OffsetBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> input;
    TensorIterator<OffsetBuffer, kTransposeConvWRank, kTransposeConvWIterRank> weights;
    TensorIterator<OffsetBuffer, kTransposeConvZPRank, kTransposeConvZPIterRank> weights_zp;
    TensorIterator<OffsetBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> output;
    
    // The layout of input
    Layout layout;

    // Encoded input zero point
    OffsetBuffer inpzp_buffer;

    // the index of quantization axis
    int inp_quant_axis;
    int wts_quant_axis;

    // Convolution config
    TransposeConv2DConfig config;
};

struct TransposeConv2DMetadata {
    TensorIterator<OffsetBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> input;
    TensorIterator<OffsetBuffer, kTransposeConvWRank, kTransposeConvWIterRank> weights;
    TensorIterator<OffsetBuffer, kTransposeConvZPRank, kTransposeConvZPIterRank> weights_zp;
    TensorIterator<OffsetBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> output;

    InternalBuffer inpzp_buffer;
    int inp_quant_axis;
    int wts_quant_axis;

    TransposeConv2DConfig cfg;
};

class FullyConnectedPrivateData : public PrivateData {

public:
    FullyConnectedPrivateData()
        : PrivateData(kFullyConnectedId, sizeof(FullyConnectedPrivateData)) {}

    // currently we support the only i8_w8_o32 case
    OffsetBuffer input_buffer;
    OffsetBuffer weights_buffer;
    OffsetBuffer output_buffer;
    OffsetBuffer wtszp_buffer;

    uint32_t input_n;
    uint32_t input_ic;

    uint32_t output_n;
    uint32_t output_oc;

    uint32_t weights_ic;
    uint32_t weights_oc;

    int32_t input_n_stride;
    int32_t input_ic_stride;

    int32_t output_n_stride;
    int32_t output_oc_stride;

    int32_t weights_ic_stride;
    int32_t weights_oc_stride;

    uint8_t stride_n;
    uint8_t stride_ic;

    int32_t qt_wtszp_axis;
};

struct FullyConnectedMetadata {
    mli_tensor input;
    mli_tensor weights;
    mli_tensor output;
};

class MovePrivateData : public PrivateData {

public:
    MovePrivateData() : PrivateData(kMoveId, sizeof(MovePrivateData)) {}

    TensorIterator<OffsetBuffer, kMoveRank, kMoveIterRank> src_it;
    TensorIterator<OffsetBuffer, kMoveRank, kMoveIterRank> dst_it;
};

/**
 *  TODO: to remove this after Pool2DPrivateData will be updated
 *  to do this SumPool2D_CS needs to be updated same was as MaxPool2D_CS
 */ 
class MaxPool2DPrivateData : public PrivateData {

public:
  MaxPool2DPrivateData(kernel_id_t id)
    : PrivateData(id, sizeof(MaxPool2DPrivateData)) {}

  TensorIterator<OffsetBuffer, kMaxpoolRank, kMaxpoolIterRank> input;
  TensorIterator<OffsetBuffer, kMaxpoolRank, kMaxpoolIterRank> output;
  PoolOpConfig config;
};

class Pool2DPrivateData : public PrivateData {

public:
    Pool2DPrivateData(kernel_id_t id)
        : PrivateData(id, sizeof(Pool2DPrivateData)) {}

    OffsetBuffer input_buffer;
    OffsetBuffer output_buffer;

    uint32_t input_b;
    uint32_t input_h;
    uint32_t input_w;
    uint32_t input_c;

    int32_t input_b_stride;
    int32_t input_h_stride;
    int32_t input_w_stride;
    int32_t input_c_stride;

    uint32_t output_b;
    uint32_t output_h;
    uint32_t output_w;
    uint32_t output_c;

    int32_t output_b_stride;
    int32_t output_h_stride;
    int32_t output_w_stride;
    int32_t output_c_stride;

    uint8_t kernel_height;
    uint8_t kernel_width;
    uint8_t stride_height;
    uint8_t stride_width;
    uint8_t padding_top;
    uint8_t padding_bottom;
    uint8_t padding_left;
    uint8_t padding_right;

    // Tile Parameters BHWC
    bool m_use_tiling;
    uint32_t m_tile_total_input_size[4];
    uint32_t m_tile_total_output_size[4];
    uint32_t m_tile_iteration_order[4];
    uint32_t m_tile_first_size[4];
    uint32_t m_tile_size[4];
    uint32_t m_tile_input_first_inc[4];
    uint32_t m_tile_input_inc[4];
    uint32_t m_tile_output_first_inc[4];
    uint32_t m_tile_output_inc[4];
};

class EltwisePrivateData : public PrivateData {

public:
    EltwisePrivateData(kernel_id_t id)
        : PrivateData(id, sizeof(EltwisePrivateData)) {}

    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_in_left_buffer;
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_in_right_buffer; 
    TensorIterator<OffsetBuffer, kEltwiseRank, kEltwiseIterRank> m_output_buffer;

    bool is_in_left_scalar;
    bool is_in_right_scalar;
    
};

class RescalePrivateData : public PrivateData {

public:
    RescalePrivateData()
        : PrivateData(kRescaleId, sizeof(RescalePrivateData)) {}

    int32_t rescale_axis;

    TensorIterator<OffsetBuffer, kRescaleRank, kRescaleIterRank> input;
    TensorIterator<OffsetBuffer, kRescaleRank, kRescaleIterRank> output;
    OffsetBuffer encoded_params_buffer;

    uint32_t params_elem_num;
    uint32_t tile_params_max_elem_num;
};

struct RescaleMetadata {
    mli_tensor input;
    mli_tensor in_bias;
    mli_tensor scale;
    mli_tensor shift;
    mli_tensor out_bias;
    int32_t rescale_axis;
    mli_tensor output;
};

struct TableBuiltinMetadata {
    TensorIterator<OffsetBuffer, kTableBuiltinIORank, kTableBuiltinIOIterRank> input;
    TensorIterator<OffsetBuffer, kBiasRank, kBiasIterRank> in_bias;
    TensorIterator<OffsetBuffer, kTableBuiltinIORank, kTableBuiltinIOIterRank> output;
    LutType lut_mode;
};

class ClipPrivateData : public PrivateData {

public:
    ClipPrivateData() : PrivateData(kClipId, sizeof(ClipPrivateData)) {}

    // currently we support the only i8_w8_o32 case
    TensorIterator<OffsetBuffer, kClipRank, kClipIterRank> input;
    TensorIterator<OffsetBuffer, kClipRank, kClipIterRank> output;
    OffsetBuffer encoded_params_buf;
};

class TableBuiltinPrivateData : public PrivateData {
public:
    TableBuiltinPrivateData() : PrivateData(kTableBuiltinId, sizeof(TableBuiltinPrivateData)) {}
    int32_t table_axis;
    uint32_t io_rank;

    TensorIterator<OffsetBuffer, kTableBuiltinIORank, kTableBuiltinIOIterRank> input;
    TensorIterator<OffsetBuffer, kTableBuiltinIORank, kTableBuiltinIOIterRank> output;
};

class ReduceMaxPrivateData : public PrivateData {

public:
    ReduceMaxPrivateData() : PrivateData(kReduceMaxId, sizeof(ReduceMaxPrivateData)) {}

    int32_t reduce_axis;

    uint32_t io_rank;

    TensorIterator<OffsetBuffer, kReduceMaxRank, kReduceMaxIterRank> input;
    TensorIterator<OffsetBuffer, kReduceMaxRank, kReduceMaxIterRank> output;
};


class PermutePrivateData : public PrivateData {

public:
    PermutePrivateData() : PrivateData(kPermuteId, sizeof(PermutePrivateData)) {}

    uint8_t perm_dim[kPermuteRank];

    uint32_t io_rank;

    TensorIterator<OffsetBuffer, kPermuteRank, kPermuteIterRank> input;
    TensorIterator<OffsetBuffer, kPermuteRank, kPermuteIterRank> output;
};

struct PermuteMetadata
{
    TensorIterator<OffsetBuffer, kPermuteRank, kPermuteIterRank> m_input;
    TensorIterator<OffsetBuffer, kPermuteRank, kPermuteIterRank> m_output;
    mli_tensor m_tile_input;
    mli_tensor m_tile_output;
};

class ReduceSumPrivateData : public PrivateData {

public:
    ReduceSumPrivateData() : PrivateData(kReduceSumId, sizeof(ReduceSumPrivateData)) {}

    int32_t reduce_axis;

    uint32_t io_rank;

    TensorIterator<OffsetBuffer, kReduceSumRank, kReduceSumIterRank> input;
    TensorIterator<OffsetBuffer, kReduceSumRank, kReduceSumIterRank> output;
};

class ArgMaxPrivateData : public PrivateData {

public:
    ArgMaxPrivateData() : PrivateData(kArgMaxId, sizeof(ArgMaxPrivateData)) {}

    int8_t argmax_axis;

    uint32_t io_rank;

    TensorIterator<OffsetBuffer, kArgMaxInRank, kArgMaxInIterRank> input;
    TensorIterator<OffsetBuffer, kArgMaxOutRank, kArgMaxOutIterRank> output;
};

class MatMulPrivateData : public PrivateData {

public:
    MatMulPrivateData() : PrivateData(kMatMulId, sizeof(MatMulPrivateData)) {}

    TensorIterator<OffsetBuffer, kMatMulRank, kMatMulIterRank> m_in_left;
    TensorIterator<OffsetBuffer, kMatMulRank, kMatMulIterRank> m_in_right;
    TensorIterator<OffsetBuffer, kMatMulRank, kMatMulIterRank> m_output;
    
    // Encoded input zero points
    OffsetBuffer encoded_params;

};

struct PreluMetadata {
    Tensor<InternalBuffer, kPreluRank> input;
    Tensor<InternalBuffer, kPreluRank> output;
    InternalBuffer in_bias;
    InternalBuffer posscale;
    InternalBuffer negscale;
    InternalBuffer posshift;
    InternalBuffer negshift;
    InternalBuffer out_bias;
    int32_t prelu_axis;
};

class PreluPrivateData : public PrivateData {

public:
    PreluPrivateData() : PrivateData(kPreluId, sizeof(PreluPrivateData)) {}
    
    int32_t prelu_axis;
    OffsetBuffer encoded_params_buffer;
    uint32_t tile_params_max_elem_num;

    TensorIterator<OffsetBuffer, kPreluRank, kPreluIterRank> input;
    TensorIterator<OffsetBuffer, kPreluRank, kPreluIterRank> output;
};

class MoveBroadcastPrivateData : public PrivateData {

public:
    MoveBroadcastPrivateData() : PrivateData(kMoveBroadcastId, sizeof(MoveBroadcastPrivateData)) {}

    TensorIterator<OffsetBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank> src;
    TensorIterator<OffsetBuffer, kMoveBroadcastRank, kMoveBroadcastIterRank> dst;
};

class ResizeBilinearPrivateData : public PrivateData {

public:
    ResizeBilinearPrivateData() : PrivateData(kResizeBilinearId, sizeof(ResizeBilinearPrivateData)) {}

    ResizeOpConfig config;

    TensorIterator<OffsetBuffer, kResizeBilinearRank, kResizeBilinearIterRank> input;
    TensorIterator<OffsetBuffer, kResizeBilinearRank, kResizeBilinearIterRank> output;
};

} // namespace snps_arc::metaware::mli::ref

#endif // _MLI_REF_PRIVATE_TYPES_HPP_

