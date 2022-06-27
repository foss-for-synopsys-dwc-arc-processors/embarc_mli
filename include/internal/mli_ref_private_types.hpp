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
#include "mli_ref_compiler_api.hpp"

namespace lib_mli = ::snps_arc::metaware::mli;

namespace snps_arc::metaware::mli::ref {
class Conv2DPrivateData : public PrivateData {
public:
    Conv2DPrivateData() : PrivateData(kConv2dId){}

    // currently we support the only i8_w8_o32 case
    OffsetBuffer input_buffer;
    OffsetBuffer weights_buffer;
    OffsetBuffer output_buffer;
    OffsetBuffer inpzp_buffer;
    OffsetBuffer wtszp_buffer;

    uint32_t input_h;
    uint32_t input_w;
    uint32_t input_c;

    uint32_t output_h;
    uint32_t output_w;
    uint32_t output_c;

    uint32_t weights_h;
    uint32_t weights_w;

    int32_t input_h_stride;
    int32_t input_w_stride;

    int32_t output_h_stride;
    int32_t output_w_stride;

    int32_t weights_h_stride;
    int32_t weights_w_stride;
    int32_t weights_c_stride;

    uint8_t stride_height;
    uint8_t stride_width;
    uint8_t dilation_height;
    uint8_t dilation_width;
    uint8_t groups;

    uint8_t padding_left;
    uint8_t padding_right;
    uint8_t padding_top;
    uint8_t padding_bottom;

};

struct Conv2dMetadata {
    mli_conv2d_cfg cfg;
    mli_tensor input;
    mli_tensor weights;
    mli_tensor output;
};

class DepthwiseConv2DPrivateData : public PrivateData {

public:
    DepthwiseConv2DPrivateData() : PrivateData(kDWConv2dId) {}

    // currently we support the only i8_w8_o32 case
    OffsetBuffer input_buffer;
    OffsetBuffer weights_buffer;
    OffsetBuffer output_buffer;
    OffsetBuffer inpzp_buffer;
    OffsetBuffer wtszp_buffer;

    uint32_t input_h;
    uint32_t input_w;
    uint32_t input_output_c;

    uint32_t output_h;
    uint32_t output_w;

    uint32_t weights_h;
    uint32_t weights_w;

    int32_t input_h_stride;
    int32_t input_w_stride;

    int32_t output_h_stride;
    int32_t output_w_stride;

    int32_t weights_h_stride;
    int32_t weights_w_stride;

    uint8_t stride_height;
    uint8_t stride_width;
    uint8_t dilation_height;
    uint8_t dilation_width;

    uint8_t padding_left;
    uint8_t padding_right;
    uint8_t padding_top;
    uint8_t padding_bottom;
};

struct DepthwiseConv2dMetadata {
    mli_conv2d_cfg cfg;
    mli_tensor input;
    mli_tensor weights;
    mli_tensor output;
};

class FullyConnectedPrivateData : public PrivateData {

public:
    FullyConnectedPrivateData() : PrivateData(kFullyConnectedId) {}

    // currently we support the only i8_w8_o32 case
    OffsetBuffer input_buffer;
    OffsetBuffer weights_buffer;
    OffsetBuffer output_buffer;
    OffsetBuffer inpzp_buffer;
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
};

struct FullyConnectedMetadata {
    mli_tensor input;
    mli_tensor weights;
    mli_tensor output;
};

class MovePrivateData : public PrivateData {

public:
    MovePrivateData() : PrivateData(kMoveId) {}

    Tensor<OffsetBuffer, Move_CS::kMaxRank> src;
    Tensor<OffsetBuffer, Move_CS::kMaxRank> dst;

    IteratorCfg<Move_CS::kMaxRank> src_cfg;
    IteratorCfg<Move_CS::kMaxRank> dst_cfg;
};

class Pool2DPrivateData : public PrivateData {

public:
    Pool2DPrivateData(kernel_id_t id) : PrivateData(id) {}

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
    EltwisePrivateData(kernel_id_t id) : PrivateData(id) {}
    OffsetBuffer m_in_left_buffer;
    OffsetBuffer m_in_right_buffer;
    OffsetBuffer m_output_buffer;

    uint32_t m_in_left_rank;
    uint32_t m_in_left_shape[4];
    int32_t m_in_left_stride[4];
    uint32_t m_in_right_rank;
    uint32_t m_in_right_shape[4];
    int32_t m_in_right_stride[4];
    uint32_t m_output_rank;
    uint32_t m_output_shape[4];
    int32_t m_output_stride[4];
};

class RescalePrivateData : public PrivateData {

public:
    RescalePrivateData() : PrivateData(kRescaleId) {}

    int32_t rescale_axis;

    uint32_t io_rank;

    OffsetBuffer input_buffer;
    OffsetBuffer output_buffer;
    OffsetBuffer encoded_params_buffer;

    uint32_t params_elem_num;
    
    uint32_t input_b;
    uint32_t input_h;
    uint32_t input_w;
    uint32_t input_c;

    uint32_t output_b;
    uint32_t output_h;
    uint32_t output_w;
    uint32_t output_c;

    int32_t input_b_stride;
    int32_t input_h_stride;
    int32_t input_w_stride;
    int32_t input_c_stride;

    int32_t output_b_stride;
    int32_t output_h_stride;
    int32_t output_w_stride;
    int32_t output_c_stride;
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

class ClipPrivateData : public PrivateData {

public:
    ClipPrivateData() : PrivateData(kClipId) {}

    uint32_t io_rank;

    // currently we support the only i8_w8_o32 case
    OffsetBuffer input_buffer;
    OffsetBuffer output_buffer;
    OffsetBuffer encoded_params_buffer;

    uint32_t params_elem_num;

    uint32_t input_b;
    uint32_t input_h;
    uint32_t input_w;
    uint32_t input_c;

    uint32_t output_b;
    uint32_t output_h;
    uint32_t output_w;
    uint32_t output_c;

    int32_t input_b_stride;
    int32_t input_h_stride;
    int32_t input_w_stride;
    int32_t input_c_stride;

    int32_t output_b_stride;
    int32_t output_h_stride;
    int32_t output_w_stride;
    int32_t output_c_stride;
};


} // namespace snps_arc::metaware::mli::ref

#endif // _MLI_REF_PRIVATE_TYPES_HPP_

