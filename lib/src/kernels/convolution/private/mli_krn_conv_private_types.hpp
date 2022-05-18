/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_CONV_PRIVATE_TYPES_HPP_
#define _MLI_KRN_CONV_PRIVATE_TYPES_HPP_

#include "mli_types.h"
#include "mli_types.hpp"

namespace snps_arc::metaware::mli {

class Conv2dPrivateData : public PrivateData {
  public:
    Conv2dPrivateData() : PrivateData(kConv2dId){}

};

namespace ref {

class DepthwiseConv2DPrivateData : public PrivateData {

public:
    DepthwiseConv2DPrivateData() : PrivateData(kDWConv2dId) {}

    // currently we support the only i8_w8_o32 case
    OffsetBuffer m_input_buffer;
    OffsetBuffer m_weights_buffer;
    OffsetBuffer m_output_buffer;
    OffsetBuffer m_inpzp_buffer;
    OffsetBuffer m_wtszp_buffer;
    OffsetBuffer m_metadata_buffer;

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

    uint32_t weights_h_stride;
    uint32_t weights_w_stride;

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
    mli_tensor input;
    mli_tensor weights;
    mli_conv2d_cfg cfg;
    mli_tensor output;
};

} // namespace ref

} // namespace mli

#endif // _MLI_KRN_CONV_PRIVATE_TYPES_HPP_