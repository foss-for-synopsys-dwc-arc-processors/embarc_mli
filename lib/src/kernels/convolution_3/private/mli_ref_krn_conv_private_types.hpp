/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_REF_KRN_CONV_PRIVATE_TYPES_HPP_
#define _MLI_REF_KRN_CONV_PRIVATE_TYPES_HPP_

#include "mli_types.h"
#include "mli_types.hpp"

namespace snps_arc::metaware::mli::ref{

class DepthwiseConv2DPrivateData : public PrivateData {

public:
    DepthwiseConv2DPrivateData() : PrivateData(kDWConv2dId) {}

    // uint32_t io_elem_size; // currently we support the only i8_w8_o32 case
    int32_t metadata_mem_id;
    int32_t input_mem_id;
    int32_t weights_mem_id;
    int32_t output_mem_id;
    uint32_t input_zp_mem_id;

    uint32_t metadata_offset;
    uint32_t input_mem_offset;
    uint32_t weights_mem_offset;
    uint32_t output_mem_offset;
    uint32_t input_zp_mem_offset;

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
    int16_t* input_zero_point;
};

}

#endif // _MLI_REF_KRN_CONV_PRIVATE_TYPES_HPP_