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

namespace lib_mli = ::snps_arc::metaware::mli;

namespace snps_arc::metaware::mli::ref {

class MaxPool2DPrivateData : public PrivateData {

public:
    MaxPool2DPrivateData() : PrivateData(kMaxPool2DId) {}

    uint32_t io_elem_size;

    uint32_t input_offset;
    uint32_t output_offset;
    uint32_t tensor_data_offset;

    uint32_t input_w;
    uint32_t input_h;
    uint32_t input_c;
    uint32_t input_b;

    uint32_t output_w;
    uint32_t output_h;
    uint32_t output_c;
    uint32_t output_b;

    int32_t descr_mem_id;
    int32_t input_mem_id;
    int32_t output_mem_id;

    int32_t input_w_stride;
    int32_t input_h_stride;
    int32_t input_c_stride;
    int32_t input_b_stride;

    int32_t output_w_stride;
    int32_t output_h_stride;
    int32_t output_c_stride;
    int32_t output_b_stride;

    uint8_t kernel_width;
    uint8_t kernel_height;
    uint8_t stride_width;
    uint8_t stride_height;
    uint8_t padding_left;
    uint8_t padding_right;
    uint8_t padding_top;
    uint8_t padding_bottom;
};

} // namespace snps_arc::metaware::mli::ref

#endif // _MLI_REF_PRIVATE_TYPES_HPP_