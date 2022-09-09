/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_TILING_HPP
#define _MLI_TILING_HPP

#include <cstdint>

// TODO: consider adding IterRank template
void strided_copy_with_offsets(uint32_t rank, uint32_t elem_size, const int8_t* src, const int32_t* src_offsets,
                               const int32_t* dst_offsets, const int32_t* strides, const uint32_t* size, int8_t* dst);

#endif	// _MLI_TILING_HPP
