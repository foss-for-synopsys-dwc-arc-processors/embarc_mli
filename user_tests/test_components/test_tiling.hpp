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


struct KernelInfo {
    uint32_t ky;
    uint32_t kx;
    uint32_t sy;
    uint32_t sx;
    uint32_t dy;
    uint32_t dx;
    uint32_t pt;
    uint32_t pb;
    uint32_t pl;
    uint32_t pr;
};

class Tiling {

    KernelInfo m_kernel_info;

    // BHWC
    uint32_t m_total_input_size[4];
    uint32_t m_total_output_size[4];
    uint32_t m_first_tile_size[4];
    uint32_t m_tile_size[4];
    uint32_t m_input_tile_first_increment[4];
    uint32_t m_output_tile_first_increment[4];
    uint32_t m_input_tile_increment[4];
    uint32_t m_output_tile_increment[4];

    uint32_t m_num_tiles;

public:
    Tiling(const uint32_t total_input_size[4], uint32_t tile_size[4],
           const KernelInfo& kernel_info, const uint32_t tile_oc, const uint32_t total_oc);
    
    uint32_t get_num_tiles() const;
    void get_io_tiles_parameters(uint32_t total_input_size[4], uint32_t total_output_size[4],
                                 uint32_t first_tile_size[4], uint32_t tile_size[4],
                                 uint32_t input_tile_first_inc[4], uint32_t output_tile_first_inc[4],
                                 uint32_t input_tile_inc[4], uint32_t output_tile_inc[4]) const;
};

void strided_copy_with_offsets(uint32_t rank, uint32_t elem_size, const int8_t* src, const uint32_t* src_offsets,
                               const uint32_t* dst_offsets, const int32_t* strides, const uint32_t* size, int8_t* dst);

#endif	// _MLI_TILING_HPP
