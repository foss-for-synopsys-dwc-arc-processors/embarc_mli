/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#include "mli_math_macros.h"
#include "mli_debug.h"
#include "test_tiling.hpp"
#include "mli_compiler_api.hpp"


using ::snps_arc::metaware::mli::kTensorBatchDim;
using ::snps_arc::metaware::mli::kTensorHeightDim;
using ::snps_arc::metaware::mli::kTensorWidthDim;
using ::snps_arc::metaware::mli::kTensorChannelDim;


static uint32_t mli_tiling_get_conv_output_size(int input_padded_size, int kernel_size, int stride, int dilation) {
    return CEIL_DIV(input_padded_size - (kernel_size - 1) * dilation, stride);
}


static void mli_tiling_get_start_end_of_tile(uint32_t input_offset, uint32_t input_size, uint32_t tile_size,
                            uint32_t overlap, uint32_t pad_1, uint32_t pad_2, uint32_t kernel_size, uint32_t stride,
                            uint32_t& start, uint32_t& end, int&effective_start, uint32_t& effective_end
                                ){
    effective_start = input_offset ? (input_offset - overlap) : (input_offset - pad_1);
    start = MAX(0, effective_start);
    effective_end = start + tile_size;
    if (effective_end >= input_size) effective_end = input_size + pad_2;
    MLI_ASSERT(effective_end - effective_start - kernel_size >= 0);
    effective_end -= ((effective_end - effective_start - kernel_size) % stride);
    MLI_ASSERT(effective_end - effective_start >= kernel_size);
    end = MIN(effective_end, input_size);
}

static uint32_t mli_tiling_check_and_fix_input_size(uint32_t overlap, uint32_t total_input_size, uint32_t tile_input_size, uint32_t pad_1, uint32_t pad_2, uint32_t kernel_size, uint32_t stride) {
    uint32_t input_offset = 0;
    uint32_t fixed_input_size = total_input_size;
    while (input_offset < total_input_size) {
        int effective_start = input_offset ? (input_offset - overlap) : (input_offset - pad_1);
        uint32_t start = MAX(0, effective_start);
        uint32_t effective_end = start + tile_input_size;
        if (effective_end >= fixed_input_size) effective_end = fixed_input_size + pad_2;
        uint32_t effective_tile_input_size = effective_end - effective_start;
        if (effective_tile_input_size < kernel_size) {
            fixed_input_size = input_offset;
            break;
        }
        effective_end -= ((effective_end - effective_start - kernel_size) % stride);
        effective_tile_input_size = effective_end - effective_start;
        if (effective_tile_input_size < kernel_size) {
            fixed_input_size = input_offset;
            break;
        }
        uint32_t end = MIN(effective_end, total_input_size);
        input_offset += (end - input_offset);
    }
    MLI_ASSERT(input_offset == fixed_input_size);
    return fixed_input_size;
}


Tiling::Tiling(const uint32_t total_input_size[4], uint32_t tile_size[4],
               const KernelInfo& kernel_info, const uint32_t tile_oc, const uint32_t total_oc) {
    
    /********************************************************************/
    // validate (and maybe fix) input size, input tile size, right/bottom padding 
    MLI_ASSERT(tile_oc > 0 && total_oc > 0);
    MLI_ASSERT(total_input_size[kTensorBatchDim] > 0 && total_input_size[kTensorHeightDim] > 0);
    MLI_ASSERT(total_input_size[kTensorWidthDim] > 0 && total_input_size[kTensorChannelDim] > 0);
    MLI_ASSERT(kernel_info.dy > 0 && kernel_info.dx > 0);
    MLI_ASSERT(kernel_info.ky > 0 && kernel_info.kx > 0);
    MLI_ASSERT(tile_size[kTensorBatchDim] > 0 && tile_size[kTensorChannelDim] > 0);
    MLI_ASSERT(kernel_info.sy > 0 && kernel_info.sx > 0);
    MLI_ASSERT(kernel_info.pt >= 0 && kernel_info.pb >= 0 && kernel_info.pl >= 0 && kernel_info.pr >= 0);
    uint32_t effective_kernel_h = (kernel_info.ky - 1) * kernel_info.dy + 1;
    MLI_ASSERT(kernel_info.pt < effective_kernel_h && kernel_info.pb < effective_kernel_h);
    uint32_t effective_kernel_w = (kernel_info.kx - 1) * kernel_info.dx + 1;
    MLI_ASSERT(kernel_info.pl < effective_kernel_w && kernel_info.pr < effective_kernel_w);
    uint32_t overlap_y = effective_kernel_h - kernel_info.sy;
    uint32_t overlap_x = effective_kernel_w - kernel_info.sx;

    m_total_input_size[kTensorBatchDim] = total_input_size[kTensorBatchDim];
    m_total_input_size[kTensorChannelDim] = total_input_size[kTensorChannelDim];

    int required_h = 0, required_w = 0;
    uint32_t padded_input_h = 0, padded_input_w = 0;
    uint32_t pad_y_descrease = 0, pad_x_descrease = 0;
    uint32_t fixed_tile_size_y = 0, fixed_tile_size_x = 0;
    while (1) {
      
      uint32_t tile_residual_y = 0;
      pad_y_descrease = 0;
      if (tile_size[kTensorHeightDim] == total_input_size[kTensorHeightDim]) {
        tile_residual_y = (tile_size[1] + kernel_info.pt + kernel_info.pb - effective_kernel_h) % kernel_info.sy;
        if (tile_residual_y && tile_residual_y <= kernel_info.pb) {
          pad_y_descrease = (kernel_info.pb - tile_residual_y);
          tile_residual_y = 0;
        }
      }
      else tile_residual_y = (tile_size[kTensorHeightDim] - effective_kernel_h) % kernel_info.sy;
      MLI_ASSERT(tile_residual_y >= 0);
      fixed_tile_size_y = tile_size[kTensorHeightDim] - tile_residual_y;
      m_total_input_size[kTensorHeightDim] = mli_tiling_check_and_fix_input_size(
        overlap_y, total_input_size[kTensorHeightDim], fixed_tile_size_y,
        kernel_info.pt, kernel_info.pb - pad_y_descrease, effective_kernel_h, kernel_info.sy
      );
      if (tile_size[kTensorHeightDim] == total_input_size[kTensorHeightDim]) {
        if (tile_size[kTensorHeightDim] + kernel_info.pt + kernel_info.pb - pad_y_descrease < effective_kernel_h) {
          tile_size[kTensorHeightDim]++;
          continue;
        }
      }
      else {
        if (tile_size[kTensorHeightDim] < effective_kernel_h) {
          tile_size[kTensorHeightDim]++;
          continue;
        }
      }
      uint32_t tile_residual_x = 0;
      pad_x_descrease = 0;
      if (tile_size[kTensorWidthDim] == total_input_size[kTensorWidthDim]) {
        tile_residual_x = (tile_size[kTensorWidthDim] + kernel_info.pl + kernel_info.pr - effective_kernel_w) % kernel_info.sx;
        if (tile_residual_x && tile_residual_x <= kernel_info.pr) {
          pad_x_descrease = (kernel_info.pr - tile_residual_x);
          tile_residual_x = 0;
        }
      }
      else tile_residual_x = (tile_size[kTensorWidthDim] - effective_kernel_w) % kernel_info.sx;
      MLI_ASSERT(tile_residual_x >= 0);
      fixed_tile_size_x = tile_size[kTensorWidthDim] - tile_residual_x;
      m_total_input_size[kTensorWidthDim] = mli_tiling_check_and_fix_input_size(
        overlap_x, total_input_size[kTensorWidthDim], fixed_tile_size_x, kernel_info.pl, kernel_info.pr - pad_x_descrease, effective_kernel_w, kernel_info.sx
      );
      if (tile_size[kTensorWidthDim] == total_input_size[kTensorWidthDim]) {
        if (tile_size[kTensorWidthDim] + kernel_info.pl + kernel_info.pr - pad_x_descrease < effective_kernel_w) {
          tile_size[kTensorWidthDim]++;
          continue;
        }
      }
      else {
        if (tile_size[kTensorWidthDim] < effective_kernel_w) {
          tile_size[kTensorWidthDim]++;
          continue;
        }
      }

      m_total_output_size[kTensorBatchDim] = m_total_input_size[kTensorBatchDim];
      padded_input_h = m_total_input_size[kTensorHeightDim] + kernel_info.pt + kernel_info.pb - pad_y_descrease;
      m_total_output_size[kTensorHeightDim] = mli_tiling_get_conv_output_size(
        padded_input_h, kernel_info.ky, kernel_info.sy, kernel_info.dy
      );
      padded_input_w = m_total_input_size[kTensorWidthDim] + kernel_info.pl + kernel_info.pr - pad_x_descrease;
      m_total_output_size[kTensorWidthDim] = mli_tiling_get_conv_output_size(
        padded_input_w, kernel_info.kx, kernel_info.sx, kernel_info.dx
      );
      m_total_output_size[kTensorChannelDim] = total_oc;

      required_w = m_total_output_size[kTensorWidthDim] * kernel_info.sx + effective_kernel_w - kernel_info.sx;
      required_h = m_total_output_size[kTensorHeightDim] * kernel_info.sy + effective_kernel_h - kernel_info.sy;

      if (required_w <= (int)padded_input_w && required_h <= (int)padded_input_h) break;
      else {
        if (required_w > (int)padded_input_w) tile_size[kTensorWidthDim]++;
        if (required_h > (int)padded_input_h) tile_size[kTensorHeightDim]++;
      }

    } 


    m_kernel_info = kernel_info;
    m_kernel_info.pb -= pad_y_descrease;
    m_kernel_info.pr -= pad_x_descrease;

    /********************************************************************/
    // calculate tiling parameters

    m_tile_size[kTensorBatchDim] = tile_size[kTensorBatchDim];
    m_first_tile_size[kTensorBatchDim] = tile_size[kTensorBatchDim];
    m_input_tile_increment[kTensorBatchDim] = tile_size[kTensorBatchDim];
    m_output_tile_increment[kTensorBatchDim] = tile_size[kTensorBatchDim];
    m_input_tile_first_increment[kTensorBatchDim] = tile_size[kTensorBatchDim];
    m_output_tile_first_increment[kTensorBatchDim] = tile_size[kTensorBatchDim];


    m_tile_size[kTensorHeightDim] = fixed_tile_size_y;
    m_input_tile_increment[kTensorHeightDim] = fixed_tile_size_y;
    if (fixed_tile_size_y < m_total_input_size[kTensorHeightDim]) {
      m_input_tile_increment[kTensorHeightDim] -= overlap_y;
      m_output_tile_increment[kTensorHeightDim] = mli_tiling_get_conv_output_size(
        fixed_tile_size_y, kernel_info.ky, m_kernel_info.sy, m_kernel_info.dy
      );
    }
    else {
      m_input_tile_increment[kTensorHeightDim] = fixed_tile_size_y;
      m_output_tile_increment[kTensorHeightDim] = mli_tiling_get_conv_output_size(
        padded_input_h, kernel_info.ky, m_kernel_info.sy, m_kernel_info.dy
      );
    }
    uint32_t start_y = 0, end_y = 0, effective_end_y = 0;
    int effective_start_y = 0;
    mli_tiling_get_start_end_of_tile(0, m_total_input_size[kTensorHeightDim], fixed_tile_size_y,
        overlap_y, m_kernel_info.pt, m_kernel_info.pb, effective_kernel_h, m_kernel_info.sy,
        start_y, end_y, effective_start_y, effective_end_y
    );
    uint32_t first_tile_size_y = end_y - start_y;
    m_first_tile_size[kTensorHeightDim] = first_tile_size_y;
    m_input_tile_first_increment[kTensorHeightDim] = first_tile_size_y;
    if (fixed_tile_size_y < m_total_input_size[kTensorHeightDim]) m_input_tile_first_increment[kTensorHeightDim] -= overlap_y;
    m_output_tile_first_increment[kTensorHeightDim] = mli_tiling_get_conv_output_size(
        effective_end_y - effective_start_y, kernel_info.ky, m_kernel_info.sy, m_kernel_info.dy
    );


    m_tile_size[kTensorWidthDim] = fixed_tile_size_x;
    m_input_tile_increment[kTensorWidthDim] = fixed_tile_size_x;
    if (fixed_tile_size_x < m_total_input_size[kTensorWidthDim]) {
      m_input_tile_increment[kTensorWidthDim] -= overlap_x;
      m_output_tile_increment[kTensorWidthDim] = mli_tiling_get_conv_output_size(
        fixed_tile_size_x, m_kernel_info.kx, m_kernel_info.sx, m_kernel_info.dx
      );
    }
    else {
      m_input_tile_increment[kTensorWidthDim] = fixed_tile_size_x;
      m_output_tile_increment[kTensorWidthDim] = mli_tiling_get_conv_output_size(
        padded_input_w, kernel_info.kx, m_kernel_info.sx, m_kernel_info.dx
      );
    }
    uint32_t start_x = 0, end_x = 0, effective_end_x = 0;
    int effective_start_x = 0;
    mli_tiling_get_start_end_of_tile(0, m_total_input_size[kTensorWidthDim], fixed_tile_size_x,
        overlap_x, m_kernel_info.pl, m_kernel_info.pr, effective_kernel_w, m_kernel_info.sx,
        start_x, end_x, effective_start_x, effective_end_x
    );
    uint32_t first_tile_size_x = end_x - start_x;
    m_first_tile_size[kTensorWidthDim] = first_tile_size_x;
    m_input_tile_first_increment[kTensorWidthDim] = first_tile_size_x;
    if (fixed_tile_size_x < m_total_input_size[kTensorWidthDim]) m_input_tile_first_increment[kTensorWidthDim] -= overlap_x;
    m_output_tile_first_increment[kTensorWidthDim] = mli_tiling_get_conv_output_size(
        effective_end_x - effective_start_x, m_kernel_info.kx, m_kernel_info.sx, m_kernel_info.dx
    );


    m_tile_size[kTensorChannelDim] = tile_size[kTensorChannelDim];
    m_first_tile_size[kTensorChannelDim] = tile_size[kTensorChannelDim];
    m_input_tile_increment[kTensorChannelDim] = tile_size[kTensorChannelDim];
    m_output_tile_increment[kTensorChannelDim] = MIN(tile_oc, total_oc);
    m_input_tile_first_increment[kTensorChannelDim] = tile_size[kTensorChannelDim];
    m_output_tile_first_increment[kTensorChannelDim] = MIN(tile_oc, total_oc);

    /********************************************************************/
    // calculate number of tiles needed
    m_num_tiles = 1;
    for (int i = 0; i < 4; i++) {
        uint32_t tiles_per_dim = 1 + CEIL_DIV(m_total_output_size[i] - m_output_tile_first_increment[i], m_output_tile_increment[i]);
        m_num_tiles *= tiles_per_dim;
    }

}

uint32_t Tiling::get_num_tiles() const {
    return m_num_tiles;
}

void Tiling::get_io_tiles_parameters(uint32_t total_input_size[4], uint32_t total_output_size[4],
                                     uint32_t first_tile_size[4], uint32_t tile_size[4],
                                     uint32_t input_tile_first_inc[4], uint32_t output_tile_first_inc[4],
                                     uint32_t input_tile_inc[4], uint32_t output_tile_inc[4]) const {
    for (int i = 0; i < 4; i++) {
        total_input_size[i] = m_total_input_size[i];
        total_output_size[i] = m_total_output_size[i];
        input_tile_first_inc[i] = m_input_tile_first_increment[i];
        output_tile_first_inc[i] = m_output_tile_first_increment[i];
        input_tile_inc[i] = m_input_tile_increment[i];
        output_tile_inc[i] = m_output_tile_increment[i];
        first_tile_size[i] = m_first_tile_size[i];
        tile_size[i] = m_tile_size[i];
    }
}

void strided_copy_with_offsets(uint32_t rank, uint32_t elem_size, const int8_t* src, const uint32_t* src_offsets,
                               const uint32_t* dst_offsets, const int32_t* strides, const uint32_t* size, int8_t* dst) {
  assert(rank > 0 && rank <= 5);
  uint32_t offsets[5]{};
  uint32_t total = 1;
  for (int i = 0; i < rank; i++) {
    total *= size[i];
  }
  uint32_t cnt = 0;
  do {
    uint32_t src_ind = 0, dst_ind = 0;
    for (int i = 0; i < rank; i++) {
      src_ind += (offsets[i] + src_offsets[i]) * (uint32_t)strides[i] * elem_size;
      dst_ind += (offsets[i] + dst_offsets[i]) * (uint32_t)strides[i] * elem_size;
    }
    for (int i = 0; i < elem_size; i++) {
      dst[dst_ind + i] = src[src_ind + i];
    }
    cnt++;

    for (int i = rank - 1; i >= 0; i--) {
      offsets[i]++;

      if (offsets[i] >= size[i]) offsets[i] = 0;
      else break;
    }
  } while (cnt < total);
}
