/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#include <cassert>
#include "test_tiling.hpp"

void strided_copy_with_offsets(uint32_t rank, uint32_t elem_size, const int8_t* src, const int32_t* src_offsets,
                               const int32_t* dst_offsets, const int32_t* strides, const uint32_t* size, int8_t* dst) {
  assert(rank > 0 && rank <= 5);
  int32_t offsets[5]{};
  uint32_t total = 1;
  for (int i = 0; i < rank; i++) {
    total *= size[i];
  }
  uint32_t cnt = 0;
  do {
    int32_t src_ind = 0, dst_ind = 0;
    for (int i = 0; i < rank; i++) {
      src_ind += (offsets[i] + src_offsets[i]) * strides[i] * elem_size;
      dst_ind += (offsets[i] + dst_offsets[i]) * strides[i] * elem_size;
    }
    for (int i = 0; i < elem_size; i++) {
      dst[dst_ind + i] = src[src_ind + i];
    }
    cnt++;

    for (int i = rank - 1; i >= 0; i--) {
      offsets[i]++;

      if (offsets[i] >= (int32_t) size[i]) offsets[i] = 0;
      else break;
    }
  } while (cnt < total);
}