/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_MATMUL_REF_H_
#define _MLI_KRN_MATMUL_REF_H_

#include <cstring>
#include "mli_debug.h"
#include "mli_ref_runtime_api.hpp"
#include "mli_types.hpp"

using snps_arc::metaware::mli::InternalBuffer;
using snps_arc::metaware::mli::Tensor;
using snps_arc::metaware::mli::OffsetBuffer;
using snps_arc::metaware::mli::kMatMulHeightDim;
using snps_arc::metaware::mli::kMatMulWidthDim;
using snps_arc::metaware::mli::kMatMulRank;

namespace mli {
namespace krn {
namespace ref {

#pragma MLI_CODE_SECTION_START(".mli_lib")


template <typename in1_t, typename in2_t,typename out_t>
void MatMul_prepare_and_run(Tensor<InternalBuffer, kMatMulRank> &in_left, 
                            Tensor<InternalBuffer, kMatMulRank> &in_right,
                            Tensor<InternalBuffer, kMatMulRank> &output,
                            InternalBuffer &encoded_params) {
  /**
  * layout = HW
  * H of left = W of right
  * output shape must be of shape Hr * Wl
  * rank = 2
  */
  MLI_ASSERT(in_left.get_dim(kMatMulWidthDim) == in_right.get_dim(kMatMulHeightDim));
  MLI_ASSERT(output.get_dim(kMatMulHeightDim) == in_left.get_dim(kMatMulHeightDim));
  MLI_ASSERT(output.get_dim(kMatMulWidthDim) == in_right.get_dim(kMatMulWidthDim));
  MLI_ASSERT(encoded_params.get_elem_size() == sizeof(int8_t));
  MLI_ASSERT(encoded_params.get_size() == kMatMulRank);

  in1_t val1;
  in2_t val2;
  out_t acc;

  /**
  * leftzp is the first element of the encoded buffer.
  * rightzp is the second element of the encoded buffer.
  */
  int8_t in_left_zp = encoded_params.read<int8_t>(kMatMulHeightDim);
  int8_t in_right_zp = encoded_params.read<int8_t>(kMatMulWidthDim);
  uint32_t left_h = in_left.get_dim(kMatMulHeightDim);
  uint32_t right_w = in_right.get_dim(kMatMulWidthDim);
  uint32_t left_w = in_left.get_dim(kMatMulWidthDim);
  int32_t left_mem_strides[kMatMulRank];
  int32_t right_mem_strides[kMatMulRank];
  in_left.get_mem_strides(left_mem_strides);
  in_right.get_mem_strides(right_mem_strides);
  for(uint32_t left_height_index = 0; left_height_index < left_h; ++left_height_index) {
    for (uint32_t right_width_index = 0; right_width_index < right_w; ++right_width_index) {
      acc = 0;
      for (uint32_t left_width_index = 0, right_height_index = 0; left_width_index < left_w; ++left_width_index, ++right_height_index) {
          val1 = in_left.template read<in1_t>(left_height_index * left_mem_strides[0] + left_width_index) - in_left_zp;
          val2 = in_right.template read<in2_t>(right_height_index * right_mem_strides[0] + right_width_index) - in_right_zp;
          acc += val1 * val2;
      }
        output.template write<out_t>( left_height_index * right_w + right_width_index, static_cast<out_t>(acc) );
    }
  }
}

#pragma MLI_CODE_SECTION_END()
} // namespace snps_arc::metaware::mli::ref
}
}
#endif // _MLI_KRN_CONVOLUTION_REF_H_
