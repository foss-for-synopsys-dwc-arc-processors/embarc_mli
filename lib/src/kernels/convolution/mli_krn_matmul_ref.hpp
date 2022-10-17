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


template <typename in1_t, typename in2_t,typename out_t, uint32_t rank>
void MatMul_prepare_and_run(Tensor<InternalBuffer, rank> &in_left, 
                            Tensor<InternalBuffer, rank> &in_right,
                            Tensor<InternalBuffer, rank> &output,
                            InternalBuffer &encoded_params) {
  /**
  * layout = HW
  * H of left = W of right
  * output shape must be of shape Hr * Wl
  * rank = 2
  */
  MLI_ASSERT(rank == kMatMulRank);
  MLI_ASSERT(in_left.get_dim(kMatMulWidthDim) == in_right.get_dim(kMatMulHeightDim));
  MLI_ASSERT(output.get_dim(kMatMulHeightDim) == in_left.get_dim(kMatMulHeightDim));
  MLI_ASSERT(output.get_dim(kMatMulWidthDim) == in_right.get_dim(kMatMulWidthDim));
  MLI_ASSERT(encoded_params.get_elem_size() == sizeof(int8_t));
  MLI_ASSERT(encoded_params.get_size() == kMatMulRank);

  in1_t val1;
  in2_t val2;
  out_t acc;
  int8_t in_left_zp = encoded_params.read<int8_t>(kMatMulHeightDim);
  int8_t in_right_zp = encoded_params.read<int8_t>(kMatMulWidthDim);
  uint32_t left_h = in_left.get_dim(kMatMulHeightDim);
  uint32_t right_w = in_right.get_dim(kMatMulWidthDim);
  uint32_t left_w = in_left.get_dim(kMatMulWidthDim);
  for(uint32_t i = 0; i < left_h; ++i) {
    for (uint32_t j = 0; j < right_w; ++j) {
      acc = 0;
      for (uint32_t k = 0; k < left_w; ++k) {
          val1 = in_left.template read<in1_t>(i * left_w + k) - in_left_zp;
          val2 = in_right.template read<in2_t>(k * right_w + j) - in_right_zp;
          acc += val1 * val2;
      }
        output.template write<out_t>( i * right_w + j, static_cast<out_t>(acc) );
    }
  }
}

#pragma MLI_CODE_SECTION_END()
} // namespace snps_arc::metaware::mli::ref
}
}
#endif // _MLI_KRN_CONVOLUTION_REF_H_
