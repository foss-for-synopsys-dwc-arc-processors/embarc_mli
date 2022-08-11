/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_SERVICE_FUNCTIONS_HPP_
#define _MLI_SERVICE_FUNCTIONS_HPP_

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#define DEPRECATED_METHOD printf("%s in %s:%d is deprecated. "\
                                 "It will be removed in future version of library.\n", __func__, __FILE__, __LINE__);

namespace snps_arc::metaware::mli::service {

inline const unsigned GetBufferSize(int rank, const uint32_t* shape,
                                    const int32_t* stride) {
  unsigned ret_val = 0;
  for (int dim = rank - 1; dim >= 0; --dim) {
    ret_val += stride[dim] * (shape[dim] - 1);
  }
  ret_val += 1;
  return ret_val;
}

template <unsigned rank>
mli_status EncodeWeights(const Tensor<Buffer, rank> &weights,
                         Buffer &encoded_weights) {
  // the element size of source should eqaul to the encoded one's
  MLI_ASSERT(weights.get_buf().get_size() == encoded_weights.get_size());

  if (weights.get_elem_size() == sizeof(int8_t)) {
    for (uint32_t i = 0; i < weights.get_buf().get_size(); ++i) {
      encoded_weights.write(i, weights.template read<int8_t>(i));
    }
  } else {
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}


template <int channel_axis>
mli_status EncodeZeroPts(const Tensor<Buffer, 1>& zeropts,
  Buffer& encoded_zeropts,
  int& quant_axis,
  uint32_t channel_length) {
  // should have the same total size
  MLI_ASSERT(zeropts.get_buf().get_size() == encoded_zeropts.get_size());
  // the element size of source should less than or equal to the encoded one's
  MLI_ASSERT(zeropts.get_elem_size() <= encoded_zeropts.get_elem_size());
  // should have the same number of elements
  MLI_ASSERT(zeropts.get_dim(0) ==
    encoded_zeropts.get_size() / encoded_zeropts.get_elem_size());

  if (zeropts.get_dim(0) == 1) {
    // per-tensor quantization
    quant_axis = -1;
  }
  else if (zeropts.get_dim(0) == channel_length) {
    // per-channel quantization
    quant_axis = channel_axis;
  }
  else {
    return MLI_STATUS_SHAPE_MISMATCH;
  }

  if (zeropts.get_elem_size() == sizeof(int8_t)) {
    for (uint32_t i = 0; i < zeropts.get_dim(0); ++i) {
      encoded_zeropts.write(i, static_cast<int16_t>(zeropts.read<int8_t>(i)));
    }
  }
  else {
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}


inline const uint32_t get_conv_input_size(uint32_t output_size, uint32_t padding,
                                          uint32_t kernel_size, uint32_t dilation, uint32_t stride) {
  int32_t input_size = output_size * stride + (kernel_size - 1) * dilation;
  input_size -= padding;
  MLI_ASSERT(input_size > 0);
  return (uint32_t)input_size;
}



inline const int32_t get_last_increment(uint32_t number_of_tiles,  int32_t first_increment, int32_t increment){
  /**
   * Following formula is not very intuitive, so here is explanation:
   * increment[i] * (number_of_tiles - 1)                   - offset of last tile in case of first_increment == increment
   * increment[i] * (number_of_tiles - 2) + first_increment - offset of last tile in case of first_increment != increment
   */
  return -(increment * ((int32_t)number_of_tiles - 2) + first_increment);
}

}  // namespace snps_arc::metaware::mli::service

#endif /* _MLI_SERVICE_FUNCTIONS_HPP_ */