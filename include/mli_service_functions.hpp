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
#include "mli_debug.h"

#define DEPRECATED_METHOD printf("%s in %s:%d is deprecated. "\
                                 "It will be removed in future version of library.\n", __func__, __FILE__, __LINE__);

#define NOT_IMPLEMENTED_METHOD printf("%s in %s:%d is not implemented yet. \n", __func__, __FILE__, __LINE__);

namespace snps_arc::metaware::mli::service {

// TODO: return bytes instead of number of elements (bytes = number of elements  * size of element)
template<typename buf_T, unsigned tensorRank>
inline const unsigned GetBufferSize(const Tensor<buf_T, tensorRank>& tensor) {
  
  uint32_t shape[tensorRank];
  int32_t stride[tensorRank];
  tensor.get_dims(shape);
  tensor.get_mem_strides(stride);
  unsigned ret_val = 0;
  for (int dim = tensorRank - 1; dim >= 0; --dim) {
    ret_val += stride[dim] * (shape[dim] - 1);
  }
  ret_val += 1;
  return ret_val;
}

// TODO: return bytes instead of number of elements (bytes = number of elements  * size of element)
inline const unsigned GetBufferSize(int rank, const uint32_t* shape,
                                    const int32_t* stride) {
  unsigned ret_val = 0;
  for (int dim = rank - 1; dim >= 0; --dim) {
    ret_val += stride[dim] * (shape[dim] - 1);
  }
  ret_val += 1;
  return ret_val;
}

// TODO: return bytes instead of number of elements (bytes = number of elements  * size of element)
inline const unsigned GetBufferSize(int rank, const uint32_t* shape) {
  unsigned ret_val = 0;
  int32_t stride = 1;
  for (int dim = rank - 1; dim >= 0; --dim) {
    ret_val += stride * (shape[dim] - 1);
    stride *= shape[dim];
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


template <int channel_axis, unsigned rank>
mli_status EncodeZeroPts(const Tensor<Buffer, rank>& zeropts,
  Buffer& encoded_zeropts,
  int& quant_axis,
  uint32_t channel_length) {
  // should have the same total size, since the zp are not compressed.
  MLI_ASSERT(zeropts.get_buf().get_size() == encoded_zeropts.get_size());
  // the element size of source should equal to the encoded one's
  MLI_ASSERT(zeropts.get_elem_size() == encoded_zeropts.get_elem_size());
  // should have the same number of elements
  MLI_ASSERT(zeropts.get_dim(0) ==
    encoded_zeropts.get_size() / encoded_zeropts.get_elem_size());

  static_assert(rank > 0);
  unsigned tensor_channel_axis = rank - 1;
  if (zeropts.get_dim(tensor_channel_axis) == 1) {
    // per-tensor quantization
    quant_axis = -1;
  }
  else if (zeropts.get_dim(tensor_channel_axis) == channel_length) {
    // per-channel quantization
    quant_axis = channel_axis;
  }
  else {
    return MLI_STATUS_SHAPE_MISMATCH;
  }

  // NOTE: if we have more data types, we should think about
  // how to encode the zp, such as always using 8b in the encoded buffer.
  // Besides, we need to keep the consistency of ZP Encoding
  // between different platforms.
  //
  // For example, when we support 16b input tensor (zp has 16b as well),
  // here we still encode the zp as 8b. At runtime, we should restore the
  // data type based on the tensor's they belong to.
  //
  // All above are under the assumption that ZP are copied without compression.
  if (zeropts.get_elem_size() == sizeof(int8_t)) {
    for (uint32_t i = 0; i < zeropts.get_dim(tensor_channel_axis); ++i) {
      encoded_zeropts.write(i, zeropts.template read<int8_t>(i));
    }
  }
  else {
    return MLI_STATUS_NOT_SUPPORTED;
  }

  return MLI_STATUS_OK;
}

template <unsigned weights_rank>
mli_status EncodeWeightsAndZeroPts(const Tensor<Buffer, weights_rank>& weights,
                                   const Tensor<Buffer, kWZPRank>& weights_zp,
                                   Buffer& encoded) {

  Buffer w_buf = weights.get_buf();
  Buffer wzp_buf = weights_zp.get_buf();
  uint32_t w_el_size = w_buf.get_elem_size();
  uint32_t wzp_el_size = wzp_buf.get_elem_size();

  MLI_ASSERT(w_buf.get_size() + wzp_buf.get_size() == encoded.get_size());
  MLI_ASSERT(w_el_size == wzp_el_size);
  MLI_ASSERT(w_el_size == encoded.get_elem_size());

  if (w_el_size == sizeof(int8_t) && wzp_el_size == sizeof(int8_t)) {
    for (uint32_t i = 0; i < w_buf.get_size(); ++i) {
      encoded.write(i, w_buf.template read<int8_t>(i));
    }
    for (uint32_t i = 0; i < wzp_buf.get_size(); ++i) {
      encoded.write(w_buf.get_size() + i, wzp_buf.template read<int8_t>(i));
    }
    return MLI_STATUS_OK;
  }
  return MLI_STATUS_NOT_SUPPORTED;
}

inline const uint32_t get_conv_input_size(uint32_t output_size, uint32_t padding,
                                          uint32_t kernel_size, uint32_t dilation, uint32_t stride) {
  int32_t input_size = output_size * stride + (kernel_size - 1) * dilation;
  input_size -= padding;
  MLI_ASSERT(input_size > 0);
  return (uint32_t)input_size;
}

inline const uint32_t get_effective_kernel_size(uint32_t kernel_size, uint32_t dilation) {
    MLI_ASSERT(kernel_size > 0 && dilation > 0);
    return (kernel_size - 1) * dilation + 1;
}

inline const int32_t get_last_increment(uint32_t number_of_tiles,  int32_t first_increment, int32_t increment){
  /**
   * Following formula is not very intuitive, so here is explanation:
   * increment[i] * (number_of_tiles - 1)                   - offset of last tile in case of first_increment == increment
   * increment[i] * (number_of_tiles - 2) + first_increment - offset of last tile in case of first_increment != increment
   */
  return -(increment * ((int32_t)number_of_tiles - 2) + first_increment);
}

template <typename T>
void ReconstructTensor(InternalBuffer& internal_buffer, mli_tensor& tensor,
                         uint32_t rank, uint32_t* shape, int32_t* stride ) {
  tensor.rank = rank;
  // assign pointer or val
  if (tensor.rank != 0) {
    MLI_ASSERT(internal_buffer.get_ptr<T>() != nullptr);
    mli_prv_tensor_set_data_ptr(&tensor, internal_buffer.get_ptr<T>());
  } else {
    if constexpr(sizeof(T) == sizeof(int8_t)) {
      tensor.data.mem.i8 = internal_buffer.read<T>(0);
    } else if constexpr(sizeof(T) == sizeof(int16_t)) {
      tensor.data.mem.i16 = internal_buffer.read<T>(0);
    } else if constexpr(sizeof(T) == sizeof(int32_t)) {
      tensor.data.mem.i32 = internal_buffer.read<T>(0);
    } else {
      MLI_ASSERT(false);
    }
  }
  // assgin shape and stride
  for (uint32_t i = 0; i < tensor.rank; ++i) {
    tensor.shape[i] = shape[i];
    tensor.mem_stride[i] = stride[i];
  }
}

template<int rank>
inline void set_default_align(uint32_t align[rank]) {
  for (uint32_t i = 0; i < rank; i++) {
    align[i] = 1;
  }
}

}  // namespace snps_arc::metaware::mli::service

#endif /* _MLI_SERVICE_FUNCTIONS_HPP_ */
