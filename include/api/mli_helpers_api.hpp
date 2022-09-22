/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_HELPERS_API_HPP_
#define _MLI_HELPERS_API_HPP_

#include <tuple>
#include "mli_iterator.hpp"

namespace snps_arc::metaware::mli {

typedef TensorIterator<NoBuffer, kTransposeConvIORank, kTransposeConvIOIterRank> DeconvIOTensorIterator;
typedef TensorIterator<NoBuffer, kTransposeConvWRank, kTransposeConvWIterRank> DeconvWTensorIterator;
typedef TensorIterator<NoBuffer, kTransposeConvZPRank, kTransposeConvZPIterRank> DeconvZPTensorIterator;
typedef std::tuple<DeconvIOTensorIterator, DeconvWTensorIterator,
                   DeconvZPTensorIterator, DeconvIOTensorIterator> DeconvTensorIteratorTuple;

DeconvTensorIteratorTuple GetDeconvTensorIterators(Tensor<NoBuffer, kTransposeConvIORank> input_tensor,
                                                   uint32_t input_tile_size[kTransposeConvIORank],
                                                   const Tensor<NoBuffer, kTransposeConvIORank>& output_tensor,
                                                   uint32_t output_tile_oc,
                                                   const Tensor<NoBuffer, kTransposeConvWRank>& weights_tensor,
                                                   uint32_t effective_kernel_size[kTransposeConvIORank],
                                                   uint32_t stride[kTransposeConvIORank],
                                                   uint32_t pre_padding[kTransposeConvIORank],
                                                   const int32_t order[kTransposeConvIORank]);

} // namespace mli

#endif // _MLI_HELPERS_API_HPP_
