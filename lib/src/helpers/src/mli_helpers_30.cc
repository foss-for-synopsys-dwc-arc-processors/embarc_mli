#include "mli_helpers_api.hpp"

namespace snps_arc::metaware::mli {

DeconvTensorIteratorTuple GetDeconvTensorIterators(Tensor<NoBuffer, kTransposeConvIORank> input_tensor,
                                                   uint32_t input_tile_size[kTransposeConvIORank],
                                                   const Tensor<NoBuffer, kTransposeConvIORank>& output_tensor,
                                                   uint32_t output_tile_oc,
                                                   const Tensor<NoBuffer, kTransposeConvWRank>& weights_tensor,
                                                   uint32_t effective_kernel_size[kTransposeConvIORank],
                                                   uint32_t stride[kTransposeConvIORank],
                                                   uint32_t pre_padding[kTransposeConvIORank],
                                                   const int32_t order[kTransposeConvIORank]) {

  DeconvIOTensorIterator in_tensor_it(input_tensor, input_tile_size, order);
  uint32_t num_tiles_oc = CEIL_DIV(output_tensor.get_dim(kGroupTensorChannelDim), output_tile_oc);
  in_tensor_it.SetCount(num_tiles_oc, kGroupTensorChannelDim);
  auto in_tensor_it_cfg = in_tensor_it.get_config();

  // TODO: fix this it some how better than just disabling tiling
  if (in_tensor_it_cfg.get_last_size(kGroupTensorHeightDim) < weights_tensor.get_dim(kKernelHeightDim)) {
    const int32_t disable_mask[kTransposeConvIORank]{ 0, 1, 0, 0, 0 };
    in_tensor_it_cfg.template DisableTiling<NoBuffer, kTransposeConvIOIterRank>(disable_mask, in_tensor_it.get_tensor());
    in_tensor_it_cfg.SetCount(1, kGroupTensorHeightDim);
  }
  if (in_tensor_it_cfg.get_last_size(kGroupTensorWidthDim) < weights_tensor.get_dim(kKernelWidthDim)) {
    const int32_t disable_mask[kTransposeConvIORank]{ 0, 0, 1, 0, 0 };
    in_tensor_it_cfg.template DisableTiling<NoBuffer, kTransposeConvIOIterRank>(disable_mask, in_tensor_it.get_tensor());
    in_tensor_it_cfg.SetCount(1, kGroupTensorWidthDim);
  }
  in_tensor_it.set_config(in_tensor_it_cfg);

  DeconvIOTensorIterator out_tensor_it(output_tensor, in_tensor_it,
                                       effective_kernel_size, stride, pre_padding);

  // TODO: change TensorIterator ctor in a way this is not needed
  const int32_t inc_override[kTransposeConvIORank]{ -1, -1, -1, -1, (int32_t)output_tile_oc };
  out_tensor_it.OverrideIncrements(inc_override);


  in_tensor_it_cfg = in_tensor_it.get_config();
  const int32_t in_zero_inc_mask[kTransposeConvIORank] = {0, 0, 0, 0, 1};
  in_tensor_it_cfg.SetZeroIncrements(in_zero_inc_mask);
  in_tensor_it.set_config(in_tensor_it_cfg);

  const int32_t w_zero_inc_mask[kTransposeConvWRank]{ 1, 1, 1, 1, 0 };
  DeconvWTensorIterator w_tensor_it(weights_tensor, out_tensor_it, nullptr, w_zero_inc_mask);

  uint32_t wzp_shape[kTransposeConvZPRank]{ output_tensor.get_dim(kGroupTensorChannelDim) };
  const Tensor<NoBuffer, kTransposeConvZPRank> wzp_tensor(wzp_shape);
  const int32_t wzp_it_order[kTransposeConvZPIterRank]{ -1, -1, -1, -1, 0 };
  DeconvZPTensorIterator wzp_tensor_it(wzp_tensor, out_tensor_it, wzp_it_order, w_zero_inc_mask);

  return std::make_tuple(in_tensor_it, w_tensor_it, wzp_tensor_it, out_tensor_it);
}

}  // namespace snps_arc::metaware::mli