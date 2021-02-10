/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_GROUP_CONV2D_VDSP_H_
#define _MLI_KRN_GROUP_CONV2D_VDSP_H_

#include "mli_api.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"
#include "mli_math.h"
#include "mli_private_types.h"
#include "mli_types.h"
#include "mli_krn_dotprod.h"
#include "mli_prv_layout.h"
#include "mli_krn_convolution.h"

namespace mli {
namespace krn {
namespace vdsp {

#pragma MLI_CODE_SECTION_START(".mli_lib")

//========================================================
// Group Convolution 2D
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void group_convolution2D(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &weights,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        quant_T quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {
    mli::krn::convolution2D<io_T, w_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height, /*group_conv2d*/ true>(
            in, weights, biases, out, perception_area, quant_params,
            val_min_limit, val_max_limit,
            stride_height, stride_width, dilation_height, dilation_width,
            padding_top, padding_left,
            padding_bot, padding_right);
}

#pragma MLI_CODE_SECTION_END()
} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_GROUP_CONV2D_VDSP_H_
