/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_CONVOLUTION_DECL_REF_H_
#define _MLI_KRN_CONVOLUTION_DECL_REF_H_

#include "mli_config.h"
#include "mli_prv_quant.h"
#include "mli_types.h"
#include "mli_prv_layout.h"

namespace mli {
typedef enum {
    CONV_GENERAL = 0,
    CONV_DEPTHWISE
} mli_conv_type;

namespace krn {
////////////////////////////////////////////////////////////////////////////////
// Functions (in *_ref/*_dsp/*vdsp) that can be called from outside their own
// file must be declared here. This includes all overloads. For example, if we
// have: io_T f(io_T a) and int8_t f(int8_t a), then both must be declared.
// Not doing so, can cause the compiler to use the wrong overload.
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// REF
////////////////////////////////////////////////////////////////////////////////
namespace ref {
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void convolution2D(
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
        const int padding_bot, const int padding_right);

template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void depthwise_convolution2D(
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
        const int padding_bot, const int padding_right);

template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T,
          mli_layout_type data_layout, mli_conv_type conv_type>
MLI_FORCE_INLINE void conv2d_prepare_and_run(
        const mli_tensor *in,
        const mli_tensor *weights,
        const mli_tensor *bias,
        const mli_conv2d_cfg *cfg,
        mli_tensor *out,
        const int fix_kernel_width = 0,
        const int fix_kernel_height = 0);
} // namespace ref

////////////////////////////////////////////////////////////////////////////////
// DSP
////////////////////////////////////////////////////////////////////////////////
namespace dsp {
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void depthwise_convolution2D(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &w,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        quant_T quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right);
} // namespace dsp

////////////////////////////////////////////////////////////////////////////////
// VDSP
////////////////////////////////////////////////////////////////////////////////
namespace vdsp {
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void convolution2D(
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
        const int padding_bot, const int padding_right);

template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T>
MLI_FORCE_INLINE void depthwise_convolution2D(
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
        const int padding_bot, const int padding_right);

} // namespace vdsp

} // namespace krn
} // namespace mli

#endif // _MLI_KRN_CONVOLUTION_DECL_REF_H_
