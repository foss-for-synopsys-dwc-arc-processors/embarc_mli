/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_TRANSPOSE_CONV_REF_H_
#define _MLI_KRN_TRANSPOSE_CONV_REF_H_

#include "mli_api.h"
#include "mli_krn_convolution.h"
#include "mli_private_types.h"
#include "mli_prv_quant.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace ref {

#pragma MLI_CODE_SECTION_START(".mli_lib")

//===================================================================
// Constructing mirrored weights subtensor for a specific pattern
//===================================================================
template <typename T>
static MLI_FORCE_INLINE conv2d_weights_tensor_private_t<T> get_mirrored_weights_subtensor_hwcn(
        const conv2d_weights_tensor_private_t<T> &weights,
        const int weights_stride_width,
        const int weights_stride_height,
        int krn_left_offset,
        int krn_top_offset) {
    MLI_ASSERT(weights_stride_width > 0);
    MLI_ASSERT(weights_stride_height > 0);
    MLI_ASSERT(krn_left_offset >= 0);
    MLI_ASSERT(krn_top_offset >= 0);

    auto result_subtensor = weights;

    // Find out kernel size for sub tensor first.
    result_subtensor.kernel_height = CEIL_DIV(weights.kernel_height - krn_top_offset, weights_stride_height);
    result_subtensor.kernel_width = CEIL_DIV(weights.kernel_width - krn_left_offset, weights_stride_width);

    // Get offsets to mirror original tensor
    int mem_offset = 0;
    mem_offset += weights.row_mem_stride * (weights.kernel_height - 1);
    mem_offset += weights.col_mem_stride * (weights.kernel_width - 1);
    result_subtensor.col_mem_stride = -result_subtensor.col_mem_stride;
    result_subtensor.row_mem_stride = -result_subtensor.row_mem_stride;

    // Do offset adjustment according to a specific subkernel.
    mem_offset += result_subtensor.row_mem_stride * krn_top_offset;
    mem_offset += result_subtensor.col_mem_stride * krn_left_offset;
    
    // Update ptr and memstrides to take strides across weights into account.
    result_subtensor.ptr += mem_offset;
    result_subtensor.col_mem_stride *= weights_stride_width;
    result_subtensor.row_mem_stride *= weights_stride_height;
    return result_subtensor;
}

//========================================================
// Main Transpose Convolution Routine
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T, int conv_fix_kernel_width, int conv_fix_kernel_height>
MLI_FORCE_INLINE void transpose_convolution2D(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &weights,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        quant_T quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {
    mli::krn::convolution2D<io_T, w_T, b_T, acc_T, quant_T, conv_fix_kernel_width, conv_fix_kernel_height>(
        in, weights, biases, out, perception_area, quant_params,
        val_min_limit, val_max_limit,
        /*stride_height = */1, /*stride_width = */1,
        /*dilation_height=  */1, /*dilation_width =  */1,
        padding_top, padding_left, padding_bot, padding_right);
}

//====================================================================================
// Common routine  for pre-calculation of various convolution parameters and running it.
//====================================================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height, int fix_stride>
MLI_FORCE_INLINE void transpose_conv2d_prepare_and_run(
        const mli_tensor *in,
        const mli_tensor *weights,
        const mli_tensor *bias,
        const mli_conv2d_cfg *cfg,
        mli_tensor *out) {
    mli_prv_fx_init_dsp_ctrl();

    constexpr int conv_fix_kernel_width = (fix_stride == 2) ? fix_kernel_width / 2 : KRN_SZ_VAR;
    constexpr int conv_fix_kernel_height = (fix_stride == 2) ? fix_kernel_height / 2 : KRN_SZ_VAR;

    // Define output val limits (may affect built in ReLU)
    mli_minmax_t val_limit = mli_prv_get_relu_min_max(&cfg->relu, out);

    const MLI_PTR(b_T) bs = (MLI_PTR(b_T))(bias->data.mem.void_p);
    const auto in_prv = mli_prv_get_tensor_hwc<MLI_PTR(io_T)>(in);
    const auto weights_prv = mli_prv_get_conv2d_weights_tensor_hwcn<MLI_PTR(w_T)>(weights);

    // To calculate transpose convolution using general convolution we
    // need to derive effective generic convolution parameters from transpose ones
    const int stride_width = cfg->stride_width;
    const int stride_height = cfg->stride_height;
    const int effective_padding_top = weights_prv.kernel_height - cfg->padding_top - 1;
    const int effective_padding_bot = weights_prv.kernel_height - cfg->padding_bottom - 1;
    const int effective_padding_left = weights_prv.kernel_width - cfg->padding_left - 1;
    const int effective_padding_right = weights_prv.kernel_width - cfg->padding_right - 1;
    const int effective_in_width = (in_prv.width - 1) * stride_width + 1;
    const int effective_in_height = (in_prv.height - 1) * stride_height + 1;

    const int out_width  = effective_in_width + effective_padding_left + effective_padding_right
                           - weights_prv.kernel_width + 1;
    const int out_height = effective_in_height + effective_padding_top + effective_padding_bot 
                           - weights_prv.kernel_height + 1;
    const int out_ch = weights_prv.out_ch;

    out->el_type = in->el_type;
    out->shape[FMAP_H_DIM_HWC] = out_height;
    out->shape[FMAP_W_DIM_HWC] = out_width;
    out->shape[FMAP_C_DIM_HWC] = out_ch;

    const auto out_prv = mli_prv_get_tensor_hwc<MLI_CONV_OUT_PTR(io_T)>(out);

    quant_T quant_params;
    define_quant_params(in, weights, bias, out, &quant_params);

    // Applying main convolution for each subtensor of weights pattern independently
    //=======================================================================
    // There are two main ways to calculate transpose convolution using general conv2d:
    // 1) mirror weights, extend input with zeroes between each pixel according to strides params, 
    //    apply a general convolution
    // 2) mirror weights, create several patterns from it according to strides, and apply
    //      general convolution of each weights subtensor on input. Results need to be concatenated
    // 2nd option looks more complicated, but doesn't require extra memory for input, can be implemented
    // using memstrides for weights and output, and more efficient as no extra multiplications with 0 
    // is needed. 
    for (int krn_h_offset = 0; krn_h_offset < stride_height; krn_h_offset++) {
        for (int krn_w_offset = 0; krn_w_offset < stride_width; krn_w_offset++) {
            const auto weights_subtensor = 
                get_mirrored_weights_subtensor_hwcn(weights_prv, stride_width, stride_height, 
                                                    krn_w_offset, krn_h_offset);
            
            // Loping across kernel patterns defined by krn_*_offset we need to define exact out subtensor
            // which is being filled with current weights subtensor. For this we need to define out subtensor size 
            // and offset in the whole output tensor using paddings and strides.
            auto cur_out = out_prv;
            const int out_w_offset = (stride_width - krn_w_offset + effective_padding_left) % stride_width;
            const int out_h_offset = (stride_height - krn_h_offset + effective_padding_top) % stride_height;
            const int cur_out_height = CEIL_DIV(out_height - out_h_offset, stride_height);
            const int cur_out_width = CEIL_DIV(out_width - out_w_offset, stride_width);
            int out_mem_offset = out_prv.row_mem_stride * out_h_offset;
            out_mem_offset += out_prv.col_mem_stride * out_w_offset;
            cur_out.height = cur_out_height;
            cur_out.width = cur_out_width;
            cur_out.col_mem_stride *= stride_width;
            cur_out.row_mem_stride *= stride_height;
            cur_out.ptr += out_prv.row_mem_stride * out_h_offset;
            cur_out.ptr += out_prv.col_mem_stride * out_w_offset;

            // Having small or no padding at top/left, krn_*_offset may reduce input area for calculations.
            auto cur_in = in_prv;
            const int in_w_offset = MAX(0, krn_w_offset - effective_padding_left); 
            const int in_h_offset = MAX(0, krn_h_offset - effective_padding_top); 
            cur_in.height -= in_h_offset;
            cur_in.width -= in_w_offset;
            cur_in.ptr += cur_in.row_mem_stride * in_h_offset + cur_in.col_mem_stride * in_w_offset;

            // Define paddings for the current calculations according to the new kernel size and offset index.
            // Right and bottom paddings are derived from other already known parameters:
            // current input size, current output size, current kernel size, and current left or top padding.
            const int cur_pad_left = MAX(0, effective_padding_left - krn_w_offset) / stride_width;
            const int cur_pad_top = MAX(0, effective_padding_top - krn_h_offset) / stride_height;
            const int cur_pad_bot = 
                (cur_out_height + (weights_subtensor.kernel_height - 1)) - cur_in.height - cur_pad_top;
            const int cur_pad_right = 
                (cur_out_width + (weights_subtensor.kernel_width - 1)) // how much in point are required for cur output
                - cur_in.width - cur_pad_left;                         // subtract real input points and left padding

            rect_t cent_area;
            cent_area.row_beg = 0;  cent_area.row_end = cur_out_height;
            cent_area.clmn_beg = 0; cent_area.clmn_end = cur_out_width;
            transpose_convolution2D<io_T, w_T, b_T, acc_T, quant_T, conv_fix_kernel_width, conv_fix_kernel_height>(
                cur_in, weights_subtensor, bs, cur_out, cent_area, quant_params,
                (io_T)val_limit.min, (io_T)val_limit.max,
                cur_pad_top, cur_pad_left, cur_pad_bot, cur_pad_right);
        }
    }
}
#pragma MLI_CODE_SECTION_END()
} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_TRANSPOSE_CONV_REF_H_
