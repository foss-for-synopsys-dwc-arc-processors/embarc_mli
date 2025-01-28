/*
* Copyright 2020-2025, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_KRN_CONVOLUTION_DSP_H_
#define _MLI_KRN_CONVOLUTION_DSP_H_

#include "mli_api.h"
#include "mli_prv_tensor.h"
#include "mli_prv_quant.h"
#include "mli_math.h"
#include "mli_mem_info.h"
#include "mli_private_types.h"
#include "mli_types.h"
#include "mli_krn_dotprod.h"
#include "mli_prv_layout.h"

namespace mli {
namespace krn {
namespace dsp {

#pragma MLI_CODE_SECTION_START(".mli_lib")
//========================================================
// Unified Generic Convolution 2D template
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
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
        const int padding_bot, const int padding_right) {
    // Unified Generic convolution for all layouts (CHW/HWC/HWCN) and quantization scheme:  
    // MLI_FX (symmetric data, scales are power of two) and s8asym (assymetric data, scales of any value)
    // For each output point Calculation implies dotproduct and bias add:
    //            out_val = sum_i(x_r * w_r) + b_r
    //
    // Considering assymetric types(x_r = (x - x_zp) and w_r = (w - w_zp)
    //                    out_val = sum_i((x-x_zp)*(w-w_zp)) + b_r
    //
    // after opening brackets:
    //      out_val = sum(x*w) - sum_i(w*x_zp) - sum_i(x*w_zp) + sum_i(w_zp*x_zp) + b_r
    // where:
    //      sum(x*w)       - generic dotproduct which can't be avoided for any type
    //      -sum_i(w*x_zp) - weights_additive. 
    //                       Allways Zero for FX and can be reused in output channel calculations for s8asym
    //      -sum_i(x*w_zp) - in_additive
    //                       Allways Zero for both FX and TF_s8asym assuming symmetric weights (w_zp == 0)
    //     sum_i(w_zp*x_zp)- zp_additive
    //                       Allways Zero for both FX and TF_s8asym assuming symmetric weights (w_zp == 0)
    //      b_r             - bias_additive
    //                        (must be of the same type as accumulator, that may require bias re-quantization)
    //
    // IMPORTANT NOTE: For border areas with padding, weights/input/zp can be reused only in case of explicitly padded values. 
    //                 In other case, these additives must be calculatid for valid area of dotproduct only.
    //================================================================================================
#ifdef __Xdsp_wide
    const int row_begin = perception_area.row_beg;
    const int row_end = perception_area.row_end;
    const int clmn_begin = perception_area.clmn_beg;
    const int clmn_end = perception_area.clmn_end;
	if (std::is_same<io_T, int16_t>::value && std::is_same<w_T, int16_t>::value && std::is_same<b_T, int16_t>::value && std::is_same<acc_T, mli_acc40_t>::value && 
		weights.kernel_height == 1 && weights.kernel_width == 1 && dilation_height == 1 && dilation_width == 1 && padding_top == 0 && padding_left == 0 && 
        biases != nullptr && in.ch == 8)
	{
		for (int out_ch_idx = 0; out_ch_idx < out.ch; out_ch_idx++) {
			int32_t bias_m = biases[out_ch_idx];
			int32_t bias_e = 1 << (((const fx_quant_specific_params *)(void*)&quant_params)->bias_shift);
			int32_t bias_s2 = 0;
			if (bias_e == (1 << 15)) {
				bias_e >>= 1;
				bias_s2 = 1;
			}
			int32_t out_shift = ((const fx_quant_specific_params *)(void*)&quant_params)->out_shift;
			const MLI_PTR(int16_t) w_ptr = (const MLI_PTR(int16_t))(void*)weights.ptr
								         + weights.out_ch_mem_stride * out_ch_idx;
			v4q15_t w0, w1;
			for (int i = 0; i < 4; i++) {
				w0[i] = w_ptr[(i + 0) * out.ch];
				w1[i] = w_ptr[(i + 4) * out.ch];
			}
			for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
				const int h_idx_in = (H_idx * stride_height);
				for (int W_idx = clmn_begin; W_idx < clmn_end - 1; W_idx += 2) {
					const int w_idx_in0 = ((W_idx + 0) * stride_width);
					const MLI_PTR(int16_t) in_ptr0 = (const MLI_PTR(int16_t))(void*)in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in0;

					// This impl is only for kernel size 1x1, in order to improve the performance
					const v4q15_t *__restrict _in_ptr0 = (const v4q15_t *__restrict)(void *)in_ptr0;
					mli_acc40_t accu0 = fx_a40_mpy_q15(fx_asl_q15(bias_m, bias_s2), bias_e);
					accu0 = fx_a40_qmac_v4q15(accu0, _in_ptr0[0], w0);
					accu0 = fx_a40_qmac_v4q15(accu0, _in_ptr0[1], w1);

					// Cast result to output type, apply built-in ReLU Applying and write result
					int32_t out_val0 = fx_q31_cast_nf_asr_rnd_a40(accu0, out_shift);

					out_val0 = MIN(out_val0, val_max_limit);
					out_val0 = MAX(out_val0, val_min_limit);

					MLI_CONV_OUT_PTR(int16_t)__restrict out_ptr0 = (MLI_CONV_OUT_PTR(int16_t)__restrict)(void*)out.ptr
							+ out.row_mem_stride * H_idx
							+ out.col_mem_stride * (W_idx + 0)
							+ out.ch_mem_stride * out_ch_idx;

					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

					const int w_idx_in1 = ((W_idx + 1) * stride_width);
					const MLI_PTR(int16_t) in_ptr1 = (const MLI_PTR(int16_t))(void*)in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in1;

					// This impl is only for kernel size 1x1, in order to improve the performance
					const v4q15_t *__restrict _in_ptr1 = (const v4q15_t *__restrict)(void *)in_ptr1;
					mli_acc40_t accu1 = fx_a40_mpy_q15(fx_asl_q15(bias_m, bias_s2), bias_e);
					accu1 = fx_a40_qmac_v4q15(accu1, _in_ptr1[0], w0);
					accu1 = fx_a40_qmac_v4q15(accu1, _in_ptr1[1], w1);

					// Cast result to output type, apply built-in ReLU Applying and write result
					int32_t out_val1 = fx_q31_cast_nf_asr_rnd_a40(accu1, out_shift);

					out_val1 = MIN(out_val1, val_max_limit);
					out_val1 = MAX(out_val1, val_min_limit);

					MLI_CONV_OUT_PTR(int16_t)__restrict out_ptr1 = (MLI_CONV_OUT_PTR(int16_t)__restrict)(void*)out.ptr
							+ out.row_mem_stride * H_idx
							+ out.col_mem_stride * (W_idx+1)
							+ out.ch_mem_stride * out_ch_idx;

					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

					*out_ptr0 = out_val0;
					*out_ptr1 = out_val1;
				}

				if ((clmn_end - clmn_begin) & 1) {
					int W_idx = clmn_end - 1;
					const int w_idx_in0 = ((W_idx + 0) * stride_width);
					const MLI_PTR(int16_t) in_ptr0 = (const MLI_PTR(int16_t))(void*)in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in0;

					// This impl is only for kernel size 1x1, in order to improve the performance
					const v4q15_t *__restrict _in_ptr0 = (const v4q15_t *__restrict)(void *)in_ptr0;
					mli_acc40_t accu0 = fx_a40_mpy_q15(fx_asl_q15(bias_m, bias_s2), bias_e);
					accu0 = fx_a40_qmac_v4q15(accu0, _in_ptr0[0], w0);
					accu0 = fx_a40_qmac_v4q15(accu0, _in_ptr0[1], w1);

					// Cast result to output type, apply built-in ReLU Applying and write result
					int32_t out_val0 = fx_q31_cast_nf_asr_rnd_a40(accu0, out_shift);

					out_val0 = MIN(out_val0, val_max_limit);
					out_val0 = MAX(out_val0, val_min_limit);

					MLI_CONV_OUT_PTR(int16_t)__restrict out_ptr0 = (MLI_CONV_OUT_PTR(int16_t)__restrict)(void*)out.ptr
							+ out.row_mem_stride * H_idx
							+ out.col_mem_stride * (W_idx + 0)
							+ out.ch_mem_stride * out_ch_idx;
					*out_ptr0 = out_val0;
				}
			}
		}
	}
	else if (std::is_same<io_T, int16_t>::value && std::is_same<w_T, int16_t>::value && std::is_same<b_T, int16_t>::value && std::is_same<acc_T, mli_acc40_t>::value && 
		     weights.kernel_height == 1 && weights.kernel_width == 1 && dilation_height == 1 && dilation_width == 1 && padding_top == 0 && padding_left == 0 && 
             biases != nullptr && (in.ch & 7) == 0 && in.ch <= 64)
	{
		for (int out_ch_idx = 0; out_ch_idx < out.ch; out_ch_idx++) {
			int32_t bias_m = biases[out_ch_idx];
			int32_t bias_e = 1 << (((const fx_quant_specific_params *)(void*)&quant_params)->bias_shift);
			int32_t bias_s2 = 0;
			if (bias_e == (1 << 15)) {
				bias_e >>= 1;
				bias_s2 = 1;
			}
			int32_t out_shift = ((const fx_quant_specific_params *)(void*)&quant_params)->out_shift;
			const MLI_PTR(int16_t) w_ptr = (const MLI_PTR(int16_t))(void*)weights.ptr
								         + weights.out_ch_mem_stride * out_ch_idx;
			v4q15_t w[16];
			for (int i = 0; i < in.ch; i += 4) {
				w[i >> 2][0] = w_ptr[(i + 0) * out.ch];
				w[i >> 2][1] = w_ptr[(i + 1) * out.ch];
				w[i >> 2][2] = w_ptr[(i + 2) * out.ch];
				w[i >> 2][3] = w_ptr[(i + 3) * out.ch];
			}
			for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
				const int h_idx_in = (H_idx * stride_height);
				for (int W_idx = clmn_begin; W_idx < clmn_end - 1; W_idx += 2) {
					const int w_idx_in0 = ((W_idx + 0) * stride_width);
					const MLI_PTR(int16_t) in_ptr0 = (const MLI_PTR(int16_t))(void*)in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in0;

					// This impl is only for kernel size 1x1, in order to improve the performance
					const v4q15_t *__restrict _in_ptr0 = (const v4q15_t *__restrict)(void *)in_ptr0;
					mli_acc40_t accu0 = fx_a40_mpy_q15(fx_asl_q15(bias_m, bias_s2), bias_e);

					for (int i = 0; i < (in.ch >> 2); i += 2) {
						accu0 = fx_a40_qmac_v4q15(accu0, _in_ptr0[i + 0], w[i + 0]);
						accu0 = fx_a40_qmac_v4q15(accu0, _in_ptr0[i + 1], w[i + 1]);
					}

					// Cast result to output type, apply built-in ReLU Applying and write result
					int32_t out_val0 = fx_q31_cast_nf_asr_rnd_a40(accu0, out_shift);

					out_val0 = MIN(out_val0, val_max_limit);
					out_val0 = MAX(out_val0, val_min_limit);

					MLI_CONV_OUT_PTR(int16_t)__restrict out_ptr0 = (MLI_CONV_OUT_PTR(int16_t)__restrict)(void*)out.ptr
							+ out.row_mem_stride * H_idx
							+ out.col_mem_stride * (W_idx + 0)
							+ out.ch_mem_stride * out_ch_idx;

					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

					const int w_idx_in1 = ((W_idx + 1) * stride_width);
					const MLI_PTR(int16_t) in_ptr1 = (const MLI_PTR(int16_t))(void*)in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in1;

					// This impl is only for kernel size 1x1, in order to improve the performance
					const v4q15_t *__restrict _in_ptr1 = (const v4q15_t *__restrict)(void *)in_ptr1;
					mli_acc40_t accu1 = fx_a40_mpy_q15(fx_asl_q15(bias_m, bias_s2), bias_e);
					for (int i = 0; i < (in.ch >> 2); i += 2) {
						accu1 = fx_a40_qmac_v4q15(accu1, _in_ptr1[i + 0], w[i + 0]);
						accu1 = fx_a40_qmac_v4q15(accu1, _in_ptr1[i + 1], w[i + 1]);
					}

					// Cast result to output type, apply built-in ReLU Applying and write result
					int32_t out_val1 = fx_q31_cast_nf_asr_rnd_a40(accu1, out_shift);

					out_val1 = MIN(out_val1, val_max_limit);
					out_val1 = MAX(out_val1, val_min_limit);

					MLI_CONV_OUT_PTR(int16_t)__restrict out_ptr1 = (MLI_CONV_OUT_PTR(int16_t)__restrict)(void*)out.ptr
							+ out.row_mem_stride * H_idx
							+ out.col_mem_stride * (W_idx + 1)
							+ out.ch_mem_stride * out_ch_idx;

					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

					*out_ptr0 = out_val0;
					*out_ptr1 = out_val1;
				}

				if ((clmn_end - clmn_begin) & 1){
					int W_idx = clmn_end - 1;
					const int w_idx_in0 = ((W_idx + 0) * stride_width);
					const MLI_PTR(int16_t) in_ptr0 = (const MLI_PTR(int16_t))(void*)in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in0;

					// This impl is only for kernel size 1x1, in order to improve the performance
					const v4q15_t *__restrict _in_ptr0 = (const v4q15_t *__restrict)(void *)in_ptr0;
					mli_acc40_t accu0 = fx_a40_mpy_q15(fx_asl_q15(bias_m, bias_s2), bias_e);

					for (int i = 0; i < (in.ch >> 2); i += 2) {
						accu0 = fx_a40_qmac_v4q15(accu0, _in_ptr0[i + 0], w[i + 0]);
						accu0 = fx_a40_qmac_v4q15(accu0, _in_ptr0[i + 1], w[i + 1]);
					}

					// Cast result to output type, apply built-in ReLU Applying and write result
					int32_t out_val0 = fx_q31_cast_nf_asr_rnd_a40(accu0, out_shift);

					out_val0 = MIN(out_val0, val_max_limit);
					out_val0 = MAX(out_val0, val_min_limit);

					MLI_CONV_OUT_PTR(int16_t)__restrict out_ptr0 = (MLI_CONV_OUT_PTR(int16_t)__restrict)(void*)out.ptr
							+ out.row_mem_stride * H_idx
							+ out.col_mem_stride * (W_idx + 0)
							+ out.ch_mem_stride * out_ch_idx;
					*out_ptr0 = out_val0;
				}
			}
		}
	}
	else
#endif // #ifdef __Xdsp_wide
	{
        mli::krn::ref::convolution2D<io_T, w_T, b_T, acc_T, quant_T, fix_kernel_width, fix_kernel_height>(
                in, weights, biases, out, perception_area, quant_params,
                val_min_limit, val_max_limit,
                stride_height, stride_width, 
                dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);
	}
}

//========================================================
// Depthwise convolution 2D template
//========================================================
template <typename io_T, typename w_T, typename b_T, typename acc_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void depthwise_convolution2D_hwcn_nopad(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &w,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {
    const int row_begin = perception_area.row_beg;
    const int row_end = perception_area.row_end;
    const int clmn_begin = perception_area.clmn_beg;
    const int clmn_end = perception_area.clmn_end;
    const int filters = 1;
    const int ch_mul = out.ch / in.ch;
    const int amount_rows = row_end - row_begin;
    const int amount_columns = clmn_end - clmn_begin;
    const int in_compensation_row_loop   = in.row_mem_stride * stride_height * amount_rows;
    const int out_compensation_row_loop  = out.row_mem_stride * amount_rows;
    const int in_compensation_clmn_loop  = in.col_mem_stride * stride_width * amount_columns;
    const int out_compensation_clmn_loop = out.col_mem_stride * amount_columns;
    const int in_increment_clmn_loop     = in.col_mem_stride * stride_width;
    const int in_increment_row_loop      = in.row_mem_stride * stride_height - in_compensation_clmn_loop;
    const int out_increment_row_loop     = out.row_mem_stride  - out_compensation_clmn_loop;
    const int channel_per_loop = 1;
    const int channels_per_loop_v = 2;
    const int out_increment_in_ch_loop = channel_per_loop - out_compensation_row_loop;
    const int out_increment_in_ch_loop_v = channels_per_loop_v - out_compensation_row_loop;

    const MLI_PTR(w_T) __restrict w_ptr = w.ptr;
    int out_ch_idx = 0;

    const MLI_PTR(io_T) __restrict in_ptr = in.ptr
            + in.col_mem_stride * (clmn_begin * stride_width - padding_left) // move to column
            + in.row_mem_stride * (row_begin * stride_height - padding_top); // move to row

    MLI_CONV_OUT_PTR(io_T) __restrict out_ptr = out.ptr
            + out.col_mem_stride * clmn_begin // move to column
            + out.row_mem_stride * row_begin; // move to row

    s8asym_quant_specific_params v2quant_params[] = {quant_params, quant_params};
    for (int in_ch_idx = 0; in_ch_idx < in.ch-1; in_ch_idx += 2) {
        for (int ch_mult_idx = 0; ch_mult_idx < ch_mul; ch_mult_idx++) {
            mli::krn::adjust_quant_params(&v2quant_params[0], out_ch_idx++);
            mli::krn::adjust_quant_params(&v2quant_params[1], out_ch_idx++);

            acc_T bias_add_ch1 = bias_additive(biases++, 0x0, &v2quant_params[0]);
            acc_T bias_add_ch2 = bias_additive(biases++, 0x0, &v2quant_params[1]);

            __v2i32_t v2acc_weights_add = {bias_add_ch1, bias_add_ch2};
            v2acc_weights_add = weights_additive_v(w_ptr, &v2acc_weights_add, &quant_params, w.kernel_width, w.kernel_height,
                    w.col_mem_stride, w.row_mem_stride);
            __builtin_assume(amount_rows > 0);
            for (int H_idx = 0; H_idx < amount_rows; H_idx++) {
                __builtin_assume(amount_columns > 0);

                __v2i32_t v2accu_dotprod = v2acc_weights_add;
                for (int W_idx = 0; W_idx < amount_columns; W_idx++) {
                    dotprod2D_hwc_v(&in_ptr, &w_ptr, &v2accu_dotprod, w.kernel_width, w.kernel_height,
                                    in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                                    w.col_mem_stride, w.row_mem_stride);
                    //compensite increment of input tensor pointer from dotprod2D_hwc_v function
                    in_ptr += in_increment_clmn_loop - w.kernel_height * in.row_mem_stride * dilation_height;
                    //compensite increment of weights pointer from dotprod2D_hwc_v function
                    w_ptr -= w.kernel_height * w.row_mem_stride;

                    // Cast result to output type
                    result_cast_relu_store_v(out_ptr, &v2accu_dotprod, v2quant_params, val_min_limit, val_max_limit);
                    out_ptr += out.col_mem_stride;

                    v2accu_dotprod = v2acc_weights_add;
                } // for W_idx
                in_ptr += in_increment_row_loop;
                out_ptr += out_increment_row_loop;
            } // for H_idx
            in_ptr -= in_compensation_row_loop;
            out_ptr += out_increment_in_ch_loop_v;
            // out_ch_idx++;
            w_ptr += 2;
            // biases++;
        } // for ch_mult_idx
        in_ptr += 2 * filters;
    } // for in_ch_idx

    if (in.ch & 1) {
        for (int ch_mult_idx = 0; ch_mult_idx < ch_mul; ch_mult_idx++) {
            mli::krn::adjust_quant_params(&quant_params, out_ch_idx);

            acc_T global_other_additives = weights_additive(w_ptr, 0x0, &quant_params, w.kernel_width, w.kernel_height,
                    w.col_mem_stride, w.row_mem_stride);
            global_other_additives += bias_additive(biases, 0x0, &quant_params);
            __v2i32_t v2global_other_additives = {global_other_additives, global_other_additives};

            for (int H_idx = 0; H_idx < amount_rows; H_idx++) {
                for (int W_idx = 0; W_idx < amount_columns - 1; W_idx += 2) {
                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    __v2i32_t accu = v2global_other_additives;
                    accu = dotprod2D_inp_width_v(&in_ptr, &w_ptr, &accu, w.kernel_width, w.kernel_height,
                            in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                            w.col_mem_stride, w.row_mem_stride, in_increment_clmn_loop);
                    //compensite increment of input tensor pointer from dotprod2D_hwc_v function
                    in_ptr += 2 * in_increment_clmn_loop - w.kernel_height * in.row_mem_stride * dilation_height;
                    //compensite increment of weights pointer from dotprod2D_hwc_v function
                    w_ptr -= w.kernel_height * w.row_mem_stride;
                    // Cast result to output type
                    result_cast_relu_store_inp_width_v(out_ptr, &accu, &quant_params, val_min_limit, val_max_limit, out.col_mem_stride);
                    out_ptr += 2 * out.col_mem_stride;
                } // for W_idx
                if (amount_columns & 0x1) {
                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    acc_T accu = 0;
                    accu = dotprod2D(&in_ptr, &w_ptr, accu, w.kernel_width, w.kernel_height,
                            in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height, w.col_mem_stride, w.row_mem_stride);
                    //compensite increment of input tensor pointer from dotprod2D_hwc_v function
                    in_ptr += in_increment_clmn_loop - w.kernel_height * in.row_mem_stride * dilation_height;
                    //compensite increment of weights pointer from dotprod2D_hwc_v function
                    w_ptr -= w.kernel_height * w.row_mem_stride;
                    accu += global_other_additives;

                    // Cast result to output type
                    result_cast_relu_store(out_ptr, accu, &quant_params, val_min_limit, val_max_limit);
                    out_ptr += out.col_mem_stride;
                }
                in_ptr += in_increment_row_loop;
                out_ptr += out_increment_row_loop;
            } // for H_idx
            in_ptr -= in_compensation_row_loop;
            out_ptr += out_increment_in_ch_loop;
            out_ch_idx++;
            w_ptr++;
            biases++;
        } // for in_ch_idx
        in_ptr += filters;
    } // for ch_mult_idx
}

template <typename io_T, typename w_T, typename b_T, typename acc_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void depthwise_convolution2D_hwcn(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &w,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        s8asym_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {

    const int row_begin = perception_area.row_beg;
    const int row_end = perception_area.row_end;
    const int clmn_begin = perception_area.clmn_beg;
    const int clmn_end = perception_area.clmn_end;
    const int ch_mul = out.ch / in.ch;
    const int amount_rows = row_end - row_begin;
    const int amount_columns = clmn_end - clmn_begin;
    const int out_compensation_row_loop  = out.row_mem_stride * amount_rows;
    const int out_compensation_clmn_loop = out.col_mem_stride * amount_columns;
    const int out_increment_row_loop     = out.row_mem_stride  - out_compensation_clmn_loop;
    const int channel_per_loop = 1;
    const int channels_per_loop_v = 2;
    const int out_increment_in_ch_loop = channel_per_loop - out_compensation_row_loop;
    const int out_increment_in_ch_loop_v = channels_per_loop_v - out_compensation_row_loop;
    int out_ch_idx = 0;

    MLI_CONV_OUT_PTR(io_T) __restrict out_ptr = out.ptr
            + out.col_mem_stride * clmn_begin // move to columm
            + out.row_mem_stride * row_begin; // move to row

    s8asym_quant_specific_params v2quant_params[] = {quant_params, quant_params};
    for (int in_ch_idx = 0; in_ch_idx < in.ch - 1; in_ch_idx += 2) {
        for (int ch_mult_idx = 0; ch_mult_idx < ch_mul; ch_mult_idx++) {
            mli::krn::adjust_quant_params(&v2quant_params[0], out_ch_idx);
            mli::krn::adjust_quant_params(&v2quant_params[1], out_ch_idx + 1);

            acc_T bias_add_ch1 = bias_additive(biases++, 0x0, &v2quant_params[0]);
            acc_T bias_add_ch2 = bias_additive(biases++, 0x0, &v2quant_params[1]);
            __v2i32_t v2_bias_add = {bias_add_ch1, bias_add_ch2};

            for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    // comp - compensation values for valid area definition
                    const mli_compensations comp = mli_prv_valid_area_compensations(
                        H_idx, W_idx, in.height, in.width,
                        w.kernel_height, w.kernel_width,
                        stride_height, stride_width, padding_left, padding_top,
                        dilation_height, dilation_width);
                    
                    const int rows = w.kernel_height - comp.kernel_top - comp.kernel_bottom;
                    const int clmns = w.kernel_width - comp.kernel_right - comp.kernel_left;
                    const int h_idx_in = (H_idx * stride_height - padding_top + comp.in_top);
                    const int w_idx_in = (W_idx * stride_width - padding_left + comp.in_left);

                    const MLI_PTR(io_T) __restrict in_ptr = in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in
                        + in.ch_mem_stride * in_ch_idx;

                    const MLI_PTR(w_T) __restrict w_ptr = w.ptr
                        + w.row_mem_stride * comp.kernel_top
                        + w.col_mem_stride * comp.kernel_left
                        + w.in_ch_mem_stride * 0
                        + w.out_ch_mem_stride * out_ch_idx;

                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    __v2i32_t v2accu_dotprod = {0, 0};
                    dotprod2D_hwc_v(in_ptr, w_ptr, &v2accu_dotprod, clmns, rows,
                            in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                            w.col_mem_stride, w.row_mem_stride);

                    __v2i32_t v2acc_weights_add = {0, 0};
                    v2acc_weights_add = weights_additive_v(w_ptr, &v2acc_weights_add, &quant_params, clmns, rows,
                            w.col_mem_stride, w.row_mem_stride);

                    v2accu_dotprod += v2acc_weights_add;
                    v2accu_dotprod += v2_bias_add;

                    // Cast result to output type
                    result_cast_relu_store_v(out_ptr, &v2accu_dotprod, v2quant_params, val_min_limit, val_max_limit);

                    out_ptr += out.col_mem_stride;
                } // for W_idx
                out_ptr += out.row_mem_stride - out_compensation_clmn_loop;
            } // for H_idx
            out_ptr += out_increment_in_ch_loop_v;
            out_ch_idx += 2;
        } // for ch_mult_idx
    } // for in_ch_idx

    if (in.ch & 1) {
        for (int ch_mult_idx = 0; ch_mult_idx < ch_mul; ch_mult_idx++) {

            mli::krn::adjust_quant_params(&quant_params, out_ch_idx);
            int32_t bias_add = bias_additive(biases++, 0x0, &quant_params);

            for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
                for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                    // Define area of input and filter for convolution
                    // comp - compensation values for valid area definition
                    const mli_compensations comp = mli_prv_valid_area_compensations(
                        H_idx, W_idx, in.height, in.width,
                        w.kernel_height, w.kernel_width,
                        stride_height, stride_width, padding_left, padding_top,
                        dilation_height, dilation_width);
                    
                    const int rows = w.kernel_height - comp.kernel_top - comp.kernel_bottom;
                    const int clmns = w.kernel_width - comp.kernel_right - comp.kernel_left;
                    const int h_idx_in = (H_idx * stride_height - padding_top + comp.in_top);
                    const int w_idx_in = (W_idx * stride_width - padding_left + comp.in_left);
                    const MLI_PTR(io_T) __restrict in_ptr = in.ptr
                        + in.row_mem_stride * h_idx_in
                        + in.col_mem_stride * w_idx_in
                        + in.ch_mem_stride * (in.ch - 1);

                    const MLI_PTR(w_T) __restrict w_ptr = w.ptr
                        + w.row_mem_stride * comp.kernel_top
                        + w.col_mem_stride * comp.kernel_left
                        + w.in_ch_mem_stride * 0
                        + w.out_ch_mem_stride * out_ch_idx;

                    // Convolution core. Here calculations performes in a unfolded expression way: 
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    acc_T accu = mli_math_mul_fx<io_T, acc_T>(0, 0);
                    accu = dotprod2D(in_ptr, w_ptr, accu, clmns, rows,
                            in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                            w.col_mem_stride, w.row_mem_stride);

                    int32_t w_adds = weights_additive(w_ptr, 0x0, &quant_params, clmns, rows,
                            w.col_mem_stride, w.row_mem_stride);

                    accu += w_adds;
                    accu += bias_add;

                    // Cast result to output type
                    io_T out_val = mli::krn::result_cast<io_T, acc_T, s8asym_quant_specific_params>(accu, &quant_params);
                    out_val = MIN(out_val, val_max_limit);
                    out_val = MAX(out_val, val_min_limit);
                    *out_ptr = out_val;

                    out_ptr += out.col_mem_stride;
                } // for W_idx
                out_ptr += out_increment_row_loop;
            } // for H_idx
            out_ptr += out_increment_in_ch_loop;
            out_ch_idx++;
        } // for ch_mult_idx
    }
}

template <typename io_T, typename w_T, typename b_T, typename acc_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void depthwise_convolution2D_hwcn_nopad(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &weights,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        fx_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {
#ifdef __Xdsp_wide
    if(std::is_same<io_T, int16_t>::value && std::is_same<w_T, int16_t>::value && std::is_same<b_T, int16_t>::value && std::is_same<acc_T, mli_acc40_t>::value && 
       fix_kernel_width == 3 && fix_kernel_height == 3) {
        // This optimization is only for dwconv_k3x3_fx16_fx16_fx16 on ARCv2HS 
        //================================================================================================
        const int row_begin = perception_area.row_beg;
        const int row_end = perception_area.row_end;
        const int clmn_begin = perception_area.clmn_beg;
        const int clmn_end = perception_area.clmn_end;

        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                // Define area of input and filter for convolution
                // comp - compensation values for valid area definition
                const mli_compensations comp = mli_prv_valid_area_compensations(
                        H_idx, W_idx, in.height, in.width,
                        weights.kernel_height, weights.kernel_width,
                        stride_height, stride_width, padding_left, padding_top,
                        dilation_height, dilation_width);

                const int rows = weights.kernel_height - comp.kernel_top - comp.kernel_bottom;
                const int clmns = weights.kernel_width - comp.kernel_right - comp.kernel_left;
                const int h_idx_in = (H_idx * stride_height - padding_top + comp.in_top);
                const int w_idx_in = (W_idx * stride_width - padding_left + comp.in_left);

                v2q15_t vval_max_limit = fx_replic_v2q15(val_max_limit);
                v2q15_t vval_min_limit = fx_replic_v2q15(val_min_limit);

                int remaining = (in.ch & 1);
                for (int in_ch_idx = 0; in_ch_idx < (in.ch - remaining); in_ch_idx += 2) {
                    const MLI_PTR(int16_t)__restrict in_ptr = (const MLI_PTR(int16_t)__restrict)(void*)in.ptr
                            + in.row_mem_stride * h_idx_in
                            + in.col_mem_stride * w_idx_in
                            + in.ch_mem_stride * in_ch_idx;
                    const int out_ch_idx = in_ch_idx;
                    const MLI_PTR(int16_t)__restrict w_ptr = (const MLI_PTR(int16_t)__restrict)(void*)weights.ptr
                            + weights.row_mem_stride * comp.kernel_top
                            + weights.col_mem_stride * comp.kernel_left
                            + weights.in_ch_mem_stride * 0
                            + weights.out_ch_mem_stride * out_ch_idx;
                    const MLI_PTR(int16_t)__restrict b_ptr = (const MLI_PTR(int16_t)__restrict)(void *)biases + out_ch_idx;

                    mli_acc40_t acc1 = mli_math_cast_fx<int16_t, mli_acc40_t>(*(b_ptr), -quant_params.bias_shift);
                    mli_acc40_t acc2 = mli_math_cast_fx<int16_t, mli_acc40_t>(*(b_ptr+1), -quant_params.bias_shift);
                    v2accum40_t accu = fx_create_v2a40(acc1, acc2);

                    accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride * dilation_height) * 0 + (in.col_mem_stride * dilation_width) * 0], 
                                                    *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * 0 + weights.col_mem_stride * 0]);
                    accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride * dilation_height) * 0 + (in.col_mem_stride * dilation_width) * 1], 
                                                    *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * 0 + weights.col_mem_stride * 1]);
                    accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride * dilation_height) * 0 + (in.col_mem_stride * dilation_width) * 2], 
                                                    *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * 0 + weights.col_mem_stride * 2]);
                    accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride * dilation_height) * 1 + (in.col_mem_stride * dilation_width) * 0], 
                                                    *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * 1 + weights.col_mem_stride * 0]);
                    accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride * dilation_height) * 1 + (in.col_mem_stride * dilation_width) * 1], 
                                                    *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * 1 + weights.col_mem_stride * 1]);
                    accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride * dilation_height) * 1 + (in.col_mem_stride * dilation_width) * 2], 
                                                    *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * 1 + weights.col_mem_stride * 2]);
                    accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride * dilation_height) * 2 + (in.col_mem_stride * dilation_width) * 0], 
                                                    *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * 2 + weights.col_mem_stride * 0]);
                    accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride * dilation_height) * 2 + (in.col_mem_stride * dilation_width) * 1], 
                                                    *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * 2 + weights.col_mem_stride * 1]);
                    accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride * dilation_height) * 2 + (in.col_mem_stride * dilation_width) * 2], 
                                                    *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * 2 + weights.col_mem_stride * 2]);

                    v2q15_t out_val;
                    // Cast result to output type, apply built-in ReLU Applying and write result
                    out_val = fx_v2q15_cast_nf_asr_rnd_v2a40(accu, ((const fx_quant_specific_params *)(void*)&quant_params)->out_shift - 16);
                    out_val = fx_min_v2q15(out_val, vval_max_limit);
                    out_val = fx_max_v2q15(out_val, vval_min_limit);
                
                    v2q15_t *__restrict out_ptr = (v2q15_t *__restrict)(void*)(out.ptr
                            + out.row_mem_stride * H_idx
                            + out.col_mem_stride * W_idx
                            + out.ch_mem_stride * out_ch_idx);
                    *out_ptr = out_val;
                } // for in_ch_idx

                if (remaining) {
                    const MLI_PTR(int16_t)__restrict in_ptr = (const MLI_PTR(int16_t)__restrict)(void*)in.ptr
                            + in.row_mem_stride * h_idx_in
                            + in.col_mem_stride * w_idx_in
                            + in.ch_mem_stride * (in.ch - 1);

                    const MLI_PTR(int16_t)__restrict w_ptr = (const MLI_PTR(int16_t)__restrict)(void*)weights.ptr
                            + weights.row_mem_stride * comp.kernel_top
                            + weights.col_mem_stride * comp.kernel_left
                            + weights.in_ch_mem_stride * 0
                            + weights.out_ch_mem_stride * (in.ch - 1);

                    const MLI_PTR(int16_t)__restrict b_ptr = (const MLI_PTR(int16_t)__restrict)(void *)biases + (in.ch - 1);
                    mli::krn::adjust_quant_params(&quant_params, (in.ch - 1));

                    // Convolution core. Here calculations performes in a unfolded expression way:
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    mli_acc40_t accu = mli_math_mul_fx<int16_t, mli_acc40_t>(0, 0);
                    accu = dotprod2D(in_ptr, w_ptr, accu, clmns, rows,
                                    in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                                    weights.col_mem_stride,
                                    weights.row_mem_stride);
                    accu = weights_additive(w_ptr, accu, &quant_params, clmns, rows,
                                            weights.col_mem_stride,
                                            weights.row_mem_stride);
                    accu = bias_additive(b_ptr, accu, &quant_params);

                    // Cast result to output type, apply built-in ReLU Applying and write result
                    int16_t out_val = mli::krn::result_cast<int16_t, mli_acc40_t, fx_quant_specific_params>(accu, &quant_params);
                    out_val = MIN(out_val, val_max_limit);
                    out_val = MAX(out_val, val_min_limit);

                    MLI_CONV_OUT_PTR(int16_t) out_ptr = out.ptr
                                                      + out.row_mem_stride * H_idx
                                                      + out.col_mem_stride * W_idx
                                                      + out.ch_mem_stride * (in.ch - 1);
                    *out_ptr = out_val;
                }
            } // for W_idx
        } // for H_idx
    } else {
        mli::krn::ref::depthwise_convolution2D<io_T, w_T, b_T, acc_T, fx_quant_specific_params, fix_kernel_width, fix_kernel_height>(
                    in, weights, biases, out, perception_area, quant_params,
                    val_min_limit, val_max_limit,
                    stride_height, stride_width,
                    dilation_height, dilation_width,
                    padding_top, padding_left,
                    padding_bot, padding_right);        
    }
#else
    mli::krn::ref::depthwise_convolution2D<io_T, w_T, b_T, acc_T, fx_quant_specific_params, fix_kernel_width, fix_kernel_height>(
                in, weights, biases, out, perception_area, quant_params,
                val_min_limit, val_max_limit,
                stride_height, stride_width,
                dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);
#endif
}

template <typename io_T, typename w_T, typename b_T, typename acc_T, int fix_kernel_width, int fix_kernel_height>
MLI_FORCE_INLINE void depthwise_convolution2D_hwcn(
        const tensor_private_t<MLI_PTR(io_T)> &in,
        const conv2d_weights_tensor_private_t<MLI_PTR(w_T)> &weights,
        const MLI_PTR(b_T)  __restrict biases,
        const tensor_private_t<MLI_CONV_OUT_PTR(io_T)> &out,
        const rect_t &perception_area,
        fx_quant_specific_params quant_params,
        const io_T val_min_limit,
        const io_T val_max_limit,
        const int stride_height, const int stride_width,
        const int dilation_height, const int dilation_width,
        const int padding_top, const int padding_left,
        const int padding_bot, const int padding_right) {
#ifdef __Xdsp_wide
    if(std::is_same<io_T, int16_t>::value && std::is_same<w_T, int16_t>::value && std::is_same<b_T, int16_t>::value && std::is_same<acc_T, mli_acc40_t>::value && 
       fix_kernel_width == 3 && fix_kernel_height == 3) {
        // This optimization is only for dwconv_k3x3_fx16_fx16_fx16 with padding on ARCv2HS 
        //================================================================================================
        const int row_begin = perception_area.row_beg;
        const int row_end = perception_area.row_end;
        const int clmn_begin = perception_area.clmn_beg;
        const int clmn_end = perception_area.clmn_end;

        for (int H_idx = row_begin; H_idx < row_end; H_idx++) {
            for (int W_idx = clmn_begin; W_idx < clmn_end; W_idx++) {
                // Define area of input and filter for convolution
                // comp - compensation values for valid area definition
                const mli_compensations comp = mli_prv_valid_area_compensations(
                        H_idx, W_idx, in.height, in.width,
                        weights.kernel_height, weights.kernel_width,
                        stride_height, stride_width, padding_left, padding_top,
                        dilation_height, dilation_width);

                const int rows = weights.kernel_height - comp.kernel_top - comp.kernel_bottom;
                const int clmns = weights.kernel_width - comp.kernel_right - comp.kernel_left;
                const int h_idx_in = (H_idx * stride_height - padding_top + comp.in_top);
                const int w_idx_in = (W_idx * stride_width - padding_left + comp.in_left);

                v2q15_t vval_max_limit = fx_replic_v2q15(val_max_limit);
                v2q15_t vval_min_limit = fx_replic_v2q15(val_min_limit);

                int remaining = (in.ch & 1);

			    if (clmns == 3 && rows == 2) {
                    for (int in_ch_idx = 0; in_ch_idx < (in.ch - remaining); in_ch_idx += 2) {
                        const MLI_PTR(int16_t)__restrict in_ptr = (const MLI_PTR(int16_t)__restrict)(void*)in.ptr
                                                                + in.row_mem_stride * h_idx_in
                                                                + in.col_mem_stride * w_idx_in
                                                                + in.ch_mem_stride * in_ch_idx;

                        const int out_ch_idx = in_ch_idx;
                        const MLI_PTR(int16_t)__restrict w_ptr = (const MLI_PTR(int16_t)__restrict)(void*)weights.ptr
                                                               + weights.row_mem_stride * comp.kernel_top
                                                               + weights.col_mem_stride * comp.kernel_left
                                                               + weights.in_ch_mem_stride * 0
                                                               + weights.out_ch_mem_stride * out_ch_idx;
                        const MLI_PTR(int16_t)__restrict b_ptr = (const MLI_PTR(int16_t)__restrict)(void *)biases + out_ch_idx;

                        mli_acc40_t acc1 = mli_math_cast_fx<int16_t, mli_acc40_t>(*(b_ptr), -quant_params.bias_shift);
                        mli_acc40_t acc2 = mli_math_cast_fx<int16_t, mli_acc40_t>(*(b_ptr+1), -quant_params.bias_shift);
                        v2accum40_t accu = fx_create_v2a40(acc1, acc2);

                        accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride * dilation_height) * 0 + (in.col_mem_stride * dilation_width) * 0], 
                                                        *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * 0 + weights.col_mem_stride * 0]);
                        accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride * dilation_height) * 0 + (in.col_mem_stride * dilation_width) * 1], 
                                                        *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * 0 + weights.col_mem_stride * 1]);
                        accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride * dilation_height) * 0 + (in.col_mem_stride * dilation_width) * 2], 
                                                        *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * 0 + weights.col_mem_stride * 2]);
                        accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride * dilation_height) * 1 + (in.col_mem_stride * dilation_width) * 0], 
                                                        *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * 1 + weights.col_mem_stride * 0]);
                        accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride * dilation_height) * 1 + (in.col_mem_stride * dilation_width) * 1], 
                                                        *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * 1 + weights.col_mem_stride * 1]);
                        accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride * dilation_height) * 1 + (in.col_mem_stride * dilation_width) * 2], 
                                                        *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * 1 + weights.col_mem_stride * 2]);

                        v2q15_t out_val;

                        // Cast result to output type, apply built-in ReLU Applying and write result
                        out_val = fx_v2q15_cast_nf_asr_rnd_v2a40(accu, ((const fx_quant_specific_params *)(void*)&quant_params)->out_shift - 16);
                        out_val = fx_min_v2q15(out_val, vval_max_limit);
                        out_val = fx_max_v2q15(out_val, vval_min_limit);
                    

                        v2q15_t *__restrict out_ptr = (v2q15_t *__restrict)(void*)(out.ptr
                                                    + out.row_mem_stride * H_idx
                                                    + out.col_mem_stride * W_idx
                                                    + out.ch_mem_stride * out_ch_idx);
                        *out_ptr = out_val;
                    } // for in_ch_idx
                } else {
                    for (int in_ch_idx = 0; in_ch_idx < (in.ch - remaining); in_ch_idx += 2) {
                        const MLI_PTR(int16_t)__restrict in_ptr = (const MLI_PTR(int16_t)__restrict)(void*)in.ptr
                                                                + in.row_mem_stride * h_idx_in
                                                                + in.col_mem_stride * w_idx_in
                                                                + in.ch_mem_stride * in_ch_idx;

                        const int out_ch_idx = in_ch_idx;
                        const MLI_PTR(int16_t)__restrict w_ptr = (const MLI_PTR(int16_t)__restrict)(void*)weights.ptr
                                                               + weights.row_mem_stride * comp.kernel_top
                                                               + weights.col_mem_stride * comp.kernel_left
                                                               + weights.in_ch_mem_stride * 0
                                                               + weights.out_ch_mem_stride * out_ch_idx;
                        const MLI_PTR(int16_t)__restrict b_ptr = (const MLI_PTR(int16_t)__restrict)(void *)biases + out_ch_idx;

                        mli_acc40_t acc1 = mli_math_cast_fx<int16_t, mli_acc40_t>(*(b_ptr), -quant_params.bias_shift);
                        mli_acc40_t acc2 = mli_math_cast_fx<int16_t, mli_acc40_t>(*(b_ptr+1), -quant_params.bias_shift);
                        v2accum40_t accu = fx_create_v2a40(acc1, acc2);
                        for(int r = 0; r < rows; r++) {
                            for(int c = 0; c < clmns; c++) {
                                accu = fx_v2a40_mac_v2q15(accu, *(const v2q15_t *__restrict)(void*)&in_ptr[(in.row_mem_stride*dilation_height) * r + (in.col_mem_stride * dilation_width) * c], 
                                                                *(const v2q15_t *__restrict)(void*)&w_ptr[weights.row_mem_stride * r + weights.col_mem_stride * c]);
                            }
                        }

                        v2q15_t out_val;
                        // Cast result to output type, apply built-in ReLU Applying and write result
                        out_val = fx_v2q15_cast_nf_asr_rnd_v2a40(accu, ((const fx_quant_specific_params *)(void*)&quant_params)->out_shift - 16);
                        out_val = fx_min_v2q15(out_val, vval_max_limit);
                        out_val = fx_max_v2q15(out_val, vval_min_limit);
                    

                        v2q15_t *__restrict out_ptr = (v2q15_t *__restrict)(void*)(out.ptr
                                                    + out.row_mem_stride * H_idx
                                                    + out.col_mem_stride * W_idx
                                                    + out.ch_mem_stride * out_ch_idx);
                        *out_ptr = out_val;
                    } // for in_ch_idx
                }

                if (remaining) {
                    const MLI_PTR(int16_t)__restrict in_ptr = (const MLI_PTR(int16_t)__restrict)(void*)in.ptr
                            + in.row_mem_stride * h_idx_in
                            + in.col_mem_stride * w_idx_in
                            + in.ch_mem_stride * (in.ch - 1);

                    const MLI_PTR(int16_t)__restrict w_ptr = (const MLI_PTR(int16_t)__restrict)(void*)weights.ptr
                            + weights.row_mem_stride * comp.kernel_top
                            + weights.col_mem_stride * comp.kernel_left
                            + weights.in_ch_mem_stride * 0
                            + weights.out_ch_mem_stride * (in.ch - 1);

                    const MLI_PTR(int16_t)__restrict b_ptr = (const MLI_PTR(int16_t)__restrict)(void *)biases + (in.ch - 1);
                    mli::krn::adjust_quant_params(&quant_params, (in.ch - 1));

                    // Convolution core. Here calculations performes in a unfolded expression way:
                    // out_val = (x-x_zp)*(w) + b) = -sum_i(w*x_zp) + sum(x*w) + b
                    //============================================
                    mli_acc40_t accu = mli_math_mul_fx<int16_t, mli_acc40_t>(0, 0);
                    accu = dotprod2D(in_ptr, w_ptr, accu, clmns, rows,
                                    in.col_mem_stride * dilation_width, in.row_mem_stride * dilation_height,
                                    weights.col_mem_stride,
                                    weights.row_mem_stride);
                    accu = weights_additive(w_ptr, accu, &quant_params, clmns, rows,
                                            weights.col_mem_stride,
                                            weights.row_mem_stride);
                    accu = bias_additive(b_ptr, accu, &quant_params);

                    // Cast result to output type, apply built-in ReLU Applying and write result
                    int16_t out_val = mli::krn::result_cast<int16_t, mli_acc40_t, fx_quant_specific_params>(accu, &quant_params);
                    out_val = MIN(out_val, val_max_limit);
                    out_val = MAX(out_val, val_min_limit);

                    MLI_CONV_OUT_PTR(int16_t) out_ptr = out.ptr
                                                      + out.row_mem_stride * H_idx
                                                      + out.col_mem_stride * W_idx
                                                      + out.ch_mem_stride * (in.ch - 1);
                    *out_ptr = out_val;
                }
            } // for W_idx
        } // for H_idx
    } else {
        mli::krn::ref::depthwise_convolution2D<io_T, w_T, b_T, acc_T, fx_quant_specific_params, fix_kernel_width, fix_kernel_height>(
                    in, weights, biases, out, perception_area, quant_params,
                    val_min_limit, val_max_limit,
                    stride_height, stride_width,
                    dilation_height, dilation_width,
                    padding_top, padding_left,
                    padding_bot, padding_right);
    }
#else 
    mli::krn::ref::depthwise_convolution2D<io_T, w_T, b_T, acc_T, fx_quant_specific_params, fix_kernel_width, fix_kernel_height>(
                in, weights, biases, out, perception_area, quant_params,
                val_min_limit, val_max_limit,
                stride_height, stride_width,
                dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);
#endif
}

template <typename io_T, typename w_T, typename b_T, typename acc_T, typename quant_T, int fix_kernel_width, int fix_kernel_height>
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
        const int padding_bot, const int padding_right) {

    // Phase 1: Process central part (without border effects - padding free)
    //=======================================================================
    rect_t perception_area_nopad;
    perception_area_nopad.row_beg = CEIL_DIV(padding_top, stride_height);
    perception_area_nopad.row_end = out.height - CEIL_DIV(padding_bot, stride_height);
    perception_area_nopad.clmn_beg = CEIL_DIV(padding_left, stride_width);
    perception_area_nopad.clmn_end = out.width - CEIL_DIV(padding_right, stride_width);
    
    if ((perception_area_nopad.row_end > perception_area_nopad.row_beg)
        && (perception_area_nopad.clmn_end > perception_area_nopad.clmn_beg)){
    depthwise_convolution2D_hwcn_nopad<io_T, w_T, b_T, acc_T, fix_kernel_width, fix_kernel_height>(
                in, w, biases, out, perception_area_nopad, quant_params,
                val_min_limit, val_max_limit,
                stride_height, stride_width,
                dilation_height, dilation_width,
                padding_top, padding_left,
                padding_bot, padding_right);
    }
    // Phase 2: Process border part with more complex algorithm
    // (usually significantly smaller part of computations)
    //=======================================================================
    if (padding_top || padding_left || padding_bot || padding_right) {
        rect_t perc_areas[4];
        int areas_num = 0;
        if (padding_top) {
            perc_areas[areas_num].row_beg = 0;
            perc_areas[areas_num].row_end = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = out.width;
        }
        if (padding_bot) {
            perc_areas[areas_num].row_beg = out.height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].row_end = out.height;
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = out.width;
        }
        if (padding_left) {
            perc_areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].row_end = out.height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = 0;
            perc_areas[areas_num++].clmn_end = CEIL_DIV(padding_left, stride_width);
        }
        if (padding_right) {
            perc_areas[areas_num].row_beg = CEIL_DIV(padding_top, stride_height);
            perc_areas[areas_num].row_end = out.height - CEIL_DIV(padding_bot, stride_height);
            perc_areas[areas_num].clmn_beg = out.width - CEIL_DIV(padding_right, stride_width);
            perc_areas[areas_num++].clmn_end = out.width;
        }
        for(int i = 0; i < areas_num; i ++) {
            depthwise_convolution2D_hwcn<io_T, w_T, b_T, acc_T, fix_kernel_width, fix_kernel_height>(
                    in, w, biases, out, perc_areas[i], quant_params,
                    val_min_limit, val_max_limit,
                    stride_height, stride_width,
                    dilation_height, dilation_width,
                    padding_top, padding_left,
                    padding_bot, padding_right);
        }
    }
}

#pragma MLI_CODE_SECTION_END()
} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_CONVOLUTION_DSP_H_
