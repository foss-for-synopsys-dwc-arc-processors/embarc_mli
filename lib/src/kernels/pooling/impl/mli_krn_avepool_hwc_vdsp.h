/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_AVEPOOL_HWC_VDSP_H_
#define _MLI_KRN_AVEPOOL_HWC_VDSP_H_

#include "mli_krn_reduce_sum2d.h"
#include "mli_prv_load_store.h"
#include "mli_prv_dsp.h"
#include "mli_math.h"

namespace mli {
namespace krn {
namespace vdsp {

static MLI_FORCE_INLINE void compute_avepool_func(
        const MLI_PTR(int8_t) __restrict in,
        MLI_OUT_PTR(int8_t) __restrict out,
        const int16_t mul,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const int32_t zp,
        const int shift_value,
        const int channels)
{
    vNx4char_t res = mli::krn::reduce_sum2D_v(in, mul, zp, width, height,
                                              col_mem_stride, row_mem_stride, shift_value);

    mli_prv_store_n_samples(out, res, channels);
}

static MLI_FORCE_INLINE void compute_avepool_func(
        const MLI_PTR(int16_t) __restrict in,
        MLI_OUT_PTR(int16_t) __restrict out,
        const int16_t mul,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const int32_t zp,
        const int shift_value,
        const int channels)
{
    vNx2accint_t accu = mli_prv_init_accu_with_bias_v<vNx2accint_t>(zp, shift_value);
#if (__Xvec_guard_bit_option == 0) && !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
    int accum_shift_amout = 0;
    vNx2int_t res = mli::krn::reduce_sum2D_v(in, mul, accu, width, height, 
		                                col_mem_stride, row_mem_stride, &accum_shift_amout);

    mli_prv_clip_and_store_output_v(out, res, shift_value - accum_shift_amout, channels);
#else
    accu = mli::krn::reduce_sum2D_v(in, mul, accu, width, height, col_mem_stride, row_mem_stride);
    
    mli_prv_clip_and_store_output_v(out, accu, shift_value, channels);
#endif
}

template<typename io_T>
static MLI_FORCE_INLINE void compute_avepool_func_k2x2_padding_kernel_unroll(
        const MLI_PTR(io_T) __restrict in,
        MLI_OUT_PTR(io_T) __restrict out,
        const int16_t mul,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const int32_t zp,
        const int shift_value,
        const int channels) {

    switch (height) {
    case 1:
        switch (width) {
        case 1:
            compute_avepool_func(
              in, out, mul, 1, 1, col_mem_stride, row_mem_stride, zp ,shift_value, channels);
            break;

        case 2:
            compute_avepool_func(
              in, out, mul, 2, 1, col_mem_stride, row_mem_stride, zp ,shift_value, channels);
            break;

        default:
            MLI_ASSERT(0);
            break;
        }
        break;

    case 2:
        switch (width) {
        case 1:
            compute_avepool_func(
              in, out, mul, 1, 2, col_mem_stride, row_mem_stride, zp ,shift_value, channels);
            break;

        default:
            MLI_ASSERT(0);
            break;
        }
        break;
    }
}

template<typename io_T>
static MLI_FORCE_INLINE void compute_avepool_func_k3x3_padding_kernel_unroll(
        const MLI_PTR(io_T) __restrict in,
        MLI_OUT_PTR(io_T) __restrict out,
        const int16_t mul,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const int32_t zp,
        const int shift_value,
        const int channels) {

    switch (height) {
    case 1:
        switch (width) {
        case 1:
            compute_avepool_func(
              in, out, mul, 1, 1, col_mem_stride, row_mem_stride, zp ,shift_value, channels);
            break;

        case 2:
            compute_avepool_func(
              in, out, mul, 2, 1, col_mem_stride, row_mem_stride, zp ,shift_value, channels);
            break;

        case 3:
            compute_avepool_func(
              in, out, mul, 3, 1, col_mem_stride, row_mem_stride, zp ,shift_value, channels);
            break;

        default:
            MLI_ASSERT(0);
            break;
        }
        break;

    case 2:
        switch (width) {
        case 1:
            compute_avepool_func(
              in, out, mul, 1, 2, col_mem_stride, row_mem_stride, zp ,shift_value, channels);
            break;

        case 2:
            compute_avepool_func(
              in, out, mul, 2, 2, col_mem_stride, row_mem_stride, zp ,shift_value, channels);
            break;

        case 3:
            compute_avepool_func(
              in, out, mul, 3, 2, col_mem_stride, row_mem_stride, zp ,shift_value, channels);
            break;

        default:
            MLI_ASSERT(0);
            break;
        }
        break;

    case 3:
        switch (width) {
        case 1:
            compute_avepool_func(
              in, out, mul, 1, 3, col_mem_stride, row_mem_stride, zp ,shift_value, channels);
            break;

        case 2:
            compute_avepool_func(
              in, out, mul, 2, 3, col_mem_stride, row_mem_stride, zp ,shift_value, channels);
            break;

        default:
            MLI_ASSERT(0);
            break;
        }
        break;

        default:
            MLI_ASSERT(0);
            break;
    }
}

template<typename io_T, int fixed_kernel_size, bool varying_kernel>
static MLI_FORCE_INLINE void compute_avepool(
        const MLI_PTR(io_T) __restrict in,
        MLI_OUT_PTR(io_T) __restrict out,
        const int16_t mul,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        const int32_t zp,
        const int shift_value,
        const int channels) {

    if (varying_kernel && fixed_kernel_size == 3) {
        compute_avepool_func_k3x3_padding_kernel_unroll<io_T>(
            in, out, mul, width, height, col_mem_stride, row_mem_stride, zp ,shift_value, channels);
    } else if (varying_kernel && fixed_kernel_size == 2) {
        compute_avepool_func_k2x2_padding_kernel_unroll<io_T>(
            in, out, mul, width, height, col_mem_stride, row_mem_stride, zp ,shift_value, channels);
    } else {
        compute_avepool_func(
              in, out, mul, width, height, col_mem_stride, row_mem_stride, zp ,shift_value, channels);
    }
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_AVEPOOL_HWC_VDSP_H_
