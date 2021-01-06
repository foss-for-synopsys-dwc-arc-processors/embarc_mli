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

template<bool remaining_channels>
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
    vNx4accint_t accu = mli_prv_init_accu_with_bias_v<vNx4accint_t>(zp, shift_value);
    
    accu = mli::krn::reduce_sum2D_v(in, mul, accu, width, height, col_mem_stride, row_mem_stride);

    if (remaining_channels) {
        mli_prv_clip_and_store_output_v(out, accu, shift_value, channels);
	} else {
	    mli_prv_clip_and_store_output_v(out, accu, shift_value);
	}

}

template<bool remaining_channels>
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

    shift_value -= accum_shift_amout;
    if (remaining_channels) {
        mli_prv_clip_and_store_output_v(out, res, shift_value, channels);
	} else {
	    mli_prv_clip_and_store_output_v(out, res, shift_value);
	}
#else
    accu = mli::krn::reduce_sum2D_v(in, mul, accu, width, height, col_mem_stride, row_mem_stride);
    
    if (remaining_channels) {
        mli_prv_clip_and_store_output_v(out, accu, shift_value, channels);
	} else {
	    mli_prv_clip_and_store_output_v(out, accu, shift_value);
	}

#endif    
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_AVEPOOL_HWC_VDSP_H_