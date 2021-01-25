/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_REDUCE_SUM2D_VDSP_H_
#define _MLI_KRN_REDUCE_SUM2D_VDSP_H_

#include "mli_prv_load_store.h"
#include "mli_prv_dsp.h"
#include "mli_math.h"

namespace mli {
namespace krn {
namespace vdsp {

template <typename acc_T>
static MLI_FORCE_INLINE acc_T reduce_sum2D_v(
        const MLI_PTR(int8_t) in,
        const int16_t mul,
        acc_T accu,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        int *accum_shift_amout) {

    int row_inc = row_mem_stride - width * col_mem_stride;
    vNx4accshort_t acc_short = mli_prv_init_accu<vNx4accshort_t>();

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            acc_short = mli_math_mac_fx(acc_short, mli_prv_load_nx4_samples(in), (int8_t)1);
            in += col_mem_stride;
        }
        in += row_inc;
    }
#pragma clang diagnostic pop

    vNx4short_t acc_casted = mli_math_acc_cast(acc_short); 
    accu = mli_math_mac_fx(accu, acc_casted, mul);
    return accu;
}

#if (__Xvec_guard_bit_option == 0) && !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
static MLI_FORCE_INLINE vNx4char_t reduce_sum2D_v(
        const MLI_PTR(int8_t) in,
        const int8_t mul,
        const int16_t accu_init,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        int shift_value) {

    constexpr int mul_hi_shift = 16;
    constexpr int mul_pre_shift = 8;
    /* To avoid using guardbits and have some space for bit growth
     * and aligning the accu result on msb to avoid lossing precision from mul_hi
     * accu_preshift = 16(size_of_short) - (log2(width * height) + 8(in_size))
     * For Kernels: WxH = 16 and less, accu_preshift = 4
     * For Kernels: WxH = 64 and less, accu_preshift = 2
     * Otherwise, accu_preshift = 1
     * */
    int accu_preshift = 1;
    if (width * height <= 16) {
        accu_preshift = 4;
    } else if (width * height <= 64) {
        accu_preshift = 2;
    }

    int row_inc = row_mem_stride - width * col_mem_stride;

    vNx4accshort_t acc_short = mli_math_mul_fx<vNx4char_t, vNx4accshort_t>
                                              (mli_prv_load_1vec(in), (int8_t)(1 << accu_preshift));
    in += col_mem_stride;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang loop unroll(full)
    for (int row = 0, clmn = 1; row < height; row++, clmn = 0) {
#pragma clang loop unroll(full)
        for (; clmn < width; clmn++) {
            acc_short = mli_math_mac_fx(acc_short, mli_prv_load_1vec(in), (int8_t)(1 << accu_preshift));
            in += col_mem_stride;
        }
        in += row_inc;
    }
#pragma clang diagnostic pop

    shift_value -= (mul_hi_shift - mul_pre_shift - accu_preshift);
    vNx4short_t acc_casted = mli_math_acc_cast(acc_short);
    acc_casted = mli_math_mul_fx_high(acc_casted, (((int16_t)mul) << mul_pre_shift));
    acc_casted = mli_math_asr_rnd_fx(acc_casted, shift_value);
    acc_casted = mli_math_add_fx<vNx4short_t>(acc_casted, accu_init);
    return mli_math_cast_fx<vNx4short_t, vNx4char_t>(acc_casted);
}
#else
static MLI_FORCE_INLINE vNx4char_t reduce_sum2D_v(
        const MLI_PTR(int8_t) in,
        const int8_t mul,
        const int16_t accu_init,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        int shift_value) {

    int row_inc = row_mem_stride - width * col_mem_stride;
    int16_t round = (1 << shift_value) >> 1;
    
    vNx4accshort_t acc_short = mli_math_init_accu<int16_t, vNx4accshort_t>(accu_init);
                   acc_short = mli_math_asl_fx(acc_short, shift_value);
                   acc_short = mli_math_add(acc_short, (vNx4short_t)round);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            acc_short = mli_math_mac_fx(acc_short, mli_prv_load_1vec(in), mul);
            in += col_mem_stride;
        }
        in += row_inc;
    }
#pragma clang diagnostic pop
    
    return mli_math_acc_cast_fx<vNx4char_t, vNx4accshort_t,/*round = */ false>(acc_short, shift_value);
}
#endif

#if (__Xvec_guard_bit_option == 0) && !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
#define MULTIPLY_FX_HI_SHIFT 32     // right shift done by mul_fx_hi operation
#define ACUMM_SUM_PRE_SHIFT 8       // elements left shift before accumulation
#define MUL_SHIFT 16                // Mul shifted left to be casted to 32 bits
static MLI_FORCE_INLINE vNx2int_t reduce_sum2D_v(
        const MLI_PTR(int16_t) in,
        const int16_t mul,
        vNx2accint_t accu,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        int *accum_shift_amout) {

    int row_inc = row_mem_stride - width * col_mem_stride;
    /* The sum elements (int16) are left shifted with 8 bits to have max (24 bits) leaving 8 bits of 32 as guard bits,
     * This is done to enlarge the multiplication value as we use multiply high.
     * Note: The output of this function is shifted right by 8 bits as result of the following:
     *       1- accum sum elements preshift left 8  (<< 8)
     *       2- mul shift left 16                   (<< 16)
     *       3- multiply high                       (>> 32)
     * So it need to be shifted back left by 8 (accum_shift_amout).
     */
    *accum_shift_amout = MULTIPLY_FX_HI_SHIFT - ACUMM_SUM_PRE_SHIFT - MUL_SHIFT;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            accu = mli_math_mac_fx(accu, mli_prv_load_nx2_samples(in), (int16_t) (1 << ACUMM_SUM_PRE_SHIFT));
            in += col_mem_stride;
        }
        in += row_inc;
    }
#pragma clang diagnostic pop

    vNx2int_t sum = mli_math_acc_cast_fx<vNx2int_t, vNx2accint_t>(accu);
    vNx2int_t average = mli_math_mul_fx_high(sum, (int32_t) (mul << MUL_SHIFT));
    return average;
}
#else

template <typename acc_T>
static MLI_FORCE_INLINE acc_T reduce_sum2D_v(
        const MLI_PTR(int16_t) in,
        const int16_t mul,
        acc_T accu,
        const int width,
        const int height,
        const int col_mem_stride,
        const int row_mem_stride,
        int *accum_shift_amout) {
    int row_inc = row_mem_stride - width * col_mem_stride;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang loop unroll(full)
    for (int row = 0; row < height; row++) {
#pragma clang loop unroll(full)
        for (int clmn = 0; clmn < width; clmn++) {
            accu = mli_math_mac_fx(accu, mli_prv_load_nx2_samples(in), mul);
            in += col_mem_stride;
        }
        in += row_inc;
    }
#pragma clang diagnostic pop

    return accu;
}

#endif

template <typename io_T, typename acc_T>
static MLI_FORCE_INLINE acc_T reduce_sum2D(
        const MLI_PTR(io_T) in,
        const int16_t mul,
        acc_T accu,
        const int width,
        const int height,
        const int channels,
        const int col_mem_stride,
        const int row_mem_stride) {

    return reduce_sum2D_v(in, mul, accu, width, height, col_mem_stride, row_mem_stride);
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_REDUCE_SUM2D_VDSP_H_
