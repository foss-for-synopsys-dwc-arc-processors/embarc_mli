/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_L2_NORMALIZE_VDSP_H_
#define _MLI_KRN_L2_NORMALIZE_VDSP_H_

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace vdsp {

template<bool convert>
static MLI_FORCE_INLINE vNx4short_t convert_input(
        vNx4char_t input,
        int16_t in_zp,
        int remaining_part = 0) {
    
    
    if (remaining_part) {
        pvNx4 predicate = init_predicate(remaining_part, input);
        input = mli_math_select_fx(predicate, input, (vNx4char_t)in_zp);
    }

    vNx4short_t input_cast = mli_math_cast_fx<vNx4char_t, vNx4short_t>(input);

    if (convert) {
        input_cast = mli_math_sub_fx<vNx4short_t>(input_cast, in_zp);
    }

    return mli_math_abs_fx<vNx4short_t>(input_cast);
}

template<bool convert>
static MLI_FORCE_INLINE vNx2short_t convert_input(
        vNx2short_t input,
        int16_t in_zp,
        int remaining_part = 0) {
    
    if (remaining_part) {
        pvNx2 predicate = init_predicate(remaining_part, input);
        input = mli_math_select_fx(predicate, input, (vNx2short_t)in_zp);
    }

    if (convert) {
        input = mli_math_sub_fx<vNx2short_t>(input, in_zp);
    }

    return mli_math_abs_fx<vNx2short_t>(input);
}

static MLI_FORCE_INLINE vNx4accint_t init_sum_acc(vNx4char_t input) {
    return mli_prv_init_accu<vNx4accint_t>();
}

static MLI_FORCE_INLINE vNx2accint_t init_sum_acc(vNx2short_t input) {
    // Update Accu initialization when a known issue is solved.
    // return mli_prv_init_accu<vNx2accint_t>();
    return mli_math_mul_fx<vNx2short_t, vNx2accint_t>(input, (vNx2short_t) 0);
}

static MLI_FORCE_INLINE int16_t normalize_sum(
        vNx2accint_t sum_acc_hi,
        vNx2accint_t sum_acc_mid,
        vNx2accint_t sum_acc_lo,
        int *norm_shift) {

    constexpr int acc_hi_shift  = 16;
    constexpr int acc_mid_shift = 9;
    constexpr int acc_shift = acc_hi_shift;
    constexpr int acc_head_room = 1;

    mli_acc32_t acc_hi  = mli_math_intra_sum(sum_acc_hi);
    mli_acc32_t acc_mid = mli_math_intra_sum(sum_acc_mid);
    mli_acc32_t acc_lo  = mli_math_intra_sum(sum_acc_lo);

    int acc_hi_norm_shift = mli_math_norm_fx<mli_acc32_t, int>(acc_hi) - acc_head_room;

    mli_acc32_t acc = mli_math_asl_fx(acc_hi, acc_hi_norm_shift);
                acc = mli_math_add_fx(acc, mli_math_asr_fx(acc_mid, acc_shift - acc_mid_shift - acc_hi_norm_shift));
                acc = mli_math_add_fx(acc, mli_math_asr_fx(acc_lo, acc_shift - acc_hi_norm_shift));

    int norm_shift_val = mli_math_norm_fx<mli_acc32_t, int>(acc);
    /* To Cast mli_acc32_t to int16_t */
    norm_shift_val = acc_shift - acc_hi_norm_shift + (sizeof(mli_acc32_t) - sizeof(int16_t)) * 8 - norm_shift_val;
    /* Adjust norm_shift to even number because we are going to divide it by 2 */
    if ((norm_shift_val & 0x1) == 0x1) {
        norm_shift_val += 1;
    }

    *norm_shift = norm_shift_val;
    /* Cast Sum_acc to Q7.8 to bring it to LUT input range */
    return mli_math_cast_fx<mli_acc32_t, int16_t>(acc, norm_shift_val - (acc_shift - acc_hi_norm_shift));
}

template<typename acc_t>
static MLI_FORCE_INLINE int16_t normalize_sum(
        acc_t sum_acc,
        int *norm_shift) {

    mli_acc32_t acc = mli_math_intra_sum(sum_acc);

    int norm_shift_val = mli_math_norm_fx<mli_acc32_t, int>(acc);
    /* To Cast mli_acc32_t to int16_t */
    norm_shift_val = (sizeof(mli_acc32_t) - sizeof(int16_t)) * 8 - norm_shift_val;
    /* Adjust norm_shift to even number because we are going to divide it by 2 */
    if ((norm_shift_val & 0x1) == 0x1) {
        norm_shift_val += 1;
    }

    *norm_shift = norm_shift_val;
    /* Cast Sum_acc to Q7.8 to bring it to LUT input range */
    return mli_math_cast_fx<mli_acc32_t, int16_t>(acc, norm_shift_val);
}

template<bool is_remaining_part = false>
static MLI_FORCE_INLINE void accumlate_sum(
        vNx2accint_t &sum_acc_hi,
        vNx2accint_t &sum_acc_mid,
        vNx2accint_t &sum_acc_lo,
        vNx2short_t input,
        int16_t in_zp,
        int remaining_part = 0)
{
    auto converted_input = (is_remaining_part) ? 
                            convert_input<false>(input, in_zp, remaining_part) :
                            convert_input<false>(input, in_zp);
    auto input_lo = converted_input & 0xFF;
    auto input_hi = (converted_input >> 8) & 0xFF;
    sum_acc_hi  = mli_math_mac_fx(sum_acc_hi, input_hi, input_hi);
    sum_acc_lo  = mli_math_mac_fx(sum_acc_lo, input_lo, input_lo);
    sum_acc_mid = mli_math_mac_fx(sum_acc_mid, input_hi, input_lo);
}

template<typename acc_T, typename in_T, bool convert, bool is_remaining_part = false>
static MLI_FORCE_INLINE void accumlate_sum(
        acc_T &sum_acc,
        in_T input,
        int16_t in_zp,
        int remaining_part = 0)
{

    auto converted_input = (is_remaining_part) ? 
                            convert_input<convert>(input, in_zp, remaining_part) :
                            convert_input<convert>(input, in_zp);
    sum_acc  = mli_math_mac_fx(sum_acc, converted_input, converted_input);
}

template<typename io_T, bool convert, bool one_dim_with_mem_stride>
static MLI_FORCE_INLINE int16_t compute_normalized_sum_square_one_dim(
        const MLI_PTR(io_T) vec_in,
        int16_t in_zp,
        int *norm_shift,
        const int one_dim_shape,
        const int one_dim_mem_stride) {
    
    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(vec_in);
    int num_lanes = get_number_lanes(input);
    int remaining_part = one_dim_shape & (num_lanes - 1);

    /* Accumulation through MAC */
    auto sum_acc = init_sum_acc(input);

    if (one_dim_with_mem_stride) {
        if (remaining_part) {
            input = mli_prv_stride_load_1vec(vec_in, one_dim_mem_stride);
            accumlate_sum<decltype(sum_acc), decltype(input), convert, true>(sum_acc, input, in_zp, remaining_part);
            vec_in  += remaining_part;
        }
        for(int idx = remaining_part; idx < one_dim_shape; idx += num_lanes) {
            input = mli_prv_stride_load_1vec(vec_in, one_dim_mem_stride);
            accumlate_sum<decltype(sum_acc), decltype(input), convert>(sum_acc, input, in_zp);
            vec_in  += num_lanes;
        }
    } else {
        if (remaining_part) {
            input = mli_prv_load_1vec(vec_in);
            accumlate_sum<decltype(sum_acc), decltype(input), convert, true>(sum_acc, input, in_zp, remaining_part);
            vec_in  += remaining_part;
        }
        for(int idx = remaining_part; idx < one_dim_shape; idx += num_lanes) {
            input = mli_prv_load_1vec(vec_in);
            accumlate_sum<decltype(sum_acc), decltype(input), convert>(sum_acc, input, in_zp);
            vec_in  += num_lanes;
        }
    }

    return normalize_sum<decltype(sum_acc)>(sum_acc, norm_shift);
}

template<>
MLI_FORCE_INLINE int16_t compute_normalized_sum_square_one_dim<int16_t, false, false>(
        const MLI_PTR(int16_t) vec_in,
        int16_t in_zp,
        int *norm_shift,
        const int one_dim_shape,
        const int one_dim_mem_stride) {
    
    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(vec_in);
    int num_lanes = get_number_lanes(input);
    int remaining_part = one_dim_shape & (num_lanes - 1);

    /* To increase range of the result of sum(x^2), it's calculated as following:
    *      sum(x^2) = sum((abs(x))^2)
    *               = sum((xh * 2^8 + xl)^2)
    *               = sum( (xh^2)*(2^16) + (2*xh*xl) * (2^8) + (xl^2))
    *               = 2^16 * sum(xh^2) + 2^9 * sum(xh*xl) + sum(xl^2)
    * */

    /* Accumulation through MAC */
    auto zero_acc = init_sum_acc(input);
    auto sum_acc_hi  = zero_acc;
    auto sum_acc_mid = zero_acc;
    auto sum_acc_lo  = zero_acc;

    if (remaining_part) {
        input = mli_prv_load_1vec(vec_in);
        accumlate_sum<true>(sum_acc_hi, sum_acc_mid, sum_acc_lo, input, in_zp, remaining_part);
        vec_in  += remaining_part;
    }
    for(int idx = remaining_part; idx < one_dim_shape; idx += num_lanes) {
        input = mli_prv_load_1vec(vec_in);
        accumlate_sum(sum_acc_hi, sum_acc_mid, sum_acc_lo, input, in_zp);
        vec_in  += num_lanes;
    }

    return normalize_sum(sum_acc_hi, sum_acc_mid, sum_acc_lo, norm_shift);
}

template<>
MLI_FORCE_INLINE int16_t compute_normalized_sum_square_one_dim<int16_t, false, true>(
        const MLI_PTR(int16_t) vec_in,
        int16_t in_zp,
        int *norm_shift,
        const int one_dim_shape,
        const int one_dim_mem_stride) {
    
    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(vec_in);
    int num_lanes = get_number_lanes(input);
    int remaining_part = one_dim_shape & (num_lanes - 1);

    /* Accumulation through MAC */
    auto zero_acc = init_sum_acc(input);
    auto sum_acc_hi  = zero_acc;
    auto sum_acc_mid = zero_acc;
    auto sum_acc_lo  = zero_acc;

    if (remaining_part) {
        input = mli_prv_stride_load_1vec(vec_in, one_dim_mem_stride);
        accumlate_sum<true>(sum_acc_hi, sum_acc_mid, sum_acc_lo, input, in_zp, remaining_part);
        vec_in  += remaining_part;
    }
    for(int idx = remaining_part; idx < one_dim_shape; idx += num_lanes) {
        input = mli_prv_stride_load_1vec(vec_in, one_dim_mem_stride);
        accumlate_sum(sum_acc_hi, sum_acc_mid, sum_acc_lo, input, in_zp);
        vec_in  += num_lanes;
    }

    return normalize_sum(sum_acc_hi, sum_acc_mid, sum_acc_lo, norm_shift);
}

template<typename io_T, bool convert>
static MLI_FORCE_INLINE int16_t compute_normalized_sum_square(
        struct generic_tensor_private_t<MLI_PTR(io_T)> *in_prv,
        const MLI_PTR(io_T) vec_in,
        int16_t in_zp,
        int *norm_shift) {

    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(vec_in);
    int num_lanes = get_number_lanes(input);
    int remaining_part = in_prv->shape[3] & (num_lanes - 1);

    /* Accumulation through MAC */
    auto sum_acc = init_sum_acc(input);

    const MLI_PTR(io_T) orig_vec_in = vec_in;
    for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                vec_in  = (MLI_PTR(io_T))orig_vec_in + POS(in_prv,  pos0, pos1, pos2, 0);
                if (remaining_part) {
                    input = mli_prv_load_1vec(vec_in);
                    accumlate_sum<decltype(sum_acc), decltype(input), convert, true>
                                                        (sum_acc, input, in_zp, remaining_part);
                    vec_in  += remaining_part;
                }
                for (int pos3 = remaining_part; pos3 < in_prv->shape[3]; pos3 += num_lanes) {
                    input = mli_prv_load_1vec(vec_in);
                    accumlate_sum<decltype(sum_acc), decltype(input), convert>(sum_acc, input, in_zp);
                    vec_in  += num_lanes;
                }
            }
        }
    }

    return normalize_sum<decltype(sum_acc)>(sum_acc, norm_shift);
}

template<>
MLI_FORCE_INLINE int16_t compute_normalized_sum_square<int16_t, false>(
        struct generic_tensor_private_t<MLI_PTR(int16_t)> *in_prv,
        const MLI_PTR(int16_t) vec_in,
        int16_t in_zp,
        int *norm_shift) {

    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(vec_in);
    int num_lanes = get_number_lanes(input);
    int remaining_part = in_prv->shape[3] & (num_lanes - 1);

    /* Accumulation through MAC */
    auto zero_acc = init_sum_acc(input);
    auto sum_acc_hi  = zero_acc;
    auto sum_acc_mid = zero_acc;
    auto sum_acc_lo  = zero_acc;

    const MLI_PTR(int16_t) orig_vec_in = vec_in;
    for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                vec_in  = (MLI_PTR(int16_t))orig_vec_in + POS(in_prv,  pos0, pos1, pos2, 0);
                if (remaining_part) {
                    input = mli_prv_load_1vec(vec_in);
                    accumlate_sum<true>(sum_acc_hi, sum_acc_mid, sum_acc_lo, input, in_zp, remaining_part);
                    vec_in  += remaining_part;
                }
                for (int pos3 = remaining_part; pos3 < in_prv->shape[3]; pos3 += num_lanes) {
                    input = mli_prv_load_1vec(vec_in);
                    accumlate_sum(sum_acc_hi, sum_acc_mid, sum_acc_lo, input, in_zp);
                    vec_in  += num_lanes;
                }
            }
        }
    }

    return normalize_sum(sum_acc_hi, sum_acc_mid, sum_acc_lo, norm_shift);
}

template<bool convert>
static MLI_FORCE_INLINE vNx4char_t compute_normalize(
        vNx4char_t input, 
        int16_t scale,
        int16_t in_zp,
        int shift) {

    /*
     * shifting more than 24 is not needed
     * as the scaled result = ((input - in_offset) * scale) will be limited by 24 bits.
     */
    constexpr int max_shift = 24;
    constexpr int mul_hi_shift = 16;
    shift = mli_math_min_fx(shift, max_shift);
    shift -= mul_hi_shift;
    int shift_right = mli_math_max_fx(shift, 1);
    int shift_left = mli_math_max_fx(1 - shift, 0);
    vNx4short_t input_cast = mli_math_cast_fx<vNx4char_t, vNx4short_t>(input);
    
    if (convert) {
        input_cast = mli_math_sub(input_cast, in_zp);
    }

    input_cast = mli_math_asl_fx(input_cast, shift_left);
    vNx4short_t res = mli_math_mul_fx_high(input_cast, scale);

    return mli_math_cast_fx<vNx4short_t, vNx4char_t>(res, shift_right);
}

template<bool convert>
static MLI_FORCE_INLINE vNx2short_t compute_normalize(
        vNx2short_t input, 
        int16_t scale,
        int16_t in_zp,
        int shift) {
    
    int shift_right = mli_math_max_fx(shift, 0);
    int shift_left  = mli_math_max_fx(-shift, 0);

    if (convert) {
        input = mli_math_sub_fx<vNx2short_t>(input, in_zp);
    }
    
    vNx2accint_t res = mli_math_mul_fx<vNx2short_t, vNx2accint_t>(input, scale);
    res = mli_math_asl_fx(res, shift_left);

    return mli_math_acc_cast_fx<vNx2short_t, vNx2accint_t>(res, shift_right);
}

template<typename io_T, bool convert, bool one_dim_with_mem_stride>
static MLI_FORCE_INLINE void normalize_tensor_one_dim(
        const MLI_PTR(io_T) vec_in,
        MLI_PTR(io_T) vec_out,
        int16_t scale,
        int16_t in_zp,
        int shift, 
        const int one_dim_shape,
        const int one_dim_in_mem_stride,
        const int one_dim_out_mem_stride) {
    
    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(vec_in);
    int num_lanes = get_number_lanes(input);
    int remaining_part = one_dim_shape & (num_lanes - 1);

    if (one_dim_with_mem_stride) {
        if (remaining_part) {
            input = mli_prv_stride_load_1vec(vec_in, one_dim_in_mem_stride);
            mli_prv_stride_store_n_samples(vec_out, 
                                           compute_normalize<convert>(input, scale, in_zp, shift),
                                           one_dim_out_mem_stride,
                                           remaining_part);
            vec_in  += remaining_part;
            vec_out += remaining_part;
        }
        for(int idx = remaining_part; idx < one_dim_shape; idx += num_lanes) {
            input = mli_prv_stride_load_1vec(vec_in, one_dim_in_mem_stride);
            mli_prv_stride_store_n_samples(vec_out, 
                                           compute_normalize<convert>(input, scale, in_zp, shift),
                                           one_dim_out_mem_stride);
            vec_in  += num_lanes;
            vec_out += num_lanes;
        }
    } else {
        if (remaining_part) {
            input = mli_prv_load_1vec(vec_in);
            mli_prv_store_n_samples(vec_out,
                                    compute_normalize<convert>(input, scale, in_zp, shift),
                                    remaining_part);
            vec_in  += remaining_part;
            vec_out += remaining_part;
        }
        for(int idx = remaining_part; idx < one_dim_shape; idx += num_lanes) {
            input = mli_prv_load_1vec(vec_in);
            mli_prv_store_n_samples(vec_out, compute_normalize<convert>(input, scale, in_zp, shift));
            vec_in  += num_lanes;
            vec_out += num_lanes;
        }
    }
}

template<typename io_T, bool convert>
static MLI_FORCE_INLINE void normalize_tensor(
        struct generic_tensor_private_t<MLI_PTR(io_T)> *in_prv,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *out_prv,
        const MLI_PTR(io_T) vec_in,
        MLI_PTR(io_T) vec_out,
        int16_t scale,
        int16_t in_zp,
        int shift) {

    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(vec_in);
    int num_lanes = get_number_lanes(input);

    const MLI_PTR(io_T) orig_vec_in = vec_in;
    MLI_OUT_PTR(io_T) orig_vec_out = vec_out;

    for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                vec_in  = (MLI_PTR(io_T))orig_vec_in + POS(in_prv,  pos0, pos1, pos2, 0);
                vec_out = orig_vec_out + POS(out_prv, pos0, pos1, pos2, 0);
                for (int pos3 = 0; pos3 < in_prv->shape[3]; pos3 += num_lanes) {
                    int remaining_el = in_prv->shape[3] - pos3;
                    /* current_el remaining elements computed in this loop iteration */
                    int current_el = MIN(remaining_el, num_lanes);
                    input = mli_prv_load_1vec(vec_in);
                    mli_prv_store_n_samples(vec_out, 
                        compute_normalize<convert>(input, scale, in_zp, shift), current_el);
                    vec_in  += num_lanes;
                    vec_out += num_lanes;
                }
            }
        }
    }
}


} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_L2_NORMALIZE_VDSP_H_
