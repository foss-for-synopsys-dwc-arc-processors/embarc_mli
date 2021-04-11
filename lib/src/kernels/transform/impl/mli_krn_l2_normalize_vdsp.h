/*
* Copyright 2020, Synopsys, Inc.
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

    const MLI_PTR(io_T) orig_vec_in = vec_in;

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
    for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                vec_in  = (MLI_PTR(io_T))orig_vec_in + POS(in_prv,  pos0, pos1, pos2, 0);
                if (remaining_part) {
                    input = mli_prv_load_1vec(vec_in);
                    auto converted_input = convert_input<convert>(input, in_zp, remaining_part);
                    auto input_lo = converted_input & 0xFF;
                    auto input_hi = (converted_input >> 8) & 0xFF;
                    sum_acc_hi  = mli_math_mac_fx(sum_acc_hi, input_hi, input_hi);
                    sum_acc_lo  = mli_math_mac_fx(sum_acc_lo, input_lo, input_lo);
                    sum_acc_mid = mli_math_mac_fx(sum_acc_mid, input_hi, input_lo);
                    vec_in  += remaining_part;
                }
                for (int pos3 = remaining_part; pos3 < in_prv->shape[3]; pos3 += num_lanes) {
                    input = mli_prv_load_1vec(vec_in);
                    auto converted_input = convert_input<convert>(input, in_zp);
                    auto input_lo = converted_input & 0xFF;
                    auto input_hi = (converted_input >> 8) & 0xFF;
                    sum_acc_hi  = mli_math_mac_fx(sum_acc_hi, input_hi, input_hi);
                    sum_acc_lo  = mli_math_mac_fx(sum_acc_lo, input_lo, input_lo);
                    sum_acc_mid = mli_math_mac_fx(sum_acc_mid, input_hi, input_lo);
                    vec_in  += num_lanes;
                }
            }
        }
    }

    constexpr int acc_hi_shift  = 16;
    constexpr int acc_mid_shift = 9;
    mli_acc32_t acc_hi  = mli_math_intra_sum(sum_acc_hi);
    mli_acc32_t acc_mid = mli_math_intra_sum(sum_acc_mid);
    mli_acc32_t acc_lo  = mli_math_intra_sum(sum_acc_lo);

    typedef typename std::conditional<convert == false, mli_acc40_t, mli_acc32_t>::type acc_type;
    acc_type acc = mli_math_add_fx((acc_type)acc_lo, mli_math_asl_fx((acc_type)acc_hi, acc_hi_shift));
             acc = mli_math_add_fx(acc, mli_math_asl_fx((acc_type)acc_mid, acc_mid_shift));

    int norm_shift_val = mli_math_norm_fx<acc_type, int>(acc);
    /* To Cast mli_acc32_t to int16_t */
    norm_shift_val = (sizeof(acc_type) - sizeof(int16_t)) * 8 - norm_shift_val;
    /* Adjust norm_shift to even number because we are going to divide it by 2 */
    if ((norm_shift_val & 0x1) == 0x1) {
        norm_shift_val += 1;
    }

    *norm_shift = norm_shift_val;
    /* Cast Sum_acc to Q7.8 to bring it to LUT input range */
    return mli_math_cast_fx<acc_type, int16_t>(acc, norm_shift_val);
}

template<bool convert>
static MLI_FORCE_INLINE vNx4char_t compute_normalize(
        vNx4char_t input, 
        int16_t scale,
        int16_t in_zp,
        int shift) {

    int shift_right = MAX(shift, 0);
    int shift_left  = MAX(-shift, 0);
    vNx4short_t input_cast = mli_math_cast_fx<vNx4char_t, vNx4short_t>(input);
    
    if (convert) {
        input_cast = mli_math_sub_fx<vNx4short_t>(input_cast, in_zp);
    }

    vNx4accint_t res = mli_math_mul_fx<vNx4short_t, vNx4accint_t>(input_cast, scale);
    res = mli_math_asl_fx(res, shift_left);

    return mli_math_acc_cast_fx<vNx4char_t, vNx4accint_t>(res, shift_right);
}

template<bool convert>
static MLI_FORCE_INLINE vNx2short_t compute_normalize(
        vNx2short_t input, 
        int16_t scale,
        int16_t in_zp,
        int shift) {
    
    int shift_right = MAX(shift, 0);
    int shift_left  = MAX(-shift, 0);

    if (convert) {
        input = mli_math_sub_fx<vNx2short_t>(input, in_zp);
    }

    vNx2accint_t res = mli_math_mul_fx<vNx2short_t, vNx2accint_t>(input, scale);
    res = mli_math_asl_fx(res, shift_left);

    return mli_math_acc_cast_fx<vNx2short_t, vNx2accint_t>(res, shift_right);
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
                    int current_el = MIN(remaining_el, num_lanes); /* nr remaining elements computed in this loop iteration */
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
