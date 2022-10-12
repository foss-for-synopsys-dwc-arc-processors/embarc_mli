/*
 * Copyright 2022, Synopsys, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-3-Clause license found in
 * the LICENSE file in the root directory of this source tree.
 *
 */

#ifndef _MLI_KRN_RESCALE_REF_HPP_
#define _MLI_KRN_RESCALE_REF_HPP_

#include "mli_math.h"
#include "mli_types.h"

namespace snps_arc::metaware::mli {
namespace krn {
namespace ref {

// First priority is 32accum to T value operation.
template <typename i_T, typename o_T>
static MLI_FORCE_INLINE o_T rescale_value(
        const i_T in_val,
        const i_T in_bias,
        const o_T out_bias,
        const int16_t scale,
        const int shift_right) {
    constexpr int max_shift_right = 63;
    constexpr int max_shift_left = -63;
    int32_t shift = MAX(max_shift_left, MIN(shift_right, max_shift_right));

    i_T value = mli_math_sub_fx(in_val, in_bias);
    int64_t scaled_value = mli_math_mul_fx<int32_t, int64_t> (static_cast<int32_t>(value),
                                                              static_cast<int32_t>(scale));
    scaled_value = mli_math_ashift_right_fx(scaled_value, shift);
    scaled_value = mli_math_add_fx(scaled_value, static_cast<int64_t>(out_bias));
    o_T result = mli_math_cast_fx<int64_t, o_T>(scaled_value, 0);
    return result;
}

template <typename i_T, typename o_T>
static MLI_FORCE_INLINE void compute_rescale_per_axis(
        const generic_tensor_private_t<MLI_PTR(i_T)> &in_data,
        const int32_t rescale_axis,
        const i_T *in_bias,
        const o_T *out_bias,
        const int16_t *scale,
        const int8_t *shift,
        generic_tensor_private_t<MLI_OUT_PTR(o_T)> &out_data) {
    constexpr int kMaxSupportedRank = 4;

    MLI_ASSERT(rescale_axis < kMaxSupportedRank);
    MLI_ASSERT(kMaxSupportedRank <=
            sizeof(in_data.shape) / sizeof(in_data.shape[0]));
    MLI_ASSERT(kMaxSupportedRank
            <= sizeof(out_data.shape) / sizeof(out_data.shape[0]));

    int pos[kMaxSupportedRank] = {0};
    for (pos[0] = 0; pos[0] < in_data.shape[0]; pos[0]++) {
        for (pos[1] = 0; pos[1] < in_data.shape[1]; pos[1]++) {
            for (pos[2] = 0; pos[2] < in_data.shape[2]; pos[2]++) {
                for (pos[3] = 0; pos[3] < in_data.shape[3]; pos[3]++) {
                    const int param_idx = pos[rescale_axis];
                    i_T in_val = mli_prv_tensor_read(in_data, pos[0], pos[1],
                            pos[2], pos[3]);
                    o_T out_val = rescale_value(in_val, in_bias[param_idx],
                            out_bias[param_idx], scale[param_idx],
                            shift[param_idx]);
                    mli_prv_tensor_write(out_val, out_data, pos[0],
                            pos[1], pos[2], pos[3]);
                }
            }
        }
    }
}

template <typename i_T, typename o_T>
static MLI_FORCE_INLINE void compute_rescale_per_tensor(
        const generic_tensor_private_t<MLI_PTR(i_T)> &in_data,
        const i_T in_bias,
        const o_T out_bias,
        const int16_t scale,
        const int8_t shift,
        generic_tensor_private_t<MLI_OUT_PTR(o_T)> &out_data) {
    constexpr int kMaxSupportedRank = 4;

    MLI_ASSERT(kMaxSupportedRank <=
            sizeof(in_data.shape) / sizeof(in_data.shape[0]));
    MLI_ASSERT(kMaxSupportedRank
            <= sizeof(out_data.shape) / sizeof(out_data.shape[0]));

    int pos[kMaxSupportedRank] = {0};
    for (pos[0] = 0; pos[0] < in_data.shape[0]; pos[0]++) {
        for (pos[1] = 0; pos[1] < in_data.shape[1]; pos[1]++) {
            for (pos[2] = 0; pos[2] < in_data.shape[2]; pos[2]++) {
                for (pos[3] = 0; pos[3] < in_data.shape[3]; pos[3]++) {
                    i_T in_val = mli_prv_tensor_read(in_data, pos[0], pos[1],
                            pos[2], pos[3]);
                    o_T out_val = rescale_value(in_val, in_bias,
                            out_bias, scale, shift);
                    mli_prv_tensor_write(out_val, out_data, pos[0],
                            pos[1], pos[2], pos[3]);
                }
            }
        }
    }
}

template <typename i_T, typename o_T>
mli_status MLI_FORCE_INLINE mli_krn_rescale(const mli_tensor *in,
                                            const mli_tensor *bias_in,
                                            const mli_tensor *scale,
                                            const mli_tensor *shift,
                                            const mli_tensor *bias_out,
                                            const int32_t rescale_axis,
                                            mli_tensor *out) {
    mli_prv_fx_init_dsp_ctrl();

    const auto in_prv = mli_prv_get_generic_tensor<MLI_PTR(i_T)>(in);
    auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(o_T)>(out);
    if (rescale_axis < 0) {
        // Asserts are in checkers
        const i_T in_bias_val = mli_prv_tensor_data_val<i_T>(bias_in);
        const o_T out_bias_val = mli_prv_tensor_data_val<o_T>(bias_out);

        const int8_t shift_val = mli_prv_tensor_data_val<int8_t>(shift);
        const int16_t scale_val = mli_prv_tensor_data_val<int16_t>(scale);
        compute_rescale_per_tensor(in_prv, in_bias_val, out_bias_val,
                                   scale_val, shift_val, out_prv);
    } else {
        // Asserts are in checkers
        const i_T *in_bias_ptr = mli_prv_tensor_data_ptr<i_T *>(bias_in);
        const o_T *out_bias_ptr = mli_prv_tensor_data_ptr<o_T *>(bias_out);

        const int8_t *shift_ptr = mli_prv_tensor_data_ptr<int8_t *>(shift);
        const int16_t *scale_ptr = mli_prv_tensor_data_ptr<int16_t *>(scale);
        compute_rescale_per_axis(in_prv, rescale_axis, in_bias_ptr, out_bias_ptr,
                                 scale_ptr, shift_ptr, out_prv);
    }

    return MLI_STATUS_OK;
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RESCALE_REF_HPP_
