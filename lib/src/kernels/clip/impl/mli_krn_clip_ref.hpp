/*
 * Copyright 2022, Synopsys, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-3-Clause license found in
 * the LICENSE file in the root directory of this source tree.
 *
 */

#ifndef _MLI_KRN_CLIP_REF_HPP_
#define _MLI_KRN_CLIP_REF_HPP_

#include "mli_math.h"
#include "mli_types.h"
#include <iostream>
namespace snps_arc::metaware::mli {
namespace krn {
namespace ref {

// First priority is 32accum to T value operation.
template <typename o_T>
static MLI_FORCE_INLINE o_T clip_value(const int8_t in_val,
                                       const int8_t min_value,
                                       const int8_t max_value) {
    o_T result = mli_math_max_fx(in_val, min_value);
    result = mli_math_min_fx(result, max_value);
    return result;
}

template <typename i_T, typename o_T>
static MLI_FORCE_INLINE void compute_clip_per_tensor(
        const generic_tensor_private_t<MLI_PTR(i_T)> &in_data,
        const i_T min_value,
        const i_T max_value,
        generic_tensor_private_t<MLI_OUT_PTR(o_T)> &out_data) {

    MLI_ASSERT(MLI_MAX_RANK <=
            sizeof(in_data.shape) / sizeof(in_data.shape[0]));
    MLI_ASSERT(MLI_MAX_RANK
            <= sizeof(out_data.shape) / sizeof(out_data.shape[0]));

    // Note: Here pos introduced in array form Just for alignement with
    // compute_rescale_per_axis
    int pos[MLI_MAX_RANK] = {0};
    for (pos[0] = 0; pos[0] < in_data.shape[0]; pos[0]++) {
        for (pos[1] = 0; pos[1] < in_data.shape[1]; pos[1]++) {
            for (pos[2] = 0; pos[2] < in_data.shape[2]; pos[2]++) {
                for (pos[3] = 0; pos[3] < in_data.shape[3]; pos[3]++) {
                    i_T in_val = mli_prv_tensor_read(in_data, pos[0], pos[1],
                            pos[2], pos[3]);
                    o_T out_val = clip_value<o_T>(in_val, min_value, max_value);
                    mli_prv_tensor_write(out_val, out_data, pos[0],
                            pos[1], pos[2], pos[3]);
                }
            }
        }
    }
}

template <typename i_T, typename o_T>
mli_status MLI_FORCE_INLINE mli_krn_clip(const mli_tensor *in,
                                         const mli_tensor *min,
                                         const mli_tensor *max,
                                         mli_tensor *out) {

    mli_prv_fx_init_dsp_ctrl();

    const auto in_prv = mli_prv_get_generic_tensor<MLI_PTR(i_T)>(in);
    auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(o_T)>(out);

    // Asserts are in checkers
    const i_T min_value = mli_prv_tensor_data_val<i_T>(min);
    const i_T max_value = mli_prv_tensor_data_val<i_T>(max);
    compute_clip_per_tensor(in_prv, min_value, max_value, out_prv);

    return MLI_STATUS_OK;
}

}  // namespace ref
}  // namespace krn
}  // namespace snps_arc::metaware::mli

#endif // _MLI_KRN_RESCALE_REF_HPP_
