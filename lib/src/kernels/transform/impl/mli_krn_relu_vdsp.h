/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_RELU_VDSP_H_
#define _MLI_KRN_RELU_VDSP_H_

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

template<typename io_T>
static MLI_FORCE_INLINE void compute_relu_inner_loop(
        const MLI_PTR(io_T) __restrict vec_in,
        MLI_OUT_PTR(io_T) __restrict vec_out,
        const io_T min_val,
        const io_T max_val,
        const int count) {
    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(vec_in);
    int num_lanes = get_number_lanes(input);
    int remaining_part = count & (num_lanes - 1);

    if (remaining_part) {
        input = mli_prv_load_1vec(vec_in);
        mli_prv_store_n_samples(vec_out, mli_math_min_fx(
            mli_math_max_fx(input, min_val), max_val), remaining_part);
        vec_in  += remaining_part;
        vec_out += remaining_part;
    }
#pragma clang loop unroll_count(4)
    for (int pos3 = remaining_part; pos3 < count; pos3 += num_lanes) {
        input = mli_prv_load_1vec(vec_in);
        mli_prv_store_n_samples(vec_out, mli_math_min_fx(
            mli_math_max_fx(input, min_val), max_val));
        vec_in  += num_lanes;
        vec_out += num_lanes;
    }
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RELU_VDSP_H_
