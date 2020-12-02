/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_PRELU_DSP_H_
#define _MLI_KRN_PRELU_DSP_H_

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
namespace dsp {

static MLI_FORCE_INLINE v2q15_t calc_prelu(
        v2q15_t input,
        v2q15_t scale_v,
        const int shift) {

    /* out  = max(0, in) + alpha * min(0, in) */
    v2q15_t zero = mli_prv_init_v(0);
    v2q15_t pos = mli_math_max_fx(zero, input);
    v2q15_t neg = mli_math_acc_cast_fx<v2q15_t, v2accum40_t>(
                  mli_math_mul_fx<v2q15_t, v2accum40_t>(scale_v, mli_math_min_fx(zero, input)), shift);
    return mli_math_add_fx(pos, neg);
}

template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift) {

    v2q15_t input = mli_prv_load_1vec(vec_in);
    mli_prv_store_n_samples(vec_out, calc_prelu(input, scale, shift));
}

template <typename io_T, typename scale_T>
static MLI_FORCE_INLINE void compute_prelu(
        const MLI_PTR(io_T) vec_in,
        const scale_T scale,
        MLI_OUT_PTR(io_T) vec_out,
        const int shift,
        const int remaining_part) {

    MLI_ASSERT(remaining_part == 1);
    v2q15_t input = mli_prv_load_1vec(vec_in);
    mli_prv_store_1_sample(vec_out, calc_prelu(input, scale, shift));
}

} // namespace dsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_PRELU_DSP_H_