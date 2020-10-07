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

template <typename io_T, prelu_elem_func_type func_type>
static MLI_FORCE_INLINE void mli_krn_scale_elem_v(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift) {
    v2q15_t input = mli_prv_load_1vec(vec_in);
    v2q15_t scale_v = mli_prv_init_v(scale);
    v2q15_t output;
    v2accum40_t tmp_acc = mli_math_mul_fx<v2q15_t, v2accum40_t>(scale_v, input);
    if (func_type == PRELU_ELEM_FUNC_MAX) {
        output = mli_math_max_fx(input, 
                mli_math_acc_cast_fx<v2q15_t, v2accum40_t>(tmp_acc,  shift));
    } else {
        output = mli_math_min_fx(input, 
                mli_math_acc_cast_fx<v2q15_t, v2accum40_t>(tmp_acc,  shift));
    }
    mli_prv_store_n_samples(vec_out, output);
}

template <typename io_T, prelu_elem_func_type func_type>
static MLI_FORCE_INLINE void mli_krn_scale_elem_v(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift,
        const int remaining_part) {

    MLI_ASSERT(remaining_part == 1);
    v2q15_t input = mli_prv_load_1_sample(vec_in);
    v2q15_t scale_v = mli_prv_init_v(scale);
    v2q15_t output;
    v2accum40_t tmp_acc = mli_math_mul_fx<v2q15_t, v2accum40_t>(scale_v, input);
    if (func_type == PRELU_ELEM_FUNC_MAX) {
        output = mli_math_max_fx(input, 
                mli_math_acc_cast_fx<v2q15_t, v2accum40_t>(tmp_acc,  shift));
    } else {
        output = mli_math_min_fx(input, 
                mli_math_acc_cast_fx<v2q15_t, v2accum40_t>(tmp_acc,  shift));
    }
    mli_prv_store_1_sample(vec_out, output);
}

} // namespace dsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_PRELU_DSP_H_