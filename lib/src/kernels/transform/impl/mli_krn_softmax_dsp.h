/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_SOFTMAX_DSP_H_
#define _MLI_KRN_SOFTMAX_DSP_H_

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_lut.h"
#include "mli_prv_activation_lut.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace dsp {

const int kTanhAsymZeroPoint = 0;
const int kTanhOutputShift = 7;

template <typename io_T>
void sum1D(accum40_t *acc40, MLI_PTR(io_T) in, int size) {
    const v2q15_t one_v = {1, 1};

    if (size & 1) {
        *acc40 = fx_a40_dmac_v2q15(*acc40, mli_prv_load_1_sample(in), one_v);
        in += 1;
    }

    for (int i = 0; i < (size >> 1); i++) {
        *acc40 = fx_a40_dmac_v2q15(*acc40, mli_prv_load_2_samples(in), one_v);
        in += 2;
    }
}

template <typename io_T>
static MLI_FORCE_INLINE mli_status mli_krn_softmax_fx_run(const mli_tensor *in, const mli_softmax_cfg* cfg,
        mli_tensor *out) {
    mli_status ret;

    const MLI_PTR(io_T) vec_in = (MLI_PTR(io_T))in->data;
    MLI_OUT_PTR(io_T) vec_out = (MLI_OUT_PTR(io_T))out->data;

    const int el_num = (int)mli_prv_count_elem_num(in);
    int in_frac = (int)in->el_params.fx.frac_bits;

    // look for max & min values
    v2q15_t max_val = fx_replic_v2q15(mli_prv_load_1_sample(vec_in)[0]);
    v2q15_t min_val = fx_replic_v2q15(mli_prv_load_1_sample(vec_in)[0]);
    if (el_num & 1) {
        // nothing to do, max & min already contain first value
        vec_in++;
    }
    for (int idx = 0; idx < (el_num >> 1); idx++) {
        v2q15_t val = mli_prv_load_2_samples(vec_in);
        max_val = fx_max_v2q15(max_val, val);
        min_val = fx_min_v2q15(min_val, val);
        vec_in += 2;
    }
    max_val = fx_replic_v2q15(MAX(max_val[0], max_val[1]));
    min_val = fx_replic_v2q15(MIN(min_val[0], min_val[1]));

    // reset data pointers
    vec_in = (MLI_PTR(io_T))in->data;
    vec_out = (MLI_OUT_PTR(io_T))out->data;

    // subtract maximum from each element,
    // free one more bit if saturation is expected
    int biased_min = (int)min_val[0] - (int)max_val[0];
    int min_limit = -(1 << (sizeof(io_T) * 8 - 1));

    if (biased_min < min_limit) {
        v2q15_t unit = {1, 1};
        max_val = fx_asr_v2q15(max_val, unit);

        in_frac -= 1;
        if (el_num & 1) {
            mli_prv_store_1_sample(vec_out, fx_sub_v2q15(fx_asr_v2q15(mli_prv_load_1_sample(vec_in), unit), max_val));
            vec_in += 1;
            vec_out += 1;
        }
        for (int idx = 0; idx < (el_num >> 1); idx++) {
            mli_prv_store_2_samples(vec_out, fx_sub_v2q15(fx_asr_v2q15(mli_prv_load_2_samples(vec_in), unit), max_val));
            vec_in += 2;
            vec_out += 2;
        }
    } else {
        if (el_num & 1) {
            mli_prv_store_1_sample(vec_out, fx_sub_v2q15(mli_prv_load_1_sample(vec_in), max_val));
            vec_in += 1;
            vec_out += 1;
        }
        for (int idx = 0; idx < (el_num >> 1); idx++) {
            mli_prv_store_2_samples(vec_out, fx_sub_v2q15(mli_prv_load_2_samples(vec_in), max_val));
            vec_in += 2;
            vec_out += 2;
        }
    }

    // reset data pointers
    vec_in = (MLI_PTR(io_T))in->data;
    vec_out = (MLI_OUT_PTR(io_T))out->data;

    mli_prv_activation_lut_fx16(vec_out, vec_out, &expneg_lut_fx16, in_frac, el_num);

    // accumulate and calculate reciprocal
    accum40_t sum_acc = fx_create_a40(0, 0);
    sum1D(&sum_acc, vec_out, el_num);
    int sum_exp = fx_norm_a40(sum_acc) + 1;
    q15_t sum_mnt = fx_q15_cast_nf_asl_rnd_a40(sum_acc, sum_exp);
    // sum_mnt is normalized (that is inside [0.5, 1) range)
    // so we use Q30(0.5) as a dividend to get Q15 result inside (0.5, 1)
    // saturation prevents it from reaching 1
    v2q15_t sum_recip = fx_replic_v2q15((q15_t)fx_sat_q31((1L << 29) / sum_mnt, 16));

    // final result: normalizing
    if (el_num & 1) {
        v2accum40_t tmp_acc = fx_v2a40_mpy_nf_v2q15(sum_recip, mli_prv_load_1_sample(vec_out));
        mli_prv_store_1_sample(vec_out, fx_v2q15_cast_nf_asl_rnd_v2a40(tmp_acc, sum_exp - 30 + sizeof(io_T) * 8));
        vec_out += 1;
    }
    for (int idx = 0; idx < (el_num >> 1); idx++) {
        v2accum40_t tmp_acc = fx_v2a40_mpy_nf_v2q15(sum_recip, mli_prv_load_2_samples(vec_out));
        mli_prv_store_2_samples(vec_out, fx_v2q15_cast_nf_asl_rnd_v2a40(tmp_acc, sum_exp - 30 + sizeof(io_T) * 8));
        vec_out += 2;
    }

    ret = mli_prv_copy_tensor_format(in, out);
    out->el_params.fx.frac_bits = (sizeof(vec_out[0]) * 8) - kTransfFuncIntBits - 1;
}

} // namespace dsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_SOFTMAX_DSP_H_
