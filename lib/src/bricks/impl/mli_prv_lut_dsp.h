/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_LUT_DSP_H_
#define _MLI_PRV_LUT_DSP_H_

#include "mli_config.h"
#include "mli_prv_lut_decl.h"
#include "mli_math.h"
#include "mli_prv_quant.h"

#include "mli_prv_dsp.h"
#include "mli_prv_load_store.h"

namespace mli {
namespace krn {
namespace dsp {

template <typename io_T, bool convert>
static void activation_lut(
        const MLI_PTR(io_T) in,
        MLI_OUT_PTR(io_T) out,
        const mli_lut *lut,
        int8_t scale_frac_bits,
        int length,
        int scale,
        int16_t zero_point) {

    MLI_ASSERT(scale_frac_bits >= -1);  // -1 may be required by softmax
    MLI_ASSERT(length >= 0);
    MLI_ASSERT(lut->frac_bits >= 0);
    MLI_ASSERT(lut->length >= 0);

    int8_t in_frac_bits;
    int32_t scale_fx = 0;

    if (convert) {
        /* Calculating Scaling Factor to transform SA8 to Qmn (FX16) */
        int lut_int_bits_fx8 = kMaxFracBitsFx8 - lut->frac_bits;
        int frac_bits_fx16 = kMaxFracBitsFx16 - lut_int_bits_fx8;
        scale_fx = mli_math_acc_ashift_fx<int32_t>(scale, ((int32_t) scale_frac_bits - frac_bits_fx16));
        in_frac_bits = frac_bits_fx16;
    } else {
        in_frac_bits = scale_frac_bits;
    }

    int shift_in = in_frac_bits - lut->frac_bits;
    int16_t *lut_data = (int16_t *)lut->data;
    // if shift amount is too high, preshift argument itself and
    // limit shift amount to prevent overflows
    int preshift_in = MAX(shift_in - (int)kMaxFracBitsFx16, 0);
    shift_in = MIN(shift_in, (int)kMaxFracBitsFx16);

    v2q15_t offset = mli_prv_init_v(lut->offset);
    v2q15_t lower = mli_prv_init_v(0);

    if (shift_in > 0) {
        // input data is more precise than LUT
        v2q15_t mask = mli_prv_init_v((1 << shift_in) - 1);
        v2q15_t upper = mli_prv_init_v(lut->length - 2);

        if (length & 1) {
            v2q15_t x = mli_prv_load_1_sample(in);
            if (convert) {
                x = mli_prv_convert_sa8_fx16<v2q15_t, v2q15_t>(x, zero_point, scale_fx);
            }
            x = mli_math_acc_ashift_fx(x, preshift_in);
            v2q15_t lut_idx = mli_math_add_fx(mli_math_acc_ashift_fx(x, shift_in), offset);
            lut_idx = mli_math_bound_range_fx(lut_idx, lower, upper);
            // perform linear interpolation
            v2q15_t frac = x & mask;
            v2q15_t res = mli_prv_init_v(lut_data[lut_idx[0]], lut_data[lut_idx[1]]);
            v2q15_t next = mli_prv_init_v(lut_data[lut_idx[0] + 1], lut_data[lut_idx[1] + 1]);
            v2q15_t diff = mli_math_sub_fx(res, next);
            res = mli_math_sub_fx(res, mli_math_acc_cast_fx<v2q15_t, v2accum40_t>(
                                  mli_math_mul_fx<v2q15_t, v2accum40_t>(diff, frac), shift_in - 16));
            if (convert) {
                res = mli_prv_convert_fx16_sa8<v2q15_t, v2q15_t>(
                                  res, kAsymZeroPointFx16, kLutOutFracBits - kAsymScalePowerFx16);
            } else {
                if(sizeof(io_T) == 1) {
                    res = mli_prv_v2q7_cast_rnd_v2q15(res);
                }
            }
            mli_prv_store_1_sample(out, res);
            in += 1;
            out += 1;
        }
        for (int idx = 0; idx < (length >> 1); idx++) {
            v2q15_t x = mli_prv_load_2_samples(in);
            if (convert) {
                x = mli_prv_convert_sa8_fx16<v2q15_t, v2q15_t>(x, zero_point, scale_fx);
            }
            x = mli_math_acc_ashift_fx(x, preshift_in);
            v2q15_t lut_idx = mli_math_add_fx(mli_math_acc_ashift_fx(x, shift_in), offset);
            lut_idx = mli_math_bound_range_fx(lut_idx, lower, upper);
            // perform linear interpolation
            v2q15_t frac = x & mask;
            v2q15_t res = mli_prv_init_v(lut_data[lut_idx[0]], lut_data[lut_idx[1]]);
            v2q15_t next = mli_prv_init_v(lut_data[lut_idx[0] + 1], lut_data[lut_idx[1] + 1]);
            v2q15_t diff = mli_math_sub_fx(res, next);
            res = mli_math_sub_fx(res, mli_math_acc_cast_fx<v2q15_t, v2accum40_t>(
                                  mli_math_mul_fx<v2q15_t, v2accum40_t>(diff, frac), shift_in - 16));
            if (convert) {
                res = mli_prv_convert_fx16_sa8<v2q15_t, v2q15_t>(
                                  res, kAsymZeroPointFx16, kLutOutFracBits - kAsymScalePowerFx16);
            } else {
                if(sizeof(io_T) == 1) {
                    res = mli_prv_v2q7_cast_rnd_v2q15(res);
                }
            }
            mli_prv_store_2_samples(out, res);
            in += 2;
            out += 2;
        }
    } else {
        // input data isn't more precise than LUT
        v2q15_t upper = mli_prv_init_v(lut->length - 1);

        if (length & 1) {
            v2q15_t x = mli_prv_load_1_sample(in);
            if (convert) {
                x = mli_prv_convert_sa8_fx16<v2q15_t, v2q15_t>(x, zero_point, scale_fx);
            }
            v2q15_t lut_idx = mli_math_add_fx(mli_math_acc_ashift_fx(x, shift_in), offset);
            lut_idx = mli_math_bound_range_fx(lut_idx, lower, upper);
            // no interpolation
            v2q15_t res = mli_prv_init_v(lut_data[lut_idx[0]], lut_data[lut_idx[1]]);
            if (convert) {
                res = mli_prv_convert_fx16_sa8<v2q15_t, v2q15_t>(
                                    res, kAsymZeroPointFx16, kLutOutFracBits - kAsymScalePowerFx16);
            } else {
                if(sizeof(io_T) == 1) {
                    res = mli_prv_v2q7_cast_rnd_v2q15(res);
                }
            }
            mli_prv_store_1_sample(out, res);
            in += 1;
            out += 1;
        }
        for (int idx = 0; idx < (length >> 1); idx++) {
            v2q15_t x = mli_prv_load_2_samples(in);
            if (convert) {
                x = mli_prv_convert_sa8_fx16<v2q15_t, v2q15_t>(x, zero_point, scale_fx);
            }
            v2q15_t lut_idx = mli_math_add_fx(mli_math_acc_ashift_fx(x, shift_in), offset);
            lut_idx = mli_math_bound_range_fx(lut_idx, lower, upper);
            // no interpolation
            v2q15_t res = mli_prv_init_v(lut_data[lut_idx[0]], lut_data[lut_idx[1]]);
            if (convert) {
                res = mli_prv_convert_fx16_sa8<v2q15_t, v2q15_t>(
                                    res, kAsymZeroPointFx16, kLutOutFracBits - kAsymScalePowerFx16);
            } else {
                if(sizeof(io_T) == 1) {
                    res = mli_prv_v2q7_cast_rnd_v2q15(res);
                }
            }
            mli_prv_store_2_samples(out, res);
            in += 2;
            out += 2;
        }
    }
}

} // namespace dsp
} // namespace krn
} // namespace mli

#endif // _MLI_PRV_LUT_DSP_H_
