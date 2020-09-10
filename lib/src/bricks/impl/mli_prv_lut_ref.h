/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_LUT_REF_H_
#define _MLI_PRV_LUT_REF_H_

#include "mli_config.h"
#include "mli_prv_lut_decl.h"
#include "mli_math.h"
#include "mli_prv_quant.h"

namespace mli {
namespace krn {
namespace ref {

template <typename io_T, bool convert>
static void activation_lut(
        const struct generic_tensor_private_t<io_T *> *in,
        struct generic_tensor_private_t<io_T *> *out,
        const mli_lut *lut,
        int8_t in_frac_bits,
        struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params) {

    MLI_ASSERT(in_frac_bits >= -1);  // -1 may be required by softmax
    MLI_ASSERT(lut->frac_bits >= 0);
    MLI_ASSERT(lut->length >= 0);
    MLI_ASSERT(MLI_MAX_RANK == 4);

    int32_t scale_fx = 0;

    if (convert) {
        MLI_ASSERT(in_params != nullptr);
        MLI_ASSERT(out_params != nullptr);
        /* Calculating Scaling Factor to transform SA8 to Qmn (FX16) */
        int lut_int_bits_fx8 = kMaxFracBitsFx8 - lut->frac_bits;
        int frac_bits_fx16 = kMaxFracBitsFx16 - lut_int_bits_fx8;
        scale_fx = mli_math_acc_ashift_fx<int32_t>(in_params->scale, ((int32_t) in_params->shift - frac_bits_fx16));
        in_frac_bits = frac_bits_fx16;
    }

    int shift_in = in_frac_bits - lut->frac_bits;
    int16_t *lut_data = (int16_t *)lut->data;
    // if shift amount is too high, preshift argument itself and
    // limit shift amount to prevent overflows
    int preshift_in = mli_math_max_fx(shift_in - (int)kMaxFracBitsFx16, 0);
    shift_in = mli_math_min_fx(shift_in, (int)kMaxFracBitsFx16);

    if (shift_in > 0) {
        // input data is more precise than LUT
        int16_t mask = (1 << shift_in) - 1;

        for (int pos0 = 0; pos0 < in->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in->shape[2]; pos2++) {
                    for (int pos3 = 0; pos3 < in->shape[3]; pos3++) {
                        /* Convert Input SA8 to FX */
                        int16_t input;
                        if (convert) {
                            input = mli_prv_convert_sa8_fx16<io_T, int16_t>(in->ptr[pos(in, pos0, pos1, pos2, pos3)],
                                                 in_params->offset, scale_fx);
                        } else {
                            input = in->ptr[pos(in, pos0, pos1, pos2, pos3)];
                        }

                        int16_t x = input >> preshift_in;
                        int lut_idx = (x >> shift_in) + lut->offset;
                        lut_idx = mli_math_bound_range_fx(lut_idx, 0, lut->length - 2);
                        // perform linear interpolation
                        int16_t frac = x & mask;
                        int16_t res = lut_data[lut_idx];
                        int16_t diff = res - lut_data[lut_idx + 1];
                        res -= mli_math_acc_cast_fx<int16_t, mli_acc32_t>(
                                mli_math_mul_fx<int16_t, mli_acc32_t>(diff, frac), shift_in);

                        if (convert) {
                            MLI_ASSERT(out_params->scale == 1);
                            out->ptr[pos(out, pos0, pos1, pos2, pos3)] =
                                    mli_prv_convert_fx16_sa8<int16_t, io_T>(res, out_params->offset,
                                    kLutOutFracBits - out_params->shift);
                        } else {
                            out->ptr[pos(out, pos0, pos1, pos2, pos3)] =
                                    mli_math_cast_fx<int16_t, io_T>(res, 16 - sizeof(io_T) * 8);
                        }
                    }
                }
            }
        }
    } else {
        // input data isn't more precise than LUT
        for (int pos0 = 0; pos0 < in->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in->shape[2]; pos2++) {
                    for (int pos3 = 0; pos3 < in->shape[3]; pos3++) {
                        /* Convert Input SA8 to FX */
                        int16_t input;
                        if (convert) {
                            input = mli_prv_convert_sa8_fx16<io_T, int16_t>(in->ptr[pos(in, pos0, pos1, pos2, pos3)],
                                    in_params->offset, scale_fx);
                        } else {
                            input = in->ptr[pos(in, pos0, pos1, pos2, pos3)];
                        }
                        int x = (int)input;
                        int lut_idx = (x << -shift_in) + lut->offset;
                        lut_idx = mli_math_bound_range_fx(lut_idx, 0, lut->length - 1);
                        // no interpolation
                        int16_t res = lut_data[lut_idx];

                        if (convert) {
                            MLI_ASSERT(out_params->scale == 1);
                            out->ptr[pos(out, pos0, pos1, pos2, pos3)] =
                                    mli_prv_convert_fx16_sa8<int16_t, io_T>(res, out_params->offset,
                                    kLutOutFracBits - out_params->shift );
                        } else {
                            out->ptr[pos(out, pos0, pos1, pos2, pos3)] =
                                    mli_math_cast_fx<int16_t, io_T>(res, 16 - sizeof(io_T) * 8);
                        }
                    }
                }
            }
        }
    }
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_PRV_LUT_REF_H_
