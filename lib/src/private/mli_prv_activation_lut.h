/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRIVATE_ACTIVATION_LUT_H_
#define _MLI_PRIVATE_ACTIVATION_LUT_H_

#include "mli_config.h" /* for MLI_PTR */
#include "mli_prv_quant.h"
using mli::krn::s8asym_quant_params;

extern const mli_lut tanh_lut_fx16;
extern const mli_lut sigmoid_lut_fx16;
extern const mli_lut expneg_lut_fx16;

#ifdef __cplusplus
extern "C" {
#endif

void mli_prv_activation_lut_fx8(
        const MLI_PTR(int8_t) in,
        MLI_OUT_PTR(int8_t) out,
        const mli_lut *lut,
        int in_frac_bits,
        int length);

void mli_prv_activation_lut_fx16(
        const MLI_PTR(int16_t) in,
        MLI_OUT_PTR(int16_t) out,
        const mli_lut *lut,
        int in_frac_bits,
        int length);

void mli_prv_activation_lut_sa8(
        const MLI_PTR(int8_t) in,
        MLI_OUT_PTR(int8_t) out,
        const mli_lut *lut,
        struct s8asym_quant_params *in_params,
        struct s8asym_quant_params *out_params,
        int length);
#ifdef __cplusplus
}
#endif

#endif  //_MLI_PRIVATE_ACTIVATION_LUT_H_
