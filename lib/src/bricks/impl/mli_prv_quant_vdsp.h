/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_QUANT_VDSP_H_
#define _MLI_PRV_QUANT_VDSP_H_

#include "mli_config.h"

// Convert between SA8 and FX16
//=========================================================================
MLI_FORCE_INLINE vNx4short_t mli_prv_convert_sa8_fx16(
        const vNx4short_t in_val,
        const int16_t zero_point,
        const int scale) {
    return (in_val - zero_point) * scale;
}

MLI_FORCE_INLINE vNx4char_t mli_prv_convert_fx16_sa8(
        const vNx4short_t in_val,
        const int16_t zero_point,
        const int scale) {
    vNx4short_t res = mli_math_cast_fx<vNx4short_t, vNx4short_t>(in_val, scale) + zero_point;
    return to_vNx4char_t(mli_math_bound_range_fx(res, INT8_MIN, INT8_MAX));
}

#endif /* _MLI_PRV_QUANT_VDSP_H_ */