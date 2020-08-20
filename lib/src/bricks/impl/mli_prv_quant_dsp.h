/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_QUANT_DSP_H_
#define _MLI_PRV_QUANT_DSP_H_

#include "mli_prv_quant_decl.h"

#include "mli_config.h"
#include "mli_check.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math_macros.h"
#include "mli_private_types.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"
#include "mli_krn_reduce_sum2d.h"

#include <arc/arc_intrinsics.h>
#include <assert.h>

namespace mli {
namespace krn {
namespace dsp {

static const int kPreDivShiftS16 = 14;
static const int kPreDivShiftS32 = 30;

// Convert between SA8 and FX16
//=========================================================================
template<>
MLI_FORCE_INLINE v2q15_t mli_prv_convert_sa8_fx16(
    const v2q15_t in,
    const int16_t zero_point,
    const int scale) {
    v2q15_t zero_point_v = fx_replic_v2q15(zero_point);
    v2q15_t in_biased_shifted_no_zp = fx_sub_v2q15(in, zero_point_v);
    int16_t res_1 = mli_math_cast_fx<int64_t, int16_t>(mli_math_mul_fx<int32_t, int64_t>((int32_t)in_biased_shifted_no_zp[0], scale), 0);
    int16_t res_2 = mli_math_cast_fx<int64_t, int16_t>(mli_math_mul_fx<int32_t, int64_t>((int32_t)in_biased_shifted_no_zp[1], scale), 0);
    return fx_create_v2q15(res_1, res_2);
}

template<>
MLI_FORCE_INLINE v2q15_t mli_prv_convert_fx16_sa8(
    const v2q15_t in,
    const int16_t zero_point,
    const int scale) {

    mli_acc32_t fx_output32 = (mli_acc32_t) in[0];
    // Converting to float and back to asym8
    mli_acc32_t fx_output32_shifted = mli_math_acc_ashift_fx<mli_acc32_t>(fx_output32, scale) + zero_point;
    int8_t res_1 = mli_math_acc_cast_fx<int8_t, mli_acc32_t>(fx_output32_shifted, 0);
    fx_output32 = (mli_acc32_t) in[1];
    // Converting to float and back to asym8
    fx_output32_shifted = mli_math_acc_ashift_fx<mli_acc32_t>(fx_output32, scale) + zero_point;
    int8_t res_2 = mli_math_acc_cast_fx<int8_t, mli_acc32_t>(fx_output32_shifted, 0);
    return fx_create_v2q15(res_1, res_2);
}

} // namespace dsp
} // namespace krn
} // namespace mli

#endif /* _MLI_PRV_QUANT_DSP_H_ */