/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_check.h"
#include "mli_krn_l2_normalize.h"
#include <string.h>

using mli::krn::mli_krn_l2_normalize_run;
#ifdef __cplusplus
extern "C" {
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")

mli_status mli_krn_l2_normalize_fx16(const mli_tensor *in,
        const mli_tensor *epsilon,
        const mli_lut *lut,
        const mli_l2_normalize_cfg *cfg, 
        mli_tensor *out) {

    mli_status ret = MLI_CHECK_STATUS(mli_chk_l2_normalize_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    return mli_krn_l2_normalize_run<int16_t>(in, epsilon, cfg, out, lut);
}

mli_status mli_krn_l2_normalize_sa8(const mli_tensor *in, 
        const mli_tensor *epsilon,
        const mli_lut *lut,
        const mli_l2_normalize_cfg *cfg, 
        mli_tensor *out) {

    mli_status ret = MLI_CHECK_STATUS(mli_chk_l2_normalize_sa8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    return mli_krn_l2_normalize_run<int8_t, true>(in, epsilon, cfg, out, lut);
}

int32_t mli_krn_l2_normalize_get_lut_size() {
    return (invsqrt_lut_fx16.length * sizeof(int16_t));
}

mli_status mli_krn_l2_normalize_create_lut(mli_lut *lut) {
    lut->type = invsqrt_lut_fx16.type;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_lut(lut, invsqrt_lut_fx16.data.capacity), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    memcpy(lut->data.mem.pi16, invsqrt_lut_fx16.data.mem.pi16, invsqrt_lut_fx16.length * sizeof(int16_t));
    lut->in_frac_bits = invsqrt_lut_fx16.in_frac_bits;
    lut->length = invsqrt_lut_fx16.length;
    lut->input_offset = invsqrt_lut_fx16.input_offset;
    lut->output_offset = invsqrt_lut_fx16.output_offset;
    lut->out_frac_bits = invsqrt_lut_fx16.out_frac_bits;
    return MLI_STATUS_OK;
}

#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
}
#endif
