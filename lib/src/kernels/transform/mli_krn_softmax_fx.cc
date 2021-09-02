/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_prv_dsp.h"
#include "mli_krn_softmax.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")


/* DEPRECATED */
mli_status mli_krn_softmax_fx8(const mli_tensor *in, const mli_lut *lut, const mli_softmax_cfg *cfg, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_softmax_fx8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();
    mli_prv_fx_init_dsp_ctrl();

    ret = mli::krn::mli_krn_softmax_run<int8_t, false>(in, lut, cfg, out);
    return ret;
}

mli_status mli_krn_softmax_fx16(const mli_tensor *in, const mli_lut *lut, const mli_softmax_cfg *cfg, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_softmax_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();
    mli_prv_fx_init_dsp_ctrl();

    ret = mli::krn::mli_krn_softmax_run<int16_t, false>(in, lut, cfg, out);
    return ret;
}

mli_status mli_krn_softmax_sa8(const mli_tensor *in, const mli_lut *lut, const mli_softmax_cfg *cfg, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_softmax_sa8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();
    mli_prv_fx_init_dsp_ctrl();

    ret = mli::krn::mli_krn_softmax_run<int8_t, true>(in, lut, cfg, out);
    return ret;
}

int32_t mli_krn_softmax_get_lut_size() {
    return (expneg_lut_fx16.length * sizeof(int16_t));
}

mli_status mli_krn_softmax_create_lut(mli_lut *lut) {
    lut->type = expneg_lut_fx16.type;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_lut(lut, expneg_lut_fx16.data.capacity), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    memcpy(lut->data.mem.pi16, expneg_lut_fx16.data.mem.pi16, expneg_lut_fx16.length * sizeof(int16_t));
    lut->in_frac_bits = expneg_lut_fx16.in_frac_bits;
    lut->length = expneg_lut_fx16.length;
    lut->input_offset = expneg_lut_fx16.input_offset;
    lut->output_offset = expneg_lut_fx16.output_offset;
    lut->out_frac_bits = expneg_lut_fx16.out_frac_bits;
    return MLI_STATUS_OK;
}


#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
}
#endif
