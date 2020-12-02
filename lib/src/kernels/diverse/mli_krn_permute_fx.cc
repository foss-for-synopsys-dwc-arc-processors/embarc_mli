/*
* Copyright 2019-2020, Synopsys, Inc.
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
#include "mli_krn_permute.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")


/* DEPRECATED */
mli_status mli_krn_permute_fx8(const mli_tensor *in, const mli_permute_cfg* cfg, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_permute_fx8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli_prv_fx_init_dsp_ctrl();

    ret = mli::krn::mli_krn_permute_run<int8_t, false>(in, cfg, out);
    return ret;
}

mli_status mli_krn_permute_fx16(const mli_tensor *in, const mli_permute_cfg* cfg, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_permute_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli_prv_fx_init_dsp_ctrl();

    ret = mli::krn::mli_krn_permute_run<int16_t, false>(in, cfg, out);
    return ret;
}

mli_status mli_krn_permute_sa8(const mli_tensor* in, const mli_permute_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_permute_sa8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli_prv_fx_init_dsp_ctrl();

    ret = mli::krn::mli_krn_permute_run<int8_t, true>(in, cfg, out);
    return ret;
}

#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
}
#endif
