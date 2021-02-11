/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_check.h"
#include "mli_debug.h"
#include "mli_krn_prelu.h"
#include "mli_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")

mli_status mli_krn_prelu_fx8(const mli_tensor *in,
        const mli_tensor *slope_coeff,
        const mli_prelu_cfg *cfg,
        mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_prelu_fx8(in, slope_coeff, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    return mli::krn::prelu_fx_run<int8_t>(in, slope_coeff, cfg, out);
}

mli_status mli_krn_prelu_fx16(const mli_tensor *in,
        const mli_tensor *slope_coeff,
        const mli_prelu_cfg *cfg,
        mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_prelu_fx16(in, slope_coeff, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    return mli::krn::prelu_fx_run<int16_t>(in, slope_coeff, cfg, out);
}

mli_status mli_krn_prelu_sa8(const mli_tensor *in,
        const mli_tensor *slope_coeff,
        const mli_prelu_cfg *cfg,
        mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_prelu_sa8(in, slope_coeff, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    return mli::krn::prelu_sa8_run(in, slope_coeff, cfg, out);
}
#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
}
#endif
