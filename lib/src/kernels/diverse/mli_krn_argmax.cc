/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_krn_argmax.h"

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")

mli_status mli_krn_argmax_sa8(
    const mli_tensor *in,
    const mli_argmax_cfg *cfg,
    mli_tensor *out) {

    mli_status ret = MLI_CHECK_STATUS(mli_chk_argmax_sa8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    mli::krn::argmax_prepare_and_run<int8_t>(in, cfg, out);
    return MLI_STATUS_OK;
}

mli_status mli_krn_argmax_fx16(
    const mli_tensor *in,
    const mli_argmax_cfg *cfg,
    mli_tensor *out) {

    mli_status ret = MLI_CHECK_STATUS(mli_chk_argmax_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    mli::krn::argmax_prepare_and_run<int16_t>(in, cfg, out);
    return MLI_STATUS_OK;
}

#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
} //extern "C"
#endif