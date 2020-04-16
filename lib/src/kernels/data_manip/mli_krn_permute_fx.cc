/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_krn_data_manip.h"

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma Code(".mli_lib")
/*******************************************************************************
 *
 * Placeholders for kernels (for future optimizations)
 *
 *******************************************************************************/

mli_status mli_krn_permute_fx8(const mli_tensor* in, const mli_permute_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_permute_fx8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::permute_data<int8_t>(in, cfg, out);

    return MLI_STATUS_OK;
}

mli_status mli_krn_permute_fx16(const mli_tensor* in, const mli_permute_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_permute_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::permute_data<int16_t>(in, cfg, out);

    return MLI_STATUS_OK;
}

#pragma code()

#ifdef __cplusplus
}  // extern "C"
#endif