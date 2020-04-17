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

mli_status mli_krn_concat_fx8(const mli_tensor** inputs, const mli_concat_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_concat_fx8(inputs, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::concat_data<int8_t>(inputs, cfg, out);

    return MLI_STATUS_OK;
}

mli_status mli_krn_concat_fx16(const mli_tensor** inputs, const mli_concat_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_concat_fx16(inputs, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::concat_data<int16_t>(inputs, cfg, out);

    return MLI_STATUS_OK;
}

#pragma code()

#ifdef __cplusplus
}  // extern "C"
#endif