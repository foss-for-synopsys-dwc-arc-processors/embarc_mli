/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_check.h"
#include "mli_config.h"
#include "mli_krn_maxpool_hwc.h"

/**
 * Function Short Description
 *
 * \param[in]
 * \param[in/out]
 * \param[out]
 * \result
 *
 * Some Details
 */

#ifdef __cplusplus
extern "C" {
#endif

#pragma Code(".mli_lib")

/******************************************************************************
 *
 * Version & platform description
 * Targets:
 *
 ******************************************************************************/

mli_status mli_krn_maxpool_hwc_fx8_generic(const mli_tensor* in, const mli_pool_cfg* cfg, mli_tensor* out)
{
    mli_status ret = MLI_CHECK_STATUS(mli_chk_maxpool_hwc_fx8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli_krn_maxpool_hwc<int8_t>(in, cfg, out);
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;
    return MLI_STATUS_OK;
}

mli_status mli_krn_maxpool_hwc_fx8(const mli_tensor* in, const mli_pool_cfg* cfg, mli_tensor* out) {
    return mli_krn_maxpool_hwc_fx8_generic(in, cfg, out);
}

#pragma code()
#ifdef __cplusplus
}
#endif
