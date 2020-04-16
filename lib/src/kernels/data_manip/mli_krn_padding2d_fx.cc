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

mli_status mli_krn_padding2d_chw_fx8(const mli_tensor* in, const mli_padding2d_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_padding2d_chw_fx8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::padding2D_data<int8_t, mli::LAYOUT_CHW>(in, cfg, out);

    return MLI_STATUS_OK;
}

mli_status mli_krn_padding2d_chw_fx16(const mli_tensor* in, const mli_padding2d_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_padding2d_chw_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::padding2D_data<int16_t, mli::LAYOUT_CHW>(in, cfg, out);

    return MLI_STATUS_OK;
}

mli_status mli_krn_padding2d_hwc_fx8(const mli_tensor* in, const mli_padding2d_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_padding2d_hwc_fx8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::padding2D_data<int8_t, mli::LAYOUT_HWC>(in, cfg, out);

    return MLI_STATUS_OK;
}

mli_status mli_krn_padding2d_hwc_fx16(const mli_tensor* in, const mli_padding2d_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_padding2d_hwc_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::padding2D_data<int16_t, mli::LAYOUT_HWC>(in, cfg, out);

    return MLI_STATUS_OK;
}

#pragma code()

#ifdef __cplusplus
}  // extern "C"
#endif