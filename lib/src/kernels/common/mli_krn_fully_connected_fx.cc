/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_krn_fully_connected.h"

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")


 /* DEPRECATED */
mli_status mli_krn_fully_connected_fx8(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_fully_connected_fx8(in, weights, bias, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    fully_connected_prepare_and_run_fx<int8_t, int8_t>(in, weights, bias, out);

    return MLI_STATUS_OK;
}

mli_status mli_krn_fully_connected_fx16(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_fully_connected_cfg cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_fully_connected_fx16(in, weights, bias, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    fully_connected_prepare_and_run_fx<int16_t, int16_t>(in, weights, bias, out);

    return MLI_STATUS_OK;
}

/* DEPRECATED */
mli_status mli_krn_fully_connected_fx8w16d(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_fully_connected_fx8w16d(in, weights, bias, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    fully_connected_prepare_and_run_fx<int16_t, int8_t>(in, weights, bias, out);

    return MLI_STATUS_OK;
}

mli_status mli_krn_fully_connected_fx16_fx8_fx8(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_fully_connected_cfg cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_fully_connected_fx8w16d(in, weights, bias, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    fully_connected_prepare_and_run_fx<int16_t, int8_t>(in, weights, bias, out);

    return MLI_STATUS_OK;
}

mli_status mli_krn_fully_connected_sa8_sa8_sa32(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_fully_connected_cfg cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_fully_connected_sa8_sa8_sa32(in, weights, bias, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    fully_connected_prepare_and_run<int8_t, int8_t, int32_t>(in, weights, bias, out);
    return MLI_STATUS_OK;
}

#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
}
#endif
