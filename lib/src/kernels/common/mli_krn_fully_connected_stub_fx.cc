/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_config.h"
#include "mli_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma Code(".mli_lib")


 /* DEPRECATED */
mli_status mli_krn_fully_connected_fx8(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        mli_tensor* out) {

    return MLI_STATUS_NOT_SUPPORTED;
}

mli_status mli_krn_fully_connected_fx16(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_fully_connected_cfg cfg,
        mli_tensor* out) {

    return MLI_STATUS_NOT_SUPPORTED;
}

/* DEPRECATED */
mli_status mli_krn_fully_connected_fx8w16d(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        mli_tensor* out) {

    return MLI_STATUS_NOT_SUPPORTED;
}

mli_status mli_krn_fully_connected_fx16_fx8_fx8(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_fully_connected_cfg cfg,
        mli_tensor* out) {

    return MLI_STATUS_NOT_SUPPORTED;
}

mli_status mli_krn_fully_connected_sa8_sa8_sa32(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_fully_connected_cfg cfg,
        mli_tensor* out) {

    return MLI_STATUS_NOT_SUPPORTED;
}

#pragma code()

#ifdef __cplusplus
}
#endif
