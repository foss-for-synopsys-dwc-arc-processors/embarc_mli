/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_krn_rnn_dense.h"

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef mli_acc32_t mli_sa8_sa8_sa32_accu_t;
typedef mli_acc40_t mli_fx16_accu_t;
typedef mli_acc32_t mli_fx16_fx8_fx8_accu_t;

#pragma MLI_CODE_SECTION_START(".mli_lib")

//========================================================
//
//        MLI 2.0
//
//========================================================

mli_status mli_krn_rnn_dense_fx16(
        const mli_tensor **in,
        const mli_tensor **weights,
        const mli_tensor *bias,
        const mli_rnn_dense_cfg *cfg,
        mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_rnn_dense_fx16(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::krn::rnn_dense_prepare_and_run<int16_t, int16_t, int16_t, mli_fx16_accu_t, 
        mli::krn::fx_quant_specific_params>(in, weights, bias, cfg, out);

    return ret;
}

mli_status mli_krn_rnn_dense_fx16_fx8_fx8(
        const mli_tensor **in,
        const mli_tensor **weights,
        const mli_tensor *bias,
        const mli_rnn_dense_cfg *cfg,
        mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_rnn_dense_fx16_fx8_fx8(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::krn::rnn_dense_prepare_and_run<int16_t, int8_t, int8_t, mli_fx16_fx8_fx8_accu_t, 
        mli::krn::fx_quant_specific_params>(in, weights, bias, cfg, out);

    return ret;
}

mli_status mli_krn_rnn_dense_sa8_sa8_sa32(
        const mli_tensor **in,
        const mli_tensor **weights,
        const mli_tensor *bias,
        const mli_rnn_dense_cfg *cfg,
        mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_rnn_dense_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::krn::rnn_dense_prepare_and_run<int8_t, int8_t, int32_t, mli_sa8_sa8_sa32_accu_t, 
        mli::krn::s8asym_quant_specific_params>(in, weights, bias, cfg, out);

    return ret;
}

#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
} //extern "C"
#endif
