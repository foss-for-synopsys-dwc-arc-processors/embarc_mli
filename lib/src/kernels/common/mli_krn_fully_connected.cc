/*
* Copyright 2019-2021, Synopsys, Inc.
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

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
typedef vNx4accshort_t mli_sa8_sa8_sa32_accu_t;
typedef vNx4accshort_t mli_fx8_accu_t;
typedef vNx2accint_t mli_fx16_accu_t;
typedef vNx4accint_t mli_fx16_fx8_fx8_accu_t;
#else
typedef mli_acc32_t mli_sa8_sa8_sa32_accu_t;
typedef mli_acc32_t mli_fx8_accu_t;
typedef mli_acc40_t mli_fx16_accu_t;
typedef mli_acc32_t mli_fx16_fx8_fx8_accu_t;
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")

mli_status mli_krn_fully_connected_fx8(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_fully_connected_cfg* cfg,
        mli_tensor* out) {

    mli_status ret = MLI_CHECK_STATUS(mli_chk_fully_connected_fx8(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::fully_connected_prepare_and_run
        <int8_t, int8_t, int8_t, mli_fx8_accu_t, mli::krn::fx_quant_specific_params, /*is_bias_ext = */ false>
        (in, weights, bias, cfg, out);

    return MLI_STATUS_OK;
}

mli_status mli_krn_fully_connected_fx16(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_fully_connected_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_fully_connected_fx16(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::fully_connected_prepare_and_run
        <int16_t, int16_t, int16_t, mli_fx16_accu_t, mli::krn::fx_quant_specific_params, /*is_bias_ext = */ false>
        (in, weights, bias, cfg, out);

    return ret;
}

mli_status mli_krn_fully_connected_fx16_fx8_fx8(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_fully_connected_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_fully_connected_fx8w16d(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::fully_connected_prepare_and_run
        <int16_t, int8_t, int8_t, mli_fx16_fx8_fx8_accu_t, mli::krn::fx_quant_specific_params, /*is_bias_ext = */ false>
        (in, weights, bias, cfg, out);

    return ret;
}

mli_status mli_krn_fully_connected_sa8_sa8_sa32(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_fully_connected_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_fully_connected_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::fully_connected_prepare_and_run
        <int8_t, int8_t, int32_t, mli_sa8_sa8_sa32_accu_t, mli::krn::s8asym_quant_specific_params, /*is_bias_ext = */ false>
        (in, weights, bias, cfg, out);
    
    return ret;
}

mli_status mli_krn_fully_connected_sa8_sa8_sa32_ext_bias(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_fully_connected_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_fully_connected_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::fully_connected_prepare_and_run
        <int8_t, int8_t, int32_t, mli_sa8_sa8_sa32_accu_t, mli::krn::s8asym_quant_specific_params, /*is_bias_ext = */ true>
        (in, weights, bias, cfg, out);

    return ret;
}
#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
} //extern "C"
#endif
