/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#include "mli_api.h"
#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_krn_convolution.h"

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
typedef vNx4accshort_t mli_sa8_sa8_sa32_accu_t;
typedef vNx2accint_t mli_fx16_accu_t;
typedef vNx4accint_t mli_fx16_fx8_fx8_accu_t;
#elif !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
typedef __v2i32_t mli_sa8_sa8_sa32_accu_t;
typedef __v2i32_t mli_fx16_accu_t;
typedef __v2i32_t mli_fx16_fx8_fx8_accu_t;
#else
typedef mli_acc32_t mli_sa8_sa8_sa32_accu_t;
typedef mli_acc40_t mli_fx16_accu_t;
typedef mli_acc32_t mli_fx16_fx8_fx8_accu_t;
#endif


#pragma MLI_CODE_SECTION_START(".mli_lib")

//========================================================
//
//        MLI 2.0
//
//========================================================

mli_status mli_krn_conv2d_hwcn_fx16(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_conv2d_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_fx16(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::conv2d_prepare_and_run
            <int16_t, int16_t, int16_t, mli_fx16_accu_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL, KRN_SZ_VAR, KRN_SZ_VAR>
            (in, weights, bias, cfg, out);
    return ret;
}

mli_status mli_krn_conv2d_hwcn_fx16_fx8_fx8(
    const mli_tensor* in,
    const mli_tensor* weights,
    const mli_tensor* bias,
    const mli_conv2d_cfg* cfg,
    mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_fx16_fx8_fx8(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::conv2d_prepare_and_run
            <int16_t, int8_t, int8_t, mli_fx16_fx8_fx8_accu_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL, KRN_SZ_VAR, KRN_SZ_VAR>
            (in, weights, bias, cfg, out);
    return ret;
}

mli_status mli_krn_conv2d_hwcn_sa8_sa8_sa32(
    const mli_tensor* in,
    const mli_tensor* weights,
    const mli_tensor* bias,
    const mli_conv2d_cfg* cfg,
    mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::conv2d_prepare_and_run
            <int8_t, int8_t, int32_t, mli_sa8_sa8_sa32_accu_t, mli::krn::s8asym_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL, KRN_SZ_VAR, KRN_SZ_VAR>
            (in, weights, bias, cfg, out);
    return ret;
}

//========================================================
// Specializations for k1x1
//========================================================
mli_status mli_krn_conv2d_hwcn_fx16_k1x1(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_conv2d_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_fx16(in, weights, bias, cfg, out, KRN_SZ_1), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::conv2d_prepare_and_run
            <int16_t, int16_t, int16_t, mli_fx16_accu_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL, KRN_SZ_1, KRN_SZ_1>
            (in, weights, bias, cfg, out);
    return ret;
}

mli_status mli_krn_conv2d_hwcn_fx16_fx8_fx8_k1x1(
    const mli_tensor* in,
    const mli_tensor* weights,
    const mli_tensor* bias,
    const mli_conv2d_cfg* cfg,
    mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_fx16_fx8_fx8(in, weights, bias, cfg, out, KRN_SZ_1), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::conv2d_prepare_and_run
            <int16_t, int8_t, int8_t, mli_fx16_fx8_fx8_accu_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL, KRN_SZ_1, KRN_SZ_1>
            (in, weights, bias, cfg, out);
    return ret;
}

mli_status mli_krn_conv2d_hwcn_sa8_sa8_sa32_k1x1(
    const mli_tensor* in,
    const mli_tensor* weights,
    const mli_tensor* bias,
    const mli_conv2d_cfg* cfg,
    mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_sa8_sa8_sa32(in, weights, bias, cfg, out, KRN_SZ_1), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::conv2d_prepare_and_run
            <int8_t, int8_t, int32_t, mli_sa8_sa8_sa32_accu_t, mli::krn::s8asym_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL, KRN_SZ_1, KRN_SZ_1>
            (in, weights, bias, cfg, out);
    return ret;
}

//========================================================
// Specializations for k3x3
//========================================================
mli_status mli_krn_conv2d_hwcn_fx16_k3x3(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_conv2d_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_fx16(in, weights, bias, cfg, out, KRN_SZ_3), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::conv2d_prepare_and_run
            <int16_t, int16_t, int16_t, mli_fx16_accu_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL, KRN_SZ_3, KRN_SZ_3>
            (in, weights, bias, cfg, out);
    return ret;
}

mli_status mli_krn_conv2d_hwcn_fx16_fx8_fx8_k3x3(
    const mli_tensor* in,
    const mli_tensor* weights,
    const mli_tensor* bias,
    const mli_conv2d_cfg* cfg,
    mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_fx16_fx8_fx8(in, weights, bias, cfg, out, KRN_SZ_3), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::conv2d_prepare_and_run
            <int16_t, int8_t, int8_t, mli_fx16_fx8_fx8_accu_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL, KRN_SZ_3, KRN_SZ_3>
            (in, weights, bias, cfg, out);
    return ret;
}

mli_status mli_krn_conv2d_hwcn_sa8_sa8_sa32_k3x3(
    const mli_tensor* in,
    const mli_tensor* weights,
    const mli_tensor* bias,
    const mli_conv2d_cfg* cfg,
    mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_sa8_sa8_sa32(in, weights, bias, cfg, out, KRN_SZ_3), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::conv2d_prepare_and_run
            <int8_t, int8_t, int32_t, mli_sa8_sa8_sa32_accu_t, mli::krn::s8asym_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL, KRN_SZ_3, KRN_SZ_3>
            (in, weights, bias, cfg, out);
    return ret;
}

//========================================================
// Specializations for k5x5
//========================================================
mli_status mli_krn_conv2d_hwcn_fx16_k5x5(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_conv2d_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_fx16(in, weights, bias, cfg, out, KRN_SZ_5), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::conv2d_prepare_and_run
            <int16_t, int16_t, int16_t, mli_fx16_accu_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL, KRN_SZ_5, KRN_SZ_5>
            (in, weights, bias, cfg, out);
    return ret;
}

mli_status mli_krn_conv2d_hwcn_fx16_fx8_fx8_k5x5(
    const mli_tensor* in,
    const mli_tensor* weights,
    const mli_tensor* bias,
    const mli_conv2d_cfg* cfg,
    mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_fx16_fx8_fx8(in, weights, bias, cfg, out, KRN_SZ_5), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::conv2d_prepare_and_run
            <int16_t, int8_t, int8_t, mli_fx16_fx8_fx8_accu_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL, KRN_SZ_5, KRN_SZ_5>
            (in, weights, bias, cfg, out);
    return ret;
}

mli_status mli_krn_conv2d_hwcn_sa8_sa8_sa32_k5x5(
    const mli_tensor* in,
    const mli_tensor* weights,
    const mli_tensor* bias,
    const mli_conv2d_cfg* cfg,
    mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_sa8_sa8_sa32(in, weights, bias, cfg, out, KRN_SZ_5), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::conv2d_prepare_and_run
            <int8_t, int8_t, int32_t, mli_sa8_sa8_sa32_accu_t, mli::krn::s8asym_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL, KRN_SZ_5, KRN_SZ_5>
            (in, weights, bias, cfg, out);
    return ret;
}


#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
} //extern "C"
#endif
