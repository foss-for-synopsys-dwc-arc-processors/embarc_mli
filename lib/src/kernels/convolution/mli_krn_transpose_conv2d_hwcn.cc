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
#include "mli_krn_transpose_conv.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
typedef vNx4accshort_t mli_sa8_sa8_sa32_accu_t;
typedef vNx2accint_t mli_fx16_accu_t;
typedef vNx4accint_t mli_fx16_fx8_fx8_accu_t;
#else
typedef mli_acc32_t mli_sa8_sa8_sa32_accu_t;
typedef mli_acc40_t mli_fx16_accu_t;
typedef mli_acc32_t mli_fx16_fx8_fx8_accu_t;
#endif

//========================================================
// Generic transpose  convolution 2D
//========================================================
mli_status mli_krn_transpose_conv2d_hwcn_fx16(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_conv2d_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn_fx16(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::transpose_conv2d_prepare_and_run
        <int16_t, int16_t, int16_t, mli_fx16_accu_t, mli::krn::fx_quant_specific_params, KRN_SZ_VAR, KRN_SZ_VAR, STR_VAR>
        (in, weights, bias, cfg, out);
    return ret;
}

mli_status mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_conv2d_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn_fx16_fx8_fx8(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::transpose_conv2d_prepare_and_run
        <int16_t, int8_t, int8_t, mli_fx16_fx8_fx8_accu_t, mli::krn::fx_quant_specific_params, KRN_SZ_VAR, KRN_SZ_VAR, STR_VAR>
        (in, weights, bias, cfg, out);
    return ret;
}

mli_status mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_conv2d_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::transpose_conv2d_prepare_and_run
        <int8_t, int8_t, int32_t, mli_sa8_sa8_sa32_accu_t, mli::krn::s8asym_quant_specific_params, KRN_SZ_VAR, KRN_SZ_VAR, STR_VAR>
        (in, weights, bias, cfg, out);
    return ret;
}


//========================================================
// Specializations for k2x2 and stride 2x2
//========================================================
mli_status mli_krn_transpose_conv2d_hwcn_fx16_k2x2_str2(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_conv2d_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn_fx16(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn_k2x2_str2(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::transpose_conv2d_prepare_and_run
        <int16_t, int16_t, int16_t, mli_fx16_accu_t, mli::krn::fx_quant_specific_params, KRN_SZ_2, KRN_SZ_2, STR_2>
        (in, weights, bias, cfg, out);
    return ret;
}

mli_status mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8_k2x2_str2(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_conv2d_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn_fx16_fx8_fx8(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn_k2x2_str2(in, weights, bias, cfg, out), __func__);    
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::transpose_conv2d_prepare_and_run
        <int16_t, int8_t, int8_t, mli_fx16_fx8_fx8_accu_t, mli::krn::fx_quant_specific_params, KRN_SZ_2, KRN_SZ_2, STR_2>
        (in, weights, bias, cfg, out);
    return ret;
}

mli_status mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32_k2x2_str2(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_conv2d_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn_k2x2_str2(in, weights, bias, cfg, out), __func__);    
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::transpose_conv2d_prepare_and_run
        <int8_t, int8_t, int32_t, mli_sa8_sa8_sa32_accu_t, mli::krn::s8asym_quant_specific_params, KRN_SZ_2, KRN_SZ_2, STR_2>
        (in, weights, bias, cfg, out);
    return ret;
}

//========================================================
// Specializations for k4x4 and stride 2x2
//========================================================
mli_status mli_krn_transpose_conv2d_hwcn_fx16_k4x4_str2(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_conv2d_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn_fx16(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn_k4x4_str2(in, weights, bias, cfg, out), __func__);    
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::transpose_conv2d_prepare_and_run
        <int16_t, int16_t, int16_t, mli_fx16_accu_t, mli::krn::fx_quant_specific_params, KRN_SZ_4, KRN_SZ_4, STR_2>
        (in, weights, bias, cfg, out);
    return ret;
}

mli_status mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8_k4x4_str2(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_conv2d_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn_fx16_fx8_fx8(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn_k4x4_str2(in, weights, bias, cfg, out), __func__);    
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::transpose_conv2d_prepare_and_run
        <int16_t, int8_t, int8_t, mli_fx16_fx8_fx8_accu_t, mli::krn::fx_quant_specific_params, KRN_SZ_4, KRN_SZ_4, STR_2>
        (in, weights, bias, cfg, out);
    return ret;
}

mli_status mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32_k4x4_str2(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_tensor* bias,
        const mli_conv2d_cfg* cfg,
        mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn_k4x4_str2(in, weights, bias, cfg, out), __func__);    
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    mli::krn::transpose_conv2d_prepare_and_run
        <int8_t, int8_t, int32_t, mli_sa8_sa8_sa32_accu_t, mli::krn::s8asym_quant_specific_params, KRN_SZ_4, KRN_SZ_4, STR_2>
        (in, weights, bias, cfg, out);
    return ret;
}

#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
} //extern "C"
#endif
