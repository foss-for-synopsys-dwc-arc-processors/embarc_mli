/*
//
// CONFIDENTIAL AND PROPRIETARY INFORMATION
//
// Copyright (c) 2018 Synopsys, Inc. All rights reserved.
// This software and documentation contain confidential and
// proprietary information that is the property of
// Synopsys, Inc. The software and documentation are
// furnished under a license agreement and may be used
// or copied only in accordance with the terms of the license
// agreement. No part of the software and documentation
// may be reproduced, transmitted, or translated, in any
// form or by any means, electronic, mechanical, manual,
// optical, or otherwise, without prior written permission
// of Synopsys, Inc., or as expressly provided by the license agreement.
// Reverse engineering is prohibited, and reproduction,
// disclosure or use without specific written authorization
// of Synopsys Inc. is strictly forbidden.
//
//
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
#else
typedef mli_acc32_t mli_sa8_sa8_sa32_accu_t;
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
    mli::krn::conv2d_prepare_and_run
            <int16_t, int16_t, int16_t, mli_acc40_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL>
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
    mli::krn::conv2d_prepare_and_run
            <int16_t, int8_t, int8_t, mli_acc32_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL>
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
    mli::krn::conv2d_prepare_and_run
            <int8_t, int8_t, int32_t, mli_sa8_sa8_sa32_accu_t, mli::krn::s8asym_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL>
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
    const int fix_k_width = 1;
    const int fix_k_height = 1;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_fx16(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli::krn::conv2d_prepare_and_run
            <int16_t, int16_t, int16_t, mli_acc40_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL>
            (in, weights, bias, cfg, out, fix_k_width, fix_k_height);
    return ret;
}

mli_status mli_krn_conv2d_hwcn_fx16_fx8_fx8_k1x1(
    const mli_tensor* in,
    const mli_tensor* weights,
    const mli_tensor* bias,
    const mli_conv2d_cfg* cfg,
    mli_tensor* out) {
    const int fix_k_width = 1;
    const int fix_k_height = 1;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_fx16_fx8_fx8(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli::krn::conv2d_prepare_and_run
            <int16_t, int8_t, int8_t, mli_acc32_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL>
            (in, weights, bias, cfg, out, fix_k_width, fix_k_height);
    return ret;
}

mli_status mli_krn_conv2d_hwcn_sa8_sa8_sa32_k1x1(
    const mli_tensor* in,
    const mli_tensor* weights,
    const mli_tensor* bias,
    const mli_conv2d_cfg* cfg,
    mli_tensor* out) {
    const int fix_k_width = 1;
    const int fix_k_height = 1;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli::krn::conv2d_prepare_and_run
            <int8_t, int8_t, int32_t, mli_sa8_sa8_sa32_accu_t, mli::krn::s8asym_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL>
            (in, weights, bias, cfg, out, fix_k_width, fix_k_height);
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
    const int fix_k_width = 3;
    const int fix_k_height = 3;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_fx16(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli::krn::conv2d_prepare_and_run
            <int16_t, int16_t, int16_t, mli_acc40_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL>
            (in, weights, bias, cfg, out, fix_k_width, fix_k_height);
    return ret;
}

mli_status mli_krn_conv2d_hwcn_fx16_fx8_fx8_k3x3(
    const mli_tensor* in,
    const mli_tensor* weights,
    const mli_tensor* bias,
    const mli_conv2d_cfg* cfg,
    mli_tensor* out) {
    const int fix_k_width = 3;
    const int fix_k_height = 3;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_fx16_fx8_fx8(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli::krn::conv2d_prepare_and_run
            <int16_t, int8_t, int8_t, mli_acc32_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL>
            (in, weights, bias, cfg, out, fix_k_width, fix_k_height);
    return ret;
}

mli_status mli_krn_conv2d_hwcn_sa8_sa8_sa32_k3x3(
    const mli_tensor* in,
    const mli_tensor* weights,
    const mli_tensor* bias,
    const mli_conv2d_cfg* cfg,
    mli_tensor* out) {
    const int fix_k_width = 3;
    const int fix_k_height = 3;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli::krn::conv2d_prepare_and_run
            <int8_t, int8_t, int32_t, mli_sa8_sa8_sa32_accu_t, mli::krn::s8asym_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL>
            (in, weights, bias, cfg, out, fix_k_width, fix_k_height);
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
    const int fix_k_width = 5;
    const int fix_k_height = 5;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_fx16(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli::krn::conv2d_prepare_and_run
            <int16_t, int16_t, int16_t, mli_acc40_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL>
            (in, weights, bias, cfg, out, fix_k_width, fix_k_height);
    return ret;
}

mli_status mli_krn_conv2d_hwcn_fx16_fx8_fx8_k5x5(
    const mli_tensor* in,
    const mli_tensor* weights,
    const mli_tensor* bias,
    const mli_conv2d_cfg* cfg,
    mli_tensor* out) {
    const int fix_k_width = 5;
    const int fix_k_height = 5;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_fx16_fx8_fx8(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli::krn::conv2d_prepare_and_run
            <int16_t, int8_t, int8_t, mli_acc32_t, mli::krn::fx_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL>
            (in, weights, bias, cfg, out, fix_k_width, fix_k_height);
    return ret;
}

mli_status mli_krn_conv2d_hwcn_sa8_sa8_sa32_k5x5(
    const mli_tensor* in,
    const mli_tensor* weights,
    const mli_tensor* bias,
    const mli_conv2d_cfg* cfg,
    mli_tensor* out) {
    const int fix_k_width = 5;
    const int fix_k_height = 5;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn_sa8_sa8_sa32(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli::krn::conv2d_prepare_and_run
            <int8_t, int8_t, int32_t, mli_sa8_sa8_sa32_accu_t, mli::krn::s8asym_quant_specific_params, LAYOUT_HWCN, mli::CONV_GENERAL>
            (in, weights, bias, cfg, out, fix_k_width, fix_k_height);
    return ret;
}


#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
} //extern "C"
#endif
