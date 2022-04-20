/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#include "mli_api.h"
// #include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli3_krn_convolution.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
typedef vNx4accshort_t mli_8x8_accu_t;
typedef vNx2accint_t mli_16x16_accu_t;
typedef vNx4accint_t mli_8x16_accu_t;
#else
typedef mli_acc32_t mli_8x8_accu_t;
typedef mli_acc40_t mli_16x16_accu_t;
typedef mli_acc32_t mli_8x16_accu_t;
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")

//========================================================
//
//        MLI 3.0 Bare semantic functions
//
//========================================================

mli_status mli3_krn_depthwise_conv2d_hwcn_w8_i8_o32(
        const mli_tensor* in,
        const mli_tensor* weights,
        const mli_conv2d_cfg* cfg,
        mli_tensor* out) {
     mli_status ret = MLI_STATUS_OK;
//     mli_status ret =  MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_hwcn_fx16(in, weights, bias, cfg, out), __func__);
//     if (ret != MLI_STATUS_OK) return ret;
//     MLI_PRINT_COMPILE_OPTIONS();

    snps_arc::metaware::mli::krn::conv2d_prepare_and_run
            <int8_t, int8_t, int32_t, mli_8x8_accu_t, mli::krn::s8asym_quant_specific_params, LAYOUT_HW1N, 
            snps_arc::metaware::mli::CONV_DEPTHWISE, KRN_SZ_VAR, KRN_SZ_VAR>
            (in, weights, cfg, out);
    return ret;
}

#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
} //extern "C"
#endif
