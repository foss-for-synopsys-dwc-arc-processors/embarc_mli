/* This file is generated, do not edit!
 * edit following template files instead:
 * filetemplate.txt
 * mli_krn_maxpool_hwc_func_body.txt
 */
/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_krn_maxpool_hwc.h"

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_tensor.h"
#include "mli_private_types.h"



#ifdef __cplusplus
extern "C" {
#endif

#pragma Code(".mli_lib")

mli_status mli_krn_maxpool_hwc_fx8_k2x2(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
	return mli_krn_maxpool_hwc<int8_t, MAXPOOL_FIXED_KRN_SIZE_2>(in, cfg, out);
}

mli_status mli_krn_maxpool_hwc_fx8_k3x3(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
	return mli_krn_maxpool_hwc<int8_t, MAXPOOL_FIXED_KRN_SIZE_3>(in, cfg, out);
}

mli_status mli_krn_maxpool_hwc_fx8(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    int kernel_w = cfg->kernel_width;
    int kernel_h = cfg->kernel_height;

    if ((kernel_w == 3) && (kernel_h == 3)) {
        return mli_krn_maxpool_hwc_fx8_k3x3(in, cfg, out);
    } else if ((kernel_w == 2) && (kernel_h == 2)) {
        return mli_krn_maxpool_hwc_fx8_k2x2(in, cfg, out);
    } else {
    	return mli_krn_maxpool_hwc<int8_t, MAXPOOL_NO_FIXED_KRN_SIZE>(in, cfg, out);
    }
}

char * mli_debug_krn_maxpool_hwc_fx8(const mli_tensor * in, const mli_pool_cfg * cfg, mli_tensor * out) {
    int kernel_w = cfg->kernel_width;
    int kernel_h = cfg->kernel_height;

    if ((kernel_w == 3) && (kernel_h == 3)) {
        return (char*)"mli_krn_maxpool_hwc_fx8_k3x3";
    } else if ((kernel_w == 2) && (kernel_h == 2)) {
        return (char*)"mli_krn_maxpool_hwc_fx8_k2x2";
    } else {
        return (char*)"mli_krn_maxpool_hwc_fx8";
    }
}

#pragma code()

#ifdef __cplusplus
    }
#endif
