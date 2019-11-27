/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <assert.h>
#include <stdio.h>

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_krn_reduce_sum2d.h"
#include "mli_krn_avepool_hwc.h"
#include "mli_prv_dsp.h"

#ifdef __FXAPI__
#include <fxarc.h>
#endif

/**
 * Function Short Description
 *
 * \param[in]
 * \param[in/out]
 * \param[out]
 * \result
 *
 * Some Details
 */

#ifdef __cplusplus
extern "C" {
#endif

#pragma Code(".mli_lib")

/******************************************************************************
 *
 * Version & platform description
 * Targets:
 *
 ******************************************************************************/

mli_status mli_krn_avepool_hwc_sa8(const mli_tensor* in, const mli_pool_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_hwc_sa8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    
    mli_prv_fx_init_dsp_ctrl();

    // Extract general maxpooling parameters
    const uint8_t stride_width = cfg->stride_width;
    const uint8_t stride_height = cfg->stride_height;
    const uint8_t padding_top = cfg->padding_top;
    const uint8_t padding_bot = cfg->padding_bottom;
    const uint8_t padding_left = cfg->padding_left;
    const uint8_t padding_right = cfg->padding_right;
    
    // Data pointers
    MLI_PTR(int8_t) in_ftrs = (MLI_PTR(int8_t))(in->data);
    MLI_OUT_PTR(int8_t) out_ftrs = (MLI_OUT_PTR(int8_t))(out->data);

    // Define Data dimensions
    int32_t channels = in->shape[FMAP_C_DIM_HWC];

    const uint8_t kernel_height = cfg->kernel_height;
    const uint8_t kernel_width = cfg->kernel_width;

    int32_t in_height = in->shape[FMAP_H_DIM_HWC];
    int32_t in_width = in->shape[FMAP_W_DIM_HWC];

    const int32_t out_width  = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    const int32_t out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    const int row_beg = 0;
    const int row_end = out_height;
    const int clmn_beg = 0;
    const int clmn_end = out_width;

    // Pooling
    //=======================================================================
    avepool_hwc_krnpad(
        row_beg, row_end,
        clmn_beg, clmn_end,
        in_ftrs, out_ftrs,
        channels, in_width, in_height,
        out_width, out_height,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_top, padding_left, padding_right, padding_bot);

    // fill output tensor parameters
    out->el_type = in->el_type;
    out->rank = in->rank;
    out->el_params.asym = in->el_params.asym;

    out->shape[FMAP_H_DIM_HWC] = out_height;
    out->shape[FMAP_W_DIM_HWC] = out_width;
    out->shape[FMAP_C_DIM_HWC] = channels;
	return MLI_STATUS_OK;
}

#pragma code()

#ifdef __cplusplus
}
#endif
