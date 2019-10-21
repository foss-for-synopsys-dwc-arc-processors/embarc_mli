/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_check.h"
#include "mli_types.h"
#include "mli_krn_maxpool_hwc.h"

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

mli_status mli_krn_maxpool_hwc_fx16_generic(const mli_tensor* in, const mli_pool_cfg* cfg, mli_tensor* out)
{
    mli_status ret = MLI_CHECK_STATUS(mli_chk_maxpool_hwc_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    // Extract general maxpooling parameters
    int stride_width = cfg->stride_width;
    int stride_height = cfg->stride_height;
    int padding_top = cfg->padding_top;
    int padding_bot = cfg->padding_bottom;
    int padding_left = cfg->padding_left;
    int padding_right = cfg->padding_right;

    // Data pointers
    const MLI_PTR(int16_t) in_ftrs = (const MLI_PTR(int16_t))in->data;
    MLI_OUT_PTR(int16_t) out_ftrs = (MLI_OUT_PTR(int16_t))out->data;

    // Define Data dimensions
    int channels_num = (int)in->shape[FMAP_C_DIM_HWC];

    int kernel_height = cfg->kernel_height;
    int kernel_width = cfg->kernel_width;

    int in_height = (int)in->shape[FMAP_H_DIM_HWC];
    int in_width = (int)in->shape[FMAP_W_DIM_HWC];

    int out_width = CEIL_DIV(in_width + padding_left + padding_right - kernel_width + 1, stride_width);
    int out_height = CEIL_DIV(in_height + padding_top + padding_bot - kernel_height + 1, stride_height);

    mli_krn_maxpool_hwc<int16_t>(
        stride_width, stride_height, padding_top,
        padding_bot, padding_left, padding_right, 
        in_ftrs, out_ftrs, channels_num, kernel_height,
        kernel_width, in_height, in_width, out_width,
        out_height);
    
    // fill output tensor parameters
    out->rank = in->rank;
    out->shape[FMAP_H_DIM_HWC] = (unsigned)out_height;
    out->shape[FMAP_W_DIM_HWC] = (unsigned)out_width;
    out->shape[FMAP_C_DIM_HWC] = (unsigned)channels_num;
    out->el_type = in->el_type;
    out->el_params.fx.frac_bits = in->el_params.fx.frac_bits;
    return MLI_STATUS_OK;
}

mli_status mli_krn_maxpool_hwc_fx16(const mli_tensor* in, const mli_pool_cfg* cfg, mli_tensor* out) {
    return mli_krn_maxpool_hwc_fx16_generic(in, cfg, out);
}

#pragma code()
#ifdef __cplusplus
}
#endif
