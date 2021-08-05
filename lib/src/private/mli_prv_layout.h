/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_LAYOUT_H_
#define _MLI_PRV_LAYOUT_H_

#include "mli_check.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math_macros.h"
#include "mli_types.h"
#include "mli_private_types.h"

/**
 * @brief Data layout type for vision kernels (convolutions/pooloing mostly).
 *
 * Provide information on how to interprete dimensions in input and params tensors:
 * which dimension are height/ width/ channels
 *
 * LAYOUT_CHW - Data is stored in next order: [Channels; Height; Width]
 *              weights in [Filters(out channel); in Channels; Height; Width]
 * LAYOUT_HWC - Data is stored in next order: [Height; Width; Channels]
 *              weights in [Filters(out channel); Height; Width; In Channels]
 * LAYOUT_HWCN - Data is stored as for HWC
 *              weights are [Height; Width; In Channels; Filters(out channel)]
 * LAYOUT_HW1N - Data is stored as for HWC
 *              weights are [Height; Width; Filters(out channel)]
 */
typedef enum {
    LAYOUT_CHW = 0,
    LAYOUT_HWC,
    LAYOUT_HWCN,
    LAYOUT_HW1N
} mli_layout_type;

/**
 * @brief Structure with compensation values to define valid area for 2d calculations.
 *
 * How much rows/columns need to be skipped from each side to leave only valid area of filter or input
 */
struct mli_compensations {
    int in_left;
    int in_right;
    int in_top;
    int in_bottom;
    int kernel_left;
    int kernel_right;
    int kernel_top;
    int kernel_bottom;
};

//====================================================================================================
// Compensation values definition: 
// How much rows/columns need to be skipped from each side to left only valid area of filter or input
//====================================================================================================
MLI_FORCE_INLINE mli_compensations mli_prv_valid_area_compensations(int out_h_idx, int out_w_idx, 
                                                                    int in_height, int in_width,
                                                                    int kernel_height, int kernel_width,
                                                                    int stride_h, int stride_w,
                                                                    int pad_left, int pad_top,
                                                                    int dilation_h, int dilation_w) {
    MLI_ASSERT(dilation_h > 0 && dilation_w > 0 && stride_h > 0 && stride_w > 0);
    MLI_ASSERT(kernel_height > 0 && kernel_width > 0 && in_height > 0 && in_width > 0);
    MLI_ASSERT(pad_left >= 0 && pad_top >= 0 && out_h_idx >= 0 && out_w_idx >= 0);
    mli_compensations comp;
    if (dilation_h == 1 && dilation_w == 1) {
        comp.in_left   = comp.kernel_left   = -MIN((out_w_idx * stride_w)- pad_left, 0);
        comp.in_right  = comp.kernel_right  = -MIN(in_width - ((out_w_idx * stride_w)- pad_left + kernel_width), 0);
        comp.in_top    = comp.kernel_top    = -MIN((out_h_idx * stride_h)- pad_top, 0);
        comp.in_bottom = comp.kernel_bottom = -MIN(in_height - ((out_h_idx * stride_h)- pad_top + kernel_height), 0);
    } else {
        const int effective_kernel_width = (kernel_width - 1) * dilation_w + 1;
        const int effective_kernel_height = (kernel_height - 1) * dilation_h + 1;
        comp.kernel_left   = CEIL_DIV(-MIN((out_w_idx * stride_w)- pad_left, 0), dilation_w);
        comp.kernel_top    = CEIL_DIV(-MIN((out_h_idx * stride_h)- pad_top, 0), dilation_h);
        comp.kernel_right  = CEIL_DIV(-MIN(in_width - ((out_w_idx * stride_w)- pad_left + effective_kernel_width), 0),
                                      dilation_w);
        comp.kernel_bottom = CEIL_DIV(-MIN(in_height - ((out_h_idx * stride_h)- pad_top + effective_kernel_height), 0),
                                      dilation_h);
        comp.in_left = comp.kernel_left * dilation_w;
        comp.in_right  = comp.kernel_right * dilation_w;
        comp.in_top = comp.kernel_top * dilation_h;
        comp.in_bottom = comp.kernel_bottom * dilation_h;
    }
    return comp;
}

#endif //_MLI_PRV_LAYOUT_H_
