/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_AUX_CALC_H_
#define _MLI_PRV_AUX_CALC_H_

#include <assert.h>

#include "mli_check.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math_macros.h"
#include "mli_types.h"
#include "mli_private_types.h"
#include <arc/arc_intrinsics.h>
 
 /**
 * @brief Structure with compensation values to define valid area for 2d calculations.
 *
 * How much rows/columns need to be skipped from each side to leave only valid area of filter or input
 */
struct mli_compensations {
    int left;
    int right;
    int top;
    int bottom;
};

//====================================================================================================
// Step in linear array to get next column (array is considered as n-dimensional with particular layout)
//====================================================================================================
template <mli_layout_type layout>
inline int mli_prv_column_step(int height = 1, int width = 1, int channels = 1, int filters = 1) {
    int column_step = 1;
    if (layout == LAYOUT_HWC) {
        column_step = channels;
    } else if (layout == LAYOUT_HWCN) {
        column_step = channels * filters;
    }
    return column_step;
}

//====================================================================================================
// Step in linear array to get next row (array is considered as n-dimensional with particular layout)
//====================================================================================================
template <mli_layout_type layout>
inline int mli_prv_row_step(int height = 1, int width = 1, int channels = 1, int filters = 1) {
    int row_step = 1;
    if (layout == LAYOUT_HWC) {
        row_step = width * channels;
    } else if (layout == LAYOUT_HWCN) {
        row_step = width * channels * filters;
    }
    return row_step;
}

//====================================================================================================
// Compensation values definition: 
// How much rows/columns need to be skipped from each side to left only valid area of filter or input
//====================================================================================================
inline mli_compensations mli_prv_valid_area_compensations(int out_h_idx, int out_w_idx, 
                                                              int in_height, int in_width,
                                                              int kernel_height, int kernel_width,
                                                              int stride_h, int stride_w,
                                                              int pad_left, int pad_top) {
    mli_compensations comp;
    comp.left   = -MIN((out_w_idx * stride_w)- pad_left, 0);
    comp.right  = -MIN(in_width - ((out_w_idx * stride_w)- pad_left + kernel_width), 0);
    comp.top    = -MIN((out_h_idx * stride_h)- pad_top, 0);
    comp.bottom = -MIN(in_height - ((out_h_idx * stride_h)- pad_top + kernel_height), 0);

    return comp;
}

//====================================================================================================
// Element index definition in linear array which is considered as n-dimensional with particular layout
//====================================================================================================
template <mli_layout_type layout>
inline int mli_prv_calc_index(int height = 1, int width = 1, int channels = 1, int filters = 1,
                                      int h_idx = 0, int w_idx = 0, int c_idx = 0, int f_idx = 0) {
    int result_idx = 0; // starting point
    if (layout == LAYOUT_HWC) {
        result_idx += f_idx * height * width * channels + // move to filter
                      h_idx * width * channels +          // move to row
                      w_idx * channels +                  // move to column
                      c_idx;                              // move to channel
    } else if (layout == LAYOUT_HWCN) {
        result_idx += h_idx * width * channels * filters + // move to row
                      w_idx * channels * filters +         // move to column
                      c_idx * filters +                    // move to channel
                      f_idx;                               // move to filter
    }
    return result_idx;
}


#endif //_MLI_PRV_AUX_CALC_H_