/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRIVATE_TYPES_API_H_
#define _MLI_PRIVATE_TYPES_API_H_

#include "mli_config.h"
#include "mli_types.h"

/**
 * Private tensor struct typically used for pooling/conv2d/depthwise_conv2d
 * inputs and outputs.
 */
template <typename T>
struct tensor_private_t {
    T __restrict ptr;
    int32_t width;
    int32_t height;
    int32_t ch;
    int32_t col_mem_stride;
    int32_t row_mem_stride;
    int32_t ch_mem_stride;
};

/**
 * Private weights tensor struct typically used for conv2d/depthwise_conv2d.
 */
template <typename T>
struct conv2d_weights_tensor_private_t {
    T __restrict ptr;
    int32_t kernel_width;
    int32_t kernel_height;
    int32_t in_ch;
    int32_t out_ch;
    int32_t col_mem_stride;
    int32_t row_mem_stride;
    int32_t in_ch_mem_stride;
    int32_t out_ch_mem_stride;
};

/**
 * Private tensor struct typically used for transform functions (i.e. softmax).
 */
template <typename T>
struct generic_tensor_private_t {
    T __restrict ptr;
    int rank;
    int shape[MLI_MAX_RANK];
    int mem_stride[MLI_MAX_RANK];
};

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t row_beg;
    uint32_t row_end;
    uint32_t clmn_beg;
    uint32_t clmn_end;
} rect_t;

/**
 * Lookup table config definition 
 */
typedef struct {
    const void* data;
    mli_element_type type;
    int length;
    int frac_bits;
    int offset;
} mli_lut;

// Value range for applying ReLU 
typedef struct {
    int16_t min;
    int16_t max;
} mli_minmax_t;

#if (PLATFORM == V2DSP) || \
    (PLATFORM == V2DSP_XY) || \
    (PLATFORM == V2DSP_WIDE)
typedef signed char v2i8_t __attribute__((__vector_size__(2)));
#endif

#ifdef __cplusplus
}
#endif
#endif                          //_MLI_HELPERS_API_H_
