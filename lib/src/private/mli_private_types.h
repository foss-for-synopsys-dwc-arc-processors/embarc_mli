/*
* Copyright 2019, Synopsys, Inc.
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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
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

typedef int32_t   mli_acc32_t;
typedef accum40_t mli_acc40_t;

typedef signed char v2i8_t __attribute__((__vector_size__(2)));

#ifdef __cplusplus
}
#endif
#endif                          //_MLI_HELPERS_API_H_
