/*
 *  Copyright (c) 2019, Synopsys, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1) Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 
 * 2)  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3) Neither the name of the <ORGANIZATION> nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
