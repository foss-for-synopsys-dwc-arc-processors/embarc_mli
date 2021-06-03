/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _DEFS_H
#define _DEFS_H

#include <cinttypes>

#include "mli_config.h"


typedef int8_t w_type;
typedef int32_t b_type;
typedef int8_t d_type;
typedef int16_t s_type;
typedef int8_t f_type;

#if (PLATFORM == V2DSP_XY)
    #define _W __xy __attribute__((section(".Xdata")))
    #define _X __xy __attribute__((section(".Xdata")))
    #define _Y __xy __attribute__((section(".Ydata")))
    #define _Z __xy __attribute__((section(".Zdata")))
    #define _PTR __xy
#elif (PLATFORM == V2DSP_VECTOR)
    #define _W __vccm __attribute__((section(".vecmem_data")))
    #define _X __vccm __attribute__((section(".vecmem_data")))
    #define _Y __vccm __attribute__((section(".vecmem_data")))
    #define _Z __vccm __attribute__((section(".vecmem_data")))
    #define _PTR __vccm
#else
    #define _W
    #define _X
    #define _Y
    #define _Z
    #define _PTR
#endif

#define CONV_W_RANK 4
#define CONV_B_RANK 1
#define FC_W_RANK 2
#define FC_B_RANK 1

#define W_EL_TYPE (MLI_EL_SA_8)
#define B_EL_TYPE (MLI_EL_SA_32)
#define D_EL_TYPE (MLI_EL_SA_8)


#endif  // _DEFS_H