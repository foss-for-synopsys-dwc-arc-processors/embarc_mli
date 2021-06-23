/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

/**
 *  @author Yaroslav Donskov <yaroslav@synopsys.com>
 */


#ifndef _UTIL_H
#define _UTIL_H

#include <cstdio>
#include <cstdint>
#include <cfloat>
#include "mli_api.h"

float fx2float(int16_t x, uint8_t fraqBits);

int8_t float2sa(float x, float scale, float z);

int8_t float2sa_(float x, int16_t scale, int8_t fraqBits, int16_t zp);

float sa2float(int8_t x, int16_t z, int16_t scale, int8_t fraqBits);

#endif  // _UTIL_H