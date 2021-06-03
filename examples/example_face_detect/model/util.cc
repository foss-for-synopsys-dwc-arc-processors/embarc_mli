/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "util.h"

#include "defs.h"


float fx2float(int16_t x, uint8_t fraqBits) {
    return (float) x / (float) (1u << fraqBits);
}


int8_t float2sa(float x, float scale, float z) {
    float val = x / scale + z;
    return (int8_t)( val + ((val >= 0) ? 0.5f : -0.5f) );
}

int8_t float2sa_(float x, int16_t scale, int8_t fraq_bits, int16_t zp) {
    float scale_float =  fx2float(scale, fraq_bits);
    return float2sa(x, scale_float, (float) zp);
}

float sa2float(int8_t x, int16_t z, int16_t scale, int8_t fraq_bits) {
    return ((float)x - (float)z) * ( float (scale) / (float)(1u << fraq_bits) );
}
