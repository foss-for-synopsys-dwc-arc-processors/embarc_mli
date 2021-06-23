/* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

/**
 *  @author Yaroslav Donskov <yaroslav@synopsys.com>
 */


#ifndef _MODEL_H
#define _MODEL_H

#include <cinttypes>
#include <vector>
#include "mli_types.h"
#include "util.h"


#define IMAGE_SIDE 128
#define NUM_CHANNELS 3
#define IMAGE_SIZE (IMAGE_SIDE * IMAGE_SIDE * NUM_CHANNELS)

#define NUM_ANCHORS 896
#define NUM_ANCHORS_COORDS 4
#define ANCHORS_SIZE (NUM_ANCHORS * NUM_ANCHORS_COORDS)
#define NUM_COORDS 16
#define RESULT_SIZE (NUM_COORDS + 1) 
#define MAX_DETECTIONS 100

#define LAYER_1_KX 5
#define LAYER_1_KY 5
#define LAYER_1_IC 3
#define LAYER_1_OC 24

#define BB_DCONV_KX 3
#define BB_DCONV_KY 3
#define BB_CONV_KX 1
#define BB_CONV_KY 1

#define BB_1_IC 24
#define BB_1_OC 24

#define BB_2_IC 24
#define BB_2_OC 28

#define BB_3_IC 28
#define BB_3_OC 32

#define BB_4_IC 32
#define BB_4_OC 36

#define BB_5_IC 36
#define BB_5_OC 42

#define BB_6_IC 42
#define BB_6_OC 48

#define BB_7_IC 48
#define BB_7_OC 56

#define BB_8_IC 56
#define BB_8_OC 64

#define BB_9_IC 64
#define BB_9_OC 72

#define BB_10_IC 72
#define BB_10_OC 80

#define BB_11_IC 80
#define BB_11_OC 88

#define BB_12_IC 88
#define BB_12_OC 96

#define BB_13_IC 96
#define BB_13_OC 96

#define BB_14_IC 96
#define BB_14_OC 96

#define BB_15_IC 96
#define BB_15_OC 96

#define BB_16_IC 96
#define BB_16_OC 96

#define OCONV_1_IC 96
#define OCONV_1_OC 6

#define OCONV_2_IC 88
#define OCONV_2_OC 2

#define OCONV_3_IC 96
#define OCONV_3_OC 96

#define OCONV_4_IC 88
#define OCONV_4_OC 32

#define NUM_OUTPUT_PARTS 4

extern const float anchors[];

struct DeQuantizeInfo {
    uint32_t size;
    int16_t scale;
    int16_t zp;
    int8_t fraq_bits;

    explicit DeQuantizeInfo(){}

    DeQuantizeInfo(const mli_tensor * tensor) : 
			scale(tensor->el_params.sa.scale.mem.i16),
			fraq_bits(tensor->el_params.sa.scale_frac_bits.mem.i8), 
            zp(tensor->el_params.sa.zero_point.mem.i16){
        size = (uint32_t) mli_hlp_count_elem_num(tensor, 0);
    }		
};

void blazenet(int8_t * input, int8_t * output, DeQuantizeInfo * dequantize_info);

void dequantize(int8_t * quantized, const DeQuantizeInfo * dequantize_info,  float * dequantized);


#endif  // _MODEL_H