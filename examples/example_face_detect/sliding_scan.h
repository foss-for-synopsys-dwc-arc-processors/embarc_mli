/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _RESCALE_H_
#define _RESCALE_H_

#include <stdint.h>

typedef struct {
    int32_t top_left_x;
    int32_t top_left_y;
    int32_t bot_right_x;
    int32_t bot_right_y;
} square_coord;

typedef struct {
    int32_t offset_x;
    int32_t offset_y;
} offset_coord;


/* 
* Image rescaling to the output resolution by defined factors
*/
void fd_rescale(const uint8_t*in_image, const int32_t width, const int32_t height, const int32_t nwidth, 
                const int32_t nheight, uint8_t*out_image, const int32_t nxfactor, const int32_t nyfactor);

/* 
* Face detector sliding scan function
*/
void fd_scan(const uint8_t* input, const int32_t x_res, const int32_t y_res,
             const int32_t* steps_x, const int32_t steps_x_len, 
             const int32_t* steps_y, const int32_t steps_y_len,
             offset_coord* coords, int* coords_len);

/* 
* Transform offset coordinates to cartesian coordinates (X,Y)
*/
square_coord offset_to_coord_scaled(int32_t offset_x, int32_t offset_y, int32_t in_resx, int32_t in_resy,
                                    int32_t x_scale_rate, int32_t y_scale_rate);

#endif // _RESCALE_H_

