/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "sliding_scan.h"

#include <stdint.h>

#include "face_trigger_model.h"

/* Prototypes of internal functions */
static inline int32_t clip_val(int32_t val, int32_t min, int32_t max);
static void copy_image(const uint8_t*src, uint8_t*dst, int32_t offset_x, int32_t offset_y, int32_t in_resx, int32_t in_resy);

#define LOCAL_FRAQ_BITS (8)
/* 
* Image rescaling to the output resolution by defined factors
*/
void fd_rescale(
        const uint8_t*in_image,
        const int32_t width,
        const int32_t height,
        const int32_t nwidth,
        const int32_t nheight,
        uint8_t*out_image,
        const int32_t nxfactor,
        const int32_t nyfactor) {
    int32_t x,y;
    int32_t ceil_x, ceil_y, floor_x, floor_y;

    int32_t fraction_x,fraction_y,one_min_x,one_min_y;
    int32_t pix[4];//4 pixels for the bilinear interpolation
    int32_t out_image_fix;

    for (y = 0; y < nheight; y++) {//compute new pixels
        for (x = 0; x < nwidth; x++) {
            floor_x = (x*nxfactor) >> LOCAL_FRAQ_BITS;//left pixels of the window
            floor_y = (y*nyfactor) >> LOCAL_FRAQ_BITS;//upper pixels of the window

            ceil_x = floor_x+1;//right pixels of the window
            if (ceil_x >= width) ceil_x=floor_x;//stay in image

            ceil_y = floor_y+1;//bottom pixels of the window
            if (ceil_y >= height) ceil_y=floor_y;

            fraction_x = x*nxfactor-(floor_x << LOCAL_FRAQ_BITS);//strength coefficients
            fraction_y = y*nyfactor-(floor_y << LOCAL_FRAQ_BITS);

            one_min_x = (1 << LOCAL_FRAQ_BITS)-fraction_x;
            one_min_y = (1 << LOCAL_FRAQ_BITS)-fraction_y;

            pix[0] = in_image[floor_y * width + floor_x];//store window
            pix[1] = in_image[floor_y * width + ceil_x];
            pix[2] = in_image[ceil_y * width + floor_x];
            pix[3] = in_image[ceil_y * width + ceil_x];

            //interpolate new pixel and truncate it's integer part
            out_image_fix = one_min_y*(one_min_x*pix[0]+fraction_x*pix[1])+fraction_y*(one_min_x*pix[2]+fraction_x*pix[3]);
            out_image_fix = out_image_fix >> (LOCAL_FRAQ_BITS * 2);
            out_image[nwidth*y+x] = out_image_fix;
        }
    }
}

/* 
* Face detector sliding scan function
*/
void fd_scan(const uint8_t* input, const int32_t x_res, const int32_t y_res,
                        const int32_t* steps_x, const int32_t steps_x_len, 
                        const int32_t* steps_y, const int32_t steps_y_len,
                        offset_coord* coords, int* coords_len) {
    uint8_t face_image[FT_MODEL_IN_POINTS];
    int coord_idx = 0;

    /* Try detect */
    for (int32_t offset_y_idx = 0; offset_y_idx < steps_y_len; offset_y_idx += 1) {
        for (int32_t offset_x_idx = 0; offset_x_idx < steps_x_len; offset_x_idx += 1) {
            /* Extracting a fragment of a scaled image to analyze */
            copy_image(input, face_image, steps_x[offset_x_idx], steps_y[offset_y_idx], x_res, y_res);

            /* ...and processing it */
            const int detected = mli_face_trigger_process(face_image);

            /* any face detected? */
            if (detected > 0) {
                if(coord_idx < *coords_len) {
                    coords[coord_idx].offset_x = steps_x[offset_x_idx];
                    coords[coord_idx].offset_y = steps_y[offset_y_idx];
                }
                ++coord_idx;
            }
        }
    }
    *coords_len = coord_idx;
}

/* 
* Transform offset coordinates to cartesian coordinates (X,Y)
*/
square_coord offset_to_coord_scaled(int32_t offset_x, int32_t offset_y, int32_t in_resx, int32_t in_resy,
                                    int32_t x_scale_rate, int32_t y_scale_rate) {  
    const int32_t q8_half = 1 << (LOCAL_FRAQ_BITS - 1);
    const int32_t frame_res_out_x = (FT_MODEL_IN_DIM_SZ * x_scale_rate + q8_half) >>8;
    const int32_t frame_res_out_y = (FT_MODEL_IN_DIM_SZ * y_scale_rate + q8_half) >>8;
    const int32_t img_res_out_x = (in_resx * x_scale_rate + q8_half) >>8;
    const int32_t img_res_out_y = (in_resy * y_scale_rate + q8_half) >>8;
    offset_x = (offset_x * x_scale_rate + q8_half) >>8;
    offset_y = (offset_y * y_scale_rate + q8_half) >>8;
    
    square_coord result = {
        .top_left_x = clip_val(((in_resx - frame_res_out_x)/2) + offset_x, 0, img_res_out_x),
        .top_left_y = clip_val(((in_resy - frame_res_out_y)/2) + offset_y, 0, img_res_out_y)
    };
    result.bot_right_x = clip_val(result.top_left_x + frame_res_out_x, 0, img_res_out_x);
    result.bot_right_y = clip_val(result.top_left_y + frame_res_out_y, 0, img_res_out_y);
    return result;
}

/* 
* Copy image from input according to face trigger size requirements
*/
static void copy_image(const uint8_t*src, uint8_t*dst, int32_t offset_x, int32_t offset_y, int32_t in_resx, int32_t in_resy) {
    int x, y;
    const int skip_y = ((in_resy - FT_MODEL_IN_DIM_SZ)/2) + offset_y;
    const int skip_x1 = ((in_resx - FT_MODEL_IN_DIM_SZ)/2) + offset_x;
    const int skip_x2 = in_resx - skip_x1 - FT_MODEL_IN_DIM_SZ;

    src += skip_y * in_resx;
    for (y = 0; y < FT_MODEL_IN_DIM_SZ; y++) {
        src += skip_x1;
        for (x = 0; x < FT_MODEL_IN_DIM_SZ; x++) {
            *dst++ = *src++;
        }
        src += skip_x2;
    }
}

/* 
* Clip value to certain range
*/
static inline int32_t clip_val(int32_t val, int32_t min, int32_t max) {
    val = (min > val)? min: val;
    return (max < val)? max: val;
}

