/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bmp_file_io.h"
#include "face_trigger_model.h"
#include "sliding_scan.h"

#if defined (__GNUC__) && !defined (__CCAC__)
extern int start_init(void);

// emulation of argv for GNU toolchain
char in_param[256];
#endif

/*
 * Recognition step structure 
 */
typedef struct {
    int32_t do_resize;

    int32_t x_res_out;
    int32_t y_res_out;

    int32_t x_rescale_rate;
    int32_t y_rescale_rate;

    const int32_t* steps_x;
    const int32_t* steps_y;

    int32_t steps_x_len;
    int32_t steps_y_len;
} scan_step_params;

#define ASIZE(a)    ((int32_t)(sizeof(a)/sizeof(a[0])))

/*
 * Offset arrays for sliding scan on image pyramide
 */
static const int32_t scale60_offsety[] = {-11, -5, 0, 5, 11};
static const int32_t scale80_1_offsetx[] = {-18, -9, 0};
static const int32_t scale80_2_offsetx[] = {9, 18};

static const int32_t scale55_offsety[] = {-8, -4, 0, 4, 8};
static const int32_t scale74_1_offsetx[] = {-14, -7, 0};
static const int32_t scale74_2_offsetx[] = {7, 14};

static const int32_t scale51_offsety[] = {-6, -3, 0, 3, 6};
static const int32_t scale68_1_offsetx[] = {-10, -5, 0};
static const int32_t scale68_2_offsetx[] = {5, 10};

static const int32_t scale47_offsety[] = {-3, 0, 3};
static const int32_t scale63_offsetx[] = {-6, -3, 0, 3, 6};

static const int32_t scale43_offsety[] = {-3, 0, 3};
static const int32_t scale58_offsetx[] = {-5, 0, 5};

static const int32_t scale40_offsety[] = {-2, 0, 2};
static const int32_t scale53_offsetx[] = {-4, 0, 4};

/*
 * Face detection sliding scan scheme
 */
#define SC(A, B) ((A<<8)/B)
static const scan_step_params fd_scan_scheme[] = {
    //             res_out              rescale_rate         
    // do_resize    X    Y          X                      Y              steps_x           steps_y           steps_x_len               steps_y_len
    {      0,       80,  60,  SC(FD_IN_XRES, 80), SC(FD_IN_YRES, 60), scale80_1_offsetx, scale60_offsety, ASIZE(scale80_1_offsetx), ASIZE(scale60_offsety)},
    {      0,       80,  60,  SC(FD_IN_XRES, 80), SC(FD_IN_YRES, 60), scale80_2_offsetx, scale60_offsety, ASIZE(scale80_2_offsetx), ASIZE(scale60_offsety)},

    {      1,       74,  55,  SC(FD_IN_XRES, 74), SC(FD_IN_YRES, 55), scale74_1_offsetx, scale55_offsety, ASIZE(scale74_1_offsetx), ASIZE(scale55_offsety)},
    {      0,       74,  55,  SC(FD_IN_XRES, 74), SC(FD_IN_YRES, 55), scale74_2_offsetx, scale55_offsety, ASIZE(scale74_2_offsetx), ASIZE(scale55_offsety)},

    {      1,       68,  51,  SC(FD_IN_XRES, 68), SC(FD_IN_YRES, 51), scale68_1_offsetx, scale51_offsety, ASIZE(scale68_1_offsetx), ASIZE(scale51_offsety)},
    {      0,       68,  51,  SC(FD_IN_XRES, 68), SC(FD_IN_YRES, 51), scale68_2_offsetx, scale51_offsety, ASIZE(scale68_2_offsetx), ASIZE(scale51_offsety)},

    {      1,       63,  47,  SC(FD_IN_XRES, 63), SC(FD_IN_YRES, 47), scale63_offsetx,   scale47_offsety, ASIZE(scale63_offsetx),   ASIZE(scale47_offsety)},

    {      1,       58,  43,  SC(FD_IN_XRES, 58), SC(FD_IN_YRES, 43), scale58_offsetx,   scale43_offsety, ASIZE(scale58_offsetx),   ASIZE(scale43_offsety)},

    {      1,       53,  40,  SC(FD_IN_XRES, 53), SC(FD_IN_YRES, 40), scale53_offsetx,   scale40_offsety, ASIZE(scale53_offsetx),   ASIZE(scale40_offsety)},
};


/* Prototypes */
static inline void draw_a_frame(square_coord coord, const uint8_t val, 
                                uint8_t* output, const int32_t x_res, const int32_t y_res);

/*
 * Main function
 */
int main(int argc, char *argv[]) {
    int ret_code = 0;
    uint8_t* input_image_data = NULL;
    uint8_t* image_mask_buffer = NULL;
    uint8_t* rescaled_image_buffer = NULL;
    uint8_t* mem_for_images = NULL;
    const size_t image_mask_size = FD_IN_XRES * FD_IN_YRES * sizeof(int8_t);
    const size_t rescaled_image_size = FD_IN_XRES * FD_IN_YRES * sizeof(int8_t);
    char* input_path = NULL;

    offset_coord offset_coords[15] = {0};
    const int offset_arr_size = sizeof(offset_coords)/ sizeof(offset_coords[0]);

#if defined (__GNUC__) && !defined (__CCAC__)
//ARC GNU tools
    if (0 != start_init() ){
        //Error init proccesor;
        printf("ERROR: init proccesor\n");
        ret_code = -1;
        goto free_res;
    }
    input_path = in_param;
#else
    /* Checking the command line */
    if (argc != 2) {
        printf("Missing command line argument: input filename\n");
        ret_code = -1;
        goto free_res;
    }
    input_path = argv[1];
#endif
    /* Reading the input data */
    input_image_data = bmp_read(input_path);
    if (input_image_data == NULL) {
        printf("Faild to read input image\n");
#if defined (__GNUC__) && !defined (__CCAC__)
        printf("Please check that you use mdb_com_gnu script with correct setups\n");
#endif
        ret_code = -1;
        goto free_res;
    }

    mem_for_images = malloc(image_mask_size + rescaled_image_size);
    if (mem_for_images == NULL) {
        printf("Failed to allocate %d bytes of RAM\n", image_mask_size + rescaled_image_size);
        ret_code = -1;
        goto free_res;
    }
    rescaled_image_buffer = &mem_for_images[0];
    image_mask_buffer = &mem_for_images[rescaled_image_size];


    /* Initialize resize image buffer */
    memcpy((void*)rescaled_image_buffer, (void*)input_image_data, FD_IN_XRES * FD_IN_XRES);
    memset((void*)image_mask_buffer, 0, FD_IN_XRES * FD_IN_YRES);

     /* Image processing */
    const int total_scan_steps = ASIZE(fd_scan_scheme);
    for (int step = 0; step < total_scan_steps; step++) {
        printf("Detection step #%d\n", step);
        const scan_step_params* params = &fd_scan_scheme[step];

        /* Image rescalinï (if necessairy) */
        if (params->do_resize) {
            fd_rescale(input_image_data, FD_IN_XRES, FD_IN_XRES, params->x_res_out, params->y_res_out, 
                rescaled_image_buffer, params->x_rescale_rate, params->y_rescale_rate);
        }

        /* Scan Image to find faces on it*/
        int found_coords = offset_arr_size;
        fd_scan(rescaled_image_buffer, params->x_res_out, params->y_res_out, 
            params->steps_x, params->steps_x_len, params->steps_y, params->steps_y_len,
            offset_coords, &found_coords);
        if (found_coords > offset_arr_size) {
            printf(" Capacity of coord list is not enough\n");
            found_coords = offset_arr_size;
        }

        /* Draw frames on mask for it's further output into file*/
        for (int i = 0; i < found_coords; i++) {
            square_coord coord = offset_to_coord_scaled(
                offset_coords[i].offset_x, offset_coords[i].offset_y, 
                params->x_res_out, params->y_res_out, params->x_rescale_rate, params->y_rescale_rate);
            draw_a_frame(coord, 255, image_mask_buffer, FD_IN_XRES, FD_IN_YRES);
            printf(" Found a face at ([X:%d, Y:%d]; [X:%d, Y:%d])\n",
                coord.top_left_x, coord.top_left_y, coord.bot_right_x, coord.bot_right_y);
        }
    }
#ifdef PROFILE_ON
    printf("\nTotal summary:\n"
      "\tNN was run %d times\n" 
      "\tNumber of cycles: %d\n", run_num, total_cycles);
#endif
    bmp_write_gray("result.bmp", input_image_data, image_mask_buffer);

    /* Free allocated buffers */
free_res:
    if (input_image_data != NULL) 
        free(input_image_data);
    if (mem_for_images  != NULL) 
        free(mem_for_images);
    return ret_code;
}

 /*
 * Draw a frame defined by corner coordinates on the output 8bit image
 */
static inline void draw_a_frame(square_coord coord, const uint8_t val, 
                                uint8_t* output, const int32_t x_res, const int32_t y_res) {
    int idx = coord.top_left_y * x_res + coord.top_left_x;
    for (int i = 0; i <= coord.bot_right_x - coord.top_left_x; i++, idx++)
        output[idx] = val;
            
    idx = coord.bot_right_y* x_res + coord.top_left_x;
    for (int i = 0; i <= coord.bot_right_x - coord.top_left_x; i++, idx++)
        output[idx] = val;
            
    idx = coord.top_left_y * x_res + coord.top_left_x;
    for (int i = 0; i < coord.bot_right_y - coord.top_left_y; i++, idx+=FD_IN_XRES)
        output[idx] = val;

    idx = coord.top_left_y * x_res + coord.bot_right_x;
    for (int i = 0; i < coord.bot_right_y - coord.top_left_y; i++, idx+=FD_IN_XRES)
        output[idx] = val;
}


