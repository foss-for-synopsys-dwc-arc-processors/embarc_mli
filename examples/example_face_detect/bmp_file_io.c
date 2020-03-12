/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "bmp_file_io.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Luma coefficients (ITU-R Recommendation BT. 709) for RGB->GS conversion
#define LUMA_COEFF_R (0.2126f)
#define LUMA_COEFF_G (0.7152f)
#define LUMA_COEFF_B (0.0722f)

#pragma pack(1)
typedef struct {
    uint16_t    type;
    uint32_t    size;
    uint16_t    reserved1;
    uint16_t    reserved2;
    uint32_t    offset;
} bmp_header_t;

typedef struct {
   uint32_t     size;
   int32_t      width;
   int32_t      height;
   uint16_t     planes;
   uint16_t     bits;
   uint32_t     compression;
   uint32_t     size_bytes;
   int32_t      x_res;
   int32_t      y_res;
   uint32_t     colors;
   uint32_t     important_colors;
} bmp_info_header_t;

typedef struct {
    uint8_t    b;
    uint8_t    g;
    uint8_t    r;
    uint8_t alpha;
} rgb_pixel_t;
#pragma pack()

/*
 * Reading input image from BMP file with RGB->gray transformation
 */
uint8_t *bmp_read(const char* filename) {
    bmp_header_t bmp_header;
    bmp_info_header_t bmp_info_header;
    rgb_pixel_t rgb_pixel;
    int32_t x, y;
    uint32_t pixel_size_bytes;
    uint8_t *grayscale_image;
    FILE *f;

    uint8_t* image = NULL;

    /* opening the input file */
    f = fopen(filename,"rb");

    if (f == NULL) {
        printf("Failed to open the input file '%s': %s\n", filename, strerror(errno));
        return NULL;
    }

    /* reading BMP headers */
    fread(&bmp_header, sizeof(bmp_header), 1, f);
    fread(&bmp_info_header, sizeof(bmp_info_header), 1, f);

    /* checking file format: FD_IN_XRESxFD_IN_YRES 24bit BMP */
    if (bmp_header.type         != 0x4D42 ||
        bmp_info_header.width   != FD_IN_XRES ||
        bmp_info_header.height  != FD_IN_YRES ||
        bmp_info_header.bits    != 24 ) {
        printf("Invaid input file format: 24bit color BMP is supported\n");
        fclose(f);
        return NULL;
    }

    /* Allocating the buffer and reading input file */
    grayscale_image = malloc(FD_IN_XRES * FD_IN_YRES);
    if (grayscale_image == NULL) {
        printf("Failed to allocate %d bytes to read the input file\n", FD_IN_XRES * FD_IN_YRES);
        fclose(f);
        return NULL;
    }

    image = grayscale_image;
    pixel_size_bytes = bmp_info_header.bits >> 3;

    /* Jump to the beginning of the pixels data */
    //fseek(f, -1 * bmp_info_header.size_bytes, SEEK_END);

    /* Read the data pixel-by-pixel and convert to 8-bit grayscale.
       Note: if "height" is positive, BMP stores the pixel rows "bottom-up" */
    for (y = (FD_IN_YRES - 1); y >= 0; y--) {
        grayscale_image = (uint8_t*)((uint32_t)image + (FD_IN_XRES * y));

        for (x = 0; x < FD_IN_XRES; x++) {
            fread(&rgb_pixel, pixel_size_bytes, 1, f);
            grayscale_image[x] = 
                (uint8_t)(rgb_pixel.r * LUMA_COEFF_R + rgb_pixel.g * LUMA_COEFF_G + rgb_pixel.b * LUMA_COEFF_B);
        }
    }
    fclose(f);
    return image;
}

/*
 * Writing input image to BMP file with gray->RGB transformation and mask applying.
 */
void bmp_write_gray(const char* filename, const uint8_t *image, const uint8_t *mask) {
    const bmp_header_t bmp_header = {
        .type = 0x4D42,
        .size = FD_IN_XRES * FD_IN_YRES * 3 + (sizeof(bmp_header_t) + sizeof(bmp_info_header_t)),
        .offset = sizeof(bmp_header_t) + sizeof(bmp_info_header_t)
    };
    
    const bmp_info_header_t bmp_info_header = {
        .size = sizeof(bmp_info_header_t),
        .width = FD_IN_XRES,
        .height = FD_IN_YRES,
        .planes = 1,
        .bits = 24,
        .compression = 0,
        .size_bytes = FD_IN_XRES * FD_IN_YRES * 3
    };
    
    /* open the input file */
    FILE* f = fopen(filename, "wb");

    if (f == NULL) {
        printf("Failed to open the output file '%s': %s\n", filename, strerror(errno));
        return;
    }

    /* Write BMP headers */
    fwrite((void*)&bmp_header, sizeof(bmp_header), 1, f);
    fwrite((void*)&bmp_info_header, sizeof(bmp_info_header), 1, f);

    /* Write BMP pixel-by-pixel with gray -> rgb transformation */
    const int pixel_size = bmp_info_header.bits >> 3;
    rgb_pixel_t rgb_pixel = {0};
    for (int y_idx = bmp_info_header.height - 1; y_idx >= 0; y_idx--) {
        for (int x_idx = 0; x_idx < bmp_info_header.width; x_idx++) {
            const int idx = y_idx * bmp_info_header.width + x_idx;
            const int8_t val = image[idx];

            /* Put non-zero pixels from mask to input as red pixels*/
            if(mask[idx]) {
                rgb_pixel.r = 255;
                rgb_pixel.g = rgb_pixel.b = 0;
            } else {
                rgb_pixel.r = rgb_pixel.g = rgb_pixel.b = val;
            }
            fwrite((void*)&rgb_pixel, pixel_size, 1, f);
        }
    }
    fclose(f);
}