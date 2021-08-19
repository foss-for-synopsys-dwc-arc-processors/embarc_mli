/*
* Copyright 2019-2021, Synopsys, Inc.
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

#ifdef __cplusplus
extern "C" {
#endif

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
uint8_t *bmp_rgb_read(const char* filename, int32_t width, int32_t height) {
    const uint32_t pixel_size_bytes = 3;
    bmp_header_t bmp_header;
    bmp_info_header_t bmp_info_header;
    rgb_pixel_t rgb_pixel;
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

    /* checking file format: width X height 24bit BMP */
    if (bmp_header.type         != 0x4D42 ||
            bmp_info_header.width   != width ||
            bmp_info_header.height  != height ||
            bmp_info_header.bits    != 8 * pixel_size_bytes) {
        printf("Invaid input file format: %dx%d 24bit color BMP is expected\n", height, width);
        fclose(f);
        return NULL;
    }

    /* Allocating the buffer and reading input file */
    size_t img_size = (size_t)width * height * pixel_size_bytes;
    image = malloc(img_size);
    if (image == NULL) {
        printf("Failed to allocate %d bytes to read the input file\n", width * height);
        fclose(f);
        return NULL;
    }

    /* Read the data pixel-by-pixel and convert BGR->RGB.
    Note: if "height" is positive, BMP stores the pixel rows "bottom-up" */
    for (int y = (height - 1); y >= 0; y--) {
        uint8_t *image_row = &image[width * y * pixel_size_bytes];

        for (int x = 0; x < width; x++) {
            const int idx = x * pixel_size_bytes;
            fread(&rgb_pixel, pixel_size_bytes, 1, f);
            image_row[idx + 0] = rgb_pixel.r;
            image_row[idx + 1] = rgb_pixel.g;
            image_row[idx + 2] = rgb_pixel.b;
        }
    }
    fclose(f);
    return image;
}


/*
* Writing input image to BMP file with required transformations.
*/
void bmp_rgb_write(const char* filename, const uint8_t* image, int32_t width, int32_t height) {
    const int32_t pixel_size_bytes = 3;
    const bmp_header_t bmp_header = {
        .type =  0x4D42,
        .size =  width * height * pixel_size_bytes + (sizeof(bmp_header_t) + sizeof(bmp_info_header_t)),
        .offset =  sizeof(bmp_header_t) + sizeof(bmp_info_header_t)
    };

    const bmp_info_header_t bmp_info_header = {
        .size =  sizeof(bmp_info_header_t),
        .width =  width,
        .height =  height,
        .planes =  1,
        .bits =  8 * pixel_size_bytes,
        .compression =  0,
        .size_bytes =  width * height * pixel_size_bytes
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

    /* Write BMP pixel-by-pixel with rgb->bgr transformation in bottom-up rows order*/
    rgb_pixel_t rgb_pixel = {0};
    for (int y_idx = bmp_info_header.height - 1; y_idx >= 0; y_idx--) {
        for (int x_idx = 0; x_idx < bmp_info_header.width; x_idx++) {
            const int idx = (y_idx * bmp_info_header.width + x_idx) * pixel_size_bytes;
            rgb_pixel.r = image[idx];
            rgb_pixel.g = image[idx + 1];
            rgb_pixel.b = image[idx + 2];

            fwrite((void*)&rgb_pixel, pixel_size_bytes, 1, f);
        }
    }
    fclose(f);
}
#ifdef __cplusplus
}
#endif
