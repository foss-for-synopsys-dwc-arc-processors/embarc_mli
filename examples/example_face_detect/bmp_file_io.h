/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _BMP_FILE_IO_H_
#define _BMP_FILE_IO_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/*
* Reading input image from BMP file with BGR->RGB transformation
* 
* Function allocates memory for image and fill it with transformed to the 
* raw 3 channel RGB data from input <filename> BMP file. Function can work only with 24bit BMP files. 
* it also ensures that input image of (<height> * <width>) size
*/
uint8_t *bmp_rgb_read(const char* filename, int32_t width, int32_t height);

/*
* Writing input image to BMP file.
* 
* Function creates (rewrites) <filename> output file.
* <image> must contain 8bit 3 channel (RGB) data of (<width> * <height>) size.
* 
*/
void bmp_rgb_write(const char* filename, const uint8_t* image, int32_t width, int32_t height);

#ifdef __cplusplus
}
#endif

#endif // _BMP_FILE_IO_H_