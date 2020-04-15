/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _BMP_READER_H_
#define _BMP_READER_H_

#include <stdint.h>

// Image resolution
#define FD_IN_XRES        (80)
#define FD_IN_YRES        (60)

/*
 * Reading input image from BMP file with RGB->gray transformation
 * 
 * Function allocates memory for image and fill it with tranformed to the 
 * gray-scale data from input BMP image.Function can work only with 24bit BMP files. 
 * Image must be of FD_IN_XRES * FD_IN_YRES resolution
 */
uint8_t *bmp_read(const char* filename);


/*
 * Writing input image to BMP file with gray->RGB transformation and mask applying.
 * 
 * Function creates (rewrites) output file.
 * Image and mask must be 8bit images of FD_IN_XRES * FD_IN_YRES resolution
 * Mask is an 8bit image of the same size as input with non-zero values for pixels which 
 * should be output as emphesized (red pixels)
 * 
 */
void bmp_write_gray(const char* filename, const uint8_t *image, const uint8_t *mask);

#endif // _BMP_READER_H_