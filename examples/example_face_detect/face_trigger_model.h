/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _FACE_TRIGGER_MODEL_H_
#define _FACE_TRIGGER_MODEL_H_

#include "stdint.h"
#include <stdbool.h>
#include <stdio.h>

// Face Trigger Input dimmensions size (same for X and Y)
#define FT_MODEL_IN_DIM_SZ (36)

// Face Trigger Input size
#define FT_MODEL_IN_POINTS (1 * FT_MODEL_IN_DIM_SZ * FT_MODEL_IN_DIM_SZ)


//Setup profiling
//#define PROFILE_ON 

#ifdef PROFILE_ON
extern int total_cycles;
extern int run_num;
static bool print_summary = true;
#endif

// Face Trigger Inference function
//
// Binary classifier which determines whether the input is a face image or not.
//
// params:
// image_buffer -  Input gray-scale image of (FT_MODEL_IN_DIM_SZ x FT_MODEL_IN_DIM_SZ) size
//
// returns: 1 if input is a face image, 0 otherwise.
int mli_face_trigger_process(const uint8_t *image_buffer);


#endif    // _FACE_TRIGGER_MODEL_H_
