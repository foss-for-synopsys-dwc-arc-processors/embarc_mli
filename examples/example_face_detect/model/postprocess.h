/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/


#ifndef _POSTPROCESS_H
#define _POSTPROCESS_H

#include <vector>


void parse_detections(float * raw_output, const float * anchors,
                      float * final_detections, int & num_detections);

void print_detections(float * detections, int num_detections, int image_id);

#endif  // _POSTPROCESS_H
