/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _FACE_DETECT_MODULE_H_
#define _FACE_DETECT_MODULE_H_

#include "stdint.h"
//#include <stdbool.h>
//#include <stdio.h>
//#include <map>

namespace mli_fd {

// (Row; Column) coordinates of a point on the image
struct point {
    int row;
    int clmn;
};

// Detections result container.
struct fd_result {
    void* source_mem;
    uint32_t source_mem_size;
    int detections_num;
};

//=======================================================================
// Face Detect Module 
//
// This module does a face localization on the provided raw RGB image. 
// Invoke method returns a detections object which can be used to extract 
// coordinates of bounding boxes and key points (eyes, nose and etc.)
//
// Usage example:
//      fd_module detector;  
//      uint32_t input_image_size = 0;
//      uint8_t *input_image_data = read_img_from_file_rgb(image_path, &input_image_size);
//      assert(input_image_size == fd_module::kInSize);
//
//      uint32_t detections_mem_size = fd_module::get_single_detection_mem_size() * 5;
//      auto *detections_mem = = new char[detections_mem_size];
//      assert(detections_mem != nullptr);
//      
//      auto detections = detector.invoke(input_image_data, detections_mem, detections_mem_size);
//      for (int det_idx = 0; det_idx < detections.detections_num; ++det_idx) {
//         const auto bbox_bot_right = 
//             fd_module::get_coordinate(detections, det_idx, fd_module::coord_id::kCoordBboxBotRight);
//         const auto bbox_bot_right = 
//             fd_module::get_coordinate(detections, det_idx, fd_module::coord_id::kCoordBboxBotRight);
//         // Do something with coordinates (bbox_bot_right.row; bbox_bot_right.clmn)
//      }
//      delete detections_mem;
//=======================================================================
class fd_module {
public:

    // Expected Input image Width and Height
    static constexpr int kInSpatialDimSize = 128;   

    // Expected Input image number of channels (RGB)
    static constexpr int kInChannels  = 3;

    // Expected Input image size in bytes
    static constexpr int kInSize  = kInSpatialDimSize * kInSpatialDimSize * kInChannels;

    // Coordinate ID to extract from the detection object
    enum class coord_id {
        kCoordBboxTopLeft = 0,
        kCoordBboxBotRight,
        kCoordLeftEye,
        kCoordRightEye,
        kCoordNose,
        kCoordMouth,
        kCoordLeftEar,
        kCoordRightEar,
    };

    // Default constructor to initialize module
    fd_module();

    // Invoke face detection processing on the provided image and return detections container
    // params:
    // [IN] in_image -      pointer to the source image buffer 
    //                      of expected size (kInSpatialDimSize X kInSpatialDimSize X  kInChannels). 
    // [IN] result_memory   - memory for the result object to store list of detections for input image.
    //                        To be able to detect up to N faces on the input image, user must provide
    //                        N * get_single_detection_mem_size() bytes
    // [IN] result_mem_size - size of the result_memory region.
    // return: fd_result container which contains number of detections for provided image. 
    //         Returns invalid object with nullptr and zero detections number in case of invalid input parameters.
    const fd_result invoke(const uint8_t *in_image, void* result_memory, uint32_t result_mem_size) const;

    // Invoke face detection processing on the provided image and return detections object
    // return: Size of the single detection in bytes. It's a minimal size of memory user must provide 
    //          to invoke() method. 
    static uint32_t get_single_detection_mem_size();

    // Get the coordinates of requested point type for a specific detection from the res container.
    // [IN] res      -  detections container returned from the invoke() method
    // [IN] detet_idx   - index of a specific detection with set of key points
    // [IN] point_id    - coord_id identifier of the key point requested for extract.
    // return: point object with coordinates (row; column) of the requested keypoint. 
    //         Returns invalid object with negative coordinates in case of invalid input parameters.
    static point get_coordinate(const fd_result &res, int detect_idx, coord_id point_id);

    // Get a score of specific detection from the res container.
    // [IN] res      -  detections container returned from the invoke() method
    // [IN] detet_idx   - index of a specific detection with set of key points
    // return: score of a specific detection in range of [0 : 1]. Might be interpreted as a confidence
    //         Returns -1 in case of invalid input parameters.
    static float get_score(const fd_result &res, int detect_idx);

private:
    struct fd_detection;
};

} // namespace mli_fd

#endif    // _FACE_DETECT_MODULE_H_
