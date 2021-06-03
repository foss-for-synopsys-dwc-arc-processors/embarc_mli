/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "face_detect_module.h"

//#include "mli_api.h"

namespace mli_fd {

// fd_result object sanity check.
//===============================================================================================
static inline bool is_fd_result_valid(const fd_result& res) {
    return (res.detections_num > 0 && res.source_mem != nullptr
        && res.source_mem_size >= res.detections_num * fd_module::get_single_detection_mem_size());
}

//===============================================================================================
//
// Methods and data of the Module
//
//===============================================================================================

// Privat structure with detections result.
//===============================================================================================
struct fd_module::fd_detection {
    point coords[8];
    float score;
};


// Default constructor (do nothing at the moment)
//===============================================================================================
fd_module::fd_module() {};


// Get a detection size
//===============================================================================================
uint32_t fd_module::get_single_detection_mem_size() {
    return sizeof(fd_detection);
}


// Main inference function (not so smart at the moment)
//===============================================================================================
const fd_result fd_module::invoke(const uint8_t* in_image, void* result_memory, uint32_t result_mem_size) const {
    fd_result out {nullptr, 0, 0};
    const int detections_mock = 3;
    const uint32_t required_size = get_single_detection_mem_size() * detections_mock;
    if (result_mem_size >= required_size) {
        out.detections_num = 3;
        auto detections = static_cast<fd_detection *>(result_memory);

        // Mock Data Original
        //detections[0] = { {{40, 57}, {59, 76}, {62, 45}, {70, 45}, {65, 50}, {65, 53}, {59, 46}, {75, 47}}, 0.963 };
        //detections[1] = { {{32, 20}, {51, 39}, {28, 36}, {36, 37}, {33, 41}, {31, 45}, {21, 37}, {37, 40}}, 0.901};
        //detections[2] = { {{30, 94}, {50, 113}, {98, 35}, {105, 36}, {100, 39}, {101, 43}, {96,38}, {112, 38}}, 0.832 };

        // Mock Data (transposed keypoints)
        detections[0] = { {{40, 57}, {59, 76}, {45, 62}, {45, 70}, {50, 65}, {53, 65}, {46, 59}, {47, 75}}, 0.963 };
        detections[1] = { {{32, 20}, {51, 39}, {36, 28}, {37, 36}, {41, 33}, {45, 31}, {37, 21}, {40, 37}}, 0.901};
        detections[2] = { {{30, 94}, {50, 113}, {35, 98}, {36, 105}, {39, 100}, {43, 101}, {38, 96}, {38, 112}}, 0.832 };

        out.source_mem = static_cast<void *>(detections);
        out.source_mem_size = required_size;
    }
    return out;
}


// Extract coordinates from detection
//===============================================================================================
point fd_module::get_coordinate(const fd_result& res, int detect_idx, coord_id point_id) {
    const auto point_idx = static_cast<int>(point_id);
    point out = { -1, -1 };
    if (is_fd_result_valid(res) && res.detections_num > detect_idx ) {
        auto detections = static_cast<fd_detection *>(res.source_mem);
        out = detections[detect_idx].coords[point_idx];
    }
    return out;
};


// Extract score from detection
//===============================================================================================
float fd_module::get_score(const fd_result& res, int detect_idx) {
    float out = -1.f;
    if (is_fd_result_valid(res) && res.detections_num > detect_idx ) {
        auto detections = static_cast<fd_detection *>(res.source_mem);
        out = detections[detect_idx].score;
    }
    return out;
};

} // namespace mli_fd
