/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "face_detect_module.h"

#include <algorithm>
#include <cmath>

#include "tests_aux.h"
#include "model/util.h"
#include "model/model.h"
#include "model/postprocess.h"


#ifdef _ARC
#include <arc/arc_timer.h>
#else
#include <ctime>
#endif
uint64_t time_basepoint;
static inline void fd_timer_reset() {
#ifdef _ARC
    _timer_default_reset();
    time_basepoint = _timer_default_read();
#else
    time_basepoint = clock();
#endif
}

static inline uint64_t fd_timer_read() {
#ifdef _ARC
    return _timer_default_read() - time_basepoint;
#else
    return clock() - time_basepoint;
#endif
}

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


// Default constructor
//===============================================================================================
fd_module::fd_module()
    : pre_process_ticks(0)
    , model_ticks(0)
    , post_process_ticks(0)
    , total_ticks(0)
{};


// Get a detection size
//===============================================================================================
uint32_t fd_module::get_single_detection_mem_size() {
    return sizeof(fd_detection);
}


// Main inference function (not so smart at the moment)
//===============================================================================================
const fd_result fd_module::invoke(const uint8_t* in_image, void* result_memory, uint32_t result_mem_size) const {
    fd_result out {nullptr, 0, 0};
    // PreProcess
    fd_timer_reset();
    static int8_t quantized_image[kInSize];
    for (int i = 0; i < kInSize; ++i){
        float val = (float) in_image[i] / 127.5f - 1.0f;
        quantized_image[i] = float2sa(val, 0.00784314f, -1.0f);
    }

    pre_process_ticks += fd_timer_read();

    // Model
    fd_timer_reset();
    static int8_t quantized_output[NUM_ANCHORS * RESULT_SIZE];
    DeQuantizeInfo dequantize_info[NUM_OUTPUT_PARTS];
    blazenet(quantized_image, quantized_output, dequantize_info);
    model_ticks += fd_timer_read();

    //Postprocess
    fd_timer_reset();
    static float output[NUM_ANCHORS * RESULT_SIZE];
    dequantize(quantized_output, dequantize_info, output);

    static float detections[MAX_DETECTIONS * RESULT_SIZE];
    int num_detections = 0;
    parse_detections(output, anchors, detections, num_detections);

    int num_detections_out = static_cast<int>(result_mem_size / get_single_detection_mem_size());
    num_detections_out = std::min(num_detections, num_detections_out);

    auto detections_out = static_cast<fd_detection *>(result_memory);
    for (int idx = 0; idx < num_detections_out; idx++){
        float * det = detections + idx * RESULT_SIZE;
        detections_out[idx].score = det[16];
        for (int j = 0; j < 2; j++)
            detections_out[idx].coords[j] = { (int) roundf(det[j * 2] * IMAGE_SIDE), (int) roundf(det[(j * 2) + 1] * IMAGE_SIDE) };
        for (int j = 2; j < 8; j++)
            detections_out[idx].coords[j] = { (int) roundf(det[(j * 2) + 1] * IMAGE_SIDE), (int) roundf(det[(j * 2)] * IMAGE_SIDE) };
    }

    out.detections_num = num_detections_out;
    out.source_mem = static_cast<void *>(detections_out);
    out.source_mem_size = out.detections_num * get_single_detection_mem_size();
    post_process_ticks += fd_timer_read();
    total_ticks = pre_process_ticks + model_ticks + post_process_ticks;
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


// Get a Profiler ticks according to ID.
//===============================================================================================
uint64_t fd_module::get_prof_ticks(prof_tick_id id) {
    switch (id) {
    case mli_fd::fd_module::prof_tick_id::kProfPreProcess:
        return pre_process_ticks;
        break;
    case mli_fd::fd_module::prof_tick_id::kProfModel:
        return model_ticks;
        break;
    case mli_fd::fd_module::prof_tick_id::kProfPostProcess:
        return post_process_ticks;
        break;
    case mli_fd::fd_module::prof_tick_id::kProfTotal:
        return total_ticks;
        break;
    default:
        return 0;
        break;
    }

}

// Reset profiler.
//===============================================================================================
void fd_module::reset_prof_ticks() {
    pre_process_ticks = 0;
    model_ticks = 0;
    post_process_ticks = 0;
    total_ticks = 0;
}
} // namespace mli_fd
