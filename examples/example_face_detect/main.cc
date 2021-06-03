/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <memory>

#include "bmp_file_io.h"
#include "face_detect_module.h"

using mli_fd::fd_module;
using mli_fd::point;

constexpr uint32_t kDetectionsMemBudget = 1024;


// Draw a key point mark defined by coordinates on the output RGB image
//=====================================================================
static inline void draw_a_keypoint_rgb(point coord, uint8_t* output,
                                       const int32_t x_res, const int32_t y_res) {
    const uint8_t val_r = 0xFF;
    const uint8_t val_g = 0x00;
    const uint8_t val_b = 0x00;
    const int channels = 3;

    coord.row = std::min(coord.row, y_res - 1);
    coord.clmn = std::min(coord.clmn, x_res - 1);
    int idx = (coord.row * x_res + coord.clmn) * channels;
    output[idx++] = val_r; output[idx++] = val_g; output[idx++] = val_b;
}


// Draw a frame defined by corner coordinates on the output RGB image
//=====================================================================
static inline void draw_a_frame_rgb(point top_left, point bot_right,
                                    uint8_t* output, const int32_t x_res, const int32_t y_res) {
    const uint8_t val_r = 0xFF;
    const uint8_t val_g = 0x00;
    const uint8_t val_b = 0x00;
    const int channels = 3;

    top_left.row = std::min(top_left.row, y_res - 1);
    top_left.clmn = std::min(top_left.clmn, x_res - 1);
    bot_right.row = std::min(bot_right.row, y_res - 1);
    bot_right.clmn = std::min(bot_right.clmn, x_res - 1);

    int idx = (top_left.row * x_res + top_left.clmn) * channels;
    for (int i = 0; i <= bot_right.clmn - top_left.clmn; i++) {
        output[idx++] = val_r; output[idx++] = val_g; output[idx++] = val_b;
    }

    idx = (bot_right.row * x_res + top_left.clmn) * channels;
    for (int i = 0; i <= bot_right.clmn - top_left.clmn; i++) {
        output[idx++] = val_r; output[idx++] = val_g; output[idx++] = val_b;
    }

    idx = (top_left.row * x_res + top_left.clmn) * channels;
    for (int i = 0; i < bot_right.row - top_left.row; i++, idx += x_res * channels) {
        output[idx + 0] = val_r; output[idx + 1] = val_g; output[idx + 2] = val_b;
    }

    idx = (top_left.row * x_res + bot_right.clmn) * channels;
    for (int i = 0; i < bot_right.row - top_left.row; i++, idx += x_res * channels) {
        output[idx + 0] = val_r; output[idx + 1] = val_g; output[idx + 2] = val_b;
    }
}


//=====================================================================
// Main function
//=====================================================================
int main(int argc, char *argv[]) {
    std::unique_ptr<uint8_t[]> input_image_data;  // empty
    const char* input_path = NULL;

    // Checking the command line
    if (argc != 2) {
        printf("Missing command line argument: input filename\n");
        return -1;
    }
    input_path = argv[1];

    // Reading the input data
    static_assert(fd_module::kInChannels == 3, "kInChannels != 3: RGB input processing is only supported");
    const int img_width = fd_module::kInSpatialDimSize;
    const int img_height = fd_module::kInSpatialDimSize;
    input_image_data.reset(bmp_rgb_read(input_path, img_width, img_width));
    if (!input_image_data) {
        printf("Failed to read input image\n");
        return -1;
    }

    // Invoking detector to write result into statically allocated memory
    const uint32_t max_detections = kDetectionsMemBudget / fd_module::get_single_detection_mem_size();
    static int8_t detections_mem[kDetectionsMemBudget];
    static fd_module detector;  // Dynamic?
    auto detections = detector.invoke(input_image_data.get(), detections_mem, sizeof(detections_mem));
    if (detections.detections_num >= max_detections) {
        printf("WARNING: Static Detections Buffer is full. Consider to increase it.\n");
    } else if (detections.detections_num <= 0) {
        printf("No detections in input image.\n");
    }

    printf("Pre_process ticks: %lld\n", detector.get_prof_ticks(fd_module::prof_tick_id::kProfPreProcess));
    printf("Model ticks: %lld\n", detector.get_prof_ticks(fd_module::prof_tick_id::kProfModel));
    printf("Post_process ticks: %lld\n", detector.get_prof_ticks(fd_module::prof_tick_id::kProfPostProcess));
    printf("Total ticks: %lld\n", detector.get_prof_ticks(fd_module::prof_tick_id::kProfTotal));
    for (int det_idx = 0; det_idx < detections.detections_num; ++det_idx) {
        const auto bbox_top_left =
            fd_module::get_coordinate(detections, det_idx, fd_module::coord_id::kCoordBboxTopLeft);
        const auto bbox_bot_right =
            fd_module::get_coordinate(detections, det_idx, fd_module::coord_id::kCoordBboxBotRight);

        const float bbox_score = fd_module::get_score(detections, det_idx);
        printf(" Found a face at ([X:%d, Y:%d]; [X:%d, Y:%d]) with (%f) score\n",
            bbox_top_left.clmn, bbox_top_left.row, bbox_bot_right.clmn, bbox_bot_right.row, bbox_score);

        draw_a_frame_rgb(bbox_top_left, bbox_bot_right, input_image_data.get(), img_width, img_height);
        draw_a_keypoint_rgb(fd_module::get_coordinate(detections, det_idx, fd_module::coord_id::kCoordLeftEye),
                            input_image_data.get(), img_width, img_height);
        draw_a_keypoint_rgb(fd_module::get_coordinate(detections, det_idx, fd_module::coord_id::kCoordRightEye),
                            input_image_data.get(), img_width, img_height);
        draw_a_keypoint_rgb(fd_module::get_coordinate(detections, det_idx, fd_module::coord_id::kCoordNose),
                            input_image_data.get(), img_width, img_height);
        draw_a_keypoint_rgb(fd_module::get_coordinate(detections, det_idx, fd_module::coord_id::kCoordMouth),
                            input_image_data.get(), img_width, img_height);
    }

    bmp_rgb_write("result.bmp", input_image_data.get(), img_width, img_height);
    printf("Done. See result in \"result.bmp\" file.\n");
    return 0;
}
