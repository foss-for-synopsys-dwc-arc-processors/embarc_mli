/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

/**
 *  @author Yaroslav Donskov <yaroslav@synopsys.com>
 */


#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
//#include "profile.h"
#include "model.h"
#include "postprocess.h"


#define NUM_KPTS 6
#define MIN_NMS_TRESHOLD 0.3f
#define score_clipping_thresh 100.0f
#define min_score_thresh 0.75f
#define SKIP_INDEX -1


static void decode_boxes(float * raw_output, const float * anchors){

    for (int i = 0; i < NUM_ANCHORS; i++){
        float * box = raw_output + NUM_ANCHORS + i * NUM_COORDS;
        const float * anchor = anchors + NUM_ANCHORS_COORDS * i;
        float x_center = box[0] / IMAGE_SIDE *  anchor[2] + anchor[0];
        float y_center = box[1] / IMAGE_SIDE *  anchor[3] + anchor[1];
        float w = box[2] / IMAGE_SIDE * anchor[2];
        float h = box[3] / IMAGE_SIDE * anchor[3];
        box[0] = y_center - h / 2.0f;
        box[1] = x_center - w / 2.0f;
        box[2] = y_center + h / 2.0f;
        box[3] = x_center + w / 2.0f;
        for (int j = 0; j < NUM_KPTS; j++){
            int offset = 4 + j * 2;
            box[offset] = box[offset] / IMAGE_SIDE * anchor[2] + anchor[0];
            box[offset + 1] = box[offset + 1] / IMAGE_SIDE * anchor[3] + anchor[1];
        }
    }
}

static float jaccard(const float * box_a, const float * box_b){
    float max_x = std::max(box_a[2], box_b[2]);
    float max_y = std::max(box_a[3], box_b[3]);
    float min_x = std::min(box_a[0], box_b[0]);
    float min_y = std::min(box_a[1], box_b[1]);
    float inter_x = std::max(0.0f, max_x - min_x);
    float inter_y = std::max(0.0f, max_y - min_y);
    float inter =  inter_x * inter_y;

    float area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
    float area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);
    float union_ab = area_a + area_b - inter;
    return inter / union_ab;

}

static float sigmoid(float val){
    static const float cutoff_upper = 16.619047164916992188f;
    static const float cutoff_lower = -9.f;

    float result;
    if (val > cutoff_upper) {
        result = 1.0f;
    } else if (val < cutoff_lower) {
        result = std::exp(val);
    } else {
        result = 1.f / (1.f + std::exp(-val));
    }
    return result;
}

static int get_first_valid(int * arr, int size, int invalid_val){
    for (int i = 0; i < size; i++){
        if (arr[i] != invalid_val) return arr[i];
    }
    return invalid_val;
}

static void weighted_nms(const float * detections,  int num_detections, float * final_detections, int & num_final_detections){
    if (!num_detections) return;

    int remaining[MAX_DETECTIONS]; 
    std::iota(remaining, remaining + num_detections, 0);
    std::sort(
        remaining, remaining + num_detections,
        [&detections](int i, int j){ return detections[i * RESULT_SIZE + NUM_COORDS] > detections[j * RESULT_SIZE + NUM_COORDS]; }
    );
    int size = num_detections; 
    while ( size ){
        int first_index = get_first_valid(remaining, num_detections, SKIP_INDEX);
        assert( first_index != SKIP_INDEX );
        const float * detection = detections + first_index * RESULT_SIZE;
        float total_score = 0.0f;
        int num_overlaps = 0;
		float weighted[RESULT_SIZE] = {};
        for (int i = 0; i < num_detections; i++) {
            if (remaining[i] == SKIP_INDEX) continue;
            int index = remaining[i];
			const float * det = detections + index * RESULT_SIZE;
			float iou = jaccard(detection, det);
            if (iou > MIN_NMS_TRESHOLD){
                remaining[i] = SKIP_INDEX;
                size--;
                num_overlaps += 1;
                float weight = det[NUM_COORDS];
                total_score += weight;
                for (int j = 0; j < NUM_COORDS; j++) weighted[j] += ( det[j] * weight );
            }
        }

        for (int i = 0; i < NUM_COORDS; i++) weighted[i] /= total_score;
        weighted[NUM_COORDS] = total_score / (float) num_overlaps;
        memcpy(final_detections + num_final_detections * RESULT_SIZE, weighted, RESULT_SIZE * sizeof(float));
        num_final_detections++;
    }
}

void parse_detections(float * raw_output, const float * anchors, float * final_detections, int & num_final_detections){

#ifdef PROFILE
    unsigned t2 = tick();
#endif

    decode_boxes(raw_output, anchors);

#ifdef PROFILE
    unsigned t3 = tick(t2, "decode boxes");
#endif

    float filtered_detections[MAX_DETECTIONS * RESULT_SIZE];
    int num_detections = 0;
    for (int i = 0; i < NUM_ANCHORS; i++){
        if (num_detections == MAX_DETECTIONS) break;
        float score = std::max(-score_clipping_thresh, raw_output[i]);
        score = std::min(score_clipping_thresh, score);
        score = sigmoid(score);
        if (score >= min_score_thresh){
            memcpy(
                filtered_detections + num_detections * RESULT_SIZE, 
                raw_output + NUM_ANCHORS + i * NUM_COORDS, sizeof(float) * NUM_COORDS
            );
			filtered_detections[num_detections * RESULT_SIZE + NUM_COORDS] = score;
            num_detections++;
        }
    }

#ifdef PROFILE
    unsigned t4 = tick(t3, "filtering");
#endif

    weighted_nms(filtered_detections,  num_detections, final_detections, num_final_detections);

#ifdef PROFILE
    tick(t4, "nms");
#endif
}


void print_detections(float * detections, int num_detections, int image_id){
    for (int j = 0; j < num_detections; j++){
        float * det = detections + j * RESULT_SIZE;
        printf("%d,", image_id);
        for (int i = 0; i < RESULT_SIZE; i++){
            if (i != 16) printf("%d,", (int) roundf(det[i] * IMAGE_SIDE));
            else printf("%.3f\n", det[i]);
        }
    }
}