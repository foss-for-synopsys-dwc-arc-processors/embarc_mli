/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <memory>

#include "kws_types.h"


// Index of maximum value in the array 
int arg_max(const float* arr, const int size);

//===============================================================================================
// Basic postprocessing for sequential KWS results.
//===============================================================================================
class simple_kws_postprocess {
 public:
    // Constructor gets total number of classes, ID of silence class, and number of sequential silences to output result.
    simple_kws_postprocess(const int classes_num, const int silence_id, const unsigned int silence_to_finalize); 
    ~simple_kws_postprocess() = default;

    // Process current KWS result (prob vector). Outputs decision - whether keyword found or not.
    bool process_kw_result(const kws_result &in); 

    // If there are no data for postprocessing, finalize current state (output keyword if there is a candidate)
    bool finalize();

    // Returns current saved keyword 
    kws_result get_kw_result() const;

private:
    std::unique_ptr<float[]> kw_probs;
    int kw_id;
    timestamp_t kw_start;
    timestamp_t kw_end;
    const int classes_num;
    const int silence_id;
    unsigned long silence_series;
    const unsigned int silence_to_finalize;
};
