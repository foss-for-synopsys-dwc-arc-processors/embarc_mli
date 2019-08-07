/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "simple_kws_postprocessor.h"

using namespace std;

//========================================================================================
// Index of maximum value in the array 
//========================================================================================
int arg_max(const float* arr, const int size) {
    int arg_max = 0;
    float max = arr[0];
    for (int idx = 1; idx < size; ++idx)
        if (max < arr[idx]) {
            arg_max = idx;
            max = arr[idx];
    }
    return arg_max;
}

//========================================================================================
//  Basic Postprocessor class methods
//========================================================================================

//  Constructor 
//========================================================================================
simple_kws_postprocess::simple_kws_postprocess(const int classes_num, const int silence_id, 
                                               const unsigned int silence_to_finalize) 
        : kw_probs(new float[classes_num])
        , kw_id(silence_id)
        , kw_start(0)
        , kw_end(0)
        , classes_num(classes_num)
        , silence_id(silence_id)
        , silence_series(silence_to_finalize)
        , silence_to_finalize(silence_to_finalize)
{}

// Process current KWS result (prob vector).
//========================================================================================
bool  simple_kws_postprocess::process_kw_result(const kws_result &in) {
    if(!kw_probs)
        return false;

    const int in_best_class = arg_max(in.results, classes_num);
    if(in_best_class == silence_id) {
        ++silence_series;
    } else {
        if(silence_series >= silence_to_finalize || kw_probs[kw_id] < in.results[in_best_class]) {
            kw_id = in_best_class;
            kw_start = in.start;
            kw_end = in.end;
            for(int i = 0; i < classes_num; ++i)
                kw_probs[i] = in.results[i];
        }
        silence_series = 0;
    }
    return (silence_series == silence_to_finalize) && (kw_id != silence_id);
}

// Finalize prosprocessing (No further data is expected)
//========================================================================================
bool simple_kws_postprocess::finalize() {
    if (silence_series < silence_to_finalize) {
        silence_series = silence_to_finalize;
        return true;
    } else {
        return false;
    }
}

//  Return saved keyword result 
//========================================================================================
kws_result simple_kws_postprocess::get_kw_result() const {
    return kws_result{kw_start, kw_end, kw_probs.get()};
}

