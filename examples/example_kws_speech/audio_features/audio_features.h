/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _AUDIO_FEATURES_H_
#define _AUDIO_FEATURES_H_

#include <math.h>
#include <stdint.h>
#include <memory>

#ifdef __Xxy
#define __XY __xy
#else
#define __XY
#endif

//
// Audio features extraction function and types.
//

namespace au_fex {

//==============================================================
// FBANK Features extraction class
// 
// For better understanding see next (implementation might slightly deffers:
//  https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
//
// Usage example:
//      fbank_features::params cfg = { /*setup parameters*/ };
//      fbank_features::info fex_info = fbank_features::get_info(cfg);
//
//      float* state_fast_mem = ...; // Allocate fast memory for state
//      fbank_features* fbank_fex = fbank_features::create_fbank_fex_module(cfg, state_fastbuf, state_fastbuf_len);
//
//      // work with module
//      ... 
//      fbank_fex.compute(samples, temp_mam_a, temp_mem_b);
//
//      delete fbank_fex; // Module is allocated in heap, so it should be managed
//==============================================================
class fbank_features {
 public:
    // Structure to provide info and mem requirements for designated module parameters
    struct info {
        int state_fastbuf_a_len;
        int temp_fastbuf_a_len;
        int temp_fastbuf_b_len;
        uint32_t dynamic_mem_sz;
    };

    // Set of module configuration parameters
    struct params {
        int num_fbank_bins;
        int frame_len;
        int sample_rate;
        int low_freq;
        int high_freq; 
    };

    // Get module information and requirements for defined parameters
    //
    // return: info structure with memory requirements 
    static info get_info(const params& config);
    
    // Factory method for module construction and initialization
    //
    // return: nullptr if construction failed or pointer to allocated and initialized fbank features module. 
    // Caller is responsible to manage module afterward.
    static fbank_features* create_fbank_fex_module(const params& config, float* state_fastbuf_a,  
                                                   int state_fastbuf_a_len);
    
    // Destructor
    ~fbank_features();

    // Fbank features computation for input frame.
    //
    // params:
    // samples          -   buffer with input samples (16 bit depth)
    // out_features     -   buffer to keep calculated fbanks (result will be stored here)
    // temp_fast_mem_a  -   pointer to fast buffer A to keep intermediate result. 
    //                      Size of buffer must be the info::temp_fastbuf_a_len. For better performance mem A 
    //                      should be allocated in the different bank to memory B
    // temp_fast_mem_B  -   pointer to fast buffer B to keep intermediate result. 
    //                      Size of buffer must be the info::temp_fastmem_b_sz. For better performance mem B 
    //                      should be allocated in the different bank to memory A

    // return: nullptr if construction failed or pointer to allocated and initialized fbank features module. 
    // Caller is responsible to manage module afterward.
    int compute(const int16_t* samples, float* out_features, float* temp_fastbuf_a, float* temp_fastbuf_b);

    // Parameters fbank module was constructed for
    const int in_frame_len;
    const int out_fbanks_num;
    const int sample_rate;
    const int low_freq;
    const int high_freq;

 private:
    // FFT functionality wrapper class (incmplete type)
    class rfft_wrapper;

    // Constructor is not accessable for user (constructin by factory method)
    fbank_features(const params& config);
    
    int frame_len_padded_;
    std::unique_ptr<int[]> fbins_first_idxs_;       // Start index set for applying triangle filters
    std::unique_ptr<int[]> fbins_last_idxs_;        // End index set for applying triangle filters
    std::unique_ptr<float*[]> mel_filters_set_;     // pointers set to triangle filters
    rfft_wrapper* rfft_;                            // FFT functionality wrapper (pointer to incomplete type)
    __XY float* window_func_fast_buf_a_;            // Storage for window function
    __XY float* mel_filters_storage_fast_buf_a_;    // Storage for triangle filters
};


// Transform input integer samples into float values with normalization (scaling samples with norm_scale factor)
int transform_normalize(const __XY int16_t* __restrict in, const int vals, const float norm_scale, 
                        __XY float*  __restrict out);

// Calculate power spectrum of Fourier transform results
int power_spect(const __XY float* __restrict in_spect, const int spect_bins, __XY float*  __restrict out_pow_spect);

// Calculate amplitude spectrum of Fourier transform results
int ampl_spect(const __XY float* __restrict in_spect, const int spect_bins, __XY float*  __restrict out_ampl_spect);

// Calculate window function (Hanning) 
int create_hann_window(const int vals, float* out_window);

// Apply window function on input frame element wise
int apply_window(const __XY float* __restrict in, const __XY float* __restrict window, const int vals, 
                 __XY float*  __restrict out);

// Transform mel frequency value into linear scale
float mel_to_freq(const float mel_freq);

// Transform linear frequency value into mel scale
float freq_to_mel(const float freq);

// Apply log function on input element wise
int log_norm(const __XY float* __restrict in, const int vals, __XY float* __restrict out);

// matrix-vector multiplication
int matrix_vector_mul(const __XY float* __restrict in_matrix, const __XY float* __restrict in_vector, 
                      const int rows, const int columns, __XY float*  __restrict out);

// Create set of triangle filter in mel scale according to parameters
int create_mel_triangle_filters(
    const int filters_num,
    const int sample_rate,
    const int freq_bins,
    const int low_freq,
    const int high_freq,

    float** pointers_to_filters,
    int *fbins_first_idxs,
    int *fbins_last_idxs,
    float*  buf_for_filters,
    int* buf_for_filters_sz);

// Calculate buffer length requirement for set of triangle filter in mel scale according to parameters
int mel_triangle_filters_buf_length(
    const int filters_num,
    const int sample_rate,
    const int freq_bins,
    const int low_freq,
    const int high_freq);

// Apply set of triangle filter to input spectrum 
int apply_filter_banks(
    const __XY float* __restrict in, 
    float** filters,
    const int *fbins_first_idxs,
    const int *fbins_last_idxs,
    const int bins_num,
    __XY float*  __restrict out);

} // namespace au_fex 

#endif //#ifndef _AUDIO_FEATURES_H_
