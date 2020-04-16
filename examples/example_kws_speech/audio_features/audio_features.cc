/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#include "audio_features.h"

#include <float.h>
#include <math.h>

#include <dsplib.h>

#ifdef __Xxy
#define __XY __xy
#else
#define __XY
#endif

using namespace au_fex;


//========================================================================================
//
// General Feature extraction functions
//
//========================================================================================

//====================================================
// Mel/Frequency scale forward/backward transformation
//====================================================
static inline float mel_to_freq(const float mel_freq) {
    return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f);
}
static inline float freq_to_mel(const float freq) {
    return 1127.0f * logf(1.0f + freq / 700.0f);
}

//====================================================
// Transform and scale integer samples into float values
//====================================================
static inline int transform_normalize(const __XY int16_t* __restrict in, const int vals, const float norm_scale, 
                                      __XY float*  __restrict out) {
    int i = 0;
    for(; i < vals; i++)
        out[i] = in[i] * norm_scale; 
    return i;
}

//====================================================
// Calculate power spectrum of Fourier transform results
//====================================================
static inline int power_spect(const __XY float* __restrict in_spect, const int spect_bins, 
                              __XY float*  __restrict out_pow_spect) {
    // handle this special case
    const float first_energy = in_spect[0] * in_spect[0];
    const float last_energy =  in_spect[1] * in_spect[1];

    const int half_dim = spect_bins/2;
    for (int i = 1; i < half_dim; i++) {
        const float real = in_spect[i*2];
        const float im = in_spect[i*2 + 1];
        out_pow_spect[i] = real*real + im*im;
    }
    out_pow_spect[0] = first_energy;
    out_pow_spect[half_dim] = last_energy; 
    return half_dim + 1;
}

//====================================================
// Calculate amplitude spectrum of Fourier transform results
//====================================================
static inline int ampl_spect(const __XY float* __restrict in_spect, const int spect_bins,
                             __XY float*  __restrict out_ampl_spect) {
    // handle this special case
    const float first_energy = in_spect[0] * in_spect[0];
    const float last_energy =  in_spect[1] * in_spect[1];

    const int half_dim = spect_bins/2;
    for (int i = 1; i < half_dim; i++) {
        const float real = in_spect[i*2];
        const float im = in_spect[i*2 + 1];
        out_ampl_spect[i] = sqrtf(real*real + im*im);
    }
    out_ampl_spect[0] = first_energy;
    out_ampl_spect[half_dim] = last_energy; 
    return half_dim + 1;
}

//====================================================
// Calculate window function (Hanning) 
//====================================================
static inline int create_hann_window(const int vals, float*  out_window) {
    int idx = 0;
    const float math_2pi = M_2PI;
    for(; idx < vals; idx++) {
        out_window[idx] = 0.5f - 0.5f*cosf(math_2pi * ((float)idx) / (vals));
    }
    return idx;
}

//====================================================
// Apply window function on input frame element wise
//====================================================
static inline int apply_window(const __XY float* __restrict in, const __XY float* __restrict window,
                               const int vals, __XY float*  __restrict out) {
    int idx = 0;
    for(; idx < vals; idx++) {
        out[idx] = in[idx] * window[idx]; 
    }
    return idx;
}

//====================================================
// Apply log function on input element wise
//====================================================
static inline int log_norm(const __XY float* __restrict in, const int vals, __XY float* __restrict out) {
    int idx = 0;
    for (; idx < vals; idx++) {
        out[idx] = logf(in[idx]);
    }
    return idx;
}

//====================================================
// Matrix-vector multiplication
//====================================================
static inline int matrix_vector_mul(const __XY float* __restrict in_matrix, const __XY float* __restrict in_vector,
                                    const int rows, const int columns, __XY float*  __restrict out) {
    int out_idx = 0;
    for (; out_idx < rows; out_idx++) {
        __XY float *__restrict mat_row = (__XY float *__restrict) (&in_matrix[out_idx * columns]);
        float accum = 0.0f;
        for (int in_idx = 0; in_idx < columns; in_idx++) {
            accum += mat_row[in_idx] * in_vector[in_idx];
        }
        out[out_idx] = accum;
    }
    return out_idx;
}

//====================================================
// Create set of triangle filter in mel scale 
//====================================================
static int create_mel_triangle_filters(const int filters_num, const int sample_rate, const int freq_bins,
                                        const int low_freq, const int high_freq, 
                                        float** pointers_to_filters, int *fbins_first_idxs, int *fbins_last_idxs, 
                                        float*  buf_for_filters, int* buf_for_filters_sz) {
    const int half_dim = freq_bins / 2;
    const int buf_len_limit = *buf_for_filters_sz;
    const float freq_bin_width = ((float)sample_rate) / freq_bins;
    const float mel_low_freq = freq_to_mel(low_freq);
    const float mel_high_freq = freq_to_mel(high_freq); 
    const float mel_freq_delta = (mel_high_freq - mel_low_freq) / (filters_num + 1);

    // outer loop on triangle filters
    float* this_filter = buf_for_filters;
    int filter_idx = 0;
    int buf_for_filters_total_len = 0;
    for (; filter_idx < filters_num && buf_for_filters_total_len < buf_len_limit; filter_idx++) {
        const float left_mel = mel_low_freq + filter_idx * mel_freq_delta;
        const float center_mel = mel_low_freq + (filter_idx + 1) * mel_freq_delta;
        const float right_mel = mel_low_freq + (filter_idx + 2) * mel_freq_delta;

        pointers_to_filters[filter_idx] = this_filter;

        // Inner loop on frequency - search for freq bins related to filter and fill it
        int first_freq_bin_index = -1, last_freq_bin_index = -1;
        int weight_idx = 0;
        for (int i = 0; i < half_dim; i++) {
            // center freq of this fft bin.
            const float freq = (freq_bin_width * i);
            const float mel = freq_to_mel(freq);

            // if frequency bin is related to filter and there are enough mem,
            // calculate one more filter coefficient (weight
            if(mel > left_mel && mel < right_mel && buf_for_filters_total_len < buf_len_limit) {
                const float weight = (mel <= center_mel)?
                    (mel - left_mel) / (center_mel - left_mel): // left slope
                    (right_mel-mel) / (right_mel-center_mel);   // right slope

                this_filter[weight_idx] = weight;
                weight_idx++;
                buf_for_filters_total_len++;
                first_freq_bin_index = (first_freq_bin_index == -1)? i: first_freq_bin_index;
                last_freq_bin_index = i;
            } else if (mel > right_mel) {
                break;
            }

        }
        fbins_first_idxs[filter_idx] = first_freq_bin_index;
        fbins_last_idxs[filter_idx] = last_freq_bin_index;
        this_filter = &buf_for_filters[buf_for_filters_total_len];
    }
    *buf_for_filters_sz = buf_for_filters_total_len;
    return filter_idx;
}

//==================================================================================
// Calculate buffer length requirement for set of triangle filter in mel scale 
//==================================================================================
static int mel_triangle_filters_buf_length(const int filters_num, const int sample_rate,
                                            const int freq_bins, const int low_freq, const int high_freq) {
    const int half_dim = freq_bins / 2;
    const float freq_bin_width = ((float)sample_rate) / freq_bins;
    const float mel_low_freq = freq_to_mel(low_freq);
    const float mel_high_freq = freq_to_mel(high_freq); 
    const float mel_freq_delta = (mel_high_freq - mel_low_freq) / (filters_num + 1);

    // outer loop on triangle filters
    int buf_for_filters_total_len = 0;
    for (int filter_idx = 0; filter_idx < filters_num; filter_idx++) {
        const float left_mel = mel_low_freq + filter_idx * mel_freq_delta;
        const float right_mel = mel_low_freq + (filter_idx + 2) * mel_freq_delta;

        // Inner loop on frequency - search for freq bins related to filter and fill it
        for (int i = 0; i < half_dim; i++) {
            // center freq of this fft bin.
            const float freq = (freq_bin_width * i);
            const float mel = freq_to_mel(freq);

            // if frequency bin is related to filter and there are enough mem,
            // calculate one more filter coefficient (weight
            if (mel > left_mel && mel < right_mel) {
                buf_for_filters_total_len++;
            } else if (mel > right_mel) {
                break;
            }
        }
    }
    return buf_for_filters_total_len;
}

//====================================================
// Apply set of triangle filter to input spectrum 
//====================================================
static int apply_filter_banks(const __XY float* __restrict in, float** filters, 
                               const int *fbins_first_idxs, const int *fbins_last_idxs, 
                               const int bins_num, __XY float*  __restrict out) {
    int filter_idx = 0;
    for (; filter_idx < bins_num; filter_idx++) {
        __XY float *__restrict filter = (__XY float *__restrict) filters[filter_idx];
        float accum = 0.0f;
        const int first_index = fbins_first_idxs[filter_idx];
        const int last_index = fbins_last_idxs[filter_idx];

        for(int freq_idx = first_index, weight_idx = 0; freq_idx <= last_index; freq_idx++, weight_idx++) {
            accum += in[freq_idx] * filter[weight_idx];
        }
        out[filter_idx] = (accum != 0.0f)? accum: FLT_MIN; //std::numeric_limits<float>::min();
    }
    return filter_idx;
}


//=========================================================================
//
// FBANK features class related declarations
//
//=========================================================================

//=========================================================================
// Wrapper class for specific FFT implementation 
//=========================================================================
class fbank_features::rfft_wrapper {
public:
    rfft_f32_t handle;
    __XY float* rfft_state;

    rfft_wrapper(__XY float* state_buf, int frame_len) 
    : handle()
    , rfft_state(state_buf) {
        dsp_rfft_init_f32(&handle, static_cast<uint32_t>(frame_len), rfft_state);
    };

    static int get_state_len(int frame_len) {
        return dsp_rfft_getsize_f32(frame_len) / sizeof(float);
    };

    void run(const __XY float* in, __XY float* out) {
        dsp_rfft_f32(&handle, (const __AGU f32_t *)in, (__AGU f32_t *)out);
    }
};

//=========================================================================
// Get module information and requirements for defined parameters
//=========================================================================
fbank_features::info fbank_features::get_info(const params& config) {
    int frame_len_padded = 1;
    for (; frame_len_padded < config.frame_len; frame_len_padded = frame_len_padded << 1)
        ;

    const int window_buf_len = config.frame_len;
    const int rfft_state_buf_len = rfft_wrapper::get_state_len(frame_len_padded);
    const uint32_t dyn_mem_sz = sizeof(rfft_wrapper) + 2 * (config.num_fbank_bins * sizeof(int)) 
                                + (config.num_fbank_bins * sizeof(float *));
    const int filters_buf_len = mel_triangle_filters_buf_length(
        config.num_fbank_bins, config.sample_rate, frame_len_padded, config.low_freq, config.high_freq);

    return info {
        window_buf_len + rfft_state_buf_len + filters_buf_len,
        frame_len_padded,
        frame_len_padded,
        dyn_mem_sz
    };
}

//=========================================================================
// Fbanks features: Module constructor
//=========================================================================
fbank_features::fbank_features(const params& config) 
        : in_frame_len(config.frame_len)
        , out_fbanks_num(config.num_fbank_bins)
        , sample_rate(config.sample_rate)
        , low_freq(config.low_freq)
        , high_freq(config.high_freq)
        , frame_len_padded_(0)
        , fbins_first_idxs_(new int[config.num_fbank_bins])
        , fbins_last_idxs_(new int[config.num_fbank_bins])
        , mel_filters_set_(new float*[config.num_fbank_bins])
        , rfft_(nullptr)
        , window_func_fast_buf_a_(nullptr)
        , mel_filters_storage_fast_buf_a_(nullptr)
{}


//=========================================================================
// Desctructor (release memory for rfft wrapper)
//=========================================================================
// We need it explicitly defined to prevent compiler from inplace generating the default one 
// to keep "rfft_wrapper" type incomplete and wrap it into unique pointer. 
// Desctructor of unique_ptr<rfft_wrapper> type need to know "rfft_wrapper" size for correct deletion. 
// For this we are forcing ~unique_ptr<rfft_wrapper> to be generated beside rfft_wrapper definition as part 
// of ~fbank_features()). Default version of ~fbank_features() is enough for this
fbank_features::~fbank_features() = default;


//=========================================================================
// Fbanks features: Factory method
//=========================================================================
fbank_features* fbank_features::create_fbank_fex_module(const params& config, float* state_fastbuf_a,
                                                        int state_fastbuf_a_len) {
    // Initial params checke
    fbank_features* out = nullptr;
    bool is_valid = (state_fastbuf_a_len > config.frame_len && config.num_fbank_bins > 0 && config.frame_len > 0 &&
                    config.low_freq >= 0 && config.high_freq > 0 && config.sample_rate > 0);
    
    // FEX module allocation
    if (is_valid) {
        out = new fbank_features(config);
        is_valid &= (out != nullptr && out->fbins_first_idxs_ != nullptr  &&
                     out->fbins_last_idxs_ != nullptr && out->mel_filters_set_ != nullptr);
    }

    // Define pow_2 frame len and fill window function 
    int state_fastbuf_occupied = 0;
    if(is_valid) {
        // Round-up frame_len to nearest power of 2.
        int frame_len_padded = 1;
        for (; frame_len_padded < config.frame_len; frame_len_padded = frame_len_padded << 1);
        out->frame_len_padded_ = frame_len_padded;

        out->window_func_fast_buf_a_ = (float*)state_fastbuf_a;
        state_fastbuf_occupied = create_hann_window(out->in_frame_len, out->window_func_fast_buf_a_);
        state_fastbuf_a_len -= state_fastbuf_occupied;
        is_valid &= (state_fastbuf_occupied == out->in_frame_len);
    }

    // Create filter banks 
    if(is_valid) {
        out->mel_filters_storage_fast_buf_a_ = (float*)&state_fastbuf_a[state_fastbuf_occupied];
        int fastbuf_for_filters_len = state_fastbuf_a_len;
        const int filters_filled = create_mel_triangle_filters(
            out->out_fbanks_num, out->sample_rate, out->frame_len_padded_, out->low_freq, out->high_freq, 
            out->mel_filters_set_.get(), out->fbins_first_idxs_.get(), out->fbins_last_idxs_.get(), 
            out->mel_filters_storage_fast_buf_a_, &fastbuf_for_filters_len);

        state_fastbuf_occupied += fastbuf_for_filters_len;
        state_fastbuf_a_len -= fastbuf_for_filters_len;
        is_valid &= (filters_filled == out->out_fbanks_num);
    }

    // Build FFT object 
    is_valid &= (state_fastbuf_a_len >= rfft_wrapper::get_state_len(out->frame_len_padded_));
    if(is_valid) {
        out->rfft_.reset(new rfft_wrapper((__XY float*)&state_fastbuf_a[state_fastbuf_occupied], out->frame_len_padded_));
        is_valid &= (out->rfft_ != nullptr);
    }

    // Check object and return
    if (!is_valid && out != nullptr) {
        delete out;
        out = nullptr;
    }
    return out;
}


//=========================================================================
// Fbank features computation function
//=========================================================================
int fbank_features::compute(const int16_t* samples, float* out_features, float* temp_fast_buf_a, float* temp_fast_buf_b) {
    const float norm_val = 1.0f/(1<<15);
    int values = transform_normalize((const __XY int16_t*)samples, in_frame_len, norm_val,
                                     (__XY float*)temp_fast_buf_b);
    values = apply_window((__XY float*)temp_fast_buf_b, (__XY float*)window_func_fast_buf_a_, in_frame_len,
                          (__XY float*)temp_fast_buf_a);
    for (; values < frame_len_padded_; ++values)
        temp_fast_buf_a[values] = 0.0f;
    rfft_->run((__XY float*)temp_fast_buf_a, (__XY float*)temp_fast_buf_b);
    values = power_spect((__XY float*)temp_fast_buf_b, values, (__XY float*)temp_fast_buf_a);
    values = apply_filter_banks((__XY float*)temp_fast_buf_a, mel_filters_set_.get(), fbins_first_idxs_.get(),
                                fbins_last_idxs_.get(), out_fbanks_num, (__XY float*)temp_fast_buf_b);
    return log_norm((__XY float*)temp_fast_buf_b, values, (__XY float*)out_features);
}

