/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <stdio.h>

//===============================================================================================
// WAV file guard: provides RAII functionality for file resource and basic checking/reading logic.
//===============================================================================================
class wav_file_guard {
 public:
    
    struct waveheader_t{
        char chunkId[4];
        unsigned long chunkSize;
        char format[4];
        char subchunk1Id[4];
        unsigned long subchunk1Size;
        unsigned short audioFormat;
        unsigned short numChannels;
        unsigned long sampleRate;
        unsigned long byteRate;
        unsigned short blockAlign;
        unsigned short bitsPerSample;
        char subchunk2Id[4];
        unsigned long subchunk2Size;
    };

    enum wav_file_status {
        END_OF_FILE = 2,
        OPENED = 1,
        CLOSED = 0,
        CANT_OPEN_FILE = -1,
        NOT_A_WAV = -2,
        NOT_SUPPORTED_WAV = -3
    };
    
    static constexpr unsigned int kExpectedSampleRate = 16000;
    static constexpr unsigned int kExpectedAudioFormat = 1;
    static constexpr unsigned short kExpectedChannelsNum = 1;
    static constexpr unsigned short kExpectedBitsPerSample = 16;

    // Consuctor opens and checks file. Result will be reflected in the status.
    wav_file_guard(const char* file_name);
    ~wav_file_guard();

    // Read function always returns the requested number of sample. 
    // If there are not enough samples in stream, it fills the rest with zeroes (status is changed to EOF).
    size_t read_samples(void* samples, const size_t num);
    const wav_file_status &status() const {return status_;}
    const waveheader_t &header() const {return header_;}

 private:
    waveheader_t header_;
    wav_file_status status_;
    FILE* file_desc_;
};

