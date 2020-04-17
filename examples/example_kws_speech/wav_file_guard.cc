/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <string.h>

#include "wav_file_guard.h"



using namespace std;

//========================================================================================
//  WAV File Guard Methods
//========================================================================================

//  Constructor: opening and checking wav file 
//========================================================================================
wav_file_guard::wav_file_guard(const char* file_name) 
        : header_()
        , status_(CLOSED) 
        , file_desc_(NULL) {
    FILE* f;
    if ((f = fopen(file_name, "rb")) == NULL) {
        status_ = CANT_OPEN_FILE;
        return;
    }

    // Read header and check it for supported type
    fread(&header_, sizeof(waveheader_t), 1, f);
    if ((header_.chunkId[0] != 'R' && header_.chunkId[1] != 'I' && 
            header_.chunkId[2] != 'F' && header_.chunkId[3] != 'F') ||
            (header_.format[0] != 'W' && header_.format[1] != 'A' && 
            header_.format[2] != 'V' && header_.format[3] != 'E')) {
        fclose(f);
        status_ = NOT_A_WAV;
        return;
    }

    if (header_.numChannels != kExpectedChannelsNum || 
            header_.audioFormat != kExpectedAudioFormat || 
            header_.bitsPerSample != kExpectedBitsPerSample ||
            header_.sampleRate != kExpectedSampleRate) {
        fclose(f);
        status_ = NOT_SUPPORTED_WAV;
        return;
    }
    status_ = OPENED;
    file_desc_ = f;
}

//  Desctructor: closing the assigned file 
//========================================================================================
wav_file_guard::~wav_file_guard() {
    if (file_desc_ != NULL)
        fclose(file_desc_);
}

//  Reading required number of samples (if not enough in stream - pad the rest with 0)
//========================================================================================
size_t wav_file_guard::read_samples(void* samples, const size_t num) {
    size_t itm = 0;
    if (status_ == OPENED) 
        itm = fread(samples, header_.bitsPerSample / 8, num, file_desc_);
    if (itm < num) {
        samples = (void *)((char *)samples + (header_.bitsPerSample / 8) * itm);
        status_ = END_OF_FILE;
        memset(samples, 0, (num - itm) * header_.bitsPerSample / 8);
    }
    return itm;
}
