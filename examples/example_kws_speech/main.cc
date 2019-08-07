/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory>

#include "kws_factory.h"
#include "simple_kws_postprocessor.h"
#include "tests_aux.h"
#include "wav_file_guard.h"

using namespace std;
using mli_kws::kws_factory;
using mli_kws::kws_module;

// Define memory attribtes for fast arrays allocation purposes
#define __Xdata_attr __attribute__((section(".Xdata")))
#define __Ydata_attr __attribute__((section(".Ydata")))
#define __Zdata_attr __attribute__((section(".Zdata")))
#if (defined __Xxy) && defined (__CCAC__) && !defined(__GNUC__) 	
// Bank X (XCCM) attribute
#define _X __xy __Xdata_attr

// Bank Y (YCCM) attribute
#define _Y __xy __Ydata_attr

// Bank Z (DCCM) attribute
#define _Z __xy __Zdata_attr
#else
#define _X __Xdata_attr
#define _Y __Ydata_attr
#define _Z __Zdata_attr
#endif //(defined __Xxy) && defined (__CCAC__) && !defined(__GNUC__) 	


// Define occupyed fast memory sizes [in bytes]
#if !defined(STATE_MEM_SZ)
#define STATE_MEM_SZ (6500)
#endif
#if !defined(X_MEM_SZ)
#define X_MEM_SZ (18000)
#endif
#if !defined(Y_MEM_SZ)
#define Y_MEM_SZ (23000)
#endif

// Static fast memory buffers for further usage
static char _X    state_fastmem[STATE_MEM_SZ];
static char _X    temp_fastmem_a[X_MEM_SZ];
static char _Y    temp_fastmem_b[Y_MEM_SZ];

// Factory instance for KeyWordSpotting module building
#if 1
static mli_kws::dsconv_lstm_factory kws_builder;
#else
static mli_kws::some_other_kws_factory kws_builder;
#endif

// Structure for parsed application arguments
struct app_args {
    const char * file_path;
    bool to_profile;
    bool args_ok;
};

//========================================================================================
//
// Internal functions, routines
//
//========================================================================================

//========================================================================================
// Argument parser function
//========================================================================================
static app_args parse_arguments(int argc, char** argv) {
    const char* file_name = NULL;
    bool to_profile = false;
    bool args_ok = true;

    switch (argc) {
    case 2: 
        file_name = argv[1];
        break;

    case 3: 
        file_name = argv[1];
        if(strcmp(argv[2], "-info") == 0) {
            to_profile = true;
        } else {
            printf("Unkonown argument (\"%s\")\n", argv[2]);
            args_ok = false;
        }
        break;

    default: 
        args_ok = false;
        break;
    }

    if(!args_ok)
        printf("App command line: %s <input_wave_file.wav> [-info]\n\n", argv[0]);
    return {file_name, to_profile, args_ok};
}

//========================================================================================
// Print formatted keyword info
//========================================================================================
static void print_keyword_info(const kws_result &output, const kws_info &kw_info, const kws_module &kw_detector) {
    const int best_id = arg_max(output.results, kw_info.output_values_num);
    const size_t ms_per_ts = kw_info.timestamp_duration_ms;
    const float best_prob_percentage = output.results[best_id] * 100.f;
    const float start_seconds = static_cast<float>(output.start * ms_per_ts) / 1000.0f;
    const float end_seconds = static_cast<float>(output.end * ms_per_ts) / 1000.0f;
    const char *key_word = kw_detector.label_id_to_str(best_id);
    printf("%.3f\t%.3f\t\"%s\"(%.3f%%)\n", start_seconds, end_seconds, key_word, best_prob_percentage);
}

//========================================================================================
//
// MAIN
//
//========================================================================================
int main(int argc, char ** argv ) {
    app_args args = parse_arguments(argc, argv);
    if (!args.args_ok)
        return -1;

    // Phase 1: allocate resources and init modules
    //==============================================
    const kws_info detector_info = kws_builder.info();
    if (detector_info.state_fastmem_a_sz > sizeof(state_fastmem)
            || detector_info.temp_fastmem_a_sz > sizeof(temp_fastmem_a)
            || detector_info.temp_fastmem_b_sz > sizeof(temp_fastmem_b)) {
        printf("Not enough pre-alocated fast memory for chosen KWS module\n\n");
        return -1;
    }

    std::unique_ptr<sample_t[]> in_samples; 
    in_samples.reset(new sample_t[detector_info.input_samples_num]);
    if (!in_samples) {
        printf("Error in memory allocation for KWS\n\n");
        return -1;
    }

    std::unique_ptr<kws_module> kw_detector;
    kw_detector.reset(kws_builder.create_module(state_fastmem));
    if (!kw_detector) {
        printf("Error in creating KWS module\n\n");
        return -1;
    }

    wav_file_guard wav_file(args.file_path);
    if (wav_file.status() != wav_file_guard::OPENED) {
        printf("Error in open file\n %s: ", args.file_path);
        switch (wav_file.status()) {
        case wav_file_guard::CANT_OPEN_FILE :
            printf("%s\n", strerror(errno));
            break;
        case wav_file_guard::NOT_A_WAV :
            printf("Not a WAV file\n");
            break;
        case wav_file_guard::NOT_SUPPORTED_WAV :
            printf("Not supported WAV format\n");
            break;
        default:
            printf("Other error(%d)\n", wav_file.status()); break;
        }
        return -1;
    }

    // Phase 2: Reading and processing input file in loop
    //======================================================
    bool is_end_of_stream = false;
    bool is_result_produced = false;
    int results_count = 0;
    unsigned int cycles_max = 0;
    unsigned int cycles_min = (unsigned int)(-1);
    unsigned long cycles_total = 0;
    unsigned int process_calls = 0;
    simple_kws_postprocess result_analyzer(detector_info.output_values_num, 0, 1);
    do {
        wav_file.read_samples((void *)in_samples.get(), detector_info.input_samples_num);
        is_end_of_stream = (wav_file.status() == wav_file_guard::END_OF_FILE);

        kws_status status;

        PROFILE(status = kw_detector->process(in_samples.get(), temp_fastmem_a, temp_fastmem_b));
        ++process_calls;
        cycles_max = MAX(cycle_cnt, cycles_max);
        // 1000 - threshold to sort-out service work of detector
        const unsigned int considerable_cycles_count = 1000; 
        cycles_min = (cycle_cnt < considerable_cycles_count)? cycles_min: MIN(cycle_cnt, cycles_min); 
        cycles_total += cycle_cnt;

        if (status == KWS_STATUS_RESULT_READY) {
            is_result_produced = true;
            ++results_count;
            bool kw_found = result_analyzer.process_kw_result(kw_detector->get_result());            
            if(kw_found) {
                const kws_result output = result_analyzer.get_kw_result();
                print_keyword_info(output, detector_info, *kw_detector);
            }
        }
        else if (status != KWS_STATUS_NONE) {
            printf("Error of kws module in processing\n");
            break;
        }
    } while (!(is_end_of_stream && is_result_produced));

    // Phase 3: Finalize and output performance (if required)
    //======================================================
    if(result_analyzer.finalize()) {
        print_keyword_info(result_analyzer.get_kw_result(), detector_info, *kw_detector);
    }
    if(args.to_profile) {
        printf("\n\nModel: %s\n", detector_info.name);
        
        printf("Dynamic Memory requirements [bytes]: %d\n", detector_info.dynamic_mem_sz);
        printf("Fast Memory requirements [bytes]\n");
        printf("\tCoefficients Memory: %d\n", detector_info.coeff_fastmem_sz);
        printf("\tState Memory: %d\n", detector_info.state_fastmem_a_sz);
        printf("\tIR Memory A: %d\n", detector_info.temp_fastmem_a_sz);
        printf("\tIR Memory B: %d\n", detector_info.temp_fastmem_b_sz);
        printf("\tTotal: %d\n", detector_info.temp_fastmem_b_sz + detector_info.temp_fastmem_a_sz 
                                + detector_info.state_fastmem_a_sz + detector_info.coeff_fastmem_sz);

        const float stream_total_seconds = (detector_info.input_samples_num * process_calls) / 16000.0f;
        printf("Performance [cycles]:\n");
        printf("\tMax per frame: %d\n", cycles_max);
        printf("\tMin per frame: %d\n", cycles_min);
        printf("\tAverage per inference: %lu\n", cycles_total/results_count);
        printf("\tAverage per second: %.1f\n", cycles_total/stream_total_seconds);
        printf("\tTotal for stream(%.3f sec): %lu\n", stream_total_seconds, cycles_total);
    }
    return 0;
}



