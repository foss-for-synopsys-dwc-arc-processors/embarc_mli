/*
 *  Copyright (c) 2019, Synopsys, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1) Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2)  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3) Neither the name of the <ORGANIZATION> nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//
// LSTM Based NN Example for UCI Smartphones HAR Dataset
//
// Based on the project of Guillaume Chevalie:
// https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition
//
// Dataset info:
// https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "mli_types.h"

#include "har_smartphone_model.h"
#include "har_smartphone_ref_inout.h"
#include "examples_aux.h"
#include "tensor_transform.h"
#include "tests_aux.h"

// Root to referenc IR vectors for comparison
// pass "./ir_idx_300" for debug
static const char kHarIrRefRoot[] = "";
static const char kOutFilePostfix[] =  "_out";

static float kSingleInSeq[IN_POINTS] = IN_SEQ_300;
static float kSingleOutRef[OUT_POINTS] = OUT_SCORES_300;

static void har_smartphone_preprocessing(const void * raw_input_, mli_tensor * net_input_);

//========================================================================================
//
// MAIN
//
//========================================================================================
int main(int argc, char ** argv ) {
    switch (argc) {
    // No Arguments for app. Process single hardcoded input
    // Print various measures to stdout
    //=========================================================
    case 1:
        printf("HARDCODED INPUT PROCESSING\n");
        model_run_single_in(kSingleInSeq, kSingleOutRef,
                har_smartphone_net_input, har_smartphone_net_output,
                har_smartphone_preprocessing, har_smartphone_net,
                kHarIrRefRoot);
        break;

    // APP <input_test_base.idx>
    // Output vectors will be written to <input_test_base.idx_out> file
    //=================================================================
    case 2:
        printf("Input IDX testset to output IDX set\n");
        char * out_path = malloc(strlen(argv[1]) + strlen(kOutFilePostfix) + 1);
        if (out_path == NULL) {
            printf("mem allocation failed\n");
            break;
        }
        out_path[0] = 0;
        strcat(out_path, argv[1]);
        strcat(out_path, kOutFilePostfix);

        model_run_idx_base_to_idx_out(argv[1], out_path,
                har_smartphone_net_input, har_smartphone_net_output,
                har_smartphone_preprocessing, har_smartphone_net,
                NULL);
        free(out_path);
        break;

    // APP <input_test_base.idx> <input_test_labels.idx>
    // Calculate accuracy of the model
    //=================================================================
    case 3:
        printf("ACCURACY CALCULATION on Input IDX testset according to IDX labels set\n");
        model_run_acc_on_idx_base(argv[1], argv[2],
                har_smartphone_net_input, har_smartphone_net_output,
                har_smartphone_preprocessing, har_smartphone_net,
                NULL);
        break;

    // Unknown format
    //=================================================================
    default:
        printf("App command line:\n"
                "\t%s \n\t\tProcess single hardcoded vector\n\n"
                "\t%s <input_test_base.idx> \n\t\tProcess testset from file and \n"
                "\t\t output model results to <input_test_base.idx_out> file\n\n", argv[0], argv[0]);
        break;
    }
    printf("FINISHED\n");

    return 0;
}

//========================================================================================
//
// Other internal functions and routines
//
//========================================================================================

//========================================================================================
// Data pre-processing for HAR Smartphone net
//========================================================================================
static void har_smartphone_preprocessing(const void * raw_input_, mli_tensor * net_input_) {
    const float * in = raw_input_;
    mli_hlp_float_to_fx_tensor(in, IN_POINTS, net_input_);
}
