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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "mli_types.h"

#include "cifar10_ref_inout.h"
#include "cifar10_model.h"
#include "examples_aux.h"
#include "tests_aux.h"

// Root to referenc IR vectors for comparison
// pass "./ir_idx_12_chw_small" or "./ir_idx_12_chw_big" for debug (regarding to used modelCHW and HWC layout accordingly)
static const char kCifar10RootIR[] = "";
static const char kOutFilePostfix[] = "_out";

const unsigned char kSingleIn[IN_POINTS] = IN_IMG_12;
const float kSingleOutRef[OUT_POINTS] = OUT_PROB_12;

static void cifar10_preprocessing(const void * image_, mli_tensor * net_input_);

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
        model_run_single_in(kSingleIn, kSingleOutRef,
                cifar10_cf_net_input, cifar10_cf_net_output,
                cifar10_preprocessing, cifar10_cf_net,
                kCifar10RootIR);
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
                cifar10_cf_net_input, cifar10_cf_net_output,
                cifar10_preprocessing, cifar10_cf_net,
                NULL);
        free(out_path);
        break;

    // APP <input_test_base.idx> <input_test_labels.idx>
    // Calculate accuracy of the model
    //=================================================================
    case 3:
        printf("ACCURACY CALCULATION on Input IDX testset according to IDX labels set\n");
        model_run_acc_on_idx_base(argv[1], argv[2],
                cifar10_cf_net_input, cifar10_cf_net_output,
                cifar10_preprocessing, cifar10_cf_net,
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
// Image pre-processing for CIFAR-10 net
//========================================================================================
static void cifar10_preprocessing(const void * image_, mli_tensor * net_input_) {
    const unsigned char * in = image_;
    d_type * const dst = (d_type * const)net_input_->data;

    // Copying data  to input tensor with subtraction of average.
    // Data shft may be required depending on tensor format
    if (net_input_->el_params.fx.frac_bits == 7) {
        for (int idx = 0; idx < IN_POINTS; idx++)
            dst[idx] = (d_type)((int)in[idx] - 128);
    } else if (net_input_->el_params.fx.frac_bits > 7) {
        int shift_left = net_input_->el_params.fx.frac_bits - 7;
        for (int idx = 0; idx < IN_POINTS; idx++)
            dst[idx] = (d_type)((int)in[idx] - 128) << shift_left;
    } else {
        int shift_right = 7 - net_input_->el_params.fx.frac_bits;
        for (int idx = 0; idx < IN_POINTS; idx++)
            dst[idx] = (d_type)((int)in[idx] - 128)  >> shift_right; // w/o rounding
    }
}
