#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "mli_types.h"

#include "model/emnist_ref_inout.h"
#include "model/emnist_model.h"
#include "examples_aux.h"
#include "tests_aux.h"

#if defined (__GNUC__) && !defined (__CCAC__)
extern int start_init(void);
#endif // if defined (__GNUC__) && !defined (__CCAC__)
static const char kOutFilePostfix[] = "_out";

const unsigned char kSingleIn[IN_POINTS] = IN_IMG_1;
const float kSingleOutRef[OUT_POINTS] = OUT_PROB_1;

static void emnist_preprocessing(const void * image_, mli_tensor * net_input_);

void run_emnist_cf_net();

#define EXAMPLE_MAX_MODE (3)
int mode=3; // emulation argc for GNU toolchain
char param[EXAMPLE_MAX_MODE][256];// emulation argv for GNU toolchain
//========================================================================================
//
// MAIN
//
//========================================================================================
int main(int argc, char ** argv ) {
#if defined (__GNUC__) && !defined (__CCAC__)
//ARC GNU tools
    if (0 != start_init() ){
        printf("ERROR: init proccesor\n");
        //Error init proccesor;
        return 1;
    }
//fill mode and param from cmd line script before use

#else
//Metaware tools
//fill mode and param from argc and argv
    if ( argc <= EXAMPLE_MAX_MODE) {
        mode=argc;
    
        for(int i=0; i < mode; i++) {
            memcpy( &param[i][0], argv[i], strlen(argv[i]) );
        }
    }        
#endif // if defined (__GNUC__) && !defined (__CCAC__)
   
    //checking that variables are set
    if(mode == 0){
        printf("ERROR: mode not set up\n");
#if defined (__GNUC__) && !defined (__CCAC__)        
//ARC GNU tools
        printf("Please set up mode \n");
        printf("Please check that you use mdb_com_gnu script with correct setups\n");
#else
//Metaware tools    
        printf("App command line:\n"
                "\t%s \n\t\tProcess single hardcoded vector\n\n"
                "\t%s <input_test_base.idx> \n\t\tProcess testset from file and \n"
                "\t\t output model results to <input_test_base.idx_out> file\n\n", argv[0], argv[0]);
#endif // if defined (__GNUC__) && !defined (__CCAC__)                 
        return 2; //Error: mode not set       
    }  
    
    for(int i=0; i < mode; i++) {
       if (param[i][0] == 0){
           printf("param[%d][0] not set.\n", i);
           if (i==0) printf("Please set up dummy string for check.\n");
           if (i==1) printf("Please set up input IDX file.\n");
           if (i==2) printf("Please set up labels IDX file.\n");
           return 2; //Error: param not set
       }
    }

    mli_status status = emnist_init();
    if (status != MLI_STATUS_OK) {
    	printf("Failed to initialize lut for softmax\n");
    	return 2; // Error: lut couldn't be initialized
    }

    switch (mode) {
    // No Arguments for app. Process single hardcoded input
    // Print various measures to stdout
    //=========================================================
    case 1:
        printf("HARDCODED INPUT PROCESSING\n");
        model_run_single_in(kSingleIn, kSingleOutRef,
                emnist_cf_net_input, emnist_cf_net_output,
                emnist_preprocessing, run_emnist_cf_net, "NN Performance");
        break;

    // APP <input_test_base.idx>
    // Output vectors will be written to <input_test_base.idx_out> file
    //=================================================================
    case 2:
        printf("Input IDX testset to output IDX set\n");
        char * out_path = malloc(strlen(param[1]) + strlen(kOutFilePostfix) + 1);
        if (out_path == NULL) {
            printf("mem allocation failed\n");
            break;
        }
        out_path[0] = 0;
        strcat(out_path, param[1]);
        strcat(out_path, kOutFilePostfix);

        model_run_idx_base_to_idx_out(param[1], out_path,
                emnist_cf_net_input, emnist_cf_net_output,
                emnist_preprocessing, run_emnist_cf_net, NULL);
        free(out_path);
        break;

    // APP <input_test_base.idx> <input_test_labels.idx>
    // Calculate accuracy of the model
    //=================================================================
    case 3:
        printf("ACCURACY CALCULATION on Input IDX testset according to IDX labels set\n");
        model_run_acc_on_idx_base(param[1], param[2],
                emnist_cf_net_input, emnist_cf_net_output,
                emnist_preprocessing, run_emnist_cf_net, NULL);
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
static void emnist_preprocessing(const void* image_, mli_tensor* net_input_) {
    const unsigned char* in = image_;
    d_type* const dst = (d_type * const)net_input_->data.mem.D_FIELD;
    for (int idx = 0; idx < IN_POINTS; idx++) {
        dst[idx] = (d_type)((int)in[idx]);
    }
}


void run_emnist_cf_net(const char * info_str) {
    PROFILE(emnist_cf_net());
    if (info_str != NULL) {
        printf("\n%s\n\tTotal: %u cycles \n\n", info_str, cycle_cnt);
    }
}