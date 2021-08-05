/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#include "examples_aux.h"

#include <assert.h>
#include <stdlib.h>

#include "idx_file.h"
#include "tensor_transform.h"
#include "mli_api.h"

// Find maximum value in the whole tensor and return it's argument (index)
// Tensor data considered as linear array. Index corresponds to number of element in this array
static inline int arg_max(mli_tensor * net_output_);

// Return label (int) stored in label container.
// Function casts label_container_ to appropriate type according to type_ and return it's value as integer
// It returns -1 if type is unknown.
static inline int get_label(void * label_container_, tIdxDataType type_);


//========================================================================================
// Single vector processing for debug
//========================================================================================
test_status model_run_single_in(
        const void * data_in,
        const float * ref_out,
        mli_tensor * model_input,
        mli_tensor * model_output,
        preproc_func_t preprocess,
        model_inference_t inference,
        const char * inf_param) {
    test_status ret_val = TEST_PASSED;
    size_t output_elements = mli_hlp_count_elem_num(model_output, 0);

    float * pred_data = malloc(output_elements * sizeof(float));
    if (pred_data == NULL) {
        printf("ERROR: Can't allocate memory for output\n");
        return TEST_NOT_ENOUGH_MEM;
    }

    mli_tensor pred_tensor = *model_output;
    pred_tensor.data.mem.pf32 = pred_data;
    pred_tensor.data.capacity = output_elements * sizeof(float);
    pred_tensor.el_type = MLI_EL_FP_32;

    // Data preprocessing and model inference
    preprocess(data_in, model_input);
    inference(inf_param);

    // Check result
    if (MLI_STATUS_OK == mli_hlp_convert_tensor(model_output, &pred_tensor)) {
        ref_to_pred_output err;
        measure_err_vfloat(ref_out, pred_data, output_elements, &err);
        printf("Result Quality: S/N=%-10.1f (%-4.1f db)\n", err.ref_vec_length / err.noise_vec_length, err.ref_to_noise_snr);
    } else {
        printf("ERROR: Can't transform out tensor to float\n");
        ret_val = TEST_SUIT_ERROR;
    }
    free(pred_data);
    return ret_val;
}


//========================================================================================
// Multiple inputs from IDX file processing. Writes output for each tensor into output file
//========================================================================================
test_status model_run_idx_base_to_idx_out(
        const char * input_idx_path,
        const char * output_idx_path,
        mli_tensor * model_input,
        mli_tensor * model_output,
        preproc_func_t preprocess,
        model_inference_t inference,
        const char * inf_param) {
    test_status ret_val = TEST_PASSED;
    tIdxDescr descr_in = {0, 0, 0, NULL};
    tIdxDescr descr_out = {0, 0, 0, NULL};
    uint32_t shape[4] = {0, 0, 0, 0};
    void * input_data = NULL;
    float * output_data = NULL;
    size_t output_elements = mli_hlp_count_elem_num(model_output, 0);
    size_t input_elements = mli_hlp_count_elem_num(model_input, 0);

    // Step 1: Resources preparations
    //================================================
    //Open and check input file
    if ((descr_in.opened_file = fopen(input_idx_path, "rb")) == NULL ||
            (idx_file_check_and_get_info(&descr_in)) != IDX_ERR_NONE ||
            descr_in.num_dim != model_input->rank + 1) {
        printf("ERROR: Problems with input idx file format.\n Requirements:\n"
                "\t tensor rank must be equal to model input rank + 1\n");
        ret_val =  TEST_SUIT_ERROR;
        goto free_ret_lbl;
    }

    // Read test base shape
    descr_in.num_elements = 0;
    if (idx_file_read_data(&descr_in, NULL, shape) != IDX_ERR_NONE) {
        printf("ERROR: Can't read input file shape\n");
        ret_val =  TEST_SUIT_ERROR;
        goto free_ret_lbl;
    }

    // Check compatibility between shapes of idx file and model input
    printf("IDX test file shape: [");
    for (int i = 0; i < descr_in.num_dim; i++) printf("%d,", shape[i]);
    printf("]\nModel input shape: [");
    for (int i = 0; i < model_input->rank; i++) printf("%d,", model_input->shape[i]);
    printf("]\n\n");
    for (int i = 1; i < descr_in.num_dim; i++)
        if ( shape[i] != model_input->shape[i-1]) {
            printf("ERROR: Shapes mismatch.\n");
            ret_val =  TEST_SUIT_ERROR;
            goto free_ret_lbl;
        }

    // Memory allocation for input/output (for it's external representations)
    input_data = malloc((input_elements * data_elem_size(descr_in.data_type)) + (output_elements * sizeof(float)));
    output_data = (float *)((char*)input_data + input_elements * data_elem_size(descr_in.data_type));
    if (input_data == NULL) {
        printf("ERROR: Can't allocate memory for input and output\n");
        ret_val =  TEST_NOT_ENOUGH_MEM;
        goto free_ret_lbl;
    }

    // Open output file
    if ( (descr_out.opened_file = fopen(output_idx_path, "wb")) == NULL) {
        printf("ERROR: Can't open output idx file\n");
        ret_val =  TEST_SUIT_ERROR;
        goto free_ret_lbl;
    }


    // Step 2: Process vectors from input file one-by-another
    //================================================
    descr_out.data_type = IDX_DT_FLOAT_4B;
    descr_out.num_dim = model_output->rank + 1;
    uint32_t test_idx = 0;
    for (; test_idx < shape[0]; test_idx++) {
        // Get next input vector from file
        descr_in.num_elements = input_elements;
        if (idx_file_read_data(&descr_in, input_data, NULL) != IDX_ERR_NONE) {
            printf("ERROR: While reading test vector %u\n", test_idx);
            ret_val =  TEST_SUIT_ERROR;
            goto free_ret_lbl;
        }

        // Model inference for the vector
        preprocess(input_data, model_input);
        inference(inf_param);

        // Output results to idx file
        descr_out.num_elements = output_elements;
        if (mli_hlp_fx_tensor_to_float(model_output, output_data, output_elements) != MLI_STATUS_OK ||
                idx_file_write_data(&descr_out, (const void *)output_data) != IDX_ERR_NONE) {
            printf("ERROR: While writing result for test vector %u\n", test_idx);
            ret_val =  TEST_SUIT_ERROR;
            goto free_ret_lbl;
        }

        // Notify User on progress (10% step)
        if (test_idx % (shape[0] / 10) == 0)
            printf("%10u of %u test vectors are processed\n", test_idx, shape[0]);
    }

    // Step 3: Fill output file header and free resources
    //==================================================
    shape[0] = test_idx;
    for (int i = 0; i < model_output->rank; i++)
        shape[i+1] = model_output->shape[i];
    if (idx_file_write_header(&descr_out, shape) != IDX_ERR_NONE) {
        printf("ERROR: While final header writing of test out file \n");
        ret_val =  TEST_SUIT_ERROR;
    }

free_ret_lbl:
    if (input_data != NULL)
        free(input_data);
    if (descr_in.opened_file != NULL)
        fclose(descr_in.opened_file);
    if (descr_out.opened_file != NULL)
        fclose(descr_out.opened_file);
    return ret_val;
}


//========================================================================================
// Multiple inputs from IDX file processing. Calculate accuracy
//========================================================================================
test_status model_run_acc_on_idx_base(
        const char * input_idx_path,
        const char * labels_idx_path,
        mli_tensor * model_input,
        mli_tensor * model_output,
        preproc_func_t preprocess,
        model_inference_t inference,
        const char * inf_param) {
    test_status ret = TEST_PASSED;
    tIdxDescr descr_in = {0, 0, 0, NULL};
    tIdxDescr descr_labels = {0, 0, 0, NULL};
    uint32_t shape[4] = {0, 0, 0, 0};
    uint32_t labels_total = 0;
    uint32_t labels_correct = 0;
    int label = 0;
    size_t input_elements = mli_hlp_count_elem_num(model_input, 0);
    void * input_data = NULL;

    // Step 1: Resources preparations
    //================================================
    //Open and check input labels file
    if ((descr_labels.opened_file = fopen(labels_idx_path, "rb")) == NULL ||
            (idx_file_check_and_get_info(&descr_labels)) != IDX_ERR_NONE ||
            descr_labels.data_type == IDX_DT_FLOAT_4B ||
            descr_labels.data_type == IDX_DT_DOUBLE_8B ||
            descr_labels.num_dim != 1) {
        printf("ERROR: Problems with labels idx file format.\n Requirements:\n"
                "\t Non-float format\n"
                "\t 1 dimensional tensor\n"
                );
        ret =  TEST_SUIT_ERROR;
        goto free_ret_lbl;
    }

    // Read labels shape
    descr_labels.num_elements = 0;
    if (idx_file_read_data(&descr_labels, NULL, shape) != IDX_ERR_NONE) {
        printf("ERROR: Problems with input idx file format.\n Requirements:\n"
                "\t tensors shape must be [N], where N is amount of tests)\n");
        ret =  TEST_SUIT_ERROR;
        goto free_ret_lbl;
    }
    labels_total = shape[0];

    //Open and check input test idxfile
    if ((descr_in.opened_file = fopen(input_idx_path, "rb")) == NULL ||
            (idx_file_check_and_get_info(&descr_in)) != IDX_ERR_NONE ||
            descr_in.num_dim != model_input->rank + 1) {
        printf("ERROR: Problems with input idx file format.\n Requirements:\n"
                "\t tensor rank must be equal to model input rank + 1\n");
        ret =  TEST_SUIT_ERROR;
        goto free_ret_lbl;
    }

    // Read test base shape
    descr_in.num_elements = 0;
    if (idx_file_read_data(&descr_in, NULL, shape) != IDX_ERR_NONE) {
        printf("ERROR: Can't read input file shape\n");
        ret =  TEST_SUIT_ERROR;
        goto free_ret_lbl;
    }

    // Check compatibility between shapes of idx file and model input
    printf("IDX test file shape: [");
    for (int i = 0; i < descr_in.num_dim; i++) printf("%d,", shape[i]);
    printf("]\nModel input shape: [");
    for (int i = 0; i < model_input->rank; i++) printf("%d,", model_input->shape[i]);
    printf("]\n\n");
    for (int i = 1; i < descr_in.num_dim; i++)
        if (shape[i] != model_input->shape[i-1]) {
            printf("ERROR: Shapes mismatch.\n");
            ret =  TEST_SUIT_ERROR;
            goto free_ret_lbl;
        }

    if (shape[0] != labels_total) {
        printf("ERROR: Amount of labels(%d) and test inputs(%d) are not the same\n", labels_total, shape[0]);
        ret =  TEST_SUIT_ERROR;
        goto free_ret_lbl;
    }

    // Memory allocation for raw input
    input_data = malloc(input_elements * data_elem_size(descr_in.data_type));
    if (input_data == NULL) {
        printf("ERROR: Can't allocate memory for input\n");
        ret =  TEST_NOT_ENOUGH_MEM;
        goto free_ret_lbl;
    }

    // Step 2: Process vectors from input file one-by-another
    //================================================
    uint32_t test_idx = 0;
    for (; test_idx < labels_total; test_idx++) {
        // Get next input vector from file and related label
        descr_in.num_elements = input_elements;
        descr_labels.num_elements = 1;
        if (idx_file_read_data(&descr_in, input_data, NULL) != IDX_ERR_NONE ||
                idx_file_read_data(&descr_labels, (void *)&label, NULL) != IDX_ERR_NONE) {
            printf("ERROR: While reading idx files content #%u\n", test_idx);
            ret =  TEST_SUIT_ERROR;
            goto free_ret_lbl;
        }
        label = get_label(&label, descr_labels.data_type);

        // Model inference
        preprocess(input_data, model_input);
        inference(inf_param);

        labels_correct += (arg_max(model_output) == label)? 1: 0;

        // Notify User on progress (10% step)
        if (((test_idx+1) % (labels_total / 10)) == 0)
            printf("%10u of %u test vectors are processed (%u are correct: %.3f %%)\n",
                    test_idx+1, labels_total, labels_correct, (labels_correct*100.f)/(test_idx+1));
    }
    printf("Final Accuracy: %.3f %% (%u are correct of %u)\n",
            (labels_correct*100.f)/test_idx, labels_correct, labels_total);

    // Step 3: Fill output file header and free resources
    //==================================================
free_ret_lbl:
    if (input_data != NULL)
        free(input_data);
    if (descr_in.opened_file != NULL)
        fclose(descr_in.opened_file);
    if (descr_labels.opened_file != NULL)
        fclose(descr_labels.opened_file);
    return ret;
}


//========================================================================================
//
// Internal routines
//
//========================================================================================

//=================================================
// Find argument (index) of maximum value in tensor
//=================================================
static inline int arg_max(mli_tensor * net_output_) {
    
    const mli_argmax_cfg argmax_cfg = {
        /* axis = */ -1,
        /* topk = */ 1
    };

    int pred_label = 0;
    mli_tensor out_tensor;
    out_tensor.data.mem.pi32 = &pred_label;
    out_tensor.el_type = MLI_EL_SA_32;
    out_tensor.rank = 2;
    out_tensor.shape[0] = 1;
    out_tensor.shape[1] = 1;
    out_tensor.mem_stride[0] = 1;
    out_tensor.mem_stride[1] = 1;

    if (net_output_->el_type == MLI_EL_SA_8)
        mli_krn_argmax_sa8(net_output_, &argmax_cfg, &out_tensor);
    else
        mli_krn_argmax_fx16(net_output_, &argmax_cfg, &out_tensor);

    return pred_label;
}

//=================================================
// Label type cast
//=================================================
static inline int get_label(void * label_container_, tIdxDataType type_) {
    switch(type_) {
    case IDX_DT_UBYTE_1B:
        return (int)*((unsigned char *)label_container_);

    case IDX_DT_BYTE_1B:
        return (int)*((char *)label_container_);

    case IDX_DT_SHORT_2B:
        return (int)*((short*)label_container_);

    case IDX_DT_INT_4B:
        return (int)*((int*)label_container_);

    default:
        return -1;
    }
    return -1;
}
