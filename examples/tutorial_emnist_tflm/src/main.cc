/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model_settings.h"
#include "model.h"
#include "test_samples.h"

//tensor_arena has to be 16 bytes aligned
typedef uint8_t aligned_uint8_t __attribute__((aligned(16)));
constexpr int kTensorArenaSize = 50 * 1024; 
aligned_uint8_t tensor_arena[kTensorArenaSize] = {0};

int main() {
    tflite::ErrorReporter* reporter = nullptr;
    tflite::MicroErrorReporter error_reporter;
    reporter = &error_reporter;
    reporter->Report("Run EMNIST letter recognition");

    //Load Model
    const tflite::Model* model = tflite::GetModel(emnist_model_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        reporter->Report( "Model is schema version: %d\n"
          "Supported schema version is: %d", model->version(), TFLITE_SCHEMA_VERSION );
        return 1;
    }

    // Setup OpResolver 
    // Add Builtins corresponding to Model layers
    // Note: If you change the model structure/layer types, you'll need to make 
    // equivalent changes to the resolver
    tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();

    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, reporter);

    // Allocate memory from the tensor_arena for the model's tensors.
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        reporter->Report( "AllocateTensors() failed" );
        return 1;
    }

    // Obtain pointers to the model's input and output tensors.
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    
    // Obtain quantization parameters for result dequantization
    float scale = input->params.scale;
    int32_t zero_point = input->params.zero_point;

    // Invoke interpreter for each test sample and process results
    for (int j = 0; j < kNumSamples; j++) {
        // Perform image thinning (round values to either -128 or 127)
        // Write image to input data
        for (int i = 0; i < kImageSize; i++) {
          input->data.int8[i] = (test_samples[j].image[i] <= 210) ? -128 : 127;
        }

        // Run model
        if (interpreter.Invoke() != kTfLiteOk) {
            reporter->Report("Invoke failed");
            return 1;
        }

        // Get max result from output array and calculate confidence
        int8_t* results_ptr = output->data.int8;
        int result = std::distance(results_ptr, std::max_element(results_ptr, results_ptr + 26));
        float confidence = ((results_ptr[result] - zero_point)*scale + 1) / 2;
        const char *status = result == test_samples[j].label ? "SUCCESS" : "FAIL";

        reporter->Report("Test sample \"%s\":\n"
          "Predicted %s (%d%%) - %s\n",
          kCategoryLabels[test_samples[j].label], 
          kCategoryLabels[result], (int)(confidence * 100), status);
    }
    return 0;
}
