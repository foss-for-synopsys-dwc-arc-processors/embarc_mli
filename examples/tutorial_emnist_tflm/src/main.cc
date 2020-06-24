#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model_settings.h"
#include "model.h"
#include "test_samples.h"

// Globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::ErrorReporter* reporter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 50 * 1024; 
uint8_t tensor_arena[ kTensorArenaSize ] = { 0 };

int main() {
    static tflite::MicroErrorReporter error_reporter;
    reporter = &error_reporter;
    reporter->Report( "Run EMNIST letter recognition" );

    //Load Model
    model = tflite::GetModel( emnist_model_int8_tflite );
    if( model->version() != TFLITE_SCHEMA_VERSION ) {
        reporter->Report( "Model is schema version: %d\nSupported schema version is: %d", model->version(), TFLITE_SCHEMA_VERSION );
        return 1;
    }

    // Setup OpResolver 
    // Add Builtins corresponding to Model layers
    static  tflite::MicroOpResolver<12> resolver;
    resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
      tflite::ops::micro::Register_CONV_2D(), 1, 3);
    resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
      tflite::ops::micro::Register_MAX_POOL_2D(),
      1, 2);
    resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
      tflite::ops::micro::Register_RESHAPE());
    resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
      tflite::ops::micro::Register_FULLY_CONNECTED(), 1, 4);
    resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
      tflite::ops::micro::Register_SOFTMAX(), 1, 2);

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, reporter );
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if( allocate_status != kTfLiteOk ) {
        reporter->Report( "AllocateTensors() failed" );
        return 1;
    }

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // Obtain quantization parameters for result dequantization
    float scale = input->params.scale;
    int32_t zero_point = input->params.zero_point;

    // Invoke interpreter for each test sample and process results
    for (int j = 0; j < kNumSamples; j++){
        // Perform image thinning (round values to either -128 or 127)
        // Write image to input data
        for (int i = 0; i < kImageSize; i++) {
          input->data.int8[i] = (test_samples[j].image[i] <= 210) ? -128 : 127;
         }

        // Run model
        TfLiteStatus invoke_status = interpreter->Invoke();
        if( invoke_status != kTfLiteOk ) {
            reporter->Report( "Invoke failed" );
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