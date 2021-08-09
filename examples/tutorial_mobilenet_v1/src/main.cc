/*
 * Copyright 2021, Synopsys, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-3-Clause license found in
 * the LICENSE file in the root directory of this source tree.
 *
 */
#include "jpgd.h"
#include "mobilenet_labels.h"
#include "model.h"
#include "model_settings.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define IMAGE_SIDE 128
#define NUM_CHANNELS 3
#define IMAGE_SIZE (IMAGE_SIDE * IMAGE_SIDE * NUM_CHANNELS)
#define EXAMPLES_NUM 5

const char* image_paths[EXAMPLES_NUM] = {"img/531.jpg", "img/699.jpg", 
                                         "img/795.jpg", "img/867.jpg", 
                                         "img/908.jpg",};

// tensor_arena has to be 16 bytes aligned
// typedef uint8_t aligned_uint8_t __attribute__((aligned(16)));
constexpr int __attribute__((aligned(16))) kTensorArenaSize = 258 * 1024;
uint8_t tensor_arena[kTensorArenaSize] = {0};

// Globals
namespace {
tflite::ErrorReporter* reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

float input_scale = 0.0;
int32_t input_zero_point = 0;
}  // namespace

TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter,
                      const char* img_file_path, int image_width,
                      int image_height, int image_channels, int8_t* image_data,
                      float scale, int32_t zero_point) {
  int width = 0;
  int height = 0;
  int channels = 0;
  unsigned char* image_buf = jpgd::decompress_jpeg_image_from_file(
      img_file_path, &width, &height, &channels, 3);
  assert(image_buf != NULL && width == image_width && height == image_height &&
         channels == image_channels);

  for (int i = 0; i < IMAGE_SIZE; i++) {
    float preprocessed_image_buf = (float)image_buf[i] / 127.5f - 1.0f;
    image_data[i] = int8_t(preprocessed_image_buf / scale) + int8_t(zero_point);
  }
  free(image_buf);

  return kTfLiteOk;
}

void loop() {
  reporter->Report("Running MobileNet v1");
  for (int i = 0; i < EXAMPLES_NUM; i++) {
    if (kTfLiteOk != GetImage(reporter, image_paths[i], kNumCols, kNumRows,
                              kNumChannels, input->data.int8, input_scale,
                              input_zero_point)) {
      TF_LITE_REPORT_ERROR(reporter, "Image capture failed.");
    }
    // Run the model on this input and make sure it succeeds.
    if (kTfLiteOk != interpreter->Invoke()) {
      TF_LITE_REPORT_ERROR(reporter, "Invoke failed.");
    }

    output = interpreter->output(0);
    int8_t* results_ptr = output->data.int8;
    int output_size = output->dims->data[output->dims->size - 1];
    int result = std::distance(
                     results_ptr,
                     std::max_element(results_ptr, results_ptr + output_size)) -
                 1;
    if (result < 1000 && result > -1) {
      printf("Result: %d (%s).\n", result, test_labels[result + 1]);
    } else {
      TF_LITE_REPORT_ERROR(reporter, "Bad result.");
    }
  }
}

int main() {
  static tflite::MicroErrorReporter micro_error_reporter;
  reporter = &micro_error_reporter;

  // Load Model
  model = tflite::GetModel(mobilenet_v1_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    reporter->Report(
        "Model is schema version: %d\n"
        "Supported schema version is: %d",
        model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  // Setup OpResolver
  // Add Builtins corresponding to Model layers
  // Note: If you change the model structure/layer types, you'll need to make
  // equivalent changes to the resolver
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAveragePool2D();
  resolver.AddReshape();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    reporter->Report("AllocateTensors() failed");
    return 1;
  }

  input = interpreter->input(0);
  input_scale = input->params.scale;
  input_zero_point = input->params.zero_point;
  while (true) {
    loop();
  }

  return 0;
}
