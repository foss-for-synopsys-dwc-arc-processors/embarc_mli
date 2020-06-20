#include <cstdint>
#include "model_settings.h"

struct TestSample {
  uint8_t image[kImageSize];
  int label;
};

extern const int kNumSamples;
extern const TestSample test_samples[];