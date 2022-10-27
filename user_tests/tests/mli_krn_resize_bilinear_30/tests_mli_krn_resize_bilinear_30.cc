/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#include <cmath>
#include "mli_ref_runtime_api.hpp"
#include "test_report.h"
#include "tests_aux.h"
#include "vectors_mli_krn_resize_bilinear_30.inc"

using lib_mli::kResizeBilinearRank;
using lib_mli::kTensorBatchDim;
using lib_mli::kTensorHeightDim;
using lib_mli::kTensorWidthDim;
using lib_mli::kTensorChannelDim;
using mli::tst::reporter_basic;

constexpr uint32_t k_input_h = 62;
constexpr uint32_t k_input_w = 208;
constexpr uint32_t num_h_strides_cases = 4;
static const uint32_t output_h_array[num_h_strides_cases]{ k_input_h / 2, k_input_h * 2, k_input_h * 4, k_input_h * 8};
constexpr uint32_t num_w_strides_cases = 4;
static const uint32_t output_w_array[num_w_strides_cases]{ k_input_w * 2, k_input_w / 2, k_input_w / 4, k_input_w / 8 };

constexpr int16_t k_sa_io_scale = 32767;
constexpr int16_t k_sa_io_zp = -128;
constexpr int8_t k_sa_io_frac_bits = 15;
constexpr int8_t k_shift = 10;
constexpr uint32_t k_max_num_input_image_elements = k_input_h * k_input_w;
static int8_t g_mem_input_values_quantized[k_max_num_input_image_elements];
constexpr uint32_t k_max_num_output_image_elements = k_input_h * 8 * k_input_w * 2;
static int8_t g_mem_output[k_max_num_output_image_elements * sizeof(int32_t)];
static int8_t g_mem_output_quantized_rescaled[k_max_num_output_image_elements];
static float g_mem_reference_output[k_max_num_output_image_elements];

void apply_tensor_rshift(int32_t* src, uint32_t n, int8_t shift, int8_t* dst) {
  for (unsigned i = 0; i < n; i++) {
    dst[i] = src[i] / (1 << shift);
  }
}

void sa8_to_float(int8_t* src, uint32_t n, int16_t k_sa_io_scale, int16_t k_sa_io_zp, int8_t k_sa_io_frac_bits, float* dst) {
  float scale_fl = (float)k_sa_io_scale / (float)(1 << k_sa_io_frac_bits);
  float zp_fl = (float)k_sa_io_zp;
  for (uint32_t i = 0; i < n; i++) {
    dst[i] = ((float)src[i] - zp_fl) * scale_fl;
  }
}
void float_to_sa8(const float* src, uint32_t n, int16_t k_sa_io_scale, int16_t k_sa_io_zp, int8_t k_sa_io_frac_bits, int8_t* dst) {
  float scale_fl = (float)k_sa_io_scale / (float)(1 << k_sa_io_frac_bits);
  float zp_fl = (float)k_sa_io_zp;
  for (uint32_t i = 0; i < n; i++) {
    dst[i] = (int8_t)roundf(src[i] / scale_fl + zp_fl);
  }
}

mli_status reference_resize_bilinear_float(const float* input, float* output, const float strides[2], int Hi, int Wi, int Ho, int Wo) {

  int input_row0, input_row1, input_col0, input_col1;
  float v00, v01, v10, v11;
  float out_val;
  int h, w;
  float dx, dy;
  for (h = 0; h < Ho; h++) {
    input_row0 = MIN(MAX(floor(h * strides[0]), 0.f), Hi - 1);
    input_row1 = MIN(input_row0 + 1, Hi - 1);
    for (w = 0; w < Wo; w++) {
      input_col0 = MIN(MAX(floor(w * strides[1]), 0.f), Wi - 1);
      input_col1 = MIN(input_col0 + 1, Wi - 1);
        // read the nearest 4 input values around the output point
        v00 = input[input_row0 * Wi + input_col0];
        v01 = input[input_row0 * Wi + input_col1];
        v10 = input[input_row1 * Wi + input_col0];
        v11 = input[input_row1 * Wi + input_col1];
        dy = strides[0] * h - input_row0;
        dx = strides[1] * w - input_col0;

        // compute and write output point
        out_val = v00 * (1.f - dy) * (1.f - dx) +
                  v01 * (1.f - dy) * dx +
                  v10 * dy         * (1.f - dx) +
                  v11 * dy         * dx;
        output[h * Wo + w] = out_val;
    }
  }

  return MLI_STATUS_OK;
}

void prepare_phase(uint32_t output_h, uint32_t output_w, mli_tensor& input_tensor,
                   lib_mli::ResizeOpConfig& cfg, mli_tensor& output_tensor) {
  input_tensor.rank = kResizeBilinearRank;
  input_tensor.shape[kTensorBatchDim] = 1;
  input_tensor.shape[kTensorHeightDim] = k_input_h;
  input_tensor.shape[kTensorWidthDim] = k_input_w;
  input_tensor.shape[kTensorChannelDim] = 1;
  mli_hlp_set_tensor_mem_strides(&input_tensor);
  input_tensor.el_type = MLI_EL_SA_8;
  input_tensor.data.mem.pi8 = g_mem_input_values_quantized;
  input_tensor.data.capacity = mli_hlp_count_elem_num(&input_tensor, 0) * sizeof(int8_t);
  input_tensor.el_params.sa.dim = -1;
  input_tensor.el_params.sa.type = MLI_EL_PARAM_SC16_ZP16;
  input_tensor.el_params.sa.scale.mem.i16 = k_sa_io_scale;
  input_tensor.el_params.sa.scale_frac_bits.mem.i8 = k_sa_io_frac_bits;
  input_tensor.el_params.sa.zero_point.mem.i16 = k_sa_io_zp;

  output_tensor.rank = kResizeBilinearRank;
  output_tensor.shape[kTensorBatchDim] = 1;
  output_tensor.shape[kTensorHeightDim] = output_h;
  output_tensor.shape[kTensorWidthDim] = output_w;
  output_tensor.shape[kTensorChannelDim] = 1;
  mli_hlp_set_tensor_mem_strides(&output_tensor);
  output_tensor.el_type = MLI_EL_SA_32;
  output_tensor.data.mem.pi8 = g_mem_output;
  output_tensor.data.capacity = mli_hlp_count_elem_num(&output_tensor, 0) * sizeof(int32_t);
  output_tensor.el_params.sa.dim = -1;
  output_tensor.el_params.sa.type = MLI_EL_PARAM_SC16_ZP16;
  output_tensor.el_params.sa.scale.mem.i16 = k_sa_io_scale;
  output_tensor.el_params.sa.scale_frac_bits.mem.i8 = k_sa_io_frac_bits;
  output_tensor.el_params.sa.zero_point.mem.i16 = k_sa_io_zp;

  int16_t strides[2]{(int16_t)roundf((float)((input_tensor.shape[kTensorHeightDim] - 1) << k_shift) / (float)(output_tensor.shape[kTensorHeightDim] - 1)),
                     (int16_t)roundf((float)((input_tensor.shape[kTensorWidthDim] - 1) << k_shift) / (float)(output_tensor.shape[kTensorWidthDim] - 1))};
  int16_t offsets[2] = { 0, 0 };
  cfg = lib_mli::ResizeOpConfig(strides, offsets, k_shift);
}


void execution_phase(mli_tensor& input_tensor, lib_mli::ResizeOpConfig& cfg, mli_tensor& output_tensor) {
  snps_arc::metaware::mli::ref::run_mli_resize_bilinear_standalone(&input_tensor, cfg, &output_tensor);
}

bool postprocess_phase(const reporter_basic& reporter, uint32_t output_h, uint32_t output_w, uint32_t n_test_case,
                       mli_tensor& input_tensor, lib_mli::ResizeOpConfig& cfg, mli_tensor& output_tensor) {

  // dequantize output of fx-algorithm
  uint32_t num_o_elem = mli_hlp_count_elem_num(&output_tensor, 0);
  apply_tensor_rshift(output_tensor.data.mem.pi32, num_o_elem, cfg.shift * 2, g_mem_output_quantized_rescaled);
  sa8_to_float(g_mem_output_quantized_rescaled, num_o_elem, k_sa_io_scale, k_sa_io_zp, k_sa_io_frac_bits, (float*) g_mem_output);

  // get float output of reference float algorithm
  const float strides_float[2]{
    (float)(k_input_h - 1) / (float)(output_h - 1),
    (float)(k_input_w - 1) / (float)(output_w - 1)
  };
  reference_resize_bilinear_float(g_mem_input_float, g_mem_reference_output, strides_float, 
                                   k_input_h, k_input_w, output_h, output_w);


  // compare dequantized output of fx-algorithm with float output of reference float algorithm
  ref_to_pred_output metrics;
  test_status status = measure_err_vfloat(g_mem_reference_output, (float*)g_mem_output, num_o_elem, &metrics);
  bool passed = status == TEST_PASSED && metrics.ref_to_noise_snr > 44.f;
  char descr[256]{};
  sprintf(descr, "Test %d %dx%d -> %dx%d", n_test_case, k_input_h, k_input_w, output_h, output_w);
  char message[256]{};
  sprintf(message, "MaxErr = %.3f, SNR = %f", metrics.max_abs_err, metrics.ref_to_noise_snr);
  reporter.report_case(descr, message, passed);
  return passed;
}

int main(){
  const reporter_basic reporter;
  reporter.report_header("MLI3.0|Kernels|Resize Bilinear Function Tests");

  float_to_sa8(g_mem_input_float, k_max_num_input_image_elements, k_sa_io_scale, k_sa_io_zp, k_sa_io_frac_bits, g_mem_input_values_quantized);

  bool final_status = true;
  for (unsigned i = 0; i < num_h_strides_cases; i++) {
    for (unsigned j = 0; j < num_w_strides_cases; j++) {
      unsigned n_test_case = 1 + i * num_w_strides_cases + j;

      mli_tensor input_tensor;
      lib_mli::ResizeOpConfig cfg;
      mli_tensor output_tensor;
      prepare_phase(output_h_array[i], output_w_array[j], input_tensor, cfg, output_tensor);

      execution_phase(input_tensor, cfg, output_tensor);

      final_status &= postprocess_phase(reporter, output_h_array[i], output_w_array[j], n_test_case, input_tensor, cfg, output_tensor);

    }
  }

  reporter.report_outline("[AUTO] Group: mli_krn_resize_bilinear_30", final_status);
  return 0;
}