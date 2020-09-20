/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_USER_TESTS_TEST_QUANTIZER_H_
#define _MLI_USER_TESTS_TEST_QUANTIZER_H_

#include <stdint.h>

#include <mli_api.h>

namespace mli {
namespace tst {
//======================================================================================================
// Quantizer class: implies to quantize float data to MLI compatible data using quantization parameters.
//======================================================================================================
class tensor_quantizer {
 public:

    enum tensor_state {
        kOk = 0,
        kBad,
        kIncompleteMem,
    };

    // Consuctor gets all necessairy inputs for quantization
    // Tensor parameters are expected to be complete enough for quantization and output valid tensor.
    // For FX types next fields must be filled: Shape, rank, memstride, el_type, el_params.fx.frac_bits
    // For SA types next fields must be filled: Shape, rank, memstride, el_type, el_params.sa.dim, el_params.sa.scale_frac_bits, 
    tensor_quantizer(mli_tensor tsr, const float *data, uint32_t data_size, const float *scales, uint32_t scales_size,
                     const float* zero_points, uint32_t zero_points_size);

    tensor_quantizer(mli_tensor tsr, const float* data, uint32_t data_size);

    //tensor_quantizer() = delete; // not sure we need it
    //tensor_quantizer& operator=(const tensor_quantizer& other) = default; // copy assignment


    // I think we don't need a reset since it can be done using copy constructor or operator. Or Move
    //quantizer_status reset(mli_tensor tsr, const float* data, uint32_t data_size,
    //                       const float* scales = nullptr, uint32_t scales_size = 0,
    //                       const float* zero_points = nullptr, uint32_t zero_points_size = 0);
    
    mli_tensor get_quantized_tensor(mli_data_container memory) const;
    mli_tensor get_not_quantized_tensor(mli_data_container memory) const;
    const mli_tensor get_source_float_tensor() const;
    
    bool is_valid() const { return is_valid_; }
    
    // This function intend to validate state of quantizer instance output tensors only.
    // It returns status of returned tensor according to knowledge on how it was filled:
    // Bad tensor, without assigned memory but with memory requirements (kNotEnoughMemory),
    // or tensor is ok (kOk)
    static tensor_state validate_tensor(const mli_tensor& tsr);

    //static quantizer_status dequantize(const mli_tensor& input, mli_tensor* out_fp);

    // Read function always returns the requested number of sample. 
    // If there are not enough samples in stream, it fills the rest with zeroes (status is changed to EOF).
    //uint32_t read_samples(void* samples, const uint32_t num);
    //const wav_file_status &status() const {return status_;}
    //const waveheader_t &header() const {return header_;}


 private:

     static uint32_t get_required_data_capacity(const mli_tensor& tsr);
     static bool  tensor_assign_data_ptr(mli_tensor* tsr, void* ptr);
     static bool spread_memory(mli_tensor* tsr, const mli_data_container* data_mem,
                       const mli_data_container* scales_mem = nullptr, const  mli_data_container* zp_mem = nullptr);


    // mli_tensor output_tsr_; 
    const mli_tensor source_tsr_;
    const float* source_data_;
    const float* source_scales_;
    const float* source_zero_points_;
    bool is_valid_;
    //bool output_tsr_quantized;
};

} // namespace tst
} // namespace mli

#endif //_MLI_USER_TESTS_TEST_QUANTIZER_H_
