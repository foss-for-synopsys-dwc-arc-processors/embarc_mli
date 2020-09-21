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
                     const float* zero_points, uint32_t zero_points_size, 
                     const int8_t* scales_fraq_bits, uint32_t scales_fraq_bits_size);
    tensor_quantizer(mli_tensor tsr, const float* data, uint32_t data_size);

    // Get mli_tensor with assigned memory only
    //
    // takes mli_data_container with memory to be spread across data and quantization params
    // returns: 1) mli_tensor with ASSIGNED memory and prepared quant params memory, if quantizer instance 
    //             was properly initialized and input data container contains enough memory, 
    //          2) mli_tensor with memory requirements in capacity fields of data container 
    //              if inputs contains not enough memory
    //          3) empty mli_tensor if quantizer instance was badly initialized    
    mli_tensor get_not_quantized_tensor(mli_data_container memory) const;

    // Get mli_tensor with quantized data
    //
    // takes mli_data_container with memory to be used for data and quantization params
    // returns: 1) mli_tensor with QUANTIZED data and prepared quant params memory according to 
    //             kept tensor and quant params, if quantizer instance 
    //             was properly initialized and input data container contains enough memory, 
    //           2-3) same as for get_not_quantized_tensor
    mli_tensor get_quantized_tensor(mli_data_container memory) const;

    // Get const mli_tensor of source float data which instance was initialized with.
    //
    // takes mli_data_container with memory to be used for data and quantization params
    // returns: 1) mli_tensor with QUANTIZED data and prepared quant params memory according to 
    //             kept tensor and quant params, if quantizer instance 
    //             was properly initialized and input data container contains enough memory, 
    //           2-3) same as for get_not_quantized_tensor
    const mli_tensor get_source_float_tensor() const;
    
    // Is quantizer instance valid and was initialized properly
    bool is_valid() const { return is_valid_; }
    
    // This function intend to validate state of quantizer instance output tensors.
    // It returns status of returned tensor: Bad tensor (kBad), without assigned 
    // memory but with memory requirements (kNotEnoughMemory), or tensor is ok (kOk)
    static tensor_state validate_tensor(const mli_tensor& tsr);

 private:
     // Source data used for quantization
     const mli_tensor source_tsr_;
     const float* source_data_;
     const float* source_scales_;
     const float* source_zero_points_;
     const int8_t* source_scales_fraq_;
     
     // Is instance valid status
     bool is_valid_;

     // Internal methods for quantization
     static uint32_t get_required_data_capacity(const mli_tensor& tsr);
     static bool  tensor_assign_data_ptr(mli_tensor* tsr, void* ptr);
     static bool spread_memory(mli_tensor* tsr, const mli_data_container* data_mem,
                       const mli_data_container* quant_params_mem = nullptr);
};

} // namespace tst
} // namespace mli

#endif //_MLI_USER_TESTS_TEST_QUANTIZER_H_
