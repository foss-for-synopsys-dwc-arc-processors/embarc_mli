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

    // Tensor state after validation
    enum tensor_state {
        kOk = 0,           /**< Tensor is complete. All felds are properly initialized and not contradictory*/
        kBad,              /**< Some of general tensor fields (shape, rank, memstride, el_type)
                                are contradictory or badly initialized. Bad memory containers not considered*/
        kIncompleteMem,    /**< General tensor fields (shape, rank, memstride, el_type) are valid, but some of memory 
                                containers are not properly initialized to reflect designated structure 
                                (pointed to nowhere or not enough capacity to keep required data)*/
    };

    // Constructor gets all necessary inputs for quantization
    // Tensor parameters are expected to be complete enough for quantization and output valid tensor.
    // For FX types next fields must be filled: Shape, rank, memstride, el_type, el_params.fx.frac_bits
    // For SA types next fields must be filled: Shape, rank, memstride, el_type, el_params.sa.dim, el_params.sa.scale_frac_bits, 
    
    // This Constructor intended to be used with FX type of tensors ONLY
    tensor_quantizer(mli_tensor tsr, const float* data, uint32_t data_size);
    
    // This Constructor intended to be used (but not limited) with SA type of tensors
    tensor_quantizer(mli_tensor tsr, const float *data, uint32_t data_size, const float *scales, uint32_t scales_size,
                     const float* zero_points, uint32_t zero_points_size, 
                     const int8_t* scales_fraq_bits, uint32_t scales_fraq_bits_size);


    // Get mli_tensor with assigned memory only
    //
    // params:
    // [IN] memory - mli_data_container with memory to be spread across data and quantization params
    // returns: 1) mli_tensor with ASSIGNED memory and prepared quant params memory, if quantizer instance 
    //             was properly initialized and input data container contains enough memory, 
    //          2) mli_tensor with memory requirements in capacity fields of data container 
    //              if input contains not enough memory
    //          3) empty mli_tensor if quantizer instance was badly initialized    
    mli_tensor get_not_quantized_tensor(mli_data_container memory) const;

    // Get mli_tensor with quantized data
    //
    // params:
    // [IN] memory - mli_data_container with memory to be used for data and quantization params
    // returns: 1) mli_tensor with QUANTIZED data and prepared quant params memory according to 
    //             kept tensor and quant params, if quantizer instance 
    //             was properly initialized and input data container contains enough memory, 
    //           2-3) same as for get_not_quantized_tensor
    mli_tensor get_quantized_tensor(mli_data_container memory) const;

    // Get const mli_tensor of source float data which instance was initialized with.
    //
    // returns: 1) mli_tensor with constant source float (MLI_EL_FP32) data 
    //             if quantizer instance was properly initialized (valid) 
    //          2) empty mli_tensor if quantizer instance was badly initialized    
    const mli_tensor get_source_float_tensor() const;
    
    // Is quantizer instance valid and was initialized properly
    bool is_valid() const { return is_valid_; }
    
    // Validate state of tensors in terms of 3 state (see tensor_state enum).
    //
    // params:
    // [IN] tsr - mli_tensor which state need to be defined
    // It returns status of returned tensor (see tensor_state enum): Bad tensor (kBad), 
    // memory requirements don't met (kNotEnoughMemory), or tensor is ok (kOk)
    static tensor_state validate_tensor(const mli_tensor& tsr);

    // Quantize float data to destination tensor according to tensor's internal format
    // Note: This function should be replaced by MLI API transform kernel when it will be done
    //
    // params:
    // [IN] src - pointer to float array that keeps input data to be quantized. src array must 
    //            contain exactly the same number of values (defined by src_size) as required 
    //            to populate dst tensor (defined by shape)
    // [IN] src_size - uint32_t number of values in src array
    // [IN] dst - mli_tensor to keep result of quantization. Tensor structure must be fully valid, 
    //            including all quantization parameters and containers.
    //
    // returns status of the dst tensor - kOk if quantization was performed properly, kBad
    //         if input tensor not valid at all, kNotEnoughMemory if tensor memory containers don't 
    //         have enough memory for quantization. Function modifies only memory pointed by tensor's data container. 
    static tensor_state quantize_float_data(const float* src, uint32_t src_size, mli_tensor* dst);
    
    // De-Quantize tensor data to float values 
    // Note: This function should be replaced by MLI API transform kernel when it will be done
    //
    // params:
    // [IN] src - mli_tensor with source data. Tensor structure must be fully valid, 
    //            and contains required data.
    //            contains exactly the same number of values (defined by src_size) as required 
    //            to populate dst tensor (defined by shape)
    // [IN] dst - pointer to float array to keeps dequantization results. dst array must 
    //            contains enough number of values (defined by dst_size) to kee de-quantization results of 
    //            src array
    // [IN] dst_size - uint32_t number of values in dst array
    //
    // returns status of the src tensor - kOk if de-quantization was performed properly, kBad
    //         if input tensor not valid at all, kNotEnoughMemory if dst array contains not enough memory 
    //         for de-quantization. Function modifies only memory pointed by dst. 
    static tensor_state dequantize_tensor_data(const mli_tensor* src, float* dst, uint32_t dst_size);

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
     
     template <mli_element_type src_el_type>
     static void dequantize_tensor_data_routine(const mli_tensor* src, float* dst, uint32_t dst_size);
     
     template <mli_element_type dst_el_type>
     static void quantize_float_data_routine(const float* src, uint32_t src_size, mli_tensor* dst);
};

} // namespace tst
} // namespace mli

#endif //_MLI_USER_TESTS_TEST_QUANTIZER_H_
