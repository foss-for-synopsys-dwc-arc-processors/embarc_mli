/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_USER_TESTS_TEST_INFRA_H_
#define _MLI_USER_TESTS_TEST_INFRA_H_

#include <limits>

#include "mli_api.h"
#include "test_quantizer.h"

namespace mli {
namespace tst {
//===============================================================================================
// Module to calculate and handle result quality metrics.
//===============================================================================================
class quality_metrics {
 public:

    enum metric_id {
        kMetricMaxAbsErr = 0,          /**< Maximum absolute error. 
                                            Shows maximum diff between individual 
                                           value of predicted vector and appropriate value in reference vector*/

        kMetricSignalToNoiseRatio,     /**< Signal-to-noise ratio (ref_vec_length)/(noise_vec_length + eps)  
                                            where ref_vec_length is a total length of reference vector 
                                                 SQRT(SUM_i(X_ref_i^2))
                                            and noise_vec_length is a total length of noise vector 
                                                 SQRT(SUM_i((X_ref_i - X_pred_i)^2))*/

        kMetricSignalToNoiseRatioDb,   /**< Signal-to-noise ratio in decibels
                                            10*log_10((ref_vec_length)/(noise_vec_length + eps))  [dB]*/

        kMetricQuantErrorPercent,      /**< Share of quantiazation error in total noise in percentage
                                            what share of total noise in output is quantization noise 
                                            (not influence of calculations or precision of input ), 
                                            ((quant_err_vec_length)/(noise_vec_length + eps)) * 100  */
    };

    // Values which always under any threshold in comparison with it
    static constexpr auto kPassValueMaxAbsErr    = std::numeric_limits<float>::max();
    static constexpr auto kPassValueSnr          = std::numeric_limits<float>::min();
    static constexpr auto kPassValueSnrDb        = std::numeric_limits<float>::min();
    static constexpr auto kPassValueQuantErrPerc = std::numeric_limits<float>::min();

    // Default and parametrized constructors
    quality_metrics();
    quality_metrics(float max_abs_err, float ref_to_noise_ratio, float ref_to_noise_snr, float quant_err_percent);

    // Populate instance metrics with measured ones.
    //
    // Calculate metrics during comparison of predicted tensor, and reference data kept in tensor quantizer
    // returns true in case of calculation was fine, and false otherwise 
    // (pred\ref structures mismatch or any of them are not valid)
    bool calculate_metrics(const mli_tensor& pred_tsr, const tensor_quantizer& ref_keeper);
    
    // Compare instance metric with some threshold (float or another instance).
    // returns true if internal value is "under" threshold (bigger or lower depending on metric)
    bool is_threshold_met(const metric_id id, const float threshold) const;
    bool is_threshold_met(const metric_id id, const quality_metrics& threshold) const;
    
    // Get metric itself
    float get_metric_float(const metric_id id) const;

 private:
    // Fields with metrics
    float max_abs_err_;
    float ref_to_noise_ratio_;
    float ref_to_noise_snr_;
    float quant_error_percentage_;
};


//===============================================================================================
// Module to calculate and handle CRC32 sum.
//===============================================================================================
class crc32_calc {
public:
    // Default and parametrized constructors to initialize crc state
    crc32_calc();
    crc32_calc(uint32_t init_val);

    // Reset instance state making it invalid
    void reset();
    
    // Reset instance with new state
    void reset(uint32_t new_val);

    // Get accumulated crc sum (state)
    uint32_t get() const;
    
    // Is crc state valid
    bool is_valid() const;

    // Process input tensor data to calculate CRC32 sum
    uint32_t operator()(const mli_tensor& in);

    // Process input tensor data to calculate CRC32 sum
    uint32_t operator()(const int8_t* in, uint32_t size);

private:
    // State and flag whether it valid or not
    uint32_t crc32_sum_;
    bool valid_crc32_;

    // look-up table for half-byte
    static const uint32_t crc32_lookup_table_[];
};


//=======================================================================
// Module to handle and check externally allocated memory for test needs
//=======================================================================
class memory_keeper {
    // This keeper intend to handle some memory region for test needs. 
    // - It returns container with requested memory size 
    // - It can return the only one container. return_memory() method should 
    //   be called to mark memory as unused and able to be afforded again.
    // - it initialize all the memory with some pre-defined pattern
    // - It checks that memory beside the afforded region is not corrupted

public:
    // Parametrized constructors to initialize keeper.
    memory_keeper(int8_t* memory, uint32_t mem_size);

    // Methods to get(afford) memory from internally handled area by keeper.
    // The whole memory is initialized with fill_pattern. if requested size less than internal memory,
    // returns container that points to the middle of the region, e.g.:
    // /*                   All handeled memory region                          */
    // /*------head_region----*------memory_to_return------*-----tail_region----*/
    // CRC32 sum for head and tail regions are kept to check that it wasn't modifyed after a while
    // If requested memory is bigger, then returns emty container (nullptr and 0 capacity)
    
    // * Method to afford memory of exact size
    mli_data_container afford_memory(uint32_t size, uint32_t fill_pattern = 0xDEADBEEF);
    // * Method to afford memory according to tensor_quantizer requirements
    mli_data_container afford_memory(const tensor_quantizer& quant_unit, uint32_t fill_pattern = 0xBEADED37);
    
    // Reset module to mark memory as unused and get an opportunity to afford one more time
    void return_memory();

    // Check that head and tail regions are not corrupted using pre-calculated CRC32 checksums. 
    // Return false (not corrupted) if regions aren't corrupted OR memory wasn't properly afforded before OR 
    // memory keeper don't handle a valid memory region
    bool is_memory_corrupted() const;

private:
    int8_t* source_memory_;
    int8_t* afforded_memory_start_;
    uint32_t source_mem_size_;
    uint32_t afforded_mem_size_;
    crc32_calc head_mem_crc_;
    crc32_calc tail_mem_crc_;
};

} // namespace tst 
} // namespace mli

#endif //_MLI_USER_TESTS_TEST_INFRA_H_

