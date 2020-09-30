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
#include "mli_config.h"
#include "test_quantizer.h"

// Attributes for data arrays allocation. 
// const W_DATA_ATTR int arr[] will be placed in a HW specific memory according to MLI requirements
// W_DATA_ATTR for weights and IO_DATA_ATTR for activations (feature maps).
// Allocation works only with CCAC copiler. For others #else branch will be used
#if (PLATFORM == V2DSP_XY)
#define W_DATA_ATTR __xy __attribute__((section(".Xdata")))
#define IO_DATA_ATTR __xy __attribute__((section(".Ydata")))

#elif (PLATFORM == V2DSP_VECTOR)
#define W_DATA_ATTR __vccm __attribute__((section(".vecmem_data")))
#define IO_DATA_ATTR __vccm __attribute__((section(".vecmem_data")))

#else
#define W_DATA_ATTR 
#define IO_DATA_ATTR 
#endif

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
    static constexpr auto kPassValueSnr          = std::numeric_limits<float>::lowest();
    static constexpr auto kPassValueSnrDb        = std::numeric_limits<float>::lowest();
    static constexpr auto kPassValueQuantErrPerc = std::numeric_limits<float>::lowest();

    // Default and parametrized constructors
    quality_metrics();
    quality_metrics(float max_abs_err, float ref_to_noise_ratio, float ref_to_noise_snr, float quant_err_percent);

    // Populate instance metrics with measured ones.
    // Calculate metrics during comparison of predicted tensor, and reference data kept in tensor quantizer
    //
    // params:
    // [IN] pred_tsr - valid mli_tensor with predicted results to compare with expected reference data
    // [IN] ref_keeper - valid tensor_quantizer instance that keeps reference data to compare with
    // 
    // returns true in case of calculation was fine, and false otherwise 
    // (pred\ref structures mismatch or any of them are not valid)
    bool calculate_metrics(const mli_tensor& pred_tsr, const tensor_quantizer& ref_keeper);
    
    // Compare instance metric with some threshold (float or another instance).
    //
    // params:
    // [IN] id - metric_id that reflects which particular metrics to be compared with threshold
    // [IN] threshold - float or another quality_metrics instance that keeps threshold value.
    //                  Same internal value is expected to be 'below' the threshold.
    //
    // returns true if internal value is "under" threshold (bigger or lower depending on metric)
    bool is_threshold_met(const metric_id id, const float threshold) const;
    bool is_threshold_met(const metric_id id, const quality_metrics& threshold) const;
    
    // Get specific metric value from instance
    //
    // params:
    // [IN] id - metric_id that reflects which particular metrics to be returned
    //
    // returns metric_id value kept by the instance
    float get_metric_float(const metric_id id) const;

 private:
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
    // Default and parametrized constructors to initialize CRC state
    crc32_calc();
    crc32_calc(uint32_t init_val);

    // Reset instance state making it invalid
    // No return
    void reset();
    
    // Reset instance with new state and make it valid.
    //
    // params:
    // [IN] new_val - state to be kept and used as initial for the following calculations
    // No return
    void reset(uint32_t new_val);

    // Get current accumulated crc sum (state) of instance
    uint32_t get() const;
    
    // Is crc state valid
    // returns true if instance/reset  with a particular value OR it was invoked to calculate 
    // CRC at least for one byte.
    bool is_valid() const;

    // Process input tensor data to calculate CRC32 sum
    //
    // params:
    // [IN] in - valid mli_tensor which data array will be used to accumulate CRC32 sum
    //           Uses all data container not taking shape or memstides into account.
    // returns current instance state after calculations (currently accumulated  crc32 sum)
    uint32_t operator()(const mli_tensor& in);

    // Process in array to calculate CRC32 sum
    //
    // params:
    // [IN] in - array with data to accumulate CRC32 sum
    // [IN] size - size of 'in' array (bytes)
    // returns current instance state after calculations (currently accumulated  crc32 sum)
    uint32_t operator()(const int8_t* in, uint32_t size);

private:
    // State and flag whether crc instance valid or not
    uint32_t crc32_sum_;
    bool valid_crc32_;

    // look-up table for half-byte
    static const uint32_t crc32_lookup_table_[];
};


//=======================================================================
// Module to handle and check externally allocated memory for test needs
//
// This memory_manager intend to handle some memory region for test needs. 
// - It returns container with requested memory size 
// - It can return the only one container. return_memory() method should 
//   be called to mark memory as unused and able to be allocated again.
// - it initialize all the memory with some pre-defined pattern
// - It checks that memory beside the allocated region is not corrupted
//=======================================================================
class memory_manager {
public:
    // Parametrized constructors to initialize memory_manager.
    memory_manager(int8_t* memory, uint32_t mem_size);

    // Methods to get (allocate) memory from internally handled area by memory_manager.
    // The whole memory is initialized with fill_pattern. if requested size less than internal memory,
    // returns container that points to the middle of the region, e.g.:
    // /*                   All handled memory region                           */
    // /*------head_region----*------memory_to_return------*-----tail_region----*/
    // CRC32 sum for head and tail regions are kept to check that it wasn't modifyed after a while
    //
    // If requested memory is bigger, then returns empty container (nullptr and 0 capacity)
    
    // Method to allocate memory of exact size
    // params:
    // [IN] size - uint32_t requested size of memory
    // [IN] fill_pattern - optional uint32_t value to initialize all memory with
    mli_data_container allocate_memory(uint32_t size, uint32_t fill_pattern = 0xDEADBEEF);
    // Method to allocate memory according to tensor_quantizer requirements
    // params:
    // [IN] quant_unit - valid tensor_quantizer Which will be analyzed to return memory container 
    //                   to exactly fit it's requirements
    // [IN] fill_pattern - optional uint32_t value to initialize all memory with
    mli_data_container allocate_memory(const tensor_quantizer& quant_unit, uint32_t fill_pattern = 0xBEADED37);
    
    // Reset module to mark memory as unused and get an opportunity to allocate again
    // No return
    void return_memory();

    // Check that head and tail regions are not corrupted using pre-calculated CRC32 checksums. 
    // Return false (not corrupted) if regions aren't corrupted OR memory wasn't properly allocated before OR 
    // memory manager don't handle a valid memory region
    bool is_memory_corrupted() const;

private:
    int8_t* source_memory_;
    int8_t* allocated_memory_start_;
    uint32_t source_mem_size_;
    uint32_t allocated_mem_size_;
    crc32_calc head_mem_crc_;
    crc32_calc tail_mem_crc_;
};

} // namespace tst 
} // namespace mli

#endif //_MLI_USER_TESTS_TEST_INFRA_H_

