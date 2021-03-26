/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_USER_TESTS_TEST_QUALITY_METRICS_H_
#define _MLI_USER_TESTS_TEST_QUALITY_METRICS_H_

#include <limits>

#include "mli_api.h"
#include "test_tensor_quantizer.h"

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

} // namespace tst 
} // namespace mli

#endif //_MLI_USER_TESTS_TEST_QUALITY_METRICS_H_

