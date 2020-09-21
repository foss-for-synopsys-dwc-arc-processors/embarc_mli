/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "test_infra.h"

#include <memory>
#include <algorithm>

#include "tensor_transform.h"
#include "tests_aux.h"
#include "mli_api.h"

//using namespace std;

namespace mli {
namespace tst {

//===============================================================================================
//
// Methods of the module to calculate and handle result quality metrics
//
//===============================================================================================


// Constructors
//====================================================
quality_metrics::quality_metrics(float max_abs_err, float ref_to_noise_ratio, 
                                 float ref_to_noise_snr, float quant_err_percent)
    : max_abs_err_(max_abs_err)
    , ref_to_noise_ratio_(ref_to_noise_ratio)
    , ref_to_noise_snr_(ref_to_noise_snr)
    , quant_error_percentage(quant_err_percent)
{}

quality_metrics::quality_metrics()
    : max_abs_err_(kPassValueMaxAbsErr)
    , ref_to_noise_ratio_(kPassValueSnr)
    , ref_to_noise_snr_(kPassValueSnrDb)
    , quant_error_percentage(kPassValueQuantErrPerc)
{}


// Get metric itself
//====================================================
float quality_metrics::get_metric_float(const metric_id id) const {
    switch (id) {
    case kMetricMaxAbsErr:
        return max_abs_err_;
    case kMetricSignalToNoiseRatio:
        return ref_to_noise_ratio_;
    case kMetricSignalToNoiseRatioDb:
        return ref_to_noise_snr_;
    case kMetricQuantErrorPercent:
        return quant_error_percentage;
    }
}

// Compare instance metric with some threshold (float)
//====================================================
bool quality_metrics::is_threshold_met(const metric_id id, const float threshold) const {
    switch (id) {
    case kMetricMaxAbsErr:
        return max_abs_err_ <= threshold;
    case kMetricSignalToNoiseRatio:
        return ref_to_noise_ratio_ >= threshold;
    case kMetricSignalToNoiseRatioDb:
        return ref_to_noise_snr_ >= threshold;
    case kMetricQuantErrorPercent:
        return quant_error_percentage >= threshold;
    }
}

// Compare instance metric with some threshold (another instance).
//====================================================
bool quality_metrics::is_threshold_met(const metric_id id, const quality_metrics& threshold) const {
    switch (id) {
    case kMetricMaxAbsErr:
        return is_threshold_met(id, threshold.max_abs_err_);
    case kMetricSignalToNoiseRatio:
        return is_threshold_met(id, threshold.ref_to_noise_ratio_);
    case kMetricSignalToNoiseRatioDb:
        return is_threshold_met(id, threshold.ref_to_noise_snr_);
    case kMetricQuantErrorPercent:
        return is_threshold_met(id, threshold.quant_error_percentage);
    }
}

// Populate instance metrics with measured ones.
//===================================================
bool quality_metrics::calculate_metrics(const mli_tensor& pred_tsr, const tensor_quantizer& ref_keeper) {
    const uint32_t elem_num = mli_hlp_count_elem_num(&pred_tsr, 0);

    if (elem_num == 0 || ref_keeper.is_valid() == false)
        return false; 

    // To measure percentage of quantization error in total noise, 
    // we also need to quantize and dequantize ref data to get know how quantization affects values 
    // (apply quantization error using current parameters)
    // we need extra memory which size will be known after first "get quantized tensor" go, 
    // and do forward - backward transformation

    // First get memory requirements and allocate it.
    // Additionally we will get source float data in tensor form
    //===============================================
    uint32_t mem_required = 0;
    mli_data_container quntized_out_container{ 0 };
    mli_tensor quantized_ref = ref_keeper.get_quantized_tensor(quntized_out_container);
    const mli_tensor ref_tensor = ref_keeper.get_source_float_tensor();
    if (ref_keeper.validate_tensor(quantized_ref) != tensor_quantizer::kIncompleteMem ||
            ref_keeper.validate_tensor(ref_tensor) != tensor_quantizer::kOk)
        return false;

    // Check shapes;
    bool is_shape_ok = ref_tensor.rank == pred_tsr.rank;
    for (int i = 0; is_shape_ok && i < ref_tensor.rank; ++i)
        is_shape_ok &= ref_tensor.shape[i] == pred_tsr.shape[i];
    
    if (!is_shape_ok)
        return false;

    mem_required += quantized_ref.data.capacity;
    if (quantized_ref.el_type == MLI_EL_SA_8 || quantized_ref.el_type == MLI_EL_SA_32)
        mem_required += quantized_ref.el_params.sa.scale.capacity + quantized_ref.el_params.sa.zero_point.capacity;

    std::unique_ptr<float[]> pred_values(new float[elem_num]);
    std::unique_ptr<int8_t[]> quantized_out_mem(new int8_t[mem_required]);
    if (quantized_out_mem == nullptr || pred_values == nullptr)
        return false;

    // Dequantize predicted tensor and get metrics for it 
    // in comparison with reference
    //===============================================
    constexpr float eps = 0.000000000000000001f;
    ref_to_pred_output metrics_pred = { 0 };
    if (MLI_STATUS_OK != mli_hlp_fx_tensor_to_float(&pred_tsr, pred_values.get(), elem_num) ||
            TEST_PASSED != measure_err_vfloat(ref_tensor.data.mem.pf32, pred_values.get(), elem_num, &metrics_pred))
        return false;

    max_abs_err_ = metrics_pred.max_abs_err;
    ref_to_noise_snr_ = metrics_pred.ref_to_noise_snr;
    ref_to_noise_ratio_ = metrics_pred.ref_vec_length / std::max(metrics_pred.noise_vec_length, eps);


    // Do forward/backward quantization of reference data 
    // and calculate the level of quantization noise
    //===============================================
    quntized_out_container.capacity = mem_required;
    quntized_out_container.mem.pi8 = quantized_out_mem.get();
    quantized_ref = std::move(ref_keeper.get_quantized_tensor(quntized_out_container));
    ref_to_pred_output metrics_quant = { 0 };
    if (ref_keeper.validate_tensor(quantized_ref) != tensor_quantizer::kOk ||
            mli_hlp_fx_tensor_to_float(&quantized_ref, pred_values.get(), elem_num) != MLI_STATUS_OK ||
            measure_err_vfloat(ref_tensor.data.mem.pf32, pred_values.get(), elem_num, &metrics_quant) != TEST_PASSED)
        return false;

    // Originaly, quntization noise were reflected as ratio against total noise
    // (how many times the total noise is greater than the quantization noise)
    //noise_to_quant_ratio_ = metrics_pred.noise_vec_length / std::max(metrics_quant.noise_vec_length, eps);

    // Currently the same information is output in form of percentage which more understandable 
    // (what share of total noise in output is quantization noise, 
    // not calculations in quantized form and quantization of input operands) 
    quant_error_percentage = metrics_quant.noise_vec_length / std::max(metrics_pred.noise_vec_length, eps) * 100.f;
    
    return true;
}

//===============================================================================================
//
// Methods and dataof the Module to calculate and handle CRC32 sum.
//
//===============================================================================================

// Half-byte lookup table
//=========================================
const uint32_t crc32_calc::crc32_lookup_table_[16] = {
  0x00000000,0x1DB71064,0x3B6E20C8,0x26D930AC,0x76DC4190,0x6B6B51F4,0x4DB26158,0x5005713C,
  0xEDB88320,0xF00F9344,0xD6D6A3E8,0xCB61B38C,0x9B64C2B0,0x86D3D2D4,0xA00AE278,0xBDBDF21C
};


// Constructors
//================================
crc32_calc::crc32_calc()
    : crc32_sum_(0x00000000)
    , valid_crc32_(false)
{}
crc32_calc::crc32_calc(uint32_t init_val)
    : crc32_sum_(init_val)
    , valid_crc32_(true)
{}

// Reset instance state
//================================
void crc32_calc::reset() {
    crc32_sum_ = 0x00000000;
    valid_crc32_ = false;
}
void crc32_calc::reset(uint32_t new_val) {
    crc32_sum_ = new_val;
    valid_crc32_ = true;
}

// Get accumulatec CRC sum
//================================
uint32_t crc32_calc::get() const {
    return crc32_sum_;
}

// Get status of CRC instance
//================================
bool crc32_calc::is_valid() const {
    return valid_crc32_;
}

// Get accumulatec CRC sum
//================================
uint32_t crc32_calc::operator()(const mli_tensor& in) {
    const int8_t* current = in.data.mem.pi8;
    uint32_t length = mli_hlp_count_elem_num(&in, 0) * mli_hlp_tensor_element_size(&in);

    if (current != nullptr && length != 0) {
        uint32_t crc = ~crc32_sum_;  // same as previousCrc32 ^ 0xFFFFFFFF

        while (length-- != 0) {
            const uint8_t current_val = static_cast<uint8_t>(*current);
            crc = crc32_lookup_table_[(crc ^ current_val) & 0x0F] ^ (crc >> 4);
            crc = crc32_lookup_table_[(crc ^ (current_val >> 4)) & 0x0F] ^ (crc >> 4);
            current++;
        }

        crc32_sum_ = ~crc; // same as crc ^ 0xFFFFFFFF
        valid_crc32_ = true;
    }
    return crc32_sum_;
}


} // namespace mli {
} // namespace tst {
