/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "test_tensor_quantizer.h"

// Standard asserts should be intentionally turned-on by defenition of TEST_DEBUG.
#if !defined(TEST_DEBUG)
#define NDEBUG
#endif

#include <assert.h>
#include <math.h>

#include <algorithm>
#include <memory>
#include <type_traits>


namespace mli {
namespace tst {

//======================================================================================================
//
// Interface Methods of Quantizer class
//
//======================================================================================================

// Default constructor
//=========================================
tensor_quantizer::tensor_quantizer()
    : source_tsr_(mli_tensor())
    , source_data_(nullptr)
    , source_scales_(nullptr)
    , source_zero_points_(nullptr)
    , source_scales_fraq_(nullptr)
    , is_valid_(false)
{}

// Constructor intended to be used with SA type ONLY
//==========================================
tensor_quantizer::tensor_quantizer(mli_tensor tsr, const int quant_dim, const float* data, const uint32_t data_size,
                                   const float* scales, const uint32_t scales_size,
                                   const float* zero_points, const uint32_t zero_points_size,
                                   const int8_t* scales_fraq_bits, const uint32_t scales_fraq_bits_size)
        : source_tsr_(tsr)
        , source_data_(data)
        , source_scales_(scales)
        , source_zero_points_(zero_points)
        , source_scales_fraq_(scales_fraq_bits)
        , is_valid_(false) {
    // if tensor not of a type that constructor intend to work with, keep is_valid_ state false 
    // and skip the rest initialization code
    if (source_tsr_.el_type == MLI_EL_SA_8 || source_tsr_.el_type == MLI_EL_SA_32) {
        source_tsr_.el_params.sa.dim = quant_dim;

        // check that input tensor in general not bad
        const tensor_state state = validate_tensor(source_tsr_);
        is_valid_ = (state == kIncompleteMem || state == kOk);

        // check that arrays are of expectd size
        if (is_valid_) {
            is_valid_ = (data_size == mli_hlp_count_elem_num(&source_tsr_, 0));
            const uint32_t expected_vals = (source_tsr_.el_params.sa.dim < 0) ? 
                                            1 : source_tsr_.shape[source_tsr_.el_params.sa.dim];
            is_valid_ &= (scales_size == expected_vals);
            is_valid_ &= (zero_points_size == expected_vals);
            is_valid_ &= (scales_fraq_bits_size == expected_vals);
            for (int i = 0; is_valid_ && i < scales_size; ++i)
                is_valid_ &= (scales[i] > 0.0f);
        }
    }
}

// Constructor intended to be used with FX type of tensors ONLY
//==========================================
tensor_quantizer::tensor_quantizer(mli_tensor tsr, const int frac_bits, const float* data, const uint32_t data_size)
        : source_tsr_(tsr)
        , source_data_(data)
        , source_scales_(nullptr)
        , source_zero_points_(nullptr)
        , source_scales_fraq_(nullptr)
        , is_valid_(false) {
    // Similar as for SA constructor - keep is_valid false if not FX
    if (source_tsr_.el_type == MLI_EL_FX_8 || source_tsr_.el_type == MLI_EL_FX_16) {
        source_tsr_.el_params.fx.frac_bits = frac_bits;
        // check that input tensor in general not bad
        const tensor_state state = validate_tensor(source_tsr_);
        is_valid_ = (state == kIncompleteMem || state == kOk);
        if(is_valid_)
            is_valid_ = (data_size == mli_hlp_count_elem_num(&source_tsr_, 0));
    }
}

// Constructor intended to be used with FP32 type of tensors ONLY
//==========================================
tensor_quantizer::tensor_quantizer(mli_tensor tsr, const float* data, const uint32_t data_size)
    : source_tsr_(tsr)
    , source_data_(data)
    , source_scales_(nullptr)
    , source_zero_points_(nullptr)
    , source_scales_fraq_(nullptr)
    , is_valid_(false) {
    // SImilar as for FX constructor - keep is_valid false if not FP32
    if (source_tsr_.el_type == MLI_EL_FP_32) {
        // check that input tensor in general not bad
        const tensor_state state = validate_tensor(source_tsr_);
        is_valid_ = (state == kIncompleteMem || state == kOk);
        if (is_valid_)
            is_valid_ = (data_size == mli_hlp_count_elem_num(&source_tsr_, 0));
    }
}



// Return complete tensor with source float point data
//=================================================================================
const mli_tensor tensor_quantizer::get_source_float_tensor() const {
    if (!is_valid_)
        return mli_tensor{ 0 };

    assert(validate_tensor(source_tsr_) != kBad);
    mli_tensor ret_tsr = source_tsr_;
    ret_tsr.data.capacity = sizeof(source_data_[0]) * mli_hlp_count_elem_num(&ret_tsr, 0);
    uint32_t cur_memstr = 1;
    for (int i = static_cast<int>(ret_tsr.rank) - 1; i >= 0; --i) {
        assert(i < MLI_MAX_RANK);
        ret_tsr.mem_stride[i] = cur_memstr;
        cur_memstr *= ret_tsr.shape[i];
    }
    ret_tsr.el_type = MLI_EL_FP_32;

    // As we return const tensor just after this assignment, 
    // removing cv qualfiers here is considered as and acceptable.
    ret_tsr.data.mem.pf32 = const_cast<float*>(source_data_);
    return ret_tsr;
}


// Return complete quantized tensor using source tensor, data and quant params
//=================================================================================
mli_tensor tensor_quantizer::get_quantized_tensor(mli_data_container memory) const {
    if (!is_valid_)
        return mli_tensor{ 0 };

    assert(validate_tensor(source_tsr_) != kBad);
    mli_tensor ret_tsr = get_not_quantized_tensor(memory);
    if (ret_tsr.data.mem.pi8 != nullptr) {
        // Note: it's better to replace it with API function when time will come
        tensor_state fp_to_fx_stat;
        fp_to_fx_stat = quantize_float_data(source_data_, mli_hlp_count_elem_num(&ret_tsr, 0), &ret_tsr);
        assert(fp_to_fx_stat == kOk);
        if (ret_tsr.rank == 0) {
            switch (ret_tsr.el_type) {
                case MLI_EL_FX_4:
                case MLI_EL_FX_8:
                case MLI_EL_SA_8:
                    ret_tsr.data.mem.i8 = ret_tsr.data.mem.pi8[0];
                    break;
                case MLI_EL_FX_16:
                case MLI_EL_FP_16:
                    ret_tsr.data.mem.i16 = ret_tsr.data.mem.pi16[0];
                    break;
                case MLI_EL_SA_32:
                    ret_tsr.data.mem.i32 = ret_tsr.data.mem.pi32[0];
                    break;
                case MLI_EL_FP_32:
                    ret_tsr.data.mem.f32 = ret_tsr.data.mem.pf32[0];
                    break;
                default:
                    assert(ret_tsr.el_type == MLI_EL_FP_32); // at least last case must match
                    ret_tsr.data.mem.pi8 = nullptr;
                    break;
            }
        }
    }
    return ret_tsr;
}


// Return tensor structure populated with source tensor fields and quant params ONLY
//=================================================================================
mli_tensor tensor_quantizer::get_not_quantized_tensor(mli_data_container memory) const {
    if (!is_valid_)
        return mli_tensor{ 0 };
    
    assert(validate_tensor(source_tsr_) != kBad);
    mli_tensor ret_tsr = source_tsr_;
    const bool is_mem_spread_ok = spread_memory(&ret_tsr, &memory);

    // For SA format preparation of quantization parameters is required
    if (is_mem_spread_ok && (ret_tsr.el_type == MLI_EL_SA_8 || ret_tsr.el_type == MLI_EL_SA_32)) {
        int num_vals = static_cast<int>((ret_tsr.el_params.sa.dim < 0) ? 1 : ret_tsr.shape[ret_tsr.el_params.sa.dim]);
        int16_t* scale_dst = (ret_tsr.el_params.sa.dim >= 0) ? ret_tsr.el_params.sa.scale.mem.pi16
                                                             : &ret_tsr.el_params.sa.scale.mem.i16;
        int16_t* zp_dst = (ret_tsr.el_params.sa.dim >= 0) ? ret_tsr.el_params.sa.zero_point.mem.pi16
                                                          : &ret_tsr.el_params.sa.zero_point.mem.i16;
        int8_t* frac_dst = (ret_tsr.el_params.sa.dim >= 0) ? ret_tsr.el_params.sa.scale_frac_bits.mem.pi8
                                                           : &ret_tsr.el_params.sa.scale_frac_bits.mem.i8;
        assert(num_vals >= 0);
        for (int i = 0; i < num_vals; i++) {
            const int8_t scale_fraq_bits = source_scales_fraq_[i];
            const float mult = (scale_fraq_bits >= 0) ? (float)((int64_t)1l << scale_fraq_bits)
                                                      : (float)(1.f / ((int64_t)1l << abs(scale_fraq_bits)));
            const float round_val = 0.5f;
            const int32_t dst_val = static_cast<int32_t>(mult * source_scales_[i] + round_val);
            const int32_t zero_val = static_cast<int32_t>(-source_zero_points_[i] / source_scales_[i] + round_val);

            auto sat_short = [](int32_t val) -> int16_t{
                constexpr int32_t lim_min = std::numeric_limits<int16_t>::min();
                constexpr int32_t lim_max = std::numeric_limits<int16_t>::max();
                return static_cast<int16_t>(std::min(std::max(val, lim_min), lim_max));
            };
            frac_dst[i] = scale_fraq_bits;
            scale_dst[i] = sat_short(dst_val);
            zp_dst[i] = sat_short(zero_val);
        }
    }
    return ret_tsr;
}


// Validate tensor in terms of three state: Bad, Incomplete, Ok
//=======================================================
tensor_quantizer::tensor_state tensor_quantizer::validate_tensor(const mli_tensor& tsr) {
    // First check general tensor parameters like shape, rank, memstride, 
    // quantization dimension of sa type if applicable.
    // If they are broken - tensor is considered as bad and can't be used at all
    if (mli_hlp_tensor_element_size(&tsr) == 0 || tsr.rank < 0 || tsr.rank > MLI_MAX_RANK ||
        ((tsr.el_type == MLI_EL_SA_8 || tsr.el_type == MLI_EL_SA_32) && tsr.el_params.sa.dim >= (int32_t)tsr.rank)) {
        return kBad;
    }

    bool is_memstride_ok = true;
    bool is_memstride_zero = true;
    uint32_t memstride_min = 1;
    int idx = static_cast<int>(tsr.rank) - 1;
    for (; idx >= 0; --idx) {
        is_memstride_zero &= (tsr.mem_stride[idx] == 0);
        is_memstride_ok &= tsr.mem_stride[idx] >= memstride_min;
        memstride_min = tsr.mem_stride[idx] * tsr.shape[idx];
    }
    if (!is_memstride_zero && !is_memstride_ok) {
        return kBad;
    }

    // Next need to define whether memory is properly assigned to data 
    // and additional fields like scales/zero_points.
    // If they are broken - tensor is considered as incomplete and requires proper memory assignment
    if (tsr.data.mem.pi16 == nullptr || get_required_data_capacity(tsr) > tsr.data.capacity)
        return kIncompleteMem;
    if ((tsr.el_type == MLI_EL_SA_8 || tsr.el_type == MLI_EL_SA_32) && tsr.el_params.sa.dim >= 0) {
        const uint32_t num_values = tsr.shape[tsr.el_params.sa.dim];
        if (tsr.el_params.sa.scale.mem.pi16 == nullptr ||
            tsr.el_params.sa.zero_point.mem.pi16 == nullptr ||
            tsr.el_params.sa.scale_frac_bits.mem.pi8 == nullptr ||
            tsr.el_params.sa.scale.capacity < num_values * sizeof(int16_t) ||
            tsr.el_params.sa.zero_point.capacity < num_values * sizeof(int16_t) ||
            tsr.el_params.sa.scale_frac_bits.capacity < num_values * sizeof(int8_t)) {
            return kIncompleteMem;
        }
    }

    // Otherwise tensor is Ok
    return kOk;
}

// Quantize float data to destination tensor according to it's format
// Instantiation of quantization routines depending on type
// Note: This function should be replaced by MLI API transform kernel when it will be done
//====================================================================
tensor_quantizer::tensor_state tensor_quantizer::quantize_float_data(const float* src, uint32_t src_size,
                                                                     mli_tensor* dst) {
    tensor_state ret_status = (src == nullptr || dst == nullptr) ? kBad : kOk;
    if (ret_status == kOk)
        ret_status = validate_tensor(*dst);
    if (ret_status == kOk && src_size > mli_hlp_count_elem_num(dst, 0))
        ret_status = kIncompleteMem;

    if (ret_status == kOk) {
        switch (dst->el_type) {
        case MLI_EL_FX_8:
            quantize_float_data_routine<MLI_EL_FX_8>(src, src_size, dst);
            break;
        case MLI_EL_SA_8:
            quantize_float_data_routine<MLI_EL_SA_8>(src, src_size, dst);
            break;
        case MLI_EL_FX_16:
            quantize_float_data_routine<MLI_EL_FX_16>(src, src_size, dst);
            break;
        case MLI_EL_SA_32:
            quantize_float_data_routine<MLI_EL_SA_32>(src, src_size, dst);
            break;
        case MLI_EL_FP_32:
            quantize_float_data_routine<MLI_EL_FP_32>(src, src_size, dst);
            break;
        default:
            ret_status = kBad;
        }
    }
    return ret_status;
}


// De-Quantize tensor data to float values 
// Instantiation of de-quantization routines depending on type
// Note: This function should be replaced by MLI API transform kernel when it will be done
//====================================================================
tensor_quantizer::tensor_state tensor_quantizer::dequantize_tensor_data(const mli_tensor* src, float* dst,
                                                                        uint32_t dst_size) {
    tensor_state ret_status = (src == nullptr || dst == nullptr) ? kBad : kOk;
    if (ret_status == kOk)
        ret_status = validate_tensor(*src);
    if (ret_status == kOk && dst_size < mli_hlp_count_elem_num(src, 0))
        ret_status = kIncompleteMem;

    if (ret_status == kOk) {
        switch (src->el_type) {
        case MLI_EL_FX_8:
            dequantize_tensor_data_routine<MLI_EL_FX_8>(src, dst, dst_size);
            break;
        case MLI_EL_SA_8:
            dequantize_tensor_data_routine<MLI_EL_SA_8>(src, dst, dst_size);
            break;
        case MLI_EL_FX_16:
            dequantize_tensor_data_routine<MLI_EL_FX_16>(src, dst, dst_size);
            break;
        case MLI_EL_SA_32:
            dequantize_tensor_data_routine<MLI_EL_SA_32>(src, dst, dst_size);
            break;
        case MLI_EL_FP_32:
            dequantize_tensor_data_routine<MLI_EL_FP_32>(src, dst, dst_size);
            break;
        default:
            assert(src->el_type == MLI_EL_FP_32); // at least last case must match
            ret_status = kBad;
        }
    }

    return ret_status;
}


//======================================================================================================
//
// Internal Functions and routines of Quantizer class
//
//======================================================================================================

// Calculate capacity requirement of data field of the tensor taking memstride into account
//=================================================================================
uint32_t tensor_quantizer::get_required_data_capacity(const mli_tensor& tsr) {
    assert(mli_hlp_tensor_element_size(&tsr) != 0);
    assert(tsr.rank >= 0);
    assert(tsr.rank <= MLI_MAX_RANK);

    uint32_t ret_val = 0;
    if (tsr.mem_stride[0] == 0) {
        ret_val = mli_hlp_tensor_element_size(&tsr) * mli_hlp_count_elem_num(&tsr, 0);
    } else {
        // Method that implies removing unused tail
        for (int dim = static_cast<int>(tsr.rank - 1); dim >= 0; --dim) {
            assert(tsr.mem_stride[dim] > 0);
            assert(tsr.shape[dim] > 0);
            ret_val += tsr.mem_stride[dim] * (tsr.shape[dim] - 1);
        }
        ret_val += 1;
        ret_val *= mli_hlp_tensor_element_size(&tsr);

        // Rough method which implies allocation for tail
        //ret_val = tsr.mem_stride[0] * tsr.shape[0]; 
    }
    return ret_val;
}


// Assign void pointer to a proper union field of tensor
//=================================================================================
bool  tensor_quantizer::tensor_assign_data_ptr(mli_tensor* tsr, void* ptr) {
    assert(tsr != nullptr);
    switch (tsr->el_type) {
    case MLI_EL_FX_4:
    case MLI_EL_FX_8:
    case MLI_EL_SA_8:
        tsr->data.mem.pi8 = static_cast<int8_t*>(ptr);
        return true;
    case MLI_EL_FX_16:
    case MLI_EL_FP_16:
        tsr->data.mem.pi16 = static_cast<int16_t*>(ptr);
        return true;
    case MLI_EL_SA_32:
        tsr->data.mem.pi32 = static_cast<int32_t*>(ptr);
        return true;
    case MLI_EL_FP_32:
        tsr->data.mem.pf32 = static_cast<float*>(ptr);
        return true;
    default:
        assert(tsr->el_type == MLI_EL_FP_32); // at least last case must match
        tsr->data.mem.pi8 = nullptr;
        return false;
    };
}

// Spread provided memory across tensor's containers: data and quantization params
//=================================================================================
bool tensor_quantizer::spread_memory(mli_tensor* tsr, const mli_data_container* data_mem, 
                                     const mli_data_container* quant_params_mem) {
    // This functions assigns provided memory to tensor data according to tensor type and alignment needs.
    // It takes tensor to be populated with data pointers and data containers as memory donors.
    // tsr structure must contains properly filled shape, rank and elemet type data. 
    // data_mem container must keep pointer and capacity to a valid memory which is sufficint for tsr needs 
    // quant_params memory container is optional even for SA type of tensor. 
    // For SA tensors if no quant_params_mem is provided, data_mem is used as the only donor for all
    // 
    // Function retrns bool value which reflects whether assignment was completed successfully or not.
    // if function fails due to nullptrs of tsr or data container, operands wasn't changed.
    // If spreading wasn't successful because of lack of memory, .capacity fields of all data containers in tsr structure
    // will keep minimal requirements for containers. 

    // Data container and target tensors must be provided. Others are opional
    assert(tsr != nullptr && validate_tensor(*tsr) != kBad);
    if (tsr == nullptr && validate_tensor(*tsr) == kBad)
        return false;

    // First fill required minimal mem capacity
    //=================================================
    tsr->data.capacity = get_required_data_capacity(*tsr);
    if ((tsr->el_type == MLI_EL_SA_8 || tsr->el_type == MLI_EL_SA_32)) {
        const uint32_t num_vals = (tsr->el_params.sa.dim >= 0) ? tsr->shape[tsr->el_params.sa.dim] : 1;
        tsr->el_params.sa.scale.capacity = (tsr->el_params.sa.dim >= 0) ? sizeof(int16_t) * num_vals : 0;
        tsr->el_params.sa.zero_point.capacity = (tsr->el_params.sa.dim >= 0) ? sizeof(int16_t) * num_vals : 0;
        tsr->el_params.sa.scale_frac_bits.capacity = (tsr->el_params.sa.dim >= 0) ? sizeof(int8_t) * num_vals : 0;
    }

    // If no data container is provided, just return memory requirements
    bool success = !(data_mem == nullptr || data_mem->mem.pi8 == nullptr);
    
    // Assign aligned memory for scales if it is needed
    //=================================================
    // If there are no specific containers for qantization params we will use data memory for it
    const bool use_data_mem_spare = (quant_params_mem == nullptr);
    void* mem_for_spare = (use_data_mem_spare) ? data_mem->mem.pi8 : quant_params_mem->mem.pi8;
    size_t available_cap = (use_data_mem_spare) ? data_mem->capacity : quant_params_mem->capacity;
    if ((tsr->el_type == MLI_EL_SA_8 || tsr->el_type == MLI_EL_SA_32)) {
        if (success && tsr->el_params.sa.scale.capacity != 0) {
            tsr->el_params.sa.scale.mem.pi16 = static_cast<int16_t*>(
                std::align(alignof(int16_t), tsr->el_params.sa.scale.capacity, mem_for_spare, available_cap));
            if (tsr->el_params.sa.scale.mem.pi16 == nullptr) {
                success = false;
            } else {
                mem_for_spare = static_cast<int8_t*>(mem_for_spare) + tsr->el_params.sa.scale.capacity;
                available_cap = static_cast<uint32_t>(available_cap) - tsr->el_params.sa.scale.capacity;
            }
        }
        if (success && tsr->el_params.sa.zero_point.capacity != 0) {
            tsr->el_params.sa.zero_point.mem.pi16 = static_cast<int16_t*>(
                std::align(alignof(int16_t), tsr->el_params.sa.zero_point.capacity, mem_for_spare, available_cap));
            if (tsr->el_params.sa.zero_point.mem.pi16 == nullptr) {
                success = false;
            } else {
                mem_for_spare = static_cast<int8_t*>(mem_for_spare) + tsr->el_params.sa.zero_point.capacity;
                available_cap = static_cast<uint32_t>(available_cap) - tsr->el_params.sa.zero_point.capacity;
            }
        }
        if (success && tsr->el_params.sa.scale_frac_bits.capacity != 0) {
            tsr->el_params.sa.scale_frac_bits.mem.pi8 = static_cast<int8_t*>(
                std::align(alignof(int8_t), tsr->el_params.sa.scale_frac_bits.capacity, mem_for_spare, available_cap));
            success = (tsr->el_params.sa.scale_frac_bits.mem.pi8 != nullptr);
            if (tsr->el_params.sa.scale_frac_bits.mem.pi8 == nullptr) {
                success = false;
            } else {
                mem_for_spare = static_cast<int8_t*>(mem_for_spare) + tsr->el_params.sa.scale_frac_bits.capacity;
                available_cap = static_cast<uint32_t>(available_cap) - tsr->el_params.sa.scale_frac_bits.capacity;
            }
        }
    }
    
    // Align memory for data and assign it to tensor
    //=================================================
    void* data_ptr = static_cast<void*>((use_data_mem_spare) ? mem_for_spare : data_mem->mem.pi8);
    size_t data_cap = static_cast<size_t>((use_data_mem_spare) ? available_cap : data_mem->capacity);
    if (success && std::align(mli_hlp_tensor_element_size(tsr), tsr->data.capacity, data_ptr, data_cap) != nullptr) {
        success &= tensor_assign_data_ptr(tsr, data_ptr);
    } else {
        success = false;
    }

    if (!success) {
        // Provide info on required memory and clear pointer to data
        tsr->data.mem.pi8 = tsr->el_params.sa.scale.mem.pi8 = tsr->el_params.sa.zero_point.mem.pi8 = nullptr;
        tsr->data.capacity += mli_hlp_tensor_element_size(tsr) - 1;
        if ((tsr->el_type == MLI_EL_SA_8 || tsr->el_type == MLI_EL_SA_32)) {
            tsr->el_params.sa.scale_frac_bits.capacity += alignof(int8_t) - 1;
            tsr->el_params.sa.zero_point.capacity += alignof(int16_t) - 1;
            tsr->el_params.sa.scale.capacity += alignof(int16_t) - 1;
        }
    }
    return success;
}

// Saturate 32bit value to the final element type range.
//====================================================================
template <typename i_T, typename o_T>
o_T tensor_quantizer::saturate(i_T val) {
    constexpr int32_t lim_min = static_cast<int32_t>(std::numeric_limits<o_T>::min());
    constexpr int32_t lim_max = static_cast<int32_t>(std::numeric_limits<o_T>::max());
    return static_cast<o_T>(std::min(std::max(val, lim_min), lim_max));
}

template <>
float tensor_quantizer::saturate(int32_t val) {
    return val;
}

// Quantize float data to destination tensor according to it's format
// Main template of quantization routine
//====================================================================
template <mli_element_type dst_el_type>
void tensor_quantizer::quantize_float_data_routine(const float* src, uint32_t src_size, mli_tensor* dst) {
    assert(src != nullptr && dst != nullptr);
    assert(validate_tensor(*dst) == kOk);
    assert(dst_el_type == dst->el_type);
    assert(src_size <= mli_hlp_count_elem_num(dst, 0));
    assert(MLI_MAX_RANK == 4);

    // Put type traits magic to derive output type (derived_T) 
    typedef typename std::conditional<dst_el_type == MLI_EL_FX_16, int16_t,
        typename std::conditional<dst_el_type == MLI_EL_FX_8 || dst_el_type == MLI_EL_SA_8, int8_t,
        typename std::conditional<dst_el_type == MLI_EL_SA_32, int32_t, 
        typename std::conditional<dst_el_type == MLI_EL_FP_32, float, void>::type >::type >::type >::type
        derived_T;

    // And use derived_T as o_T if it's in supported set (not a void). Otherwise, break compilation
    typedef typename std::enable_if<!std::is_same<derived_T, void>::value, derived_T>::type    o_T;

    // Extend shape to the MLI_MAX_RANK complimenting it with 1s in a front.
    // Calculate strides on input float array and output tensor 
    // for easier definition of element position in total arrays
    int dst_strides[MLI_MAX_RANK] = { 0 };
    int src_strides[MLI_MAX_RANK] = { 0 };
    int dst_extended_shape[MLI_MAX_RANK] = { 0 };
    int extended_shape_idx = MLI_MAX_RANK - 1;
    int shape_idx = dst->rank - 1;
    int src_memstride = 1;
    for (; extended_shape_idx >= 0; --extended_shape_idx, --shape_idx) {
        if (shape_idx >= 0) {
            src_strides[extended_shape_idx] = src_memstride;
            dst_extended_shape[extended_shape_idx] = dst->shape[shape_idx];
            dst_strides[extended_shape_idx] = (dst->mem_stride[shape_idx] == 0) ? src_memstride
                : dst->mem_stride[shape_idx];
            src_memstride *= dst->shape[shape_idx];
        } else {
            src_strides[extended_shape_idx] = src_memstride;
            dst_extended_shape[extended_shape_idx] = 1;
            dst_strides[extended_shape_idx] = dst_strides[extended_shape_idx + 1];
        }
    }

    // Lambda to define a linear element position in memory using strides
    auto val_pos = [](int strides[MLI_MAX_RANK], int dim0_idx, int dim1_idx, int dim2_idx, int dim3_idx) -> int {
        return (strides[0] * dim0_idx) + (strides[1] * dim1_idx) + (strides[2] * dim2_idx) + (strides[3] * dim3_idx);
    };

    o_T* dst_arr = static_cast<o_T*>(dst->data.mem.void_p);
    int scale_dim = -1;
    int scales_num = 1;
    if ((dst_el_type == MLI_EL_SA_8 || dst_el_type == MLI_EL_SA_32) && dst->el_params.sa.dim >= 0) {
        scale_dim = dst->el_params.sa.dim + (MLI_MAX_RANK - dst->rank);
        scales_num = dst_extended_shape[scale_dim];
    }

    // Transformation will be applied on slices across scales dimension (or all tensor)
    for (int scale_idx = 0; scale_idx < scales_num; ++scale_idx) {
        // calculate current scale and zero offset.
        const int32_t scale_fraq_bits = mli_hlp_tensor_scale_shift(dst, scale_idx);
        float scale_val = (float)((int64_t)1l << abs(scale_fraq_bits));
        scale_val = scale_fraq_bits >= 0 ? scale_val : 1.f / scale_val;
        scale_val = scale_val / (float)mli_hlp_tensor_scale(dst, scale_idx);
        int16_t zero_offset = mli_hlp_tensor_zero_offset(dst, scale_idx);

        // calculate borders across all dimensions for slice where this scale is applicable.
        int dim_start[MLI_MAX_RANK] = { 0 };
        int dim_end[MLI_MAX_RANK] = { 0 };
        for (int i = 0; i < MLI_MAX_RANK; ++i) {
            dim_start[i] = (scale_dim == i) ? scale_idx : 0;
            dim_end[i] = (scale_dim == i) ? scale_idx + 1 : dst_extended_shape[i];
        }

        // Apply transformation of defined slice
        for (int dim0_idx = dim_start[0]; dim0_idx < dim_end[0]; ++dim0_idx) {
            for (int dim1_idx = dim_start[1]; dim1_idx < dim_end[1]; ++dim1_idx) {
                for (int dim2_idx = dim_start[2]; dim2_idx < dim_end[2]; ++dim2_idx) {
                    for (int dim3_idx = dim_start[3]; dim3_idx < dim_end[3]; ++dim3_idx) {
                        const int src_pos = val_pos(src_strides, dim0_idx, dim1_idx, dim2_idx, dim3_idx);
                        const int dst_pos = val_pos(dst_strides, dim0_idx, dim1_idx, dim2_idx, dim3_idx);
                        assert(src_pos < src_size);
                        assert(dst_pos < dst->data.capacity / sizeof(dst_arr[0]));
                        if (dst_el_type == MLI_EL_FP_32) {
                            dst_arr[dst_pos] = src[src_pos];
                        } else {
                            const float round_val = (src[src_pos] > 0) ? 0.5f : -0.5f;
                            const int32_t dst_val = static_cast<int32_t>(scale_val * src[src_pos] + round_val);
                            dst_arr[dst_pos] = saturate<int32_t, o_T>(dst_val + zero_offset);
                        }
                    }
                }
            }
        }
    }
}

//============================================================================================
// De-Quantize tensor data to float values 
// Main template of de-quantization routine
//====================================================================
template <mli_element_type src_el_type>
void tensor_quantizer::dequantize_tensor_data_routine(const mli_tensor* src, float* dst, uint32_t dst_size) {
    assert(src != nullptr && dst != nullptr);
    assert(validate_tensor(*src) == kOk);
    assert(src_el_type == src->el_type);
    assert(dst_size >= mli_hlp_count_elem_num(src, 0));
    assert(MLI_MAX_RANK == 4);

    // Put type traits magic to derive input type (derived_T) 
    typedef typename std::conditional<src_el_type == MLI_EL_FX_16, int16_t,
        typename std::conditional<src_el_type == MLI_EL_FX_8 || src_el_type == MLI_EL_SA_8, int8_t,
        typename std::conditional<src_el_type == MLI_EL_SA_32, int32_t, 
        typename std::conditional<src_el_type == MLI_EL_FP_32, float, void>::type >::type >::type >::type
        derived_T;

    // And use derived_T as o_T if it's in supported set (not a void). Otherwise, break compilation
    typedef typename std::enable_if<!std::is_same<derived_T, void>::value, derived_T>::type    i_T;

    // Extend shape to the MLI_MAX_RANK complimenting it with 1s in a front.
    // Calculate strides on input float array and output tensor 
    // for easier definition of element position in total arrays
    int dst_strides[MLI_MAX_RANK] = { 0 };
    int src_strides[MLI_MAX_RANK] = { 0 };
    int src_extended_shape[MLI_MAX_RANK] = { 0 };
    int extended_shape_idx = MLI_MAX_RANK - 1;
    int shape_idx = src->rank - 1;
    int dst_memstride = 1;
    for (; extended_shape_idx >= 0; --extended_shape_idx, --shape_idx) {
        if (shape_idx >= 0) {
            dst_strides[extended_shape_idx] = dst_memstride;
            src_extended_shape[extended_shape_idx] = src->shape[shape_idx];
            src_strides[extended_shape_idx] = (src->mem_stride[shape_idx] == 0) ? dst_memstride : src->mem_stride[shape_idx];
            dst_memstride *= src->shape[shape_idx];
        } else {
            dst_strides[extended_shape_idx] = dst_memstride;
            src_extended_shape[extended_shape_idx] = 1;
            src_strides[extended_shape_idx] = src_strides[extended_shape_idx + 1];
        }
    }

    // Lambda to define a linear element position in memory using strides
    auto val_pos = [](int strides[MLI_MAX_RANK], int dim0_idx, int dim1_idx, int dim2_idx, int dim3_idx) -> int {
        return (strides[0] * dim0_idx) + (strides[1] * dim1_idx) + (strides[2] * dim2_idx) + (strides[3] * dim3_idx);
    };

    i_T* src_arr = static_cast<i_T*>(src->data.mem.void_p);
    int scale_dim = -1;
    int scales_num = 1;
    if ((src_el_type == MLI_EL_SA_8 || src_el_type == MLI_EL_SA_32) && src->el_params.sa.dim >= 0) {
        scale_dim = src->el_params.sa.dim + (MLI_MAX_RANK - src->rank);
        scales_num = src_extended_shape[scale_dim];
    }

    // Transformation will be applied on slices across scales dimension (or all tensor)
    for (int scale_idx = 0; scale_idx < scales_num; ++scale_idx) {
        // calculate current scale and zero offset.
        const int32_t scale_fraq_bits = mli_hlp_tensor_scale_shift(src, scale_idx);
        float scale_val = (float)((int64_t)1l << abs(scale_fraq_bits));
        scale_val = scale_fraq_bits >= 0 ? scale_val : 1.f / scale_val;
        scale_val = (float)mli_hlp_tensor_scale(src, scale_idx) / scale_val;
        int16_t zero_offset = mli_hlp_tensor_zero_offset(src, scale_idx);

        // calculate borders across all dimensions for slice where this scale is applicable.
        int dim_start[MLI_MAX_RANK] = { 0 };
        int dim_end[MLI_MAX_RANK] = { 0 };
        for (int i = 0; i < MLI_MAX_RANK; ++i) {
            dim_start[i] = (scale_dim == i) ? scale_idx : 0;
            dim_end[i] = (scale_dim == i) ? scale_idx + 1 : src_extended_shape[i];
        }

        // Apply transformation of defined slice
        for (int dim0_idx = dim_start[0]; dim0_idx < dim_end[0]; ++dim0_idx) {
            for (int dim1_idx = dim_start[1]; dim1_idx < dim_end[1]; ++dim1_idx) {
                for (int dim2_idx = dim_start[2]; dim2_idx < dim_end[2]; ++dim2_idx) {
                    for (int dim3_idx = dim_start[3]; dim3_idx < dim_end[3]; ++dim3_idx) {
                        const int dst_pos = val_pos(dst_strides, dim0_idx, dim1_idx, dim2_idx, dim3_idx);
                        const int src_pos = val_pos(src_strides, dim0_idx, dim1_idx, dim2_idx, dim3_idx);
                        assert(dst_pos < dst_size);
                        assert(src_pos < src->data.capacity / sizeof(src_arr[0]));
                        if (src_el_type == MLI_EL_FP_32) {
                            dst[dst_pos] = src_arr[src_pos];
                        } else {
                            const float dst_val_unscaled = static_cast<float>(src_arr[src_pos]) - zero_offset;
                            dst[dst_pos] = dst_val_unscaled * scale_val;
                        }
                    }
                }
            }
        }
    }
}

} // namespace tst
} // namespace mli
