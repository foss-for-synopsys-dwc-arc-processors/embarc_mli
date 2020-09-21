/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "test_quantizer.h"

#include <algorithm>
#include <limits>
#include <memory>

#include "tensor_transform.h"
#include "tests_aux.h"

// Assert wrapper: works only in DBG_MODE_FULL and DBG_MODE_DEBUG
// TODO: Replace with something external
#if defined(DEBUG)
#include <assert.h>
#define ASSERT(cond) assert(cond)
#else
#define ASSERT(cond) (void)(cond)
#endif

namespace mli {
namespace tst {

//======================================================================================================
//
// Methods of Quantizer class
//
//======================================================================================================

// Constructor intended to be used (but not limited) with SA type of tensors
//==========================================
tensor_quantizer::tensor_quantizer(mli_tensor tsr, const float* data, uint32_t data_size, 
                                   const float* scales, uint32_t scales_size, 
                                   const float* zero_points, uint32_t zero_points_size, 
                                   const int8_t* scales_fraq_bits, uint32_t scales_fraq_bits_size)
        : source_tsr_(tsr)
        , source_data_(data)
        , source_scales_(scales)
        , source_zero_points_(zero_points)
        , source_scales_fraq_(scales_fraq_bits)
        , is_valid_(false) {
    // check that input tensor in general not bad
    const tensor_state state = validate_tensor(tsr);
    is_valid_ = (state == kIncompleteMem || state == kOk);

    // check that arrays are of expectd size
    if (is_valid_) {
        is_valid_ = (data_size == mli_hlp_count_elem_num(&tsr, 0));
        if ((tsr.el_type == MLI_EL_SA_8 || tsr.el_type == MLI_EL_SA_32)) {
            const uint32_t expected_vals = (tsr.el_params.sa.dim < 0) ? 1 : tsr.shape[tsr.el_params.sa.dim];
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
tensor_quantizer::tensor_quantizer(mli_tensor tsr, const float* data, uint32_t data_size)
        : source_tsr_(tsr)
        , source_data_(data)
        , source_scales_(nullptr)
        , source_zero_points_(nullptr)
        , source_scales_fraq_(nullptr)
        , is_valid_(false) {
    // check that input tensor in general not bad
    const tensor_state state = validate_tensor(tsr);
    is_valid_ = (state == kIncompleteMem || state == kOk);
    is_valid_ &= (tsr.el_type != MLI_EL_SA_8 && tsr.el_type != MLI_EL_SA_32);
}


// Return complete tensor with source float point data
//=================================================================================
const mli_tensor tensor_quantizer::get_source_float_tensor() const {
    mli_tensor ret_tsr = source_tsr_; //TODO return 0 if not ok
    if (is_valid_) {
        ret_tsr.data.capacity = sizeof(source_data_[0]) * mli_hlp_count_elem_num(&ret_tsr, 0);
        uint32_t cur_memstr = 1;
        for (int i = ret_tsr.rank - 1; i >= 0; --i) {
            ret_tsr.mem_stride[i] = cur_memstr;
            cur_memstr *= ret_tsr.shape[i];
        }
        ret_tsr.el_type = MLI_EL_FP_32;

        // As we return const tensor just after this assignment, 
        // removing cv qualfiers here is considered as and acceptable.
        ret_tsr.data.mem.pf32 = const_cast<float*>(source_data_);
    }
    return ret_tsr;
}

// Return complete quantized tensor using source tensor, data and quant params
//=================================================================================
mli_tensor tensor_quantizer::get_quantized_tensor(mli_data_container memory) const {
    if (!is_valid_)
        return mli_tensor{ 0 };

    mli_tensor ret_tsr = get_not_quantized_tensor(memory);
    if (ret_tsr.data.mem.pi8 != nullptr) {
        mli_status fp_to_fx_stat = mli_hlp_float_to_fx_tensor(source_data_, mli_hlp_count_elem_num(&ret_tsr, 0), 
                                                              &ret_tsr);
        ASSERT(fp_to_fx_stat == MLI_STATUS_OK);
    }

    return ret_tsr;
}

// Return tensor structure populated with source tensor fields and quant params ONLY
//=================================================================================
mli_tensor tensor_quantizer::get_not_quantized_tensor(mli_data_container memory) const {
    if (!is_valid_)
        return mli_tensor{ 0 };
    
    mli_tensor ret_tsr = source_tsr_;
    const bool is_mem_spread_ok = spread_memory(&ret_tsr, &memory);

    // For SA format preparation of quantization parameters is required
    if (is_mem_spread_ok && (ret_tsr.el_type == MLI_EL_SA_8 || ret_tsr.el_type == MLI_EL_SA_32)) {
        int num_vals = (ret_tsr.el_params.sa.dim < 0) ? 1 : ret_tsr.shape[ret_tsr.el_params.sa.dim];
        int16_t* scale_dst = (num_vals > 1) ? ret_tsr.el_params.sa.scale.mem.pi16 
                                            : &ret_tsr.el_params.sa.scale.mem.i16;
        int16_t* zp_dst = (num_vals > 1) ? ret_tsr.el_params.sa.zero_point.mem.pi16
                                         : &ret_tsr.el_params.sa.zero_point.mem.i16;
        int8_t* frac_dst = (num_vals > 1) ? ret_tsr.el_params.sa.scale_frac_bits.mem.pi8
                                          : &ret_tsr.el_params.sa.scale_frac_bits.mem.i8;

        for (int i = 0; i < num_vals; i++) {
            const int8_t scale_fraq_bits = source_scales_fraq_[i];
            const uint32_t mult = (uint32_t)1l << scale_fraq_bits;
            const float round_val = 0.5f;
            const int32_t dst_val = (int32_t)(mult * source_scales_[i] + round_val);
            const int32_t zero_val = (int32_t)(-source_zero_points_[i] / source_scales_[i] + round_val);

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

// Calculate capacity requirement of data field of the tensor taking memstride into account
//=================================================================================
uint32_t tensor_quantizer::get_required_data_capacity(const mli_tensor& tsr) {
    // TODO: Need to check tensor
    uint32_t ret_val = 0;
    if (tsr.mem_stride[0] == 0) {
        ret_val = mli_hlp_tensor_element_size(&tsr) * mli_hlp_count_elem_num(&tsr, 0);
    } else {
        // Method that implie removing "trash tail
        for (int idx = 0; idx < tsr.rank; ++idx)
            ret_val += tsr.mem_stride[idx] * (tsr.shape[idx] - 1);
        ret_val += 1;
        ret_val *= mli_hlp_tensor_element_size(&tsr);

        // ROugh method which implies allocation for tail
        //ret_val = tsr.mem_stride[0] * tsr.shape[0]; 
    }
    return ret_val;
}


// Assign void pointer to a proper union field of tensor
//=================================================================================
bool  tensor_quantizer::tensor_assign_data_ptr(mli_tensor* tsr, void* ptr) {
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
    case MLI_EL_LARGE_ENUM:
        tsr->data.mem.pi8 = nullptr;
        return false;
    };
}

// Spread provided memory across tensor's containers: data and quantization params
//=================================================================================
bool tensor_quantizer::spread_memory(mli_tensor* tsr, const mli_data_container* data_mem, 
                                     const mli_data_container* quant_params_mem) {
    // This functions assigns provided memory to tensor data according to tensor type and alignment needs.
    // It takes tensor to be populated with data pointers and data containers as a memory donors.
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
    if (tsr == nullptr /*|| data_mem == nullptr || data_mem->mem.pi8 == nullptr*/)
        return false;

    // First fill required minimal mem capacity
    //=================================================
    tsr->data.capacity = get_required_data_capacity(*tsr);
    if ((tsr->el_type == MLI_EL_SA_8 || tsr->el_type == MLI_EL_SA_32)) {
        const int num_vals = (tsr->el_params.sa.dim >= 0) ? tsr->shape[tsr->el_params.sa.dim] : 1;
        tsr->el_params.sa.scale.capacity = (num_vals > 1) ? sizeof(int16_t) * num_vals : 0;
        tsr->el_params.sa.zero_point.capacity = (num_vals > 1) ? sizeof(int16_t) * num_vals : 0;
        tsr->el_params.sa.scale_frac_bits.capacity = (num_vals > 1) ? sizeof(int8_t) * num_vals : 0;
    }

    // If no data container is provided, just return memory requirements
    bool success = !(data_mem == nullptr || data_mem->mem.pi8 == nullptr);
    
    // Asign aligned memory for scales if it is needed
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
        tsr->el_params.sa.scale_frac_bits.capacity += alignof(int8_t) - 1;
        tsr->el_params.sa.zero_point.capacity += alignof(int16_t) - 1;
        tsr->el_params.sa.scale.capacity += alignof(int16_t) - 1;
    }
    return success;
}


// Validate tensor in terms of three state: Bad, Incomplete, Ok
//=======================================================
tensor_quantizer::tensor_state tensor_quantizer::validate_tensor(const mli_tensor& tsr) {
    // First check general tensor parameters like shape, rank, memstride, 
    // quantization dimension of sa type if applicable.
    // If they are broken - tensor is considered as bad and can't be used at all
    if (mli_hlp_tensor_element_size(&tsr) == 0 || tsr.rank <= 0 || tsr.rank > MLI_MAX_RANK ||
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

    // Next need to define whether memory is properly assignet for data 
    // and additional fields like scales/zero_points.
    // If they are broken - tensor is considered as incomplete and requires proper memory assignment
    if (tsr.data.mem.pi16 == nullptr || get_required_data_capacity(tsr) < tsr.data.capacity)
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


} // namespace tst
} // namespace mli
