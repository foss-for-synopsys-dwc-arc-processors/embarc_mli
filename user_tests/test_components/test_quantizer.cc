/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "test_quantizer.h"

#include <memory>

// Assert wrapper: works only in DBG_MODE_FULL and DBG_MODE_DEBUG
#if defined(DEBUG)
#include <assert.h>
#define ASSERT(cond) assert(cond)
#else
#define ASSERT(cond) (void)(cond)
#endif

#include "tensor_transform.h"
#include "tests_aux.h"

namespace mli {
namespace tst {

//======================================================================================================
// Quantizer class: implies to quantize float data to MLI compatible data using quantization parameters.
//======================================================================================================
tensor_quantizer::tensor_quantizer(mli_tensor tsr, const float* data, uint32_t data_size, 
                                   const float* scales, uint32_t scales_size, 
                                   const float* zero_points, uint32_t zero_points_size)
        : source_tsr_(tsr)
        , source_data_(data)
        , source_scales_(scales)
        , source_zero_points_(zero_points)
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
        }
    }
}

tensor_quantizer::tensor_quantizer(mli_tensor tsr, const float* data, uint32_t data_size)
        : source_tsr_(tsr)
        , source_data_(data)
        , source_scales_(nullptr)
        , source_zero_points_(nullptr)
        , is_valid_(false) {
    // check that input tensor in general not bad
    const tensor_state state = validate_tensor(tsr);
    is_valid_ = (state == kIncompleteMem || state == kOk);
    is_valid_ &= (tsr.el_type != MLI_EL_SA_8 && tsr.el_type != MLI_EL_SA_32);
}


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

mli_tensor tensor_quantizer::get_not_quantized_tensor(mli_data_container memory) const {
    if (!is_valid_)
        return mli_tensor{ 0 };
    
    mli_tensor ret_tsr = source_tsr_;
    const bool is_mem_spread_ok = spread_memory(&ret_tsr, &memory);

    // For SA format preparation of quantization parameters is required
    if (is_mem_spread_ok && (ret_tsr.el_type == MLI_EL_SA_8 || ret_tsr.el_type == MLI_EL_SA_32)) {
        int num_vals = (ret_tsr.el_params.sa.dim < 0) ? 1 : ret_tsr.shape[ret_tsr.el_params.sa.dim];
        const int int_bits = sizeof(int32_t) * 8 - ret_tsr.el_params.sa.scale_frac_bits - 1;

        test_status fill_sa_stat = fill_asym_tensor_element_params(source_scales_, source_zero_points_, 
                                                                   num_vals, int_bits, &ret_tsr);
        ASSERT(fill_sa_stat == TEST_PASSED);
    }

    return ret_tsr;
}

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

        // ROugh method which implies allocation for "trash" tail
        //ret_val = tsr.mem_stride[0] * tsr.shape[0]; 
    }
    
    /*
    ret_val += alignof(mli_hlp_tensor_element_size(&tsr)) - 1; // To take alignment into account,

    if ((ret_tsr.el_type == MLI_EL_SA_8 || ret_tsr.el_type == MLI_EL_SA_32) && ret_tsr.el_params.sa.dim >= 0) {
        // Memory for scale ratio and zero points also required
        ret_val += sizeof(int32_t) * ret_tsr.shape[ret_tsr.el_params.sa.dim];
        ret_val += sizeof(int16_t) * ret_tsr.shape[ret_tsr.el_params.sa.dim];
        
        ret_val += alignof(int32_t) - 1; // To take alignment of scales into account,
        ret_val += alignof(int16_t) - 1; // To take alignment of zero_point into account
    }
    */
    return ret_val;

}

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

bool tensor_quantizer::spread_memory(mli_tensor* tsr, const mli_data_container* data_mem, 
                                     const mli_data_container* scales_mem,
                                     const  mli_data_container* zp_mem) {

    // This functions assigns provided memory to tensor data according to tensor type and alignment needs.
    // It takes tensor to be populated with data pointers and several data containers as a memory donors.
    // tsr structure must contains properly filled shape, rank and elemet type data. 
    // data_mem container must keerp pointer and capacity to a valid memory which is sufficint for tsr needs 
    // Scales and zero point memory containers are optional even for SA type of tensor. 
    // For SA tensors if no scales_mem and zp_mem is provided, data_mem is used as the only donor for all
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
        tsr->el_params.sa.scale.capacity = (num_vals > 1) ? sizeof(int32_t) * num_vals : 0;
        tsr->el_params.sa.zero_point.capacity = (num_vals > 1) ? sizeof(int16_t) * num_vals : 0;
    }

    // If no data container is provided, just return memory requirements
    bool success = !(data_mem == nullptr || data_mem->mem.pi8 == nullptr);
    
    // Asign aligned memory for scales if it is needed
    //=================================================
    // If there are no specific containers for scales or zero points we will use data memory for it
    int8_t* mem_for_spare = data_mem->mem.pi8;
    uint32_t available_cap = data_mem->capacity;
    if (success && (tsr->el_type == MLI_EL_SA_8 || tsr->el_type == MLI_EL_SA_32)
            && tsr->el_params.sa.scale.capacity != 0) {
        const bool use_spare_mem = (scales_mem == nullptr);
        void* scale_ptr = static_cast<void*>((use_spare_mem)? mem_for_spare : scales_mem->mem.pi8);
        size_t scale_cap = static_cast<size_t>((use_spare_mem) ? available_cap : scales_mem->capacity);

        tsr->el_params.sa.scale.mem.pi32 = static_cast<int32_t *>(
            std::align(alignof(int32_t), tsr->el_params.sa.scale.capacity, scale_ptr, scale_cap));
        
        if (tsr->el_params.sa.scale.mem.pi32 == nullptr) {
            success = false;
        } else if (use_spare_mem) {
            // if spare memory was succesfully used, we need to update ptr and capacity
            mem_for_spare = static_cast<int8_t*>(scale_ptr) + tsr->el_params.sa.scale.capacity;
            available_cap = static_cast<uint32_t>(scale_cap) - tsr->el_params.sa.scale.capacity;
        }
    }

    // Asign aligned memory in a similar way for zero points if it is needed
    //=================================================
    if (success && (tsr->el_type == MLI_EL_SA_8 || tsr->el_type == MLI_EL_SA_32)
            && tsr->el_params.sa.zero_point.capacity != 0) {
        const bool use_spare_mem = (zp_mem == nullptr);
        void* zp_ptr = static_cast<void*>((use_spare_mem) ? mem_for_spare : zp_mem->mem.pi8);
        size_t zp_cap = static_cast<size_t>((use_spare_mem) ? available_cap : zp_mem->capacity);

        tsr->el_params.sa.zero_point.mem.pi16 = static_cast<int16_t*>(
            std::align(alignof(int16_t), tsr->el_params.sa.zero_point.capacity, zp_ptr, zp_cap));
        if (tsr->el_params.sa.zero_point.mem.pi16 == nullptr) {
            success = false;
        } else if (use_spare_mem) {
            mem_for_spare = static_cast<int8_t*>(zp_ptr) + tsr->el_params.sa.zero_point.capacity;
            available_cap = static_cast<uint32_t>(zp_cap) - tsr->el_params.sa.zero_point.capacity;
        }
    }
    
    // Align memory for data and assign it to tensor
    //=================================================
    void* data_ptr = static_cast<void*>(mem_for_spare);
    size_t data_cap = static_cast<size_t>(available_cap);
    if (success && std::align(mli_hlp_tensor_element_size(tsr), tsr->data.capacity, data_ptr, data_cap) != nullptr) {
        success &= tensor_assign_data_ptr(tsr, data_ptr);
    } else {
        success = false;
    }

    if (!success) {
        // Provide info on required memory and clear pointer to data
        tsr->data.mem.pi8 = tsr->el_params.sa.scale.mem.pi8 = tsr->el_params.sa.zero_point.mem.pi8 = nullptr;
        tsr->data.capacity += mli_hlp_tensor_element_size(tsr) - 1;
        tsr->el_params.sa.zero_point.capacity += alignof(int16_t) - 1;
        tsr->el_params.sa.scale.capacity += alignof(int32_t) - 1;
    }
    return success;
}

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
                tsr.el_params.sa.scale.capacity < num_values * sizeof(int32_t) ||
                tsr.el_params.sa.zero_point.capacity < num_values * sizeof(int16_t)) {
            return kIncompleteMem;
        }
    }

    // Otherwise tensor is Ok
    return kOk;
}


} // namespace tst
} // namespace mli
