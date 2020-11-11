/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "test_memory_manager.h"

// Standard asserts should be intentionally turned-on by defenition of TEST_DEBUG.
#if !defined(TEST_DEBUG)
#define NDEBUG
#endif

#include <assert.h>

#include "mli_api.h"

namespace mli {
namespace tst {

//=======================================================================
//
// Module to handle and check externally allocated memory for test needs
//
//=======================================================================

// Default constructor
//=========================================
memory_manager::memory_manager()
    : source_memory_(nullptr)
    , source_mem_size_(0)
    , allocated_memory_start_(nullptr)
    , allocated_mem_size_(0)
    , head_mem_crc_()
    , tail_mem_crc_()
{}

// Parametrized constructor
//=========================================
memory_manager::memory_manager(int8_t* memory, uint32_t mem_size)
    : source_memory_(memory)
    , source_mem_size_(mem_size)
    , allocated_memory_start_(nullptr)
    , allocated_mem_size_(0)
    , head_mem_crc_()
    , tail_mem_crc_()
{}

// Allocate memory of exact size
//=========================================
mli_data_container memory_manager::allocate_memory(uint32_t size, uint32_t fill_pattern) {
    mli_data_container ret_val{ 0 };
    if (source_memory_ == nullptr || allocated_memory_start_ != nullptr || source_mem_size_ < size )
        return ret_val;

    // Fill the whole memory region with a pre-defined pattern
    int pattern_byte = 0;
    for (uint32_t idx = 0; idx < source_mem_size_; ++idx, ++pattern_byte) {
        pattern_byte = pattern_byte % sizeof(fill_pattern);
        int shift = (sizeof(fill_pattern) - 1 - pattern_byte) * 8;
        source_memory_[idx] = static_cast<int8_t>((fill_pattern >> shift) & 0xFF);
    }

    if (size == source_mem_size_) {
        // If the whole memory region is requested, there is no need to keep valid CRC32 for head and tail
        // Return the whole memory
        head_mem_crc_.reset();
        tail_mem_crc_.reset();
        allocated_memory_start_ = source_memory_;
        allocated_mem_size_ = source_mem_size_;
    } else {
        // otherwise, we need to return middle sub-region and keep CRC32 checksums for head and tail 
        const uint32_t head_size = (source_mem_size_ - size) / 2;
        const uint32_t tail_size = source_mem_size_ - head_size - size;
        allocated_memory_start_ = source_memory_ + head_size;
        allocated_mem_size_ = size;
        head_mem_crc_(source_memory_, head_size);
        tail_mem_crc_(allocated_memory_start_ + allocated_mem_size_, tail_size);
    }

    ret_val.capacity = allocated_mem_size_;
    ret_val.mem.pi8 = allocated_memory_start_;
    return ret_val;
}


// Allocate memory according to quantizer requirements
//===================================================
mli_data_container memory_manager::allocate_memory(const tensor_quantizer& quant_unit, uint32_t fill_pattern) {
    // First get memory requirements from quantizer
    mli_data_container empty_container{ 0 };
    mli_tensor tensor_with_requirements = quant_unit.get_not_quantized_tensor(empty_container);

    // Sort out cases with bad or complete tensors
    if (tensor_quantizer::validate_tensor(tensor_with_requirements) != tensor_quantizer::kIncompleteMem)
        return empty_container;

    // calculate how much memory is needed
    uint32_t required_size = tensor_with_requirements.data.capacity;
    if (tensor_with_requirements.el_type == MLI_EL_SA_8 || tensor_with_requirements.el_type == MLI_EL_SA_32) {
        required_size += tensor_with_requirements.el_params.sa.scale.capacity;
        required_size += tensor_with_requirements.el_params.sa.scale_frac_bits.capacity;
        required_size += tensor_with_requirements.el_params.sa.zero_point.capacity;
    }

    return allocate_memory(required_size, fill_pattern);
}

// Mark memory as unused 
//===================================================
void memory_manager::return_memory(){
    head_mem_crc_.reset();
    tail_mem_crc_.reset();
    allocated_memory_start_ = nullptr;
    allocated_mem_size_ = 0;
}

// Check that Head and tail regions are not corrupted
//===================================================
bool memory_manager::is_memory_corrupted() const {
    if (source_memory_ == nullptr || allocated_memory_start_ == nullptr ||
            allocated_memory_start_ <= source_memory_  ||
            allocated_mem_size_ >= source_mem_size_)
        return false;
    
    const uint32_t head_size = static_cast<uint32_t>(allocated_memory_start_ - source_memory_);
    const uint32_t tail_size = source_mem_size_ - (head_size + allocated_mem_size_);
    crc32_calc head_current_crc, tail_current_crc;
    head_current_crc(source_memory_, head_size);
    tail_current_crc(allocated_memory_start_ + allocated_mem_size_, tail_size);

    assert(head_mem_crc_.is_valid() == head_current_crc.is_valid()); 
    assert(tail_mem_crc_.is_valid() == tail_current_crc.is_valid());
    bool is_corrupted = head_mem_crc_.get() != head_current_crc.get();
    is_corrupted |= tail_mem_crc_.get() != tail_current_crc.get();
    return is_corrupted;
}

} // namespace tst
} // namespace mli
