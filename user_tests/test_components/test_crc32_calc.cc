/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "test_crc32_calc.h"

// Standard asserts should be intentionally turned-on by defenition of TEST_DEBUG.
#if !defined(TEST_DEBUG)
#define NDEBUG
#endif

#include <assert.h>

#include "mli_api.h"

namespace mli {
namespace tst {


//===============================================================================================
//
// Methods and data of the Module to calculate and handle CRC32 sum.
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

// Get accumulated CRC sum
//================================
uint32_t crc32_calc::get() const {
    return crc32_sum_;
}

// Get status of CRC instance
//================================
bool crc32_calc::is_valid() const {
    return valid_crc32_;
}

// Accumulate CRC using tensor data
//=========================================
// Note: It was planned to calculate CRC only using valid values of tensor via proper 
// handling of memstrides. But as we have an opportunity to initialize memory before usage
// it found out that calculating CRC over all tensor's data container (pointer + capacity)
// is beneficial as we can track memory corruption of inner tensor data ("holes" that excluded by memstride)
// TODO: add an optional flag to calculated CRC on valid data, if needed
uint32_t crc32_calc::operator()(const mli_tensor& in) {
    const int8_t* current = in.data.mem.pi8;
    uint32_t length = in.data.capacity;

    return (*this)(current, length);
}

// Accumulate CRC using array
//=========================================
uint32_t crc32_calc::operator()(const int8_t* in, uint32_t size) {
    if (in != nullptr && size != 0) {
        uint32_t crc = ~crc32_sum_;  // same as previousCrc32 ^ 0xFFFFFFFF

        while (size-- != 0) {
            const uint8_t current_val = static_cast<uint8_t>(*in);
            crc = crc32_lookup_table_[(crc ^ current_val) & 0x0F] ^ (crc >> 4);
            crc = crc32_lookup_table_[(crc ^ (current_val >> 4)) & 0x0F] ^ (crc >> 4);
            in++;
        }

        crc32_sum_ = ~crc; // same as crc ^ 0xFFFFFFFF
        valid_crc32_ = true;
    }
    return crc32_sum_;
}

} // namespace tst
} // namespace mli
