/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_USER_TESTS_TEST_CRC32_CALC_H_
#define _MLI_USER_TESTS_TEST_CRC32_CALC_H_

#include "mli_api.h"

namespace mli {
namespace tst {

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

} // namespace tst 
} // namespace mli

#endif //_MLI_USER_TESTS_TEST_CRC32_CALC_H_

