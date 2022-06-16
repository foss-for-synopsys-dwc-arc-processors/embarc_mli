/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_USER_TESTS_TEST_RESCALE_UTILITY_H_
#define _MLI_USER_TESTS_TEST_RESCALE_UTILITY_H_

#include <vector>

#include "mli_api.h"

namespace mli {
namespace tst {


//===============================================================================================
// Module to calculate scale values for the Rescale operation
//===============================================================================================
class scales_calc {
public:
    scales_calc() {}
    // Parametrised constructor for just in-to-out
    scales_calc(float in_scale, float out_scale, float multiplier = 1);
    scales_calc(float in_scale, float out_scale,
                const float* w_scales, size_t w_scales_num);

    const std::vector<int16_t>& get_scales_vec() const {return scales_val_vec;}
    const std::vector<int8_t>& get_shift_vec() const {return scales_shift_vec;}

    const mli_tensor& get_scales_tsr() const {return scales_val_tsr;}
    const mli_tensor& get_shift_tsr() const {return scales_shift_tsr;}

private:
    std::vector<int16_t> scales_val_vec;
    std::vector<int8_t> scales_shift_vec;
    mli_tensor scales_val_tsr;
    mli_tensor scales_shift_tsr;
};


class bias_folder {
public:
    bias_folder() {}
    // Parametrised constructor for just in-to-out
    bias_folder(const mli_tensor& b_tsr);
    bias_folder(const mli_tensor& b_tsr, const mli_tensor& in_tsr,
                const mli_tensor& w_tsr);

    // TODO: Support double-wide accum
    const std::vector<int32_t>& get_bias_vec() const {return bias_vec;};
    const mli_tensor& get_bias_tsr() const {return bias_tsr;};

private:
    std::vector<int32_t> bias_vec;
    mli_tensor bias_tsr;
};

void vectorize_single_elem_tensor(mli_tensor& dst_tsr,
                                  const mli_tensor& src_tsr,
                                  void* data);

} // namespace tst
} // namespace mli

#endif //_MLI_USER_TESTS_TEST_RESCALE_UTILITY_H_

