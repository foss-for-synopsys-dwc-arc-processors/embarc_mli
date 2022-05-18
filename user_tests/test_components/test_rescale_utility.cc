/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "test_rescale_utility.h"

// Standard asserts should be intentionally turned-on by defenition of TEST_DEBUG.
#if !defined(TEST_DEBUG)
#define NDEBUG
#endif

#include<assert.h>
#include<math.h>

namespace mli {
namespace tst {

//======================================================================================================
//
// Internal helper functions
//
//======================================================================================================

inline int16_t val_to_fx(float val, int8_t &val_shift) {
    // TODO: Subject for review (asserts at least and shifting ranges)
    int exp_value = 0;
    frexpf(val, &exp_value);
    val_shift = 15 - exp_value;
    int16_t fx_val = (int16_t)((1ll << val_shift) * val + 0.5f);
    return fx_val;
}

static mli_tensor flat_tensor_common(size_t tsr_size) {
    mli_tensor ret_tsr{0};
    ret_tsr.el_params.fx.frac_bits = 0;
    ret_tsr.rank = 1;
    ret_tsr.shape[0] = tsr_size;
    ret_tsr.mem_stride[0] = 1;
    return ret_tsr;
}

static mli_tensor flat_tensor(int16_t* data, size_t tsr_size) {
    assert(data != nullptr);
    mli_tensor ret_tsr = flat_tensor_common(tsr_size);
    ret_tsr.data.mem.pi16 = data;
    ret_tsr.data.capacity = tsr_size * sizeof(data[0]);
    ret_tsr.el_type = MLI_EL_FX_16;
    return ret_tsr;
}

static mli_tensor flat_tensor(int8_t* data, size_t tsr_size) {
    assert(data != nullptr);
    mli_tensor ret_tsr = flat_tensor_common(tsr_size);
    ret_tsr.data.capacity = tsr_size * sizeof(data[0]);
    ret_tsr.data.mem.pi8 = data;
    ret_tsr.el_type = MLI_EL_FX_8;
    return ret_tsr;
}

static mli_tensor flat_tensor(int32_t* data, size_t tsr_size) {
    assert(data != nullptr);
    mli_tensor ret_tsr = flat_tensor_common(tsr_size);
    ret_tsr.data.capacity = tsr_size * sizeof(data[0]);
    ret_tsr.data.mem.pi32 = data;
    ret_tsr.el_type = MLI_EL_SA_32;
    ret_tsr.el_params.sa.dim = -1;
    ret_tsr.el_params.sa.scale.capacity = 0;
    ret_tsr.el_params.sa.scale.mem.i16 = 1;
    ret_tsr.el_params.sa.type = MLI_EL_PARAM_SC16_ZP16;
    ret_tsr.el_params.sa.scale_frac_bits.capacity = 0;
    ret_tsr.el_params.sa.scale_frac_bits.mem.i8 = 0;
    ret_tsr.el_params.sa.zero_point.capacity = 0;
    ret_tsr.el_params.sa.zero_point.mem.i16 = 0;
    return ret_tsr;
}

//======================================================================================================
//
// Internal helper functions
//
//======================================================================================================

scales_calc::scales_calc(float in_scale, float out_scale)
        : scales_val_vec(1)
        , scales_shift_vec(1)
        , scales_val_tsr(flat_tensor(scales_val_vec.data(), 1))
        , scales_shift_tsr(flat_tensor(scales_shift_vec.data(), 1)) {
    assert(out_scale != 0.0f);
    scales_val_vec[0] = val_to_fx(in_scale/out_scale, scales_shift_vec[0]);
}

scales_calc::scales_calc(float in_scale, float out_scale,
                       const float* w_scales, size_t w_scales_num)
        : scales_val_vec(w_scales_num)
        , scales_shift_vec(w_scales_num)
        , scales_val_tsr(flat_tensor(scales_val_vec.data(), w_scales_num))
        , scales_shift_tsr(flat_tensor(scales_shift_vec.data(), w_scales_num)) {
    // Vector on the dynamic memory might looks like a  bad choice since in the past
    // we used such operands in calculations and asumed they are in the fast memory.
    // But nowadays we first may use them on the CS side, which dont care about fast memory and any
    // other optimization - we just prepare data to the opaque form for the following inference
    assert(w_scales_num > 0);
    assert(w_scales != nullptr);
    assert(out_scale != 0.0f);
    const float in_to_out_scale = in_scale / out_scale;
    for (size_t i = 0; i < w_scales_num; ++i)
        scales_val_vec[i] = val_to_fx(in_to_out_scale* w_scales[i], scales_shift_vec[i]);
}



bias_folder::bias_folder(const mli_tensor& b_tsr, const mli_tensor& in_tsr,
                         const mli_tensor& w_tsr)
        : bias_vec(b_tsr.shape[0])
        , bias_tsr(flat_tensor(bias_vec.data(), bias_vec.size())){
    assert(b_tsr.rank == 1);
    assert(b_tsr.el_type == MLI_EL_SA_32);
    assert(in_tsr.el_type == MLI_EL_SA_8);
    assert(w_tsr.el_type == MLI_EL_SA_8);

    // Extend shape to the MLI_MAX_RANK complimenting it with 1s in a front.
    // for easier definition of element position in total array
    int w_strides[MLI_MAX_RANK] = { 0 };
    int w_extended_shape[MLI_MAX_RANK] = { 0 };
    int extended_shape_idx = MLI_MAX_RANK - 1;
    int shape_idx = w_tsr.rank - 1;
    int dst_memstride = 1;
    for (; extended_shape_idx >= 0; --extended_shape_idx, --shape_idx) {
        if (shape_idx >= 0) {
            w_extended_shape[extended_shape_idx] = w_tsr.shape[shape_idx];
            dst_memstride = w_tsr.mem_stride[shape_idx];
            w_strides[extended_shape_idx] = dst_memstride;
        } else {
            w_strides[extended_shape_idx] = dst_memstride;
            w_extended_shape[extended_shape_idx] = 1;
        }
    }

    // Lambda to define a linear element position in memory using strides
    auto val_pos = [](int strides[MLI_MAX_RANK], int pos[MLI_MAX_RANK]) -> int {
        return (strides[0] * pos[0]) + (strides[1] * pos[1]) + (strides[2] * pos[2]) + (strides[3] * pos[3]);
    };

    // currently we asume only 8bit weights.
    const auto in_zp = in_tsr.el_params.sa.zero_point.mem.i8;

    // Init bias values. Negative as rescale op implies subtraction of bias_in
    for (size_t i = 0; i < bias_vec.size(); ++i)
        bias_vec[i] = -b_tsr.data.mem.pi32[i];

    // We don't care about performance right now.
    assert(MLI_MAX_RANK == 4);
    int pos[MLI_MAX_RANK] = {0};
    for (pos[0] = 0; pos[0] < w_extended_shape[0]; ++pos[0]) {
        for (pos[1] = 0; pos[1] < w_extended_shape[1]; ++pos[1]) {
            for (pos[2] = 0; pos[2] < w_extended_shape[2]; ++pos[2]) {
                for (pos[3] = 0; pos[3] < w_extended_shape[3]; ++pos[3]) {

                    const int w_pos = val_pos(w_strides, pos);
                    const int b_pos = pos[3]; // Asum filters is an innermost channel
                    bias_vec[b_pos] += in_zp * w_tsr.data.mem.pi8[w_pos];

                    // TODO: weights zero point is expected to be equal to zero.
                    // Consider to support in future (by uncommenting with array check).
                    // bias_vec[b_pos] -= in_zp * w_tsr.el_params.sa.zero_point.mem.pi8[b_pos]
                }
            }
        }
    }
}

} // namespace tst
} // namespace mli
