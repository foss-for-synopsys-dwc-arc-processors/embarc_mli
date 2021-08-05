/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_RELU_REF_H_
#define _MLI_KRN_RELU_REF_H_

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_mem_info.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace ref {

template<typename io_T>
static MLI_FORCE_INLINE void compute_relu_inner_loop(
        const MLI_PTR(io_T) __restrict vec_in,
        MLI_OUT_PTR(io_T) __restrict vec_out,
        const io_T min_val,
        const io_T max_val,
        const int count) {
    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(vec_in);
    int num_lanes = get_number_lanes(input);
    int remaining_part = count & (num_lanes - 1);

    if (remaining_part) {
        input = mli_prv_load_1vec(vec_in);
        mli_prv_store_n_samples(vec_out, mli_math_min_fx(
            mli_math_max_fx(input, min_val), max_val), remaining_part);
        vec_in  += remaining_part;
        vec_out += remaining_part;
    }
    for (int pos3 = remaining_part; pos3 < count; pos3 += num_lanes) {
        input = mli_prv_load_1vec(vec_in);
        mli_prv_store_n_samples(vec_out, mli_math_min_fx(
            mli_math_max_fx(input, min_val), max_val));
        vec_in  += num_lanes;
        vec_out += num_lanes;
    }
}

template <typename io_T, bool asym>
static MLI_FORCE_INLINE mli_status mli_krn_relu_fx_run(const mli_tensor *in, 
        const mli_relu_cfg *cfg, mli_tensor *out) {

    mli_prv_fx_init_dsp_ctrl();
    
    const MLI_PTR(io_T) __restrict vec_in = mli_prv_tensor_data_ptr<MLI_PTR(io_T)>(in);
    MLI_OUT_PTR(io_T) __restrict vec_out = mli_prv_tensor_data_ptr<MLI_OUT_PTR(io_T)>(out);

    /* Get Relu Limits */
    const mli_minmax_t limits = mli_prv_get_relu_limits<io_T, asym>(cfg, in);
    const io_T min_val = static_cast<io_T>(limits.min);
    const io_T max_val = static_cast<io_T>(limits.max);

    int shape = mli_prv_squash_tensor_to_one_dim(in, out);

    out->el_params = in->el_params;

    if(shape) {
        mli::krn::compute_relu_inner_loop<io_T>(vec_in, vec_out, min_val, max_val, shape);
    } else {
        /* Get Generic Private Tensor */
        auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);
        auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(io_T)>(out);
        /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
        mli_prv_squash_generic_tensor<MLI_PTR(io_T)>(&in_prv, &out_prv);

        const MLI_PTR(io_T) __restrict orig_vec_in = vec_in;
        MLI_OUT_PTR(io_T) __restrict orig_vec_out = vec_out;

        for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                    vec_in  = (MLI_PTR(io_T))orig_vec_in + POS(&in_prv,  pos0, pos1, pos2, 0);
                    vec_out = orig_vec_out + POS(&out_prv, pos0, pos1, pos2, 0);
                    mli::krn::compute_relu_inner_loop<io_T>(vec_in, vec_out, min_val, max_val, in_prv.shape[3]);
                }
            }
        }
    }
    return MLI_STATUS_OK;
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RELU_REF_H_
