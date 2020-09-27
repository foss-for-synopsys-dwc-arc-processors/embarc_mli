/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_RELU_DSP_H_
#define _MLI_KRN_RELU_DSP_H_

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace dsp {

template <typename io_T>
static MLI_FORCE_INLINE mli_status mli_krn_relu_fx_run(const mli_tensor *in, 
        const mli_relu_cfg *cfg, mli_tensor *out) {

    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(io_T) vec_in = (MLI_PTR(io_T))(in->data.mem.void_p);
    MLI_PTR(io_T) vec_out = (MLI_PTR(io_T))(out->data.mem.void_p);

    /* Copy tensor format */
    mli_prv_copy_tensor_format_except_mem_strides(in, out);
    /* Get Generic Private Tensor */
    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);
    auto out_prv = mli_prv_get_generic_tensor<MLI_PTR(io_T)>(out);
    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&in_prv );
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&out_prv);

    const MLI_PTR(io_T) orig_vec_in = vec_in;
    MLI_PTR(io_T) orig_vec_out = vec_out;

    const mli_minmax_t limits = mli_prv_get_relu_min_max(cfg, in);
    const v2q15_t min_val = mli_prv_init_v(static_cast<io_T>(limits.min));
    const v2q15_t max_val = mli_prv_init_v(static_cast<io_T>(limits.max));

    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                vec_in  = (MLI_PTR(io_T))orig_vec_in + POS(&in_prv,  pos0, pos1, pos2, 0);
                vec_out = orig_vec_out + POS(&out_prv, pos0, pos1, pos2, 0);
                if (in_prv.shape[3] & 1) {
                    mli_prv_store_1_sample(vec_out, mli_math_min_fx(max_val, 
                        mli_math_max_fx(min_val ,mli_prv_load_1_sample(vec_in))));
                    vec_in  += 1;
                    vec_out += 1;
                }
                for (int pos3 = 0; pos3 < in_prv.shape[3] >> 1; pos3++) {
                    mli_prv_store_2_samples(vec_out, mli_math_min_fx(max_val,
                        mli_math_max_fx(min_val, mli_prv_load_2_samples(vec_in))));
                    vec_in  += 2;
                    vec_out += 2;
                }
            }
        }
    }

    return MLI_STATUS_OK;
}


} // namespace dsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_RELU_DSP_H_