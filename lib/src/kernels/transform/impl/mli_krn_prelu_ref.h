/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_PRELU_REF_H_
#define _MLI_KRN_PRELU_REF_H_

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
namespace ref {

template <typename io_T, prelu_elem_func_type func_type>
static MLI_FORCE_INLINE void mli_krn_scale_elem_v(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift) {
    io_T input = mli_prv_load_1vec(vec_in);
    io_T output;
    if (func_type == PRELU_ELEM_FUNC_MAX) {
        output = mli_math_max_fx(input, mli_math_acc_cast_fx<io_T, mli_acc32_t>(
                                      mli_math_mul_fx<io_T, mli_acc32_t>(input, scale), shift));
    } else {
        output = mli_math_min_fx(input,mli_math_acc_cast_fx<io_T, mli_acc32_t>(
                                      mli_math_mul_fx<io_T, mli_acc32_t>(input, scale), shift));
    }

    mli_prv_store_n_samples(vec_out, output);
}

template <typename io_T, prelu_elem_func_type func_type>
static MLI_FORCE_INLINE void mli_krn_scale_elem_v(
        const MLI_PTR(io_T) vec_in,
        MLI_OUT_PTR(io_T) vec_out,
        const io_T scale,
        const int shift,
        const int remaining_part) {

    MLI_ASSERT(remaining_part == 1);
    mli_krn_scale_elem_v<io_T, func_type>(vec_in, vec_out, scale, shift);
}

template <typename io_T>
static MLI_FORCE_INLINE mli_status mli_krn_prelu_fx_run(const mli_tensor *in, 
        const mli_tensor *slope_coeff,
        const mli_prelu_cfg *cfg, 
        mli_tensor *out) {
   
    mli_prv_fx_init_dsp_ctrl();

    const MLI_PTR(io_T) vec_in = nullptr;
    MLI_OUT_PTR(io_T) vec_out = nullptr;

    const MLI_PTR(io_T) in_ptr = (MLI_PTR(io_T))(in->data.mem.void_p);
    MLI_OUT_PTR(io_T) out_ptr = (MLI_OUT_PTR(io_T)) (out->data.mem.void_p);

    /* Copy tensor format */
    mli_prv_copy_tensor_format_except_mem_strides(in, out);
    /* Get Generic Private Tensor */
    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);
    auto out_prv = mli_prv_get_generic_tensor<MLI_OUT_PTR(io_T)>(out);
    /* Get Non Axis Tensor */
    auto in_non_axis_prv  = mli_prv_get_non_axis_tensor<MLI_PTR(io_T)>(&in_prv,  cfg->axis);
    auto out_non_axis_prv = mli_prv_get_non_axis_tensor<MLI_OUT_PTR(io_T)>(&out_prv, cfg->axis);
    /* Get Axis Tensor */
    in_prv  = mli_prv_get_axis_tensor<MLI_PTR(io_T)>(&in_prv,  cfg->axis);
    out_prv = mli_prv_get_axis_tensor<MLI_OUT_PTR(io_T)>(&out_prv, cfg->axis);
    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&in_prv );
    mli_prv_reorder_generic_tensor<MLI_OUT_PTR(io_T)>(&out_prv);

    /* Dummy Load to get num_lanes, remaining part */
    auto input = mli_prv_load_1vec(in_ptr);
    int num_lanes = get_number_lanes(input);
    int remaining_part = in_prv.shape[3] & (num_lanes - 1);

    int shift = mli_prv_calc_shift(in, slope_coeff, out);
    io_T scale;
    // Getscalar value (casting or getting from memory)
    if (slope_coeff->rank == 0) {
        // value is stored in tensor`s field: analog of reinterpret_cast
        scale = mli_math_cast_ptr_to_scalar_fx<io_T>(slope_coeff->data.mem.void_p);
    } else {
        // pointer access to value
        scale = static_cast<io_T *>(slope_coeff->data.mem.void_p)[0];
    }

    /* For applying the function to specific axis dimension, we should first loop across other dimensions then process
     * axis dimension elements.
     * For applying the function to the whole tensor, loop body is executed only one time. (i.e. shape[i] = 1).
     */
    if (mli_prv_less_than_1(scale, slope_coeff->el_params.fx.frac_bits)) {
        for (int dim0 = 0; dim0 < in_non_axis_prv.shape[0]; dim0++) {
            for (int dim1 = 0; dim1 < in_non_axis_prv.shape[1]; dim1++) {
                for (int dim2 = 0; dim2 < in_non_axis_prv.shape[2]; dim2++) {

                    vec_in = &in_ptr[dim0 * in_non_axis_prv.mem_stride[0] + 
                                    dim1 * in_non_axis_prv.mem_stride[1] + 
                                    dim2 * in_non_axis_prv.mem_stride[2]];
                    vec_out = &out_ptr[dim0 * out_non_axis_prv.mem_stride[0] + 
                                    dim1 * out_non_axis_prv.mem_stride[1] + 
                                    dim2 * out_non_axis_prv.mem_stride[2]];

                    const MLI_PTR(io_T) orig_vec_in = vec_in;
                    MLI_OUT_PTR(io_T) orig_vec_out = vec_out;
                    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
                        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                                vec_in  = (MLI_PTR(io_T))orig_vec_in  + POS(&in_prv,  pos0, pos1, pos2, 0);
                                vec_out = orig_vec_out + POS(&out_prv, pos0, pos1, pos2, 0);
                                if (remaining_part) {
                                    mli::krn::mli_krn_scale_elem_v<io_T, PRELU_ELEM_FUNC_MAX>(vec_in, vec_out, 
                                                                   scale, shift, remaining_part);
                                    vec_in  += remaining_part;
                                    vec_out += remaining_part;
                                }
                                for (int pos3 = remaining_part; pos3 < in_prv.shape[3]; pos3 += num_lanes) {
                                    mli::krn::mli_krn_scale_elem_v<io_T, PRELU_ELEM_FUNC_MAX>(vec_in, vec_out, 
                                                                   scale, shift);
                                    vec_in  += num_lanes;
                                    vec_out += num_lanes;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (int dim0 = 0; dim0 < in_non_axis_prv.shape[0]; dim0++) {
            for (int dim1 = 0; dim1 < in_non_axis_prv.shape[1]; dim1++) {
                for (int dim2 = 0; dim2 < in_non_axis_prv.shape[2]; dim2++) {

                    vec_in = &in_ptr[dim0 * in_non_axis_prv.mem_stride[0] + 
                                    dim1 * in_non_axis_prv.mem_stride[1] + 
                                    dim2 * in_non_axis_prv.mem_stride[2]];
                    vec_out = &out_ptr[dim0 * out_non_axis_prv.mem_stride[0] + 
                                    dim1 * out_non_axis_prv.mem_stride[1] + 
                                    dim2 * out_non_axis_prv.mem_stride[2]];

                    const MLI_PTR(io_T) orig_vec_in = vec_in;
                    MLI_OUT_PTR(io_T) orig_vec_out = vec_out;
                    for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
                        for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                            for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                                vec_in  = (MLI_PTR(io_T))orig_vec_in  + POS(&in_prv,  pos0, pos1, pos2, 0);
                                vec_out = orig_vec_out + POS(&out_prv, pos0, pos1, pos2, 0);
                                if (remaining_part) {
                                    mli::krn::mli_krn_scale_elem_v<io_T, PRELU_ELEM_FUNC_MIN>(vec_in, vec_out, 
                                                                   scale, shift, remaining_part);
                                    vec_in  += remaining_part;
                                    vec_out += remaining_part;
                                }
                                for (int pos3 = remaining_part; pos3 < in_prv.shape[3]; pos3 += num_lanes) {
                                    mli::krn::mli_krn_scale_elem_v<io_T, PRELU_ELEM_FUNC_MIN>(vec_in, vec_out, 
                                                                   scale, shift);
                                    vec_in  += num_lanes;
                                    vec_out += num_lanes;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return MLI_STATUS_OK;
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_PRELU_REF_H_