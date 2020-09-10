/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_SOFTMAX_REF_H_
#define _MLI_KRN_SOFTMAX_REF_H_

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_lut.h"
#include "mli_prv_activation_lut.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace ref {

const int kTanhAsymZeroPoint = 0;
const int kTanhOutputShift = 7;

#include "stdio.h"

template <typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_subtract_max(const io_T *vec_in, io_T *vec_out,
        struct generic_tensor_private_t<io_T*> *in_prv,
        struct generic_tensor_private_t<io_T*> *out_prv,
        int *in_frac_p) {
    // Looking for maximum value
    io_T max_val = vec_in[0];
    io_T min_val = vec_in[0];
    for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                for (int pos3 = 0; pos3 < in_prv->shape[3]; pos3++) {
                    max_val = mli_math_max_fx(max_val, vec_in[pos(in_prv, pos0, pos1, pos2, pos3)]);
                    min_val = mli_math_min_fx(min_val, vec_in[pos(in_prv, pos0, pos1, pos2, pos3)]);
                }
            }
        }
    }

    // Subtract maximum from each element
    // free one more bit if saturation is expected.
    const int biased_min = static_cast<int>(min_val) - max_val;
    const int min_limit = -(1 << ((sizeof(io_T) * 8) - 1));
    if (biased_min < min_limit) {
        max_val = max_val >> 1;
        for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                    for (int pos3 = 0; pos3 < in_prv->shape[3]; pos3++) {
                        vec_out[pos(out_prv, pos0, pos1, pos2, pos3)] =
                                (vec_in[pos(in_prv, pos0, pos1, pos2, pos3)] >> 1) - max_val;
                    }
                }
            }
        }
        *in_frac_p -= 1;
    } else {
        for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
            for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
                for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                    for (int pos3 = 0; pos3 < in_prv->shape[3]; pos3++) {
                        vec_out[pos(out_prv, pos0, pos1, pos2, pos3)] =
                                vec_in[pos(in_prv, pos0, pos1, pos2, pos3)] - max_val;
                    }
                }
            }
        }
    }
}

template <typename io_T>
static MLI_FORCE_INLINE void mli_krn_softmax_convert_tensor(const mli_tensor *in,
        const mli_softmax_cfg* cfg,
        mli_tensor *out,
        generic_tensor_private_t<io_T *> *in_prv,
        generic_tensor_private_t<io_T *> *out_prv,
        int *shape,
        int *in_mem_str,
        int *out_mem_str) {

    /* (1) Convert the input tensor to a private tensor that contains only the axis in case of per axis,
     * and the complete tensor in case of per tensor, this is used in inner 4 loops.
     * (2) Also convert this input tensor into a tensor that is basically the opposite of this
     * which we use in outer 3 loops.
     */

    /* conversion (1) is done here */
    *in_prv =  mli_prv_get_generic_tensor<io_T *>(in, cfg->axis);
    *out_prv = mli_prv_get_generic_tensor<io_T *>(out, cfg->axis);

    for (int i = 0; i < MLI_MAX_RANK - 1; i++) {
        shape[i] = 1;
    }

    /* conversion (2) is done here,
     * This loop is just added to eliminate axis shape, to iterate across the other dimensions(without axis dimension),
     * as the loop body will handle the axis dimension
     */
    if (cfg->axis >= 0) {
        for (int all_dim_idx = 0, not_axis_dim_idx = 0; all_dim_idx < MLI_MAX_RANK; all_dim_idx++) {
            if (all_dim_idx != cfg->axis) {
                shape[not_axis_dim_idx] = (all_dim_idx < in->rank) ? in->shape[all_dim_idx] : 1;
                in_mem_str[not_axis_dim_idx] = in_prv->mem_stride[all_dim_idx];
                out_mem_str[not_axis_dim_idx] = out_prv->mem_stride[all_dim_idx];
                not_axis_dim_idx++;
            }
        }
    }
}

template <typename io_T>
static MLI_FORCE_INLINE mli_status mli_krn_softmax_fx_run(const mli_tensor *in, const mli_softmax_cfg* cfg,
        mli_tensor *out) {

    MLI_ASSERT(MLI_MAX_RANK == 4);

    const io_T *vec_in = nullptr;
    io_T *vec_out = nullptr;

    const io_T *in_ptr = (io_T *)(in->data.mem.void_p);
    io_T *out_ptr = (io_T *) (out->data.mem.void_p);

    int shape[MLI_MAX_RANK - 1];
    int in_mem_str[MLI_MAX_RANK - 1];
    int out_mem_str[MLI_MAX_RANK - 1];

    struct generic_tensor_private_t<io_T *> in_prv;
    struct generic_tensor_private_t<io_T *> out_prv;

    /* Copy tensor format */
    mli_prv_copy_tensor_format_except_mem_strides(in, out);
    out->el_params.fx.frac_bits = (sizeof(vec_out[0]) * 8) - kTransfFuncIntBits - 1;

    /* (1) Convert the input tensor to a private tensor that contains only the axis in case of per axis,
     * and the complete tensor in case of per tensor, this is used in inner 4 loops.
     * (2) Also convert this input tensor into a tensor that is basically the opposite of this
     * which we use in outer 3 loops.
     */
    mli_krn_softmax_convert_tensor(in, cfg, out, &in_prv, &out_prv, shape, in_mem_str, out_mem_str);

    int in_frac = static_cast<int>(in->el_params.fx.frac_bits);

    /* For applying the function to specific axis dimension, we should first loop across other dimensions then process
     * axis dimension elements.
     * For applying the function to the whole tensor, loop body is executed only one time. (i.e. shape[i] = 1).
     */
    for (int dim0 = 0; dim0 < shape[0]; dim0++) {
        for (int dim1 = 0; dim1 < shape[1]; dim1++) {
            for (int dim2 = 0; dim2 < shape[2]; dim2++) {

                vec_in = &in_ptr[dim0 * in_mem_str[0] + dim1 * in_mem_str[1] + dim2 * in_mem_str[2]];
                vec_out = &out_ptr[dim0 * out_mem_str[0] + dim1 * out_mem_str[1] + dim2 * out_mem_str[2]];

                /* Subtract maximum from each element */
                mli_krn_softmax_subtract_max(vec_in, vec_out, &in_prv, &out_prv, &in_frac);

                /* Activation lookup table */
                struct generic_tensor_private_t<io_T *> out_vec_tensor = out_prv;
                out_vec_tensor.ptr = vec_out;
                mli::krn::activation_lut<io_T, false>(&out_vec_tensor, &out_vec_tensor, &expneg_lut_fx16, in_frac);

                // Accumulation through MAC and reciprocal calculation
                mli_acc40_t sum_acc = mli_math_mul_fx<io_T, mli_acc40_t>(0, 0);
                for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
                    for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                        for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                            for (int pos3 = 0; pos3 < in_prv.shape[3]; pos3++) {
                                sum_acc = mli_math_mac_fx(sum_acc,
                                        vec_out[pos(&out_prv, pos0, pos1, pos2, pos3)], static_cast<io_T>(1));
                            }
                        }
                    }
                }

                int sum_exp = mli_math_norm_fx<mli_acc40_t, int>(sum_acc) + 1;
                io_T sum_mnt = mli_math_acc_cast_fx<io_T, mli_acc40_t>(sum_acc, 16 - sum_exp);
                // sum_mnt is normalized (that is inside [0.5, 1) range)
                // so we use Q30(0.5) as a dividend to get Q15 result inside (0.5, 1)
                // saturation prevents it from reaching 1
                io_T sum_recip = (io_T)MIN((1L << 29) / sum_mnt, 32767L);

                // final result: normalizing
                for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
                    for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                        for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                            for (int pos3 = 0; pos3 < in_prv.shape[3]; pos3++) {
                                mli_acc40_t tmp_acc = mli_math_mul_fx<io_T, mli_acc40_t>(sum_recip,
                                        static_cast<io_T>(vec_out[pos(&out_prv, pos0, pos1, pos2, pos3)]));
                                vec_out[pos(&out_prv, pos0, pos1, pos2, pos3)] =
                                        mli_math_acc_cast_fx<io_T, mli_acc40_t>(tmp_acc,
                                                30 - sum_exp + 16 - sizeof(io_T) * 8);
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

#endif // _MLI_KRN_SOFTMAX_REF_H_
