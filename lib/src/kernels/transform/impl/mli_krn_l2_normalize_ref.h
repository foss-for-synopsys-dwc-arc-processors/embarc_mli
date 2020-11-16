/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_L2_NORMALIZE_REF_H_
#define _MLI_KRN_L2_NORMALIZE_REF_H_

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"
#include "mli_prv_activation_lut.h"
#include "mli_prv_lut.h"

const int kL2NormAsymZeroPoint = 0;
const int kL2NormOutputShift = 7;

namespace mli {
namespace krn {
namespace ref {

template <typename io_T, bool convert>
static MLI_FORCE_INLINE mli_status mli_krn_l2_normalize_run(const mli_tensor *in, 
        const mli_tensor *epsilon, 
        const mli_l2_normalize_cfg *cfg, 
        mli_tensor *out) {

    mli_prv_fx_init_dsp_ctrl();
    
    MLI_ASSERT(MLI_MAX_RANK == 4);

    const MLI_PTR(io_T) vec_in = nullptr;
    MLI_PTR(io_T) vec_out = nullptr;

    const MLI_PTR(io_T) in_ptr = (MLI_PTR(io_T))(in->data.mem.void_p);
    MLI_PTR(io_T) out_ptr = (MLI_PTR(io_T)) (out->data.mem.void_p);

    /* Copy tensor format */
    mli_prv_copy_tensor_format_except_mem_strides(in, out);

    /* Get Generic Private Tensor */
    auto in_prv =  mli_prv_get_generic_tensor<MLI_PTR(io_T)>(in);
    auto out_prv = mli_prv_get_generic_tensor<MLI_PTR(io_T)>(out);
    /* Get Non Axis Tensor */
    auto in_non_axis_prv  = mli_prv_get_non_axis_tensor<MLI_PTR(io_T)>(&in_prv,  cfg->axis);
    auto out_non_axis_prv = mli_prv_get_non_axis_tensor<MLI_PTR(io_T)>(&out_prv, cfg->axis);
    /* Get Axis Tensor */
    in_prv  = mli_prv_get_axis_tensor<MLI_PTR(io_T)>(&in_prv,  cfg->axis);
    out_prv = mli_prv_get_axis_tensor<MLI_PTR(io_T)>(&out_prv, cfg->axis);
    /* Reordering shapes/mem_stirde to place the inner most dim at last shape */
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&in_prv );
    mli_prv_reorder_generic_tensor<MLI_PTR(io_T)>(&out_prv);

    int16_t in_zp = 0;
    int out_shift = 0;
    if (convert) {
        in_zp = in->el_params.sa.zero_point.mem.i16;
        out->el_params.sa.zero_point.mem.i16 = kL2NormAsymZeroPoint;
        out->el_params.sa.scale.mem.i16 = 1;
        out->el_params.sa.scale_frac_bits.mem.i8 = kL2NormOutputShift;
        out_shift = out->el_params.sa.scale_frac_bits.mem.i8;
    } else {
        out->el_params.fx.frac_bits = (sizeof(io_T) * 8) - kTransfFuncIntBits - 1;
        out_shift = out->el_params.fx.frac_bits;
    }

    /* For applying the function to specific axis dimension, we should first loop across other dimensions then process
     * axis dimension elements.
     * For applying the function to the whole tensor, loop body is executed only one time. (i.e. shape[i] = 1).
     */
    for (int dim0 = 0; dim0 < in_non_axis_prv.shape[0]; dim0++) {
        for (int dim1 = 0; dim1 < in_non_axis_prv.shape[1]; dim1++) {
            for (int dim2 = 0; dim2 < in_non_axis_prv.shape[2]; dim2++) {

                vec_in = &in_ptr[dim0 * in_non_axis_prv.mem_stride[0] + 
                                 dim1 * in_non_axis_prv.mem_stride[1] + 
                                 dim2 * in_non_axis_prv.mem_stride[2]];
                vec_out = &out_ptr[dim0 * out_non_axis_prv.mem_stride[0] + 
                                   dim1 * out_non_axis_prv.mem_stride[1] + 
                                   dim2 * out_non_axis_prv.mem_stride[2]];

                /* Accumulation through MAC */
                int64_t sum_acc = mli_math_mul_fx<int32_t, int64_t>(0, 0);
                for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
                    for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                        for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                            for (int pos3 = 0; pos3 < in_prv.shape[3]; pos3++) {
                                int16_t input = vec_in[POS(&out_prv, pos0, pos1, pos2, pos3)];
                                if (convert) {
                                    input -= in_zp;
                                }
                                sum_acc = mli_math_mac_fx(sum_acc, input, input);
                            }
                        }
                    }
                }
                /* inv_sqrt = 1/sqrt(sum_acc)
                 * sum_acc can be approximated to sum_acc = (sum_acc_cast_8b * 2^norm_shift)
                 * inv_sqrt = 1/sqrt(sum_acc * 2^-norm_shift * 2^norm_shift)
                 *          = 1/sqrt(sum_acc_cast_8b * 2^norm_shift)
                 *          = 1/(2^(norm_shift/2) * sqrt(sum_acc_cast_8b))
                 *          = 2^(-norm_shift/2) * (1/sqrt(sum_acc_cast_8b))
                 */
                int norm_shift = mli_math_norm_fx<int64_t, int>(sum_acc);
                /* To Cast int64_t to int8_t */
                norm_shift = (sizeof(int64_t) - sizeof(int8_t)) * 8 - norm_shift;
                /* Adjust norm_shift to even number because we are going to divide it by 2 */
                if ((norm_shift & 0x1) == 0x1) {
                    norm_shift +=1;
                }
                /* Cast Sum_acc to 8 bit to bring it to LUT input range */
                const int8_t sum_acc_cast_8b = mli_math_cast_fx<int64_t, int8_t>(sum_acc, norm_shift);
                /* Norm_shift is divided by 2 because of the square root */
                norm_shift >>= 1;
                /* Activation lookup table */
                int16_t out_lut = mli::krn::activation_lut_one_elem_interpolate<int8_t, int16_t, false, false>(
                    sum_acc_cast_8b, &invsqrt_lut_fx16, 0);

                // final result: normalizing
                for (int pos0 = 0; pos0 < in_prv.shape[0]; pos0++) {
                    for (int pos1 = 0; pos1 < in_prv.shape[1]; pos1++) {
                        for (int pos2 = 0; pos2 < in_prv.shape[2]; pos2++) {
                            for (int pos3 = 0; pos3 < in_prv.shape[3]; pos3++) {
                                int16_t input = vec_in[POS(&in_prv, pos0, pos1, pos2, pos3)];
                                if (convert) {
                                    input -= in_zp;
                                }
                                mli_acc32_t tmp_acc = mli_math_mul_fx<int16_t, mli_acc32_t>(out_lut, input);
                                vec_out[POS(&out_prv, pos0, pos1, pos2, pos3)] =
                                        mli_math_acc_cast_fx<io_T, mli_acc32_t>(tmp_acc,
                                            kLutOutFracBits + norm_shift - out_shift);
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

#endif // _MLI_KRN_L2_NORMALIZE_REF_H_