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

#if defined(__FXAPI__)
/* FIXME: Remove this when known problem with dsp mli_math will be solved */
typedef int64_t acc_t;
#else
typedef mli_acc40_t acc_t;
#endif

const int kL2NormAsymZeroPoint = 0;
const int kL2NormOutputShift = 7;
const int kL2NormLutFracBits = 8;

namespace mli {
namespace krn {
namespace ref {

template<typename io_T, bool convert>
static MLI_FORCE_INLINE int16_t compute_normalized_sum_square(
        struct generic_tensor_private_t<MLI_PTR(io_T)> *in_prv,
        const MLI_PTR(io_T) vec_in,
        int16_t in_zp,
        int *norm_shift) {

    /* Accumulation through MAC */
    acc_t sum_acc = mli_math_mul_fx<int16_t, acc_t>(0, 0);
    for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                for (int pos3 = 0; pos3 < in_prv->shape[3]; pos3++) {
                    int16_t input = vec_in[POS(in_prv, pos0, pos1, pos2, pos3)];
                    if (convert) {
                        input = mli_math_sub_fx(input, in_zp);
                    }
                    sum_acc = mli_math_mac_fx(sum_acc, input, input);
                }
            }
        }
    }

    int norm_shift_val = mli_math_norm_fx<acc_t, int>(sum_acc);
    /* To Cast acc_t to int16_t */
    norm_shift_val = (sizeof(acc_t) - sizeof(int16_t)) * 8 - norm_shift_val;
    /* Adjust norm_shift to even number because we are going to divide it by 2 */
    if ((norm_shift_val & 0x1) == 0x1) {
        norm_shift_val += 1;
    }

    *norm_shift = norm_shift_val;
    /* Cast Sum_acc to Q7.8 to bring it to LUT input range */
    return mli_math_cast_fx<acc_t, int16_t>(sum_acc, norm_shift_val);
}

template<typename io_T, bool convert>
static MLI_FORCE_INLINE void normalize_tensor(
        struct generic_tensor_private_t<MLI_PTR(io_T)> *in_prv,
        struct generic_tensor_private_t<MLI_PTR(io_T)> *out_prv,
        const MLI_PTR(io_T) vec_in,
        MLI_PTR(io_T) vec_out,
        int16_t scale,
        int16_t in_zp,
        int shift) {

    // final result: normalizing
    for (int pos0 = 0; pos0 < in_prv->shape[0]; pos0++) {
        for (int pos1 = 0; pos1 < in_prv->shape[1]; pos1++) {
            for (int pos2 = 0; pos2 < in_prv->shape[2]; pos2++) {
                for (int pos3 = 0; pos3 < in_prv->shape[3]; pos3++) {
                    int16_t input = vec_in[POS(in_prv, pos0, pos1, pos2, pos3)];
                    if (convert) {
                        input = mli_math_sub_fx(input, in_zp);
                    }
                    mli_acc32_t tmp_acc = mli_math_mul_fx<int16_t, mli_acc32_t>(scale, input);
                    vec_out[POS(out_prv, pos0, pos1, pos2, pos3)] =
                            mli_math_acc_cast_fx<io_T, mli_acc32_t>(tmp_acc, shift);
                }
            }
        }
    }
}

template <typename io_T, bool convert>
static MLI_FORCE_INLINE mli_status mli_krn_l2_normalize_run(const mli_tensor *in, 
        const mli_tensor *epsilon, 
        const mli_l2_normalize_cfg *cfg, 
        mli_tensor *out) {
    
    /* Epsilon Tensor is Unused */
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

                int norm_shift;
                /* inv_sqrt = 1/sqrt(sum_acc)
                 * sum_acc can be approximated to sum_acc = (sum_acc_cast * 2^norm_shift)
                 * inv_sqrt = 1/sqrt(sum_acc * 2^-norm_shift * 2^norm_shift)
                 *          = 1/sqrt(sum_acc_cast * 2^norm_shift)
                 *          = 1/(2^(norm_shift/2) * sqrt(sum_acc_cast))
                 *          = 2^(-norm_shift/2) * (1/sqrt(sum_acc_cast))
                 */
                const int16_t sum_acc_cast = mli::krn::compute_normalized_sum_square<io_T, convert>(&in_prv, vec_in, in_zp, &norm_shift);
                /* Activation lookup table of input Q7.8 */
                int16_t out_lut = mli::krn::activation_lut_one_elem_interpolate<int16_t, int16_t, false, false>(
                    sum_acc_cast, &invsqrt_lut_fx16, kL2NormLutFracBits);

                /* (Norm_shift + kL2NormLutFracBits) is divided by 2 because of the square root */
                int shift = invsqrt_lut_fx16.out_frac_bits + ((norm_shift + kL2NormLutFracBits) >> 1) - out_shift;
                
                // final result: normalizing
                mli::krn::normalize_tensor<io_T, convert>(&in_prv, &out_prv, vec_in, vec_out, out_lut, in_zp, shift);

            }
        }
    }

    return MLI_STATUS_OK;
}

} // namespace ref
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_L2_NORMALIZE_REF_H_