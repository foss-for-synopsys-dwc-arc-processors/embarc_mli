/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_TENSOR_H_
#define _MLI_PRV_TENSOR_H_

#include <assert.h>

#include "mli_check.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math_macros.h"
#include "mli_types.h"
#include "mli_private_types.h"
#include <arc/arc_intrinsics.h>

#ifdef __cplusplus
extern "C" {
#endif

// with a shift of 31, we cannot represent the value one. So we shift only 30
// and an extra multiplication of 2 is done when bias is loaded.
#define MLI_BIAS_MUL_SHIFT 30

static inline uint32_t __attribute__ ((always_inline)) mli_prv_norm(int32_t val) {
    return _norm(val);
}

static inline mli_status __attribute__ ((always_inline)) mli_prv_copy_tensor_format (
        const mli_tensor * src, 
        mli_tensor * dst) {
    mli_status check = MLI_CHECK_STATUS(mli_chk_tensor (src), __func__);
    if (check != MLI_STATUS_OK)
          return check;

    for (int idx = 0; idx < src->rank; idx++)
        dst->shape[idx] = src->shape[idx];

    dst->rank = src->rank;
    dst->el_type = src->el_type;
    dst->el_params = src->el_params;
    return MLI_STATUS_OK;
}

static int inline __attribute__((always_inline)) mli_prv_calc_shift(
        const mli_tensor *in0,
        const mli_tensor *in1,
        const mli_tensor *out){
    if ((in0->el_type == MLI_EL_FX_8) || (in0->el_type == MLI_EL_FX_16)) {
        /* mix of FX and asym datatypes is not supported */
        MLI_ASSERT((in1->el_type == MLI_EL_FX_8) || (in1->el_type == MLI_EL_FX_16));
        MLI_ASSERT((out->el_type == MLI_EL_FX_8) || (out->el_type == MLI_EL_FX_16));
        return (in0->el_params.fx.frac_bits + in1->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;
    } else if (in0->el_type == MLI_EL_ASYM_I8) {
        /* mix of FX and asym datatypes is not supported */
        MLI_ASSERT(in1->el_type == MLI_EL_ASYM_I8);
        MLI_ASSERT((out->el_type == MLI_EL_ASYM_I8) || (out->el_type == MLI_EL_ASYM_I32));
        return (in0->el_params.asym.scale_frac_bits + in1->el_params.asym.scale_frac_bits) - out->el_params.asym.scale_frac_bits;
    } else {
        MLI_ASSERT(0);
        return 0;
    }
}

static int32_t inline __attribute__((always_inline)) mli_prv_calc_out_mul(
        const mli_tensor *in0,
        const mli_tensor *in1,
        const mli_tensor *out,
        int * shift){
    if ((in0->el_type == MLI_EL_FX_8) || (in0->el_type == MLI_EL_FX_16)) {
        /* mix of FX and asym datatypes is not supported */
        MLI_ASSERT((in1->el_type == MLI_EL_FX_8) || (in1->el_type == MLI_EL_FX_16));
        MLI_ASSERT((out->el_type == MLI_EL_FX_8) || (out->el_type == MLI_EL_FX_16));
        return 1;
    } else if (in0->el_type == MLI_EL_ASYM_I8) {
        /* mix of FX and asym datatypes is not supported */
        MLI_ASSERT(in1->el_type == MLI_EL_ASYM_I8);
        MLI_ASSERT((out->el_type == MLI_EL_ASYM_I8) || (out->el_type == MLI_EL_ASYM_I32));
        MLI_ASSERT((in0->el_params.asym.dim < 0) && (in1->el_params.asym.dim < 0));
        int32_t out_mul = (int32_t)in0->el_params.asym.scale.i16 * (int32_t)in1->el_params.asym.scale.i16;
        int norm = mli_prv_norm(out_mul);
        out_mul <<= norm;
        *shift += norm;
        out_mul = out_mul / (int32_t)out->el_params.asym.scale.i16;
        norm = mli_prv_norm(out_mul);
        out_mul <<= norm;
        *shift += norm;
        *shift -= MLI_MAT_MUL_Q31_SHIFT; // compensate for the fact that fractional mul is used (the mull does internal shift right with 31)
        return out_mul;
    } else {
        MLI_ASSERT(0);
        return 0;
    }
}

static int32_t inline __attribute__((always_inline)) mli_prv_calc_bias_mul(
        const mli_tensor *in0,
        const mli_tensor *in1,
        const mli_tensor *bias){
    if ((in0->el_type == MLI_EL_FX_8) || (in0->el_type == MLI_EL_FX_16)) {
        /* mix of FX and asym datatypes is not supported */
        MLI_ASSERT((in1->el_type == MLI_EL_FX_8) || (in1->el_type == MLI_EL_FX_16));
        MLI_ASSERT((bias->el_type == MLI_EL_FX_8) || (bias->el_type == MLI_EL_FX_16));
        return 1;
    } else if (in0->el_type == MLI_EL_ASYM_I8) {
        /* mix of FX and asym datatypes is not supported */
        MLI_ASSERT(in1->el_type == MLI_EL_ASYM_I8);
        MLI_ASSERT((bias->el_type == MLI_EL_ASYM_I8) || (bias->el_type == MLI_EL_ASYM_I32));
        int32_t bias_mul = (1 << MLI_BIAS_MUL_SHIFT) / ((int32_t)in0->el_params.asym.scale.i16 * (int32_t)in1->el_params.asym.scale.i16);
        return bias_mul;
    } else {
        MLI_ASSERT(0);
        return 0;
    }
}

/* partial element counting. starting at startrank */
static uint32_t inline __attribute__((always_inline)) mli_prv_count_elem_num_part(
        const mli_tensor *in,
        uint32_t startrank) {
    const uint32_t *shape = &in->shape[startrank];
    uint32_t rank = in->rank - startrank;
    uint32_t elem_num = 1;

    for (int idx = 0; idx < rank; idx++)
        elem_num *= shape[idx];

    return elem_num;
}

/* full element counting */
static uint32_t inline __attribute__((always_inline)) mli_prv_count_elem_num(const mli_tensor *in) {
    return mli_prv_count_elem_num_part(in, 0);
}

static inline mli_minmax_t __attribute__((always_inline))
mli_prv_get_relu_min_max (const mli_relu_cfg * cfg, const mli_tensor * out) {
    mli_minmax_t val_limit;
    int min_val, max_val;
    int zero, one, neg_one, six;
    if (out->el_type == MLI_EL_ASYM_I8 || out->el_type == MLI_EL_ASYM_I32) {
        MLI_ASSERT(out->el_params.asym.dim < 0);
        zero = out->el_params.asym.zero_point.i16;
        // In theory it is possible that scale of input is really small value and shift might be bigger than 16 bit to 
        // represent six and one in such format before int div (may exceed 32 bits). 
        // One and six are not casted to 16bit directly, only after comparison with min_val and max_val and all of them are int.
        // Min val and max val always fit to container range, while six and one don't have to.
        // TODO: think about how to define whether it is required to extract six and one at all or not.
        six = ((int64_t)6l << mli_hlp_tensor_scale_shift(out)) /  mli_hlp_tensor_scale(out, 0);
        one = ((int64_t)1l << mli_hlp_tensor_scale_shift(out)) /  mli_hlp_tensor_scale(out, 0);
        six = six + zero;
        neg_one = -one + zero;
        one = one + zero;
    } else {
        zero = 0;
        six = 6 << (int) out->el_params.fx.frac_bits;
        one = 1 << (int) out->el_params.fx.frac_bits;
        neg_one = -one;
    }

    switch (out->el_type) {
    case MLI_EL_FX_8:
    case MLI_EL_ASYM_I8:
        min_val = INT8_MIN;
        max_val = INT8_MAX;
        break;
    case MLI_EL_FX_16:
        min_val = INT16_MIN;
        max_val = INT16_MAX;
        break;
    default:
        MLI_ASSERT(0);             /* unsupported element type */
    }

    switch (cfg->type) {
    case MLI_RELU_GEN:
        val_limit.min = (int16_t) zero;
        val_limit.max = (int16_t) max_val;
        break;
    case MLI_RELU_6:
        val_limit.min = (int16_t) zero;
        val_limit.max = (int16_t) MIN (six, max_val);
        break;
    case MLI_RELU_1:
        val_limit.min = (int16_t) MAX(neg_one, min_val);
        val_limit.max = (int16_t) MIN(one, max_val);
        break;
    default:
        // For leaky and param relu there is no saturation in the function domain.
        // only container type limitations (8bit or 16 bit)
        val_limit.min = (int16_t) min_val;
        val_limit.max = (int16_t) max_val;
    }

    return val_limit;
}


#ifdef __cplusplus
}
#endif

#endif //_MLI_PRV_TENSOR_H_
