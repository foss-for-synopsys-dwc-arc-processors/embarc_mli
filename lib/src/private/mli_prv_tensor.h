/*
* Copyright 2019-2020, Synopsys, Inc.
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

// with a shift of 31, we cannot represent the value one. So we shift only 30
// and an extra multiplication of 2 is done when bias is loaded.
#define MLI_BIAS_MUL_SHIFT 30

// Enable the below define to let the functions in this file
// assume that the user has set all the mli_tensor_t mem_strides to zero.
// Using this define can result in slight performance increase.
//#define MLI_PRV_TENSOR_CALC_MEM_STRIDES

#ifdef MLI_PRV_TENSOR_CALC_MEM_STRIDES
#define MLI_PRV_TENSOR_CALC_MEM_STRIDES_VAL true
#else
#define MLI_PRV_TENSOR_CALC_MEM_STRIDES_VAL false
#endif

// To prevent a compiler name mangling issue, type_is_xy should be true if and only if T has __xy.
template <typename T, bool type_is_xy> __attribute__((always_inline))
static inline tensor_private_t<T> mli_prv_get_tensor_chw(
        const mli_tensor *in,
        const int fix_ch = 0) {
    int ch             = (int)in->shape[FMAP_C_DIM_CHW];
    const int height   = (int)in->shape[FMAP_H_DIM_CHW];
    const int width    = (int)in->shape[FMAP_W_DIM_CHW];
    int ch_mem_stride  = in->mem_stride[FMAP_C_DIM_CHW];
    int row_mem_stride = in->mem_stride[FMAP_H_DIM_CHW];
    int col_mem_stride = in->mem_stride[FMAP_W_DIM_CHW];

    if (fix_ch != 0) {
        MLI_CHECK_AND_FIX(ch, fix_ch);
    }

    if (MLI_PRV_TENSOR_CALC_MEM_STRIDES_VAL || row_mem_stride == 0) {
        // User does not supply memory strides (all must be zero), so we calculate them here.
        MLI_ASSERT(ch_mem_stride == 0);
        MLI_ASSERT(row_mem_stride == 0);
        MLI_ASSERT(col_mem_stride == 0);

        ch_mem_stride  = width * height;
        row_mem_stride = width;
        col_mem_stride = 1;
    } else {
        // The inner-most memory stride should be 1.
        MLI_CHECK_AND_FIX(col_mem_stride, 1);
    }

    return tensor_private_t<T> {
            (T)in->data, width, height, ch,
            col_mem_stride, row_mem_stride, ch_mem_stride };
}

// To prevent a compiler name mangling issue, type_is_xy should be true if and only if T has __xy.
template <typename T, bool type_is_xy> __attribute__((always_inline))
static inline tensor_private_t<T> mli_prv_get_tensor_hwc(
        const mli_tensor *in,
        const int fix_ch = 0) {
    const int height   = (int)in->shape[FMAP_H_DIM_HWC];
    const int width    = (int)in->shape[FMAP_W_DIM_HWC];
    int ch             = (int)in->shape[FMAP_C_DIM_HWC];
    int row_mem_stride = in->mem_stride[FMAP_H_DIM_HWC];
    int col_mem_stride = in->mem_stride[FMAP_W_DIM_HWC];
    int ch_mem_stride  = in->mem_stride[FMAP_C_DIM_HWC];

    if (fix_ch != 0) {
        MLI_CHECK_AND_FIX(ch, fix_ch);
    }

    if (MLI_PRV_TENSOR_CALC_MEM_STRIDES_VAL || col_mem_stride == 0) {
        // User does not supply memory strides (all must be zero), so we calculate them here.
        MLI_ASSERT(row_mem_stride == 0);
        MLI_ASSERT(col_mem_stride == 0);
        MLI_ASSERT(ch_mem_stride == 0);

        row_mem_stride = ch * width;
        col_mem_stride = ch;
        ch_mem_stride  = 1;
    } else {
        // The inner-most memory stride should be 1.
        MLI_CHECK_AND_FIX(ch_mem_stride, 1);
    }

    return tensor_private_t<T> {
            (T)in->data, width, height, ch,
            col_mem_stride, row_mem_stride, ch_mem_stride };
}

// To prevent a compiler name mangling issue, type_is_xy should be true if and only if T has __xy.
template <typename T, bool type_is_xy> __attribute__((always_inline))
static inline conv2d_weights_tensor_private_t<T> mli_prv_get_conv2d_weights_tensor_nhwc(
        const mli_tensor *weights,
        const int fix_in_ch = 0,
        const int fix_width = 0,
        const int fix_height = 0) {
    const int out_ch = (int)weights->shape[KRNL_C_DIM_HWC];
    int height       = (int)weights->shape[KRNL_H_DIM_HWC];
    int width        = (int)weights->shape[KRNL_W_DIM_HWC];
    int in_ch        = (int)weights->shape[KRNL_D_DIM_HWC];
    int out_ch_mem_stride = weights->mem_stride[KRNL_C_DIM_HWC];
    int row_mem_stride    = weights->mem_stride[KRNL_H_DIM_HWC];
    int col_mem_stride    = weights->mem_stride[KRNL_W_DIM_HWC];
    int in_ch_mem_stride  = weights->mem_stride[KRNL_D_DIM_HWC];

    if (fix_width != 0) {
        MLI_CHECK_AND_FIX(width, fix_width);
    }
    if (fix_height != 0) {
        MLI_CHECK_AND_FIX(height, fix_height);
    }
    if (fix_in_ch != 0) {
        MLI_CHECK_AND_FIX(in_ch, fix_in_ch);
    }

    if (MLI_PRV_TENSOR_CALC_MEM_STRIDES_VAL || col_mem_stride == 0) {
        // User does not supply memory strides (all must be zero), so we calculate them here.
        MLI_ASSERT(out_ch_mem_stride == 0);
        MLI_ASSERT(row_mem_stride == 0);
        MLI_ASSERT(col_mem_stride == 0);
        MLI_ASSERT(in_ch_mem_stride == 0);

        out_ch_mem_stride = in_ch * width * height;
        row_mem_stride    = in_ch * width;
        col_mem_stride    = in_ch;
        in_ch_mem_stride  = 1;
    } else {
        // The inner-most memory stride should be 1.
        MLI_CHECK_AND_FIX(in_ch_mem_stride, 1);
    }

    return conv2d_weights_tensor_private_t<T> {
            (T)weights->data, width, height, in_ch, out_ch,
            col_mem_stride, row_mem_stride, in_ch_mem_stride, out_ch_mem_stride };
}

// To prevent a compiler name mangling issue, type_is_xy should be true if and only if T has __xy.
template <typename T, bool type_is_xy> __attribute__((always_inline))
static inline conv2d_weights_tensor_private_t<T> mli_prv_get_conv2d_weights_tensor_1hwn(
        const mli_tensor *weights,
        const int fix_width = 0,
        const int fix_height = 0) {
    int in_ch        = (int)weights->shape[KRNL_DW_D_DIM_HWC];
    int height       = (int)weights->shape[KRNL_DW_H_DIM_HWC];
    int width        = (int)weights->shape[KRNL_DW_W_DIM_HWC];
    const int out_ch = (int)weights->shape[KRNL_DW_C_DIM_HWC];
    int in_ch_mem_stride  = weights->mem_stride[KRNL_DW_D_DIM_HWC];
    int row_mem_stride    = weights->mem_stride[KRNL_DW_H_DIM_HWC];
    int col_mem_stride    = weights->mem_stride[KRNL_DW_W_DIM_HWC];
    int out_ch_mem_stride = weights->mem_stride[KRNL_DW_C_DIM_HWC];

    MLI_CHECK_AND_FIX(in_ch, 1);

    if (fix_width != 0) {
        MLI_CHECK_AND_FIX(width, fix_width);
    }
    if (fix_height != 0) {
        MLI_CHECK_AND_FIX(height, fix_height);
    }

    if (MLI_PRV_TENSOR_CALC_MEM_STRIDES_VAL || col_mem_stride == 0) {
        // User does not supply memory strides (all must be zero), so we calculate them here.
        MLI_ASSERT(in_ch_mem_stride == 0);
        MLI_ASSERT(row_mem_stride == 0);
        MLI_ASSERT(col_mem_stride == 0);
        MLI_ASSERT(out_ch_mem_stride == 0);

        in_ch_mem_stride  = out_ch * width * height;
        row_mem_stride    = out_ch * width;
        col_mem_stride    = out_ch;
        out_ch_mem_stride = 1;
    } else {
        // The inner-most memory stride should be 1.
        MLI_CHECK_AND_FIX(out_ch_mem_stride, 1);
    }

    return conv2d_weights_tensor_private_t<T> {
            (T)weights->data, width, height, in_ch, out_ch,
            col_mem_stride, row_mem_stride, in_ch_mem_stride, out_ch_mem_stride };
}

#ifdef __cplusplus
extern "C" {
#endif

static inline uint32_t __attribute__ ((always_inline)) mli_prv_norm(int32_t val) {
    return _norm(val);
}

static inline mli_status __attribute__ ((always_inline)) mli_prv_copy_tensor_format(
        const mli_tensor * src, 
        mli_tensor * dst) {
    mli_status check = MLI_CHECK_STATUS(mli_chk_tensor (src), __func__);
    if (check != MLI_STATUS_OK)
          return check;

    for (int idx = 0; idx < src->rank; idx++) {
        dst->shape[idx] = src->shape[idx];
        dst->mem_stride[idx] = src->mem_stride[idx];
    }

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
