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

#define POS mli_prv_get_tensor_idx_pos

/* To move inside tensor using memory strides (using 4 nested loops with counters pos0 pos1 pos2 pos4) */
template <typename io_T>
static MLI_FORCE_INLINE int mli_prv_get_tensor_idx_pos(
        const struct generic_tensor_private_t<MLI_PTR(io_T)> *in,
        int pos0, int pos1, int pos2, int pos3) {

    int res = pos0 * in->mem_stride[0] + pos1 * in->mem_stride[1] +
              pos2 * in->mem_stride[2] + pos3 * in->mem_stride[3];

    return res;
}

template <typename T>
static MLI_FORCE_INLINE tensor_private_t<T> mli_prv_get_tensor_chw(
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
            (T)in->data.mem.void_p, width, height, ch,
            col_mem_stride, row_mem_stride, ch_mem_stride };
}

template <typename T>
static MLI_FORCE_INLINE tensor_private_t<T> mli_prv_get_tensor_hwc(
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
            (T)in->data.mem.void_p, width, height, ch,
            col_mem_stride, row_mem_stride, ch_mem_stride };
}

template <typename T>
static MLI_FORCE_INLINE generic_tensor_private_t<T> mli_prv_get_generic_tensor(
        const mli_tensor *in) {
    generic_tensor_private_t<T> tensor;
    int rank = in->rank;

    tensor.ptr = (T)in->data.mem.void_p;
    tensor.rank = rank;

    if (rank) {
        for (int i = 0; i < rank; i++) {
                tensor.shape[i] = in->shape[i];
        }

        tensor.mem_stride[rank - 1] = in->mem_stride[rank - 1] != 0 ? in->mem_stride[rank - 1] : 1;
        for (int i = rank - 2; i >= 0; i--) {
            tensor.mem_stride[i] = in->mem_stride[i] != 0 ? in->mem_stride[i] : tensor.mem_stride[i+1] * in->shape[i+1];
        }

        for (int i = rank; i < MLI_MAX_RANK; i++) {
            tensor.shape[i] = 1;
            tensor.mem_stride[i] = 0;
        }
    }

    return tensor;
}

template <typename T>
static MLI_FORCE_INLINE generic_tensor_private_t<T> mli_prv_get_axis_tensor(
        generic_tensor_private_t<T> *in,
        const int axis) {
    generic_tensor_private_t<T> axis_prv_tensor;

    MLI_ASSERT(axis < in->rank);

    axis_prv_tensor = *in;

    /* Convert input tensor to a tensor that has the Axis parameter only in case of axis.
     * and the whole tensor in case of no axis.
     */
    if( axis > -1) {
        for (int i = 0; i < in->rank; i++) {
            if (i != axis) {
                axis_prv_tensor.shape[i] = 1;
            }
        }
    }

    return axis_prv_tensor;
}

template <typename T>
static MLI_FORCE_INLINE generic_tensor_private_t<T> mli_prv_get_non_axis_tensor(
        generic_tensor_private_t<T> *in,
        const int axis) {
    generic_tensor_private_t<T> non_axis_prv_tensor;

    MLI_ASSERT(axis < in->rank);

    for (int i = 0; i < MLI_MAX_RANK - 1; i++) {
        non_axis_prv_tensor.shape[i] = 1;
        non_axis_prv_tensor.rank = 0;
    }
    
    if (axis > -1) {
        /* Convert input tensor to a tensor that has the Non Axis parameters only in case of axis. */
        non_axis_prv_tensor.rank = in->rank - 1;
        for (int all_dim_idx = 0, not_axis_dim_idx = 0; all_dim_idx < MLI_MAX_RANK; all_dim_idx++) {
            if (all_dim_idx != axis) {
                non_axis_prv_tensor.shape[not_axis_dim_idx] = (all_dim_idx < (int)in->rank) ? 
                                                               in->shape[all_dim_idx] : 1;
                non_axis_prv_tensor.mem_stride[not_axis_dim_idx] = in->mem_stride[all_dim_idx];
                not_axis_dim_idx++;
            }
        }

    }
    return non_axis_prv_tensor;
}

/* To move the inner most dim with mem_stride = 1 to be at shape MLI_MAX_RANK - 1
 * so we can vectorize the inner most loop for kernels with mem_strides.
 */
template <typename T>
static MLI_FORCE_INLINE void mli_prv_reorder_generic_tensor(
        generic_tensor_private_t<T> *in_prv) {
    
    int i = MLI_MAX_RANK - 1;
    for (int j = in_prv->rank - 1 ; j >= 0; i--, j--) {
        in_prv->shape[i]  = in_prv->shape[j];
        in_prv->mem_stride[i] = in_prv->mem_stride[j];
    }

    for(; i >= 0; i--) {
        in_prv->shape[i]  = 1;
        in_prv->mem_stride[i] = 0;
    }
}

//This function squash tensor dimentions, where it's possible (where data is adjacent in memory)
//it's needed for better vectorizing of some kernels
//we also need to squash input and output tensors together because data in one of them can be not adjacent in memory
template <typename T>
static MLI_FORCE_INLINE void mli_prv_squash_generic_tensor(
        generic_tensor_private_t<T> *in_prv,
        generic_tensor_private_t<T> *out_prv) {
    int shift = 0;
    for (int i = in_prv->rank - 1; i > 0; i--){
        if (((in_prv->mem_stride[i - 1] == 0) || (in_prv->mem_stride[i - 1] == in_prv->shape[i])) &&
           ((out_prv->mem_stride[i - 1] == 0) || (out_prv->mem_stride[i - 1] == out_prv->shape[i]))){
            in_prv->mem_stride[i - 1] = 1;
            in_prv->mem_stride[i] = 1;
            in_prv->shape[i - 1] *= in_prv->shape[i];
            in_prv->shape[i] = 1;
            out_prv->mem_stride[i - 1] = 1;
            out_prv->mem_stride[i] = 1;
            out_prv->shape[i - 1] *= out_prv->shape[i];
            out_prv->shape[i] = 1;
            shift++;
        }
        else {
            break;
        }
    }
    int i = MLI_MAX_RANK - 1;
    for (int j = in_prv->rank - shift - 1 ; j >= 0; i--, j--) {
        in_prv->shape[i]  = in_prv->shape[j];
        in_prv->mem_stride[i] = in_prv->mem_stride[j];
        out_prv->shape[i]  = out_prv->shape[j];
        out_prv->mem_stride[i] = out_prv->mem_stride[j];
    }

    for(; i >= 0; i--) {
        in_prv->shape[i]  = 1;
        in_prv->mem_stride[i] = 0;
        out_prv->shape[i]  = 1;
        out_prv->mem_stride[i] = 0;
    }
}

template <typename T>
static MLI_FORCE_INLINE conv2d_weights_tensor_private_t<T> mli_prv_get_conv2d_weights_tensor_nhwc(
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
            (T)weights->data.mem.void_p, width, height, in_ch, out_ch,
            col_mem_stride, row_mem_stride, in_ch_mem_stride, out_ch_mem_stride };
}

template <typename T>
static MLI_FORCE_INLINE conv2d_weights_tensor_private_t<T> mli_prv_get_conv2d_weights_tensor_hwcn(
        const mli_tensor *weights,
        const int fix_in_ch = 0,
        const int fix_width = 0,
        const int fix_height = 0) {
    const int out_ch      = (int)weights->shape[KRNL_C_DIM_HWCN];
    int height            = (int)weights->shape[KRNL_H_DIM_HWCN];
    int width             = (int)weights->shape[KRNL_W_DIM_HWCN];
    int in_ch             = (int)weights->shape[KRNL_D_DIM_HWCN];
    int out_ch_mem_stride = weights->mem_stride[KRNL_C_DIM_HWCN];
    int row_mem_stride    = weights->mem_stride[KRNL_H_DIM_HWCN];
    int col_mem_stride    = weights->mem_stride[KRNL_W_DIM_HWCN];
    int in_ch_mem_stride  = weights->mem_stride[KRNL_D_DIM_HWCN];

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
        // user does not supply memory strides, hence we calculate them.
        MLI_ASSERT(out_ch_mem_stride == 0);
        MLI_ASSERT(row_mem_stride == 0);
        MLI_ASSERT(col_mem_stride == 0);
        MLI_ASSERT(out_ch_mem_stride == 0);
        out_ch_mem_stride = 1;
        in_ch_mem_stride  = out_ch;
        col_mem_stride    = out_ch * in_ch;
        row_mem_stride    = out_ch * in_ch * width;
    } else {
        // The inner-most memory stride should be 1.
        MLI_CHECK_AND_FIX(out_ch_mem_stride, 1);
    }

    return conv2d_weights_tensor_private_t<T> {
        (T)weights->data.mem.void_p, width, height, in_ch, out_ch,
        col_mem_stride, row_mem_stride, in_ch_mem_stride, out_ch_mem_stride };
}

template <typename T>
static MLI_FORCE_INLINE conv2d_weights_tensor_private_t<T> mli_prv_get_conv2d_weights_tensor_hw1n(
        const mli_tensor *weights,
        const int fix_width = 0,
        const int fix_height = 0) {
    int in_ch        = (int)weights->shape[KRNL_DW_D_DIM_HW1N];
    int height       = (int)weights->shape[KRNL_DW_H_DIM_HW1N];
    int width        = (int)weights->shape[KRNL_DW_W_DIM_HW1N];
    const int out_ch = (int)weights->shape[KRNL_DW_N_DIM_HW1N];
    int in_ch_mem_stride  = weights->mem_stride[KRNL_DW_D_DIM_HW1N];
    int row_mem_stride    = weights->mem_stride[KRNL_DW_H_DIM_HW1N];
    int col_mem_stride    = weights->mem_stride[KRNL_DW_W_DIM_HW1N];
    int out_ch_mem_stride = weights->mem_stride[KRNL_DW_N_DIM_HW1N];

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
            (T)weights->data.mem.void_p, width, height, in_ch, out_ch,
            col_mem_stride, row_mem_stride, in_ch_mem_stride, out_ch_mem_stride };
}

template <typename T>
static MLI_FORCE_INLINE conv2d_weights_tensor_private_t<T> mli_prv_get_conv2d_weights_tensor_nchw(
        const mli_tensor *weights) {
    const int out_ch      = (int)weights->shape[KRNL_C_DIM_CHW];
    const int in_ch       = (int)weights->shape[KRNL_D_DIM_CHW];
    const int height      = (int)weights->shape[KRNL_H_DIM_CHW];
    const int width       = (int)weights->shape[KRNL_W_DIM_CHW];
    int out_ch_mem_stride = weights->mem_stride[KRNL_C_DIM_CHW];
    int in_ch_mem_stride  = weights->mem_stride[KRNL_D_DIM_CHW];
    int row_mem_stride    = weights->mem_stride[KRNL_H_DIM_CHW];
    int col_mem_stride    = weights->mem_stride[KRNL_W_DIM_CHW];

    if (row_mem_stride == 0) {
        // user does not supply memory strides, hence we calculate them.
        MLI_ASSERT(out_ch_mem_stride == 0);
        MLI_ASSERT(in_ch_mem_stride == 0);
        MLI_ASSERT(col_mem_stride == 0);
        col_mem_stride    = 1;
        row_mem_stride    = width;
        in_ch_mem_stride  = width * height;
        out_ch_mem_stride = in_ch * width * height;
    }

    return conv2d_weights_tensor_private_t<T> {
        (T)weights->data.mem.void_p, width, height, in_ch, out_ch,
        col_mem_stride, row_mem_stride, in_ch_mem_stride, out_ch_mem_stride };
}

#ifdef __cplusplus
extern "C" {
#endif

static MLI_FORCE_INLINE mli_status mli_prv_copy_tensor_format(
        const mli_tensor * src, 
        mli_tensor * dst) {
    mli_status check = MLI_CHECK_STATUS(mli_chk_tensor (src), __func__);
    if (check != MLI_STATUS_OK)
          return check;

    for (int idx = 0; idx < (int)src->rank; idx++) {
        dst->shape[idx] = src->shape[idx];
        dst->mem_stride[idx] = src->mem_stride[idx];
    }

    dst->rank = src->rank;
    dst->el_type = src->el_type;
    dst->el_params = src->el_params;
    return MLI_STATUS_OK;
}

static MLI_FORCE_INLINE mli_status mli_prv_copy_tensor_format_except_mem_strides(
        const mli_tensor * src,
        mli_tensor * dst) {
    mli_status check = MLI_CHECK_STATUS(mli_chk_tensor (src), __func__);
    if (check != MLI_STATUS_OK)
          return check;

    for (int idx = 0; idx < (int)src->rank; idx++) {
        dst->shape[idx] = src->shape[idx];
    }

    dst->rank = src->rank;
    dst->el_type = src->el_type;
    dst->el_params = src->el_params;
    return MLI_STATUS_OK;
}

static MLI_FORCE_INLINE int mli_prv_calc_shift(
        const mli_tensor *in,
        const mli_tensor *w,
        const mli_tensor *out){
    if ((in->el_type == MLI_EL_FX_8) || (in->el_type == MLI_EL_FX_16)) {
        /* mix of FX and asym datatypes is not supported */
        MLI_ASSERT((w->el_type == MLI_EL_FX_8) || (w->el_type == MLI_EL_FX_16));
        MLI_ASSERT((out->el_type == MLI_EL_FX_8) || (out->el_type == MLI_EL_FX_16));
        return (in->el_params.fx.frac_bits + w->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;
    } else if (in->el_type == MLI_EL_SA_8) {
        /* mix of FX and asym datatypes is not supported */
        MLI_ASSERT(w->el_type == MLI_EL_SA_8);
        MLI_ASSERT((out->el_type == MLI_EL_SA_8) || (out->el_type == MLI_EL_SA_32));
        MLI_ASSERT(in->el_params.sa.dim < 0); // this function can only be used for per tensor quantization.
        MLI_ASSERT(w->el_params.sa.dim < 0); // this function can only be used for per tensor quantization.
        MLI_ASSERT(out->el_params.sa.dim < 0); // this function can only be used for per tensor quantization.
        return (in->el_params.sa.scale_frac_bits.mem.i8 + w->el_params.sa.scale_frac_bits.mem.i8) - out->el_params.sa.scale_frac_bits.mem.i8;
    } else {
        MLI_ASSERT(0);
        return 0;
    }
}

static MLI_FORCE_INLINE int mli_prv_calc_shift_idx(
        const mli_tensor *in,
        const mli_tensor *w,
        const mli_tensor *out,
        const int idx){
    if ((in->el_type == MLI_EL_FX_8) || (in->el_type == MLI_EL_FX_16)) {
        /* mix of FX and asym datatypes is not supported */
        MLI_ASSERT((w->el_type == MLI_EL_FX_8) || (w->el_type == MLI_EL_FX_16));
        MLI_ASSERT((out->el_type == MLI_EL_FX_8) || (out->el_type == MLI_EL_FX_16));
        return (in->el_params.fx.frac_bits + w->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;
    } else if (in->el_type == MLI_EL_SA_8) {
        /* mix of FX and asym datatypes is not supported */
        MLI_ASSERT(w->el_type == MLI_EL_SA_8);
        MLI_ASSERT((out->el_type == MLI_EL_SA_8) || (out->el_type == MLI_EL_SA_32));
        int in_shift = (in->el_params.sa.dim < 0) ? in->el_params.sa.scale_frac_bits.mem.i8 : in->el_params.sa.scale_frac_bits.mem.pi8[idx];
        int w_shift = (w->el_params.sa.dim < 0) ? w->el_params.sa.scale_frac_bits.mem.i8 : w->el_params.sa.scale_frac_bits.mem.pi8[idx];
        int out_shift = (out->el_params.sa.dim < 0) ? out->el_params.sa.scale_frac_bits.mem.i8 : out->el_params.sa.scale_frac_bits.mem.pi8[idx];
        return in_shift + w_shift - out_shift;
    } else {
        MLI_ASSERT(0);
        return 0;
    }
}

static MLI_FORCE_INLINE int32_t mli_prv_calc_bias_mul(
        const mli_tensor *in0,
        const mli_tensor *in1,
        const mli_tensor *bias){
    if ((in0->el_type == MLI_EL_FX_8) || (in0->el_type == MLI_EL_FX_16)) {
        /* mix of FX and asym datatypes is not supported */
        MLI_ASSERT((in1->el_type == MLI_EL_FX_8) || (in1->el_type == MLI_EL_FX_16));
        MLI_ASSERT((bias->el_type == MLI_EL_FX_8) || (bias->el_type == MLI_EL_FX_16));
        return 1;
    } else if (in0->el_type == MLI_EL_SA_8) {
        /* mix of FX and asym datatypes is not supported */
        MLI_ASSERT(in1->el_type == MLI_EL_SA_8);
        MLI_ASSERT((bias->el_type == MLI_EL_SA_8) || (bias->el_type == MLI_EL_SA_32));
        int32_t bias_mul = (1 << MLI_BIAS_MUL_SHIFT) / ((int32_t)in0->el_params.sa.scale.mem.i16 * (int32_t)in1->el_params.sa.scale.mem.i16);
        return bias_mul;
    } else {
        MLI_ASSERT(0);
        return 0;
    }
}

/* partial element counting. starting at startrank */
static MLI_FORCE_INLINE uint32_t mli_prv_count_elem_num_part(
        const mli_tensor *in,
        uint32_t startrank) {
    const uint32_t *shape = &in->shape[startrank];
    uint32_t rank = in->rank - startrank;
    uint32_t elem_num = 1;

    for (int idx = 0; idx < (int)rank; idx++)
        elem_num *= shape[idx];

    return elem_num;
}

/* full element counting */
static MLI_FORCE_INLINE uint32_t mli_prv_count_elem_num(const mli_tensor *in) {
    return mli_prv_count_elem_num_part(in, 0);
}

static MLI_FORCE_INLINE mli_minmax_t
mli_prv_get_relu_min_max (const mli_relu_cfg * cfg, const mli_tensor * out) {
    mli_minmax_t val_limit;
    int min_val, max_val;
    int zero, one, neg_one, six;
    if (out->el_type == MLI_EL_SA_8 || out->el_type == MLI_EL_SA_32) {
        MLI_ASSERT(out->el_params.sa.dim < 0);
        zero = out->el_params.sa.zero_point.mem.i16;
        // In theory it is possible that scale of input is really small value and shift might be bigger than 16 bit to 
        // represent six and one in such format before int div (may exceed 32 bits). 
        // One and six are not casted to 16bit directly, only after comparison with min_val and max_val and all of them are int.
        // Min val and max val always fit to container range, while six and one don't have to.
        // TODO: think about how to define whether it is required to extract six and one at all or not.
        six = ((int64_t)6l << mli_hlp_tensor_scale_shift(out, 0)) /  mli_hlp_tensor_scale(out, 0);
        one = ((int64_t)1l << mli_hlp_tensor_scale_shift(out, 0)) /  mli_hlp_tensor_scale(out, 0);
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
    case MLI_EL_SA_8:
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
        val_limit.min = (int16_t) MAX(zero, min_val);
        val_limit.max = (int16_t) max_val;
        break;
    case MLI_RELU_6:
        val_limit.min = (int16_t) MAX(zero, min_val);
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
