/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_TENSOR_H_
#define _MLI_PRV_TENSOR_H_

#include <assert.h>
#include <limits>

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

MLI_FORCE_INLINE void* mli_prv_tensor_cast_data_ptr(
        const mli_tensor *tensor) {
    void* ptr;
    MLI_ASSERT(tensor->rank > 0);
    switch (tensor->el_type) {
    case MLI_EL_FX_8:
    case MLI_EL_SA_8:
        ptr = static_cast<void*>(tensor->data.mem.pi8);
        break;
    case MLI_EL_FX_16:
        ptr = static_cast<void*>(tensor->data.mem.pi16);
        break;
    case MLI_EL_SA_32:
        ptr = static_cast<void*>(tensor->data.mem.pi32);
        break;
    case MLI_EL_FP_32:
        ptr = static_cast<void*>(tensor->data.mem.pf32);
        break;
    default:
        MLI_ASSERT(0);
        ptr = static_cast<void*>(tensor->data.mem.pi8);
        break;
    };
    return ptr;
}

template <typename T>
MLI_FORCE_INLINE T mli_prv_tensor_data_ptr(
        const mli_tensor *tensor);

template <>
MLI_FORCE_INLINE int8_t* mli_prv_tensor_data_ptr(
        const mli_tensor *tensor) {
    MLI_ASSERT((tensor->el_type == MLI_EL_FX_8) || (tensor->el_type == MLI_EL_SA_8));
    MLI_ASSERT(tensor->rank > 0);
    return tensor->data.mem.pi8;
}

template <>
MLI_FORCE_INLINE int16_t* mli_prv_tensor_data_ptr(
        const mli_tensor *tensor) {
    MLI_ASSERT(tensor->el_type == MLI_EL_FX_16);
    MLI_ASSERT(tensor->rank > 0);
    return tensor->data.mem.pi16;
}

template <>
MLI_FORCE_INLINE int32_t* mli_prv_tensor_data_ptr(
        const mli_tensor *tensor) {
    MLI_ASSERT(tensor->el_type == MLI_EL_SA_32);
    MLI_ASSERT(tensor->rank > 0);
    return tensor->data.mem.pi32;
}

template <>
MLI_FORCE_INLINE float* mli_prv_tensor_data_ptr(
        const mli_tensor *tensor) {
    MLI_ASSERT(tensor->el_type == MLI_EL_FP_32);
    MLI_ASSERT(tensor->rank > 0);
    return tensor->data.mem.pf32;
}
#if defined(__Xvec_width)
template <>
MLI_FORCE_INLINE __vccm int8_t* mli_prv_tensor_data_ptr(
        const mli_tensor *tensor) {
    MLI_ASSERT((tensor->el_type == MLI_EL_FX_8) || (tensor->el_type == MLI_EL_SA_8));
    MLI_ASSERT(tensor->rank > 0);
    return (__vccm int8_t*)tensor->data.mem.pi8;
}

template <>
MLI_FORCE_INLINE __vccm int16_t* mli_prv_tensor_data_ptr(
        const mli_tensor *tensor) {
    MLI_ASSERT(tensor->el_type == MLI_EL_FX_16);
    MLI_ASSERT(tensor->rank > 0);
    return (__vccm int16_t*)tensor->data.mem.pi16;
}

template <>
MLI_FORCE_INLINE __vccm int32_t* mli_prv_tensor_data_ptr(
        const mli_tensor *tensor) {
    MLI_ASSERT(tensor->el_type == MLI_EL_SA_32);
    MLI_ASSERT(tensor->rank > 0);
    return (__vccm int32_t*)tensor->data.mem.pi32;
}

template <>
MLI_FORCE_INLINE __vccm float* mli_prv_tensor_data_ptr(
        const mli_tensor *tensor) {
    MLI_ASSERT(tensor->el_type == MLI_EL_FP_32);
    MLI_ASSERT(tensor->rank > 0);
    return (__vccm float*)tensor->data.mem.pf32;
}
#endif

#ifdef __Xxy
template <>
MLI_FORCE_INLINE __xy int8_t* mli_prv_tensor_data_ptr(
        const mli_tensor *tensor) {
    MLI_ASSERT((tensor->el_type == MLI_EL_FX_8) || (tensor->el_type == MLI_EL_SA_8));
    MLI_ASSERT(tensor->rank > 0);
    return (__xy int8_t*)tensor->data.mem.pi8;
}

template <>
MLI_FORCE_INLINE __xy int16_t* mli_prv_tensor_data_ptr(
        const mli_tensor *tensor) {
    MLI_ASSERT(tensor->el_type == MLI_EL_FX_16);
    MLI_ASSERT(tensor->rank > 0);
    return (__xy int16_t*)tensor->data.mem.pi16;
}

template <>
MLI_FORCE_INLINE __xy int32_t* mli_prv_tensor_data_ptr(
        const mli_tensor *tensor) {
    MLI_ASSERT(tensor->el_type == MLI_EL_SA_32);
    MLI_ASSERT(tensor->rank > 0);
    return (__xy int32_t*)tensor->data.mem.pi32;
}

template <>
MLI_FORCE_INLINE __xy float* mli_prv_tensor_data_ptr(
        const mli_tensor *tensor) {
    MLI_ASSERT(tensor->el_type == MLI_EL_FP_32);
    MLI_ASSERT(tensor->rank > 0);
    return (__xy float*)tensor->data.mem.pf32;
}
#endif

template <typename T>
MLI_FORCE_INLINE T mli_prv_tensor_data_val(
        const mli_tensor *tensor);

template <>
MLI_FORCE_INLINE int8_t mli_prv_tensor_data_val(
        const mli_tensor *tensor) {
    MLI_ASSERT((tensor->el_type == MLI_EL_FX_8) || (tensor->el_type == MLI_EL_SA_8));
    if (tensor->rank == 0) {
        return tensor->data.mem.i8;
    } else {
        return tensor->data.mem.pi8[0];
    }
}

template <>
MLI_FORCE_INLINE int16_t mli_prv_tensor_data_val(
        const mli_tensor *tensor) {
    MLI_ASSERT(tensor->el_type == MLI_EL_FX_16);
    if (tensor->rank == 0) {
        return tensor->data.mem.i16;
    } else {
        return tensor->data.mem.pi16[0];
    }
}

template <>
MLI_FORCE_INLINE int32_t mli_prv_tensor_data_val(
        const mli_tensor *tensor) {
    MLI_ASSERT(tensor->el_type == MLI_EL_SA_32);
    if (tensor->rank == 0) {
        return tensor->data.mem.i32;
    } else {
        return tensor->data.mem.pi32[0];
    }
}

template <>
MLI_FORCE_INLINE float mli_prv_tensor_data_val(
        const mli_tensor *tensor) {
    MLI_ASSERT(tensor->el_type == MLI_EL_FP_32);
    if (tensor->rank == 0) {
        return tensor->data.mem.f32;
    } else {
        return tensor->data.mem.pf32[0];
    }
}

template <typename T>
MLI_FORCE_INLINE void mli_prv_tensor_inc_data_ptr
        (mli_tensor *in, int elements);


template <>
MLI_FORCE_INLINE void mli_prv_tensor_inc_data_ptr<int8_t*>
        (mli_tensor *tensor, int elements)
{
    int element_size = sizeof(int8_t);
    MLI_ASSERT((tensor->el_type == MLI_EL_FX_8) ||
               (tensor->el_type == MLI_EL_SA_8));
    MLI_ASSERT(element_size * elements <= (int)(tensor->data.capacity));
    tensor->data.mem.pi8 += elements;
    tensor->data.capacity -= elements * element_size;
}

template <>
MLI_FORCE_INLINE void mli_prv_tensor_inc_data_ptr<int16_t*>
        (mli_tensor *tensor, int elements)
{
    int element_size = sizeof(int16_t);
    MLI_ASSERT(tensor->el_type == MLI_EL_FX_16);
    MLI_ASSERT(element_size * elements <= (int)(tensor->data.capacity));
    tensor->data.mem.pi16 += elements;
    tensor->data.capacity -= elements * element_size;
}

template <>
MLI_FORCE_INLINE void mli_prv_tensor_inc_data_ptr<int32_t*>
        (mli_tensor *tensor, int elements)
{
    int element_size = sizeof(int32_t);
    MLI_ASSERT(tensor->el_type == MLI_EL_SA_32);
    MLI_ASSERT(element_size * elements <= (int)(tensor->data.capacity));
    tensor->data.mem.pi32 += elements;
    tensor->data.capacity -= elements * element_size;
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

    // The inner-most memory stride should be 1.
    MLI_CHECK_AND_FIX(col_mem_stride, 1);

    return tensor_private_t<T> {
            mli_prv_tensor_data_ptr<T>(in), width, height, ch,
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

    // The inner-most memory stride should be 1.
    MLI_CHECK_AND_FIX(ch_mem_stride, 1);

    return tensor_private_t<T> {
            mli_prv_tensor_data_ptr<T>(in), width, height, ch,
            col_mem_stride, row_mem_stride, ch_mem_stride };
}

template <typename T>
static MLI_FORCE_INLINE generic_tensor_private_t<T> mli_prv_get_generic_tensor(
        const mli_tensor *in) {
    generic_tensor_private_t<T> tensor;
    int rank = in->rank;

    tensor.ptr = mli_prv_tensor_data_ptr<T>(in);
    tensor.rank = rank;

    if (rank) {
        for (int i = 0; i < rank; i++) {
            tensor.shape[i] = in->shape[i];
            tensor.mem_stride[i] = in->mem_stride[i];
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

//This function squash tensor dimensions, where it's possible (where data is adjacent in memory)
//it's needed for better vectorizing of some kernels
//we also need to squash input and output tensors together because data in one of them can be not adjacent in memory

template <typename T>
static MLI_FORCE_INLINE void mli_prv_squash_generic_tensor(
        generic_tensor_private_t<T> *in_prv,
        generic_tensor_private_t<T> *out_prv) {
    int shift = 0;
    for (int i = in_prv->rank - 1; i > 0; i--){
        if ((in_prv->mem_stride[i - 1] == in_prv->shape[i]) &&
            (out_prv->mem_stride[i - 1] == out_prv->shape[i])){
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

static MLI_FORCE_INLINE int mli_prv_squash_tensor_to_one_dim(
        const mli_tensor *in,
        mli_tensor *out) {

    int rank = in->rank;
    int shape = in->shape[rank - 1];
    MLI_ASSERT(rank > 0);

    for (int i = rank - 1; i > 0; i--) {
         if ((in->mem_stride[i - 1] == shape) &&
             (out->mem_stride[i - 1] == shape)){
            shape *= in->shape[i - 1];
        } else {
             return 0;
        }
    }

    return shape;
}

static MLI_FORCE_INLINE int mli_prv_squash_tensor_to_one_dim(
        const mli_tensor *in1,
        const mli_tensor *in2,
        mli_tensor *out) {

    int rank = in1->rank;
    int shape = in1->shape[rank - 1];
    MLI_ASSERT(rank > 0);

    for (int i = rank - 1; i > 0; i--) {
         if ((in1->mem_stride[i - 1] == shape) &&
             (in2->mem_stride[i - 1] == shape) &&
             (out->mem_stride[i - 1] == shape)){

            shape *= in1->shape[i - 1];
        } else {
             return 0;
        }
    }

    return shape;
}

template <typename T>
static MLI_FORCE_INLINE void mli_prv_squash_generic_tensor(
        generic_tensor_private_t<T> *in1_prv,
        generic_tensor_private_t<T> *in2_prv,
        generic_tensor_private_t<T> *out_prv) {
    int shift = 0;
    for (int i = in1_prv->rank - 1; i > 0; i--){
        if ((in1_prv->mem_stride[i - 1] == in1_prv->shape[i]) &&
            (in2_prv->mem_stride[i - 1] == in2_prv->shape[i]) &&
            (out_prv->mem_stride[i - 1] == out_prv->shape[i])){
            in1_prv->mem_stride[i - 1] = 1;
            in1_prv->mem_stride[i] = 1;
            in1_prv->shape[i - 1] *= in1_prv->shape[i];
            in1_prv->shape[i] = 1;
            in2_prv->mem_stride[i - 1] = 1;
            in2_prv->mem_stride[i] = 1;
            in2_prv->shape[i - 1] *= in2_prv->shape[i];
            in2_prv->shape[i] = 1;
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
    for (int j = in1_prv->rank - shift - 1 ; j >= 0; i--, j--) {
        in1_prv->shape[i]  = in1_prv->shape[j];
        in1_prv->mem_stride[i] = in1_prv->mem_stride[j];
        in2_prv->shape[i]  = in2_prv->shape[j];
        in2_prv->mem_stride[i] = in2_prv->mem_stride[j];
        out_prv->shape[i]  = out_prv->shape[j];
        out_prv->mem_stride[i] = out_prv->mem_stride[j];
    }

    for(; i >= 0; i--) {
        in1_prv->shape[i]  = 1;
        in1_prv->mem_stride[i] = 0;
        in2_prv->shape[i]  = 1;
        in2_prv->mem_stride[i] = 0;
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

    // The inner-most memory stride should be 1.
    MLI_CHECK_AND_FIX(in_ch_mem_stride, 1);

    return conv2d_weights_tensor_private_t<T> {
            mli_prv_tensor_data_ptr<T>(weights), width, height, in_ch, out_ch,
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

    // The inner-most memory stride should be 1.
    MLI_CHECK_AND_FIX(out_ch_mem_stride, 1);

    return conv2d_weights_tensor_private_t<T> {
        mli_prv_tensor_data_ptr<T>(weights), width, height, in_ch, out_ch,
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

    // The inner-most memory stride should be 1.
    MLI_CHECK_AND_FIX(out_ch_mem_stride, 1);

    return conv2d_weights_tensor_private_t<T> {
            mli_prv_tensor_data_ptr<T>(weights), width, height, in_ch, out_ch,
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

    return conv2d_weights_tensor_private_t<T> {
        mli_prv_tensor_data_ptr<T>(weights), width, height, in_ch, out_ch,
        col_mem_stride, row_mem_stride, in_ch_mem_stride, out_ch_mem_stride };
}

/* rotate n quadrants counter clockwise */
template <int n, typename T>
static MLI_FORCE_INLINE tensor_private_t<T> mli_prv_rotate_tensor_private(
        const tensor_private_t<T> in) {
    auto out = in;
    if (n == 1) {
        out.ptr += in.col_mem_stride * (in.width - 1);
        out.width = in.height;
        out.height = in.width;
        out.row_mem_stride = -in.col_mem_stride;
        out.col_mem_stride = in.row_mem_stride;
    } else if (n == 2) {
        out.ptr += in.col_mem_stride * (in.width - 1);
        out.ptr += in.row_mem_stride * (in.height - 1);
        out.col_mem_stride = -in.col_mem_stride;
        out.row_mem_stride = -in.row_mem_stride;
    } else if (n == 3) {
        out.ptr += in.row_mem_stride * (in.height - 1);
        out.width = in.height;
        out.height = in.width;
        out.row_mem_stride = in.col_mem_stride;
        out.col_mem_stride = -in.row_mem_stride;
    }
    return out;
}

/* rotate n quadrants counter clockwise */
template <int n, typename T>
static MLI_FORCE_INLINE conv2d_weights_tensor_private_t<T> mli_prv_rotate_weights_tensor_private(
        const conv2d_weights_tensor_private_t<T> weights) {
    auto out = weights;
    if (n == 1) {
        out.ptr += weights.col_mem_stride * (weights.kernel_width - 1);
        out.kernel_width = weights.kernel_height;
        out.kernel_height = weights.kernel_width;
        out.row_mem_stride = -weights.col_mem_stride;
        out.col_mem_stride = weights.row_mem_stride;
    } else if (n == 2) {
        out.ptr += weights.col_mem_stride * (weights.kernel_width - 1);
        out.ptr += weights.row_mem_stride * (weights.kernel_height - 1);
        out.col_mem_stride = -weights.col_mem_stride;
        out.row_mem_stride = -weights.row_mem_stride;
    } else if (n == 3) {
        out.ptr += weights.row_mem_stride * (weights.kernel_height - 1);
        out.kernel_width = weights.kernel_height;
        out.kernel_height = weights.kernel_width;
        out.row_mem_stride = weights.col_mem_stride;
        out.col_mem_stride = -weights.row_mem_stride;
    }
    return out;
}


#ifdef __cplusplus
extern "C" {
#endif

static MLI_FORCE_INLINE mli_status mli_prv_copy_tensor_format(
        const mli_tensor * src, 
        mli_tensor * dst) {
    mli_status check = MLI_CHECK_STATUS(mli_chk_tensor (src, /*check_bank=*/false), __func__);
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
    mli_status check = MLI_CHECK_STATUS(mli_chk_tensor (src, /*check_bank=*/false), __func__);
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

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
    if (rank == 0) elem_num = 1;
    if (rank == 1) elem_num = shape[0];
    if (rank == 2) elem_num = shape[0]*shape[1];
    if (rank == 3) elem_num = shape[0]*shape[1]*shape[2];
    if (rank == 4) elem_num = shape[0]*shape[1]*shape[2]*shape[3];
#else
    for (int idx = 0; idx < (int)rank; idx++)
        elem_num *= shape[idx];
#endif

    return elem_num;
}

/* full element counting */
static MLI_FORCE_INLINE uint32_t mli_prv_count_elem_num(const mli_tensor *in) {
    return mli_prv_count_elem_num_part(in, 0);
}

#ifdef __cplusplus
}
#endif

template <typename out_T, bool asym>
static MLI_FORCE_INLINE mli_minmax_t
mli_prv_get_relu_limits (const mli_relu_cfg * cfg, const mli_tensor * out) {
    mli_minmax_t val_limit;
    int min_val, max_val;
    int zero, one, neg_one, six;
    int16_t scale;
    int shift;
    if (asym) {
        MLI_ASSERT((out->el_type == MLI_EL_SA_8 || out->el_type == MLI_EL_SA_32));
        // only per tensor quantization for output tensor supported.
        MLI_ASSERT(out->el_params.sa.dim < 0);
        zero = out->el_params.sa.zero_point.mem.i16;
        scale = out->el_params.sa.scale.mem.i16;
        shift = out->el_params.sa.scale_frac_bits.mem.i8;
    } else {
        zero = 0;
        scale = 1;
        shift = out->el_params.fx.frac_bits;
    }

    min_val = std::numeric_limits<out_T>::min();
    max_val = std::numeric_limits<out_T>::max();

    // In theory it is possible that scale of input is really small value and shift might be bigger than 16 bit to
    // represent six and one in such format before int div (may exceed 32 bits).
    // One and six are not casted to 16bit directly, only after comparison with min_val and max_val and all of them are int.
    // Min val and max val always fit to container range, while six and one don't have to.
    // when six doesn't fit in the container range, it will be clipped to the container range.

    switch (cfg->type) {
    case MLI_RELU_GEN:
        val_limit.min = (int16_t) MAX(zero, min_val);
        val_limit.max = (int16_t) max_val;
        break;
    case MLI_RELU_6:
        if (shift >= 0) {
            six = (shift < 28) ? ((int32_t)6 << shift) / scale : max_val;
        }
        else {
            six = (shift > -3) ?((int32_t)6 >> (-shift)) / scale : 0;
        }

        six = six + zero;
        val_limit.min = (int16_t) MAX(zero, min_val);
        val_limit.max = (int16_t) MIN (six, max_val);
        break;
    case MLI_RELU_1:
        if (shift >= 0) {
            one = (shift < 30) ? ((int32_t)1 << shift) / scale : max_val;
        }
        else {
            one = 0;
        }

        neg_one = -one + zero;
        one = one + zero;
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

#endif //_MLI_PRV_TENSOR_H_
