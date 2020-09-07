/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_check.h"

#include <stdio.h>
#include <assert.h>

#include "mli_types.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math_macros.h"
#include "mli_prv_tensor.h"

#pragma Code(".mli_lib")

static inline mli_status check_tensor_private(
        const uint32_t *shape,
        const uint32_t *mem_stride,
        uint32_t rank,
        uint32_t capacity,
        uint32_t element_size) {
    bool fail = false;

    fail |= MLI_CHECK(rank <= MLI_MAX_RANK, "Wrong tensor rank");
    for (int i = 0; i < rank; i++) {
        fail |= MLI_CHECK(mem_stride[i] >= 0, "Negative memory strides are not supported");
        fail |= MLI_CHECK(shape[i] > 0, "Shape invalid");
    }
    if (fail) return MLI_STATUS_BAD_TENSOR;

    bool strides_set = true;
    for (int i = 0; i < rank; i++) {
        strides_set &= (mem_stride[i] != 0);
    }
    uint32_t size = 1;
    if (strides_set) {
        uint32_t previous_shape = 1;
        int32_t previous_mem_stride = 1;
        for (int i = rank - 1; i >= 0; i--) {
            fail |= MLI_CHECK(mem_stride[i] >= (previous_shape * previous_mem_stride), "Tensor mem stride too small");
            previous_shape = shape[i];
            previous_mem_stride = mem_stride[i];
            size += (previous_shape - 1) * previous_mem_stride;
        }
    } else {
        for (int i = rank - 1; i >= 0; i--) {
            size *= shape[i];
        }
    }
    size *= element_size;

    if (fail) return MLI_STATUS_BAD_TENSOR;
    fail |= MLI_CHECK(capacity >= size, "Insufficient tensor capacity");
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;
    return MLI_STATUS_OK;
}

/******************************************************
 *  mli_tensor data structure correctness checking
 ******************************************************/
#ifdef __cplusplus
extern "C" {
#endif
mli_status mli_chk_tensor (const mli_tensor * in) {
    if (MLI_CHECK(in != NULL, "Bad tensor null pointer")) return MLI_STATUS_BAD_TENSOR;
    return check_tensor_private(in->shape, in->mem_stride, in->rank, in->data.capacity, mli_hlp_tensor_element_size(in));
}
#ifdef __cplusplus
}
#endif


mli_status mli_chk_scalar_tensor (const mli_tensor * in) {
    mli_status stat = MLI_STATUS_OK;
    // Scalar tensor is:
    // tensor of rank 0 and actual value in data field (shape and capacity isn't considered)
    //     or
    // typical tensor of any rank but with the only element.

    // .el_type - any value (any valid element type)
    // .el_params - any value (any valid for primitive)

    if (MLI_CHECK(in != NULL, "Bad tensor null pointer")) return MLI_STATUS_BAD_TENSOR;

    // for tensors with rank 0 no need for extra checks.
    if (in->rank == 0) return MLI_STATUS_OK;

    // for tensors with any rank, it has to be a valid tensor with 1 element.
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(mli_prv_count_elem_num(in) == 1, "Scalar tensor has to have only 1 element")) return MLI_STATUS_SHAPE_MISMATCH;

    return MLI_STATUS_OK;
}

bool mli_tensor_is_scalar (const mli_tensor * in) {
    if (in->rank == 0) return true;
    if (mli_prv_count_elem_num(in) == 1) return true;
    return false;
}

mli_status mli_chk_bias_frac_fx(const mli_tensor * in, const mli_tensor * weights, const mli_tensor * bias) {
    if (MLI_CHECK(bias->el_params.fx.frac_bits <= in->el_params.fx.frac_bits + weights->el_params.fx.frac_bits,
                      "The number of fractional bits of the accumulator will be the sum of the frac bits of in and weights. If bias has more frac bits, precision will be lost."))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    return MLI_STATUS_OK;
}

mli_status mli_chk_bias_scale_asym(const mli_tensor * in, const mli_tensor * weights, const mli_tensor * bias) {
    // For FX in runtime, before adding bias, we requantize it to intermediate result format. 
    // For FX it's just the matter of shift. That's why we are ok to not store bias in this format in advance (but it could be done).
    // For asym it is assumed that bias is stored in intermediate result format to avoid this requantization in runtime, 
    // which is not a matter of shifts anymore. Here we check bias format is similar to IR
    MLI_ASSERT(in->el_params.sa.dim < 0);
    MLI_ASSERT(in->el_type == MLI_EL_SA_8);
    MLI_ASSERT(weights->el_type == MLI_EL_SA_8);
    MLI_ASSERT(bias->el_type == MLI_EL_SA_32);
    const bool is_per_axis = weights->el_params.sa.dim >= 0;
    const int num_scale_vals = (is_per_axis)? weights->shape[weights->el_params.sa.dim]: 1;
    const int32_t* w_scales = (is_per_axis)? weights->el_params.sa.scale.mem.pi32: &weights->el_params.sa.scale.mem.i32;
    const int32_t* b_scales = (is_per_axis)? bias->el_params.sa.scale.mem.pi32: &bias->el_params.sa.scale.mem.i32;
    const int scale_in = (int)in->el_params.sa.scale.mem.i32;
    const int out_shift = mli_prv_calc_shift(in, weights, bias);
    for (int idx = 0; idx < num_scale_vals; idx++) {
        long long bias_scale_expected = scale_in * w_scales[idx];
        bias_scale_expected = (out_shift > 0)
                ? bias_scale_expected >> out_shift
                : bias_scale_expected << out_shift;
        const long long scales_diff = bias_scale_expected - b_scales[idx];
        // Check that diff is about the rounding error
        if (MLI_CHECK(scales_diff <= 1 && scales_diff >= -1, "Bias scale must be the multiplication of input and weights scales for correct calculations in current quanization scheme"))
            return MLI_STATUS_INCOMPATEBLE_TENSORS;
    }
    return MLI_STATUS_OK;
}

static inline bool check_inner_most_dimension_is_one(const mli_tensor *t) {
    return (t->mem_stride[t->rank - 1] == 1) || (t->mem_stride[t->rank - 1] == 0);
}

static inline bool check_layout_is_contiguous(const uint32_t *mem_stride, uint32_t rank) {
    // When only mem_stride and rank is under considiration, contiguous means 
    // all memory strides are zero OR rank is 1 and memory stride between elements is 1
    // If all memory strides are zero, the kernel itself will calculate the actual memory
    // strides such that all data is contiguous.
    bool strides_set = true;
    for (int i = 0; i < rank; i++) {
        strides_set &= (mem_stride[i] != 0);
    }
    
    if (!strides_set || (rank == 1 && mem_stride[0] == 1))
        return true;
    else
        return false;
}

static inline bool check_layout_is_contiguous(const uint32_t *shape, const uint32_t *mem_stride, uint32_t rank) {
    // This function either requires that all memory strides are zero,
    // or that the memory strides are set such that it results in the
    // same memory layout. If all memory strides are zero, the kernel itself
    // will calculate the actual memory strides such that all data is contiguous.
    bool strides_set = true;
    for (int i = 0; i < rank; i++) {
        strides_set &= (mem_stride[i] != 0);
    }
    if (!strides_set) return true;

    bool fail = false;
    uint32_t previous_shape = 1;
    int32_t previous_mem_stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
        fail |= MLI_CHECK(mem_stride[i] == (previous_shape * previous_mem_stride),
                "Tensor mem stride set incorrectly");
        previous_shape = shape[i];
        previous_mem_stride = mem_stride[i];
    }
    return !fail;
}

static inline bool check_layout_is_contiguous(const mli_tensor *t) {
    return check_layout_is_contiguous(t->shape, t->mem_stride, t->rank);
}

/******************************************************
 *  mli_krn_conv2d_hwc parameters checking function
 ******************************************************/
#ifdef __cplusplus
extern "C" {
#endif
mli_status mli_chk_conv2d_hwc (
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights), "Bad weights tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (bias), "Bad bias tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL, "Bad Output tensor pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL, "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(in->rank == 3, "Wrong input rank");
    fail |= MLI_CHECK(weights->rank == 4, "Wrong weights rank");
    fail |= MLI_CHECK(bias->rank == 1, "Wrong bias rank");
    fail |= MLI_CHECK(in->shape[FMAP_C_DIM_HWC] == weights->shape[KRNL_D_DIM_HWC], "Shape mismatch in and weights");
    fail |= MLI_CHECK(bias->shape[0] == weights->shape[KRNL_C_DIM_HWC], "Shape mismatch bias and weights");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in), "Memory stride for inner most dimension of input must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(weights), "Memory stride for inner most dimension of weights must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(bias), "Memory stride for inner most dimension of bias must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out), "Memory stride for inner most dimension of output must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    int kernel_width = weights->shape[KRNL_W_DIM_HWC];
    int kernel_height = weights->shape[KRNL_H_DIM_HWC];
    fail |= MLI_CHECK(cfg->padding_left < kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_right < kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_top < kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_bottom < kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->stride_height > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(cfg->stride_width > 0, "Stride should be greater than zero");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    int in_height = in->shape[FMAP_H_DIM_HWC];
    int in_width = in->shape[FMAP_W_DIM_HWC];
    uint32_t out_shape[3] = {
            (uint32_t)CEIL_DIV(in_height + cfg->padding_top + cfg->padding_bottom - kernel_height + 1,
                    cfg->stride_height), // h
            (uint32_t)CEIL_DIV(in_width + cfg->padding_left + cfg->padding_right - kernel_width + 1,
                    cfg->stride_width), // w
            weights->shape[KRNL_C_DIM_HWC]}; // c
    stat = check_tensor_private(out_shape, out->mem_stride, 3, out->data.capacity, mli_hlp_tensor_element_size(out));

    return stat;
}

mli_status mli_chk_conv2d_hwc_fx8(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwc(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_conv2d_hwc_fx16(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_16, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_16, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwc(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_conv2d_hwc_fx8w16d(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwc(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_conv2d_hwcn (
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights), "Bad weights tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (bias), "Bad bias tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL, "Bad Output tensor pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL, "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(in->rank == 3, "Wrong input rank");
    fail |= MLI_CHECK(weights->rank == 4, "Wrong weights rank");
    fail |= MLI_CHECK(bias->rank == 1, "Wrong bias rank");
    fail |= MLI_CHECK(in->shape[FMAP_C_DIM_HWC] == weights->shape[KRNL_D_DIM_HWCN], "Shape mismatch in and weights");
    fail |= MLI_CHECK(bias->shape[0] == weights->shape[KRNL_C_DIM_HWCN], "Shape mismatch bias and weights");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in), "Memory stride for inner most dimension of input must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(weights), "Memory stride for inner most dimension of weights must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(bias), "Memory stride for inner most dimension of bias must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out), "Memory stride for inner most dimension of output must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    int kernel_width = weights->shape[KRNL_W_DIM_HWCN];
    int kernel_height = weights->shape[KRNL_H_DIM_HWCN];
    fail |= MLI_CHECK(cfg->padding_left < kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_right < kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_top < kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_bottom < kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->stride_height > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(cfg->stride_width > 0, "Stride should be greater than zero");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    int in_height = in->shape[FMAP_H_DIM_HWC];
    int in_width = in->shape[FMAP_W_DIM_HWC];
    uint32_t out_shape[3] = {
            (uint32_t)CEIL_DIV(in_height + cfg->padding_top + cfg->padding_bottom - kernel_height + 1,
                    cfg->stride_height), // h
            (uint32_t)CEIL_DIV(in_width + cfg->padding_left + cfg->padding_right - kernel_width + 1,
                    cfg->stride_width), // w
            weights->shape[KRNL_C_DIM_HWCN]}; // c
    stat = check_tensor_private(out_shape, out->mem_stride, 3, out->data.capacity, mli_hlp_tensor_element_size(out));

    return stat;
}

mli_status mli_chk_conv2d_hwcn_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_SA_8, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_SA_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_SA_32, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_conv2d_hwcn_fx16(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_16, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_16, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_conv2d_hwcn_fx16_fx8_fx8(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_conv2d_nhwc_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_SA_8, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_SA_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_SA_32, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    if (MLI_CHECK(in->el_params.sa.zero_point.mem.i16 != INT16_MIN,"Input tensor: INT16_MIN doesn't support as offset value") ||
        MLI_CHECK(out->el_params.sa.zero_point.mem.i16 != INT16_MIN,"Input tensor: INT16_MIN doesn't support as offset value"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;

    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwc(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    if (MLI_CHECK(weights->el_params.sa.dim == KRNL_C_DIM_HWC, "Weights tensor: per output channels quantization is expected") ||
        MLI_CHECK(bias->el_params.sa.dim == 0, "Bias tensor: per output channels quantization is expected") || 
        MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    ret = MLI_CHECK_STATUS(mli_chk_bias_scale_asym(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    return MLI_STATUS_OK;
}

mli_status mli_chk_conv2d_chw (
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights), "Bad weights tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (bias), "Bad bias tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(in->rank == 3, "Wrong input rank");
    fail |= MLI_CHECK(weights->rank == 4, "Wrong weights rank");
    fail |= MLI_CHECK(bias->rank == 1, "Wrong bias rank");
    fail |= MLI_CHECK(in->shape[FMAP_C_DIM_CHW] == weights->shape[KRNL_D_DIM_CHW], "Shape mismatch in and weights");
    fail |= MLI_CHECK(bias->shape[0] == weights->shape[KRNL_C_DIM_CHW], "Shape mismatch bias and weights");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_layout_is_contiguous(in), "Memory Layout of input tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(weights), "Memory Layout of weights tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(out), "Memory Layout of output tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(bias), "Memory Layout of bias tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    fail |= MLI_CHECK(cfg->padding_left < kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_right < kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_top < kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_bottom < kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->stride_height > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(cfg->stride_width > 0, "Stride should be greater than zero");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];
    uint32_t out_shape[3] = {
            weights->shape[KRNL_C_DIM_CHW], // c
            (uint32_t)CEIL_DIV(in_height + cfg->padding_top + cfg->padding_bottom - kernel_height + 1,
                    cfg->stride_height), // h
            (uint32_t)CEIL_DIV(in_width + cfg->padding_left + cfg->padding_right - kernel_width + 1,
                    cfg->stride_width)}; // w

    fail |= MLI_CHECK(check_layout_is_contiguous(out_shape, out->mem_stride, 3),
                      "Memory Layout of output tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    stat = check_tensor_private(out_shape, out->mem_stride, 3, out->data.capacity, mli_hlp_tensor_element_size(out));

    return stat;
}

mli_status mli_chk_conv2d_chw_fx8(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_conv2d_chw(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_conv2d_chw_fx16(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_16, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_16, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_conv2d_chw(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_conv2d_chw_fx8w16d(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_conv2d_chw(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_depthwise_conv2d_chw (
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights), "Bad weights tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (bias), "Bad bias tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(in->rank == 3, "Wrong input rank");
    fail |= MLI_CHECK(weights->rank == 4, "Wrong weights rank");
    fail |= MLI_CHECK(bias->rank == 1, "Wrong bias rank");
    fail |= MLI_CHECK(weights->shape[KRNL_D_DIM_CHW] == 1, "Wrong weights shape");
    fail |= MLI_CHECK(in->shape[FMAP_C_DIM_CHW] == weights->shape[KRNL_C_DIM_CHW], "Shape mismatch in and weights");
    fail |= MLI_CHECK(bias->shape[0] == weights->shape[KRNL_C_DIM_CHW], "Shape mismatch bias and weights");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_layout_is_contiguous(in), "Memory Layout of input tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(weights), "Memory Layout of weights tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(bias), "Memory Layout of bias tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    int kernel_width = weights->shape[KRNL_W_DIM_CHW];
    int kernel_height = weights->shape[KRNL_H_DIM_CHW];
    fail |= MLI_CHECK(cfg->padding_left < kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_right < kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_top < kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_bottom < kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->stride_height > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(cfg->stride_width > 0, "Stride should be greater than zero");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];
    uint32_t out_shape[3] = {
            weights->shape[KRNL_C_DIM_CHW], // c
            (uint32_t)CEIL_DIV(in_height + cfg->padding_top + cfg->padding_bottom - kernel_height + 1,
                    cfg->stride_height), // h
            (uint32_t)CEIL_DIV(in_width + cfg->padding_left + cfg->padding_right - kernel_width + 1,
                    cfg->stride_width)}; // w

    fail |= MLI_CHECK(check_layout_is_contiguous(out_shape, out->mem_stride, 3), 
        "Memory Layout of out tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    stat = check_tensor_private(out_shape, out->mem_stride, 3, out->data.capacity, mli_hlp_tensor_element_size(out));

    return stat;
}

mli_status mli_chk_depthwise_conv2d_chw_fx8(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_depthwise_conv2d_chw_fx16(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_16, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_16, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_depthwise_conv2d_chw_fx8w16d(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_chw(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}


mli_status mli_chk_depthwise_conv2d_hwc(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights), "Bad weights tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (bias), "Bad bias tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(in->rank == 3, "Wrong input rank");
    fail |= MLI_CHECK(weights->rank == 4, "Wrong weights rank");
    fail |= MLI_CHECK(bias->rank == 1, "Wrong bias rank");
    fail |= MLI_CHECK(weights->shape[0] == 1, "Wrong weights shape");
    fail |= MLI_CHECK(bias->shape[0] == weights->shape[KRNL_DW_C_DIM_HWC], "Shape mismatch bias and weights");
    fail |= MLI_CHECK(weights->shape[KRNL_DW_C_DIM_HWC] == in->shape[FMAP_C_DIM_HWC], "Shape mismatch in and weights");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in), "Memory stride for inner most dimension of input must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(weights), "Memory stride for inner most dimension of weights must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(bias), "Memory stride for inner most dimension of bias must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out), "Memory stride for inner most dimension of output must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    int kernel_width = weights->shape[KRNL_DW_W_DIM_HWC];
    int kernel_height = weights->shape[KRNL_DW_H_DIM_HWC];
    fail |= MLI_CHECK(cfg->padding_left < kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_right < kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_top < kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_bottom < kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->stride_height > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(cfg->stride_width > 0, "Stride should be greater than zero");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    int in_height = in->shape[FMAP_H_DIM_HWC];
    int in_width = in->shape[FMAP_W_DIM_HWC];
    uint32_t out_shape[3] = {
            (uint32_t)CEIL_DIV(in_height + cfg->padding_top + cfg->padding_bottom - kernel_height + 1,
                    cfg->stride_height), // h
            (uint32_t)CEIL_DIV(in_width + cfg->padding_left + cfg->padding_right - kernel_width + 1,
                    cfg->stride_width), // w
            weights->shape[KRNL_DW_C_DIM_HWC]}; // c
    stat = check_tensor_private(out_shape, out->mem_stride, 3, out->data.capacity, mli_hlp_tensor_element_size(out));

    return stat;
}

mli_status mli_chk_depthwise_conv2d_hwc_fx8(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_hwc(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_depthwise_conv2d_hwc_fx16(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_16, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_16, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_hwc(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_depthwise_conv2d_hwc_fx8w16d(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_hwc(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}


mli_status mli_chk_depthwise_conv2d_hwcn_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_SA_8, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_SA_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_SA_32, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_hwc(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    if (MLI_CHECK(weights->el_params.sa.dim == KRNL_DW_C_DIM_HWC, "Weights tensor: per output channels quantization is expected") ||
        MLI_CHECK(bias->el_params.sa.dim == 0, "Bias tensor: per output channels quantization is expected") || 
        MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;

    if (MLI_CHECK(in->el_params.sa.zero_point.mem.i16 != INT16_MIN,"Input tensor: INT16_MIN doesn't support as offset value") ||
        MLI_CHECK(out->el_params.sa.zero_point.mem.i16 != INT16_MIN,"Input tensor: INT16_MIN doesn't support as offset value"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;

    ret = MLI_CHECK_STATUS(mli_chk_bias_scale_asym(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_maxpool_chw (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(in->rank == 3, "Wrong input rank");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in), "Memory stride for inner most dimension of input must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out), "Memory stride for inner most dimension of output must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    fail |= MLI_CHECK(cfg->padding_left < cfg->kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_right < cfg->kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_top < cfg->kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_bottom < cfg->kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->stride_height > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(cfg->stride_width > 0, "Stride should be greater than zero");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];
    uint32_t out_shape[3] = {
            in->shape[FMAP_C_DIM_CHW], // c
            (uint32_t)CEIL_DIV(in_height + cfg->padding_top + cfg->padding_bottom - cfg->kernel_height + 1,
                    cfg->stride_height), // h
            (uint32_t)CEIL_DIV(in_width + cfg->padding_left + cfg->padding_right - cfg->kernel_width + 1,
                    cfg->stride_width)}; // w

    stat = check_tensor_private(out_shape, out->mem_stride, 3, out->data.capacity, mli_hlp_tensor_element_size(out));

    return stat;
}

mli_status mli_chk_maxpool_chw_fx8 (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_maxpool_chw(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_maxpool_chw_fx16 (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_maxpool_chw(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}


mli_status mli_chk_maxpool_hwc (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(in->rank == 3, "Wrong input rank");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in), "Memory stride for inner most dimension of input must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out), "Memory stride for inner most dimension of output must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    fail |= MLI_CHECK(cfg->padding_left < cfg->kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_right < cfg->kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_top < cfg->kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_bottom < cfg->kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->stride_height > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(cfg->stride_width > 0, "Stride should be greater than zero");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    int in_height = in->shape[FMAP_H_DIM_HWC];
    int in_width = in->shape[FMAP_W_DIM_HWC];
    uint32_t out_shape[3] = {
            (uint32_t)CEIL_DIV(in_height + cfg->padding_top + cfg->padding_bottom - cfg->kernel_height + 1,
                    cfg->stride_height), // h
            (uint32_t)CEIL_DIV(in_width + cfg->padding_left + cfg->padding_right - cfg->kernel_width + 1,
                    cfg->stride_width), // w
            in->shape[FMAP_C_DIM_HWC]}; // c
    stat = check_tensor_private(out_shape, out->mem_stride, 3, out->data.capacity, mli_hlp_tensor_element_size(out));

    return stat;
}

mli_status mli_chk_maxpool_hwc_fx8 (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_maxpool_hwc(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_maxpool_hwc_fx16 (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_maxpool_hwc(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_maxpool_hwc_sa8 (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_maxpool_hwc(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_SA_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;    
    if (MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;

    return MLI_STATUS_OK;
}

mli_status mli_chk_avepool_chw (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(in->rank == 3, "Wrong input rank");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in), "Memory stride for inner most dimension of input must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out), "Memory stride for inner most dimension of output must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    fail |= MLI_CHECK(cfg->padding_left < cfg->kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_right < cfg->kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_top < cfg->kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_bottom < cfg->kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->stride_height > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(cfg->stride_width > 0, "Stride should be greater than zero");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    int in_height = in->shape[FMAP_H_DIM_CHW];
    int in_width = in->shape[FMAP_W_DIM_CHW];
    uint32_t out_shape[3] = {
            in->shape[FMAP_C_DIM_CHW], // c
            (uint32_t)CEIL_DIV(in_height + cfg->padding_top + cfg->padding_bottom - cfg->kernel_height + 1,
                    cfg->stride_height), // h
            (uint32_t)CEIL_DIV(in_width + cfg->padding_left + cfg->padding_right - cfg->kernel_width + 1,
                    cfg->stride_width)}; // w
    stat = check_tensor_private(out_shape, out->mem_stride, 3, out->data.capacity, mli_hlp_tensor_element_size(out));

    return stat;
}

mli_status mli_chk_avepool_chw_fx8 (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_avepool_chw_fx16 (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_chw(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_avepool_hwc (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(in->rank == 3, "Wrong input rank");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in), "Memory stride for inner most dimension of input must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out), "Memory stride for inner most dimension of output must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    fail |= MLI_CHECK(cfg->padding_left < cfg->kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_right < cfg->kernel_width, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_top < cfg->kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->padding_bottom < cfg->kernel_height, "Padding should be smaller than kernelsize");
    fail |= MLI_CHECK(cfg->stride_height > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(cfg->stride_width > 0, "Stride should be greater than zero");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    int in_height = in->shape[FMAP_H_DIM_HWC];
    int in_width = in->shape[FMAP_W_DIM_HWC];
    uint32_t out_shape[3] = {
            (uint32_t)CEIL_DIV(in_height + cfg->padding_top + cfg->padding_bottom - cfg->kernel_height + 1,
                    cfg->stride_height), // h
            (uint32_t)CEIL_DIV(in_width + cfg->padding_left + cfg->padding_right - cfg->kernel_width + 1,
                    cfg->stride_width), // w
            in->shape[FMAP_C_DIM_HWC]}; // c
    stat = check_tensor_private(out_shape, out->mem_stride, 3, out->data.capacity, mli_hlp_tensor_element_size(out));

    return MLI_STATUS_OK;
}

mli_status mli_chk_avepool_hwc_fx8 (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_hwc(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_avepool_hwc_fx16 (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_hwc(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_avepool_hwc_sa8 (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_avepool_hwc(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_SA_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    if (MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    return MLI_STATUS_OK;
}

mli_status mli_chk_fully_connected (
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights), "Bad weights tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (bias), "Bad bias tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(weights->rank == 2, "Wrong weights rank");
    fail |= MLI_CHECK(bias->rank == 1, "Wrong bias rank");
    fail |= MLI_CHECK(mli_prv_count_elem_num (in) == weights->shape[1], "weights shape doesn't match number of input elements");
    fail |= MLI_CHECK(bias->shape[0] == weights->shape[0], "Shape mismatch bias and weights");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_layout_is_contiguous(in), "Memory Layout of input tensor must be contiguous");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(weights), "Memory stride for inner most dimension of weights must be 1");
    fail |= MLI_CHECK(check_layout_is_contiguous(bias), "Memory Layout of bias tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(out->mem_stride, 1), "Memory Layout of output tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    fail |= MLI_CHECK((weights->shape[0] * mli_hlp_tensor_element_size (in)) <= out->data.capacity, "capacity of output tensor is too small");
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;

    return MLI_STATUS_OK;
}

mli_status mli_chk_fully_connected_fx8w16d(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_fully_connected(in, weights, bias, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_fully_connected_fx8(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_fully_connected(in, weights, bias, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_fully_connected_fx16(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_16, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_16, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_fully_connected(in, weights, bias, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_fully_connected_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_SA_8, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_SA_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_SA_32, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_fully_connected(in, weights, bias, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    if (MLI_CHECK(weights->el_params.sa.dim < 0, "Weights tensor: Per-tensor quantization is expected") ||
        MLI_CHECK(bias->el_params.sa.dim < 0, "Bias tensor: Per-tensor quantization is expected") || 
        MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    ret = MLI_CHECK_STATUS(mli_chk_bias_scale_asym(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_relu(const mli_tensor * in, const mli_relu_cfg * cfg, mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    // Check that tensors are valid
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;
    fail |= MLI_CHECK(check_layout_is_contiguous(in), "Memory Layout of input tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(in->shape, out->mem_stride, in->rank),
                      "Memory Layout of output tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check that output contains enough space
    if (MLI_CHECK((mli_prv_count_elem_num(in) * mli_hlp_tensor_element_size(in)) <= out->data.capacity, "capacity of output tensor is too small"))
        return MLI_STATUS_NOT_ENGH_MEM;

    return MLI_STATUS_OK;
}

mli_status mli_chk_relu_fx8(const mli_tensor * in, const mli_relu_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_relu(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_relu_fx16(const mli_tensor * in, const mli_relu_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_relu(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise (
        const mli_tensor * in1,
        const mli_tensor * in2,
        mli_tensor * out,
        bool check_same_fx_notation,
        const char *funcname) {
    bool fail = false;
    mli_status stat = MLI_STATUS_OK;

    // Check that tensors are valid
    // One of tensors may be scalar - check through it at first
    if (mli_tensor_is_scalar(in1)){
        fail |= MLI_CHECK2(mli_chk_scalar_tensor(in1) == MLI_STATUS_OK, "bad in1 tensor", funcname);
    }
    if (mli_tensor_is_scalar(in2)){
        fail |= MLI_CHECK2(mli_chk_scalar_tensor(in2) == MLI_STATUS_OK, "bad in2 tensor", funcname);
    }
    fail |= MLI_CHECK2(out != NULL , "Bad Output tensor  pointer", funcname);
    fail |= MLI_CHECK2(out->data.mem.void_p != NULL , "Bad data pointer of output", funcname);
    if (fail) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(check_layout_is_contiguous(in1), "Memory Layout of in1 tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(in2), "Memory Layout of in2 tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // One of the tensors must be typical (even with single element)
    fail |= MLI_CHECK2((in1->rank > 0) || (in2->rank > 0),
                       "One of the tensors must be typical (even with single element)", funcname);
    if (fail) return MLI_STATUS_NOT_SUPPORTED;
    if (!(mli_tensor_is_scalar(in1))) {
        stat = MLI_CHECK_STATUS(mli_chk_tensor(in1), "Bad input1 tensor");
        fail |= MLI_CHECK(check_layout_is_contiguous(in1->shape, out->mem_stride, in1->rank), 
                          "Memory Layout of output tensor must be contiguous");
    }
    if (!(mli_tensor_is_scalar(in2))) {
        stat = MLI_CHECK_STATUS(mli_chk_tensor(in2), "Bad input2 tensor");
        fail |= MLI_CHECK(check_layout_is_contiguous(in2->shape, out->mem_stride, in2->rank),
                          "Memory Layout of output tensor must be contiguous");
    }
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;
    if (stat != MLI_STATUS_OK) return stat;

    // Element wise may be performed only for tensors of the same element type
    fail |= MLI_CHECK2(in1->el_type == in2->el_type,
            "ADD may be performed only for tensors of the same FX notation", funcname);
    // sub, add, min and max may be performed only for tensors of the same FX notation
    if (check_same_fx_notation) {
        fail |= MLI_CHECK2(in1->el_params.fx.frac_bits == in2->el_params.fx.frac_bits,
                "this eltwise function may be performed only for tensors of the same FX notation", funcname);
    }
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // If both tensors are not scalar their shapes must be exactly the same.
    if (!mli_tensor_is_scalar(in1) && !mli_tensor_is_scalar(in2)) {
        fail |= MLI_CHECK2(in1->rank == in2->rank,
                "If both tensors are not scalar their shapes must be exactly the same.", funcname);
        for (int idx = 0; idx < in1->rank; idx++) {
            fail |= MLI_CHECK2(in1->shape[idx] == in2->shape[idx],
                    "If both tensors are not scalar their shapes must be exactly the same.", funcname);
        }
        if (fail) return MLI_STATUS_SHAPE_MISMATCH;
    }

    // Check that output contains enough space
    int in1_sz = mli_prv_count_elem_num(in1);
    int in2_sz = mli_prv_count_elem_num(in2);
    fail |= MLI_CHECK2((MAX (in1_sz, in2_sz) * mli_hlp_tensor_element_size (in1)) <= out->data.capacity,
            "capacity of output tensor is too small", funcname);
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;

    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise_add_fx8 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, true, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in1->el_type == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise_add_fx16 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, true, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in1->el_type == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise_sub_fx8 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, true, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in1->el_type == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise_sub_fx16 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, true, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in1->el_type == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise_min_fx8 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, true, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in1->el_type == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise_min_fx16 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, true, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in1->el_type == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise_max_fx8 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, true, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in1->el_type == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise_max_fx16 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, true, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in1->el_type == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise_mul_fx8 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    // elementwisw multiply does accept different FX points in its inputs
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, false, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in1->el_type == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise_mul_fx16 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    // elementwisw multiply does accept different FX points in its inputs
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, false, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in1->el_type == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_basic_activation(const mli_tensor * in, mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    // Check that tensors are valid
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;
    fail |= MLI_CHECK(check_layout_is_contiguous(in), "Memory Layout of input tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(in->shape, out->mem_stride, in->rank),  // output has shape of input
                      "Memory Layout of output tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check that output contains enough space
    fail |= MLI_CHECK((mli_prv_count_elem_num (in) * mli_hlp_tensor_element_size (in)) <= out->data.capacity,
                      "capacity of output tensor is too small");
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;

    return MLI_STATUS_OK;
}

mli_status mli_chk_basic_activation_fx8(const mli_tensor * in, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation(in, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_basic_activation_fx16(const mli_tensor * in, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation(in, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_basic_activation_sa8(const mli_tensor * in, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation(in, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_SA_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_leaky_relu (const mli_tensor * in, const mli_tensor * slope_coeff, mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    // Check that tensors are valid
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;
    fail |= MLI_CHECK(check_layout_is_contiguous(in), "Memory Layout of input tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(in->shape, out->mem_stride, in->rank),  // output has shape of input
                      "Memory Layout of output tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check that slope tensors is valid scalar
    stat = MLI_CHECK_STATUS(mli_chk_scalar_tensor (slope_coeff), "Slope should be scalar tensor");
    if (stat != MLI_STATUS_OK) return stat;

    // Slope must be scalar tensor of the same el_type as input
    fail |= MLI_CHECK(slope_coeff->el_type == in->el_type, "element type has to be the same");
    if (fail) return MLI_STATUS_TYPE_MISMATCH;

    // Check that output contains enough space
    fail |= MLI_CHECK((mli_prv_count_elem_num (in) * mli_hlp_tensor_element_size (in)) <= out->data.capacity,
                      "capacity of output tensor is too small");
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;

    return MLI_STATUS_OK;
}

mli_status mli_chk_leaky_relu_fx8 (const mli_tensor * in, const mli_tensor * slope_coeff, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_leaky_relu(in, slope_coeff, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(slope_coeff->el_type == MLI_EL_FX_8, "Wrong slope_coeff tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_leaky_relu_fx16 (const mli_tensor * in, const mli_tensor * slope_coeff, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_leaky_relu(in, slope_coeff, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(slope_coeff->el_type == MLI_EL_FX_16, "Wrong slope_coeff tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_basic_rnn_cell (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    // Check that tensors are valid
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (prev_out), "Bad prev_out tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights), "Bad weights tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (bias), "Bad bias tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(check_layout_is_contiguous(in), "Memory Layout of input tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(prev_out), "Memory Layout of prev_out tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(weights), "Memory Layout of weights tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(bias), "Memory Layout of bias tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(
                          out->mem_stride,
                          (cfg->mode == RNN_ONE_TO_ONE || cfg->mode == RNN_BATCH_TO_LAST) ? bias->rank : 2),
                      "Memory Layout of output tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check config and IR tensors are valid
    if (MLI_CHECK(cfg != NULL , "Bad cfg pointer")) return MLI_STATUS_BAD_FUNC_CFG;
    fail |= MLI_CHECK(!(cfg->mode == RNN_BATCH_TO_LAST && (cfg->ir_tsr == NULL || cfg->ir_tsr->data.mem.void_p == NULL)),
                      "BasicRNN in Batch-to-last mode requires IR of sufficient capacity");
    fail |= MLI_CHECK(!(cfg->mode != RNN_ONE_TO_ONE && in->rank < 2),
                      "Input tensor isn't capable for batch mode of BasicRNN");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    // Get number of elements and check input
    int in_elements;
    int out_elements = mli_prv_count_elem_num (bias);
    int prev_out_elements = mli_prv_count_elem_num (prev_out);
    int out_batches = (cfg->mode == RNN_BATCH_TO_BATCH) ? in->shape[0] : 1;
    if (cfg->mode == RNN_ONE_TO_ONE)
        in_elements = mli_prv_count_elem_num (in);
    else
        in_elements = mli_prv_count_elem_num_part (in, 1);

    // Check shapes of Learned parameters inputs
    if (cfg->mode == RNN_ONE_TO_ONE && weights->rank == 3) {
        // Stacked weights -> stacked output case
        fail |= MLI_CHECK(bias->rank == 2, "bias should have rank 2 in stacked output case");
        fail |= MLI_CHECK(weights->shape[0] == bias->shape[0], "shape mismatch weights and bias in stacked output case");
        fail |= MLI_CHECK(weights->shape[1] == bias->shape[1], "shape mismatch weights and bias in stacked output case");
        fail |= MLI_CHECK(bias->shape[1] == prev_out_elements, "shape mismatch weights and prev in stacked output case");
        fail |= MLI_CHECK(in_elements + prev_out_elements == weights->shape[2],
                          "weights shape mismatch in stacked output case");
        if (fail) return MLI_STATUS_SHAPE_MISMATCH;
    } else {
        // Basic RNN general case
        fail |= MLI_CHECK(weights->rank == 2, "weights should have rank 2 in RNN general case");
        fail |= MLI_CHECK(bias->rank == 1, "bias should have rank 1 in RNN general case");
        fail |= MLI_CHECK(weights->shape[0] == bias->shape[0], "shape mismatch weights and bias in RNN general case");
        fail |= MLI_CHECK(bias->shape[0] == prev_out_elements, "shape mismatch weights and prev in RNN general case");
        fail |= MLI_CHECK(in_elements + prev_out_elements == weights->shape[1],
                          "weights shape mismatch in RNN general case");
        if (fail) return MLI_STATUS_SHAPE_MISMATCH;
    }

    // Check shapes of variable tensors
    if (MLI_CHECK(prev_out->rank == 1, "prev_out rank should be 1")) return MLI_STATUS_SHAPE_MISMATCH;

    // Check data type of tensors
    fail |= MLI_CHECK(weights->el_type == bias->el_type, "element type of weights and bias has to be the same");
    fail |= MLI_CHECK(in->el_type == prev_out->el_type, "element type of in and prev_out has to be the same");
    if (fail) return MLI_STATUS_TYPE_MISMATCH;

    fail |= MLI_CHECK(bias->el_params.fx.frac_bits <= in->el_params.fx.frac_bits + weights->el_params.fx.frac_bits,
                      "The number of fractional bits of the accumulator will be the sum of the frac bits of in and weights. If bias has more frac bits, precision will be lost.");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check capacity of output and IR
    fail |= MLI_CHECK((out_elements * out_batches * mli_hlp_tensor_element_size (in)) <= out->data.capacity,
                      "capacity of output tensor is too small");

    // Check capacity of IR
    if (cfg->mode == RNN_BATCH_TO_LAST){
        fail |= MLI_CHECK((out_elements * mli_hlp_tensor_element_size (in)) <= cfg->ir_tsr->data.capacity,
                          "capacity of IR tensor is too small");
        if (fail) return MLI_STATUS_NOT_ENGH_MEM;
    }

    return MLI_STATUS_OK;
}

mli_status mli_chk_basic_rnn_cell_fx8 (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_rnn_cell(in, prev_out, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type       == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(prev_out->el_type == MLI_EL_FX_8, "Wrong prev_out tensor type") ||
        MLI_CHECK(weights->el_type  == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type     == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_basic_rnn_cell_fx16 (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_rnn_cell(in, prev_out, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type       == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(prev_out->el_type == MLI_EL_FX_16, "Wrong prev_out tensor type") ||
        MLI_CHECK(weights->el_type  == MLI_EL_FX_16, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type     == MLI_EL_FX_16, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_basic_rnn_cell_fx8w16d (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_rnn_cell(in, prev_out, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type       == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(prev_out->el_type == MLI_EL_FX_16, "Wrong prev_out tensor type") ||
        MLI_CHECK(weights->el_type  == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type     == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_lstm_cell (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * cell,
        mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    // Check that tensors are valid
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (prev_out), "Bad prev_out tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights), "Bad weights tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (bias), "Bad bias tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (cell), "Bad cell tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    // Check config and IR tensors are valid
    if (MLI_CHECK(cfg != NULL , "Bad cfg pointer")) return MLI_STATUS_BAD_FUNC_CFG;
    if (MLI_CHECK(cfg->ir_tsr != NULL, "bad cfg->ir_tsr pointer")) return MLI_STATUS_BAD_FUNC_CFG;
    if (MLI_CHECK(cfg->ir_tsr->data.mem.void_p != NULL, "bad cfg->ir_tsr->data.mem.void_p pointer")) return MLI_STATUS_BAD_FUNC_CFG;
    if (MLI_CHECK(!(cfg->mode != RNN_ONE_TO_ONE && in->rank < 2), "bad rank")) return MLI_STATUS_BAD_FUNC_CFG;

    // Get number of elements and check input
    uint32_t in_elements;
    uint32_t out_elements = mli_prv_count_elem_num (prev_out);
    uint32_t out_batches = (cfg->mode == RNN_BATCH_TO_BATCH) ? in->shape[0] : 1;
    if (cfg->mode == RNN_ONE_TO_ONE)
        in_elements = mli_prv_count_elem_num (in);
    else
        in_elements = mli_prv_count_elem_num_part (in, 1);

    // Check shapes of Learned parameters input tensors
    fail |= MLI_CHECK(weights->rank == 3, "Wrong weights rank");
    fail |= MLI_CHECK(bias->rank == 2, "Wrong bias rank");
    fail |= MLI_CHECK(weights->shape[0] == 4, "Wrong weights shape");
    fail |= MLI_CHECK(bias->shape[0] == 4, "Wrong bias shape");
    fail |= MLI_CHECK(weights->shape[1] == bias->shape[1], "weights and bias shape mismatch");
    fail |= MLI_CHECK(bias->shape[1] == out_elements, "Wrong bias shape");
    fail |= MLI_CHECK(in_elements + out_elements == weights->shape[2], "Wrong weights shape");

    // Check shapes of variable tensors
    fail |= MLI_CHECK(cell->rank == 1, "Wrong cell rank");
    fail |= MLI_CHECK(prev_out->rank == 1, "Wrong prev_out rank");
    fail |= MLI_CHECK(prev_out->shape[0] == cell->shape[0], "prev_out and cell shape mismatch");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_layout_is_contiguous(in), "Memory Layout of input tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(prev_out), "Memory Layout of prev_out tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(weights), "Memory Layout of weights tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(bias), "Memory Layout of bias tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(
                          out->mem_stride, (cfg->mode == RNN_ONE_TO_ONE || cfg->mode == RNN_BATCH_TO_LAST) ? 1 : 2),
                     "Memory Layout of output tensor must be contiguous"); 
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check data type of tensors
    fail |= MLI_CHECK(weights->el_type == bias->el_type, "element type of weights and bias has to be the same");
    fail |= MLI_CHECK(in->el_type == prev_out->el_type, "element type of in and prev_out has to be the same");
    fail |= MLI_CHECK(in->el_type == cell->el_type, "element type of in and cell has to be the same");
    if (fail) return MLI_STATUS_TYPE_MISMATCH;

    fail |= MLI_CHECK(bias->el_params.fx.frac_bits <= in->el_params.fx.frac_bits + weights->el_params.fx.frac_bits,
                      "The number of fractional bits of the accumulator will be the sum of the frac bits of in and weights. If bias has more frac bits, precision will be lost.");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check capacity of output and IR
    fail |= MLI_CHECK((4 * out_elements * mli_hlp_tensor_element_size (in)) <= cfg->ir_tsr->data.capacity,
                      "capacity of IR tensor is too small");
    fail |= MLI_CHECK((out_batches * out_elements * mli_hlp_tensor_element_size (in)) <= out->data.capacity,
                      "capacity of output tensor is too small");
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;

    return MLI_STATUS_OK;
}
mli_status mli_chk_lstm_cell_fx8 (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * cell,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_lstm_cell(in, prev_out, weights, bias, cfg, cell, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type       == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(prev_out->el_type == MLI_EL_FX_8, "Wrong prev_out tensor type") ||
        MLI_CHECK(weights->el_type  == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type     == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_lstm_cell_fx16 (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * cell,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_lstm_cell(in, prev_out, weights, bias, cfg, cell, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type       == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(prev_out->el_type == MLI_EL_FX_16, "Wrong prev_out tensor type") ||
        MLI_CHECK(weights->el_type  == MLI_EL_FX_16, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type     == MLI_EL_FX_16, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_lstm_cell_fx8w16d (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * cell,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_lstm_cell(in, prev_out, weights, bias, cfg, cell, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type       == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(prev_out->el_type == MLI_EL_FX_16, "Wrong prev_out tensor type") ||
        MLI_CHECK(weights->el_type  == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type     == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_concat (const mli_tensor ** inputs, const mli_concat_cfg * cfg, mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    // Check first input and output tensors
    if (MLI_CHECK(inputs != NULL , "Bad inputs tensor array")) return MLI_STATUS_BAD_TENSOR;

    stat = MLI_CHECK_STATUS(mli_chk_tensor (inputs[0]), "Bad inputs[0] tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;
    fail |= MLI_CHECK(check_layout_is_contiguous(out->mem_stride, inputs[0]->rank), // output rank is input rank
                      "Memory Layout of output tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check config structure
    if (MLI_CHECK(cfg != NULL , "Bad cfg pointer")) return MLI_STATUS_BAD_FUNC_CFG;

    fail |= MLI_CHECK(cfg->axis < inputs[0]->rank, "wrong axis configuration");
    fail |= MLI_CHECK(cfg->tensors_num <= MLI_CONCAT_MAX_TENSORS, "wrong number of tensors");
    fail |= MLI_CHECK(cfg->tensors_num > 0, "wrong number of tensors");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    const mli_tensor *anchor_tsr = inputs[0];
    const int tsr_num = cfg->tensors_num;
    const int conc_axis = cfg->axis;
    uint32_t tot_elem = mli_prv_count_elem_num (anchor_tsr);

    // Check each tensor in the array (in comparison with anchor)
    for (int idx = 1; idx < tsr_num; idx++) {
        stat = MLI_CHECK_STATUS(mli_chk_tensor(inputs[idx]), "Bad input tensor");
        if (stat != MLI_STATUS_OK) return stat;

        fail |= MLI_CHECK(inputs[idx]->el_type == anchor_tsr->el_type, "element type mismatch");
        fail |= MLI_CHECK(inputs[idx]->el_params.fx.frac_bits == anchor_tsr->el_params.fx.frac_bits, "FX mismatch");
        fail |= MLI_CHECK(inputs[idx]->rank == anchor_tsr->rank, "rank mismatch");
        
        fail |= MLI_CHECK(check_layout_is_contiguous(inputs[idx]), "Memory Layout of all input tensors must be contiguous");
        if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

        for (int dim_idx = 0; dim_idx < anchor_tsr->rank; dim_idx++) {
            fail |= MLI_CHECK(dim_idx == conc_axis || inputs[idx]->shape[dim_idx] == anchor_tsr->shape[dim_idx],
                              "shape mismatch");
            if (fail) return MLI_STATUS_SHAPE_MISMATCH;
        }

        tot_elem += mli_prv_count_elem_num (inputs[idx]);
    }

    // Check that output contains enough space
    fail |= MLI_CHECK((tot_elem * mli_hlp_tensor_element_size(anchor_tsr)) <= out->data.capacity,
                      "capacity of output tensor is too small");
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;

    return MLI_STATUS_OK;
}

mli_status mli_chk_concat_fx8 (const mli_tensor ** inputs, const mli_concat_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_concat(inputs, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    // check of first input is enough because mli_chk_concat() checked that all inputs have same type
    if (MLI_CHECK(inputs[0]->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_concat_fx16 (const mli_tensor ** inputs, const mli_concat_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_concat(inputs, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    // check of first input is enough because mli_chk_concat() checked that all inputs have same type
    if (MLI_CHECK(inputs[0]->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_padding2d_chw (const mli_tensor * in, const mli_padding2d_cfg * cfg, mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    // Check that in tensor is valid and out provides valid pointers
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    if (MLI_CHECK(in->rank == 3, "in rank should be 3")) return MLI_STATUS_SHAPE_MISMATCH;
    fail |= MLI_CHECK(check_layout_is_contiguous(in), "Memory Layout of input tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(out->mem_stride, in->rank), // output rank is input rank
                      "Memory Layout of output tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check config structure
    if (MLI_CHECK(cfg != NULL , "Bad cfg pointer")) return MLI_STATUS_BAD_FUNC_CFG;

    // Check that output contains enough space
    unsigned out_elements = 0;
    out_elements = in->shape[FMAP_C_DIM_CHW];
    out_elements *= (in->shape[FMAP_H_DIM_CHW] + cfg->padding_top + cfg->padding_bottom);
    out_elements *= (in->shape[FMAP_W_DIM_CHW] + cfg->padding_left + cfg->padding_right);

    fail |= MLI_CHECK((out_elements * mli_hlp_tensor_element_size(in)) <= out->data.capacity,
                      "capacity of output tensor is too small");
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;

    return MLI_STATUS_OK;
}

mli_status mli_chk_padding2d_chw_fx8 (const mli_tensor * in, const mli_padding2d_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_padding2d_chw(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_padding2d_chw_fx16 (const mli_tensor * in, const mli_padding2d_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_padding2d_chw(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_padding2d_hwc (const mli_tensor * in, const mli_padding2d_cfg * cfg, mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    // Check that in tensor is valid and out provides valid pointers
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    if (MLI_CHECK(in->rank == 3, "in rank should be 3")) return MLI_STATUS_SHAPE_MISMATCH;
    fail |= MLI_CHECK(check_layout_is_contiguous(in), "Memory Layout of input tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(out->mem_stride, in->rank), // output rank is input rank
                      "Memory Layout of output tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check config structure
    if (MLI_CHECK(cfg != NULL , "Bad cfg pointer")) return MLI_STATUS_BAD_FUNC_CFG;

    // Check that output contains enough space
    unsigned out_elements = 0;
    out_elements = in->shape[FMAP_C_DIM_HWC];
    out_elements *= (in->shape[FMAP_H_DIM_HWC] + cfg->padding_top + cfg->padding_bottom);
    out_elements *= (in->shape[FMAP_W_DIM_HWC] + cfg->padding_left + cfg->padding_right);

    fail |= MLI_CHECK((out_elements * mli_hlp_tensor_element_size(in)) <= out->data.capacity,
                      "capacity of output tensor is too small");
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;

    return MLI_STATUS_OK;
}

mli_status mli_chk_padding2d_hwc_fx8 (const mli_tensor * in, const mli_padding2d_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_padding2d_hwc(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_padding2d_hwc_fx16 (const mli_tensor * in, const mli_padding2d_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_padding2d_hwc(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_permute (const mli_tensor * in, const mli_permute_cfg * cfg, mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    // Check that in tensor is valid and out provides valid pointers
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(check_layout_is_contiguous(in), "Memory Layout of input tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(out->mem_stride, in->rank), // output rank is input rank
                      "Memory Layout of output tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check config structure
    if (MLI_CHECK(cfg != NULL , "Bad cfg pointer")) return MLI_STATUS_BAD_FUNC_CFG;

    for (int idx = 0; idx < in->rank; idx++) {
        if (MLI_CHECK(cfg->perm_dim[idx] < in->rank, "rank mismatch"))
            return MLI_STATUS_BAD_FUNC_CFG;

        // Each permute dimension must be unique
        for (int jdx = idx + 1; jdx < in->rank; jdx++)
            if (MLI_CHECK(cfg->perm_dim[idx] != cfg->perm_dim[jdx], "Each permute dimension must be unique"))
                return MLI_STATUS_BAD_FUNC_CFG;
    }

    // Check that output contains enough space
    fail |= MLI_CHECK((mli_prv_count_elem_num (in) * mli_hlp_tensor_element_size (in)) <= out->data.capacity,
                      "capacity of output tensor is too small");
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;

    return MLI_STATUS_OK;
}

mli_status mli_chk_permute_fx8 (const mli_tensor * in, const mli_permute_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_permute(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_permute_fx16 (const mli_tensor * in, const mli_permute_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_permute(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_count_elem_num(const mli_tensor *in, uint32_t start_dim) {
    if (MLI_CHECK(in->rank <= MLI_MAX_RANK, "rank should not exceed MAX_RANK"))
        return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(start_dim < in->rank, "start_dim should be smaller than rank"))
        return MLI_STATUS_BAD_FUNC_CFG;
    return MLI_STATUS_OK;
}

mli_status mli_chk_convert_tensor(mli_tensor *in, mli_tensor *out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    // Check that in tensor is valid and out provides valid pointers
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(out->data.mem.void_p != NULL , "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(check_layout_is_contiguous(in), "Memory Layout of input tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(out->mem_stride, in->rank), // output rank is input rank
        "Memory Layout of output tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check that output contains enough space
    const unsigned out_elements = mli_prv_count_elem_num(in);
    fail |= MLI_CHECK((out_elements * mli_hlp_tensor_element_size(out)) <= out->data.capacity,
                      "capacity of output tensor is too small");
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;

    // TODO: Check fraq bits and etc.
    return MLI_STATUS_OK;
}

mli_status mli_chk_point_to_subtensor(const mli_tensor *in, const mli_point_to_subtsr_cfg *cfg, mli_tensor *out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    // Check that in tensor is valid and out provides valid pointers
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;

    if (MLI_CHECK(cfg != NULL , "Bad cfg pointer")) return MLI_STATUS_BAD_FUNC_CFG;
    if (MLI_CHECK(cfg->coord_num <= in->rank, "incorrect number of coordinates"))
        return MLI_STATUS_BAD_FUNC_CFG;

    for (int i = 0; i < cfg->coord_num; i++) {
        if (MLI_CHECK(cfg->start_coord[i] < in->shape[i], "bad config"))
            return MLI_STATUS_BAD_FUNC_CFG;
    }

    const uint32_t subtensor_end = cfg->start_coord[cfg->coord_num - 1] + cfg->first_out_dim_size;
    if (MLI_CHECK(subtensor_end <= in->shape[cfg->coord_num - 1], "bad config"))
        return MLI_STATUS_BAD_FUNC_CFG;

    const uint32_t subtsr_start_axis = cfg->coord_num - 1;
    const uint32_t out_rank = in->rank - subtsr_start_axis;
    fail |= MLI_CHECK(check_layout_is_contiguous(in), "Memory Layout of input tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(out->mem_stride, out_rank), 
                      "Memory Layout of output tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    return MLI_STATUS_OK;
}

mli_status mli_chk_create_subtensor(const mli_tensor *in, const mli_sub_tensor_cfg *cfg, mli_tensor *out) {
    mli_status stat = MLI_STATUS_OK;

    // Check that in tensor is valid and out provides valid pointers
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;

    if (MLI_CHECK(cfg != NULL , "Bad cfg pointer")) return MLI_STATUS_BAD_FUNC_CFG;
    if (MLI_CHECK(cfg->sub_tensor_rank <= in->rank, "incorrect number of coordinates"))
        return MLI_STATUS_BAD_FUNC_CFG;

    for (int i = 0; i < in->rank; i++) {
        if (MLI_CHECK(cfg->offset[i] < in->shape[i], "bad config"))
            return MLI_STATUS_BAD_FUNC_CFG;
        if (MLI_CHECK(cfg->offset[i] + cfg->size[i] <= in->shape[i], "bad config"))
            return MLI_STATUS_BAD_FUNC_CFG;
    }

    return MLI_STATUS_OK;
}

#pragma code()

#ifdef __cplusplus
}
#endif
