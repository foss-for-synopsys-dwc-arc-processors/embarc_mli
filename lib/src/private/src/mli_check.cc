/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_check.h"

#include <stdio.h>
#include <assert.h>

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_math_macros.h"
#include "mli_mem_info.h"
#include "mli_prv_activation_lut.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

#pragma MLI_CODE_SECTION_START(".mli_lib")

/*check_bank checks whether the tensor buffer have to be allocated in ccm memory or not and its value will be:
 *  -false: if the out_buffer uses MLI_CONV_OUT_PTR which means for the output buffers of all weights based kernels.
 *  -true: Otherwise.
 */
MLI_FORCE_INLINE mli_status mli_mem_chk(const mli_tensor *t, bool check_bank) {
#if (PLATFORM == V2DSP_XY) || (PLATFORM == V2DSP_VECTOR)
    void *p = mli_prv_tensor_cast_data_ptr(t);
    uint32_t align_mask = mli_hlp_tensor_element_size(t) - 1;
    return mli_mem_chk_ptr(p, align_mask, check_bank);
#else
    return MLI_STATUS_OK;
#endif
}

static MLI_FORCE_INLINE bool check_ptr_not_null(mli_data_container data, mli_element_type el_type) {
    bool val = false;
    switch (el_type) {
		case MLI_EL_FX_8:
		case MLI_EL_SA_8:
			val = (data.mem.pi8 != NULL);
			break;
		case MLI_EL_FX_16:
			val = (data.mem.pi16 != NULL);
			break;
		case MLI_EL_SA_32:
			val = (data.mem.pi32 != NULL);
			break;
		case MLI_EL_FP_32:
			val = (data.mem.pf32 != NULL);
			break;
		default:
			MLI_ASSERT(0);
			val = (data.mem.pi8 != NULL);
			break;
	}
	return val;
}

static MLI_FORCE_INLINE bool check_ptr_not_null(const mli_tensor *t) {
    return check_ptr_not_null(t->data, t->el_type);
}

static MLI_FORCE_INLINE mli_status check_tensor_private(
        const uint32_t *shape,
        const int32_t *mem_stride,
        uint32_t rank,
        uint32_t capacity,
        uint32_t element_size) {
    bool fail = false;

    fail |= MLI_CHECK(rank <= MLI_MAX_RANK, "Wrong tensor rank");
    fail |= MLI_CHECK(capacity <= INT32_MAX, "Capacity too big");
    for (int i = 0; i < (int)rank; i++) {
        fail |= MLI_CHECK(mem_stride[i] != 0, "Memory stride invalid");
        fail |= MLI_CHECK(mem_stride[i] > 0, "Negative memory strides are not supported");
        fail |= MLI_CHECK(shape[i] > 0, "Shape invalid");
    }
    if (fail) return MLI_STATUS_BAD_TENSOR;

    uint32_t size = 1;
    uint32_t previous_shape = 1;
    uint32_t previous_mem_stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
        fail |= MLI_CHECK(mem_stride[i] >= ((int)previous_shape * (int)previous_mem_stride), "Tensor mem stride too small");
        previous_shape = shape[i];
        previous_mem_stride = mem_stride[i];
        size += (previous_shape - 1) * previous_mem_stride;
    }
    size *= element_size;

    if (fail) return MLI_STATUS_BAD_TENSOR;
    fail |= MLI_CHECK(capacity >= size, "Insufficient tensor capacity");
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;
    return MLI_STATUS_OK;
}

mli_status mli_chk_lut(const mli_lut * lut, int buff_size) {
    if (MLI_CHECK((int)lut->data.capacity >= buff_size, "Insufficient lut data capacity")) {
        return MLI_STATUS_NOT_ENGH_MEM;
    }
#if (PLATFORM == V2DSP_XY) || (PLATFORM == V2DSP_VECTOR)
    void *p = (void*)lut->data.mem.pi16;
    mli_tensor dummy = { .el_type = lut->type };
    uint32_t align_mask = mli_hlp_tensor_element_size(&dummy) - 1;

    mli_status ret = mli_mem_chk_ptr(p, align_mask, /*check_bank=*/true);
    return ret;
#else
    return MLI_STATUS_OK;
#endif
}

/************************************************************
*  mli_tensor quantization parameters correctness checking
*************************************************************/
#if MLI_DBG_ENABLE_RETURNCODES
// These constants are used in debug mode only.
// To not rise a warning in release mode we exclude it by pre-processor.
static constexpr unsigned kZeroPointBitsZero = 0;
static constexpr unsigned kZeroPointBitsByteRange = (sizeof(int8_t) * 8) - 1;
#endif
static constexpr unsigned kZeroPointBitsMaxRange = (sizeof(int16_t) * 8) - 1;

mli_status mli_chk_tensor_quant_params(const mli_tensor* in, unsigned zp_used_bits = kZeroPointBitsMaxRange) {
    MLI_ASSERT(zp_used_bits <= kZeroPointBitsMaxRange);
    MLI_ASSERT(in != nullptr);
    switch (in->el_type) {
    case MLI_EL_FX_4:
    case MLI_EL_FX_8:
    case MLI_EL_FX_16:
    case MLI_EL_FP_16:
    case MLI_EL_FP_32:
        return MLI_STATUS_OK;
        break;

    case MLI_EL_SA_8:
    case MLI_EL_SA_32:
    {
        const bool is_per_axis = in->el_params.sa.dim >= 0;
        const int num_quant_vals = static_cast<int>((is_per_axis)? in->shape[in->el_params.sa.dim]: 1);
        const int16_t* scales = (is_per_axis)? in->el_params.sa.scale.mem.pi16: &in->el_params.sa.scale.mem.i16;
        const int16_t* zp = (is_per_axis)? in->el_params.sa.zero_point.mem.pi16: &in->el_params.sa.zero_point.mem.i16;
        const int8_t* scale_frac_bits = (is_per_axis)? in->el_params.sa.scale_frac_bits.mem.pi8: &in->el_params.sa.scale_frac_bits.mem.i8;
        bool fail = false;
        MLI_ASSERT(scales != nullptr && zp != nullptr && scale_frac_bits != nullptr);

        const int16_t zp_min = (zp_used_bits == 0)? 0 : static_cast<int16_t>(-(1 << zp_used_bits));
        const int16_t zp_max = (zp_used_bits == 0)? 0 : static_cast<int16_t>((1 << zp_used_bits) - 1);
        fail |= MLI_CHECK(is_per_axis ? scales != nullptr && in->el_params.sa.scale.capacity >= num_quant_vals * sizeof(scales[0])
            : in->el_params.sa.scale.capacity == 0,
            "SA Tensor Scales data container is invalid");
        fail |= MLI_CHECK(is_per_axis ? zp != nullptr && in->el_params.sa.zero_point.capacity >= num_quant_vals * sizeof(zp[0])
            : in->el_params.sa.zero_point.capacity == 0,
            "SA Tensor zero points data container is invalid");
        fail |= MLI_CHECK(is_per_axis ? scale_frac_bits != nullptr
            && in->el_params.sa.scale_frac_bits.capacity >= num_quant_vals * sizeof(scale_frac_bits[0])
            : in->el_params.sa.scale_frac_bits.capacity == 0,
            "SA Tensor scales frac bits data container is invalid");
        if (fail) return MLI_STATUS_BAD_TENSOR;

        for (int idx = 0; idx < num_quant_vals; idx++) {
            fail = MLI_CHECK(scales[idx] > 0, "SA Tensor contains non-positive scale value");
            if (fail) return MLI_STATUS_BAD_TENSOR;

            // Check flexible requirements for tensor, and return "incompatible" status if it doesnt met.
            fail |= MLI_CHECK(zp[idx] >= zp_min && zp[idx] <= zp_max,
                    "Additional requirement: SA Tensor contains zero point out of supported  range");
            if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;
        }
        return MLI_STATUS_OK;
        break;
    }
    default:
        MLI_ASSERT(0);
    };
    return MLI_STATUS_OK;
}


/******************************************************
 *  mli_tensor data structure correctness checking
 ******************************************************/
#ifdef __cplusplus
extern "C" {
#endif
mli_status mli_chk_tensor (const mli_tensor * in, bool check_bank) {
	mli_status stat = MLI_STATUS_OK;
    if (MLI_CHECK(in != NULL, "Bad tensor null pointer")) stat = MLI_STATUS_BAD_TENSOR;
    if (stat == MLI_STATUS_OK) stat = check_tensor_private(in->shape, in->mem_stride, in->rank, in->data.capacity, mli_hlp_tensor_element_size(in));
    if (stat == MLI_STATUS_OK) stat = mli_chk_tensor_quant_params(in);
    if (stat == MLI_STATUS_OK) stat = MLI_CHECK_STATUS(mli_mem_chk(in, check_bank), "Memory check error");
    return stat;
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

    if (MLI_CHECK(in != NULL, "Bad tensor null pointer")) stat = MLI_STATUS_BAD_TENSOR;

    if (in->rank == 0) {
        // for tensors with rank 0 no need for extra checks beside quant params.
        if (stat == MLI_STATUS_OK) stat = mli_chk_tensor_quant_params(in);
    } else {
        // for tensors with any rank, it has to be a valid tensor with 1 element.
        if (stat == MLI_STATUS_OK) stat = MLI_CHECK_STATUS(mli_chk_tensor(in), "Bad input tensor");
        if (stat == MLI_STATUS_OK) {
            if (MLI_CHECK(mli_prv_count_elem_num(in) == 1, "Scalar tensor has to have only 1 element"))
                stat = MLI_STATUS_SHAPE_MISMATCH;
        }
    }

    return stat;
}

bool mli_tensor_is_scalar (const mli_tensor * in) {
    if (in == NULL) return false;
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

int scale_fluctuations_headroom(int32_t scale) {
    // This functions finds acceptabla fluctuations of scale representation
    // assuming that it was transformed from another representation
    // (for instance float number transformed to q31)

    // assuming scale > 0, we try to find unused MSB (zero bits in the front)
    // Note: an external norm operation can be used for it
    MLI_ASSERT(scale > 0);
    int unused_bits = 0;
    int32_t norm_value = 1 << 30;
    for (; norm_value > scale; norm_value = norm_value >> 1)
        ++unused_bits;

    // Knowing unused bits we can estimate fluctuations
    // For instance: int32 have 31 significant bits, while floats have only 24.
    // It means that (31 - 24) = 7 bits of Q31 might keep some trash after float->q31 transfromation
    // But if 3 MSB of q31 number are not used, than only (7 - 3) = 4 bits are expected to have noise.
    // We conclude that some fluctuations in range of (+16 : -16) are acceptable (2**4 = 16).
    constexpr int kSignifacntBitsFloatSP = 24;
    constexpr int kSignifacntBitsScale = 31;
    const int fluctuations_bits = MAX(0, (kSignifacntBitsScale - kSignifacntBitsFloatSP) - unused_bits);
    const int fluctuations_headroom = 1 << fluctuations_bits;
    return fluctuations_headroom;
};

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
    const int16_t* w_scales = (is_per_axis)? weights->el_params.sa.scale.mem.pi16: &weights->el_params.sa.scale.mem.i16;
    const int16_t* b_scales = (is_per_axis)? bias->el_params.sa.scale.mem.pi16: &bias->el_params.sa.scale.mem.i16;
    const int16_t scale_in = in->el_params.sa.scale.mem.i16;
    for (int idx = 0; idx < num_scale_vals; idx++) {
        int32_t bias_scale_expected = scale_in * w_scales[idx];
        int out_shift = mli_prv_calc_shift_idx(in, weights, bias, idx);
        bias_scale_expected = (out_shift > 0)
                ? mli_math_asr_rnd_fx(bias_scale_expected, out_shift)
                : bias_scale_expected << -out_shift;
        const int32_t scales_diff = bias_scale_expected - b_scales[idx];
        // Check that diff is about the rounding error
        if (MLI_CHECK(scales_diff <= 1 && scales_diff >= -1, "Bias scale must be the multiplication of input and weights scales for correct calculations in current quanization scheme"))
            return MLI_STATUS_INCOMPATEBLE_TENSORS;
    }
    return MLI_STATUS_OK;
}

static MLI_FORCE_INLINE bool check_inner_most_dimension_is_one(const mli_tensor *t) {
    return (t->mem_stride[t->rank - 1] == 1);
}

static MLI_FORCE_INLINE bool check_layout_is_contiguous(const int32_t *mem_stride, uint32_t rank) {
    // When only mem_stride and rank is under considiration, contiguous means
    // all memory strides are zero OR rank is 1 and memory stride between elements is 1

    if (rank == 1 && mem_stride[0] == 1)
        return true;
    else
        return false;
}

static MLI_FORCE_INLINE bool check_layout_is_contiguous(const uint32_t *shape, const int32_t *mem_stride, uint32_t rank) {
    // This function either requires that the memory strides are set such that it results in the
    // same memory layout.

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

static MLI_FORCE_INLINE bool check_layout_is_contiguous(const mli_tensor *t) {
    return check_layout_is_contiguous(t->shape, t->mem_stride, t->rank);
}

static MLI_FORCE_INLINE bool check_mem_stride_matches(const mli_tensor *t1, const mli_tensor *t2) {
    bool fail = false;
    fail |= MLI_CHECK(t1->rank == t2->rank, "Tensors ranks doesn't match");
    if (fail) return !fail;
    for (uint32_t i = 0; i < t1->rank - 1; ++i) {
        fail |= MLI_CHECK(t1->mem_stride[i] == t2->mem_stride[i],
                "Tensors memstrides doesn't match");
    }
    return !fail;
}

static MLI_FORCE_INLINE bool check_quantized_on_tensor(const mli_tensor *t1, const mli_tensor *t2) {
    bool fail = false;
    if (t1->el_type == MLI_EL_SA_8 || t1->el_type == MLI_EL_SA_32) {
        fail |= ((t1->el_params.sa.dim >= 0) || (t2->el_params.sa.dim >= 0));
    }

    return !fail;
}

/******************************************************
 *  mli_krn_conv2d_hwc parameters checking function
 ******************************************************/
#ifdef __cplusplus
extern "C" {
#endif
mli_status mli_chk_conv2d_hwcn (
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_CONV_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights), "Bad weights tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (bias), "Bad bias tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL, "Bad Output tensor pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

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
    int dilation_width = cfg->dilation_width;
    int dilation_height = cfg->dilation_height;
    int effective_kernel_width = (kernel_width - 1) * dilation_width + 1;
    int effective_kernel_height = (kernel_height - 1) * dilation_height + 1;
    int effective_input_width = in->shape[FMAP_W_DIM_HWC] + cfg->padding_left + cfg->padding_right;
    int effective_input_height = in->shape[FMAP_H_DIM_HWC] + cfg->padding_top + cfg->padding_bottom;
    int required_width = out->shape[FMAP_W_DIM_HWC] * cfg->stride_width + effective_kernel_width - cfg->stride_width;
    int required_height = out->shape[FMAP_H_DIM_HWC] * cfg->stride_height + effective_kernel_height - cfg->stride_height;
    fail |= MLI_CHECK(cfg->dilation_height > 0, "Dilation should be greater than zero");
    fail |= MLI_CHECK(cfg->dilation_width > 0, "Dilation should be greater than zero");
    fail |= MLI_CHECK(cfg->padding_left < effective_kernel_width, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->padding_right < effective_kernel_width, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->padding_top < effective_kernel_height, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->padding_bottom < effective_kernel_height, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->stride_height > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(cfg->stride_width > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(required_height <= effective_input_height, "incorrect output height");
    fail |= MLI_CHECK(required_width <= effective_input_width, "incorrect output width");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;
    stat = check_tensor_private(out->shape, out->mem_stride, 3, out->data.capacity, mli_hlp_tensor_element_size(out));

    return stat;
}

mli_status mli_chk_conv2d_hwcn_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_conv2d_hwcn(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    if (MLI_CHECK(in->el_type          == MLI_EL_SA_8, "Wrong input tensor type") ||
            MLI_CHECK(weights->el_type == MLI_EL_SA_8, "Wrong weights tensor type") ||
            MLI_CHECK(bias->el_type    == MLI_EL_SA_32, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    // Check additional requrements for tensor params.
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in, kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(out, kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(weights, kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(bias,    kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    if (weights->el_params.sa.dim < 0) {
        if (MLI_CHECK(bias->el_params.sa.dim < 0, "Bias tensor: per tensor quantization is expected (similar to weights)"))
            return MLI_STATUS_INCOMPATEBLE_TENSORS;
    } else {
        if (MLI_CHECK(weights->el_params.sa.dim == KRNL_C_DIM_HWCN, "Weights tensor: per output channels quantization is expected") ||
            MLI_CHECK(bias->el_params.sa.dim == 0, "Bias tensor: per output channels quantization is expected"))
            return MLI_STATUS_INCOMPATEBLE_TENSORS;
    }

    if (MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected") ||
        MLI_CHECK(out->el_params.sa.dim < 0, "Output tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;

    ret = MLI_CHECK_STATUS(mli_chk_bias_scale_asym(in, weights, bias), __func__);
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

mli_status mli_chk_depthwise_conv2d_hwcn(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_CONV_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights), "Bad weights tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (bias), "Bad bias tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(in->rank == 3, "Wrong input rank");
    fail |= MLI_CHECK(weights->rank == 4, "Wrong weights rank");
    fail |= MLI_CHECK(bias->rank == 1, "Wrong bias rank");
    fail |= MLI_CHECK(weights->shape[KRNL_DW_D_DIM_HW1N] == 1, "Wrong weights shape");
    fail |= MLI_CHECK(bias->shape[0] == weights->shape[KRNL_DW_N_DIM_HW1N], "Shape mismatch bias and weights");
    fail |= MLI_CHECK(weights->shape[KRNL_DW_N_DIM_HW1N] == in->shape[FMAP_C_DIM_HWC], "Shape mismatch in and weights");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in), "Memory stride for inner most dimension of input must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(weights), "Memory stride for inner most dimension of weights must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(bias), "Memory stride for inner most dimension of bias must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out), "Memory stride for inner most dimension of output must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    int kernel_width = weights->shape[KRNL_DW_W_DIM_HW1N];
    int kernel_height = weights->shape[KRNL_DW_H_DIM_HW1N];
    int dilation_width = cfg->dilation_width;
    int dilation_height = cfg->dilation_height;
    int effective_kernel_width = (kernel_width - 1) * dilation_width + 1;
    int effective_kernel_height = (kernel_height - 1) * dilation_height + 1;
    int effective_input_width = in->shape[FMAP_W_DIM_HWC] + cfg->padding_left + cfg->padding_right;
    int effective_input_height = in->shape[FMAP_H_DIM_HWC] + cfg->padding_top + cfg->padding_bottom;
    int required_width = out->shape[FMAP_W_DIM_HWC] * cfg->stride_width + effective_kernel_width - cfg->stride_width;
    int required_height = out->shape[FMAP_H_DIM_HWC] * cfg->stride_height + effective_kernel_height - cfg->stride_height;
    fail |= MLI_CHECK(cfg->dilation_height > 0, "Dilation should be greater than zero");
    fail |= MLI_CHECK(cfg->dilation_width > 0, "Dilation should be greater than zero");
    fail |= MLI_CHECK(cfg->padding_left < effective_kernel_width, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->padding_right < effective_kernel_width, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->padding_top < effective_kernel_height, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->padding_bottom < effective_kernel_height, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->stride_height > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(cfg->stride_width > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(required_height <= effective_input_height, "incorrect output height");
    fail |= MLI_CHECK(required_width <= effective_input_width, "incorrect output width");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;
    stat = check_tensor_private(out->shape, out->mem_stride, 3, out->data.capacity, mli_hlp_tensor_element_size(out));

    return stat;
}
mli_status mli_chk_depthwise_conv2d_hwcn_fx16(
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
    ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_hwcn(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_depthwise_conv2d_hwcn_fx16_fx8_fx8(
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
    ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_hwcn(in, weights, bias, cfg, out), __func__);
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
    mli_status ret = MLI_CHECK_STATUS(mli_chk_depthwise_conv2d_hwcn(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    if (MLI_CHECK(in->el_type          == MLI_EL_SA_8, "Wrong input tensor type") ||
            MLI_CHECK(weights->el_type == MLI_EL_SA_8, "Wrong weights tensor type") ||
            MLI_CHECK(bias->el_type    == MLI_EL_SA_32, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    // Check additional requrements for tensor params.
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in, kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(out, kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(weights,  kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(bias,     kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    if (weights->el_params.sa.dim < 0) {
        if (MLI_CHECK(bias->el_params.sa.dim < 0, "Bias tensor: per tensor quantization is expected (similar to weights)"))
            return MLI_STATUS_INCOMPATEBLE_TENSORS;
    } else {
        if (MLI_CHECK(weights->el_params.sa.dim == KRNL_DW_N_DIM_HW1N, "Weights tensor: per output channels quantization is expected") ||
            MLI_CHECK(bias->el_params.sa.dim == 0, "Bias tensor: per output channels quantization is expected"))
            return MLI_STATUS_INCOMPATEBLE_TENSORS;
    }

    if (MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected") ||
        MLI_CHECK(out->el_params.sa.dim < 0, "Output tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;

    ret = MLI_CHECK_STATUS(mli_chk_bias_scale_asym(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_group_conv2d_hwcn(
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
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(in->rank == 3, "Wrong input rank");
    fail |= MLI_CHECK(weights->rank == 4, "Wrong weights rank");
    fail |= MLI_CHECK(bias->rank == 1, "Wrong bias rank");
    fail |= MLI_CHECK(bias->shape[0] == weights->shape[KRNL_C_DIM_HWCN], "Shape mismatch bias and weights");

    int group_count = in->shape[FMAP_C_DIM_HWC] / weights->shape[KRNL_D_DIM_HWCN];
    fail |= MLI_CHECK(in->shape[FMAP_C_DIM_HWC] % weights->shape[KRNL_D_DIM_HWCN] == 0, "Input channel count must be multiple of group count");
    fail |= MLI_CHECK(weights->shape[KRNL_C_DIM_HWCN] % group_count == 0, "Number of filters must be multiple of group count");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in), "Memory stride for inner most dimension of input must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(weights), "Memory stride for inner most dimension of weights must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(bias), "Memory stride for inner most dimension of bias must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out), "Memory stride for inner most dimension of output must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    int kernel_width = weights->shape[KRNL_W_DIM_HWCN];
    int kernel_height = weights->shape[KRNL_H_DIM_HWCN];
    int dilation_width = cfg->dilation_width;
    int dilation_height = cfg->dilation_height;
    int effective_kernel_width = (kernel_width - 1) * dilation_width + 1;
    int effective_kernel_height = (kernel_height - 1) * dilation_height + 1;
    int effective_input_width = in->shape[FMAP_W_DIM_HWC] + cfg->padding_left + cfg->padding_right;
    int effective_input_height = in->shape[FMAP_H_DIM_HWC] + cfg->padding_top + cfg->padding_bottom;
    int required_width = out->shape[FMAP_W_DIM_HWC] * cfg->stride_width + effective_kernel_width - cfg->stride_width;
    int required_height = out->shape[FMAP_H_DIM_HWC] * cfg->stride_height + effective_kernel_height - cfg->stride_height;
    fail |= MLI_CHECK(cfg->dilation_height > 0, "Dilation should be greater than zero");
    fail |= MLI_CHECK(cfg->dilation_width > 0, "Dilation should be greater than zero");
    fail |= MLI_CHECK(cfg->padding_left < effective_kernel_width, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->padding_right < effective_kernel_width, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->padding_top < effective_kernel_height, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->padding_bottom < effective_kernel_height, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->stride_height > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(cfg->stride_width > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(required_height <= effective_input_height, "incorrect output height");
    fail |= MLI_CHECK(required_width <= effective_input_width, "incorrect output width");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;
    stat = check_tensor_private(out->shape, out->mem_stride, 3, out->data.capacity, mli_hlp_tensor_element_size(out));

    return stat;
}

mli_status mli_chk_group_conv2d_hwcn_fx16(
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
    ret = MLI_CHECK_STATUS(mli_chk_group_conv2d_hwcn(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_group_conv2d_hwcn_fx16_fx8_fx8(
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
    ret = MLI_CHECK_STATUS(mli_chk_group_conv2d_hwcn(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_group_conv2d_hwcn_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_group_conv2d_hwcn(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    if (MLI_CHECK(in->el_type          == MLI_EL_SA_8, "Wrong input tensor type") ||
            MLI_CHECK(weights->el_type == MLI_EL_SA_8, "Wrong weights tensor type") ||
            MLI_CHECK(bias->el_type    == MLI_EL_SA_32, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    // Check additional requrements for tensor params.
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in,      kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(out,     kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(weights,  kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(bias,     kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    if (weights->el_params.sa.dim < 0) {
        if (MLI_CHECK(bias->el_params.sa.dim < 0, "Bias tensor: per tensor quantization is expected (similar to weights)"))
            return MLI_STATUS_INCOMPATEBLE_TENSORS;
    } else {
        if (MLI_CHECK(weights->el_params.sa.dim == KRNL_C_DIM_HWCN, "Weights tensor: per output channels quantization is expected") ||
            MLI_CHECK(bias->el_params.sa.dim == 0, "Bias tensor: per output channels quantization is expected"))
            return MLI_STATUS_INCOMPATEBLE_TENSORS;
    }

    if (MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected") ||
        MLI_CHECK(out->el_params.sa.dim < 0, "Output tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;

    ret = MLI_CHECK_STATUS(mli_chk_bias_scale_asym(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    return MLI_STATUS_OK;
}

mli_status mli_chk_transpose_conv2d_hwcn (
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_CONV_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights), "Bad weights tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (bias), "Bad bias tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL, "Bad Output tensor pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

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

    const int kernel_width = weights->shape[KRNL_W_DIM_HWCN];
    const int kernel_height = weights->shape[KRNL_H_DIM_HWCN];

    fail |= MLI_CHECK(cfg->dilation_width == 1, "Dilation ratio isn't supported by transpose convolution");
    fail |= MLI_CHECK(cfg->dilation_height == 1, "Dilation ratio isn't supported by transpose convolution");
    fail |= MLI_CHECK(cfg->padding_left < kernel_width, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->padding_right < kernel_width, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->padding_top < kernel_height, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->padding_bottom < kernel_height, "Padding should be smaller than effective kernel size");
    fail |= MLI_CHECK(cfg->stride_height > 0, "Stride should be greater than zero");
    fail |= MLI_CHECK(cfg->stride_width > 0, "Stride should be greater than zero");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    const int in_height = in->shape[FMAP_H_DIM_HWC];
    const int in_width = in->shape[FMAP_W_DIM_HWC];
    const int stride_width = cfg->stride_width;
    const int stride_height = cfg->stride_height;
    const int effective_padding_top = kernel_height - cfg->padding_top - 1;
    const int effective_padding_bot = kernel_height - cfg->padding_bottom - 1;
    const int effective_padding_left = kernel_width - cfg->padding_left - 1;
    const int effective_padding_right = kernel_width - cfg->padding_right - 1;
    const int effective_in_width = (in_width - 1) * stride_width + 1;
    const int effective_in_height = (in_height - 1) * stride_height + 1;

    uint32_t out_shape[3] = {
        (uint32_t)(effective_in_height + effective_padding_top + effective_padding_bot - kernel_height + 1), // h
        (uint32_t)(effective_in_width + effective_padding_left + effective_padding_right - kernel_width + 1), // w
        weights->shape[KRNL_C_DIM_HWCN]}; // c
    stat = check_tensor_private(out_shape, out->mem_stride, 3, out->data.capacity, mli_hlp_tensor_element_size(out));

    return stat;
}

mli_status mli_chk_transpose_conv2d_hwcn_fx16(
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
    ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_transpose_conv2d_hwcn_fx16_fx8_fx8(
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
    ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_transpose_conv2d_hwcn_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_transpose_conv2d_hwcn(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    if (MLI_CHECK(in->el_type          == MLI_EL_SA_8, "Wrong input tensor type") ||
            MLI_CHECK(weights->el_type == MLI_EL_SA_8, "Wrong weights tensor type") ||
            MLI_CHECK(bias->el_type    == MLI_EL_SA_32, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    // Check additional requrements for tensor params.
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in,      kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(out,     kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(weights,  kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(bias,     kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    if (weights->el_params.sa.dim < 0) {
        if (MLI_CHECK(bias->el_params.sa.dim < 0, "Bias tensor: per tensor quantization is expected (similar to weights)"))
            return MLI_STATUS_INCOMPATEBLE_TENSORS;
    } else {
        if (MLI_CHECK(weights->el_params.sa.dim == KRNL_C_DIM_HWCN, "Weights tensor: per output channels quantization is expected") ||
            MLI_CHECK(bias->el_params.sa.dim == 0, "Bias tensor: per output channels quantization is expected"))
            return MLI_STATUS_INCOMPATEBLE_TENSORS;
    }

    if (MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected") ||
        MLI_CHECK(out->el_params.sa.dim < 0, "Output tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;

    ret = MLI_CHECK_STATUS(mli_chk_bias_scale_asym(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_transpose_conv2d_hwcn_k2x2_str2(
        const mli_tensor * /*in*/,
        const mli_tensor * weights,
        const mli_tensor * /*bias*/,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * /*out*/) {
    bool fail = false;

    const int kernel_width = weights->shape[KRNL_W_DIM_HWCN];
    const int kernel_height = weights->shape[KRNL_H_DIM_HWCN];
    const int stride_width = cfg->stride_width;
    const int stride_height = cfg->stride_height;
    fail |= MLI_CHECK(stride_width == 2, "Stride width should be 2 for k2x2_str2 specialization");
    fail |= MLI_CHECK(stride_height == 2, "Stride height should be 2 for k2x2_str2 specialization");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    fail |= MLI_CHECK(kernel_width == 2, "Kernel width should be 2 for k2x2_str2 specialization");
    fail |= MLI_CHECK(kernel_height == 2, "Kernel height should be 2 for k2x2_str2 specialization");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    return MLI_STATUS_OK;
}

mli_status mli_chk_transpose_conv2d_hwcn_k4x4_str2(
        const mli_tensor * /*in*/,
        const mli_tensor * weights,
        const mli_tensor * /*bias*/,
        const mli_conv2d_cfg * cfg,
        const mli_tensor * /*out*/) {
    bool fail = false;

    const int kernel_width = weights->shape[KRNL_W_DIM_HWCN];
    const int kernel_height = weights->shape[KRNL_H_DIM_HWCN];
    const int stride_width = cfg->stride_width;
    const int stride_height = cfg->stride_height;
    fail |= MLI_CHECK(stride_width == 2, "Stride width should be 2 for k4x4_str2 specialization");
    fail |= MLI_CHECK(stride_height == 2, "Stride height should be 2 for k4x4_str2 specialization");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    fail |= MLI_CHECK(kernel_width == 4, "Kernel width should be 4 for k4x4_str2 specialization");
    fail |= MLI_CHECK(kernel_height == 4, "Kernel height should be 4 for k4x4_str2 specialization");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    return MLI_STATUS_OK;
}

mli_status mli_chk_maxpool_hwc (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

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

mli_status mli_chk_avepool_hwc (const mli_tensor * in, const mli_pool_cfg * cfg, const mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

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

    if (MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected") ||
            MLI_CHECK(out->el_params.sa.dim < 0, "Output tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check additional requrements for tensor params.
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in,      kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(out,     kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    return MLI_STATUS_OK;
}

mli_status mli_chk_fully_connected (
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_fully_connected_cfg * cfg,
        mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;
    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_CONV_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights), "Bad weights tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (bias), "Bad bias tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(weights->rank == 2, "Wrong weights rank");
    fail |= MLI_CHECK(bias->rank == 1, "Wrong bias rank");
    fail |= MLI_CHECK(mli_prv_count_elem_num (in) == weights->shape[0], "weights shape doesn't match number of input elements");
    fail |= MLI_CHECK(bias->shape[0] == weights->shape[1], "Shape mismatch bias and weights");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_layout_is_contiguous(in), "Memory Layout of input tensor must be contiguous");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(weights), "Memory stride for inner most dimension of weights must be 1");
    fail |= MLI_CHECK(check_layout_is_contiguous(bias), "Memory Layout of bias tensor must be contiguous");
    fail |= MLI_CHECK(check_layout_is_contiguous(out->mem_stride, 1), "Memory Layout of output tensor must be contiguous");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    fail |= MLI_CHECK((weights->shape[1] * mli_hlp_tensor_element_size (in)) <= out->data.capacity, "capacity of output tensor is too small");
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;

    return MLI_STATUS_OK;
}

mli_status mli_chk_fully_connected_fx8w16d(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_fully_connected_cfg * cfg,
        mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_fully_connected(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_fully_connected_fx8(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_fully_connected_cfg * cfg,
        mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_fully_connected(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_fully_connected_fx16(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_fully_connected_cfg * cfg,
        mli_tensor * out) {
    if (MLI_CHECK(in->el_type      == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(weights->el_type == MLI_EL_FX_16, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type    == MLI_EL_FX_16, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    ret = MLI_CHECK_STATUS(mli_chk_fully_connected(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_fully_connected_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_fully_connected_cfg * cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_fully_connected(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    if (MLI_CHECK(in->el_type          == MLI_EL_SA_8, "Wrong input tensor type") ||
            MLI_CHECK(weights->el_type == MLI_EL_SA_8, "Wrong weights tensor type") ||
            MLI_CHECK(bias->el_type    == MLI_EL_SA_32, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    // Check additional requrements for tensor params.
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in,      kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(out,     kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(weights,  kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(bias,     kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    if (weights->el_params.sa.dim < 0) {
        if (MLI_CHECK(bias->el_params.sa.dim < 0, "Bias tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    }
    else if (MLI_CHECK(weights->el_params.sa.dim == 1, "Weights tensor: per output channels quantization is expected") ||
                MLI_CHECK(bias->el_params.sa.dim == 0, "Bias tensor: per output channels quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;

    if (MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected") ||
            MLI_CHECK(out->el_params.sa.dim < 0, "Output tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;

    ret = MLI_CHECK_STATUS(mli_chk_bias_scale_asym(in, weights, bias), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    return MLI_STATUS_OK;
}

mli_status mli_chk_relu(const mli_tensor * in, const mli_relu_cfg * cfg, mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;
    // Check that tensors are valid
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in),
                      "Memory stride of the innermost dimension should be equal to 1 for the input tensor");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out),
                      "Memory stride of the innermost dimension should be equal to 1 for the output tensor");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check that output contains enough space
    if (MLI_CHECK((mli_prv_count_elem_num(in) * mli_hlp_tensor_element_size(in)) <= out->data.capacity, "Capacity of output tensor is too small"))
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

mli_status mli_chk_relu_sa8(const mli_tensor * in, const mli_relu_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_relu(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_SA_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    // Check additional requrements for tensor params.
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in,      kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    if (MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise (
        const mli_tensor * in1,
        const mli_tensor * in2,
        mli_tensor * out,
        const char *funcname) {
    bool fail = false;
    mli_status stat = MLI_STATUS_OK;

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;
    // Check that tensors are valid
    // One of tensors may be scalar - check through it at first
    if (mli_tensor_is_scalar(in1)){
        fail |= MLI_CHECK2(mli_chk_scalar_tensor(in1) == MLI_STATUS_OK, "bad in1 tensor", funcname);
    }
    if (mli_tensor_is_scalar(in2)){
        fail |= MLI_CHECK2(mli_chk_scalar_tensor(in2) == MLI_STATUS_OK, "bad in2 tensor", funcname);
    }
    fail |= MLI_CHECK2(out != NULL , "Bad Output tensor  pointer", funcname);
    fail |= MLI_CHECK2(check_ptr_not_null(out), "Bad data pointer of output", funcname);
    if (fail) return MLI_STATUS_BAD_TENSOR;

    // One of the tensors must be typical (even with single element)
    fail |= MLI_CHECK2((in1->rank > 0) || (in2->rank > 0),
                       "One of the tensors must be typical (even with single element)", funcname);
    if (fail) return MLI_STATUS_NOT_SUPPORTED;
    if (!(mli_tensor_is_scalar(in1))) {
        stat = MLI_CHECK_STATUS(mli_chk_tensor(in1), "Bad input1 tensor");
    }
    if (!(mli_tensor_is_scalar(in2))) {
        stat = MLI_CHECK_STATUS(mli_chk_tensor(in2), "Bad input2 tensor");
    }
    if (stat != MLI_STATUS_OK) return stat;

    /* Memory stride for inner most dimension of non scalar input must be 1 */
    if (!(mli_tensor_is_scalar(in1))) {
        fail |= MLI_CHECK(check_inner_most_dimension_is_one(in1),
                          "Memory stride for inner most dimension of input must be 1");
    }
    if (!(mli_tensor_is_scalar(in2))) {
        fail |= MLI_CHECK(check_inner_most_dimension_is_one(in2),
                          "Memory stride for inner most dimension of input must be 1");
    }
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Output tensor holds the same element type as the input tensors
    fail |= MLI_CHECK2(out->el_type == in1->el_type,
            "Output tensor holds the same element type as the input tensors", funcname);
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // If both tensors are not scalar their shapes must be exactly the same.
    if (!mli_tensor_is_scalar(in1) && !mli_tensor_is_scalar(in2)) {
        fail |= MLI_CHECK2(in1->rank == in2->rank,
                "If both tensors are not scalar their shapes must be exactly the same.", funcname);
        for (int idx = 0; idx < (int)in1->rank; idx++) {
            fail |= MLI_CHECK2(in1->shape[idx] == in2->shape[idx],
                    "If both tensors are not scalar their shapes must be exactly the same.", funcname);
        }
        if (fail) return MLI_STATUS_SHAPE_MISMATCH;
    }

    fail |= MLI_CHECK(check_quantized_on_tensor(in1, in2), "Tensors should be quantized on tensor level");
    if (fail) return MLI_STATUS_NOT_SUPPORTED;

    // Check that output contains enough space
    int in1_sz = mli_prv_count_elem_num(in1);
    int in2_sz = mli_prv_count_elem_num(in2);
    fail |= MLI_CHECK2((MAX (in1_sz, in2_sz) * mli_hlp_tensor_element_size (in1)) <= out->data.capacity,
            "Capacity of output tensor is too small", funcname);
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;

    return MLI_STATUS_OK;
}

static MLI_FORCE_INLINE bool check_same_data_format(const mli_tensor *in1, const mli_tensor *in2) {
    /* check both have same element type */
    if (in1->el_type != in2->el_type) {
        return false;
    }

    /* check both have same data format */
    if (in1->el_type == MLI_EL_FX_4 || in1->el_type == MLI_EL_FX_8 || in1->el_type == MLI_EL_FX_16) {
        if (in1->el_params.fx.frac_bits != in2->el_params.fx.frac_bits) {
            return false;
        }

    } else if (in1->el_type == MLI_EL_SA_8 || in1->el_type == MLI_EL_SA_32) {
        if ((in1->el_params.sa.dim != in2->el_params.sa.dim) ||
            (in1->el_params.sa.zero_point.mem.i16 != in2->el_params.sa.zero_point.mem.i16) ||
            (in1->el_params.sa.scale.mem.i16 != in2->el_params.sa.scale.mem.i16) ||
            (in1->el_params.sa.scale_frac_bits.mem.i8 != in2->el_params.sa.scale_frac_bits.mem.i8) ||
            (in1->el_params.sa.type != in2->el_params.sa.type)) {
            return false;
        }
    }

    return true;
}

mli_status mli_chk_eltwise_maxmin_fx8 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (!check_same_data_format(in1, in2))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    if (MLI_CHECK(in1->el_type == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise_maxmin_fx16 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (!check_same_data_format(in1, in2))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    if (MLI_CHECK(in1->el_type == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise_maxmin_sa8 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (!check_same_data_format(in1, in2))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    if (MLI_CHECK(in1->el_type == MLI_EL_SA_8, "Wrong input1 tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_SA_8, "Wrong input1 tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    // Check additional requrements for tensor params.
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in1,      kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in2,      kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(out,     kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    if (MLI_CHECK(in1->el_params.sa.dim < 0, "Input1 tensor: Per-tensor quantization is expected") ||
            MLI_CHECK(in2->el_params.sa.dim < 0, "Input1 tensor: Per-tensor quantization is expected") ||
            MLI_CHECK(out->el_params.sa.dim < 0, "Output tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise_fx8 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in1->el_type == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise_fx16 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in1->el_type == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_eltwise_sa8 (const mli_tensor * in1, const mli_tensor * in2, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise(in1, in2, out, __func__), "");
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in1->el_type == MLI_EL_SA_8, "Wrong input tensor type") ||
        MLI_CHECK(in2->el_type == MLI_EL_SA_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    // Check additional requrements for tensor params.
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in1,      kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in2,      kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(out,     kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    if (MLI_CHECK(in1->el_params.sa.dim < 0, "Input1 tensor: Per-tensor quantization is expected") ||
            MLI_CHECK(in2->el_params.sa.dim < 0, "Input1 tensor: Per-tensor quantization is expected") ||
            MLI_CHECK(out->el_params.sa.dim < 0, "Output tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;

    return MLI_STATUS_OK;
}

mli_status mli_chk_basic_activation(const mli_tensor * in, mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;
    // Check that tensors are valid
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in),
                      "Memory stride of the innermost dimension should be equal to 1 for the input tensor");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out),
                      "Memory stride of the innermost dimension should be equal to 1 for the output tensor");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;
    // Check that output contains enough space
    fail |= MLI_CHECK((mli_prv_count_elem_num (in) * mli_hlp_tensor_element_size (in)) <= out->data.capacity,
                      "Capacity of output tensor is too small");
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

    // Check additional requrements for tensor params.
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in,      kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    if (MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    return MLI_STATUS_OK;
}

mli_status mli_chk_softmax_fx8(const mli_tensor * in, const mli_softmax_cfg* cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation(in, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    if (MLI_CHECK(cfg->axis < (int)in->rank, "Wrong axis parameter, axis parameter must be less than in tensor rank"))
        return MLI_STATUS_BAD_FUNC_CFG;
    return MLI_STATUS_OK;
}

mli_status mli_chk_softmax_fx16(const mli_tensor * in, const mli_softmax_cfg* cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation(in, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    if (MLI_CHECK(cfg->axis < (int)in->rank, "Wrong axis parameter, axis parameter must be less than in tensor rank"))
        return MLI_STATUS_BAD_FUNC_CFG;
    return MLI_STATUS_OK;
}

mli_status mli_chk_softmax_sa8(const mli_tensor * in, const mli_softmax_cfg* cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation(in, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_SA_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    if (MLI_CHECK(cfg->axis < (int)in->rank, "Wrong axis parameter, axis parameter must be less than in tensor rank"))
        return MLI_STATUS_BAD_FUNC_CFG;

    // Check additional requrements for tensor params.
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in,      kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    return MLI_STATUS_OK;
}

mli_status mli_chk_l2_normalize_fx16(const mli_tensor * in, const mli_l2_normalize_cfg* cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation(in, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    if (MLI_CHECK(cfg->axis < (int)in->rank, "Wrong axis parameter, axis parameter must be less than in tensor rank"))
        return MLI_STATUS_BAD_FUNC_CFG;
    return MLI_STATUS_OK;
}

mli_status mli_chk_l2_normalize_sa8(const mli_tensor * in, const mli_l2_normalize_cfg* cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation(in, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_SA_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    if (MLI_CHECK(cfg->axis < (int)in->rank, "Wrong axis parameter, axis parameter must be less than in tensor rank"))
        return MLI_STATUS_BAD_FUNC_CFG;

    // Check additional requrements for tensor params.
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in,      kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    return MLI_STATUS_OK;
}

mli_status mli_chk_leaky_relu (const mli_tensor * in, const mli_tensor * slope_coeff, mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;
    // Check that tensors are valid
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in),
                      "Memory stride of the innermost dimension should be equal to 1 for the input tensor");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out),
                      "Memory stride of the innermost dimension should be equal to 1 for the output tensor");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check that slope tensors is valid scalar
    stat = MLI_CHECK_STATUS(mli_chk_scalar_tensor (slope_coeff), "Slope should be scalar tensor");
    if (stat != MLI_STATUS_OK) return stat;

    // Slope must be scalar tensor of the same el_type as input
    fail |= MLI_CHECK(slope_coeff->el_type == in->el_type, "Element type has to be the same");
    if (fail) return MLI_STATUS_TYPE_MISMATCH;

    // Check that output contains enough space
    fail |= MLI_CHECK((mli_prv_count_elem_num (in) * mli_hlp_tensor_element_size (in)) <= out->data.capacity,
                      "Capacity of output tensor is too small");
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

mli_status mli_chk_leaky_relu_sa8 (const mli_tensor * in, const mli_tensor * slope_coeff, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_leaky_relu(in, slope_coeff, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_SA_8, "Wrong input tensor type") ||
        MLI_CHECK(slope_coeff->el_type == MLI_EL_SA_8, "Wrong slope_coeff tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    // Check additional requrements for tensor params.
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in,      kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(out,     kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(slope_coeff, kZeroPointBitsMaxRange - 1), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    if (MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    if (MLI_CHECK(out->el_params.sa.dim < 0, "Output tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    return MLI_STATUS_OK;
}

mli_status mli_chk_prelu (
        const mli_tensor * in,
        const mli_tensor * slope_coeff,
        const mli_prelu_cfg *cfg,
        mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;
    // Check that tensors are valid
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (cfg->axis == -1) {
        // Check that slope tensors is valid scalar
        stat = MLI_CHECK_STATUS(mli_chk_scalar_tensor (slope_coeff), "Slope should be scalar tensor");
    } else {
        stat = MLI_CHECK_STATUS(mli_chk_tensor (slope_coeff), "Bad slope_coeff tensor");
    }
    if (stat != MLI_STATUS_OK) return stat;

    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in),
                      "Memory stride of the innermost dimension should be equal to 1 for the input tensor");
    if (cfg->axis != -1) {
        fail |= MLI_CHECK(check_inner_most_dimension_is_one(slope_coeff),
                      "Memory stride of the innermost dimension should be equal to 1 for the slope_coeff tensor");
        /* slope tensor must be of the same shape of input tensor at axis and others should be 1 */
        for(uint32_t i = 0; i < in->rank; i++) {
            if( i == cfg->axis) {
                fail |= MLI_CHECK(in->shape[i] == slope_coeff->shape[i], "Bad Slope_Coeff Shape");
            } else {
                fail |= MLI_CHECK(slope_coeff->shape[i] == 1, "Bad Slope_Coeff Shape");
            }
        }
    }
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out),
                      "Memory stride of the innermost dimension should be equal to 1 for the output tensor");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Slope must be  of the same el_type as input
    fail |= MLI_CHECK(slope_coeff->el_type == in->el_type, "Element type has to be the same");
    if (fail) return MLI_STATUS_TYPE_MISMATCH;

    // Check that output contains enough space
    fail |= MLI_CHECK((mli_prv_count_elem_num (in) * mli_hlp_tensor_element_size (in)) <= out->data.capacity,
                      "Capacity of output tensor is too small");
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;

    return MLI_STATUS_OK;
}

mli_status mli_chk_prelu_fx8 (
        const mli_tensor * in,
        const mli_tensor * slope_coeff,
        const mli_prelu_cfg *cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_prelu(in, slope_coeff, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_8, "Wrong input tensor type") ||
        MLI_CHECK(slope_coeff->el_type == MLI_EL_FX_8, "Wrong slope_coeff tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_prelu_fx16 (
        const mli_tensor * in,
        const mli_tensor * slope_coeff,
        const mli_prelu_cfg *cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_prelu(in, slope_coeff, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(slope_coeff->el_type == MLI_EL_FX_16, "Wrong slope_coeff tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

mli_status mli_chk_prelu_sa8 (
        const mli_tensor * in,
        const mli_tensor * slope_coeff,
        const mli_prelu_cfg *cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_prelu(in, slope_coeff, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_SA_8, "Wrong input tensor type") ||
        MLI_CHECK(slope_coeff->el_type == MLI_EL_SA_8, "Wrong slope_coeff tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    // Check additional requrements for tensor params.
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in,      kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(out,     kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(slope_coeff, kZeroPointBitsMaxRange - 1), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    if (MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    if (MLI_CHECK(out->el_params.sa.dim < 0, "Output tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    return MLI_STATUS_OK;
}

mli_status mli_chk_rnn_dense (
        const mli_tensor **in,
        const mli_tensor **weights,
        const mli_tensor *bias,
        const mli_rnn_dense_cfg *cfg,
        mli_tensor *out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_CONV_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;

    // Check config is valid
    fail |= MLI_CHECK(cfg != NULL , "Bad cfg pointer");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    const int inputs_num = cfg->inputs_num;
    fail |= MLI_CHECK(inputs_num > 0, "number of inputs should be > 0");
    fail |= MLI_CHECK(inputs_num <= MLI_RNN_MAX_INPUT, "number of inputs should not exceed MLI_RNN_MAX_INPUT");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    for (int idx = 0; idx < inputs_num; idx++) {
        // Check that input and weights are valid
        stat = MLI_CHECK_STATUS(mli_chk_tensor (in[idx]), "Bad input tensor");
        if (stat != MLI_STATUS_OK) return stat;
        stat = MLI_CHECK_STATUS(mli_chk_tensor (weights[idx]), "Bad weights tensor");
        if (stat != MLI_STATUS_OK) return stat;

        fail |= MLI_CHECK(weights[idx]->rank == 2, "weights should have rank 2 in RNN general case");
        // Check that number of input and weights is equal
        fail |= MLI_CHECK(mli_prv_count_elem_num(in[idx]) == weights[idx]->shape[0], "number of input and weights tensors should be equal");
        // Check that all weights have the same number of output neurons
        fail |= MLI_CHECK(weights[idx]->shape[1] == weights[0]->shape[1], "number of outputs should be the same for all weights tensors");
        if (fail) return MLI_STATUS_SHAPE_MISMATCH;

        fail |= MLI_CHECK(check_inner_most_dimension_is_one(in[idx]), "Memory stride for inner most dimension of input must be 1");
        fail |= MLI_CHECK(check_inner_most_dimension_is_one(weights[idx]), "Memory stride for inner most dimension of weights must be 1");
        if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;
    }

    // Check bias
    stat = MLI_CHECK_STATUS(mli_chk_tensor (bias), "Bad bias tensor");
    if (stat != MLI_STATUS_OK) return stat;

    fail |= MLI_CHECK(bias->rank == 1, "bias should have rank 1 in RNN general case");
    fail |= MLI_CHECK(weights[0]->shape[1] == bias->shape[0], "shape mismatch weights and bias in RNN general case");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    // Check output
    uint32_t out_elements = mli_prv_count_elem_num (bias);
    fail |= MLI_CHECK(out != NULL, "Bad Output tensor  pointer");
    fail |= MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output");
    fail |= MLI_CHECK((out_elements * mli_hlp_tensor_element_size (in[0])) <= out->data.capacity,
                      "capacity of output tensor is too small");
    if (fail) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(bias), "Memory stride for inner most dimension of bias must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out), "Memory stride for inner most dimension of output must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    return MLI_STATUS_OK;
}

mli_status mli_chk_rnn_dense_fx16 (
        const mli_tensor **in,
        const mli_tensor **weights,
        const mli_tensor *bias,
        const mli_rnn_dense_cfg *cfg,
        mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_rnn_dense(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    bool fail = false;
    const int inputs_num = cfg->inputs_num;
    for (int idx = 0; idx < inputs_num; idx++) {
        fail |= MLI_CHECK(in[idx]->el_type == MLI_EL_FX_16, "Wrong input tensor type");
        fail |= MLI_CHECK(weights[idx]->el_type == MLI_EL_FX_16, "Wrong weights tensor type");
    }

    fail |= MLI_CHECK(bias->el_type == MLI_EL_FX_16, "Wrong bias tensor type");
    if (fail) return MLI_STATUS_TYPE_MISMATCH;

    ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in[0], weights[0], bias), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    return MLI_STATUS_OK;
}

mli_status mli_chk_rnn_dense_fx16_fx8_fx8 (
        const mli_tensor **in,
        const mli_tensor **weights,
        const mli_tensor *bias,
        const mli_rnn_dense_cfg *cfg,
        mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_rnn_dense(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    bool fail = false;
    const int inputs_num = cfg->inputs_num;
    for (int idx = 0; idx < inputs_num; idx++) {
        fail |= MLI_CHECK(in[idx]->el_type == MLI_EL_FX_16, "Wrong input tensor type");
        fail |= MLI_CHECK(weights[idx]->el_type == MLI_EL_FX_8, "Wrong weights tensor type");
    }

    fail |= MLI_CHECK(bias->el_type == MLI_EL_FX_8, "Wrong bias tensor type");
    if (fail) return MLI_STATUS_TYPE_MISMATCH;

    ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in[0], weights[0], bias), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    return MLI_STATUS_OK;
}

mli_status mli_chk_rnn_dense_sa8_sa8_sa32(
        const mli_tensor **in,
        const mli_tensor **weights,
        const mli_tensor *bias,
        const mli_rnn_dense_cfg *cfg,
        mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_rnn_dense(in, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    bool fail = false;
    const int inputs_num = cfg->inputs_num;
    for (int idx = 0; idx < inputs_num; idx++) {
        fail |= MLI_CHECK(in[idx]->el_type == MLI_EL_SA_8, "Wrong input tensor type");
        fail |= MLI_CHECK(weights[idx]->el_type == MLI_EL_SA_8, "Wrong weights tensor type");
    }
    fail |= MLI_CHECK(bias->el_type == MLI_EL_SA_32, "Wrong bias tensor type");
    if (fail) return MLI_STATUS_TYPE_MISMATCH;

    // Check additional requrements for tensor params.
    for (int idx = 0; idx < inputs_num; idx++) {
        ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in[idx],       kZeroPointBitsByteRange), __func__);
        if (ret != MLI_STATUS_OK) return ret;
        ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(weights[idx],  kZeroPointBitsZero), __func__);
        if (ret != MLI_STATUS_OK) return ret;

        fail |= MLI_CHECK(in[idx]->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected");
        fail |= MLI_CHECK(weights[idx]->el_params.sa.dim < 0, "Weights tensor: Per-tensor quantization is expected");
    }

    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(out,     kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(bias,    kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    fail |= MLI_CHECK(out->el_params.sa.dim < 0, "Output tensor: Per-tensor quantization is expected");
    fail |= MLI_CHECK(bias->el_params.sa.dim < 0, "Bias tensor: Per-tensor quantization is expected");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    ret = MLI_CHECK_STATUS(mli_chk_bias_scale_asym(in[0], weights[0], bias), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    return MLI_STATUS_OK;
}

mli_status mli_chk_lstm_cell (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * cell,
        mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_CONV_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;

    // Check tanh_lut
    stat = MLI_CHECK_STATUS(mli_chk_lut(tanh_lut, tanh_lut_fx16.data.capacity), "Tanh LUT error");
    if (stat != MLI_STATUS_OK) return stat;

    // Check sigm_lut
    stat = MLI_CHECK_STATUS(mli_chk_lut(sigm_lut, sigmoid_lut_fx16.data.capacity), "Sigm LUT error");
    if (stat != MLI_STATUS_OK) return stat;

    // Check that tensors are valid
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (prev_out), "Bad prev_out tensor");
    if (stat != MLI_STATUS_OK) return stat;

    fail |= MLI_CHECK(in->rank == 2, "Wrong input rank");
    fail |= MLI_CHECK(prev_out->rank == 1, "Wrong prev_out rank");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in), "Memory stride for inner most dimension of input must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(prev_out), "Memory stride for inner most dimension of input must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    uint32_t in_elements = mli_prv_count_elem_num_part(in, 1);
    uint32_t out_elements = mli_prv_count_elem_num (prev_out);
    uint32_t out_seq_len = (cfg->results == RNN_OUT_ALL) ? in->shape[0] : 1;

    // Check weights
    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights_in), "Bad weights_in tensor");
    if (stat != MLI_STATUS_OK) return stat;

    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights_out), "Bad weights_out tensor");
    if (stat != MLI_STATUS_OK) return stat;

    fail |= MLI_CHECK(weights_in->rank == 3, "Wrong weights_in rank");
    fail |= MLI_CHECK(weights_out->rank == 3, "Wrong weights_out rank");
    fail |= MLI_CHECK(weights_in->shape[0] == 4 && weights_in->shape[1] == in_elements && weights_in->shape[2] == out_elements, "Wrong weights_in shape");
    fail |= MLI_CHECK(weights_out->shape[0] == 4 && weights_out->shape[1] == out_elements && weights_out->shape[2] == out_elements, "Wrong weights_out shape");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(weights_in), "Memory stride for inner most dimension of weights must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(weights_out), "Memory stride for inner most dimension of weights must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check bias
    stat = MLI_CHECK_STATUS(mli_chk_tensor (bias), "Bad bias tensor");
    if (stat != MLI_STATUS_OK) return stat;

    fail |= MLI_CHECK(bias->rank == 2, "Wrong bias rank");
    fail |= MLI_CHECK(bias->shape[0] == 4 && bias->shape[1] == out_elements, "Wrong bias shape");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    // Check cell
    stat = MLI_CHECK_STATUS(mli_chk_tensor (cell), "Bad cell tensor");
    if (stat != MLI_STATUS_OK) return stat;

    fail |= MLI_CHECK(cell->rank == 1, "Wrong cell rank");
    fail |= MLI_CHECK(cell->shape[0] == out_elements, "Wrong cell shape");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    // Check output
    fail |= MLI_CHECK(out != NULL, "Bad Output tensor  pointer");
    fail |= MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output");
    fail |= MLI_CHECK((out_seq_len * out_elements * mli_hlp_tensor_element_size (in)) <= out->data.capacity,
                      "capacity of output tensor is too small");
    if (fail) return MLI_STATUS_BAD_TENSOR;

    // Check scratch data
    fail |= MLI_CHECK(check_ptr_not_null(cfg->scratch_data, in->el_type), "Bad data pointer of scratch data");
    fail |= MLI_CHECK((4 * out_elements * mli_hlp_tensor_element_size (in)) <= cfg->scratch_data.capacity,
                      "capacity of scratch data is too small");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(cell), "Memory stride for inner most dimension of cell must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(bias), "Memory stride for inner most dimension of bias must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out), "Memory stride for inner most dimension of output must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    return MLI_STATUS_OK;
}

mli_status mli_chk_lstm_cell_fx16 (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * cell,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_lstm_cell(in, prev_out, weights_in, weights_out, bias, tanh_lut, sigm_lut,
                                                        cfg, cell, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type       == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(prev_out->el_type == MLI_EL_FX_16, "Wrong prev_out tensor type") ||
        MLI_CHECK(weights_in->el_type  == MLI_EL_FX_16, "Wrong weights_in tensor type") ||
        MLI_CHECK(weights_out->el_type  == MLI_EL_FX_16, "Wrong weights_out tensor type") ||
        MLI_CHECK(cell->el_type  == MLI_EL_FX_16, "Wrong cell tensor type") ||
        MLI_CHECK(bias->el_type     == MLI_EL_FX_16, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights_in, bias), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    return MLI_STATUS_OK;
}

mli_status mli_chk_lstm_cell_fx16_fx8_fx8 (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * cell,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_lstm_cell(in, prev_out, weights_in, weights_out, bias, tanh_lut, sigm_lut,
                                                        cfg, cell, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type       == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(prev_out->el_type == MLI_EL_FX_16, "Wrong prev_out tensor type") ||
        MLI_CHECK(weights_in->el_type  == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(weights_out->el_type  == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(cell->el_type == MLI_EL_FX_16, "Wrong cell tensor type") ||
        MLI_CHECK(bias->el_type     == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights_in, bias), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    return MLI_STATUS_OK;
}

mli_status mli_chk_lstm_cell_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * cell,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_lstm_cell(in, prev_out, weights_in, weights_out, bias, tanh_lut, sigm_lut,
                                                        cfg, cell, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    bool fail = false;

    if (MLI_CHECK(in->el_type       == MLI_EL_SA_8, "Wrong input tensor type") ||
            MLI_CHECK(prev_out->el_type == MLI_EL_SA_8, "Wrong prev_out tensor type") ||
            MLI_CHECK(weights_in->el_type  == MLI_EL_SA_8, "Wrong weights tensor type") ||
            MLI_CHECK(weights_out->el_type  == MLI_EL_SA_8, "Wrong weights tensor type") ||
            MLI_CHECK(cell->el_type == MLI_EL_SA_8, "Wrong cell tensor type") ||
            MLI_CHECK(bias->el_type     == MLI_EL_SA_32, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    fail |= MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected");
    fail |= MLI_CHECK(prev_out->el_params.sa.dim < 0, "Prev out tensor: Per-tensor quantization is expected");
    fail |= MLI_CHECK(cell->el_params.sa.dim < 0, "Cell tensor: Per-tensor quantization is expected");

    fail |= MLI_CHECK(weights_in->el_params.sa.dim == 0, "Weights_in tensor: Per-axis quantization is expected");
    fail |= MLI_CHECK(weights_out->el_params.sa.dim == 0, "Weights_out tensor: Per-axis quantization is expected");
    fail |= MLI_CHECK(bias->el_params.sa.dim == 0, "Bias tensor: Per-axis quantization is expected");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in,        kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(prev_out,  kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(cell,      kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(weights_in, kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(weights_out, kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(bias,       kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    ret = MLI_CHECK_STATUS(mli_chk_bias_scale_asym(in, weights_in, bias), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    return MLI_STATUS_OK;
}

mli_status mli_chk_gru_cell (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_CONV_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;

    // Check tanh_lut
    stat = MLI_CHECK_STATUS(mli_chk_lut(tanh_lut, tanh_lut_fx16.data.capacity), "Tanh LUT error");
    if (stat != MLI_STATUS_OK) return stat;

    // Check sigm_lut
    stat = MLI_CHECK_STATUS(mli_chk_lut(sigm_lut, sigmoid_lut_fx16.data.capacity), "Sigm LUT error");
    if (stat != MLI_STATUS_OK) return stat;

    // Check that tensors are valid
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    stat = MLI_CHECK_STATUS(mli_chk_tensor (prev_out), "Bad prev_out tensor");
    if (stat != MLI_STATUS_OK) return stat;

    fail |= MLI_CHECK(in->rank == 2, "Wrong input rank");
    fail |= MLI_CHECK(prev_out->rank == 1, "Wrong prev_out rank");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in), "Memory stride for inner most dimension of input must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(prev_out), "Memory stride for inner most dimension of input must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    uint32_t in_elements = mli_prv_count_elem_num_part(in, 1);
    uint32_t out_elements = mli_prv_count_elem_num (prev_out);
    uint32_t out_seq_len = (cfg->results == RNN_OUT_ALL) ? in->shape[0] : 1;

    // Check weights
    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights_in), "Bad weights_in tensor");
    if (stat != MLI_STATUS_OK) return stat;

    stat = MLI_CHECK_STATUS(mli_chk_tensor (weights_out), "Bad weights_out tensor");
    if (stat != MLI_STATUS_OK) return stat;

    fail |= MLI_CHECK(weights_in->rank == 3, "Wrong weights_in rank");
    fail |= MLI_CHECK(weights_out->rank == 3, "Wrong weights_out rank");
    fail |= MLI_CHECK(weights_in->shape[0] == 3 && weights_in->shape[1] == in_elements && weights_in->shape[2] == out_elements, "Wrong weights_in shape");
    fail |= MLI_CHECK(weights_out->shape[0] == 3 && weights_out->shape[1] == out_elements && weights_out->shape[2] == out_elements, "Wrong weights_out shape");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(weights_in), "Memory stride for inner most dimension of weights must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(weights_out), "Memory stride for inner most dimension of weights must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check bias
    stat = MLI_CHECK_STATUS(mli_chk_tensor (bias), "Bad bias tensor");
    if (stat != MLI_STATUS_OK) return stat;

    fail |= MLI_CHECK(bias->rank == 2, "Wrong bias rank");
    fail |= MLI_CHECK(bias->shape[0] == 3 && bias->shape[1] == out_elements, "Wrong bias shape");
    if (fail) return MLI_STATUS_SHAPE_MISMATCH;

    // Check output
    fail |= MLI_CHECK(out != NULL, "Bad Output tensor  pointer");
    fail |= MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output");
    fail |= MLI_CHECK((out_seq_len * out_elements * mli_hlp_tensor_element_size (in)) <= out->data.capacity,
                      "capacity of output tensor is too small");
    if (fail) return MLI_STATUS_BAD_TENSOR;

    // Check scratch data
    fail |= MLI_CHECK(check_ptr_not_null(cfg->scratch_data, in->el_type), "Bad data pointer of scratch data");
    fail |= MLI_CHECK((3 * out_elements * mli_hlp_tensor_element_size (in)) <= cfg->scratch_data.capacity,
                      "capacity of scratch data is too small");
    if (fail) return MLI_STATUS_BAD_FUNC_CFG;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(bias), "Memory stride for inner most dimension of bias must be 1");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out), "Memory stride for inner most dimension of output must be 1");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    return MLI_STATUS_OK;
}

mli_status mli_chk_gru_cell_fx16 (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_gru_cell(in, prev_out, weights_in, weights_out, bias, tanh_lut, sigm_lut,
                                                        cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type       == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(prev_out->el_type == MLI_EL_FX_16, "Wrong prev_out tensor type") ||
        MLI_CHECK(weights_in->el_type  == MLI_EL_FX_16, "Wrong weights_in tensor type") ||
        MLI_CHECK(weights_out->el_type  == MLI_EL_FX_16, "Wrong weights_out tensor type") ||
        MLI_CHECK(bias->el_type     == MLI_EL_FX_16, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights_in, bias), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    return MLI_STATUS_OK;
}

mli_status mli_chk_gru_cell_fx16_fx8_fx8 (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_gru_cell(in, prev_out, weights_in, weights_out, bias, tanh_lut, sigm_lut,
                                                        cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type       == MLI_EL_FX_16, "Wrong input tensor type") ||
        MLI_CHECK(prev_out->el_type == MLI_EL_FX_16, "Wrong prev_out tensor type") ||
        MLI_CHECK(weights_in->el_type  == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(weights_out->el_type  == MLI_EL_FX_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type     == MLI_EL_FX_8, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    ret = MLI_CHECK_STATUS(mli_chk_bias_frac_fx(in, weights_in, bias), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    return MLI_STATUS_OK;
}

mli_status mli_chk_gru_cell_sa8_sa8_sa32(
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out) {

    mli_status ret = MLI_CHECK_STATUS(mli_chk_gru_cell(in, prev_out, weights_in, weights_out, bias, tanh_lut, sigm_lut,
                                                        cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    bool fail = false;

    if (MLI_CHECK(in->el_type       == MLI_EL_SA_8, "Wrong input tensor type") ||
        MLI_CHECK(prev_out->el_type == MLI_EL_SA_8, "Wrong prev_out tensor type") ||
        MLI_CHECK(weights_in->el_type  == MLI_EL_SA_8, "Wrong weights tensor type") ||
        MLI_CHECK(weights_out->el_type  == MLI_EL_SA_8, "Wrong weights tensor type") ||
        MLI_CHECK(bias->el_type     == MLI_EL_SA_32, "Wrong bias tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;

    fail |= MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected");
    fail |= MLI_CHECK(prev_out->el_params.sa.dim < 0, "Prev out tensor: Per-tensor quantization is expected");

    if (weights_in->el_params.sa.dim == 0) {
        if (MLI_CHECK(weights_out->el_params.sa.dim == 0, "Weights_out tensor: per first dimension quantization is expected")) {
            if (MLI_CHECK(bias->el_params.sa.dim == 0, "Bias tensor: per first dimension quantization is expected (similar to weights)"))
                return MLI_STATUS_INCOMPATEBLE_TENSORS;
        }
    } else {
        if (MLI_CHECK(weights_in->el_params.sa.dim == -1, "Weights_in tensor: per tensor quantization is expected")) {
            if (MLI_CHECK(weights_out->el_params.sa.dim == -1, "Weights_out tensor: per tensor quantization is expected")) {
                if (MLI_CHECK(bias->el_params.sa.dim == -1, "Bias tensor: per tensor quantization is expected (similar to weights)"))
                    return MLI_STATUS_INCOMPATEBLE_TENSORS;
            }
        }
    }

    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(in,        kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(prev_out,  kZeroPointBitsByteRange), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(weights_in, kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(weights_out, kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    ret = MLI_CHECK_STATUS(mli_chk_tensor_quant_params(bias,       kZeroPointBitsZero), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    ret = MLI_CHECK_STATUS(mli_chk_bias_scale_asym(in, weights_in, bias), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    return MLI_STATUS_OK;
}

mli_status mli_chk_concat (const mli_tensor ** inputs, const mli_concat_cfg * cfg, mli_tensor * out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;
    // Check first input and output tensors
    if (MLI_CHECK(inputs != NULL , "Bad inputs tensor array")) return MLI_STATUS_BAD_TENSOR;

    stat = MLI_CHECK_STATUS(mli_chk_tensor (inputs[0]), "Bad inputs[0] tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;
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

        for (int dim_idx = 0; dim_idx < (int)anchor_tsr->rank; dim_idx++) {
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

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;
    // Check that in tensor is valid and out provides valid pointers
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

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

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;
    // Check that in tensor is valid and out provides valid pointers
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

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

    stat = MLI_CHECK_STATUS(mli_mem_chk(out, MLI_OUT_PTR_IS_XY), "Memory check error");
    if (stat != MLI_STATUS_OK) return stat;
    // Check that in tensor is valid and out provides valid pointers
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    // Check config structure
    if (MLI_CHECK(cfg != NULL , "Bad cfg pointer")) return MLI_STATUS_BAD_FUNC_CFG;

    for (int idx = 0; idx < (int)in->rank; idx++) {
        if (MLI_CHECK(cfg->perm_dim[idx] < in->rank, "Rank mismatch"))
            return MLI_STATUS_BAD_FUNC_CFG;

        // Each permute dimension must be unique
        for (int jdx = idx + 1; jdx < (int)in->rank; jdx++)
            if (MLI_CHECK(cfg->perm_dim[idx] != cfg->perm_dim[jdx], "Each permute dimension must be unique"))
                return MLI_STATUS_BAD_FUNC_CFG;
    }

    // Check that output contains enough space
    fail |= MLI_CHECK((mli_prv_count_elem_num (in) * mli_hlp_tensor_element_size (in)) <= out->data.capacity,
                      "Capacity of output tensor is too small");
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

mli_status mli_chk_permute_sa8 (const mli_tensor * in, const mli_permute_cfg * cfg, mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_permute(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;
    if (MLI_CHECK(in->el_type == MLI_EL_SA_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    if (in->el_params.sa.dim >= 0) {
        bool fail = false;
        if (out->el_params.sa.zero_point.mem.pi16 == in->el_params.sa.zero_point.mem.pi16)
            fail |= MLI_CHECK((out->el_params.sa.zero_point.mem.pi16 == in->el_params.sa.zero_point.mem.pi16) \
                    && (out->el_params.sa.scale.mem.pi16 == in->el_params.sa.scale.mem.pi16) \
                    && (out->el_params.sa.scale_frac_bits.mem.pi8 == in->el_params.sa.scale_frac_bits.mem.pi8),
                    "El_params data for out tensor wasn`t initialized in a consistent way");
        if (out->el_params.sa.zero_point.mem.pi16 != in->el_params.sa.zero_point.mem.pi16)
            fail |= MLI_CHECK((out->el_params.sa.zero_point.mem.pi16 != in->el_params.sa.zero_point.mem.pi16) \
                    && (out->el_params.sa.scale.mem.pi16 != in->el_params.sa.scale.mem.pi16) \
                    && (out->el_params.sa.scale_frac_bits.mem.pi8 != in->el_params.sa.scale_frac_bits.mem.pi8),
                    "El_params data for out tensor wasn`t initialized in a consistent way");
        if (out->el_params.sa.zero_point.mem.pi16 == nullptr)
            fail |= MLI_CHECK((out->el_params.sa.zero_point.mem.pi16 == nullptr) \
                    && (out->el_params.sa.scale.mem.pi16 == nullptr) \
                    && (out->el_params.sa.scale_frac_bits.mem.pi8 == nullptr),
                    "El_params data for out tensor wasn`t initialized in a consistent way");

        if (!fail && out->el_params.sa.zero_point.mem.pi16 != in->el_params.sa.zero_point.mem.pi16 \
                && out->el_params.sa.zero_point.mem.pi16 != nullptr)
            fail |= MLI_CHECK(out->el_params.sa.zero_point.capacity >= in->el_params.sa.zero_point.capacity \
                    && out->el_params.sa.scale.capacity >= in->el_params.sa.scale.capacity \
                    && out->el_params.sa.scale_frac_bits.capacity >= in->el_params.sa.scale_frac_bits.capacity,
                    "Not enough memory allocated for quantization parameters");
        if (fail) return MLI_STATUS_SPEC_PARAM_MISMATCH;
    }

    return MLI_STATUS_OK;
}

mli_status mli_chk_count_elem_num(const mli_tensor *in, uint32_t start_dim) {
    if (MLI_CHECK(in->rank <= MLI_MAX_RANK, "rank should not exceed MAX_RANK"))
        return MLI_STATUS_BAD_TENSOR;
    if (in->rank != 0) {
        if (MLI_CHECK(start_dim < in->rank, "start_dim should be smaller than rank"))
            return MLI_STATUS_BAD_FUNC_CFG;
    }
    return MLI_STATUS_OK;
}

mli_status mli_chk_convert_tensor(const mli_tensor *in, mli_tensor *out) {
    mli_status stat = MLI_STATUS_OK;
    bool fail = false;

    // Check that in tensor is valid and out provides valid pointers
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(check_ptr_not_null(in), "Bad data pointer of input")) return MLI_STATUS_BAD_TENSOR;

    // Rank and shapes are copied from input to output tensor, so they don't need to be checked.
    // But mem_strides of output tensor still must be valid.
    if (MLI_CHECK(out != NULL , "Bad Output tensor pointer")) return MLI_STATUS_BAD_TENSOR;
    stat = MLI_CHECK_STATUS(check_tensor_private(in->shape, out->mem_stride, in->rank, out->data.capacity, mli_hlp_tensor_element_size(out)), "Bad output tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    fail |= MLI_CHECK(check_inner_most_dimension_is_one(in), "mem_stride of the innermost dimension for input tensor must be not more than 1.");
    fail |= MLI_CHECK(check_inner_most_dimension_is_one(out), "mem_stride of the innermost dimension for output tensor must be not more than 1.");

    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check when output data points to the same memory as input, they have an equal container size.
    if (mli_hlp_tensor_data_ptr_cmp(in, out))
        fail |= MLI_CHECK(mli_hlp_tensor_element_size(in) == mli_hlp_tensor_element_size(out) && check_mem_stride_matches(in, out),
                          "In-place computation is permitted only when tensors have the same container size and mem_stride.");
    if (fail) return MLI_STATUS_INCOMPATEBLE_TENSORS;

    // Check that output contains enough space
    const unsigned out_elements = mli_prv_count_elem_num(in);
    fail |= MLI_CHECK((out_elements * mli_hlp_tensor_element_size(out)) <= out->data.capacity,
        "capacity of output tensor is too small");
    if (fail) return MLI_STATUS_NOT_ENGH_MEM;

    if ((in->el_type == MLI_EL_SA_8 || in->el_type == MLI_EL_SA_32) &&
        (out->el_type == MLI_EL_SA_8 || out->el_type == MLI_EL_SA_32))
        fail |= MLI_CHECK(in->el_params.sa.dim == out->el_params.sa.dim || in->el_params.sa.dim < 0 || out->el_params.sa.dim < 0, "Scale axises doesn't match.");

    if (fail) return MLI_STATUS_SIZE_MISMATCH;

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
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in, /*check_bank=*/false), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;

    if (MLI_CHECK(cfg != NULL , "Bad cfg pointer")) return MLI_STATUS_BAD_FUNC_CFG;
    if (MLI_CHECK(cfg->sub_tensor_rank <= in->rank, "incorrect number of coordinates"))
        return MLI_STATUS_BAD_FUNC_CFG;

    for (int i = 0; i < (int)in->rank; i++) {
        if (MLI_CHECK(cfg->offset[i] < in->shape[i], "bad config"))
            return MLI_STATUS_BAD_FUNC_CFG;
        if (MLI_CHECK(cfg->offset[i] + cfg->size[i] <= in->shape[i], "bad config"))
            return MLI_STATUS_BAD_FUNC_CFG;
    }

    return MLI_STATUS_OK;
}

mli_status mli_chk_data_movement(const mli_tensor *in, const mli_mov_cfg_t *cfg, mli_tensor *out) {
    mli_status stat = MLI_STATUS_OK;
    // For data movement the tensor data can be allocated in external memory.
    // Check that in tensor is valid and out provides valid pointers
    stat = MLI_CHECK_STATUS(mli_chk_tensor (in, false), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;

    if ((in->el_type == MLI_EL_SA_8 || in->el_type == MLI_EL_SA_32) && (in->el_params.sa.dim != -1)) {
    	bool fail = false;
    	if (out->el_params.sa.zero_point.mem.pi16 == in->el_params.sa.zero_point.mem.pi16)
    	    fail |= MLI_CHECK((out->el_params.sa.scale.mem.pi16 == in->el_params.sa.scale.mem.pi16)
    	            && (out->el_params.sa.scale_frac_bits.mem.pi8 == in->el_params.sa.scale_frac_bits.mem.pi8),
    	            "El_params data for out tensor wasn`t initialized in a consistent way");

    	if (out->el_params.sa.zero_point.mem.pi16 != in->el_params.sa.zero_point.mem.pi16)
    	    fail |= MLI_CHECK((out->el_params.sa.scale.mem.pi16 != in->el_params.sa.scale.mem.pi16)
    	            && (out->el_params.sa.scale_frac_bits.mem.pi8 != in->el_params.sa.scale_frac_bits.mem.pi8),
    	            "El_params data for out tensor wasn`t initialized in a consistent way");

    	if (out->el_params.sa.zero_point.mem.pi16 == nullptr)
    	    fail |= MLI_CHECK((out->el_params.sa.scale.mem.pi16 == nullptr)
    	            && (out->el_params.sa.scale_frac_bits.mem.pi8 == nullptr),
    	            "El_params data for out tensor wasn`t initialized in a consistent way");

    	if (!fail && out->el_params.sa.zero_point.mem.pi16 != in->el_params.sa.zero_point.mem.pi16
    	          && out->el_params.sa.zero_point.mem.pi16 != nullptr)
    	    fail |= MLI_CHECK(out->el_params.sa.zero_point.capacity >= in->el_params.sa.zero_point.capacity
    	            && out->el_params.sa.scale.capacity >= in->el_params.sa.scale.capacity
    	            && out->el_params.sa.scale_frac_bits.capacity >= in->el_params.sa.scale_frac_bits.capacity,
    	            "Not enough memory allocated for quantization parameters");

    	if (fail) return MLI_STATUS_SPEC_PARAM_MISMATCH;

    }

    if (MLI_CHECK(out != NULL , "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    //check that the configurations are valid
    if (MLI_CHECK(cfg != NULL , "Bad cfg pointer")) return MLI_STATUS_BAD_FUNC_CFG;
    for (uint32_t i=0; i < in->rank; i++) {
    	if (MLI_CHECK((cfg->size[i] + cfg->offset[i]) <= in->shape[i],"Bad configurations"))
		    return MLI_STATUS_BAD_FUNC_CFG;
        if (MLI_CHECK(cfg->dst_offset[i] <= out->shape[i],"Bad configurations"))
        	return MLI_STATUS_BAD_FUNC_CFG;
        //check that in case cfg->size is provided the post padding should be 0 in case we will slice from the tensor
        if (MLI_CHECK(((cfg->size[i] == 0) || ((cfg->size[i] + cfg->offset[i]) == in->shape[i]) || (cfg->padding_post[i] == 0)),
        		"Bad Configurations"))
        	return MLI_STATUS_BAD_FUNC_CFG;
    }

    //check that input and output are not overlapped
    if (MLI_CHECK((out->data.mem.i32 >= (in->data.mem.i32 + (int)(in->data.capacity))) ||
    		(in->data.mem.i32 >= (out->data.mem.i32 + (int)(out->data.capacity))),"in and out buffer are overlapped")) {
    	return MLI_STATUS_INCOMPATEBLE_TENSORS;
    }

    return MLI_STATUS_OK;
}

mli_status mli_chk_data_movement_dst_tensor(const mli_tensor *t) {
    mli_status stat = MLI_STATUS_OK;
    // For data movement the tensor data can be allocated in external memory.
    // Check that in tensor is valid and out provides valid pointers
    stat = MLI_CHECK_STATUS(mli_chk_tensor (t, false), "Bad input tensor");
    return stat;

}

mli_status mli_chk_argmax(const mli_tensor *in, const mli_argmax_cfg *cfg, mli_tensor *out) {
    mli_status stat = MLI_STATUS_OK;

    // Check that in tensor is valid and out provides valid pointers
    stat = MLI_CHECK_STATUS(mli_chk_tensor(in), "Bad input tensor");
    if (stat != MLI_STATUS_OK) return stat;
    if (MLI_CHECK(check_ptr_not_null(in), "Bad data pointer of input")) return MLI_STATUS_BAD_TENSOR;

    if (MLI_CHECK(out != NULL, "Bad Output tensor  pointer")) return MLI_STATUS_BAD_TENSOR;
    if (MLI_CHECK(check_ptr_not_null(out), "Bad data pointer of output")) return MLI_STATUS_BAD_TENSOR;

    // Check if cfg is valid
    if (MLI_CHECK(cfg != NULL, "Bad cfg pointer")) return MLI_STATUS_BAD_FUNC_CFG;
    if (MLI_CHECK(cfg->axis <= (int32_t)in->rank, "Incorrect axis")) return MLI_STATUS_BAD_FUNC_CFG;

    if (MLI_CHECK(check_inner_most_dimension_is_one(in), "mem_stride of the innermost dimension for input tensor must be not more than 1."))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;

    uint32_t dim_size = 1;
    if (cfg->axis >= 0)
        dim_size = in->shape[cfg->axis];
    if (MLI_CHECK(out->data.capacity == cfg->topk * dim_size * sizeof(int32_t), "Insufficient output buffer."))
        return MLI_STATUS_NOT_ENGH_MEM;

    if (in->el_type == MLI_EL_SA_8 || in->el_type == MLI_EL_SA_32)
        if (MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor must be quantized on the tensor level."))
            return MLI_STATUS_INCOMPATEBLE_TENSORS;

    return MLI_STATUS_OK;
}

mli_status mli_chk_argmax_sa8(const mli_tensor *in, const mli_argmax_cfg *cfg, mli_tensor *out) {
    mli_status ret = mli_chk_argmax(in, cfg, out);
    if (ret != MLI_STATUS_OK)
        return ret;

    if (MLI_CHECK(in->el_type == MLI_EL_SA_8, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    if (MLI_CHECK(in->el_params.sa.dim < 0, "Input tensor: Per-tensor quantization is expected"))
        return MLI_STATUS_INCOMPATEBLE_TENSORS;
    return MLI_STATUS_OK;
}

mli_status mli_chk_argmax_fx16(const mli_tensor *in, const mli_argmax_cfg *cfg, mli_tensor *out) {
    mli_status ret = mli_chk_argmax(in, cfg, out);
    if (ret != MLI_STATUS_OK)
        return ret;

    if (MLI_CHECK(in->el_type == MLI_EL_FX_16, "Wrong input tensor type"))
        return MLI_STATUS_TYPE_MISMATCH;
    return MLI_STATUS_OK;
}

#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
}
#endif
