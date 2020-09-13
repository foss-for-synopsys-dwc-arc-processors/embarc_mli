/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_math.h"
#include "mli_helpers_api.h"
#include "mli_prv_tensor.h"

#pragma MLI_CODE_SECTION_START(".mli_lib")

#ifdef __cplusplus
extern "C" {
#endif

static void convert_tensor_fx8_to_fx8(
        const MLI_PTR(int8_t) __restrict in, 
        MLI_PTR(int8_t) __restrict out, 
        int count, 
        int shift_right) {
    __builtin_assume(count > 0);
    for (int i = 0; i < count; i++)
        out[i] = mli_math_cast_fx<int16_t, int8_t>((int16_t)in[i], shift_right);
}

static void convert_tensor_fx16_to_fx16(
        const MLI_PTR(int16_t) __restrict in, 
        MLI_PTR(int16_t) __restrict out, 
        int count, 
        int shift_right) {
    __builtin_assume(count > 0);
    for (int i = 0; i < count; i++)
        out[i] = mli_math_asr_rnd_fx<int16_t>(in[i], shift_right);
}

static void convert_tensor_fx8_to_fx16(
        const MLI_PTR(int8_t) __restrict in, 
        MLI_PTR(int16_t) __restrict out, 
        int count, 
        int shift_right) {
    __builtin_assume(count > 0);
    for (int i = 0; i < count; i++)
        out[i] = mli_math_cast_fx<int8_t, int16_t>((int16_t)in[i], shift_right);
}

static void convert_tensor_fx16_to_fx8(
        const MLI_PTR(int16_t) __restrict in, 
        MLI_PTR(int8_t) __restrict out, 
        int count, 
        int shift_right) {
    for (int i = 0; i < count; i++)
        out[i] = mli_math_cast_fx<int16_t, int8_t>(in[i], shift_right);
}

uint32_t mli_hlp_count_elem_num(const mli_tensor *in, uint32_t start_dim) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_count_elem_num(in, start_dim), __func__);
    if (ret != MLI_STATUS_OK)
        return 0;
    return mli_prv_count_elem_num_part(in, start_dim);
}



uint32_t mli_hlp_tensor_element_size(const mli_tensor *in) {
    switch (in->el_type) {
        case MLI_EL_FX_8:  return sizeof(int8_t);
        case MLI_EL_FX_16: return sizeof(int16_t);
        case MLI_EL_SA_8:  return sizeof(int8_t);
        case MLI_EL_SA_32:  return sizeof(int32_t);
        default:
            MLI_ASSERT(0);
            return 0;
    }
}

uint32_t mli_hlp_tensor_scale_shift(const mli_tensor *in) {
    switch (in->el_type) {
        case MLI_EL_FX_8:
        case MLI_EL_FX_16:
            return in->el_params.fx.frac_bits;
        case MLI_EL_SA_8:
        case MLI_EL_SA_32:
            return in->el_params.sa.scale_frac_bits;
        default:
            MLI_ASSERT(0);
            return 0;
    }
}

int32_t mli_hlp_tensor_scale(const mli_tensor *in, const uint32_t scale_idx) {
    switch (in->el_type) {
        case MLI_EL_FX_8:
        case MLI_EL_FX_16:
            return 1;
        case MLI_EL_SA_8:
        case MLI_EL_SA_32:
            return (in->el_params.sa.dim >= 0)? in->el_params.sa.scale.mem.pi32[scale_idx]: in->el_params.sa.scale.mem.i32;
        default:
            MLI_ASSERT(0);
            return 0;
    }
}

int16_t mli_hlp_tensor_zero_offset(const mli_tensor *in, const uint32_t zero_idx) {
    switch (in->el_type) {
        case MLI_EL_FX_8:
        case MLI_EL_FX_16:
            return 0;
        case MLI_EL_SA_8:
        case MLI_EL_SA_32:
            return (in->el_params.sa.dim >= 0)? in->el_params.sa.zero_point.mem.pi16[zero_idx]: in->el_params.sa.zero_point.mem.i16;
        default:
            MLI_ASSERT(0);
            return 0;
    }
}

mli_status mli_hlp_point_to_subtensor(const mli_tensor *in, const mli_point_to_subtsr_cfg *cfg, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_point_to_subtensor(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    const uint32_t subtsr_start_axis = cfg->coord_num - 1;
    const uint32_t elem_size = mli_hlp_tensor_element_size(in);
    const uint32_t out_rank = in->rank - subtsr_start_axis;
    uint32_t dimension_sizes[MLI_MAX_RANK];

    uint32_t size = elem_size;
    for (int i = in->rank - 1; i >= 0; i--) {
        dimension_sizes[i] = size;
        size *= in->shape[i];
    }

    size = cfg->start_coord[0] * dimension_sizes[0];
    for (int i = 1; i < cfg->coord_num; i++)
        size += cfg->start_coord[i] * dimension_sizes[i];

    out->data.mem.void_p = (void *)((char *)in->data.mem.void_p + size);
    size = out->shape[0] = cfg->first_out_dim_size;
    for (int i = 1; i < out_rank; i++) {
        out->shape[i] = in->shape[subtsr_start_axis + i];
        size *= in->shape[subtsr_start_axis + i];
    }
    out->rank = out_rank;
    out->data.capacity = size * elem_size;
    out->el_params = in->el_params;
    out->el_type = in->el_type;

    return MLI_STATUS_OK;
}

mli_status mli_hlp_create_subtensor(const mli_tensor *in, const mli_sub_tensor_cfg *cfg, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_create_subtensor(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    const uint32_t elem_size = mli_hlp_tensor_element_size(in);
    const uint32_t out_rank = cfg->sub_tensor_rank;
    uint32_t mem_strides[MLI_MAX_RANK];
    const uint32_t input_rank = in->rank;
    const bool isAsym = (in->el_type == MLI_EL_SA_8) || (in->el_type == MLI_EL_SA_32);

    // compute memory strides for the input tensor if not yet provided by the input tensor.
    mem_strides[input_rank - 1] = in->mem_stride[input_rank - 1] != 0 ? in->mem_stride[input_rank - 1] : 1;
    for (int i = input_rank - 2; i >= 0; i--) {
        mem_strides[i] = in->mem_stride[i] != 0 ? in->mem_stride[i] : mem_strides[i+1] * in->shape[i+1];
    }

    // compute the offset inside the buffer
    int buf_offset = 0;
    for (int i = 0; i < input_rank; i++) {
        buf_offset += cfg->offset[i] * mem_strides[i];
    }
    buf_offset *= elem_size;
    out->data.mem.void_p = (void *)((char *)in->data.mem.void_p + buf_offset);
    out->data.capacity = in->data.capacity - buf_offset;

    // Fill the shape[] of the output tensor.
    // If the sub_tensor_rank is smaller than the input rank, the dimensions with
    // a size of 1 will be removed in the output shape starting from the first dimension
    // until the requested sub_tensor_rank value is reached.
    int out_idx = 0;
    int skip_cnt = input_rank - out_rank;
    int out_asym_dim = -1;
    int out_asym_offset = 0;
    for (int in_idx = 0; in_idx < input_rank; in_idx++) {
        if ((skip_cnt > 0) && (cfg->size[in_idx] == 1)) {
            skip_cnt--;
            continue;
        }
        out->shape[out_idx] = cfg->size[in_idx];
        out->mem_stride[out_idx] = mem_strides[in_idx];
        if (isAsym && (in->el_params.sa.dim == in_idx)) {
            out_asym_dim = out_idx;
            out_asym_offset = cfg->offset[in_idx];
        }
        out_idx++;
    }

    out->rank = out_rank;
    out->el_params = in->el_params;
    out->el_type = in->el_type;

    if (isAsym){
        if (out->el_params.sa.dim >= 0) {
            out->el_params.sa.scale.mem.pi32 += out_asym_offset;
            out->el_params.sa.dim = out_asym_dim;
            out->el_params.sa.zero_point.mem.pi16 += out_asym_offset;
        }
    }
    return MLI_STATUS_OK;
}

mli_status mli_hlp_convert_tensor(mli_tensor *in, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_convert_tensor(in, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    const int in_sz = (int)mli_prv_count_elem_num(in);
    const int out_shift = (int)(in->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;

    if(in_sz <= 0)
        return MLI_STATUS_BAD_TENSOR;

    // Switchnig functionality depending on tensors type
    if (in->el_type == MLI_EL_FX_8 && out->el_type == MLI_EL_FX_8)
        convert_tensor_fx8_to_fx8((MLI_PTR(int8_t))in->data.mem.void_p, (MLI_PTR(int8_t))out->data.mem.void_p, in_sz, out_shift);
    else if (in->el_type == MLI_EL_FX_16 && out->el_type == MLI_EL_FX_16)
        convert_tensor_fx16_to_fx16((MLI_PTR(int16_t))in->data.mem.void_p, (MLI_PTR(int16_t))out->data.mem.void_p, in_sz, out_shift);
    else if (in->el_type == MLI_EL_FX_8 && out->el_type == MLI_EL_FX_16)
        convert_tensor_fx8_to_fx16((MLI_PTR(int8_t))in->data.mem.void_p, (MLI_PTR(int16_t))out->data.mem.void_p, in_sz, out_shift);
    else if (in->el_type == MLI_EL_FX_16 && out->el_type == MLI_EL_FX_8)
        convert_tensor_fx16_to_fx8((MLI_PTR(int16_t))in->data.mem.void_p, (MLI_PTR(int8_t))out->data.mem.void_p, in_sz, out_shift);

    // Fill the rest output tensor params
    for (int idx = 0; idx < in->rank; idx++)
        out->shape[idx] = in->shape[idx];
    out->rank = in->rank;
    return MLI_STATUS_OK;
}

#ifdef __cplusplus
}
#endif

#pragma MLI_CODE_SECTION_END()
