/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "mli_hlp_convert_tensor.h"

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_math.h"
#include "mli_helpers_api.h"
#include "mli_prv_tensor.h"

#pragma MLI_CODE_SECTION_START(".mli_lib")

#ifdef __cplusplus
extern "C" {
#endif

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
        case MLI_EL_FP_32: return sizeof(float);
        default:
            MLI_ASSERT(0);
            return 0;
    }
}

int32_t mli_hlp_tensor_scale_shift(const mli_tensor *in, const uint32_t scale_idx) {
    switch (in->el_type) {
        case MLI_EL_FX_8:
        case MLI_EL_FX_16:
            return in->el_params.fx.frac_bits;
        case MLI_EL_SA_8:
        case MLI_EL_SA_32:
            return (in->el_params.sa.dim >= 0)? in->el_params.sa.scale_frac_bits.mem.pi8[scale_idx]: in->el_params.sa.scale_frac_bits.mem.i8;
        case MLI_EL_FP_32:
            return 0;
        default:
            MLI_ASSERT(0);
            return 0;
    }
}

int32_t mli_hlp_tensor_scale(const mli_tensor *in, const uint32_t scale_idx) {
    switch (in->el_type) {
        case MLI_EL_FX_8:
        case MLI_EL_FX_16:
        case MLI_EL_FP_32:
            return 1;
        case MLI_EL_SA_8:
        case MLI_EL_SA_32:
            return (in->el_params.sa.dim >= 0)? in->el_params.sa.scale.mem.pi16[scale_idx]: in->el_params.sa.scale.mem.i16;
        default:
            MLI_ASSERT(0);
            return 0;
    }
}

int16_t mli_hlp_tensor_zero_offset(const mli_tensor *in, const uint32_t zero_idx) {
    switch (in->el_type) {
        case MLI_EL_FX_8:
        case MLI_EL_FX_16:
        case MLI_EL_FP_32:
            return 0;
        case MLI_EL_SA_8:
        case MLI_EL_SA_32:
            return (in->el_params.sa.dim >= 0)? in->el_params.sa.zero_point.mem.pi16[zero_idx]: in->el_params.sa.zero_point.mem.i16;
        default:
            MLI_ASSERT(0);
            return 0;
    }
}

bool mli_hlp_tensor_data_ptr_cmp(const mli_tensor *in1, const mli_tensor *in2) {
    bool val = false;
    if (in1->el_type != in2->el_type) return false;
    if ((in1->rank == 0) || (in2->rank == 0)) return false;
    switch (in1->el_type) {
        case MLI_EL_FX_8:
        case MLI_EL_SA_8:
            val = in1->data.mem.pi8 == in2->data.mem.pi8;
            break;
        case MLI_EL_FX_16:
            val = in1->data.mem.pi16 == in2->data.mem.pi16;
            break;
        case MLI_EL_FP_32:
            val = in1->data.mem.pf32 == in2->data.mem.pf32;
            break;
        case MLI_EL_SA_32:
            val = in1->data.mem.pi32 == in2->data.mem.pi32;
            break;
        default:
            MLI_ASSERT(0);
            val = false;
    }
    return val;
}

/* DEPRECATED */
mli_status mli_hlp_point_to_subtensor(const mli_tensor *in, const mli_point_to_subtsr_cfg *cfg, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_point_to_subtensor(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    const uint32_t subtsr_start_axis = cfg->coord_num - 1;
    const uint32_t elem_size = mli_hlp_tensor_element_size(in);
    const uint32_t out_rank = in->rank - subtsr_start_axis;
    uint32_t dimension_sizes[MLI_MAX_RANK];

    uint32_t size = 1;
    for (int i = in->rank - 1; i >= 0; i--) {
        dimension_sizes[i] = size;
        size *= in->shape[i];
    }

    size = cfg->start_coord[0] * dimension_sizes[0];
    for (int i = 1; i < cfg->coord_num; i++)
        size += cfg->start_coord[i] * dimension_sizes[i];

    switch(in->el_type) {
    case MLI_EL_FX_8:
    case MLI_EL_SA_8:
        out->data.mem.pi8 = in->data.mem.pi8 + size;
        break;
    case MLI_EL_FX_16:
        out->data.mem.pi16 = in->data.mem.pi16 + size;
        break;
    case MLI_EL_FP_32:
        out->data.mem.f32 = in->data.mem.f32 + size;
        break;
    case MLI_EL_SA_32:
        out->data.mem.pi32 = in->data.mem.pi32 + size;
        break;
    default:
        MLI_ASSERT(0);
        return MLI_STATUS_NOT_SUPPORTED;
    }

    size = out->shape[0] = cfg->first_out_dim_size;
    for (int i = 1; i < (int)out_rank; i++) {
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

    const int out_rank = cfg->sub_tensor_rank;
    int mem_strides[MLI_MAX_RANK];
    const int input_rank = in->rank;
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
    out->data = in->data;
    out->el_type = in->el_type;
    switch (in->el_type) {
    case MLI_EL_FX_8:
    case MLI_EL_SA_8:
        mli_prv_tensor_inc_data_ptr<int8_t*>(out, buf_offset);
        break;
    case MLI_EL_FX_16:
        mli_prv_tensor_inc_data_ptr<int16_t*>(out, buf_offset);
        break;
    case MLI_EL_SA_32:
        mli_prv_tensor_inc_data_ptr<int32_t*>(out, buf_offset);
        break;
    default:
        MLI_ASSERT(0);
        return MLI_STATUS_NOT_SUPPORTED;
    }


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

    if (isAsym){
        if (out->el_params.sa.dim >= 0) {
            out->el_params.sa.scale.mem.pi16 += out_asym_offset;
            out->el_params.sa.scale_frac_bits.mem.pi8 += out_asym_offset;
            out->el_params.sa.dim = out_asym_dim;
            out->el_params.sa.zero_point.mem.pi16 += out_asym_offset;
        }
    }
    return MLI_STATUS_OK;
}

mli_status mli_hlp_convert_tensor_safx(const mli_tensor * src, mli_tensor * dst) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_convert_tensor(src, dst), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    return mli::hlp::convert_quantized_data(src, dst);
}

mli_status mli_hlp_convert_tensor(const mli_tensor * src, mli_tensor * dst) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_convert_tensor(src, dst), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    // To check if output tensor has enough memory.
    if (src->el_type != MLI_EL_FP_32 && dst->el_type != MLI_EL_FP_32) {
        ret = mli_hlp_convert_tensor_safx(src, dst);
        return ret;
    } else {
        mli::hlp::convert_mode mode;
        if (src->el_type == MLI_EL_FP_32) {
            mode = mli::hlp::QUANTIZE;
        } else {
            mode = mli::hlp::DEQUANTIZE;
        }
        if ((src->el_type == MLI_EL_FX_8 || src->el_type == MLI_EL_SA_8) ||
            (dst->el_type == MLI_EL_FX_8 || dst->el_type == MLI_EL_SA_8)) {
            ret = mli::hlp::convert_float_data<int8_t>(src, dst, mode);
        } else if (src->el_type == MLI_EL_FX_16 || dst->el_type == MLI_EL_FX_16) {
            ret = mli::hlp::convert_float_data<int16_t>(src, dst, mode);
        } else if (src->el_type == MLI_EL_SA_32 || dst->el_type == MLI_EL_SA_32) {
            ret = mli::hlp::convert_float_data<int32_t>(src, dst, mode);
        } else if (src->el_type == MLI_EL_FP_32 || dst->el_type == MLI_EL_FP_32) {
            ret = mli::hlp::convert_float_data<float>(src, dst, mode);
        } else {
            ret = MLI_STATUS_TYPE_MISMATCH;
        }
    }
    return MLI_STATUS_OK;
}

const char* mli_hlp_compile_options_string() {
    return OPTIONS_STR;
}

#if defined(__FXAPI__)
// FXAPI have an interface level contract on number of bits. 
// MLI Implementation uses functions with guardbits. it's the only currently allowed case. 
uint8_t mli_hlp_accu_guard_bits_sa8_sa8() { 
    return (sizeof(mli_acc32_t)*8) - ((sizeof(int8_t)*8) * 2); 
}
uint8_t mli_hlp_accu_guard_bits_fx16_fx16() { 
    constexpr uint8_t kWiderAccBits = 40;
    return (kWiderAccBits - (sizeof(int16_t)*8) * 2);
}
uint8_t mli_hlp_accu_guard_bits_fx16_fx8() { 
    return (sizeof(mli_acc32_t)*8) - ((sizeof(int8_t)*8) + (sizeof(int16_t)*8));
}


#elif defined(__Xvec_width)
// Vector DSP Code. Number of guard bits depends on the guard bits option

#if (__Xvec_guard_bit_option == 0)
constexpr uint8_t extra_bits = 0;
#elif (__Xvec_guard_bit_option == 1)
constexpr uint8_t extra_bits = 4;
#elif (__Xvec_guard_bit_option == 2)
constexpr uint8_t extra_bits = 8;
#else 
#error "mli_hlp_accu_guard_bits: unknown case for __Xvec_guard_bit_option"
#endif

uint8_t mli_hlp_accu_guard_bits_sa8_sa8() { 
    const uint8_t basic_guard_bits = (sizeof(int16_t)*8) - ((sizeof(int8_t)*8) * 2);
    return basic_guard_bits + extra_bits;
}
uint8_t mli_hlp_accu_guard_bits_fx16_fx16() { 
    const uint8_t basic_guard_bits = (sizeof(int32_t)*8) - ((sizeof(int16_t)*8) * 2);
    return basic_guard_bits + extra_bits;
}
uint8_t mli_hlp_accu_guard_bits_fx16_fx8() { 
    const uint8_t basic_guard_bits = (sizeof(int32_t)*8) - ((sizeof(int8_t)*8) + (sizeof(int16_t)*8));
    return basic_guard_bits + extra_bits;
}

#else 
// Reference use usual C built-in types through the typedef prism. 
uint8_t mli_hlp_accu_guard_bits_sa8_sa8() {
    return (sizeof(mli_acc32_t)*8) - ((sizeof(int8_t)*8) * 2); 
}
uint8_t mli_hlp_accu_guard_bits_fx16_fx16() {
    return (sizeof(mli_acc40_t)*8) - ((sizeof(int16_t)*8) * 2);
}
uint8_t mli_hlp_accu_guard_bits_fx16_fx8() {
    return (sizeof(mli_acc32_t)*8) - ((sizeof(int8_t)*8) + (sizeof(int16_t)*8));
}
#endif


// set the memstrides based on the rank and shape
void mli_hlp_set_tensor_mem_strides(mli_tensor* in) {
    int rank = in->rank;
    in->mem_stride[rank-1] = 1;
    for (int i = rank - 2; i >=0; i--) {
        in->mem_stride[i] = in->mem_stride[i+1] * in->shape[i+1];
    }
}

#ifdef __cplusplus
}
#endif

#pragma MLI_CODE_SECTION_END()
