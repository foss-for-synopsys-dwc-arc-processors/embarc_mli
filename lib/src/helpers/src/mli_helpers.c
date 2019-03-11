/*
*  Copyright (c) 2019, Synopsys, Inc. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without modification,
* are permitted provided that the following conditions are met:
*
* 1) Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 
* 2)  Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
* 
* 3) Neither the name of the <ORGANIZATION> nor the names of its contributors
* may be used to endorse or promote products derived from this software
* without specific prior written permission.
* 
* THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
* ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_prv_tensor.h"

#pragma Code(".mli_lib")


static void convert_tensor_fx8_to_fx8(const int8_t *in, int8_t *out, int count, int shift_right);
static void convert_tensor_fx16_to_fx16(const int16_t *in, int16_t *out, int count, int shift_right);
static void convert_tensor_fx8_to_fx16(const int8_t *in, int16_t *out, int count, int shift_right);
static void convert_tensor_fx16_to_fx8(const int16_t *in, int8_t *out, int count, int shift_right);


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
        default: return 0;
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

    out->data = (void *)((char *)in->data + size);
    size = out->shape[0] = cfg->first_out_dim_size;
    for (int i = 1; i < out_rank; i++) {
        out->shape[i] = in->shape[subtsr_start_axis + i];
        size *= in->shape[subtsr_start_axis + i];
    }
    out->rank = out_rank;
    out->capacity = size * elem_size;
    out->el_params = in->el_params;
    out->el_type = in->el_type;

    return MLI_STATUS_OK;
}


mli_status mli_hlp_convert_tensor(mli_tensor *in, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_convert_tensor(in, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    const int in_sz = (int)mli_prv_count_elem_num(in);
    const int out_shift = (int)(in->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;

    // Switchnig functionality depending on tensors type
    if (in->el_type == out->el_type == MLI_EL_FX_8)
        convert_tensor_fx8_to_fx8((int8_t *)in->data, (int8_t *)out->data, in_sz, out_shift);
    else if (in->el_type == out->el_type == MLI_EL_FX_16)
        convert_tensor_fx16_to_fx16((int16_t *)in->data, (int16_t *)out->data, in_sz, out_shift);
    else if (in->el_type == MLI_EL_FX_8 && out->el_type == MLI_EL_FX_16)
        convert_tensor_fx8_to_fx16((int8_t *)in->data, (int16_t *)out->data, in_sz, out_shift);
    else if (in->el_type == MLI_EL_FX_16 && out->el_type == MLI_EL_FX_8)
        convert_tensor_fx16_to_fx8((int16_t *)in->data, (int8_t *)out->data, in_sz, out_shift);

    return MLI_STATUS_OK;
}


static void convert_tensor_fx8_to_fx8(const int8_t *in, int8_t *out, int count, int shift_right) {
    for (int i = 0; i < count; ++i)
        out[i] = (int8_t)fx_sat_q15(fx_asr_rnd_q15((int16_t)in[i], shift_right), 8);
}

static void convert_tensor_fx16_to_fx16(const int16_t *in, int16_t *out, int count, int shift_right) {
    for (int i = 0; i < count; ++i)
        out[i] = fx_asr_rnd_q15(in[i], shift_right);
}

static void convert_tensor_fx8_to_fx16(const int8_t *in, int16_t *out, int count, int shift_right) {
    for (int i = 0; i < count; ++i)
        out[i] = (int16_t)fx_asr_rnd_q15((int16_t)in[i], shift_right);
}

static void convert_tensor_fx16_to_fx8(const int16_t *in, int8_t *out, int count, int shift_right) {
    for (int i = 0; i < count; ++i)
        out[i] = (int8_t)fx_sat_q15(fx_asr_rnd_q15(in[i], shift_right), 8);
}

#pragma code()
