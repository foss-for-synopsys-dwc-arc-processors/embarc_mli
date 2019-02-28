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
#include "tensor_transform.h"

#include <stdint.h>

#include "mli_api.h"
#include "tests_aux.h"

//================================================================================
// Transform float array to MLI FX tensor according to it's element type parameters
//=================================================================================
mli_status mli_hlp_float_to_fx_tensor (const float *src, uint32_t src_size, mli_tensor * dst) {
    mli_status ret = MLI_STATUS_OK;
    const uint32_t scale_val = 1u << (dst->el_params.fx.frac_bits);

    if (dst->el_type == MLI_EL_FX_16) {
        if (dst->capacity < src_size * sizeof (int16_t))
            return MLI_STATUS_LENGTH_ERROR;

        int16_t *dst_arr = dst->data;
        for (int idx = 0; idx < src_size; idx++) {
            const float round_val = (src[idx] > 0) ? 0.5f : -0.5f;
            int32_t dst_val = (int32_t) (scale_val * src[idx] + round_val);
            dst_arr[idx] = (int16_t) (MIN (MAX (dst_val, INT16_MIN), INT16_MAX));
        }
    } else {
        if (dst->capacity < src_size * sizeof (int8_t))
            return MLI_STATUS_LENGTH_ERROR;

        int8_t *dst_arr = dst->data;
        for (int idx = 0; idx < src_size; idx++) {
            const float round_val = (src[idx] > 0) ? 0.5f : -0.5f;
            const int32_t dst_val = (int32_t) (scale_val * src[idx] + round_val);
            dst_arr[idx] = (int8_t) (MIN (MAX (dst_val, INT8_MIN), INT8_MAX));
        }
    }
    return ret;
}

//================================================================================
// Transform MLI FX tensor to float array
//=================================================================================
mli_status mli_hlp_fx_tensor_to_float (const mli_tensor * src, float *dst, uint32_t dst_size) {
    uint32_t elem_num = mli_hlp_count_elem_num(src, 0);
    if (elem_num > dst_size)
        return MLI_STATUS_LENGTH_ERROR;
    if (elem_num == 0)
        return MLI_STATUS_BAD_TENSOR;

    const float scale_val = 1.0f / (float) (1u << (src->el_params.fx.frac_bits));
    if (src->el_type == MLI_EL_FX_16) {
        int16_t *src_arr = src->data;
        for (int idx = 0; idx < elem_num; idx++)
            dst[idx] = (float) (scale_val * src_arr[idx]);
    } else {
        int8_t *src_arr = src->data;
        for (int idx = 0; idx < elem_num; idx++)
            dst[idx] = (float) (scale_val * src_arr[idx]);
    }
    return MLI_STATUS_OK;
}
