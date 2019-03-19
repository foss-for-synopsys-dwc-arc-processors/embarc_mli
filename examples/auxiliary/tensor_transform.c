/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
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
