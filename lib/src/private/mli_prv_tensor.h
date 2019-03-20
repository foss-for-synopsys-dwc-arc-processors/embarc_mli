/*
* Copyright 2019, Synopsys, Inc.
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
#include "mli_types.h"
#include "mli_private_types.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline mli_status __attribute__ ((always_inline)) mli_prv_copy_tensor_format (
        const mli_tensor * src, 
        mli_tensor * dst) {
    mli_status check = MLI_CHECK_STATUS(mli_chk_tensor (src), __func__);
    if (check != MLI_STATUS_OK)
          return check;

    for (int idx = 0; idx < src->rank; idx++)
          dst->shape[idx] = src->shape[idx];
      dst->rank = src->rank;
      dst->el_type = src->el_type;
      dst->el_params.fx.frac_bits = src->el_params.fx.frac_bits;
      return MLI_STATUS_OK;
}

static int inline __attribute__((always_inline)) mli_prv_calc_shift(
        const mli_tensor *in0,
        const mli_tensor *in1,
        const mli_tensor *out){
    return (in0->el_params.fx.frac_bits + in1->el_params.fx.frac_bits) - out->el_params.fx.frac_bits;
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

mli_minmax_t mli_prv_get_relu_min_max (const mli_relu_cfg * cfg, const mli_tensor * out);


#ifdef __cplusplus
}
#endif

#endif //_MLI_PRV_TENSOR_H_
