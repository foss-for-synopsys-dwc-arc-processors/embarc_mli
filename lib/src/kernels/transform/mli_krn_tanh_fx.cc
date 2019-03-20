/*
* Copyright 2019, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_lut.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma Code(".mli_lib")

mli_status mli_krn_tanh_fx8(const mli_tensor* in, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation_fx8(in, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli_prv_fx_init_dsp_ctrl();

    mli_prv_activation_lut_fx8(
            (MLI_PTR(int8_t))in->data, (MLI_PTR(int8_t))out->data, &tanh_lut_fx16, in->el_params.fx.frac_bits,
            (int)mli_prv_count_elem_num(in));
    mli_prv_copy_tensor_format(in, out);
    out->el_params.fx.frac_bits = 7;

    return MLI_STATUS_OK;
}

mli_status mli_krn_tanh_fx16(const mli_tensor* in, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation_fx16(in, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli_prv_fx_init_dsp_ctrl();

    mli_prv_activation_lut_fx16(
            (MLI_PTR(int16_t))in->data, (MLI_PTR(int16_t))out->data, &tanh_lut_fx16, in->el_params.fx.frac_bits,
            (int)mli_prv_count_elem_num(in));
    mli_prv_copy_tensor_format(in, out);
    out->el_params.fx.frac_bits = 15;

    return MLI_STATUS_OK;
}

#pragma code()

#ifdef __cplusplus
}
#endif
