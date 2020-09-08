/*
* Copyright 2019-2020, Synopsys, Inc.
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
#include "mli_prv_dsp.h"
#include "mli_prv_lut.h"
#include "mli_prv_activation_lut.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

const int kSigmAsymZeroPoint = -128;
const int kSigmOutputShift = 8;

#ifdef __cplusplus
extern "C" {
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")

mli_status mli_krn_sigm_fx8(const mli_tensor* in, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation_fx8(in, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli_prv_fx_init_dsp_ctrl();

    mli_prv_copy_tensor_format_except_mem_strides(in, out);
    out->el_params.fx.frac_bits = 7;

    mli_prv_activation_lut_fx8(in, out, &sigmoid_lut_fx16, in->el_params.fx.frac_bits,
            (int)mli_prv_count_elem_num(in));

    return MLI_STATUS_OK;
}

mli_status mli_krn_sigm_fx16(const mli_tensor* in, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation_fx16(in, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli_prv_fx_init_dsp_ctrl();

    mli_prv_copy_tensor_format_except_mem_strides(in, out);
    out->el_params.fx.frac_bits = 15;

    mli_prv_activation_lut_fx16(in, out, &sigmoid_lut_fx16, in->el_params.fx.frac_bits,
            (int)mli_prv_count_elem_num(in));

    return MLI_STATUS_OK;
}

mli_status mli_krn_sigm_sa8(const mli_tensor* in, mli_tensor* out) {
    struct s8asym_quant_params in_params;
    struct s8asym_quant_params out_params;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation_sa8(in, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli_prv_fx_init_dsp_ctrl();

    in_params.offset = in->el_params.sa.zero_point.mem.i16;
    in_params.scale  = in->el_params.sa.scale.mem.i32;
    in_params.shift = in->el_params.sa.scale_frac_bits.mem.i8;
    out_params.offset = kSigmAsymZeroPoint;
    out_params.scale  = 1;
    out_params.shift = kSigmOutputShift;

    // Update output shape
    mli_prv_copy_tensor_format_except_mem_strides(in, out);

    mli_prv_activation_lut_sa8(in, out, &sigmoid_lut_fx16,
            &in_params, &out_params, (int)mli_prv_count_elem_num(in));

    out->el_params.sa.zero_point.mem.i16 = out_params.offset;
    out->el_params.sa.scale.mem.i32 = out_params.scale;
    out->el_params.sa.scale_frac_bits.mem.i8 = (uint8_t)out_params.shift;

    return MLI_STATUS_OK;
}

#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
}
#endif
