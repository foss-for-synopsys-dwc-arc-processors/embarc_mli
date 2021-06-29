/*
* Copyright 2019-2021, Synopsys, Inc.
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
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")

mli_status mli_krn_sigm_fx8(const mli_tensor *in, const mli_lut *lut, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation_fx8(in, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();
    mli_prv_fx_init_dsp_ctrl();

    if (in != out) mli_prv_copy_tensor_format_except_mem_strides(in, out);
    
    mli_prv_activation_lut_fx8(in, out, lut, in->el_params.fx.frac_bits);

    out->el_params.fx.frac_bits = 7;

    return MLI_STATUS_OK;
}

mli_status mli_krn_sigm_fx16(const mli_tensor *in, const mli_lut *lut, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation_fx16(in, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();
    mli_prv_fx_init_dsp_ctrl();

    if (in != out)  mli_prv_copy_tensor_format_except_mem_strides(in, out);

    mli_prv_activation_lut_fx16(in, out, lut, in->el_params.fx.frac_bits);

    out->el_params.fx.frac_bits = 15;

    return MLI_STATUS_OK;
}

mli_status mli_krn_sigm_sa8(const mli_tensor *in, const mli_lut *lut, mli_tensor *out) {
    struct s8asym_quant_params in_params;
    struct s8asym_quant_params out_params;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation_sa8(in, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();
    mli_prv_fx_init_dsp_ctrl();

    in_params.offset = in->el_params.sa.zero_point.mem.i16;
    in_params.scale  = in->el_params.sa.scale.mem.i16;
    in_params.shift  = in->el_params.sa.scale_frac_bits.mem.i8;
    out_params.offset = K_SIGM_ASYM_ZERO_POINT;
    out_params.scale  = 1;
    out_params.shift  = K_SIGM_OUTPUT_SHIFT;

    // Update output shape
    if (in != out) mli_prv_copy_tensor_format_except_mem_strides(in, out);

    mli_prv_activation_lut_sa8(in, out, lut, &in_params, &out_params);

    out->el_params.sa.zero_point.mem.i16 = out_params.offset;
    out->el_params.sa.scale.mem.i16 = out_params.scale;
    out->el_params.sa.scale_frac_bits.mem.i8 = (int8_t)out_params.shift;

    return MLI_STATUS_OK;
}

int32_t mli_krn_sigm_get_lut_size() {
    return (sigmoid_lut_fx16.length * sizeof(int16_t));
}

mli_status mli_krn_sigm_create_lut(mli_lut *lut) {
    lut->type = sigmoid_lut_fx16.type;
    mli_status ret = MLI_CHECK_STATUS(mli_chk_lut(lut, sigmoid_lut_fx16.data.capacity), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    MLI_PRINT_COMPILE_OPTIONS();

    memcpy(lut->data.mem.pi16, sigmoid_lut_fx16.data.mem.pi16, sigmoid_lut_fx16.length * sizeof(int16_t));
    lut->in_frac_bits = sigmoid_lut_fx16.in_frac_bits;
    lut->length = sigmoid_lut_fx16.length;
    lut->input_offset = sigmoid_lut_fx16.input_offset;
    lut->output_offset = sigmoid_lut_fx16.output_offset;
    lut->out_frac_bits = sigmoid_lut_fx16.out_frac_bits;
    return MLI_STATUS_OK;
}

#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
}
#endif
