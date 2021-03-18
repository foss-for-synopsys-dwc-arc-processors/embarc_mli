/*
* Copyright 2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_krn_gru_cell.h"

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_prv_activation_lut.h"
#include "mli_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef mli_acc32_t mli_sa8_sa8_sa32_accu_t;
typedef mli_acc40_t mli_fx16_accu_t;
typedef mli_acc32_t mli_fx16_fx8_fx8_accu_t;

#pragma MLI_CODE_SECTION_START(".mli_lib")

mli_status mli_krn_gru_cell_fx16 (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_gru_cell_fx16
        (in, prev_out, weights_in, weights_out, bias, tanh_lut, sigm_lut, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::krn::gru_cell_prepare_and_run<int16_t, int16_t, int16_t, mli_fx16_accu_t, 
        mli::krn::fx_quant_specific_params>
        (in, prev_out, weights_in, weights_out, bias, tanh_lut, sigm_lut, cfg, out);

    return ret;
}

mli_status mli_krn_gru_cell_fx16_fx8_fx8 (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_gru_cell_fx16_fx8_fx8
        (in, prev_out, weights_in, weights_out, bias, tanh_lut, sigm_lut, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::krn::gru_cell_prepare_and_run<int16_t, int8_t, int8_t, mli_fx16_fx8_fx8_accu_t, 
        mli::krn::fx_quant_specific_params>
        (in, prev_out, weights_in, weights_out, bias, tanh_lut, sigm_lut, cfg, out);

    return ret;
}

mli_status mli_krn_gru_cell_sa8_sa8_sa32 (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights_in,
        const mli_tensor * weights_out,
        const mli_tensor * bias,
        const mli_lut * tanh_lut,
        const mli_lut * sigm_lut,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_gru_cell_sa8_sa8_sa32
        (in, prev_out, weights_in, weights_out, bias, tanh_lut, sigm_lut, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::krn::gru_cell_prepare_and_run<int8_t, int8_t, int32_t, mli_sa8_sa8_sa32_accu_t, 
        mli::krn::s8asym_quant_specific_params>
        (in, prev_out, weights_in, weights_out, bias, tanh_lut, sigm_lut, cfg, out);

    return ret;
}

#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
}
#endif
