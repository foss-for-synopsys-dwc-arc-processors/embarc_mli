/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_api.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_krn_common.h"

#ifdef __cplusplus
extern "C" {
#endif


#pragma Code(".mli_lib")

mli_status mli_krn_basic_rnn_cell_fx8 (
        const mli_tensor * in,
        const mli_tensor * prev_out, 
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_rnn_cell_fx8(in, prev_out, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    mli::basic_rnn_cell_prepare_and_run_fx < int8_t, int8_t > (in, prev_out, weights, bias, cfg, out);

    return MLI_STATUS_OK;
} 

mli_status mli_krn_basic_rnn_cell_fx16 (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights,
        const mli_tensor * bias, 
        const mli_rnn_cell_cfg * cfg, 
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_rnn_cell_fx16(in, prev_out, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    mli::basic_rnn_cell_prepare_and_run_fx < int16_t, int16_t > (in, prev_out, weights, bias, cfg, out);
    return MLI_STATUS_OK;
}


mli_status mli_krn_basic_rnn_cell_fx8w16d (
        const mli_tensor * in,
        const mli_tensor * prev_out,
        const mli_tensor * weights,
        const mli_tensor * bias,
        const mli_rnn_cell_cfg * cfg,
        mli_tensor * out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_rnn_cell_fx8w16d(in, prev_out, weights, bias, cfg, out), __func__);
    if (ret != MLI_STATUS_OK)
        return ret;

    mli::basic_rnn_cell_prepare_and_run_fx < int16_t, int8_t > (in, prev_out, weights, bias, cfg, out);
    return MLI_STATUS_OK;
}

#pragma code()

#ifdef __cplusplus
}                               //extern "C"
#endif
