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

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_prv_dsp.h"
#include "mli_prv_lut.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma Code(".mli_lib")

mli_status mli_krn_sigm_fx8(const mli_tensor* in, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation_fx8(in, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli_prv_fx_init_dsp_ctrl();

    mli_prv_activation_lut_fx8(
            (MLI_PTR(int8_t))in->data, (MLI_PTR(int8_t))out->data, &sigmoid_lut_fx16, in->el_params.fx.frac_bits,
            (int)mli_prv_count_elem_num(in));
    mli_prv_copy_tensor_format(in, out);
    out->el_params.fx.frac_bits = 7;

    return MLI_STATUS_OK;
}

mli_status mli_krn_sigm_fx16(const mli_tensor* in, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_basic_activation_fx16(in, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    mli_prv_fx_init_dsp_ctrl();

    mli_prv_activation_lut_fx16(
            (MLI_PTR(int16_t))in->data, (MLI_PTR(int16_t))out->data, &sigmoid_lut_fx16, in->el_params.fx.frac_bits,
            (int)mli_prv_count_elem_num(in));
    mli_prv_copy_tensor_format(in, out);
    out->el_params.fx.frac_bits = 15;

    return MLI_STATUS_OK;
}

#pragma code()

#ifdef __cplusplus
}
#endif
