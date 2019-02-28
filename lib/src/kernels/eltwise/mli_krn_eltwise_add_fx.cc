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
#include "mli_krn_eltwise.h"

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma Code(".mli_lib")

/*******************************************************************************
 *
 * Placeholders for kernels (for future optimizations)
 *
 *******************************************************************************/

mli_status mli_krn_eltwise_add_fx8(const mli_tensor* in1, const mli_tensor* in2, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise_add_fx8(in1, in2, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::eltwise_prepare_and_run_fx<int8_t, mli::ELTWISE_ADD>(in1, in2, out);

    return MLI_STATUS_OK;
}

mli_status mli_krn_eltwise_add_fx16(const mli_tensor* in1, const mli_tensor* in2, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise_add_fx16(in1, in2, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::eltwise_prepare_and_run_fx<int16_t, mli::ELTWISE_ADD>(in1, in2, out);

    return MLI_STATUS_OK;
}

#pragma code()

#ifdef __cplusplus
}  // extern "C"
#endif
