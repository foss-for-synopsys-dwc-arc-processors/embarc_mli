/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_krn_eltwise.h"

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")

/*******************************************************************************
 *
 * Placeholders for kernels (for future optimizations)
 *
 *******************************************************************************/

mli_status mli_krn_eltwise_mul_fx8(const mli_tensor* in1, const mli_tensor* in2, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise_mul_fx8(in1, in2, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::eltwise_prepare_and_run_fx<int8_t, mli::ELTWISE_MUL>(in1, in2, out);

    return MLI_STATUS_OK;
}

mli_status mli_krn_eltwise_mul_fx16(const mli_tensor* in1, const mli_tensor* in2, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_eltwise_mul_fx16(in1, in2, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    mli::eltwise_prepare_and_run_fx<int16_t, mli::ELTWISE_MUL>(in1, in2, out);

    return MLI_STATUS_OK;
}

#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
}  // extern "C"
#endif
