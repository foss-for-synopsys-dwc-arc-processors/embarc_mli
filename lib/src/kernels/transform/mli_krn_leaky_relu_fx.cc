/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_load_store.h"
#include "mli_prv_tensor.h"
#include "mli_krn_prelu.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")

mli_status mli_krn_leaky_relu_fx8(const mli_tensor *in, const mli_tensor *slope_coeff, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_leaky_relu_fx8(in, slope_coeff, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    const mli_prelu_cfg cfg = {/*axis*/ -1};
    return mli::krn::prelu_fx_run<int8_t>(in, slope_coeff, &cfg, out);
}

mli_status mli_krn_leaky_relu_fx16(const mli_tensor *in, const mli_tensor *slope_coeff, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_leaky_relu_fx16(in, slope_coeff, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    const mli_prelu_cfg cfg = {/*axis*/ -1};
    return mli::krn::prelu_fx_run<int16_t>(in, slope_coeff, &cfg, out);
}

mli_status mli_krn_leaky_relu_sa8(const mli_tensor *in, const mli_tensor *slope_coeff, mli_tensor *out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_leaky_relu_sa8(in, slope_coeff, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;

    const mli_prelu_cfg cfg = {/*axis*/ -1};
    return mli::krn::prelu_sa8_run(in, slope_coeff, &cfg, out);
}
#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
}
#endif
