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
#include "mli_prv_load_store.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"
#include "mli_krn_relu.h"

using mli::krn::mli_krn_relu_fx_run;

#ifdef __cplusplus
extern "C" {
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")

mli_status mli_krn_relu_fx8(const mli_tensor* in, const mli_relu_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_relu_fx8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    
    ret = mli_krn_relu_fx_run<int8_t, /*asym = */ false>(in, cfg, out);
    return ret;
}

mli_status mli_krn_relu_fx16(const mli_tensor* in, const mli_relu_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_relu_fx16(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    
    ret = mli_krn_relu_fx_run<int16_t, /*asym = */ false>(in, cfg, out);
    return ret;
}


mli_status mli_krn_relu_sa8(const mli_tensor* in, const mli_relu_cfg* cfg, mli_tensor* out) {
    mli_status ret = MLI_CHECK_STATUS(mli_chk_relu_sa8(in, cfg, out), __func__);
    if (ret != MLI_STATUS_OK) return ret;
    
    ret = mli_krn_relu_fx_run<int8_t, /*asym = */ true>(in, cfg, out);

    out->el_params.sa.zero_point.mem.i16 = in->el_params.sa.zero_point.mem.i16;
    out->el_params.sa.scale.mem.i16 = in->el_params.sa.scale.mem.i16;
    out->el_params.sa.scale_frac_bits.mem.i8 = in->el_params.sa.scale_frac_bits.mem.i8;

    return ret;
}

#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
}
#endif
