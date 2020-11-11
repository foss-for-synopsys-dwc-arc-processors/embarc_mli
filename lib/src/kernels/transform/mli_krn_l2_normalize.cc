/*
* Copyright 2020, Synopsys, Inc.
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
#include "mli_prv_tensor.h"
#include "mli_types.h"
#include "mli_krn_l2_normalize.h"

const int kL2NormAsymZeroPoint = 0;
const int kL2NormOutputShift = 7;

using mli::krn::mli_krn_l2_normalize_run;
#ifdef __cplusplus
extern "C" {
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")

mli_status mli_krn_l2_normalize_fx16(const mli_tensor *in, 
        const mli_tensor *epsilon, 
        const mli_l2_normalize_cfg *cfg, 
        mli_tensor *out) {

    return mli_krn_l2_normalize_run<int16_t>(in, epsilon, cfg, out);
}

mli_status mli_krn_l2_normalize_sa8(const mli_tensor *in, 
        const mli_tensor *epsilon, 
        const mli_l2_normalize_cfg *cfg, 
        mli_tensor *out) {

    return mli_krn_l2_normalize_run<int8_t, true>(in, epsilon, cfg, out);
}

#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
}
#endif
