/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_L2_NORMALIZE_VDSP_H_
#define _MLI_KRN_L2_NORMALIZE_VDSP_H_

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace vdsp {

template <typename io_T>
static MLI_FORCE_INLINE mli_status mli_krn_l2_normalize_run(const mli_tensor *in, 
        const mli_tensor *epsilon, 
        const mli_l2_normalize_cfg *cfg, 
        mli_tensor *out) {

    mli_prv_fx_init_dsp_ctrl();
    

    return MLI_STATUS_NOT_SUPPORTED;
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_L2_NORMALIZE_VDSP_H_