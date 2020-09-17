/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_SOFTMAX_VDSP_H_
#define _MLI_KRN_SOFTMAX_VDSP_H_

#include "mli_check.h"
#include "mli_config.h"
#include "mli_debug.h"
#include "mli_helpers_api.h"
#include "mli_math.h"
#include "mli_prv_dsp.h"
#include "mli_prv_lut.h"
#include "mli_prv_activation_lut.h"
#include "mli_prv_tensor.h"
#include "mli_types.h"

namespace mli {
namespace krn {
namespace vdsp {

template <typename io_T>
static MLI_FORCE_INLINE mli_status mli_krn_softmax_fx_run(const mli_tensor *in, const mli_softmax_cfg* cfg,
        mli_tensor *out) {
    /* TODO */
    return MLI_STATUS_NOT_SUPPORTED;
}

} // namespace vdsp
} // namespace krn
} // namespace mli

#endif // _MLI_KRN_SOFTMAX_VDSP_H_
