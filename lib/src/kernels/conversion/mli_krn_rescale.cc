/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "mli_check.h"
#include "mli_debug.h"
#include "mli_krn_rescale.hpp"
#include "mli_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma MLI_CODE_SECTION_START(".mli_lib")

//========================================================
//
//        MLI 3.0 Bare semantic functions
//
//========================================================
using snps_arc::metaware::mli::krn::rescale_prepare_and_run;

mli_status mli_krn_rescale_i32_o8(const mli_tensor *in,
                                  const mli_tensor *bias_in,
                                  const mli_tensor *scale,
                                  const mli_tensor *shift,
                                  const mli_tensor *bias_out,
                                  mli_tensor *out) {
    // TODO: Make sense to have checker even for this form.
    // mli_status ret = MLI_CHECK_STATUS(mli_chk_rescale_i32_o8(in, scale, shift, out), __func__);
    // if (ret != MLI_STATUS_OK) return ret;
    // MLI_PRINT_COMPILE_OPTIONS();

    return rescale_prepare_and_run<int32_t, int8_t>(in, bias_in, scale, shift,
                                                    bias_out, out);
}

#pragma MLI_CODE_SECTION_END()

#ifdef __cplusplus
}
#endif
