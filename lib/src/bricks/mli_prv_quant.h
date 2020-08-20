/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_QUANT_H_
#define _MLI_PRV_QUANT_H_

#include "mli_config.h"
#include "mli_math.h"

template <typename in_T, typename out_T> MLI_FORCE_INLINE out_T mli_prv_convert_sa8_fx16(
    const in_T in, 
    const int16_t zero_point, 
    const int scale);
template <typename in_T, typename out_T> MLI_FORCE_INLINE out_T mli_prv_convert_fx16_sa8(
    const in_T in, 
    const int16_t zero_point, 
    const int scale);

#if defined(MLI_BUILD_REFERENCE)
#include "impl/mli_prv_quant_ref.h"
#elif defined(__Xvec_width)
#include "impl/mli_prv_quant_vdsp.h"
#elif defined(__FXAPI__)
#include "impl/mli_prv_quant_dsp.h"
#endif

#endif /* _MLI_PRV_QUANT_H_ */