/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_DSP_H_
#define _MLI_PRV_DSP_H_

#if defined(__Xvec_width)
#include "vdsp/mli_prv_dsp.h"
#elif defined(__FXAPI__) && !defined(MLI_BUILD_REFERENCE)
#include "dsp/mli_prv_dsp.h"
#else
#include "ref/mli_prv_dsp.h"
#endif

#endif // _MLI_PRV_DSP_H_