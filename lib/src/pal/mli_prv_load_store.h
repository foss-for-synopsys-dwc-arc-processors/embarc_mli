/*
* Copyright 2020-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRV_LOAD_STORE_H_
#define _MLI_PRV_LOAD_STORE_H_

#if defined(__Xvec_width)
#include "vdsp/mli_prv_load_store.h"
#elif defined(__FXAPI__) //&& !defined(MLI_BUILD_REFERENCE)
// not ported kernels running EM/HS; always require dsp/* version of PAL.
#include "dsp/mli_prv_load_store.h"
#else
#include "ref/mli_prv_load_store.h"
#endif

#endif // _MLI_PRV_LOAD_STORE_H_