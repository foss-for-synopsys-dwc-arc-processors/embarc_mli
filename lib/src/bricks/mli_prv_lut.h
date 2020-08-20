/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_PRIVATE_LUT_H_
#define _MLI_PRIVATE_LUT_H_

#include "mli_config.h" /* for MLI_PTR */
#include "mli_private_types.h"
#include "mli_prv_lut_decl.h"

////////////////////////////////////////////////////////////////////////////////
// Setting up namespace
////////////////////////////////////////////////////////////////////////////////
namespace mli {
namespace krn {
#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
using mli::krn::vdsp::activation_lut;

#elif !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
using mli::krn::dsp::activation_lut;

#else
using mli::krn::ref::activation_lut;

#endif
} // krn
} // mli

////////////////////////////////////////////////////////////////////////////////
// Include implementation
////////////////////////////////////////////////////////////////////////////////
#include "impl/mli_prv_lut_ref.h"

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
#include "impl/mli_prv_lut_vdsp.h"
#endif

#if !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
#include "impl/mli_prv_lut_dsp.h"
#endif

#endif  //_MLI_PRIVATE_LUT_H_
