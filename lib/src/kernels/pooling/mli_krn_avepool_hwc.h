/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_AVEPOOL_HWC_H_
#define _MLI_KRN_AVEPOOL_HWC_H_

#include "mli_krn_avepool_hwc_decl.h"

////////////////////////////////////////////////////////////////////////////////
// Setting up namespace
////////////////////////////////////////////////////////////////////////////////
//namespace mli { // TODO: callers of below functions expect global namespace
//namespace krn {
#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
using mli::krn::ref::avepool_hwc_nopad;
using mli::krn::ref::avepool_hwc;
using mli::krn::ref::avepool_hwc_krnpad;

#elif !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
using mli::krn::dsp::avepool_hwc_nopad;
using mli::krn::dsp::avepool_hwc;
using mli::krn::dsp::avepool_hwc_krnpad;

#else
using mli::krn::ref::avepool_hwc_nopad;
using mli::krn::ref::avepool_hwc;
using mli::krn::ref::avepool_hwc_krnpad;

#endif
//} // krn
//} // mli

////////////////////////////////////////////////////////////////////////////////
// Include implementation
////////////////////////////////////////////////////////////////////////////////
#include "impl/mli_krn_avepool_hwc_ref.h"

// TODO: !defined(MLI_BUILD_REFERENCE) && ... if reference version is available
#if defined(__Xvec_width)
// no implementation
#endif

// TODO: !defined(MLI_BUILD_REFERENCE) && ... if reference version is available
#if defined(__FXAPI__)
#include "impl/mli_krn_avepool_hwc_dsp.h"
#endif

#endif // _MLI_KRN_AVEPOOL_HWC_H_
