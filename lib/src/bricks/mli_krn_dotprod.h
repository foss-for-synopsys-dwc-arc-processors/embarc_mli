/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_DOTPROD_H_
#define _MLI_KRN_DOTPROD_H_

#include "mli_krn_dotprod_decl.h"

// This header file must be included by users inside MLI library that depend
// on mli_krn_eltwise. Depending on platform capabilities, the right
// implementation with 'using' is chosen. This header file is responsible for
// including *_dsp (FXAPI) and *_vdsp (vector DSP) variants of mli_krn_eltwise.

////////////////////////////////////////////////////////////////////////////////
// Setting up namespace
////////////////////////////////////////////////////////////////////////////////
// Selecting between different variants (depending on hardware features) is
// done with 'using'. A completely different implementation can be used/'using'.
// However, also only a part of the reference together with optimized functions
// (from example *_dsp) can be used/'using'.

namespace mli {
namespace krn {
#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
using mli::krn::ref::dotprod1D;
using mli::krn::ref::dotprod2D;
using mli::krn::vdsp::dotprod2D_vv;
using mli::krn::ref::dotprod3D;
using mli::krn::vdsp::dotprod3D_v;

#elif !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
using mli::krn::ref::dotprod1D;
using mli::krn::dsp::dotprod2D;
using mli::krn::dsp::dotprod2D_hwc_v;
using mli::krn::dsp::dotprod2D_inp_width_v;
using mli::krn::dsp::dotprod2D_hwc_v_point;
using mli::krn::dsp::dotprod3D_v_simple;
using mli::krn::dsp::dotprod2D_inp_width_v;
using mli::krn::ref::dotprod3D;

#else
using mli::krn::ref::dotprod1D;
using mli::krn::ref::dotprod2D;
using mli::krn::ref::dotprod3D;

#endif
} // namespace krn
} // namespace mli

////////////////////////////////////////////////////////////////////////////////
// Include implementation
////////////////////////////////////////////////////////////////////////////////
// The reference (*_ref.h) implementation can run on all platforms and is always
// included. Other variants are included based on capabilities. Implementations
// below can depend on each other through declarations in *_decl.h.
#include "impl/mli_krn_dotprod_ref.h"

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
#include "impl/mli_krn_dotprod_vdsp.h"
#endif

#if !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
#include "impl/mli_krn_dotprod_dsp.h"
#endif

#endif // _MLI_KRN_DOTPROD_H_
