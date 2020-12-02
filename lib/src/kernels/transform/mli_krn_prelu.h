/*
* Copyright 2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_PRELU_H_
#define _MLI_KRN_PRELU_H_

#include "mli_krn_prelu_decl.h"

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
using mli::krn::vdsp::compute_prelu;
using mli::krn::ref::leaky_relu_sa8_run;
using mli::krn::ref::prelu_fx_run;
using mli::krn::ref::prelu_sa8_run;

#elif !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
using mli::krn::dsp::compute_prelu;
using mli::krn::ref::leaky_relu_sa8_run;
using mli::krn::ref::prelu_fx_run;
using mli::krn::ref::prelu_sa8_run;

#else
using mli::krn::ref::compute_prelu;
using mli::krn::ref::leaky_relu_sa8_run;
using mli::krn::ref::prelu_fx_run;
using mli::krn::ref::prelu_sa8_run;

#endif
} // namespace krn
} // namespace mli

////////////////////////////////////////////////////////////////////////////////
// Include implementation
////////////////////////////////////////////////////////////////////////////////
// The reference (*_ref.h) implementation can run on all platforms and is always
// included. Other variants are included based on capabilities. Implementations
// below can depend on each other through declarations in *_decl.h.

#include "impl/mli_krn_prelu_ref.h"

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
#include "impl/mli_krn_prelu_vdsp.h"
#endif

#if !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
#include "impl/mli_krn_prelu_dsp.h"
#endif

#endif // _MLI_KRN_PRELU_H_
