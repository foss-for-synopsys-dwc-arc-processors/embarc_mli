/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_REDUCE_SUM2D_H_
#define _MLI_KRN_REDUCE_SUM2D_H_

#include "mli_krn_reduce_sum2d_decl.h"

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
using mli::krn::vdsp::reduce_sum2D_v;
using mli::krn::ref::reduce_sum2D;

#elif !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
using mli::krn::dsp::reduce_sum2D_v;
using mli::krn::dsp::reduce_sum2D_d;
using mli::krn::ref::reduce_sum2D;

#else
using mli::krn::ref::reduce_sum2D_v;
using mli::krn::ref::reduce_sum2D;
using mli::krn::ref::reduce_sum;

#endif
} // namespace krn
} // namespace mli

////////////////////////////////////////////////////////////////////////////////
// Include implementation
////////////////////////////////////////////////////////////////////////////////
// The reference (*_ref.h) implementation can run on all platforms and is always
// included. Other variants are included based on capabilities. Implementations
// below can depend on each other through declarations in *_decl.h.
#include "impl/mli_krn_reduce_sum2d_ref.h"

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
#include "impl/mli_krn_reduce_sum2d_vdsp.h"
#endif

#if !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
#include "impl/mli_krn_reduce_sum2d_dsp.h"
#endif

#endif // _MLI_KRN_REDUCE_SUM2D_H_
