/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_AVEPOOL_HWC_H_
#define _MLI_KRN_AVEPOOL_HWC_H_

#include "mli_config.h"
#include "mli_math_macros.h"
#include "mli_private_types.h"
#include "mli_krn_avepool_hwc_decl.h"

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
using mli::krn::vdsp::compute_avepool_func;
using mli::krn::ref::get_mul_shift_value;
using mli::krn::ref::compute_avepool;

#elif !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
using mli::krn::dsp::compute_avepool_func;
using mli::krn::ref::get_mul_shift_value;
using mli::krn::ref::compute_avepool;

#else
using mli::krn::ref::compute_avepool_func;
using mli::krn::ref::get_mul_shift_value;
using mli::krn::ref::compute_avepool;

#endif
} // krn
} // mli

////////////////////////////////////////////////////////////////////////////////
// Include implementation
////////////////////////////////////////////////////////////////////////////////
// The reference (*_ref.h) implementation can run on all platforms and is always
// included. Other variants are included based on capabilities. Implementations
// below can depend on each other through declarations in *_decl.h.

#include "impl/mli_krn_avepool_hwc_ref.h"

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
#include "impl/mli_krn_avepool_hwc_vdsp.h"
#endif

#if !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
#include "impl/mli_krn_avepool_hwc_dsp.h"
#endif

#endif // _MLI_KRN_AVEPOOL_HWC_H_
