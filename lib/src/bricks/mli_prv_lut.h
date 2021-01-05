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
// Selecting between different variants (depending on hardware features) is
// done with 'using'. A completely different implementation can be used/'using'.
// However, also only a part of the reference together with optimized functions
// (from example *_dsp) can be used/'using'.

namespace mli {
namespace krn {
#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
using mli::krn::vdsp::compute_activation_lut;
using mli::krn::vdsp::activation_lut_vec_elem_interpolate;
using mli::krn::vdsp::activation_lut_vec_elem_no_interpolate;
using mli::krn::ref::activation_lut;
using mli::krn::ref::activation_lut_one_elem_interpolate;
using mli::krn::ref::activation_lut_one_elem_no_interpolate;

#elif !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
using mli::krn::dsp::compute_activation_lut;
using mli::krn::dsp::activation_lut_two_elem_interpolate;
using mli::krn::dsp::activation_lut_two_elem_no_interpolate;
using mli::krn::ref::activation_lut;
using mli::krn::ref::activation_lut_one_elem_interpolate;
using mli::krn::ref::activation_lut_one_elem_no_interpolate;

#else
using mli::krn::ref::activation_lut;
using mli::krn::ref::compute_activation_lut;
using mli::krn::ref::activation_lut_one_elem_interpolate;
using mli::krn::ref::activation_lut_one_elem_no_interpolate;

#endif
} // krn
} // mli

////////////////////////////////////////////////////////////////////////////////
// Include implementation
////////////////////////////////////////////////////////////////////////////////
// The reference (*_ref.h) implementation can run on all platforms and is always
// included. Other variants are included based on capabilities. Implementations
// below can depend on each other through declarations in *_decl.h.

#include "impl/mli_prv_lut_ref.h"

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
#include "impl/mli_prv_lut_vdsp.h"
#endif

#if !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
#include "impl/mli_prv_lut_dsp.h"
#endif

#endif  //_MLI_PRIVATE_LUT_H_
