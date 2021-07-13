/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_HLP_CONVERT_TENSOR_H_
#define _MLI_HLP_CONVERT_TENSOR_H_

#include "mli_hlp_convert_tensor_decl.h"

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
namespace hlp {
#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
using mli::hlp::vdsp::convert_quantized_data;
using mli::hlp::ref::convert_float_data;

#elif !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
using mli::hlp::ref::convert_quantized_data;
using mli::hlp::ref::convert_float_data;

#else
using mli::hlp::ref::convert_quantized_data;
using mli::hlp::ref::convert_float_data;

#endif
} // namespace hlp
} // namespace mli

////////////////////////////////////////////////////////////////////////////////
// Include implementation
////////////////////////////////////////////////////////////////////////////////
// The reference (*_ref.h) implementation can run on all platforms and is always
// included. Other variants are included based on capabilities. Implementations
// below can depend on each other through declarations in *_decl.h.
#include "impl/mli_hlp_convert_tensor_ref.h"

#if !defined(MLI_BUILD_REFERENCE) && defined(__Xvec_width)
#include "impl/mli_hlp_convert_tensor_vdsp.h"
#endif

// #if !defined(MLI_BUILD_REFERENCE) && defined(__FXAPI__)
// #include "impl/mli_hlp_convert_tensor_dsp.h"
// #endif

#endif  //_MLI_HLP_CONVERT_TENSOR_H_
