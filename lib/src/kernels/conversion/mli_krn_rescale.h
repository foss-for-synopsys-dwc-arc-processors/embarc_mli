/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_KRN_RESCALE_H_
#define _MLI_KRN_RESCALE_H_

#include "mli_krn_rescale_decl.h"

////////////////////////////////////////////////////////////////////////////////
// Setting up namespace
////////////////////////////////////////////////////////////////////////////////
// Selecting between different variants (depending on hardware features) is
// done with 'using'. A completely different implementation can be used/'using'.
// However, also only a part of the reference together with optimized functions
// (from example *_dsp) can be used/'using'.

namespace snps_arc::metaware::mli {
namespace krn {
// using mli::krn::ref::compute_prelu;
// using mli::krn::ref::prelu_define_requant_alpha_params;
// using mli::krn::ref::compute_prelu_no_broadcast;
// using mli::krn::ref::compute_prelu_broadcast;
// using mli::krn::ref::prelu_fx_run;
using snps_arc::metaware::mli::krn::ref::rescale_prepare_and_run;
} // namespace krn
} // namespace snps_arc::metaware::mli

////////////////////////////////////////////////////////////////////////////////
// Include implementation
////////////////////////////////////////////////////////////////////////////////
// The reference (*_ref.h) implementation can run on all platforms and is always
// included. Other variants are included based on capabilities. Implementations
// below can depend on each other through declarations in *_decl.h.
#include "impl/mli_krn_rescale_ref.h"

#endif // _MLI_KRN_RESCALE_H_
